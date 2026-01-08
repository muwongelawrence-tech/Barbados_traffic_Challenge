"""
Train Enhanced Model v3.0 with PROPERLY MATCHED Video Features
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

import config

print("="*60)
print("ENHANCED MODEL v3.0 - WITH MATCHED VIDEO FEATURES")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
video_features_df = pd.read_csv('video_features.csv')

print(f"\nTrain data: {len(train_df)}")
print(f"Video features: {len(video_features_df)}")

# Extract timestamp and location from video filename
def extract_timestamp_from_filename(filename):
    parts = filename.replace('.mp4', '').split('_')
    if len(parts) >= 2:
        date_time = parts[1]
        components = date_time.split('-')
        if len(components) == 6:
            date = f"{components[0]}-{components[1]}-{components[2]}"
            time = f"{components[3]}:{components[4]}:{components[5]}"
            return f"{date} {time}"
    return None

def extract_location_from_filename(filename):
    parts = filename.split('_')[0]
    location_map = {
        'normanniles1': 'Norman Niles #1',
        'normanniles2': 'Norman Niles #2',
        'normanniles3': 'Norman Niles #3',
        'normanniles4': 'Norman Niles #4'
    }
    return location_map.get(parts, parts)

def simplify_timestamp(ts_str):
    """Simplify to YYYY-MM-DD HH:MM"""
    try:
        parts = str(ts_str).split()
        if len(parts) >= 2:
            date = parts[0]
            time_parts = parts[1].split(':')
            if len(time_parts) >= 2:
                return f"{date} {time_parts[0]}:{time_parts[1]}"
    except:
        pass
    return None

# Prepare video features
video_features_df['timestamp'] = video_features_df['video_filename'].apply(extract_timestamp_from_filename)
video_features_df['location'] = video_features_df['video_filename'].apply(extract_location_from_filename)
video_features_df['timestamp_simple'] = video_features_df['timestamp'].apply(simplify_timestamp)
video_features_df['match_key'] = video_features_df['location'] + '|' + video_features_df['timestamp_simple'].fillna('')

# Prepare train data
train_df['timestamp_simple'] = train_df['datetimestamp_start'].apply(simplify_timestamp)
train_df['match_key'] = train_df['view_label'] + '|' + train_df['timestamp_simple'].fillna('')

# Merge with proper matching
print("\n" + "="*60)
print("MATCHING VIDEO FEATURES")
print("="*60)

video_cols = ['match_key', 'motion_mean', 'motion_std', 'motion_max',
              'occupancy_mean', 'occupancy_std', 'occupancy_max',
              'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']

train_enhanced = train_df.merge(
    video_features_df[video_cols],
    on='match_key',
    how='left'
)

matched_count = train_enhanced['motion_mean'].notna().sum()
print(f"Matched segments: {matched_count}/{len(train_enhanced)} ({matched_count/len(train_enhanced)*100:.1f}%)")

# Fill missing with median
video_feature_cols = ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                      'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                      'flow_rate_proxy', 'congestion_proxy']

for col in video_feature_cols:
    median_val = train_enhanced[col].median()
    filled_count = train_enhanced[col].isna().sum()
    train_enhanced[col].fillna(median_val, inplace=True)
    print(f"  {col}: {matched_count} matched, {filled_count} filled with median")

# Create forecasting dataset
print("\n" + "="*60)
print("CREATING FORECASTING DATASET")
print("="*60)

def create_enhanced_forecasting_dataset(df, forecast_horizon=5):
    df = df.sort_values(['view_label', 'time_segment_id']).reset_index(drop=True)
    
    features_list = []
    targets_enter = []
    
    for location in df['view_label'].unique():
        location_df = df[df['view_label'] == location].reset_index(drop=True)
        
        for i in range(15, len(location_df) - forecast_horizon):
            future_point = location_df.iloc[i + forecast_horizon]
            current_point = location_df.iloc[i]
            
            feature_dict = {
                'current_segment': current_point['time_segment_id'],
                'location': location,
                'hour': int(str(current_point['datetimestamp_start']).split()[1].split(':')[0]),
                'signaling': current_point['signaling'],
            }
            
            # Video features (current)
            for col in video_feature_cols:
                feature_dict[f'current_{col}'] = current_point[col]
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                if i - lag >= 0:
                    past_row = location_df.iloc[i - lag]
                    feature_dict[f'enter_lag_{lag}'] = past_row.get('congestion_enter_rating', 'free flowing')
                    feature_dict[f'motion_lag_{lag}'] = past_row.get('motion_mean', 0)
                    feature_dict[f'occupancy_lag_{lag}'] = past_row.get('occupancy_mean', 0)
            
            features_list.append(feature_dict)
            targets_enter.append(future_point.get('congestion_enter_rating', 'free flowing'))
    
    return pd.DataFrame(features_list), targets_enter

X_train, y_enter_train = create_enhanced_forecasting_dataset(train_enhanced, forecast_horizon=5)

print(f"Dataset: {len(X_train)} samples, {len(X_train.columns)} features")

# Encode
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

le_target = LabelEncoder()
y_enter_encoded = le_target.fit_transform(y_enter_train)

# Split
split_idx = int(len(X_train) * 0.8)
X_train_split = X_train.iloc[:split_idx]
X_val_split = X_train.iloc[split_idx:]
y_train_split = y_enter_encoded[:split_idx]
y_val_split = y_enter_encoded[split_idx:]

print(f"Train: {len(X_train_split)}, Val: {len(X_val_split)}")

# Train
print("\n" + "="*60)
print("TRAINING MODEL v3.0")
print("="*60)

model = LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=300,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=63,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    verbose=-1
)

model.fit(X_train_split, y_train_split)

# Evaluate
y_pred_val = model.predict(X_val_split)

acc = accuracy_score(y_val_split, y_pred_val)
f1_macro = f1_score(y_val_split, y_pred_val, average='macro')
f1_weighted = f1_score(y_val_split, y_pred_val, average='weighted')

print(f"\nValidation Performance:")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 (macro): {f1_macro:.4f}")
print(f"  F1 (weighted): {f1_weighted:.4f}")

print("\nClassification Report:")
y_pred_labels = le_target.inverse_transform(y_pred_val)
y_val_labels = le_target.inverse_transform(y_val_split)
print(classification_report(y_val_labels, y_pred_labels, zero_division=0))

# Feature importance
print("\n" + "="*60)
print("TOP 15 FEATURE IMPORTANCE")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# Save
os.makedirs(config.MODELS_DIR, exist_ok=True)
model_path = os.path.join(config.MODELS_DIR, 'forecasting_model_v3.0.pkl')

model_data = {
    'model': model,
    'label_encoder': le_target,
    'feature_columns': X_train.columns.tolist(),
    'forecast_horizon': 5,
    'video_feature_cols': video_feature_cols,
    'matched_count': matched_count
}

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to {model_path}")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"v2.0 (CSV only):           F1 = 0.4677")
print(f"v2.1 (video structure):    F1 = 0.4821")
print(f"v3.0 (matched video):      F1 = {f1_macro:.4f}")

if f1_macro > 0.4821:
    improvement = ((f1_macro - 0.4821) / 0.4821) * 100
    print(f"✓ Improvement over v2.1: {improvement:+.1f}%")
else:
    print("⚠️  No improvement - need more matched data")

print("\n" + "="*60)
print("SUMMARY - v3.0")
print("="*60)
print(f"✓ Real video features: {matched_count} segments ({matched_count/len(train_enhanced)*100:.1f}%)")
print(f"✓ Validation F1 (macro): {f1_macro:.4f}")
print(f"✓ Total features: {len(X_train.columns)}")
print("="*60)
