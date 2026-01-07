"""
Enhanced Forecasting Model v2.1
Combines CSV features + Video features for better predictions
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
print("ENHANCED FORECASTING MODEL v2.1")
print("="*60)
print("Combining CSV + Video Features")
print("="*60)

# Load data
print("\nLoading data...")
train_df = pd.read_csv('Train.csv')
video_features_df = pd.read_csv('video_features.csv')

print(f"Training data: {len(train_df)} segments")
print(f"Video features: {len(video_features_df)} videos")

# Match video features to training data
print("\n" + "="*60)
print("MATCHING VIDEO FEATURES TO TRAINING DATA")
print("="*60)

# Extract timestamp from video filename to match with training data
def extract_timestamp_from_filename(filename):
    """Extract timestamp from video filename"""
    # Format: normanniles1_2025-10-20-06-00-45.mp4
    parts = filename.replace('.mp4', '').split('_')
    if len(parts) >= 5:
        date = parts[1]
        time = f"{parts[2]}:{parts[3]}:{parts[4]}"
        return f"{date} {time}"
    return None

video_features_df['timestamp'] = video_features_df['video_filename'].apply(extract_timestamp_from_filename)

# Match with training data
train_df['timestamp'] = train_df['datetimestamp_start']

# Merge
print("Merging video features with training data...")
train_enhanced = train_df.merge(
    video_features_df[['timestamp', 'motion_mean', 'motion_std', 'motion_max',
                       'occupancy_mean', 'occupancy_std', 'occupancy_max',
                       'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']],
    on='timestamp',
    how='left'
)

print(f"Merged data: {len(train_enhanced)} segments")
print(f"Segments with video features: {train_enhanced['motion_mean'].notna().sum()}")

# Fill missing video features with median (for segments without videos)
video_feature_cols = ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                      'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                      'flow_rate_proxy', 'congestion_proxy']

for col in video_feature_cols:
    median_val = train_enhanced[col].median()
    train_enhanced[col].fillna(median_val, inplace=True)
    print(f"  {col}: {train_enhanced[col].notna().sum()} filled")

# Create forecasting dataset
print("\n" + "="*60)
print("CREATING FORECASTING DATASET")
print("="*60)

def create_enhanced_forecasting_dataset(df, forecast_horizon=5):
    """Create forecasting dataset with CSV + video features"""
    df = df.sort_values(['view_label', 'time_segment_id']).reset_index(drop=True)
    
    features_list = []
    targets_enter = []
    
    for location in df['view_label'].unique():
        location_df = df[df['view_label'] == location].reset_index(drop=True)
        
        for i in range(15, len(location_df) - forecast_horizon):
            # Target: 5 minutes ahead
            future_point = location_df.iloc[i + forecast_horizon]
            current_point = location_df.iloc[i]
            
            # CSV features
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
                    # Add video features from past
                    feature_dict[f'motion_lag_{lag}'] = past_row.get('motion_mean', 0)
                    feature_dict[f'occupancy_lag_{lag}'] = past_row.get('occupancy_mean', 0)
            
            features_list.append(feature_dict)
            targets_enter.append(future_point.get('congestion_enter_rating', 'free flowing'))
    
    return pd.DataFrame(features_list), targets_enter

print("Creating enhanced forecasting dataset...")
X_train, y_enter_train = create_enhanced_forecasting_dataset(train_enhanced, forecast_horizon=5)

print(f"\nDataset created:")
print(f"  Samples: {len(X_train)}")
print(f"  Features: {len(X_train.columns)}")
print(f"  Feature columns: {X_train.columns.tolist()}")

# Encode features
print("\n" + "="*60)
print("ENCODING FEATURES")
print("="*60)

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

# Encode targets
le_target = LabelEncoder()
y_enter_encoded = le_target.fit_transform(y_enter_train)

print(f"Features encoded: {X_train.shape}")
print(f"Target classes: {le_target.classes_}")

# Split for validation
split_idx = int(len(X_train) * 0.8)
X_train_split = X_train.iloc[:split_idx]
X_val_split = X_train.iloc[split_idx:]
y_train_split = y_enter_encoded[:split_idx]
y_val_split = y_enter_encoded[split_idx:]

print(f"\nTrain/Val split:")
print(f"  Train: {len(X_train_split)}")
print(f"  Val: {len(X_val_split)}")

# Train enhanced model
print("\n" + "="*60)
print("TRAINING ENHANCED MODEL")
print("="*60)

model = LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=300,  # Increased
    learning_rate=0.03,
    max_depth=8,  # Increased
    num_leaves=63,  # Increased
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    verbose=-1
)

print("Training with CSV + video features...")
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
print("TOP 10 FEATURE IMPORTANCE")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Save model
os.makedirs(config.MODELS_DIR, exist_ok=True)
model_path = os.path.join(config.MODELS_DIR, 'forecasting_model_v2.1_enhanced.pkl')

model_data = {
    'model': model,
    'label_encoder': le_target,
    'feature_columns': X_train.columns.tolist(),
    'forecast_horizon': 5,
    'video_feature_cols': video_feature_cols
}

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to {model_path}")

# Compare with v2.0
print("\n" + "="*60)
print("COMPARISON WITH v2.0")
print("="*60)
print(f"v2.0 (CSV only):     F1 macro = 0.4677")
print(f"v2.1 (CSV + video):  F1 macro = {f1_macro:.4f}")
improvement = ((f1_macro - 0.4677) / 0.4677) * 100
print(f"Improvement: {improvement:+.1f}%")

print("\n" + "="*60)
print("SUMMARY - v2.1")
print("="*60)
print(f"✓ Enhanced with video features")
print(f"✓ Validation F1 (macro): {f1_macro:.4f}")
print(f"✓ Total features: {len(X_train.columns)}")
print(f"✓ Ready for submission generation")
print("="*60)
