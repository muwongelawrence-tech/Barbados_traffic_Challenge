"""
Phase 2a: Hyperparameter Tuning with Optuna
Optimize LightGBM for video features
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler
import pickle
import warnings
warnings.filterwarnings('ignore')

import config

print("="*60)
print("PHASE 2a: HYPERPARAMETER TUNING WITH OPTUNA")
print("="*60)

# Load and prepare data (same as v3.0)
train_df = pd.read_csv('Train.csv')
video_features_df = pd.read_csv('video_features.csv')

print(f"\nTrain data: {len(train_df)}")
print(f"Video features: {len(video_features_df)}")

# Timestamp and location extraction functions
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

# Merge
video_cols = ['match_key', 'motion_mean', 'motion_std', 'motion_max',
              'occupancy_mean', 'occupancy_std', 'occupancy_max',
              'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']

train_enhanced = train_df.merge(video_features_df[video_cols], on='match_key', how='left')

matched_count = train_enhanced['motion_mean'].notna().sum()
print(f"\nMatched segments: {matched_count}/{len(train_enhanced)} ({matched_count/len(train_enhanced)*100:.1f}%)")

# Fill missing
video_feature_cols = ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                      'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                      'flow_rate_proxy', 'congestion_proxy']

for col in video_feature_cols:
    median_val = train_enhanced[col].median()
    train_enhanced[col].fillna(median_val, inplace=True)

# Create dataset
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
            
            # Video features
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

print("\n" + "="*60)
print("CREATING DATASET")
print("="*60)

X_train, y_enter_train = create_enhanced_forecasting_dataset(train_enhanced, forecast_horizon=5)

print(f"Dataset: {len(X_train)} samples, {len(X_train.columns)} features")

# Encode
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

le_target = LabelEncoder()
y_enter_encoded = le_target.fit_transform(y_enter_train)

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_enter_encoded), y=y_enter_encoded)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

print(f"\nClass weights: {class_weight_dict}")

# Optuna objective
def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 4,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1,
        'class_weight': class_weight_dict
    }
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_enter_encoded):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_enter_encoded[train_idx], y_enter_encoded[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)
print("Running Optuna optimization (50 trials)...")
print("This will take ~30-45 minutes")

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)
print(f"Best F1 (macro): {study.best_value:.4f}")
print(f"\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best params
print("\n" + "="*60)
print("TRAINING FINAL MODEL")
print("="*60)

best_params = study.best_params.copy()
best_params.update({
    'objective': 'multiclass',
    'num_class': 4,
    'random_state': 42,
    'verbose': -1,
    'class_weight': class_weight_dict
})

# Split for final evaluation
split_idx = int(len(X_train) * 0.8)
X_train_split = X_train.iloc[:split_idx]
X_val_split = X_train.iloc[split_idx:]
y_train_split = y_enter_encoded[:split_idx]
y_val_split = y_enter_encoded[split_idx:]

model = LGBMClassifier(**best_params)
model.fit(X_train_split, y_train_split)

# Evaluate
y_pred_val = model.predict(X_val_split)

acc = accuracy_score(y_val_split, y_pred_val)
f1_macro = f1_score(y_val_split, y_pred_val, average='macro')
f1_weighted = f1_score(y_val_split, y_pred_val, average='weighted')

print(f"\nFinal Validation Performance:")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 (macro): {f1_macro:.4f}")
print(f"  F1 (weighted): {f1_weighted:.4f}")

from sklearn.metrics import classification_report
y_pred_labels = le_target.inverse_transform(y_pred_val)
y_val_labels = le_target.inverse_transform(y_val_split)
print("\nClassification Report:")
print(classification_report(y_val_labels, y_pred_labels, zero_division=0))

# Save model
os.makedirs(config.MODELS_DIR, exist_ok=True)
model_path = os.path.join(config.MODELS_DIR, 'forecasting_model_v3.1_tuned.pkl')

model_data = {
    'model': model,
    'label_encoder': le_target,
    'feature_columns': X_train.columns.tolist(),
    'forecast_horizon': 5,
    'video_feature_cols': video_feature_cols,
    'matched_count': matched_count,
    'best_params': best_params,
    'optuna_study': study
}

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to {model_path}")

# Comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"v2.1 (video structure):    F1 = 0.4821")
print(f"v3.0 (matched video):      F1 = 0.4568")
print(f"v3.1 (tuned + weighted):   F1 = {f1_macro:.4f}")

if f1_macro > 0.4821:
    improvement = ((f1_macro - 0.4821) / 0.4821) * 100
    print(f"✓ Improvement over v2.1: {improvement:+.1f}%")

print("\n" + "="*60)
print("SUMMARY - v3.1")
print("="*60)
print(f"✓ Hyperparameter tuning: Complete")
print(f"✓ Class weighting: Applied")
print(f"✓ Validation F1 (macro): {f1_macro:.4f}")
print(f"✓ Best trial: {study.best_trial.number}")
print("="*60)
