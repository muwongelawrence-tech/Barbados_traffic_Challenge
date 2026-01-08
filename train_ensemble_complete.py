"""
Phase 2: Complete Pipeline - Feature Engineering + Ensemble
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle
import warnings
warnings.filterwarnings('ignore')

import config
from feature_engineering import create_advanced_features

print("="*60)
print("PHASE 2: COMPLETE PIPELINE")
print("="*60)

# Load tuned model
print("\nLoading tuned model v3.1...")
with open('models/forecasting_model_v3.1_tuned.pkl', 'rb') as f:
    tuned_model_data = pickle.load(f)

best_params = tuned_model_data['best_params']
print(f"✓ Loaded best parameters (F1: 0.5219)")

# Load and prepare data (same as before)
train_df = pd.read_csv('Train.csv')
video_features_df = pd.read_csv('video_features.csv')

# Timestamp functions
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

# Prepare and merge
video_features_df['timestamp'] = video_features_df['video_filename'].apply(extract_timestamp_from_filename)
video_features_df['location'] = video_features_df['video_filename'].apply(extract_location_from_filename)
video_features_df['timestamp_simple'] = video_features_df['timestamp'].apply(simplify_timestamp)
video_features_df['match_key'] = video_features_df['location'] + '|' + video_features_df['timestamp_simple'].fillna('')

train_df['timestamp_simple'] = train_df['datetimestamp_start'].apply(simplify_timestamp)
train_df['match_key'] = train_df['view_label'] + '|' + train_df['timestamp_simple'].fillna('')

video_cols = ['match_key', 'motion_mean', 'motion_std', 'motion_max',
              'occupancy_mean', 'occupancy_std', 'occupancy_max',
              'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']

train_enhanced = train_df.merge(video_features_df[video_cols], on='match_key', how='left')

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
            
            for col in video_feature_cols:
                feature_dict[f'current_{col}'] = current_point[col]
            
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
print("CREATING DATASET WITH FEATURE ENGINEERING")
print("="*60)

X_train, y_enter_train = create_enhanced_forecasting_dataset(train_enhanced, forecast_horizon=5)

# Encode categorical before feature engineering
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

# Apply feature engineering
print("\nApplying advanced feature engineering...")
X_train_enhanced = create_advanced_features(X_train, video_feature_cols)

le_target = LabelEncoder()
y_enter_encoded = le_target.fit_transform(y_enter_train)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_enter_encoded), y=y_enter_encoded)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Split
split_idx = int(len(X_train_enhanced) * 0.8)
X_train_split = X_train_enhanced.iloc[:split_idx]
X_val_split = X_train_enhanced.iloc[split_idx:]
y_train_split = y_enter_encoded[:split_idx]
y_val_split = y_enter_encoded[split_idx:]

print(f"\nDataset: {len(X_train_enhanced)} samples, {len(X_train_enhanced.columns)} features")
print(f"Train: {len(X_train_split)}, Val: {len(X_val_split)}")

# Train ensemble
print("\n" + "="*60)
print("TRAINING ENSEMBLE MODELS")
print("="*60)

models = {}
predictions = {}

# 1. LightGBM (tuned)
print("\n1. Training LightGBM (tuned)...")
lgbm_model = LGBMClassifier(**best_params)
lgbm_model.fit(X_train_split, y_train_split)
lgbm_pred = lgbm_model.predict(X_val_split)
lgbm_pred_proba = lgbm_model.predict_proba(X_val_split)

lgbm_f1 = f1_score(y_val_split, lgbm_pred, average='macro')
print(f"   LightGBM F1: {lgbm_f1:.4f}")

models['lgbm'] = lgbm_model
predictions['lgbm'] = lgbm_pred
predictions['lgbm_proba'] = lgbm_pred_proba

# 2. XGBoost
print("\n2. Training XGBoost...")
sample_weights = np.array([class_weight_dict[y] for y in y_train_split])

xgb_params = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbosity': 0
}

xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(X_train_split, y_train_split, sample_weight=sample_weights)
xgb_pred = xgb_model.predict(X_val_split)
xgb_pred_proba = xgb_model.predict_proba(X_val_split)

xgb_f1 = f1_score(y_val_split, xgb_pred, average='macro')
print(f"   XGBoost F1: {xgb_f1:.4f}")

models['xgb'] = xgb_model
predictions['xgb'] = xgb_pred
predictions['xgb_proba'] = xgb_pred_proba

# 3. CatBoost
print("\n3. Training CatBoost...")
catboost_params = {
    'loss_function': 'MultiClass',
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': False,
    'class_weights': class_weight_dict
}

catboost_model = CatBoostClassifier(**catboost_params)
catboost_model.fit(X_train_split, y_train_split)
catboost_pred = catboost_model.predict(X_val_split).flatten().astype(int)
catboost_pred_proba = catboost_model.predict_proba(X_val_split)

catboost_f1 = f1_score(y_val_split, catboost_pred, average='macro')
print(f"   CatBoost F1: {catboost_f1:.4f}")

models['catboost'] = catboost_model
predictions['catboost'] = catboost_pred
predictions['catboost_proba'] = catboost_pred_proba

# Ensemble
print("\n" + "="*60)
print("CREATING ENSEMBLE")
print("="*60)

total_f1 = lgbm_f1 + xgb_f1 + catboost_f1
weights = {
    'lgbm': lgbm_f1 / total_f1,
    'xgb': xgb_f1 / total_f1,
    'catboost': catboost_f1 / total_f1
}

print(f"\nModel weights:")
for model, weight in weights.items():
    print(f"  {model}: {weight:.3f}")

ensemble_proba = (
    weights['lgbm'] * predictions['lgbm_proba'] +
    weights['xgb'] * predictions['xgb_proba'] +
    weights['catboost'] * predictions['catboost_proba']
)

ensemble_pred = np.argmax(ensemble_proba, axis=1)

ensemble_f1 = f1_score(y_val_split, ensemble_pred, average='macro')
ensemble_acc = accuracy_score(y_val_split, ensemble_pred)

print(f"\nEnsemble Performance:")
print(f"  Accuracy: {ensemble_acc:.4f}")
print(f"  F1 (macro): {ensemble_f1:.4f}")

y_pred_labels = le_target.inverse_transform(ensemble_pred)
y_val_labels = le_target.inverse_transform(y_val_split)
print("\nClassification Report:")
print(classification_report(y_val_labels, y_pred_labels, zero_division=0))

# Save ensemble
os.makedirs(config.MODELS_DIR, exist_ok=True)
ensemble_path = os.path.join(config.MODELS_DIR, 'ensemble_model_v3.2.pkl')

ensemble_data = {
    'models': models,
    'weights': weights,
    'label_encoder': le_target,
    'feature_columns': X_train_enhanced.columns.tolist(),
    'forecast_horizon': 5,
    'video_feature_cols': video_feature_cols
}

with open(ensemble_path, 'wb') as f:
    pickle.dump(ensemble_data, f)

print(f"\n✓ Ensemble saved to {ensemble_path}")

# Final comparison
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"v2.1 (video structure):    F1 = 0.4821")
print(f"v3.0 (matched video):      F1 = 0.4568")
print(f"v3.1 (tuned + weighted):   F1 = 0.5219")
print(f"v3.2 (ensemble):           F1 = {ensemble_f1:.4f}")

if ensemble_f1 > 0.5219:
    improvement = ((ensemble_f1 - 0.5219) / 0.5219) * 100
    print(f"✓ Improvement over v3.1: {improvement:+.1f}%")

print("\n" + "="*60)
print("PHASE 2 COMPLETE!")
print("="*60)
print(f"✓ Feature engineering: {len(X_train_enhanced.columns)} features")
print(f"✓ Ensemble models: LightGBM + XGBoost + CatBoost")
print(f"✓ Final F1 (macro): {ensemble_f1:.4f}")
print(f"✓ Expected leaderboard: 0.53-0.57")
print("="*60)
