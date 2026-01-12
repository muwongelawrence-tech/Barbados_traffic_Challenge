"""
STACKING ENSEMBLE v7.7
Train a meta-learner on top of base models
Expected: 0.59 → 0.62-0.68
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import re
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STACKING ENSEMBLE v7.7")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')
video_features_df = pd.read_csv('video_features.csv')

print(f"Training data: {len(train_df)}")

# Prepare video features matching
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

# Merge training with video features
train_df['timestamp_simple'] = train_df['datetimestamp_start'].apply(simplify_timestamp)
train_df['match_key'] = train_df['view_label'] + '|' + train_df['timestamp_simple'].fillna('')

video_cols = ['match_key', 'motion_mean', 'motion_std', 'motion_max',
              'occupancy_mean', 'occupancy_std', 'occupancy_max',
              'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']

train_enhanced = train_df.merge(video_features_df[video_cols], on='match_key', how='left')

# Fill missing
for col in ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
            'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
            'flow_rate_proxy', 'congestion_proxy']:
    median_val = video_features_df[col].median()
    train_enhanced[col].fillna(median_val, inplace=True)

# Extract hour
train_enhanced['hour'] = train_enhanced['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0])
)

def create_features(df):
    features = pd.DataFrame()
    
    le_location = LabelEncoder()
    le_signaling = LabelEncoder()
    
    features['location'] = le_location.fit_transform(df['view_label'])
    features['signaling'] = le_signaling.fit_transform(df['signaling'])
    features['hour'] = df['hour']
    features['time_segment_id'] = df['time_segment_id']
    
    for col in ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                'flow_rate_proxy', 'congestion_proxy']:
        features[col] = df[col]
    
    features['hour_x_signaling'] = features['hour'] * features['signaling']
    features['location_x_signaling'] = features['location'] * features['signaling']
    features['motion_x_occupancy'] = features['motion_mean'] * features['occupancy_mean']
    features['flow_x_occupancy'] = features['flow_rate_proxy'] * features['occupancy_mean']
    features['is_peak_hour'] = ((features['hour'] >= 10) & (features['hour'] <= 17)).astype(int)
    features['is_morning'] = (features['hour'] < 10).astype(int)
    features['motion_squared'] = features['motion_mean'] ** 2
    features['occupancy_squared'] = features['occupancy_mean'] ** 2
    
    return features, le_location, le_signaling

X, le_location, le_signaling = create_features(train_enhanced)
y_enter = train_enhanced['congestion_enter_rating']

le_target = LabelEncoder()
y_enter_encoded = le_target.fit_transform(y_enter)

print(f"Features: {X.shape[1]}")
print(f"Classes: {le_target.classes_}")

# STACKING: Train base models and meta-learner
print("\n" + "="*60)
print("LEVEL 1: TRAINING BASE MODELS")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store out-of-fold predictions for meta-learner
meta_train = np.zeros((len(X), 3 * 4))  # 3 models × 4 classes
base_models = {'lgbm': [], 'xgb': [], 'lgbm2': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enter_encoded)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_enter_encoded[train_idx], y_enter_encoded[val_idx]
    
    # LightGBM model 1
    lgbm1 = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        class_weight='balanced'
    )
    lgbm1.fit(X_train, y_train)
    base_models['lgbm'].append(lgbm1)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    base_models['xgb'].append(xgb_model)
    
    # LightGBM model 2 (different params)
    lgbm2 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=43,
        class_weight='balanced'
    )
    lgbm2.fit(X_train, y_train)
    base_models['lgbm2'].append(lgbm2)
    
    # Get out-of-fold predictions
    lgbm1_proba = lgbm1.predict_proba(X_val)
    xgb_proba = xgb_model.predict_proba(X_val)
    lgbm2_proba = lgbm2.predict_proba(X_val)
    
    # Stack predictions
    meta_train[val_idx, :4] = lgbm1_proba
    meta_train[val_idx, 4:8] = xgb_proba
    meta_train[val_idx, 8:12] = lgbm2_proba
    
    print(f"Fold {fold+1}: Models trained")

print("✓ Base models trained")

# LEVEL 2: Train meta-learner
print("\n" + "="*60)
print("LEVEL 2: TRAINING META-LEARNER")
print("="*60)

meta_learner = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
meta_learner.fit(meta_train, y_enter_encoded)

# Evaluate stacking
meta_pred = meta_learner.predict(meta_train)
stacking_score = f1_score(y_enter_encoded, meta_pred, average='macro')
print(f"✓ Stacking F1: {stacking_score:.4f}")

# Train EXIT model
exit_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=6,
    random_state=42
)
y_exit_encoded = le_target.transform(train_enhanced['congestion_exit_rating'])
exit_model.fit(X, y_exit_encoded)

# Generate predictions
print("\n" + "="*60)
print("GENERATING PREDICTIONS WITH STACKING")
print("="*60)

predictions = []

for idx, row in sample_df.iterrows():
    id_str = row['ID']
    
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        rating_type = match.group(3)
        
        estimated_hour = 6 + ((segment_id // 60) % 12)
        estimated_signaling = 'none'
        
        test_features = pd.DataFrame([{
            'location': le_location.transform([location])[0],
            'signaling': le_signaling.transform([estimated_signaling])[0],
            'hour': estimated_hour,
            'time_segment_id': segment_id,
            'motion_mean': video_features_df['motion_mean'].median(),
            'motion_std': video_features_df['motion_std'].median(),
            'motion_max': video_features_df['motion_max'].median(),
            'occupancy_mean': video_features_df['occupancy_mean'].median(),
            'occupancy_std': video_features_df['occupancy_std'].median(),
            'occupancy_max': video_features_df['occupancy_max'].median(),
            'vehicle_count_proxy': video_features_df['vehicle_count_proxy'].median(),
            'flow_rate_proxy': video_features_df['flow_rate_proxy'].median(),
            'congestion_proxy': video_features_df['congestion_proxy'].median(),
        }])
        
        test_features['hour_x_signaling'] = test_features['hour'] * test_features['signaling']
        test_features['location_x_signaling'] = test_features['location'] * test_features['signaling']
        test_features['motion_x_occupancy'] = test_features['motion_mean'] * test_features['occupancy_mean']
        test_features['flow_x_occupancy'] = test_features['flow_rate_proxy'] * test_features['occupancy_mean']
        test_features['is_peak_hour'] = int((estimated_hour >= 10) and (estimated_hour <= 17))
        test_features['is_morning'] = int(estimated_hour < 10)
        test_features['motion_squared'] = test_features['motion_mean'] ** 2
        test_features['occupancy_squared'] = test_features['occupancy_mean'] ** 2
        
        if rating_type == 'enter':
            # Get predictions from all base models
            meta_test = np.zeros((1, 12))
            
            for i in range(5):
                lgbm1_proba = base_models['lgbm'][i].predict_proba(test_features)
                xgb_proba = base_models['xgb'][i].predict_proba(test_features)
                lgbm2_proba = base_models['lgbm2'][i].predict_proba(test_features)
                
                meta_test[0, :4] += lgbm1_proba[0]
                meta_test[0, 4:8] += xgb_proba[0]
                meta_test[0, 8:12] += lgbm2_proba[0]
            
            meta_test /= 5  # Average across folds
            
            # Meta-learner prediction
            pred_encoded = meta_learner.predict(meta_test)[0]
            prediction = le_target.inverse_transform([pred_encoded])[0]
        else:
            pred_encoded = exit_model.predict(test_features)[0]
            prediction = le_target.inverse_transform([pred_encoded])[0]
        
        predictions.append({
            'ID': id_str,
            'Target': prediction,
            'Target_Accuracy': prediction
        })
    else:
        predictions.append({
            'ID': id_str,
            'Target': 'free flowing',
            'Target_Accuracy': 'free flowing'
        })

submission_df = pd.DataFrame(predictions)

output_path = 'submissions/v7.7_stacking.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}")

print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nPrediction distribution:")
print(submission_df['Target'].value_counts(normalize=True).sort_index())

print("\nPrediction counts:")
print(submission_df['Target'].value_counts())

print("\n" + "="*60)
print("STACKING SUMMARY")
print("="*60)
print("✓ 3 diverse base models (LightGBM×2, XGBoost)")
print("✓ Logistic Regression meta-learner")
print(f"✓ Stacking F1: {stacking_score:.4f}")
print("\nExpected score: 0.60-0.65")
print("="*60)
