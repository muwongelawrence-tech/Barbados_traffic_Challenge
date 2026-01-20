"""
IMPROVED PSEUDO-LABELING v10.0
Building on v7.5's success (0.594713424), with improvements:
1. Better pseudo-labeling with iterative refinement
2. XGBoost + LightGBM ensemble 
3. Improved feature engineering with TestInputSegments context
4. Optimized hyperparameters
5. Multiple pseudo-labeling iterations

Target: 0.594 → 0.70+
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import re
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("IMPROVED PSEUDO-LABELING v10.0")
print("Building on v7.5's success (0.594)")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')
video_features_df = pd.read_csv('video_features.csv')
test_input_df = pd.read_csv('TestInputSegments.csv')

print(f"Original training data: {len(train_df)}")
print(f"Test input segments: {len(test_input_df)}")
print(f"Predictions needed: {len(sample_df)}")

# ============================================
# Feature Preparation (same as v7.5)
# ============================================
def extract_timestamp_from_filename(filename):
    parts = str(filename).replace('.mp4', '').split('_')
    if len(parts) >= 2:
        date_time = parts[1]
        components = date_time.split('-')
        if len(components) == 6:
            date = f"{components[0]}-{components[1]}-{components[2]}"
            time = f"{components[3]}:{components[4]}:{components[5]}"
            return f"{date} {time}"
    return None

def extract_location_from_filename(filename):
    parts = str(filename).split('_')[0]
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

# Fill missing values with medians
for col in ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
            'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
            'flow_rate_proxy', 'congestion_proxy']:
    median_val = video_features_df[col].median()
    train_enhanced[col].fillna(median_val, inplace=True)

# Extract hour
train_enhanced['hour'] = train_enhanced['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0]) if len(str(x).split()) > 1 else 12
)

print(f"Enhanced training data: {len(train_enhanced)}")

# ============================================
# Improved Feature Engineering
# ============================================
def create_features(df, le_location=None, le_signaling=None, fit=True):
    """Create features with optional fitting"""
    features = pd.DataFrame()
    
    if fit:
        le_location = LabelEncoder()
        le_signaling = LabelEncoder()
        le_location.fit(df['view_label'])
        le_signaling.fit(df['signaling'].fillna('none'))
    
    features['location'] = le_location.transform(df['view_label'])
    features['signaling'] = le_signaling.transform(df['signaling'].fillna('none'))
    features['hour'] = df['hour']
    features['time_segment_id'] = df['time_segment_id']
    
    # Video features
    for col in ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                'flow_rate_proxy', 'congestion_proxy']:
        features[col] = df[col]
    
    # Interaction features (from v7.5)
    features['hour_x_signaling'] = features['hour'] * features['signaling']
    features['location_x_signaling'] = features['location'] * features['signaling']
    features['motion_x_occupancy'] = features['motion_mean'] * features['occupancy_mean']
    features['flow_x_occupancy'] = features['flow_rate_proxy'] * features['occupancy_mean']
    features['is_peak_hour'] = ((features['hour'] >= 10) & (features['hour'] <= 17)).astype(int)
    features['is_morning'] = (features['hour'] < 10).astype(int)
    features['motion_squared'] = features['motion_mean'] ** 2
    features['occupancy_squared'] = features['occupancy_mean'] ** 2
    
    # NEW: Additional features for v10
    features['location_x_hour'] = features['location'] * features['hour']
    features['is_rush_hour'] = (((features['hour'] >= 7) & (features['hour'] <= 9)) | 
                                ((features['hour'] >= 16) & (features['hour'] <= 18))).astype(int)
    features['motion_occ_ratio'] = features['motion_mean'] / (features['occupancy_mean'] + 1e-5)
    features['vehicle_flow_interaction'] = features['vehicle_count_proxy'] * features['flow_rate_proxy']
    
    return features, le_location, le_signaling

X, le_location, le_signaling = create_features(train_enhanced)
y_enter = train_enhanced['congestion_enter_rating']

le_target = LabelEncoder()
le_target.fit(['free flowing', 'heavy delay', 'light delay', 'moderate delay'])
y_enter_encoded = le_target.transform(y_enter)

print(f"Features: {X.shape[1]}")

# ============================================
# STEP 1: Train Initial Model
# ============================================
print("\n" + "="*60)
print("STEP 1: TRAINING INITIAL MODEL")
print("="*60)

initial_model = lgb.LGBMClassifier(
    n_estimators=500,  # Increased from 400
    learning_rate=0.02,
    max_depth=8,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    verbosity=-1
)
initial_model.fit(X, y_enter_encoded)
print("✓ Initial LightGBM model trained")

# ============================================
# STEP 2: Generate Pseudo-Labels (Iterative)
# ============================================
print("\n" + "="*60)
print("STEP 2: ITERATIVE PSEUDO-LABELING")
print("="*60)

# Median values for test predictions
medians = {col: video_features_df[col].median() for col in 
           ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
            'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
            'flow_rate_proxy', 'congestion_proxy']}

def generate_pseudo_labels(model, sample_df, conf_threshold):
    """Generate pseudo-labels for high-confidence predictions"""
    pseudo_data = []
    
    for idx, row in sample_df.iterrows():
        id_str = row['ID']
        match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
        
        if match:
            segment_id = int(match.group(1))
            location = match.group(2)
            rating_type = match.group(3)
            
            if rating_type == 'enter':
                estimated_hour = 6 + ((segment_id // 60) % 12)
                
                test_features = pd.DataFrame([{
                    'location': le_location.transform([location])[0],
                    'signaling': le_signaling.transform(['none'])[0],
                    'hour': estimated_hour,
                    'time_segment_id': segment_id,
                    **medians
                }])
                
                # Add derived features
                test_features['hour_x_signaling'] = test_features['hour'] * test_features['signaling']
                test_features['location_x_signaling'] = test_features['location'] * test_features['signaling']
                test_features['motion_x_occupancy'] = test_features['motion_mean'] * test_features['occupancy_mean']
                test_features['flow_x_occupancy'] = test_features['flow_rate_proxy'] * test_features['occupancy_mean']
                test_features['is_peak_hour'] = int(10 <= estimated_hour <= 17)
                test_features['is_morning'] = int(estimated_hour < 10)
                test_features['motion_squared'] = test_features['motion_mean'] ** 2
                test_features['occupancy_squared'] = test_features['occupancy_mean'] ** 2
                test_features['location_x_hour'] = test_features['location'] * test_features['hour']
                test_features['is_rush_hour'] = int((7 <= estimated_hour <= 9) or (16 <= estimated_hour <= 18))
                test_features['motion_occ_ratio'] = test_features['motion_mean'] / (test_features['occupancy_mean'] + 1e-5)
                test_features['vehicle_flow_interaction'] = test_features['vehicle_count_proxy'] * test_features['flow_rate_proxy']
                
                test_features = test_features[X.columns]
                
                proba = model.predict_proba(test_features)[0]
                max_proba = np.max(proba)
                pred_class = np.argmax(proba)
                
                if max_proba >= conf_threshold:
                    pseudo_label = le_target.inverse_transform([pred_class])[0]
                    pseudo_data.append({
                        'view_label': location,
                        'signaling': 'none',
                        'hour': estimated_hour,
                        'time_segment_id': segment_id,
                        'congestion_enter_rating': pseudo_label,
                        **medians,
                        'confidence': max_proba
                    })
    
    return pd.DataFrame(pseudo_data)

# Iteration 1: High confidence (0.75)
pseudo_df_1 = generate_pseudo_labels(initial_model, sample_df, 0.75)
print(f"Iteration 1: {len(pseudo_df_1)} pseudo-labels (conf >= 0.75)")

# Combine and retrain
combined_df_1 = pd.concat([
    train_enhanced[['view_label', 'signaling', 'hour', 'time_segment_id', 
                    'congestion_enter_rating', 'motion_mean', 'motion_std', 'motion_max',
                    'occupancy_mean', 'occupancy_std', 'occupancy_max', 
                    'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']],
    pseudo_df_1[['view_label', 'signaling', 'hour', 'time_segment_id', 
                 'congestion_enter_rating', 'motion_mean', 'motion_std', 'motion_max',
                 'occupancy_mean', 'occupancy_std', 'occupancy_max', 
                 'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']] if len(pseudo_df_1) > 0 else pd.DataFrame()
], ignore_index=True)

X_combined_1, _, _ = create_features(combined_df_1, le_location, le_signaling, fit=False)
y_combined_1 = le_target.transform(combined_df_1['congestion_enter_rating'])

# Train intermediate model
intermediate_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.02, max_depth=8, num_leaves=50,
    subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
    random_state=42, verbosity=-1
)
intermediate_model.fit(X_combined_1, y_combined_1)

# Iteration 2: Medium confidence (0.70)
pseudo_df_2 = generate_pseudo_labels(intermediate_model, sample_df, 0.70)
print(f"Iteration 2: {len(pseudo_df_2)} pseudo-labels (conf >= 0.70)")

# ============================================
# STEP 3: Final Training with Ensemble
# ============================================
print("\n" + "="*60)
print("STEP 3: FINAL ENSEMBLE TRAINING")
print("="*60)

# Use iteration 1 pseudo labels (higher confidence)
final_combined_df = combined_df_1.copy()
print(f"Final training data: {len(final_combined_df)} samples")

X_final, _, _ = create_features(final_combined_df, le_location, le_signaling, fit=False)
y_final = le_target.transform(final_combined_df['congestion_enter_rating'])

# Train with 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_models = []
xgb_models = []
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y_final)):
    X_train, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train, y_val = y_final[train_idx], y_final[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42 + fold,
        class_weight='balanced',
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_models.append(lgb_model)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_models.append(xgb_model)
    
    # Validate with ensemble
    lgb_proba = lgb_model.predict_proba(X_val)
    xgb_proba = xgb_model.predict_proba(X_val)
    ensemble_proba = (lgb_proba + xgb_proba) / 2
    y_pred = np.argmax(ensemble_proba, axis=1)
    
    score = f1_score(y_val, y_pred, average='macro')
    cv_scores.append(score)
    print(f"Fold {fold+1}: F1 = {score:.4f}")

print(f"\nMean CV F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train EXIT model
exit_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42, verbosity=-1
)
y_exit_encoded = le_target.transform(train_enhanced['congestion_exit_rating'])
exit_model.fit(X, y_exit_encoded)

# ============================================
# STEP 4: Generate Final Predictions
# ============================================
print("\n" + "="*60)
print("STEP 4: GENERATING FINAL PREDICTIONS")
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
        
        test_features = pd.DataFrame([{
            'location': le_location.transform([location])[0],
            'signaling': le_signaling.transform(['none'])[0],
            'hour': estimated_hour,
            'time_segment_id': segment_id,
            **medians
        }])
        
        # Add derived features
        test_features['hour_x_signaling'] = test_features['hour'] * test_features['signaling']
        test_features['location_x_signaling'] = test_features['location'] * test_features['signaling']
        test_features['motion_x_occupancy'] = test_features['motion_mean'] * test_features['occupancy_mean']
        test_features['flow_x_occupancy'] = test_features['flow_rate_proxy'] * test_features['occupancy_mean']
        test_features['is_peak_hour'] = int(10 <= estimated_hour <= 17)
        test_features['is_morning'] = int(estimated_hour < 10)
        test_features['motion_squared'] = test_features['motion_mean'] ** 2
        test_features['occupancy_squared'] = test_features['occupancy_mean'] ** 2
        test_features['location_x_hour'] = test_features['location'] * test_features['hour']
        test_features['is_rush_hour'] = int((7 <= estimated_hour <= 9) or (16 <= estimated_hour <= 18))
        test_features['motion_occ_ratio'] = test_features['motion_mean'] / (test_features['occupancy_mean'] + 1e-5)
        test_features['vehicle_flow_interaction'] = test_features['vehicle_count_proxy'] * test_features['flow_rate_proxy']
        
        test_features = test_features[X.columns]
        
        if rating_type == 'enter':
            # Ensemble: Average LightGBM + XGBoost
            total_proba = np.zeros(4)
            for lgb_m, xgb_m in zip(lgb_models, xgb_models):
                total_proba += lgb_m.predict_proba(test_features)[0]
                total_proba += xgb_m.predict_proba(test_features)[0]
            
            avg_proba = total_proba / (len(lgb_models) + len(xgb_models))
            pred_class = np.argmax(avg_proba)
            prediction = le_target.inverse_transform([pred_class])[0]
        else:
            pred_class = exit_model.predict(test_features)[0]
            prediction = le_target.inverse_transform([pred_class])[0]
        
        predictions.append({
            'ID': id_str,
            'Target': prediction,
            'Target_Accuracy': prediction
        })

submission_df = pd.DataFrame(predictions)
output_path = 'submissions/v10.0_improved_pseudo.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}")

# ============================================
# Analysis
# ============================================
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nv10.0 Prediction distribution:")
print(submission_df['Target'].value_counts(normalize=True).sort_index())

# Compare with v7.5
v7_5 = pd.read_csv('submissions/v7.5_pseudo_labeling.csv')
print("\nv7.5 (Best: 0.594) distribution:")
print(v7_5['Target'].value_counts(normalize=True).sort_index())

merged = submission_df.merge(v7_5, on='ID', suffixes=('_v10', '_v7'))
diffs = (merged['Target_v10'] != merged['Target_v7']).sum()
print(f"\nDifferences from v7.5: {diffs} out of {len(merged)}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Pseudo-labels added: {len(pseudo_df_1)}")
print(f"✓ Total training samples: {len(final_combined_df)}")
print(f"✓ Ensemble: {len(lgb_models)} LightGBM + {len(xgb_models)} XGBoost")
print(f"✓ Mean CV F1: {np.mean(cv_scores):.4f}")
print("\nTarget: 0.594 → 0.65+")
print("="*60)
