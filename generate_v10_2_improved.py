"""
IMPROVED v10.2 - Building on v7.5's verified success
Changes from v7.5:
1. More estimators (500 vs 400)
2. Lower learning rate (0.015 vs 0.02)
3. Temperature-based probability smoothing
4. Slightly lower confidence for pseudo-labels (0.72 vs 0.75)
5. Additional interaction features

Target: 0.594 → 0.65+
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
import re
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("IMPROVED v10.2 - Targeted Improvements")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')
video_features_df = pd.read_csv('video_features.csv')

# Feature preparation (same as v7.5)
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

train_df['timestamp_simple'] = train_df['datetimestamp_start'].apply(simplify_timestamp)
train_df['match_key'] = train_df['view_label'] + '|' + train_df['timestamp_simple'].fillna('')

video_cols = ['match_key', 'motion_mean', 'motion_std', 'motion_max',
              'occupancy_mean', 'occupancy_std', 'occupancy_max',
              'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']

train_enhanced = train_df.merge(video_features_df[video_cols], on='match_key', how='left')

for col in ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
            'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
            'flow_rate_proxy', 'congestion_proxy']:
    median_val = video_features_df[col].median()
    train_enhanced[col].fillna(median_val, inplace=True)

train_enhanced['hour'] = train_enhanced['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0]) if len(str(x).split()) > 1 else 12
)

# ENHANCED Feature engineering (v7.5 + improvements)
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
    
    # Original v7.5 features
    features['hour_x_signaling'] = features['hour'] * features['signaling']
    features['location_x_signaling'] = features['location'] * features['signaling']
    features['motion_x_occupancy'] = features['motion_mean'] * features['occupancy_mean']
    features['flow_x_occupancy'] = features['flow_rate_proxy'] * features['occupancy_mean']
    features['is_peak_hour'] = ((features['hour'] >= 10) & (features['hour'] <= 17)).astype(int)
    features['is_morning'] = (features['hour'] < 10).astype(int)
    features['motion_squared'] = features['motion_mean'] ** 2
    features['occupancy_squared'] = features['occupancy_mean'] ** 2
    
    # NEW v10.2 features
    features['location_x_hour'] = features['location'] * features['hour']
    features['motion_cv'] = features['motion_std'] / (features['motion_mean'] + 1e-5)  # Coefficient of variation
    features['occupancy_range'] = features['occupancy_max'] - features['occupancy_mean']
    features['is_rush_hour'] = (((features['hour'] >= 7) & (features['hour'] <= 9)) | 
                                ((features['hour'] >= 16) & (features['hour'] <= 18))).astype(int)
    
    return features, le_location, le_signaling

X, le_location, le_signaling = create_features(train_enhanced)
y_enter = train_enhanced['congestion_enter_rating']

le_target = LabelEncoder()
y_enter_encoded = le_target.fit_transform(y_enter)

print(f"Training: {len(X)} samples, {X.shape[1]} features")

# Train initial model with IMPROVED hyperparameters
print("\n" + "="*60)
print("STEP 1: TRAINING IMPROVED INITIAL MODEL")
print("="*60)

initial_model = lgb.LGBMClassifier(
    n_estimators=500,      # Increased from 400
    learning_rate=0.015,   # Reduced from 0.02 for better generalization
    max_depth=7,           # Slightly reduced from 8
    num_leaves=40,         # Reduced from 50
    subsample=0.85,        # Increased from 0.8
    colsample_bytree=0.85,
    reg_alpha=0.15,        # Increased regularization
    reg_lambda=0.15,
    random_state=42,
    class_weight='balanced',
    verbosity=-1
)
initial_model.fit(X, y_enter_encoded)
print("✓ Initial model trained")

# Generate pseudo-labels with SLIGHTLY LOWER threshold
print("\n" + "="*60)
print("STEP 2: PSEUDO-LABELING (threshold=0.72)")
print("="*60)

confidence_threshold = 0.72  # Lowered from 0.75 for more pseudo-labels
pseudo_data = []

medians = {col: video_features_df[col].median() for col in 
           ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
            'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
            'flow_rate_proxy', 'congestion_proxy']}

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
            
            test_features['hour_x_signaling'] = test_features['hour'] * test_features['signaling']
            test_features['location_x_signaling'] = test_features['location'] * test_features['signaling']
            test_features['motion_x_occupancy'] = test_features['motion_mean'] * test_features['occupancy_mean']
            test_features['flow_x_occupancy'] = test_features['flow_rate_proxy'] * test_features['occupancy_mean']
            test_features['is_peak_hour'] = int(10 <= estimated_hour <= 17)
            test_features['is_morning'] = int(estimated_hour < 10)
            test_features['motion_squared'] = test_features['motion_mean'] ** 2
            test_features['occupancy_squared'] = test_features['occupancy_mean'] ** 2
            test_features['location_x_hour'] = test_features['location'] * test_features['hour']
            test_features['motion_cv'] = test_features['motion_std'] / (test_features['motion_mean'] + 1e-5)
            test_features['occupancy_range'] = test_features['occupancy_max'] - test_features['occupancy_mean']
            test_features['is_rush_hour'] = int((7 <= estimated_hour <= 9) or (16 <= estimated_hour <= 18))
            test_features = test_features[X.columns]
            
            proba = initial_model.predict_proba(test_features)[0]
            max_proba = np.max(proba)
            pred_class = np.argmax(proba)
            
            if max_proba >= confidence_threshold:
                pseudo_label = le_target.inverse_transform([pred_class])[0]
                pseudo_data.append({
                    'view_label': location,
                    'signaling': 'none',
                    'hour': estimated_hour,
                    'time_segment_id': segment_id,
                    'congestion_enter_rating': pseudo_label,
                    **medians
                })

pseudo_df = pd.DataFrame(pseudo_data)
print(f"✓ Generated {len(pseudo_df)} pseudo-labels")

if len(pseudo_df) > 0:
    print("\nPseudo-label distribution:")
    print(pseudo_df['congestion_enter_rating'].value_counts(normalize=True).sort_index())

# Combine and train
print("\n" + "="*60)
print("STEP 3: TRAINING WITH PSEUDO-LABELS")
print("="*60)

combined_df = pd.concat([
    train_enhanced[['view_label', 'signaling', 'hour', 'time_segment_id', 
                    'congestion_enter_rating', 'motion_mean', 'motion_std', 'motion_max',
                    'occupancy_mean', 'occupancy_std', 'occupancy_max', 
                    'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']],
    pseudo_df[['view_label', 'signaling', 'hour', 'time_segment_id', 
               'congestion_enter_rating', 'motion_mean', 'motion_std', 'motion_max',
               'occupancy_mean', 'occupancy_std', 'occupancy_max', 
               'vehicle_count_proxy', 'flow_rate_proxy', 'congestion_proxy']] if len(pseudo_df) > 0 else pd.DataFrame()
], ignore_index=True)

print(f"Combined: {len(combined_df)} samples ({len(train_enhanced)} + {len(pseudo_df)} pseudo)")

X_combined, le_loc_new, le_sig_new = create_features(combined_df)
y_combined = combined_df['congestion_enter_rating']
y_combined_encoded = le_target.fit_transform(y_combined)

# 5-fold CV with improved hyperparameters
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_models = []
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined_encoded)):
    X_train, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
    y_train, y_val = y_combined_encoded[train_idx], y_combined_encoded[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.015,
        max_depth=7,
        num_leaves=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=0.15,
        random_state=42 + fold,
        class_weight='balanced',
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro')
    cv_scores.append(score)
    final_models.append(model)
    print(f"Fold {fold+1}: F1 = {score:.4f}")

print(f"\nMean CV F1: {np.mean(cv_scores):.4f}")

# Exit model
exit_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.02, max_depth=6, random_state=42, verbosity=-1
)
y_exit_encoded = le_target.transform(train_enhanced['congestion_exit_rating'])
exit_model.fit(X, y_exit_encoded)

# Generate predictions with temperature scaling
print("\n" + "="*60)
print("STEP 4: PREDICTIONS WITH TEMPERATURE SCALING")
print("="*60)

# Temperature for calibration (>1 softens probabilities)
temperature = 1.2

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
            'location': le_loc_new.transform([location])[0],
            'signaling': le_sig_new.transform(['none'])[0],
            'hour': estimated_hour,
            'time_segment_id': segment_id,
            **medians
        }])
        
        test_features['hour_x_signaling'] = test_features['hour'] * test_features['signaling']
        test_features['location_x_signaling'] = test_features['location'] * test_features['signaling']
        test_features['motion_x_occupancy'] = test_features['motion_mean'] * test_features['occupancy_mean']
        test_features['flow_x_occupancy'] = test_features['flow_rate_proxy'] * test_features['occupancy_mean']
        test_features['is_peak_hour'] = int(10 <= estimated_hour <= 17)
        test_features['is_morning'] = int(estimated_hour < 10)
        test_features['motion_squared'] = test_features['motion_mean'] ** 2
        test_features['occupancy_squared'] = test_features['occupancy_mean'] ** 2
        test_features['location_x_hour'] = test_features['location'] * test_features['hour']
        test_features['motion_cv'] = test_features['motion_std'] / (test_features['motion_mean'] + 1e-5)
        test_features['occupancy_range'] = test_features['occupancy_max'] - test_features['occupancy_mean']
        test_features['is_rush_hour'] = int((7 <= estimated_hour <= 9) or (16 <= estimated_hour <= 18))
        test_features = test_features[X.columns]
        
        if rating_type == 'enter':
            proba_sum = np.zeros(len(le_target.classes_))
            for model in final_models:
                raw_proba = model.predict_proba(test_features)[0]
                # Apply temperature scaling
                scaled_proba = np.exp(np.log(raw_proba + 1e-10) / temperature)
                scaled_proba = scaled_proba / scaled_proba.sum()
                proba_sum += scaled_proba
            
            avg_proba = proba_sum / len(final_models)
            pred_encoded = np.argmax(avg_proba)
            prediction = le_target.inverse_transform([pred_encoded])[0]
        else:
            pred_encoded = exit_model.predict(test_features)[0]
            prediction = le_target.inverse_transform([pred_encoded])[0]
        
        predictions.append({
            'ID': id_str,
            'Target': prediction,
            'Target_Accuracy': prediction
        })

submission_df = pd.DataFrame(predictions)
output_path = 'submissions/v10.2_improved.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")

# Compare distributions
print("\n" + "="*60)
print("DISTRIBUTION COMPARISON")
print("="*60)

print("\nv10.2 (this run):")
print(submission_df['Target'].value_counts(normalize=True).sort_index())

v7_5 = pd.read_csv('submissions/v7.5_pseudo_labeling.csv')
print("\nv7.5 (best: 0.594):")
print(v7_5['Target'].value_counts(normalize=True).sort_index())

# Count differences
merged = submission_df.merge(v7_5, on='ID', suffixes=('_new', '_old'))
diff_count = (merged['Target_new'] != merged['Target_old']).sum()
print(f"\nDifferences from v7.5: {diff_count} predictions ({100*diff_count/len(merged):.1f}%)")

# Show what changed
if diff_count > 0:
    diffs = merged[merged['Target_new'] != merged['Target_old']]
    print("\nChange breakdown:")
    for old_val in ['free flowing', 'heavy delay', 'light delay', 'moderate delay']:
        for new_val in ['free flowing', 'heavy delay', 'light delay', 'moderate delay']:
            if old_val != new_val:
                count = ((diffs['Target_old'] == old_val) & (diffs['Target_new'] == new_val)).sum()
                if count > 0:
                    print(f"  {old_val} -> {new_val}: {count}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ New features: 4 additional")
print(f"✓ Pseudo-labels: {len(pseudo_df)} (vs 395 in v7.5)")
print(f"✓ CV F1: {np.mean(cv_scores):.4f}")
print(f"✓ Changes from v7.5: {diff_count}")
print("\nTarget: 0.594 → 0.62+")
print("="*60)
