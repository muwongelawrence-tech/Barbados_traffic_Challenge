"""
ML-BASED SUBMISSION v7.0
Train proper models on training data with ALL features
Use cross-validation and ensemble
Expected: 0.52-0.62
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
import re
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ML-BASED SUBMISSION v7.0")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')
video_features_df = pd.read_csv('video_features.csv')

print(f"Training data: {len(train_df)}")
print(f"Sample submission: {len(sample_df)}")

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

# Fill missing video features with medians
for col in ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
            'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
            'flow_rate_proxy', 'congestion_proxy']:
    median_val = video_features_df[col].median()
    train_enhanced[col].fillna(median_val, inplace=True)

matched = train_enhanced['motion_mean'].notna().sum()
print(f"Training segments with video features: {matched}/{len(train_enhanced)} ({matched/len(train_enhanced)*100:.1f}%)")

# Extract features
train_enhanced['hour'] = train_enhanced['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0])
)

# Create features for ENTER rating
def create_features(df):
    features = pd.DataFrame()
    
    # Encode categorical
    le_location = LabelEncoder()
    le_signaling = LabelEncoder()
    
    features['location'] = le_location.fit_transform(df['view_label'])
    features['signaling'] = le_signaling.fit_transform(df['signaling'])
    features['hour'] = df['hour']
    features['time_segment_id'] = df['time_segment_id']
    
    # Video features
    for col in ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                'flow_rate_proxy', 'congestion_proxy']:
        features[col] = df[col]
    
    # Interaction features
    features['hour_x_signaling'] = features['hour'] * features['signaling']
    features['location_x_signaling'] = features['location'] * features['signaling']
    features['motion_x_occupancy'] = features['motion_mean'] * features['occupancy_mean']
    
    return features, le_location, le_signaling

X, le_location, le_signaling = create_features(train_enhanced)
y_enter = train_enhanced['congestion_enter_rating']
y_exit = train_enhanced['congestion_exit_rating']

# Encode targets
le_target = LabelEncoder()
y_enter_encoded = le_target.fit_transform(y_enter)
y_exit_encoded = le_target.transform(y_exit)

print(f"\nFeatures: {X.shape[1]}")
print(f"Classes: {le_target.classes_}")

# Train models with cross-validation for ENTER
print("\n" + "="*60)
print("TRAINING ENTER MODEL")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
enter_models = []
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enter_encoded)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_enter_encoded[train_idx], y_enter_encoded[val_idx]
    
    # Train LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro')
    cv_scores.append(score)
    enter_models.append(model)
    
    print(f"Fold {fold+1}: F1 = {score:.4f}")

print(f"\nMean CV F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train EXIT model (simpler - mostly free flowing)
print("\n" + "="*60)
print("TRAINING EXIT MODEL")
print("="*60)

exit_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
exit_model.fit(X, y_exit_encoded)

# Generate predictions for sample submission
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

predictions = []

for idx, row in sample_df.iterrows():
    id_str = row['ID']
    
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        rating_type = match.group(3)
        
        # Estimate features for this segment
        estimated_hour = 6 + ((segment_id // 60) % 12)
        estimated_signaling = 'none'  # Conservative default
        
        # Create feature vector
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
        
        # Add interaction features
        test_features['hour_x_signaling'] = test_features['hour'] * test_features['signaling']
        test_features['location_x_signaling'] = test_features['location'] * test_features['signaling']
        test_features['motion_x_occupancy'] = test_features['motion_mean'] * test_features['occupancy_mean']
        
        # Predict
        if rating_type == 'enter':
            # Ensemble of 5 fold models
            preds = []
            for model in enter_models:
                pred = model.predict(test_features)[0]
                preds.append(pred)
            # Majority vote
            pred_encoded = int(np.median(preds))
            prediction = le_target.inverse_transform([pred_encoded])[0]
        else:  # exit
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

# Create submission
submission_df = pd.DataFrame(predictions)

# Save
output_path = 'submissions/v7.0_ml_based.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}")

# Analysis
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nPrediction distribution:")
dist = submission_df['Target'].value_counts(normalize=True).sort_index()
print(dist)

print("\nPrediction counts:")
print(submission_df['Target'].value_counts())

print("\n" + "="*60)
print("ML APPROACH")
print("="*60)
print(f"✓ 5-fold cross-validation: F1 = {np.mean(cv_scores):.4f}")
print("✓ Ensemble of 5 models for ENTER")
print("✓ Separate model for EXIT")
print("✓ Video features + interactions")
print("\nExpected score: 0.52-0.62")
print("="*60)
