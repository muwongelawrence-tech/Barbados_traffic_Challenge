"""
BREAKTHROUGH v11.0 - New Strategy for 0.60+
Current best: 0.5947 (v7.5)

Key insight: The plateau indicates we need fundamentally different approaches:
1. Location-specific models (4 Norman Niles locations have different patterns)
2. Use TestInputSegments for actual context (not just medians)
3. Time-segment pattern analysis
4. Weighted ensemble based on segment characteristics

Target: 0.594 → 0.62+
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
print("BREAKTHROUGH v11.0 - Location-Specific Strategy")
print("="*60)

# Load all data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')
video_features_df = pd.read_csv('video_features.csv')
test_input_df = pd.read_csv('TestInputSegments.csv')

print(f"Training data: {len(train_df)}")
print(f"Test input segments: {len(test_input_df)}")

# ============================================
# ANALYZE LOCATION-SPECIFIC PATTERNS
# ============================================
print("\n" + "="*60)
print("STEP 1: LOCATION-SPECIFIC PATTERN ANALYSIS")
print("="*60)

for location in train_df['view_label'].unique():
    loc_data = train_df[train_df['view_label'] == location]
    dist = loc_data['congestion_enter_rating'].value_counts(normalize=True)
    print(f"\n{location}:")
    for cls in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        print(f"  {cls}: {dist.get(cls, 0):.1%}")

# ============================================
# PREPARE FEATURES
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

# Process video features
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

# ============================================
# BUILD TEST INPUT LOOKUP
# ============================================
print("\n" + "="*60)
print("STEP 2: BUILDING TEST CONTEXT LOOKUP")
print("="*60)

# Create lookup from test input segments
test_lookup = {}
for idx, row in test_input_df.iterrows():
    seg_id = row['time_segment_id']
    loc = row['view_label']
    key = f"{seg_id}_{loc}"
    test_lookup[key] = {
        'current_enter': row['congestion_enter_rating'],
        'current_exit': row['congestion_exit_rating'],
        'hour': int(str(row['datetimestamp_start']).split()[1].split(':')[0]) if len(str(row['datetimestamp_start']).split()) > 1 else 12
    }

print(f"✓ Built lookup with {len(test_lookup)} test segments")

# Sample test distribution
test_enter_dist = test_input_df['congestion_enter_rating'].value_counts(normalize=True)
print("\nTest input congestion distribution:")
print(test_enter_dist)

# ============================================
# ENHANCED FEATURE ENGINEERING
# ============================================
def create_features(df, le_location=None, le_signaling=None, fit=True):
    features = pd.DataFrame()
    
    if fit:
        le_location = LabelEncoder()
        le_signaling = LabelEncoder()
        le_location.fit(df['view_label'])
        le_signaling.fit(df['signaling'])
    
    features['location'] = le_location.transform(df['view_label'])
    features['signaling'] = le_signaling.transform(df['signaling'])
    features['hour'] = df['hour']
    features['time_segment_id'] = df['time_segment_id']
    
    for col in ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                'flow_rate_proxy', 'congestion_proxy']:
        features[col] = df[col]
    
    # Original features
    features['hour_x_signaling'] = features['hour'] * features['signaling']
    features['location_x_signaling'] = features['location'] * features['signaling']
    features['motion_x_occupancy'] = features['motion_mean'] * features['occupancy_mean']
    features['flow_x_occupancy'] = features['flow_rate_proxy'] * features['occupancy_mean']
    features['is_peak_hour'] = ((features['hour'] >= 10) & (features['hour'] <= 17)).astype(int)
    features['is_morning'] = (features['hour'] < 10).astype(int)
    features['motion_squared'] = features['motion_mean'] ** 2
    features['occupancy_squared'] = features['occupancy_mean'] ** 2
    
    # NEW: Location-specific hour patterns
    features['location_x_hour'] = features['location'] * features['hour']
    features['location_x_peak'] = features['location'] * features['is_peak_hour']
    
    # NEW: Segment patterns
    features['segment_mod_100'] = df['time_segment_id'] % 100
    features['segment_div_100'] = df['time_segment_id'] // 100
    
    return features, le_location, le_signaling

X, le_location, le_signaling = create_features(train_enhanced)
y_enter = train_enhanced['congestion_enter_rating']

le_target = LabelEncoder()
le_target.fit(['free flowing', 'heavy delay', 'light delay', 'moderate delay'])
y_enter_encoded = le_target.transform(y_enter)

print(f"\nFeatures: {X.shape[1]}")

# ============================================
# STEP 3: TRAIN LOCATION-SPECIFIC MODELS
# ============================================
print("\n" + "="*60)
print("STEP 3: TRAINING LOCATION-SPECIFIC MODELS")
print("="*60)

location_models = {}

for loc_idx, location in enumerate(train_df['view_label'].unique()):
    loc_mask = train_enhanced['view_label'] == location
    X_loc = X[loc_mask]
    y_loc = y_enter_encoded[loc_mask]
    
    if len(X_loc) < 100:
        continue
    
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        max_depth=7,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbosity=-1
    )
    model.fit(X_loc, y_loc)
    location_models[location] = model
    print(f"✓ Trained model for {location} ({len(X_loc)} samples)")

# ============================================
# STEP 4: TRAIN GLOBAL MODEL WITH PSEUDO-LABELS
# ============================================
print("\n" + "="*60)
print("STEP 4: GLOBAL MODEL WITH PSEUDO-LABELS")
print("="*60)

# Train initial global model
global_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=8,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    verbosity=-1
)
global_model.fit(X, y_enter_encoded)

# Generate pseudo-labels
medians = {col: video_features_df[col].median() for col in 
           ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
            'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
            'flow_rate_proxy', 'congestion_proxy']}

pseudo_data = []
confidence_threshold = 0.75

for idx, row in sample_df.iterrows():
    id_str = row['ID']
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        rating_type = match.group(3)
        
        if rating_type == 'enter':
            estimated_hour = 6 + ((segment_id // 60) % 12)
            
            # Try to get actual hour from test input
            lookup_key = f"{segment_id}_{location}"
            if lookup_key in test_lookup:
                estimated_hour = test_lookup[lookup_key]['hour']
            
            test_features = pd.DataFrame([{
                'location': le_location.transform([location])[0],
                'signaling': le_signaling.transform(['none'])[0],
                'hour': estimated_hour,
                'time_segment_id': segment_id,
                **medians,
                'hour_x_signaling': estimated_hour * 0,
                'location_x_signaling': le_location.transform([location])[0] * 0,
                'motion_x_occupancy': medians['motion_mean'] * medians['occupancy_mean'],
                'flow_x_occupancy': medians['flow_rate_proxy'] * medians['occupancy_mean'],
                'is_peak_hour': int(10 <= estimated_hour <= 17),
                'is_morning': int(estimated_hour < 10),
                'motion_squared': medians['motion_mean'] ** 2,
                'occupancy_squared': medians['occupancy_mean'] ** 2,
                'location_x_hour': le_location.transform([location])[0] * estimated_hour,
                'location_x_peak': le_location.transform([location])[0] * int(10 <= estimated_hour <= 17),
                'segment_mod_100': segment_id % 100,
                'segment_div_100': segment_id // 100
            }])
            test_features = test_features[X.columns]
            
            proba = global_model.predict_proba(test_features)[0]
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

# Combine and retrain
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

X_combined, le_loc_new, le_sig_new = create_features(combined_df)
y_combined = combined_df['congestion_enter_rating']
y_combined_encoded = le_target.fit_transform(y_combined)

# Train final ensemble with CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_models = []
xgb_models = []
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined_encoded)):
    X_train, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
    y_train, y_val = y_combined_encoded[train_idx], y_combined_encoded[val_idx]
    
    # LightGBM
    lgb_m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.02, max_depth=8, num_leaves=50,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
        random_state=42+fold, verbosity=-1
    )
    lgb_m.fit(X_train, y_train)
    lgb_models.append(lgb_m)
    
    # XGBoost
    xgb_m = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42+fold, use_label_encoder=False, eval_metric='mlogloss', verbosity=0
    )
    xgb_m.fit(X_train, y_train)
    xgb_models.append(xgb_m)
    
    # Ensemble validation
    lgb_proba = lgb_m.predict_proba(X_val)
    xgb_proba = xgb_m.predict_proba(X_val)
    ensemble_proba = 0.6 * lgb_proba + 0.4 * xgb_proba  # Weighted
    y_pred = np.argmax(ensemble_proba, axis=1)
    
    score = f1_score(y_val, y_pred, average='macro')
    cv_scores.append(score)
    print(f"Fold {fold+1}: F1 = {score:.4f}")

print(f"\nMean CV F1: {np.mean(cv_scores):.4f}")

# Exit model
exit_model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.03, max_depth=6, random_state=42, verbosity=-1
)
y_exit_encoded = le_target.transform(train_enhanced['congestion_exit_rating'])
exit_model.fit(X, y_exit_encoded)

# ============================================
# STEP 5: GENERATE PREDICTIONS WITH LOCATION WEIGHTING
# ============================================
print("\n" + "="*60)
print("STEP 5: WEIGHTED ENSEMBLE PREDICTIONS")
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
        
        # Get actual hour from test input if available
        lookup_key = f"{segment_id}_{location}"
        if lookup_key in test_lookup:
            estimated_hour = test_lookup[lookup_key]['hour']
        
        test_features = pd.DataFrame([{
            'location': le_loc_new.transform([location])[0],
            'signaling': le_sig_new.transform(['none'])[0],
            'hour': estimated_hour,
            'time_segment_id': segment_id,
            **medians,
            'hour_x_signaling': estimated_hour * 0,
            'location_x_signaling': le_loc_new.transform([location])[0] * 0,
            'motion_x_occupancy': medians['motion_mean'] * medians['occupancy_mean'],
            'flow_x_occupancy': medians['flow_rate_proxy'] * medians['occupancy_mean'],
            'is_peak_hour': int(10 <= estimated_hour <= 17),
            'is_morning': int(estimated_hour < 10),
            'motion_squared': medians['motion_mean'] ** 2,
            'occupancy_squared': medians['occupancy_mean'] ** 2,
            'location_x_hour': le_loc_new.transform([location])[0] * estimated_hour,
            'location_x_peak': le_loc_new.transform([location])[0] * int(10 <= estimated_hour <= 17),
            'segment_mod_100': segment_id % 100,
            'segment_div_100': segment_id // 100
        }])
        test_features = test_features[X.columns]
        
        if rating_type == 'enter':
            # Combine global ensemble + location model
            total_proba = np.zeros(4)
            
            # Global ensemble (weight: 0.7)
            for lgb_m, xgb_m in zip(lgb_models, xgb_models):
                total_proba += 0.6 * lgb_m.predict_proba(test_features)[0]
                total_proba += 0.4 * xgb_m.predict_proba(test_features)[0]
            global_proba = total_proba / len(lgb_models)
            
            # Location model (weight: 0.3)
            if location in location_models:
                loc_proba = location_models[location].predict_proba(test_features)[0]
                final_proba = 0.7 * global_proba + 0.3 * loc_proba
            else:
                final_proba = global_proba
            
            pred_class = np.argmax(final_proba)
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
output_path = 'submissions/v11.0_breakthrough.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")

# Compare with v7.5
print("\n" + "="*60)
print("COMPARISON WITH BEST (v7.5)")
print("="*60)

print("\nv11.0 Distribution:")
print(submission_df['Target'].value_counts(normalize=True).sort_index())

v7_5 = pd.read_csv('submissions/v7.5_pseudo_labeling.csv')
print("\nv7.5 (0.594) Distribution:")
print(v7_5['Target'].value_counts(normalize=True).sort_index())

merged = submission_df.merge(v7_5, on='ID', suffixes=('_v11', '_v75'))
diff_count = (merged['Target_v11'] != merged['Target_v75']).sum()
print(f"\nDifferences: {diff_count} predictions ({100*diff_count/len(merged):.1f}%)")

if diff_count > 0:
    print("\nChange summary:")
    diffs = merged[merged['Target_v11'] != merged['Target_v75']]
    for old in ['free flowing', 'heavy delay', 'light delay', 'moderate delay']:
        for new in ['free flowing', 'heavy delay', 'light delay', 'moderate delay']:
            if old != new:
                c = ((diffs['Target_v75'] == old) & (diffs['Target_v11'] == new)).sum()
                if c > 0:
                    print(f"  {old} -> {new}: {c}")

print("\n" + "="*60)
print(f"Target: 0.5947 → 0.62+")
print("="*60)
