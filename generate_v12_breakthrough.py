"""
BREAKTHROUGH v12.0 - Using Test Input Segments as Features
KEY INSIGHT: Sample predictions are offset by 7-17 segments from test inputs!
- Sample segment 129 uses test input segment 112 (offset ~17)
- Test input has ACTUAL congestion ratings we can use as features!

Strategy:
1. For each prediction segment N, find test input segment N-offset
2. Use the ACTUAL congestion rating from test input as primary feature
3. Build transition model: current_state -> future_state

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
print("BREAKTHROUGH v12.0 - Test Input Feature Strategy")  
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')
video_features_df = pd.read_csv('video_features.csv')
test_input_df = pd.read_csv('TestInputSegments.csv')

print(f"Training data: {len(train_df)}")
print(f"Test input segments: {len(test_input_df)}")
print(f"Predictions needed: {len(sample_df)}")

# ============================================
# Build Test Input Lookup
# ============================================
print("\n" + "="*60)
print("BUILDING TEST INPUT LOOKUP")
print("="*60)

# Create lookup: (segment_id, location) -> congestion data
test_lookup = {}
for _, row in test_input_df.iterrows():
    key = (row['time_segment_id'], row['view_label'])
    
    # Extract hour
    try:
        hour = int(str(row['datetimestamp_start']).split()[1].split(':')[0])
    except:
        hour = 12
    
    test_lookup[key] = {
        'enter_rating': row['congestion_enter_rating'],
        'exit_rating': row['congestion_exit_rating'],
        'hour': hour,
        'signaling': row['signaling'] if pd.notna(row['signaling']) else 'none'
    }

print(f"✓ Built lookup with {len(test_lookup)} test segments")

# ============================================
# Create Training Data with State Transitions
# ============================================
print("\n" + "="*60)
print("CREATING TRANSITION TRAINING DATA")
print("="*60)

# In training data, we can model transitions:
# Current congestion -> Next segment's congestion

# Extract hour from training data
train_df['hour'] = train_df['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0]) if len(str(x).split()) > 1 else 12
)

# Encode ratings
le_rating = LabelEncoder()
le_rating.fit(['free flowing', 'heavy delay', 'light delay', 'moderate delay'])
le_location = LabelEncoder()
le_location.fit(['Norman Niles #1', 'Norman Niles #2', 'Norman Niles #3', 'Norman Niles #4'])

# Create transition features from training data
# For each segment, look at previous segment's rating
features_list = []
targets_list = []

# Sort by location and time_segment_id
for location in train_df['view_label'].unique():
    loc_data = train_df[train_df['view_label'] == location].sort_values('time_segment_id')
    
    for i in range(1, len(loc_data)):
        prev_row = loc_data.iloc[i-1]
        curr_row = loc_data.iloc[i]
        
        # Only use if segments are close (within 20 segments)
        if curr_row['time_segment_id'] - prev_row['time_segment_id'] <= 20:
            features_list.append({
                'location': le_location.transform([location])[0],
                'hour': curr_row['hour'],
                'prev_enter_rating': le_rating.transform([prev_row['congestion_enter_rating']])[0],
                'prev_exit_rating': le_rating.transform([prev_row['congestion_exit_rating']])[0],
                'segment_diff': curr_row['time_segment_id'] - prev_row['time_segment_id'],
                'is_peak': int(10 <= curr_row['hour'] <= 17),
                'is_morning': int(curr_row['hour'] < 10),
            })
            targets_list.append(curr_row['congestion_enter_rating'])

X_transition = pd.DataFrame(features_list)
y_transition = le_rating.transform(targets_list)

print(f"Transition training samples: {len(X_transition)}")
print(f"Features: {list(X_transition.columns)}")

# ============================================
# Train Transition Model
# ============================================
print("\n" + "="*60)
print("TRAINING TRANSITION MODEL")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
transition_models = []
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_transition, y_transition)):
    X_train, X_val = X_transition.iloc[train_idx], X_transition.iloc[val_idx]
    y_train, y_val = y_transition[train_idx], y_transition[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42 + fold,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    transition_models.append(model)
    
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro')
    cv_scores.append(score)
    print(f"Fold {fold+1}: F1 = {score:.4f}")

print(f"\nMean CV F1: {np.mean(cv_scores):.4f}")

# ============================================
# Generate Predictions Using Test Input Context
# ============================================
print("\n" + "="*60)
print("GENERATING PREDICTIONS WITH TEST INPUT CONTEXT")
print("="*60)

def parse_id(id_str):
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    if match:
        return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None

predictions = []
used_test_input = 0
fallback_count = 0

for _, row in sample_df.iterrows():
    id_str = row['ID']
    segment_id, location, rating_type = parse_id(id_str)
    
    if segment_id and location:
        # Try to find relevant test input segment
        # Look back with offsets 7-17 (where we found 100% match)
        test_context = None
        for offset in range(7, 18):
            key = (segment_id - offset, location)
            if key in test_lookup:
                test_context = test_lookup[key]
                break
        
        if test_context:
            used_test_input += 1
            
            # Create features using test input context
            test_features = pd.DataFrame([{
                'location': le_location.transform([location])[0],
                'hour': test_context['hour'],
                'prev_enter_rating': le_rating.transform([test_context['enter_rating']])[0],
                'prev_exit_rating': le_rating.transform([test_context['exit_rating']])[0],
                'segment_diff': 10,  # Average offset
                'is_peak': int(10 <= test_context['hour'] <= 17),
                'is_morning': int(test_context['hour'] < 10),
            }])
            
            if rating_type == 'enter':
                # Ensemble prediction
                proba_sum = np.zeros(4)
                for model in transition_models:
                    proba_sum += model.predict_proba(test_features)[0]
                avg_proba = proba_sum / len(transition_models)
                pred_class = np.argmax(avg_proba)
                prediction = le_rating.inverse_transform([pred_class])[0]
            else:
                # For exit, use high free flowing (95%+ in training)
                prediction = 'free flowing'
        else:
            fallback_count += 1
            # Fallback: use most common class
            estimated_hour = 6 + ((segment_id // 60) % 12)
            if estimated_hour < 10:
                prediction = 'free flowing'
            else:
                prediction = 'free flowing'  # Most common
        
        predictions.append({
            'ID': id_str,
            'Target': prediction,
            'Target_Accuracy': prediction
        })

print(f"\n✓ Used test input context: {used_test_input}/{len(sample_df)}")
print(f"  Fallback predictions: {fallback_count}")

# Create submission
submission_df = pd.DataFrame(predictions)

# ============================================
# Analysis
# ============================================
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nv12.0 Distribution:")
print(submission_df['Target'].value_counts(normalize=True).sort_index())

# Compare with v7.5
v7_5 = pd.read_csv('submissions/v7.5_pseudo_labeling.csv')
print("\nv7.5 (0.594) Distribution:")
print(v7_5['Target'].value_counts(normalize=True).sort_index())

# Count differences
merged = submission_df.merge(v7_5, on='ID', suffixes=('_v12', '_v75'))
diff_count = (merged['Target_v12'] != merged['Target_v75']).sum()
print(f"\nDifferences from v7.5: {diff_count}")

# Save
output_path = 'submissions/v12.0_test_input_features.csv'
submission_df.to_csv(output_path, index=False)
print(f"\n✓ Saved: {output_path}")

print("\n" + "="*60)
print(f"Target: 0.594 → 0.65+ (using actual test observations!)")
print("="*60)
