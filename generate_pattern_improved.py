"""
Improved Pattern-Based Submission v3.4
Use actual pattern frequencies, not just most common
This should match training distribution better
"""
import pandas as pd
import numpy as np
import re
import os

print("="*60)
print("IMPROVED PATTERN-BASED SUBMISSION v3.4")
print("="*60)
print("Using actual pattern frequencies from training data")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')

print(f"\nData loaded:")
print(f"  Training: {len(train_df)} records")
print(f"  Sample: {len(sample_df)} predictions needed")

# Extract hour
train_df['hour'] = train_df['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0])
)

print(f"\nTraining data distribution:")
print(train_df['congestion_enter_rating'].value_counts(normalize=True))

# Build segment-to-hour mapping
segment_hour_map = {}
for seg_id in train_df['time_segment_id'].unique():
    seg_data = train_df[train_df['time_segment_id'] == seg_id]
    most_common_hour = seg_data['hour'].mode()[0] if len(seg_data) > 0 else 12
    segment_hour_map[seg_id] = most_common_hour

print(f"\n✓ Mapped {len(segment_hour_map)} segments to hours")

# Build pattern lookup with ACTUAL DISTRIBUTION
# Store all values, not just mode
print("\nLearning congestion patterns with distributions...")
pattern_lookup = {}

for location in train_df['view_label'].unique():
    for hour in range(6, 18):
        # Enter ratings
        enter_data = train_df[(train_df['view_label'] == location) & (train_df['hour'] == hour)]
        if len(enter_data) >= 5:  # Need enough data
            # Store actual distribution
            enter_values = enter_data['congestion_enter_rating'].tolist()
            pattern_lookup[(location, hour, 'enter')] = enter_values
        
        # Exit ratings
        if len(enter_data) >= 5:
            exit_values = enter_data['congestion_exit_rating'].tolist()
            pattern_lookup[(location, hour, 'exit')] = exit_values

print(f"✓ Learned {len(pattern_lookup)} pattern distributions")

# Generate predictions using actual distributions
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

np.random.seed(42)  # For reproducibility
predictions = []
prediction_sources = {'pattern': 0, 'fallback': 0}

for idx, row in sample_df.iterrows():
    id_str = row['ID']
    
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        rating_type = match.group(3)
        
        hour = segment_hour_map.get(segment_id, 12)
        
        pattern_key = (location, hour, rating_type)
        
        if pattern_key in pattern_lookup:
            # Sample from actual distribution
            values = pattern_lookup[pattern_key]
            prediction = np.random.choice(values)
            prediction_sources['pattern'] += 1
        else:
            # Fallback: use overall distribution for this location
            location_data = train_df[train_df['view_label'] == location]
            if len(location_data) > 0:
                if rating_type == 'enter':
                    values = location_data['congestion_enter_rating'].tolist()
                else:
                    values = location_data['congestion_exit_rating'].tolist()
                prediction = np.random.choice(values)
                prediction_sources['fallback'] += 1
            else:
                prediction = 'free flowing'
                prediction_sources['fallback'] += 1
        
        predictions.append(prediction)
    else:
        prediction = 'free flowing'
        prediction_sources['fallback'] += 1
        predictions.append(prediction)
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(sample_df)}...")

# Update submission
sample_df['Target'] = predictions
sample_df['Target_Accuracy'] = predictions

# Save
output_path = 'submissions/v3.4_pattern_improved.csv'
sample_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}")

# Analysis
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print(f"\nPrediction sources:")
print(f"  From patterns: {prediction_sources['pattern']}")
print(f"  Fallback: {prediction_sources['fallback']}")

print("\nPrediction distribution:")
pred_dist = sample_df['Target'].value_counts(normalize=True).sort_index()
print(pred_dist)

print("\nPrediction counts:")
print(sample_df['Target'].value_counts())

print(f"\nUnique classes: {sample_df['Target'].nunique()}")

# Compare with training
print("\nTraining distribution (for comparison):")
train_dist = train_df['congestion_enter_rating'].value_counts(normalize=True).sort_index()
print(train_dist)

print("\n" + "="*60)
print("READY FOR SUBMISSION!")
print("="*60)
print(f"File: {output_path}")
print(f"Method: Pattern-based with actual distributions")
print(f"Prediction diversity: {sample_df['Target'].nunique()} classes")
print(f"Expected score: 0.48-0.54 (should match training distribution)")
print("="*60)
