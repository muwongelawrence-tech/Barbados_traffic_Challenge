"""
Pattern-Based Submission v3.3
Use training data patterns directly - NO ML models
This should give prediction diversity and better score
"""
import pandas as pd
import numpy as np
import re
import os

print("="*60)
print("PATTERN-BASED SUBMISSION v3.3")
print("="*60)
print("Using training patterns directly (no ML)")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')

print(f"\nData loaded:")
print(f"  Training: {len(train_df)} records")
print(f"  Sample: {len(sample_df)} predictions needed")

# Extract hour from training data
train_df['hour'] = train_df['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0])
)

print(f"\nTraining data covers hours: {train_df['hour'].min()} to {train_df['hour'].max()}")

# Build segment-to-hour mapping from training data
print("\nBuilding segment-to-hour mapping...")
segment_hour_map = {}
for seg_id in train_df['time_segment_id'].unique():
    seg_data = train_df[train_df['time_segment_id'] == seg_id]
    most_common_hour = seg_data['hour'].mode()[0] if len(seg_data) > 0 else 12
    segment_hour_map[seg_id] = most_common_hour

print(f"✓ Mapped {len(segment_hour_map)} segments to hours")

# Build pattern lookup: (location, hour, rating_type) -> most common congestion
print("\nLearning congestion patterns...")
pattern_lookup = {}

for location in train_df['view_label'].unique():
    for hour in range(6, 18):  # Data covers 6 AM to 5 PM
        # Enter rating patterns
        enter_data = train_df[(train_df['view_label'] == location) & (train_df['hour'] == hour)]
        if len(enter_data) > 0:
            most_common_enter = enter_data['congestion_enter_rating'].mode()[0]
            pattern_lookup[(location, hour, 'enter')] = most_common_enter
        
        # Exit rating patterns
        if len(enter_data) > 0:
            most_common_exit = enter_data['congestion_exit_rating'].mode()[0]
            pattern_lookup[(location, hour, 'exit')] = most_common_exit

print(f"✓ Learned {len(pattern_lookup)} patterns")

# Analyze patterns
print("\nPattern distribution:")
pattern_values = list(pattern_lookup.values())
unique_patterns = pd.Series(pattern_values).value_counts()
print(unique_patterns)

# Generate predictions
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

predictions = []
prediction_sources = {'pattern': 0, 'default': 0}

for idx, row in sample_df.iterrows():
    id_str = row['ID']
    
    # Parse ID: time_segment_XXX_Location_congestion_enter/exit_rating
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        rating_type = match.group(3)
        
        # Get hour from segment mapping
        hour = segment_hour_map.get(segment_id, 12)  # default to noon
        
        # Look up pattern
        pattern_key = (location, hour, rating_type)
        
        if pattern_key in pattern_lookup:
            prediction = pattern_lookup[pattern_key]
            prediction_sources['pattern'] += 1
        else:
            # Fallback: use any hour for this location
            found = False
            for h in range(6, 18):
                fallback_key = (location, h, rating_type)
                if fallback_key in pattern_lookup:
                    prediction = pattern_lookup[fallback_key]
                    prediction_sources['pattern'] += 1
                    found = True
                    break
            
            if not found:
                prediction = 'free flowing'
                prediction_sources['default'] += 1
        
        predictions.append(prediction)
    else:
        predictions.append('free flowing')
        prediction_sources['default'] += 1
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(sample_df)}...")

# Update submission
sample_df['Target'] = predictions
sample_df['Target_Accuracy'] = predictions

# Save
output_path = 'submissions/v3.3_pattern_based.csv'
sample_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}")

# Analysis
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print(f"\nPrediction sources:")
print(f"  From patterns: {prediction_sources['pattern']}")
print(f"  Default (free flowing): {prediction_sources['default']}")

print("\nPrediction distribution:")
pred_dist = sample_df['Target'].value_counts()
print(pred_dist)

print(f"\nUnique classes: {sample_df['Target'].nunique()}")

non_free = (sample_df['Target'] != 'free flowing').sum()
print(f"Non-free-flowing: {non_free} ({non_free/len(sample_df)*100:.1f}%)")

# Compare with training distribution
print("\nTraining data distribution (for comparison):")
train_dist = train_df['congestion_enter_rating'].value_counts()
print(train_dist)

print("\n" + "="*60)
print("READY FOR SUBMISSION!")
print("="*60)
print(f"File: {output_path}")
print(f"Method: Pattern-based (training data patterns)")
print(f"Prediction diversity: {sample_df['Target'].nunique()} classes")
print(f"Expected score: 0.48-0.52 (better than 0.4612)")
print("="*60)
