"""
Complete Submission - Cover ALL Required Segments
Based on error messages, we need to predict segments not in TestInputSegments.csv
"""
import pandas as pd
import numpy as np

print("="*60)
print("CREATING COMPLETE SUBMISSION")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('TestInputSegments.csv')
sample_df = pd.read_csv('SampleSubmission.csv')

print(f"Train segments: {train_df['time_segment_id'].nunique()}")
print(f"Test segments: {test_df['time_segment_id'].nunique()}")
print(f"Sample submission rows: {len(sample_df)}")

# Extract hour from training
train_df['hour'] = train_df['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0])
)

# Build comprehensive pattern lookup
print("\nBuilding pattern lookup from training...")
pattern_lookup = {}

for location in train_df['view_label'].unique():
    for hour in range(6, 18):
        data = train_df[(train_df['view_label'] == location) & (train_df['hour'] == hour)]
        if len(data) >= 3:
            enter_values = data['congestion_enter_rating'].tolist()
            exit_values = data['congestion_exit_rating'].tolist()
            pattern_lookup[(location, hour, 'enter')] = enter_values
            pattern_lookup[(location, hour, 'exit')] = exit_values

# Also store overall location patterns as fallback
location_patterns = {}
for location in train_df['view_label'].unique():
    data = train_df[train_df['view_label'] == location]
    location_patterns[(location, 'enter')] = data['congestion_enter_rating'].tolist()
    location_patterns[(location, 'exit')] = data['congestion_exit_rating'].tolist()

print(f"✓ Learned {len(pattern_lookup)} hour/location patterns")
print(f"✓ Learned {len(location_patterns)} location patterns")

# Get all unique segment IDs and locations from sample submission
# Parse sample to understand what's expected
import re

required_predictions = []
seen_ids = set()

# First, add all IDs from sample submission
for id_str in sample_df['ID']:
    if id_str not in seen_ids:
        required_predictions.append(id_str)
        seen_ids.add(id_str)

# Then add all IDs from test data
for _, row in test_df.iterrows():
    for id_str in [row['ID_enter'], row['ID_exit']]:
        if id_str not in seen_ids:
            required_predictions.append(id_str)
            seen_ids.add(id_str)

print(f"\nTotal unique IDs to predict: {len(required_predictions)}")

# Generate predictions
print("\nGenerating predictions...")
np.random.seed(42)
predictions = []

for id_str in required_predictions:
    # Parse ID to get segment, location, and rating type
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        rating_type = match.group(3)
        
        # Estimate hour (use middle of day as default)
        hour = 12
        
        # Try to get hour from test data if segment exists
        test_match = test_df[test_df['time_segment_id'] == segment_id]
        if len(test_match) > 0:
            hour = int(str(test_match.iloc[0]['datetimestamp_start']).split()[1].split(':')[0])
        
        # Look up pattern
        pattern_key = (location, hour, rating_type)
        
        if pattern_key in pattern_lookup:
            values = pattern_lookup[pattern_key]
            prediction = np.random.choice(values)
        elif (location, rating_type) in location_patterns:
            # Fallback to location average
            values = location_patterns[(location, rating_type)]
            prediction = np.random.choice(values)
        else:
            # Ultimate fallback
            prediction = 'free flowing'
        
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
output_path = 'submissions/v4.1_COMPLETE.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}")
print(f"✓ Total predictions: {len(submission_df)}")

# Analysis
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nPrediction distribution:")
print(submission_df['Target'].value_counts(normalize=True))

print("\nPrediction counts:")
print(submission_df['Target'].value_counts())

print("\n" + "="*60)
print("READY FOR SUBMISSION!")
print("="*60)
print(f"File: {output_path}")
print(f"Predictions: {len(submission_df)}")
print("="*60)
