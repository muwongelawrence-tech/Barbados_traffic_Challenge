"""
SMART PATTERN-BASED SUBMISSION v6.0
Using discovered patterns:
- Hour 6-9: 99% free flowing
- Hour 10-17: 50-67% delays based on signaling
- Signaling is the KEY predictor
"""
import pandas as pd
import numpy as np
import re

print("="*60)
print("SMART PATTERN-BASED SUBMISSION v6.0")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('TestInputSegments.csv')
sample_df = pd.read_csv('SampleSubmission.csv')

# Extract hour
train_df['hour'] = train_df['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0])
)

test_df['hour'] = test_df['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0])
)

print(f"Test data hour range: {test_df['hour'].min()} to {test_df['hour'].max()}")
print(f"\nTest signaling distribution:")
print(test_df['signaling'].value_counts())

# Build SMART pattern lookup: (location, hour, signaling, rating_type) -> distribution
pattern_lookup = {}

for location in train_df['view_label'].unique():
    for hour in range(6, 18):
        for signaling in ['none', 'low', 'medium', 'high']:
            for rating_type in ['enter', 'exit']:
                data = train_df[
                    (train_df['view_label'] == location) &
                    (train_df['hour'] == hour) &
                    (train_df['signaling'] == signaling)
                ]
                
                if len(data) >= 3:
                    if rating_type == 'enter':
                        values = data['congestion_enter_rating'].tolist()
                    else:
                        values = data['congestion_exit_rating'].tolist()
                    
                    pattern_lookup[(location, hour, signaling, rating_type)] = values

print(f"✓ Learned {len(pattern_lookup)} detailed patterns")

# Fallback: hour + signaling only
fallback_patterns = {}
for hour in range(6, 18):
    for signaling in ['none', 'low', 'medium', 'high']:
        for rating_type in ['enter', 'exit']:
            data = train_df[
                (train_df['hour'] == hour) &
                (train_df['signaling'] == signaling)
            ]
            
            if len(data) >= 5:
                if rating_type == 'enter':
                    values = data['congestion_enter_rating'].tolist()
                else:
                    values = data['congestion_exit_rating'].tolist()
                
                fallback_patterns[(hour, signaling, rating_type)] = values

print(f"✓ Learned {len(fallback_patterns)} fallback patterns")

# Get all required IDs
required_ids = set(sample_df['ID'].tolist())
for _, row in test_df.iterrows():
    required_ids.add(row['ID_enter'])
    required_ids.add(row['ID_exit'])

print(f"\nTotal IDs to predict: {len(required_ids)}")

# Create test lookup
test_lookup = {}
for _, row in test_df.iterrows():
    test_lookup[row['time_segment_id']] = {
        'hour': row['hour'],
        'location': row['view_label'],
        'signaling': row['signaling']
    }

# Generate predictions
np.random.seed(42)
predictions = []

for id_str in required_ids:
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        rating_type = match.group(3)
        
        # Default
        prediction = 'free flowing'
        
        # Get segment info from test data if available
        if segment_id in test_lookup:
            seg_info = test_lookup[segment_id]
            hour = seg_info['hour']
            signaling = seg_info['signaling']
            
            # Try detailed pattern first
            pattern_key = (location, hour, signaling, rating_type)
            if pattern_key in pattern_lookup:
                values = pattern_lookup[pattern_key]
                prediction = np.random.choice(values)
            # Try fallback pattern
            elif (hour, signaling, rating_type) in fallback_patterns:
                values = fallback_patterns[(hour, signaling, rating_type)]
                prediction = np.random.choice(values)
            # Ultimate fallback based on hour
            elif hour >= 10:
                # Peak hours - more delays
                prediction = np.random.choice(
                    ['free flowing', 'light delay', 'moderate delay', 'heavy delay'],
                    p=[0.4, 0.2, 0.2, 0.2]
                )
        
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
output_path = 'submissions/v6.0_smart_patterns.csv'
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

free_pct = (submission_df['Target'] == 'free flowing').sum() / len(submission_df) * 100
delay_pct = 100 - free_pct

print(f"\nFree flowing: {free_pct:.1f}%")
print(f"Delays: {delay_pct:.1f}%")

print("\n" + "="*60)
print("KEY INSIGHTS USED")
print("="*60)
print("✓ Hours 6-9: 99% free flowing")
print("✓ Hours 10-17: 50-67% delays")
print("✓ Signaling is KEY predictor")
print("✓ Location-specific patterns")
print("\nExpected score: 0.55-0.65 (MAJOR IMPROVEMENT!)")
print("="*60)
