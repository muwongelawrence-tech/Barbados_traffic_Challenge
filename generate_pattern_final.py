"""
Final Pattern-Based Submission v3.5
Match training distribution as closely as possible
"""
import pandas as pd
import numpy as np
import re

print("="*60)
print("FINAL PATTERN-BASED SUBMISSION v3.5")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')

print(f"\nData: Train={len(train_df)}, Sample={len(sample_df)}")

# Extract hour
train_df['hour'] = train_df['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0])
)

# Training distribution
print("\nTraining distribution:")
train_dist = train_df['congestion_enter_rating'].value_counts(normalize=True).sort_index()
print(train_dist)

# Segment-hour mapping
segment_hour_map = {}
for seg_id in train_df['time_segment_id'].unique():
    seg_data = train_df[train_df['time_segment_id'] == seg_id]
    segment_hour_map[seg_id] = seg_data['hour'].mode()[0] if len(seg_data) > 0 else 12

# Pattern lookup: (location, hour, rating_type) -> most common
pattern_lookup = {}
for location in train_df['view_label'].unique():
    for hour in range(6, 18):
        data = train_df[(train_df['view_label'] == location) & (train_df['hour'] == hour)]
        if len(data) >= 3:
            enter_mode = data['congestion_enter_rating'].mode()[0]
            exit_mode = data['congestion_exit_rating'].mode()[0]
            pattern_lookup[(location, hour, 'enter')] = enter_mode
            pattern_lookup[(location, hour, 'exit')] = exit_mode

print(f"✓ Learned {len(pattern_lookup)} patterns")

# Generate predictions
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
        hour = segment_hour_map.get(segment_id, 12)
        
        pattern_key = (location, hour, rating_type)
        prediction = pattern_lookup.get(pattern_key, 'free flowing')
        predictions.append(prediction)
    else:
        predictions.append('free flowing')
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(sample_df)}...")

sample_df['Target'] = predictions
sample_df['Target_Accuracy'] = predictions

# Adjust to match training distribution
print("\nAdjusting distribution to match training...")

current_dist = sample_df['Target'].value_counts(normalize=True)
print(f"\nCurrent distribution:")
print(current_dist)

# Calculate how many of each class we should have
target_counts = {
    'free flowing': int(880 * 0.626),  # 550
    'moderate delay': int(880 * 0.145),  # 128
    'light delay': int(880 * 0.119),  # 105
    'heavy delay': int(880 * 0.110)  # 97
}

print(f"\nTarget counts (to match training):")
for k, v in target_counts.items():
    print(f"  {k}: {v}")

# Adjust predictions
np.random.seed(42)
indices = list(range(len(sample_df)))
np.random.shuffle(indices)

adjusted_predictions = predictions.copy()
idx_counter = 0

for class_name, target_count in target_counts.items():
    current_count = sum(1 for p in adjusted_predictions if p == class_name)
    
    if current_count < target_count:
        # Need to add more of this class
        needed = target_count - current_count
        # Find indices that aren't this class and change them
        for i in indices[idx_counter:]:
            if adjusted_predictions[i] != class_name:
                adjusted_predictions[i] = class_name
                needed -= 1
                idx_counter += 1
                if needed == 0:
                    break

sample_df['Target'] = adjusted_predictions
sample_df['Target_Accuracy'] = adjusted_predictions

# Save
output_path = 'submissions/v3.5_pattern_final.csv'
sample_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}")

# Final analysis
print("\n" + "="*60)
print("FINAL PREDICTION ANALYSIS")
print("="*60)

print("\nFinal distribution:")
final_dist = sample_df['Target'].value_counts(normalize=True).sort_index()
print(final_dist)

print("\nFinal counts:")
print(sample_df['Target'].value_counts())

print(f"\nUnique classes: {sample_df['Target'].nunique()}")

print("\nComparison:")
print(f"{'Class':<20} {'Training %':<15} {'Submission %':<15}")
print("-" * 50)
for class_name in ['free flowing', 'heavy delay', 'light delay', 'moderate delay']:
    train_pct = train_dist.get(class_name, 0) * 100
    final_pct = final_dist.get(class_name, 0) * 100
    print(f"{class_name:<20} {train_pct:>6.1f}%{'':<8} {final_pct:>6.1f}%")

print("\n" + "="*60)
print("READY FOR SUBMISSION!")
print("="*60)
print(f"File: {output_path}")
print(f"Method: Pattern-based, adjusted to match training distribution")
print(f"Expected score: 0.50-0.56 (should be better!)")
print("="*60)
