"""
v11.2 - Time Pattern-Based Selective Changes
Key insight: Morning (6-10) is 99%+ free flowing across all locations
            Afternoon (10-18) has varying congestion levels by location

Strategy:
1. Start from v7.5 (best score: 0.594)
2. For morning hours: Increase free flowing predictions
3. For Norman Niles #4 afternoon: Increase congestion predictions
4. Make only high-confidence targeted changes
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import re
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("v11.2 - Time Pattern-Based Strategy")
print("="*60)

# Load all data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')
video_features_df = pd.read_csv('video_features.csv')
v7_5 = pd.read_csv('submissions/v7.5_pseudo_labeling.csv')

# Feature preparation
def extract_timestamp_from_filename(filename):
    parts = str(filename).replace('.mp4', '').split('_')
    if len(parts) >= 2:
        date_time = parts[1]
        components = date_time.split('-')
        if len(components) == 6:
            return f"{components[0]}-{components[1]}-{components[2]} {components[3]}:{components[4]}:{components[5]}"
    return None

def simplify_timestamp(ts_str):
    try:
        parts = str(ts_str).split()
        if len(parts) >= 2:
            return f"{parts[0]} {parts[1].split(':')[0]}:{parts[1].split(':')[1]}"
    except:
        pass
    return None

video_features_df['timestamp'] = video_features_df['video_filename'].apply(extract_timestamp_from_filename)
video_features_df['timestamp_simple'] = video_features_df['timestamp'].apply(simplify_timestamp)

# Build pattern lookup from training
print("\n" + "="*60)
print("BUILDING TIME-LOCATION PATTERN LOOKUP")
print("="*60)

# For each location and time range, calculate most likely congestion
pattern_lookup = {}
locations = ['Norman Niles #1', 'Norman Niles #2', 'Norman Niles #3', 'Norman Niles #4']
time_ranges = [(6, 10), (10, 14), (14, 18), (18, 22)]

for loc in locations:
    loc_data = train_df[train_df['view_label'] == loc].copy()
    loc_data['hour'] = loc_data['datetimestamp_start'].apply(
        lambda x: int(str(x).split()[1].split(':')[0]) if len(str(x).split()) > 1 else 12
    )
    
    for h_start, h_end in time_ranges:
        hour_data = loc_data[(loc_data['hour'] >= h_start) & (loc_data['hour'] < h_end)]
        if len(hour_data) > 0:
            dist = hour_data['congestion_enter_rating'].value_counts(normalize=True)
            most_common = dist.idxmax()
            confidence = dist.max()
            pattern_lookup[(loc, h_start, h_end)] = {
                'most_common': most_common,
                'confidence': confidence,
                'distribution': dist.to_dict(),
                'samples': len(hour_data)
            }
            print(f"{loc} ({h_start}-{h_end}h): {most_common} ({confidence:.1%})")

# Parse sample submission IDs
def parse_id(id_str):
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    if match:
        return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None

# Make targeted changes to v7.5
print("\n" + "="*60)
print("MAKING TARGETED CHANGES TO v7.5")
print("="*60)

new_predictions = []
changes = {'morning_to_free': 0, 'evening_to_congestion': 0, 'other': 0}

for idx, row in v7_5.iterrows():
    id_str = row['ID']
    v75_pred = row['Target']
    segment_id, location, rating_type = parse_id(id_str)
    
    final_pred = v75_pred  # Default to v7.5
    
    if segment_id and location and rating_type == 'enter':
        # Estimate hour from segment_id
        estimated_hour = 6 + ((segment_id // 60) % 12)
        
        # Apply time-pattern rules
        if 6 <= estimated_hour < 10:
            # Morning: Should almost always be free flowing
            if v75_pred != 'free flowing':
                # Check pattern
                key = (location, 6, 10)
                if key in pattern_lookup:
                    if pattern_lookup[key]['confidence'] > 0.95:
                        final_pred = 'free flowing'
                        changes['morning_to_free'] += 1
        
        elif 10 <= estimated_hour < 18:
            # Midday/Afternoon: Check location-specific patterns
            h_start = 10 if estimated_hour < 14 else 14
            h_end = 14 if estimated_hour < 14 else 18
            key = (location, h_start, h_end)
            
            if key in pattern_lookup:
                pattern = pattern_lookup[key]
                # For Norman Niles #4 (most congested), be more aggressive
                if location == 'Norman Niles #4' and v75_pred == 'free flowing':
                    if pattern['confidence'] < 0.4:  # Less than 40% free flowing
                        # Consider changing to most common congestion type
                        if pattern['most_common'] != 'free flowing':
                            # Only change with some probability to avoid overdoing it
                            # Actually, let's be conservative and not change
                            pass
    
    new_predictions.append({
        'ID': id_str,
        'Target': final_pred,
        'Target_Accuracy': final_pred
    })

# Create submission
submission_df = pd.DataFrame(new_predictions)

print(f"\nChanges made:")
print(f"  Morning -> Free flowing: {changes['morning_to_free']}")
print(f"  Other: {changes['other']}")

# Compare distributions
print("\nv11.2 Distribution:")
print(submission_df['Target'].value_counts(normalize=True).sort_index())

print("\nv7.5 Distribution:")
print(v7_5['Target'].value_counts(normalize=True).sort_index())

# Count total differences
merged = submission_df.merge(v7_5, on='ID', suffixes=('_new', '_old'))
total_diff = (merged['Target_new'] != merged['Target_old']).sum()
print(f"\nTotal differences from v7.5: {total_diff}")

# Save
output_path = 'submissions/v11.2_time_patterns.csv'
submission_df.to_csv(output_path, index=False)
print(f"\n✓ Saved: {output_path}")
