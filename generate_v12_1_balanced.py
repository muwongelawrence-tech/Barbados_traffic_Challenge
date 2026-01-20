"""
v12.1 - Balanced Test Input Features
Fix v12.0's over-prediction of congestion

Strategy:
1. Use test input context for ENTER predictions
2. If current state is "free flowing", most likely stays "free flowing"
3. Apply transition probabilities learned from training

Key insight from training data:
- When current is free flowing → ~91% stays free flowing
- When current is congested → ~50% becomes free flowing
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import re
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("v12.1 - Balanced Test Input Features")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')
test_input_df = pd.read_csv('TestInputSegments.csv')

# Build test lookup
test_lookup = {}
for _, row in test_input_df.iterrows():
    key = (row['time_segment_id'], row['view_label'])
    try:
        hour = int(str(row['datetimestamp_start']).split()[1].split(':')[0])
    except:
        hour = 12
    test_lookup[key] = {
        'enter_rating': row['congestion_enter_rating'],
        'exit_rating': row['congestion_exit_rating'],
        'hour': hour
    }

print(f"✓ Built lookup with {len(test_lookup)} test segments")

# Calculate transition probabilities from training data
print("\n" + "="*60)
print("CALCULATING TRANSITION PROBABILITIES")
print("="*60)

train_df['hour'] = train_df['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0]) if len(str(x).split()) > 1 else 12
)

# Build transition matrix: P(next_state | current_state)
states = ['free flowing', 'heavy delay', 'light delay', 'moderate delay']
transition_counts = {s1: {s2: 0 for s2 in states} for s1 in states}

for location in train_df['view_label'].unique():
    loc_data = train_df[train_df['view_label'] == location].sort_values('time_segment_id')
    
    for i in range(1, len(loc_data)):
        prev_row = loc_data.iloc[i-1]
        curr_row = loc_data.iloc[i]
        
        if curr_row['time_segment_id'] - prev_row['time_segment_id'] <= 20:
            prev_state = prev_row['congestion_enter_rating']
            curr_state = curr_row['congestion_enter_rating']
            transition_counts[prev_state][curr_state] += 1

# Convert to probabilities
print("\nTransition Probabilities (current -> next):")
transition_probs = {}
for s1 in states:
    total = sum(transition_counts[s1].values())
    if total > 0:
        transition_probs[s1] = {s2: transition_counts[s1][s2] / total for s2 in states}
        print(f"\nFrom '{s1}':")
        for s2 in states:
            print(f"  → {s2}: {100*transition_probs[s1][s2]:.1f}%")

# Generate predictions using transition probabilities
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

def parse_id(id_str):
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    if match:
        return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None

predictions = []
used_context = 0

for _, row in sample_df.iterrows():
    id_str = row['ID']
    segment_id, location, rating_type = parse_id(id_str)
    
    if segment_id and location:
        # Find test input context
        test_context = None
        for offset in range(7, 18):
            key = (segment_id - offset, location)
            if key in test_lookup:
                test_context = test_lookup[key]
                break
        
        if test_context:
            used_context += 1
            current_state = test_context['enter_rating']
            hour = test_context['hour']
            
            if rating_type == 'enter':
                # Use transition probabilities
                if current_state in transition_probs:
                    probs = transition_probs[current_state]
                    
                    # Apply hour-based adjustment
                    # Morning (before 10): favor free flowing more
                    if hour < 10:
                        # Morning is almost always free flowing
                        prediction = 'free flowing'
                    else:
                        # Use weighted sampling based on probabilities
                        prediction = max(probs, key=probs.get)
                else:
                    prediction = 'free flowing'
            else:
                # Exit is almost always free flowing
                prediction = 'free flowing'
        else:
            # Fallback
            estimated_hour = 6 + ((segment_id // 60) % 12)
            if estimated_hour < 10:
                prediction = 'free flowing'
            else:
                prediction = 'free flowing'
        
        predictions.append({
            'ID': id_str,
            'Target': prediction,
            'Target_Accuracy': prediction
        })

print(f"✓ Used context: {used_context}/{len(sample_df)}")

submission_df = pd.DataFrame(predictions)

print("\nv12.1 Distribution:")
print(submission_df['Target'].value_counts(normalize=True).sort_index())

v7_5 = pd.read_csv('submissions/v7.5_pseudo_labeling.csv')
print("\nv7.5 Distribution:")
print(v7_5['Target'].value_counts(normalize=True).sort_index())

# Save
submission_df.to_csv('submissions/v12.1_balanced.csv', index=False)
print("\n✓ Saved: submissions/v12.1_balanced.csv")
