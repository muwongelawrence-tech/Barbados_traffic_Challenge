"""
v11.3 - Targeted Corrections for Norman Niles #4
Based on analysis:
- Midday (10-14h): v7.5 predicts 50% free, training shows 35.8% → reduce free
- Afternoon (14-18h): v7.5 predicts 9.1% free, training shows 35.2% → increase free

Strategy: Probabilistically adjust predictions to match training distribution
"""
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("v11.3 - Targeted Norman Niles #4 Corrections")
print("="*60)

# Load data
v7_5 = pd.read_csv('submissions/v7.5_pseudo_labeling.csv')
train_df = pd.read_csv('Train.csv')

np.random.seed(42)

# Get training distribution for Norman Niles #4 afternoon
train_df['hour'] = train_df['datetimestamp_start'].apply(
    lambda x: int(str(x).split()[1].split(':')[0]) if len(str(x).split()) > 1 else 12
)

nn4_afternoon = train_df[(train_df['view_label'] == 'Norman Niles #4') & 
                         (train_df['hour'] >= 14) & (train_df['hour'] < 18)]
nn4_afternoon_dist = nn4_afternoon['congestion_enter_rating'].value_counts(normalize=True)
print("\nNorman Niles #4 Afternoon Training Distribution:")
print(nn4_afternoon_dist)

nn4_midday = train_df[(train_df['view_label'] == 'Norman Niles #4') & 
                      (train_df['hour'] >= 10) & (train_df['hour'] < 14)]
nn4_midday_dist = nn4_midday['congestion_enter_rating'].value_counts(normalize=True)
print("\nNorman Niles #4 Midday Training Distribution:")
print(nn4_midday_dist)

# Parse and adjust
def parse_id(id_str):
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    if match:
        return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None

new_predictions = []
changes = {
    'midday_to_congestion': 0,
    'afternoon_to_free': 0
}

for idx, row in v7_5.iterrows():
    id_str = row['ID']
    v75_pred = row['Target']
    segment_id, location, rating_type = parse_id(id_str)
    
    final_pred = v75_pred  # Default to v7.5
    
    if segment_id and location == 'Norman Niles #4' and rating_type == 'enter':
        estimated_hour = 6 + ((segment_id // 60) % 12)
        
        # Midday: Too much "free flowing" - change some to congestion
        if 10 <= estimated_hour < 14:
            if v75_pred == 'free flowing':
                # v7.5 has 50% free, training has 35.8%
                # Need to change ~28% of free flowing predictions to congestion
                # 0.50 * (1 - x) = 0.358 → x = 0.284
                if np.random.random() < 0.25:  # Change ~25% of free flowing
                    # Pick from training distribution (excluding free flowing)
                    non_free = nn4_midday[nn4_midday['congestion_enter_rating'] != 'free flowing']
                    if len(non_free) > 0:
                        final_pred = np.random.choice(non_free['congestion_enter_rating'].values)
                        changes['midday_to_congestion'] += 1
        
        # Afternoon: Not enough "free flowing" - change some to free
        elif 14 <= estimated_hour < 18:
            if v75_pred != 'free flowing':
                # v7.5 has 9.1% free, training has 35.2%
                # Need to change ~29% of congestion predictions to free
                if np.random.random() < 0.25:  # Change ~25% of congestion
                    final_pred = 'free flowing'
                    changes['afternoon_to_free'] += 1
    
    new_predictions.append({
        'ID': id_str,
        'Target': final_pred,
        'Target_Accuracy': final_pred
    })

submission_df = pd.DataFrame(new_predictions)

print("\n" + "="*60)
print("CHANGES MADE")
print("="*60)
print(f"Midday (free -> congestion): {changes['midday_to_congestion']}")
print(f"Afternoon (congestion -> free): {changes['afternoon_to_free']}")
print(f"Total changes: {sum(changes.values())}")

# Compare distributions
print("\n" + "="*60)
print("DISTRIBUTION COMPARISON")
print("="*60)

print("\nv11.3 Distribution:")
print(submission_df['Target'].value_counts(normalize=True).sort_index())

print("\nv7.5 Distribution:")
print(v7_5['Target'].value_counts(normalize=True).sort_index())

# Save
output_path = 'submissions/v11.3_nn4_targeted.csv'
submission_df.to_csv(output_path, index=False)
print(f"\n✓ Saved: {output_path}")

# Show a few changed predictions
merged = submission_df.merge(v7_5, on='ID', suffixes=('_new', '_old'))
changed = merged[merged['Target_new'] != merged['Target_old']]
print(f"\nSample changes (first 10):")
print(changed[['ID', 'Target_old', 'Target_new']].head(10))
