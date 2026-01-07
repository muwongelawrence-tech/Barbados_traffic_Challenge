"""
Conservative Submission - v2.2
Predict mostly "free flowing", only delays when very confident
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import config

print("="*60)
print("CONSERVATIVE SUBMISSION - v2.2")
print("="*60)
print("Strategy: Predict 'free flowing' unless VERY confident of delay")
print("="*60)

# Load model
model_path = os.path.join(config.MODELS_DIR, 'forecasting_model_v2.1_enhanced.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le = model_data['label_encoder']
feature_cols = model_data['feature_columns']

print(f"✓ Loaded model v2.1")

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
video_features_df = pd.read_csv('video_features.csv')

# Video medians
video_feature_cols = ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                      'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                      'flow_rate_proxy', 'congestion_proxy']
video_medians = {col: video_features_df[col].median() for col in video_feature_cols}

# Learn patterns
train_df['hour'] = train_df['datetimestamp_start'].apply(lambda x: int(str(x).split()[1].split(':')[0]))

# Prepare features by hour/location
hour_location_features = {}

for location in train_df['view_label'].unique():
    location_df = train_df[train_df['view_label'] == location].sort_values('time_segment_id')
    
    for hour in range(24):
        hour_data = location_df[location_df['hour'] == hour]
        
        if len(hour_data) >= 10:
            median_idx = len(hour_data) // 2
            median_row = hour_data.iloc[median_idx]
            
            feature_dict = {
                'current_segment': median_row['time_segment_id'],
                'location': location,
                'hour': hour,
                'signaling': median_row['signaling'],
            }
            
            for col in video_feature_cols:
                feature_dict[f'current_{col}'] = video_medians[col]
            
            for lag in [1, 2, 3, 5, 10]:
                feature_dict[f'enter_lag_{lag}'] = 'free flowing'  # Conservative
                feature_dict[f'motion_lag_{lag}'] = video_medians['motion_mean']
                feature_dict[f'occupancy_lag_{lag}'] = video_medians['occupancy_mean']
            
            hour_location_features[(location, hour)] = feature_dict

print(f"✓ Prepared features for {len(hour_location_features)} combinations")

# Fill sample submission with CONSERVATIVE predictions
print("\n" + "="*60)
print("GENERATING CONSERVATIVE PREDICTIONS")
print("="*60)

submission_df = sample_df.copy()

CONFIDENCE_THRESHOLD = 0.95  # Very high threshold

predictions_made = 0
free_flowing_count = 0
delay_count = 0

for idx, row in submission_df.iterrows():
    id_str = row['ID']
    
    # Parse ID
    import re
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    prediction = 'free flowing'  # Default
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        
        # Estimate hour
        estimated_hour = (segment_id // 60) % 24
        
        # Try to get model prediction with probabilities
        if (location, estimated_hour) in hour_location_features:
            feature_dict = hour_location_features[(location, estimated_hour)]
            
            # Encode
            feature_row = pd.DataFrame([feature_dict])
            for col in feature_row.columns:
                if feature_row[col].dtype == 'object':
                    le_temp = LabelEncoder()
                    feature_row[col] = le_temp.fit_transform(feature_row[col].astype(str))
            
            try:
                # Get prediction probabilities
                proba = model.predict_proba(feature_row[feature_cols])[0]
                pred_class = np.argmax(proba)
                max_proba = proba[pred_class]
                
                # Only predict delay if VERY confident
                if max_proba >= CONFIDENCE_THRESHOLD:
                    pred_label = le.inverse_transform([pred_class])[0]
                    if pred_label != 'free flowing':
                        prediction = pred_label
                        delay_count += 1
                    else:
                        free_flowing_count += 1
                else:
                    # Not confident enough - default to free flowing
                    free_flowing_count += 1
                
                predictions_made += 1
            except:
                free_flowing_count += 1
        else:
            free_flowing_count += 1
    
    submission_df.at[idx, 'Target'] = prediction
    submission_df.at[idx, 'Target_Accuracy'] = prediction
    
    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx+1}/{len(submission_df)}...")

print(f"\nPrediction breakdown:")
print(f"  Free flowing: {free_flowing_count}")
print(f"  Delays: {delay_count}")
print(f"  Delay rate: {delay_count/len(submission_df)*100:.1f}%")

# Save
output_path = os.path.join(config.SUBMISSIONS_DIR, 'v2.2_conservative.csv')
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved to {output_path}")

# Analyze
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nPrediction distribution:")
print(submission_df['Target'].value_counts())

print("\n" + "="*60)
print("READY FOR SUBMISSION")
print("="*60)
print(f"File: {output_path}")
print(f"Strategy: {CONFIDENCE_THRESHOLD*100:.0f}% confidence threshold")
print(f"Expected: Should beat v1.0 (0.4612) by being more accurate")
print("="*60)
