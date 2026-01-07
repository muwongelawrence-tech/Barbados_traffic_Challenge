"""
Time-Based Forecasting Submission
Predict based on hour/location patterns, not segment IDs
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
print("TIME-BASED FORECASTING SUBMISSION - v2.1")
print("="*60)
print("Predicting by hour/location patterns")
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

print(f"\nData loaded:")
print(f"  Train: {len(train_df)}")
print(f"  Sample: {len(sample_df)}")

# Video medians
video_feature_cols = ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                      'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                      'flow_rate_proxy', 'congestion_proxy']
video_medians = {col: video_features_df[col].median() for col in video_feature_cols}

# Calculate average congestion by hour and location from training data
print("\n" + "="*60)
print("LEARNING PATTERNS FROM TRAINING DATA")
print("="*60)

train_df['hour'] = train_df['datetimestamp_start'].apply(lambda x: int(str(x).split()[1].split(':')[0]))

# Calculate most common congestion by hour and location
pattern_lookup = {}

for location in train_df['view_label'].unique():
    for hour in range(24):
        loc_hour_data = train_df[(train_df['view_label'] == location) & (train_df['hour'] == hour)]
        
        if len(loc_hour_data) > 0:
            # Most common congestion for this hour/location
            enter_mode = loc_hour_data['congestion_enter_rating'].mode()
            exit_mode = loc_hour_data['congestion_exit_rating'].mode()
            
            if len(enter_mode) > 0:
                pattern_lookup[(location, hour, 'enter')] = enter_mode[0]
            if len(exit_mode) > 0:
                pattern_lookup[(location, hour, 'exit')] = exit_mode[0]

print(f"✓ Learned patterns for {len(pattern_lookup)} hour/location combinations")

# Also prepare model-based predictions using aggregated features
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

# Prepare aggregated features by hour/location
hour_location_features = {}

for location in train_df['view_label'].unique():
    location_df = train_df[train_df['view_label'] == location].sort_values('time_segment_id')
    
    for hour in range(24):
        hour_data = location_df[location_df['hour'] == hour]
        
        if len(hour_data) >= 10:  # Need enough data
            # Use median segment as representative
            median_idx = len(hour_data) // 2
            median_row = hour_data.iloc[median_idx]
            
            # Create features
            feature_dict = {
                'current_segment': median_row['time_segment_id'],
                'location': location,
                'hour': hour,
                'signaling': median_row['signaling'],
            }
            
            # Video features
            for col in video_feature_cols:
                feature_dict[f'current_{col}'] = video_medians[col]
            
            # Lag features from typical patterns
            for lag in [1, 2, 3, 5, 10]:
                # Use mode of congestion at this hour
                typical_congestion = pattern_lookup.get((location, hour, 'enter'), 'free flowing')
                feature_dict[f'enter_lag_{lag}'] = typical_congestion
                feature_dict[f'motion_lag_{lag}'] = video_medians['motion_mean']
                feature_dict[f'occupancy_lag_{lag}'] = video_medians['occupancy_mean']
            
            hour_location_features[(location, hour)] = feature_dict

print(f"✓ Prepared features for {len(hour_location_features)} hour/location combinations")

# Fill sample submission
print("\n" + "="*60)
print("FILLING SAMPLE SUBMISSION")
print("="*60)

submission_df = sample_df.copy()

pattern_used = 0
model_used = 0
default_used = 0

for idx, row in submission_df.iterrows():
    id_str = row['ID']
    
    # Parse ID
    import re
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        rating_type = match.group(3)
        
        # Estimate hour based on segment ID (rough approximation)
        # Assume segments are roughly sequential in time
        estimated_hour = (segment_id // 60) % 24  # Very rough estimate
        
        # Try model-based prediction
        if (location, estimated_hour) in hour_location_features:
            feature_dict = hour_location_features[(location, estimated_hour)]
            
            # Encode
            feature_row = pd.DataFrame([feature_dict])
            for col in feature_row.columns:
                if feature_row[col].dtype == 'object':
                    le_temp = LabelEncoder()
                    feature_row[col] = le_temp.fit_transform(feature_row[col].astype(str))
            
            # Predict
            try:
                pred_encoded = model.predict(feature_row[feature_cols])[0]
                prediction = le.inverse_transform([pred_encoded])[0]
                model_used += 1
            except:
                # Fall back to pattern
                prediction = pattern_lookup.get((location, estimated_hour, rating_type), 'free flowing')
                pattern_used += 1
        else:
            # Use pattern lookup
            prediction = pattern_lookup.get((location, estimated_hour, rating_type), 'free flowing')
            if prediction == 'free flowing' and (location, estimated_hour, rating_type) not in pattern_lookup:
                default_used += 1
            else:
                pattern_used += 1
        
        submission_df.at[idx, 'Target'] = prediction
        submission_df.at[idx, 'Target_Accuracy'] = prediction
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(submission_df)}...")

print(f"\nPrediction sources:")
print(f"  Model-based: {model_used}")
print(f"  Pattern-based: {pattern_used}")
print(f"  Default: {default_used}")

# Save
output_path = os.path.join(config.SUBMISSIONS_DIR, 'v2.1_time_based.csv')
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved to {output_path}")

# Analyze
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nPrediction distribution:")
print(submission_df['Target'].value_counts())

print(f"\nUnique classes: {submission_df['Target'].nunique()}")

print("\n" + "="*60)
print("READY FOR SUBMISSION")
print("="*60)
print(f"File: {output_path}")
print("Method: Time-based forecasting (hour/location patterns)")
print("Expected: Should perform better than random!")
print("="*60)
