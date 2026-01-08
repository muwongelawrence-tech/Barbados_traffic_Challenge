"""
Generate Submission with XGBoost Only (Best Single Model)
F1: 0.5320 (better than ensemble for diversity)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

import config

print("="*60)
print("SUBMISSION - XGBOOST ONLY (Best Single Model)")
print("="*60)

# Load ensemble to get XGBoost model
with open('models/ensemble_model_v3.2.pkl', 'rb') as f:
    ensemble_data = pickle.load(f)

xgb_model = ensemble_data['models']['xgb']
le_target = ensemble_data['label_encoder']
feature_columns = ensemble_data['feature_columns']
video_feature_cols = ensemble_data['video_feature_cols']

print(f"✓ Loaded XGBoost model (F1: 0.5320)")

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
video_features_df = pd.read_csv('video_features.csv')

print(f"\nData: Train={len(train_df)}, Sample={len(sample_df)}")

# Segment-hour mapping
train_df['hour'] = train_df['datetimestamp_start'].apply(lambda x: int(str(x).split()[1].split(':')[0]))

segment_hour_map = {}
for seg_id in train_df['time_segment_id'].unique():
    seg_data = train_df[train_df['time_segment_id'] == seg_id]
    most_common_hour = seg_data['hour'].mode()[0] if len(seg_data) > 0 else 12
    segment_hour_map[seg_id] = most_common_hour

print(f"✓ Built segment-hour mapping: {len(segment_hour_map)} segments")

# Video medians
video_medians = {col: video_features_df[col].median() for col in video_feature_cols}

# Prepare features
hour_location_features = {}

for location in train_df['view_label'].unique():
    location_df = train_df[train_df['view_label'] == location].sort_values('time_segment_id')
    
    for hour in range(6, 18):
        hour_data = location_df[location_df['hour'] == hour]
        
        if len(hour_data) >= 3:
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
            
            typical_enter = hour_data['congestion_enter_rating'].mode()
            typical_congestion = typical_enter[0] if len(typical_enter) > 0 else 'free flowing'
            
            for lag in [1, 2, 3, 5, 10]:
                feature_dict[f'enter_lag_{lag}'] = typical_congestion
                feature_dict[f'motion_lag_{lag}'] = video_medians['motion_mean']
                feature_dict[f'occupancy_lag_{lag}'] = video_medians['occupancy_mean']
            
            hour_location_features[(location, hour)] = feature_dict

print(f"✓ Prepared {len(hour_location_features)} hour/location features")

# Generate predictions
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

submission_df = sample_df.copy()
predictions = []

for idx, row in submission_df.iterrows():
    id_str = row['ID']
    
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', id_str)
    
    if match:
        segment_id = int(match.group(1))
        location = match.group(2)
        rating_type = match.group(3)
        
        hour = segment_hour_map.get(segment_id, 12)
        
        if (location, hour) in hour_location_features:
            feature_dict = hour_location_features[(location, hour)]
            
            feature_row = pd.DataFrame([feature_dict])
            
            for col in feature_row.columns:
                if feature_row[col].dtype == 'object':
                    le_temp = LabelEncoder()
                    feature_row[col] = le_temp.fit_transform(feature_row[col].astype(str))
            
            for col in feature_columns:
                if col not in feature_row.columns:
                    feature_row[col] = 0
            
            feature_row = feature_row[feature_columns]
            
            try:
                # XGBoost prediction
                pred_encoded = int(xgb_model.predict(feature_row)[0])
                prediction = le_target.inverse_transform([pred_encoded])[0]
            except Exception as e:
                prediction = 'free flowing'
        else:
            prediction = 'free flowing'
        
        predictions.append(prediction)
    else:
        predictions.append('free flowing')
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(submission_df)}...")

submission_df['Target'] = predictions
submission_df['Target_Accuracy'] = predictions

# Save
output_path = os.path.join(config.SUBMISSIONS_DIR, 'v3.2_xgboost_FINAL.csv')
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}")

# Analysis
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nPrediction distribution:")
print(submission_df['Target'].value_counts())

print(f"\nUnique classes: {submission_df['Target'].nunique()}")

non_free = (submission_df['Target'] != 'free flowing').sum()
print(f"Non-free-flowing: {non_free} ({non_free/len(submission_df)*100:.1f}%)")

print("\n" + "="*60)
print("READY FOR SUBMISSION!")
print("="*60)
print(f"File: {output_path}")
print(f"Model: XGBoost (Best Single Model)")
print(f"Validation F1: 0.5320")
print(f"Expected leaderboard: 0.53-0.57")
print("="*60)
