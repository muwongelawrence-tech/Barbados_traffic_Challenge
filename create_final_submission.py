"""
FINAL CORRECT Submission Generation
Use SampleSubmission.csv as template and fill with our predictions
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
print("FINAL SUBMISSION - v2.1 (Matching Sample Format)")
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
test_df = pd.read_csv('TestInputSegments.csv')
sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
video_features_df = pd.read_csv('video_features.csv')

print(f"\nData loaded:")
print(f"  Train: {len(train_df)}")
print(f"  Test: {len(test_df)}")
print(f"  Sample submission: {len(sample_df)} rows")

# Combine for lag features
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df = combined_df.sort_values(['view_label', 'time_segment_id']).reset_index(drop=True)

# Video medians
video_feature_cols = ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                      'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                      'flow_rate_proxy', 'congestion_proxy']
video_medians = {col: video_features_df[col].median() for col in video_feature_cols}

# Create prediction lookup from test data
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

prediction_lookup = {}

for idx, row in test_df.iterrows():
    segment_id = row['time_segment_id']
    location = row['view_label']
    
    # Find in combined data
    location_df = combined_df[combined_df['view_label'] == location].reset_index(drop=True)
    segment_idx = location_df[location_df['time_segment_id'] == segment_id].index
    
    if len(segment_idx) == 0 or segment_idx[0] < 15:
        prediction = 'free flowing'
    else:
        idx_val = segment_idx[0]
        
        # Create features
        feature_dict = {
            'current_segment': segment_id,
            'location': location,
            'hour': int(str(row['datetimestamp_start']).split()[1].split(':')[0]),
            'signaling': row['signaling'],
        }
        
        # Video features
        for col in video_feature_cols:
            feature_dict[f'current_{col}'] = video_medians[col]
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            if idx_val - lag >= 0:
                past_row = location_df.iloc[idx_val - lag]
                feature_dict[f'enter_lag_{lag}'] = past_row.get('congestion_enter_rating', 'free flowing')
                feature_dict[f'motion_lag_{lag}'] = video_medians['motion_mean']
                feature_dict[f'occupancy_lag_{lag}'] = video_medians['occupancy_mean']
            else:
                feature_dict[f'enter_lag_{lag}'] = 'free flowing'
                feature_dict[f'motion_lag_{lag}'] = video_medians['motion_mean']
                feature_dict[f'occupancy_lag_{lag}'] = video_medians['occupancy_mean']
        
        # Encode
        feature_row = pd.DataFrame([feature_dict])
        for col in feature_row.columns:
            if feature_row[col].dtype == 'object':
                le_temp = LabelEncoder()
                feature_row[col] = le_temp.fit_transform(feature_row[col].astype(str))
        
        # Predict
        pred_encoded = model.predict(feature_row[feature_cols])[0]
        prediction = le.inverse_transform([pred_encoded])[0]
    
    # Store predictions for both enter and exit
    enter_id = f"time_segment_{segment_id}_{location}_congestion_enter_rating"
    exit_id = f"time_segment_{segment_id}_{location}_congestion_exit_rating"
    
    prediction_lookup[enter_id] = prediction
    prediction_lookup[exit_id] = prediction
    
    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx+1}/{len(test_df)}...")

print(f"\n✓ Generated predictions for {len(prediction_lookup)} IDs")

# Fill sample submission with our predictions
print("\n" + "="*60)
print("FILLING SAMPLE SUBMISSION")
print("="*60)

submission_df = sample_df.copy()

filled = 0
defaulted = 0

for idx, row in submission_df.iterrows():
    id_str = row['ID']
    
    if id_str in prediction_lookup:
        submission_df.at[idx, 'Target'] = prediction_lookup[id_str]
        submission_df.at[idx, 'Target_Accuracy'] = prediction_lookup[id_str]
        filled += 1
    else:
        # Keep default from sample or use 'free flowing'
        submission_df.at[idx, 'Target'] = 'heavy delay'  # Use sample default
        submission_df.at[idx, 'Target_Accuracy'] = 'heavy delay'
        defaulted += 1

print(f"Filled with predictions: {filled}")
print(f"Used defaults: {defaulted}")
print(f"Total: {len(submission_df)}")

# Save
output_path = os.path.join(config.SUBMISSIONS_DIR, 'v2.1_final.csv')
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved to {output_path}")

# Analyze
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nPrediction distribution:")
print(submission_df['Target'].value_counts())

print(f"\nCoverage: {filled}/{len(submission_df)} ({filled/len(submission_df)*100:.1f}%)")

print("\n" + "="*60)
print("READY FOR SUBMISSION")
print("="*60)
print(f"File: {output_path}")
print("Format: Matches SampleSubmission.csv exactly")
print("="*60)
