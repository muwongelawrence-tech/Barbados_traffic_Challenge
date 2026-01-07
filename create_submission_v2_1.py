"""
Generate submission for v2.1 enhanced model
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
print("GENERATING v2.1 SUBMISSION")
print("="*60)

# Load enhanced model
model_path = os.path.join(config.MODELS_DIR, 'forecasting_model_v2.1_enhanced.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le = model_data['label_encoder']
feature_cols = model_data['feature_columns']

print(f"✓ Loaded model v2.1")
print(f"  Features: {len(feature_cols)}")
print(f"  Classes: {le.classes_}")

# Load data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('TestInputSegments.csv')
video_features_df = pd.read_csv('video_features.csv')

# Combine for continuous time series
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df = combined_df.sort_values(['view_label', 'time_segment_id']).reset_index(drop=True)

print(f"\nCombined dataset: {len(combined_df)} segments")

# Load sample submission
sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
print(f"Sample submission: {len(sample_df)} predictions needed")

# Generate predictions
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

# Use median video features as defaults
video_feature_cols = ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                      'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                      'flow_rate_proxy', 'congestion_proxy']

video_medians = {col: video_features_df[col].median() for col in video_feature_cols}

predictions = []

for idx, row in sample_df.iterrows():
    id_str = row['ID']
    
    # Parse ID
    parts = id_str.split('_')
    segment_id = int(parts[2])
    location = '_'.join(parts[3:-2])
    
    # Find segment
    segment_data = combined_df[
        (combined_df['time_segment_id'] == segment_id) &
        (combined_df['view_label'] == location)
    ]
    
    if len(segment_data) == 0:
        prediction = 'free flowing'
    else:
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
                'hour': int(str(location_df.iloc[idx_val]['datetimestamp_start']).split()[1].split(':')[0]),
                'signaling': location_df.iloc[idx_val]['signaling'],
            }
            
            # Add video features (use medians)
            for col in video_feature_cols:
                feature_dict[f'current_{col}'] = video_medians[col]
            
            # Add lag features
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
    
    predictions.append(prediction)
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(sample_df)}...")

# Create submission
submission_df = sample_df.copy()
submission_df['Target'] = predictions
submission_df['Target_Accuracy'] = predictions

# Save
output_path = os.path.join(config.SUBMISSIONS_DIR, 'v2.1_enhanced.csv')
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved to {output_path}")

# Analyze
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nPrediction distribution:")
print(submission_df['Target'].value_counts())

print(f"\nTotal predictions: {len(submission_df)}")
print(f"Unique classes: {submission_df['Target'].nunique()}")

print("\n" + "="*60)
print("READY FOR SUBMISSION")
print("="*60)
print(f"File: {output_path}")
print("Model: v2.1 Enhanced (F1: 0.4821)")
print("Expected: Better than v2.0 (F1: 0.4677)")
print("="*60)
