"""
CORRECT Submission Generation
Predict on TestInputSegments.csv and create submission file
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
print("CORRECT SUBMISSION GENERATION - v2.1")
print("="*60)
print("Predicting on TestInputSegments.csv")
print("="*60)

# Load model
model_path = os.path.join(config.MODELS_DIR, 'forecasting_model_v2.1_enhanced.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le = model_data['label_encoder']
feature_cols = model_data['feature_columns']

print(f"✓ Loaded model v2.1")
print(f"  Features: {len(feature_cols)}")

# Load data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('TestInputSegments.csv')
video_features_df = pd.read_csv('video_features.csv')

print(f"\nData loaded:")
print(f"  Train: {len(train_df)} segments")
print(f"  Test: {len(test_df)} segments")

# Combine for lag features
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df = combined_df.sort_values(['view_label', 'time_segment_id']).reset_index(drop=True)

print(f"  Combined: {len(combined_df)} segments")

# Video feature medians
video_feature_cols = ['motion_mean', 'motion_std', 'motion_max', 'occupancy_mean', 
                      'occupancy_std', 'occupancy_max', 'vehicle_count_proxy', 
                      'flow_rate_proxy', 'congestion_proxy']
video_medians = {col: video_features_df[col].median() for col in video_feature_cols}

# Generate predictions for TEST data
print("\n" + "="*60)
print("GENERATING PREDICTIONS ON TEST DATA")
print("="*60)

predictions_enter = []
predictions_exit = []
test_indices = []

for idx, row in test_df.iterrows():
    segment_id = row['time_segment_id']
    location = row['view_label']
    
    # Find in combined data
    location_df = combined_df[combined_df['view_label'] == location].reset_index(drop=True)
    segment_idx = location_df[location_df['time_segment_id'] == segment_id].index
    
    if len(segment_idx) == 0 or segment_idx[0] < 15:
        # Not enough history - use most common class
        pred_enter = 'free flowing'
        pred_exit = 'free flowing'
    else:
        idx_val = segment_idx[0]
        
        # Create features
        feature_dict = {
            'current_segment': segment_id,
            'location': location,
            'hour': int(str(row['datetimestamp_start']).split()[1].split(':')[0]),
            'signaling': row['signaling'],
        }
        
        # Video features (medians)
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
        pred_enter = le.inverse_transform([pred_encoded])[0]
        pred_exit = pred_enter  # Use same prediction for both
    
    predictions_enter.append(pred_enter)
    predictions_exit.append(pred_exit)
    test_indices.append(idx)
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(test_df)}...")

print(f"\n✓ Generated {len(predictions_enter)} predictions")

# Create submission in sample format
print("\n" + "="*60)
print("CREATING SUBMISSION FILE")
print("="*60)

submission_rows = []

for idx, row in test_df.iterrows():
    segment_id = row['time_segment_id']
    location = row['view_label']
    
    pred_enter = predictions_enter[idx]
    pred_exit = predictions_exit[idx]
    
    # Create IDs in sample submission format
    id_enter = f"time_segment_{segment_id}_{location}_congestion_enter_rating"
    id_exit = f"time_segment_{segment_id}_{location}_congestion_exit_rating"
    
    submission_rows.append({
        'ID': id_enter,
        'Target': pred_enter,
        'Target_Accuracy': pred_enter
    })
    
    submission_rows.append({
        'ID': id_exit,
        'Target': pred_exit,
        'Target_Accuracy': pred_exit
    })

submission_df = pd.DataFrame(submission_rows)

# Save
output_path = os.path.join(config.SUBMISSIONS_DIR, 'v2.1_correct.csv')
submission_df.to_csv(output_path, index=False)

print(f"✓ Submission saved to {output_path}")
print(f"  Total rows: {len(submission_df)}")

# Analyze
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

print("\nEnter rating predictions:")
print(pd.Series(predictions_enter).value_counts())

print("\nExit rating predictions:")
print(pd.Series(predictions_exit).value_counts())

print(f"\nUnique classes predicted: {pd.Series(predictions_enter).nunique()}")

print("\n" + "="*60)
print("READY FOR SUBMISSION")
print("="*60)
print(f"File: {output_path}")
print("This submission uses ACTUAL TEST DATA predictions!")
print("Expected: Much better than previous submissions")
print("="*60)
