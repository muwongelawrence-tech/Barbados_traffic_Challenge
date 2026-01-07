"""
Generate submission using forecasting model
This creates predictions for FUTURE time segments (5 min ahead)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
import config

print("="*60)
print("GENERATING FORECASTING SUBMISSION - v2.0")
print("="*60)

# Load model
model_path = os.path.join(config.MODELS_DIR, 'forecasting_model_v2.0.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le = model_data['label_encoder']
feature_cols = model_data['feature_columns']

print(f"✓ Loaded forecasting model")
print(f"  Features: {len(feature_cols)}")
print(f"  Classes: {le.classes_}")

# Load data
from src.data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
train_df, test_df = preprocessor.load_data()

# Combine train and test for continuous time series
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df = combined_df.sort_values(['view_label', 'time_segment_id']).reset_index(drop=True)

print(f"\nCombined dataset: {len(combined_df)} segments")

# Load sample submission to see what we need to predict
sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
print(f"Sample submission: {len(sample_df)} predictions needed")

# For each required prediction, we need to:
# 1. Find the corresponding time segment
# 2. Use past 15 minutes as features
# 3. Predict 5 minutes ahead

print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

from sklearn.preprocessing import LabelEncoder

predictions = []

for idx, row in sample_df.iterrows():
    id_str = row['ID']
    
    # Parse ID to get segment and location
    # Format: time_segment_XXX_Location_congestion_enter/exit_rating
    parts = id_str.split('_')
    segment_id = int(parts[2])
    location = '_'.join(parts[3:-2])  # Handle "Norman Niles #X"
    rating_type = parts[-2]  # 'enter' or 'exit'
    
    # Find this segment in our data
    segment_data = combined_df[
        (combined_df['time_segment_id'] == segment_id) &
        (combined_df['view_label'] == location)
    ]
    
    if len(segment_data) == 0:
        # Segment not in our data - use most common class
        prediction = 'free flowing'
    else:
        # Get past 15 minutes for this location
        location_df = combined_df[combined_df['view_label'] == location].reset_index(drop=True)
        segment_idx = location_df[location_df['time_segment_id'] == segment_id].index
        
        if len(segment_idx) == 0 or segment_idx[0] < 15:
            # Not enough history
            prediction = 'free flowing'
        else:
            idx_val = segment_idx[0]
            
            # Create features (same as training)
            feature_dict = {
                'current_segment': segment_id,
                'location': location,
                'hour': str(location_df.iloc[idx_val]['datetimestamp_start']).split()[1].split(':')[0] if 'datetimestamp_start' in location_df.columns else '12',
                'signaling': location_df.iloc[idx_val]['signaling'] if 'signaling' in location_df.columns else 'none',
            }
            
            # Add lag features
            for lag in [1, 2, 3, 5, 10, 15]:
                if idx_val - lag >= 0:
                    past_row = location_df.iloc[idx_val - lag]
                    feature_dict[f'enter_lag_{lag}'] = past_row.get('congestion_enter_rating', 'free flowing')
                    feature_dict[f'exit_lag_{lag}'] = past_row.get('congestion_exit_rating', 'free flowing')
                else:
                    feature_dict[f'enter_lag_{lag}'] = 'free flowing'
                    feature_dict[f'exit_lag_{lag}'] = 'free flowing'
            
            # Encode features
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
        print(f"  Processed {idx+1}/{len(sample_df)} predictions...")

# Create submission
submission_df = sample_df.copy()
submission_df['Target'] = predictions
submission_df['Target_Accuracy'] = predictions

# Save
output_path = os.path.join(config.SUBMISSIONS_DIR, 'v2.0_forecasting.csv')
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved to {output_path}")

# Analyze predictions
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
print("This uses the CORRECT forecasting approach!")
print("Expected to perform better than v1.x versions")
print("="*60)
