"""
Create submission matching sample format exactly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
import config

print("="*60)
print("CREATING FINAL SUBMISSION")
print("="*60)

# Load sample submission to get required IDs
sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
print(f"\nSample submission has {len(sample_df)} rows")

# Load test data
test_df = pd.read_csv(config.TEST_FILE)
print(f"Test data has {len(test_df)} rows")

# Load models
print("\nLoading models...")
with open(os.path.join(config.MODELS_DIR, 'quick_enter_model.pkl'), 'rb') as f:
    model_enter = pickle.load(f)
with open(os.path.join(config.MODELS_DIR, 'quick_exit_model.pkl'), 'rb') as f:
    model_exit = pickle.load(f)

# Load preprocessor and process data
from src.data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()

train_df, _ = preprocessor.load_data()
train_processed = preprocessor.preprocess(train_df, is_train=True)

# Get feature columns from training
X_train_temp, _, _, _, _, _ = preprocessor.prepare_train_test_split(train_processed, test_size=0.2)
feature_cols = list(X_train_temp.columns)

# Process test data with training data for lag features
train_for_lags = train_df.copy()
test_for_lags = test_df.copy()
train_for_lags['is_test'] = 0
test_for_lags['is_test'] = 1

combined = pd.concat([train_for_lags, test_for_lags], ignore_index=True)
combined_processed = preprocessor.preprocess(combined, is_train=True)
test_processed = combined_processed[combined_processed['is_test'] == 1].copy()

# Get test features
X_test = test_processed[feature_cols].fillna(0)

# Generate predictions
print("\nGenerating predictions...")
predictions_enter = model_enter.predict(X_test)
predictions_exit = model_exit.predict(X_test)

# Create lookup dictionaries
enter_preds = {}
exit_preds = {}

for idx, row in test_df.iterrows():
    time_segment = row['time_segment_id']
    view_label = row['view_label']
    
    key = f'time_segment_{time_segment}_{view_label}'
    enter_preds[key + '_congestion_enter_rating'] = config.CONGESTION_LEVELS_REVERSE[predictions_enter[idx]]
    exit_preds[key + '_congestion_exit_rating'] = config.CONGESTION_LEVELS_REVERSE[predictions_exit[idx]]

# Fill in sample submission
print("\nFilling submission template...")
submission_df = sample_df.copy()

for idx, row in submission_df.iterrows():
    id_str = row['ID']
    
    if id_str in enter_preds:
        submission_df.at[idx, 'Target'] = enter_preds[id_str]
        submission_df.at[idx, 'Target_Accuracy'] = enter_preds[id_str]
    elif id_str in exit_preds:
        submission_df.at[idx, 'Target'] = exit_preds[id_str]
        submission_df.at[idx, 'Target_Accuracy'] = exit_preds[id_str]
    else:
        # Default to free flowing if not found
        submission_df.at[idx, 'Target'] = 'free flowing'
        submission_df.at[idx, 'Target_Accuracy'] = 'free flowing'

# Save submission
output_path = os.path.join(config.SUBMISSIONS_DIR, 'submission.csv')
submission_df.to_csv(output_path, index=False)

print(f"\n✅ Submission saved to: {output_path}")
print(f"Total predictions: {len(submission_df)}")

# Validate
print("\nValidation:")
print(f"  Shape matches: {submission_df.shape == sample_df.shape}")
print(f"  Columns match: {list(submission_df.columns) == list(sample_df.columns)}")
print(f"  No missing values: {not submission_df.isna().any().any()}")

print("\nPrediction distribution:")
print(submission_df['Target'].value_counts())

print("\n" + "="*60)
print("✅ SUBMISSION READY!")
print("="*60)
print(f"File: {output_path}")
print("You can now upload this to Zindi!")
