"""
Quick training script for first submission
Simplified version for faster execution
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

import config
from src.data_preprocessing import DataPreprocessor

print("="*60)
print("BARBADOS TRAFFIC - QUICK TRAINING")
print("="*60)

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load and preprocess data
print("\nLoading data...")
train_df, test_df = preprocessor.load_data()
train_processed = preprocessor.preprocess(train_df, is_train=True)

# Prepare train/val split
X_train, X_val, y_enter_train, y_enter_val, y_exit_train, y_exit_val = \
    preprocessor.prepare_train_test_split(train_processed, test_size=0.2)

# Save feature columns for later use
feature_cols = list(X_train.columns)

# Fill missing values
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)

print(f"Training samples: {len(X_train)}")
print(f"Features: {len(X_train.columns)}")

# Train model for ENTER rating
print("\n" + "="*60)
print("Training model for ENTER rating...")
print("="*60)

model_enter = LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

model_enter.fit(X_train, y_enter_train)
y_pred_enter = model_enter.predict(X_val)

acc_enter = accuracy_score(y_enter_val, y_pred_enter)
f1_enter = f1_score(y_enter_val, y_pred_enter, average='weighted')

print(f"Validation Accuracy: {acc_enter:.4f}")
print(f"Validation F1 (weighted): {f1_enter:.4f}")

# Train model for EXIT rating
print("\n" + "="*60)
print("Training model for EXIT rating...")
print("="*60)

model_exit = LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

model_exit.fit(X_train, y_exit_train)
y_pred_exit = model_exit.predict(X_val)

acc_exit = accuracy_score(y_exit_val, y_pred_exit)
f1_exit = f1_score(y_exit_val, y_pred_exit, average='weighted')

print(f"Validation Accuracy: {acc_exit:.4f}")
print(f"Validation F1 (weighted): {f1_exit:.4f}")

# Save models
os.makedirs(config.MODELS_DIR, exist_ok=True)

with open(os.path.join(config.MODELS_DIR, 'quick_enter_model.pkl'), 'wb') as f:
    pickle.dump(model_enter, f)

with open(os.path.join(config.MODELS_DIR, 'quick_exit_model.pkl'), 'wb') as f:
    pickle.dump(model_exit, f)

print(f"\n✅ Models saved to {config.MODELS_DIR}")

# Generate predictions for test set
print("\n" + "="*60)
print("Generating test predictions...")
print("="*60)

# Need to process test data with training data to get proper lag features
# Create a combined dataset
train_for_lags = train_df.copy()
test_for_lags = test_df.copy()

# Mark which is which
train_for_lags['is_test'] = 0
test_for_lags['is_test'] = 1

# Combine
combined = pd.concat([train_for_lags, test_for_lags], ignore_index=True)

# Process combined (this will create lag features based on full history)
combined_processed = preprocessor.preprocess(combined, is_train=True)

# Split back
test_processed = combined_processed[combined_processed['is_test'] == 1].copy()

# Get features (same as training)
X_test = test_processed[feature_cols].fillna(0)

print(f"Test features shape: {X_test.shape}")
print(f"Training features shape: {X_train.shape}")

# Predict
predictions_enter = model_enter.predict(X_test)
predictions_exit = model_exit.predict(X_test)

print(f"Generated {len(predictions_enter)} predictions for each target")

# Create submission file
print("\n" + "="*60)
print("Creating submission file...")
print("="*60)

submission_rows = []

for idx, row in test_df.iterrows():
    time_segment = row['time_segment_id']
    view_label = row['view_label']
    
    # Entry rating
    enter_pred = config.CONGESTION_LEVELS_REVERSE[predictions_enter[idx]]
    submission_rows.append({
        'ID': f'time_segment_{time_segment}_{view_label}_congestion_enter_rating',
        'Target': enter_pred,
        'Target_Accuracy': enter_pred
    })
    
    # Exit rating
    exit_pred = config.CONGESTION_LEVELS_REVERSE[predictions_exit[idx]]
    submission_rows.append({
        'ID': f'time_segment_{time_segment}_{view_label}_congestion_exit_rating',
        'Target': exit_pred,
        'Target_Accuracy': exit_pred
    })

submission_df = pd.DataFrame(submission_rows)

# Save submission
os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)
submission_path = os.path.join(config.SUBMISSIONS_DIR, 'first_submission.csv')
submission_df.to_csv(submission_path, index=False)

print(f"\n✅ Submission file created: {submission_path}")
print(f"Total predictions: {len(submission_df)}")

# Validate format
sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
print(f"\nValidation:")
print(f"  Expected rows: {len(sample_df)}, Got: {len(submission_df)}")
print(f"  Expected columns: {sample_df.columns.tolist()}")
print(f"  Got columns: {submission_df.columns.tolist()}")

if len(submission_df) == len(sample_df) and set(submission_df.columns) == set(sample_df.columns):
    print(f"\n✅ Submission format is valid!")
else:
    print(f"\n⚠️  Submission format may have issues")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Enter Rating - Val F1: {f1_enter:.4f}")
print(f"Exit Rating  - Val F1: {f1_exit:.4f}")
print(f"Average F1: {(f1_enter + f1_exit) / 2:.4f}")
print(f"\nSubmission ready at: {submission_path}")
print("="*60)
