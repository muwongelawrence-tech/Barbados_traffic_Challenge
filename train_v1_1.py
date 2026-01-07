"""
Improved training script for v1.1
Addresses issues from v1.0:
1. Better handling of test data preprocessing
2. Class balancing with SMOTE
3. More estimators
4. Better validation strategy
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

import config
from src.data_preprocessing import DataPreprocessor

print("="*60)
print("BARBADOS TRAFFIC - v1.1 TRAINING")
print("="*60)
print("Improvements:")
print("- SMOTE for class balancing")
print("- Increased estimators (100 → 200)")
print("- Better test data handling")
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

# Save feature columns
feature_cols = list(X_train.columns)

# Fill missing values
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)

print(f"Training samples: {len(X_train)}")
print(f"Features: {len(X_train.columns)}")

# Check class distribution
print("\n" + "="*60)
print("Class Distribution (Enter Rating)")
print("="*60)
unique, counts = np.unique(y_enter_train, return_counts=True)
for u, c in zip(unique, counts):
    label = config.CONGESTION_LEVELS_REVERSE[u]
    print(f"  {label}: {c} ({c/len(y_enter_train)*100:.1f}%)")

# Apply SMOTE for class balancing
print("\n" + "="*60)
print("Applying SMOTE for class balancing...")
print("="*60)

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_enter_balanced = smote.fit_resample(X_train, y_enter_train)

print(f"Original samples: {len(X_train)}")
print(f"Balanced samples: {len(X_train_balanced)}")

print("\nBalanced distribution:")
unique, counts = np.unique(y_enter_balanced, return_counts=True)
for u, c in zip(unique, counts):
    label = config.CONGESTION_LEVELS_REVERSE[u]
    print(f"  {label}: {c} ({c/len(y_enter_balanced)*100:.1f}%)")

# Train model for ENTER rating
print("\n" + "="*60)
print("Training LightGBM model...")
print("="*60)

model_enter = LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=200,  # Increased from 100
    learning_rate=0.05,
    num_leaves=31,
    max_depth=8,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    verbose=-1
)

model_enter.fit(X_train_balanced, y_enter_balanced)

# Evaluate on validation
y_pred_val = model_enter.predict(X_val)

acc = accuracy_score(y_enter_val, y_pred_val)
f1 = f1_score(y_enter_val, y_pred_val, average='weighted')

print(f"\nValidation Performance:")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 (weighted): {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_enter_val, y_pred_val,
    target_names=list(config.CONGESTION_LEVELS.keys())
))

# Save model
os.makedirs(config.MODELS_DIR, exist_ok=True)
model_path = os.path.join(config.MODELS_DIR, 'v1.1_enter_model_smote.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model_enter, f)

print(f"\n✓ Model saved to {model_path}")

# Generate predictions for test set
print("\n" + "="*60)
print("Generating test predictions...")
print("="*60)

# Process test data properly
train_for_lags = train_df.copy()
test_for_lags = test_df.copy()
train_for_lags['is_test'] = 0
test_for_lags['is_test'] = 1

combined = pd.concat([train_for_lags, test_for_lags], ignore_index=True)
combined_processed = preprocessor.preprocess(combined, is_train=True)
test_processed = combined_processed[combined_processed['is_test'] == 1].copy()

# Get features
X_test = test_processed[feature_cols].fillna(0)

print(f"Test samples: {len(X_test)}")

# Predict
predictions_enter = model_enter.predict(X_test)

print(f"\nPrediction distribution:")
unique, counts = np.unique(predictions_enter, return_counts=True)
for u, c in zip(unique, counts):
    label = config.CONGESTION_LEVELS_REVERSE[u]
    print(f"  {label}: {c} ({c/len(predictions_enter)*100:.1f}%)")

# Create submission file matching sample format
print("\n" + "="*60)
print("Creating submission file...")
print("="*60)

# Load sample to get exact IDs needed
sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)

# Create mapping from test data
test_predictions = {}
for idx, row in test_df.iterrows():
    time_segment = row['time_segment_id']
    view_label = row['view_label']
    key = f'time_segment_{time_segment}_{view_label}_congestion_enter_rating'
    test_predictions[key] = config.CONGESTION_LEVELS_REVERSE[predictions_enter[idx]]

# Fill submission
submission_df = sample_df.copy()
for idx, row in submission_df.iterrows():
    id_str = row['ID']
    if id_str in test_predictions:
        pred = test_predictions[id_str]
        submission_df.at[idx, 'Target'] = pred
        submission_df.at[idx, 'Target_Accuracy'] = pred

# Save
output_path = os.path.join(config.SUBMISSIONS_DIR, 'v1.1_smote_balanced.csv')
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved to {output_path}")
print(f"Total predictions: {len(submission_df)}")

# Validate
print("\nSubmission validation:")
print(f"  Shape: {submission_df.shape}")
print(f"  Missing values: {submission_df.isna().sum().sum()}")
print(f"\nPrediction distribution in submission:")
print(submission_df['Target'].value_counts())

print("\n" + "="*60)
print("SUMMARY - v1.1")
print("="*60)
print(f"Validation F1: {f1:.4f}")
print(f"Estimators: 200")
print(f"Class balancing: SMOTE")
print(f"Submission: {output_path}")
print("="*60)
