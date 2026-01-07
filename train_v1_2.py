"""
Training script for v1.2 - Conservative Predictions
Strategy:
1. NO SMOTE - respect natural distribution
2. Use class weights to handle imbalance
3. Predict delays only when highly confident
4. Bias toward "free flowing" (the majority class)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle
import warnings
warnings.filterwarnings('ignore')

import config
from src.data_preprocessing import DataPreprocessor

print("="*60)
print("BARBADOS TRAFFIC - v1.2 TRAINING")
print("="*60)
print("Strategy: Conservative Predictions")
print("- NO SMOTE (respect natural distribution)")
print("- Class weights for imbalance")
print("- High confidence threshold for delays")
print("- More estimators (300)")
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
print("Class Distribution (Natural - No SMOTE)")
print("="*60)
unique, counts = np.unique(y_enter_train, return_counts=True)
for u, c in zip(unique, counts):
    label = config.CONGESTION_LEVELS_REVERSE[u]
    print(f"  {label}: {c} ({c/len(y_enter_train)*100:.1f}%)")

# Calculate class weights (but make them less aggressive)
print("\n" + "="*60)
print("Calculating Conservative Class Weights...")
print("="*60)

# Standard balanced weights
standard_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_enter_train),
    y=y_enter_train
)

# Make weights less aggressive (reduce minority class weights)
conservative_weights = standard_weights.copy()
conservative_weights[0] = 1.0  # Free flowing gets weight 1.0
conservative_weights[1:] *= 0.3  # Reduce minority class weights by 70%

weight_dict = {i: w for i, w in enumerate(conservative_weights)}

print("Standard balanced weights:")
for i, w in enumerate(standard_weights):
    label = config.CONGESTION_LEVELS_REVERSE[i]
    print(f"  {label}: {w:.3f}")

print("\nConservative weights (70% reduction for minorities):")
for i, w in enumerate(conservative_weights):
    label = config.CONGESTION_LEVELS_REVERSE[i]
    print(f"  {label}: {w:.3f}")

# Train model
print("\n" + "="*60)
print("Training LightGBM with Conservative Weights...")
print("="*60)

model_enter = LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=300,  # Increased from 200
    learning_rate=0.03,  # Slightly lower for better generalization
    num_leaves=31,
    max_depth=6,  # Reduced from 8 to prevent overfitting
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,  # Prevent overfitting on minority classes
    random_state=42,
    verbose=-1,
    class_weight=weight_dict
)

model_enter.fit(X_train, y_enter_train)

# Evaluate with standard predictions
y_pred_val_standard = model_enter.predict(X_val)

acc_standard = accuracy_score(y_enter_val, y_pred_val_standard)
f1_standard = f1_score(y_enter_val, y_pred_val_standard, average='weighted')

print(f"\nStandard Predictions (argmax):")
print(f"  Accuracy: {acc_standard:.4f}")
print(f"  F1 (weighted): {f1_standard:.4f}")

# Get probability predictions
y_proba_val = model_enter.predict_proba(X_val)

# Apply conservative threshold: only predict delay if very confident
print("\n" + "="*60)
print("Applying Conservative Threshold Strategy...")
print("="*60)

def conservative_predict(probas, threshold=0.75):
    """
    Predict conservatively:
    - Predict 'free flowing' (class 0) unless very confident about delay
    - Require threshold probability for delay classes
    """
    predictions = []
    for proba in probas:
        # If free flowing probability > 0.5, predict it
        if proba[0] > 0.5:
            predictions.append(0)
        # Otherwise, only predict delay if very confident
        else:
            max_delay_idx = np.argmax(proba[1:]) + 1  # Best delay class
            max_delay_prob = proba[max_delay_idx]
            
            if max_delay_prob > threshold:
                predictions.append(max_delay_idx)
            else:
                predictions.append(0)  # Default to free flowing
    
    return np.array(predictions)

# Try different thresholds
thresholds = [0.6, 0.7, 0.75, 0.8, 0.85]
best_threshold = 0.75
best_f1 = 0

print("\nTesting different confidence thresholds:")
for threshold in thresholds:
    y_pred_conservative = conservative_predict(y_proba_val, threshold)
    f1 = f1_score(y_enter_val, y_pred_conservative, average='weighted')
    
    # Count predictions
    unique, counts = np.unique(y_pred_conservative, return_counts=True)
    free_flowing_pct = counts[0] / len(y_pred_conservative) * 100 if 0 in unique else 0
    
    print(f"  Threshold {threshold:.2f}: F1={f1:.4f}, Free flowing={free_flowing_pct:.1f}%")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n✓ Best threshold: {best_threshold} (F1: {best_f1:.4f})")

# Use best threshold for final predictions
y_pred_val = conservative_predict(y_proba_val, best_threshold)

acc = accuracy_score(y_enter_val, y_pred_val)
f1 = f1_score(y_enter_val, y_pred_val, average='weighted')

print(f"\nFinal Validation Performance:")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 (weighted): {f1:.4f}")

print("\nValidation Prediction Distribution:")
unique, counts = np.unique(y_pred_val, return_counts=True)
for u, c in zip(unique, counts):
    label = config.CONGESTION_LEVELS_REVERSE[u]
    print(f"  {label}: {c} ({c/len(y_pred_val)*100:.1f}%)")

print("\nClassification Report:")
print(classification_report(
    y_enter_val, y_pred_val,
    target_names=list(config.CONGESTION_LEVELS.keys()),
    zero_division=0
))

# Save model and threshold
os.makedirs(config.MODELS_DIR, exist_ok=True)
model_path = os.path.join(config.MODELS_DIR, 'v1.2_conservative_model.pkl')

model_data = {
    'model': model_enter,
    'threshold': best_threshold,
    'feature_cols': feature_cols
}

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to {model_path}")

# Generate predictions for test set
print("\n" + "="*60)
print("Generating Conservative Test Predictions...")
print("="*60)

# Process test data
train_for_lags = train_df.copy()
test_for_lags = test_df.copy()
train_for_lags['is_test'] = 0
test_for_lags['is_test'] = 1

combined = pd.concat([train_for_lags, test_for_lags], ignore_index=True)
combined_processed = preprocessor.preprocess(combined, is_train=True)
test_processed = combined_processed[combined_processed['is_test'] == 1].copy()

X_test = test_processed[feature_cols].fillna(0)

# Get probabilities
test_proba = model_enter.predict_proba(X_test)

# Apply conservative threshold
predictions_enter = conservative_predict(test_proba, best_threshold)

print(f"\nTest Prediction Distribution:")
unique, counts = np.unique(predictions_enter, return_counts=True)
for u, c in zip(unique, counts):
    label = config.CONGESTION_LEVELS_REVERSE[u]
    print(f"  {label}: {c} ({c/len(predictions_enter)*100:.1f}%)")

# Create submission
print("\n" + "="*60)
print("Creating Submission File...")
print("="*60)

sample_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)

test_predictions = {}
for idx, row in test_df.iterrows():
    time_segment = row['time_segment_id']
    view_label = row['view_label']
    key = f'time_segment_{time_segment}_{view_label}_congestion_enter_rating'
    test_predictions[key] = config.CONGESTION_LEVELS_REVERSE[predictions_enter[idx]]

submission_df = sample_df.copy()
for idx, row in submission_df.iterrows():
    id_str = row['ID']
    if id_str in test_predictions:
        pred = test_predictions[id_str]
        submission_df.at[idx, 'Target'] = pred
        submission_df.at[idx, 'Target_Accuracy'] = pred

output_path = os.path.join(config.SUBMISSIONS_DIR, 'v1.2_conservative.csv')
submission_df.to_csv(output_path, index=False)

print(f"\n✓ Submission saved to {output_path}")
print(f"Total predictions: {len(submission_df)}")

print("\nSubmission Distribution:")
print(submission_df['Target'].value_counts())

print("\n" + "="*60)
print("SUMMARY - v1.2")
print("="*60)
print(f"Validation F1: {f1:.4f}")
print(f"Strategy: Conservative (threshold={best_threshold})")
print(f"Estimators: 300")
print(f"Class weights: Conservative (30% of balanced)")
print(f"Submission: {output_path}")
print("\nExpected improvement over v1.0 (0.4612): +0.03 to +0.08")
print("="*60)
