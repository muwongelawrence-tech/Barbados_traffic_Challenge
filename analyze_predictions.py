"""
Analyze predictions from v1.0 submission
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import config

print("="*60)
print("ANALYZING v1.0 PREDICTIONS")
print("="*60)

# Load submission
submission = pd.read_csv('submissions/v1.0_baseline_lightgbm.csv')
print(f"\nSubmission shape: {submission.shape}")

# Analyze prediction distribution
print("\n" + "="*60)
print("PREDICTION DISTRIBUTION")
print("="*60)

print("\nOverall:")
print(submission['Target'].value_counts())
print(f"\nPercentages:")
print(submission['Target'].value_counts(normalize=True) * 100)

# Separate enter vs exit
enter_preds = submission[submission['ID'].str.contains('enter')]
exit_preds = submission[submission['ID'].str.contains('exit')]

print("\n" + "="*60)
print("ENTER RATING PREDICTIONS")
print("="*60)
print(enter_preds['Target'].value_counts())
print(f"\nPercentages:")
print(enter_preds['Target'].value_counts(normalize=True) * 100)

print("\n" + "="*60)
print("EXIT RATING PREDICTIONS")
print("="*60)
print(exit_preds['Target'].value_counts())
print(f"\nPercentages:")
print(exit_preds['Target'].value_counts(normalize=True) * 100)

# Compare with training distribution
print("\n" + "="*60)
print("COMPARISON WITH TRAINING DATA")
print("="*60)

train = pd.read_csv(config.TRAIN_FILE)

print("\nTraining Enter Rating:")
print(train['congestion_enter_rating'].value_counts(normalize=True) * 100)

print("\nTraining Exit Rating:")
print(train['congestion_exit_rating'].value_counts(normalize=True) * 100)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Enter predictions
enter_preds['Target'].value_counts().plot(kind='bar', ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Enter Rating Predictions (Test)')
axes[0, 0].set_xlabel('Congestion Level')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=45)

# Exit predictions
exit_preds['Target'].value_counts().plot(kind='bar', ax=axes[0, 1], color='coral')
axes[0, 1].set_title('Exit Rating Predictions (Test)')
axes[0, 1].set_xlabel('Congestion Level')
axes[0, 1].set_ylabel('Count')
axes[0, 1].tick_params(axis='x', rotation=45)

# Training enter
train['congestion_enter_rating'].value_counts().plot(kind='bar', ax=axes[1, 0], color='lightblue')
axes[1, 0].set_title('Enter Rating Distribution (Train)')
axes[1, 0].set_xlabel('Congestion Level')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# Training exit
train['congestion_exit_rating'].value_counts().plot(kind='bar', ax=axes[1, 1], color='lightsalmon')
axes[1, 1].set_title('Exit Rating Distribution (Train)')
axes[1, 1].set_xlabel('Congestion Level')
axes[1, 1].set_ylabel('Count')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('docs/analysis/v1.0_prediction_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to docs/analysis/v1.0_prediction_analysis.png")

# Key insights
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

if submission['Target'].nunique() == 1:
    print("⚠️  WARNING: All predictions are the same class!")
    print(f"   All predictions: {submission['Target'].iloc[0]}")
    print("   This explains the low public score.")
    print("\n   ISSUE: Model is predicting only 'free flowing'")
    print("   CAUSE: Likely due to:")
    print("   1. Test time segments not in training data")
    print("   2. Lag features all NaN for test data")
    print("   3. Default fallback to most common class")
else:
    print("✓ Model is predicting multiple classes")
    
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
print("1. Fix test data preprocessing to properly create lag features")
print("2. Ensure test data uses training history for lag calculation")
print("3. Implement proper handling of missing lag features")
print("4. Add class balancing (SMOTE) for exit ratings")
print("5. Implement cross-validation to better estimate performance")
