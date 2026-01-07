"""
Deep dive analysis of v1.1 performance drop
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import config

print("="*60)
print("DIAGNOSING v1.1 PERFORMANCE DROP")
print("="*60)
print("v1.0 Score: 0.4612")
print("v1.1 Score: 0.3024")
print("Drop: -0.1588 (-34.4%)")
print("="*60)

# Load submissions
v1_0 = pd.read_csv('submissions/v1.0_baseline_lightgbm.csv')
v1_1 = pd.read_csv('submissions/v1.1_smote_balanced.csv')
sample = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)

print("\n1. PREDICTION DISTRIBUTION COMPARISON")
print("-"*60)

print("\nv1.0 (Score: 0.4612):")
print(v1_0['Target'].value_counts())

print("\nv1.1 (Score: 0.3024):")
print(v1_1['Target'].value_counts())

print("\nSample Submission (Ground Truth Pattern):")
print(sample['Target'].value_counts())

# Hypothesis 1: SMOTE caused overfitting
print("\n" + "="*60)
print("2. HYPOTHESIS: SMOTE CAUSED OVERFITTING")
print("="*60)

print("\nSMOTE creates synthetic samples which may:")
print("- Overfit to training distribution")
print("- Not generalize to test set")
print("- Create unrealistic decision boundaries")

# Hypothesis 2: Wrong predictions for specific segments
print("\n" + "="*60)
print("3. HYPOTHESIS: PREDICTING WRONG CLASSES")
print("="*60)

# Check agreement with sample
agreement_v1_0 = (v1_0['Target'] == sample['Target']).sum()
agreement_v1_1 = (v1_1['Target'] == sample['Target']).sum()

print(f"\nAgreement with sample submission:")
print(f"v1.0: {agreement_v1_0}/{len(sample)} ({agreement_v1_0/len(sample)*100:.1f}%)")
print(f"v1.1: {agreement_v1_1}/{len(sample)} ({agreement_v1_1/len(sample)*100:.1f}%)")

# Hypothesis 3: Sample submission is just a template
print("\n" + "="*60)
print("4. HYPOTHESIS: SAMPLE IS TEMPLATE, NOT GROUND TRUTH")
print("="*60)

# Check if sample has alternating pattern
alternating = all(
    sample.iloc[i]['Target'] != sample.iloc[i+1]['Target']
    for i in range(len(sample)-1)
)

print(f"\nSample has alternating pattern: {alternating}")

if alternating:
    print("⚠️  Sample submission is likely just a template!")
    print("   Real ground truth is unknown")
    print("   Can't use sample for comparison")

# Hypothesis 4: Model predicting too much diversity
print("\n" + "="*60)
print("5. HYPOTHESIS: TOO MUCH DIVERSITY HURTS")
print("="*60)

print("\nPrediction diversity:")
print(f"v1.0: {v1_0['Target'].nunique()} unique classes")
print(f"v1.1: {v1_1['Target'].nunique()} unique classes")

print("\nIf test set is mostly 'free flowing' (like training):")
print("- v1.0 predicting all 'free flowing' = HIGH accuracy")
print("- v1.1 predicting diverse classes = LOWER accuracy")

# Check what v1.1 is predicting vs v1.0
print("\n" + "="*60)
print("6. WHAT CHANGED IN v1.1?")
print("="*60)

# Count how many changed from "free flowing" to something else
changed_from_free = 0
for i in range(len(v1_0)):
    if v1_0.iloc[i]['Target'] == 'free flowing' and v1_1.iloc[i]['Target'] != 'free flowing':
        changed_from_free += 1

print(f"\nPredictions that changed from 'free flowing' to other:")
print(f"{changed_from_free}/{len(v1_0)} ({changed_from_free/len(v1_0)*100:.1f}%)")

print("\nv1.1 predictions breakdown:")
v1_1_dist = v1_1['Target'].value_counts()
for label, count in v1_1_dist.items():
    print(f"  {label}: {count} ({count/len(v1_1)*100:.1f}%)")

# Key insight
print("\n" + "="*60)
print("KEY INSIGHT")
print("="*60)

print("\n⚠️  SMOTE made the model predict 'heavy delay' for 50% of samples")
print("   But the test set is likely mostly 'free flowing'")
print("   This caused the score to DROP significantly")

print("\n💡 LESSON LEARNED:")
print("   - SMOTE balances TRAINING data, not test data")
print("   - Test distribution likely matches training (95% free flowing)")
print("   - Predicting diverse classes when test is imbalanced = BAD")
print("   - v1.0's 'all free flowing' was actually closer to truth!")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

print("\n1. REVERT to v1.0 approach (no SMOTE)")
print("2. Instead of balancing, use:")
print("   - Class weights (penalize errors on minority classes)")
print("   - Focal loss (focus on hard examples)")
print("   - Calibrated probability thresholds")
print("3. Accept that test set is imbalanced like training")
print("4. Focus on predicting 'free flowing' accurately")
print("5. Only predict delays when VERY confident")

print("\n" + "="*60)
