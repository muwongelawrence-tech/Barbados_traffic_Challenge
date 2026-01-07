# Barbados Traffic Challenge - Experiment Log

## Overview

This document tracks all experiments, model iterations, and their performance on the Zindi leaderboard.

**Goal**: Beat current best score and reach top of leaderboard

**Current Best Score**: 0.461234377 (Baseline - Submission v1.0)

---

## Submission History

### v1.0 - Baseline LightGBM (2026-01-07)

**Public Score**: 0.461234377  
**Private Score**: TBD  
**Rank**: TBD

**Model Details**:
- Algorithm: LightGBM Classifier
- Features: 37 engineered features
- Training samples: 12,860
- Validation samples: 3,216

**Features**:
- Temporal: hour, minute, day_of_week, is_weekend, is_rush_hour
- Cyclical: hour_sin/cos, day_sin/cos
- Lag features: 1, 2, 3, 5, 10 time segments
- Rolling statistics: mean/std over 5, 10, 20 windows
- Location encoding
- Signal encoding

**Hyperparameters**:
```python
{
    'objective': 'multiclass',
    'num_class': 4,
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
}
```

**Validation Performance**:
- Enter Rating F1: 0.8293
- Exit Rating F1: 0.9601
- Average F1: 0.8947

**Observations**:
- High validation F1 (0.8947) but lower public score (0.4612)
- Significant gap suggests:
  1. Possible overfitting to validation set
  2. Test distribution differs from train
  3. All predictions defaulting to "free flowing" for certain segments
  4. Need to investigate prediction distribution

**Next Steps**:
1. Analyze prediction distribution on test set
2. Implement proper cross-validation (TimeSeriesSplit)
3. Address class imbalance with SMOTE
4. Try ensemble methods
5. Increase model complexity (more estimators)

**Files**:
- Model: `models/quick_enter_model.pkl`, `models/quick_exit_model.pkl`
- Submission: `submissions/v1.0_baseline_lightgbm.csv`
- Code: `quick_train.py`, `create_submission.py`

---

### v1.1 - SMOTE Class Balancing (2026-01-07) ❌ FAILED

**Public Score**: 0.302442736 ⬇️ (-34.4% from v1.0)  
**Private Score**: TBD  
**Rank**: Worse than v1.0

**Status**: ❌ **FAILED - Score decreased significantly**

**Model Details**:
- Algorithm: LightGBM Classifier with SMOTE
- Features: 37 engineered features (same as v1.0)
- Training samples: 33,348 (after SMOTE from 12,860)
- Validation samples: 3,216

**Key Changes**:
- ✅ Applied SMOTE for class balancing (25% each class)
- ✅ Increased estimators: 100 → 200
- ✅ Added max_depth: 8
- ✅ Better test data preprocessing

**Hyperparameters**:
```python
{
    'objective': 'multiclass',
    'num_class': 4,
    'n_estimators': 200,  # Increased
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 8,  # Added
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
}
```

**Validation Performance**:
- Accuracy: 0.8305
- F1 (weighted): 0.8278
- Per-class F1: free flowing (0.94), light (0.59), moderate (0.72), heavy (0.76)

**Test Predictions**:
- free flowing: 64.3%
- moderate delay: 16.4%
- light delay: 10.3%
- heavy delay: 9.1%

**Observations**:
- ❌ **MAJOR FAILURE**: Score dropped 34.4%
- ❌ SMOTE caused model to predict 50% "heavy delay"
- ❌ Test set is likely imbalanced (like training: ~95% free flowing)
- ❌ Predicting delays when there aren't any = many false positives
- ✅ Model predictions were diverse (as intended)
- ✅ But diversity hurt performance on imbalanced test set

**Root Cause**:
SMOTE balanced the TRAINING data but test data remained imbalanced. Model learned to predict all classes equally, but test set is ~95% "free flowing". Result: massive false positive rate on delays.

**Key Insight**:
v1.0's "all free flowing" predictions were actually CLOSER to the truth than v1.1's balanced predictions!

**Lessons Learned**:
1. ❌ Don't use SMOTE when test set is imbalanced
2. ❌ Forcing diversity in predictions can hurt if test is imbalanced  
3. ✅ Respect natural class distribution
4. ✅ Simple baselines (predict most common) can be hard to beat
5. ✅ Use class weights instead of resampling for imbalanced data

**Expected Improvement**: +0.05 to +0.10 from v1.0  
**Actual Result**: -0.16 from v1.0 ❌

**Files**:
- Model: `models/v1.1_enter_model_smote.pkl`
- Submission: `submissions/v1.1_smote_balanced.csv`
- Code: `train_v1_1.py`
- Documentation: `docs/experiments/v1.1_smote_balancing.md`
- Post-mortem: `docs/analysis/v1.1_postmortem.md`

---

### v1.2 - Conservative Predictions (2026-01-07)

**Public Score**: TBD (Pending submission)  
**Private Score**: TBD  
**Rank**: TBD

**Model Details**:
- Algorithm: LightGBM with conservative class weights
- Features: 37 engineered features (same as v1.0/v1.1)
- Training samples: 12,860 (NO SMOTE - natural distribution)
- Validation samples: 3,216

**Key Changes from v1.1**:
- ❌ NO SMOTE - respect natural distribution
- ✅ Conservative class weights (70% reduction from balanced)
- ✅ Probability thresholding (threshold=0.6)
- ✅ More estimators: 200 → 300
- ✅ Lower learning rate: 0.05 → 0.03

**Hyperparameters**:
```python
{
    'objective': 'multiclass',
    'num_class': 4,
    'n_estimators': 300,
    'learning_rate': 0.03,  # Reduced
    'max_depth': 6,  # Reduced from 8
    'min_child_samples': 20,  # Added
    'class_weight': conservative_weights  # Custom
}
```

**Validation Performance**:
- Accuracy: 0.7422 (lower due to conservative strategy)
- F1 (weighted): 0.7133
- Free flowing recall: **0.99** (catches almost all!)
- Delay precision: 0.83-0.86 (high confidence when predicting delays)

**Test Predictions**:
- free flowing: 83.8% ← Much better than v1.1's 50%!
- heavy delay: 5.9%
- moderate delay: 6.7%
- light delay: 3.6%

**Strategy**:
Conservative prediction with probability thresholding:
1. If P(free flowing) > 0.5 → predict free flowing
2. Else: only predict delay if P(delay) > 0.6
3. Otherwise → default to free flowing

**Observations**:
- ✅ Much more conservative than v1.1 (84% vs 50% free flowing)
- ✅ Respects natural distribution (training: 65% free flowing)
- ✅ High recall on majority class (99%)
- ✅ High precision on delays (83-86%)
- ⚠️ Lower validation F1 (expected with conservative strategy)

**Expected Improvement**: +0.03 to +0.08 from v1.0 (target: 0.49-0.54)

**Hypothesis**: Should perform better than both v1.0 and v1.1 by:
- Predicting mostly free flowing (like v1.0)
- But catching some delays correctly (unlike v1.0)
- Without false positives (unlike v1.1)

**Files**:
- Model: `models/v1.2_conservative_model.pkl`
- Submission: `submissions/v1.2_conservative.csv`
- Code: `train_v1_2.py`
- Documentation: `docs/experiments/v1.2_conservative.md`

---

## Planned Experiments

### v1.1 - Improved Feature Engineering
- [ ] Add more lag features (15, 20, 30 segments)
- [ ] Create location-specific features
- [ ] Add day-of-week × hour interaction features
- [ ] Implement exponentially weighted moving averages

### v1.2 - Hyperparameter Tuning
- [ ] Increase n_estimators to 500
- [ ] Grid search for optimal learning_rate
- [ ] Tune num_leaves and max_depth
- [ ] Optimize feature_fraction and bagging_fraction

### v1.3 - Class Imbalance Handling
- [ ] Implement SMOTE for minority classes
- [ ] Try different class weighting strategies
- [ ] Use focal loss

### v1.4 - Ensemble Models
- [ ] Train XGBoost model
- [ ] Train CatBoost model
- [ ] Create voting ensemble
- [ ] Try stacking with meta-learner

### v2.0 - Advanced Techniques
- [ ] Implement proper time-series CV
- [ ] Add video feature extraction (if feasible)
- [ ] Try neural networks (LSTM/Transformer)
- [ ] Implement pseudo-labeling

---

## Performance Tracking

| Version | Public Score | Private Score | Rank | Improvement | Date |
|---------|-------------|---------------|------|-------------|------|
| v1.0    | 0.4612      | TBD          | TBD  | Baseline    | 2026-01-07 |
| v1.1    | -           | -            | -    | -           | - |
| v1.2    | -           | -            | -    | -           | - |

---

## Key Learnings

### What Worked
- Time-based features (hour, day_of_week) are important
- Lag features capture temporal dependencies
- Separate models for enter/exit ratings

### What Didn't Work
- TBD

### Insights
- Large gap between validation (0.8947) and public score (0.4612)
- Need to investigate test set predictions
- May need different validation strategy

---

## Notes

- Competition metric: TBD (likely F1-score or accuracy)
- Target: Beat 0.653151551 (mentioned earlier)
- Current best on leaderboard: TBD
