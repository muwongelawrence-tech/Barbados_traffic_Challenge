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
