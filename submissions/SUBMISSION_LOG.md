# Submission Tracking

## Naming Convention

All submissions follow this format:
```
submission_v{VERSION}_{DESCRIPTION}.csv
```

Examples:
- `submission_v1_baseline_temporal_only.csv`
- `submission_v2_tuned_hyperparams.csv`
- `submission_v3_with_video_features.csv`
- `submission_v4_ensemble_lgb_xgb.csv`

---

## Submission Log

| Version | Date | Time | Description | Score | F1 | Accuracy | Notes |
|---------|------|------|-------------|-------|----|---------|----|
| v1 | 2026-01-06 | 10:27 | Baseline - Temporal features only | TBD | 0.5659* | 0.7307* | *Validation scores. LightGBM, 17 features, no video processing |
| v2 | - | - | - | - | - | - | - |
| v3 | - | - | - | - | - | - | - |

*Validation scores (not leaderboard scores)

---

## Submission History

### v1 - Baseline (Temporal Features Only)

**File:** `submission_v1_baseline_temporal_only.csv`

**Model:** LightGBM  
**Features:** 17 temporal features
- Hour, minute, day of week
- Cyclical encoding (sin/cos)
- Rush hour indicators
- Camera ID
- Turn signal usage
- Time segment ID

**Performance (Validation):**
- Entrance: F1 = 0.5659, Acc = 0.7307
- Exit: F1 = 0.3545, Acc = 0.9515
- Combined: 0.5745

**Leaderboard Score:** _[Update after submission]_

**What worked:**
- Hour of day is strongest predictor
- Turn signal usage is important
- Fast training (2 minutes)

**What didn't work:**
- Struggles with minority classes
- Exit congestion prediction poor (class imbalance)
- No visual information

**Next steps:**
- Add video features (vehicle counts, optical flow)
- Handle class imbalance
- Hyperparameter tuning

---

### v2 - [Next Submission]

**File:** `submission_v2_[description].csv`

**Changes from v1:**
- TBD

**Performance:**
- TBD

---

## Improvement Ideas Backlog

### High Priority
- [ ] Add classical CV features from videos
  - Vehicle counts (entrance/exit rates)
  - Queue length estimation
  - Optical flow magnitude
  - Spatial occupancy
- [ ] Handle class imbalance
  - Class weights in LightGBM
  - SMOTE for minority classes
  - Focal loss

### Medium Priority
- [ ] Hyperparameter tuning
  - Grid search on num_leaves, learning_rate
  - Bayesian optimization
- [ ] Ensemble methods
  - LightGBM + XGBoost + CatBoost
  - Weighted averaging
- [ ] Advanced temporal features
  - Lag features (previous segments)
  - Rolling statistics
  - Rate of change

### Low Priority
- [ ] Feature engineering
  - Interaction features
  - Polynomial features
  - Camera-specific patterns
- [ ] Cross-validation
  - Time-series CV
  - Stratified K-fold

---

## Best Practices

### Before Each Submission:
1. ✅ Train model and evaluate on validation set
2. ✅ Document changes from previous version
3. ✅ Generate submission file
4. ✅ Validate format (3 columns, correct classes)
5. ✅ Name file with version and description
6. ✅ Save to `submissions/` folder
7. ✅ Update this log with validation scores
8. ✅ Upload to Zindi
9. ✅ Record leaderboard score in this log

### After Each Submission:
1. Analyze what worked / didn't work
2. Update "Next steps" section
3. Plan improvements for next version
4. Keep best model in `models/best/`

---

## Quick Commands

```bash
# List all submissions
ls -lh submissions/

# Count submissions
ls submissions/ | wc -l

# View latest submission
tail -20 submissions/submission_v*.csv | head -20

# Compare two submissions
diff submissions/submission_v1_*.csv submissions/submission_v2_*.csv | head -20
```

---

## Submission Limits

- **Daily limit:** 20 submissions
- **Total limit:** Check competition rules
- **Strategy:** Use validation scores to filter - only submit when improvement is expected

---

## Version Control

All submission files are tracked in git:
```bash
git add submissions/
git commit -m "Add submission vX: [description]"
```

Keep models corresponding to each submission:
```
models/
├── v1_baseline/
├── v2_tuned/
├── v3_video_features/
└── best/  # Copy of best performing model
```
