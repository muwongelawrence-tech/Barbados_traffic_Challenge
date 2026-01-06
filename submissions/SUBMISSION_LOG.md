# Submission Log

Track all submissions to Zindi platform for the Barbados Traffic Analysis Challenge.

## Naming Convention

`submission_v{N}_{description}.csv`

Example: `submission_v1_baseline_temporal_only.csv`

---

## Submission History

### v1 - Baseline (Temporal Features Only)
- **File:** `submission_v1_baseline_temporal_only.csv`
- **Date:** 2026-01-06
- **Model:** LightGBM baseline
- **Features:** Temporal features only (hour, minute, cyclical encoding, rush hour)
- **Validation Score:** 0.5745
- **Leaderboard Score:** _Pending upload_
- **Notes:** First corrected submission with entrance predictions only (2,640 rows)

### v2 - Increased Complexity
- **File:** `submission_v2_increased_complexity.csv`
- **Date:** 2026-01-06
- **Model:** LightGBM with increased complexity
- **Features:** Temporal features only
- **Hyperparameters:** 
  - `num_leaves=50` (vs 31 baseline)
  - `learning_rate=0.03` (vs 0.05 baseline)
  - `max_depth=8`
  - `min_child_samples=10`
- **Validation Score:** 0.5324
- **Leaderboard Score:** _Pending upload_
- **Notes:** Higher complexity model, scored lower on validation

---

## Next Submissions (Planned)

### Phase 1: Quick Wins (Target: 0.60-0.63)
- [ ] v3: Lower learning rate (lr=0.01, 1000 rounds)
- [ ] v4: Regularization (L1/L2)
- [ ] v5: Class weights (expected best performer)
- [ ] v6: Deep trees (100 leaves)
- [ ] v7: XGBoost alternative
- [ ] v8: CatBoost alternative

### Phase 2: Video Features (Target: 0.65-0.68)
- [ ] v9+: Classical CV features (vehicle counting, optical flow, spatial analysis)

### Phase 3: Advanced (Target: 0.70-0.72+)
- [ ] Ensemble methods
- [ ] Bayesian optimization
- [ ] Pseudo-labeling

---

## Submission Checklist

Before uploading to Zindi:
- [ ] Verify file has 2,640 rows (entrance predictions only)
- [ ] Verify no exit IDs in submission
- [ ] Verify columns: ID, Target, Target_Accuracy
- [ ] Verify all class names are valid
- [ ] Record validation score
- [ ] Update this log with leaderboard score after upload

---

## Best Submission Tracker

| Rank | Version | Leaderboard Score | Date |
|------|---------|-------------------|------|
| 1 | _TBD_ | _TBD_ | - |

---

**Last Updated:** 2026-01-06
