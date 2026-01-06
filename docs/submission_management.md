# Submission Management Guide

## Quick Start

### Create a New Submission
```bash
# After training a model and generating submission.csv
python scripts/manage_submissions.py create "description_of_changes" submission.csv 0.5745

# Example:
python scripts/manage_submissions.py create "baseline_temporal_only" submission.csv 0.5745
python scripts/manage_submissions.py create "tuned_hyperparams" submission.csv 0.6234
python scripts/manage_submissions.py create "with_video_features" submission.csv 0.7123
```

### List All Submissions
```bash
python scripts/manage_submissions.py list
```

---

## Workflow for Each Submission

### 1. Train Model
```bash
source barbados/bin/activate
python scripts/train.py
```

### 2. Generate Predictions
```bash
python scripts/predict.py
# Creates: submission.csv
```

### 3. Version the Submission
```bash
python scripts/manage_submissions.py create "description" submission.csv 0.XXXX
# Creates: submissions/submission_vX_description.csv
```

### 4. Upload to Zindi
- Go to competition page
- Upload `submissions/submission_vX_description.csv`
- Wait for score

### 5. Record Score
Update `submissions/SUBMISSION_LOG.md` with leaderboard score

---

## Naming Convention

**Format:** `submission_v{VERSION}_{DESCRIPTION}.csv`

**Good descriptions:**
- `baseline_temporal_only` - Clear what features used
- `tuned_lr005_leaves50` - Specific hyperparameters
- `ensemble_lgb_xgb_catboost` - Model combination
- `video_features_optical_flow` - New features added
- `class_weights_balanced` - Technique used

**Bad descriptions:**
- `test` - Not descriptive
- `final` - You'll make more submissions
- `v2` - Version is already in filename
- `submission` - Too generic

---

## File Structure

```
submissions/
├── SUBMISSION_LOG.md                          # Master tracking document
├── submission_v1_baseline_temporal_only.csv   # First submission
├── submission_v1_metadata.txt                 # Auto-generated metadata
├── submission_v2_tuned_hyperparams.csv        # Second submission
├── submission_v2_metadata.txt
├── submission_v3_with_video_features.csv
├── submission_v3_metadata.txt
└── ...
```

---

## Tracking Improvements

### After Each Submission:

1. **Record in SUBMISSION_LOG.md:**
   - Leaderboard score
   - What changed from previous version
   - What worked / didn't work
   - Next steps

2. **Save corresponding model:**
   ```bash
   cp -r models/baseline models/v1_baseline
   ```

3. **Git commit:**
   ```bash
   git add submissions/
   git commit -m "Add submission v1: baseline temporal only - Score: 0.XXXX"
   ```

---

## Daily Submission Strategy

**You have 20 submissions per day. Use them wisely!**

### Morning (5 submissions)
- Test 3-5 quick variations
- Hyperparameter tweaks
- Different random seeds
- Submit best 2-3

### Afternoon (10 submissions)
- Implement major improvements
- Add new features
- Try different models
- Test on validation first
- Submit top 5-7

### Evening (5 submissions)
- Ensemble best models
- Final tuning
- Reserve 2-3 for emergencies

### Strategy:
- ✅ **Always validate locally first**
- ✅ **Only submit if validation improves**
- ✅ **Keep 2-3 submissions as buffer**
- ❌ **Don't waste on random experiments**

---

## Version Control

### Git Workflow
```bash
# After each submission
git add submissions/submission_v*
git add models/v*
git commit -m "v{X}: {description} - Score: {score}"
git push
```

### Model Versioning
```bash
# Save model for each submission
cp -r models/baseline models/v1_baseline
cp -r models/baseline models/v2_tuned
cp -r models/baseline models/v3_video_features

# Keep best model
cp -r models/v5_best models/best
```

---

## Useful Commands

```bash
# Count submissions today
ls submissions/submission_v*.csv | wc -l

# View latest submission
ls -lt submissions/submission_v*.csv | head -1

# Compare two submissions (first 20 lines)
diff submissions/submission_v1_*.csv submissions/submission_v2_*.csv | head -20

# Check submission file size
ls -lh submissions/submission_v*.csv

# Find best submission (manually check SUBMISSION_LOG.md)
cat submissions/SUBMISSION_LOG.md | grep "Score"
```

---

## Troubleshooting

### "Too many submissions"
- You've hit the 20/day limit
- Wait until next day (UTC time)
- Use validation scores to filter

### "Invalid submission format"
- Run validation: `python scripts/predict.py`
- Check columns: ID, Target, Target_Accuracy
- Check classes: free flowing, light delay, moderate delay, heavy delay

### "Duplicate submission"
- Zindi may reject identical files
- Make sure you've actually changed something
- Check diff between versions

---

## Best Practices

✅ **DO:**
- Validate locally before submitting
- Use descriptive names
- Track all changes in SUBMISSION_LOG.md
- Save models corresponding to submissions
- Test incrementally

❌ **DON'T:**
- Submit without validation
- Use generic names like "test" or "final"
- Submit same file twice
- Waste submissions on random experiments
- Forget to record scores

---

## Example Session

```bash
# Morning: Quick hyperparameter test
python scripts/train.py  # Modify params first
python scripts/predict.py
python scripts/manage_submissions.py create "lr01_leaves40" submission.csv 0.5823
# Upload to Zindi → Score: 0.5654 (worse, don't pursue)

# Afternoon: Add new features
# ... implement video features ...
python scripts/train.py
python scripts/predict.py
python scripts/manage_submissions.py create "video_vehicle_counts" submission.csv 0.6234
# Upload to Zindi → Score: 0.6123 (better! ✅)

# Evening: Ensemble
# ... create ensemble ...
python scripts/train.py
python scripts/predict.py
python scripts/manage_submissions.py create "ensemble_lgb_xgb" submission.csv 0.6456
# Upload to Zindi → Score: 0.6389 (best so far! ✅✅)
```

---

## Summary

1. **Create submissions folder** ✅
2. **Use naming convention** ✅
3. **Track in SUBMISSION_LOG.md** ✅
4. **Validate before submitting** ✅
5. **Record leaderboard scores** ✅
6. **Iterate and improve** ✅

**Goal:** Systematic improvement from baseline (0.57) to competitive score (0.75+)
