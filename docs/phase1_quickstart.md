# Phase 1 Quick Start Guide

## What We've Built

**Rapid testing infrastructure for Phase 1:**
- ✅ 6 hyperparameter configurations
- ✅ Quick training script (`train_quick.py`)
- ✅ Automated batch testing (`phase1_rapid_test.sh`)
- ✅ Submission management integration

---

## Quick Commands

### Option 1: Run All 6 Tests Automatically (Recommended)
```bash
source barbados/bin/activate
./scripts/phase1_rapid_test.sh
```

**What this does:**
- Trains 6 different model configurations
- Generates 6 submission files
- Takes ~15-20 minutes total
- Creates: `submissions/submission_v2_*.csv` through `v7_*.csv`

**Then:**
- Upload all 6 to Zindi
- Record scores in `submissions/SUBMISSION_LOG.md`
- Identify best configuration

---

### Option 2: Test Individual Configurations

```bash
source barbados/bin/activate

# v2: Increased complexity
python scripts/train_quick.py --config v2_complexity --output-dir models/v2
python scripts/predict.py
python scripts/manage_submissions.py create "v2_increased_complexity" submission.csv

# v3: Lower learning rate
python scripts/train_quick.py --config v3_lower_lr --num-boost-round 1000 --output-dir models/v3
python scripts/predict.py
python scripts/manage_submissions.py create "v3_lower_lr" submission.csv

# v4: Regularization
python scripts/train_quick.py --config v4_regularized --output-dir models/v4
python scripts/predict.py
python scripts/manage_submissions.py create "v4_regularized" submission.csv

# v5: Class weights (EXPECTED BIG WIN)
python scripts/train_quick.py --config v2_complexity --class-weights --output-dir models/v5
python scripts/predict.py
python scripts/manage_submissions.py create "v5_class_weights" submission.csv

# v6: Deep trees
python scripts/train_quick.py --config v6_deep_trees --output-dir models/v6
python scripts/predict.py
python scripts/manage_submissions.py create "v6_deep_trees" submission.csv
```

---

## The 6 Configurations

| Version | Key Changes | Expected Impact |
|---------|-------------|-----------------|
| **v2** | `num_leaves=50`, `lr=0.03` | +0.01-0.02 |
| **v3** | `lr=0.01`, 1000 rounds | +0.01-0.02 |
| **v4** | L1/L2 regularization | +0.01 |
| **v5** | Class weights (balanced) | **+0.02-0.03** 🎯 |
| **v6** | Deep trees (100 leaves) | +0.01-0.02 |
| **v7** | Ensemble (after seeing results) | +0.01-0.02 |

**Expected best:** v5 (class weights) - should help minority classes significantly

---

## Timeline

### Today (Next 2 hours):
1. ✅ Infrastructure ready
2. ⏳ Run batch testing (~20 min)
3. ⏳ Upload 6 submissions to Zindi
4. ⏳ Record scores

### Tomorrow (Day 2):
- Add advanced temporal features
- Lag features, rolling stats
- 8-10 more submissions
- Target: 0.61-0.62

### Day 3:
- Model optimization
- Custom loss function
- Final Phase 1 push
- Target: 0.62-0.63

---

## What to Expect

**Baseline (v1):** 0.5745 validation

**After Phase 1 (v2-v7):**
- Worst case: 0.58
- Expected: 0.60-0.61
- Best case: 0.62

**Key improvements:**
- Better minority class performance (F1 boost)
- More stable predictions
- Less overfitting

---

## Next Steps After Phase 1

Once we hit 0.60-0.63:

**Phase 2 (Days 4-8):**
- Video feature extraction
- Classical CV implementation
- Target: 0.65-0.68

**Phase 3 (Days 9-14):**
- Ensemble methods
- Advanced optimization
- Target: **0.70-0.72+ (#1)**

---

## Ready to Run?

**Recommended approach:**

```bash
# 1. Activate environment
source barbados/bin/activate

# 2. Run all 6 tests (takes ~20 min)
./scripts/phase1_rapid_test.sh

# 3. Wait for completion, then upload all submissions
# 4. Record scores and identify best config
```

**Alternative (if you want to watch progress):**
Run each configuration individually to see results as they complete.

---

## Troubleshooting

**If script fails:**
- Check virtual environment is activated
- Ensure all dependencies installed
- Run individual commands to isolate issue

**If training is slow:**
- Reduce `num_boost_round` to 300
- Use fewer configurations (just v2, v5, v6)

**If out of memory:**
- Close other applications
- Reduce batch size (train one at a time)

---

## Expected Output

After running `phase1_rapid_test.sh`:

```
submissions/
├── submission_v2_increased_complexity.csv
├── submission_v3_lower_lr_1000rounds.csv
├── submission_v4_regularized_l1_l2.csv
├── submission_v5_class_weights_balanced.csv  ← LIKELY BEST
├── submission_v6_deep_trees_100leaves.csv
└── submission_v7_ensemble.csv (manual)

models/
├── v2_complexity/
├── v3_lower_lr/
├── v4_regularized/
├── v5_class_weights/  ← LIKELY BEST
└── v6_deep_trees/
```

**Ready to accelerate to #1!** 🚀
