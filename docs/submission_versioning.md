# Submission Versioning Guide

## Naming Convention

All submission files should follow this format:
```
vX.Y_description.csv
```

Where:
- `X` = Major version (significant model/approach changes)
- `Y` = Minor version (hyperparameter tuning, small improvements)
- `description` = Brief description of the approach

## Version History

### v1.0_baseline_lightgbm.csv
- **Date**: 2026-01-07
- **Public Score**: 0.461234377
- **Description**: Baseline LightGBM with 37 features
- **Model**: LightGBM (100 estimators)
- **Status**: ✅ Submitted

### Future Versions

#### v1.1_improved_features.csv
- Planned: Enhanced feature engineering
- Expected improvement: +0.05-0.10

#### v1.2_tuned_hyperparams.csv
- Planned: Hyperparameter optimization
- Expected improvement: +0.02-0.05

#### v1.3_class_balanced.csv
- Planned: SMOTE + class weighting
- Expected improvement: +0.10-0.15

#### v1.4_ensemble.csv
- Planned: XGBoost + LightGBM + CatBoost ensemble
- Expected improvement: +0.05-0.10

#### v2.0_advanced.csv
- Planned: Neural networks or advanced techniques
- Expected improvement: TBD

## Submission Checklist

Before submitting:
- [ ] Version number assigned
- [ ] Description added to experiment_log.md
- [ ] Model saved to models/ directory
- [ ] Validation scores documented
- [ ] Submission file validated (880 rows, correct format)
- [ ] Previous best score noted
- [ ] Expected improvement estimated

## Best Practices

1. **Never overwrite previous submissions** - Always create new versioned files
2. **Document everything** - Update experiment_log.md with each submission
3. **Save models** - Keep model files for reproducibility
4. **Track scores** - Record both public and private scores
5. **Note observations** - Document what worked and what didn't

## Quick Commands

### Create new submission
```bash
# Copy template
cp submissions/v1.0_baseline_lightgbm.csv submissions/vX.Y_description.csv

# Or generate new
python create_submission.py --output submissions/vX.Y_description.csv
```

### List all submissions
```bash
ls -lh submissions/v*.csv
```

### Compare submissions
```bash
diff submissions/v1.0_baseline_lightgbm.csv submissions/v1.1_improved_features.csv | head -20
```
