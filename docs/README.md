# Documentation Index

## Quick Links

- [Experiment Log](experiment_log.md) - Track all submissions and performance
- [Submission Versioning](submission_versioning.md) - Naming conventions and best practices
- [Data Analysis](analysis/data_analysis.md) - Dataset insights and recommendations

## Model Documentation

- [v1.0 Baseline](models/v1.0_baseline.md) - LightGBM baseline model

## Experiments

*Experiment documentation will be added here as we iterate*

## Analysis

- [Data Analysis](analysis/data_analysis.md) - Initial data exploration and insights

## Project Structure

```
docs/
├── README.md                    # This file
├── experiment_log.md            # Master log of all experiments
├── submission_versioning.md     # Versioning guide
├── experiments/                 # Individual experiment docs
├── analysis/                    # Data analysis and insights
│   └── data_analysis.md
└── models/                      # Model-specific documentation
    └── v1.0_baseline.md
```

## How to Use This Documentation

### When Starting a New Experiment

1. Review [experiment_log.md](experiment_log.md) to see what's been tried
2. Check [data_analysis.md](analysis/data_analysis.md) for insights
3. Plan your experiment
4. Document your approach in `experiments/vX.Y_description.md`
5. Update [experiment_log.md](experiment_log.md) with planned experiment

### After Running an Experiment

1. Update [experiment_log.md](experiment_log.md) with results
2. Create/update model documentation in `models/`
3. Add observations and learnings
4. Version your submission file properly
5. Commit changes to Git

### Before Submitting to Zindi

1. Check [submission_versioning.md](submission_versioning.md)
2. Ensure submission file is properly versioned
3. Update [experiment_log.md](experiment_log.md)
4. Document expected vs actual performance

## Current Status

**Latest Submission**: v1.0 Baseline  
**Public Score**: 0.461234377  
**Goal**: Improve iteratively to reach top of leaderboard

## Next Steps

1. Analyze v1.0 predictions
2. Implement class balancing
3. Add cross-validation
4. Try ensemble methods
