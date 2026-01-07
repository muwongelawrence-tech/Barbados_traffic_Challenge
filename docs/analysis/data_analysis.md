# Data Analysis - Barbados Traffic Challenge

## Dataset Overview

### Training Data
- **Size**: 16,076 samples
- **Locations**: 4 camera positions (Norman Niles #1-4)
- **Time Period**: 7 days (2025-10-20 to 2025-10-26)
- **Samples per location**: 4,019 each
- **Temporal resolution**: 1-minute segments

### Test Data
- **Size**: 2,640 samples
- **Samples per location**: 660 each
- **Submission required**: 880 predictions (440 enter + 440 exit)

## Target Variable Distribution

### Congestion Enter Rating (Training)
```
free flowing:    10,056 (63%)
moderate delay:   2,328 (14%)
light delay:      1,919 (12%)
heavy delay:      1,773 (11%)
```

**Observations**:
- Relatively balanced distribution
- Majority class (free flowing) is 63%
- Minority classes well-represented (11-14%)

### Congestion Exit Rating (Training)
```
free flowing:    15,353 (95%)
moderate delay:     283 (2%)
light delay:        241 (2%)
heavy delay:        199 (1%)
```

**Observations**:
- **Severe class imbalance**
- 95% of samples are "free flowing"
- Minority classes very rare (1-2%)
- This explains high validation F1 (0.9601) - model predicts mostly "free flowing"

## Feature Analysis

### Temporal Patterns

**Signaling Distribution**:
```
none:     8,811 (55%)
low:      4,074 (25%)
medium:   2,594 (16%)
high:       597 (4%)
```

### Missing Data
- Lag features have missing values for first N segments per location
- Total missing values in features: 192 (handled by filling with 0)

## Key Insights

### 1. Class Imbalance Issue
The exit rating is severely imbalanced (95% free flowing). This likely causes:
- Model to predict "free flowing" for most test samples
- High validation scores but poor generalization
- Need for:
  - SMOTE or other oversampling techniques
  - Adjusted class weights
  - Different evaluation metrics

### 2. Validation vs Test Gap
- Validation F1: 0.8947
- Public score: 0.4612
- **Gap**: 0.4335 (48% drop)

Possible reasons:
1. Test distribution differs from train
2. Time-based split may not represent test data well
3. Overfitting to validation set
4. Submission format issues (all predictions "free flowing"?)

### 3. Feature Importance
Top features (to be analyzed):
- Time-based features (hour, day_of_week)
- Lag features (recent congestion history)
- Location encoding
- Signal status

## Recommendations

### Immediate Actions
1. **Analyze test predictions**: Check distribution of predictions in submission file
2. **Implement proper CV**: Use TimeSeriesSplit with 5 folds
3. **Address class imbalance**: 
   - SMOTE for exit ratings
   - Adjust class weights
   - Try focal loss

### Feature Engineering
1. Add more sophisticated lag features
2. Create location-specific patterns
3. Add interaction features (location × time)
4. Implement exponentially weighted moving averages

### Model Improvements
1. Increase model complexity (more estimators)
2. Try ensemble methods
3. Separate strategies for enter vs exit (due to different distributions)
4. Consider treating exit rating as binary (flowing vs delayed) first

## Next Analysis Steps

1. [ ] Analyze prediction distribution on test set
2. [ ] Perform feature importance analysis
3. [ ] Investigate temporal patterns in errors
4. [ ] Compare train/test distributions
5. [ ] Analyze per-location performance
