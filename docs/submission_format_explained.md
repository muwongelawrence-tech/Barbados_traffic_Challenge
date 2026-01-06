# Submission File Explained

## What You're Submitting

You're predicting **congestion levels** for test video segments.

## The 4 Congestion Classes

Your predictions must be one of these **exact strings**:
1. `free flowing` - No congestion
2. `light delay` - Minor congestion
3. `moderate delay` - Moderate congestion  
4. `heavy delay` - Severe congestion

## Submission File Structure

```csv
ID,Target,Target_Accuracy
time_segment_112_Norman Niles #1_congestion_enter_rating,free flowing,free flowing
time_segment_112_Norman Niles #1_congestion_exit_rating,free flowing,free flowing
time_segment_113_Norman Niles #1_congestion_enter_rating,light delay,light delay
time_segment_113_Norman Niles #1_congestion_exit_rating,moderate delay,moderate delay
```

### Column Details:

| Column | Purpose | Weight | Values |
|--------|---------|--------|--------|
| **ID** | Identifies what you're predicting | N/A | From test data |
| **Target** | Prediction for F1-score | 70% | One of 4 classes |
| **Target_Accuracy** | Prediction for Accuracy | 30% | One of 4 classes |

## What Each Row Represents

Each test video segment needs **2 predictions**:

```
Test Segment #112 (Norman Niles #1 camera)
├── Row 1: Entrance congestion → "free flowing"
└── Row 2: Exit congestion → "free flowing"

Test Segment #113 (Norman Niles #1 camera)
├── Row 1: Entrance congestion → "light delay"
└── Row 2: Exit congestion → "moderate delay"
```

## Example Breakdown

```csv
ID,Target,Target_Accuracy
time_segment_112_Norman Niles #1_congestion_enter_rating,free flowing,free flowing
```

Breaking down the ID:
- `time_segment_112` - This is test segment #112
- `Norman Niles #1` - Camera view #1
- `congestion_enter_rating` - Predicting ENTRANCE congestion
- Prediction: `free flowing` (no congestion at entrance)

```csv
time_segment_112_Norman Niles #1_congestion_exit_rating,free flowing,free flowing
```

- Same segment #112, same camera
- `congestion_exit_rating` - Predicting EXIT congestion
- Prediction: `free flowing` (no congestion at exit)

## How Our Code Generates This

```python
# Model predicts integers (0, 1, 2, 3)
pred_enter = [0, 1, 2, 3]  # Example predictions
pred_exit = [0, 1, 1, 2]

# Convert to class names
class_map = {
    0: 'free flowing',
    1: 'light delay', 
    2: 'moderate delay',
    3: 'heavy delay'
}

# For each test segment, create 2 rows:
for segment in test_data:
    # Row 1: Entrance
    submission.append({
        'ID': segment.ID_enter,
        'Target': class_map[pred_enter[i]],
        'Target_Accuracy': class_map[pred_enter[i]]
    })
    
    # Row 2: Exit
    submission.append({
        'ID': segment.ID_exit,
        'Target': class_map[pred_exit[i]],
        'Target_Accuracy': class_map[pred_exit[i]]
    })
```

## Scoring

Your final score is calculated as:

```
Final Score = 0.7 × Macro-F1(Target) + 0.3 × Accuracy(Target_Accuracy)
```

**Macro F1-Score (70% weight):**
- Calculated from `Target` column
- Treats all 4 classes equally
- Good for imbalanced datasets

**Accuracy (30% weight):**
- Calculated from `Target_Accuracy` column
- Overall percentage correct
- Simple metric

## Common Questions

### Q: Why are Target and Target_Accuracy the same?
**A:** For simplicity. You *could* use different strategies for each metric, but usually the same prediction works best.

### Q: How many rows will my submission have?
**A:** `2 × number_of_test_segments`
- If test has 1,000 segments → 2,000 rows
- If test has 500 segments → 1,000 rows

### Q: What if I predict the wrong class name?
**A:** Submission will be rejected! Must be exact:
- ✅ `free flowing`
- ❌ `Free Flowing` (wrong capitalization)
- ❌ `free_flowing` (wrong format)
- ❌ `no congestion` (not a valid class)

### Q: Can entrance and exit have different congestion?
**A:** Yes! They're independent predictions:
- Entrance could be `heavy delay` (cars waiting to enter)
- Exit could be `free flowing` (cars leaving smoothly)

## Validation

Our code automatically validates:
```python
✅ Submission format is valid!
   - 2000 predictions
   - Columns: ['ID', 'Target', 'Target_Accuracy']
   - No missing values
   - All classes valid
```

## What Happens After Submission

1. You upload `submission.csv` to Zindi
2. Zindi compares your predictions to true labels (hidden)
3. Calculates Macro F1-Score from `Target` column
4. Calculates Accuracy from `Target_Accuracy` column
5. Combines: `0.7 × F1 + 0.3 × Accuracy`
6. Shows your score on leaderboard!

## Summary

**You're predicting:** Congestion levels (entrance + exit) for each test segment

**Format:** 3 columns (ID, Target, Target_Accuracy)

**Classes:** `free flowing`, `light delay`, `moderate delay`, `heavy delay`

**Rows:** 2 per test segment (entrance + exit)

**Our code handles all of this automatically!** ✅
