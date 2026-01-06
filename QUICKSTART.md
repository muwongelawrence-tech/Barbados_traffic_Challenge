# Quick Start Guide - Get Your First Submission Today!

## Prerequisites
Make sure your virtual environment is activated:
```bash
source barbados/bin/activate
```

## Step 1: Install Dependencies (5 minutes)
```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy, scikit-learn (data processing)
- lightgbm (our model)
- opencv-python (for future video features)

## Step 2: Train Baseline Model (2-5 minutes)
```bash
python scripts/train.py
```

**What this does:**
- Loads `Train.csv` (16,077 samples)
- Engineers temporal features (hour, rush hour, cyclical encoding, etc.)
- Trains LightGBM model for entrance and exit congestion
- Evaluates on validation set
- Saves model to `models/baseline/`

**Expected output:**
```
Average Combined Score: 0.XXXX ⭐⭐
```

## Step 3: Generate Submission (30 seconds)
```bash
python scripts/predict.py
```

**What this does:**
- Loads `TestInputSegments.csv`
- Applies same feature engineering
- Loads trained model
- Generates predictions
- Creates `submission.csv` in correct format

**Output file:** `submission.csv`

## Step 4: Submit to Zindi
1. Go to https://zindi.africa/competitions/barbados-traffic-analysis-challenge
2. Click "Make a submission"
3. Upload `submission.csv`
4. See your score on the leaderboard!

---

## Complete Workflow (One Command)
```bash
# Activate environment
source barbados/bin/activate

# Train and predict
python scripts/train.py && python scripts/predict.py

# Your submission.csv is ready!
```

---

## What's in the Baseline Model?

### Features Used (No Video Processing):
- **Time features**: hour, minute, day of week
- **Cyclical encoding**: sin/cos for hour and minute
- **Rush hour indicators**: morning (7-9am), evening (4-6pm)
- **Camera ID**: which camera view (1-4)
- **Turn signal usage**: encoded as 0-3
- **Time segment ID**: sequential identifier

### Model:
- **LightGBM** (gradient boosting decision trees)
- **Dual output**: separate models for entrance and exit
- **No backpropagation** ✅
- **Fast inference**: <5ms per prediction

### Expected Performance:
- **Baseline (temporal only)**: F1 ~0.50-0.60, Accuracy ~0.60-0.70
- **Combined Score**: ~0.55-0.65

This gets you on the leaderboard! We'll improve it later with video features.

---

## Troubleshooting

### ImportError: No module named 'lightgbm'
```bash
source barbados/bin/activate
pip install lightgbm
```

### FileNotFoundError: Train.csv
Make sure you're in the project root directory:
```bash
cd /home/muwongepro/Mlops/Barbados_traffic_Challenge
ls Train.csv  # Should exist
```

### Model training takes too long
The baseline should train in 2-5 minutes. If it's slower:
- Check CPU usage
- Reduce `num_boost_round` in `scripts/train.py` (line with `num_boost_round=500`)

---

## Next Steps (After First Submission)

1. **Analyze results**: Check which classes are hardest to predict
2. **Add video features**: Extract vehicle counts, optical flow
3. **Tune hyperparameters**: Optimize LightGBM settings
4. **Ensemble models**: Combine multiple models
5. **Feature engineering**: Create more sophisticated temporal patterns

---

## File Structure
```
Barbados_traffic_Challenge/
├── Train.csv                    # Training data
├── TestInputSegments.csv        # Test data
├── SampleSubmission.csv         # Format reference
├── submission.csv               # YOUR SUBMISSION (generated)
├── requirements.txt             # Dependencies
├── scripts/
│   ├── train.py                # Training script
│   └── predict.py              # Prediction script
├── src/
│   ├── data/
│   │   └── data_loader.py      # Data loading
│   ├── features/
│   │   └── temporal_features.py # Feature engineering
│   ├── models/
│   │   └── baseline_model.py   # LightGBM model
│   ├── utils/
│   │   └── metrics.py          # Evaluation
│   └── inference/
│       └── submission_generator.py # Submission creation
└── models/
    └── baseline/               # Saved models (after training)
```

---

## Quick Commands Reference

```bash
# Setup
source barbados/bin/activate
pip install -r requirements.txt

# Train
python scripts/train.py

# Predict
python scripts/predict.py

# Check submission format
head -20 submission.csv

# Count predictions
wc -l submission.csv
```

---

**Ready to start? Run:**
```bash
source barbados/bin/activate && python scripts/train.py
```

Good luck! 🚀
