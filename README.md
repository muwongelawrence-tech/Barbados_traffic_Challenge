# Barbados Traffic Prediction Challenge

Machine learning solution for predicting traffic congestion levels in Barbados.

## Project Overview

This project predicts traffic congestion ratings (free flowing, light delay, moderate delay, heavy delay) at 4 camera locations across Barbados using time-series data and gradient boosting models.

**Validation Performance:**
- Enter Rating F1: **0.8293**
- Exit Rating F1: **0.9601**
- Average F1: **0.8947**

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv barbados
source barbados/bin/activate  # On Linux/Mac
# barbados\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Quick Training & Submission

Train models and generate submission in one step:

```bash
python quick_train.py
```

This will:
- Train LightGBM models for enter and exit ratings
- Display validation metrics
- Save models to `models/` directory

### Generate Submission

```bash
python create_submission.py
```

Output: `submissions/submission.csv` (ready to upload to Zindi)

## Project Structure

```
├── config.py                 # Configuration and hyperparameters
├── quick_train.py            # Quick training script
├── create_submission.py      # Submission generation
├── train.py                  # Full training pipeline (XGBoost, CatBoost, LightGBM)
├── predict.py                # Prediction script
├── src/
│   ├── data_preprocessing.py # Data processing (37 features)
│   └── utils.py              # Utilities (metrics, visualization)
├── models/                   # Saved models
├── submissions/              # Generated submissions
└── requirements.txt          # Python dependencies
```

## Features Engineered (37 total)

- **Temporal**: hour, minute, day_of_week, is_weekend, is_rush_hour
- **Cyclical**: hour_sin/cos, day_sin/cos
- **Lag features**: Previous 1, 2, 3, 5, 10 time segments
- **Rolling stats**: Mean and std over 5, 10, 20 windows
- **Location**: Encoded camera locations
- **Signals**: Traffic light status encoding

## Models

- **Primary**: LightGBM (100 estimators)
- **Available**: XGBoost, CatBoost (in `train.py`)
- **Strategy**: Separate models for enter and exit ratings

## Data

- Training: 16,076 samples across 4 locations, 7 days
- Test: 2,640 samples
- Submission: 880 predictions required

## Next Steps

1. Upload `submissions/submission.csv` to Zindi
2. Based on leaderboard score, iterate with:
   - Ensemble models
   - Hyperparameter tuning
   - Advanced feature engineering
   - Class imbalance handling (SMOTE)

## License

MIT
