"""
Configuration file for Barbados Traffic Prediction Challenge
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')

# Data files
TRAIN_FILE = os.path.join(DATA_DIR, 'Train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'TestInputSegments.csv')
SAMPLE_SUBMISSION_FILE = os.path.join(DATA_DIR, 'SampleSubmission.csv')

# Random seed for reproducibility
RANDOM_SEED = 42

# Target columns
TARGET_ENTER = 'congestion_enter_rating'
TARGET_EXIT = 'congestion_exit_rating'

# Congestion level mapping
CONGESTION_LEVELS = {
    'free flowing': 0,
    'light delay': 1,
    'moderate delay': 2,
    'heavy delay': 3
}

CONGESTION_LEVELS_REVERSE = {v: k for k, v in CONGESTION_LEVELS.items()}

# Feature engineering parameters
RUSH_HOUR_MORNING = (7, 9)  # 7 AM to 9 AM
RUSH_HOUR_EVENING = (16, 18)  # 4 PM to 6 PM
LAG_PERIODS = [1, 2, 3, 5, 10]  # Lag features to create

# Model hyperparameters
LIGHTGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': RANDOM_SEED,
    'n_estimators': 100  # Reduced for faster training
}

XGBOOST_PARAMS = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'random_state': RANDOM_SEED,
    'n_estimators': 500,
    'early_stopping_rounds': 50
}

CATBOOST_PARAMS = {
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'depth': 6,
    'learning_rate': 0.05,
    'iterations': 500,
    'random_seed': RANDOM_SEED,
    'verbose': False,
    'early_stopping_rounds': 50
}

# Cross-validation
CV_FOLDS = 5

# Class weights (to handle imbalance)
# Will be calculated dynamically based on class distribution
USE_CLASS_WEIGHTS = True
