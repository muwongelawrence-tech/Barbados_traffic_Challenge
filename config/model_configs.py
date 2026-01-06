"""
Configuration file for model hyperparameters
Easily switch between different configurations for rapid testing
"""

# Baseline configuration (v1)
BASELINE_PARAMS = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': ['multi_logloss', 'multi_error'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': -1,
    'seed': 42
}

# v2: Increased tree complexity
V2_INCREASED_COMPLEXITY = {
    **BASELINE_PARAMS,
    'num_leaves': 50,
    'learning_rate': 0.03,
    'min_child_samples': 10,
    'max_depth': 8
}

# v3: Lower learning rate, more iterations
V3_LOWER_LR = {
    **BASELINE_PARAMS,
    'num_leaves': 50,
    'learning_rate': 0.01,
    'min_child_samples': 10
}

# v4: Add regularization
V4_REGULARIZED = {
    **BASELINE_PARAMS,
    'num_leaves': 50,
    'learning_rate': 0.03,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_child_samples': 10
}

# v5: Stronger regularization
V5_STRONG_REG = {
    **BASELINE_PARAMS,
    'num_leaves': 50,
    'learning_rate': 0.03,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'min_child_samples': 20
}

# v6: Deep trees
V6_DEEP_TREES = {
    **BASELINE_PARAMS,
    'num_leaves': 100,
    'learning_rate': 0.02,
    'max_depth': 10,
    'min_child_samples': 5
}

# XGBoost parameters
XGBOOST_PARAMS = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'gamma': 0.1,
    'min_child_weight': 5,
    'random_state': 42,
    'n_jobs': -1
}

# CatBoost parameters
CATBOOST_PARAMS = {
    'iterations': 500,
    'depth': 8,
    'learning_rate': 0.05,
    'loss_function': 'MultiClass',
    'eval_metric': 'TotalF1:average=Macro',
    'random_seed': 42,
    'verbose': False,
    'thread_count': -1
}

# Configuration selector
CONFIGS = {
    'baseline': BASELINE_PARAMS,
    'v2_complexity': V2_INCREASED_COMPLEXITY,
    'v3_lower_lr': V3_LOWER_LR,
    'v4_regularized': V4_REGULARIZED,
    'v5_strong_reg': V5_STRONG_REG,
    'v6_deep_trees': V6_DEEP_TREES,
    'xgboost': XGBOOST_PARAMS,
    'catboost': CATBOOST_PARAMS
}

def get_config(name='baseline'):
    """Get configuration by name"""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name].copy()
