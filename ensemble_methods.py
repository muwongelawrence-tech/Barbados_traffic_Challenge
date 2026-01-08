"""
Phase 2c: Ensemble Methods
Train multiple models and combine predictions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

import config

print("="*60)
print("PHASE 2c: ENSEMBLE METHODS")
print("="*60)

def train_ensemble_models(X_train, y_train, X_val, y_val, class_weight_dict, best_lgbm_params=None):
    """
    Train LightGBM, XGBoost, and CatBoost models
    """
    models = {}
    predictions = {}
    
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE MODELS")
    print("="*60)
    
    # 1. LightGBM (use tuned params if available)
    print("\n1. Training LightGBM...")
    if best_lgbm_params:
        lgbm_params = best_lgbm_params.copy()
    else:
        lgbm_params = {
            'objective': 'multiclass',
            'num_class': 4,
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 8,
            'num_leaves': 63,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'verbose': -1,
            'class_weight': class_weight_dict
        }
    
    lgbm_model = LGBMClassifier(**lgbm_params)
    lgbm_model.fit(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_val)
    lgbm_pred_proba = lgbm_model.predict_proba(X_val)
    
    lgbm_f1 = f1_score(y_val, lgbm_pred, average='macro')
    print(f"   LightGBM F1: {lgbm_f1:.4f}")
    
    models['lgbm'] = lgbm_model
    predictions['lgbm'] = lgbm_pred
    predictions['lgbm_proba'] = lgbm_pred_proba
    
    # 2. XGBoost
    print("\n2. Training XGBoost...")
    
    # Calculate sample weights for XGBoost
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    
    xgb_params = {
        'objective': 'multi:softmax',
        'num_class': 4,
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 0
    }
    
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_pred = xgb_model.predict(X_val)
    xgb_pred_proba = xgb_model.predict_proba(X_val)
    
    xgb_f1 = f1_score(y_val, xgb_pred, average='macro')
    print(f"   XGBoost F1: {xgb_f1:.4f}")
    
    models['xgb'] = xgb_model
    predictions['xgb'] = xgb_pred
    predictions['xgb_proba'] = xgb_pred_proba
    
    # 3. CatBoost
    print("\n3. Training CatBoost...")
    
    catboost_params = {
        'loss_function': 'MultiClass',
        'iterations': 300,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'class_weights': class_weight_dict
    }
    
    catboost_model = CatBoostClassifier(**catboost_params)
    catboost_model.fit(X_train, y_train)
    catboost_pred = catboost_model.predict(X_val).flatten().astype(int)
    catboost_pred_proba = catboost_model.predict_proba(X_val)
    
    catboost_f1 = f1_score(y_val, catboost_pred, average='macro')
    print(f"   CatBoost F1: {catboost_f1:.4f}")
    
    models['catboost'] = catboost_model
    predictions['catboost'] = catboost_pred
    predictions['catboost_proba'] = catboost_pred_proba
    
    return models, predictions

def create_ensemble_predictions(predictions, y_val, method='weighted_avg'):
    """
    Combine predictions from multiple models
    """
    print("\n" + "="*60)
    print(f"ENSEMBLE METHOD: {method.upper()}")
    print("="*60)
    
    if method == 'weighted_avg':
        # Weight by individual model F1 scores
        lgbm_f1 = f1_score(y_val, predictions['lgbm'], average='macro')
        xgb_f1 = f1_score(y_val, predictions['xgb'], average='macro')
        catboost_f1 = f1_score(y_val, predictions['catboost'], average='macro')
        
        total_f1 = lgbm_f1 + xgb_f1 + catboost_f1
        weights = {
            'lgbm': lgbm_f1 / total_f1,
            'xgb': xgb_f1 / total_f1,
            'catboost': catboost_f1 / total_f1
        }
        
        print(f"\nModel weights:")
        for model, weight in weights.items():
            print(f"  {model}: {weight:.3f}")
        
        # Weighted average of probabilities
        ensemble_proba = (
            weights['lgbm'] * predictions['lgbm_proba'] +
            weights['xgb'] * predictions['xgb_proba'] +
            weights['catboost'] * predictions['catboost_proba']
        )
        
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
    elif method == 'voting':
        # Majority voting
        votes = np.vstack([
            predictions['lgbm'],
            predictions['xgb'],
            predictions['catboost']
        ])
        
        ensemble_pred = []
        for i in range(votes.shape[1]):
            counts = np.bincount(votes[:, i].astype(int), minlength=4)
            ensemble_pred.append(np.argmax(counts))
        
        ensemble_pred = np.array(ensemble_pred)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    ensemble_f1 = f1_score(y_val, ensemble_pred, average='macro')
    ensemble_acc = accuracy_score(y_val, ensemble_pred)
    
    print(f"\nEnsemble Performance:")
    print(f"  Accuracy: {ensemble_acc:.4f}")
    print(f"  F1 (macro): {ensemble_f1:.4f}")
    
    return ensemble_pred, ensemble_proba if method == 'weighted_avg' else None

if __name__ == "__main__":
    print("Ensemble module loaded")
    print("Use: train_ensemble_models() and create_ensemble_predictions()")
