"""
Training script for Barbados Traffic Challenge
Trains gradient boosting models for congestion prediction
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import argparse
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

import config
from src.data_preprocessing import DataPreprocessor
from src.utils import (
    calculate_metrics, plot_confusion_matrix, plot_feature_importance,
    save_model, save_metrics, get_class_weights
)


def train_model(X_train, y_train, X_val, y_val, model_type='lightgbm', 
                target_name='', use_class_weights=True):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} for {target_name}")
    print(f"{'='*60}")
    
    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = get_class_weights(y_train)
    
    # Handle missing values
    X_train_clean = X_train.fillna(0)
    X_val_clean = X_val.fillna(0)
    
    # Initialize model
    if model_type == 'lightgbm':
        model = LGBMClassifier(**config.LIGHTGBM_PARAMS)
        if class_weights:
            model.set_params(class_weight=class_weights)
        model.fit(
            X_train_clean, y_train,
            eval_set=[(X_val_clean, y_val)]
        )
    
    elif model_type == 'xgboost':
        # Calculate sample weights for XGBoost
        sample_weights = None
        if class_weights:
            sample_weights = np.array([class_weights[y] for y in y_train])
        
        model = XGBClassifier(**config.XGBOOST_PARAMS)
        model.fit(
            X_train_clean, y_train,
            eval_set=[(X_val_clean, y_val)],
            sample_weight=sample_weights
        )
    
    elif model_type == 'catboost':
        model = CatBoostClassifier(**config.CATBOOST_PARAMS)
        if class_weights:
            model.set_params(class_weights=list(class_weights.values()))
        model.fit(
            X_train_clean, y_train,
            eval_set=(X_val_clean, y_val)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Make predictions
    y_pred_train = model.predict(X_train_clean)
    y_pred_val = model.predict(X_val_clean)
    
    # Calculate metrics
    print("\n--- Training Set ---")
    train_metrics = calculate_metrics(y_train, y_pred_train, f"{target_name} (Train)")
    
    print("\n--- Validation Set ---")
    val_metrics = calculate_metrics(y_val, y_pred_val, f"{target_name} (Val)")
    
    # Plot confusion matrix
    cm_path = os.path.join(
        config.RESULTS_DIR,
        f'confusion_matrix_{model_type}_{target_name.replace(" ", "_")}.png'
    )
    plot_confusion_matrix(y_val, y_pred_val, target_name, save_path=cm_path)
    
    # Plot feature importance
    fi_path = os.path.join(
        config.RESULTS_DIR,
        f'feature_importance_{model_type}_{target_name.replace(" ", "_")}.png'
    )
    importance_df = plot_feature_importance(
        model, X_train.columns.tolist(), top_n=20, save_path=fi_path
    )
    
    return model, train_metrics, val_metrics


def train_all_models(model_types=['lightgbm', 'xgboost', 'catboost']):
    """Train all models for both enter and exit ratings"""
    print("="*60)
    print("BARBADOS TRAFFIC PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    train_df, test_df = preprocessor.load_data()
    train_processed = preprocessor.preprocess(train_df, is_train=True)
    
    # Prepare train/val split
    X_train, X_val, y_enter_train, y_enter_val, y_exit_train, y_exit_val = \
        preprocessor.prepare_train_test_split(train_processed, test_size=0.2)
    
    print(f"\nFeature columns: {len(X_train.columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Store results
    all_results = {}
    all_models = {}
    
    # Train models for each target
    for target_type in ['enter', 'exit']:
        if target_type == 'enter':
            y_train = y_enter_train
            y_val = y_enter_val
            target_name = 'Congestion Enter Rating'
        else:
            y_train = y_exit_train
            y_val = y_exit_val
            target_name = 'Congestion Exit Rating'
        
        all_results[target_type] = {}
        all_models[target_type] = {}
        
        for model_type in model_types:
            model, train_metrics, val_metrics = train_model(
                X_train, y_train, X_val, y_val,
                model_type=model_type,
                target_name=target_name,
                use_class_weights=config.USE_CLASS_WEIGHTS
            )
            
            all_results[target_type][model_type] = {
                'train': train_metrics,
                'val': val_metrics
            }
            all_models[target_type][model_type] = model
            
            # Save model
            model_path = os.path.join(
                config.MODELS_DIR,
                f'{model_type}_{target_type}_rating.pkl'
            )
            save_model(model, model_path)
    
    # Save all metrics
    metrics_path = os.path.join(config.RESULTS_DIR, 'all_metrics.json')
    save_metrics(all_results, metrics_path)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for target_type in ['enter', 'exit']:
        print(f"\n{target_type.upper()} RATING - Validation F1 Scores (Weighted):")
        for model_type in model_types:
            f1 = all_results[target_type][model_type]['val']['f1_weighted']
            print(f"  {model_type:12s}: {f1:.4f}")
    
    # Find best models
    best_enter_model = max(
        model_types,
        key=lambda m: all_results['enter'][m]['val']['f1_weighted']
    )
    best_exit_model = max(
        model_types,
        key=lambda m: all_results['exit'][m]['val']['f1_weighted']
    )
    
    print(f"\nBest model for ENTER rating: {best_enter_model}")
    print(f"Best model for EXIT rating: {best_exit_model}")
    
    # Save best models with special naming
    save_model(
        all_models['enter'][best_enter_model],
        os.path.join(config.MODELS_DIR, 'best_enter_model.pkl')
    )
    save_model(
        all_models['exit'][best_exit_model],
        os.path.join(config.MODELS_DIR, 'best_exit_model.pkl')
    )
    
    return all_models, all_results, preprocessor


def main():
    parser = argparse.ArgumentParser(description='Train traffic prediction models')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['lightgbm', 'xgboost', 'catboost'],
        choices=['lightgbm', 'xgboost', 'catboost'],
        help='Models to train'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Train models
    models, results, preprocessor = train_all_models(model_types=args.models)
    
    print("\n✅ Training completed successfully!")
    print(f"Models saved to: {config.MODELS_DIR}")
    print(f"Results saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
