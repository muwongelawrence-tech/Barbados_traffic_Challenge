"""
Baseline model using LightGBM with temporal features only
Fast training for quick submission
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import joblib
from pathlib import Path


class BaselineModel:
    """
    Baseline traffic congestion classifier using temporal features only
    Uses LightGBM for fast training and inference
    """
    
    def __init__(self, params: Optional[dict] = None):
        """
        Initialize baseline model
        
        Args:
            params: LightGBM parameters (optional)
        """
        self.params = params or self._get_default_params()
        self.model_enter = None
        self.model_exit = None
        self.feature_columns = None
        
    def _get_default_params(self) -> dict:
        """Get default LightGBM parameters"""
        return {
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
            'num_threads': -1,  # Use all available threads
            'seed': 42
        }
    
    def fit(self, X_train: pd.DataFrame, y_train_enter: np.ndarray, y_train_exit: np.ndarray,
            X_val: Optional[pd.DataFrame] = None, y_val_enter: Optional[np.ndarray] = None,
            y_val_exit: Optional[np.ndarray] = None, num_boost_round: int = 500) -> dict:
        """
        Train both entrance and exit models
        
        Args:
            X_train: Training features
            y_train_enter: Training labels for entrance
            y_train_exit: Training labels for exit
            X_val: Validation features (optional)
            y_val_enter: Validation labels for entrance (optional)
            y_val_exit: Validation labels for exit (optional)
            num_boost_round: Number of boosting rounds
            
        Returns:
            Dictionary with training history
        """
        self.feature_columns = X_train.columns.tolist()
        
        # Prepare training data
        train_data_enter = lgb.Dataset(X_train, label=y_train_enter)
        train_data_exit = lgb.Dataset(X_train, label=y_train_exit)
        
        valid_sets_enter = [train_data_enter]
        valid_sets_exit = [train_data_exit]
        valid_names = ['train']
        
        # Add validation data if provided
        if X_val is not None and y_val_enter is not None:
            valid_data_enter = lgb.Dataset(X_val, label=y_val_enter, reference=train_data_enter)
            valid_data_exit = lgb.Dataset(X_val, label=y_val_exit, reference=train_data_exit)
            valid_sets_enter.append(valid_data_enter)
            valid_sets_exit.append(valid_data_exit)
            valid_names.append('valid')
        
        print("Training entrance congestion model...")
        self.model_enter = lgb.train(
            self.params,
            train_data_enter,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets_enter,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
        )
        
        print("\nTraining exit congestion model...")
        self.model_exit = lgb.train(
            self.params,
            train_data_exit,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets_exit,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
        )
        
        return {
            'enter_best_iteration': self.model_enter.best_iteration,
            'exit_best_iteration': self.model_exit.best_iteration
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict congestion for both entrance and exit
        
        Args:
            X: Features dataframe
            
        Returns:
            Tuple of (entrance_predictions, exit_predictions)
        """
        if self.model_enter is None or self.model_exit is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Ensure features are in correct order
        X = X[self.feature_columns]
        
        # Predict probabilities
        pred_enter_proba = self.model_enter.predict(X)
        pred_exit_proba = self.model_exit.predict(X)
        
        # Get class predictions
        pred_enter = np.argmax(pred_enter_proba, axis=1)
        pred_exit = np.argmax(pred_exit_proba, axis=1)
        
        return pred_enter, pred_exit
    
    def get_feature_importance(self, importance_type: str = 'gain') -> dict:
        """
        Get feature importance for both models
        
        Args:
            importance_type: Type of importance ('gain', 'split', or 'cover')
            
        Returns:
            Dictionary with feature importance for both models
        """
        if self.model_enter is None or self.model_exit is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance_enter = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model_enter.feature_importance(importance_type=importance_type)
        }).sort_values('importance', ascending=False)
        
        importance_exit = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model_exit.feature_importance(importance_type=importance_type)
        }).sort_values('importance', ascending=False)
        
        return {
            'enter': importance_enter,
            'exit': importance_exit
        }
    
    def save(self, path: str):
        """Save model to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        self.model_enter.save_model(str(path / 'model_enter.txt'))
        self.model_exit.save_model(str(path / 'model_exit.txt'))
        
        # Save feature columns
        joblib.dump(self.feature_columns, path / 'feature_columns.pkl')
        joblib.dump(self.params, path / 'params.pkl')
        
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        path = Path(path)
        
        # Load models
        self.model_enter = lgb.Booster(model_file=str(path / 'model_enter.txt'))
        self.model_exit = lgb.Booster(model_file=str(path / 'model_exit.txt'))
        
        # Load feature columns
        self.feature_columns = joblib.load(path / 'feature_columns.pkl')
        self.params = joblib.load(path / 'params.pkl')
        
        print(f"Model loaded from {path}")
