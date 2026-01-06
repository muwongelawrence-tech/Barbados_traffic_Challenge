#!/usr/bin/env python3
"""
Quick training script for baseline model
Trains on temporal features only for fast submission
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.data_loader import load_training_data, get_class_mapping
from src.features.temporal_features import prepare_features, get_feature_columns
from src.models.baseline_model import BaselineModel
from src.utils.metrics import evaluate_dual_output


def main():
    print("=" * 80)
    print("BARBADOS TRAFFIC CHALLENGE - BASELINE MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    print("\n[1/6] Loading training data...")
    train_df = load_training_data('Train.csv')
    print(f"   Loaded {len(train_df)} training samples")
    
    # Prepare features
    print("\n[2/6] Engineering temporal features...")
    train_df = prepare_features(train_df, is_training=True)
    print(f"   Created features for {len(train_df)} samples")
    
    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"   Using {len(feature_cols)} features")
    
    # Handle missing values from lag features
    print("\n[3/6] Handling missing values...")
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    
    # Prepare X and y
    X = train_df[feature_cols]
    
    # Encode targets
    class_map = get_class_mapping()
    y_enter = train_df['congestion_enter_rating'].map(class_map).values
    y_exit = train_df['congestion_exit_rating'].map(class_map).values
    
    print(f"   Features shape: {X.shape}")
    print(f"   Entrance labels: {len(y_enter)}")
    print(f"   Exit labels: {len(y_exit)}")
    
    # Split data
    print("\n[4/6] Splitting data (80/20 train/val)...")
    X_train, X_val, y_train_enter, y_val_enter, y_train_exit, y_val_exit = train_test_split(
        X, y_enter, y_exit, test_size=0.2, random_state=42, stratify=y_enter
    )
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Val samples: {len(X_val)}")
    
    # Train model
    print("\n[5/6] Training LightGBM model...")
    model = BaselineModel()
    
    history = model.fit(
        X_train, y_train_enter, y_train_exit,
        X_val, y_val_enter, y_val_exit,
        num_boost_round=500
    )
    
    print(f"\n   Entrance model best iteration: {history['enter_best_iteration']}")
    print(f"   Exit model best iteration: {history['exit_best_iteration']}")
    
    # Evaluate
    print("\n[6/6] Evaluating model...")
    pred_enter, pred_exit = model.predict(X_val)
    
    metrics = evaluate_dual_output(
        y_val_enter, pred_enter,
        y_val_exit, pred_exit,
        verbose=True
    )
    
    # Save model
    print("\nSaving model...")
    model.save('models/baseline')
    
    # Show feature importance
    print("\n" + "=" * 80)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 80)
    
    importance = model.get_feature_importance()
    
    print("\nFor ENTRANCE congestion:")
    print(importance['enter'].head(10).to_string(index=False))
    
    print("\nFor EXIT congestion:")
    print(importance['exit'].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Model saved to: models/baseline/")
    print(f"✅ Average Combined Score: {metrics['average']['avg_combined_score']:.4f}")
    print(f"\nNext step: Run 'python scripts/predict.py' to generate submission")
    print("=" * 80)


if __name__ == '__main__':
    main()
