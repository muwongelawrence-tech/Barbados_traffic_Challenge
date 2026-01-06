#!/usr/bin/env python3
"""
Quick training script with configurable hyperparameters
Usage: python scripts/train_quick.py --config v2_complexity
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.data.data_loader import load_training_data, get_class_mapping
from src.features.temporal_features import prepare_features, get_feature_columns
from src.models.baseline_model import BaselineModel
from src.utils.metrics import evaluate_dual_output
from config.model_configs import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline', 
                       help='Config name: baseline, v2_complexity, v3_lower_lr, etc.')
    parser.add_argument('--class-weights', action='store_true',
                       help='Use class weights for imbalance')
    parser.add_argument('--num-boost-round', type=int, default=500,
                       help='Number of boosting rounds')
    parser.add_argument('--output-dir', type=str, default='models/quick',
                       help='Output directory for model')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"QUICK TRAINING - Config: {args.config}")
    print("=" * 80)
    
    # Load data
    print("\n[1/7] Loading training data...")
    train_df = load_training_data('Train.csv')
    print(f"   Loaded {len(train_df)} training samples")
    
    # Prepare features
    print("\n[2/7] Engineering temporal features...")
    train_df = prepare_features(train_df, is_training=True)
    
    # Get feature columns
    feature_cols = get_feature_columns()
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    
    X = train_df[feature_cols]
    
    # Encode targets
    class_map = get_class_mapping()
    y_enter = train_df['congestion_enter_rating'].map(class_map).values
    y_exit = train_df['congestion_exit_rating'].map(class_map).values
    
    print(f"   Features shape: {X.shape}")
    
    # Split data
    print("\n[3/7] Splitting data (80/20 train/val)...")
    X_train, X_val, y_train_enter, y_val_enter, y_train_exit, y_val_exit = train_test_split(
        X, y_enter, y_exit, test_size=0.2, random_state=42, stratify=y_enter
    )
    
    # Class weights
    if args.class_weights:
        print("\n[4/7] Computing class weights...")
        weights_enter = compute_class_weight('balanced', classes=np.unique(y_train_enter), y=y_train_enter)
        weights_exit = compute_class_weight('balanced', classes=np.unique(y_train_exit), y=y_train_exit)
        print(f"   Entrance weights: {weights_enter}")
        print(f"   Exit weights: {weights_exit}")
        # Note: LightGBM uses sample weights, not class weights directly
        # We'll need to map these to samples
    else:
        print("\n[4/7] No class weights (using default)")
    
    # Get config
    print(f"\n[5/7] Loading config: {args.config}")
    params = get_config(args.config)
    print(f"   Params: {params}")
    
    # Train model
    print("\n[6/7] Training LightGBM model...")
    model = BaselineModel(params=params)
    
    history = model.fit(
        X_train, y_train_enter, y_train_exit,
        X_val, y_val_enter, y_val_exit,
        num_boost_round=args.num_boost_round
    )
    
    # Evaluate
    print("\n[7/7] Evaluating model...")
    pred_enter, pred_exit = model.predict(X_val)
    
    metrics = evaluate_dual_output(
        y_val_enter, pred_enter,
        y_val_exit, pred_exit,
        verbose=True
    )
    
    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    model.save(args.output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Combined Score: {metrics['average']['avg_combined_score']:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 80)
    
    return metrics['average']['avg_combined_score']


if __name__ == '__main__':
    score = main()
    sys.exit(0)
