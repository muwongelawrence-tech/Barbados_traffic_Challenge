#!/usr/bin/env python3
"""
Generate predictions and submission file
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.data.data_loader import load_test_data, get_class_mapping
from src.features.temporal_features import extract_temporal_features, encode_categorical_features, get_feature_columns
from src.models.baseline_model import BaselineModel
from src.inference.submission_generator import generate_submission, preview_submission


def main():
    print("=" * 80)
    print("BARBADOS TRAFFIC CHALLENGE - GENERATING SUBMISSION")
    print("=" * 80)
    
    # Load test data
    print("\n[1/5] Loading test data...")
    test_df = load_test_data('TestInputSegments.csv')
    print(f"   Loaded {len(test_df)} test samples")
    
    # Prepare features (no lag features for test data)
    print("\n[2/5] Engineering temporal features...")
    test_df = extract_temporal_features(test_df)
    test_df = encode_categorical_features(test_df)
    
    # Get feature columns
    feature_cols = get_feature_columns()
    
    # Handle missing features (lag features won't exist in test)
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    
    X_test = test_df[feature_cols]
    print(f"   Test features shape: {X_test.shape}")
    
    # Load model
    print("\n[3/5] Loading trained model...")
    model = BaselineModel()
    model.load('models/baseline')
    print("   Model loaded successfully")
    
    # Generate predictions
    print("\n[4/5] Generating predictions...")
    pred_enter, pred_exit = model.predict(X_test)
    print(f"   Generated {len(pred_enter)} entrance predictions")
    print(f"   Generated {len(pred_exit)} exit predictions")
    
    # Create submission
    print("\n[5/5] Creating submission file...")
    submission_df = generate_submission(
        pred_enter, pred_exit,
        test_df,
        output_path='submission.csv'
    )
    
    # Preview
    preview_submission(submission_df, n=20)
    
    print("\n" + "=" * 80)
    print("SUBMISSION GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Submission file: submission.csv")
    print(f"✅ Total predictions: {len(submission_df)}")
    print(f"\nNext step: Upload 'submission.csv' to Zindi platform")
    print("=" * 80)


if __name__ == '__main__':
    main()
