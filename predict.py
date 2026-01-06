"""
Prediction script for Barbados Traffic Challenge
Generates predictions for test set and creates submission file
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import argparse

import config
from src.data_preprocessing import DataPreprocessor
from src.utils import load_model, create_submission_file, validate_submission


def generate_predictions(test_df, preprocessor, enter_model_path, exit_model_path):
    """Generate predictions for test set"""
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    # Load models
    print(f"\nLoading models...")
    enter_model = load_model(enter_model_path)
    exit_model = load_model(exit_model_path)
    
    # Preprocess test data
    # First, we need to combine with train data to create lag features properly
    print("\nPreprocessing test data...")
    train_df, _ = preprocessor.load_data()
    
    # Combine train and test for proper lag feature creation
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_processed = preprocessor.preprocess(combined_df, is_train=False)
    
    # Extract only test portion
    test_processed = combined_processed.iloc[len(train_df):].reset_index(drop=True)
    
    # Get feature columns
    feature_cols = preprocessor.get_feature_columns(combined_processed)
    
    # Prepare features
    X_test = test_processed[feature_cols].fillna(0)
    
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions_enter = enter_model.predict(X_test)
    predictions_exit = exit_model.predict(X_test)
    
    print(f"✓ Predictions generated")
    
    # Show prediction distribution
    print(f"\nEnter rating predictions:")
    unique, counts = np.unique(predictions_enter, return_counts=True)
    for u, c in zip(unique, counts):
        label = config.CONGESTION_LEVELS_REVERSE[u]
        print(f"  {label}: {c} ({c/len(predictions_enter)*100:.1f}%)")
    
    print(f"\nExit rating predictions:")
    unique, counts = np.unique(predictions_exit, return_counts=True)
    for u, c in zip(unique, counts):
        label = config.CONGESTION_LEVELS_REVERSE[u]
        print(f"  {label}: {c} ({c/len(predictions_exit)*100:.1f}%)")
    
    return predictions_enter, predictions_exit, test_processed


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for test set')
    parser.add_argument(
        '--enter-model',
        default=os.path.join(config.MODELS_DIR, 'best_enter_model.pkl'),
        help='Path to enter rating model'
    )
    parser.add_argument(
        '--exit-model',
        default=os.path.join(config.MODELS_DIR, 'best_exit_model.pkl'),
        help='Path to exit rating model'
    )
    parser.add_argument(
        '--output',
        default=os.path.join(config.SUBMISSIONS_DIR, 'submission.csv'),
        help='Output submission file path'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load test data
    _, test_df = preprocessor.load_data()
    
    # Generate predictions
    predictions_enter, predictions_exit, test_processed = generate_predictions(
        test_df, preprocessor, args.enter_model, args.exit_model
    )
    
    # Create submission file
    submission_df = create_submission_file(
        test_df, predictions_enter, predictions_exit, args.output
    )
    
    # Validate submission
    is_valid = validate_submission(args.output)
    
    if is_valid:
        print(f"\n✅ Submission file ready: {args.output}")
        print(f"You can now upload this file to the Zindi platform!")
    else:
        print(f"\n❌ Submission validation failed. Please check the errors above.")
    
    return submission_df


if __name__ == "__main__":
    main()
