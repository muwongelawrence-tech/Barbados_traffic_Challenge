"""
Generate submission file for Zindi competition
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.data_loader import get_inverse_class_mapping


def generate_submission(predictions_enter: np.ndarray, predictions_exit: np.ndarray,
                       test_df: pd.DataFrame, output_path: str = 'submission.csv') -> pd.DataFrame:
    """
    Generate submission file in required format
    
    Args:
        predictions_enter: Predicted entrance congestion classes (integers)
        predictions_exit: Predicted exit congestion classes (integers)
        test_df: Test dataframe with ID columns
        output_path: Path to save submission file
        
    Returns:
        Submission dataframe
    """
    # Get class mapping
    class_map = get_inverse_class_mapping()
    
    # Convert predictions to class names
    pred_enter_names = [class_map[p] for p in predictions_enter]
    pred_exit_names = [class_map[p] for p in predictions_exit]
    
    # Create submission rows
    submission_rows = []
    
    for idx, row in test_df.iterrows():
        # Entrance prediction
        submission_rows.append({
            'ID': row['ID_enter'],
            'Target': pred_enter_names[idx],
            'Target_Accuracy': pred_enter_names[idx]
        })
        
        # Exit prediction
        submission_rows.append({
            'ID': row['ID_exit'],
            'Target': pred_exit_names[idx],
            'Target_Accuracy': pred_exit_names[idx]
        })
    
    # Create dataframe
    submission_df = pd.DataFrame(submission_rows)
    
    # Save to file
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    
    # Validate format
    validate_submission(submission_df)
    
    return submission_df


def validate_submission(submission_df: pd.DataFrame) -> bool:
    """
    Validate submission format
    
    Args:
        submission_df: Submission dataframe
        
    Returns:
        True if valid, raises error otherwise
    """
    # Check columns
    required_cols = ['ID', 'Target', 'Target_Accuracy']
    if list(submission_df.columns) != required_cols:
        raise ValueError(f"Columns must be {required_cols}, got {list(submission_df.columns)}")
    
    # Check for missing values
    if submission_df.isnull().any().any():
        raise ValueError("Submission contains missing values")
    
    # Check valid classes
    valid_classes = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
    invalid_target = ~submission_df['Target'].isin(valid_classes)
    invalid_accuracy = ~submission_df['Target_Accuracy'].isin(valid_classes)
    
    if invalid_target.any():
        raise ValueError(f"Invalid classes in Target column: {submission_df[invalid_target]['Target'].unique()}")
    
    if invalid_accuracy.any():
        raise ValueError(f"Invalid classes in Target_Accuracy column: {submission_df[invalid_accuracy]['Target_Accuracy'].unique()}")
    
    print("✅ Submission format is valid!")
    print(f"   - {len(submission_df)} predictions")
    print(f"   - Columns: {list(submission_df.columns)}")
    print(f"   - No missing values")
    print(f"   - All classes valid")
    
    return True


def preview_submission(submission_df: pd.DataFrame, n: int = 10):
    """
    Preview submission file
    
    Args:
        submission_df: Submission dataframe
        n: Number of rows to preview
    """
    print("\nSubmission Preview:")
    print("=" * 80)
    print(submission_df.head(n))
    print("=" * 80)
    
    # Show class distribution
    print("\nClass Distribution:")
    print("-" * 40)
    print("Target column:")
    print(submission_df['Target'].value_counts())
    print("\nTarget_Accuracy column:")
    print(submission_df['Target_Accuracy'].value_counts())
