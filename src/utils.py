"""
Utility functions for Barbados Traffic Challenge
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, balanced_accuracy_score
)
import pickle
import json
import config


def calculate_metrics(y_true, y_pred, target_name=''):
    """Calculate comprehensive metrics for predictions"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_per_class': f1_score(y_true, y_pred, average=None).tolist()
    }
    
    print(f"\n{'='*60}")
    print(f"Metrics for {target_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=list(config.CONGESTION_LEVELS.keys())
    ))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, target_name='', save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=list(config.CONGESTION_LEVELS.keys()),
        yticklabels=list(config.CONGESTION_LEVELS.keys())
    )
    plt.title(f'Confusion Matrix - {target_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance"""
    # Get feature importance (works for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        importances = model.feature_importance()
    else:
        print("Model does not have feature importance attribute")
        return
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return importance_df


def save_model(model, filepath):
    """Save model to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def save_metrics(metrics, filepath):
    """Save metrics to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def get_class_weights(y):
    """Calculate class weights for imbalanced data"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))
    
    print(f"\nClass weights: {class_weight_dict}")
    return class_weight_dict


def analyze_predictions_by_location(df, y_true_col, y_pred_col, location_col='view_label'):
    """Analyze prediction performance by location"""
    df = df.copy()
    df['correct'] = (df[y_true_col] == df[y_pred_col]).astype(int)
    
    location_stats = df.groupby(location_col).agg({
        'correct': ['mean', 'count']
    }).round(4)
    
    print(f"\n{'='*60}")
    print("Performance by Location")
    print(f"{'='*60}")
    print(location_stats)
    
    return location_stats


def analyze_predictions_by_time(df, y_true_col, y_pred_col, time_col='hour'):
    """Analyze prediction performance by time"""
    df = df.copy()
    df['correct'] = (df[y_true_col] == df[y_pred_col]).astype(int)
    
    time_stats = df.groupby(time_col).agg({
        'correct': ['mean', 'count']
    }).round(4)
    
    print(f"\n{'='*60}")
    print(f"Performance by {time_col}")
    print(f"{'='*60}")
    print(time_stats)
    
    return time_stats


def create_submission_file(test_df, predictions_enter, predictions_exit, output_path):
    """Create submission file in the required format"""
    submission_rows = []
    
    for idx, row in test_df.iterrows():
        time_segment = row['time_segment_id']
        view_label = row['view_label']
        
        # Entry rating prediction
        enter_pred = config.CONGESTION_LEVELS_REVERSE[predictions_enter[idx]]
        submission_rows.append({
            'ID': f'time_segment_{time_segment}_{view_label}_congestion_enter_rating',
            'Target': enter_pred,
            'Target_Accuracy': enter_pred  # As per sample submission format
        })
        
        # Exit rating prediction
        exit_pred = config.CONGESTION_LEVELS_REVERSE[predictions_exit[idx]]
        submission_rows.append({
            'ID': f'time_segment_{time_segment}_{view_label}_congestion_exit_rating',
            'Target': exit_pred,
            'Target_Accuracy': exit_pred  # As per sample submission format
        })
    
    submission_df = pd.DataFrame(submission_rows)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission file created: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    
    return submission_df


def validate_submission(submission_path, sample_path=None):
    """Validate submission file format"""
    sample_path = sample_path or config.SAMPLE_SUBMISSION_FILE
    
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    
    print(f"\n{'='*60}")
    print("Submission Validation")
    print(f"{'='*60}")
    
    # Check shape
    if submission.shape[0] != sample.shape[0]:
        print(f"❌ Row count mismatch: {submission.shape[0]} vs {sample.shape[0]}")
        return False
    else:
        print(f"✓ Row count: {submission.shape[0]}")
    
    # Check columns
    if set(submission.columns) != set(sample.columns):
        print(f"❌ Column mismatch")
        print(f"Expected: {sample.columns.tolist()}")
        print(f"Got: {submission.columns.tolist()}")
        return False
    else:
        print(f"✓ Columns: {submission.columns.tolist()}")
    
    # Check for missing values
    if submission.isna().any().any():
        print(f"❌ Missing values found")
        return False
    else:
        print(f"✓ No missing values")
    
    # Check target values are valid
    valid_targets = set(config.CONGESTION_LEVELS.keys())
    invalid_targets = set(submission['Target'].unique()) - valid_targets
    if invalid_targets:
        print(f"❌ Invalid target values: {invalid_targets}")
        return False
    else:
        print(f"✓ All target values are valid")
    
    print(f"\n✅ Submission validation passed!")
    return True


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
