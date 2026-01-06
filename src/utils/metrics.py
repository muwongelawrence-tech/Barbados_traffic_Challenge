"""
Custom evaluation metrics for Barbados Traffic Challenge
Combined metric: 0.7 * Macro-F1 + 0.3 * Accuracy
"""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


def calculate_combined_score(y_true, y_pred) -> float:
    """
    Calculate the combined competition metric
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Combined score (0.7 * F1 + 0.3 * Accuracy)
    """
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    combined = 0.7 * f1 + 0.3 * acc
    
    return combined


def evaluate_model(y_true, y_pred, verbose: bool = True) -> dict:
    """
    Comprehensive model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        verbose: Whether to print results
        
    Returns:
        Dictionary with all metrics
    """
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    combined = calculate_combined_score(y_true, y_pred)
    
    metrics = {
        'macro_f1': f1_macro,
        'weighted_f1': f1_weighted,
        'accuracy': acc,
        'combined_score': combined
    }
    
    if verbose:
        print("=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Macro F1-Score:     {f1_macro:.4f} (70% weight)")
        print(f"Accuracy:           {acc:.4f} (30% weight)")
        print(f"Combined Score:     {combined:.4f} ⭐")
        print(f"Weighted F1-Score:  {f1_weighted:.4f}")
        print("=" * 60)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['free flowing', 'light delay', 
                                               'moderate delay', 'heavy delay']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
    
    return metrics


def evaluate_dual_output(y_true_enter, y_pred_enter, y_true_exit, y_pred_exit, verbose: bool = True) -> dict:
    """
    Evaluate both entrance and exit predictions
    
    Args:
        y_true_enter: True entrance labels
        y_pred_enter: Predicted entrance labels
        y_true_exit: True exit labels
        y_pred_exit: Predicted exit labels
        verbose: Whether to print results
        
    Returns:
        Dictionary with metrics for both outputs
    """
    if verbose:
        print("\n" + "=" * 60)
        print("ENTRANCE CONGESTION EVALUATION")
        print("=" * 60)
    
    enter_metrics = evaluate_model(y_true_enter, y_pred_enter, verbose=verbose)
    
    if verbose:
        print("\n" + "=" * 60)
        print("EXIT CONGESTION EVALUATION")
        print("=" * 60)
    
    exit_metrics = evaluate_model(y_true_exit, y_pred_exit, verbose=verbose)
    
    # Average metrics
    avg_metrics = {
        'avg_macro_f1': (enter_metrics['macro_f1'] + exit_metrics['macro_f1']) / 2,
        'avg_accuracy': (enter_metrics['accuracy'] + exit_metrics['accuracy']) / 2,
        'avg_combined_score': (enter_metrics['combined_score'] + exit_metrics['combined_score']) / 2
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("AVERAGE METRICS (ENTER + EXIT)")
        print("=" * 60)
        print(f"Average Macro F1:      {avg_metrics['avg_macro_f1']:.4f}")
        print(f"Average Accuracy:      {avg_metrics['avg_accuracy']:.4f}")
        print(f"Average Combined:      {avg_metrics['avg_combined_score']:.4f} ⭐⭐")
        print("=" * 60)
    
    return {
        'enter': enter_metrics,
        'exit': exit_metrics,
        'average': avg_metrics
    }
