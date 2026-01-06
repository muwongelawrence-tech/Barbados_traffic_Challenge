"""
Data loading utilities for Barbados Traffic Challenge
"""
import pandas as pd
import os
from typing import Tuple, Optional


def load_training_data(data_path: str = 'Train.csv') -> pd.DataFrame:
    """
    Load and parse training data
    
    Args:
        data_path: Path to Train.csv file
        
    Returns:
        DataFrame with training data
    """
    df = pd.read_csv(data_path)
    
    # Convert timestamps to datetime
    df['datetimestamp_start'] = pd.to_datetime(df['datetimestamp_start'])
    df['datetimestamp_end'] = pd.to_datetime(df['datetimestamp_end'])
    df['video_time'] = pd.to_datetime(df['video_time'])
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def load_test_data(data_path: str = 'TestInputSegments.csv') -> pd.DataFrame:
    """
    Load and parse test data
    
    Args:
        data_path: Path to TestInputSegments.csv file
        
    Returns:
        DataFrame with test data
    """
    df = pd.read_csv(data_path)
    
    # Convert timestamps to datetime
    df['datetimestamp_start'] = pd.to_datetime(df['datetimestamp_start'])
    df['datetimestamp_end'] = pd.to_datetime(df['datetimestamp_end'])
    df['video_time'] = pd.to_datetime(df['video_time'])
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def load_sample_submission(data_path: str = 'SampleSubmission.csv') -> pd.DataFrame:
    """
    Load sample submission file to understand format
    
    Args:
        data_path: Path to SampleSubmission.csv file
        
    Returns:
        DataFrame with sample submission format
    """
    return pd.read_csv(data_path)


def get_class_mapping() -> dict:
    """
    Get mapping of congestion classes to integers
    
    Returns:
        Dictionary mapping class names to integers
    """
    return {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }


def get_inverse_class_mapping() -> dict:
    """
    Get mapping of integers to congestion classes
    
    Returns:
        Dictionary mapping integers to class names
    """
    return {
        0: 'free flowing',
        1: 'light delay',
        2: 'moderate delay',
        3: 'heavy delay'
    }
