"""
Temporal feature engineering (no video processing required)
Fast baseline features for quick submission
"""
import pandas as pd
import numpy as np
from typing import Dict


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from dataframe
    
    Args:
        df: DataFrame with datetime columns
        
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['video_time'].dt.hour
    df['minute'] = df['video_time'].dt.minute
    df['day_of_week'] = df['video_time'].dt.dayofweek
    
    # Cyclical encoding for hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for minute (60-minute cycle)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    # Rush hour indicators
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    
    # Time of day categories
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features
    
    Args:
        df: DataFrame with categorical columns
        
    Returns:
        DataFrame with encoded features
    """
    df = df.copy()
    
    # Encode view_label (camera position)
    df['camera_id'] = df['view_label'].str.extract(r'#(\d+)').astype(int)
    
    # Encode signaling
    signal_mapping = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    df['signaling_encoded'] = df['signaling'].map(signal_mapping).fillna(0)
    
    # One-hot encode time_of_day if it exists
    if 'time_of_day' in df.columns:
        time_dummies = pd.get_dummies(df['time_of_day'], prefix='time')
        df = pd.concat([df, time_dummies], axis=1)
    
    return df


def create_lag_features(df: pd.DataFrame, target_cols: list, lags: list = [1, 2, 3, 5]) -> pd.DataFrame:
    """
    Create lag features for temporal patterns
    
    Args:
        df: DataFrame sorted by time
        target_cols: Columns to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    # Sort by camera and time
    df = df.sort_values(['camera_id', 'time_segment_id'])
    
    for col in target_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby('camera_id')[col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, target_cols: list, windows: list = [3, 5, 10]) -> pd.DataFrame:
    """
    Create rolling window statistics
    
    Args:
        df: DataFrame sorted by time
        target_cols: Columns to create rolling features for
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    # Sort by camera and time
    df = df.sort_values(['camera_id', 'time_segment_id'])
    
    for col in target_cols:
        for window in windows:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}'] = (
                df.groupby('camera_id')[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # Rolling std
            df[f'{col}_rolling_std_{window}'] = (
                df.groupby('camera_id')[col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
    
    return df


def prepare_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Complete feature engineering pipeline for baseline model
    
    Args:
        df: Raw dataframe
        is_training: Whether this is training data
        
    Returns:
        DataFrame with all engineered features
    """
    # Extract temporal features
    df = extract_temporal_features(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # If training, we can create lag features from targets
    if is_training:
        # Encode targets for lag features
        from src.data.data_loader import get_class_mapping
        class_map = get_class_mapping()
        
        df['congestion_enter_encoded'] = df['congestion_enter_rating'].map(class_map)
        df['congestion_exit_encoded'] = df['congestion_exit_rating'].map(class_map)
        
        # Create lag features
        df = create_lag_features(
            df,
            target_cols=['congestion_enter_encoded', 'congestion_exit_encoded'],
            lags=[1, 2, 3, 5]
        )
        
        # Create rolling features
        df = create_rolling_features(
            df,
            target_cols=['congestion_enter_encoded', 'congestion_exit_encoded'],
            windows=[3, 5, 10]
        )
    
    return df


def get_feature_columns() -> list:
    """
    Get list of feature columns for model training
    
    Returns:
        List of feature column names
    """
    base_features = [
        'hour', 'minute', 'day_of_week',
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
        'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
        'camera_id', 'signaling_encoded', 'time_segment_id'
    ]
    
    # Add time of day dummies
    time_features = ['time_afternoon', 'time_evening', 'time_morning', 'time_night']
    
    return base_features + time_features
