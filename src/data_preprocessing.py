"""
Data preprocessing module for Barbados Traffic Challenge
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import config


class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, train_path=None, test_path=None):
        """Load training and test data"""
        train_path = train_path or config.TRAIN_FILE
        test_path = test_path or config.TEST_FILE
        
        print(f"Loading training data from {train_path}...")
        train_df = pd.read_csv(train_path)
        
        print(f"Loading test data from {test_path}...")
        test_df = pd.read_csv(test_path)
        
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    
    def parse_datetime_features(self, df):
        """Extract datetime features from timestamp columns"""
        df = df.copy()
        
        # Convert to datetime
        df['datetimestamp_start'] = pd.to_datetime(df['datetimestamp_start'])
        df['datetimestamp_end'] = pd.to_datetime(df['datetimestamp_end'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract time features
        df['hour'] = df['datetimestamp_start'].dt.hour
        df['minute'] = df['datetimestamp_start'].dt.minute
        df['day_of_week'] = df['datetimestamp_start'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['datetimestamp_start'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        # Rush hour indicator
        df['is_rush_hour'] = (
            ((df['hour'] >= config.RUSH_HOUR_MORNING[0]) & 
             (df['hour'] < config.RUSH_HOUR_MORNING[1])) |
            ((df['hour'] >= config.RUSH_HOUR_EVENING[0]) & 
             (df['hour'] < config.RUSH_HOUR_EVENING[1]))
        ).astype(int)
        
        # Cyclical encoding for hour (to capture circular nature of time)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Duration of segment in seconds
        df['segment_duration'] = (
            df['datetimestamp_end'] - df['datetimestamp_start']
        ).dt.total_seconds()
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        df = df.copy()
        
        categorical_cols = ['view_label', 'signaling', 'time_of_day']
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            if fit:
                # Fit and transform
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Transform only
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[f'{col}_encoded'] = le.transform(df[col].astype(str))
        
        return df
    
    def create_lag_features(self, df, target_cols=None):
        """Create lag features based on historical congestion"""
        df = df.copy()
        
        if target_cols is None:
            target_cols = [config.TARGET_ENTER, config.TARGET_EXIT]
        
        # Sort by location and time
        df = df.sort_values(['view_label', 'datetimestamp_start'])
        
        for target in target_cols:
            # Only create lag features if target exists (training data)
            if target not in df.columns or df[target].isna().all():
                continue
            
            # Encode target for lag features
            if target not in self.label_encoders:
                le = LabelEncoder()
                df[f'{target}_encoded'] = le.fit_transform(df[target].astype(str))
                self.label_encoders[target] = le
            else:
                le = self.label_encoders[target]
                df[f'{target}_encoded'] = le.transform(df[target].astype(str))
            
            # Create lag features per location
            for lag in config.LAG_PERIODS:
                df[f'{target}_lag_{lag}'] = df.groupby('view_label')[f'{target}_encoded'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'{target}_rolling_mean_{window}'] = (
                    df.groupby('view_label')[f'{target}_encoded']
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                df[f'{target}_rolling_std_{window}'] = (
                    df.groupby('view_label')[f'{target}_encoded']
                    .transform(lambda x: x.rolling(window, min_periods=1).std())
                )
        
        return df
    
    def encode_targets(self, df):
        """Encode target variables to numeric"""
        df = df.copy()
        
        for target in [config.TARGET_ENTER, config.TARGET_EXIT]:
            if target in df.columns and not df[target].isna().all():
                df[f'{target}_numeric'] = df[target].map(config.CONGESTION_LEVELS)
        
        return df
    
    def preprocess(self, df, is_train=True):
        """Complete preprocessing pipeline"""
        print(f"\nPreprocessing {'training' if is_train else 'test'} data...")
        
        # Parse datetime features
        df = self.parse_datetime_features(df)
        print("✓ Datetime features extracted")
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=is_train)
        print("✓ Categorical features encoded")
        
        # Create lag features (only for train, or test with historical data)
        if is_train:
            df = self.create_lag_features(df)
            print("✓ Lag features created")
        
        # Encode targets (only for train)
        if is_train:
            df = self.encode_targets(df)
            print("✓ Targets encoded")
        
        return df
    
    def get_feature_columns(self, df):
        """Get list of feature columns for modeling"""
        # Exclude non-feature columns
        exclude_cols = [
            'responseId', 'ID_enter', 'ID_exit', 'videos', 'video_time',
            'datetimestamp_start', 'datetimestamp_end', 'date',
            'view_label', 'signaling', 'time_of_day',  # Original categorical
            config.TARGET_ENTER, config.TARGET_EXIT,  # Original targets
            f'{config.TARGET_ENTER}_numeric', f'{config.TARGET_EXIT}_numeric',  # Numeric targets
            f'{config.TARGET_ENTER}_encoded', f'{config.TARGET_EXIT}_encoded',  # Encoded targets
            'cycle_phase'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        return feature_cols
    
    def prepare_train_test_split(self, df, test_size=0.2, stratify_col=None):
        """Split data into train and validation sets"""
        feature_cols = self.get_feature_columns(df)
        
        X = df[feature_cols]
        y_enter = df[f'{config.TARGET_ENTER}_numeric']
        y_exit = df[f'{config.TARGET_EXIT}_numeric']
        
        # Time-aware split (use last test_size% as validation)
        split_idx = int(len(df) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_enter_train = y_enter.iloc[:split_idx]
        y_enter_val = y_enter.iloc[split_idx:]
        y_exit_train = y_exit.iloc[:split_idx]
        y_exit_val = y_exit.iloc[split_idx:]
        
        print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")
        
        return X_train, X_val, y_enter_train, y_enter_val, y_exit_train, y_exit_val


def main():
    """Test preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Load data
    train_df, test_df = preprocessor.load_data()
    
    # Preprocess training data
    train_processed = preprocessor.preprocess(train_df, is_train=True)
    
    # Get feature columns
    feature_cols = preprocessor.get_feature_columns(train_processed)
    print(f"\n{len(feature_cols)} feature columns created:")
    print(feature_cols[:20], "...")
    
    # Check for missing values
    print(f"\nMissing values in features: {train_processed[feature_cols].isna().sum().sum()}")
    
    # Prepare train/val split
    X_train, X_val, y_enter_train, y_enter_val, y_exit_train, y_exit_val = \
        preprocessor.prepare_train_test_split(train_processed)
    
    print("\n✓ Preprocessing pipeline test completed successfully!")


if __name__ == "__main__":
    main()
