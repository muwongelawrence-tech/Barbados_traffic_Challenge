"""
Phase 2b: Advanced Feature Engineering
Create interaction and temporal features
"""
import pandas as pd
import numpy as np

def create_advanced_features(df, video_feature_cols):
    """
    Create advanced features from existing data
    """
    df = df.copy()
    
    print("Creating advanced features...")
    
    # 1. Interaction features (motion × hour)
    df['motion_hour_interaction'] = df['current_motion_mean'] * df['hour']
    df['occupancy_hour_interaction'] = df['current_occupancy_mean'] * df['hour']
    
    # 2. Motion trend (change over time)
    if 'motion_lag_1' in df.columns and 'motion_lag_2' in df.columns:
        df['motion_trend_1'] = df['motion_lag_1'] - df['motion_lag_2']
        df['motion_trend_2'] = df['motion_lag_2'] - df.get('motion_lag_3', df['motion_lag_2'])
    
    # 3. Occupancy trend
    if 'occupancy_lag_1' in df.columns and 'occupancy_lag_2' in df.columns:
        df['occupancy_trend_1'] = df['occupancy_lag_1'] - df['occupancy_lag_2']
        df['occupancy_trend_2'] = df['occupancy_lag_2'] - df.get('occupancy_lag_3', df['occupancy_lag_2'])
    
    # 4. Rolling statistics (3-segment window)
    motion_lags = [col for col in df.columns if 'motion_lag' in col]
    if len(motion_lags) >= 3:
        df['motion_rolling_mean_3'] = df[motion_lags[:3]].mean(axis=1)
        df['motion_rolling_std_3'] = df[motion_lags[:3]].std(axis=1)
        df['motion_rolling_max_3'] = df[motion_lags[:3]].max(axis=1)
    
    # 5. Time-of-day features
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
    df['is_midday'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    df['is_night'] = ((df['hour'] >= 20) | (df['hour'] <= 6)).astype(int)
    
    # 6. Motion intensity categories
    df['motion_category'] = pd.cut(df['current_motion_mean'], 
                                    bins=[0, 2, 5, 10, 20], 
                                    labels=[0, 1, 2, 3])
    df['motion_category'] = df['motion_category'].fillna(0).astype(int)
    
    # 7. Occupancy categories
    df['occupancy_category'] = pd.cut(df['current_occupancy_mean'], 
                                       bins=[0, 0.01, 0.03, 0.05, 1], 
                                       labels=[0, 1, 2, 3])
    df['occupancy_category'] = df['occupancy_category'].fillna(0).astype(int)
    
    # 8. Location-specific features (if location is encoded)
    if 'location' in df.columns:
        for loc in df['location'].unique():
            df[f'is_location_{loc}'] = (df['location'] == loc).astype(int)
    
    # 9. Congestion proxy ratios
    if 'current_vehicle_count_proxy' in df.columns and 'current_flow_rate_proxy' in df.columns:
        df['vehicle_flow_ratio'] = df['current_vehicle_count_proxy'] / (df['current_flow_rate_proxy'] + 1e-6)
    
    # 10. Motion variability
    if 'current_motion_std' in df.columns and 'current_motion_mean' in df.columns:
        df['motion_cv'] = df['current_motion_std'] / (df['current_motion_mean'] + 1e-6)  # Coefficient of variation
    
    print(f"✓ Created {len(df.columns) - 28} new features")
    print(f"  Total features: {len(df.columns)}")
    
    return df

if __name__ == "__main__":
    print("Feature engineering module loaded")
    print("Use: create_advanced_features(df, video_feature_cols)")
