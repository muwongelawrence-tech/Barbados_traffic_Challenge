"""
Forecasting Model - Predict 5 minutes ahead
This is the CORRECT approach for the competition

Key differences from previous attempts:
1. Predict FUTURE congestion (t+5 minutes)
2. Use time-series structure properly
3. Prepare for video features (will add later)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

import config
from src.data_preprocessing import DataPreprocessor

print("="*60)
print("FORECASTING MODEL - 5 MINUTES AHEAD")
print("="*60)
print("Correct approach:")
print("- Predict congestion 5 minutes into future")
print("- Use features from past 15 minutes")
print("- Structure: t-15 to t → predict t+5")
print("="*60)

# Load data
preprocessor = DataPreprocessor()
train_df, test_df = preprocessor.load_data()

print(f"\nData loaded:")
print(f"  Train: {len(train_df)} segments")
print(f"  Test: {len(test_df)} segments")

# Create forecasting dataset
print("\n" + "="*60)
print("CREATING FORECASTING DATASET")
print("="*60)

def create_forecasting_dataset(df, forecast_horizon=5):
    """
    Create dataset for forecasting
    
    Args:
        df: DataFrame with time-series data
        forecast_horizon: Minutes ahead to predict (default: 5)
    
    Returns:
        X: Features (current + historical)
        y: Target (5 minutes ahead)
    """
    # Sort by location and time
    df = df.sort_values(['view_label', 'time_segment_id']).reset_index(drop=True)
    
    features_list = []
    targets_enter = []
    targets_exit = []
    
    # For each location
    for location in df['view_label'].unique():
        location_df = df[df['view_label'] == location].reset_index(drop=True)
        
        # For each time point (need at least 15 past + 5 future)
        for i in range(15, len(location_df) - forecast_horizon):
            # Features: past 15 minutes
            past_window = location_df.iloc[i-15:i]
            
            # Target: 5 minutes ahead
            future_point = location_df.iloc[i + forecast_horizon]
            
            # Extract features from past window
            feature_dict = {
                'current_segment': location_df.iloc[i]['time_segment_id'],
                'location': location,
                'hour': location_df.iloc[i]['datetimestamp_start'].split()[1].split(':')[0],
                'signaling': location_df.iloc[i]['signaling'],
            }
            
            # Add lag features (congestion in past minutes)
            for lag in [1, 2, 3, 5, 10, 15]:
                if i - lag >= 0:
                    past_row = location_df.iloc[i - lag]
                    feature_dict[f'enter_lag_{lag}'] = past_row.get('congestion_enter_rating', 'free flowing')
                    feature_dict[f'exit_lag_{lag}'] = past_row.get('congestion_exit_rating', 'free flowing')
            
            features_list.append(feature_dict)
            targets_enter.append(future_point.get('congestion_enter_rating', 'free flowing'))
            targets_exit.append(future_point.get('congestion_exit_rating', 'free flowing'))
    
    return pd.DataFrame(features_list), targets_enter, targets_exit

print("Creating forecasting dataset...")
print("  Using past 15 minutes to predict 5 minutes ahead")

X_train, y_enter_train, y_exit_train = create_forecasting_dataset(train_df, forecast_horizon=5)

print(f"\nForecasting dataset created:")
print(f"  Samples: {len(X_train)}")
print(f"  Features: {X_train.columns.tolist()}")

# Encode features
print("\n" + "="*60)
print("ENCODING FEATURES")
print("="*60)

from sklearn.preprocessing import LabelEncoder

# Encode categorical features
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

# Encode targets
le_target = LabelEncoder()
y_enter_encoded = le_target.fit_transform(y_enter_train)
y_exit_encoded = le_target.transform(y_exit_train)

print(f"Features encoded: {X_train.shape}")
print(f"Target classes: {le_target.classes_}")

# Split for validation (last 20% chronologically)
split_idx = int(len(X_train) * 0.8)
X_train_split = X_train.iloc[:split_idx]
X_val_split = X_train.iloc[split_idx:]
y_enter_train_split = y_enter_encoded[:split_idx]
y_enter_val_split = y_enter_encoded[split_idx:]

print(f"\nTrain/Val split:")
print(f"  Train: {len(X_train_split)}")
print(f"  Val: {len(X_val_split)}")

# Train forecasting model
print("\n" + "="*60)
print("TRAINING FORECASTING MODEL")
print("="*60)

model = LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1
)

print("Training model to forecast 5 minutes ahead...")
model.fit(X_train_split, y_enter_train_split)

# Evaluate
y_pred_val = model.predict(X_val_split)

acc = accuracy_score(y_enter_val_split, y_pred_val)
f1_macro = f1_score(y_enter_val_split, y_pred_val, average='macro')
f1_weighted = f1_score(y_enter_val_split, y_pred_val, average='weighted')

print(f"\nValidation Performance (5-min ahead forecast):")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 (macro): {f1_macro:.4f}")
print(f"  F1 (weighted): {f1_weighted:.4f}")

print("\nClassification Report:")
y_pred_labels = le_target.inverse_transform(y_pred_val)
y_val_labels = le_target.inverse_transform(y_enter_val_split)
print(classification_report(y_val_labels, y_pred_labels, zero_division=0))

# Save model
os.makedirs(config.MODELS_DIR, exist_ok=True)
model_path = os.path.join(config.MODELS_DIR, 'forecasting_model_v2.0.pkl')

model_data = {
    'model': model,
    'label_encoder': le_target,
    'feature_columns': X_train.columns.tolist(),
    'forecast_horizon': 5
}

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to {model_path}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Built forecasting model (5-min ahead)")
print(f"✓ Validation F1 (macro): {f1_macro:.4f}")
print(f"✓ This is the CORRECT approach for competition")
print(f"\nNext steps:")
print(f"  1. Add video features when available")
print(f"  2. Test on proper test structure (15-min windows)")
print(f"  3. Generate submission for future segments")
print("="*60)
