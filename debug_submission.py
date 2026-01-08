"""
Debug Submission Format Issue
Find out why predictions are all "free flowing"
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import re

print("="*60)
print("DEBUGGING SUBMISSION FORMAT")
print("="*60)

# Load ensemble
with open('models/ensemble_model_v3.2.pkl', 'rb') as f:
    ensemble_data = pickle.load(f)

models = ensemble_data['models']
le_target = ensemble_data['label_encoder']
feature_columns = ensemble_data['feature_columns']
video_feature_cols = ensemble_data['video_feature_cols']

print(f"\nModel info:")
print(f"  Features expected: {len(feature_columns)}")
print(f"  Classes: {le_target.classes_}")

# Load data
train_df = pd.read_csv('Train.csv')
sample_df = pd.read_csv('SampleSubmission.csv')
video_features_df = pd.read_csv('video_features.csv')

print(f"\nData:")
print(f"  Train: {len(train_df)}")
print(f"  Sample: {len(sample_df)}")

# Check sample submission format
print(f"\nSample submission columns: {sample_df.columns.tolist()}")
print(f"\nFirst few IDs:")
print(sample_df['ID'].head(3).tolist())

# Parse one ID
test_id = sample_df['ID'].iloc[0]
print(f"\nParsing ID: {test_id}")

location = None
segment_id = None
rating_type = None

match = re.search(r'time_segment_(\d+)_(.*?)_congestion_(enter|exit)_rating', test_id)
if match:
    segment_id = int(match.group(1))
    location = match.group(2)
    rating_type = match.group(3)
    
    print(f"  Segment ID: {segment_id}")
    print(f"  Location: {location}")
    print(f"  Rating type: {rating_type}")
    
    # Estimate hour
    estimated_hour = (segment_id // 60) % 24
    print(f"  Estimated hour: {estimated_hour}")
else:
    print("  Failed to parse ID")
    estimated_hour = 0

# Check training data for this location/hour
train_df['hour'] = train_df['datetimestamp_start'].apply(lambda x: int(str(x).split()[1].split(':')[0]))

if location:
    location_hour_data = train_df[(train_df['view_label'] == location) & (train_df['hour'] == estimated_hour)]
else:
    location_hour_data = pd.DataFrame()
print(f"\nTraining data for {location}, hour {estimated_hour}:")
print(f"  Count: {len(location_hour_data)}")

if len(location_hour_data) > 0:
    print(f"  Congestion distribution:")
    print(location_hour_data['congestion_enter_rating'].value_counts())

# Test prediction pipeline
print("\n" + "="*60)
print("TESTING PREDICTION PIPELINE")
print("="*60)

# Create test feature
video_medians = {col: video_features_df[col].median() for col in video_feature_cols}

if len(location_hour_data) > 0:
    median_row = location_hour_data.iloc[len(location_hour_data) // 2]
    
    feature_dict = {
        'current_segment': median_row['time_segment_id'],
        'location': location,
        'hour': estimated_hour,
        'signaling': median_row['signaling'],
    }
    
    for col in video_feature_cols:
        feature_dict[f'current_{col}'] = video_medians[col]
    
    typical_congestion = location_hour_data['congestion_enter_rating'].mode()[0]
    
    for lag in [1, 2, 3, 5, 10]:
        feature_dict[f'enter_lag_{lag}'] = typical_congestion
        feature_dict[f'motion_lag_{lag}'] = video_medians['motion_mean']
        feature_dict[f'occupancy_lag_{lag}'] = video_medians['occupancy_mean']
    
    print(f"\nFeature dict created: {len(feature_dict)} features")
    print(f"  Typical congestion: {typical_congestion}")
    
    # Convert to DataFrame
    feature_row = pd.DataFrame([feature_dict])
    print(f"\nBefore encoding: {feature_row.shape}")
    print(f"  Columns: {feature_row.columns.tolist()[:5]}...")
    
    # Encode
    for col in feature_row.columns:
        if feature_row[col].dtype == 'object':
            le_temp = LabelEncoder()
            feature_row[col] = le_temp.fit_transform(feature_row[col].astype(str))
    
    print(f"\nAfter encoding: {feature_row.shape}")
    
    # Check feature alignment
    print(f"\nFeature alignment:")
    print(f"  Model expects: {len(feature_columns)} features")
    print(f"  We have: {len(feature_row.columns)} features")
    
    missing = set(feature_columns) - set(feature_row.columns)
    extra = set(feature_row.columns) - set(feature_columns)
    
    if missing:
        print(f"  Missing features: {len(missing)}")
        print(f"    Examples: {list(missing)[:5]}")
    
    if extra:
        print(f"  Extra features: {len(extra)}")
        print(f"    Examples: {list(extra)[:5]}")
    
    # Align features
    for col in feature_columns:
        if col not in feature_row.columns:
            feature_row[col] = 0
    
    feature_row = feature_row[feature_columns]
    
    print(f"\nAfter alignment: {feature_row.shape}")
    
    # Try prediction
    print("\n" + "="*60)
    print("TESTING PREDICTION")
    print("="*60)
    
    try:
        lgbm_pred = models['lgbm'].predict(feature_row)
        lgbm_proba = models['lgbm'].predict_proba(feature_row)
        
        print(f"\nLightGBM prediction:")
        print(f"  Encoded: {lgbm_pred[0]}")
        print(f"  Probabilities: {lgbm_proba[0]}")
        print(f"  Decoded: {le_target.inverse_transform(lgbm_pred)[0]}")
        
        xgb_pred = models['xgb'].predict(feature_row)
        xgb_proba = models['xgb'].predict_proba(feature_row)
        
        print(f"\nXGBoost prediction:")
        print(f"  Encoded: {xgb_pred[0]}")
        print(f"  Probabilities: {xgb_proba[0]}")
        print(f"  Decoded: {le_target.inverse_transform([int(xgb_pred[0])])[0]}")
        
        catboost_pred = models['catboost'].predict(feature_row)
        catboost_proba = models['catboost'].predict_proba(feature_row)
        
        print(f"\nCatBoost prediction:")
        print(f"  Encoded: {catboost_pred[0]}")
        print(f"  Probabilities: {catboost_proba[0]}")
        print(f"  Decoded: {le_target.inverse_transform([int(catboost_pred[0])])[0]}")
        
        # Ensemble
        weights = ensemble_data['weights']
        ensemble_proba = (
            weights['lgbm'] * lgbm_proba +
            weights['xgb'] * xgb_proba +
            weights['catboost'] * catboost_proba
        )
        
        ensemble_pred = np.argmax(ensemble_proba, axis=1)[0]
        
        print(f"\nEnsemble prediction:")
        print(f"  Encoded: {ensemble_pred}")
        print(f"  Probabilities: {ensemble_proba[0]}")
        print(f"  Decoded: {le_target.inverse_transform([ensemble_pred])[0]}")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
