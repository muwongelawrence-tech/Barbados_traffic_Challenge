# Technical Approach & Model Selection

## Challenge Constraint Analysis

### Critical Constraint: No Backpropagation
> **"Back-propagation is not allowed in training or inference"**

This constraint fundamentally shapes our technical approach and eliminates entire categories of models.

### What This Constraint Eliminates:
- ❌ **Convolutional Neural Networks (CNNs)** - Require backprop for training
- ❌ **Recurrent Neural Networks (RNNs/LSTMs)** - Require backprop for training
- ❌ **Transformers** - Require backprop for training
- ❌ **Any deep learning models** - All require backpropagation
- ❌ **Transfer learning with fine-tuning** - Requires backprop
- ❌ **Pre-trained models as classifiers** - Even frozen CNNs violate the spirit of the constraint

### What This Constraint Allows:
- ✅ **Classical Computer Vision** - OpenCV, image processing algorithms
- ✅ **Gradient Boosting** - Tree-based models (XGBoost, LightGBM, CatBoost)
- ✅ **Random Forests** - Ensemble of decision trees
- ✅ **Support Vector Machines** - Kernel-based methods
- ✅ **Traditional ML algorithms** - No gradient descent involved

---

## Our Selected Approach: Classical CV + LightGBM

### Architecture Overview

```
┌─────────────────┐
│  Video Frames   │
│   (~1 minute)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Classical Computer Vision      │
│  (OpenCV - No Backprop)         │
│  • Background Subtraction       │
│  • Optical Flow Analysis        │
│  • Blob Detection               │
│  • Contour Analysis             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Feature Engineering            │
│  • Vehicle counts               │
│  • Flow metrics                 │
│  • Queue lengths                │
│  • Temporal patterns            │
│  • Camera metadata              │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  LightGBM Classifier            │
│  (No Backprop - Tree-based)     │
│  • Multi-class classification   │
│  • Dual output (enter/exit)     │
│  • Optimized for F1 + Accuracy  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Congestion Predictions         │
│  • Free flowing                 │
│  • Light delay                  │
│  • Moderate delay               │
│  • Heavy delay                  │
└─────────────────────────────────┘
```

---

## Why This Approach Achieves Challenge Goals

### 1. Satisfies "No Backpropagation" Constraint

**Classical Computer Vision:**
- Background subtraction (MOG2, KNN) - Statistical methods, no gradients
- Optical flow (Lucas-Kanade, Farneback) - Optimization-based, not backprop
- Blob detection - Threshold-based, no learning
- Contour analysis - Geometric algorithms, no gradients

**LightGBM:**
- Tree-based learning - Uses split finding, not gradient descent on weights
- No backpropagation through layers
- Gradient boosting uses gradients of loss w.r.t. predictions, NOT model weights
- Fully compliant with constraint

### 2. Identifies Root Causes of Traffic

**Feature Importance Analysis:**
LightGBM provides built-in feature importance metrics:
- **Gain**: Total reduction in loss from splits using this feature
- **Split**: Number of times feature is used for splitting
- **Cover**: Number of samples affected by splits

**Example Root Causes We Can Identify:**
```
Top Features Contributing to Congestion:
1. Vehicle entry rate (gain: 0.35) → High entry rate causes congestion
2. Turn signal usage (gain: 0.22) → Low signal usage correlates with delays
3. Hour of day (gain: 0.18) → Rush hour patterns
4. Queue length (gain: 0.15) → Backup at exits
5. Optical flow variance (gain: 0.10) → Erratic movement patterns
```

This directly addresses: *"identifying the root causes of increased time spent in the roundabout"*

### 3. Develops Features from Unstructured Video Data

**Classical CV Feature Extraction:**
We transform raw video (unstructured) into structured features:

| Video Data (Unstructured) | → | Extracted Features (Structured) |
|---------------------------|---|----------------------------------|
| Pixel values, motion      | → | Vehicle count: 15 vehicles/min   |
| Visual patterns           | → | Average speed: 25 km/h           |
| Spatial information       | → | Queue length: 8 vehicles         |
| Temporal sequences        | → | Flow rate: 12 vehicles/min       |
| Movement patterns         | → | Congestion score: 0.73           |

This addresses: *"developing features from unstructured video data"*

### 4. Predicts Traffic Congestion 5 Minutes Ahead

**Real-Time Prediction Pipeline:**
- Process 15 minutes of input video
- Extract features for each minute
- Apply 2-minute embargo (no predictions)
- Predict next 5 minutes sequentially
- No future data leakage (enforced by design)

**LightGBM Advantages for Real-Time:**
- Inference time: ~1-5ms per prediction
- Can easily predict minute-by-minute
- No GPU required for inference
- Lightweight model deployment

### 5. Optimizes for Challenge Metrics

**Multi-Metric Optimization:**
Final Score = 0.7 × Macro-F1 + 0.3 × Accuracy

**LightGBM Custom Objective:**
```python
def custom_objective(y_true, y_pred):
    # Calculate both F1 and Accuracy
    f1_score = macro_f1(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Weighted combination
    combined_score = 0.7 * f1_score + 0.3 * accuracy
    
    # Return gradient and hessian for LightGBM
    return grad, hess
```

**Why LightGBM Excels:**
- Handles class imbalance well (important for F1)
- Supports custom loss functions
- Fast hyperparameter tuning
- Built-in cross-validation

---

## Technical Implementation Details

### Classical Computer Vision Techniques

#### 1. Vehicle Detection & Counting
```python
# Background Subtraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

# Apply to each frame
fg_mask = bg_subtractor.apply(frame)

# Find contours (vehicles)
contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter by size to get actual vehicles
vehicles = [c for c in contours if cv2.contourArea(c) > MIN_VEHICLE_SIZE]
vehicle_count = len(vehicles)
```

#### 2. Optical Flow Analysis
```python
# Dense Optical Flow (Farneback method)
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray,
    None, 0.5, 3, 15, 3, 5, 1.2, 0
)

# Extract flow metrics
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

features = {
    'avg_flow_magnitude': np.mean(magnitude),
    'max_flow_magnitude': np.max(magnitude),
    'flow_variance': np.var(magnitude),
    'dominant_direction': np.median(angle)
}
```

#### 3. Queue Length Estimation
```python
# Detect stationary vehicles
stationary_mask = magnitude < STATIONARY_THRESHOLD

# Count stationary regions
stationary_contours = cv2.findContours(stationary_mask, ...)
queue_length = len(stationary_contours)

# Measure queue spatial extent
queue_area = sum(cv2.contourArea(c) for c in stationary_contours)
```

### LightGBM Configuration

```python
params = {
    'objective': 'multiclass',
    'num_class': 4,  # 4 congestion levels
    'metric': ['multi_logloss', 'multi_error'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_threads': 22  # Use all CPU cores
}

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    early_stopping_rounds=50
)
```

---

## Advantages Over Deep Learning Approaches

| Aspect | Classical CV + LightGBM | Deep Learning (CNN/LSTM) |
|--------|-------------------------|---------------------------|
| **Backpropagation** | ✅ Not required | ❌ Required (violates constraint) |
| **Training Time** | ✅ Minutes | ❌ Hours/Days |
| **Inference Speed** | ✅ 1-5ms | ❌ 50-200ms |
| **Interpretability** | ✅ Feature importance clear | ❌ Black box |
| **Data Requirements** | ✅ Works with 13K samples | ❌ Needs 100K+ samples |
| **GPU Requirement** | ✅ Optional (only for CV) | ❌ Essential |
| **Root Cause Analysis** | ✅ Direct feature attribution | ❌ Difficult to interpret |
| **Deployment** | ✅ Lightweight | ❌ Heavy (model size) |

---

## Expected Performance

### Baseline Targets:
- **Temporal features only**: F1 ~0.50, Accuracy ~0.60
- **With classical CV features**: F1 ~0.70, Accuracy ~0.75
- **Optimized ensemble**: F1 ~0.75+, Accuracy ~0.80+

### Feature Extraction Performance:
- **Processing time**: ~2-5 seconds per 1-minute video
- **Feature dimension**: ~50-100 features per segment
- **Total training time**: ~30-60 minutes for full pipeline

---

## Alignment with Challenge Goals

| Challenge Goal | Our Approach |
|----------------|--------------|
| Predict traffic congestion | ✅ LightGBM multi-class classifier |
| Identify root causes | ✅ Feature importance analysis |
| Develop features from video | ✅ Classical CV feature extraction |
| 5-minute ahead prediction | ✅ Sequential prediction pipeline |
| No backpropagation | ✅ Tree-based models only |
| Real-time constraints | ✅ Fast inference (<5ms) |
| Interpretable results | ✅ Feature importance + SHAP values |

---

## Deliverables for Top 20 Requirement

As required for top 20 solutions, we will provide:

### 1. Feature Importance Table
```
| Feature Name              | Contribution | Notes                                    |
|---------------------------|--------------|------------------------------------------|
| vehicle_entry_rate        | 0.35         | High entry rate → congestion            |
| turn_signal_compliance    | 0.22         | Low compliance → erratic behavior       |
| hour_of_day              | 0.18         | Rush hour patterns (7-9am, 4-6pm)       |
| queue_length_exit        | 0.15         | Backup at exits causes roundabout fill  |
| optical_flow_variance    | 0.10         | High variance → unpredictable movement  |
```

### 2. Root Cause Analysis
- Quantitative impact of each factor
- Temporal patterns (when congestion occurs)
- Spatial patterns (which cameras/entrances)
- Behavioral factors (turn signal usage)

### 3. Reproducible Code
- Complete feature extraction pipeline
- Model training scripts
- Inference pipeline
- All documented and tested

---

## Conclusion

This classical computer vision + LightGBM approach:
1. ✅ **Fully complies** with the "no backpropagation" constraint
2. ✅ **Directly addresses** root cause identification through feature importance
3. ✅ **Transforms** unstructured video into structured, interpretable features
4. ✅ **Achieves** real-time prediction requirements
5. ✅ **Optimizes** for the challenge's dual-metric evaluation
6. ✅ **Provides** clear, actionable insights for the Ministry of Transport

This is the correct technical approach for this challenge.
