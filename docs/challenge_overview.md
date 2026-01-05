# Barbados Traffic Analysis Challenge - Overview

## Challenge Objective
Predict traffic congestion levels at the Norman Niles roundabout in Barbados using video data from 4 camera views. The goal is to identify root causes of traffic congestion and predict congestion 5 minutes into the future.

## Problem Statement
- **Task**: Multi-class classification for traffic congestion prediction
- **Input**: Video data from 4 cameras at roundabout entrances/exits
- **Output**: Congestion ratings for both entrance and exit points
- **Prediction Window**: 5 minutes ahead with 2-minute embargo period

## Congestion Classes
1. **Free flowing** - No delays
2. **Light delay** - Minor congestion
3. **Moderate delay** - Moderate congestion
4. **Heavy delay** - Severe congestion

## Evaluation Metrics
The challenge uses **multi-metric evaluation**:
- **Macro-F1 Score (70%)**: Measures balanced performance across all 4 classes
- **Accuracy (30%)**: Overall percentage of correct predictions

**Final Score** = 0.7 × Macro-F1 + 0.3 × Accuracy

## Data Structure

### Training Data (`Train.csv`)
- **13,012 rows** with 14 columns
- Each row represents a ~1-minute video segment
- Key columns:
  - `videos`: Path to video file (e.g., `normanniles1/normanniles1_2025-10-20-06-00-45.mp4`)
  - `view_label`: Camera view (Norman Niles #1, #2, #3, #4)
  - `congestion_enter_rating`: Congestion at entrance
  - `congestion_exit_rating`: Congestion at exit
  - `time_segment_id`: Sequential time segment identifier
  - `datetimestamp_start/end`: Video timestamp
  - `signaling`: Turn signal usage (none, low, medium, high)
  - `cycle_phase`: Always "train" in training data

### Test Data (`TestInputSegments.csv`)
- Contains test video segments
- 15 minutes of input data per test period
- 2-minute embargo period (data processing lag)
- Predict 5 minutes of congestion ratings

### Submission Format (`SampleSubmission.csv`)
Must contain 3 columns:
- `ID`: Unique identifier (format: `time_segment_X_Norman Niles #Y_congestion_Z_rating`)
- `Target`: Prediction for F1 calculation
- `Target_Accuracy`: Prediction for Accuracy calculation

## Key Constraints

### Real-Time Inference
- **Sequential prediction**: Each minute must be predicted in order
- **No future data**: Cannot use data from minute N+1 to predict minute N
- **No backpropagation during inference**: Model weights cannot be updated during test time
- **Operational lag**: 2-minute embargo simulates real-world processing delay

### Data Pipeline
```
Training data → Test input (15 min) → 2-min embargo → 5-min prediction window
```

## Important Behavioral Factor
**Turn Signal Usage**: Anecdotal evidence suggests many Barbadian drivers don't use turn signals when entering/exiting roundabouts, which may contribute to congestion.

## Video Data Location
Videos are stored in Google Cloud Storage buckets:
- Camera 1: `normanniles1/`
- Camera 2: `normanniles2/`
- Camera 3: `normanniles3/`
- Camera 4: `normanniles4/`

## Deliverables for Top 20
Top 20 solutions must provide:
1. Working code (reproducible)
2. Documentation of top factors contributing to congestion
3. Table with: feature name, feature contribution, notes

## Prize
**$11,000 USD** for winning solutions

## Timeline
- **20 days remaining** (as of challenge access)
- Competition hosted on Zindi.africa
