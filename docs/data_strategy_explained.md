# Data Strategy: Videos vs Metadata

## Understanding the Data Files

### What You Have Locally (CSV Files)

Your `Train.csv` and `TestInputSegments.csv` contain **metadata** about traffic, not the actual videos.

#### Train.csv Structure:
```csv
responseId,view_label,ID_enter,ID_exit,videos,video_time,datetimestamp_start,datetimestamp_end,date,signaling,congestion_enter_rating,congestion_exit_rating,time_segment_id,cycle_phase
```

**Key columns we use for BASELINE:**

| Column | What It Contains | How We Use It |
|--------|------------------|---------------|
| `video_time` | Timestamp (e.g., "2025-10-20 06:00:45") | Extract hour, minute, day |
| `view_label` | Camera name (e.g., "Norman Niles #1") | Camera ID feature |
| `signaling` | Turn signal usage (none/low/medium/high) | Categorical feature |
| `time_segment_id` | Segment number (0, 1, 2, ...) | Sequential feature |
| `congestion_enter_rating` | **TARGET** - Entrance congestion | What we predict |
| `congestion_exit_rating` | **TARGET** - Exit congestion | What we predict |
| `videos` | Video path (e.g., "normanniles1/...mp4") | **NOT USED in baseline** |

**The `videos` column is just a PATH/REFERENCE** - it tells you where the video is stored in Google Cloud Storage, but we don't download or use it for the baseline.

---

## Phase 1: Baseline Model (TODAY)

### Data Flow - NO Videos Needed:

```
Train.csv (Local File)
├── video_time: "2025-10-20 06:00:45"
├── view_label: "Norman Niles #1"
├── signaling: "none"
├── time_segment_id: 0
├── congestion_enter_rating: "free flowing"  ← TARGET
├── congestion_exit_rating: "free flowing"   ← TARGET
└── videos: "normanniles1/normanniles1_2025-10-20-06-00-45.mp4"  ← IGNORED
         ↓
    Extract Features
         ↓
    - hour = 6
    - minute = 0
    - is_morning_rush = 0
    - camera_id = 1
    - signaling_encoded = 0
         ↓
    Train LightGBM
         ↓
    Predict Congestion
```

**We're predicting congestion based on:**
- What time it is (rush hour patterns)
- Which camera (location patterns)
- Turn signal usage
- Historical patterns (lag features)

**We're NOT using:**
- Actual video content
- Vehicle counts
- Traffic flow from videos

---

## Phase 2: Enhanced Model (LATER) - Videos Needed

### When We Add Video Features:

```
Train.csv (Local File)                    Google Cloud Storage
├── video_time: "06:00:45"                ├── normanniles1_2025-10-20-06-00-45.mp4
├── videos: "normanniles1/..."  ──────────┤
├── congestion: "free flowing"            └── [Actual video file]
         ↓                                         ↓
    Temporal Features                      Download & Process Video
    - hour = 6                                     ↓
    - is_rush = 0                          Classical Computer Vision
                                                   ↓
                                           - Vehicle count = 15
                                           - Queue length = 3
                                           - Optical flow = 0.8
         ↓                                         ↓
         └─────────────┬───────────────────────────┘
                       ↓
              Combined Features
              (Temporal + Video)
                       ↓
              Train LightGBM
                       ↓
           Better Predictions
```

---

## Why This Two-Phase Approach?

### Phase 1 Advantages (Baseline):
✅ **Fast** - Train in 2-5 minutes
✅ **No video download** - Saves bandwidth and storage
✅ **No video processing** - Saves computation time
✅ **Gets you on leaderboard TODAY**
✅ **Establishes baseline performance**

### Phase 2 Advantages (Enhanced):
✅ **Better accuracy** - Video features capture actual traffic
✅ **More interpretable** - Can see what causes congestion
✅ **Competitive score** - Needed for top 20

---

## What Data Do You Need Right Now?

### For Training (Phase 1):
```bash
Train.csv  ← YES, you have this locally
```

That's it! Just this one CSV file.

### For Prediction (Phase 1):
```bash
TestInputSegments.csv  ← YES, you have this locally
```

That's it! Just this one CSV file.

### For Submission:
```bash
SampleSubmission.csv  ← YES, you have this (for format reference)
```

---

## Summary

**TODAY (Phase 1):**
- Use ONLY CSV files (Train.csv, TestInputSegments.csv)
- Extract features from timestamps, camera IDs, turn signals
- Train LightGBM model
- Generate submission
- **NO videos needed**

**LATER (Phase 2):**
- Access videos from Google Cloud Storage
- Extract visual features (vehicle counts, flow, queues)
- Combine with temporal features
- Retrain model
- Better predictions

**Right now, you can train and submit without touching any videos!**
