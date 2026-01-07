# Google Cloud Storage Access - Quick Start

## Prerequisites
- Google Cloud SDK installed (`gcloud` command)
- Access to `brb-traffic` project

## Authentication Steps

### 1. Run Authentication Script
```bash
./authenticate_gcs.sh
```

This will:
- Authenticate your identity for gcloud CLI
- Set up application-default credentials for API access
- Configure the project to `brb-traffic`

### 2. Download Videos
```bash
source barbados/bin/activate
python setup_gcs_and_download.py
```

This will:
- Connect to GCS bucket `brb-traffic`
- Analyze bucket contents
- Download sample videos (first 10)
- Test video feature extraction

## Manual Authentication (if script fails)

```bash
# Step 1: Authenticate for CLI
gcloud auth login

# Step 2: Authenticate for API
gcloud auth application-default login

# Step 3: Set project
gcloud config set project brb-traffic
```

## Bucket Structure

Based on the notebook, videos are organized as:
```
brb-traffic/
├── normanniles1/
│   ├── normanniles1_2025-10-20-06-00-45.mp4
│   ├── normanniles1_2025-10-20-06-01-45.mp4
│   └── ...
├── normanniles2/
├── normanniles3/
└── normanniles4/
```

## Download All Videos

To download all training videos:
```python
from setup_gcs_and_download import download_all_videos
download_all_videos()  # Downloads all ~16K videos
```

## Troubleshooting

### "Permission denied" error
- Make sure you're authenticated: `gcloud auth list`
- Check project access: `gcloud projects list`

### "Quota exceeded" error
- Add quota project: `gcloud auth application-default set-quota-project brb-traffic`

### Videos not downloading
- Check bucket access: `gsutil ls gs://brb-traffic/`
- Verify blob paths in Train.csv match bucket structure

## Next Steps

After downloading videos:
1. Run `extract_video_features.py` to process all videos
2. Build feature dataset
3. Retrain forecasting model with video features
4. Generate improved submission
