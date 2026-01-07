"""
Google Cloud Storage Setup and Video Download
Based on Reading_Data_From_Google_Storage_Bucket.ipynb
"""
import os
import pandas as pd
from google.cloud import storage
from collections import defaultdict

print("="*60)
print("GOOGLE CLOUD STORAGE - VIDEO DOWNLOAD")
print("="*60)

# Configuration
PROJECT_ID = "brb-traffic"
BUCKET_NAME = "brb-traffic"
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

print(f"\nProject: {PROJECT_ID}")
print(f"Bucket: {BUCKET_NAME}")
print(f"Local directory: {VIDEO_DIR}")

# Initialize GCS client
print("\n" + "="*60)
print("INITIALIZING GOOGLE CLOUD STORAGE CLIENT")
print("="*60)

try:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    print("✓ Successfully connected to GCS")
except Exception as e:
    print(f"❌ Error connecting to GCS: {e}")
    print("\nPlease authenticate first:")
    print("  gcloud auth login")
    print("  gcloud auth application-default login")
    exit(1)

# Load training data
print("\n" + "="*60)
print("LOADING TRAINING DATA")
print("="*60)

train_df = pd.read_csv('Train.csv')
print(f"Training data: {len(train_df)} segments")

# Extract camera path (as in notebook)
def extract_camera_path(video):
    """Extract camera ID and reconstruct video path"""
    parts = video.split('/')
    filename = parts[-1]
    camera_id = filename.split('_')[0]
    return f"{camera_id}/{filename}"

train_df['video_path_gcs'] = train_df['videos'].apply(extract_camera_path)

print(f"\nSample video paths:")
for i, path in enumerate(train_df['video_path_gcs'].head(5)):
    print(f"  {i+1}. {path}")

# Analyze bucket contents
print("\n" + "="*60)
print("ANALYZING BUCKET CONTENTS")
print("="*60)

print("Fetching blob list...")
blobs = list(bucket.list_blobs())
print(f"Total blobs in bucket: {len(blobs)}")

# Track stats per folder
folder_counts = defaultdict(int)
folder_sizes = defaultdict(int)
largest_file = ("", 0)

for blob in blobs:
    folder = blob.name.split('/')[0] if '/' in blob.name else '(root)'
    
    folder_counts[folder] += 1
    folder_sizes[folder] += blob.size
    
    if blob.size > largest_file[1]:
        largest_file = (blob.name, blob.size)

print("\nFolder stats:")
for folder in sorted(folder_counts.keys()):
    size_mb = folder_sizes[folder] / (1024*1024)
    print(f"  {folder}: {folder_counts[folder]} files, {size_mb:.2f} MB")

largest_mb = largest_file[1] / (1024*1024)
print(f"\nLargest file: {largest_file[0]} ({largest_mb:.2f} MB)")

# Download sample videos
print("\n" + "="*60)
print("DOWNLOADING SAMPLE VIDEOS")
print("="*60)

# Download first 10 videos as test
sample_videos = train_df['video_path_gcs'].head(10).tolist()

downloaded = 0
for i, video_path in enumerate(sample_videos):
    local_filename = os.path.basename(video_path)
    local_path = os.path.join(VIDEO_DIR, local_filename)
    
    # Skip if already downloaded
    if os.path.exists(local_path):
        print(f"[{i+1}/{len(sample_videos)}] ✓ Already exists: {local_filename}")
        downloaded += 1
        continue
    
    try:
        blob = bucket.blob(video_path)
        print(f"[{i+1}/{len(sample_videos)}] Downloading: {local_filename}...", end=' ')
        blob.download_to_filename(local_path)
        
        size_mb = os.path.getsize(local_path) / (1024*1024)
        print(f"✓ ({size_mb:.2f} MB)")
        downloaded += 1
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n✓ Downloaded {downloaded}/{len(sample_videos)} videos")

# Test video feature extraction
print("\n" + "="*60)
print("TESTING VIDEO FEATURE EXTRACTION")
print("="*60)

import cv2
import numpy as np

def extract_quick_features(video_path):
    """Quick feature extraction test"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample a few frames
    motion_scores = []
    prev_gray = None
    
    for i in range(0, min(frame_count, 100), 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = np.mean(diff)
            motion_scores.append(motion)
        
        prev_gray = gray
    
    cap.release()
    
    return {
        'fps': fps,
        'frames': frame_count,
        'duration': frame_count / fps if fps > 0 else 0,
        'motion_mean': np.mean(motion_scores) if motion_scores else 0,
        'motion_std': np.std(motion_scores) if motion_scores else 0,
    }

# Test on first downloaded video
local_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]

if local_videos:
    test_video = os.path.join(VIDEO_DIR, local_videos[0])
    print(f"Testing on: {local_videos[0]}")
    
    features = extract_quick_features(test_video)
    
    if features:
        print("\n✓ Extracted features:")
        for key, value in features.items():
            print(f"  {key}: {value:.2f}")
    else:
        print("❌ Failed to extract features")
else:
    print("No videos available for testing")

print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)
print(f"✓ GCS connection established")
print(f"✓ Downloaded {downloaded} sample videos")
print(f"✓ Video feature extraction tested")
print("\nNext steps:")
print("  1. Download more videos (or all)")
print("  2. Extract features from all videos")
print("  3. Build enhanced forecasting model")
print("="*60)
