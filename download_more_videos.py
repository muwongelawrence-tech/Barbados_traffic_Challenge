"""
Download more videos from GCS for better model training
Downloads a specified number of videos across all cameras
"""
import os
import pandas as pd
from google.cloud import storage
from collections import defaultdict

print("="*60)
print("DOWNLOAD MORE VIDEOS FOR TRAINING")
print("="*60)

# Configuration
PROJECT_ID = "brb-traffic"
BUCKET_NAME = "brb-traffic"
VIDEO_DIR = "videos"
NUM_VIDEOS_TO_DOWNLOAD = 100  # Adjust this number

os.makedirs(VIDEO_DIR, exist_ok=True)

print(f"\nTarget: Download {NUM_VIDEOS_TO_DOWNLOAD} videos")
print(f"Directory: {VIDEO_DIR}")

# Connect to GCS
try:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    print("✓ Connected to GCS")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Load training data
train_df = pd.read_csv('Train.csv')

# Extract camera path
def extract_camera_path(video):
    parts = video.split('/')
    filename = parts[-1]
    camera_id = filename.split('_')[0]
    return f"{camera_id}/{filename}"

train_df['video_path_gcs'] = train_df['videos'].apply(extract_camera_path)

# Sample videos evenly across cameras and time
print("\n" + "="*60)
print("SAMPLING VIDEOS")
print("="*60)

# Group by camera
cameras = train_df['view_label'].unique()
videos_per_camera = NUM_VIDEOS_TO_DOWNLOAD // len(cameras)

print(f"Cameras: {len(cameras)}")
print(f"Videos per camera: {videos_per_camera}")

sampled_videos = []
for camera in cameras:
    camera_df = train_df[train_df['view_label'] == camera]
    # Sample evenly across time
    sample = camera_df.sample(n=min(videos_per_camera, len(camera_df)), random_state=42)
    sampled_videos.extend(sample['video_path_gcs'].tolist())

print(f"Total videos to download: {len(sampled_videos)}")

# Download videos
print("\n" + "="*60)
print("DOWNLOADING VIDEOS")
print("="*60)

downloaded = 0
skipped = 0
failed = 0

for i, video_path in enumerate(sampled_videos):
    local_filename = os.path.basename(video_path)
    local_path = os.path.join(VIDEO_DIR, local_filename)
    
    # Skip if exists
    if os.path.exists(local_path):
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(sampled_videos)}] Skipped (exists): {local_filename}")
        skipped += 1
        continue
    
    try:
        blob = bucket.blob(video_path)
        blob.download_to_filename(local_path)
        
        size_mb = os.path.getsize(local_path) / (1024*1024)
        if (i + 1) % 10 == 0 or i < 10:
            print(f"[{i+1}/{len(sampled_videos)}] ✓ {local_filename} ({size_mb:.2f} MB)")
        downloaded += 1
    except Exception as e:
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(sampled_videos)}] ❌ {local_filename}: {e}")
        failed += 1

print("\n" + "="*60)
print("DOWNLOAD SUMMARY")
print("="*60)
print(f"Downloaded: {downloaded}")
print(f"Skipped (exists): {skipped}")
print(f"Failed: {failed}")
print(f"Total in directory: {len([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])}")

# Calculate total size
total_size = sum(os.path.getsize(os.path.join(VIDEO_DIR, f)) 
                 for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4'))
print(f"Total size: {total_size / (1024*1024):.2f} MB")

print("\n✓ Ready for feature extraction!")
print("  Run: python extract_all_video_features.py")
