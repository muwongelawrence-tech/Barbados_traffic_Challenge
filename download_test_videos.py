"""
Download ALL test videos from GCS
"""
import os
import pandas as pd
from google.cloud import storage

print("="*60)
print("DOWNLOADING TEST VIDEOS FROM GCS")
print("="*60)

# Configuration
PROJECT_ID = "brb-traffic"
BUCKET_NAME = "brb-traffic"
VIDEO_DIR = "videos"

os.makedirs(VIDEO_DIR, exist_ok=True)

# Load test data
test_df = pd.read_csv('TestInputSegments.csv')

print(f"\nTest segments: {len(test_df)}")
print(f"Unique videos needed: {test_df['videos'].nunique()}")

# Extract unique video paths
test_videos = test_df['videos'].unique()

print(f"\nTotal test videos to download: {len(test_videos)}")

# Connect to GCS
try:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    print("✓ Connected to GCS")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Download videos
print("\n" + "="*60)
print("DOWNLOADING VIDEOS")
print("="*60)

downloaded = 0
skipped = 0
failed = 0

for i, video_path in enumerate(test_videos):
    local_filename = os.path.basename(video_path)
    local_path = os.path.join(VIDEO_DIR, local_filename)
    
    # Skip if exists
    if os.path.exists(local_path):
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(test_videos)}] Skipped (exists): {local_filename}")
        skipped += 1
        continue
    
    try:
        blob = bucket.blob(video_path)
        blob.download_to_filename(local_path)
        
        size_mb = os.path.getsize(local_path) / (1024*1024)
        if (i + 1) % 100 == 0 or i < 10:
            print(f"[{i+1}/{len(test_videos)}] ✓ {local_filename} ({size_mb:.2f} MB)")
        downloaded += 1
    except Exception as e:
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(test_videos)}] ❌ {local_filename}: {str(e)[:50]}")
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
print(f"Total size: {total_size / (1024*1024*1024):.2f} GB")

print("\n✓ Ready for feature extraction!")
print("  Run: python extract_all_video_features.py")
