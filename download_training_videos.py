"""
Phase 1: Download Training-Matching Videos
Identify which training timestamps have videos in GCS and download them
"""
import os
import pandas as pd
from google.cloud import storage
from datetime import datetime

print("="*60)
print("PHASE 1: DOWNLOADING TRAINING-MATCHING VIDEOS")
print("="*60)

# Configuration
PROJECT_ID = "brb-traffic"
BUCKET_NAME = "brb-traffic"
VIDEO_DIR = "videos"

os.makedirs(VIDEO_DIR, exist_ok=True)

# Load training data
train_df = pd.read_csv('Train.csv')

print(f"\nTraining data: {len(train_df)} segments")
print(f"Unique timestamps: {train_df['datetimestamp_start'].nunique()}")

# Extract unique training timestamps
def simplify_timestamp(ts_str):
    """Simplify to YYYY-MM-DD HH:MM"""
    try:
        parts = str(ts_str).split()
        if len(parts) >= 2:
            date = parts[0]
            time_parts = parts[1].split(':')
            if len(time_parts) >= 2:
                return f"{date} {time_parts[0]}:{time_parts[1]}"
    except:
        pass
    return None

train_df['timestamp_simple'] = train_df['datetimestamp_start'].apply(simplify_timestamp)

# Get unique location-timestamp combinations
train_df['match_key'] = train_df['view_label'] + '|' + train_df['timestamp_simple'].fillna('')
unique_keys = train_df['match_key'].unique()

print(f"Unique location-timestamp combinations: {len(unique_keys)}")

# Create video filename mapping
def create_video_filename(location, timestamp):
    """Create expected video filename from location and timestamp"""
    # Map location to camera name
    location_map = {
        'Norman Niles #1': 'normanniles1',
        'Norman Niles #2': 'normanniles2',
        'Norman Niles #3': 'normanniles3',
        'Norman Niles #4': 'normanniles4'
    }
    
    camera = location_map.get(location, '')
    if not camera or not timestamp:
        return None
    
    # Parse timestamp: "2025-10-20 06:00"
    try:
        parts = timestamp.split()
        date = parts[0]  # 2025-10-20
        time = parts[1]  # 06:00
        
        # We need to guess the seconds (videos are at :45 seconds typically)
        # Try common second values
        for seconds in ['45', '44', '46', '43', '47']:
            filename = f"{camera}_{date.replace('-', '-')}-{time.replace(':', '-')}-{seconds}.mp4"
            yield filename
    except:
        pass

# Build list of videos to download
print("\n" + "="*60)
print("BUILDING VIDEO LIST")
print("="*60)

videos_to_download = {}

for key in unique_keys:
    if '|' not in key:
        continue
    
    location, timestamp = key.split('|')
    
    for filename in create_video_filename(location, timestamp):
        if filename:
            # Determine path in GCS
            camera = filename.split('_')[0]
            video_path = f"{camera}/{filename}"
            videos_to_download[video_path] = key

print(f"Potential videos to check: {len(videos_to_download)}")

# Connect to GCS
print("\n" + "="*60)
print("CONNECTING TO GCS")
print("="*60)

try:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    print("✓ Connected to GCS")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Check which videos exist and download
print("\n" + "="*60)
print("DOWNLOADING TRAINING VIDEOS")
print("="*60)

downloaded = 0
skipped = 0
not_found = 0
failed = 0

for i, (video_path, match_key) in enumerate(videos_to_download.items()):
    local_filename = os.path.basename(video_path)
    local_path = os.path.join(VIDEO_DIR, local_filename)
    
    # Skip if exists
    if os.path.exists(local_path):
        skipped += 1
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(videos_to_download)}] Skipped (exists): {local_filename}")
        continue
    
    try:
        blob = bucket.blob(video_path)
        
        # Check if exists
        if not blob.exists():
            not_found += 1
            continue
        
        # Download
        blob.download_to_filename(local_path)
        
        size_mb = os.path.getsize(local_path) / (1024*1024)
        if (downloaded + 1) % 50 == 0 or downloaded < 10:
            print(f"[{i+1}/{len(videos_to_download)}] ✓ {local_filename} ({size_mb:.2f} MB)")
        downloaded += 1
        
    except Exception as e:
        if "404" not in str(e):
            if (failed + 1) % 100 == 0:
                print(f"[{i+1}/{len(videos_to_download)}] ❌ {local_filename}: {str(e)[:50]}")
        failed += 1

print("\n" + "="*60)
print("DOWNLOAD SUMMARY")
print("="*60)
print(f"Downloaded: {downloaded}")
print(f"Skipped (exists): {skipped}")
print(f"Not found in GCS: {not_found}")
print(f"Failed: {failed}")

# Count total videos
total_videos = len([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
print(f"\nTotal videos in directory: {total_videos}")

# Calculate total size
total_size = sum(os.path.getsize(os.path.join(VIDEO_DIR, f)) 
                 for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4'))
print(f"Total size: {total_size / (1024*1024*1024):.2f} GB")

# Estimate coverage
estimated_coverage = (downloaded + skipped) / len(unique_keys) * 100
print(f"\nEstimated training coverage: {estimated_coverage:.1f}%")

if downloaded > 0:
    print("\n✓ Ready to extract features from new videos!")
    print("  Run: python extract_all_video_features.py")
else:
    print("\n⚠️  No new videos downloaded")
    print("  Most training videos may not be available in GCS")
