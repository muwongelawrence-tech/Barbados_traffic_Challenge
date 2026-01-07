"""
Extract features from downloaded videos
Process all videos in the videos/ directory
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

print("="*60)
print("VIDEO FEATURE EXTRACTION")
print("="*60)

VIDEO_DIR = "videos"
OUTPUT_FILE = "video_features.csv"

# Get all videos
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"\nFound {len(video_files)} videos to process")

def extract_features_from_video(video_path):
    """Extract comprehensive traffic features from video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Feature collectors
    motion_scores = []
    occupancy_scores = []
    brightness_scores = []
    prev_gray = None
    
    # Sample every 5th frame for efficiency
    sample_rate = 5
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness (average pixel intensity)
            brightness = np.mean(gray)
            brightness_scores.append(brightness)
            
            if prev_gray is not None:
                # Motion detection
                diff = cv2.absdiff(gray, prev_gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                
                # Motion intensity
                motion = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])
                motion_scores.append(motion)
                
                # Occupancy (percentage of frame with motion)
                occupancy = np.count_nonzero(thresh) / thresh.size
                occupancy_scores.append(occupancy)
            
            prev_gray = gray.copy()
        
        frame_idx += 1
    
    cap.release()
    
    # Aggregate features
    features = {
        # Video metadata
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'resolution': f'{width}x{height}',
        
        # Motion features (proxy for traffic flow)
        'motion_mean': np.mean(motion_scores) if motion_scores else 0,
        'motion_std': np.std(motion_scores) if motion_scores else 0,
        'motion_max': np.max(motion_scores) if motion_scores else 0,
        'motion_min': np.min(motion_scores) if motion_scores else 0,
        'motion_median': np.median(motion_scores) if motion_scores else 0,
        
        # Occupancy features (proxy for congestion)
        'occupancy_mean': np.mean(occupancy_scores) if occupancy_scores else 0,
        'occupancy_std': np.std(occupancy_scores) if occupancy_scores else 0,
        'occupancy_max': np.max(occupancy_scores) if occupancy_scores else 0,
        
        # Brightness features (time of day, weather)
        'brightness_mean': np.mean(brightness_scores) if brightness_scores else 0,
        'brightness_std': np.std(brightness_scores) if brightness_scores else 0,
        
        # Derived features
        'vehicle_count_proxy': np.mean(motion_scores) * 100 if motion_scores else 0,
        'flow_rate_proxy': np.mean(occupancy_scores) * 10 if occupancy_scores else 0,
        'congestion_proxy': (np.mean(occupancy_scores) / (np.mean(motion_scores) + 0.001)) if motion_scores and occupancy_scores else 0,
    }
    
    return features

# Process all videos
print("\nProcessing videos...")
features_list = []

for i, video_file in enumerate(video_files):
    video_path = os.path.join(VIDEO_DIR, video_file)
    print(f"[{i+1}/{len(video_files)}] {video_file}...", end=' ')
    
    features = extract_features_from_video(video_path)
    
    if features:
        # Add filename
        features['video_filename'] = video_file
        
        # Extract timestamp from filename
        # Format: normanniles1_2025-10-20-06-00-45.mp4
        parts = video_file.replace('.mp4', '').split('_')
        if len(parts) >= 4:
            features['camera'] = parts[0]
            features['date'] = parts[1]
            features['time'] = f"{parts[2]}:{parts[3]}:{parts[4]}"
        
        features_list.append(features)
        print(f"✓ (motion: {features['motion_mean']:.3f}, occupancy: {features['occupancy_mean']:.3f})")
    else:
        print("❌ Failed")

# Create DataFrame
features_df = pd.DataFrame(features_list)

print(f"\n✓ Processed {len(features_df)} videos successfully")

# Save features
features_df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved features to {OUTPUT_FILE}")

# Display summary statistics
print("\n" + "="*60)
print("FEATURE STATISTICS")
print("="*60)

print("\nMotion features:")
print(f"  Mean: {features_df['motion_mean'].mean():.4f} ± {features_df['motion_mean'].std():.4f}")
print(f"  Range: [{features_df['motion_mean'].min():.4f}, {features_df['motion_mean'].max():.4f}]")

print("\nOccupancy features:")
print(f"  Mean: {features_df['occupancy_mean'].mean():.4f} ± {features_df['occupancy_mean'].std():.4f}")
print(f"  Range: [{features_df['occupancy_mean'].min():.4f}, {features_df['occupancy_mean'].max():.4f}]")

print("\nVehicle count proxy:")
print(f"  Mean: {features_df['vehicle_count_proxy'].mean():.2f} ± {features_df['vehicle_count_proxy'].std():.2f}")

print("\nFlow rate proxy:")
print(f"  Mean: {features_df['flow_rate_proxy'].mean():.2f} ± {features_df['flow_rate_proxy'].std():.2f}")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("1. Match video features to training data")
print("2. Combine with CSV features")
print("3. Retrain forecasting model")
print("4. Generate improved submission")
print("="*60)
