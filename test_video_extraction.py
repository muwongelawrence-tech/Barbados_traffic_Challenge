"""
Quick test of video feature extraction
Tests on first few videos without GCS (using local if available)
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd

print("="*60)
print("VIDEO FEATURE EXTRACTION - QUICK TEST")
print("="*60)

# Check OpenCV installation
print(f"\nOpenCV version: {cv2.__version__}")

# Check if we have any local videos
video_dir = 'videos'
if os.path.exists(video_dir):
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"Found {len(videos)} local videos")
else:
    print(f"No videos directory found")
    videos = []

# Simple feature extraction test
def extract_simple_features(video_path):
    """Extract basic features from video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo properties:")
    print(f"  FPS: {fps}")
    print(f"  Frames: {frame_count}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {frame_count/fps:.1f}s")
    
    # Sample frames for motion detection
    motion_scores = []
    prev_gray = None
    frame_idx = 0
    
    while frame_idx < min(frame_count, 300):  # Limit to first 300 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 5 == 0:  # Sample every 5th frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Calculate motion
                diff = cv2.absdiff(gray, prev_gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])
                motion_scores.append(motion)
            
            prev_gray = gray.copy()
        
        frame_idx += 1
    
    cap.release()
    
    features = {
        'fps': fps,
        'frame_count': frame_count,
        'duration': frame_count/fps if fps > 0 else 0,
        'motion_mean': np.mean(motion_scores) if motion_scores else 0,
        'motion_std': np.std(motion_scores) if motion_scores else 0,
        'motion_max': np.max(motion_scores) if motion_scores else 0,
    }
    
    return features

# Test on local videos if available
if videos:
    print(f"\n{'='*60}")
    print("Testing on local videos...")
    print('='*60)
    
    for i, video_file in enumerate(videos[:3]):  # Test first 3
        video_path = os.path.join(video_dir, video_file)
        print(f"\n[{i+1}] Processing: {video_file}")
        
        features = extract_simple_features(video_path)
        
        if features:
            print(f"  ✓ Extracted features:")
            for key, value in features.items():
                print(f"    {key}: {value:.4f}")
else:
    print("\n⚠️  No local videos found")
    print("   Need to download from Google Cloud Storage")
    print("\nTo proceed:")
    print("1. Set up GCS authentication")
    print("2. Download sample videos")
    print("3. Run feature extraction")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
