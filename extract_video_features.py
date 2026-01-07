"""
Video Feature Extraction Pipeline
Extracts traffic features from videos for forecasting model
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Google Cloud Storage imports (optional - for downloading videos)
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Google Cloud Storage not available - will use local videos only")

class VideoFeatureExtractor:
    """Extract traffic features from video files"""
    
    def __init__(self, video_dir='videos'):
        self.video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)
        
    def download_video_from_gcs(self, bucket_name, blob_path, local_path):
        """Download video from Google Cloud Storage"""
        if not GCS_AVAILABLE:
            print("GCS not available - skipping download")
            return False
            
        try:
            client = storage.Client(project="brb-traffic")
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(local_path)
            return True
        except Exception as e:
            print(f"Error downloading {blob_path}: {e}")
            return False
    
    def extract_features_from_video(self, video_path):
        """
        Extract traffic features from a single video
        
        Features:
        - vehicle_count: Estimated number of vehicles
        - motion_intensity: Amount of movement in frame
        - occupancy: % of frame with movement
        - flow_rate: Vehicles per second
        """
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Initialize feature collectors
        motion_scores = []
        occupancy_scores = []
        prev_gray = None
        
        # Sample frames (every 5th frame to speed up)
        frame_idx = 0
        sample_rate = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Calculate motion using frame difference
                    diff = cv2.absdiff(gray, prev_gray)
                    
                    # Threshold to get moving regions
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    
                    # Calculate motion intensity
                    motion_intensity = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])
                    motion_scores.append(motion_intensity)
                    
                    # Calculate occupancy (% of frame with motion)
                    occupancy = np.count_nonzero(thresh) / thresh.size
                    occupancy_scores.append(occupancy)
                
                prev_gray = gray.copy()
            
            frame_idx += 1
        
        cap.release()
        
        # Aggregate features
        features = {
            'duration': duration,
            'fps': fps,
            'frame_count': frame_count,
            'motion_mean': np.mean(motion_scores) if motion_scores else 0,
            'motion_std': np.std(motion_scores) if motion_scores else 0,
            'motion_max': np.max(motion_scores) if motion_scores else 0,
            'occupancy_mean': np.mean(occupancy_scores) if occupancy_scores else 0,
            'occupancy_std': np.std(occupancy_scores) if occupancy_scores else 0,
            'occupancy_max': np.max(occupancy_scores) if occupancy_scores else 0,
            # Proxy for vehicle count (higher motion = more vehicles)
            'vehicle_count_proxy': np.mean(motion_scores) * 100 if motion_scores else 0,
            # Flow rate estimate
            'flow_rate_proxy': np.mean(occupancy_scores) * 10 if occupancy_scores else 0,
        }
        
        return features
    
    def process_dataset(self, csv_path, bucket_name='brb-traffic', max_videos=None):
        """
        Process entire dataset and extract features
        
        Args:
            csv_path: Path to Train.csv or TestInputSegments.csv
            bucket_name: GCS bucket name
            max_videos: Limit number of videos to process (for testing)
        """
        df = pd.read_csv(csv_path)
        
        print(f"Processing {len(df)} video segments...")
        if max_videos:
            df = df.head(max_videos)
            print(f"Limited to {max_videos} for testing")
        
        features_list = []
        
        for idx, row in df.iterrows():
            video_rel_path = row['videos']
            video_filename = os.path.basename(video_rel_path)
            local_video_path = os.path.join(self.video_dir, video_filename)
            
            # Download if not exists
            if not os.path.exists(local_video_path):
                print(f"[{idx+1}/{len(df)}] Downloading {video_filename}...")
                success = self.download_video_from_gcs(
                    bucket_name, 
                    video_rel_path, 
                    local_video_path
                )
                if not success:
                    print(f"  Skipping - download failed")
                    continue
            
            # Extract features
            print(f"[{idx+1}/{len(df)}] Extracting features from {video_filename}...")
            video_features = self.extract_features_from_video(local_video_path)
            
            if video_features:
                # Combine with metadata
                combined_features = {
                    'time_segment_id': row['time_segment_id'],
                    'view_label': row['view_label'],
                    'datetimestamp_start': row['datetimestamp_start'],
                    'signaling': row['signaling'],
                    **video_features
                }
                
                # Add labels if available (training data)
                if 'congestion_enter_rating' in row:
                    combined_features['congestion_enter_rating'] = row['congestion_enter_rating']
                if 'congestion_exit_rating' in row:
                    combined_features['congestion_exit_rating'] = row['congestion_exit_rating']
                
                features_list.append(combined_features)
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        return features_df


def main():
    """Quick test of feature extraction"""
    print("="*60)
    print("VIDEO FEATURE EXTRACTION - TEST")
    print("="*60)
    
    extractor = VideoFeatureExtractor()
    
    # Test on first 5 training videos
    print("\nExtracting features from first 5 training videos...")
    features_df = extractor.process_dataset('Train.csv', max_videos=5)
    
    print(f"\nExtracted features for {len(features_df)} videos")
    print("\nFeature columns:")
    print(features_df.columns.tolist())
    
    print("\nSample features:")
    print(features_df[['time_segment_id', 'motion_mean', 'occupancy_mean', 
                       'vehicle_count_proxy', 'congestion_enter_rating']].head())
    
    # Save features
    output_path = 'video_features_sample.csv'
    features_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved features to {output_path}")
    
    return features_df


if __name__ == "__main__":
    main()
