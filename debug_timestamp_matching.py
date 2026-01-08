"""
Debug timestamp matching between video features and training data
"""
import pandas as pd
import re
from datetime import datetime

print("="*60)
print("TIMESTAMP MATCHING DEBUG")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
video_features_df = pd.read_csv('video_features.csv')

print(f"\nTrain data: {len(train_df)} rows")
print(f"Video features: {len(video_features_df)} rows")

# Check timestamp formats
print("\n" + "="*60)
print("TIMESTAMP FORMATS")
print("="*60)

print("\nTrain CSV timestamps (first 5):")
for i, ts in enumerate(train_df['datetimestamp_start'].head()):
    print(f"  {i+1}. {ts}")

print("\nVideo filenames (first 5):")
for i, filename in enumerate(video_features_df['video_filename'].head()):
    print(f"  {i+1}. {filename}")

# Extract timestamp from video filename
def extract_timestamp_from_filename(filename):
    """Extract timestamp from video filename"""
    # Format: normanniles1_2025-10-20-06-00-45.mp4
    parts = filename.replace('.mp4', '').split('_')
    if len(parts) >= 2:
        # parts[1] should be like: 2025-10-20-06-00-45
        date_time = parts[1]
        # Split by dash
        components = date_time.split('-')
        if len(components) == 6:
            # components: ['2025', '10', '20', '06', '00', '45']
            date = f"{components[0]}-{components[1]}-{components[2]}"
            time = f"{components[3]}:{components[4]}:{components[5]}"
            return f"{date} {time}"
    return None

print("\n" + "="*60)
print("EXTRACTED TIMESTAMPS FROM VIDEOS")
print("="*60)

video_features_df['extracted_timestamp'] = video_features_df['video_filename'].apply(extract_timestamp_from_filename)

print("\nExtracted timestamps (first 5):")
for i, ts in enumerate(video_features_df['extracted_timestamp'].head()):
    print(f"  {i+1}. {ts}")

# Try matching
print("\n" + "="*60)
print("MATCHING ATTEMPT")
print("="*60)

# Create a simplified timestamp for matching (ignore seconds)
def simplify_timestamp(ts_str):
    """Simplify timestamp to YYYY-MM-DD HH:MM for matching"""
    try:
        # Parse and format to ignore seconds
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
video_features_df['timestamp_simple'] = video_features_df['extracted_timestamp'].apply(simplify_timestamp)

print(f"\nTrain timestamps (simplified, first 5):")
for i, ts in enumerate(train_df['timestamp_simple'].head()):
    print(f"  {i+1}. {ts}")

print(f"\nVideo timestamps (simplified, first 5):")
for i, ts in enumerate(video_features_df['timestamp_simple'].head()):
    print(f"  {i+1}. {ts}")

# Check overlap
train_timestamps = set(train_df['timestamp_simple'].dropna())
video_timestamps = set(video_features_df['timestamp_simple'].dropna())

overlap = train_timestamps.intersection(video_timestamps)

print(f"\n" + "="*60)
print("MATCHING RESULTS")
print("="*60)
print(f"Train unique timestamps: {len(train_timestamps)}")
print(f"Video unique timestamps: {len(video_timestamps)}")
print(f"Overlap: {len(overlap)} ({len(overlap)/len(video_timestamps)*100:.1f}%)")

if len(overlap) > 0:
    print(f"\n✓ SUCCESS! Found {len(overlap)} matching timestamps")
    print("\nSample matches:")
    for i, ts in enumerate(list(overlap)[:5]):
        print(f"  {i+1}. {ts}")
else:
    print("\n❌ NO MATCHES FOUND")
    print("\nDEBUGGING:")
    print(f"Sample train timestamp: {list(train_timestamps)[0]}")
    print(f"Sample video timestamp: {list(video_timestamps)[0]}")

# Also check by location
print("\n" + "="*60)
print("LOCATION MATCHING")
print("="*60)

# Extract location from video filename
def extract_location_from_filename(filename):
    """Extract camera location from filename"""
    # Format: normanniles1_2025-10-20-06-00-45.mp4
    parts = filename.split('_')[0]
    # Map to CSV format
    location_map = {
        'normanniles1': 'Norman Niles #1',
        'normanniles2': 'Norman Niles #2',
        'normanniles3': 'Norman Niles #3',
        'normanniles4': 'Norman Niles #4'
    }
    return location_map.get(parts, parts)

video_features_df['location'] = video_features_df['video_filename'].apply(extract_location_from_filename)

print("\nLocations in train data:")
print(train_df['view_label'].unique())

print("\nLocations in video data:")
print(video_features_df['location'].unique())

# Try matching with location + timestamp
video_features_df['match_key'] = video_features_df['location'] + '|' + video_features_df['timestamp_simple'].fillna('')
train_df['match_key'] = train_df['view_label'] + '|' + train_df['timestamp_simple'].fillna('')

train_keys = set(train_df['match_key'])
video_keys = set(video_features_df['match_key'])

overlap_keys = train_keys.intersection(video_keys)

print(f"\n" + "="*60)
print("LOCATION + TIMESTAMP MATCHING")
print("="*60)
print(f"Overlap with location+timestamp: {len(overlap_keys)} ({len(overlap_keys)/len(video_keys)*100:.1f}%)")

if len(overlap_keys) > 0:
    print(f"\n✓ SUCCESS! Found {len(overlap_keys)} matches")
    print("\nSample matches:")
    for i, key in enumerate(list(overlap_keys)[:5]):
        print(f"  {i+1}. {key}")
