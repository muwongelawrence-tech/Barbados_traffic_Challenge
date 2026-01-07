"""
Check if video features are being used in predictions
"""
import pandas as pd

print("="*60)
print("VIDEO FEATURE INTEGRATION CHECK")
print("="*60)

# Load data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('TestInputSegments.csv')
video_features_df = pd.read_csv('video_features.csv')

print("\n1. VIDEO FEATURES AVAILABLE")
print("-"*60)
print(f"Videos processed: {len(video_features_df)}")
print(f"Video filenames sample:")
for i, filename in enumerate(video_features_df['video_filename'].head(5)):
    print(f"  {i+1}. {filename}")

print("\n2. TEST DATA VIDEOS")
print("-"*60)
print(f"Test segments: {len(test_df)}")
print(f"Test video paths sample:")
for i, video_path in enumerate(test_df['videos'].head(5)):
    print(f"  {i+1}. {video_path}")

# Check matching
print("\n3. MATCHING CHECK")
print("-"*60)

# Extract filenames from test data
test_df['video_filename'] = test_df['videos'].apply(lambda x: x.split('/')[-1])

# Check overlap
video_files_set = set(video_features_df['video_filename'])
test_files_set = set(test_df['video_filename'])

overlap = video_files_set.intersection(test_files_set)

print(f"Video features available: {len(video_files_set)}")
print(f"Test videos needed: {len(test_files_set)}")
print(f"Overlap: {len(overlap)} ({len(overlap)/len(test_files_set)*100:.1f}%)")

if len(overlap) > 0:
    print(f"\n✓ We have features for {len(overlap)} test videos!")
    print("\nSample matched videos:")
    for i, filename in enumerate(list(overlap)[:5]):
        print(f"  {i+1}. {filename}")
else:
    print("\n❌ NO OVERLAP - Video features not matching test data")

print("\n4. CURRENT STATUS")
print("-"*60)
if len(overlap) > 0:
    print("✓ Video features CAN be integrated")
    print(f"  Coverage: {len(overlap)}/{len(test_files_set)} test videos")
    print("\n⚠️  BUT: Current submission uses MEDIAN values as placeholders")
    print("   Need to update code to use actual video features")
else:
    print("❌ Video features NOT integrated")
    print("   Reason: No matching filenames between videos and test data")
    print("\n   Options:")
    print("   1. Download test videos from GCS")
    print("   2. Extract features from test videos")
    print("   3. Use median values (current approach)")

print("\n5. RECOMMENDATION")
print("-"*60)
if len(overlap) > 0:
    print("Update submission code to use actual video features where available")
    print("This could improve predictions significantly!")
else:
    print("Download and process test videos to get real video features")
    print("Current submission uses model structure but not actual video data")
