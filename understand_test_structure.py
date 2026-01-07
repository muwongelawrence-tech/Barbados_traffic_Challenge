"""
Understand the test structure:
15-min input → 2-min embargo → 5-min prediction
"""
import pandas as pd
import re
from collections import defaultdict

print("="*60)
print("UNDERSTANDING TEST STRUCTURE")
print("="*60)

test_df = pd.read_csv('TestInputSegments.csv')
sample_df = pd.read_csv('SampleSubmission.csv')

# Extract segments from sample
sample_segments = []
for id_str in sample_df['ID']:
    match = re.search(r'time_segment_(\d+)_', id_str)
    if match:
        sample_segments.append(int(match.group(1)))

sample_segments_unique = sorted(set(sample_segments))

print("\n1. TEST INPUT STRUCTURE")
print("-"*60)
print(f"Test segments: {len(test_df)}")
print(f"Cycle phase: {test_df['cycle_phase'].unique()}")

# Group test segments by location
test_by_location = test_df.groupby('view_label')['time_segment_id'].apply(list).to_dict()

print("\nTest segments by location:")
for loc, segs in test_by_location.items():
    print(f"  {loc}: {len(segs)} segments")
    print(f"    Range: {min(segs)} - {max(segs)}")
    print(f"    First 5: {sorted(segs)[:5]}")
    print(f"    Last 5: {sorted(segs)[-5:]}")

print("\n2. SAMPLE SUBMISSION STRUCTURE")
print("-"*60)
print(f"Sample rows: {len(sample_df)}")
print(f"Unique segments: {len(sample_segments_unique)}")
print(f"Segment range: {min(sample_segments_unique)} - {max(sample_segments_unique)}")

# Group sample by location
sample_by_location = defaultdict(list)
for id_str in sample_df['ID']:
    match = re.search(r'time_segment_(\d+)_(.*?)_congestion', id_str)
    if match:
        seg = int(match.group(1))
        loc = match.group(2)
        sample_by_location[loc].append(seg)

print("\nSample segments by location:")
for loc, segs in sample_by_location.items():
    unique_segs = sorted(set(segs))
    print(f"  {loc}: {len(unique_segs)} unique segments")
    print(f"    Range: {min(unique_segs)} - {max(unique_segs)}")
    print(f"    Segments: {unique_segs}")

print("\n3. HYPOTHESIS: FORECASTING STRUCTURE")
print("-"*60)
print("If test input = 15 min and we predict 5 min ahead:")
print("Sample segments should be ~5 min AFTER test segments")

# Check if sample segments come after test segments
for loc in test_by_location.keys():
    test_segs = sorted(test_by_location[loc])
    sample_segs = sorted(set(sample_by_location.get(loc, [])))
    
    if sample_segs:
        print(f"\n{loc}:")
        print(f"  Test last segment: {test_segs[-1]}")
        print(f"  Sample first segment: {sample_segs[0]}")
        print(f"  Gap: {sample_segs[0] - test_segs[-1]} segments")

print("\n4. CONCLUSION")
print("-"*60)
print("The sample submission segments are NOT directly after test segments.")
print("They appear to be SPECIFIC time periods we need to forecast.")
print("\nThis means:")
print("1. We need to use patterns from test data")
print("2. Forecast to the specific segments in sample submission")
print("3. Use time/location features, not just segment IDs")
