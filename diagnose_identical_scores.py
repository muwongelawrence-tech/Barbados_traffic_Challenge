"""
Critical Analysis: Why are v1.1 and v1.2 getting the SAME score?
"""
import pandas as pd
import numpy as np

print("="*60)
print("CRITICAL ANALYSIS: IDENTICAL SCORES")
print("="*60)
print("v1.1 (SMOTE): 0.302442736")
print("v1.2 (Conservative): 0.302442736")
print("v1.0 (All free flowing): 0.461234377")
print("="*60)

# Load all submissions
v1_0 = pd.read_csv('submissions/v1.0_baseline_lightgbm.csv')
v1_1 = pd.read_csv('submissions/v1.1_smote_balanced.csv')
v1_2 = pd.read_csv('submissions/v1.2_conservative.csv')

print("\n1. SUBMISSION COMPARISON")
print("-"*60)

print("\nv1.0 predictions:")
print(v1_0['Target'].value_counts())

print("\nv1.1 predictions:")
print(v1_1['Target'].value_counts())

print("\nv1.2 predictions:")
print(v1_2['Target'].value_counts())

# Check if v1.1 and v1.2 are identical
identical = (v1_1['Target'] == v1_2['Target']).all()
print(f"\nv1.1 and v1.2 identical: {identical}")

if identical:
    print("\n⚠️  CRITICAL FINDING: v1.1 and v1.2 are IDENTICAL!")
    print("   This explains why they have the same score.")
    print("   The submission file is being created incorrectly!")

# Check the pattern
print("\n2. PATTERN ANALYSIS")
print("-"*60)

print("\nFirst 30 predictions:")
print("Index | v1.0 | v1.1 | v1.2")
print("-"*60)
for i in range(30):
    print(f"{i:3d}   | {v1_0.iloc[i]['Target'][:4]:4s} | {v1_1.iloc[i]['Target'][:4]:4s} | {v1_2.iloc[i]['Target'][:4]:4s}")

# Check if there's an alternating pattern
print("\n3. ALTERNATING PATTERN CHECK")
print("-"*60)

def check_alternating(df):
    """Check if predictions alternate between two classes"""
    values = df['Target'].values
    if len(set(values)) != 2:
        return False, None, None
    
    class1, class2 = list(set(values))
    
    # Check if it alternates
    for i in range(len(values)-1):
        if i % 2 == 0:
            if values[i] != class1:
                return False, None, None
        else:
            if values[i] != class2:
                return False, None, None
    
    return True, class1, class2

alt_v1_0, c1_v1_0, c2_v1_0 = check_alternating(v1_0)
alt_v1_1, c1_v1_1, c2_v1_1 = check_alternating(v1_1)
alt_v1_2, c1_v1_2, c2_v1_2 = check_alternating(v1_2)

print(f"v1.0 alternating: {alt_v1_0}")
print(f"v1.1 alternating: {alt_v1_1} ({c1_v1_1} / {c2_v1_1})")
print(f"v1.2 alternating: {alt_v1_2} ({c1_v1_2} / {c2_v1_2})")

print("\n4. HYPOTHESIS: SUBMISSION CREATION BUG")
print("-"*60)

print("\nThe issue is in create_submission.py logic:")
print("1. We generate diverse predictions for test data")
print("2. But when filling sample submission, IDs don't match")
print("3. Unmatched IDs keep sample submission's default values")
print("4. Sample submission has alternating pattern")
print("5. Result: Our predictions are ignored!")

print("\n5. VERIFICATION")
print("-"*60)

# Check how many IDs actually match
sample = pd.read_csv('SampleSubmission.csv')
test = pd.read_csv('TestInputSegments.csv')

# Create test IDs
test_ids = set()
for _, row in test.iterrows():
    time_segment = row['time_segment_id']
    view_label = row['view_label']
    test_ids.add(f'time_segment_{time_segment}_{view_label}_congestion_enter_rating')

sample_ids = set(sample['ID'].values)

matching_ids = test_ids.intersection(sample_ids)
print(f"\nTest IDs: {len(test_ids)}")
print(f"Sample IDs: {len(sample_ids)}")
print(f"Matching IDs: {len(matching_ids)}")
print(f"Match rate: {len(matching_ids)/len(sample_ids)*100:.1f}%")

if len(matching_ids) == 0:
    print("\n❌ CRITICAL BUG: NO MATCHING IDs!")
    print("   Our predictions are NEVER used!")
    print("   Submission file just copies sample submission!")

print("\n6. ROOT CAUSE")
print("-"*60)
print("\nThe test data time segments don't match sample submission segments!")
print("We need to:")
print("1. Find which time segments are actually needed")
print("2. Check if we have data for those segments")
print("3. If not, we need a different approach")

# Show some sample IDs
print("\nSample submission IDs (first 10):")
for id in list(sample_ids)[:10]:
    print(f"  {id}")

print("\nTest data IDs (first 10):")
for id in list(test_ids)[:10]:
    print(f"  {id}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("\n⚠️  Our predictions are being IGNORED!")
print("   The submission file is just copying the sample submission.")
print("   This is why v1.1 and v1.2 have identical scores.")
print("\n💡 SOLUTION:")
print("   1. Investigate which time segments are in sample submission")
print("   2. Check if we have those in our test data")
print("   3. If not, we need to predict for DIFFERENT data")
print("   4. Or the competition format is different than we think")
