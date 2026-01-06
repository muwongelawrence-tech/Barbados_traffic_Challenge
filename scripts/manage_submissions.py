#!/usr/bin/env python3
"""
Helper script to manage submissions
Creates properly named submission files with metadata
"""
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path


def get_next_version():
    """Get next submission version number"""
    submissions_dir = Path('submissions')
    if not submissions_dir.exists():
        return 1
    
    existing = list(submissions_dir.glob('submission_v*.csv'))
    if not existing:
        return 1
    
    versions = []
    for f in existing:
        try:
            version = int(f.stem.split('_v')[1].split('_')[0])
            versions.append(version)
        except:
            continue
    
    return max(versions) + 1 if versions else 1


def create_submission(description, source_file='submission.csv', score=None):
    """
    Create a new versioned submission file
    
    Args:
        description: Short description of this submission (e.g., 'baseline_temporal_only')
        source_file: Path to the submission file to version
        score: Optional validation score
    """
    # Get next version
    version = get_next_version()
    
    # Create filename
    filename = f"submission_v{version}_{description}.csv"
    submissions_dir = Path('submissions')
    submissions_dir.mkdir(exist_ok=True)
    
    dest_path = submissions_dir / filename
    
    # Copy file
    if not Path(source_file).exists():
        print(f"❌ Error: {source_file} not found")
        return None
    
    shutil.copy(source_file, dest_path)
    
    # Create metadata file
    metadata_path = submissions_dir / f"submission_v{version}_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Version: v{version}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {source_file}\n")
        if score:
            f.write(f"Validation Score: {score}\n")
    
    print(f"✅ Created: {dest_path}")
    print(f"📝 Metadata: {metadata_path}")
    print(f"\n📊 Next steps:")
    print(f"   1. Upload {filename} to Zindi")
    print(f"   2. Record leaderboard score in submissions/SUBMISSION_LOG.md")
    print(f"   3. Update metadata file with leaderboard score")
    
    return dest_path


def list_submissions():
    """List all submissions"""
    submissions_dir = Path('submissions')
    if not submissions_dir.exists():
        print("No submissions yet")
        return
    
    submissions = sorted(submissions_dir.glob('submission_v*.csv'))
    
    if not submissions:
        print("No submissions yet")
        return
    
    print("\n📁 Submissions:")
    print("=" * 80)
    for sub in submissions:
        size = sub.stat().st_size
        mtime = datetime.fromtimestamp(sub.stat().st_mtime)
        print(f"  {sub.name}")
        print(f"    Size: {size:,} bytes")
        print(f"    Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check for metadata
        metadata_file = sub.parent / f"{sub.stem}_metadata.txt"
        if metadata_file.exists():
            with open(metadata_file) as f:
                for line in f:
                    if line.startswith('Validation Score:'):
                        print(f"    {line.strip()}")
        print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/manage_submissions.py create <description> [source_file] [score]")
        print("  python scripts/manage_submissions.py list")
        print("\nExamples:")
        print("  python scripts/manage_submissions.py create 'baseline_temporal_only' submission.csv 0.5745")
        print("  python scripts/manage_submissions.py create 'tuned_hyperparams'")
        print("  python scripts/manage_submissions.py list")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'create':
        if len(sys.argv) < 3:
            print("❌ Error: Description required")
            print("Usage: python scripts/manage_submissions.py create <description>")
            sys.exit(1)
        
        description = sys.argv[2]
        source_file = sys.argv[3] if len(sys.argv) > 3 else 'submission.csv'
        score = sys.argv[4] if len(sys.argv) > 4 else None
        
        create_submission(description, source_file, score)
    
    elif command == 'list':
        list_submissions()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: create, list")
        sys.exit(1)
