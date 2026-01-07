"""
Submission versioning utility
Helps create and manage versioned submission files
"""
import os
import sys
import pandas as pd
from datetime import datetime
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def list_submissions():
    """List all submission files with their versions"""
    submissions_dir = config.SUBMISSIONS_DIR
    
    if not os.path.exists(submissions_dir):
        print(f"No submissions directory found at {submissions_dir}")
        return
    
    files = [f for f in os.listdir(submissions_dir) if f.endswith('.csv') and f.startswith('v')]
    
    if not files:
        print("No versioned submissions found")
        return
    
    files.sort()
    
    print("\n" + "="*80)
    print("SUBMISSION HISTORY")
    print("="*80)
    print(f"{'Version':<20} {'Size':<10} {'Rows':<10} {'Modified':<20}")
    print("-"*80)
    
    for f in files:
        filepath = os.path.join(submissions_dir, f)
        size = os.path.getsize(filepath)
        modified = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        # Count rows
        try:
            df = pd.read_csv(filepath)
            rows = len(df)
        except:
            rows = "?"
        
        print(f"{f:<20} {size:>8}B  {rows:>8}  {modified.strftime('%Y-%m-%d %H:%M')}")
    
    print("="*80)


def create_new_version(base_file, new_version, description):
    """Create a new versioned submission file"""
    submissions_dir = config.SUBMISSIONS_DIR
    
    base_path = os.path.join(submissions_dir, base_file)
    new_filename = f"v{new_version}_{description}.csv"
    new_path = os.path.join(submissions_dir, new_filename)
    
    if not os.path.exists(base_path):
        print(f"Error: Base file not found: {base_path}")
        return False
    
    if os.path.exists(new_path):
        print(f"Error: Version already exists: {new_filename}")
        return False
    
    # Copy file
    shutil.copy2(base_path, new_path)
    print(f"✓ Created new version: {new_filename}")
    
    return True


def get_latest_version():
    """Get the latest version number"""
    submissions_dir = config.SUBMISSIONS_DIR
    
    if not os.path.exists(submissions_dir):
        return "1.0"
    
    files = [f for f in os.listdir(submissions_dir) if f.endswith('.csv') and f.startswith('v')]
    
    if not files:
        return "1.0"
    
    # Extract version numbers
    versions = []
    for f in files:
        try:
            version_str = f.split('_')[0][1:]  # Remove 'v' prefix
            major, minor = version_str.split('.')
            versions.append((int(major), int(minor)))
        except:
            continue
    
    if not versions:
        return "1.0"
    
    versions.sort()
    latest = versions[-1]
    
    return f"{latest[0]}.{latest[1]}"


def suggest_next_version(increment='minor'):
    """Suggest next version number"""
    latest = get_latest_version()
    major, minor = map(int, latest.split('.'))
    
    if increment == 'major':
        return f"{major + 1}.0"
    else:  # minor
        return f"{major}.{minor + 1}"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage submission versions')
    parser.add_argument('--list', action='store_true', help='List all submissions')
    parser.add_argument('--latest', action='store_true', help='Show latest version')
    parser.add_argument('--next', choices=['major', 'minor'], help='Suggest next version')
    
    args = parser.parse_args()
    
    if args.list:
        list_submissions()
    elif args.latest:
        print(f"Latest version: v{get_latest_version()}")
    elif args.next:
        print(f"Next {args.next} version: v{suggest_next_version(args.next)}")
    else:
        # Default: list submissions
        list_submissions()


if __name__ == "__main__":
    main()
