#!/usr/bin/env python3
"""
Smart Data Refresh
==================

Intelligently updates data based on integrity check results.

Strategy (Option C - Check First):
1. Run data integrity check
2. If data is clean → Skip full download (save 10-15 min!)
3. If issues found → Run full download_data.py
4. If data is stale → Run live_data_update.py only

Usage:
    # Before any backtest
    python3 scripts/smart_data_refresh.py
    
    # Force full download
    python3 scripts/smart_data_refresh.py --force
    
    # Verbose output
    python3 scripts/smart_data_refresh.py --verbose

Exit codes:
    0 - Data is current and clean
    1 - Full download was needed and completed
    2 - Data update failed
"""

import os
import sys
import subprocess
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPTS_DIR = "scripts"
CHECK_SCRIPT = os.path.join(SCRIPTS_DIR, "check_data_integrity.py")
DOWNLOAD_SCRIPT = os.path.join(SCRIPTS_DIR, "download_data.py")
UPDATE_SCRIPT = os.path.join(SCRIPTS_DIR, "live_data_update.py")

# ============================================================================
# MAIN LOGIC
# ============================================================================

def check_data_integrity(verbose=False):
    """Run data integrity check and return results"""
    print("="*80)
    print("SMART DATA REFRESH - Checking data quality first...")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not os.path.exists(CHECK_SCRIPT):
        print(f"⚠️  Warning: {CHECK_SCRIPT} not found")
        print(f"   Proceeding with full download to be safe\n")
        return False, "check_script_missing"
    
    # Run integrity check
    cmd = ['python3', CHECK_SCRIPT]
    if verbose:
        cmd.append('--verbose')
    
    result = subprocess.run(cmd, capture_output=not verbose)
    
    # Return code 0 = all good, 1 = issues found
    if result.returncode == 0:
        return True, "clean"
    else:
        return False, "issues_found"


def run_full_download(verbose=False):
    """Run full data download"""
    print("\n" + "="*80)
    print("RUNNING FULL DATA DOWNLOAD")
    print("="*80)
    print("This will take 10-15 minutes...\n")
    
    if not os.path.exists(DOWNLOAD_SCRIPT):
        print(f"❌ Error: {DOWNLOAD_SCRIPT} not found!")
        return False
    
    cmd = ['python3', DOWNLOAD_SCRIPT]
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Full download complete!")
        return True
    else:
        print("\n❌ Full download failed!")
        return False


def run_incremental_update(verbose=False):
    """Run incremental data update"""
    print("\n" + "="*80)
    print("RUNNING INCREMENTAL DATA UPDATE")
    print("="*80)
    print("Quick update, should take 2-5 minutes...\n")
    
    if not os.path.exists(UPDATE_SCRIPT):
        print(f"❌ Error: {UPDATE_SCRIPT} not found!")
        return False
    
    cmd = ['python3', UPDATE_SCRIPT]
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Incremental update complete!")
        return True
    else:
        print("\n❌ Incremental update failed!")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart data refresh')
    parser.add_argument('--force', action='store_true', 
                       help='Force full download without checking')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Force mode - skip check, just download
    if args.force:
        print("🔄 FORCE MODE - Running full download\n")
        if run_full_download(args.verbose):
            return 0
        else:
            return 2
    
    # Smart mode - check first
    is_clean, reason = check_data_integrity(args.verbose)
    
    if is_clean:
        # Data is clean!
        print("\n" + "="*80)
        print("✅ DATA IS CLEAN AND CURRENT")
        print("="*80)
        print("No download needed - proceed with backtest!")
        print(f"Saved 10-15 minutes by skipping unnecessary download 🎉\n")
        return 0
    
    else:
        # Issues found - need to refresh
        print("\n" + "="*80)
        print("⚠️  DATA ISSUES DETECTED")
        print("="*80)
        print(f"Reason: {reason}")
        print("Refreshing data to ensure accuracy...\n")
        
        if run_full_download(args.verbose):
            print("\n" + "="*80)
            print("✅ DATA REFRESH COMPLETE")
            print("="*80)
            print("Data is now clean - proceed with backtest!\n")
            return 1
        else:
            print("\n" + "="*80)
            print("❌ DATA REFRESH FAILED")
            print("="*80)
            print("Check error messages above")
            print("You may need to run download_data.py manually\n")
            return 2


if __name__ == "__main__":
    sys.exit(main())
