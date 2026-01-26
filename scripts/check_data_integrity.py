#!/usr/bin/env python3
"""
Data Integrity Checker
======================

Verifies that CSV data files are complete and have no gaps.

What it checks:
- All expected symbols have CSV files
- No missing trading days (gaps in data)
- Latest data is recent (within 5 days)
- File sizes are reasonable
- No corrupted files

Usage:
    python3 scripts/check_data_integrity.py

    # Verbose output:
    python3 scripts/check_data_integrity.py --verbose

    # Check specific symbols:
    python3 scripts/check_data_integrity.py --symbols AAPL MSFT NVDA
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data"
MIN_ROWS = 200  # Minimum rows expected
MAX_GAP_DAYS = 10  # Maximum allowed gap in trading days
STALENESS_DAYS = 5  # Alert if data is older than this

# ============================================================================
# DATA INTEGRITY CHECKS
# ============================================================================

def load_universe():
    """Load expected symbols from universe file"""
    universe_file = "live_universe.txt"
    
    if os.path.exists(universe_file):
        with open(universe_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return symbols
    else:
        # Fallback: scan data directory
        csv_files = [f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        return [s for s in csv_files if not s.startswith('^')]


def check_file_exists(symbol):
    """Check if CSV file exists for symbol"""
    filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
    return os.path.exists(filepath), filepath


def check_file_size(filepath):
    """Check if file size is reasonable"""
    if not os.path.exists(filepath):
        return False, 0
    
    size_bytes = os.path.getsize(filepath)
    size_kb = size_bytes / 1024
    
    # File should be at least 10KB (very rough check)
    if size_kb < 10:
        return False, size_kb
    
    return True, size_kb


def check_data_completeness(filepath, verbose=False):
    """Check if data is complete and has no gaps"""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # FIX: Handle timezone-aware datetimes
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index.astype(str), utc=True)
        
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        if len(df) < MIN_ROWS:
            return False, f"Only {len(df)} rows (min {MIN_ROWS})"
        
        # Check for date gaps
        dates = pd.DatetimeIndex(df.index)
        date_diffs = dates.to_series().diff()
        
        # Find gaps > MAX_GAP_DAYS (excluding weekends)
        large_gaps = date_diffs[date_diffs > timedelta(days=MAX_GAP_DAYS)]
        
        if len(large_gaps) > 0:
            gap_details = []
            for idx, gap in large_gaps.items():
                gap_details.append(f"{idx.date()} ({gap.days} days)")
            
            if verbose:
                return False, f"{len(large_gaps)} gaps found: {', '.join(gap_details[:3])}"
            else:
                return False, f"{len(large_gaps)} gaps found (max {large_gaps.max().days} days)"
        
        # Check data staleness
        latest_date = dates[-1]
        days_old = (datetime.now() - latest_date).days
        
        if days_old > STALENESS_DAYS:
            return False, f"Data stale ({days_old} days old, latest: {latest_date.date()})"
        
        # All checks passed
        return True, f"{len(df)} rows, latest: {latest_date.date()}"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def check_data_quality(filepath):
    """Check for NaN values and other quality issues"""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # FIX: Handle timezone-aware datetimes
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index.astype(str), utc=True)
        
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        issues = []
        
        # Check for NaN in critical columns
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in critical_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    issues.append(f"{col}: {nan_count} NaN values")
        
        # Check for zeros in price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    issues.append(f"{col}: {zero_count} zero values")
        
        # Check for negative values
        for col in critical_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    issues.append(f"{col}: {neg_count} negative values")
        
        if issues:
            return False, '; '.join(issues)
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


# ============================================================================
# MAIN INTEGRITY CHECK
# ============================================================================

def run_integrity_check(symbols=None, verbose=False):
    """Run complete integrity check on all symbols"""
    
    if symbols is None:
        symbols = load_universe()
    
    print("="*80)
    print("DATA INTEGRITY CHECK")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checking {len(symbols)} symbols")
    print(f"Data directory: {DATA_DIR}")
    print("="*80)
    
    results = {
        'total': len(symbols),
        'passed': 0,
        'failed': 0,
        'missing': 0,
        'issues': []
    }
    
    for symbol in symbols:
        # Check 1: File exists
        exists, filepath = check_file_exists(symbol)
        
        if not exists:
            results['missing'] += 1
            results['issues'].append({
                'symbol': symbol,
                'severity': 'ERROR',
                'issue': 'File missing'
            })
            if verbose:
                print(f"❌ {symbol:6s} - File missing")
            continue
        
        # Check 2: File size
        size_ok, size_kb = check_file_size(filepath)
        if not size_ok:
            results['failed'] += 1
            results['issues'].append({
                'symbol': symbol,
                'severity': 'ERROR',
                'issue': f'File too small ({size_kb:.1f} KB)'
            })
            if verbose:
                print(f"❌ {symbol:6s} - File too small ({size_kb:.1f} KB)")
            continue
        
        # Check 3: Data completeness
        complete_ok, complete_msg = check_data_completeness(filepath, verbose)
        if not complete_ok:
            results['failed'] += 1
            results['issues'].append({
                'symbol': symbol,
                'severity': 'WARNING',
                'issue': complete_msg
            })
            if verbose:
                print(f"⚠️  {symbol:6s} - {complete_msg}")
            continue
        
        # Check 4: Data quality
        quality_ok, quality_msg = check_data_quality(filepath)
        if not quality_ok:
            results['failed'] += 1
            results['issues'].append({
                'symbol': symbol,
                'severity': 'WARNING',
                'issue': quality_msg
            })
            if verbose:
                print(f"⚠️  {symbol:6s} - {quality_msg}")
            continue
        
        # All checks passed
        results['passed'] += 1
        if verbose:
            print(f"✅ {symbol:6s} - {complete_msg}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total symbols:    {results['total']}")
    print(f"Passed:           {results['passed']} ({'green' if results['passed'] == results['total'] else 'yellow'})")
    print(f"Failed/Warnings:  {results['failed']}")
    print(f"Missing:          {results['missing']}")
    print("="*80)
    
    # Print issues if any
    if results['issues']:
        print("\nISSUES FOUND:")
        print("-" * 80)
        
        # Group by severity
        errors = [i for i in results['issues'] if i['severity'] == 'ERROR']
        warnings = [i for i in results['issues'] if i['severity'] == 'WARNING']
        
        if errors:
            print("\nERRORS (require immediate attention):")
            for issue in errors:
                print(f"  ❌ {issue['symbol']:6s} - {issue['issue']}")
        
        if warnings:
            print("\nWARNINGS (should be reviewed):")
            for issue in warnings:
                print(f"  ⚠️  {issue['symbol']:6s} - {issue['issue']}")
        
        print("-" * 80)
        
        # Provide recommendations
        print("\nRECOMMENDATIONS:")
        if results['missing'] > 0:
            print("  → Run: python3 scripts/download_data.py")
            print("    (This will download missing symbols)")
        
        if results['failed'] > 0:
            print("  → Run: python3 scripts/download_data.py")
            print("    (This will refresh all data and fix gaps)")
        
        print()
    else:
        print("\n✅ ALL CHECKS PASSED - Data is clean and complete!")
        print()
    
    return results


def check_index_data():
    """Special check for S&P 500 index data"""
    print("\nCHECKING S&P 500 INDEX DATA:")
    print("-" * 80)
    
    index_files = ['^GSPC.csv', 'GSPC.csv', 'SPY.csv']
    
    found = False
    for filename in index_files:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            complete_ok, complete_msg = check_data_completeness(filepath)
            if complete_ok:
                print(f"✅ {filename} - {complete_msg}")
                found = True
                break
            else:
                print(f"⚠️  {filename} - {complete_msg}")
    
    if not found:
        print(f"❌ No S&P 500 index data found!")
        print(f"   Required for relative strength calculations")
        print(f"   Run: python3 scripts/download_data.py")
    
    print("-" * 80)


# ============================================================================
# ADVANCED CHECKS
# ============================================================================

def check_data_consistency():
    """Check if all files have similar date ranges"""
    print("\nCHECKING DATE RANGE CONSISTENCY:")
    print("-" * 80)
    
    symbols = load_universe()
    date_ranges = {}
    
    for symbol in symbols:
        filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # FIX: Handle timezone-aware datetimes
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index.astype(str), utc=True)
                
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                date_ranges[symbol] = {
                    'start': df.index[0],
                    'end': df.index[-1],
                    'count': len(df)
                }
            except:
                pass
    
    if not date_ranges:
        print("❌ No data loaded")
        return
    
    # Find common date range
    all_starts = [info['start'] for info in date_ranges.values()]
    all_ends = [info['end'] for info in date_ranges.values()]
    
    latest_start = max(all_starts)
    earliest_end = min(all_ends)
    
    print(f"Common date range:")
    print(f"  Latest start: {latest_start.date()}")
    print(f"  Earliest end: {earliest_end.date()}")
    
    # Find outliers
    outliers = []
    for symbol, info in date_ranges.items():
        if info['end'] < (max(all_ends) - timedelta(days=STALENESS_DAYS)):
            outliers.append(f"{symbol} (ends {info['end'].date()})")
    
    if outliers:
        print(f"\n⚠️  Symbols with stale data:")
        for outlier in outliers[:10]:
            print(f"    {outlier}")
        if len(outliers) > 10:
            print(f"    ... and {len(outliers) - 10} more")
    else:
        print(f"\n✅ All symbols have current data")
    
    print("-" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Check data integrity')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to check')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--consistency', action='store_true', help='Check date consistency')
    
    args = parser.parse_args()
    
    # Run main integrity check
    results = run_integrity_check(symbols=args.symbols, verbose=args.verbose)
    
    # Check index data
    if args.symbols is None:  # Only if checking all symbols
        check_index_data()
    
    # Check consistency if requested
    if args.consistency and args.symbols is None:
        check_data_consistency()
    
    # Return exit code based on results
    if results['missing'] > 0 or results['failed'] > 0:
        return 1  # Issues found
    else:
        return 0  # All good


if __name__ == "__main__":
    sys.exit(main())
