#!/usr/bin/env python3
"""
Live Data Update Script
=======================

Downloads latest stock data for your trading universe.
Run daily at 6:00 AM PT / 9:00 AM ET (before market open).

What it does:
1. Downloads yesterday's close + pre-market data
2. Updates CSV files in data/ folder
3. Calculates all indicators (EMA, SMA, stage, etc.)
4. Validates data quality
5. Logs any issues

Runtime: ~5-10 minutes for 81 stocks
"""

import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
DATA_DIR = "data"
LOG_DIR = "logs/data_updates"
UNIVERSE_FILE = "live_universe.txt"  # Core stock list
USE_SCANNER_UNIVERSE = True  # Set to True to download full scanner universe (614+ stocks)

# Create directories
Path(DATA_DIR).mkdir(exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

def load_universe(use_scanner=USE_SCANNER_UNIVERSE):
    """
    Load trading universe.

    Args:
        use_scanner: If True, load full scanner universe (614+ stocks)
                    If False, load only core universe (156 stocks)
    """
    # Always include core universe
    core_symbols = []
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE, 'r') as f:
            core_symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not use_scanner:
        print(f"Loading core universe: {len(core_symbols)} stocks")
        # Add benchmark
        if '^GSPC' not in core_symbols:
            core_symbols.append('^GSPC')
        return core_symbols

    # Load full scanner universe
    try:
        print("Loading scanner universe (S&P 500 + NASDAQ-100 + growth stocks)...")
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scanner.universe_scanner import UniverseScanner

        scanner = UniverseScanner()
        scanner_symbols = scanner.get_all_us_stocks()

        # Combine with core universe
        all_symbols = list(set(scanner_symbols) | set(core_symbols))

        # Add benchmark
        if '^GSPC' not in all_symbols:
            all_symbols.append('^GSPC')

        print(f"Total universe: {len(all_symbols)} stocks")
        return all_symbols

    except Exception as e:
        print(f"Warning: Could not load scanner universe: {e}")
        print(f"Falling back to core universe: {len(core_symbols)} stocks")
        if '^GSPC' not in core_symbols:
            core_symbols.append('^GSPC')
        return core_symbols

def update_stock_data(symbol, days_back=5):
    """
    Update data for a single stock
    
    Args:
        symbol: Stock ticker
        days_back: How many days to download (default 5 for recent update)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
        
        # Check if file exists
        if os.path.exists(filepath):
            # Load existing data
            existing_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            last_date = existing_df.index[-1].to_pydatetime().date()
            
            # Check if already up-to-date
            today = datetime.now().date()
            if last_date >= today - timedelta(days=1):
                print(f"{symbol}: Already up-to-date (last: {last_date})")
                return True
            
            # Download recent data
            start_date = last_date + timedelta(days=1)
        else:
            # Download full history
            print(f"{symbol}: No existing data, downloading full history...")
            existing_df = None
            start_date = None
        
        # Download data from yfinance
        ticker = yf.Ticker(symbol)
        
        if start_date:
            df_new = ticker.history(start=start_date, auto_adjust=True)
        else:
            df_new = ticker.history(period="max", auto_adjust=True)
        
        if len(df_new) == 0:
            print(f"{symbol}: ⚠️ No new data available")
            return False
        
        # Standardize columns
        df_new = df_new.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df_new = df_new[['open', 'high', 'low', 'close', 'volume']]
        
        # Merge with existing if available
        if existing_df is not None:
            combined_df = pd.concat([existing_df, df_new])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
        else:
            combined_df = df_new
        
        # Calculate indicators
        combined_df['ema_21'] = combined_df['close'].ewm(span=21, adjust=False).mean()
        combined_df['sma_200'] = combined_df['close'].rolling(window=200).mean()
        combined_df['sma_30week'] = combined_df['close'].rolling(window=150).mean()
        
        # Volume indicators
        combined_df['avg_volume_50'] = combined_df['volume'].rolling(window=50).mean()
        combined_df['volume_ratio'] = combined_df['volume'] / combined_df['avg_volume_50']
        
        # Save updated file
        combined_df.to_csv(filepath)
        
        date_range = f"{combined_df.index[0].date()} to {combined_df.index[-1].date()}"
        print(f"{symbol}: ✅ Updated ({len(df_new)} new rows) [{date_range}]")
        
        return True
        
    except Exception as e:
        print(f"{symbol}: ❌ Error: {e}")
        return False

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Update stock data for trading universe')
    parser.add_argument('--core-only', action='store_true',
                       help='Only update core universe (156 stocks), skip scanner universe')
    parser.add_argument('--scanner', action='store_true',
                       help='Update full scanner universe (614+ stocks) - this is the default')
    args = parser.parse_args()

    # Determine which universe to use
    use_scanner = not args.core_only  # Default is scanner unless --core-only specified

    print("="*60)
    print("LIVE DATA UPDATE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Scanner Universe (614+ stocks)' if use_scanner else 'Core Universe (156 stocks)'}")
    print("="*60)

    # Create log file
    log_file = os.path.join(LOG_DIR, f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Load universe
    universe = load_universe(use_scanner=use_scanner)
    print(f"\nUpdating {len(universe)} symbols...")
    print()
    
    # Track results
    successful = []
    failed = []
    
    # Update each symbol
    for symbol in universe:
        success = update_stock_data(symbol)
        
        if success:
            successful.append(symbol)
        else:
            failed.append(symbol)
    
    # Summary
    print()
    print("="*60)
    print("UPDATE SUMMARY")
    print("="*60)
    print(f"Successful: {len(successful)}/{len(universe)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed symbols: {', '.join(failed)}")
        print("⚠️ WARNING: Some symbols failed to update!")
    
    print(f"\nData directory: {DATA_DIR}/")
    
    # Write log
    with open(log_file, 'w') as f:
        f.write(f"Data Update Log - {datetime.now()}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        if failed:
            f.write(f"\nFailed symbols:\n")
            for symbol in failed:
                f.write(f"  - {symbol}\n")
    
    print(f"\nLog saved: {log_file}")
    
    # Return status code
    if len(failed) == 0:
        print("\n✅ ALL SYMBOLS UPDATED SUCCESSFULLY!")
        return 0
    elif len(failed) < len(universe) * 0.1:  # < 10% failure
        print("\n⚠️ MOSTLY SUCCESSFUL (some failures)")
        return 0
    else:
        print("\n❌ TOO MANY FAILURES - CHECK ISSUES!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
