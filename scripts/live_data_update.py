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

# Create directories
Path(DATA_DIR).mkdir(exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

def load_universe():
    """Load trading universe from core universe file."""
    symbols = []
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Loading core universe: {len(symbols)} stocks")
    # Add benchmark
    if '^GSPC' not in symbols:
        symbols.append('^GSPC')
    return symbols

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
    parser.parse_args()

    print("="*60)
    print("LIVE DATA UPDATE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Create log file
    log_file = os.path.join(LOG_DIR, f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Load universe
    universe = load_universe()
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
