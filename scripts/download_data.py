# scripts/download_data.py
"""
What does this script do?

This script downloads historical stock/ETF data for a predefined test universe of symbols (tech stocks, ETFs, benchmarks, etc.)
using yfinance and saves each as a CSV file in a local 'data/' folder.

Downloads MAXIMUM available historical data for each symbol (typically 20+ years depending on when the stock IPO'd).
"""
# ---- Library Imports ----
import yfinance as yf      # used to download financial market data from Yahoo Finance
import pandas as pd        # used to manipulate tabular data
from datetime import datetime, timedelta   # used to handle date ranges
import os                  # used for file and directory operations

# Stocks universe.

SYMBOLS = {
    'mag7': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA'],
    
    'block1': ['AVGO', 'TSM', 'SNPS', 'NFLX', 'JPM', 'AMD', 'QCOM','ASML', 'BABA', 'CRM', 'MU', 'PM', 'ADBE', 'COST', 'LRCX', 'ZS','ANET', 'VEEV', 'FIX', 'NU','FICO','PSTG','BKNG', 'ECL', 'NKE', 'LULU', 'TMDX',
                   'CME','XYZ','VRTX','HD', 'GEHC', 'BAM','CHWY','NET','MDB','CPNG','DOCN', 'CAT', 'LMT', 'HON', 'GS', 'DE', 'RTX','COP', 'EME','U', 'TOST', 'BTC','DGX', 'KD', 'DASH', 'IONS', 'GRAB', 'TBBB', 
                   'ONON', 'AXON', 'KNSL', 'TTD', 'SHOP', 'MELI', 'ISRG', 'NOW', 'DDOG', 'PLTR', 'CRWD', 'RKLB', 'ZETA','UNH', 'NVO', 'LLY', 'ABBV', 'HOOD', 'BULL', 'CPRT', 'MSTR', 'V', 'MA', 'PYPL', 'DECK', 'LMND', 
                   'SE', 'VST', 'SNOW', 'SOFI', 'APP', 'DLO', 'BROS', 'PANW', 'MDB', 'JD', 'RDDT', 'ASTS', 'NBIS', 'UBER','BA', 'GE', 'MMM', 'TFC', 'WFC'],
    
    'robotics_drones': ['ARM', 'PRCT', 'SERV', 'PATH', 'AVAV','SYM', 'CGNX', 'TER', 'ROK', 'KTOS'],

    'high_risk': ['LUNR', 'OUST', 'EH', 'ZENA', 'RCAT', 'APLD', 'AIRO', 'ONDS', 'IREN', 'ALAB', 'IONQ', 'OKLO', 'DUOL'],
        
    'energy': ['XOM', 'CVX', 'CEG', 'SLB', 'EOG', 'MPC', 'CCJ', 'GEV', 'EOSE'],
    
    'etfs': ['QQQ', 'SPY', 'IWM', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
             'ARKK', 'ARKQ', 'IGV', 'SOXX', 'SMH', 'SOXL'],
    
    'benchmark': ['^GSPC']  # Don't trade, just for RS calculation
}


# Combine all categories into a single master list
ALL_SYMBOLS = SYMBOLS['mag7'] + SYMBOLS['block1'] + SYMBOLS['robotics_drones'] + SYMBOLS['high_risk'] + SYMBOLS['energy'] +SYMBOLS['etfs'] + SYMBOLS['benchmark']


# ---- FUNCTION DEFINITION ----
def download_historical_data(
    symbols: list,
    output_dir: str = 'data'
):

    """
        Download maximum available historical OHLCV (Open, High, Low, Close, Volume) data for each symbol.    
        Args:
            symbols: List of ticker symbols
            output_dir: Directory to save CSV files

        Returns:
            dict: A dictionary where each key = symbol, value = DataFrame of data
    """

    # ---- Create output directory ----

    os.makedirs(output_dir, exist_ok=True)
    

    # ---- Date info ----

    end_date = datetime.now()
    
    print(f"Downloading maximum available historical data (up to {end_date.date()})")
    print(f"Total symbols: {len(symbols)}\n")


 
    # ---- Initialize empty structures to track results ----   

    results = {}   # holds dataframes for successful downloads
    failed = []    # holds symbols that fail to download
    


    # ---- Iterate through every ticker symbol in the list ----

    for symbol in symbols:
        try:
            filepath = os.path.join(output_dir, f"{symbol}.csv")

            # --- Check if we already have existing data ---
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                last_date = existing_df.index[-1].to_pydatetime().date()
                days_old = (datetime.now().date() - last_date).days

                # If the data is already up-to-date (less than 1 day old), skip
                if days_old < 1:
                    print(f"{symbol}: already up-to-date (last: {last_date}) ✓")
                    results[symbol] = existing_df
                    continue
                else:
                    print(f"{symbol}: updating from {last_date}...")
                    # Download from day after last date
                    start_dl = last_date + timedelta(days=1)
            else:
                # No existing file: download full history
                print(f"{symbol}: downloading maximum available history...")
                start_dl = None  # Will use period="max"

            # --- Download from yfinance ---
            ticker = yf.Ticker(symbol)
            
            if start_dl is None:
                # Full history download using period="max"
                df_new = ticker.history(period="max", auto_adjust=True)
            else:
                # Incremental update from last date
                df_new = ticker.history(start=start_dl, end=end_date, auto_adjust=True)

            if len(df_new) == 0:
                print(f"✗ No new data")
                failed.append(symbol)
                continue

            # --- Standardize columns ---
            df_new = df_new.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            df_new = df_new[['open', 'high', 'low', 'close', 'volume']]
            
            # --- Calculate 21-day EMA ---
            df_new['ema_21'] = df_new['close'].ewm(span=21, adjust=False).mean()
            
            # --- Calculate 200-day SMA
            df_new['sma_200'] = df_new['close'].rolling(window=200).mean() #MB

            # --- Merge with existing data if it exists ---
            if os.path.exists(filepath):
                combined_df = pd.concat([existing_df, df_new])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
                # Recalculate EMA and SMA for entire dataset to ensure continuity
                combined_df['ema_21'] = combined_df['close'].ewm(span=21, adjust=False).mean()
                combined_df['sma_200'] = combined_df['close'].rolling(window=200).mean() #MB
                combined_df.to_csv(filepath)
                results[symbol] = combined_df
                date_range = f"{combined_df.index[0].date()} to {combined_df.index[-1].date()}"
                print(f"✓ Updated ({len(df_new)} new rows) [{date_range}]")
            else:
                df_new.to_csv(filepath)
                results[symbol] = df_new
                date_range = f"{df_new.index[0].date()} to {df_new.index[-1].date()}"
                print(f"✓ Saved ({len(df_new)} rows) [{date_range}]")

        except Exception as e:
            print(f"✗ Error fetching {symbol}: {e}")
            failed.append(symbol)
    
    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(results)}/{len(symbols)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed symbols: {', '.join(failed)}")
    print(f"\nData saved to: {output_dir}/")
    
    return results


# ---- MAIN EXECUTION BLOCK ----
# This ensures the script only runs when executed directly,
# not when imported as a module in another script.

if __name__ == "__main__":
    # Download all test data
    data = download_historical_data(ALL_SYMBOLS)
    
    # Print sample for verification
    if 'AAPL' in data:
        print(f"\nSample data for AAPL:")
        print(data['AAPL'].tail())
        print(f"\nTotal rows: {len(data['AAPL'])}")
        print(f"Date range: {data['AAPL'].index[0].date()} to {data['AAPL'].index[-1].date()}")
