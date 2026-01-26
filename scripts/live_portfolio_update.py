#!/usr/bin/env python3
"""
Live Portfolio Update Script
=============================

Updates your live portfolio with executed trades.
Run daily at 4:10 PM PT / 1:10 PM ET (after market close).

What it does:
1. Loads your broker's executed trades (CSV export)
2. Updates live_positions table
3. Closes positions that were sold
4. Calculates P&L
5. Updates performance metrics
6. Creates daily summary

Runtime: ~1-2 minutes
"""

import os
import sys
import pandas as pd
import sqlite3
from datetime import datetime, date
from pathlib import Path

# Configuration
DB_PATH = "database/live_trading.db"
BROKER_FILLS_DIR = "data/broker_fills"
LOG_DIR = "logs/portfolio_updates"

# Create directories
Path(BROKER_FILLS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

class LivePortfolioUpdater:
    """Updates live portfolio with executed trades"""
    
    def __init__(self):
        self.db_path = DB_PATH
    
    def get_today_fills(self, fill_date=None):
        """
        Load today's executed trades from broker CSV
        
        Expected CSV format:
        symbol,side,shares,price,timestamp,commission
        AAPL,BUY,10,150.25,2024-01-15 09:35:00,1.00
        MSFT,SELL,15,350.50,2024-01-15 14:20:00,1.50
        
        Args:
            fill_date: Date to load (default: today)
        
        Returns:
            DataFrame with executed trades
        """
        if fill_date is None:
            fill_date = date.today()
        
        # Look for CSV file
        fill_file = os.path.join(BROKER_FILLS_DIR, f"{fill_date.strftime('%Y-%m-%d')}.csv")
        
        if not os.path.exists(fill_file):
            print(f"⚠️ No fill file found for {fill_date}")
            print(f"Expected: {fill_file}")
            print("\nTo create fill file:")
            print("1. Export trades from your broker as CSV")
            print(f"2. Save to: {fill_file}")
            print("3. Format: symbol,side,shares,price,timestamp,commission")
            return None
        
        # Load CSV
        df = pd.read_csv(fill_file)
        
        # Validate columns
        required_cols = ['symbol', 'side', 'shares', 'price']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ Missing columns in CSV: {missing}")
            print(f"Required: {required_cols}")
            return None
        
        # Add optional columns if missing
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        if 'commission' not in df.columns:
            df['commission'] = 0.0
        
        print(f"✅ Loaded {len(df)} fills from {fill_file}")
        
        return df
    
    def process_buy_fill(self, fill):
        """Process a BUY execution"""
        with sqlite3.connect(self.db_path) as conn:
            # Check if we already have this position
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM live_positions 
                WHERE symbol = ? AND status = 'OPEN'
            """, (fill['symbol'],))
            
            existing = cursor.fetchone()
            
            if existing:
                print(f"{fill['symbol']}: ⚠️ Position already exists (ID: {existing[0]})")
                return False
            
            # Get the signal that triggered this trade
            cursor.execute("""
                SELECT * FROM daily_signals 
                WHERE symbol = ? AND signal_type = 'ENTRY'
                AND signal_date = ?
                ORDER BY id DESC LIMIT 1
            """, (fill['symbol'], date.today()))
            
            signal = cursor.fetchone()
            
            # Insert new position
            position_value = fill['shares'] * fill['price']
            
            cursor.execute("""
                INSERT INTO live_positions (
                    model, symbol, entry_date, entry_price,
                    shares, stop_loss, position_value, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'Weinstein_Core',
                fill['symbol'],
                date.today(),
                fill['price'],
                fill['shares'],
                signal[5] if signal else fill['price'] * 0.92,  # stop_loss from signal or -8%
                position_value,
                'OPEN'
            ))
            
            # Mark signal as executed
            if signal:
                cursor.execute("""
                    UPDATE daily_signals 
                    SET executed = 1, approved = 1
                    WHERE id = ?
                """, (signal[0],))
            
            conn.commit()
            
            print(f"✅ {fill['symbol']}: Position OPENED")
            print(f"   {fill['shares']} shares @ ${fill['price']:.2f} = ${position_value:.2f}")
            
            return True
    
    def process_sell_fill(self, fill):
        """Process a SELL execution"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Find the open position
            cursor.execute("""
                SELECT * FROM live_positions 
                WHERE symbol = ? AND status = 'OPEN'
            """, (fill['symbol'],))
            
            position = cursor.fetchone()
            
            if not position:
                print(f"{fill['symbol']}: ❌ No open position found!")
                return False
            
            # Extract position details
            pos_id = position[0]
            entry_date = position[3]
            entry_price = position[4]
            shares = position[5]
            
            # Calculate P&L
            exit_price = fill['price']
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price / entry_price - 1) * 100
            
            # Calculate hold days
            entry_dt = pd.to_datetime(entry_date).date()
            exit_dt = date.today()
            hold_days = (exit_dt - entry_dt).days
            
            # Get exit reason from signal
            cursor.execute("""
                SELECT reason FROM daily_signals 
                WHERE symbol = ? AND signal_type = 'EXIT'
                AND signal_date = ?
                ORDER BY id DESC LIMIT 1
            """, (fill['symbol'], date.today()))
            
            signal = cursor.fetchone()
            exit_reason = signal[0] if signal else 'MANUAL'
            
            # Update position status
            cursor.execute("""
                UPDATE live_positions 
                SET status = 'CLOSED'
                WHERE id = ?
            """, (pos_id,))
            
            # Record in live_trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS live_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER,
                    model TEXT,
                    symbol TEXT,
                    entry_date DATE,
                    entry_price REAL,
                    exit_date DATE,
                    exit_price REAL,
                    shares INTEGER,
                    pnl REAL,
                    pnl_pct REAL,
                    hold_days INTEGER,
                    exit_reason TEXT,
                    commission REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT INTO live_trades (
                    position_id, model, symbol, entry_date, entry_price,
                    exit_date, exit_price, shares, pnl, pnl_pct,
                    hold_days, exit_reason, commission
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos_id,
                'Weinstein_Core',
                fill['symbol'],
                entry_date,
                entry_price,
                exit_dt,
                exit_price,
                shares,
                pnl,
                pnl_pct,
                hold_days,
                exit_reason,
                fill.get('commission', 0)
            ))
            
            # Mark exit signal as executed
            cursor.execute("""
                UPDATE daily_signals 
                SET executed = 1, approved = 1
                WHERE symbol = ? AND signal_type = 'EXIT'
                AND signal_date = ?
            """, (fill['symbol'], date.today()))
            
            conn.commit()
            
            result = "WIN" if pnl > 0 else "LOSS"
            print(f"{'✅' if pnl > 0 else '❌'} {fill['symbol']}: Position CLOSED ({result})")
            print(f"   Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}")
            print(f"   P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%) | Hold: {hold_days} days")
            print(f"   Reason: {exit_reason}")
            
            return True
    
    def update_daily_performance(self):
            """Update daily performance metrics"""
            with sqlite3.connect(self.db_path) as conn:
                # Create tables if needed
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS daily_performance (
                        date DATE PRIMARY KEY,
                        portfolio_value REAL,
                        daily_pnl REAL,
                        open_positions INTEGER,
                        total_trades INTEGER,
                        notes TEXT
                    )
                """)
                
                # CREATE live_trades table if it doesn't exist yet
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS live_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id INTEGER,
                        model TEXT,
                        symbol TEXT,
                        entry_date DATE,
                        entry_price REAL,
                        exit_date DATE,
                        exit_price REAL,
                        shares INTEGER,
                        pnl REAL,
                        pnl_pct REAL,
                        hold_days INTEGER,
                        exit_reason TEXT,
                        commission REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Get current positions
                positions = pd.read_sql_query("""
                    SELECT * FROM live_positions WHERE status = 'OPEN'
                """, conn)
                
                # Get all closed trades (table now exists)
                trades = pd.read_sql_query("""
                    SELECT * FROM live_trades
                """, conn)
                
                # Calculate portfolio value
                # Start with initial capital
                initial_capital = 50000
                
                # Add cumulative P&L from closed trades
                if len(trades) > 0:
                    cumulative_pnl = trades['pnl'].sum()
                else:
                    cumulative_pnl = 0
                
                portfolio_value = initial_capital + cumulative_pnl
                
                # Today's P&L (from today's closed trades)
                today_trades = trades[trades['exit_date'] == str(date.today())]
                daily_pnl = today_trades['pnl'].sum() if len(today_trades) > 0 else 0
                
                # Save daily performance
                conn.execute("""
                    INSERT OR REPLACE INTO daily_performance (
                        date, portfolio_value, daily_pnl, open_positions, total_trades
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    date.today(),
                    portfolio_value,
                    daily_pnl,
                    len(positions),
                    len(trades)
                ))
                
                conn.commit()
                
                print("\n" + "="*60)
                print("PORTFOLIO STATUS")
                print("="*60)
                print(f"Portfolio Value: ${portfolio_value:,.2f}")
                print(f"Total P&L:       ${cumulative_pnl:,.2f} ({cumulative_pnl/initial_capital*100:+.2f}%)")
                print(f"Today's P&L:     ${daily_pnl:,.2f}")
                print(f"Open Positions:  {len(positions)}")
                print(f"Total Trades:    {len(trades)}")
    
    def run(self):
        """Main execution"""
        print("="*60)
        print("LIVE PORTFOLIO UPDATE")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Load today's fills
        fills = self.get_today_fills()
        
        if fills is None:
            print("\n⚠️ No fills to process")
            print("If you executed trades today:")
            print("1. Export from broker as CSV")
            print(f"2. Save to: {BROKER_FILLS_DIR}/{date.today().strftime('%Y-%m-%d')}.csv")
            print("3. Run this script again")
            return 1
        
        # Process each fill
        print(f"\nProcessing {len(fills)} fills...\n")
        
        for _, fill in fills.iterrows():
            side = fill['side'].upper()
            
            if side == 'BUY':
                self.process_buy_fill(fill)
            elif side == 'SELL':
                self.process_sell_fill(fill)
            else:
                print(f"⚠️ Unknown side: {side}")
        
        # Update performance metrics
        self.update_daily_performance()
        
        # Create log
        log_file = os.path.join(LOG_DIR, f"update_{date.today().strftime('%Y%m%d')}.log")
        with open(log_file, 'w') as f:
            f.write(f"Portfolio Update - {datetime.now()}\n")
            f.write(f"Fills Processed: {len(fills) if fills is not None else 0}\n")
        
        print(f"\nLog saved: {log_file}")
        print("\n✅ Portfolio update complete!")
        
        return 0

def main():
    updater = LivePortfolioUpdater()
    return updater.run()

if __name__ == "__main__":
    sys.exit(main())
