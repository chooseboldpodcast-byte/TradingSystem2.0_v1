# database/db_manager.py
"""
Database manager for Weinstein trading system
Handles saving and retrieving backtest results with run_type and config_type support

Config Types:
- A: 126 stocks, 4 original models (baseline)
- B: 126 stocks, 8 models (4 original + 4 new)
- C: Scanner + 126 stocks, 8 models (full potential)

Run Types:
- production: Validated model performance
- development: Experimental model testing
"""
import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

class DatabaseManager:
    """Manages SQLite database for backtest results"""
    
    # Valid config and run types
    VALID_CONFIG_TYPES = ['A', 'B', 'C']
    VALID_RUN_TYPES = ['production', 'development']
    
    CONFIG_DESCRIPTIONS = {
        'A': '126 stocks, 4 original models',
        'B': '126 stocks, 8 models (4+4)',
        'C': 'Scanner + 126 stocks, 8 models'
    }
    
    def __init__(self, db_path: str = "database/weinstein.db"):
        """Initialize database connection"""
        self.db_path = db_path
        
        # Create database directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Create tables if they don't exist"""
        schema_path = Path(__file__).parent / "schema.sql"
        
        with sqlite3.connect(self.db_path) as conn:
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())
            
            # Check if we need to add new columns (migration)
            self._migrate_if_needed(conn)
    
    def _migrate_if_needed(self, conn):
        """Add run_type and config_type columns if they don't exist"""
        cursor = conn.cursor()
        
        # Check existing columns
        cursor.execute("PRAGMA table_info(backtest_runs)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Add run_type if missing
        if 'run_type' not in columns:
            cursor.execute("ALTER TABLE backtest_runs ADD COLUMN run_type TEXT DEFAULT 'development'")
            print("Added run_type column to backtest_runs")
        
        # Add config_type if missing
        if 'config_type' not in columns:
            cursor.execute("ALTER TABLE backtest_runs ADD COLUMN config_type TEXT DEFAULT 'A'")
            print("Added config_type column to backtest_runs")
        
        conn.commit()
    
    def save_backtest_run(
        self, 
        results: dict, 
        run_type: str = 'development',
        config_type: str = 'A',
        description: str = None, 
        parameters: dict = None
    ) -> int:
        """
        Save a backtest run and all its trades to database
        
        Args:
            results: Dictionary from backtest._calculate_results()
            run_type: 'production' or 'development'
            config_type: 'A', 'B', or 'C'
            description: Optional description of this backtest
            parameters: Optional dict of parameters used
        
        Returns:
            run_id: ID of the saved backtest run
        """
        # Validate inputs
        if run_type not in self.VALID_RUN_TYPES:
            raise ValueError(f"Invalid run_type: {run_type}. Must be one of {self.VALID_RUN_TYPES}")
        if config_type not in self.VALID_CONFIG_TYPES:
            raise ValueError(f"Invalid config_type: {config_type}. Must be one of {self.VALID_CONFIG_TYPES}")
        
        # Use provided description or generate simple one
        # Format: "Prod-ConfigA" or "Dev-ConfigB"
        run_type_short = 'Prod' if run_type == 'production' else 'Dev'
        auto_desc = f"{run_type_short}-Config{config_type}"
        final_desc = description if description else auto_desc
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Save backtest run summary
            cursor.execute("""
                INSERT INTO backtest_runs (
                    run_type, config_type, description, initial_capital, final_value, 
                    total_pnl, total_return_pct, cagr, total_trades, winning_trades,
                    losing_trades, win_rate, profit_factor, avg_hold_days, parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_type,
                config_type,
                final_desc,
                results.get('initial_capital', 100000),
                results.get('initial_capital', 100000) + results['total_pnl'],
                results['total_pnl'],
                results['total_return_pct'],
                results.get('cagr', 0),
                results['total_trades'],
                results['winning_trades'],
                results['losing_trades'],
                results['win_rate'],
                results['profit_factor'],
                results['avg_hold_days'],
                json.dumps(parameters) if parameters else None
            ))
            
            run_id = cursor.lastrowid
            
            # Save individual trades
            if 'trades' in results and len(results['trades']) > 0:
                trades_df = results['trades']
                
                for _, trade in trades_df.iterrows():
                    # Get model safely
                    try:
                        model = trade.get('model', 'UNKNOWN')
                        if pd.isna(model) or model is None:
                            model = 'UNKNOWN'
                    except (KeyError, AttributeError):
                        model = 'UNKNOWN'
                    
                    # Get entry_method safely
                    try:
                        entry_method = trade.get('entry_method', 'UNKNOWN')
                        if pd.isna(entry_method) or entry_method is None:
                            entry_method = 'UNKNOWN'
                    except (KeyError, AttributeError):
                        entry_method = 'UNKNOWN'
                    
                    # Convert timestamps to strings for SQLite
                    entry_date_str = str(trade['entry_date']) if pd.notna(trade['entry_date']) else None
                    exit_date_str = str(trade['exit_date']) if pd.notna(trade['exit_date']) else None
                    
                    cursor.execute("""
                        INSERT INTO trades (
                            backtest_run_id, model, symbol, entry_date, entry_price,
                            exit_date, exit_price, shares, pnl, pnl_pct,
                            hold_days, exit_reason, entry_method
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        run_id,
                        str(model),
                        trade['symbol'],
                        entry_date_str,
                        float(trade['entry_price']),
                        exit_date_str,
                        float(trade['exit_price']),
                        int(trade['shares']),
                        float(trade['pnl']),
                        float(trade['pnl_pct']),
                        int(trade['hold_days']),
                        trade['exit_reason'],
                        str(entry_method)
                    ))
            
            conn.commit()
            print(f"✅ Saved {run_type} backtest (Config {config_type}) as run #{run_id}")
            return run_id
    
    def get_all_runs(
        self, 
        run_type: str = None, 
        config_type: str = None
    ) -> pd.DataFrame:
        """
        Get summary of all backtest runs, optionally filtered
        
        Args:
            run_type: Filter by 'production' or 'development' (None = all)
            config_type: Filter by 'A', 'B', or 'C' (None = all)
        """
        query = """
            SELECT 
                id,
                run_date,
                run_type,
                config_type,
                description,
                initial_capital,
                final_value,
                total_pnl,
                total_return_pct,
                cagr,
                total_trades,
                win_rate,
                profit_factor
            FROM backtest_runs
            WHERE 1=1
        """
        params = []
        
        if run_type:
            query += " AND run_type = ?"
            params.append(run_type)
        
        if config_type:
            query += " AND config_type = ?"
            params.append(config_type)
        
        query += " ORDER BY run_date DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
            if len(df) > 0:
                df['run_date'] = pd.to_datetime(df['run_date'])
            
            return df
    
    def get_production_runs(self, config_type: str = None) -> pd.DataFrame:
        """Get all production runs, optionally filtered by config"""
        return self.get_all_runs(run_type='production', config_type=config_type)
    
    def get_development_runs(self, config_type: str = None) -> pd.DataFrame:
        """Get all development runs, optionally filtered by config"""
        return self.get_all_runs(run_type='development', config_type=config_type)
    
    def get_runs_by_config(self, config_type: str) -> pd.DataFrame:
        """Get all runs for a specific config (A, B, or C)"""
        return self.get_all_runs(config_type=config_type)
    
    def get_run_details(self, run_id: int) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
        """Get detailed results for a specific backtest run"""
        with sqlite3.connect(self.db_path) as conn:
            # Get run summary
            run_df = pd.read_sql_query("""
                SELECT * FROM backtest_runs WHERE id = ?
            """, conn, params=(run_id,))
            
            if len(run_df) == 0:
                return None, None
            
            # Get all trades for this run
            trades_df = pd.read_sql_query("""
                SELECT * FROM trades WHERE backtest_run_id = ?
                ORDER BY entry_date
            """, conn, params=(run_id,))
            
            if len(trades_df) > 0:
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            
            return run_df.iloc[0], trades_df
    
    def get_latest_run(
        self, 
        run_type: str = None, 
        config_type: str = None
    ) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
        """Get the most recent backtest run, optionally filtered"""
        query = "SELECT MAX(id) FROM backtest_runs WHERE 1=1"
        params = []
        
        if run_type:
            query += " AND run_type = ?"
            params.append(run_type)
        
        if config_type:
            query += " AND config_type = ?"
            params.append(config_type)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            if result and result[0]:
                return self.get_run_details(result[0])
            
            return None, None
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """Get summary for comparing across configs and run types"""
        query = """
            SELECT 
                run_type,
                config_type,
                COUNT(*) as num_runs,
                AVG(cagr) as avg_cagr,
                AVG(total_return_pct) as avg_return,
                AVG(win_rate) as avg_win_rate,
                AVG(profit_factor) as avg_profit_factor,
                SUM(total_trades) as total_trades,
                MAX(run_date) as last_run
            FROM backtest_runs
            GROUP BY run_type, config_type
            ORDER BY run_type, config_type
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df
    
    def delete_run(self, run_id: int):
        """Delete a backtest run and all its trades"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trades WHERE backtest_run_id = ?", (run_id,))
            cursor.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
            conn.commit()
            print(f"✅ Deleted backtest run #{run_id}")
    
    def delete_all_runs(self, run_type: str = None, config_type: str = None, confirm: bool = False):
        """
        Delete multiple runs (with safeguards)
        
        Args:
            run_type: Filter by run type (None = all)
            config_type: Filter by config (None = all)
            confirm: Must be True to actually delete
        """
        if not confirm:
            print("⚠️  Set confirm=True to actually delete runs")
            return
        
        query = "SELECT id FROM backtest_runs WHERE 1=1"
        params = []
        
        if run_type:
            query += " AND run_type = ?"
            params.append(run_type)
        
        if config_type:
            query += " AND config_type = ?"
            params.append(config_type)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            run_ids = [row[0] for row in cursor.fetchall()]
            
            for run_id in run_ids:
                cursor.execute("DELETE FROM trades WHERE backtest_run_id = ?", (run_id,))
                cursor.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
            
            conn.commit()
            print(f"✅ Deleted {len(run_ids)} backtest runs")


if __name__ == "__main__":
    # Quick test
    db = DatabaseManager()
    print("✅ Database initialized successfully!")
    
    # Show all runs
    runs = db.get_all_runs()
    print(f"\nTotal backtest runs in database: {len(runs)}")
    
    if len(runs) > 0:
        print("\nRecent runs:")
        print(runs[['id', 'run_type', 'config_type', 'cagr', 'win_rate']].head(10))
        
        print("\nComparison summary:")
        print(db.get_comparison_summary())
