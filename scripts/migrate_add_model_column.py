#!/usr/bin/env python3
"""
Database Migration Script - Add Model Column
============================================

This script adds the 'model' column to the trades table in your existing database.

WHY DO I NEED THIS?
-------------------
The multi-model trading system needs to track which model (Weinstein_Core, RSI_Mean_Reversion, etc.)
generated each trade. The old database schema didn't have this column, so the dashboard can't show
per-model performance breakdowns.

WHAT DOES THIS DO?
------------------
1. Backs up your existing database
2. Adds a 'model' column to the trades table
3. Sets existing trades to 'UNKNOWN' (you'll need to re-run backtests for per-model data)
4. Creates an index on the new column for fast queries

HOW TO USE:
-----------
python scripts/migrate_add_model_column.py

SAFE TO RUN:
-----------
✅ Creates a backup before making changes
✅ Only adds column if it doesn't exist
✅ Won't lose any existing data
"""

import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

def migrate_database(db_path: str = "database/weinstein.db"):
    """
    Add model column to trades table
    
    Args:
        db_path: Path to the database file
    """
    print("="*60)
    print("DATABASE MIGRATION - ADD MODEL COLUMN")
    print("="*60)
    
    db_path = Path(db_path)
    
    # Check if database exists
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("   Nothing to migrate. Run a backtest first to create the database.")
        return
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}.db"
    
    print(f"\n1. Creating backup...")
    print(f"   {db_path} → {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"   ✅ Backup created")
    
    # Connect to database
    print(f"\n2. Connecting to database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if model column already exists
    print(f"\n3. Checking if migration is needed...")
    cursor.execute("PRAGMA table_info(trades)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'model' in columns:
        print(f"   ℹ️  Model column already exists!")
        print(f"   No migration needed.")
        conn.close()
        return
    
    print(f"   ✅ Model column not found - migration needed")
    
    # Add model column
    print(f"\n4. Adding 'model' column to trades table...")
    try:
        cursor.execute("ALTER TABLE trades ADD COLUMN model TEXT DEFAULT 'UNKNOWN'")
        print(f"   ✅ Column added")
    except sqlite3.OperationalError as e:
        print(f"   ❌ Error: {e}")
        conn.close()
        return
    
    # Create index on model column
    print(f"\n5. Creating index on model column...")
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_model ON trades(model)")
        print(f"   ✅ Index created")
    except sqlite3.OperationalError as e:
        print(f"   ⚠️  Warning: {e}")
    
    # Commit changes
    conn.commit()
    
    # Verify the change
    print(f"\n6. Verifying migration...")
    cursor.execute("SELECT COUNT(*) FROM trades")
    total_trades = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM trades WHERE model = 'UNKNOWN'")
    unknown_trades = cursor.fetchone()[0]
    
    print(f"   Total trades in database: {total_trades}")
    print(f"   Trades with model='UNKNOWN': {unknown_trades}")
    
    if unknown_trades == total_trades:
        print(f"   ✅ All existing trades set to 'UNKNOWN'")
    
    conn.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"MIGRATION COMPLETE ✅")
    print(f"{'='*60}")
    print(f"\nYour database has been updated!")
    print(f"\nNEXT STEPS:")
    print(f"1. Run a new backtest with the multi-model system:")
    print(f"   python scripts/run_4_model_backtest.py")
    print(f"\n2. View results in the dashboard:")
    print(f"   streamlit run dashboard/app.py")
    print(f"\n3. The dashboard will now show per-model performance cards!")
    print(f"\nNOTE: Existing backtest runs will show model='UNKNOWN' since they")
    print(f"      were created before the multi-model system was implemented.")
    print(f"\nBACKUP LOCATION:")
    print(f"   {backup_path}")
    print(f"   (Delete this once you confirm everything works)")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    # Allow custom database path
    db_path = sys.argv[1] if len(sys.argv) > 1 else "database/weinstein.db"
    
    migrate_database(db_path)
