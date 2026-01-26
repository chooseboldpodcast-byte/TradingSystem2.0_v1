-- Weinstein Trading System Database Schema
-- Multi-Model System Schema with Config Support

-- Backtest runs table
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_type TEXT DEFAULT 'development',  -- 'production' or 'development'
    config_type TEXT DEFAULT 'A',          -- 'A', 'B', or 'C'
    description TEXT,
    initial_capital REAL,
    final_value REAL,
    total_pnl REAL,
    total_return_pct REAL,
    cagr REAL,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate REAL,
    profit_factor REAL,
    avg_hold_days REAL,
    parameters TEXT  -- JSON string of parameters used
);

-- Index for filtering by run_type and config_type
CREATE INDEX IF NOT EXISTS idx_backtest_run_type ON backtest_runs(run_type);
CREATE INDEX IF NOT EXISTS idx_backtest_config_type ON backtest_runs(config_type);
CREATE INDEX IF NOT EXISTS idx_backtest_run_config ON backtest_runs(run_type, config_type);

-- Individual trades table
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER,
    model TEXT,  -- NEW: Which model generated this trade
    symbol TEXT,
    entry_date TIMESTAMP,
    entry_price REAL,
    exit_date TIMESTAMP,
    exit_price REAL,
    shares INTEGER,
    pnl REAL,
    pnl_pct REAL,
    hold_days INTEGER,
    exit_reason TEXT,
    entry_method TEXT,
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id)
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_trades_backtest_run ON trades(backtest_run_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON trades(entry_date);
CREATE INDEX IF NOT EXISTS idx_trades_model ON trades(model);  -- NEW: Index for model queries
