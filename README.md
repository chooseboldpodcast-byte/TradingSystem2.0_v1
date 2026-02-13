# Trading System 2.0

Multi-model trading system with 7 production models and a core universe of 126 stocks.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy data from Env1
cp -r ../Trading-System-main/data/* ./data/

# Initialize database
sqlite3 database/weinstein.db < database/schema.sql

# Run production backtest
python scripts/run_prod_backtest.py

# Launch dashboard
streamlit run dashboard/unified_dashboard.py --server.port 8502
```

## Workflow

### Daily Live Trading
```bash
# Generate today's signals
python scripts/live_signal_generator.py
```

### Production Backtest (validate models)
```bash
python scripts/run_prod_backtest.py
python scripts/run_prod_backtest.py --start-year 2015
python scripts/run_prod_backtest.py --description "Full test"
```

### Development Backtest (test changes)
```bash
python scripts/run_dev_backtest.py
python scripts/run_dev_backtest.py --start-year 2015
```

## Models

| Model | Allocation | Strategy |
|-------|-----------|----------|
| 52W High Momentum | 20% | New high breakouts |
| VCP | 20% | Volatility Contraction Pattern (Minervini) |
| RS Breakout | 10% | Relative strength + base breakout |
| Consolidation Breakout | 10% | Flag/pennant patterns |
| High Tight Flag | 10% | 100%+ move + tight consolidation |
| RSI Mean Reversion | 10% | Oversold bounce |
| Pocket Pivot | 10% | Early entry volume signature |

Total allocation: 90% (10% cash reserve)

## Database Schema

Backtest runs are stored with `run_type`:

```
run_type: 'production' or 'development'
```

**Filter in Code:**
```python
db = DatabaseManager()
db.get_all_runs(run_type='production')
db.get_production_runs()
db.get_development_runs()
```

## Directory Structure

```
TradingSystem2.0/
├── models/                    # Production models (7 active)
│   ├── weinstein_core.py
│   ├── rsi_mean_reversion.py
│   ├── momentum_52w_high.py
│   ├── consolidation_breakout.py
│   ├── vcp.py
│   ├── pocket_pivot.py
│   ├── rs_breakout.py
│   └── high_tight_flag.py
├── models_dev/                # Development/experimental models
├── scripts/
│   ├── run_prod_backtest.py
│   ├── run_dev_backtest.py
│   ├── live_signal_generator.py
│   ├── live_data_update.py
│   ├── deploy_model.py
│   └── validation_protocol.py
├── dashboard/
│   └── unified_dashboard.py
├── config/
│   └── models_config.yaml     # Model settings & allocations
├── database/
│   ├── schema.sql
│   └── weinstein.db           # Backtest results
├── data/                      # Stock CSV files
├── logs/
│   ├── signals/              # Daily signal logs
│   └── data_updates/         # Data update logs
└── live_universe.txt          # Core 126 stocks
```

## Dashboard

Launch: `streamlit run dashboard/unified_dashboard.py --server.port 8502`

### Live Trading Mode
- Today's signals with CSS grading
- Open positions
- Performance metrics
- Monthly heatmap

### Backtest Analysis Mode
- **Run Type Filter**: Production / Development / All
- Portfolio overview
- Trade analysis
- Model breakdown
- Universe heatmap

## Critical Rules

1. **Test in models_dev/** - Isolate experiments
2. **Deploy with deploy_model.py** - Automatic backup + git
3. **Two separate databases** - Live != Backtest
4. **Run prod backtest regularly** - Monitor model health
