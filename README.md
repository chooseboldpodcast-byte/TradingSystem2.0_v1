# Trading System Dev Environment

Extended trading system with 8 models, dynamic scanner, and three configurations.

## Three Configurations

| Config | Universe | Models | Description |
|--------|----------|--------|-------------|
| **A** | 126 core stocks | 4 original | Baseline (matches Env1) |
| **B** | 126 core stocks | 8 models (4+4) | Test new models on same universe |
| **C** | Scanner + 126 | 8 models | Full potential with dynamic universe |

### Config A - Baseline
- Exact replica of production environment (Env1)
- Use for comparison benchmarks
- Models: Weinstein Core, RSI Mean Reversion, 52W High Momentum, Consolidation Breakout

### Config B - Extended Models
- Same 126 stocks as Config A
- Adds 4 new models: VCP, Pocket Pivot, RS Breakout, High Tight Flag
- Use to measure incremental value from new models

### Config C - Full Scanner
- Dynamic universe: Scanner results ∪ Core 126 stocks
- All 8 models
- Use to discover opportunities beyond core universe

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
python scripts/run_prod_backtest.py --config A
python scripts/run_prod_backtest.py --config B
python scripts/run_prod_backtest.py --config C

# Launch dashboard
streamlit run dashboard/unified_dashboard.py --server.port 8502
```

## Workflow

### Daily Live Trading
```bash
# Generate today's signals for your chosen config
python scripts/live_signal_generator.py --config A --save
python scripts/live_signal_generator.py --config B --save
python scripts/live_signal_generator.py --config C --save
```

### Production Backtest (validate models)
```bash
# Run with specific config
python scripts/run_prod_backtest.py --config A
python scripts/run_prod_backtest.py --config B
python scripts/run_prod_backtest.py --config C

# With date range
python scripts/run_prod_backtest.py --config B --start 2015-01-01 --end 2023-12-31
```

### Development Backtest (test changes)
```bash
# Test all production models
python scripts/run_dev_backtest.py --config B

# Test specific dev models from models_dev/
python scripts/run_dev_backtest.py --config A --dev-models weinstein_core_v2.py
```

### Deploy Model Changes
```bash
# After testing in dev, promote to production
python scripts/deploy_model.py weinstein_core_v2.py
```

## Database Schema

Backtest runs are stored with both `run_type` and `config_type`:

```
run_type: 'production' or 'development'
config_type: 'A', 'B', or 'C'
```

**Filter in Dashboard:**
- Run Type: Production / Development / All
- Config: A / B / C / All

**Filter in Code:**
```python
db = DatabaseManager()
db.get_all_runs(run_type='production', config_type='B')
db.get_production_runs(config_type='A')
db.get_development_runs(config_type='C')
```

## Models

### Original Models (4)
| Model | Allocation (A) | Allocation (B/C) | Strategy |
|-------|----------------|------------------|----------|
| Weinstein Core | 40% | 25% | Stage Analysis breakout/pullback |
| RSI Mean Reversion | 10% | 8% | Oversold bounce |
| 52W High Momentum | 15% | 10% | New high breakouts |
| Consolidation Breakout | 10% | 8% | Flag/pennant patterns |

### New Models (4)
| Model | Allocation (B/C) | Strategy |
|-------|------------------|----------|
| VCP | 12% | Volatility Contraction Pattern (Minervini) |
| Pocket Pivot | 10% | Early entry volume signature |
| RS Breakout | 12% | Relative strength + base breakout |
| High Tight Flag | 8% | 100%+ move + tight consolidation |

## Directory Structure

```
Trading-System-Dev/
├── models/                    # Production models (8 total)
│   ├── weinstein_core.py
│   ├── rsi_mean_reversion.py
│   ├── momentum_52w_high.py
│   ├── consolidation_breakout.py
│   ├── vcp.py                 # NEW
│   ├── pocket_pivot.py        # NEW
│   ├── rs_breakout.py         # NEW
│   └── high_tight_flag.py     # NEW
├── models_dev/                # Development/experimental models
├── scanner/                   # Dynamic universe scanner
│   └── universe_scanner.py
├── scripts/
│   ├── run_prod_backtest.py   # --config A|B|C
│   ├── run_dev_backtest.py    # --config A|B|C
│   ├── live_signal_generator.py  # --config A|B|C
│   ├── live_data_update.py
│   ├── deploy_model.py
│   └── ...
├── dashboard/
│   └── unified_dashboard.py   # Filter by run_type + config_type
├── config/
│   └── models_config.yaml     # All config definitions
├── database/
│   ├── schema.sql             # With run_type, config_type columns
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
- Config selector (A/B/C)
- Today's signals
- Open positions
- Performance metrics
- Monthly heatmap

### Backtest Analysis Mode
- **Run Type Filter**: Production / Development / All
- **Config Filter**: A / B / C / All
- Portfolio overview
- Trade analysis
- Model breakdown
- Universe heatmap

## Comparison Framework

```
Config A (baseline):     CAGR: 14.2%  ← Match this with Env1
Config B (same universe): CAGR: 17.8%  ← +3.6% from new models
Config C (full scanner):  CAGR: 19.5%  ← +1.7% from scanner
```

## Critical Rules

1. **Live uses Config A initially** - Production stability
2. **Test in models_dev/** - Isolate experiments
3. **Deploy with deploy_model.py** - Automatic backup + git
4. **Two separate databases** - Live ≠ Backtest
5. **Run prod backtest regularly** - Monitor model health
6. **Compare configs weekly** - Track incremental value
