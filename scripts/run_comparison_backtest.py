# scripts/run_comparison_backtest.py
"""
Comparison Backtest Runner

Runs backtests with different configurations to compare:
1. Original 4 models (126 stocks) - matches Env1
2. All 7 models (126 stocks) - full production set

Usage:
    python scripts/run_comparison_backtest.py --config original
    python scripts/run_comparison_backtest.py --config all_core
    python scripts/run_comparison_backtest.py --all  # Run all configs
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import pandas as pd
from datetime import datetime
from typing import List, Dict

# Direct file imports (no __init__.py needed)
from models.weinstein_core import WeinsteinCore
from models.rsi_mean_reversion import RSIMeanReversion
from models.momentum_52w_high import Momentum52WeekHigh
from models.consolidation_breakout import ConsolidationBreakout
from models.vcp import VCP
from models.pocket_pivot import PocketPivot
from models.rs_breakout import RSBreakout
from models.high_tight_flag import HighTightFlag
from core.portfolio_manager import PortfolioManager
from core.weinstein_engine import WeinsteinEngine

# Define model groups locally
ORIGINAL_MODELS = ['weinstein_core', 'rsi_mean_reversion', 'momentum_52w_high', 'consolidation_breakout']
MOMENTUM_MODELS = ['vcp', 'pocket_pivot', 'rs_breakout', 'high_tight_flag']

MODEL_CLASSES = {
    'weinstein_core': WeinsteinCore,
    'rsi_mean_reversion': RSIMeanReversion,
    'momentum_52w_high': Momentum52WeekHigh,
    'consolidation_breakout': ConsolidationBreakout,
    'vcp': VCP,
    'pocket_pivot': PocketPivot,
    'rs_breakout': RSBreakout,
    'high_tight_flag': HighTightFlag,
}

def get_model(name: str, **kwargs):
    """Factory function to create model instances"""
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_CLASSES[name](**kwargs)


def load_config(config_path: str = 'config/models_config.yaml') -> dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_universe(config: dict) -> List[str]:
    """Load stock universe from core universe file."""
    core_path = config.get('paths', {}).get('core_universe', 'live_universe.txt')
    with open(core_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def create_models(model_names: List[str], config: dict) -> List:
    """Create model instances from config"""
    models = []
    model_configs = config.get('models', {})
    
    for name in model_names:
        if name in model_configs:
            model_config = model_configs[name]
            if model_config.get('enabled', True):
                allocation = model_config.get('allocation', 0.10)
                models.append(get_model(name, allocation_pct=allocation))
    
    return models


def load_stock_data(symbols: List[str], data_dir: str = 'data') -> Dict[str, pd.DataFrame]:
    """Load stock data from CSV files"""
    data = {}
    engine = WeinsteinEngine()
    
    print(f"Loading data for {len(symbols)} symbols...")
    
    for symbol in symbols:
        filepath = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                # Calculate indicators
                df = engine.calculate_indicators(df)
                data[symbol] = df
            except Exception as e:
                print(f"  Error loading {symbol}: {e}")
    
    print(f"Loaded {len(data)} symbols successfully")
    return data


def run_backtest(
    models: List,
    stock_data: Dict[str, pd.DataFrame],
    initial_capital: float = 100000,
    start_date: str = None,
    end_date: str = None,
    verbose: bool = False
) -> Dict:
    """
    Run backtest with given models and data
    
    Returns:
        Dict with results and trades
    """
    # Initialize portfolio manager
    pm = PortfolioManager(initial_capital, models, verbose=verbose)
    
    # Get date range
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    
    all_dates = sorted(list(all_dates))
    
    if start_date:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
    if end_date:
        all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
    
    print(f"\nBacktest period: {all_dates[0].date()} to {all_dates[-1].date()}")
    print(f"Trading days: {len(all_dates)}")
    
    # Collect all signals first
    all_signals = []
    
    print("Generating signals...")
    for symbol, df in stock_data.items():
        for model in models:
            for idx in range(150, len(df)):
                date = df.index[idx]
                
                if start_date and date < pd.Timestamp(start_date):
                    continue
                if end_date and date > pd.Timestamp(end_date):
                    continue
                
                # Check for entry
                signal = model.generate_entry_signals(df, idx)
                
                if signal.get('signal', False):
                    all_signals.append({
                        'date': date,
                        'symbol': symbol,
                        'model': model.name,
                        'type': 'ENTRY',
                        'price': signal['entry_price'],
                        'stop_loss': signal['stop_loss'],
                        'confidence': signal.get('confidence', 0.8),
                        'metadata': signal.get('metadata', {}),
                        'df_idx': idx
                    })
    
    # Sort signals chronologically
    all_signals.sort(key=lambda x: x['date'])
    print(f"Total signals generated: {len(all_signals)}")
    
    # Process signals in order
    print("Processing trades...")
    processed = 0
    
    for signal in all_signals:
        date = signal['date']
        symbol = signal['symbol']
        model_name = signal['model']
        model = pm.models[model_name]
        
        # Check exits first for all open positions
        for pos_symbol in list(pm.open_positions.keys()):
            if pos_symbol not in stock_data:
                continue
            
            pos_df = stock_data[pos_symbol]
            pos_model_name = pm.open_positions[pos_symbol]['model']
            pos_model = pm.models[pos_model_name]
            
            try:
                pos_idx = pos_df.index.get_loc(date)
                if isinstance(pos_idx, slice):
                    pos_idx = pos_idx.start
                
                exit_signal = pos_model.generate_exit_signals(
                    pos_df, pos_idx, pm.open_positions[pos_symbol]
                )
                
                if exit_signal.get('signal', False):
                    pm.close_position(
                        pos_symbol,
                        date,
                        exit_signal['exit_price'],
                        exit_signal['reason']
                    )
            except (KeyError, TypeError):
                continue
        
        # Check stop losses
        pm.check_stop_losses(date, stock_data)
        
        # Process entry signal
        if symbol not in pm.open_positions:
            available = pm.get_model_available_capital(model_name)
            shares = model.calculate_position_size(
                available,
                signal['price'],
                signal['confidence']
            )
            
            if shares > 0:
                position_cost = shares * signal['price']
                
                if pm.can_open_position(model_name, symbol, position_cost):
                    pm.open_position(
                        model_name,
                        symbol,
                        date,
                        signal['price'],
                        shares,
                        signal['stop_loss'],
                        signal['metadata']
                    )
        
        processed += 1
        if processed % 1000 == 0:
            print(f"  Processed {processed}/{len(all_signals)} signals...")
    
    # Close remaining positions at end
    if all_dates:
        final_date = all_dates[-1]
        for symbol in list(pm.open_positions.keys()):
            if symbol in stock_data:
                df = stock_data[symbol]
                if final_date in df.index:
                    pm.close_position(
                        symbol,
                        final_date,
                        df.loc[final_date, 'close'],
                        'END_OF_BACKTEST'
                    )
    
    # Calculate results
    results = pm.get_portfolio_summary()
    
    # Calculate CAGR
    if all_dates and len(all_dates) > 1:
        years = (all_dates[-1] - all_dates[0]).days / 365.25
        if years > 0 and results['current_capital'] > 0:
            results['cagr'] = ((results['current_capital'] / initial_capital) ** (1/years) - 1) * 100
        else:
            results['cagr'] = 0
    else:
        results['cagr'] = 0
    
    results['trades'] = pm.closed_trades
    results['model_summary'] = pm.get_all_models_summary()
    
    return results


def print_results(results: Dict, config_name: str):
    """Print backtest results"""
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS: {config_name}")
    print(f"{'='*60}")
    print(f"Initial Capital:  ${100000:,.0f}")
    print(f"Final Capital:    ${results['current_capital']:,.0f}")
    print(f"Total P&L:        ${results['total_pnl']:,.0f}")
    print(f"Total Return:     {results['total_return_pct']:.2f}%")
    print(f"CAGR:             {results.get('cagr', 0):.2f}%")
    print(f"\nTotal Trades:     {results['closed_trades']}")
    
    # Model breakdown
    if 'model_summary' in results:
        print(f"\n{'='*60}")
        print("PERFORMANCE BY MODEL")
        print(f"{'='*60}")
        model_df = results['model_summary']
        for _, row in model_df.iterrows():
            trades = row['closed_trades']
            if trades > 0:
                print(f"\n{row['name']}:")
                print(f"  Trades:      {int(trades)}")
                print(f"  P&L:         ${row['total_pnl']:,.0f}")
                print(f"  Win Rate:    {row['win_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Run comparison backtests')
    parser.add_argument('--config', choices=['original', 'all_core'],
                       help='Configuration to run')
    parser.add_argument('--all', action='store_true', help='Run all configurations')
    parser.add_argument('--start', type=str, default='2015-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Define configurations
    configs = {
        'original': {
            'name': 'Original 4 Models (Core 126)',
            'models': ORIGINAL_MODELS,
        },
        'all_core': {
            'name': 'All 7 Models (Core 126)',
            'models': ORIGINAL_MODELS + MOMENTUM_MODELS,
        },
    }

    # Determine which configs to run
    if args.all:
        configs_to_run = ['original', 'all_core']
    elif args.config:
        configs_to_run = [args.config]
    else:
        configs_to_run = ['all_core']  # Default
    
    results_summary = []
    
    for config_name in configs_to_run:
        cfg = configs[config_name]
        print(f"\n{'#'*60}")
        print(f"RUNNING: {cfg['name']}")
        print(f"{'#'*60}")
        
        # Load universe
        universe = load_universe(config)
        print(f"Universe: {len(universe)} stocks")
        
        # Load data
        stock_data = load_stock_data(universe)
        
        if not stock_data:
            print("No data loaded. Skipping.")
            continue
        
        # Create models
        models = create_models(cfg['models'], config)
        print(f"Models: {[m.name for m in models]}")
        
        # Run backtest
        results = run_backtest(
            models,
            stock_data,
            initial_capital=config['portfolio']['initial_capital'],
            start_date=args.start,
            end_date=args.end,
            verbose=args.verbose
        )
        
        # Print results
        print_results(results, cfg['name'])
        
        # Store for comparison
        results_summary.append({
            'config': cfg['name'],
            'final_capital': results['current_capital'],
            'total_return': results['total_return_pct'],
            'cagr': results.get('cagr', 0),
            'trades': results['closed_trades']
        })
    
    # Print comparison summary
    if len(results_summary) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Configuration':<35} {'Return':>10} {'CAGR':>8} {'Trades':>8}")
        print(f"{'-'*60}")
        for r in results_summary:
            print(f"{r['config']:<35} {r['total_return']:>9.1f}% {r['cagr']:>7.1f}% {r['trades']:>8}")


if __name__ == "__main__":
    main()
