# scripts/run_backtest.py
"""
Standard Backtest Runner for Dev Environment

Runs backtest with configurable models from YAML config.

Usage:
    # Run with all enabled models from config
    python scripts/run_backtest.py
    
    # Run with specific models
    python scripts/run_backtest.py --models weinstein_core vcp rs_breakout
    
    # Run with date range
    python scripts/run_backtest.py --start 2015-01-01 --end 2023-12-31
    
    # Run with custom initial capital
    python scripts/run_backtest.py --capital 50000
    
    # Verbose mode (shows individual trades)
    python scripts/run_backtest.py --verbose
    
    # Save results to database
    python scripts/run_backtest.py --save
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import argparse
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from database.db_manager import DatabaseManager

# Model registry
MODELS = {
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
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    return MODELS[name](**kwargs)


class MultiModelBacktest:
    """Multi-model backtest engine"""
    
    def __init__(
        self, 
        initial_capital: float = 100000,
        models: list = None,
        verbose: bool = False
    ):
        self.initial_capital = initial_capital
        self.verbose = verbose
        
        if models is None:
            # Default: all 8 models
            models = [
                WeinsteinCore(allocation_pct=0.25),
                RSIMeanReversion(allocation_pct=0.08),
                Momentum52WeekHigh(allocation_pct=0.10),
                ConsolidationBreakout(allocation_pct=0.08),
                VCP(allocation_pct=0.12),
                PocketPivot(allocation_pct=0.10),
                RSBreakout(allocation_pct=0.12),
                HighTightFlag(allocation_pct=0.08)
            ]
        
        self.portfolio = PortfolioManager(
            initial_capital=initial_capital,
            models=models,
            verbose=verbose
        )
        
        self.engine = WeinsteinEngine()
        
        # Track rejection stats
        self.rejection_stats = {}
        for model in models:
            self.rejection_stats[model.name] = {
                'total_signals': 0,
                'accepted': 0,
                'rejected_no_capital': 0,
                'rejected_duplicate': 0,
                'rejected_allocation': 0
            }
    
    def run(
        self, 
        stock_data: dict, 
        index_data: pd.DataFrame = None,
        start_date: str = None,
        end_date: str = None
    ) -> dict:
        """Run the backtest"""
        print(f"\n{'='*60}")
        print(f"MULTI-MODEL BACKTEST - STARTING")
        print(f"{'='*60}")
        print(f"Stocks: {len(stock_data)}")
        print(f"Models: {len(self.portfolio.models)}")
        for name, model in self.portfolio.models.items():
            print(f"  - {name}: {model.allocation_pct*100:.0f}%")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        
        # PHASE 1: Analyze stocks
        print(f"\nPhase 1: Analyzing stocks and calculating indicators...")
        analyzed_data = {}
        for symbol, df in stock_data.items():
            if self.verbose:
                print(f"  Analyzing {symbol}...")
            analyzed = self.engine.analyze_stock(df, index_data)
            analyzed_data[symbol] = analyzed
        print(f"  Analyzed {len(analyzed_data)} stocks")
        
        # PHASE 2: Generate signals
        print(f"\nPhase 2: Generating signals from all models...")
        all_signals = self._generate_all_signals(
            analyzed_data, index_data, start_date, end_date
        )
        
        if len(all_signals) == 0:
            print("❌ No signals generated!")
            return self._calculate_results()
        
        print(f"\n✅ Generated {len(all_signals)} total signals")
        print(f"  Date range: {all_signals['date'].min().date()} to {all_signals['date'].max().date()}")
        
        # Signal breakdown by model
        for model_name in self.portfolio.models.keys():
            model_signals = all_signals[all_signals['model'] == model_name]
            entry_signals = model_signals[model_signals['type'] == 'ENTRY']
            print(f"  {model_name}: {len(entry_signals)} entry signals")
        
        # PHASE 3: Process signals chronologically
        print(f"\nPhase 3: Processing signals chronologically...")
        self._process_signals_chronologically(all_signals, analyzed_data)
        
        self._print_rejection_stats()
        
        return self._calculate_results()
    
    def _generate_all_signals(
        self, 
        analyzed_data: dict, 
        index_data: pd.DataFrame,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """Generate all entry/exit signals from all models"""
        signals = []
        
        # Convert date strings to timestamps
        start_ts = pd.Timestamp(start_date) if start_date else None
        end_ts = pd.Timestamp(end_date) if end_date else None
        
        for symbol, df in analyzed_data.items():
            for model_name, model in self.portfolio.models.items():
                model_signals = self._generate_signals_for_stock_and_model(
                    symbol, df, model, index_data, start_ts, end_ts
                )
                signals.extend(model_signals)
        
        if len(signals) == 0:
            return pd.DataFrame()
        
        signals_df = pd.DataFrame(signals)
        signals_df = signals_df.sort_values('date').reset_index(drop=True)
        
        return signals_df
    
    def _generate_signals_for_stock_and_model(
        self, 
        symbol: str, 
        df: pd.DataFrame,
        model,
        index_data: pd.DataFrame,
        start_ts: pd.Timestamp = None,
        end_ts: pd.Timestamp = None
    ) -> list:
        """Generate signals for one stock and one model"""
        signals = []
        
        for i in range(150, len(df)):
            current_date = df.index[i]
            
            # Apply date filters
            if start_ts and current_date < start_ts:
                continue
            if end_ts and current_date > end_ts:
                continue
            
            # Get market data at same index if available
            try:
                if index_data is not None:
                    market_idx = index_data.index.get_loc(current_date)
                    market_df = index_data
                else:
                    market_df = None
            except:
                market_df = None
            
            # Check for ENTRY signal
            entry_signal = model.generate_entry_signals(df, i, market_df)
            
            if entry_signal.get('signal'):
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'model': model.name,
                    'type': 'ENTRY',
                    'data_idx': i,
                    'signal_data': entry_signal
                })
            
            # Check for EXIT signals based on stage transitions
            current_stage = df['stage'].iloc[i] if 'stage' in df.columns else None
            
            if current_stage is not None:
                # Stage 4 exit for Weinstein
                if model.name == 'Weinstein_Core' and current_stage == 4:
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'model': model.name,
                        'type': 'EXIT',
                        'data_idx': i,
                        'signal_data': {'reason': 'STAGE_4'}
                    })
                # Stage 3 exit for momentum models
                elif model.name in ['52W_High_Momentum', 'Consolidation_Breakout', 
                                    'VCP', 'RS_Breakout', 'High_Tight_Flag'] and current_stage == 3:
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'model': model.name,
                        'type': 'EXIT',
                        'data_idx': i,
                        'signal_data': {'reason': 'STAGE_3'}
                    })
        
        return signals
    
    def _process_signals_chronologically(
        self, 
        signals_df: pd.DataFrame,
        analyzed_data: dict
    ):
        """Process all signals in chronological order"""
        total_signals = len(signals_df)
        processed = 0
        
        for idx, signal in signals_df.iterrows():
            if processed % 500 == 0 and processed > 0:
                print(f"    Processed {processed}/{total_signals} signals...")
            
            signal_date = signal['date']
            symbol = signal['symbol']
            model_name = signal['model']
            signal_type = signal['type']
            signal_data = signal['signal_data']
            data_idx = signal['data_idx']
            
            df = analyzed_data[symbol]
            model = self.portfolio.models[model_name]
            
            # Process EXIT signals first
            if signal_type == 'EXIT':
                if symbol in self.portfolio.open_positions:
                    position = self.portfolio.open_positions[symbol]
                    
                    if position['model'] == model_name:
                        exit_price = df['close'].iloc[data_idx]
                        exit_reason = signal_data.get('reason', 'MODEL_EXIT')
                        
                        self.portfolio.close_position(
                            symbol=symbol,
                            exit_date=signal_date,
                            exit_price=exit_price,
                            exit_reason=exit_reason
                        )
            
            # Process ENTRY signals
            elif signal_type == 'ENTRY':
                self.rejection_stats[model_name]['total_signals'] += 1
                
                # Check if already have position in this symbol
                if symbol in self.portfolio.open_positions:
                    self.rejection_stats[model_name]['rejected_duplicate'] += 1
                    continue
                
                # Get model-specific available capital
                model_available_capital = self.portfolio.get_model_available_capital(model_name)
                
                entry_price = signal_data['entry_price']
                shares = model.calculate_position_size(
                    available_capital=model_available_capital,
                    price=entry_price,
                    confidence=signal_data.get('confidence', 1.0)
                )
                
                if shares <= 0:
                    self.rejection_stats[model_name]['rejected_no_capital'] += 1
                    continue
                
                success = self.portfolio.open_position(
                    model_name=model_name,
                    symbol=symbol,
                    entry_date=signal_date,
                    entry_price=entry_price,
                    shares=shares,
                    stop_loss=signal_data['stop_loss'],
                    metadata=signal_data.get('metadata', {})
                )
                
                if success:
                    self.rejection_stats[model_name]['accepted'] += 1
                else:
                    self.rejection_stats[model_name]['rejected_allocation'] += 1
            
            # Check stop losses
            self.portfolio.check_stop_losses(signal_date, analyzed_data)
            
            # Check model-specific exits for all open positions
            for pos_symbol in list(self.portfolio.open_positions.keys()):
                position = self.portfolio.open_positions[pos_symbol]
                pos_model = self.portfolio.models[position['model']]
                pos_df = analyzed_data[pos_symbol]
                
                try:
                    pos_idx = pos_df.index.get_loc(signal_date)
                    if isinstance(pos_idx, slice):
                        pos_idx = pos_idx.start
                    
                    exit_signal = pos_model.generate_exit_signals(pos_df, pos_idx, position)
                    
                    if exit_signal.get('signal'):
                        self.portfolio.close_position(
                            symbol=pos_symbol,
                            exit_date=signal_date,
                            exit_price=exit_signal['exit_price'],
                            exit_reason=exit_signal['reason']
                        )
                except (KeyError, TypeError):
                    continue
            
            processed += 1
        
        print(f"    Completed processing {total_signals} signals")
        
        # Close remaining positions at end of backtest
        print(f"\n  Closing {len(self.portfolio.open_positions)} remaining positions...")
        for symbol in list(self.portfolio.open_positions.keys()):
            df = analyzed_data[symbol]
            final_price = df['close'].iloc[-1]
            final_date = df.index[-1]
            self.portfolio.close_position(
                symbol=symbol,
                exit_date=final_date,
                exit_price=final_price,
                exit_reason='END_OF_BACKTEST'
            )
    
    def _print_rejection_stats(self):
        """Print signal acceptance statistics"""
        print(f"\n{'='*60}")
        print(f"SIGNAL ACCEPTANCE STATISTICS")
        print(f"{'='*60}")
        
        for model_name, stats in self.rejection_stats.items():
            if stats['total_signals'] > 0:
                print(f"\n{model_name}:")
                print(f"  Total Signals:       {stats['total_signals']}")
                print(f"  Accepted:            {stats['accepted']}")
                print(f"  Rejected - No Cap:   {stats['rejected_no_capital']}")
                print(f"  Rejected - Dup:      {stats['rejected_duplicate']}")
                print(f"  Rejected - Alloc:    {stats['rejected_allocation']}")
                
                acceptance_rate = (stats['accepted'] / stats['total_signals']) * 100
                print(f"  Acceptance Rate:     {acceptance_rate:.1f}%")
    
    def _calculate_results(self) -> dict:
        """Calculate and print final results"""
        self.portfolio.print_final_summary()
        
        trades_df = self.portfolio.get_trades_dataframe()
        
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'cagr': 0,
                'years': 0
            }
        
        # Calculate metrics
        total_pnl = trades_df['pnl'].sum()
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        total_trades = len(trades_df)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        
        first_date = trades_df['entry_date'].min()
        last_date = trades_df['exit_date'].max()
        years = (last_date - first_date).days / 365.25
        
        final_capital = self.initial_capital + total_pnl
        total_return_pct = (total_pnl / self.initial_capital) * 100
        cagr = (((final_capital / self.initial_capital) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        avg_hold_days = trades_df['hold_days'].mean()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(f"Period: {first_date.date()} to {last_date.date()} ({years:.1f} years)")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Final Capital:   ${final_capital:,.0f}")
        print(f"Total Return:    {total_return_pct:.2f}%")
        print(f"CAGR:            {cagr:.2f}%")
        print(f"")
        print(f"Total Trades:    {total_trades}")
        print(f"Win Rate:        {win_rate:.1f}%")
        print(f"Profit Factor:   {profit_factor:.2f}")
        print(f"Avg Win:         ${avg_win:,.0f}")
        print(f"Avg Loss:        ${avg_loss:,.0f}")
        print(f"Avg Hold Days:   {avg_hold_days:.1f}")
        
        return {
            'trades': trades_df,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'years': years,
            'cagr': cagr,
            'avg_hold_days': avg_hold_days,
            'final_value': final_capital,
            'initial_capital': self.initial_capital
        }


def load_config(config_path: str = 'config/models_config.yaml') -> dict:
    """Load configuration from YAML"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_data(data_dir: str = 'data') -> tuple:
    """Load stock data from CSV files"""
    data = {}
    index_data = None
    
    print(f"Loading data from {data_dir}/...")
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            symbol = filename.replace('.csv', '')
            filepath = os.path.join(data_dir, filename)
            
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # Handle timezone
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # Store SPY/^GSPC as index data
                if symbol in ['SPY', '^GSPC']:
                    index_data = df.copy()
                    if symbol == '^GSPC':
                        continue  # Don't trade the index
                
                data[symbol] = df
                
            except Exception as e:
                print(f"  Error loading {symbol}: {e}")
    
    print(f"Loaded {len(data)} stocks")
    
    return data, index_data


def create_models_from_config(config: dict, model_names: list = None) -> list:
    """Create model instances from config"""
    models = []
    model_configs = config.get('models', {})
    
    # If specific models requested, use those
    if model_names:
        for name in model_names:
            if name in model_configs:
                model_config = model_configs[name]
                allocation = model_config.get('allocation', 0.10)
                models.append(get_model(name, allocation_pct=allocation))
            elif name in MODELS:
                models.append(get_model(name, allocation_pct=0.10))
    else:
        # Use all enabled models from config
        for name, model_config in model_configs.items():
            if model_config.get('enabled', True) and name in MODELS:
                allocation = model_config.get('allocation', 0.10)
                models.append(get_model(name, allocation_pct=allocation))
    
    return models


def main():
    parser = argparse.ArgumentParser(description='Run multi-model backtest')
    parser.add_argument('--models', nargs='+', help='Specific models to run')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--config', type=str, default='config/models_config.yaml', help='Config file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save', action='store_true', help='Save results to database')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load data
    stock_data, index_data = load_data(args.data_dir)
    
    if not stock_data:
        print("No data loaded. Exiting.")
        return
    
    # Create models
    models = create_models_from_config(config, args.models)
    
    if not models:
        print("No models configured. Exiting.")
        return
    
    # Run backtest
    backtest = MultiModelBacktest(
        initial_capital=args.capital,
        models=models,
        verbose=args.verbose
    )
    
    results = backtest.run(
        stock_data=stock_data,
        index_data=index_data,
        start_date=args.start,
        end_date=args.end
    )
    
    # Save to database if requested
    if args.save and 'trades' in results:
        db_path = config.get('database', {}).get('path', 'database/trading_dev.db')
        db = DatabaseManager(db_path)
        
        run_id = db.save_backtest_run(
            description=f"Backtest with {len(models)} models",
            initial_capital=args.capital,
            final_value=results['final_value'],
            total_pnl=results['total_pnl'],
            total_return_pct=results['total_return_pct'],
            cagr=results['cagr'],
            total_trades=results['total_trades'],
            winning_trades=results['winning_trades'],
            losing_trades=results['losing_trades'],
            win_rate=results['win_rate'],
            profit_factor=results['profit_factor'],
            avg_hold_days=results['avg_hold_days'],
            parameters=str({'models': [m.name for m in models]})
        )
        
        db.save_trades(run_id, results['trades'])
        print(f"\n✅ Results saved to database (Run ID: {run_id})")


if __name__ == "__main__":
    main()
