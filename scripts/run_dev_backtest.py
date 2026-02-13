#!/usr/bin/env python3
"""
Development Backtest Runner - CORRECTED VERSION
================================================

Runs backtests using DEVELOPMENT models from models_dev/ directory.
This version matches the proven logic from Env1's run_multi_model_backtest.py.

FEATURES:
1. ✅ Starting index: range(150, len(df))
2. ✅ Manual stage-based exit signals (Weinstein/Momentum/Consolidation)
3. ✅ Calls model.generate_exit_signals() during processing
4. ✅ Timezone handling (UTC conversion + localize None)
5. ✅ Minimum data length: 1260 days (5 years)
6. ✅ Process EXIT before ENTRY
7. ✅ min_year filtering from config (default 1985)
8. ✅ Dynamic loading of dev models from models_dev/

Usage:
    python scripts/run_dev_backtest.py
    python scripts/run_dev_backtest.py --start-year 2015
    python scripts/run_dev_backtest.py --description "Test RSI v2"
"""

import os
import sys
import argparse
from datetime import datetime
import importlib.util

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add models_dev directory to path for importing development models
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models_dev'))

import pandas as pd
import numpy as np
import yaml

# Import DEVELOPMENT models from models_dev/ if they exist, else fall back to models/
try:
    from models_dev.weinstein_core import WeinsteinCore
    print("✅ Using DEV version: weinstein_core")
except ImportError:
    from models.weinstein_core import WeinsteinCore

try:
    from models_dev.rsi_mean_reversion import RSIMeanReversion
    print("✅ Using DEV version: rsi_mean_reversion")
except ImportError:
    from models.rsi_mean_reversion import RSIMeanReversion

try:
    from models_dev.momentum_52w_high import Momentum52WeekHigh
    print("✅ Using DEV version: momentum_52w_high")
except ImportError:
    from models.momentum_52w_high import Momentum52WeekHigh

try:
    from models_dev.consolidation_breakout import ConsolidationBreakout
    print("✅ Using DEV version: consolidation_breakout")
except ImportError:
    from models.consolidation_breakout import ConsolidationBreakout

# New models - only from models/ (no dev versions yet)
from models.vcp import VCP
from models.pocket_pivot import PocketPivot
from models.rs_breakout import RSBreakout
from models.high_tight_flag import HighTightFlag

from core.portfolio_manager import PortfolioManager
from core.weinstein_engine import WeinsteinEngine
from database.db_manager import DatabaseManager

# Model name mapping
MODEL_CLASSES = {
    'weinstein_core': WeinsteinCore,
    'rsi_mean_reversion': RSIMeanReversion,
    'momentum_52w_high': Momentum52WeekHigh,
    'consolidation_breakout': ConsolidationBreakout,
    'vcp': VCP,
    'pocket_pivot': PocketPivot,
    'rs_breakout': RSBreakout,
    'high_tight_flag': HighTightFlag
}


def load_config(config_path: str = 'config/models_config.yaml') -> dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_models(config: dict) -> list:
    """Create model instances based on config"""
    models = []
    model_names = config.get('models', [])
    allocations = config.get('allocations', {})
    
    for name in model_names:
        if name in MODEL_CLASSES:
            allocation = allocations.get(name, 0.10)
            models.append(MODEL_CLASSES[name](allocation_pct=allocation))
    
    return models


# ============================================================================
# DATA LOADING - EXACT COPY OF ENV1 WORKING LOGIC
# ============================================================================

def load_stock_data(data_dir: str = 'data', min_year: int = 1985, universe: list = None):
    """
    Load stock data from CSV files
    EXACT COPY from Env1 run_multi_model_backtest.py with universe filter
    
    Args:
        data_dir: Directory containing CSV files
        min_year: Minimum year for data (filters out older data)
        universe: Optional list of symbols to load (None = load all)
    
    Returns:
        dict of symbol -> DataFrame
    """
    data = {}
    skipped = []
    
    print(f"\n{'='*60}")
    print(f"LOADING STOCK DATA")
    print(f"{'='*60}")
    print(f"Min year: {min_year}")
    print(f"Universe: {len(universe) if universe else 'ALL'} stocks")
    
    # Get list of files to process
    if universe:
        files_to_load = [f"{symbol}.csv" for symbol in universe]
    else:
        files_to_load = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for filename in files_to_load:
        if not filename.endswith('.csv'):
            continue
            
        symbol = filename.replace('.csv', '')
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            skipped.append(symbol)
            continue
        
        try:
            # FIX #1: Proper timezone handling (EXACT COPY from Env1)
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index.astype(str), utc=True)
            
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_convert(None)  # tz_convert for tz-aware data
            
            # FIX #2: Filter by min_year (EXACT COPY from Env1)
            df = df[df.index.year >= min_year]
            
            # FIX #3: Minimum 1260 days (5 years) - EXACT COPY from Env1
            if len(df) < 1260:
                skipped.append(symbol)
                continue
            
            data[symbol] = df
            print(f"  ✅ {symbol}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
            
        except Exception as e:
            print(f"  ❌ {symbol}: Error loading - {e}")
            skipped.append(symbol)
    
    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} symbols (insufficient data or not found)")
    
    print(f"\n✅ Loaded {len(data)} stocks")
    return data


def get_index_data(stock_data: dict):
    """
    Extract index data from loaded stocks
    Looks for ^GSPC, GSPC, or SPY
    """
    for symbol in ['^GSPC', 'GSPC', 'SPY']:
        if symbol in stock_data:
            index_data = stock_data.pop(symbol)
            print(f"✅ Using {symbol} as market index ({len(index_data)} days)")
            return index_data
    
    print("⚠️  No market index found (^GSPC, GSPC, or SPY)")
    return None


# ============================================================================
# DEVELOPMENT BACKTEST ENGINE - EXACT COPY OF ENV1 WORKING LOGIC
# ============================================================================

class DevelopmentBacktest:
    """Backtest engine using DEVELOPMENT models from models_dev/"""
    
    def __init__(self, models: list, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.models = models
        
        print(f"\n{'='*60}")
        print(f"DEVELOPMENT MODELS LOADED FROM models_dev/")
        print(f"{'='*60}")
        for model in self.models:
            print(f"✅ {model.name} ({model.allocation_pct*100:.0f}% allocation)")
        print(f"Total allocation: {sum(m.allocation_pct for m in self.models)*100:.0f}%")
        
        self.portfolio = PortfolioManager(
            initial_capital=initial_capital,
            models=self.models,
            verbose=False
        )
        
        self.engine = WeinsteinEngine()
        
        self.rejection_stats = {}
        for model in self.models:
            self.rejection_stats[model.name] = {
                'total_signals': 0,
                'accepted': 0,
                'rejected_no_capital': 0,
                'rejected_duplicate': 0,
                'rejected_allocation': 0
            }
    
    def run(self, stock_data: dict, index_data: pd.DataFrame):
        """Run backtest with development models - EXACT COPY OF ENV1 LOGIC"""
        
        print(f"\n{'='*60}")
        print(f"DEVELOPMENT BACKTEST - STARTING")
        print(f"{'='*60}")
        print(f"Source: models_dev/ (DEVELOPMENT)")
        print(f"Stocks: {len(stock_data)}")
        print(f"Models: {len(self.portfolio.models)}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        
        # PHASE 1: Analyze stocks (EXACT COPY from Env1)
        print(f"\nPhase 1: Analyzing stocks and calculating indicators...")
        analyzed_data = {}
        for symbol, df in stock_data.items():
            try:
                analyzed = self.engine.analyze_stock(df, index_data)
                analyzed_data[symbol] = analyzed
            except Exception as e:
                print(f"  ❌ Error analyzing {symbol}: {e}")
        
        print(f"✅ Analyzed {len(analyzed_data)} stocks")
        
        # PHASE 2: Generate signals (EXACT COPY from Env1)
        print(f"\nPhase 2: Generating signals from all models...")
        all_signals = self._generate_all_signals(analyzed_data, index_data)
        
        if len(all_signals) == 0:
            print("❌ No signals generated!")
            return self._calculate_results()
        
        print(f"\n✅ Generated {len(all_signals)} total signals")
        print(f"  Date range: {all_signals['date'].min().date()} to {all_signals['date'].max().date()}")
        
        for model_name in self.portfolio.models.keys():
            model_signals = all_signals[all_signals['model'] == model_name]
            entry_signals = model_signals[model_signals['type'] == 'ENTRY']
            print(f"  {model_name}: {len(entry_signals)} entry signals")
        
        # PHASE 3: Process signals chronologically (EXACT COPY from Env1)
        print(f"\nPhase 3: Processing signals chronologically...")
        self._process_signals_chronologically(all_signals, analyzed_data)
        
        self._print_rejection_stats()
        
        # PHASE 4: Calculate results
        results = self._calculate_results()
        
        return results
    
    def _calculate_model_priorities(self) -> dict:
        """
        Calculate processing priority for each model.
        
        Priority rules:
        1. Higher allocation % = higher priority (lower number = processed first)
        2. Same allocation % = YAML config order (earlier in list = processed first)
        
        Returns:
            dict of model_name -> priority (lower = higher priority)
        """
        # Get models with their allocation and original order
        models_info = []
        for idx, (name, model) in enumerate(self.portfolio.models.items()):
            models_info.append({
                'name': name,
                'allocation': model.allocation_pct,
                'yaml_order': idx
            })
        
        # Sort by allocation (descending), then yaml_order (ascending)
        models_info.sort(key=lambda x: (-x['allocation'], x['yaml_order']))
        
        # Assign priority based on sorted position
        priorities = {}
        for priority, info in enumerate(models_info):
            priorities[info['name']] = priority
        
        return priorities
    
    def _generate_all_signals(self, analyzed_data: dict, index_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals from all models with deterministic sorting.
        
        Signals are sorted by (date, model_priority, symbol) to ensure:
        - Reproducible results regardless of data loading order
        - Higher allocation models get priority for same-day signals
        - Alphabetical symbol order as final tiebreaker
        """
        signals = []
        
        # Calculate model priorities once
        model_priorities = self._calculate_model_priorities()
        
        for symbol, df in analyzed_data.items():
            for model_name, model in self.portfolio.models.items():
                model_signals = self._generate_signals_for_stock_and_model(
                    symbol, df, model, index_data
                )
                # Add model_priority to each signal
                for sig in model_signals:
                    sig['model_priority'] = model_priorities[model_name]
                signals.extend(model_signals)
        
        if len(signals) == 0:
            return pd.DataFrame()
        
        signals_df = pd.DataFrame(signals)
        
        # DETERMINISTIC SORT: date, model_priority, symbol
        # This ensures reproducible results regardless of data loading order
        signals_df = signals_df.sort_values(
            ['date', 'model_priority', 'symbol']
        ).reset_index(drop=True)
        
        return signals_df
    
    def _generate_signals_for_stock_and_model(
        self, 
        symbol: str, 
        df: pd.DataFrame,
        model,
        index_data: pd.DataFrame
    ) -> list:
        """
        Generate signals for one stock and one model
        EXACT COPY OF WORKING LOGIC from Env1 run_multi_model_backtest.py
        """
        signals = []
        
        # FIX #1: Start at 150, not 200!
        for i in range(150, len(df)):
            current_date = df.index[i]
            
            # Get market data at same index if available
            try:
                market_idx = index_data.index.get_loc(current_date)
                market_df = index_data
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
            
            # ================================================================
            # MANUAL STAGE-BASED EXIT CHECKS - EXACT COPY FROM ENV1
            # NOTE: These are critical for matching Env1 results
            # ================================================================
            current_stage = df['stage'].iloc[i]
            
            # Model-specific exit triggers (EXACT COPY from working backtest)
            if model.name == 'Weinstein_Core' and current_stage == 4:
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'model': model.name,
                    'type': 'EXIT',
                    'data_idx': i,
                    'signal_data': {'reason': 'STAGE_4'}
                })
            elif model.name == '52W_High_Momentum' and current_stage == 3:
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'model': model.name,
                    'type': 'EXIT',
                    'data_idx': i,
                    'signal_data': {'reason': 'STAGE_3'}
                })
            elif model.name == 'Consolidation_Breakout' and current_stage == 3:
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'model': model.name,
                    'type': 'EXIT',
                    'data_idx': i,
                    'signal_data': {'reason': 'STAGE_3'}
                })
            # New models - add stage-based exits as needed
            elif model.name == 'VCP' and current_stage == 3:
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'model': model.name,
                    'type': 'EXIT',
                    'data_idx': i,
                    'signal_data': {'reason': 'STAGE_3'}
                })
            elif model.name == 'Pocket_Pivot' and current_stage == 3:
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'model': model.name,
                    'type': 'EXIT',
                    'data_idx': i,
                    'signal_data': {'reason': 'STAGE_3'}
                })
            elif model.name == 'RS_Breakout' and current_stage == 3:
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'model': model.name,
                    'type': 'EXIT',
                    'data_idx': i,
                    'signal_data': {'reason': 'STAGE_3'}
                })
            elif model.name == 'High_Tight_Flag' and current_stage == 3:
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
        """
        Process signals in time order
        EXACT COPY OF WORKING LOGIC from Env1 run_multi_model_backtest.py
        """
        total_signals = len(signals_df)
        processed = 0
        
        for idx, signal in signals_df.iterrows():
            processed += 1
            if processed % 1000 == 0:
                print(f"    Processed {processed}/{total_signals} signals...")
            
            signal_date = signal['date']
            symbol = signal['symbol']
            model_name = signal['model']
            signal_type = signal['type']
            signal_data = signal['signal_data']
            data_idx = signal['data_idx']
            
            df = analyzed_data[symbol]
            model = self.portfolio.models[model_name]
            
            # ================================================================
            # FIX #2: Process EXIT signals FIRST (before ENTRY) - EXACT COPY
            # ================================================================
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
            
            # Check model-specific exits (calls model.generate_exit_signals)
            # NOTE: This is redundant with manual checks above but matches working backtest
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
                except:
                    continue
        
        # Close remaining positions at end
        print(f"\nClosing {len(self.portfolio.open_positions)} remaining positions...")
        for symbol in list(self.portfolio.open_positions.keys()):
            if symbol in analyzed_data:
                df = analyzed_data[symbol]
                self.portfolio.close_position(
                    symbol=symbol,
                    exit_date=df.index[-1],
                    exit_price=df['close'].iloc[-1],
                    exit_reason='END_OF_BACKTEST'
                )
    
    def _print_rejection_stats(self):
        """Print signal acceptance/rejection statistics"""
        print(f"\n{'='*60}")
        print("SIGNAL ACCEPTANCE STATISTICS")
        print(f"{'='*60}")
        
        for name, stats in self.rejection_stats.items():
            if stats['total_signals'] > 0:
                rate = (stats['accepted'] / stats['total_signals']) * 100
                print(f"\n{name}:")
                print(f"  Total Signals: {stats['total_signals']}")
                print(f"  Accepted:      {stats['accepted']} ({rate:.1f}%)")
                print(f"  Rejected - No Capital:  {stats['rejected_no_capital']}")
                print(f"  Rejected - Duplicate:   {stats['rejected_duplicate']}")
                print(f"  Rejected - Allocation:  {stats['rejected_allocation']}")
    
    def _calculate_results(self) -> dict:
        """Calculate final results - EXACT COPY from Env1"""
        trades_df = self.portfolio.get_trades_dataframe()
        
        if len(trades_df) == 0:
            print("\n⚠️  No trades executed!")
            return {
                'trades': pd.DataFrame(),
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'years': 0,
                'cagr': 0,
                'avg_hold_days': 0,
                'final_value': self.initial_capital,
                'initial_capital': self.initial_capital
            }
        
        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 0
        
        avg_hold_days = trades_df['hold_days'].mean() if 'hold_days' in trades_df.columns else 0
        
        # Calculate CAGR
        first_trade_date = trades_df['entry_date'].min()
        last_trade_date = trades_df['exit_date'].max()
        years = (last_trade_date - first_trade_date).days / 365.25
        
        final_capital = self.initial_capital + total_pnl
        cagr = (((final_capital / self.initial_capital) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        # Print summary - EXACT FORMAT from Env1
        print(f"\n{'='*60}")
        print(f"DEV BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Return:   {total_return_pct:>8.1f}%")
        print(f"CAGR:           {cagr:>8.1f}%")
        print(f"Total Trades:   {total_trades:>8,}")
        print(f"Win Rate:       {win_rate:>8.1f}%")
        print(f"Profit Factor:  {profit_factor:>8.2f}")
        print(f"Avg Win:        ${avg_win:>8,.0f}")
        print(f"Avg Loss:       ${avg_loss:>8,.0f}")
        print(f"Final Equity:   ${final_capital:>8,.0f}")
        print(f"{'='*60}")
        
        # Breakdown by model - EXACT COPY from Env1
        print(f"\nBREAKDOWN BY MODEL:")
        for model_name in self.portfolio.models.keys():
            model_trades = trades_df[trades_df['model'] == model_name]
            if len(model_trades) > 0:
                model_win_rate = len(model_trades[model_trades['pnl'] > 0]) / len(model_trades) * 100
                model_total_pnl = model_trades['pnl'].sum()
                print(f"  {model_name}:")
                print(f"    Trades: {len(model_trades)}, Win Rate: {model_win_rate:.1f}%, P&L: ${model_total_pnl:,.0f}")
        
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run development model backtest')
    parser.add_argument('--start-year', type=int, default=1985, help='Start year for backtest data')
    parser.add_argument('--capital', type=int, default=100000, help='Initial capital')
    parser.add_argument('--description', type=str, default='', help='Run description')
    parser.add_argument('--no-save', action='store_true', help='Do not save to database')

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Get min_year from config or use argument
    min_year = config.get('portfolio', {}).get('data_start_year', args.start_year)
    if args.start_year != 1985:  # User specified a different year
        min_year = args.start_year

    # Default description
    if not args.description:
        args.description = f"Development - {datetime.now().strftime('%Y-%m-%d')}"

    print("\n" + "="*80)
    print(f"DEVELOPMENT BACKTEST")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Description: {args.description}")
    print(f"Min Year: {min_year}")
    print(f"Source: models_dev/ (DEVELOPMENT)")
    print("="*80)

    # Load universe from core universe file
    core_path = config.get('paths', {}).get('core_universe', 'live_universe.txt')
    with open(core_path, 'r') as f:
        universe = [line.strip() for line in f if line.strip()]
    # Add index symbols
    universe.extend(['^GSPC', 'GSPC', 'SPY'])
    universe = list(set(universe))  # Remove duplicates
    
    # Load data
    data_dir = config.get('paths', {}).get('data_dir', 'data')
    all_data = load_stock_data(data_dir=data_dir, min_year=min_year, universe=universe)
    
    # Separate index data
    index_data = get_index_data(all_data)
    
    if index_data is None:
        print("❌ Error: No market index data found!")
        return 1
    
    if len(all_data) == 0:
        print("❌ No stock data loaded!")
        return 1
    
    print(f"\n✅ Ready to backtest {len(all_data)} stocks")
    
    # Create models
    models = create_models(config)
    print(f"Models: {[m.name for m in models]}")
    
    # Run backtest
    backtest = DevelopmentBacktest(models, args.capital)
    results = backtest.run(all_data, index_data)
    
    # Save to database
    if not args.no_save and results.get('total_trades', 0) > 0:
        db_path = config.get('database', {}).get('path', 'database/weinstein.db')
        db = DatabaseManager(db_path)
        
        run_id = db.save_backtest_run(
            results=results,
            run_type='development',
            description=args.description,
            parameters={
                'models': [m.name for m in models],
                'universe_size': len(all_data),
                'min_year': min_year
            }
        )
        print(f"\n✅ Saved as Run #{run_id}")
        print(f"   Database: {db_path}")

    print(f"\n{'='*80}")
    print("DEVELOPMENT BACKTEST COMPLETE")
    print(f"{'='*80}")
    if not args.no_save and results.get('total_trades', 0) > 0:
        print(f"\n✅ Saved as Run #{run_id}")
        print(f"   Database: {db_path}")
    print("✅ Results saved with run_type='development'")
    print("✅ View in dashboard")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
