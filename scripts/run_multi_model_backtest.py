# scripts/run_4_model_backtest.py
"""
4-Model Backtest System - Adding Consolidation Breakout

Tests the multi-model system with:
1. Weinstein Core (40% allocation)
2. RSI Mean Reversion (10% allocation)
3. 52-Week High Momentum (15% allocation)
4. Consolidation Breakout (10% allocation) - NEW!

Total allocation: 75% of capital
Remaining 25% available for future models or cash buffer
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.weinstein_core import WeinsteinCore
from models.rsi_mean_reversion import RSIMeanReversion
from models.momentum_52w_high import Momentum52WeekHigh
from models.consolidation_breakout import ConsolidationBreakout
from core.portfolio_manager import PortfolioManager
from core.weinstein_engine import WeinsteinEngine
from database.db_manager import DatabaseManager

class MultiModelBacktest:
    """Multi-model backtest engine"""
    
    def __init__(
        self, 
        initial_capital: float = 100000,
        models: list = None
    ):
        self.initial_capital = initial_capital
        
        if models is None:
            models = [
                WeinsteinCore(allocation_pct=0.40),
                RSIMeanReversion(allocation_pct=0.10),
                Momentum52WeekHigh(allocation_pct=0.15),
                ConsolidationBreakout(allocation_pct=0.10)
            ]
        
        self.portfolio = PortfolioManager(
            initial_capital=initial_capital,
            models=models,
            verbose=False  # Suppress individual trade messages
        )
        
        self.engine = WeinsteinEngine()
        
        self.rejection_stats = {}
        for model in models:
            self.rejection_stats[model.name] = {
                'total_signals': 0,
                'accepted': 0,
                'rejected_no_capital': 0,
                'rejected_duplicate': 0,
                'rejected_allocation': 0
            }
    
    def run(self, stock_data: dict, index_data: pd.DataFrame):
        print(f"\n{'='*60}")
        print(f"4-MODEL BACKTEST - STARTING")
        print(f"{'='*60}")
        print(f"Stocks: {len(stock_data)}")
        print(f"Models: {len(self.portfolio.models)}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        
        # PHASE 1: Analyze stocks
        print(f"\nPhase 1: Analyzing stocks and calculating indicators...")
        analyzed_data = {}
        for symbol, df in stock_data.items():
            print(f"  Analyzing {symbol}...")
            analyzed = self.engine.analyze_stock(df, index_data)
            analyzed_data[symbol] = analyzed
        
        # PHASE 2: Generate signals
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
        
        # PHASE 3: Process signals chronologically
        print(f"\nPhase 3: Processing signals chronologically...")
        self._process_signals_chronologically(all_signals, analyzed_data)
        
        self._print_rejection_stats()
        
        return self._calculate_results()
    
    def _generate_all_signals(self, analyzed_data: dict, index_data: pd.DataFrame) -> pd.DataFrame:
        signals = []
        
        for symbol, df in analyzed_data.items():
            for model_name, model in self.portfolio.models.items():
                model_signals = self._generate_signals_for_stock_and_model(
                    symbol, df, model, index_data
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
        index_data: pd.DataFrame
    ) -> list:
        signals = []
        
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
            
            # Check for EXIT signal based on model-specific rules
            current_stage = df['stage'].iloc[i]
            
            # Model-specific exit triggers
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
        
        return signals
    
    def _process_signals_chronologically(
        self, 
        signals_df: pd.DataFrame,
        analyzed_data: dict
    ):
        total_signals = len(signals_df)
        processed = 0
        
        for idx, signal in signals_df.iterrows():
            if processed % 100 == 0 and processed > 0:
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
            
            # Check model-specific exits
            for pos_symbol in list(self.portfolio.open_positions.keys()):
                position = self.portfolio.open_positions[pos_symbol]
                pos_model = self.portfolio.models[position['model']]
                pos_df = analyzed_data[pos_symbol]
                
                try:
                    pos_idx = pos_df.index.get_loc(signal_date)
                    exit_signal = pos_model.generate_exit_signals(pos_df, pos_idx, position)
                    
                    if exit_signal.get('signal'):
                        self.portfolio.close_position(
                            symbol=pos_symbol,
                            exit_date=signal_date,
                            exit_price=exit_signal['exit_price'],
                            exit_reason=exit_signal['reason']
                        )
                except KeyError:
                    continue
            
            processed += 1
        
        print(f"    Completed processing {total_signals} signals")
        
        # Close remaining positions
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
        print(f"\n{'='*60}")
        print(f"SIGNAL ACCEPTANCE STATISTICS")
        print(f"{'='*60}")
        
        for model_name, stats in self.rejection_stats.items():
            print(f"\n{model_name}:")
            print(f"  Total Signals:       {stats['total_signals']}")
            print(f"  Accepted:            {stats['accepted']}")
            print(f"  Rejected - No Cap:   {stats['rejected_no_capital']}")
            print(f"  Rejected - Dup:      {stats['rejected_duplicate']}")
            print(f"  Rejected - Alloc:    {stats['rejected_allocation']}")
            
            if stats['total_signals'] > 0:
                acceptance_rate = (stats['accepted'] / stats['total_signals']) * 100
                print(f"  Acceptance Rate:     {acceptance_rate:.1f}%")
    
    def _calculate_results(self) -> dict:
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


def load_data(data_dir: str = 'data', min_year: int = 1985) -> dict:
    """Load stock data from CSV files"""
    data = {}
    skipped = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            symbol = filename.replace('.csv', '')
            filepath = os.path.join(data_dir, filename)
            
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index.astype(str), utc=True)
                
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                df = df[df.index.year >= min_year]
                
                if len(df) < 1260:
                    skipped.append(symbol)
                    continue
                
                data[symbol] = df
                print(f"  Loaded {symbol}: {len(df)} days")
                
            except Exception as e:
                print(f"  ERROR loading {symbol}: {e}")
                skipped.append(symbol)
    
    if skipped:
        print(f"\nSkipped {len(skipped)} symbols")
    
    return data


if __name__ == "__main__":
    print("="*60)
    print("4-MODEL BACKTEST SYSTEM")
    print("Testing: Weinstein + RSI + 52W High + Consolidation BO")
    print("="*60)
    
    # Load data
    print("\nLoading data (1985+)...")
    all_data = load_data(data_dir='data', min_year=1985)
    
    # Separate index
    index_data = all_data.pop('^GSPC', None)
    if index_data is None:
        index_data = all_data.pop('GSPC', None)
    
    if index_data is None:
        print("❌ Error: S&P 500 data not found!")
        sys.exit(1)
    
    print(f"\n✅ Loaded {len(all_data)} stocks for backtest")
    
    # Initialize models
    models = [
        WeinsteinCore(allocation_pct=0.40),
        RSIMeanReversion(allocation_pct=0.10),
        Momentum52WeekHigh(allocation_pct=0.15),
        ConsolidationBreakout(allocation_pct=0.10)
    ]
    
    # Run backtest
    backtest = MultiModelBacktest(
        initial_capital=100000,
        models=models
    )
    
    results = backtest.run(all_data, index_data)
    
    # Print results
    if results['total_trades'] > 0:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - 4-MODEL SYSTEM")
        print(f"{'='*60}")
        print(f"CAGR:                {results['cagr']:.2f}%")
        print(f"Total Trades:        {results['total_trades']}")
        print(f"Win Rate:            {results['win_rate']:.1f}%")
        print(f"Profit Factor:       {results['profit_factor']:.2f}")
        print(f"Avg Hold Days:       {results['avg_hold_days']:.0f}")
        
        # Model breakdown
        print(f"\n{'='*60}")
        print(f"PERFORMANCE BY MODEL")
        print(f"{'='*60}")
        
        trades_df = results['trades']
        for model_name in ['Weinstein_Core', 'RSI_Mean_Reversion', '52W_High_Momentum', 'Consolidation_Breakout']:
            model_trades = trades_df[trades_df['model'] == model_name]
            
            if len(model_trades) > 0:
                model_pnl = model_trades['pnl'].sum()
                model_wins = len(model_trades[model_trades['pnl'] > 0])
                model_total = len(model_trades)
                model_win_rate = (model_wins / model_total * 100) if model_total > 0 else 0
                model_avg_hold = model_trades['hold_days'].mean()
                
                print(f"\n{model_name}:")
                print(f"  Trades:       {model_total}")
                print(f"  Total P&L:    ${model_pnl:,.0f}")
                print(f"  Win Rate:     {model_win_rate:.1f}%")
                print(f"  Avg Hold:     {model_avg_hold:.0f} days")
        
        # Save to database
        print(f"\n{'='*60}")
        print(f"SAVING TO DATABASE")
        print(f"{'='*60}")
        
        db = DatabaseManager()
        description = f"4-Model System (W+RSI+52W+Consol) - {len(all_data)} stocks"
        
        run_id = db.save_backtest_run(results, description)
        print(f"\n✅ Saved as Run #{run_id}")
        print(f"View: streamlit run dashboard/app.py")
