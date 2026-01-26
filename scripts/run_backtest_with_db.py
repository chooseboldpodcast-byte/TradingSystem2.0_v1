# scripts/run_backtest_with_db.py
"""
COMPLETE Weinstein backtest - VERSION 5 (CHRONOLOGICAL PROCESSING)
Processes ALL trades in time order (not symbol-by-symbol) for correct capital allocation

V5 CRITICAL FIX:
- Chronological processing of all trades across all symbols
- Proper concurrent position tracking
- Realistic capital allocation based on available capital at each point in time

V4 FEATURES (PRESERVED):
1. 1985+ data filter (fixes 62.8 year time frame bug) 
2. Relative Strength filtering (Weinstein's #1 rule)
3. Candlestick confirmation (high conviction entries)
4. Support level detection (better entry timing)
5. KEEPS ORIGINAL STAGE 4 EXITS (preserves profit factor)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.weinstein_engine import WeinsteinEngine, Stage
from database.db_manager import DatabaseManager

class SimpleBacktest:
    """
    Complete Weinstein backtest with chronological processing
    Processes ALL trades in time order across all symbols
    """
    
    def __init__(self, initial_capital: float = 100000, position_size_pct: float = 0.10):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.available_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.engine = WeinsteinEngine()
        self.open_positions = {}  # {symbol: {entry_date, entry_price, shares, stop_loss, method}}
        self.closed_trades = []
        self.equity_history = []
        self.rejection_stats = {
            'no_rs': 0,
            'weak_candle': 0,
            'no_volume': 0,
            'total_signals': 0
        }
    
    def run(self, stock_data: dict, index_data: pd.DataFrame):
        """
        Run backtest with CHRONOLOGICAL PROCESSING
        
        Args:
            stock_data: Dict of {symbol: DataFrame}
            index_data: S&P 500 data for context
        """
        print(f"Starting V5 Weinstein Backtest - CHRONOLOGICAL PROCESSING")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Position Size: {self.position_size_pct*100:.0f}% of available capital")
        print(f"Stocks: {len(stock_data)}")
        print(f"\nV5 CRITICAL FIX:")
        print(f"  ✓ Chronological processing (ALL symbols)")
        print(f"  ✓ Concurrent position tracking")
        print(f"  ✓ Real-time capital allocation")
        print(f"\nV4 QUALITY FILTERS:")
        print(f"  ✓ 1985+ data only (fixes time frame)")
        print(f"  ✓ Relative Strength > 0 (vs SPY)")
        print(f"  ✓ Bullish candlestick confirmation")
        print(f"  ✓ Volume confirmation (1.4x or 1.2x)")
        print(f"  ✓ Support level detection (bonus)")
        print(f"\nPhase 1: Analyzing stocks and generating signals...\n")
        
        # PHASE 1: Analyze all stocks and generate signal dataframes
        analyzed_data = {}
        for symbol, df in stock_data.items():
            print(f"Analyzing {symbol}...")
            df = self.engine.analyze_stock(df, index_data)
            analyzed_data[symbol] = df
        
        print(f"\nPhase 2: Generating chronological signal timeline...")
        
        # PHASE 2: Generate all signals across all stocks
        all_signals = self._generate_all_signals(analyzed_data)
        
        if len(all_signals) == 0:
            print("No signals generated!")
            return self._calculate_results()
        
        print(f"\nGenerated {len(all_signals)} total signals")
        print(f"Date range: {all_signals['date'].min().date()} to {all_signals['date'].max().date()}")
        
        # PHASE 3: Process signals chronologically
        print(f"\nPhase 3: Processing signals chronologically...")
        self._process_signals_chronologically(all_signals, analyzed_data)
        
        # Print rejection stats
        print(f"\n{'='*60}")
        print(f"QUALITY FILTER STATISTICS")
        print(f"{'='*60}")
        print(f"Total signals evaluated:  {self.rejection_stats['total_signals']}")
        print(f"Rejected - No RS:         {self.rejection_stats['no_rs']}")
        print(f"Rejected - Weak candle:   {self.rejection_stats['weak_candle']}")
        print(f"Rejected - Low volume:    {self.rejection_stats['no_volume']}")
        print(f"ACCEPTED:                 {len(self.closed_trades)}")
        acceptance_rate = (len(self.closed_trades) / self.rejection_stats['total_signals'] * 100) if self.rejection_stats['total_signals'] > 0 else 0
        print(f"Acceptance rate:          {acceptance_rate:.1f}%")
        
        # Calculate results
        return self._calculate_results()
    
    def _generate_all_signals(self, analyzed_data: dict) -> pd.DataFrame:
        """
        Generate ALL potential entry and exit signals across all stocks
        Returns a DataFrame sorted by date
        """
        signals = []
        
        for symbol, df in analyzed_data.items():
            # Generate signals for this stock
            stock_signals = self._generate_signals_for_stock(symbol, df)
            signals.extend(stock_signals)
        
        if len(signals) == 0:
            return pd.DataFrame()
        
        # Create DataFrame and sort by date
        signals_df = pd.DataFrame(signals)
        signals_df = signals_df.sort_values('date').reset_index(drop=True)
        
        return signals_df
    
    def _generate_signals_for_stock(self, symbol: str, df: pd.DataFrame) -> list:
        """
        Generate entry and exit signals for a single stock
        Returns list of signal dictionaries
        """
        signals = []
        
        for i in range(150, len(df)):
            current_date = df.index[i]
            current_stage = df['stage'].iloc[i]
            prev_stage = df['stage'].iloc[i-1] if i > 0 else 0
            current_price = df['close'].iloc[i]
            
            # ENTRY SIGNAL 1: Stage 1 -> Stage 2 Breakout
            if prev_stage == Stage.STAGE_1 and current_stage == Stage.STAGE_2:
                historical = df.iloc[:i+1]
                
                # Check for valid breakout
                if self.engine.detect_stage_2_breakout(historical, volume_threshold=1.4):
                    signals.append({
                        'date': current_date,
                        'symbol': symbol,
                        'type': 'ENTRY',
                        'method': 'BREAKOUT',
                        'price': current_price,
                        'data_idx': i
                    })
            
            # ENTRY SIGNAL 2: Stage 2 Pullback to MA
            elif current_stage == Stage.STAGE_2:
                ma = df['sma_30week'].iloc[i]
                ma_slope = df['ma_slope_pct'].iloc[i]
                volume_ratio = df['volume_ratio'].iloc[i]
                
                price_to_ma_pct = ((current_price - ma) / ma) * 100
                
                # Check for valid pullback entry conditions
                if (0 <= price_to_ma_pct <= 5 and
                    ma_slope > 0.1 and
                    volume_ratio >= 1.2 and
                    i >= 20):
                    
                    # Verify this is actually a pullback
                    lookback_period = min(20, i)
                    recent_high = df['high'].iloc[i-lookback_period:i].max()
                    
                    if recent_high > current_price * 1.03:
                        signals.append({
                            'date': current_date,
                            'symbol': symbol,
                            'type': 'ENTRY',
                            'method': 'PULLBACK',
                            'price': current_price,
                            'data_idx': i
                        })
            
            # EXIT SIGNAL: Stage 4 (original rule)
            if current_stage == Stage.STAGE_4:
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'type': 'EXIT',
                    'reason': 'STAGE_4',
                    'price': current_price,
                    'data_idx': i
                })
            
            # EXIT SIGNAL: Stop loss check (if we need historical data)
            # This will be handled in the chronological processing
        
        return signals
    
    def _process_signals_chronologically(self, signals_df: pd.DataFrame, analyzed_data: dict):
        """
        Process all signals in chronological order
        This is where capital allocation happens correctly!
        """
        total_signals = len(signals_df)
        
        for idx, signal in signals_df.iterrows():
            if idx % 100 == 0:
                print(f"  Processed {idx}/{total_signals} signals...")
            
            signal_date = signal['date']
            symbol = signal['symbol']
            signal_type = signal['type']
            price = signal['price']
            data_idx = signal['data_idx']
            
            df = analyzed_data[symbol]
            
            # Process EXIT signals first (frees up capital)
            if signal_type == 'EXIT':
                if symbol in self.open_positions:
                    self._close_position(symbol, signal_date, price, signal['reason'])
            
            # Then process ENTRY signals
            elif signal_type == 'ENTRY':
                # Skip if already in position for this symbol
                if symbol in self.open_positions:
                    continue
                
                # Apply quality filters
                self.rejection_stats['total_signals'] += 1
                
                # Quality Filter 1 - Relative Strength
                has_rs, rs_value = self.engine.has_positive_relative_strength(df, data_idx)
                if not has_rs:
                    self.rejection_stats['no_rs'] += 1
                    continue
                
                # Quality Filter 2 - Bullish Candle
                is_bullish, pattern = self.engine.detect_bullish_candle(df, data_idx)
                if not is_bullish:
                    self.rejection_stats['weak_candle'] += 1
                    continue
                
                # Quality Filter 3 - Support Detection (bonus)
                at_support, support_level = self.engine.detect_support_level(df, data_idx)
                
                # ALL FILTERS PASSED - Attempt to take the trade
                self._open_position(
                    symbol=symbol,
                    date=signal_date,
                    price=price,
                    df=df,
                    data_idx=data_idx,
                    method=signal['method'],
                    rs_value=rs_value,
                    pattern=pattern,
                    at_support=at_support,
                    support_level=support_level
                )
            
            # Check stop losses for all open positions
            self._check_stop_losses(signal_date, analyzed_data)
        
        print(f"  Completed processing {total_signals} signals")
        
        # Close any remaining open positions at final prices
        print(f"\nClosing {len(self.open_positions)} remaining positions...")
        for symbol in list(self.open_positions.keys()):
            df = analyzed_data[symbol]
            final_price = df['close'].iloc[-1]
            final_date = df.index[-1]
            self._close_position(symbol, final_date, final_price, 'END_OF_BACKTEST')
    
    def _open_position(self, symbol: str, date, price: float, df: pd.DataFrame, 
                      data_idx: int, method: str, rs_value: float, pattern: str,
                      at_support: bool, support_level: float):
        """
        Open a new position - sizes based on CURRENT available capital
        """
        # Calculate position size based on AVAILABLE capital (not total capital)
        position_value = self.available_capital * self.position_size_pct
        shares = int(position_value / price)
        
        if shares <= 0:
            return  # Not enough capital
        
        # Calculate actual position cost
        position_cost = shares * price
        
        # Update available capital
        self.available_capital -= position_cost
        
        # Calculate stop loss
        historical = df.iloc[:data_idx+1]
        stop_loss = self.engine.calculate_stop_loss(historical, price)
        
        # Record position
        self.open_positions[symbol] = {
            'entry_date': date,
            'entry_price': price,
            'shares': shares,
            'stop_loss': stop_loss,
            'method': method,
            'position_cost': position_cost
        }
        
        support_str = f", At Support=${support_level:.2f}" if at_support else ""
        print(f"  ✓ OPEN {symbol}: {shares} shares @ ${price:.2f} on {date.date()}")
        print(f"    Cost: ${position_cost:,.0f}, Available: ${self.available_capital:,.0f}, Stop: ${stop_loss:.2f}")
        print(f"    RS={rs_value:.1f}, Pattern={pattern}{support_str} [{method}]")
        print(f"    Open positions: {len(self.open_positions)}")
    
    def _close_position(self, symbol: str, date, price: float, reason: str):
        """
        Close a position and return capital (with P&L)
        """
        if symbol not in self.open_positions:
            return
        
        pos = self.open_positions[symbol]
        
        # Calculate P&L
        pnl = (price - pos['entry_price']) * pos['shares']
        pnl_pct = ((price / pos['entry_price']) - 1) * 100
        
        # Return capital to available pool
        exit_value = pos['shares'] * price
        self.available_capital += exit_value
        
        # Calculate hold time
        hold_days = (date - pos['entry_date']).days
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'shares': pos['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_days': hold_days,
            'exit_reason': reason,
            'entry_method': pos['method']
        }
        
        self.closed_trades.append(trade)
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        result_symbol = "✓" if pnl > 0 else "✗"
        print(f"  {result_symbol} CLOSE {symbol}: {pos['shares']} shares @ ${price:.2f} on {date.date()}")
        print(f"    P&L: ${pnl:,.0f} ({pnl_pct:+.1f}%), Hold: {hold_days} days, Reason: {reason}")
        print(f"    Available capital: ${self.available_capital:,.0f}")
    
    def _check_stop_losses(self, current_date, analyzed_data: dict):
        """
        Check if any open positions hit their stop loss
        """
        for symbol in list(self.open_positions.keys()):
            pos = self.open_positions[symbol]
            df = analyzed_data[symbol]
            
            # Get current price (find the row for current_date)
            try:
                current_price = df.loc[current_date, 'low']
                
                # Check if stop loss hit
                if current_price <= pos['stop_loss']:
                    self._close_position(symbol, current_date, pos['stop_loss'], 'STOP_LOSS')
            except KeyError:
                # Date not in this stock's dataframe
                continue
    
    def _calculate_results(self) -> dict:
        """Calculate backtest performance metrics"""
        
        if len(self.closed_trades) == 0:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'cagr': 0,
                'years': 0
            }
        
        trades_df = pd.DataFrame(self.closed_trades)
        
        # Basic metrics
        total_pnl = trades_df['pnl'].sum()
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        total_trades = len(trades_df)
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Average win/loss
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        
        # Time metrics
        first_date = trades_df['entry_date'].min()
        last_date = trades_df['exit_date'].max()
        years = (last_date - first_date).days / 365.25
        
        # Returns
        final_capital = self.initial_capital + total_pnl
        total_return_pct = (total_pnl / self.initial_capital) * 100
        cagr = (((final_capital / self.initial_capital) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        # Average hold time
        avg_hold_days = trades_df['hold_days'].mean()
        
        return {
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
            'final_value': final_capital
        }


def load_data(data_dir: str = 'data', min_year: int = 1985) -> dict:
    """
    Load all CSV files from data directory with date filtering
    
    Args:
        data_dir: Directory with CSV files
        min_year: Minimum year to include (default 1985)
    
    Returns:
        Dict of {symbol: DataFrame} with 1985+ data only
    """
    data = {}
    skipped = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            symbol = filename.replace('.csv', '')
            filepath = os.path.join(data_dir, filename)
            
            try:
                # Load CSV with date index
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # BULLETPROOF TIMEZONE FIX: Handles ALL cases
                if not isinstance(df.index, pd.DatetimeIndex):
                    # Convert to DatetimeIndex, handling timezone strings
                    df.index = pd.to_datetime(df.index.astype(str), utc=True)
                
                # Remove timezone if present (convert to naive datetime)
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # Filter to 1985+ using year
                df = df[df.index.year >= min_year]
                
                # Need at least 5 years of data (1260 trading days)
                if len(df) < 1260:
                    skipped.append(symbol)
                    continue
                
                data[symbol] = df
                first_date = df.index[0].date()
                last_date = df.index[-1].date()
                years = (last_date - first_date).days / 365.25
                print(f"  Loaded {symbol}: {len(df)} days ({first_date} to {last_date}, {years:.1f} years)")
                
            except Exception as e:
                print(f"  ERROR loading {symbol}: {e}")
                skipped.append(symbol)
                continue
    
    if skipped:
        print(f"\nSkipped {len(skipped)} symbols due to insufficient data or errors")
    
    return data


if __name__ == "__main__":
    print("="*60)
    print("WEINSTEIN COMPLETE SYSTEM - VERSION 5")
    print("="*60)
    print("V5 CRITICAL FIX:")
    print("  ✓ Chronological processing (not symbol-by-symbol)")
    print("  ✓ Concurrent position tracking across all symbols")
    print("  ✓ Real-time capital allocation")
    print("\nV4 FEATURES (PRESERVED):")
    print("  1. 1985+ data only (fixes 62.8 year time frame)")
    print("  2. Relative Strength filtering (Weinstein's #1 rule)")
    print("  3. Candlestick confirmation (high conviction)")
    print("  4. Support level detection (better entries)")
    print("  5. Original Stage 4 exits (preserves profit factor)")
    print("="*60)
    
    # Load data with 1985+ filter
    print("\nLoading data (1985+ only)...")
    all_data = load_data(data_dir='data', min_year=1985)
    
    # Separate stocks and index
    index_data = all_data.pop('^GSPC', None)
    if index_data is None:
        index_data = all_data.pop('GSPC', None)
    
    if index_data is None:
        print("Error: S&P 500 data (^GSPC or GSPC) not found!")
        sys.exit(1)
    
    # Remove benchmarks from stock data
    #all_data.pop('SPY', None)
    #all_data.pop('QQQ', None)
    #all_data.pop('TQQQ', None)
    
    print(f"\n✓ Loaded {len(all_data)} stocks for backtest")
    if len(all_data) > 0:
        first_dates = [df.index[0] for df in all_data.values()]
        last_dates = [df.index[-1] for df in all_data.values()]
        print(f"Date range: {min(first_dates).date()} to {max(last_dates).date()}")
        years = (max(last_dates) - min(first_dates)).days / 365.25
        print(f"Time span: {years:.1f} years")
    
    # Run backtest
    print("\n" + "="*60)
    backtest = SimpleBacktest(initial_capital=100000, position_size_pct=0.10)
    results = backtest.run(all_data, index_data)
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS (V5 - CHRONOLOGICAL)")
    print("="*60)
    
    if results['total_trades'] == 0:
        print("No trades executed!")
        print("This might mean filters are too strict. Check rejection stats above.")
    else:
        print(f"Initial Capital:     ${backtest.initial_capital:,.0f}")
        print(f"Final Value:         ${results['final_value']:,.0f}")
        print(f"Total P&L:           ${results['total_pnl']:,.0f}")
        print(f"Total Return:        {results['total_return_pct']:.2f}%")
        print(f"CAGR:                {results['cagr']:.2f}%")
        print(f"Time Period:         {results['years']:.1f} years")
        
        print(f"\nTrade Statistics:")
        print(f"Total Trades:        {results['total_trades']}")
        print(f"Trades per Year:     {results['total_trades']/results['years']:.1f}")
        print(f"Winning Trades:      {results['winning_trades']}")
        print(f"Losing Trades:       {results['losing_trades']}")
        print(f"Win Rate:            {results['win_rate']:.1f}%")
        print(f"\nAverage Win:         ${results['avg_win']:,.0f}")
        print(f"Average Loss:        ${results['avg_loss']:,.0f}")
        print(f"Profit Factor:       {results['profit_factor']:.2f}")
        print(f"Avg Hold Time:       {results['avg_hold_days']:.0f} days")
        
        # Compare to V4
        print("\n" + "="*60)
        print("V5 vs V4 COMPARISON")
        print("="*60)
        print("V5 fixes capital allocation by processing chronologically")
        print("Results should now reflect realistic concurrent position limits")
    
    # Save to database
    print("\n" + "="*60)
    print("SAVING TO DATABASE")
    print("="*60)
    
    if results['total_trades'] > 0:
        db = DatabaseManager()
        
        description = f"V5 Chronological - Fixed Capital Allocation - {len(all_data)} stocks"
        
        parameters = {
            'version': 'v5',
            'min_year': 1985,
            'volume_threshold': 1.2,
            'initial_capital': 100000,
            'position_size_pct': 10,
            'num_stocks': len(all_data),
            'processing': 'chronological',
            'filters': {
                'relative_strength': True,
                'candlestick_confirmation': True,
                'support_detection': True,
                'volume_confirmation': True
            }
        }
        
        # Prepare results with trades DataFrame
        results['trades'] = pd.DataFrame(backtest.closed_trades)
        
        run_id = db.save_backtest_run(results, description, parameters)
        
        print(f"\n✅ Backtest saved as Run #{run_id}")
        print(f"View results: streamlit run dashboard/app.py")
    else:
        print("\nNo trades to save.")
        print("Filters may be too strict - consider loosening requirements.")
