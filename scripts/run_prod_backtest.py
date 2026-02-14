#!/usr/bin/env python3
"""
Production Backtest Runner - CORRECTED VERSION
==============================================

Runs backtests using PRODUCTION models from models/ directory.
This version matches the proven logic from Env1's run_multi_model_backtest.py.

FEATURES:
1. ✅ Starting index: range(150, len(df))
2. ✅ Manual stage-based exit signals (Weinstein/Momentum/Consolidation)
3. ✅ Calls model.generate_exit_signals() during processing
4. ✅ Timezone handling (UTC conversion + localize None)
5. ✅ Minimum data length: 1260 days (5 years)
6. ✅ Process EXIT before ENTRY
7. ✅ min_year filtering from config (default 1985)

Usage:
    python scripts/run_prod_backtest.py
    python scripts/run_prod_backtest.py --start-year 2015
    python scripts/run_prod_backtest.py --description "Full test"
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yaml

# Direct file imports (no __init__.py needed)
from models.weinstein_core import WeinsteinCore
from models.rsi_mean_reversion import RSIMeanReversion
from models.momentum_52w_high import Momentum52WeekHigh
from models.consolidation_breakout import ConsolidationBreakout
from models.vcp import VCP
from models.pocket_pivot import PocketPivot
from models.rs_breakout import RSBreakout
from models.high_tight_flag import HighTightFlag
# Phase 4: New models
from models.enhanced_mean_reversion import EnhancedMeanReversion

from core.portfolio_manager import PortfolioManager
from core.weinstein_engine import WeinsteinEngine
from core.backtest_logger import BacktestLogger
from core.market_context import MarketContextAnalyzer
from core.market_regime import MarketRegime
from database.db_manager import DatabaseManager
from scanner.universe_scanner import UniverseScanner

# Sector ETFs for rotation tracking
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

# Model name mapping
MODEL_CLASSES = {
    'weinstein_core': WeinsteinCore,
    'rsi_mean_reversion': RSIMeanReversion,
    'enhanced_mean_reversion': EnhancedMeanReversion,
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


def load_sector_data(data_dir: str = 'data', min_year: int = 1985) -> dict:
    """
    Load sector ETF data for rotation tracking.

    Args:
        data_dir: Directory containing CSV files
        min_year: Minimum year for data

    Returns:
        Dict of {sector_symbol: DataFrame}
    """
    sector_data = {}

    print(f"\n{'='*60}")
    print(f"LOADING SECTOR ETF DATA")
    print(f"{'='*60}")

    for symbol in SECTOR_ETFS:
        filepath = os.path.join(data_dir, f"{symbol}.csv")

        if not os.path.exists(filepath):
            print(f"  ⚠️  {symbol}: Not found")
            continue

        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index.astype(str), utc=True)

            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            # Filter by min_year
            df = df[df.index.year >= min_year]

            if len(df) >= 252:  # At least 1 year of data
                sector_data[symbol] = df
                print(f"  ✅ {symbol}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
            else:
                print(f"  ⚠️  {symbol}: Insufficient data ({len(df)} days)")

        except Exception as e:
            print(f"  ❌ {symbol}: Error - {e}")

    print(f"\n✅ Loaded {len(sector_data)} sector ETFs")
    return sector_data


# ============================================================================
# PRODUCTION BACKTEST ENGINE - EXACT COPY OF ENV1 WORKING LOGIC
# ============================================================================

class ProductionBacktest:
    """Backtest engine using PRODUCTION models from models/"""

    def __init__(self, models: list, initial_capital: float = 100000, logger: BacktestLogger = None,
                 market_context: MarketContextAnalyzer = None, sector_data: dict = None,
                 enable_regime_filter: bool = False, crisis_only: bool = False):
        self.initial_capital = initial_capital
        self.models = models
        self.logger = logger
        self.market_context = market_context
        self.sector_data = sector_data or {}
        self.enable_regime_filter = enable_regime_filter
        self.crisis_only = crisis_only

        print(f"\n{'='*60}")
        print(f"PRODUCTION MODELS LOADED FROM models/")
        print(f"{'='*60}")
        for model in self.models:
            print(f"✅ {model.name} ({model.allocation_pct*100:.0f}% allocation)")
        print(f"Total allocation: {sum(m.allocation_pct for m in self.models)*100:.0f}%")

        if self.enable_regime_filter and self.market_context:
            if self.crisis_only:
                print(f"\n🛡️  MARKET REGIME FILTER: CRISIS-ONLY (VIX > 35)")
            else:
                print(f"\n🛡️  MARKET REGIME FILTER: FULL (BEAR/CAUTION/CRISIS)")
            print(f"   Sector ETFs loaded: {len(self.sector_data)}")
        else:
            print(f"\n⚠️  MARKET REGIME FILTER: DISABLED (baseline mode)")

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
                'rejected_allocation': 0,
                'rejected_regime': 0  # NEW: Track regime rejections
            }

        # Track regime statistics
        self.regime_stats = {
            'regime_changes': [],
            'trades_by_regime': {r.value: 0 for r in MarketRegime},
            'pnl_by_regime': {r.value: 0.0 for r in MarketRegime},
            'rejected_by_regime': {r.value: 0 for r in MarketRegime}
        }
        self.current_regime = None
        self.current_context = None
    
    def run(self, stock_data: dict, index_data: pd.DataFrame):
        """Run backtest with production models - EXACT COPY OF ENV1 LOGIC"""
        
        print(f"\n{'='*60}")
        print(f"PRODUCTION BACKTEST - STARTING")
        print(f"{'='*60}")
        print(f"Source: models/ (PRODUCTION)")
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
        self._process_signals_chronologically(all_signals, analyzed_data, index_data)
        
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
        
        #signals_df.to_csv('actual_backtest_signals.csv', index=False)
        #print(f"Exported {len(signals_df)} signals to actual_backtest_signals.csv")
        
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

                # Log signal generation
                if self.logger:
                    self.logger.log_signal_generated(
                        date=current_date,
                        symbol=symbol,
                        model=model.name,
                        signal_type='ENTRY',
                        entry_price=entry_signal.get('entry_price'),
                        stop_loss=entry_signal.get('stop_loss'),
                        confidence=entry_signal.get('confidence', 1.0),
                        metadata=entry_signal.get('metadata', {})
                    )
            
            # ================================================================
            # MANUAL STAGE-BASED EXIT CHECKS - EXACT COPY FROM ENV1
            # NOTE: These are critical for matching Env1 results
            # ================================================================
            current_stage = df['stage'].iloc[i]
            
            # Model-specific exit triggers (EXACT COPY from working backtest)
            exit_signal_added = False
            exit_reason = None

            if model.name == 'Weinstein_Core' and current_stage == 4:
                exit_reason = 'STAGE_4'
                exit_signal_added = True
            elif model.name == '52W_High_Momentum' and current_stage == 3:
                exit_reason = 'STAGE_3'
                exit_signal_added = True
            elif model.name == 'Consolidation_Breakout' and current_stage == 3:
                exit_reason = 'STAGE_3'
                exit_signal_added = True
            elif model.name == 'VCP' and current_stage == 3:
                exit_reason = 'STAGE_3'
                exit_signal_added = True
            elif model.name == 'Pocket_Pivot' and current_stage == 3:
                exit_reason = 'STAGE_3'
                exit_signal_added = True
            elif model.name == 'RS_Breakout' and current_stage == 3:
                exit_reason = 'STAGE_3'
                exit_signal_added = True
            elif model.name == 'High_Tight_Flag' and current_stage == 3:
                exit_reason = 'STAGE_3'
                exit_signal_added = True
            # Phase 4 new models
            elif model.name == 'Enhanced_Mean_Reversion' and current_stage in [3, 4]:
                exit_reason = f'STAGE_{current_stage}'
                exit_signal_added = True

            if exit_signal_added:
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'model': model.name,
                    'type': 'EXIT',
                    'data_idx': i,
                    'signal_data': {'reason': exit_reason}
                })

                # Log exit signal generation
                if self.logger:
                    self.logger.log_signal_generated(
                        date=current_date,
                        symbol=symbol,
                        model=model.name,
                        signal_type='EXIT',
                        reason=exit_reason
                    )

        return signals
    
    def _process_signals_chronologically(
        self,
        signals_df: pd.DataFrame,
        analyzed_data: dict,
        index_data: pd.DataFrame = None
    ):
        """
        Process signals in time order
        EXACT COPY OF WORKING LOGIC from Env1 run_multi_model_backtest.py
        Enhanced with Phase 0 market regime filter support.
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
            # POSITION LAYERING: Multiple models can hold same stock
            # ================================================================
            if signal_type == 'EXIT':
                # Check if THIS model has a position on this symbol
                pos_key = self.portfolio.make_position_key(symbol, model_name)
                if pos_key in self.portfolio.open_positions:
                    position = self.portfolio.open_positions[pos_key]

                    exit_price = df['close'].iloc[data_idx]
                    exit_reason = signal_data.get('reason', 'MODEL_EXIT')

                    # Calculate P&L before closing
                    pnl = (exit_price - position['entry_price']) * position['shares']
                    pnl_pct = ((exit_price / position['entry_price']) - 1) * 100
                    hold_days = (signal_date - position['entry_date']).days

                    self.portfolio.close_position(
                        symbol=symbol,
                        exit_date=signal_date,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        model_name=model_name
                    )

                    # Log position closed
                    if self.logger:
                        self.logger.log_position_closed(
                            date=signal_date,
                            symbol=symbol,
                            model=model_name,
                            shares=position['shares'],
                            price=exit_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            hold_days=hold_days,
                            exit_reason=exit_reason
                        )

            # Process ENTRY signals
            elif signal_type == 'ENTRY':
                self.rejection_stats[model_name]['total_signals'] += 1

                # ================================================================
                # PHASE 0: MARKET REGIME FILTER
                # Check if market conditions allow this trade
                # ================================================================
                if self.enable_regime_filter and self.market_context:
                    # Get market context for this date (use cached if same date)
                    if not hasattr(self, '_last_context_date') or self._last_context_date != signal_date:
                        try:
                            # Filter data up to current date for backtesting
                            spy_to_date = index_data[index_data.index <= signal_date]
                            sector_to_date = {
                                k: v[v.index <= signal_date]
                                for k, v in self.sector_data.items()
                                if len(v[v.index <= signal_date]) > 0
                            }

                            if len(spy_to_date) >= 210:  # Need 200+ days for MA
                                self.current_context = self.market_context.get_full_context(
                                    spy_to_date,
                                    sector_to_date,
                                    as_of_date=signal_date
                                )
                                self._last_context_date = signal_date

                                # Track regime changes
                                new_regime = self.current_context['regime']
                                if self.current_regime != new_regime:
                                    self.regime_stats['regime_changes'].append({
                                        'date': signal_date,
                                        'from': self.current_regime.value if self.current_regime else None,
                                        'to': new_regime.value
                                    })
                                    self.current_regime = new_regime
                        except Exception as e:
                            # If context fails, allow trade (conservative)
                            pass

                    # Check if trade should be taken based on regime
                    if self.current_context:
                        current_regime = self.current_context['regime']

                        # CRISIS-ONLY mode: only filter when VIX > 35 (CRISIS regime)
                        if self.crisis_only:
                            should_trade = current_regime != MarketRegime.CRISIS
                            reason = "CRISIS regime (VIX > 35) - no new positions" if not should_trade else ""
                        else:
                            # Full regime filter mode
                            should_trade, reason = self.market_context.should_take_trade(
                                symbol, self.current_context
                            )

                        if not should_trade:
                            self.rejection_stats[model_name]['rejected_regime'] += 1
                            regime_name = current_regime.value
                            self.regime_stats['rejected_by_regime'][regime_name] += 1

                            if self.logger:
                                self.logger.log_signal_rejected(
                                    date=signal_date,
                                    symbol=symbol,
                                    model=model_name,
                                    reason=f'REGIME_FILTER: {reason}',
                                    available_capital=0,
                                    required_capital=0,
                                    model_allocation=0,
                                    model_deployed=0,
                                    details={'regime': regime_name, 'reason': reason}
                                )
                            continue

                # Get model-specific available capital
                model_available_capital = self.portfolio.get_model_available_capital(model_name)
                model_allocation = self.initial_capital * model.allocation_pct
                model_deployed = self.portfolio.capital_by_model[model_name]

                # Apply regime multiplier to position sizing if enabled
                regime_multiplier = 1.0
                if self.enable_regime_filter and self.current_context:
                    if self.crisis_only:
                        # Crisis-only mode: full position sizes unless CRISIS
                        # (but CRISIS trades are already filtered above, so this is just safety)
                        if self.current_context['regime'] == MarketRegime.CRISIS:
                            regime_multiplier = 0.0  # Should never reach here due to filter above
                    else:
                        # Full regime filter: apply deployment multiplier
                        regime_multiplier = self.current_context.get('final_deployment_multiplier', 1.0)
                    model_available_capital *= regime_multiplier

                entry_price = signal_data['entry_price']
                shares = model.calculate_position_size(
                    available_capital=model_available_capital,
                    price=entry_price,
                    confidence=signal_data.get('confidence', 1.0)
                )

                required_capital = shares * entry_price if shares > 0 else entry_price * 100

                if shares <= 0:
                    self.rejection_stats[model_name]['rejected_no_capital'] += 1

                    # Log rejection - no capital
                    if self.logger:
                        self.logger.log_signal_rejected(
                            date=signal_date,
                            symbol=symbol,
                            model=model_name,
                            reason='NO_CAPITAL',
                            available_capital=model_available_capital,
                            required_capital=required_capital,
                            model_allocation=model_allocation,
                            model_deployed=model_deployed,
                            details={
                                'entry_price': entry_price,
                                'confidence': signal_data.get('confidence', 1.0)
                            }
                        )
                    continue

                # Try to open position (layering enabled - multiple models can hold same stock)
                success, rejection_reason = self.portfolio.open_position(
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

                    # Log position opened
                    if self.logger:
                        self.logger.log_position_opened(
                            date=signal_date,
                            symbol=symbol,
                            model=model_name,
                            shares=shares,
                            price=entry_price,
                            stop_loss=signal_data['stop_loss']
                        )

                        # Log capital state after position open
                        self.logger.log_capital_state(
                            date=signal_date,
                            event_type='POSITION_OPEN',
                            total_capital=self.portfolio.current_capital,
                            available_capital=self.portfolio.available_capital,
                            deployed_capital=self.initial_capital - self.portfolio.available_capital,
                            model_allocations={
                                name: {
                                    'allocated': self.initial_capital * m.allocation_pct,
                                    'deployed': self.portfolio.capital_by_model[name],
                                    'available': self.portfolio.get_model_available_capital(name)
                                }
                                for name, m in self.portfolio.models.items()
                            },
                            open_positions_count=len(self.portfolio.open_positions),
                            trigger_symbol=symbol,
                            trigger_model=model_name
                        )
                else:
                    # Map rejection reasons to stats
                    if rejection_reason == 'DUPLICATE_POSITION_SAME_MODEL':
                        self.rejection_stats[model_name]['rejected_duplicate'] += 1
                    elif rejection_reason in ('SYMBOL_EXPOSURE_EXCEEDED', 'MODEL_ALLOCATION_EXCEEDED', 'NO_CAPITAL'):
                        self.rejection_stats[model_name]['rejected_allocation'] += 1
                    else:
                        self.rejection_stats[model_name]['rejected_allocation'] += 1

                    # Log rejection with specific reason
                    if self.logger:
                        # Get existing positions on this symbol for context
                        existing_positions = self.portfolio.get_positions_for_symbol(symbol)
                        details = {
                            'existing_models': list(existing_positions.keys()),
                            'symbol_exposure': self.portfolio.get_symbol_exposure(symbol),
                            'max_exposure': self.portfolio.initial_capital * self.portfolio.max_stock_exposure_pct
                        }
                        self.logger.log_signal_rejected(
                            date=signal_date,
                            symbol=symbol,
                            model=model_name,
                            reason=rejection_reason or 'ALLOCATION_EXCEEDED',
                            available_capital=model_available_capital,
                            required_capital=shares * entry_price,
                            model_allocation=model_allocation,
                            model_deployed=model_deployed,
                            details=details
                        )
            
            # Check stop losses
            self.portfolio.check_stop_losses(signal_date, analyzed_data)

            # Check model-specific exits (calls model.generate_exit_signals)
            # NOTE: This is redundant with manual checks above but matches working backtest
            for pos_key in list(self.portfolio.open_positions.keys()):
                position = self.portfolio.open_positions[pos_key]
                pos_symbol = position['symbol']
                pos_model_name = position['model']
                pos_model = self.portfolio.models[pos_model_name]
                pos_df = analyzed_data[pos_symbol]

                try:
                    pos_idx = pos_df.index.get_loc(signal_date)
                    if isinstance(pos_idx, slice):
                        pos_idx = pos_idx.start

                    exit_signal = pos_model.generate_exit_signals(pos_df, pos_idx, position)

                    if exit_signal.get('signal'):
                        position = self.portfolio.open_positions.get(pos_key)
                        if position:
                            exit_price = exit_signal['exit_price']
                            pnl = (exit_price - position['entry_price']) * position['shares']
                            pnl_pct = ((exit_price / position['entry_price']) - 1) * 100
                            hold_days = (signal_date - position['entry_date']).days

                            self.portfolio.close_position(
                                symbol=pos_symbol,
                                exit_date=signal_date,
                                exit_price=exit_price,
                                exit_reason=exit_signal['reason'],
                                model_name=pos_model_name
                            )

                            # Log the model-triggered exit
                            if self.logger:
                                self.logger.log_position_closed(
                                    date=signal_date,
                                    symbol=pos_symbol,
                                    model=pos_model_name,
                                    shares=position['shares'],
                                    price=exit_price,
                                    pnl=pnl,
                                    pnl_pct=pnl_pct,
                                    hold_days=hold_days,
                                    exit_reason=exit_signal['reason']
                                )
                except Exception as e:
                    # Log the actual error instead of silently continuing
                    print(f"    WARNING: Exit signal error for {pos_symbol}/{pos_model_name}: {e}")
                    continue

        # Close remaining positions at end
        remaining_count = len(self.portfolio.open_positions)
        print(f"\nClosing {remaining_count} remaining positions...")
        end_of_backtest_closed = 0
        for pos_key in list(self.portfolio.open_positions.keys()):
            position = self.portfolio.open_positions[pos_key]
            symbol = position['symbol']
            model_name = position['model']
            if symbol in analyzed_data:
                df = analyzed_data[symbol]
                exit_date = df.index[-1]
                exit_price = df['close'].iloc[-1]

                # Calculate P&L for logging
                pnl = (exit_price - position['entry_price']) * position['shares']
                pnl_pct = ((exit_price / position['entry_price']) - 1) * 100
                hold_days = (exit_date - position['entry_date']).days

                self.portfolio.close_position(
                    symbol=symbol,
                    exit_date=exit_date,
                    exit_price=exit_price,
                    exit_reason='END_OF_BACKTEST',
                    model_name=model_name
                )
                end_of_backtest_closed += 1

                # Log the END_OF_BACKTEST close
                if self.logger:
                    self.logger.log_position_closed(
                        date=exit_date,
                        symbol=symbol,
                        model=model_name,
                        shares=position['shares'],
                        price=exit_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        hold_days=hold_days,
                        exit_reason='END_OF_BACKTEST'
                    )

        print(f"  Closed {end_of_backtest_closed} positions at END_OF_BACKTEST")
    
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
                if stats.get('rejected_regime', 0) > 0:
                    print(f"  Rejected - Regime:      {stats['rejected_regime']}")

        # Print regime statistics if enabled
        if self.enable_regime_filter and self.regime_stats['regime_changes']:
            print(f"\n{'='*60}")
            print("MARKET REGIME STATISTICS")
            print(f"{'='*60}")
            print(f"\nRegime Changes: {len(self.regime_stats['regime_changes'])}")
            for change in self.regime_stats['regime_changes'][:10]:  # Show first 10
                print(f"  {change['date'].date()}: {change['from']} → {change['to']}")
            if len(self.regime_stats['regime_changes']) > 10:
                print(f"  ... and {len(self.regime_stats['regime_changes']) - 10} more")

            print(f"\nRejected by Regime:")
            for regime, count in self.regime_stats['rejected_by_regime'].items():
                if count > 0:
                    print(f"  {regime}: {count} signals")
    
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
        print(f"FINAL PORTFOLIO SUMMARY")
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
    parser = argparse.ArgumentParser(description='Run production model backtest')
    parser.add_argument('--start-year', type=int, default=1985, help='Start year for backtest data')
    parser.add_argument('--capital', type=int, default=100000, help='Initial capital')
    parser.add_argument('--description', type=str, default='', help='Run description')
    parser.add_argument('--no-save', action='store_true', help='Do not save to database')
    parser.add_argument('--enable-logs', action='store_true', help='Enable diagnostic logging (for troubleshooting)')
    parser.add_argument('--enable-regime-filter', action='store_true',
                       help='Enable Phase 0 market regime filter (reduces exposure in bear/crisis markets)')
    parser.add_argument('--crisis-only', action='store_true',
                       help='Only filter during CRISIS (VIX > 35). Ignores BEAR/CAUTION regimes.')
    parser.add_argument('--scanner', action='store_true',
                       help='Enable Minervini Trend Template scanner to expand universe')

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Get min_year from config or use argument
    min_year = config.get('portfolio', {}).get('data_start_year', args.start_year)
    if args.start_year != 1985:  # User specified a different year
        min_year = args.start_year

    # Default description
    if not args.description:
        args.description = f"Production - {datetime.now().strftime('%Y-%m-%d')}"

    # Initialize diagnostic logger (disabled by default, use --enable-logs to turn on)
    logs_dir = config.get('paths', {}).get('logs_dir', 'logs')
    if args.enable_logs:
        bt_logger = BacktestLogger(
            logs_dir=logs_dir
        )
    else:
        bt_logger = None

    print("\n" + "="*80)
    print(f"PRODUCTION BACKTEST")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Description: {args.description}")
    print(f"Min Year: {min_year}")
    if bt_logger:
        print(f"Diagnostic Logging: ENABLED (logs/{bt_logger.run_id})")
    else:
        print(f"Diagnostic Logging: DISABLED (use --enable-logs to turn on)")
    if args.enable_regime_filter:
        if args.crisis_only:
            print(f"Market Regime Filter: CRISIS-ONLY (VIX > 35)")
        else:
            print(f"Market Regime Filter: ENABLED (Phase 0 - Full)")
    else:
        print(f"Market Regime Filter: DISABLED (baseline mode)")
    if args.scanner:
        print(f"Scanner: ENABLED (Minervini Trend Template)")
    else:
        print(f"Scanner: DISABLED (using live_universe.txt only)")
    print("="*80)

    # Load universe
    core_path = config.get('paths', {}).get('core_universe', 'live_universe.txt')
    data_dir = config.get('paths', {}).get('data_dir', 'data')

    if args.scanner:
        # Scanner mode: use Minervini Trend Template to expand universe
        scanner_config = config.get('scanner', {})
        tt_config = scanner_config.get('trend_template', {})

        scanner = UniverseScanner(
            core_universe_path=core_path,
            min_price=scanner_config.get('min_price', 3.0),
            max_price=scanner_config.get('max_price', 10000.0),
            min_avg_volume=scanner_config.get('min_avg_volume', 200000),
            min_market_cap=scanner_config.get('min_market_cap', 500_000_000),
            exclude_sectors=scanner_config.get('exclude_sectors', ['Biotechnology']),
            logger=bt_logger
        )

        # Configure trend template thresholds from yaml
        scanner.trend_template.min_rs_rating = tt_config.get('rs_rating_min', 70)
        scanner.trend_template.max_pct_from_high = tt_config.get('max_pct_from_high', 25.0)
        scanner.trend_template.min_pct_above_low = tt_config.get('min_pct_above_low', 30.0)
        scanner.trend_template.ma_rising_days = tt_config.get('ma_rising_days', 20)

        print(f"\nRunning Minervini Trend Template scanner...")
        universe = scanner.quick_scan(data_dir)
        print(f"Scanner universe: {len(universe)} stocks")
    else:
        # Default: load from live_universe.txt (unchanged behavior)
        with open(core_path, 'r') as f:
            universe = [line.strip() for line in f if line.strip()]

    # Add index symbols
    universe.extend(['^GSPC', 'GSPC', 'SPY'])
    # Add sector ETFs for regime filter (will be loaded separately)
    if args.enable_regime_filter:
        universe.extend(SECTOR_ETFS)
    universe = list(set(universe))  # Remove duplicates
    
    # Load data
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

    # Load sector data if regime filter enabled
    sector_data = {}
    market_context = None
    if args.enable_regime_filter:
        sector_data = load_sector_data(data_dir=data_dir, min_year=min_year)
        if len(sector_data) >= 5:  # Need at least 5 sectors for meaningful rotation analysis
            market_context = MarketContextAnalyzer()
            print(f"\n✅ Market context analyzer initialized with {len(sector_data)} sectors")
        else:
            print(f"\n⚠️  Insufficient sector data ({len(sector_data)} ETFs), regime filter disabled")
            args.enable_regime_filter = False

    # Create models
    models = create_models(config)
    print(f"Models: {[m.name for m in models]}")

    # Run backtest
    backtest = ProductionBacktest(
        models,
        args.capital,
        logger=bt_logger,
        market_context=market_context,
        sector_data=sector_data,
        enable_regime_filter=args.enable_regime_filter,
        crisis_only=args.crisis_only
    )
    results = backtest.run(all_data, index_data)
    
    # Save to database
    if not args.no_save and results.get('total_trades', 0) > 0:
        db_path = config.get('database', {}).get('path', 'database/weinstein.db')
        db = DatabaseManager(db_path)

        run_id = db.save_backtest_run(
            results=results,
            run_type='production',
            description=args.description,
            parameters={
                'models': [m.name for m in models],
                'universe_size': len(all_data),
                'min_year': min_year,
                'scanner_enabled': args.scanner,
                'regime_filter_enabled': args.enable_regime_filter,
                'sector_etfs_loaded': len(sector_data) if args.enable_regime_filter else 0,
                'regime_changes': len(backtest.regime_stats['regime_changes']) if args.enable_regime_filter else 0
            }
        )

    
    # Save diagnostic logs
    if bt_logger:
        bt_logger.save_all()
        bt_logger.print_final_summary()

    print(f"\n{'='*80}")
    print("PRODUCTION BACKTEST COMPLETE")
    print(f"{'='*80}\n")
    if not args.no_save and results.get('total_trades', 0) > 0:
        print(f"✅ Saved as Run #{run_id}")
        print(f"   Database: {db_path}")
        print("✅ View results in dashboard")
    else:
        print("ℹ️  Results not saved to database (--no-save or no trades)")
    if bt_logger:
        print(f"✅ Diagnostic logs saved to: {bt_logger.run_log_dir}")
    print("\nThis shows how your LIVE TRADING models perform historically.")
    print("Run weekly to monitor production model health.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
