# models/dual_momentum.py
"""
Dual Momentum Model

Based on Gary Antonacci's research on Dual Momentum:
1. Absolute Momentum: Is the asset trending up? (12-month return > 0)
2. Relative Momentum: Is it outperforming the benchmark? (vs SPY)

This model acts as both:
- A standalone trend-following strategy
- A regime filter that can enhance other models

Key Features:
- Rides strong trends in individual stocks
- Avoids buying stocks in downtrends
- Scales position size by momentum strength
- Uses multiple timeframe momentum (3, 6, 12 month)

Strategy:
- Entry: Positive absolute momentum (12M > 0) + Positive relative momentum (vs SPY)
         + Stage 2 confirmed + Volume confirmation
- Exit: Absolute momentum turns negative OR relative momentum weakens significantly
- Hold Time: Variable (follows trend, typically 60-180 days)
- Position Size: 8% scaled by momentum strength
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelSignal
from core.weinstein_engine import Stage
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class DualMomentum(BaseModel):
    """
    Dual Momentum Model

    Combines absolute and relative momentum for high-probability entries.
    Only buys stocks that are both:
    1. In their own uptrend (absolute momentum > 0)
    2. Outperforming the market (relative momentum > 0)

    Expected: 45-55% win rate, 3.0-5.0 profit factor (asymmetric returns)
    """

    def __init__(self, allocation_pct: float = 0.20):
        """
        Initialize Dual Momentum model

        Args:
            allocation_pct: Portfolio allocation (default 20%)
        """
        super().__init__(
            name="Dual_Momentum",
            allocation_pct=allocation_pct
        )

        # Model parameters
        self.position_size_pct = 0.08       # 8% base position

        # Momentum lookback periods (trading days)
        self.momentum_periods = {
            'short': 63,      # 3 months
            'medium': 126,    # 6 months
            'long': 252       # 12 months
        }

        # Momentum thresholds
        self.min_absolute_momentum = 0.0    # Stock must have positive return
        self.min_relative_momentum = 0.0    # Must beat SPY

        # Momentum ranking for position sizing
        self.strong_momentum_threshold = 0.20   # 20%+ = strong momentum
        self.weak_momentum_threshold = 0.05     # 5%+ = minimum acceptable

        # Exit thresholds
        self.exit_absolute_threshold = -0.05    # Exit if 12M return < -5%
        self.exit_relative_threshold = -0.10    # Exit if trailing SPY by 10%+

        # Technical filters
        self.volume_threshold = 1.2         # Volume confirmation
        self.min_price_vs_ma = 1.0          # Price must be above MA

        # Trailing stop
        self.trailing_stop_pct = 0.15       # 15% trailing stop

        # Track highest prices for trailing stops
        self.position_highest = {}

    def generate_entry_signals(
        self,
        df: pd.DataFrame,
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Check for Dual Momentum entry

        Entry Conditions:
        1. Positive 12-month absolute momentum (stock is trending up)
        2. Positive relative momentum vs SPY (outperforming)
        3. Stage 2 confirmed (Weinstein trend filter)
        4. Price above 30-week MA
        5. Volume confirmation (participation)
        6. Multi-timeframe momentum alignment (3M, 6M, 12M all positive)
        """
        # Need enough data for 12-month lookback
        if idx < self.momentum_periods['long'] + 20:
            return ModelSignal.no_signal()

        current = df.iloc[idx]

        # ========== CONDITION 1: ABSOLUTE MOMENTUM ==========
        abs_momentum = self._calculate_absolute_momentum(df, idx)

        if abs_momentum['12m'] <= self.min_absolute_momentum:
            return ModelSignal.no_signal()

        # ========== CONDITION 2: RELATIVE MOMENTUM (vs SPY) ==========
        if market_df is not None:
            rel_momentum = self._calculate_relative_momentum(df, idx, market_df)
        else:
            # Fall back to Mansfield RS if no market data
            rs_value = current.get('mansfield_rs', 0)
            rel_momentum = {'12m': rs_value * 0.1}  # Approximate conversion

        if rel_momentum['12m'] <= self.min_relative_momentum:
            return ModelSignal.no_signal()

        # ========== CONDITION 3: STAGE 2 ==========
        is_stage_2 = current['stage'] == Stage.STAGE_2
        if not is_stage_2:
            return ModelSignal.no_signal()

        # ========== CONDITION 4: PRICE ABOVE MA ==========
        ma = current['sma_30week']
        price_vs_ma = current['close'] / ma

        if price_vs_ma < self.min_price_vs_ma:
            return ModelSignal.no_signal()

        # ========== CONDITION 5: VOLUME CONFIRMATION ==========
        volume_ratio = current.get('volume_ratio', 1.0)
        # Don't require high volume, but avoid very low volume
        if volume_ratio < 0.5:  # Below average volume
            return ModelSignal.no_signal()

        # ========== CONDITION 6: MULTI-TIMEFRAME ALIGNMENT ==========
        # All timeframes should show positive momentum
        if abs_momentum['3m'] <= 0 or abs_momentum['6m'] <= 0:
            return ModelSignal.no_signal()

        # ========== ALL CONDITIONS MET - GENERATE ENTRY ==========

        # Calculate composite momentum score
        momentum_score = self._calculate_momentum_score(abs_momentum, rel_momentum)

        # Calculate stop loss
        # Use higher of: 15% below entry OR 5% below 30-week MA
        stop_price = current['close'] * (1 - self.trailing_stop_pct)
        stop_ma = ma * 0.95
        stop_loss = max(stop_price, stop_ma)

        # Confidence based on momentum score
        base_confidence = 0.75
        momentum_boost = min(momentum_score * 0.1, 0.15)  # Up to +15%
        confidence = min(0.95, base_confidence + momentum_boost)

        return ModelSignal.entry_signal(
            entry_price=current['close'],
            stop_loss=stop_loss,
            confidence=confidence,
            method='DUAL_MOMENTUM',
            abs_momentum_12m=abs_momentum['12m'] * 100,
            abs_momentum_6m=abs_momentum['6m'] * 100,
            abs_momentum_3m=abs_momentum['3m'] * 100,
            rel_momentum_12m=rel_momentum['12m'] * 100,
            momentum_score=momentum_score,
            price_vs_ma_pct=(price_vs_ma - 1) * 100,
            rs=current.get('mansfield_rs', 0)
        )

    def generate_exit_signals(
        self,
        df: pd.DataFrame,
        idx: int,
        position: Dict
    ) -> Dict:
        """
        Check for exit signals

        Exit Conditions:
        1. Absolute momentum turns significantly negative (12M return < -5%)
        2. Relative momentum weakens significantly (trailing SPY by 10%+)
        3. Stage 3 or 4 transition (Weinstein exit)
        4. Trailing stop hit (15% from highest)
        5. Initial stop loss hit
        """
        current = df.iloc[idx]
        symbol = position['symbol']

        # Track highest high
        if symbol not in self.position_highest:
            self.position_highest[symbol] = position['entry_price']
        if current['high'] > self.position_highest[symbol]:
            self.position_highest[symbol] = current['high']

        highest = self.position_highest[symbol]

        # Calculate holding period (handle different datetime types)
        try:
            current_date = pd.Timestamp(df.index[idx])
            entry_date = pd.Timestamp(position['entry_date'])
            days_held = (current_date - entry_date).days
        except Exception:
            days_held = 0

        # ========== EXIT 1: ABSOLUTE MOMENTUM BREAKDOWN ==========
        abs_momentum = self._calculate_absolute_momentum(df, idx)

        if abs_momentum['12m'] < self.exit_absolute_threshold:
            self._cleanup_tracking(symbol)
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='MOMENTUM_BREAKDOWN',
                abs_momentum_12m=abs_momentum['12m'] * 100,
                days_held=days_held
            )

        # ========== EXIT 2: RELATIVE MOMENTUM WEAKNESS ==========
        # Check if significantly underperforming market
        # Use Mansfield RS as proxy if no market data
        rs_value = current.get('mansfield_rs', 0)
        if rs_value < -1.5:  # Significantly underperforming
            self._cleanup_tracking(symbol)
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='RS_WEAKNESS',
                rs=rs_value,
                days_held=days_held
            )

        # ========== EXIT 3: STAGE TRANSITION ==========
        current_stage = current['stage']
        if current_stage in [Stage.STAGE_3, Stage.STAGE_4]:
            self._cleanup_tracking(symbol)
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason=f'STAGE_{current_stage}',
                days_held=days_held
            )

        # ========== EXIT 4: TRAILING STOP ==========
        trailing_stop_level = highest * (1 - self.trailing_stop_pct)

        if current['low'] <= trailing_stop_level:
            exit_price = max(trailing_stop_level, current['open'])  # Realistic fill
            gain_pct = (exit_price - position['entry_price']) / position['entry_price']

            self._cleanup_tracking(symbol)
            return ModelSignal.exit_signal(
                exit_price=exit_price,
                reason='TRAILING_STOP',
                highest_high=highest,
                days_held=days_held,
                gain_pct=gain_pct * 100
            )

        # ========== EXIT 5: INITIAL STOP LOSS ==========
        if current['low'] <= position['stop_loss']:
            self._cleanup_tracking(symbol)
            return ModelSignal.exit_signal(
                exit_price=position['stop_loss'],
                reason='STOP_LOSS',
                days_held=days_held
            )

        return ModelSignal.no_signal()

    def calculate_position_size(
        self,
        available_capital: float,
        price: float,
        confidence: float = 1.0
    ) -> int:
        """
        Calculate position size scaled by momentum strength

        Base: 8% of capital
        Strong momentum (20%+): Full position
        Medium momentum (10-20%): 80% position
        Weak momentum (5-10%): 60% position
        """
        adjusted_pct = self.position_size_pct * confidence

        position_value = available_capital * adjusted_pct
        shares = int(position_value / price)
        return shares

    def _calculate_absolute_momentum(
        self,
        df: pd.DataFrame,
        idx: int
    ) -> Dict[str, float]:
        """
        Calculate absolute momentum (rate of return) over multiple periods

        Returns dict with 3m, 6m, 12m momentum values
        """
        momentum = {}

        current_price = df['close'].iloc[idx]

        for period_name, days in self.momentum_periods.items():
            if idx >= days:
                past_price = df['close'].iloc[idx - days]
                momentum[period_name.replace('short', '3m').replace('medium', '6m').replace('long', '12m')] = \
                    (current_price - past_price) / past_price
            else:
                momentum[period_name.replace('short', '3m').replace('medium', '6m').replace('long', '12m')] = 0.0

        # Map the keys correctly
        result = {
            '3m': momentum.get('3m', 0),
            '6m': momentum.get('6m', 0),
            '12m': momentum.get('12m', 0)
        }

        return result

    def _calculate_relative_momentum(
        self,
        df: pd.DataFrame,
        idx: int,
        market_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate relative momentum vs market (SPY)

        Relative momentum = stock return - market return
        """
        rel_momentum = {}

        if idx >= len(market_df):
            return {'3m': 0, '6m': 0, '12m': 0}

        stock_price = df['close'].iloc[idx]
        market_price = market_df['close'].iloc[idx]

        for period_name, days in self.momentum_periods.items():
            if idx >= days:
                stock_past = df['close'].iloc[idx - days]
                market_past = market_df['close'].iloc[idx - days]

                stock_return = (stock_price - stock_past) / stock_past
                market_return = (market_price - market_past) / market_past

                key = period_name.replace('short', '3m').replace('medium', '6m').replace('long', '12m')
                rel_momentum[key] = stock_return - market_return
            else:
                key = period_name.replace('short', '3m').replace('medium', '6m').replace('long', '12m')
                rel_momentum[key] = 0.0

        result = {
            '3m': rel_momentum.get('3m', 0),
            '6m': rel_momentum.get('6m', 0),
            '12m': rel_momentum.get('12m', 0)
        }

        return result

    def _calculate_momentum_score(
        self,
        abs_momentum: Dict[str, float],
        rel_momentum: Dict[str, float]
    ) -> float:
        """
        Calculate composite momentum score (0-2)

        Weights:
        - 12-month absolute: 40%
        - 6-month absolute: 20%
        - 3-month absolute: 10%
        - 12-month relative: 20%
        - 6-month relative: 10%
        """
        # Normalize momentum values (cap at 50% to avoid outliers dominating)
        abs_12m = min(abs_momentum['12m'], 0.50)
        abs_6m = min(abs_momentum['6m'], 0.50)
        abs_3m = min(abs_momentum['3m'], 0.50)
        rel_12m = min(rel_momentum['12m'], 0.50)
        rel_6m = min(rel_momentum.get('6m', 0), 0.50)

        # Calculate weighted score
        score = (
            abs_12m * 0.40 +
            abs_6m * 0.20 +
            abs_3m * 0.10 +
            rel_12m * 0.20 +
            rel_6m * 0.10
        )

        # Scale to 0-2 range
        return min(2.0, score * 4)

    def _cleanup_tracking(self, symbol: str):
        """Clean up position tracking"""
        if symbol in self.position_highest:
            del self.position_highest[symbol]

    def get_model_params(self) -> Dict:
        """Return model parameters for logging"""
        return {
            'position_size_pct': self.position_size_pct,
            'momentum_periods': self.momentum_periods,
            'min_absolute_momentum': self.min_absolute_momentum,
            'min_relative_momentum': self.min_relative_momentum,
            'exit_absolute_threshold': self.exit_absolute_threshold,
            'exit_relative_threshold': self.exit_relative_threshold,
            'trailing_stop_pct': self.trailing_stop_pct,
            'strategy': 'dual_momentum',
            'timeframe': 'medium_to_long_term',
            'expected_win_rate': '45-55%',
            'expected_profit_factor': '3.0-5.0',
            'expected_hold_days': '60-180',
            'key_features': [
                'Absolute momentum filter (12M return)',
                'Relative momentum filter (vs SPY)',
                'Multi-timeframe alignment (3M, 6M, 12M)',
                'Momentum-based position sizing',
                'Trailing stop protection'
            ]
        }
