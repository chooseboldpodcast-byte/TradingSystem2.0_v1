# models/enhanced_mean_reversion.py
"""
Enhanced Mean Reversion Model (Replaces RSI Mean Reversion)

A refined mean reversion strategy that buys deep pullbacks in strong Stage 2
uptrends when price tests the 30-week MA support.

Key Improvements over RSI Mean Reversion:
1. Stricter entry: RSI < 25 (deeper oversold) + price within 3% of MA
2. Larger position size: 8% (captures meaningful gains)
3. Smart stop loss: Based on MA support, not arbitrary percentage
4. Better exits: Hold until RSI > 65 or Stage transition (let winners run)
5. Trend confirmation: Requires 30+ days in Stage 2

Strategy:
- Entry: Deep oversold (RSI < 25) at MA support in strong Stage 2
- Exit: RSI > 65, Stage transition, or MA breakdown
- Hold Time: 10-30 days
- Position Size: 8% (high conviction pullback plays)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelSignal
from core.weinstein_engine import Stage
import pandas as pd
import numpy as np
from typing import Dict, Optional


class EnhancedMeanReversion(BaseModel):
    """
    Enhanced Mean Reversion Model

    Buys deep pullbacks to MA support in confirmed uptrends.
    Higher selectivity, larger positions, better risk/reward.

    Expected: 55-65% win rate, 2.5-3.5 profit factor
    """

    def __init__(self, allocation_pct: float = 0.15):
        """
        Initialize Enhanced Mean Reversion model

        Args:
            allocation_pct: Portfolio allocation (default 15%, replacing RSI's 10%)
        """
        super().__init__(
            name="Enhanced_Mean_Reversion",
            allocation_pct=allocation_pct
        )

        # Model parameters - MORE SELECTIVE than RSI Mean Reversion
        self.position_size_pct = 0.08       # 8% per position (was 5%)
        self.rsi_period = 14
        self.rsi_entry_threshold = 25       # Deeper oversold (was 30)
        self.rsi_exit_threshold = 65        # Higher exit (was 50) - let winners run
        self.max_hold_days = 30             # Longer hold (was 15)

        # MA proximity requirement
        self.ma_proximity_pct = 0.03        # Price within 3% of MA for entry

        # Trend confirmation
        self.min_stage_2_days = 30          # Must be in Stage 2 for 30+ days

        # Stop loss based on MA, not fixed percentage
        self.stop_below_ma_pct = 0.03       # Stop 3% below MA (technical stop)

    def generate_entry_signals(
        self,
        df: pd.DataFrame,
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Check for enhanced mean reversion entry

        Entry Conditions (ALL must be met):
        1. Stock in Stage 2 for at least 30 consecutive days (confirmed uptrend)
        2. RSI < 25 (deeply oversold, not just slightly oversold)
        3. Price within 3% of 30-week MA (at support, not extended)
        4. MA is rising (slope > 0.2%)
        5. Relative Strength > 0 (still outperforming despite pullback)
        6. Volume NOT spiking > 2x (no panic selling / capitulation)
        7. Higher low pattern in last 20 days (pullback in uptrend, not breakdown)
        """
        # Need enough data
        if idx < max(self.rsi_period + 10, 150):
            return ModelSignal.no_signal()

        current = df.iloc[idx]

        # ========== CONDITION 1: CONFIRMED STAGE 2 TREND ==========
        is_stage_2 = current['stage'] == Stage.STAGE_2
        if not is_stage_2:
            return ModelSignal.no_signal()

        days_in_stage_2 = self._count_consecutive_stage(df, idx, Stage.STAGE_2)
        if days_in_stage_2 < self.min_stage_2_days:
            return ModelSignal.no_signal()

        # ========== CONDITION 2: DEEPLY OVERSOLD ==========
        rsi = self._calculate_rsi(df['close'], idx)
        if rsi >= self.rsi_entry_threshold:
            return ModelSignal.no_signal()

        # ========== CONDITION 3: PRICE AT MA SUPPORT ==========
        ma = current['sma_30week']
        price_to_ma_pct = abs(current['close'] - ma) / ma

        # Must be within 3% of MA (at support, not extended above or below)
        if price_to_ma_pct > self.ma_proximity_pct:
            return ModelSignal.no_signal()

        # Price must be AT or ABOVE MA (not below - that's breakdown territory)
        if current['close'] < ma * 0.98:  # Allow 2% tolerance below MA
            return ModelSignal.no_signal()

        # ========== CONDITION 4: MA RISING ==========
        ma_slope = current.get('ma_slope_pct', 0)
        if ma_slope < 0.2:  # MA must be rising at least 0.2%
            return ModelSignal.no_signal()

        # ========== CONDITION 5: POSITIVE RELATIVE STRENGTH ==========
        rs_value = current.get('mansfield_rs', 0)
        if rs_value <= 0:
            return ModelSignal.no_signal()

        # ========== CONDITION 6: NOT PANIC SELLING ==========
        volume_ratio = current.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:  # High volume during pullback = potential capitulation
            return ModelSignal.no_signal()

        # ========== CONDITION 7: HIGHER LOW PATTERN ==========
        # Check if recent pullback is making higher lows (healthy pullback)
        if not self._has_higher_low_pattern(df, idx, lookback=20):
            return ModelSignal.no_signal()

        # ========== ALL CONDITIONS MET - GENERATE ENTRY ==========

        # Calculate stop loss: 3% below MA (technical level)
        stop_loss = ma * (1 - self.stop_below_ma_pct)

        # Calculate potential reward (target RSI 65, estimate 8-12% move)
        # Risk is current price to stop
        risk_per_share = current['close'] - stop_loss

        # Estimate reward based on pullback depth
        # Deeper RSI = larger expected bounce
        expected_bounce_pct = (30 - rsi) * 0.004  # ~0.4% per RSI point below 30
        expected_bounce_pct = max(0.05, min(0.15, expected_bounce_pct))  # 5-15% range
        reward_per_share = current['close'] * expected_bounce_pct

        # Only enter if reward > 2x risk
        risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        if risk_reward < 2.0:
            return ModelSignal.no_signal()

        # Confidence based on RSI depth and RS strength
        base_confidence = 0.80
        # Boost confidence for deeper oversold
        rsi_boost = (25 - rsi) * 0.01  # +1% per RSI point below 25
        # Boost for strong RS
        rs_boost = min(rs_value * 0.02, 0.05)  # Up to +5% for RS
        confidence = min(0.95, base_confidence + rsi_boost + rs_boost)

        return ModelSignal.entry_signal(
            entry_price=current['close'],
            stop_loss=stop_loss,
            confidence=confidence,
            method='ENHANCED_MA_PULLBACK',
            rsi=rsi,
            rs=rs_value,
            ma_proximity_pct=price_to_ma_pct * 100,
            days_in_stage_2=days_in_stage_2,
            ma_slope=ma_slope,
            risk_reward=risk_reward,
            expected_bounce_pct=expected_bounce_pct * 100
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
        1. RSI reaches 65+ (overbought, mean reversion complete)
        2. Stage transition to 3 or 4 (trend change)
        3. Price breaks below MA by more than 5% (support failed)
        4. Time stop: 30 days max hold
        5. Trailing stop: If gained 10%+, protect with 5% trailing
        """
        current = df.iloc[idx]

        # Calculate RSI
        rsi = self._calculate_rsi(df['close'], idx)

        # Calculate days held and gain
        days_held = (df.index[idx] - position['entry_date']).days
        gain_pct = (current['close'] - position['entry_price']) / position['entry_price']

        # ========== EXIT 1: RSI OVERBOUGHT (PRIMARY EXIT) ==========
        if rsi >= self.rsi_exit_threshold:
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='RSI_OVERBOUGHT',
                rsi=rsi,
                days_held=days_held,
                gain_pct=gain_pct * 100
            )

        # ========== EXIT 2: STAGE TRANSITION ==========
        current_stage = current['stage']
        if current_stage in [Stage.STAGE_3, Stage.STAGE_4]:
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason=f'STAGE_{current_stage}',
                rsi=rsi,
                days_held=days_held,
                gain_pct=gain_pct * 100
            )

        # ========== EXIT 3: MA BREAKDOWN ==========
        ma = current['sma_30week']
        price_below_ma_pct = (ma - current['close']) / ma if current['close'] < ma else 0

        if price_below_ma_pct > 0.05:  # More than 5% below MA
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='MA_BREAKDOWN',
                rsi=rsi,
                days_held=days_held,
                price_below_ma_pct=price_below_ma_pct * 100
            )

        # ========== EXIT 4: TIME STOP ==========
        if days_held >= self.max_hold_days:
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='TIME_STOP',
                rsi=rsi,
                days_held=days_held,
                gain_pct=gain_pct * 100
            )

        # ========== EXIT 5: TRAILING STOP (PROFIT PROTECTION) ==========
        if gain_pct >= 0.10:  # If up 10%+
            # Track highest price since entry (simplified - use 5-day high)
            lookback = min(days_held, 10)
            if lookback > 0:
                recent_high = df['high'].iloc[idx-lookback:idx+1].max()
                trailing_stop = recent_high * 0.95  # 5% trailing

                if current['close'] < trailing_stop:
                    return ModelSignal.exit_signal(
                        exit_price=current['close'],
                        reason='TRAILING_STOP',
                        rsi=rsi,
                        days_held=days_held,
                        gain_pct=gain_pct * 100,
                        trailing_from_high=(recent_high - current['close']) / recent_high * 100
                    )

        # ========== EXIT 6: INITIAL STOP LOSS ==========
        if current['low'] <= position['stop_loss']:
            return ModelSignal.exit_signal(
                exit_price=position['stop_loss'],
                reason='STOP_LOSS',
                rsi=rsi,
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
        Calculate position size

        Uses 8% of available capital (up from RSI's 5%)
        Adjusted by confidence level
        """
        # Scale by confidence
        adjusted_pct = self.position_size_pct * confidence

        position_value = available_capital * adjusted_pct
        shares = int(position_value / price)
        return shares

    def _calculate_rsi(self, prices: pd.Series, idx: int) -> float:
        """Calculate RSI at given index"""
        if idx < self.rsi_period:
            return 50.0

        price_window = prices.iloc[max(0, idx-self.rsi_period-1):idx+1]
        deltas = price_window.diff()

        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)

        avg_gain = gains.rolling(window=self.rsi_period).mean().iloc[-1]
        avg_loss = losses.rolling(window=self.rsi_period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _count_consecutive_stage(
        self,
        df: pd.DataFrame,
        idx: int,
        target_stage: int
    ) -> int:
        """Count consecutive days in target stage"""
        count = 0
        for i in range(idx, max(0, idx-100), -1):
            if df['stage'].iloc[i] == target_stage:
                count += 1
            else:
                break
        return count

    def _has_higher_low_pattern(
        self,
        df: pd.DataFrame,
        idx: int,
        lookback: int = 20
    ) -> bool:
        """
        Check if recent price action shows higher lows

        This indicates a healthy pullback in an uptrend,
        not the start of a breakdown.
        """
        if idx < lookback + 5:
            return False

        # Find local lows in the lookback period
        lows = df['low'].iloc[idx-lookback:idx+1].values

        # Simple check: current low > lowest low of first half of lookback
        first_half_low = min(lows[:lookback//2])
        recent_low = min(lows[lookback//2:])

        # Recent low should be at same level or higher than earlier low
        return recent_low >= first_half_low * 0.98  # Allow 2% tolerance

    def get_model_params(self) -> Dict:
        """Return model parameters for logging"""
        return {
            'position_size_pct': self.position_size_pct,
            'rsi_period': self.rsi_period,
            'rsi_entry_threshold': self.rsi_entry_threshold,
            'rsi_exit_threshold': self.rsi_exit_threshold,
            'max_hold_days': self.max_hold_days,
            'ma_proximity_pct': self.ma_proximity_pct,
            'min_stage_2_days': self.min_stage_2_days,
            'stop_below_ma_pct': self.stop_below_ma_pct,
            'strategy': 'enhanced_mean_reversion',
            'timeframe': 'short_to_medium_term',
            'expected_win_rate': '55-65%',
            'expected_profit_factor': '2.5-3.5',
            'expected_hold_days': '10-30',
            'key_improvements': [
                'Deeper oversold entry (RSI < 25 vs 30)',
                'MA proximity requirement (at support)',
                'Trend confirmation (30+ days Stage 2)',
                'Higher exit threshold (RSI 65 vs 50)',
                'Technical stop loss (MA-based)',
                'Larger position size (8% vs 5%)'
            ]
        }
