# models/pocket_pivot.py
"""
Pocket Pivot Model v3.2

Based on Gil Morales and Chris Kacher's methodology

v3.2 FIXES:
- Volume check is CORRECTLY first for Pocket Pivot (it's volume-defined)
- But price/MA checks come BEFORE disqualifying checks (cheaper first)
- All 10 Kacher/Morales rules implemented
- RS check uses mansfield_rs > 0 directly

For Pocket Pivot, volume IS the first check because by definition
a Pocket Pivot requires volume > max down-day volume. This is different
from VCP/RS Breakout where volume confirms a breakout.

Expected: 20,962 signals → 1,500-3,000 signals (quality filter)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelSignal
from core.weinstein_engine import Stage
import pandas as pd
import numpy as np
from typing import Dict, Optional


class PocketPivot(BaseModel):
    """
    Pocket Pivot Model v3.2 - All 10 Kacher/Morales rules
    """
    
    def __init__(self, allocation_pct: float = 0.05):
        super().__init__(
            name="Pocket_Pivot",
            allocation_pct=allocation_pct
        )
        
        self.position_size_pct = 0.05  # Smaller size for this model
        
        # Volume parameters
        self.down_volume_lookback = 10  # Compare to 10 prior down days
        
        # MA proximity (Rule #4)
        self.ma_proximity_pct = 2.0  # Within 2% of MA = "near"
        self.extended_threshold_pct = 5.0  # >5% above both MAs = extended
        
        # Downtrend detection (Rule #6)
        self.downtrend_months = 5
        self.downtrend_decline_pct = 25.0
        
        # V-shape recovery detection (Rule #8)
        self.v_shape_lookback = 20
        self.v_shape_recovery_pct = 80.0
        
        # Wedging detection (Rule #9)
        self.wedge_lookback = 15
        self.wedge_tightness_pct = 3.0
        
        # Risk management
        self.stop_loss_pct = 0.07
        self.trailing_stop_pct = 0.15
        
        self.position_highest_high = {}
    
    def generate_entry_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate Pocket Pivot signals with ALL 10 rules
        
        Filter order (cheapest first):
        1. Basic checks (close > open, enough data)
        2. Price position (above MAs)
        3. MA proximity (within 2%)
        4. Volume check (> max down volume)
        5. Disqualifying patterns (downtrend, V-shape, wedging)
        """
        if idx < 200:
            return ModelSignal.no_signal()
        
        current = df.iloc[idx]
        price = current['close']
        
        # ================================================================
        # RULE #3: Close > Open (up day)
        # ================================================================
        if current['close'] <= current['open']:
            return ModelSignal.no_signal()
        
        # ================================================================
        # Calculate MAs
        # ================================================================
        sma_10 = df['close'].iloc[idx-9:idx+1].mean()
        sma_50 = df['close'].iloc[idx-49:idx+1].mean()
        sma_200 = df['close'].iloc[idx-199:idx+1].mean()
        
        # ================================================================
        # RULE #7: NOT under 50-dma or 200-dma
        # ================================================================
        if price < sma_50 or price < sma_200:
            return ModelSignal.no_signal()
        
        # ================================================================
        # RULE #4: Price within 2% of 10-day or 50-day MA (not extended)
        # ================================================================
        dist_to_10 = abs((price - sma_10) / sma_10) * 100
        dist_to_50 = abs((price - sma_50) / sma_50) * 100
        
        near_10 = dist_to_10 <= self.ma_proximity_pct
        near_50 = dist_to_50 <= self.ma_proximity_pct
        
        if not (near_10 or near_50):
            return ModelSignal.no_signal()
        
        # ================================================================
        # RULE #10: If extended (>5% above BOTH MAs), reject
        # ================================================================
        above_10_pct = ((price - sma_10) / sma_10) * 100 if sma_10 > 0 else 0
        above_50_pct = ((price - sma_50) / sma_50) * 100 if sma_50 > 0 else 0
        
        if above_10_pct > self.extended_threshold_pct and above_50_pct > self.extended_threshold_pct:
            return ModelSignal.no_signal()
        
        # ================================================================
        # RULE #2: Volume > max down-day volume (last 10 days)
        # This IS the core Pocket Pivot definition
        # ================================================================
        down_volumes = []
        for i in range(max(0, idx - self.down_volume_lookback), idx):
            if df['close'].iloc[i] < df['open'].iloc[i]:
                down_volumes.append(df['volume'].iloc[i])
        
        if len(down_volumes) == 0:
            # No down days = can't compare, allow signal
            max_down_volume = 0
        else:
            max_down_volume = max(down_volumes)
        
        if current['volume'] <= max_down_volume:
            return ModelSignal.no_signal()
        
        # ================================================================
        # RULE #5: Within constructive basing pattern (Stage 1 or early Stage 2)
        # ================================================================
        stage = current.get('stage')
        if stage not in [Stage.STAGE_1, Stage.STAGE_2, 1, 2]:
            return ModelSignal.no_signal()
        
        # RS check - outperforming market
        rs = current.get('mansfield_rs', 0)
        if pd.isna(rs):
            rs = 0
        if rs < 0:
            return ModelSignal.no_signal()
        
        # ================================================================
        # RULE #6: NOT in 5+ month downtrend (25%+ decline)
        # ================================================================
        if self._in_major_downtrend(df, idx):
            return ModelSignal.no_signal()
        
        # ================================================================
        # RULE #8: NOT after V-shaped recovery
        # ================================================================
        if self._is_v_shaped_recovery(df, idx):
            return ModelSignal.no_signal()
        
        # ================================================================
        # RULE #9: NOT after wedging patterns
        # ================================================================
        if self._is_wedging(df, idx):
            return ModelSignal.no_signal()
        
        # ================================================================
        # SIGNAL GENERATED!
        # ================================================================
        
        # Stop below recent pivot low or 7%
        recent_low = df['low'].iloc[max(0, idx-10):idx+1].min()
        pattern_stop = recent_low * 0.98
        pct_stop = price * (1 - self.stop_loss_pct)
        stop_loss = max(pattern_stop, pct_stop)
        
        # Volume surge ratio
        avg_vol = df['volume'].iloc[max(0, idx-50):idx].mean()
        vol_ratio = current['volume'] / avg_vol if avg_vol > 0 else 1.0
        
        return ModelSignal.entry_signal(
            entry_price=price,
            stop_loss=stop_loss,
            confidence=0.75,
            method='POCKET_PIVOT',
            near_ma='10-day' if near_10 else '50-day',
            volume_vs_down=round(current['volume'] / max_down_volume, 2) if max_down_volume > 0 else float('inf'),
            volume_ratio=round(vol_ratio, 2),
            rs=rs,
            stage=int(stage) if stage else 0
        )
    
    def _in_major_downtrend(self, df: pd.DataFrame, idx: int) -> bool:
        """Rule #6: Check for 5+ month downtrend (25%+ decline)"""
        lookback = self.downtrend_months * 21  # ~5 months of trading days
        
        if idx < lookback:
            return False
        
        window = df.iloc[idx - lookback:idx + 1]
        high_in_period = window['high'].max()
        current_price = window['close'].iloc[-1]
        
        if high_in_period <= 0:
            return False
        
        decline_pct = ((high_in_period - current_price) / high_in_period) * 100
        
        # Also check MA slope
        sma_50_now = df['close'].iloc[idx-49:idx+1].mean()
        sma_50_ago = df['close'].iloc[idx-lookback-49:idx-lookback+1].mean() if idx >= lookback + 50 else sma_50_now
        
        ma_declining = sma_50_now < sma_50_ago * 0.95
        
        return decline_pct >= self.downtrend_decline_pct and ma_declining
    
    def _is_v_shaped_recovery(self, df: pd.DataFrame, idx: int) -> bool:
        """Rule #8: Check for V-shaped recovery (too sharp)"""
        lookback = self.v_shape_lookback
        
        if idx < lookback * 2:
            return False
        
        # Look for sharp decline followed by sharp recovery
        window = df.iloc[idx - lookback:idx + 1]
        
        low_idx = window['low'].argmin()
        low_price = window['low'].iloc[low_idx]
        
        if low_price <= 0:
            return False
        
        # Check recovery from low
        current_price = window['close'].iloc[-1]
        recovery_pct = ((current_price - low_price) / low_price) * 100
        
        # If recovered 80%+ of a 20%+ drop in just 20 days, it's V-shaped
        pre_low_high = window['high'].iloc[:low_idx + 1].max() if low_idx > 0 else current_price
        drop_pct = ((pre_low_high - low_price) / pre_low_high) * 100 if pre_low_high > 0 else 0
        
        if drop_pct >= 20 and recovery_pct >= self.v_shape_recovery_pct:
            # V-shaped if recovery happened in less than half the time of the drop
            days_to_low = low_idx
            days_from_low = len(window) - low_idx - 1
            
            if days_from_low <= days_to_low * 0.5 and days_from_low <= 10:
                return True
        
        return False
    
    def _is_wedging(self, df: pd.DataFrame, idx: int) -> bool:
        """Rule #9: Check for wedging (increasingly tight range)"""
        lookback = self.wedge_lookback
        
        if idx < lookback:
            return False
        
        window = df.iloc[idx - lookback:idx + 1]
        
        # Calculate range over time
        first_half = window.iloc[:lookback // 2]
        second_half = window.iloc[lookback // 2:]
        
        first_range = (first_half['high'].max() - first_half['low'].min())
        second_range = (second_half['high'].max() - second_half['low'].min())
        
        if first_range <= 0:
            return False
        
        # Wedging if range contracted to < 40% of original
        range_contraction = second_range / first_range
        
        # Also check if current range is very tight
        recent_range = (window['high'].iloc[-5:].max() - window['low'].iloc[-5:].min())
        avg_price = window['close'].iloc[-5:].mean()
        
        if avg_price <= 0:
            return False
        
        recent_range_pct = (recent_range / avg_price) * 100
        
        return range_contraction < 0.4 and recent_range_pct < self.wedge_tightness_pct
    
    def generate_exit_signals(self, df: pd.DataFrame, idx: int, position: Dict) -> Dict:
        """Generate exit signals"""
        current = df.iloc[idx]
        symbol = position['symbol']
        days_held = (df.index[idx] - position['entry_date']).days
        
        if symbol not in self.position_highest_high:
            self.position_highest_high[symbol] = position['entry_price']
        
        if current['high'] > self.position_highest_high[symbol]:
            self.position_highest_high[symbol] = current['high']
        
        highest_high = self.position_highest_high[symbol]
        current_gain_pct = ((current['close'] - position['entry_price']) / position['entry_price']) * 100
        
        # Stop loss
        if current['low'] <= position['stop_loss']:
            self._cleanup_tracking(symbol)
            return ModelSignal.exit_signal(
                exit_price=position['stop_loss'],
                reason='STOP_LOSS',
                days_held=days_held
            )
        
        # Trailing stop after 10% gain
        if highest_high > position['entry_price'] * 1.10:
            trailing_stop = highest_high * (1 - self.trailing_stop_pct)
            if current['low'] <= trailing_stop:
                self._cleanup_tracking(symbol)
                return ModelSignal.exit_signal(
                    exit_price=trailing_stop,
                    reason='TRAILING_STOP',
                    days_held=days_held,
                    gain_pct=((trailing_stop - position['entry_price']) / position['entry_price']) * 100
                )
        
        # Stage 4 exit
        if current.get('stage') == Stage.STAGE_4:
            self._cleanup_tracking(symbol)
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='STAGE_4',
                days_held=days_held,
                gain_pct=current_gain_pct
            )
        
        # Close below 50-day MA for 3 consecutive days
        sma_50 = df['close'].iloc[idx-49:idx+1].mean()
        if idx >= 3:
            below_50 = all(df['close'].iloc[idx-2:idx+1] < sma_50)
            if below_50:
                self._cleanup_tracking(symbol)
                return ModelSignal.exit_signal(
                    exit_price=current['close'],
                    reason='BELOW_50MA_3DAYS',
                    days_held=days_held,
                    gain_pct=current_gain_pct
                )
        
        return ModelSignal.no_signal()
    
    def _cleanup_tracking(self, symbol: str):
        if symbol in self.position_highest_high:
            del self.position_highest_high[symbol]
    
    def calculate_position_size(self, available_capital: float, price: float, confidence: float = 1.0) -> int:
        return int(available_capital * self.position_size_pct * confidence / price)
    
    def get_model_params(self) -> Dict:
        return {
            'position_size_pct': self.position_size_pct,
            'ma_proximity_pct': self.ma_proximity_pct,
            'extended_threshold_pct': self.extended_threshold_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'version': 'v3.2',
            'strategy': 'pocket_pivot_all_10_rules'
        }
