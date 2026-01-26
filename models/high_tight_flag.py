# models/high_tight_flag.py
"""
High Tight Flag Model v3.5

v3.5 FIX:
- Simplified pattern detection that ACTUALLY GENERATES SIGNALS
- Uses simple approach: look for 80%+ gain followed by consolidation
- No nested loops, O(n) algorithm
- Previous v3.4's O(n) algorithm didn't find patterns correctly

Expected: 50-200 signals
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelSignal
from core.weinstein_engine import Stage
import pandas as pd
import numpy as np
from typing import Dict, Optional


class HighTightFlag(BaseModel):
    """
    High Tight Flag Model v3.5 - Working pattern detection
    
    Pattern: 80%+ gain in recent history, now in consolidation
    """
    
    def __init__(self, allocation_pct: float = 0.08):
        super().__init__(
            name="High_Tight_Flag",
            allocation_pct=allocation_pct
        )
        
        self.position_size_pct = 0.08
        
        # Pattern parameters - SIMPLIFIED
        self.min_pole_gain_pct = 80.0  # 80%+ gain requirement
        self.max_lookback = 60  # Look back 60 days for pole
        self.flag_lookback = 15  # Flag = last 15 days
        self.min_flag_pullback = 10.0
        self.max_flag_pullback = 25.0
        
        # Volume - ONLY for breakout confirmation
        self.breakout_volume_threshold = 2.0   # HTF breakouts require strong volume
        
        # Entry
        self.breakout_buffer = 0.005
        self.max_chase = 0.03
        
        # Risk
        self.stop_loss_pct = 0.08
        self.trailing_stop_pct = 0.15
        self.profit_target_pct = 0.30
        
        self.position_highest_high = {}
    
    def generate_entry_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Generate HTF entry signals with WORKING pattern detection"""
        if idx < 100:
            return ModelSignal.no_signal()
        
        current = df.iloc[idx]
        price = current['close']
        
        # ================================================================
        # STEP 1: RS CHECK (must be outperforming)
        # ================================================================
        
        rs = current.get('mansfield_rs', 0)
        if pd.isna(rs):
            rs = 0
        if rs < 0:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 2: NEAR HIGHS CHECK
        # ================================================================
        
        recent_high_10 = df['high'].iloc[max(0, idx-10):idx].max()
        if price < recent_high_10 * 0.92:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 3: SIMPLE HTF PATTERN DETECTION
        # Look for: big gain in recent history + current consolidation
        # ================================================================
        
        htf_result = self._detect_htf_simple(df, idx)
        
        if not htf_result['found']:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 4: BREAKOUT CHECK (is TODAY a breakout?)
        # ================================================================
        
        pivot = htf_result['pole_high']
        pivot_with_buffer = pivot * (1 + self.breakout_buffer)
        max_entry = pivot * (1 + self.max_chase)
        
        if price <= pivot_with_buffer:
            return ModelSignal.no_signal()
        
        if price > max_entry:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 5: VOLUME CONFIRMATION (only on breakout day)
        # ================================================================
        
        vol_ratio = current.get('volume_ratio', 0)
        if pd.isna(vol_ratio):
            vol_ratio = 0
        
        if vol_ratio < self.breakout_volume_threshold:
            return ModelSignal.no_signal()
        
        # ================================================================
        # SIGNAL GENERATED!
        # ================================================================
        
        stop_loss = max(htf_result['flag_low'] * 0.97, price * (1 - self.stop_loss_pct))
        
        return ModelSignal.entry_signal(
            entry_price=price,
            stop_loss=stop_loss,
            confidence=0.90,
            method='HTF_BREAKOUT',
            pole_gain_pct=htf_result['pole_gain_pct'],
            flag_pullback_pct=htf_result['flag_pullback_pct'],
            pole_high=htf_result['pole_high'],
            flag_low=htf_result['flag_low'],
            volume_ratio=vol_ratio,
            rs=rs
        )
    
    def _detect_htf_simple(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        Simple HTF detection - O(n) algorithm that actually works
        
        Approach:
        1. Flag = last 15 days (current consolidation)
        2. Pole = everything before flag in lookback window
        3. Check if pole has 80%+ gain and flag is 10-25% pullback
        """
        result = {'found': False}
        
        # Flag window (last 15 days)
        flag_start = idx - self.flag_lookback
        if flag_start < self.max_lookback:
            return result
        
        flag_window = df.iloc[flag_start:idx + 1]
        flag_high = flag_window['high'].max()
        flag_low = flag_window['low'].min()
        
        # Pole window (before flag, up to max_lookback)
        pole_start = max(0, flag_start - self.max_lookback)
        pole_window = df.iloc[pole_start:flag_start]
        
        if len(pole_window) < 10:
            return result
        
        pole_high = pole_window['high'].max()
        pole_low = pole_window['low'].min()
        
        if pole_low <= 0:
            return result
        
        # Check pole gain (80%+)
        pole_gain_pct = ((pole_high - pole_low) / pole_low) * 100
        
        if pole_gain_pct < self.min_pole_gain_pct:
            return result
        
        # Check flag pullback (10-25% from pole high)
        if pole_high <= 0:
            return result
        
        flag_pullback_pct = ((pole_high - flag_low) / pole_high) * 100
        
        if not (self.min_flag_pullback <= flag_pullback_pct <= self.max_flag_pullback):
            return result
        
        # Flag should be below pole high (proper consolidation)
        if flag_high > pole_high * 1.02:
            return result
        
        # Pattern found!
        result['found'] = True
        result['pole_high'] = float(pole_high)
        result['pole_low'] = float(pole_low)
        result['pole_gain_pct'] = round(pole_gain_pct, 1)
        result['flag_high'] = float(flag_high)
        result['flag_low'] = float(flag_low)
        result['flag_pullback_pct'] = round(flag_pullback_pct, 1)
        
        return result
    
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
        
        # Profit target (30% for HTF)
        target_price = position['entry_price'] * (1 + self.profit_target_pct)
        if current['high'] >= target_price:
            self._cleanup_tracking(symbol)
            return ModelSignal.exit_signal(
                exit_price=target_price,
                reason='PROFIT_TARGET',
                days_held=days_held,
                gain_pct=self.profit_target_pct * 100
            )
        
        # Trailing stop after 15% gain
        if highest_high > position['entry_price'] * 1.15:
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
        
        return ModelSignal.no_signal()
    
    def _cleanup_tracking(self, symbol: str):
        if symbol in self.position_highest_high:
            del self.position_highest_high[symbol]
    
    def calculate_position_size(self, available_capital: float, price: float, confidence: float = 1.0) -> int:
        return int(available_capital * self.position_size_pct * confidence / price)
    
    def get_model_params(self) -> Dict:
        return {
            'position_size_pct': self.position_size_pct,
            'min_pole_gain_pct': self.min_pole_gain_pct,
            'breakout_volume_threshold': self.breakout_volume_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'version': 'v3.5',
            'strategy': 'htf_simple_pattern'
        }
