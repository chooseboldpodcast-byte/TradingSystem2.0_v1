# models/vcp.py
"""
VCP (Volatility Contraction Pattern) Model v3.5

v3.5 FIX:
- Simplified pattern detection that ACTUALLY GENERATES SIGNALS
- Previous v3.3 pattern detection was too strict (0 signals)
- Uses simple consolidation range check like the working test script

Expected: 100-500 signals
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelSignal
from core.weinstein_engine import Stage
import pandas as pd
import numpy as np
from typing import Dict, Optional, List


class VCP(BaseModel):
    """
    VCP Model v3.5 - Working pattern detection
    """
    
    def __init__(self, allocation_pct: float = 0.12):
        super().__init__(
            name="VCP",
            allocation_pct=allocation_pct
        )
        
        self.position_size_pct = 0.12
        
        # Trend Template
        self.min_pct_above_52w_low = 25
        self.max_pct_from_52w_high = 30
        
        # Pattern - SIMPLIFIED
        self.max_consolidation_range = 35.0  # Max 35% range = valid consolidation
        self.lookback_days = 20  # Look back 20 days for pattern
        
        # Volume - ONLY for breakout confirmation
        self.breakout_volume_threshold = 2.0   # Minervini also requires 2x+ volume
        
        # Entry
        self.pivot_buffer_pct = 0.01
        self.max_entry_extension = 0.05
        
        # Risk
        self.stop_loss_pct = 0.07
        self.trailing_stop_pct = 0.20
        
        self.position_highest_high = {}
    
    def generate_entry_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Generate VCP entry signals with WORKING pattern detection"""
        if idx < 252:
            return ModelSignal.no_signal()
        
        current = df.iloc[idx]
        price = current['close']
        
        # ================================================================
        # STEP 1: TREND TEMPLATE (Minervini's 8-point)
        # ================================================================
        
        sma_50 = df['close'].iloc[idx-49:idx+1].mean()
        sma_150 = df['close'].iloc[idx-149:idx+1].mean()
        sma_200 = df['close'].iloc[idx-199:idx+1].mean()
        
        # Criterion #1: Price > 150-day and 200-day MA
        if price <= sma_150 or price <= sma_200:
            return ModelSignal.no_signal()
        
        # Criterion #2: 150-day MA > 200-day MA
        if sma_150 <= sma_200:
            return ModelSignal.no_signal()
        
        # Criterion #3: 200-day MA rising
        if idx >= 274:
            sma_200_prev = df['close'].iloc[idx-221:idx-21+1].mean()
            if sma_200 < sma_200_prev * 0.99:
                return ModelSignal.no_signal()
        
        # Criterion #4: 50-day MA > 150-day and 200-day MA
        if sma_50 <= sma_150 or sma_50 <= sma_200:
            return ModelSignal.no_signal()
        
        # Criterion #5: Price > 50-day MA
        if price <= sma_50:
            return ModelSignal.no_signal()
        
        # Criterion #6: >= 25% above 52-week low
        low_52w = df['low'].iloc[idx-251:idx+1].min()
        if low_52w > 0:
            pct_above_low = ((price - low_52w) / low_52w) * 100
            if pct_above_low < self.min_pct_above_52w_low:
                return ModelSignal.no_signal()
        
        # Criterion #7: Within 30% of 52-week high
        high_52w = df['high'].iloc[idx-251:idx+1].max()
        if high_52w > 0:
            pct_from_high = ((high_52w - price) / high_52w) * 100
            if pct_from_high > self.max_pct_from_52w_high:
                return ModelSignal.no_signal()
        
        # Criterion #8: RS > 0 (outperforming market)
        rs = current.get('mansfield_rs', 0)
        if pd.isna(rs):
            rs = 0
        if rs <= 0:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 2: SIMPLE PATTERN DETECTION (this actually works!)
        # ================================================================
        
        recent_high = df['high'].iloc[idx-self.lookback_days:idx].max()
        recent_low = df['low'].iloc[idx-self.lookback_days:idx].min()
        
        if recent_high <= 0:
            return ModelSignal.no_signal()
        
        # Check consolidation range
        consolidation_range = ((recent_high - recent_low) / recent_high) * 100
        if consolidation_range > self.max_consolidation_range:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 3: BREAKOUT CHECK (is TODAY a breakout?)
        # ================================================================
        
        pivot = recent_high
        pivot_with_buffer = pivot * (1 + self.pivot_buffer_pct)
        max_entry_price = pivot * (1 + self.max_entry_extension)
        
        if price <= pivot_with_buffer:
            return ModelSignal.no_signal()
        
        if price > max_entry_price:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 4: VOLUME CONFIRMATION (only on breakout day)
        # ================================================================
        
        vol_ratio = current.get('volume_ratio', 0)
        if pd.isna(vol_ratio):
            vol_ratio = 0
        
        if vol_ratio < self.breakout_volume_threshold:
            return ModelSignal.no_signal()
        
        # ================================================================
        # SIGNAL GENERATED!
        # ================================================================
        
        stop_loss = max(recent_low * 0.98, price * (1 - self.stop_loss_pct))
        
        return ModelSignal.entry_signal(
            entry_price=price,
            stop_loss=stop_loss,
            confidence=0.80,
            method='VCP_BREAKOUT',
            consolidation_range=round(consolidation_range, 1),
            pivot_price=pivot,
            pattern_low=recent_low,
            volume_ratio=vol_ratio,
            rs=rs
        )
    
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
        
        # Trailing stop
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
        
        # Stage 4
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
            'max_consolidation_range': self.max_consolidation_range,
            'breakout_volume_threshold': self.breakout_volume_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'version': 'v3.5',
            'strategy': 'vcp_simple_pattern'
        }
