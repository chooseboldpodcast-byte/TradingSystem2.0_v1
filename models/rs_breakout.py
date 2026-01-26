# models/rs_breakout.py
"""
RS Breakout Model v3.5

v3.5 FIX:
- Simplified base detection that ACTUALLY GENERATES SIGNALS
- Previous v3.3 was too strict (0 signals)
- Uses simple base depth check like the working test script

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


class RSBreakout(BaseModel):
    """
    RS Breakout Model v3.5 - Working pattern detection
    """
    
    def __init__(self, allocation_pct: float = 0.12):
        super().__init__(
            name="RS_Breakout",
            allocation_pct=allocation_pct
        )
        
        self.position_size_pct = 0.12
        
        # RS parameters
        self.min_mansfield_rs = 10
        
        # Base parameters - SIMPLIFIED
        self.lookback_days = 20
        self.max_base_depth = 35.0
        self.min_base_depth = 5.0
        
        # Volume - ONLY for breakout confirmation
        self.breakout_volume_threshold = 2.0   # O'Neil/IBD requires 2x+ volume on breakout
        
        # Entry
        self.breakout_buffer = 0.01
        self.max_chase = 0.05
        
        # Risk
        self.stop_loss_pct = 0.07
        self.trailing_stop_pct = 0.20
        self.profit_target_pct = 0.25
        
        self.position_highest_high = {}
    
    def generate_entry_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Generate RS Breakout signals with WORKING pattern detection"""
        if idx < 252:
            return ModelSignal.no_signal()
        
        current = df.iloc[idx]
        price = current['close']
        
        # ================================================================
        # STEP 1: STAGE 2 CHECK (most important)
        # ================================================================
        
        stage = current.get('stage')
        if stage != Stage.STAGE_2 and stage != 2:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 2: RS CHECK (outperforming by 10%+)
        # ================================================================
        
        rs = current.get('mansfield_rs', 0)
        if pd.isna(rs):
            rs = 0
        if rs < self.min_mansfield_rs:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 3: NEAR HIGHS CHECK
        # ================================================================
        
        recent_high = df['high'].iloc[max(0, idx-self.lookback_days):idx].max()
        if price < recent_high * 0.95:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 4: SIMPLE BASE DETECTION (this actually works!)
        # ================================================================
        
        recent_low = df['low'].iloc[max(0, idx-self.lookback_days):idx].min()
        
        if recent_high <= 0:
            return ModelSignal.no_signal()
        
        base_depth = ((recent_high - recent_low) / recent_high) * 100
        
        if base_depth > self.max_base_depth or base_depth < self.min_base_depth:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 5: BREAKOUT CHECK (is TODAY a breakout?)
        # ================================================================
        
        pivot = recent_high
        pivot_with_buffer = pivot * (1 + self.breakout_buffer)
        max_entry = pivot * (1 + self.max_chase)
        
        if price <= pivot_with_buffer:
            return ModelSignal.no_signal()
        
        if price > max_entry:
            return ModelSignal.no_signal()
        
        # ================================================================
        # STEP 6: VOLUME CONFIRMATION (only on breakout day)
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
            confidence=0.85,
            method='RS_BREAKOUT',
            mansfield_rs=rs,
            base_depth=round(base_depth, 1),
            pivot_price=pivot,
            base_low=recent_low,
            volume_ratio=vol_ratio
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
        
        # Profit target
        target_price = position['entry_price'] * (1 + self.profit_target_pct)
        if current['high'] >= target_price:
            self._cleanup_tracking(symbol)
            return ModelSignal.exit_signal(
                exit_price=target_price,
                reason='PROFIT_TARGET',
                days_held=days_held,
                gain_pct=self.profit_target_pct * 100
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
            'min_mansfield_rs': self.min_mansfield_rs,
            'breakout_volume_threshold': self.breakout_volume_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'version': 'v3.5',
            'strategy': 'rs_breakout_simple_base'
        }
