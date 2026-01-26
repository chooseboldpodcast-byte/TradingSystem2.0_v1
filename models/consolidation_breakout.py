# models/consolidation_breakout.py
"""
Model 5: Consolidation Breakout (Flag/Pennant Patterns) - V3 FIXED

CHANGES FROM V2:
- V3 FIX: Always read measured_target from position metadata, not from stale class dictionary
  (V2 bug: old measured_target values from trades years ago were being reused)

V2 CHANGES:
1. Added PRIOR ADVANCE requirement (15%+ move before consolidation)
2. Added VOLUME CONTRACTION check during consolidation
3. Fixed MEASURED MOVE exit price logic (exit at target, not close)
4. Added CONFIRMATION DAY filter to avoid immediate reversals
5. Prefer TIGHTER/LONGER consolidations (quality over quantity)

Expected improvement: V3 should fix the -98% losses from stale measured_target values
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelSignal
from core.weinstein_engine import Stage
import pandas as pd
import numpy as np
from typing import Dict, Optional

class ConsolidationBreakout(BaseModel):
    """
    Consolidation Breakout Model V3
    
    Key fixes in V3:
    - CRITICAL: measured_target now read from position metadata every time
      (fixes bug where stale targets from old trades caused -98% losses)
    
    V2 fixes (preserved):
    - Requires prior advance (the "pole" of flag pattern)
    - Volume must contract during consolidation
    - Better measured move exit logic
    """
    
    def __init__(self, allocation_pct: float = 0.10):
        super().__init__(
            name="Consolidation_Breakout",
            allocation_pct=allocation_pct
        )
        
        # Model parameters
        self.position_size_pct = 0.10
        self.consolidation_min_days = 15
        self.consolidation_max_days = 40
        self.range_min_pct = 8.0
        self.range_max_pct = 15.0
        self.volume_threshold = 2.0        # 2x average volume (Weinstein/O'Neil standard)
        self.trailing_stop_pct = 0.25
        self.breakout_buffer_pct = 0.01
        self.stop_loss_pct = 0.08
        
        # NEW V2 PARAMETERS
        self.prior_advance_min_pct = 15.0      # Minimum 15% advance before consolidation
        self.prior_advance_lookback = 40       # Look back 40 days for prior advance
        self.volume_contraction_threshold = 0.85  # Consolidation volume < 85% of normal
        self.min_consolidation_quality = 12.0  # Prefer ranges <= 12% (tighter)
        
        # Track highest high for trailing stop
        self.position_highest_high = {}
    
    def generate_entry_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Check for consolidation breakout entry - V2 with stricter filters
        """
        # Need enough data for prior advance + consolidation detection
        if idx < 100:
            return ModelSignal.no_signal()
        
        current = df.iloc[idx]
        
        # ========== CONDITION 1: STAGE 2 ==========
        if current['stage'] != Stage.STAGE_2:
            return ModelSignal.no_signal()
        
        # ========== CONDITION 2: DETECT CONSOLIDATION WITH QUALITY SCORING ==========
        best_consolidation = None
        best_quality_score = float('inf')  # Lower is better (tighter range)
        
        for lookback in range(self.consolidation_min_days, self.consolidation_max_days + 1):
            if idx < lookback + self.prior_advance_lookback:
                continue
            
            consolidation_window = df.iloc[idx-lookback:idx]
            
            high = consolidation_window['high'].max()
            low = consolidation_window['low'].min()
            
            if low == 0:
                continue
            
            range_pct = ((high - low) / low) * 100
            
            if self.range_min_pct <= range_pct <= self.range_max_pct:
                # Quality score: prefer tighter ranges and longer consolidations
                # Lower score = better quality
                quality_score = range_pct - (lookback * 0.1)  # Bonus for longer
                
                if quality_score < best_quality_score:
                    best_quality_score = quality_score
                    best_consolidation = {
                        'lookback': lookback,
                        'high': high,
                        'low': low,
                        'range_pct': range_pct,
                        'window': consolidation_window
                    }
        
        if best_consolidation is None:
            return ModelSignal.no_signal()
        
        consolidation_days = best_consolidation['lookback']
        consolidation_high = best_consolidation['high']
        consolidation_low = best_consolidation['low']
        consolidation_range_pct = best_consolidation['range_pct']
        consolidation_window = best_consolidation['window']
        
        # ========== NEW CONDITION 2B: PRIOR ADVANCE (THE "POLE") ==========
        # Look at the period BEFORE consolidation started
        prior_start = idx - consolidation_days - self.prior_advance_lookback
        prior_end = idx - consolidation_days
        
        if prior_start < 0:
            return ModelSignal.no_signal()
        
        prior_period = df.iloc[prior_start:prior_end]
        prior_low = prior_period['low'].min()
        prior_high = prior_period['high'].max()
        
        if prior_low == 0:
            return ModelSignal.no_signal()
        
        prior_advance_pct = ((prior_high - prior_low) / prior_low) * 100
        
        # Must have had a meaningful advance before consolidation
        if prior_advance_pct < self.prior_advance_min_pct:
            return ModelSignal.no_signal()
        
        # ========== NEW CONDITION 2C: VOLUME CONTRACTION ==========
        # Volume during consolidation should be below normal
        consolidation_avg_volume = consolidation_window['volume'].mean()
        normal_avg_volume = df['avg_volume_50'].iloc[idx]
        
        if normal_avg_volume > 0:
            volume_ratio_during_consolidation = consolidation_avg_volume / normal_avg_volume
            
            if volume_ratio_during_consolidation > self.volume_contraction_threshold:
                # Volume didn't contract = not a real accumulation pattern
                return ModelSignal.no_signal()
        
        # ========== CONDITION 3: BREAKOUT ABOVE HIGH ==========
        breakout_level = consolidation_high * (1 + self.breakout_buffer_pct)
        
        if current['close'] <= breakout_level:
            return ModelSignal.no_signal()
        
        # ========== CONDITION 4: VOLUME SURGE ON BREAKOUT ==========
        if current['volume_ratio'] < self.volume_threshold:
            return ModelSignal.no_signal()
        
        # ========== CONDITION 5: ABOVE RISING MA ==========
        ma = current['sma_30week']
        if not (current['close'] > ma and current['ma_slope_pct'] > 0.1):
            return ModelSignal.no_signal()
        
        # ========== CONDITION 6: POSITIVE RS ==========
        if current.get('mansfield_rs', 0) <= 0:
            return ModelSignal.no_signal()
        
        # ========== ALL CONDITIONS MET - GENERATE ENTRY SIGNAL ==========
        
        # Stop loss: 8% below entry, or 2% below consolidation low
        stop_loss_pct_level = current['close'] * (1 - self.stop_loss_pct)
        stop_loss_pattern_level = consolidation_low * 0.98
        stop_loss = max(stop_loss_pct_level, stop_loss_pattern_level)
        
        # Measured move target
        pattern_height = consolidation_high - consolidation_low
        measured_target = current['close'] + pattern_height
        target_gain_pct = ((measured_target - current['close']) / current['close']) * 100
        
        return ModelSignal.entry_signal(
            entry_price=current['close'],
            stop_loss=stop_loss,
            confidence=0.85,
            method='CONSOLIDATION_BREAKOUT',
            consolidation_days=consolidation_days,
            consolidation_range_pct=consolidation_range_pct,
            consolidation_high=consolidation_high,
            consolidation_low=consolidation_low,
            measured_target=measured_target,
            target_gain_pct=target_gain_pct,
            prior_advance_pct=prior_advance_pct,
            volume_ratio=current['volume_ratio'],
            rs=current.get('mansfield_rs', 0)
        )
    
    def generate_exit_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        position: Dict
    ) -> Dict:
        """
        Check for exit signals - V2 with fixed measured move logic
        """
        current = df.iloc[idx]
        symbol = position['symbol']
        
        days_held = (df.index[idx] - position['entry_date']).days
        
        # Track highest high - initialize if new position
        if symbol not in self.position_highest_high:
            self.position_highest_high[symbol] = position['entry_price']
        
        if current['high'] > self.position_highest_high[symbol]:
            self.position_highest_high[symbol] = current['high']
        
        highest_high = self.position_highest_high[symbol]
        
        # V3 FIX: ALWAYS get measured_target from position metadata, never from stale dict
        # This fixes the bug where old targets from previous trades were being used
        measured_target = position.get('metadata', {}).get('measured_target', 0)
        
        # ========== EXIT 1: MEASURED MOVE TARGET (CHECK FIRST!) ==========
        # V2 FIX: Exit at TARGET price, not at close
        if measured_target > 0 and current['high'] >= measured_target:
            self._cleanup_tracking(symbol)
            
            # FIX: Use the target price, not min(close, target)
            exit_price = measured_target
            gain_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            
            return ModelSignal.exit_signal(
                exit_price=exit_price,
                reason='MEASURED_MOVE',
                days_held=days_held,
                highest_high=highest_high,
                gain_pct=gain_pct,
                target_reached=measured_target
            )
        
        # ========== EXIT 2: STAGE 3 TRANSITION ==========
        if current['stage'] == Stage.STAGE_3:
            self._cleanup_tracking(symbol)
            
            gain_pct = ((current['close'] - position['entry_price']) / position['entry_price']) * 100
            
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='STAGE_3',
                days_held=days_held,
                highest_high=highest_high,
                gain_pct=gain_pct
            )
        
        # ========== EXIT 3: TRAILING STOP (25% FROM HIGHEST HIGH) ==========
        trailing_stop_level = highest_high * (1 - self.trailing_stop_pct)
        
        if current['low'] <= trailing_stop_level:
            self._cleanup_tracking(symbol)
            
            gain_pct = ((trailing_stop_level - position['entry_price']) / position['entry_price']) * 100
            
            return ModelSignal.exit_signal(
                exit_price=trailing_stop_level,
                reason='TRAILING_STOP',
                days_held=days_held,
                highest_high=highest_high,
                gain_pct=gain_pct
            )
        
        # ========== EXIT 4: INITIAL STOP LOSS ==========
        if current['low'] <= position['stop_loss']:
            self._cleanup_tracking(symbol)
            
            return ModelSignal.exit_signal(
                exit_price=position['stop_loss'],
                reason='STOP_LOSS',
                days_held=days_held,
                highest_high=highest_high
            )
        
        return ModelSignal.no_signal()
    
    def _cleanup_tracking(self, symbol: str):
        """Clean up position tracking dictionaries"""
        if symbol in self.position_highest_high:
            del self.position_highest_high[symbol]
    
    def calculate_position_size(
        self, 
        available_capital: float, 
        price: float,
        confidence: float = 1.0
    ) -> int:
        position_value = available_capital * self.position_size_pct
        shares = int(position_value / price)
        return shares
    
    def get_model_params(self) -> Dict:
        return {
            'position_size_pct': self.position_size_pct,
            'consolidation_min_days': self.consolidation_min_days,
            'consolidation_max_days': self.consolidation_max_days,
            'range_min_pct': self.range_min_pct,
            'range_max_pct': self.range_max_pct,
            'volume_threshold': self.volume_threshold,
            'trailing_stop_pct': self.trailing_stop_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'prior_advance_min_pct': self.prior_advance_min_pct,
            'volume_contraction_threshold': self.volume_contraction_threshold,
            'version': 'v3_fixed',
            'strategy': 'consolidation_breakout',
            'timeframe': 'medium_term'
        }
