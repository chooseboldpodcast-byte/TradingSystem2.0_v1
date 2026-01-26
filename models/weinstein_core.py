# models/weinstein_core.py
"""
Weinstein Core Model (V5 Logic)

The foundational Stage Analysis model based on Stan Weinstein's methodology.
This is your existing V5 system refactored into the multi-model architecture.

Entry Methods:
1. Stage 1→2 breakout with volume
2. Stage 2 pullback to MA

Exit Method:
- Stage 4 transition (bearish breakdown)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelSignal
from core.weinstein_engine import WeinsteinEngine, Stage
import pandas as pd
from typing import Dict, Optional

class WeinsteinCore(BaseModel):
    """
    Core Weinstein Stage Analysis Model
    
    Your proven V5 system with:
    - Relative Strength filtering
    - Candlestick confirmation
    - Support level detection
    - Stage 4 exits
    """
    
    def __init__(self, allocation_pct: float = 0.40):
        """
        Initialize Weinstein Core model
        
        Args:
            allocation_pct: Portfolio allocation (default 40%)
        """
        super().__init__(
            name="Weinstein_Core",
            allocation_pct=allocation_pct
        )
        
        # Model parameters (from your V5 system)
        self.position_size_pct = 0.10  # 10% per position
        self.volume_threshold_breakout = 2.0   # Weinstein: "at least 2x average volume"
        self.volume_threshold_pullback = 1.5   # Pullback re-entry needs decent volume
        
        # Initialize Weinstein engine
        self.engine = WeinsteinEngine()
    
    def generate_entry_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Check for Weinstein entry signals
        
        Two entry methods:
        1. Stage 1→2 breakout
        2. Stage 2 pullback to MA
        """
        if idx < 150:
            return ModelSignal.no_signal()
        
        current = df.iloc[idx]
        prev_stage = df['stage'].iloc[idx-1] if idx > 0 else 0
        current_stage = current['stage']
        
        # ENTRY METHOD 1: Stage 1→2 Breakout
        if prev_stage == Stage.STAGE_1 and current_stage == Stage.STAGE_2:
            historical = df.iloc[:idx+1]
            
            if self.engine.detect_stage_2_breakout(historical, self.volume_threshold_breakout):
                # Apply quality filters
                if not self._passes_quality_filters(df, idx):
                    return ModelSignal.no_signal()
                
                # Calculate stop loss
                stop_loss = self.engine.calculate_stop_loss(historical, current['close'])
                
                # Get metadata
                has_rs, rs_value = self.engine.has_positive_relative_strength(df, idx)
                is_bullish, pattern = self.engine.detect_bullish_candle(df, idx)
                at_support, support_level = self.engine.detect_support_level(df, idx)
                
                return ModelSignal.entry_signal(
                    entry_price=current['close'],
                    stop_loss=stop_loss,
                    confidence=0.9,  # Breakouts are high confidence
                    method='BREAKOUT',
                    rs=rs_value,
                    pattern=pattern,
                    at_support=at_support,
                    support_level=support_level
                )
        
        # ENTRY METHOD 2: Stage 2 Pullback
        elif current_stage == Stage.STAGE_2:
            if self._detect_pullback_entry(df, idx):
                # Apply quality filters
                if not self._passes_quality_filters(df, idx):
                    return ModelSignal.no_signal()
                
                # Calculate stop loss
                historical = df.iloc[:idx+1]
                stop_loss = self.engine.calculate_stop_loss(historical, current['close'])
                
                # Get metadata
                has_rs, rs_value = self.engine.has_positive_relative_strength(df, idx)
                is_bullish, pattern = self.engine.detect_bullish_candle(df, idx)
                at_support, support_level = self.engine.detect_support_level(df, idx)
                
                return ModelSignal.entry_signal(
                    entry_price=current['close'],
                    stop_loss=stop_loss,
                    confidence=0.85,  # Pullbacks slightly lower confidence
                    method='PULLBACK',
                    rs=rs_value,
                    pattern=pattern,
                    at_support=at_support,
                    support_level=support_level
                )
        
        return ModelSignal.no_signal()
    
    def generate_exit_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        position: Dict
    ) -> Dict:
        """
        Check for Weinstein exit signals
        
        Exit on:
        1. Stage 4 transition (primary exit)
        2. Stop loss hit
        """
        current = df.iloc[idx]
        current_stage = current['stage']
        
        # EXIT 1: Stage 4 transition (Weinstein's exit rule)
        if current_stage == Stage.STAGE_4:
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='STAGE_4'
            )
        
        # EXIT 2: Stop loss (checked by portfolio manager, but return signal if hit)
        if current['low'] <= position['stop_loss']:
            return ModelSignal.exit_signal(
                exit_price=position['stop_loss'],
                reason='STOP_LOSS'
            )
        
        return ModelSignal.no_signal()
    
    def calculate_position_size(
        self, 
        available_capital: float, 
        price: float,
        confidence: float = 1.0
    ) -> int:
        """
        Calculate position size for Weinstein trades
        
        Uses 10% of available capital (Weinstein's rule)
        """
        position_value = available_capital * self.position_size_pct
        shares = int(position_value / price)
        return shares
    
    def _detect_pullback_entry(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Detect Stage 2 pullback entry opportunity
        
        Conditions:
        - In Stage 2
        - Price 0-5% above 30-week MA
        - MA rising
        - Volume confirmation
        - Recent pullback occurred
        """
        if idx < 20:
            return False
        
        current = df.iloc[idx]
        
        # Basic Stage 2 checks
        ma = current['sma_30week']
        ma_slope = current['ma_slope_pct']
        volume_ratio = current['volume_ratio']
        
        # Price position relative to MA
        price_to_ma_pct = ((current['close'] - ma) / ma) * 100
        
        # Check pullback conditions
        if not (0 <= price_to_ma_pct <= 5 and
                ma_slope > 0.1 and
                volume_ratio >= self.volume_threshold_pullback):
            return False
        
        # Verify this is actually a pullback (not just hovering at MA)
        lookback_period = min(20, idx)
        recent_high = df['high'].iloc[idx-lookback_period:idx].max()
        
        # Must have pulled back from a higher level
        if recent_high <= current['close'] * 1.03:
            return False
        
        return True
    
    def _passes_quality_filters(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Apply V5 quality filters
        
        Filters:
        1. Relative Strength > 0
        2. Bullish candlestick pattern
        3. Volume confirmation (already checked in entry methods)
        """
        # Filter 1: Relative Strength
        has_rs, rs_value = self.engine.has_positive_relative_strength(df, idx)
        if not has_rs:
            return False
        
        # Filter 2: Bullish Candle
        is_bullish, pattern = self.engine.detect_bullish_candle(df, idx)
        if not is_bullish:
            return False
        
        return True
    
    def get_model_params(self) -> Dict:
        """Return model parameters for logging"""
        return {
            'position_size_pct': self.position_size_pct,
            'volume_threshold_breakout': self.volume_threshold_breakout,
            'volume_threshold_pullback': self.volume_threshold_pullback,
            'filters': ['relative_strength', 'candlestick', 'volume', 'support']
        }
