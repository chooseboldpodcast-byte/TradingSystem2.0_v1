# models/momentum_52w_high.py
"""
Model 2: 52-Week High Momentum (Momentum Breakout)

Buys stocks breaking to new 52-week highs with strong volume,
riding institutional momentum and breakout power.

Strategy:
- Entry: NEW 52-week high + volume surge + Stage 2 + strong RS
- Exit: Stage 3 transition, 20% trailing stop, or 30-day stall
- Hold Time: 60-120 days (medium-term)
- Position Size: 8% (high conviction momentum plays)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelSignal
from core.weinstein_engine import Stage
import pandas as pd
import numpy as np
from typing import Dict, Optional

class Momentum52WeekHigh(BaseModel):
    """
    52-Week High Momentum Model
    
    Captures powerful breakout moves when stocks reach new highs.
    Lower win rate (40-50%) but asymmetric returns (15-30% winners).
    """
    
    def __init__(self, allocation_pct: float = 0.15):
        """
        Initialize 52-Week High Momentum model
        
        Args:
            allocation_pct: Portfolio allocation (default 15%)
        """
        super().__init__(
            name="52W_High_Momentum",
            allocation_pct=allocation_pct
        )
        
        # Model parameters
        self.position_size_pct = 0.08      # 8% per position (bigger bets)
        self.lookback_period = 252         # 52 weeks = ~252 trading days
        self.volume_threshold = 2.0        # 2x average volume (Weinstein/O'Neil standard)
        self.price_above_ma_threshold = 0.05  # 5% above 30-week MA
        self.trailing_stop_pct = 0.20      # 20% trailing stop
        self.stall_days = 30              # Exit if no new high in 30 days
        self.stop_loss_pct = 0.15         # Initial -15% stop loss
        
        # Track highest high for trailing stop
        self.position_highest_high = {}
    
    def generate_entry_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Check for 52-week high breakout entry
        
        Entry Conditions:
        1. Stock makes NEW 52-week high today
        2. Volume surge (≥2x average)
        3. Stage 2 confirmed (uptrend)
        4. Price > 30-week MA by ≥5% (well-established trend)
        5. Relative Strength accelerating (RS improving)
        6. Market context: SPY also strong (optional but preferred)
        """
        # Need enough data for 52-week lookback
        if idx < self.lookback_period + 20:
            return ModelSignal.no_signal()
        
        current = df.iloc[idx]
        
        # ========== CONDITION 1: NEW 52-WEEK HIGH ==========
        # Look back 252 days to find previous 52-week high
        lookback_data = df.iloc[max(0, idx-self.lookback_period):idx]
        prev_52w_high = lookback_data['high'].max()
        
        # Current high must exceed previous 52-week high
        is_new_high = current['high'] > prev_52w_high
        
        if not is_new_high:
            return ModelSignal.no_signal()
        
        # ========== CONDITION 2: VOLUME SURGE ==========
        volume_surge = current['volume_ratio'] >= self.volume_threshold
        
        if not volume_surge:
            return ModelSignal.no_signal()
        
        # ========== CONDITION 3: STAGE 2 ==========
        is_stage_2 = current['stage'] == Stage.STAGE_2
        
        if not is_stage_2:
            return ModelSignal.no_signal()
        
        # ========== CONDITION 4: PRICE WELL ABOVE MA ==========
        # Must be at least 5% above 30-week MA (not just barely)
        ma = current['sma_30week']
        price_to_ma_pct = (current['close'] - ma) / ma
        well_above_ma = price_to_ma_pct >= self.price_above_ma_threshold
        
        if not well_above_ma:
            return ModelSignal.no_signal()
        
        # ========== CONDITION 5: RS ACCELERATING ==========
        # Compare recent RS (last 5 days) to 20-day average RS
        # We want to see RS improving, not deteriorating
        has_rs_data = 'mansfield_rs' in df.columns and idx >= 20
        
        if has_rs_data:
            recent_rs_avg = df['mansfield_rs'].iloc[idx-5:idx+1].mean()
            baseline_rs_avg = df['mansfield_rs'].iloc[idx-20:idx-5].mean()
            rs_accelerating = recent_rs_avg > baseline_rs_avg
            current_rs = current['mansfield_rs']
        else:
            # No RS data, proceed anyway
            rs_accelerating = True
            current_rs = 0
        
        if not rs_accelerating:
            return ModelSignal.no_signal()
        
        # ========== CONDITION 6: MARKET CONTEXT (OPTIONAL) ==========
        # Check if market (SPY) is also strong
        market_strong = True  # Default to True if no market data
        
        if market_df is not None and idx < len(market_df):
            try:
                market_current = market_df.iloc[idx]
                
                # Market should be in Stage 2 or at its own 52-week high area
                market_stage_ok = market_current.get('stage', 2) == Stage.STAGE_2
                
                # Check if market near its own highs (within 5% of 52-week high)
                market_lookback = market_df.iloc[max(0, idx-self.lookback_period):idx]
                market_52w_high = market_lookback['high'].max()
                market_near_high = market_current['close'] >= market_52w_high * 0.95
                
                market_strong = market_stage_ok or market_near_high
            except:
                market_strong = True
        
        # Don't require market strength, but use it for confidence adjustment
        
        # ========== ALL CONDITIONS MET - GENERATE ENTRY SIGNAL ==========
        
        # Calculate stop loss (15% below entry, or below MA, whichever is closer)
        stop_loss_price = current['close'] * (1 - self.stop_loss_pct)
        stop_loss_ma = ma * 0.95  # 5% below MA
        stop_loss = max(stop_loss_price, stop_loss_ma)
        
        # Calculate confidence (adjust based on market strength)
        base_confidence = 0.85
        confidence = base_confidence if market_strong else base_confidence * 0.90
        
        return ModelSignal.entry_signal(
            entry_price=current['close'],
            stop_loss=stop_loss,
            confidence=confidence,
            method='52W_HIGH_BREAKOUT',
            prev_52w_high=prev_52w_high,
            volume_ratio=current['volume_ratio'],
            rs=current_rs,
            price_to_ma_pct=price_to_ma_pct * 100,
            market_strong=market_strong
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
        1. Stage 3 transition (topping pattern - primary exit)
        2. 20% trailing stop from highest high hit
        3. No new high in 30 days (momentum stalling)
        4. Initial stop loss hit (-15%)
        """
        current = df.iloc[idx]
        symbol = position['symbol']
        
        # Calculate days held
        days_held = (df.index[idx] - position['entry_date']).days
        
        # Track highest high since entry (for trailing stop)
        if symbol not in self.position_highest_high:
            self.position_highest_high[symbol] = position['entry_price']
        
        # Update highest high
        if current['high'] > self.position_highest_high[symbol]:
            self.position_highest_high[symbol] = current['high']
        
        highest_high = self.position_highest_high[symbol]
        
        # ========== EXIT 1: STAGE 3 TRANSITION (PRIMARY EXIT) ==========
        current_stage = current['stage']
        
        if current_stage == Stage.STAGE_3:
            # Clean up tracking
            if symbol in self.position_highest_high:
                del self.position_highest_high[symbol]
            
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='STAGE_3',
                days_held=days_held,
                highest_high=highest_high,
                gain_from_entry=(current['close'] - position['entry_price']) / position['entry_price'] * 100
            )
        
        # ========== EXIT 2: TRAILING STOP (20% FROM HIGHEST HIGH) ==========
        # Calculate trailing stop level
        trailing_stop_level = highest_high * (1 - self.trailing_stop_pct)
        
        if current['low'] <= trailing_stop_level:
            # Clean up tracking
            if symbol in self.position_highest_high:
                del self.position_highest_high[symbol]
            
            return ModelSignal.exit_signal(
                exit_price=trailing_stop_level,
                reason='TRAILING_STOP',
                days_held=days_held,
                highest_high=highest_high,
                gain_from_entry=(trailing_stop_level - position['entry_price']) / position['entry_price'] * 100
            )
        
        # ========== EXIT 3: MOMENTUM STALL (NO NEW HIGH IN 30 DAYS) ==========
        # Find when we last made a new high
        if days_held >= self.stall_days:
            # Check if we've made a new high in last 30 days
            recent_data = df.iloc[max(0, idx-self.stall_days):idx+1]
            recent_high = recent_data['high'].max()
            
            # If recent high is same as our tracked highest high, we're still making progress
            # If not, momentum has stalled
            if recent_high < highest_high * 0.99:  # Allow 1% tolerance
                # Clean up tracking
                if symbol in self.position_highest_high:
                    del self.position_highest_high[symbol]
                
                return ModelSignal.exit_signal(
                    exit_price=current['close'],
                    reason='MOMENTUM_STALL',
                    days_held=days_held,
                    highest_high=highest_high,
                    days_since_high=self.stall_days
                )
        
        # ========== EXIT 4: INITIAL STOP LOSS ==========
        if current['low'] <= position['stop_loss']:
            # Clean up tracking
            if symbol in self.position_highest_high:
                del self.position_highest_high[symbol]
            
            return ModelSignal.exit_signal(
                exit_price=position['stop_loss'],
                reason='STOP_LOSS',
                days_held=days_held,
                highest_high=highest_high
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
        
        Uses 8% of available capital (larger than Weinstein's 10% because
        this model runs fewer, higher-conviction trades with asymmetric upside)
        
        Confidence can scale position size:
        - High confidence (market strong): Full 8%
        - Medium confidence: 7%
        - Low confidence: 6%
        """
        # Scale position size by confidence
        adjusted_position_pct = self.position_size_pct * confidence
        
        position_value = available_capital * adjusted_position_pct
        shares = int(position_value / price)
        return shares
    
    def get_model_params(self) -> Dict:
        """Return model parameters for logging"""
        return {
            'position_size_pct': self.position_size_pct,
            'lookback_period': self.lookback_period,
            'volume_threshold': self.volume_threshold,
            'price_above_ma_threshold': self.price_above_ma_threshold,
            'trailing_stop_pct': self.trailing_stop_pct,
            'stall_days': self.stall_days,
            'stop_loss_pct': self.stop_loss_pct,
            'strategy': 'momentum_breakout',
            'timeframe': 'medium_term',
            'expected_win_rate': '40-50%',
            'expected_profit_factor': '4.0-6.0',
            'expected_hold_days': '60-120'
        }
