# models/rsi_mean_reversion.py
"""
Model 1: RSI Mean Reversion

Short-term model that buys oversold conditions in Stage 2 stocks,
expecting a bounce back to the mean.

Strategy:
- Entry: RSI < 30 in Stage 2 stocks with positive RS
- Exit: RSI reaches 50-60, or time stop (15 days), or stop loss (-5%)
- Hold Time: 5-15 days
- Position Size: 5% (smaller than Weinstein)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelSignal
from core.weinstein_engine import Stage
import pandas as pd
import numpy as np
from typing import Dict, Optional

class RSIMeanReversion(BaseModel):
    """
    RSI Mean Reversion Model
    
    Buys oversold bounces in confirmed uptrends.
    High win rate (60-70%), short hold times (5-15 days).
    """
    
    def __init__(self, allocation_pct: float = 0.10):
        """
        Initialize RSI Mean Reversion model
        
        Args:
            allocation_pct: Portfolio allocation (default 10%)
        """
        super().__init__(
            name="RSI_Mean_Reversion",
            allocation_pct=allocation_pct
        )
        
        # Model parameters
        self.position_size_pct = 0.05  # 5% per position (smaller, more trades)
        self.rsi_period = 14
        self.rsi_entry_threshold = 30  # Oversold entry
        self.rsi_exit_threshold = 50   # Neutral exit
        self.max_hold_days = 15
        self.stop_loss_pct = 0.05      # -5% stop loss
    
    def generate_entry_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Check for RSI oversold bounce entry
        
        Entry Conditions:
        1. Stock in Stage 2 (confirmed uptrend)
        2. RSI < 30 (oversold)
        3. Price still above 30-week MA
        4. Relative Strength > 0 (outperforming market)
        5. Volume NOT spiking (no panic selling)
        """
        # Need enough data for RSI calculation
        if idx < self.rsi_period + 10:
            return ModelSignal.no_signal()
        
        current = df.iloc[idx]
        
        # Calculate RSI
        rsi = self._calculate_rsi(df['close'], idx)
        
        # Entry conditions
        is_stage_2 = current['stage'] == Stage.STAGE_2
        is_oversold = rsi < self.rsi_entry_threshold
        above_ma = current['close'] > current['sma_30week']
        has_rs = current.get('mansfield_rs', 0) > 0
        low_volume = current['volume_ratio'] < 1.0  # Not panic selling
        
        # All conditions must be met
        if not all([is_stage_2, is_oversold, above_ma, has_rs, low_volume]):
            return ModelSignal.no_signal()
        
        # Calculate stop loss (5% below entry)
        stop_loss = current['close'] * (1 - self.stop_loss_pct)
        
        return ModelSignal.entry_signal(
            entry_price=current['close'],
            stop_loss=stop_loss,
            confidence=0.8,  # Good odds on mean reversion
            method='RSI_OVERSOLD',
            rsi=rsi,
            rs=current.get('mansfield_rs', 0),
            days_in_stage_2=self._count_consecutive_stage(df, idx, Stage.STAGE_2)
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
        1. RSI reaches 50-60 (back to neutral)
        2. 15-day time stop (holding too long)
        3. Stop loss hit (-5%)
        4. Makes 5-day high (momentum returned)
        """
        current = df.iloc[idx]
        
        # Calculate RSI
        rsi = self._calculate_rsi(df['close'], idx)
        
        # Calculate days held
        days_held = (df.index[idx] - position['entry_date']).days
        
        # EXIT 1: RSI back to neutral (primary exit)
        if rsi >= self.rsi_exit_threshold:
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='RSI_NEUTRAL',
                rsi=rsi,
                days_held=days_held
            )
        
        # EXIT 2: Time stop (holding too long)
        if days_held >= self.max_hold_days:
            return ModelSignal.exit_signal(
                exit_price=current['close'],
                reason='TIME_STOP',
                rsi=rsi,
                days_held=days_held
            )
        
        # EXIT 3: Made new 5-day high (momentum back)
        if idx >= 5:
            recent_high = df['high'].iloc[idx-5:idx].max()
            if current['high'] > recent_high:
                return ModelSignal.exit_signal(
                    exit_price=current['close'],
                    reason='NEW_HIGH',
                    rsi=rsi,
                    days_held=days_held
                )
        
        # EXIT 4: Stop loss (checked by portfolio manager, but signal it)
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
        
        Uses 5% of available capital (smaller than Weinstein)
        This allows more concurrent positions for diversification
        """
        position_value = available_capital * self.position_size_pct
        shares = int(position_value / price)
        return shares
    
    def _calculate_rsi(self, prices: pd.Series, idx: int) -> float:
        """
        Calculate RSI (Relative Strength Index) at given index
        
        Args:
            prices: Series of closing prices
            idx: Current index
        
        Returns:
            RSI value (0-100)
        """
        if idx < self.rsi_period:
            return 50.0  # Neutral if not enough data
        
        # Get price changes
        price_window = prices.iloc[max(0, idx-self.rsi_period-1):idx+1]
        deltas = price_window.diff()
        
        # Separate gains and losses
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gains.rolling(window=self.rsi_period).mean().iloc[-1]
        avg_loss = losses.rolling(window=self.rsi_period).mean().iloc[-1]
        
        # Avoid division by zero
        if avg_loss == 0:
            return 100.0
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _count_consecutive_stage(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        target_stage: int
    ) -> int:
        """
        Count how many consecutive days stock has been in target stage
        
        Useful for understanding trend strength
        """
        count = 0
        for i in range(idx, max(0, idx-60), -1):
            if df['stage'].iloc[i] == target_stage:
                count += 1
            else:
                break
        return count
    
    def get_model_params(self) -> Dict:
        """Return model parameters for logging"""
        return {
            'position_size_pct': self.position_size_pct,
            'rsi_period': self.rsi_period,
            'rsi_entry_threshold': self.rsi_entry_threshold,
            'rsi_exit_threshold': self.rsi_exit_threshold,
            'max_hold_days': self.max_hold_days,
            'stop_loss_pct': self.stop_loss_pct,
            'strategy': 'mean_reversion',
            'timeframe': 'short_term'
        }
