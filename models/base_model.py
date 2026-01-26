# models/base_model.py
"""
Abstract Base Class for Trading Models

All trading models must inherit from this class and implement the required methods.
This ensures consistent interface across all models and makes them interchangeable.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime

class BaseModel(ABC):
    """
    Abstract base class for all trading models
    
    Each model must implement:
    - generate_entry_signals(): When to enter trades
    - generate_exit_signals(): When to exit trades
    - calculate_position_size(): How many shares to buy
    """
    
    def __init__(self, name: str, allocation_pct: float):
        """
        Initialize model
        
        Args:
            name: Unique identifier for this model
            allocation_pct: Percentage of portfolio allocated to this model (0.0-1.0)
        """
        self.name = name
        self.allocation_pct = allocation_pct
        self.enabled = True
        
        # Performance tracking (updated by portfolio manager)
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
    
    @abstractmethod
    def generate_entry_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        market_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Check if entry conditions are met at given index
        
        Args:
            df: Stock dataframe with indicators calculated
            idx: Current index position in dataframe
            market_df: Market index dataframe (SPY) for context
        
        Returns:
            Dictionary with:
            {
                'signal': bool,           # True if entry conditions met
                'entry_price': float,     # Price to enter at
                'confidence': float,      # Confidence 0.0-1.0
                'stop_loss': float,       # Initial stop loss price
                'metadata': dict          # Additional info for tracking
            }
        """
        pass
    
    @abstractmethod
    def generate_exit_signals(
        self, 
        df: pd.DataFrame, 
        idx: int,
        position: Dict
    ) -> Dict:
        """
        Check if exit conditions are met for an open position
        
        Args:
            df: Stock dataframe with indicators
            idx: Current index position
            position: Dict with position details:
                {
                    'entry_date': datetime,
                    'entry_price': float,
                    'shares': int,
                    'stop_loss': float,
                    'metadata': dict
                }
        
        Returns:
            Dictionary with:
            {
                'signal': bool,        # True if should exit
                'exit_price': float,   # Price to exit at
                'reason': str          # Exit reason code
            }
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        available_capital: float, 
        price: float,
        confidence: float = 1.0
    ) -> int:
        """
        Calculate number of shares to buy
        
        Args:
            available_capital: Capital available for this model
            price: Current stock price
            confidence: Signal confidence (0.0-1.0), can scale position size
        
        Returns:
            Number of shares to purchase (integer)
        """
        pass
    
    def get_model_info(self) -> Dict:
        """Return model configuration and stats"""
        return {
            'name': self.name,
            'allocation_pct': self.allocation_pct,
            'enabled': self.enabled,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'win_rate': (self.winning_trades / self.total_trades * 100) 
                       if self.total_trades > 0 else 0.0
        }
    
    def update_performance(self, pnl: float):
        """Update model performance tracking"""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
    
    def __str__(self):
        return f"{self.name} (Allocation: {self.allocation_pct*100:.0f}%)"
    
    def __repr__(self):
        return (f"<{self.__class__.__name__} "
                f"name='{self.name}' "
                f"allocation={self.allocation_pct:.1%} "
                f"enabled={self.enabled}>")


class ModelSignal:
    """Helper class to standardize signal outputs"""
    
    @staticmethod
    def no_signal() -> Dict:
        """Return no signal"""
        return {'signal': False}
    
    @staticmethod
    def entry_signal(
        entry_price: float,
        stop_loss: float,
        confidence: float = 0.8,
        **metadata
    ) -> Dict:
        """
        Create entry signal
        
        Args:
            entry_price: Price to enter position
            stop_loss: Initial stop loss price
            confidence: Signal confidence (0.0-1.0)
            **metadata: Additional tracking info
        """
        return {
            'signal': True,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'confidence': confidence,
            'metadata': metadata
        }
    
    @staticmethod
    def exit_signal(
        exit_price: float,
        reason: str,
        **metadata
    ) -> Dict:
        """
        Create exit signal
        
        Args:
            exit_price: Price to exit position
            reason: Exit reason code (e.g., 'STOP_LOSS', 'PROFIT_TARGET', 'STAGE_4')
            **metadata: Additional info
        """
        return {
            'signal': True,
            'exit_price': exit_price,
            'reason': reason,
            'metadata': metadata
        }
