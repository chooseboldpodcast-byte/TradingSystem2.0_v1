# core/weinstein_engine.py
"""
Core Weinstein Stage Analysis Engine - VERSION 4
Implements the COMPLETE 4-stage classification system with quality filters

NEW V4 FEATURES:
- Relative Strength filtering (Weinstein's #1 rule)
- Candlestick pattern confirmation
- Support level detection
- Complete Weinstein methodology
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from enum import IntEnum

class Stage(IntEnum):
    """Weinstein's Four Stages"""
    UNKNOWN = 0
    STAGE_1 = 1  # Basing/Accumulation
    STAGE_2 = 2  # Advancing/Markup
    STAGE_3 = 3  # Topping/Distribution
    STAGE_4 = 4  # Declining/Markdown

class WeinsteinEngine:
    """
    Implements Stan Weinstein's Stage Analysis methodology
    VERSION 4: Complete system with quality filters
    """
    
    def __init__(self, ma_period: int = 150):
        """
        Args:
            ma_period: Moving average period (150 days = 30 weeks)
        """
        self.ma_period = ma_period
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators needed for stage analysis
        
        Args:
            df: DataFrame with OHLCV data (index must be datetime)
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Calculate moving averages
        df['sma_30week'] = df['close'].rolling(window=self.ma_period).mean()
        df['sma_50day'] = df['close'].rolling(window=50).mean()
        df['sma_10week'] = df['close'].rolling(window=50).mean()
        
        # Calculate MA slope (rate of change)
        df['ma_slope'] = df['sma_30week'].diff(5)  # 5-day change
        df['ma_slope_pct'] = (df['ma_slope'] / df['sma_30week']) * 100
        
        # Volume indicators
        df['avg_volume_50'] = df['volume'].rolling(window=50).mean()
        df['volume_ratio'] = df['volume'] / df['avg_volume_50']
        
        # Price relative to MA
        df['price_to_ma'] = (df['close'] / df['sma_30week']) * 100
        
        return df
    
    def calculate_mansfield_rs(
        self,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        period: int = 200
    ) -> pd.Series:
        """
        Calculate Mansfield Relative Strength vs market index
        
        Args:
            stock_df: Stock price data
            index_df: Market index data (e.g., S&P 500)
            period: Period for RS moving average (200 days)
        
        Returns:
            Series with Mansfield RS values
        """
        # Align dataframes by date
        combined = pd.DataFrame({
            'stock': stock_df['close'],
            'index': index_df['close']
        }).dropna()
        
        if len(combined) < period:
            return pd.Series(0, index=stock_df.index)
        
        # Calculate price relative (PR)
        pr = (combined['stock'] / combined['index']) * 100
        
        # Calculate 200-day SMA of PR
        pr_sma = pr.rolling(window=period).mean()
        
        # Calculate Mansfield RS
        mansfield_rs = ((pr / pr_sma) - 1) * 100
        
        return mansfield_rs
    
    # ========== NEW V4 METHOD: RELATIVE STRENGTH FILTER ==========
    def has_positive_relative_strength(
        self,
        df: pd.DataFrame,
        idx: int
    ) -> Tuple[bool, float]:
        """
        Check if stock has positive relative strength vs market
        
        WEINSTEIN'S #1 RULE: Only trade stocks outperforming the market
        
        Args:
            df: DataFrame with mansfield_rs column
            idx: Current index
        
        Returns:
            (has_positive_rs, rs_value)
        """
        if 'mansfield_rs' not in df.columns:
            return True, 0.0  # No RS data, don't filter
        
        if idx < 200:
            return True, 0.0  # Not enough data yet
        
        # Get current RS value
        rs = df['mansfield_rs'].iloc[idx]
        
        if pd.isna(rs):
            return True, 0.0
        
        # Weinstein rule: RS must be > 0 (outperforming market)
        return rs > 0, float(rs)
    
    # ========== NEW V4 METHOD: CANDLESTICK CONFIRMATION ==========
    def detect_bullish_candle(
        self,
        df: pd.DataFrame,
        idx: int
    ) -> Tuple[bool, str]:
        """
        Detect if current candle is bullish confirmation
        
        Strong bullish candle = high conviction entry
        
        Args:
            df: DataFrame with OHLC data
            idx: Current index
        
        Returns:
            (is_bullish, pattern_name)
        """
        if idx < 1:
            return False, "INSUFFICIENT_DATA"
        
        candle = df.iloc[idx]
        prev_candle = df.iloc[idx-1]
        
        open_price = candle['open']
        close_price = candle['close']
        high = candle['high']
        low = candle['low']
        
        # Must be bullish (close > open)
        if close_price <= open_price:
            return False, "BEARISH"
        
        # Calculate candle characteristics
        body = close_price - open_price
        upper_wick = high - close_price
        lower_wick = open_price - low
        total_range = high - low
        
        if total_range == 0:
            return False, "DOJI"
        
        body_pct = body / total_range
        lower_wick_pct = lower_wick / total_range
        
        # Pattern 1: HAMMER (strongest - bouncing off support)
        # Long lower wick (2x body), small upper wick, bullish close
        if (lower_wick >= body * 2 and 
            upper_wick < body * 0.5 and
            body_pct >= 0.25):
            return True, "HAMMER"
        
        # Pattern 2: BULLISH ENGULFING (very strong)
        # Current candle completely engulfs previous
        if (open_price < prev_candle['close'] and
            close_price > prev_candle['open'] and
            body > (prev_candle['close'] - prev_candle['open']) * 1.2):
            return True, "BULLISH_ENGULFING"
        
        # Pattern 3: STRONG BULLISH CANDLE (good conviction)
        # Body is 50%+ of range, closes near high
        if (body_pct >= 0.5 and 
            upper_wick < body * 0.3):
            return True, "STRONG_BULLISH"
        
        # Pattern 4: ACCEPTABLE BULLISH (minimum)
        # Body is 40%+ of range
        if body_pct >= 0.4:
            return True, "BULLISH"
        
        # Weak candle - no conviction
        return False, "WEAK_BULLISH"
    
    # ========== NEW V4 METHOD: SUPPORT LEVEL DETECTION ==========
    def detect_support_level(
        self,
        df: pd.DataFrame,
        idx: int
    ) -> Tuple[bool, float]:
        """
        Detect if current price is at/near support level
        
        Support = recent lows where buying pressure appeared
        Better entries at support = better risk/reward
        
        Args:
            df: DataFrame with price data
            idx: Current index
        
        Returns:
            (at_support, support_level)
        """
        if idx < 60:
            return False, 0.0
        
        current_price = df['close'].iloc[idx]
        
        # Look back 20-60 days for support
        lookback_min = 20
        lookback_max = 60
        recent = df.iloc[max(0, idx-lookback_max):idx]
        
        # Find the 3 lowest lows in this period
        lowest_lows = recent.nsmallest(3, 'low')['low'].values
        
        if len(lowest_lows) == 0:
            return False, 0.0
        
        # Support level = average of these lows
        support_level = np.mean(lowest_lows)
        
        # Check if current price is near support (within 3%)
        distance_pct = abs((current_price - support_level) / support_level) * 100
        
        at_support = distance_pct <= 3.0
        
        return at_support, float(support_level)
    
    def classify_stage(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify the stage for each row in the dataframe
        
        Args:
            df: DataFrame with indicators already calculated
        
        Returns:
            Series with stage classification (1, 2, 3, or 4)
        """
        # Need sufficient data for classification
        if len(df) < self.ma_period:
            return pd.Series(Stage.UNKNOWN, index=df.index)
        
        # IMPORTANT: Work with a copy and initialize stage column
        df_copy = df.copy()
        df_copy['stage'] = Stage.UNKNOWN
        
        # Calculate for each valid row
        for i in range(self.ma_period, len(df_copy)):
            stage = self._classify_single_row(df_copy, i)
            df_copy.loc[df_copy.index[i], 'stage'] = stage
        
        # Return just the stage series
        return df_copy['stage']

    def _classify_single_row(self, df: pd.DataFrame, idx: int) -> int:
        """
        Classify stage for a single row
        
        Stage Rules:
        - Stage 1: Price oscillates around flat MA
        - Stage 2: Price above rising MA, making higher highs/lows
        - Stage 3: Price oscillates around flat MA (after Stage 2)
        - Stage 4: Price below declining MA, making lower highs/lows
        """
        # Current values
        close = df['close'].iloc[idx]
        ma = df['sma_30week'].iloc[idx]
        ma_slope_pct = df['ma_slope_pct'].iloc[idx]
        
        # Get recent data for trend analysis
        lookback = min(50, idx)
        recent = df.iloc[idx-lookback:idx+1]
        
        # Price position
        price_above_ma = close > ma
        
        # MA direction thresholds
        MA_FLAT_THRESHOLD = 0.1  # ±0.1% per week considered flat
        ma_rising = ma_slope_pct > MA_FLAT_THRESHOLD
        ma_flat = abs(ma_slope_pct) <= MA_FLAT_THRESHOLD
        ma_falling = ma_slope_pct < -MA_FLAT_THRESHOLD
        
        # Trend analysis
        higher_highs, higher_lows = self._check_trend(recent)
        lower_highs, lower_lows = self._check_downtrend(recent)
        
        # Stage classification logic
        if price_above_ma and ma_rising and higher_highs and higher_lows:
            # Strong uptrend: Stage 2
            return Stage.STAGE_2
        
        elif not price_above_ma and ma_falling and lower_highs and lower_lows:
            # Strong downtrend: Stage 4
            return Stage.STAGE_4
        
        elif ma_flat:
            # Sideways: Stage 1 or 3
            if 'stage' in df.columns:
                prev_stages = df['stage'].iloc[max(0, idx-50):idx]
                valid_stages = prev_stages[prev_stages != Stage.UNKNOWN]
                
                if len(valid_stages) > 0:
                    stage_4_count = (valid_stages == Stage.STAGE_4).sum()
                    stage_2_count = (valid_stages == Stage.STAGE_2).sum()
                    
                    if stage_4_count > stage_2_count:
                        return Stage.STAGE_1
                    elif stage_2_count > 0:
                        return Stage.STAGE_3
            
            return Stage.STAGE_1
        
        else:
            # Transitional phase
            if 'stage' in df.columns and idx > 0:
                prev_stage = df['stage'].iloc[idx-1]
                if prev_stage != Stage.UNKNOWN:
                    return prev_stage
            return Stage.STAGE_1
    
    def _check_trend(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Check if price is making higher highs and higher lows"""
        if len(df) < 10:
            return False, False
        
        highs = df.nlargest(5, 'high')['high'].values
        lows = df.nsmallest(5, 'low')['low'].values
        
        highs_sorted = sorted(highs)
        lows_sorted = sorted(lows)
        
        higher_highs = highs_sorted[-1] > np.mean(highs_sorted[:-1])
        higher_lows = lows_sorted[-1] > np.mean(lows_sorted[:-1])
        
        return higher_highs, higher_lows
    
    def _check_downtrend(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Check if price is making lower highs and lower lows"""
        if len(df) < 10:
            return False, False
        
        highs = df.nlargest(5, 'high')['high'].values
        lows = df.nsmallest(5, 'low')['low'].values
        
        highs_sorted = sorted(highs, reverse=True)
        lows_sorted = sorted(lows, reverse=True)
        
        lower_highs = highs_sorted[-1] < np.mean(highs_sorted[:-1])
        lower_lows = lows_sorted[-1] < np.mean(lows_sorted[:-1])
        
        return lower_highs, lower_lows
    
    def detect_stage_2_breakout(
        self,
        df: pd.DataFrame,
        volume_threshold: float = 1.3
    ) -> bool:
        """
        Detect if most recent price action is a valid Stage 2 breakout
        
        Requirements:
        1. Price breaks above Stage 1 resistance
        2. Volume surge
        3. Price above 30-week MA
        4. MA starting to rise
        """
        if len(df) < self.ma_period + 10:
            return False
        
        recent = df.tail(10)
        current = recent.iloc[-1]
        
        conditions = {
            'price_above_ma': current['close'] > current['sma_30week'],
            'ma_rising': current['ma_slope_pct'] > 0,
            'volume_surge': current['volume_ratio'] >= volume_threshold,
            'stage_transition': False
        }
        
        if len(recent) >= 2:
            prev_stage = recent['stage'].iloc[-2]
            curr_stage = recent['stage'].iloc[-1]
            conditions['stage_transition'] = (
                prev_stage == Stage.STAGE_1 and 
                curr_stage == Stage.STAGE_2
            )
        
        stage1_data = df[df['stage'] == Stage.STAGE_1].tail(60)
        if len(stage1_data) > 0:
            stage1_high = stage1_data['high'].max()
            conditions['broke_resistance'] = current['close'] >= stage1_high*0.95
        else:
            conditions['broke_resistance'] = True

        return all(conditions.values())
    
    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        entry_price: float,
        method: str = 'weinstein'
    ) -> float:
        """Calculate stop loss level using Weinstein method"""
        if method == 'percentage':
            return entry_price * 0.92
        
        elif method == 'weinstein':
            recent = df.tail(50)
            recent_low = recent['low'].min()
            current_ma = recent['sma_30week'].iloc[-1]
            
            stop = min(recent_low, current_ma)
            stop = np.floor(stop) - 0.07
            stop = min(stop, entry_price * 0.95)
            
            return round(stop, 2)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def analyze_stock(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Complete analysis of a stock
        
        Args:
            df: Stock price data
            index_df: Market index data for RS calculation
        
        Returns:
            DataFrame with all indicators and stages
        """
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Calculate Mansfield RS if index provided
        if index_df is not None:
            df['mansfield_rs'] = self.calculate_mansfield_rs(df, index_df)
        
        # Classify stages
        df['stage'] = self.classify_stage(df)
        
        return df
