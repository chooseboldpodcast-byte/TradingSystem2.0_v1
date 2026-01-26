# core/weinstein_engine.py
"""
Core Weinstein Stage Analysis Engine. Implements the 4-stage classification system.
This is a conservative model with the 1.5x volume surge, without the pullback method implemented, and +/-5% breakout from resistence levels
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
        
        # Calculate price relative (PR)
        pr = (combined['stock'] / combined['index']) * 100
        
        # Calculate 200-day SMA of PR
        pr_sma = pr.rolling(window=period).mean()
        
        # Calculate Mansfield RS
        mansfield_rs = ((pr / pr_sma) - 1) * 100
        
        return mansfield_rs
    
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
            # df_copy['stage'].iloc[i] = stage
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
            # Look at recent history to determine which
            # Only check if stage column exists and has been populated
            if 'stage' in df.columns:
                prev_stages = df['stage'].iloc[max(0, idx-50):idx]
                
                # Count only non-zero stages
                valid_stages = prev_stages[prev_stages != Stage.UNKNOWN]
                
                if len(valid_stages) > 0:
                    # Count recent stages
                    stage_4_count = (valid_stages == Stage.STAGE_4).sum()
                    stage_2_count = (valid_stages == Stage.STAGE_2).sum()
                    
                    if stage_4_count > stage_2_count:
                        # Coming from decline: Stage 1 (Basing)
                        return Stage.STAGE_1
                    elif stage_2_count > 0:
                        # Coming from advance: Stage 3 (Topping)
                        return Stage.STAGE_3
            
            # Default to Stage 1 if no clear history
            return Stage.STAGE_1
        
        else:
            # Transitional phase - use previous stage if available
            if 'stage' in df.columns and idx > 0:
                prev_stage = df['stage'].iloc[idx-1]
                if prev_stage != Stage.UNKNOWN:
                    return prev_stage
            return Stage.STAGE_1
    
    def _check_trend(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Check if price is making higher highs and higher lows
        
        Returns:
            (higher_highs, higher_lows)
        """
        if len(df) < 10:
            return False, False
        
        # Get significant highs and lows
        highs = df.nlargest(5, 'high')['high'].values
        lows = df.nsmallest(5, 'low')['low'].values
        
        # Check if trending higher
        # Sort by date and check if generally increasing
        highs_sorted = sorted(highs)
        lows_sorted = sorted(lows)
        
        # Simple check: most recent high > average of earlier highs
        higher_highs = highs_sorted[-1] > np.mean(highs_sorted[:-1])
        higher_lows = lows_sorted[-1] > np.mean(lows_sorted[:-1])
        
        return higher_highs, higher_lows
    
    def _check_downtrend(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Check if price is making lower highs and lower lows
        
        Returns:
            (lower_highs, lower_lows)
        """
        if len(df) < 10:
            return False, False
        
        highs = df.nlargest(5, 'high')['high'].values
        lows = df.nsmallest(5, 'low')['low'].values
        
        highs_sorted = sorted(highs, reverse=True)
        lows_sorted = sorted(lows, reverse=True)
        
        # Most recent high < average of earlier highs
        lower_highs = highs_sorted[-1] < np.mean(highs_sorted[:-1])
        lower_lows = lows_sorted[-1] < np.mean(lows_sorted[:-1])
        
        return lower_highs, lower_lows
    
    def detect_stage_2_breakout(
        self,
        df: pd.DataFrame,
        volume_threshold: float = 2.0
        #debug: bool = False # add debug output to breakout detection
    ) -> bool:
        """
        Detect if most recent price action is a valid Stage 2 breakout
        
        Requirements:
        1. Price breaks above Stage 1 resistance
        2. Volume surge (2x+ average)
        3. Price above 30-week MA
        4. MA starting to rise
        
        Args:
            df: DataFrame with indicators and stages
            volume_threshold: Minimum volume ratio required
        
        Returns:
            True if valid Stage 2 breakout detected
        """
        if len(df) < self.ma_period + 10:
            return False
        
        # Get recent data
        recent = df.tail(10)
        current = recent.iloc[-1]
        
        # Check conditions
        conditions = {
            'price_above_ma': current['close'] > current['sma_30week'],
            'ma_rising': current['ma_slope_pct'] > 0,
            'volume_surge': current['volume_ratio'] >= volume_threshold,
            'stage_transition': False
        }
        
        # Check for Stage 1 -> Stage 2 transition
        if len(recent) >= 2:
            prev_stage = recent['stage'].iloc[-2]
            curr_stage = recent['stage'].iloc[-1]
            conditions['stage_transition'] = (
                prev_stage == Stage.STAGE_1 and 
                curr_stage == Stage.STAGE_2
            )
        
        # Check if broke above Stage 1 resistance
        stage1_data = df[df['stage'] == Stage.STAGE_1].tail(60)
        if len(stage1_data) > 0:
            stage1_high = stage1_data['high'].max()
            # Accept if within 2% of resistance and >= (breakouts often happen AT resistance)
            conditions['broke_resistance'] = current['close'] >= stage1_high*0.95
        else:
            conditions['broke_resistance'] = True
            
        # DEBUG: Show why breakouts are rejected
        if not all(conditions.values()):
            failed = [k for k, v in conditions.items() if not v]
            print(f"    Breakout REJECTED. Failed: {failed}")
            if 'volume_ratio' in current.index:
                print(f"    Volume: {current['volume_ratio']:.2f}x (need {volume_threshold}x)")


        # All conditions must be True
        return all(conditions.values())
    
    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        entry_price: float,
        method: str = 'weinstein'
    ) -> float:
        """
        Calculate stop loss level using Weinstein method
        
        Args:
            df: DataFrame with price data
            entry_price: Entry price
            method: 'weinstein' or 'percentage'
        
        Returns:
            Stop loss price
        """
        if method == 'percentage':
            # Simple 8% stop
            return entry_price * 0.92
        
        elif method == 'weinstein':
            # Below 30-week MA and recent minor low
            recent = df.tail(50)
            
            # Recent low
            recent_low = recent['low'].min()
            
            # Current 30-week MA
            current_ma = recent['sma_30week'].iloc[-1]
            
            # Stop below both
            stop = min(recent_low, current_ma)
            
            # Round down to below whole dollar
            stop = np.floor(stop) - 0.07
            
            # Don't set stop above entry price
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


if __name__ == "__main__":
    # Quick test
    import yfinance as yf
    
    print("Testing Weinstein Engine...")
    
    # Download test data
    aapl = yf.Ticker("AAPL")
    df = aapl.history(period="2y")
    df = df.rename(columns={'Close': 'close', 'Open': 'open', 
                            'High': 'high', 'Low': 'low', 'Volume': 'volume'})
    
    # Initialize engine
    engine = WeinsteinEngine()
    
    # Analyze
    df = engine.analyze_stock(df)
    
    # Print results
    print(f"\nAnalysis complete!")
    print(f"Total days: {len(df)}")
    print(f"\nCurrent Status:")
    print(f"Price: ${df['close'].iloc[-1]:.2f}")
    print(f"30-Week MA: ${df['sma_30week'].iloc[-1]:.2f}")
    print(f"Stage: {df['stage'].iloc[-1]}")
    print(f"\nLast 10 days:")
    print(df[['close', 'sma_30week', 'stage']].tail(10))
