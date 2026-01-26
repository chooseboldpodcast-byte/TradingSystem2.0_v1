# core/weinstein_engine.py
"""
Core Weinstein Stage Analysis Engine (Enhanced Version 3 - Phase 1)
Implements the 4-stage classification system with Phase 1 improvements

PHASE 1 ENHANCEMENTS:
- Market stage calculation for S&P 500 (capital deployment filter)
- Portfolio-level tracking support
- Methods to support multiple concurrent positions
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
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
    Enhanced with Phase 1 improvements for portfolio management
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
        volume_threshold: float = 1.3
    ) -> bool:
        """
        Detect if most recent price action is a valid Stage 2 breakout
        
        Requirements:
        1. Price breaks above Stage 1 resistance
        2. Volume surge (1.3x+ average)
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
            conditions['broke_resistance'] = current['close'] >= stage1_high*0.95
        else:
            conditions['broke_resistance'] = True

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
    
    def calculate_trailing_stop(
        self,
        df: pd.DataFrame,
        entry_price: float,
        current_price: float,
        highest_price: float,
        current_stop: float
    ) -> float:
        """
        Calculate trailing stop that rises with price but never falls
        
        Weinstein approach: Trail below 10-week MA once profitable
        
        Args:
            df: DataFrame with price data
            entry_price: Original entry price
            current_price: Current price
            highest_price: Highest price reached since entry
            current_stop: Current stop loss level
        
        Returns:
            New stop loss level (can only move up, never down)
        """
        # Calculate profit percentage
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        
        recent = df.tail(50)
        
        if profit_pct >= 20:
            # Up 20%+: Trail 3% below 10-week MA (tighter)
            ma_10week = recent['close'].rolling(window=50).mean().iloc[-1]
            trailing_stop = ma_10week * 0.97
            
        elif profit_pct >= 10:
            # Up 10-20%: Trail 5% below 10-week MA
            ma_10week = recent['close'].rolling(window=50).mean().iloc[-1]
            trailing_stop = ma_10week * 0.95
            
        elif profit_pct >= 5:
            # Up 5-10%: Trail 5% below 30-week MA
            ma_30week = recent['sma_30week'].iloc[-1]
            trailing_stop = ma_30week * 0.95
            
        else:
            # Not profitable enough yet, use original stop
            return current_stop
        
        # Trailing stop can only move UP, never down
        new_stop = max(trailing_stop, current_stop)
        
        return round(new_stop, 2)
    
    def detect_stage_3_from_stage_2(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Detect Stage 3 (topping/distribution) emerging from Stage 2
        
        Stage 3 indicators:
        - MA flattening (was rising)
        - Price oscillating around MA (no longer decisively above)
        - Volume declining
        - Coming from Stage 2
        
        Args:
            df: DataFrame with indicators
            idx: Current index
        
        Returns:
            True if Stage 3 detected
        """
        if idx < 60:
            return False
        
        # Check recent stage history
        recent_stages = df['stage'].iloc[idx-20:idx]
        had_stage_2 = (recent_stages == Stage.STAGE_2).any()
        
        if not had_stage_2:
            return False  # Only care if coming from Stage 2
        
        # Current conditions
        ma_slope = df['ma_slope_pct'].iloc[idx]
        close = df['close'].iloc[idx]
        ma = df['sma_30week'].iloc[idx]
        
        # MA flattening (was rising, now flat)
        ma_flattening = abs(ma_slope) < 0.15  # Less than 0.15% per week
        
        # Price near MA (within ±5%)
        price_to_ma_pct = ((close - ma) / ma) * 100
        price_near_ma = abs(price_to_ma_pct) < 5
        
        # Volume declining
        recent_20 = df.iloc[idx-20:idx+1]
        prior_20 = df.iloc[idx-40:idx-20+1]
        
        recent_vol = recent_20['volume'].mean()
        prior_vol = prior_20['volume'].mean()
        volume_declining = recent_vol < prior_vol * 0.85  # Down 15%+
        
        # All three conditions = Stage 3
        if ma_flattening and price_near_ma and volume_declining:
            return True
        
        return False
    
    def detect_distribution(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Detect distribution/topping patterns in Stage 2
        
        Distribution signals:
        1. Price making new highs, volume declining (bearish divergence)
        2. Multiple failed attempts at new highs
        3. Increasing volatility (whipsaws)
        
        Args:
            df: DataFrame with price data
            idx: Current index
        
        Returns:
            True if distribution detected
        """
        if df['stage'].iloc[idx] != Stage.STAGE_2:
            return False
        
        if idx < 40:
            return False
        
        recent_20 = df.iloc[idx-20:idx+1]
        prior_20 = df.iloc[idx-40:idx-20+1]
        
        # Signal 1: Price higher, volume lower (bearish divergence)
        recent_high = recent_20['high'].max()
        prior_high = prior_20['high'].max()
        
        recent_vol = recent_20['volume'].mean()
        prior_vol = prior_20['volume'].mean()
        
        bearish_divergence = (recent_high > prior_high and 
                            recent_vol < prior_vol * 0.8)
        
        # Signal 2: Multiple tops within 2% (resistance)
        highs_last_20 = recent_20.nlargest(5, 'high')['high'].values
        if len(highs_last_20) >= 3:
            high_range = (highs_last_20.max() - highs_last_20.min()) / highs_last_20.max()
            multiple_tops = high_range < 0.02  # All within 2%
        else:
            multiple_tops = False
        
        # Signal 3: Increasing volatility
        recent_volatility = (recent_20['high'] - recent_20['low']).mean() / recent_20['close'].mean()
        prior_volatility = (prior_20['high'] - prior_20['low']).mean() / prior_20['close'].mean()
        
        increasing_volatility = recent_volatility > prior_volatility * 1.4  # 40% increase
        
        # Any 2 of the 3 signals = distribution
        signals = sum([bearish_divergence, multiple_tops, increasing_volatility])
        
        return signals >= 2
    
    # ========== PHASE 1: NEW METHODS FOR MARKET STAGE & PORTFOLIO ==========
    
    def get_market_stage(self, index_df: pd.DataFrame, date) -> int:
        """
        Get market stage for a specific date
        
        Args:
            index_df: Market index DataFrame with stage calculated
            date: Date to check market stage
        
        Returns:
            Stage number (1, 2, 3, or 4)
        """
        try:
            if date in index_df.index:
                return index_df.loc[date, 'stage']
            else:
                # Find nearest date
                nearest_date = index_df.index[index_df.index <= date][-1]
                return index_df.loc[nearest_date, 'stage']
        except:
            return Stage.STAGE_1  # Default to conservative
    
    def get_capital_deployment_target(self, market_stage: int) -> float:
        """
        Get target capital deployment percentage based on market stage
        
        Weinstein principle: Deploy aggressively in Stage 2, conservatively otherwise
        
        Args:
            market_stage: Current market stage (1, 2, 3, or 4)
        
        Returns:
            Target deployment percentage (0.0 to 1.0)
        """
        deployment_map = {
            1: 0.20,  # Stage 1 - Conservative, early accumulation
            2: 0.90,  # Stage 2 - Aggressive, bull market
            3: 0.40,  # Stage 3 - Defensive, protect gains
            4: 0.10   # Stage 4 - Cash, preserve capital
        }
        
        # Convert Stage enum to int if needed
        stage_val = int(market_stage) if hasattr(market_stage, 'value') else market_stage
        
        return deployment_map.get(stage_val, 0.20)
    
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
    
    print("Testing Weinstein Engine v3 (Phase 1)...")
    
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
