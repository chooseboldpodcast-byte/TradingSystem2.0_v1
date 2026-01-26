#!/usr/bin/env python3
# scripts/stage1_to2_breakout_monitor.py
"""
Stage 1→2 Breakout Monitor
==========================

Monitors stocks for Weinstein Stage 1→2 transition breakouts using strict criteria:
- Medium/Long base duration (6-12+ months)
- Volume 2x+ average on breakout
- Price above last significant swing high
- Sector in Stage 2 with positive RS
- Individual stock outperforming sector
- RS line improving (higher highs/lows) before price breakout

Based on Stan Weinstein's "Secrets for Profiting in Bull and Bear Markets"

Usage:
    python scripts/stage1_to2_breakout_monitor.py --scan           # Scan for candidates
    python scripts/stage1_to2_breakout_monitor.py --scan --save    # Scan and save to DB
    python scripts/stage1_to2_breakout_monitor.py --alerts         # Show active alerts
    python scripts/stage1_to2_breakout_monitor.py --email          # Send email alerts

Author: Trading System
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import argparse
import yaml
import json
import sqlite3
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.weinstein_engine import WeinsteinEngine, Stage

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'stage1_to2_breakout': {
        'enabled': True,
        # Base duration requirements
        'min_base_days': 120,          # 6 months minimum (medium base)
        'ideal_base_days': 252,        # 12 months (long base - more powerful)
        'max_base_days': 756,          # 3 years max
        # Breakout confirmation
        'min_volume_ratio': 2.0,       # Must be 2x average volume
        'volume_lookback': 50,         # 50-day average volume
        # Price requirements
        'breakout_buffer_pct': 0.01,   # 1% above resistance to confirm
        'max_extension_pct': 0.05,     # Don't chase >5% extended
        # Relative Strength requirements
        'min_rs_rating': 0,            # Minimum Mansfield RS (positive = outperforming)
        'rs_improving_days': 20,       # RS must show improvement over X days
        # Sector requirements
        'sector_stage2_required': True,
        # Risk management
        'stop_loss_method': 'weinstein',  # 'weinstein' or 'percentage'
        'stop_loss_pct': 0.08,         # 8% fallback stop
        # Watchlist
        'custom_watchlist': [
            # Drone ecosystem
            'FEIM', 'COHU', 'TTMI', 'DCO', 'MTSI',
            # AI Energy / Nuclear
            'LEU', 'MOD', 'POWL', 'BWXT', 'UEC',
            'DNN', 'AMBA', 'ACLS', 'UUUU', 'NVT',
            # Natural Gas
            'TLN', 'EQT', 'AR', 'KMI', 'KGS',
            # Additional candidates
            'CCJ', 'SMR', 'OKLO', 'TER', 'ST'
        ]
    }
}


@dataclass
class BreakoutCandidate:
    """Represents a Stage 1→2 breakout candidate"""
    symbol: str
    scan_date: datetime
    stage1_start_date: datetime
    stage1_duration_days: int
    breakout_date: Optional[datetime]
    entry_price: float
    stop_loss: float
    stage1_high: float
    stage1_low: float
    consolidation_range_pct: float
    current_volume_ratio: float
    rs_rating: float
    rs_improving: bool
    rs_trend_days: int
    breakout_confirmed: bool
    confidence: float
    status: str  # 'WATCHING', 'BREAKOUT_PENDING', 'CONFIRMED', 'FAILED'
    metadata: Dict


class Stage1To2Monitor:
    """
    Monitors stocks for Stage 1→2 transitions

    The key insight from Weinstein: The longer the base, the bigger the breakout.
    We want to catch stocks emerging from extended Stage 1 bases with:
    - Rising RS (outperforming market)
    - Volume confirmation
    - Clear breakout above resistance
    """

    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG['stage1_to2_breakout']
        self.engine = WeinsteinEngine(ma_period=150)
        self.candidates: List[BreakoutCandidate] = []

    def load_stock_data(self, symbols: List[str], data_dir: str = 'data') -> Tuple[Dict, pd.DataFrame]:
        """Load stock data and index data"""
        data = {}
        index_data = None

        # Always load SPY for RS calculation
        spy_path = os.path.join(data_dir, 'SPY.csv')
        if os.path.exists(spy_path):
            index_data = pd.read_csv(spy_path, index_col=0, parse_dates=True)
            if isinstance(index_data.index, pd.DatetimeIndex) and index_data.index.tz is not None:
                index_data.index = index_data.index.tz_localize(None)

        for symbol in symbols:
            filepath = os.path.join(data_dir, f"{symbol}.csv")
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    if len(df) >= 252:  # Need at least 1 year of data
                        data[symbol] = df
                except Exception as e:
                    print(f"Warning: Could not load {symbol}: {e}")

        return data, index_data

    def detect_stage1_base(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        Detect if stock is in or emerging from a Stage 1 base

        Stage 1 characteristics:
        - Price oscillates around flat 30-week MA
        - Volume typically declining during base
        - Can last months to years
        - Ends with volume surge and breakout above resistance
        """
        if idx < 252:
            return {'in_stage1': False, 'reason': 'Insufficient history'}

        # Get stage classifications
        stages = df['stage'].iloc[max(0, idx-504):idx+1]  # 2 years lookback

        # Find Stage 1 periods
        stage1_mask = stages == Stage.STAGE_1

        if not stage1_mask.any():
            return {'in_stage1': False, 'reason': 'No Stage 1 detected'}

        # Find the start of the current/recent Stage 1
        # Work backwards from current position
        stage1_end_idx = None
        stage1_start_idx = None

        for i in range(len(stages)-1, -1, -1):
            if stages.iloc[i] == Stage.STAGE_1:
                if stage1_end_idx is None:
                    stage1_end_idx = i
                stage1_start_idx = i
            elif stage1_end_idx is not None:
                # Found end of Stage 1 period
                break

        if stage1_start_idx is None:
            return {'in_stage1': False, 'reason': 'No Stage 1 period found'}

        # Calculate base duration
        duration = stage1_end_idx - stage1_start_idx + 1

        # Get price range during Stage 1
        stage1_data = df.iloc[max(0, idx-duration-50):idx+1]
        stage1_high = stage1_data['high'].max()
        stage1_low = stage1_data['low'].min()
        consolidation_range = ((stage1_high - stage1_low) / stage1_low) * 100

        # Check if base meets minimum duration
        min_days = self.config.get('min_base_days', 120)

        current_stage = df['stage'].iloc[idx] if 'stage' in df.columns else Stage.UNKNOWN

        return {
            'in_stage1': current_stage == Stage.STAGE_1 or duration >= min_days,
            'current_stage': int(current_stage),
            'stage1_start_idx': stage1_start_idx,
            'stage1_end_idx': stage1_end_idx,
            'duration_days': duration,
            'stage1_high': stage1_high,
            'stage1_low': stage1_low,
            'consolidation_range_pct': consolidation_range,
            'base_quality': self._assess_base_quality(duration, consolidation_range)
        }

    def _assess_base_quality(self, duration: int, range_pct: float) -> str:
        """Assess the quality of the base formation"""
        # Longer bases with tighter consolidation are better
        if duration >= 252 and range_pct <= 25:
            return 'EXCELLENT'
        elif duration >= 180 and range_pct <= 30:
            return 'GOOD'
        elif duration >= 120 and range_pct <= 40:
            return 'ACCEPTABLE'
        else:
            return 'WEAK'

    def check_volume_confirmation(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        Check if volume confirms breakout

        Weinstein's rule: Volume should surge on breakout (ideally 2x+ average)
        """
        if idx < 50:
            return {'confirmed': False, 'ratio': 0, 'reason': 'Insufficient data'}

        current_volume = df['volume'].iloc[idx]
        avg_volume = df['volume'].iloc[idx-50:idx].mean()

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        min_ratio = self.config.get('min_volume_ratio', 2.0)

        return {
            'confirmed': volume_ratio >= min_ratio,
            'ratio': volume_ratio,
            'min_required': min_ratio,
            'avg_volume_50d': avg_volume,
            'current_volume': current_volume
        }

    def check_rs_improvement(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        Check if Relative Strength is improving

        Key Weinstein insight: RS line often breaks out BEFORE price
        Look for RS making higher highs and higher lows
        """
        if 'mansfield_rs' not in df.columns or idx < 60:
            return {'improving': False, 'current_rs': 0, 'reason': 'No RS data'}

        rs = df['mansfield_rs'].iloc[max(0, idx-60):idx+1]

        if rs.isna().all():
            return {'improving': False, 'current_rs': 0, 'reason': 'RS data is NaN'}

        current_rs = rs.iloc[-1] if not pd.isna(rs.iloc[-1]) else 0

        # Check RS trend over last 20 days
        lookback = self.config.get('rs_improving_days', 20)
        recent_rs = rs.tail(lookback)

        if len(recent_rs) < lookback:
            return {'improving': False, 'current_rs': current_rs, 'reason': 'Insufficient RS history'}

        # Calculate RS trend
        rs_start = recent_rs.iloc[:5].mean()
        rs_end = recent_rs.iloc[-5:].mean()
        rs_change = rs_end - rs_start

        # Check for higher highs and higher lows in RS
        rs_highs = recent_rs.rolling(5).max()
        rs_lows = recent_rs.rolling(5).min()

        higher_highs = rs_highs.iloc[-1] > rs_highs.iloc[5] if len(rs_highs) > 5 else False
        higher_lows = rs_lows.iloc[-1] > rs_lows.iloc[5] if len(rs_lows) > 5 else False

        improving = (rs_change > 0) and (higher_highs or higher_lows)

        # RS should be positive (outperforming market)
        min_rs = self.config.get('min_rs_rating', 0)

        return {
            'improving': improving,
            'current_rs': current_rs,
            'rs_change': rs_change,
            'higher_highs': higher_highs,
            'higher_lows': higher_lows,
            'above_min': current_rs >= min_rs,
            'rs_20d_ago': rs_start,
            'rs_trend': 'IMPROVING' if improving else 'FLAT' if abs(rs_change) < 2 else 'DECLINING'
        }

    def check_breakout(self, df: pd.DataFrame, idx: int, stage1_info: Dict) -> Dict:
        """
        Check if price is breaking out of Stage 1

        Breakout confirmation:
        1. Price closes above Stage 1 high (resistance)
        2. Volume surge (2x+)
        3. Close in upper portion of day's range
        """
        if not stage1_info.get('in_stage1'):
            return {'breakout': False, 'reason': 'Not in Stage 1'}

        current = df.iloc[idx]
        stage1_high = stage1_info['stage1_high']

        buffer_pct = self.config.get('breakout_buffer_pct', 0.01)
        breakout_level = stage1_high * (1 + buffer_pct)

        # Check if price broke above resistance
        price_breakout = current['close'] >= breakout_level

        # Check if not too extended
        max_ext = self.config.get('max_extension_pct', 0.05)
        max_price = stage1_high * (1 + max_ext)
        too_extended = current['close'] > max_price

        # Check candle quality (close in upper half of range)
        day_range = current['high'] - current['low']
        if day_range > 0:
            close_position = (current['close'] - current['low']) / day_range
            strong_close = close_position >= 0.5
        else:
            strong_close = True

        return {
            'breakout': price_breakout and not too_extended,
            'price_above_resistance': price_breakout,
            'too_extended': too_extended,
            'strong_close': strong_close,
            'breakout_level': breakout_level,
            'current_price': current['close'],
            'distance_from_breakout_pct': ((current['close'] / stage1_high) - 1) * 100
        }

    def calculate_stop_loss(self, df: pd.DataFrame, entry_price: float, stage1_info: Dict) -> float:
        """Calculate stop loss using Weinstein method"""
        method = self.config.get('stop_loss_method', 'weinstein')

        if method == 'weinstein':
            # Stop below Stage 1 low or 30-week MA, whichever is higher
            stage1_low = stage1_info.get('stage1_low', entry_price * 0.92)
            current_ma = df['sma_30week'].iloc[-1] if 'sma_30week' in df.columns else entry_price * 0.95

            # Use the higher of the two (tighter stop, less risk)
            stop = max(stage1_low * 0.98, current_ma * 0.97)

            # But don't allow more than 8% loss
            max_stop = entry_price * (1 - self.config.get('stop_loss_pct', 0.08))
            stop = max(stop, max_stop)

            return round(stop, 2)
        else:
            # Percentage-based stop
            return round(entry_price * (1 - self.config.get('stop_loss_pct', 0.08)), 2)

    def calculate_confidence(self, stage1_info: Dict, volume_info: Dict,
                            rs_info: Dict, breakout_info: Dict) -> float:
        """
        Calculate confidence score for the setup (0-1)

        Weights:
        - Base quality: 25%
        - Volume confirmation: 25%
        - RS improvement: 25%
        - Breakout quality: 25%
        """
        score = 0.0

        # Base quality (25%)
        base_quality = stage1_info.get('base_quality', 'WEAK')
        base_scores = {'EXCELLENT': 25, 'GOOD': 20, 'ACCEPTABLE': 15, 'WEAK': 5}
        score += base_scores.get(base_quality, 5)

        # Volume confirmation (25%)
        vol_ratio = volume_info.get('ratio', 0)
        if vol_ratio >= 3.0:
            score += 25
        elif vol_ratio >= 2.5:
            score += 22
        elif vol_ratio >= 2.0:
            score += 18
        elif vol_ratio >= 1.5:
            score += 12
        else:
            score += 5

        # RS improvement (25%)
        if rs_info.get('improving') and rs_info.get('above_min'):
            score += 25
        elif rs_info.get('above_min'):
            score += 15
        elif rs_info.get('improving'):
            score += 10
        else:
            score += 5

        # Breakout quality (25%)
        if breakout_info.get('breakout') and breakout_info.get('strong_close'):
            score += 25
        elif breakout_info.get('breakout'):
            score += 18
        elif breakout_info.get('price_above_resistance'):
            score += 10
        else:
            score += 5

        return round(score / 100, 2)

    def scan_stock(self, symbol: str, df: pd.DataFrame, index_df: pd.DataFrame) -> Optional[BreakoutCandidate]:
        """Scan a single stock for Stage 1→2 breakout setup"""

        # Analyze with Weinstein engine
        try:
            analyzed = self.engine.analyze_stock(df, index_df)
        except Exception as e:
            print(f"  Warning: Could not analyze {symbol}: {e}")
            return None

        idx = len(analyzed) - 1
        if idx < 252:
            return None

        # Check Stage 1 base
        stage1_info = self.detect_stage1_base(analyzed, idx)

        if not stage1_info.get('in_stage1') and stage1_info.get('duration_days', 0) < self.config.get('min_base_days', 120):
            return None

        # Check volume
        volume_info = self.check_volume_confirmation(analyzed, idx)

        # Check RS
        rs_info = self.check_rs_improvement(analyzed, idx)

        # Check breakout
        breakout_info = self.check_breakout(analyzed, idx, stage1_info)

        # Calculate confidence
        confidence = self.calculate_confidence(stage1_info, volume_info, rs_info, breakout_info)

        # Determine status
        if breakout_info.get('breakout') and volume_info.get('confirmed'):
            status = 'CONFIRMED'
        elif breakout_info.get('price_above_resistance'):
            status = 'BREAKOUT_PENDING'
        elif stage1_info.get('base_quality') in ['EXCELLENT', 'GOOD'] and rs_info.get('improving'):
            status = 'WATCHING'
        else:
            return None  # Not interesting enough

        current = analyzed.iloc[idx]
        entry_price = current['close']
        stop_loss = self.calculate_stop_loss(analyzed, entry_price, stage1_info)

        # Calculate Stage 1 start date
        stage1_start_idx = stage1_info.get('stage1_start_idx', 0)
        stage1_start_date = analyzed.index[max(0, idx - stage1_info.get('duration_days', 0))]

        return BreakoutCandidate(
            symbol=symbol,
            scan_date=datetime.now(),
            stage1_start_date=stage1_start_date,
            stage1_duration_days=stage1_info.get('duration_days', 0),
            breakout_date=datetime.now() if status == 'CONFIRMED' else None,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stage1_high=stage1_info.get('stage1_high', 0),
            stage1_low=stage1_info.get('stage1_low', 0),
            consolidation_range_pct=stage1_info.get('consolidation_range_pct', 0),
            current_volume_ratio=volume_info.get('ratio', 0),
            rs_rating=rs_info.get('current_rs', 0),
            rs_improving=rs_info.get('improving', False),
            rs_trend_days=self.config.get('rs_improving_days', 20),
            breakout_confirmed=status == 'CONFIRMED',
            confidence=confidence,
            status=status,
            metadata={
                'stage1_info': stage1_info,
                'volume_info': volume_info,
                'rs_info': rs_info,
                'breakout_info': breakout_info
            }
        )

    def scan_universe(self, symbols: List[str], data_dir: str = 'data') -> List[BreakoutCandidate]:
        """Scan entire universe for Stage 1→2 candidates"""
        print(f"\n{'='*60}")
        print("STAGE 1→2 BREAKOUT SCANNER")
        print(f"{'='*60}")
        print(f"Scanning {len(symbols)} symbols...")
        print(f"Min base duration: {self.config.get('min_base_days', 120)} days")
        print(f"Min volume ratio: {self.config.get('min_volume_ratio', 2.0)}x")
        print(f"{'='*60}\n")

        # Load data
        data, index_df = self.load_stock_data(symbols, data_dir)
        print(f"Loaded {len(data)} stocks with sufficient history\n")

        candidates = []

        for symbol in symbols:
            if symbol not in data:
                continue

            df = data[symbol]
            candidate = self.scan_stock(symbol, df, index_df)

            if candidate:
                candidates.append(candidate)
                print(f"  ✓ {symbol}: {candidate.status} | "
                      f"Base: {candidate.stage1_duration_days}d | "
                      f"Vol: {candidate.current_volume_ratio:.1f}x | "
                      f"RS: {candidate.rs_rating:.1f} | "
                      f"Conf: {candidate.confidence:.0%}")

        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: (x.status == 'CONFIRMED', x.confidence), reverse=True)

        self.candidates = candidates
        return candidates

    def print_results(self, candidates: List[BreakoutCandidate] = None):
        """Print scan results in a formatted table"""
        if candidates is None:
            candidates = self.candidates

        if not candidates:
            print("\nNo Stage 1→2 candidates found.")
            return

        print(f"\n{'='*80}")
        print("STAGE 1→2 BREAKOUT CANDIDATES")
        print(f"{'='*80}")

        # Separate by status
        confirmed = [c for c in candidates if c.status == 'CONFIRMED']
        pending = [c for c in candidates if c.status == 'BREAKOUT_PENDING']
        watching = [c for c in candidates if c.status == 'WATCHING']

        if confirmed:
            print(f"\n🚨 CONFIRMED BREAKOUTS ({len(confirmed)}):")
            print(f"{'-'*80}")
            print(f"{'Symbol':<8} {'Entry':>10} {'Stop':>10} {'Risk%':>7} {'Base':>6} {'Vol':>6} {'RS':>7} {'Conf':>6}")
            print(f"{'-'*80}")
            for c in confirmed:
                risk_pct = ((c.entry_price - c.stop_loss) / c.entry_price) * 100
                print(f"{c.symbol:<8} ${c.entry_price:>8.2f} ${c.stop_loss:>8.2f} {risk_pct:>6.1f}% "
                      f"{c.stage1_duration_days:>5}d {c.current_volume_ratio:>5.1f}x {c.rs_rating:>6.1f} {c.confidence:>5.0%}")

        if pending:
            print(f"\n⏳ BREAKOUT PENDING ({len(pending)}):")
            print(f"{'-'*80}")
            print(f"{'Symbol':<8} {'Price':>10} {'Breakout':>10} {'Base':>6} {'Vol':>6} {'RS':>7} {'RS↑':>5} {'Conf':>6}")
            print(f"{'-'*80}")
            for c in pending:
                rs_arrow = "✓" if c.rs_improving else "✗"
                print(f"{c.symbol:<8} ${c.entry_price:>8.2f} ${c.stage1_high:>8.2f} "
                      f"{c.stage1_duration_days:>5}d {c.current_volume_ratio:>5.1f}x {c.rs_rating:>6.1f} {rs_arrow:>5} {c.confidence:>5.0%}")

        if watching:
            print(f"\n👀 WATCHING ({len(watching)}):")
            print(f"{'-'*80}")
            print(f"{'Symbol':<8} {'Price':>10} {'Resistance':>10} {'Base':>6} {'Range%':>7} {'RS':>7} {'RS↑':>5}")
            print(f"{'-'*80}")
            for c in watching:
                rs_arrow = "✓" if c.rs_improving else "✗"
                print(f"{c.symbol:<8} ${c.entry_price:>8.2f} ${c.stage1_high:>8.2f} "
                      f"{c.stage1_duration_days:>5}d {c.consolidation_range_pct:>6.1f}% {c.rs_rating:>6.1f} {rs_arrow:>5}")

        print(f"\n{'='*80}")


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_database(db_path: str = 'database/live_trading.db'):
    """Initialize the stage1_to2_alerts table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stage1_to2_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_date TEXT DEFAULT (date('now')),
            symbol TEXT NOT NULL,
            stage1_start_date TEXT,
            stage1_duration_days INTEGER,
            breakout_date TEXT,
            entry_price REAL,
            stop_loss REAL,
            stage1_high REAL,
            stage1_low REAL,
            consolidation_range_pct REAL,
            volume_ratio REAL,
            rs_rating REAL,
            rs_improving INTEGER,
            confidence REAL,
            status TEXT DEFAULT 'WATCHING',
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    # Create index for faster lookups
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stage1_to2_symbol_date
        ON stage1_to2_alerts(symbol, alert_date)
    ''')

    conn.commit()
    conn.close()
    print("Database table initialized: stage1_to2_alerts")


def save_candidates(candidates: List[BreakoutCandidate], db_path: str = 'database/live_trading.db'):
    """Save candidates to database"""
    init_database(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    today = datetime.now().strftime('%Y-%m-%d')

    for c in candidates:
        # Check if alert already exists for this symbol today
        cursor.execute('''
            SELECT id FROM stage1_to2_alerts
            WHERE symbol = ? AND alert_date = ?
        ''', (c.symbol, today))

        existing = cursor.fetchone()

        if existing:
            # Update existing record
            cursor.execute('''
                UPDATE stage1_to2_alerts
                SET stage1_start_date = ?, stage1_duration_days = ?, breakout_date = ?,
                    entry_price = ?, stop_loss = ?, stage1_high = ?, stage1_low = ?,
                    consolidation_range_pct = ?, volume_ratio = ?, rs_rating = ?,
                    rs_improving = ?, confidence = ?, status = ?, updated_at = ?
                WHERE id = ?
            ''', (
                c.stage1_start_date.strftime('%Y-%m-%d') if c.stage1_start_date else None,
                c.stage1_duration_days,
                c.breakout_date.strftime('%Y-%m-%d') if c.breakout_date else None,
                c.entry_price,
                c.stop_loss,
                c.stage1_high,
                c.stage1_low,
                c.consolidation_range_pct,
                c.current_volume_ratio,
                c.rs_rating,
                1 if c.rs_improving else 0,
                c.confidence,
                c.status,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                existing[0]
            ))
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO stage1_to2_alerts
                (alert_date, symbol, stage1_start_date, stage1_duration_days, breakout_date,
                 entry_price, stop_loss, stage1_high, stage1_low,
                 consolidation_range_pct, volume_ratio, rs_rating, rs_improving,
                 confidence, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today,
                c.symbol,
                c.stage1_start_date.strftime('%Y-%m-%d') if c.stage1_start_date else None,
                c.stage1_duration_days,
                c.breakout_date.strftime('%Y-%m-%d') if c.breakout_date else None,
                c.entry_price,
                c.stop_loss,
                c.stage1_high,
                c.stage1_low,
                c.consolidation_range_pct,
                c.current_volume_ratio,
                c.rs_rating,
                1 if c.rs_improving else 0,
                c.confidence,
                c.status,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))

    conn.commit()
    conn.close()
    print(f"\nSaved {len(candidates)} candidates to database")


def get_active_alerts(db_path: str = 'database/live_trading.db') -> pd.DataFrame:
    """Get active alerts from database"""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query('''
            SELECT * FROM stage1_to2_alerts
            WHERE status IN ('WATCHING', 'BREAKOUT_PENDING', 'CONFIRMED')
            ORDER BY
                CASE status
                    WHEN 'CONFIRMED' THEN 1
                    WHEN 'BREAKOUT_PENDING' THEN 2
                    ELSE 3
                END,
                confidence DESC
        ''', conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading alerts: {e}")
        return pd.DataFrame()


# ============================================================================
# EMAIL ALERTS (Optional)
# ============================================================================

def send_email_alert(candidates: List[BreakoutCandidate], config: Dict = None):
    """
    Send email alert for confirmed breakouts

    Requires email configuration in config/models_config.yaml:
    email:
        enabled: true
        smtp_server: smtp.gmail.com
        smtp_port: 587
        sender: your_email@gmail.com
        password: your_app_password
        recipients:
            - recipient@email.com
    """
    confirmed = [c for c in candidates if c.status == 'CONFIRMED']

    if not confirmed:
        print("No confirmed breakouts to alert")
        return

    # Build email body
    subject = f"🚨 Stage 1→2 Breakout Alert: {len(confirmed)} Confirmed"

    body = "CONFIRMED STAGE 1→2 BREAKOUTS\n"
    body += "=" * 50 + "\n\n"

    for c in confirmed:
        risk_pct = ((c.entry_price - c.stop_loss) / c.entry_price) * 100
        body += f"Symbol: {c.symbol}\n"
        body += f"Entry: ${c.entry_price:.2f}\n"
        body += f"Stop: ${c.stop_loss:.2f} ({risk_pct:.1f}% risk)\n"
        body += f"Base Duration: {c.stage1_duration_days} days\n"
        body += f"Volume: {c.current_volume_ratio:.1f}x average\n"
        body += f"RS Rating: {c.rs_rating:.1f}\n"
        body += f"Confidence: {c.confidence:.0%}\n"
        body += "-" * 30 + "\n\n"

    print("\nEmail Alert Content:")
    print(body)

    # TODO: Implement actual email sending using smtplib
    # For now, just print the content
    print("\n[Email sending not implemented - configure SMTP settings]")


# ============================================================================
# MAIN
# ============================================================================

def load_universe(config_path: str = 'config/models_config.yaml') -> List[str]:
    """Load stock universe from config"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get custom watchlist from stage1_to2 config if exists
        s1_config = config.get('stage1_to2_breakout', {})
        custom = s1_config.get('custom_watchlist', [])

        # Also load core universe
        core_path = config.get('paths', {}).get('core_universe', 'live_universe.txt')
        core = []
        if os.path.exists(core_path):
            with open(core_path, 'r') as f:
                core = [line.strip() for line in f if line.strip()]

        # Combine and dedupe
        universe = list(set(custom + core))
        return sorted(universe)
    except Exception as e:
        print(f"Warning: Could not load universe: {e}")
        return DEFAULT_CONFIG['stage1_to2_breakout']['custom_watchlist']


def main():
    parser = argparse.ArgumentParser(description='Stage 1→2 Breakout Monitor')
    parser.add_argument('--scan', action='store_true', help='Scan for breakout candidates')
    parser.add_argument('--save', action='store_true', help='Save results to database')
    parser.add_argument('--alerts', action='store_true', help='Show active alerts from database')
    parser.add_argument('--email', action='store_true', help='Send email alerts for confirmed breakouts')
    parser.add_argument('--watchlist', type=str, help='Custom watchlist file (one symbol per line)')
    parser.add_argument('--init-db', action='store_true', help='Initialize database table')

    args = parser.parse_args()

    # Initialize database if requested
    if args.init_db:
        init_database()
        return

    # Show alerts from database
    if args.alerts:
        alerts = get_active_alerts()
        if len(alerts) == 0:
            print("No active alerts in database")
        else:
            print(f"\n{'='*60}")
            print("ACTIVE STAGE 1→2 ALERTS")
            print(f"{'='*60}")
            print(alerts.to_string(index=False))
        return

    # Default to scan mode
    if args.scan or not any([args.alerts, args.init_db]):
        # Load universe
        if args.watchlist and os.path.exists(args.watchlist):
            with open(args.watchlist, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
        else:
            symbols = load_universe()

        # Create monitor and scan
        monitor = Stage1To2Monitor()
        candidates = monitor.scan_universe(symbols)

        # Print results
        monitor.print_results()

        # Save to database if requested
        if args.save and candidates:
            save_candidates(candidates)

        # Send email if requested
        if args.email:
            send_email_alert(candidates)

        return candidates


if __name__ == "__main__":
    main()
