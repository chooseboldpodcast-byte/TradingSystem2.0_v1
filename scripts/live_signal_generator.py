# scripts/live_signal_generator.py
"""
Live Signal Generator

Generates trading signals for today based on the selected configuration.
Runs daily (typically at 6 AM before market open).

Usage:
    python scripts/live_signal_generator.py --config A   # 126 stocks, 4 models
    python scripts/live_signal_generator.py --config B   # 126 stocks, 8 models
    python scripts/live_signal_generator.py --config C   # Scanner + 126, 8 models

    # Generate signals for specific date (backtesting verification)
    python scripts/live_signal_generator.py --config B --date 2024-01-15
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import argparse
import yaml
import json
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct file imports (no __init__.py needed)
from models.weinstein_core import WeinsteinCore
from models.rsi_mean_reversion import RSIMeanReversion
from models.momentum_52w_high import Momentum52WeekHigh
from models.consolidation_breakout import ConsolidationBreakout
from models.vcp import VCP
from models.pocket_pivot import PocketPivot
from models.rs_breakout import RSBreakout
from models.high_tight_flag import HighTightFlag
from models.enhanced_mean_reversion import EnhancedMeanReversion
from models.dual_momentum import DualMomentum
from core.weinstein_engine import WeinsteinEngine
from scanner.universe_scanner import get_daily_universe

# Model mapping
MODEL_CLASSES = {
    'weinstein_core': WeinsteinCore,
    'rsi_mean_reversion': RSIMeanReversion,
    'momentum_52w_high': Momentum52WeekHigh,
    'consolidation_breakout': ConsolidationBreakout,
    'vcp': VCP,
    'pocket_pivot': PocketPivot,
    'rs_breakout': RSBreakout,
    'high_tight_flag': HighTightFlag,
    'enhanced_mean_reversion': EnhancedMeanReversion,
    'dual_momentum': DualMomentum
}


# ============================================================================
# CSS (Composite Signal Strength) Calculation
# ============================================================================

def load_model_quality_config(config_path: str = 'config/model_quality.yaml') -> dict:
    """Load model quality configuration"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load model_quality.yaml: {e}")
        return get_default_model_quality_config()


def get_default_model_quality_config() -> dict:
    """Default config if model_quality.yaml is missing"""
    return {
        'baseline_scores': {
            'Weinstein_Core': 100,
            '52W_High_Momentum': 62,
            'Consolidation_Breakout': 35,
            'VCP': 29,
            'Pocket_Pivot': 16,
            'RS_Breakout': 11,
            'Dual_Momentum': 10,
            'RSI_Mean_Reversion': 2,
            'Enhanced_Mean_Reversion': 2,
            'High_Tight_Flag': 0
        },
        'css_weights': {
            'rs_score': 0.35,
            'momentum_score': 0.25,
            'volume_score': 0.20,
            'tightness_score': 0.10,
            'model_quality_score': 0.10
        },
        'css_thresholds': {
            'rs_minimum': 80
        },
        'grade_ranges': {
            'A_plus': {'min': 85, 'max': 100},
            'A': {'min': 75, 'max': 84},
            'B': {'min': 65, 'max': 74},
            'C': {'min': 55, 'max': 64},
            'D': {'min': 0, 'max': 54}
        }
    }


def calculate_rs_score(rs_value: float) -> float:
    """Convert Mansfield RS to 0-100 score"""
    if rs_value >= 90:
        return 100
    elif rs_value >= 80:
        return 85
    elif rs_value >= 70:
        return 70
    elif rs_value >= 50:
        return 50
    else:
        return 25


def calculate_momentum_score(metadata: dict, rs_value: float = 0) -> float:
    """Calculate momentum score from signal metadata

    Handles different model types:
    - Dual_Momentum: uses abs_momentum_12m, 6m, 3m
    - 52W_High_Momentum: uses momentum_score or derives from RS
    - Others: falls back to RS-based estimate
    """
    # Try Dual_Momentum style metadata first
    abs_12m = metadata.get('abs_momentum_12m', 0) / 100  # Convert from %
    abs_6m = metadata.get('abs_momentum_6m', 0) / 100
    abs_3m = metadata.get('abs_momentum_3m', 0) / 100

    if abs_12m > 0 or abs_6m > 0 or abs_3m > 0:
        # IBD-style weighting: 40% recent, 60% longer-term
        raw_momentum = (0.40 * abs_3m) + (0.30 * abs_6m) + (0.30 * abs_12m)
        score = min(100, max(0, raw_momentum * 200))
        return score

    # Try momentum_score from metadata (some models provide this)
    if 'momentum_score' in metadata:
        # momentum_score is typically 0-2, scale to 0-100
        return min(100, max(0, metadata['momentum_score'] * 50))

    # Fallback: derive from RS value (RS is a proxy for relative momentum)
    # RS 100 = top performer = high momentum score
    if rs_value > 0:
        # Scale RS to momentum score: RS 50=25, RS 80=65, RS 100=85, RS 120+=100
        score = min(100, max(0, (rs_value - 50) * 1.5 + 25))
        return score

    return 50  # Neutral default


def calculate_volume_score(volume_ratio: float) -> float:
    """Convert volume ratio to 0-100 score"""
    if volume_ratio >= 2.5:
        return 100
    elif volume_ratio >= 2.0:
        return 90
    elif volume_ratio >= 1.5:
        return 75
    elif volume_ratio >= 1.2:
        return 60
    elif volume_ratio >= 1.0:
        return 40
    else:
        return 20


def calculate_tightness_score(price_vs_ma_pct: float) -> float:
    """Convert price vs MA percentage to 0-100 score (lower = tighter = better)"""
    pct = abs(price_vs_ma_pct)
    if pct <= 5:
        return 100
    elif pct <= 10:
        return 80
    elif pct <= 15:
        return 60
    elif pct <= 25:
        return 40
    else:
        return 20


def get_model_quality_score(model_name: str, mq_config: dict) -> float:
    """Get MQS for a model from config"""
    baseline_scores = mq_config.get('baseline_scores', {})
    return baseline_scores.get(model_name, 30)  # Default to 30 if not found


def calculate_css(signal: dict, mq_config: dict) -> float:
    """
    Calculate Composite Signal Strength (CSS) for a signal

    CSS = (0.35 × RS) + (0.25 × Momentum) + (0.20 × Volume) + (0.10 × Tightness) + (0.10 × MQS)
    """
    weights = mq_config.get('css_weights', {
        'rs_score': 0.35,
        'momentum_score': 0.25,
        'volume_score': 0.20,
        'tightness_score': 0.10,
        'model_quality_score': 0.10
    })

    metadata = signal.get('metadata', {})
    rs_value = signal.get('rs', 0)

    # Calculate component scores
    rs_score = calculate_rs_score(rs_value)
    momentum_score = calculate_momentum_score(metadata, rs_value)
    volume_score = calculate_volume_score(signal.get('volume_ratio', 1.0))

    # Get price vs MA from metadata
    price_vs_ma = metadata.get('price_vs_ma_pct', metadata.get('price_to_ma_pct', 15))
    tightness_score = calculate_tightness_score(price_vs_ma)

    mqs = get_model_quality_score(signal.get('model', ''), mq_config)

    # Calculate weighted CSS
    css = (
        weights['rs_score'] * rs_score +
        weights['momentum_score'] * momentum_score +
        weights['volume_score'] * volume_score +
        weights['tightness_score'] * tightness_score +
        weights['model_quality_score'] * mqs
    )

    return round(css, 1)


def assign_grade(css: float, mq_config: dict) -> str:
    """Assign letter grade based on CSS score"""
    grade_ranges = mq_config.get('grade_ranges', {
        'A_plus': {'min': 85, 'max': 100},
        'A': {'min': 75, 'max': 84},
        'B': {'min': 65, 'max': 74},
        'C': {'min': 55, 'max': 64},
        'D': {'min': 0, 'max': 54}
    })

    if css >= grade_ranges['A_plus']['min']:
        return 'A+'
    elif css >= grade_ranges['A']['min']:
        return 'A'
    elif css >= grade_ranges['B']['min']:
        return 'B'
    elif css >= grade_ranges['C']['min']:
        return 'C'
    else:
        return 'D'


def load_config(config_path: str = 'config/models_config.yaml') -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config_settings(config: dict, config_type: str) -> dict:
    """Get settings for specific config"""
    configs = config.get('configurations', {})
    if config_type not in configs:
        raise ValueError(f"Invalid config_type: {config_type}")
    return configs[config_type]


def load_universe(config_settings: dict, config: dict) -> list:
    """Load stock universe based on config"""
    universe_type = config_settings.get('universe', 'core')
    core_path = config.get('paths', {}).get('core_universe', 'live_universe.txt')
    data_dir = config.get('paths', {}).get('data_dir', 'data')
    
    with open(core_path, 'r') as f:
        core_universe = [line.strip() for line in f if line.strip()]
    
    if universe_type == 'core':
        return core_universe
    else:
        return get_daily_universe(core_universe_path=core_path, data_dir=data_dir, quick=True)


def create_models(config_settings: dict) -> list:
    """Create model instances"""
    models = []
    model_names = config_settings.get('models', [])
    allocations = config_settings.get('allocations', {})
    
    for name in model_names:
        if name in MODEL_CLASSES:
            allocation = allocations.get(name, 0.10)
            models.append(MODEL_CLASSES[name](allocation_pct=allocation))
    
    return models


def load_stock_data(symbols: list, data_dir: str = 'data') -> tuple:
    """Load and analyze stock data"""
    data = {}
    index_data = None
    engine = WeinsteinEngine()
    
    for symbol in symbols:
        filepath = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                if symbol == 'SPY':
                    index_data = df.copy()
                
                data[symbol] = df
            except:
                pass
    
    # Analyze stocks
    analyzed_data = {}
    for symbol, df in data.items():
        try:
            analyzed = engine.analyze_stock(df, index_data)
            analyzed_data[symbol] = analyzed
        except:
            pass
    
    return analyzed_data, index_data


def generate_signals(
    models: list,
    analyzed_data: dict,
    index_data: pd.DataFrame,
    signal_date: datetime = None,
    mq_config: dict = None
) -> dict:
    """
    Generate signals for all models

    Returns dict with entry signals, exit signals, and summary
    """
    if signal_date is None:
        signal_date = datetime.now()

    if mq_config is None:
        mq_config = load_model_quality_config()

    rs_minimum = mq_config.get('css_thresholds', {}).get('rs_minimum', 80)

    entry_signals = []
    exit_signals = []
    filtered_count = 0  # Track signals filtered by RS

    for symbol, df in analyzed_data.items():
        try:
            # Convert index to tz-naive if needed
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)

            # Find the index for signal_date (or closest prior date)
            signal_ts = pd.Timestamp(signal_date).tz_localize(None) if pd.Timestamp(signal_date).tz else pd.Timestamp(signal_date)

            # Get dates up to signal date
            mask = df.index <= signal_ts
            if not mask.any():
                continue
            idx = mask.sum() - 1

            if idx < 150:
                continue
        except Exception:
            # Fallback: use last available data point
            idx = len(df) - 1
            if idx < 150:
                continue

        if idx < 150:
            continue

        current = df.iloc[idx]

        for model in models:
            # Check for entry signals
            entry_signal = model.generate_entry_signals(df, idx, index_data)

            if entry_signal.get('signal'):
                metadata = entry_signal.get('metadata', {})
                rs_value = current.get('mansfield_rs', 0)

                # Tier 1 Filter: RS must be >= rs_minimum (default 80)
                if rs_value < rs_minimum:
                    filtered_count += 1
                    continue

                # Build signal dict - FIX: get method from metadata
                signal = {
                    'symbol': symbol,
                    'model': model.name,
                    'date': df.index[idx],
                    'price': current['close'],
                    'entry_price': entry_signal['entry_price'],
                    'stop_loss': entry_signal['stop_loss'],
                    'confidence': entry_signal.get('confidence', 0.8),
                    'method': metadata.get('method', 'UNKNOWN'),  # FIX: get from metadata
                    'stage': current.get('stage', 'UNKNOWN'),
                    'rs': rs_value,
                    'volume_ratio': current.get('volume_ratio', 1.0),
                    'metadata': metadata
                }

                # Calculate CSS and Grade
                css = calculate_css(signal, mq_config)
                grade = assign_grade(css, mq_config)

                signal['css'] = css
                signal['grade'] = grade

                # Add component scores for transparency
                signal['css_components'] = {
                    'rs_score': calculate_rs_score(rs_value),
                    'momentum_score': calculate_momentum_score(metadata, rs_value),
                    'volume_score': calculate_volume_score(signal['volume_ratio']),
                    'tightness_score': calculate_tightness_score(
                        metadata.get('price_vs_ma_pct', metadata.get('price_to_ma_pct', 15))
                    ),
                    'model_quality_score': get_model_quality_score(model.name, mq_config)
                }

                entry_signals.append(signal)

    # Sort by CSS descending (best signals first)
    entry_signals.sort(key=lambda x: x['css'], reverse=True)

    # Count by grade
    grade_counts = {'A+': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for sig in entry_signals:
        grade_counts[sig['grade']] = grade_counts.get(sig['grade'], 0) + 1

    return {
        'entry_signals': entry_signals,
        'exit_signals': exit_signals,
        'summary': {
            'date': signal_date.strftime('%Y-%m-%d'),
            'stocks_analyzed': len(analyzed_data),
            'total_entry_signals': len(entry_signals),
            'signals_filtered_by_rs': filtered_count,
            'rs_minimum': rs_minimum,
            'signals_by_model': {},
            'signals_by_grade': grade_counts
        }
    }


def save_signals(signals: dict, config_type: str, output_dir: str = 'logs/signals'):
    """Save signals to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    date_str = signals['summary']['date']
    filename = f"signals_{config_type}_{date_str}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert datetime objects to strings for JSON and ensure all fields are included
    output = {
        'config_type': config_type,
        'summary': signals['summary'],
        'entry_signals': [
            {
                'symbol': s['symbol'],
                'model': s['model'],
                'date': str(s['date']),
                'price': s['price'],
                'entry_price': s['entry_price'],
                'stop_loss': s['stop_loss'],
                'confidence': s['confidence'],
                'method': s['method'],
                'stage': s.get('stage'),
                'rs': s['rs'],
                'volume_ratio': s['volume_ratio'],
                'css': s.get('css', 0),
                'grade': s.get('grade', 'D'),
                'css_components': s.get('css_components', {}),
                'metadata': s.get('metadata', {})
            }
            for s in signals['entry_signals']
        ],
        'exit_signals': signals['exit_signals']
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    return filepath


def print_signals(signals: dict, config_type: str):
    """Print signals to console"""
    summary = signals['summary']

    print(f"\n{'='*70}")
    print(f"LIVE SIGNALS - CONFIG {config_type}")
    print(f"Date: {summary['date']}")
    print(f"Stocks Analyzed: {summary['stocks_analyzed']}")
    print(f"RS Filter: >= {summary.get('rs_minimum', 80)}")
    print(f"{'='*70}")

    entry_signals = signals['entry_signals']
    filtered = summary.get('signals_filtered_by_rs', 0)

    if len(entry_signals) == 0:
        print("\nNo entry signals today")
        if filtered > 0:
            print(f"({filtered} signals filtered out by RS < {summary.get('rs_minimum', 80)})")
    else:
        grade_counts = summary.get('signals_by_grade', {})
        print(f"\nENTRY SIGNALS: {len(entry_signals)} total")
        print(f"  Grades: A+={grade_counts.get('A+', 0)}, A={grade_counts.get('A', 0)}, "
              f"B={grade_counts.get('B', 0)}, C={grade_counts.get('C', 0)}, D={grade_counts.get('D', 0)}")
        if filtered > 0:
            print(f"  Filtered by RS: {filtered}")
        print(f"{'-'*70}")

        # Print signals sorted by CSS (already sorted)
        print(f"\n{'Symbol':<8} {'Grade':<6} {'CSS':<6} {'Model':<22} {'Price':>10} {'RS':>6} {'Vol':>5} {'Conf':>6}")
        print(f"{'-'*70}")

        for sig in entry_signals:
            grade_display = sig.get('grade', '?')
            css_display = sig.get('css', 0)
            print(f"{sig['symbol']:<8} {grade_display:<6} {css_display:<6.1f} {sig['model']:<22} "
                  f"${sig['entry_price']:>8.2f} {sig['rs']:>5.1f} {sig['volume_ratio']:>4.1f}x {sig['confidence']:>5.0%}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Generate live trading signals')
    parser.add_argument('--config', type=str, required=True, choices=['A', 'B', 'C'],
                       help='Configuration: A (4 models), B (8 models), C (scanner + 8 models)')
    parser.add_argument('--date', type=str, default=None,
                       help='Signal date (YYYY-MM-DD), default is today')
    parser.add_argument('--save', action='store_true', help='Save signals to file')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Parse date
    if args.date:
        signal_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        signal_date = datetime.now()

    # Load configs
    config = load_config()
    config_settings = get_config_settings(config, args.config)
    mq_config = load_model_quality_config()

    if not args.quiet:
        print(f"\n{'#'*60}")
        print(f"LIVE SIGNAL GENERATOR - CONFIG {args.config}")
        print(f"{config_settings['description']}")
        print(f"Signal Date: {signal_date.strftime('%Y-%m-%d')}")
        print(f"{'#'*60}")

    # Load universe
    universe = load_universe(config_settings, config)
    if not args.quiet:
        print(f"\nUniverse: {len(universe)} stocks")

    # Load data
    data_dir = config.get('paths', {}).get('data_dir', 'data')
    analyzed_data, index_data = load_stock_data(universe, data_dir)

    if not analyzed_data:
        print("No data loaded. Exiting.")
        return

    if not args.quiet:
        print(f"Analyzed: {len(analyzed_data)} stocks")

    # Create models
    models = create_models(config_settings)
    if not args.quiet:
        print(f"Models: {[m.name for m in models]}")

    # Generate signals with CSS scoring
    signals = generate_signals(models, analyzed_data, index_data, signal_date, mq_config)

    # Count signals by model
    for sig in signals['entry_signals']:
        model = sig['model']
        if model not in signals['summary']['signals_by_model']:
            signals['summary']['signals_by_model'][model] = 0
        signals['summary']['signals_by_model'][model] += 1

    # Output
    if not args.quiet:
        print_signals(signals, args.config)

    # Save if requested
    if args.save:
        logs_dir = config.get('paths', {}).get('logs_dir', 'logs')
        filepath = save_signals(signals, args.config, os.path.join(logs_dir, 'signals'))
        print(f"\nSignals saved to: {filepath}")

    # Return for programmatic use
    return signals


if __name__ == "__main__":
    main()
