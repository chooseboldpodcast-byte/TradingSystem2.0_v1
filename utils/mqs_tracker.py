# utils/mqs_tracker.py
"""
Model Quality Score (MQS) Tracker

Tracks live trading performance and blends with backtest MQS scores
to provide dynamic model quality rankings.

Usage:
    python utils/mqs_tracker.py --update    # Update MQS from live trades
    python utils/mqs_tracker.py --report    # Show MQS report
    python utils/mqs_tracker.py --recalc    # Recalculate from backtest DB
"""

import sqlite3
import pandas as pd
import yaml
import os
from datetime import datetime
from typing import Dict, Tuple


# Paths
LIVE_DB_PATH = "database/live_trading.db"
BACKTEST_DB_PATH = "database/weinstein.db"
MQ_CONFIG_PATH = "config/model_quality.yaml"


def load_mq_config() -> dict:
    """Load model quality configuration"""
    try:
        with open(MQ_CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: {MQ_CONFIG_PATH} not found")
        return {}


def save_mq_config(config: dict):
    """Save model quality configuration"""
    with open(MQ_CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Updated {MQ_CONFIG_PATH}")


def get_sample_size_factor(n_trades: int) -> float:
    """Get sample size factor based on number of trades"""
    if n_trades < 50:
        return 0.25
    elif n_trades < 500:
        return 0.50
    elif n_trades < 2000:
        return 0.75
    elif n_trades < 10000:
        return 0.90
    else:
        return 1.00


def get_blending_weights(n_live_trades: int) -> Tuple[float, float]:
    """Get blending weights based on number of live trades"""
    if n_live_trades < 10:
        return 1.0, 0.0  # 100% backtest
    elif n_live_trades < 50:
        return 0.8, 0.2  # 80% backtest, 20% live
    elif n_live_trades < 100:
        return 0.6, 0.4  # 60% backtest, 40% live
    else:
        return 0.5, 0.5  # 50% backtest, 50% live


def calculate_mqs_from_trades(trades_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Calculate MQS for each model from trades dataframe

    MQS = Avg_Return × Win_Rate_Factor × Sample_Size_Factor
    Win_Rate_Factor = 0.5 + (Win_Rate / 100)
    """
    if len(trades_df) == 0 or 'model' not in trades_df.columns:
        return {}

    results = {}

    for model in trades_df['model'].unique():
        model_trades = trades_df[trades_df['model'] == model]
        n_trades = len(model_trades)

        if n_trades == 0:
            continue

        # Calculate metrics
        avg_return = model_trades['pnl_pct'].mean()
        win_rate = (model_trades['pnl'] > 0).sum() / n_trades * 100
        total_pnl = model_trades['pnl'].sum()

        # Calculate factors
        win_rate_factor = 0.5 + (win_rate / 100)
        sample_size_factor = get_sample_size_factor(n_trades)

        # Calculate raw MQS
        raw_mqs = avg_return * win_rate_factor * sample_size_factor

        results[model] = {
            'trades': n_trades,
            'avg_return_pct': round(avg_return, 2),
            'win_rate_pct': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'win_rate_factor': round(win_rate_factor, 3),
            'sample_size_factor': sample_size_factor,
            'raw_mqs': round(raw_mqs, 2)
        }

    return results


def normalize_mqs_scores(mqs_data: Dict[str, dict]) -> Dict[str, int]:
    """Normalize raw MQS scores to 0-100 scale"""
    if not mqs_data:
        return {}

    # Get max raw MQS for normalization
    max_raw = max(d['raw_mqs'] for d in mqs_data.values())

    if max_raw <= 0:
        return {model: 0 for model in mqs_data}

    normalized = {}
    for model, data in mqs_data.items():
        # Normalize to 0-100
        score = int((data['raw_mqs'] / max_raw) * 100)
        score = max(0, min(100, score))  # Clamp to 0-100
        normalized[model] = score

    return normalized


def get_live_trades() -> pd.DataFrame:
    """Get all live trades from database"""
    if not os.path.exists(LIVE_DB_PATH):
        return pd.DataFrame()

    try:
        with sqlite3.connect(LIVE_DB_PATH) as conn:
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='live_trades'
            """)
            if cursor.fetchone() is None:
                return pd.DataFrame()

            df = pd.read_sql_query("""
                SELECT * FROM live_trades
                WHERE pnl IS NOT NULL
            """, conn)
            return df
    except Exception as e:
        print(f"Error reading live trades: {e}")
        return pd.DataFrame()


def get_backtest_trades() -> pd.DataFrame:
    """Get production backtest trades from database"""
    if not os.path.exists(BACKTEST_DB_PATH):
        return pd.DataFrame()

    try:
        with sqlite3.connect(BACKTEST_DB_PATH) as conn:
            df = pd.read_sql_query("""
                SELECT t.*
                FROM trades t
                JOIN backtest_runs r ON t.backtest_run_id = r.id
                WHERE r.run_type = 'production'
            """, conn)
            return df
    except Exception as e:
        print(f"Error reading backtest trades: {e}")
        return pd.DataFrame()


def update_mqs_from_live():
    """Update MQS scores by blending backtest and live performance"""
    print("=" * 60)
    print("MQS UPDATE - Blending Backtest + Live Performance")
    print("=" * 60)

    # Load current config
    config = load_mq_config()
    if not config:
        print("Error: Could not load model_quality.yaml")
        return

    # Get backtest baseline scores
    backtest_scores = config.get('baseline_scores', {})

    # Get live trades
    live_trades = get_live_trades()
    print(f"\nLive trades found: {len(live_trades)}")

    if len(live_trades) == 0:
        print("No live trades yet. Using backtest scores only.")
        return

    # Calculate live MQS
    live_mqs_data = calculate_mqs_from_trades(live_trades)
    live_scores = normalize_mqs_scores(live_mqs_data)

    print("\nLive Performance by Model:")
    print("-" * 60)
    for model, data in live_mqs_data.items():
        print(f"{model}:")
        print(f"  Trades: {data['trades']}, Avg Return: {data['avg_return_pct']:.2f}%")
        print(f"  Win Rate: {data['win_rate_pct']:.1f}%, Raw MQS: {data['raw_mqs']:.2f}")
        print(f"  Normalized Score: {live_scores.get(model, 0)}")

    # Blend scores
    print("\n" + "=" * 60)
    print("BLENDED SCORES")
    print("=" * 60)

    blended_scores = {}
    for model in set(list(backtest_scores.keys()) + list(live_scores.keys())):
        bt_score = backtest_scores.get(model, 30)  # Default 30 if not in backtest
        live_score = live_scores.get(model, bt_score)  # Default to backtest if no live

        n_live = live_mqs_data.get(model, {}).get('trades', 0)
        bt_weight, live_weight = get_blending_weights(n_live)

        blended = int(bt_weight * bt_score + live_weight * live_score)
        blended_scores[model] = blended

        print(f"{model}:")
        print(f"  Backtest: {bt_score} ({bt_weight:.0%}), Live: {live_score} ({live_weight:.0%})")
        print(f"  Blended: {blended}")

    # Update config
    if 'blended_scores' not in config:
        config['blended_scores'] = {}

    config['blended_scores'] = blended_scores
    config['blended_scores_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Store live stats for reference
    if 'live_stats' not in config:
        config['live_stats'] = {}

    config['live_stats'] = {
        model: {
            'trades': data['trades'],
            'avg_return_pct': data['avg_return_pct'],
            'win_rate_pct': data['win_rate_pct'],
            'total_pnl': data['total_pnl']
        }
        for model, data in live_mqs_data.items()
    }
    config['live_stats_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    save_mq_config(config)

    print("\n" + "=" * 60)
    print("MQS scores updated successfully!")


def recalculate_baseline_mqs():
    """Recalculate baseline MQS from backtest database"""
    print("=" * 60)
    print("RECALCULATING BASELINE MQS FROM BACKTEST DATA")
    print("=" * 60)

    # Get backtest trades
    trades = get_backtest_trades()
    print(f"\nBacktest trades found: {len(trades)}")

    if len(trades) == 0:
        print("No backtest trades found.")
        return

    # Calculate MQS
    mqs_data = calculate_mqs_from_trades(trades)
    normalized_scores = normalize_mqs_scores(mqs_data)

    print("\nModel Performance:")
    print("-" * 60)
    for model, data in sorted(mqs_data.items(), key=lambda x: x[1]['raw_mqs'], reverse=True):
        print(f"{model}:")
        print(f"  Trades: {data['trades']}, Avg Return: {data['avg_return_pct']:.2f}%")
        print(f"  Win Rate: {data['win_rate_pct']:.1f}%, Total PnL: ${data['total_pnl']:,.0f}")
        print(f"  Normalized Score: {normalized_scores.get(model, 0)}")

    # Update config
    config = load_mq_config()
    if not config:
        config = {}

    config['baseline_scores'] = normalized_scores
    config['backtest_stats'] = {
        model: {
            'trades': data['trades'],
            'avg_return_pct': data['avg_return_pct'],
            'win_rate_pct': data['win_rate_pct'],
            'total_pnl': data['total_pnl'],
            'profit_factor': round(
                trades[trades['model'] == model][trades['pnl'] > 0]['pnl'].sum() /
                abs(trades[trades['model'] == model][trades['pnl'] <= 0]['pnl'].sum())
                if abs(trades[trades['model'] == model][trades['pnl'] <= 0]['pnl'].sum()) > 0 else 0, 2
            )
        }
        for model, data in mqs_data.items()
    }
    config['last_updated'] = datetime.now().strftime('%Y-%m-%d')

    save_mq_config(config)


def show_mqs_report():
    """Display current MQS report"""
    config = load_mq_config()
    if not config:
        print("No MQS configuration found.")
        return

    print("=" * 60)
    print("MODEL QUALITY SCORE (MQS) REPORT")
    print(f"Last Updated: {config.get('last_updated', 'Unknown')}")
    print("=" * 60)

    print("\n--- BASELINE SCORES (from backtest) ---")
    baseline = config.get('baseline_scores', {})
    for model, score in sorted(baseline.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {score}")

    if 'blended_scores' in config:
        print(f"\n--- BLENDED SCORES (backtest + live) ---")
        print(f"Updated: {config.get('blended_scores_updated', 'Unknown')}")
        blended = config.get('blended_scores', {})
        for model, score in sorted(blended.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {score}")

    if 'live_stats' in config:
        print(f"\n--- LIVE TRADING STATS ---")
        print(f"Updated: {config.get('live_stats_updated', 'Unknown')}")
        live_stats = config.get('live_stats', {})
        for model, stats in live_stats.items():
            print(f"  {model}: {stats['trades']} trades, "
                  f"{stats['avg_return_pct']:.1f}% avg, "
                  f"{stats['win_rate_pct']:.0f}% win rate")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MQS Tracker')
    parser.add_argument('--update', action='store_true', help='Update MQS from live trades')
    parser.add_argument('--report', action='store_true', help='Show MQS report')
    parser.add_argument('--recalc', action='store_true', help='Recalculate baseline from backtest')

    args = parser.parse_args()

    if args.update:
        update_mqs_from_live()
    elif args.recalc:
        recalculate_baseline_mqs()
    elif args.report:
        show_mqs_report()
    else:
        # Default: show report
        show_mqs_report()
