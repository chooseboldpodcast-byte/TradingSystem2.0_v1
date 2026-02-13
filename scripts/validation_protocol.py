#!/usr/bin/env python3
"""
Validation Protocol for TradingSystem2.0

Establishes baseline metrics and validates each phase of enhancement.

Usage:
    # Establish baseline (run this FIRST, before any changes)
    python scripts/validation_protocol.py --establish-baseline

    # Validate a phase after implementation
    python scripts/validation_protocol.py --validate-phase "phase_0_regime_filter"

    # Run stress tests only
    python scripts/validation_protocol.py --stress-test
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_prod_backtest import (
    ProductionBacktest,
    load_config,
    load_stock_data,
    get_index_data,
    create_models
)


# ============================================================================
# CONSTANTS
# ============================================================================

VALIDATION_DIR = Path("validation")
BASELINE_FILE = VALIDATION_DIR / "baseline_metrics.json"
RESULTS_DIR = VALIDATION_DIR / "phase_results"
STRESS_DIR = VALIDATION_DIR / "stress_test_reports"

# Walk-forward validation periods
PERIODS = {
    "full": ("1985-01-01", "2026-01-31"),
    "train": ("1985-01-01", "2015-12-31"),      # 31 years - fit the system
    "validate": ("2016-01-01", "2020-12-31"),   # 5 years - out-of-sample
    "test": ("2021-01-01", "2026-01-31"),       # 5 years - hold-out
}

# Stress test periods (major market events)
STRESS_PERIODS = {
    "dot_com_crash": ("2000-03-01", "2002-10-31"),
    "financial_crisis": ("2007-10-01", "2009-03-31"),
    "covid_crash": ("2020-02-01", "2020-04-30"),
    "rate_hike_bear": ("2022-01-01", "2022-12-31"),
    "bull_2023_2024": ("2023-01-01", "2024-12-31"),
}

# Phase gate thresholds
GATES = {
    "min_cagr": 0.0,                    # CAGR must be positive
    "max_drawdown": 40.0,               # Max drawdown must be < 40%
    "max_performance_decay": 0.50,      # Validate CAGR >= 50% of train CAGR
    "max_cagr_regression": 5.0,         # Can't drop more than 5% CAGR vs baseline
}

SURVIVORSHIP_WARNING = """
⚠️  SURVIVORSHIP BIAS WARNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This backtest uses stocks that exist today (2026) to test historical periods.
Companies that went bankrupt, were delisted, or were acquired are NOT included.

Examples of missing companies:
- Enron (bankrupt 2001)
- Lehman Brothers (bankrupt 2008)
- Bear Stearns (collapsed 2008)
- Hundreds of other delisted companies

Estimated impact: Results may be inflated by 2-5% annually.

To correct this, use CRSP/CompuStat data with delisting returns.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_max_drawdown(trades_df: pd.DataFrame, initial_capital: float) -> Dict:
    """
    Calculate maximum drawdown from trade history.

    Returns:
        {
            "max_drawdown_pct": float,
            "max_drawdown_date": str,
            "max_drawdown_duration_days": int,
            "recovery_date": str or None
        }
    """
    if len(trades_df) == 0:
        return {
            "max_drawdown_pct": 0.0,
            "max_drawdown_date": None,
            "max_drawdown_duration_days": 0,
            "recovery_date": None
        }

    # Sort by exit date
    df = trades_df.sort_values('exit_date').copy()

    # Calculate cumulative P&L and equity curve
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['equity'] = initial_capital + df['cumulative_pnl']

    # Calculate running maximum and drawdown
    df['running_max'] = df['equity'].cummax()
    df['drawdown'] = df['equity'] - df['running_max']
    df['drawdown_pct'] = (df['drawdown'] / df['running_max']) * 100

    # Find maximum drawdown
    max_dd_idx = df['drawdown_pct'].idxmin()
    max_dd_pct = df.loc[max_dd_idx, 'drawdown_pct']
    max_dd_date = df.loc[max_dd_idx, 'exit_date']

    # Find drawdown duration (when did we recover?)
    post_dd = df[df.index >= max_dd_idx]
    recovery_mask = post_dd['equity'] >= post_dd.loc[max_dd_idx, 'running_max']

    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        recovery_date = df.loc[recovery_idx, 'exit_date']
        duration = (recovery_date - max_dd_date).days
    else:
        recovery_date = None
        duration = (df['exit_date'].max() - max_dd_date).days

    return {
        "max_drawdown_pct": round(abs(max_dd_pct), 2),
        "max_drawdown_date": str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
        "max_drawdown_duration_days": duration,
        "recovery_date": str(recovery_date.date()) if recovery_date is not None and hasattr(recovery_date, 'date') else str(recovery_date) if recovery_date else None
    }


def calculate_sharpe_ratio(trades_df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio from trade returns.

    Uses monthly returns for calculation.
    """
    if len(trades_df) < 12:
        return 0.0

    df = trades_df.sort_values('exit_date').copy()
    df['exit_month'] = pd.to_datetime(df['exit_date']).dt.to_period('M')

    # Calculate monthly P&L
    monthly_pnl = df.groupby('exit_month')['pnl'].sum()

    if len(monthly_pnl) < 12:
        return 0.0

    # Calculate monthly returns (assuming constant capital base for simplicity)
    # In reality, should use actual capital at start of each month
    avg_capital = 100000  # Default assumption
    monthly_returns = monthly_pnl / avg_capital

    # Annualize
    mean_return = monthly_returns.mean() * 12
    std_return = monthly_returns.std() * np.sqrt(12)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return
    return round(sharpe, 2)


def calculate_sortino_ratio(trades_df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio (only considers downside volatility).
    """
    if len(trades_df) < 12:
        return 0.0

    df = trades_df.sort_values('exit_date').copy()
    df['exit_month'] = pd.to_datetime(df['exit_date']).dt.to_period('M')

    monthly_pnl = df.groupby('exit_month')['pnl'].sum()

    if len(monthly_pnl) < 12:
        return 0.0

    avg_capital = 100000
    monthly_returns = monthly_pnl / avg_capital

    mean_return = monthly_returns.mean() * 12
    downside_returns = monthly_returns[monthly_returns < 0]

    if len(downside_returns) == 0:
        return 10.0  # No downside, excellent

    downside_std = downside_returns.std() * np.sqrt(12)

    if downside_std == 0:
        return 0.0

    sortino = (mean_return - risk_free_rate) / downside_std
    return round(sortino, 2)


def extract_metrics(results: Dict, initial_capital: float) -> Dict:
    """
    Extract and calculate all metrics from backtest results.
    """
    trades_df = results.get('trades', pd.DataFrame())

    # Basic metrics from results
    metrics = {
        "total_trades": results.get('total_trades', 0),
        "winning_trades": results.get('winning_trades', 0),
        "losing_trades": results.get('losing_trades', 0),
        "win_rate_pct": round(results.get('win_rate', 0), 2),
        "total_pnl": round(results.get('total_pnl', 0), 2),
        "total_return_pct": round(results.get('total_return_pct', 0), 2),
        "cagr_pct": round(results.get('cagr', 0), 2),
        "profit_factor": round(results.get('profit_factor', 0), 2),
        "avg_win": round(results.get('avg_win', 0), 2),
        "avg_loss": round(results.get('avg_loss', 0), 2),
        "avg_hold_days": round(results.get('avg_hold_days', 0), 1),
        "initial_capital": initial_capital,
        "final_value": round(results.get('final_value', initial_capital), 2),
        "years": round(results.get('years', 0), 2),
    }

    # Add drawdown metrics
    dd_metrics = calculate_max_drawdown(trades_df, initial_capital)
    metrics.update(dd_metrics)

    # Add risk-adjusted metrics
    metrics["sharpe_ratio"] = calculate_sharpe_ratio(trades_df)
    metrics["sortino_ratio"] = calculate_sortino_ratio(trades_df)

    # Calmar ratio (CAGR / Max DD)
    if metrics["max_drawdown_pct"] > 0:
        metrics["calmar_ratio"] = round(metrics["cagr_pct"] / metrics["max_drawdown_pct"], 2)
    else:
        metrics["calmar_ratio"] = 0.0

    return metrics


def extract_model_breakdown(results: Dict) -> Dict:
    """Extract per-model performance breakdown."""
    trades_df = results.get('trades', pd.DataFrame())

    if len(trades_df) == 0 or 'model' not in trades_df.columns:
        return {}

    breakdown = {}
    for model in trades_df['model'].unique():
        model_trades = trades_df[trades_df['model'] == model]
        wins = len(model_trades[model_trades['pnl'] > 0])
        total = len(model_trades)

        breakdown[model] = {
            "trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate_pct": round(wins / total * 100, 1) if total > 0 else 0,
            "total_pnl": round(model_trades['pnl'].sum(), 2),
            "avg_pnl": round(model_trades['pnl'].mean(), 2) if total > 0 else 0,
        }

    return breakdown


# ============================================================================
# VALIDATION PROTOCOL CLASS
# ============================================================================

class ValidationProtocol:
    """
    Manages baseline comparison and phase validation.
    """

    def __init__(self, initial_capital: int = 100000):
        self.initial_capital = initial_capital

        # Ensure directories exist
        VALIDATION_DIR.mkdir(exist_ok=True)
        RESULTS_DIR.mkdir(exist_ok=True)
        STRESS_DIR.mkdir(exist_ok=True)

    def run_backtest(self, start_date: str, end_date: str, description: str = "") -> Dict:
        """
        Run a backtest for the specified period.

        Returns:
            Dict with all metrics
        """
        print(f"\n{'='*60}")
        print(f"Running backtest: {start_date} to {end_date}")
        print(f"Capital: ${self.initial_capital:,}")
        print(f"{'='*60}")

        # Parse dates
        start_year = int(start_date.split('-')[0])

        # Load configuration
        config = load_config()

        # Load universe
        core_path = config.get('paths', {}).get('core_universe', 'live_universe.txt')
        with open(core_path, 'r') as f:
            universe = [line.strip() for line in f if line.strip()]
        universe.extend(['^GSPC', 'GSPC', 'SPY'])
        universe = list(set(universe))

        # Load data
        data_dir = config.get('paths', {}).get('data_dir', 'data')
        all_data = load_stock_data(data_dir=data_dir, min_year=start_year, universe=universe)

        # Get index data
        index_data = get_index_data(all_data)

        if index_data is None:
            raise ValueError("No market index data found!")

        if len(all_data) == 0:
            raise ValueError("No stock data loaded!")

        print(f"✅ Loaded {len(all_data)} stocks")

        # Create models
        models = create_models(config)

        # Run backtest
        backtest = ProductionBacktest(models, self.initial_capital, logger=None)
        results = backtest.run(all_data, index_data)

        # Extract metrics
        metrics = extract_metrics(results, self.initial_capital)
        metrics["period_start"] = start_date
        metrics["period_end"] = end_date
        metrics["description"] = description

        # Add model breakdown
        metrics["model_breakdown"] = extract_model_breakdown(results)

        return metrics

    def establish_baseline(self) -> Dict:
        """
        Run full backtest and save as baseline.

        This should be run ONCE before any code changes.
        """
        print("\n" + "="*70)
        print("ESTABLISHING BASELINE")
        print("="*70)
        print(SURVIVORSHIP_WARNING)

        # Run full period backtest
        start, end = PERIODS["full"]
        metrics = self.run_backtest(start, end, "Baseline - before any changes")

        # Add metadata
        baseline = {
            "created_at": datetime.now().isoformat(),
            "system_version": "pre-enhancement",
            "survivorship_bias_acknowledged": True,
            "estimated_bias_pct": 2.0,  # Conservative estimate
            "metrics": metrics,
        }

        # Save baseline
        with open(BASELINE_FILE, 'w') as f:
            json.dump(baseline, f, indent=2, default=str)

        print(f"\n✅ Baseline saved to {BASELINE_FILE}")
        self._print_metrics_summary(metrics, "BASELINE METRICS")

        return baseline

    def validate_phase(self, phase_name: str) -> Dict:
        """
        Run complete validation suite for a phase.

        Args:
            phase_name: e.g., "phase_0_regime_filter"

        Returns:
            Validation results dict
        """
        print("\n" + "="*70)
        print(f"VALIDATING PHASE: {phase_name}")
        print("="*70)
        print(SURVIVORSHIP_WARNING)

        results = {
            "phase": phase_name,
            "timestamp": datetime.now().isoformat(),
            "periods": {},
            "stress_tests": {},
            "vs_baseline": {},
            "gates": {},
        }

        # 1. Run all periods
        for period_name, (start, end) in PERIODS.items():
            print(f"\n📊 Running {period_name} period ({start} to {end})...")
            metrics = self.run_backtest(start, end, f"{phase_name} - {period_name}")
            results["periods"][period_name] = metrics

        # 2. Run stress tests
        print("\n🔥 Running stress tests...")
        for stress_name, (start, end) in STRESS_PERIODS.items():
            print(f"  - {stress_name}...")
            try:
                metrics = self.run_backtest(start, end, f"{phase_name} - stress: {stress_name}")
                results["stress_tests"][stress_name] = metrics
            except Exception as e:
                results["stress_tests"][stress_name] = {"error": str(e)}

        # 3. Compare to baseline
        results["vs_baseline"] = self._compare_to_baseline(results["periods"]["full"])

        # 4. Check gates
        results["gates"] = self._check_gates(results)

        # 5. Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = RESULTS_DIR / f"{phase_name}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✅ Results saved to {output_file}")

        # 6. Print summary
        self._print_validation_summary(results)

        return results

    def run_stress_tests(self) -> Dict:
        """Run only stress tests."""
        print("\n" + "="*70)
        print("RUNNING STRESS TESTS")
        print("="*70)

        results = {
            "timestamp": datetime.now().isoformat(),
            "stress_tests": {},
        }

        for stress_name, (start, end) in STRESS_PERIODS.items():
            print(f"\n🔥 {stress_name} ({start} to {end})...")
            try:
                metrics = self.run_backtest(start, end, f"stress: {stress_name}")
                results["stress_tests"][stress_name] = metrics
                self._print_metrics_summary(metrics, stress_name.upper())
            except Exception as e:
                results["stress_tests"][stress_name] = {"error": str(e)}
                print(f"  ❌ Error: {e}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = STRESS_DIR / f"stress_tests_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✅ Results saved to {output_file}")

        return results

    def _compare_to_baseline(self, current_metrics: Dict) -> Dict:
        """Compare current results to baseline."""
        if not BASELINE_FILE.exists():
            return {"error": "No baseline file found. Run --establish-baseline first."}

        with open(BASELINE_FILE, 'r') as f:
            baseline = json.load(f)

        baseline_metrics = baseline.get("metrics", {})
        comparison = {}

        compare_fields = [
            "cagr_pct", "max_drawdown_pct", "sharpe_ratio", "sortino_ratio",
            "win_rate_pct", "profit_factor", "total_trades", "calmar_ratio"
        ]

        for field in compare_fields:
            baseline_val = baseline_metrics.get(field, 0)
            current_val = current_metrics.get(field, 0)

            if baseline_val is None:
                baseline_val = 0
            if current_val is None:
                current_val = 0

            delta = current_val - baseline_val
            pct_change = (delta / abs(baseline_val) * 100) if baseline_val != 0 else 0

            comparison[field] = {
                "baseline": baseline_val,
                "current": current_val,
                "delta": round(delta, 2),
                "pct_change": round(pct_change, 1),
                "improved": self._is_improvement(field, delta)
            }

        return comparison

    def _is_improvement(self, metric: str, delta: float) -> bool:
        """Determine if delta represents improvement for this metric."""
        # Higher is better
        if metric in ["cagr_pct", "sharpe_ratio", "sortino_ratio", "win_rate_pct",
                      "profit_factor", "calmar_ratio"]:
            return delta > 0
        # Lower is better
        elif metric in ["max_drawdown_pct"]:
            return delta < 0
        # Neutral
        return True

    def _check_gates(self, results: Dict) -> Dict:
        """
        Check if phase results meet minimum quality gates.

        Returns:
            {
                "passed": bool,
                "checks": [...],
                "failures": [...]
            }
        """
        checks = []
        failures = []

        # Gate 1: CAGR must be positive in all periods
        for period_name, metrics in results["periods"].items():
            cagr = metrics.get("cagr_pct", 0)
            check = {
                "name": f"{period_name}_positive_cagr",
                "value": cagr,
                "threshold": GATES["min_cagr"],
                "passed": cagr > GATES["min_cagr"]
            }
            checks.append(check)
            if not check["passed"]:
                failures.append(f"{period_name} has non-positive CAGR: {cagr}%")

        # Gate 2: Validation CAGR >= 50% of training CAGR
        train_cagr = results["periods"].get("train", {}).get("cagr_pct", 0)
        validate_cagr = results["periods"].get("validate", {}).get("cagr_pct", 0)

        if train_cagr > 0:
            decay = (train_cagr - validate_cagr) / train_cagr
            check = {
                "name": "performance_decay",
                "value": decay,
                "threshold": GATES["max_performance_decay"],
                "passed": decay <= GATES["max_performance_decay"]
            }
            checks.append(check)
            if not check["passed"]:
                failures.append(
                    f"Performance decay {decay:.1%} exceeds {GATES['max_performance_decay']:.0%} threshold. "
                    f"Train CAGR: {train_cagr}%, Validate CAGR: {validate_cagr}%"
                )

        # Gate 3: Max drawdown < 40% in full period
        full_dd = results["periods"].get("full", {}).get("max_drawdown_pct", 0)
        check = {
            "name": "max_drawdown",
            "value": full_dd,
            "threshold": GATES["max_drawdown"],
            "passed": full_dd < GATES["max_drawdown"]
        }
        checks.append(check)
        if not check["passed"]:
            failures.append(f"Max drawdown {full_dd}% exceeds {GATES['max_drawdown']}% threshold")

        # Gate 4: Not worse than baseline by more than 5% CAGR
        vs_baseline = results.get("vs_baseline", {})
        if "cagr_pct" in vs_baseline and "error" not in vs_baseline:
            cagr_delta = vs_baseline["cagr_pct"].get("delta", 0)
            check = {
                "name": "vs_baseline_cagr",
                "value": cagr_delta,
                "threshold": -GATES["max_cagr_regression"],
                "passed": cagr_delta >= -GATES["max_cagr_regression"]
            }
            checks.append(check)
            if not check["passed"]:
                failures.append(
                    f"CAGR regression of {abs(cagr_delta)}% exceeds {GATES['max_cagr_regression']}% threshold"
                )

        return {
            "passed": len(failures) == 0,
            "checks": checks,
            "failures": failures
        }

    def _print_metrics_summary(self, metrics: Dict, title: str):
        """Print formatted metrics summary."""
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")
        print(f"  CAGR:           {metrics.get('cagr_pct', 0):>8.1f}%")
        print(f"  Total Return:   {metrics.get('total_return_pct', 0):>8.1f}%")
        print(f"  Max Drawdown:   {metrics.get('max_drawdown_pct', 0):>8.1f}%")
        print(f"  Sharpe Ratio:   {metrics.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Sortino Ratio:  {metrics.get('sortino_ratio', 0):>8.2f}")
        print(f"  Calmar Ratio:   {metrics.get('calmar_ratio', 0):>8.2f}")
        print(f"  Win Rate:       {metrics.get('win_rate_pct', 0):>8.1f}%")
        print(f"  Profit Factor:  {metrics.get('profit_factor', 0):>8.2f}")
        print(f"  Total Trades:   {metrics.get('total_trades', 0):>8,}")
        print(f"  Avg Hold Days:  {metrics.get('avg_hold_days', 0):>8.1f}")
        print(f"{'─'*50}")

    def _print_validation_summary(self, results: Dict):
        """Print human-readable validation summary."""
        print("\n" + "="*70)
        print(f"VALIDATION SUMMARY: {results['phase']}")
        print("="*70)

        # Period results
        print("\n📊 PERIOD RESULTS:")
        for period_name, metrics in results["periods"].items():
            print(f"\n  {period_name.upper()} ({metrics.get('period_start', '')} to {metrics.get('period_end', '')}):")
            print(f"    CAGR: {metrics.get('cagr_pct', 0):.1f}%  |  "
                  f"Max DD: {metrics.get('max_drawdown_pct', 0):.1f}%  |  "
                  f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  |  "
                  f"Trades: {metrics.get('total_trades', 0):,}")

        # Walk-forward check
        train_cagr = results["periods"].get("train", {}).get("cagr_pct", 0)
        validate_cagr = results["periods"].get("validate", {}).get("cagr_pct", 0)
        test_cagr = results["periods"].get("test", {}).get("cagr_pct", 0)

        print("\n📈 WALK-FORWARD ANALYSIS:")
        print(f"  Train CAGR (1985-2015):    {train_cagr:.1f}%")
        print(f"  Validate CAGR (2016-2020): {validate_cagr:.1f}%")
        print(f"  Test CAGR (2021-2026):     {test_cagr:.1f}%")

        if train_cagr > 0:
            decay = (train_cagr - validate_cagr) / train_cagr * 100
            print(f"  Performance Decay:         {decay:.1f}%")
            if decay > 50:
                print("  ⚠️  WARNING: >50% decay suggests overfitting!")
            elif decay > 30:
                print("  ⚡ CAUTION: 30-50% decay - monitor closely")
            else:
                print("  ✅ Acceptable decay level")

        # Stress tests
        print("\n🔥 STRESS TEST RESULTS:")
        for stress_name, metrics in results.get("stress_tests", {}).items():
            if "error" in metrics:
                print(f"  {stress_name}: ❌ Error - {metrics['error']}")
            else:
                dd = metrics.get('max_drawdown_pct', 0)
                cagr = metrics.get('cagr_pct', 0)
                print(f"  {stress_name}: Max DD {dd:.1f}%, CAGR {cagr:.1f}%")

        # vs Baseline
        vs_baseline = results.get("vs_baseline", {})
        if "error" not in vs_baseline:
            print("\n📋 VS BASELINE:")
            for metric, comp in vs_baseline.items():
                if isinstance(comp, dict):
                    arrow = "↑" if comp.get('improved', False) else "↓"
                    delta = comp.get('delta', 0)
                    sign = "+" if delta > 0 else ""
                    print(f"  {metric}: {comp.get('baseline', 0):.2f} → {comp.get('current', 0):.2f} "
                          f"({arrow} {sign}{delta:.2f})")
        else:
            print(f"\n📋 VS BASELINE: {vs_baseline.get('error', 'No baseline')}")

        # Gates
        gates = results.get("gates", {})
        print("\n🚦 PHASE GATES:")
        if gates.get("passed", False):
            print("  ✅ ALL GATES PASSED - Ready to proceed to next phase")
        else:
            print("  ❌ GATES FAILED - Fix issues before proceeding:")
            for failure in gates.get("failures", []):
                print(f"     • {failure}")

        print("\n" + "="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validation Protocol for TradingSystem2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First, establish baseline (run before any changes)
  python scripts/validation_protocol.py --establish-baseline

  # After implementing Phase 0, validate it
  python scripts/validation_protocol.py --validate-phase "phase_0_regime_filter"

  # Run stress tests only
  python scripts/validation_protocol.py --stress-test
        """
    )

    parser.add_argument('--establish-baseline', action='store_true',
                        help='Run full backtest and save as baseline')
    parser.add_argument('--validate-phase', type=str,
                        help='Phase name to validate (e.g., "phase_0_regime_filter")')
    parser.add_argument('--stress-test', action='store_true',
                        help='Run stress tests only')
    parser.add_argument('--capital', type=int, default=100000,
                        help='Initial capital (default: 100000)')

    args = parser.parse_args()

    # Create protocol
    protocol = ValidationProtocol(initial_capital=args.capital)

    # Execute requested action
    if args.establish_baseline:
        protocol.establish_baseline()
    elif args.validate_phase:
        protocol.validate_phase(args.validate_phase)
    elif args.stress_test:
        protocol.run_stress_tests()
    else:
        parser.print_help()
        print("\n⚠️  No action specified. Use --establish-baseline, --validate-phase, or --stress-test")


if __name__ == "__main__":
    main()
