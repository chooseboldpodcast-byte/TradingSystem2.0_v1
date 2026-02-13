# core/backtest_logger.py
"""
Backtest Diagnostic Logger
==========================

Comprehensive logging for troubleshooting backtest performance.

LOGS:
1. Signal generation - every entry/exit signal with full context
2. Signal rejection - detailed reasons why signals were not executed
3. Capital allocation - state at each decision point
4. Position events - opens, closes, stop losses

OUTPUT:
- Console: Summary level information
- JSON files: Detailed machine-readable logs for analysis
- Log files: Human-readable detailed logs

Usage:
    from core.backtest_logger import BacktestLogger

    logger = BacktestLogger(run_id='20260121_143022')
    logger.log_signal_generated(...)
    ...
    logger.save_all()
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class SignalType(Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


class RejectionReason(Enum):
    NO_CAPITAL = "NO_CAPITAL"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"  # Legacy: any model has position
    DUPLICATE_POSITION_SAME_MODEL = "DUPLICATE_POSITION_SAME_MODEL"  # Same model already has position
    SYMBOL_EXPOSURE_EXCEEDED = "SYMBOL_EXPOSURE_EXCEEDED"  # Total exposure cap hit
    MODEL_ALLOCATION_EXCEEDED = "MODEL_ALLOCATION_EXCEEDED"  # Model's allocation exhausted
    ALLOCATION_EXCEEDED = "ALLOCATION_EXCEEDED"
    POSITION_LOCKED = "POSITION_LOCKED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    BELOW_MINIMUM_SHARES = "BELOW_MINIMUM_SHARES"


class ExitReason(Enum):
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    STAGE_CHANGE = "STAGE_CHANGE"
    MODEL_EXIT = "MODEL_EXIT"
    END_OF_BACKTEST = "END_OF_BACKTEST"
    TIME_EXIT = "TIME_EXIT"


@dataclass
class SignalEvent:
    """Record of a signal generation event"""
    timestamp: str
    date: str  # Trading date
    symbol: str
    model: str
    signal_type: str  # ENTRY or EXIT
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    reason: Optional[str] = None  # For exits


@dataclass
class RejectionEvent:
    """Record of a signal rejection"""
    timestamp: str
    date: str
    symbol: str
    model: str
    reason: str
    details: Dict = field(default_factory=dict)
    available_capital: Optional[float] = None
    required_capital: Optional[float] = None
    model_allocation: Optional[float] = None
    model_deployed: Optional[float] = None


@dataclass
class CapitalStateSnapshot:
    """Snapshot of capital allocation state"""
    timestamp: str
    date: str
    event_type: str  # SIGNAL_PROCESSING, POSITION_OPEN, POSITION_CLOSE
    total_capital: float
    available_capital: float
    deployed_capital: float
    model_allocations: Dict[str, Dict] = field(default_factory=dict)
    open_positions_count: int = 0
    trigger_symbol: Optional[str] = None
    trigger_model: Optional[str] = None


@dataclass
class PositionEvent:
    """Record of a position open/close event"""
    timestamp: str
    date: str
    event_type: str  # OPEN, CLOSE
    symbol: str
    model: str
    shares: int
    price: float
    position_value: float
    stop_loss: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    hold_days: Optional[int] = None
    exit_reason: Optional[str] = None


class BacktestLogger:
    """
    Comprehensive backtest logging system

    Provides structured logging for:
    - Scanner execution and results
    - Signal generation and processing
    - Capital allocation decisions
    - Position management
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        logs_dir: str = 'logs',
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialize the backtest logger

        Args:
            run_id: Unique identifier for this run (auto-generated if not provided)
            logs_dir: Base directory for log files
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logs_dir = logs_dir

        # Create run-specific log directory
        self.run_log_dir = os.path.join(logs_dir, 'backtest_runs', f'run_{self.run_id}')
        os.makedirs(self.run_log_dir, exist_ok=True)

        # Initialize data stores
        self.signal_events: List[SignalEvent] = []
        self.rejection_events: List[RejectionEvent] = []
        self.capital_snapshots: List[CapitalStateSnapshot] = []
        self.position_events: List[PositionEvent] = []

        # Setup Python logging
        self._setup_logging(console_level, file_level)

        # Log initialization
        self.logger.info(f"{'='*60}")
        self.logger.info(f"BACKTEST LOGGER INITIALIZED")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Log Directory: {self.run_log_dir}")
        self.logger.info(f"{'='*60}")

    def _setup_logging(self, console_level: int, file_level: int):
        """Setup Python logging with console and file handlers"""
        self.logger = logging.getLogger(f'backtest_{self.run_id}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        # Console handler - summary level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler - detailed level
        log_file = os.path.join(self.run_log_dir, 'backtest.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    # =========================================================================
    # SIGNAL LOGGING
    # =========================================================================

    def log_signal_generated(
        self,
        date: datetime,
        symbol: str,
        model: str,
        signal_type: str,
        entry_price: float = None,
        stop_loss: float = None,
        confidence: float = None,
        metadata: Dict = None,
        reason: str = None
    ):
        """Log a generated signal"""
        event = SignalEvent(
            timestamp=datetime.now().isoformat(),
            date=str(date.date()) if hasattr(date, 'date') else str(date),
            symbol=symbol,
            model=model,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=confidence,
            metadata=metadata or {},
            reason=reason
        )
        self.signal_events.append(event)

        if signal_type == 'ENTRY':
            self.logger.debug(
                f"SIGNAL ENTRY: {date.date() if hasattr(date, 'date') else date} | {symbol} | {model} | "
                f"Price=${entry_price:.2f} | Stop=${stop_loss:.2f} | Conf={confidence:.2f}"
            )
        else:
            self.logger.debug(
                f"SIGNAL EXIT: {date.date() if hasattr(date, 'date') else date} | {symbol} | {model} | "
                f"Reason={reason}"
            )

    def log_signal_rejected(
        self,
        date: datetime,
        symbol: str,
        model: str,
        reason: str,
        available_capital: float = None,
        required_capital: float = None,
        model_allocation: float = None,
        model_deployed: float = None,
        details: Dict = None
    ):
        """Log a rejected signal with detailed reason"""
        event = RejectionEvent(
            timestamp=datetime.now().isoformat(),
            date=str(date.date()) if hasattr(date, 'date') else str(date),
            symbol=symbol,
            model=model,
            reason=reason,
            details=details or {},
            available_capital=available_capital,
            required_capital=required_capital,
            model_allocation=model_allocation,
            model_deployed=model_deployed
        )
        self.rejection_events.append(event)

        # Build log message with optional capital info
        msg = f"SIGNAL REJECTED: {date.date() if hasattr(date, 'date') else date} | {symbol} | {model} | Reason={reason}"
        if available_capital is not None and required_capital is not None:
            msg += f" | Avail=${available_capital:,.0f} | Req=${required_capital:,.0f}"
        self.logger.debug(msg)

    # =========================================================================
    # CAPITAL ALLOCATION LOGGING
    # =========================================================================

    def log_capital_state(
        self,
        date: datetime,
        event_type: str,
        total_capital: float,
        available_capital: float,
        deployed_capital: float,
        model_allocations: Dict[str, Dict],
        open_positions_count: int,
        trigger_symbol: str = None,
        trigger_model: str = None
    ):
        """Log capital allocation state snapshot"""
        snapshot = CapitalStateSnapshot(
            timestamp=datetime.now().isoformat(),
            date=str(date.date()) if hasattr(date, 'date') else str(date),
            event_type=event_type,
            total_capital=total_capital,
            available_capital=available_capital,
            deployed_capital=deployed_capital,
            model_allocations=model_allocations,
            open_positions_count=open_positions_count,
            trigger_symbol=trigger_symbol,
            trigger_model=trigger_model
        )
        self.capital_snapshots.append(snapshot)

        self.logger.debug(
            f"CAPITAL STATE: {event_type} | Total=${total_capital:,.0f} | "
            f"Avail=${available_capital:,.0f} | Deployed=${deployed_capital:,.0f} | "
            f"Positions={open_positions_count}"
        )

    # =========================================================================
    # POSITION LOGGING
    # =========================================================================

    def log_position_opened(
        self,
        date: datetime,
        symbol: str,
        model: str,
        shares: int,
        price: float,
        stop_loss: float
    ):
        """Log position open event"""
        position_value = shares * price
        event = PositionEvent(
            timestamp=datetime.now().isoformat(),
            date=str(date.date()) if hasattr(date, 'date') else str(date),
            event_type='OPEN',
            symbol=symbol,
            model=model,
            shares=shares,
            price=price,
            position_value=position_value,
            stop_loss=stop_loss
        )
        self.position_events.append(event)

        self.logger.info(
            f"POSITION OPEN: {date.date() if hasattr(date, 'date') else date} | {symbol} | {model} | "
            f"{shares} shares @ ${price:.2f} = ${position_value:,.0f}"
        )

    def log_position_closed(
        self,
        date: datetime,
        symbol: str,
        model: str,
        shares: int,
        price: float,
        pnl: float,
        pnl_pct: float,
        hold_days: int,
        exit_reason: str
    ):
        """Log position close event"""
        position_value = shares * price
        event = PositionEvent(
            timestamp=datetime.now().isoformat(),
            date=str(date.date()) if hasattr(date, 'date') else str(date),
            event_type='CLOSE',
            symbol=symbol,
            model=model,
            shares=shares,
            price=price,
            position_value=position_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_days=hold_days,
            exit_reason=exit_reason
        )
        self.position_events.append(event)

        result = "WIN" if pnl > 0 else "LOSS"
        self.logger.info(
            f"POSITION CLOSE: {date.date() if hasattr(date, 'date') else date} | {symbol} | {model} | "
            f"{result} ${pnl:,.0f} ({pnl_pct:+.1f}%) | {hold_days}d | {exit_reason}"
        )

    # =========================================================================
    # DATA SUFFICIENCY LOGGING
    # =========================================================================

    def log_data_insufficient(
        self,
        symbol: str,
        available_days: int,
        required_days: int,
        reason: str = "Below minimum threshold"
    ):
        """Log when a symbol is skipped due to insufficient data"""
        self.logger.debug(
            f"DATA INSUFFICIENT: {symbol} | {available_days} days available, "
            f"{required_days} required | {reason}"
        )

    # =========================================================================
    # SUMMARY AND EXPORT
    # =========================================================================

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'run_id': self.run_id,
            'signals': {
                'total_generated': len(self.signal_events),
                'entries': len([s for s in self.signal_events if s.signal_type == 'ENTRY']),
                'exits': len([s for s in self.signal_events if s.signal_type == 'EXIT'])
            },
            'rejections': {
                'total': len(self.rejection_events),
                'by_reason': self._count_rejections_by_reason()
            },
            'positions': {
                'total_opened': len([p for p in self.position_events if p.event_type == 'OPEN']),
                'total_closed': len([p for p in self.position_events if p.event_type == 'CLOSE'])
            }
        }

    def _count_rejections_by_reason(self) -> Dict[str, int]:
        """Count rejections by reason"""
        counts = {}
        for event in self.rejection_events:
            reason = event.reason
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _count_rejections_by_model(self) -> Dict[str, Dict[str, int]]:
        """Count rejections by model and reason"""
        counts = {}
        for event in self.rejection_events:
            model = event.model
            reason = event.reason
            if model not in counts:
                counts[model] = {}
            counts[model][reason] = counts[model].get(reason, 0) + 1
        return counts

    def save_all(self):
        """Save all logs to files"""
        self.logger.info(f"Saving logs to {self.run_log_dir}")

        # Save signal events
        if self.signal_events:
            signals_file = os.path.join(self.run_log_dir, 'signals.json')
            with open(signals_file, 'w') as f:
                json.dump([asdict(s) for s in self.signal_events], f, indent=2, default=str)
            self.logger.info(f"  Signals: {signals_file}")

        # Save rejection events
        if self.rejection_events:
            rejections_file = os.path.join(self.run_log_dir, 'rejections.json')
            with open(rejections_file, 'w') as f:
                json.dump({
                    'summary': {
                        'total': len(self.rejection_events),
                        'by_reason': self._count_rejections_by_reason(),
                        'by_model': self._count_rejections_by_model()
                    },
                    'events': [asdict(r) for r in self.rejection_events]
                }, f, indent=2, default=str)
            self.logger.info(f"  Rejections: {rejections_file}")

        # Save capital snapshots
        if self.capital_snapshots:
            capital_file = os.path.join(self.run_log_dir, 'capital_snapshots.json')
            with open(capital_file, 'w') as f:
                json.dump([asdict(c) for c in self.capital_snapshots], f, indent=2, default=str)
            self.logger.info(f"  Capital snapshots: {capital_file}")

        # Save position events
        if self.position_events:
            positions_file = os.path.join(self.run_log_dir, 'positions.json')
            with open(positions_file, 'w') as f:
                json.dump([asdict(p) for p in self.position_events], f, indent=2, default=str)
            self.logger.info(f"  Position events: {positions_file}")

        # Save summary
        summary_file = os.path.join(self.run_log_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(self.get_summary(), f, indent=2, default=str)
        self.logger.info(f"  Summary: {summary_file}")

        self.logger.info(f"All logs saved successfully")

    def print_final_summary(self):
        """Print final summary to console"""
        summary = self.get_summary()

        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC LOG SUMMARY")
        print(f"{'='*60}")

        # Signal summary
        print(f"\nSIGNALS:")
        print(f"  Total generated: {summary['signals']['total_generated']}")
        print(f"  Entry signals: {summary['signals']['entries']}")
        print(f"  Exit signals: {summary['signals']['exits']}")

        # Rejection summary
        print(f"\nREJECTIONS:")
        print(f"  Total rejected: {summary['rejections']['total']}")
        if summary['rejections']['by_reason']:
            for reason, count in sorted(
                summary['rejections']['by_reason'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"    - {reason}: {count}")

        # Position summary
        print(f"\nPOSITIONS:")
        print(f"  Opened: {summary['positions']['total_opened']}")
        print(f"  Closed: {summary['positions']['total_closed']}")

        print(f"\nLog files saved to: {self.run_log_dir}")
        print(f"{'='*60}\n")
