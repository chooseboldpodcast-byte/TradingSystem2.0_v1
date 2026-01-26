# core/portfolio_manager.py
"""
Multi-Model Portfolio Manager - POSITION LAYERING VERSION

Manages capital allocation across multiple trading models, ensuring:
- No capital allocation conflicts
- Proper tracking of which model owns which position
- Fair capital distribution based on model allocations
- Chronological trade processing

POSITION LAYERING: Multiple models can now hold positions in the same stock.
- Each model can have at most ONE position per symbol
- Total exposure per symbol is capped at MAX_STOCK_EXPOSURE_PCT
- This allows investment and trading strategies to coexist on same stocks

KEY FIX: Added get_model_available_capital() method to properly calculate
         model-specific available capital for position sizing
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Maximum exposure to any single stock across all models
# Note: Using 25% allows position layering without overly constraining the winners
# The cap scales with current_capital (not initial) to allow for compounding
MAX_STOCK_EXPOSURE_PCT = 0.25


class PortfolioManager:
    """
    Manages portfolio across multiple trading models with POSITION LAYERING

    Key responsibilities:
    - Track available capital
    - Allocate capital to models based on allocation percentages
    - Allow multiple models to hold same stock (position layering)
    - Prevent same model from having duplicate positions on same symbol
    - Cap total exposure per stock at MAX_STOCK_EXPOSURE_PCT
    - Track which model owns which position
    - Calculate portfolio-level metrics

    Position Key Format: "{symbol}_{model_name}" (e.g., "AAPL_Weinstein_Core")
    """

    def __init__(self, initial_capital: float, models: List, verbose: bool = False,
                 max_stock_exposure_pct: float = None):
        """
        Initialize portfolio manager

        Args:
            initial_capital: Starting capital
            models: List of model instances (must inherit from BaseModel)
            verbose: If True, print detailed trade open/close messages
            max_stock_exposure_pct: Override default MAX_STOCK_EXPOSURE_PCT (0.12)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.verbose = verbose
        self.max_stock_exposure_pct = max_stock_exposure_pct or MAX_STOCK_EXPOSURE_PCT

        # Store models by name
        self.models = {model.name: model for model in models}

        # Track open positions: {position_key: position_dict}
        # Position key format: "{symbol}_{model_name}"
        self.open_positions = {}

        # Track closed trades
        self.closed_trades = []

        # Capital allocation tracking
        self.capital_by_model = {model.name: 0.0 for model in models}

        print(f"\n{'='*60}")
        print(f"PORTFOLIO MANAGER INITIALIZED (Position Layering Enabled)")
        print(f"{'='*60}")
        print(f"Initial Capital: ${initial_capital:,.0f}")
        print(f"Max Stock Exposure: {self.max_stock_exposure_pct*100:.0f}%")
        print(f"Models: {len(models)}")
        for model in models:
            max_allocation = initial_capital * model.allocation_pct
            print(f"  - {model.name}: {model.allocation_pct*100:.0f}% (max ${max_allocation:,.0f})")
        print(f"{'='*60}\n")

    @staticmethod
    def make_position_key(symbol: str, model_name: str) -> str:
        """Create position key from symbol and model name"""
        return f"{symbol}_{model_name}"

    @staticmethod
    def parse_position_key(position_key: str) -> Tuple[str, str]:
        """Parse position key into (symbol, model_name)"""
        # Find the last underscore to handle symbols with underscores
        parts = position_key.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return position_key, ""

    def get_symbol_exposure(self, symbol: str) -> float:
        """
        Get total capital exposure for a symbol across all models

        Args:
            symbol: Stock symbol

        Returns:
            Total position cost for this symbol across all models
        """
        total_exposure = 0.0
        for pos_key, position in self.open_positions.items():
            if position['symbol'] == symbol:
                total_exposure += position['position_cost']
        return total_exposure

    def get_positions_for_symbol(self, symbol: str) -> Dict[str, Dict]:
        """
        Get all open positions for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dict of {model_name: position_dict} for all models holding this symbol
        """
        positions = {}
        for pos_key, position in self.open_positions.items():
            if position['symbol'] == symbol:
                positions[position['model']] = position
        return positions

    def has_position(self, symbol: str, model_name: str = None) -> bool:
        """
        Check if a position exists

        Args:
            symbol: Stock symbol
            model_name: If provided, check if this specific model has a position.
                       If None, check if ANY model has a position on this symbol.

        Returns:
            True if position exists
        """
        if model_name:
            pos_key = self.make_position_key(symbol, model_name)
            return pos_key in self.open_positions
        else:
            # Check if any model has this symbol
            for position in self.open_positions.values():
                if position['symbol'] == symbol:
                    return True
            return False
    
    def get_model_available_capital(self, model_name: str) -> float:
        """
        Calculate available capital for a specific model
        
        â­ THIS IS THE KEY FIX FOR THE CAPITAL ALLOCATION BUG â­
        
        This method properly calculates how much capital a model has available
        by considering:
        1. Model's initial allocation (% of starting capital)
        2. Capital currently deployed in open positions
        3. Profits accumulated by this model (can reinvest)
        
        Args:
            model_name: Name of the model (e.g., 'Weinstein_Core')
        
        Returns:
            Amount of capital this model can still deploy
        
        Example:
            Initial capital: $100,000
            Weinstein allocation: 40% = $40,000
            Current positions: 3 positions totaling $12,000
            Weinstein profits: $5,000
            
            Available = $40,000 - $12,000 + $5,000 = $33,000
            
        Why this fixes the bug:
            Before: Position sizing used total portfolio capital ($100K)
            After:  Position sizing uses model's available capital ($40K for Weinstein)
            
            This allows the model to take ~10 positions of $4K each
            instead of only 4-5 positions of $10K each before hitting limits.
        """
        model = self.models[model_name]
        
        # Calculate model's total allocation from initial capital
        allocated_capital = self.initial_capital * model.allocation_pct
        
        # Calculate capital currently deployed by this model
        capital_deployed = self.capital_by_model[model_name]
        
        # Calculate accumulated profits for this model (can be reinvested)
        model_trades = [t for t in self.closed_trades if t['model'] == model_name]
        model_profits = sum(t['pnl'] for t in model_trades)
        
        # Available = Allocated - Deployed + Profits
        available = allocated_capital - capital_deployed + model_profits
        
        # Can't exceed total portfolio available capital
        available = min(available, self.available_capital)
        
        return max(0, available)  # Never return negative
    
    def can_open_position(
        self,
        model_name: str,
        symbol: str,
        position_cost: float
    ) -> Tuple[bool, str]:
        """
        Check if model can open a new position

        POSITION LAYERING: Multiple models can hold same stock, but:
        - Same model cannot have duplicate position on same symbol
        - Total exposure per symbol capped at MAX_STOCK_EXPOSURE_PCT

        Args:
            model_name: Name of model requesting position
            symbol: Stock symbol
            position_cost: Cost of the position (shares * price)

        Returns:
            Tuple of (can_open: bool, rejection_reason: str or None)
        """
        # Check 1: This model doesn't already have a position on this symbol
        pos_key = self.make_position_key(symbol, model_name)
        if pos_key in self.open_positions:
            return False, 'DUPLICATE_POSITION_SAME_MODEL'

        # Check 2: Total symbol exposure doesn't exceed cap
        # Use CURRENT capital (not initial) so cap scales with portfolio growth
        current_exposure = self.get_symbol_exposure(symbol)
        max_exposure = self.current_capital * self.max_stock_exposure_pct
        if current_exposure + position_cost > max_exposure:
            return False, 'SYMBOL_EXPOSURE_EXCEEDED'

        # Check 3: Model has enough available capital
        model_available = self.get_model_available_capital(model_name)
        if position_cost > model_available:
            return False, 'MODEL_ALLOCATION_EXCEEDED'

        # Check 4: Sufficient total portfolio capital available
        if position_cost > self.available_capital:
            return False, 'NO_CAPITAL'

        return True, None
    
    def open_position(
        self,
        model_name: str,
        symbol: str,
        entry_date: datetime,
        entry_price: float,
        shares: int,
        stop_loss: float,
        metadata: Dict = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Open a new position

        Args:
            model_name: Name of model opening position
            symbol: Stock symbol
            entry_date: Date of entry
            entry_price: Entry price
            shares: Number of shares
            stop_loss: Stop loss price
            metadata: Additional tracking info

        Returns:
            Tuple of (success: bool, rejection_reason: str or None)
        """
        position_cost = shares * entry_price

        # Verify we can open this position
        can_open, rejection_reason = self.can_open_position(model_name, symbol, position_cost)
        if not can_open:
            return False, rejection_reason

        # Create position key for layered positions
        pos_key = self.make_position_key(symbol, model_name)

        # Create position record
        position = {
            'model': model_name,
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'position_cost': position_cost,
            'metadata': metadata or {}
        }

        # Update tracking - use position key instead of symbol
        self.open_positions[pos_key] = position
        self.available_capital -= position_cost
        self.capital_by_model[model_name] += position_cost

        if self.verbose:
            # Show if this is a layered position
            other_positions = self.get_positions_for_symbol(symbol)
            layer_info = f" [Layer {len(other_positions)}]" if len(other_positions) > 1 else ""
            print(f"  ✓ OPEN {symbol}: {shares} shares @ ${entry_price:.2f} [{model_name}]{layer_info}")
            print(f"    Cost: ${position_cost:,.0f}, Available: ${self.available_capital:,.0f}")
            print(f"    Model allocation: ${self.capital_by_model[model_name]:,.0f} / "
                  f"${self.initial_capital * self.models[model_name].allocation_pct:,.0f}")
            if len(other_positions) > 1:
                total_exposure = self.get_symbol_exposure(symbol)
                max_exposure = self.initial_capital * self.max_stock_exposure_pct
                print(f"    Symbol exposure: ${total_exposure:,.0f} / ${max_exposure:,.0f}")

        return True, None
    
    def close_position(
        self,
        symbol: str,
        exit_date: datetime,
        exit_price: float,
        exit_reason: str,
        model_name: str = None
    ) -> Optional[Dict]:
        """
        Close an open position

        Args:
            symbol: Stock symbol to close
            exit_date: Exit date
            exit_price: Exit price
            exit_reason: Reason for exit
            model_name: Model that owns the position (required for layered positions)

        Returns:
            Trade record dict, or None if position not found
        """
        # For position layering, we need model_name to find the right position
        if model_name:
            pos_key = self.make_position_key(symbol, model_name)
        else:
            # Backward compatibility: if no model_name, try to find by symbol
            # (only works if there's exactly one position for this symbol)
            matching_keys = [k for k, p in self.open_positions.items() if p['symbol'] == symbol]
            if len(matching_keys) == 1:
                pos_key = matching_keys[0]
            elif len(matching_keys) > 1:
                # Multiple positions - can't determine which to close without model_name
                if self.verbose:
                    print(f"  ⚠ Cannot close {symbol}: multiple positions exist, model_name required")
                return None
            else:
                return None  # No position found

        if pos_key not in self.open_positions:
            return None

        position = self.open_positions[pos_key]

        # Calculate P&L
        exit_value = position['shares'] * exit_price
        pnl = exit_value - position['position_cost']
        pnl_pct = (pnl / position['position_cost']) * 100
        hold_days = (exit_date - position['entry_date']).days

        # Create trade record
        trade = {
            'model': position['model'],
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_days': hold_days,
            'exit_reason': exit_reason,
            'entry_metadata': position['metadata']
        }

        # Update capital tracking
        self.available_capital += exit_value
        self.capital_by_model[position['model']] -= position['position_cost']
        self.current_capital = self.initial_capital + sum(t['pnl'] for t in self.closed_trades) + pnl

        # Update model performance
        self.models[position['model']].update_performance(pnl)

        # Record trade and remove position
        self.closed_trades.append(trade)
        del self.open_positions[pos_key]

        result_symbol = "✓" if pnl > 0 else "✗"

        if self.verbose:
            print(f"  {result_symbol} CLOSE {symbol}: {position['shares']} shares @ ${exit_price:.2f} [{position['model']}]")
            print(f"    P&L: ${pnl:,.0f} ({pnl_pct:+.1f}%), Hold: {hold_days} days, Reason: {exit_reason}")
            print(f"    Available capital: ${self.available_capital:,.0f}")

        return trade
    
    def check_stop_losses(
        self,
        current_date: datetime,
        stock_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """
        Check if any open positions hit their stop loss

        Args:
            current_date: Current date to check
            stock_data: Dictionary of stock dataframes {symbol: df}

        Returns:
            List of closed trade records
        """
        closed_trades = []

        # Iterate over position keys (which now include model name)
        for pos_key in list(self.open_positions.keys()):
            position = self.open_positions[pos_key]
            symbol = position['symbol']
            model_name = position['model']

            # Get current price data
            if symbol not in stock_data:
                continue

            df = stock_data[symbol]

            try:
                current_price = df.loc[current_date, 'low']

                # Check stop loss
                if current_price <= position['stop_loss']:
                    trade = self.close_position(
                        symbol=symbol,
                        exit_date=current_date,
                        exit_price=position['stop_loss'],
                        exit_reason='STOP_LOSS',
                        model_name=model_name
                    )
                    if trade:
                        closed_trades.append(trade)
            except KeyError:
                # Date not in this stock's dataframe
                continue

        return closed_trades
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary statistics"""
        total_position_value = sum(
            pos['shares'] * pos['entry_price']
            for pos in self.open_positions.values()
        )

        total_pnl = sum(t['pnl'] for t in self.closed_trades)

        # Count unique symbols with positions (for layering stats)
        unique_symbols = set(pos['symbol'] for pos in self.open_positions.values())
        layered_symbols = [s for s in unique_symbols
                          if len(self.get_positions_for_symbol(s)) > 1]

        return {
            'current_capital': self.current_capital,
            'available_capital': self.available_capital,
            'deployed_capital': total_position_value,
            'deployment_pct': (total_position_value / self.current_capital * 100)
                             if self.current_capital > 0 else 0,
            'open_positions': len(self.open_positions),
            'unique_symbols': len(unique_symbols),
            'layered_symbols': len(layered_symbols),
            'closed_trades': len(self.closed_trades),
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / self.initial_capital * 100)
        }

    def get_model_summary(self, model_name: str) -> Dict:
        """Get summary for specific model"""
        model = self.models[model_name]

        # Get trades for this model
        model_trades = [t for t in self.closed_trades if t['model'] == model_name]

        # Count open positions for this model
        open_count = sum(1 for p in self.open_positions.values() if p['model'] == model_name)

        return {
            'name': model_name,
            'allocation_pct': model.allocation_pct,
            'capital_used': self.capital_by_model[model_name],
            'open_positions': open_count,
            'closed_trades': len(model_trades),
            'total_pnl': sum(t['pnl'] for t in model_trades),
            'win_rate': model_trades and
                       (sum(1 for t in model_trades if t['pnl'] > 0) / len(model_trades) * 100) or 0
        }
    
    def get_all_models_summary(self) -> pd.DataFrame:
        """Get summary dataframe for all models"""
        summaries = [self.get_model_summary(name) for name in self.models.keys()]
        return pd.DataFrame(summaries)
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get all closed trades as dataframe"""
        if not self.closed_trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.closed_trades)
    
    def print_final_summary(self):
        """Print final portfolio summary"""
        summary = self.get_portfolio_summary()
        
        print(f"\n{'='*60}")
        print(f"FINAL PORTFOLIO SUMMARY")
        print(f"{'='*60}")
        print(f"Initial Capital:     ${self.initial_capital:,.0f}")
        print(f"Final Capital:       ${summary['current_capital']:,.0f}")
        print(f"Total P&L:           ${summary['total_pnl']:,.0f}")
        print(f"Total Return:        {summary['total_return_pct']:.2f}%")
        print(f"\nClosed Trades:       {summary['closed_trades']}")
        print(f"Open Positions:      {summary['open_positions']}")
        print(f"Deployed Capital:    ${summary['deployed_capital']:,.0f} ({summary['deployment_pct']:.1f}%)")
        
        # Model breakdown
        print(f"\n{'='*60}")
        print(f"PERFORMANCE BY MODEL")
        print(f"{'='*60}")
        
        model_df = self.get_all_models_summary()
        for _, row in model_df.iterrows():
            print(f"\n{row['name']}:")
            print(f"  Allocation:      {row['allocation_pct']*100:.0f}%")
            print(f"  Capital Used:    ${row['capital_used']:,.0f}")
            print(f"  Closed Trades:   {row['closed_trades']:.0f}")
            print(f"  Total P&L:       ${row['total_pnl']:,.0f}")
            print(f"  Win Rate:        {row['win_rate']:.1f}%")
        
        print(f"\n{'='*60}\n")
