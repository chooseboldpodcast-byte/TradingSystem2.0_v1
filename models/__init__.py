# models/__init__.py
"""
Trading Models Package v3.1

Contains all trading model implementations:
- weinstein_core: Stan Weinstein Stage Analysis
- rsi_mean_reversion: RSI-based mean reversion (legacy, being replaced)
- enhanced_mean_reversion: Improved mean reversion with stricter filters
- dual_momentum: Absolute + Relative momentum regime filter
- momentum_52w_high: 52-week high momentum breakouts
- consolidation_breakout: Consolidation pattern breakouts
- vcp: Volatility Contraction Pattern (Minervini)
- pocket_pivot: Pocket Pivot (Kacher/Morales)
- rs_breakout: Relative Strength Breakout (O'Neil)
- high_tight_flag: High Tight Flag (O'Neil/Bulkowski)
"""

from models.base_model import BaseModel, ModelSignal

# Original models
from models.weinstein_core import WeinsteinCore
from models.rsi_mean_reversion import RSIMeanReversion
from models.momentum_52w_high import Momentum52WeekHigh
from models.consolidation_breakout import ConsolidationBreakout

# New models v3.0 (research-backed)
from models.vcp import VCP
from models.pocket_pivot import PocketPivot
from models.rs_breakout import RSBreakout
from models.high_tight_flag import HighTightFlag

# New models v3.1 (Phase 4 improvements)
from models.enhanced_mean_reversion import EnhancedMeanReversion
from models.dual_momentum import DualMomentum

__all__ = [
    'BaseModel',
    'ModelSignal',
    'WeinsteinCore',
    'RSIMeanReversion',
    'Momentum52WHighModel',
    'ConsolidationBreakout',
    'VCP',
    'PocketPivot',
    'RSBreakout',
    'HighTightFlag',
    'EnhancedMeanReversion',
    'DualMomentum',
]

# Model registry for dynamic loading
MODEL_REGISTRY = {
    'weinstein_core': WeinsteinCore,
    'rsi_mean_reversion': RSIMeanReversion,
    'enhanced_mean_reversion': EnhancedMeanReversion,
    'dual_momentum': DualMomentum,
    'momentum_52w_high': Momentum52WeekHigh,
    'consolidation_breakout': ConsolidationBreakout,
    'vcp': VCP,
    'pocket_pivot': PocketPivot,
    'rs_breakout': RSBreakout,
    'high_tight_flag': HighTightFlag,
}


def get_model(model_name: str, allocation_pct: float = None):
    """
    Factory function to get model instance by name
    
    Args:
        model_name: Name of the model (e.g., 'weinstein_core', 'vcp')
        allocation_pct: Optional allocation percentage override
        
    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    
    if allocation_pct is not None:
        return model_class(allocation_pct=allocation_pct)
    return model_class()
