"""
Signal Output Contract — Standard dataclass for all sentiment signals.

Every signal function in this module MUST return a SignalOutput instance.
Downstream consumers (backtester, live execution, risk module) depend on
this interface remaining stable.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalOutput:
    """
    Standardised output contract for all sentiment signals.

    Parameters
    ----------
    name : str
        Unique signal identifier, e.g. 'vol.iv_rank'.
    value : float
        Raw computed value in the signal's native units.
    z_score : float
        Rolling z-score over the configured lookback window.
    normalised : float
        Value mapped to [-1, +1] via tanh soft-clip.
    regime : str
        Market regime: 'risk_on' | 'risk_off' | 'neutral'.
    confidence : float
        Data quality / completeness score in [0, 1].
    timestamp : int
        Unix timestamp in milliseconds.
    lookback_bars : int
        Number of bars used in the calculation.
    is_stale : bool
        True if the last valid data point exceeds the staleness threshold.
    meta : dict, optional
        Signal-specific extras (e.g. per-strike GEX breakdown).
    """

    name: str
    value: float
    z_score: float
    normalised: float
    regime: str
    confidence: float
    timestamp: int
    lookback_bars: int
    is_stale: bool
    meta: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        """Validate invariants on construction."""
        if self.regime not in ('risk_on', 'risk_off', 'neutral'):
            raise ValueError(
                f"regime must be 'risk_on', 'risk_off', or 'neutral', got '{self.regime}'"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not (-1.0 <= self.normalised <= 1.0):
            raise ValueError(f"normalised must be in [-1, +1], got {self.normalised}")
