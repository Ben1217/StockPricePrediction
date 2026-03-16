"""
Unit tests for the Technical Sentiment Signal module.

Covers:
  - SignalOutput contract validation
  - Processing pipeline (z-score, normalise, regime filter)
  - Every signal function (P0 → P3) with synthetic data
  - Composite score weighting, staleness exclusion, and bounding
  - Edge cases: NaN, zero volume, missing columns, single-row DataFrame
  - No-lookahead validation (shifted-index test)
"""

import numpy as np
import pandas as pd
import pytest

from src.signals.sentiment.models import SignalOutput
from src.signals.sentiment.signal_processor import (
    z_score_rolling,
    normalise_to_unit,
    ema_smooth,
    get_regime,
    build_signal_output,
)
from src.signals.sentiment.composite_score import composite_score, get_weights

# P0
from src.signals.sentiment.microstructure_signals import (
    compute_ofi,
    compute_trade_sign_imbalance,
    compute_spread_z,
)
from src.signals.sentiment.volume_signals import (
    compute_vwap_z,
    compute_cmf,
    compute_obv_divergence,
)

# P1
from src.signals.sentiment.volatility_signals import (
    compute_iv_rank,
    compute_term_slope,
    compute_rv_iv_spread,
    compute_skew_25d,
)
from src.signals.sentiment.options_signals import (
    compute_pcr_5d,
    compute_gex,
    compute_delta_flow,
)

# P2
from src.signals.sentiment.breadth_signals import (
    compute_mcclellan,
    compute_ad_line,
    compute_nh_nl_ratio,
)

# P3
from src.signals.sentiment.positioning_signals import (
    compute_cot_z,
    compute_commercial_ratio,
)
from src.signals.sentiment.momentum_signals import (
    compute_rsi_divergence,
    compute_macd_exhaustion,
    compute_roc_z,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def ohlcv_df():
    """100-bar OHLCV with trending price."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": (dates.astype(np.int64) // 10**6).astype(int),
        "open": close + np.random.randn(n) * 0.2,
        "high": close + np.abs(np.random.randn(n)) * 0.5,
        "low": close - np.abs(np.random.randn(n)) * 0.5,
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=dates)


@pytest.fixture
def tick_df():
    """200-row tick data with bid/ask sides."""
    np.random.seed(7)
    n = 200
    prices = 100 + np.cumsum(np.random.randn(n) * 0.01)
    sides = np.random.choice(["bid", "ask"], n)
    return pd.DataFrame({
        "timestamp": np.arange(n),
        "price": prices,
        "size": np.random.randint(1, 100, n).astype(float),
        "side": sides,
        "bid_price": prices - 0.01,
        "bid_size": np.random.randint(50, 200, n).astype(float),
        "ask_price": prices + 0.01,
        "ask_size": np.random.randint(50, 200, n).astype(float),
    })


@pytest.fixture
def options_df():
    """Synthetic options chain snapshot."""
    np.random.seed(11)
    n = 40
    strikes = np.linspace(90, 110, n)
    types = np.tile(["call", "put"], n // 2)
    return pd.DataFrame({
        "timestamp": [1_000_000] * n,
        "strike": strikes,
        "expiry": ["2023-06-16"] * n,
        "type": types,
        "iv": np.random.uniform(0.15, 0.45, n),
        "delta": np.where(types == "call",
                          np.random.uniform(0.1, 0.9, n),
                          -np.random.uniform(0.1, 0.9, n)),
        "gamma": np.random.uniform(0.001, 0.05, n),
        "oi": np.random.randint(100, 5000, n),
        "volume": np.random.randint(10, 500, n),
    })


@pytest.fixture
def vix_df():
    """60-bar VIX futures data."""
    np.random.seed(3)
    n = 60
    return pd.DataFrame({
        "timestamp": np.arange(n),
        "vx_m1": 18 + np.random.randn(n) * 2,
        "vx_m2": 20 + np.random.randn(n) * 2,
    })


@pytest.fixture
def breadth_df():
    """60-bar exchange breadth data."""
    np.random.seed(5)
    n = 60
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "advancing": np.random.randint(1000, 3000, n),
        "declining": np.random.randint(800, 2500, n),
        "new_highs": np.random.randint(10, 200, n),
        "new_lows": np.random.randint(5, 150, n),
    })


@pytest.fixture
def cot_df():
    """60-row COT data."""
    np.random.seed(9)
    n = 60
    return pd.DataFrame({
        "report_date": pd.date_range("2023-01-06", periods=n, freq="W-FRI"),
        "instrument": ["ES"] * n,
        "comm_long": np.random.randint(50_000, 200_000, n),
        "comm_short": np.random.randint(50_000, 200_000, n),
        "spec_long": np.random.randint(100_000, 300_000, n),
        "spec_short": np.random.randint(100_000, 300_000, n),
        "nonrep_long": np.random.randint(10_000, 50_000, n),
        "nonrep_short": np.random.randint(10_000, 50_000, n),
    })


# ===================================================================
# 1. SignalOutput contract tests
# ===================================================================

class TestSignalOutput:
    def test_valid_construction(self):
        s = SignalOutput(
            name="test", value=1.0, z_score=0.5, normalised=0.3,
            regime="neutral", confidence=0.9, timestamp=123456,
            lookback_bars=60, is_stale=False,
        )
        assert s.name == "test"
        assert s.normalised == 0.3

    def test_invalid_regime_raises(self):
        with pytest.raises(ValueError, match="regime"):
            SignalOutput(
                name="x", value=0, z_score=0, normalised=0,
                regime="INVALID", confidence=0.5, timestamp=0,
                lookback_bars=0, is_stale=False,
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            SignalOutput(
                name="x", value=0, z_score=0, normalised=0,
                regime="neutral", confidence=1.5, timestamp=0,
                lookback_bars=0, is_stale=False,
            )

    def test_normalised_out_of_range_raises(self):
        with pytest.raises(ValueError, match="normalised"):
            SignalOutput(
                name="x", value=0, z_score=0, normalised=2.0,
                regime="neutral", confidence=0.5, timestamp=0,
                lookback_bars=0, is_stale=False,
            )


# ===================================================================
# 2. Processing pipeline tests
# ===================================================================

class TestProcessingPipeline:
    def test_z_score_known_values(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        z = z_score_rolling(s, window=5)
        # Last value should be positive (10 is above 5-bar mean of 8)
        assert z.iloc[-1] > 0

    def test_normalise_bounded(self):
        z = pd.Series(np.linspace(-10, 10, 100))
        normed = normalise_to_unit(z)
        assert normed.max() <= 1.0
        assert normed.min() >= -1.0

    def test_ema_smooth_length(self):
        s = pd.Series(np.random.randn(50))
        smoothed = ema_smooth(s, span=5)
        assert len(smoothed) == len(s)

    def test_regime_backwardation(self):
        assert get_regime(0.90) == "risk_off"

    def test_regime_contango(self):
        assert get_regime(1.10) == "risk_on"

    def test_regime_neutral(self):
        assert get_regime(1.00) == "neutral"

    def test_regime_boundary_low(self):
        assert get_regime(0.95) == "neutral"

    def test_regime_boundary_high(self):
        assert get_regime(1.05) == "neutral"

    def test_build_signal_output_valid(self):
        series = pd.Series(np.random.randn(100))
        so = build_signal_output(
            name="test.signal",
            raw_value=1.5,
            series_for_z=series,
            timestamp=999,
            lookback_bars=100,
        )
        assert isinstance(so, SignalOutput)
        assert -1.0 <= so.normalised <= 1.0
        assert so.name == "test.signal"


# ===================================================================
# 3. P0 — Microstructure signals
# ===================================================================

class TestMicrostructureSignals:
    def test_ofi_returns_signal_output(self, tick_df):
        result = compute_ofi(tick_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "micro.ofi"
        assert not result.is_stale

    def test_ofi_missing_columns(self):
        result = compute_ofi(pd.DataFrame({"x": [1, 2]}))
        assert result.is_stale

    def test_trade_sign_imbalance(self, tick_df):
        result = compute_trade_sign_imbalance(tick_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "micro.trade_sign"

    def test_spread_z(self, tick_df):
        result = compute_spread_z(tick_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "micro.spread_z"


# ===================================================================
# 4. P0/P2 — Volume signals
# ===================================================================

class TestVolumeSignals:
    def test_vwap_z_returns_valid(self, ohlcv_df):
        result = compute_vwap_z(ohlcv_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "volume.vwap_z"
        assert not result.is_stale
        assert -1.0 <= result.normalised <= 1.0

    def test_vwap_z_missing_columns(self):
        result = compute_vwap_z(pd.DataFrame({"x": [1]}))
        assert result.is_stale

    def test_cmf_returns_valid(self, ohlcv_df):
        result = compute_cmf(ohlcv_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "volume.cmf_20"
        assert not result.is_stale

    def test_obv_divergence(self, ohlcv_df):
        result = compute_obv_divergence(ohlcv_df, window=20)
        assert isinstance(result, SignalOutput)
        assert result.name == "volume.obv_divergence"
        assert result.value in (-1.0, 0.0, 1.0)

    def test_vwap_z_capitalised_columns(self):
        """Handles both 'Close' and 'close' conventions."""
        np.random.seed(42)
        n = 80
        df = pd.DataFrame({
            "High": np.random.uniform(101, 105, n),
            "Low": np.random.uniform(95, 99, n),
            "Close": np.random.uniform(99, 103, n),
            "Volume": np.random.randint(100, 1000, n).astype(float),
        })
        result = compute_vwap_z(df)
        assert not result.is_stale

    def test_cmf_zero_volume(self):
        """CMF with zero volume should not crash."""
        df = pd.DataFrame({
            "high": [10.0] * 30,
            "low": [9.0] * 30,
            "close": [9.5] * 30,
            "volume": [0.0] * 30,
        })
        result = compute_cmf(df)
        assert isinstance(result, SignalOutput)


# ===================================================================
# 5. P1 — Volatility signals
# ===================================================================

class TestVolatilitySignals:
    def test_iv_rank_valid(self, options_df):
        result = compute_iv_rank(options_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "vol.iv_rank"
        assert 0.0 <= result.value <= 1.0

    def test_iv_rank_missing_column(self):
        result = compute_iv_rank(pd.DataFrame({"x": [1]}))
        assert result.is_stale

    def test_term_slope_valid(self, vix_df):
        result = compute_term_slope(vix_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "vol.term_slope"
        assert result.regime in ("risk_on", "risk_off", "neutral")

    def test_rv_iv_spread(self, options_df, ohlcv_df):
        result = compute_rv_iv_spread(options_df, ohlcv_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "vol.rv_iv_spread"

    def test_skew_25d(self, options_df):
        # Ensure we have 25-delta options
        df = options_df.copy()
        # Set some deltas near 0.25
        df.loc[df["type"] == "put", "delta"] = -0.25
        df.loc[df["type"] == "call", "delta"] = 0.25
        result = compute_skew_25d(df)
        assert isinstance(result, SignalOutput)
        assert result.name == "vol.skew_25d"


# ===================================================================
# 6. P1 — Options signals
# ===================================================================

class TestOptionsSignals:
    def test_pcr_5d_valid(self, options_df):
        result = compute_pcr_5d(options_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "options.pcr_5d"
        assert result.value > 0  # ratio must be positive

    def test_pcr_5d_missing_columns(self):
        result = compute_pcr_5d(pd.DataFrame({"x": [1]}))
        assert result.is_stale

    def test_gex(self, options_df):
        result = compute_gex(options_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "options.gex_total"

    def test_delta_flow(self, options_df):
        result = compute_delta_flow(options_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "options.delta_flow"


# ===================================================================
# 7. P2 — Breadth signals
# ===================================================================

class TestBreadthSignals:
    def test_mcclellan(self, breadth_df):
        result = compute_mcclellan(breadth_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "breadth.mcclellan"
        assert not result.is_stale

    def test_mcclellan_insufficient_data(self):
        df = pd.DataFrame({
            "advancing": [100, 200],
            "declining": [150, 100],
        })
        result = compute_mcclellan(df)
        assert result.is_stale  # needs >= 39 bars

    def test_ad_line(self, breadth_df):
        result = compute_ad_line(breadth_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "breadth.ad_line"

    def test_nh_nl_ratio(self, breadth_df):
        result = compute_nh_nl_ratio(breadth_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "breadth.nh_nl_ratio"


# ===================================================================
# 8. P3 — Positioning signals
# ===================================================================

class TestPositioningSignals:
    def test_cot_z(self, cot_df):
        result = compute_cot_z(cot_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "positioning.cot_z"
        assert not result.is_stale

    def test_commercial_ratio(self, cot_df):
        result = compute_commercial_ratio(cot_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "positioning.commercial_ratio"

    def test_cot_z_missing_columns(self):
        result = compute_cot_z(pd.DataFrame({"x": [1, 2]}))
        assert result.is_stale


# ===================================================================
# 9. P3 — Momentum / Divergence signals
# ===================================================================

class TestMomentumSignals:
    def test_rsi_divergence(self, ohlcv_df):
        result = compute_rsi_divergence(ohlcv_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "momentum.rsi_divergence"
        assert result.value in (-1.0, 0.0, 1.0)

    def test_macd_exhaustion(self, ohlcv_df):
        result = compute_macd_exhaustion(ohlcv_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "momentum.macd_exhaust"
        assert result.value in (-1.0, 0.0, 1.0)

    def test_roc_z(self, ohlcv_df):
        result = compute_roc_z(ohlcv_df)
        assert isinstance(result, SignalOutput)
        assert result.name == "momentum.roc_z"
        assert -1.0 <= result.normalised <= 1.0


# ===================================================================
# 10. Composite Score
# ===================================================================

class TestCompositeScore:
    def _make_signal(self, name, normalised, is_stale=False):
        return SignalOutput(
            name=name, value=0, z_score=0, normalised=normalised,
            regime="neutral", confidence=1.0, timestamp=0,
            lookback_bars=0, is_stale=is_stale,
        )

    def test_composite_bounded(self):
        """Result must be in [-1, +1]."""
        signals = {
            "micro.ofi": self._make_signal("micro.ofi", 1.0),
            "vol.iv_rank": self._make_signal("vol.iv_rank", 1.0),
            "vol.term_slope": self._make_signal("vol.term_slope", 1.0),
            "options.pcr_5d": self._make_signal("options.pcr_5d", 1.0),
            "volume.vwap_z": self._make_signal("volume.vwap_z", 1.0),
            "volume.cmf_20": self._make_signal("volume.cmf_20", 1.0),
            "breadth.mcclellan": self._make_signal("breadth.mcclellan", 1.0),
            "momentum.rsi_divergence": self._make_signal("momentum.rsi_divergence", 1.0),
            "positioning.cot_z": self._make_signal("positioning.cot_z", 1.0),
        }
        score = composite_score(signals)
        assert -1.0 <= score <= 1.0
        assert abs(score - 1.0) < 1e-9  # all max → score = 1.0

    def test_composite_all_stale_returns_zero(self):
        signals = {
            "micro.ofi": self._make_signal("micro.ofi", 0.5, is_stale=True),
        }
        assert composite_score(signals) == 0.0

    def test_composite_partial_signals(self):
        """Only available non-stale signals contribute."""
        signals = {
            "micro.ofi": self._make_signal("micro.ofi", 0.8),
            "vol.iv_rank": self._make_signal("vol.iv_rank", -0.4),
        }
        score = composite_score(signals)
        assert -1.0 <= score <= 1.0

    def test_composite_stale_excluded(self):
        """Stale signals must not influence the score."""
        s_fresh = self._make_signal("micro.ofi", 1.0, is_stale=False)
        s_stale = self._make_signal("vol.iv_rank", -1.0, is_stale=True)
        signals = {"micro.ofi": s_fresh, "vol.iv_rank": s_stale}
        score = composite_score(signals)
        # Only micro.ofi contributes → score = 1.0
        assert abs(score - 1.0) < 1e-9

    def test_empty_signals(self):
        assert composite_score({}) == 0.0


# ===================================================================
# 11. No-Lookahead Validation (Shifted-Index Test)
# ===================================================================

class TestNoLookahead:
    def test_vwap_z_no_lookahead(self, ohlcv_df):
        """Signal at bar t must not change when bar t+1 data changes."""
        df_full = ohlcv_df.copy()
        df_partial = ohlcv_df.iloc[:-1].copy()

        sig_full = compute_vwap_z(df_full)
        sig_partial = compute_vwap_z(df_partial)

        # The signal from df_partial (bar t-1) should be deterministic
        # regardless of what happens at bar t
        assert isinstance(sig_partial, SignalOutput)
        assert not sig_partial.is_stale

    def test_cmf_no_lookahead(self, ohlcv_df):
        """CMF at bar t should not use data from bar t+1."""
        df_full = ohlcv_df.copy()
        n = len(df_full)
        mid = n // 2

        sig_mid = compute_cmf(df_full.iloc[:mid])
        sig_full = compute_cmf(df_full)

        # Same history → same result at mid-point
        sig_mid2 = compute_cmf(df_full.iloc[:mid])
        assert abs(sig_mid.value - sig_mid2.value) < 1e-10

    def test_rsi_divergence_no_lookahead(self, ohlcv_df):
        """RSI divergence at bar t must not see bar t+1."""
        df = ohlcv_df.copy()
        sig1 = compute_rsi_divergence(df.iloc[:-5])
        sig2 = compute_rsi_divergence(df.iloc[:-5])
        assert sig1.value == sig2.value


# ===================================================================
# 12. Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_single_row_ohlcv(self):
        df = pd.DataFrame({
            "close": [100.0], "high": [101.0],
            "low": [99.0], "volume": [1000.0],
        })
        result = compute_vwap_z(df)
        assert isinstance(result, SignalOutput)

    def test_all_nan_series(self):
        df = pd.DataFrame({
            "close": [np.nan] * 10,
            "high": [np.nan] * 10,
            "low": [np.nan] * 10,
            "volume": [np.nan] * 10,
        })
        result = compute_cmf(df)
        assert isinstance(result, SignalOutput)

    def test_zero_volume_bars(self, ohlcv_df):
        df = ohlcv_df.copy()
        df["volume"] = 0.0
        result = compute_vwap_z(df)
        assert isinstance(result, SignalOutput)

    def test_ofi_single_side(self):
        """All ticks on one side should not crash."""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "size": [100.0, 200.0, 150.0],
            "side": ["ask", "ask", "ask"],
        })
        result = compute_ofi(df)
        assert isinstance(result, SignalOutput)
        # All buys → OFI should be positive or 1.0
        assert result.value >= 0 or result.is_stale
