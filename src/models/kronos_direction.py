"""
Kronos candlestick foundation model — the sequence slot.

Kronos is a decoder-only transformer pre-trained on ~12 billion K-line records
from 45 exchanges. A Binary-Spherical-Quantisation tokeniser turns each candle
into discrete tokens, and the model continues the token sequence
autoregressively. Sampling it repeatedly gives a distribution over tomorrow's
candle, which is exactly the two outputs this project needs from one forward
pass:

    P(up)      = share of sampled closes above today's close
    price band = percentiles of those same sampled closes

Unlike the tabular slots, Kronos eats raw OHLCV. It is not given the 46
engineered columns; the tokeniser's own vocabulary *is* the chart-pattern
representation, learned from 12 billion candles rather than hand-specified. That
makes the tabular-vs-sequence comparison in the report a real question rather
than a formality.

How the per-sample paths are obtained
-------------------------------------
``KronosPredictor.predict()`` averages over ``sample_count`` internally
(``auto_regressive_inference`` ends with ``np.mean(preds, axis=1)``), which
destroys exactly the spread this module needs. Rather than reimplement the
sampler — brittle, and it would silently drift from the vendored code — this
module uses the **public** ``generate()`` with the batch replicated ``N`` times
and ``sample_count=1``:

    generate(x=repeat(x, N), ..., sample_count=1) -> (N, pred_len, 6)

With ``sample_count=1`` the internal replicate-and-average is the identity, so
every one of the N batch rows is an independent draw (``sample_from_logits``
calls ``torch.multinomial`` per row). Verified against the library: 26 distinct
closes out of 32 draws, and the mean of the draws matches ``predict()``'s own
output to within Monte-Carlo error.

Cost, stated plainly
--------------------
This is a transformer forward pass per test row, not a tree lookup. Measured on
6 CPU threads: lookback 128 / 128 samples is ~12.8 s per row, so a 504-row
walk-forward is close to two hours. Batching several test dates into one call
is sublinear (4x the batch costs ~2.5x the time), which ``chunk_size`` exploits.
On CUDA it is minutes. ``sample_count`` also sets the Monte-Carlo error on
P(up): at N draws the standard error is sqrt(p(1-p)/N), about 4.4pp at N=128,
and that number is recorded in ``fit_info_`` so it is not mistaken for signal.

Public API:
    KronosDirection(...)  — DirectionEstimator subclass
    kronos_availability() -> (bool, reason)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .direction_models import DirectionEstimator, _ConstantProbabilityMixin

logger = get_logger(__name__)

# The vendored checkout, added to sys.path so `from model import ...` resolves.
# Kronos is distributed as a repository, not a package on PyPI.
VENDOR_PATH = Path(__file__).resolve().parents[2] / "vendor" / "kronos"

_KRONOS_IMPORT_ERROR: Optional[str] = None
_KRONOS_AVAILABLE = False

if VENDOR_PATH.is_dir():
    if str(VENDOR_PATH) not in sys.path:
        sys.path.insert(0, str(VENDOR_PATH))
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: F401
        from model.kronos import calc_time_stamps
        _KRONOS_AVAILABLE = True
    except Exception as exc:  # noqa: BLE001 - torch/einops may be missing too
        _KRONOS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _KRONOS_IMPORT_ERROR = f"vendor directory not found at {VENDOR_PATH}"

SETUP_HINT = (
    "Run `python scripts/setup_kronos.py` to vendor the model code, and "
    "`pip install -e \".[foundation]\"` for its dependencies."
)

DEFAULT_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_MODEL_ID = "NeoQuasar/Kronos-small"

# Bars of context per prediction. Kronos-small has max_context 512; 128 daily
# bars is half a trading year, enough for the tokeniser to place the current
# regime, and costs a quarter of what 252 does (attention is quadratic in
# sequence length).
DEFAULT_LOOKBACK = 128
DEFAULT_SAMPLE_COUNT = 128
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 0
MAX_CONTEXT = 512

# Ceiling on batch rows per generate() call, so chunking cannot exhaust memory
# on a large sample_count.
DEFAULT_MAX_BATCH = 512

# Percentiles reported as the price band.
BAND_PERCENTILES = (5.0, 50.0, 95.0)

# Column order Kronos expects, and the index of `close` within it.
_KRONOS_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
_CLOSE_INDEX = 3
_COLUMN_MAP = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def kronos_availability() -> Tuple[bool, Optional[str]]:
    """Whether the vendored model code imported, and why not if it did not."""
    return _KRONOS_AVAILABLE, _KRONOS_IMPORT_ERROR


class KronosDirection(DirectionEstimator, _ConstantProbabilityMixin):
    """
    Zero-shot Kronos direction estimator.

    ``fit()`` performs no gradient updates — the model is pre-trained — but it
    is not a no-op: it records the training base rate used by the degenerate
    fallback, and asserts the OHLCV side-channel has been supplied.

    ``predict_proba_up()`` runs one sampling pass per test date, over the bars
    ending at that date. That is the per-day model call the project needs:
    tomorrow's forecast is recomputed from today's chart, not interpolated from
    a single number.
    """

    name = "kronos"

    # Pre-trained and never fine-tuned here: fit() reads y only to detect a
    # degenerate single-class window and to record the fallback base rate. The
    # shuffled-label leakage check is therefore meaningless for this slot, and
    # the pipeline skips it rather than reporting a pass it did not earn.
    learns_from_labels = False

    def __init__(
        self,
        seed: int = 42,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        lookback: int = DEFAULT_LOOKBACK,
        tokenizer_id: str = DEFAULT_TOKENIZER_ID,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        max_batch: int = DEFAULT_MAX_BATCH,
        device: Optional[str] = None,
        predictor: Optional[Any] = None,
    ):
        super().__init__(seed=seed)
        self.sample_count = max(1, int(sample_count))
        self.lookback = max(8, int(lookback))
        self.tokenizer_id = str(tokenizer_id)
        self.model_id = str(model_id)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.max_batch = max(self.sample_count, int(max_batch))
        self.device = device

        self._predictor = predictor
        self._ohlcv: Optional[pd.DataFrame] = None
        self.degenerate_: bool = False
        self.price_bands_: Optional[np.ndarray] = None

        if self._predictor is None and not _KRONOS_AVAILABLE:
            raise ImportError(f"Kronos is unavailable ({_KRONOS_IMPORT_ERROR}). {SETUP_HINT}")

    # -- OHLCV side-channel --------------------------------------------------

    def set_ohlcv_context(self, ohlcv: pd.DataFrame) -> None:
        """
        Supply the full raw bars the sampler draws its context from.

        Must be the *whole* series, not the current fold's slice: a test row
        needs the ``lookback`` bars preceding it, which live in the training
        window. Slicing per fold would leave the first test rows with almost no
        context. Only bars at or before each prediction date are ever read, so
        passing the full frame does not leak — the slicing happens per row, by
        date, in :meth:`_predict_proba_up`.
        """
        required = {"Open", "High", "Low", "Close"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise KeyError(f"set_ohlcv_context requires {sorted(missing)}")
        frame = ohlcv.copy()
        frame.index = pd.to_datetime(frame.index)
        self._ohlcv = frame.sort_index()

    # -- model loading -------------------------------------------------------

    def _ensure_predictor(self):
        """Load tokeniser and weights once, on first prediction."""
        if self._predictor is not None:
            return self._predictor

        import torch

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info("Loading Kronos %s + %s on %s", self.model_id, self.tokenizer_id, device)
        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
        model = Kronos.from_pretrained(self.model_id)
        self._predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)
        self.device = device
        return self._predictor

    # -- DirectionEstimator hooks -------------------------------------------

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.degenerate_ = len(np.unique(y)) < 2
        if self.degenerate_:
            logger.warning(
                "Training window for %s holds one class only; emitting its base rate", self.name
            )
            self.fit_info_["degenerate_single_class"] = True
            return

        if self._ohlcv is None:
            raise RuntimeError(
                "KronosDirection.set_ohlcv_context() must be called before fit(); "
                "the pipeline passes DirectionDataset.ohlcv."
            )

        # The Monte-Carlo standard error of P(up) at its worst (p = 0.5). This
        # is measurement noise on the model's own output and belongs in the
        # report next to any probability it produces.
        monte_carlo_se = float(np.sqrt(0.25 / self.sample_count))
        self.fit_info_.update({
            "pretrained": True,
            "gradient_updates": 0,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "sample_count": self.sample_count,
            "lookback": self.lookback,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "monte_carlo_se_of_p_up": round(monte_carlo_se, 6),
        })

    def _context_frame(self, as_of: pd.Timestamp) -> Optional[pd.DataFrame]:
        """The last ``lookback`` bars at or before ``as_of``, in Kronos layout."""
        assert self._ohlcv is not None
        history = self._ohlcv.loc[self._ohlcv.index <= as_of]
        if len(history) < 8:
            return None
        window = history.tail(self.lookback)
        frame = window.rename(columns=_COLUMN_MAP)

        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        # Kronos was trained with a turnover channel alongside volume; the
        # library derives it the same way when it is absent.
        frame["amount"] = frame["volume"] * frame[["open", "high", "low", "close"]].mean(axis=1)
        return frame[_KRONOS_COLUMNS]

    def _next_session(self, as_of: pd.Timestamp) -> pd.Timestamp:
        """
        The date of the session being predicted.

        Taken from the bar index when the next bar exists, so the temporal
        embedding sees the real weekday. This is a *calendar* lookup, not a
        price one — the exchange calendar is known in advance — so it carries no
        information about the outcome.
        """
        assert self._ohlcv is not None
        later = self._ohlcv.index[self._ohlcv.index > as_of]
        if len(later):
            return pd.Timestamp(later[0])
        return pd.Timestamp(as_of) + pd.tseries.offsets.BDay(1)

    def _predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        if self.degenerate_ or self._ohlcv is None:
            self.price_bands_ = np.full((len(X), 3), np.nan)
            return self._constant(len(X), self.train_base_rate_)

        predictor = self._ensure_predictor()
        n_rows = len(X)
        probabilities = np.full(n_rows, self.train_base_rate_, dtype=np.float64)
        bands = np.full((n_rows, 3), np.nan, dtype=np.float64)

        # Rows are grouped into chunks whose contexts share a length, so several
        # test dates ride in one forward pass.
        chunk_size = max(1, self.max_batch // self.sample_count)
        prepared: List[Optional[Dict[str, Any]]] = [
            self._prepare_row(pd.Timestamp(date)) for date in X.index
        ]

        failures = 0
        for start in range(0, n_rows, chunk_size):
            group = [(i, prepared[i]) for i in range(start, min(start + chunk_size, n_rows))
                     if prepared[i] is not None]
            if not group:
                continue
            # All contexts in a batch must share a sequence length; the warm-up
            # rows at the very start of a series are shorter, so group by it.
            for length in sorted({item[1]["x"].shape[0] for item in group}):
                same = [item for item in group if item[1]["x"].shape[0] == length]
                try:
                    closes = self._sample_chunk(predictor, [item[1] for item in same])
                except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the run
                    failures += len(same)
                    logger.warning("Kronos sampling failed for a chunk of %d rows: %s", len(same), exc)
                    continue
                for (row_index, item), sampled in zip(same, closes):
                    last_close = item["last_close"]
                    probabilities[row_index] = float(np.mean(sampled > last_close))
                    bands[row_index] = np.percentile(sampled, BAND_PERCENTILES)

        if failures:
            logger.warning(
                "Kronos fell back to the training base rate on %d/%d rows", failures, n_rows
            )
        self.price_bands_ = bands
        self.fit_info_["rows_predicted"] = int(n_rows - failures)
        return probabilities

    def _prepare_row(self, as_of: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """
        Normalise one context window exactly as ``KronosPredictor.predict()`` does.

        The mean and standard deviation are per-window, so the model always sees
        a standardised series and the de-normalisation back to price uses the
        same statistics. Both are computed from context bars only.
        """
        frame = self._context_frame(as_of)
        if frame is None or frame.isnull().values.any():
            return None

        values = frame.to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        normalised = np.clip((values - mean) / (std + 1e-5), -5.0, 5.0)

        x_timestamp = pd.Series(pd.to_datetime(frame.index))
        y_timestamp = pd.Series([self._next_session(as_of)])
        return {
            "x": normalised.astype(np.float32),
            "x_stamp": calc_time_stamps(x_timestamp).values.astype(np.float32),
            "y_stamp": calc_time_stamps(y_timestamp).values.astype(np.float32),
            "mean": mean,
            "std": std,
            "last_close": float(frame["close"].iloc[-1]),
        }

    def _sample_chunk(self, predictor, items: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Sample ``sample_count`` next-day closes for each prepared context.

        Each context is repeated ``sample_count`` times in the batch and
        ``sample_count=1`` is passed to ``generate()``, which makes the library's
        internal replicate-then-average the identity and leaves every batch row
        an independent draw.
        """
        import torch

        # Seeded per call so a re-run of the same fold reproduces its samples.
        torch.manual_seed(self.seed)

        repeats = self.sample_count
        x = np.concatenate([np.repeat(item["x"][None], repeats, axis=0) for item in items], axis=0)
        x_stamp = np.concatenate(
            [np.repeat(item["x_stamp"][None], repeats, axis=0) for item in items], axis=0
        )
        y_stamp = np.concatenate(
            [np.repeat(item["y_stamp"][None], repeats, axis=0) for item in items], axis=0
        )

        predictions = predictor.generate(
            x, x_stamp, y_stamp,
            pred_len=1,
            T=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            sample_count=1,
            verbose=False,
        )
        predictions = np.asarray(predictions)
        expected = len(items) * repeats
        if predictions.shape[0] != expected:
            raise RuntimeError(
                f"Kronos returned {predictions.shape[0]} rows, expected {expected}; "
                f"the vendored generate() contract has changed"
            )

        results: List[np.ndarray] = []
        for position, item in enumerate(items):
            block = predictions[position * repeats:(position + 1) * repeats, 0, _CLOSE_INDEX]
            # Undo the per-window standardisation to get prices back.
            closes = block * (item["std"][_CLOSE_INDEX] + 1e-5) + item["mean"][_CLOSE_INDEX]
            results.append(np.asarray(closes, dtype=np.float64))
        return results
