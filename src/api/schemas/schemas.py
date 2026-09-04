"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Literal, Optional
from datetime import date, datetime
from enum import Enum

from src.defaults import DEFAULT_INDEX_SYMBOL


SUPPORTED_FORECAST_HORIZONS = [7, 15, 30, 60]

# The 1-day model is not a horizon the UI offers — it is the step model the
# recursive forecast rolls forward to produce a genuine prediction for every day.
# It has to be trainable through the API or per-step forecasting can never be
# switched on: the tab's train button is the only way bundles ever get built.
STEP_FORECAST_HORIZON = 1
TRAINABLE_HORIZONS = [STEP_FORECAST_HORIZON, *SUPPORTED_FORECAST_HORIZONS]


# ── Enums ──────────────────────────────────────────────────────
class DataSourceEnum(str, Enum):
    yfinance = "yfinance"
    alpha_vantage = "alpha_vantage"


class ModelTypeEnum(str, Enum):
    """
    Every model the prediction route can serve.

    The bare names are the legacy multi-horizon regression bundles. The
    ``unified_`` names are the next-timeframe models that return a price and a
    direction probability together. Not all of them can be trained: the
    foundation models are pre-trained and zero-shot, so only the rest appear in
    :data:`TRAINABLE_MODEL_TYPES`.
    """

    lstm = "lstm"
    xgboost = "xgboost"
    random_forest = "random_forest"
    unified_lstm = "unified_lstm"
    unified_xgboost = "unified_xgboost"
    unified_random_forest = "unified_random_forest"
    unified_kronos = "unified_kronos"
    unified_timesfm = "unified_timesfm"
    unified_chronos = "unified_chronos"
    baseline_rw = "baseline_rw"


#: Model types a training request can target. The foundation models are
#: pre-trained and do no gradient updates here, so asking to train one is a
#: request the API cannot honour and rejects rather than silently no-ops.
TRAINABLE_MODEL_TYPES = frozenset(
    {
        ModelTypeEnum.lstm,
        ModelTypeEnum.xgboost,
        ModelTypeEnum.random_forest,
        ModelTypeEnum.unified_lstm,
        ModelTypeEnum.unified_xgboost,
        ModelTypeEnum.unified_random_forest,
    }
)


class ValidationModeEnum(str, Enum):
    single_period = "single_period"
    walk_forward = "walk_forward"


class OptimizationMethod(str, Enum):
    max_sharpe = "max_sharpe"
    min_volatility = "min_volatility"
    max_return = "max_return"
    risk_parity = "risk_parity"


class TargetTypeEnum(str, Enum):
    price = "price"
    ret = "return"
    direction = "direction"


# ── Data Schemas ───────────────────────────────────────────────
class PriceBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class IndicatorValues(BaseModel):
    date: str
    values: Dict[str, Optional[float]]


class PriceResponse(BaseModel):
    symbol: str
    source: str
    bars: List[PriceBar]
    count: int


class IndicatorResponse(BaseModel):
    symbol: str
    indicators: List[str]
    data: List[Dict]
    count: int


class SP500Response(BaseModel):
    symbols: List[Dict[str, str]]
    count: int


class UploadResponse(BaseModel):
    filename: str
    rows: int
    columns: List[str]
    date_range: Dict[str, str]
    message: str


# ── Training Schemas ───────────────────────────────────────────
class TrainRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_INDEX_SYMBOL, description="Ticker symbol")
    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.xgboost)
    horizons: List[int] = Field(default_factory=lambda: [1])
    lookback_days: int = Field(default=756)
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    params: Optional[Dict] = None


class TrainResponse(BaseModel):
    job_id: str
    status: str
    model_type: str
    symbol: str
    message: str


class BootstrapTrainRequest(BaseModel):
    symbols: List[str] = Field(
        default_factory=lambda: [
            DEFAULT_INDEX_SYMBOL,
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "TSLA",
            "META",
            "NFLX",
        ]
    )
    model_types: List[ModelTypeEnum] = Field(default_factory=lambda: [ModelTypeEnum.xgboost, ModelTypeEnum.random_forest, ModelTypeEnum.lstm])
    horizons: List[int] = Field(default_factory=lambda: [1])
    lookback_days: int = Field(default=756)
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    use_sp500: bool = Field(default=False)
    skip_fresh_hours: Optional[int] = Field(default=24, ge=0)
    params: Optional[Dict[str, Dict[str, Any]]] = None


class BootstrapTrainResponse(BaseModel):
    job_id: str
    status: str
    symbols: List[str]
    model_types: List[str]
    message: str


class TrainStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = 0.0
    metrics: Optional[Dict] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    symbol: str
    trained_at: str
    horizons: List[int]
    metrics: Dict
    params: Dict


# ── Prediction Schemas ─────────────────────────────────────────
class PredictRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_INDEX_SYMBOL)
    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.xgboost)
    horizon: int = Field(default=1, ge=1, le=120)
    data_source: DataSourceEnum = Field(default=DataSourceEnum.yfinance)


class ForecastPoint(BaseModel):
    date: str
    predicted: float
    upper95: float
    lower95: float
    upper68: float
    lower68: float


class PredictResponse(BaseModel):
    symbol: str
    model_type: str
    horizon: int
    current_price: float
    current_price_source: Optional[str] = None
    predicted_price: Optional[float] = None
    target_price: Optional[float] = None
    expected_change_pct: Optional[float] = None
    upper95: Optional[float] = None
    lower95: Optional[float] = None
    upper68: Optional[float] = None
    lower68: Optional[float] = None
    direction: Optional[str] = None
    signal: Optional[str] = None
    confidence: Optional[float] = None
    probability_up: Optional[float] = None
    probability_down: Optional[float] = None
    expected_move: Optional[str] = None
    prediction_date: Optional[str] = None
    forecasts: List[ForecastPoint] = Field(default_factory=list)
    model_info: Dict
    status: str = Field(default="ok")
    model_available: bool = Field(default=True)
    reason: Optional[str] = None
    message: Optional[str] = None
    can_train: bool = Field(default=False)
    scenario_paths: Optional[List[List[float]]] = None  # Monte Carlo price paths for fan chart

    # Set when the symbol had nothing servable and training was started for it.
    # `status` is then "preparing", and this is the job to poll — so a client that
    # knows only this endpoint still learns that work is underway, instead of
    # being handed a dead end that says "train the model yourself".
    preparation: Optional[Dict] = None


class HistoricalSignal(BaseModel):
    date: str
    type: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    predicted_return: Optional[float] = None
    probability_up: Optional[float] = None
    direction: Optional[str] = None


# -- Best-performing-model chart overlay -----------------------
#
# What the chart draws on top of the candles. Two winners rather than one,
# because the evaluation suite scores price and direction separately and a
# model that tracks the level is often not the model that calls the sign
# (see src.models.model_selection).


class ChartForecastPoint(BaseModel):
    """
    One future bar, with the band around it and the step that produced it.

    ``direction`` is the sign of this point's move away from the previous one,
    which is what lets the chart colour the line segment by segment. It is the
    *price* model's implied step, and is not the direction model's call - that
    one is served once, for the next bar, in ``DirectionCall``.
    """

    date: str
    predicted: float
    upper95: float
    lower95: float
    upper68: float
    lower68: float
    direction: str  # "up" | "down" | "flat"
    change_pct: float


class SelectedModel(BaseModel):
    """The winner of one metric family, with the evidence it won on."""

    model_type: str
    label: str
    #: "walk_forward_benchmark" | "bundle_holdout" | "direction_walk_forward"
    evidence: str
    #: The horizon the metrics were measured at, which is not always the
    #: horizon requested: the unified and foundation models answer for the next
    #: bar whatever window the dashboard is showing.
    scored_horizon: int
    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    #: The metrics the ranking actually used. Shorter than the full scorecard
    #: whenever some candidate was never measured on one of them.
    metrics_used: List[str] = Field(default_factory=list)
    #: Which model won each individual metric, so a split decision is visible.
    metric_winners: Dict[str, str] = Field(default_factory=dict)
    mean_rank: Optional[float] = None
    n_candidates: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)


class DirectionCall(BaseModel):
    """
    The direction winner's call for the next bar.

    ``tradeable`` carries the same gate the direction API applies: a model that
    failed its walk-forward ship criteria still reports a probability, and the
    client is told not to present it as actionable.
    """

    #: Always "UP" or "DOWN", whichever model produced it. The panels below
    #: the chart word the same fact as Bullish/Bearish; the chart contract is
    #: fixed here so a client never has to match on two vocabularies.
    direction: Optional[str] = None
    probability_up: Optional[float] = None
    probability_down: Optional[float] = None
    confidence: Optional[float] = None
    #: BUY / SELL / HOLD, on the thresholds in src.models.direction_utils.
    signal: Optional[str] = None
    prediction_date: Optional[str] = None
    tradeable: bool = True
    gate_reason: Optional[str] = None
    #: How the number was produced: "unified_bundle", "direction_classifier",
    #: "regression_sign" or "unavailable".
    source: str = "unavailable"
    message: Optional[str] = None


class BestModelForecastResponse(BaseModel):
    """Everything the chart overlay needs, in one request."""

    symbol: str
    horizon: int
    as_of: Optional[str] = None
    current_price: Optional[float] = None
    current_price_source: Optional[str] = None

    price_model: Optional[SelectedModel] = None
    direction_model: Optional[SelectedModel] = None

    forecast: List[ChartForecastPoint] = Field(default_factory=list)
    direction: Optional[DirectionCall] = None

    #: Provenance for the trajectory: "recursive_per_step" means every point is
    #: a model output, "compounded_interpolation" means one number was produced
    #: for the whole horizon and the days between are a path drawn to it. The
    #: chart labels the line accordingly rather than implying a daily forecast.
    path_type: Optional[str] = None
    per_step_predictions: bool = False

    #: The full ranking tables behind both winners.
    selection: Dict[str, Any] = Field(default_factory=dict)

    status: str = Field(default="ok")  # "ok" | "partial" | "unavailable" | "preparing"
    reason: Optional[str] = None
    message: Optional[str] = None
    preparation: Optional[Dict] = None


class ForecastHistoryResponse(BaseModel):
    """
    The candles the Predictions tab draws, with no forecast attached.

    Split out from :class:`SimpleForecastResponse` because the two halves cost
    three orders of magnitude apart -- the bars are a cached download, the
    forecast is ~7s of transformer sampling -- and bundling them made the chart
    wait on the box. ``as_of`` is the last bar, so a client can tell whether a
    forecast it holds was built on this same frame.
    """

    symbol: str
    as_of: Optional[str] = None
    bars: List[PriceBar] = Field(default_factory=list)


class SimpleForecastPoint(BaseModel):
    """
    One future bar and the band around it.

    Named for the coverage the bounds actually have. ``ChartForecastPoint`` --
    which this deliberately does not reuse -- calls its outer pair
    ``upper95``/``lower95``, and the foundation ensemble's outer pair is the
    q0.05/q0.95 quantiles, which bracket 90% of the mass rather than 95%.
    Reusing that schema would have published a 90% interval under a name
    claiming five points more coverage than it has.
    """

    date: str
    predicted: float
    upper_90: float
    lower_90: float
    upper_68: float
    lower_68: float
    #: Sign of the move away from ``SimpleForecastResponse.anchor_price``, so
    #: the chart can colour the segment. The chart hangs this line off the last
    #: candle's close, so it has to be measured from that same close: against a
    #: live quote from a later session it once painted a falling segment green.
    #: Not the direction call -- that is one probability, served once.
    direction: str  # "up" | "down"
    change_pct: float


class SimpleForecastResponse(BaseModel):
    """
    The whole Predictions tab, in one payload.

    Deliberately small. The tab answers two questions -- what is the next price,
    and is it up or down -- so this carries the candles to draw, the one forecast
    point that continues them, and the four numbers in the forecast box. The
    feature assembly, the three model runs and the aggregation that produced them
    stay on the server; none of that reaches this schema, because none of it
    belongs on the screen.

    The candles are NOT here: they come from ``GET /predict/history/{symbol}``,
    because they are ready in milliseconds and this is not. ``as_of`` names the
    bar the models read, so a client holding both can tell they describe the
    same frame -- and the chart drops any forecast point that is not strictly
    after its last candle, so a mismatch degrades to "no forecast drawn" rather
    than to a point anchored on the wrong bar.
    """

    symbol: str
    #: "ok" | "unavailable". Unavailable is a normal answer with a `message`.
    status: str = "ok"
    message: Optional[str] = None

    #: Date of the last historical bar -- the session `anchor_price` closed at.
    as_of: Optional[str] = None
    #: The close the models measured against. `direction` is P(next bar > this)
    #: thresholded at 0.5, and `expected_change_pct` is the move away from it,
    #: so the two are readings of one bar rather than of two reference prices.
    anchor_price: Optional[float] = None
    #: The live quote. It can be a whole session later than `anchor_price` --
    #: the download window ends at today's date and yfinance treats that end as
    #: exclusive, so the frame always stops at the previous session -- and no
    #: model saw it. Context for the reader, never the reference for a number.
    current_price: Optional[float] = None
    current_price_source: Optional[str] = None

    forecast_price: Optional[float] = None
    #: The forecast move away from `anchor_price`.
    expected_change_pct: Optional[float] = None
    #: The same forecast against the live quote. This is the figure that reads
    #: +3.8% when a stock has fallen 4% since the bar the models read, and it
    #: is named separately so it can never be printed as "Expected Change".
    quote_change_pct: Optional[float] = None
    #: "UP" | "DOWN", from the aggregated probability rather than the price.
    direction: Optional[str] = None
    #: The bar being forecast, and how it is written in the box.
    forecast_date: Optional[str] = None
    horizon_label: str = "Next 1 Day"

    #: Display names of the members behind this number, for the one-line footer.
    models: List[str] = Field(default_factory=list)
    #: Set when the rows above would otherwise read as self-contradictory. One
    #: short line in the UI, not a panel -- and `split_reason` says which line,
    #: because the two causes need different sentences and writing the rarer
    #: one over the commoner case described a model disagreement that was not
    #: happening.
    split: bool = False
    #: Why `split` is set, or None:
    #:
    #:   "heads"  the aggregated probability and the aggregated price disagree
    #:            on `anchor_price`. Rare, and genuinely the models disagreeing.
    #:   "quote"  the two agree, but the live quote sits on the other side of
    #:            the forecast from `anchor_price`, so "Current Price" next to
    #:            "Forecast Price" implies the opposite of the call. Common
    #:            after a gap, and not a disagreement about anything.
    split_reason: Optional[str] = None

    forecast: List[SimpleForecastPoint] = Field(default_factory=list)


# ── Ensemble Prediction Schemas ────────────────────────────────

class EnsemblePredictRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_INDEX_SYMBOL)
    horizon: Literal[7, 15, 30, 60] = Field(default=30)


class EnsembleTrainRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_INDEX_SYMBOL)
    # Defaults to every trainable horizon, step model included. Training only the
    # four display horizons leaves the per-step forecast permanently unavailable,
    # because the 1-day bundles it rolls forward would never be built.
    horizons: List[int] = Field(default_factory=lambda: TRAINABLE_HORIZONS.copy())
    lookback_days: int = Field(default=1825, ge=365, le=3650)
    model_types: List[str] = Field(default_factory=lambda: ["xgboost", "random_forest", "lstm"])

    @field_validator("horizons")
    @classmethod
    def validate_supported_horizons(cls, value: List[int]) -> List[int]:
        requested = sorted({int(item) for item in value})
        unsupported = [item for item in requested if item not in TRAINABLE_HORIZONS]
        if unsupported:
            raise ValueError(
                f"Unsupported forecast horizon(s): {unsupported}. "
                f"Trainable horizons are {TRAINABLE_HORIZONS} "
                f"(forecast horizons {SUPPORTED_FORECAST_HORIZONS} plus the "
                f"{STEP_FORECAST_HORIZON}-day step model used for per-step forecasts)."
            )
        return requested

    @field_validator("model_types")
    @classmethod
    def validate_model_types(cls, value: List[str]) -> List[str]:
        supported = {"xgboost", "random_forest", "lstm"}
        requested = [str(item) for item in value]
        unsupported = [item for item in requested if item not in supported]
        if unsupported:
            raise ValueError(
                f"Unsupported model type(s): {unsupported}. "
                "Supported model types are xgboost, random_forest, and lstm."
            )
        return requested


class ModelPredictionDetail(BaseModel):
    prediction: float
    weight: float
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float


class EnsembleSummary(BaseModel):
    target: float
    #: The move from `EnsemblePredictResponse.anchor_price` to `target`, so this
    #: and `signal` are two readings of one bar. The Analysis tab prints them in
    #: a single string, so measuring them from different prices is visible at a
    #: glance: against a live quote a session away this tile read "+3.82% DOWN".
    change_pct: float
    #: The same move measured against the live quote, which no model saw. Kept
    #: apart from `change_pct` for the reason above.
    quote_change_pct: Optional[float] = None
    upper_90: float
    lower_90: float
    upper_95: Optional[float] = None
    lower_95: Optional[float] = None
    upper_68: Optional[float] = None
    lower_68: Optional[float] = None
    reliability: str
    consensus: str
    signal: Optional[str] = None


class EnsembleForecastPoint(BaseModel):
    date: str
    ensemble: float
    prediction: Optional[float] = None
    lstm: Optional[float] = None
    xgboost: Optional[float] = None
    random_forest: Optional[float] = None
    unified_kronos: Optional[float] = None
    unified_chronos: Optional[float] = None
    unified_timesfm: Optional[float] = None
    model_predictions: Optional[Dict[str, float]] = None
    upper_90: float
    lower_90: float
    upper_95: Optional[float] = None
    lower_95: Optional[float] = None
    upper_68: Optional[float] = None
    lower_68: Optional[float] = None


class EnsemblePredictResponse(BaseModel):
    symbol: str
    #: Date of the last bar the models read -- the session `anchor_price`
    #: closed at.
    as_of: Optional[str] = None
    #: The close the ensemble measured against. `EnsembleSummary.change_pct`
    #: and `EnsembleSummary.signal` both belong to it. Same two prices, and the
    #: same reason for serving both, as `SimpleForecastResponse`.
    anchor_price: Optional[float] = None
    #: The live quote. It can be a whole session later than `anchor_price` and
    #: no model saw it: context for the reader, not a reference for a number.
    current_price: float
    current_price_source: Optional[str] = None
    horizon: int
    ensemble: Optional[EnsembleSummary] = None
    weights: Dict[str, float] = Field(default_factory=dict)
    forecast_points: List[EnsembleForecastPoint] = Field(default_factory=list)
    status: str = Field(default="ok")
    model_available: bool = Field(default=True)
    message: Optional[str] = None
    scenario_paths: List[List[float]] = Field(default_factory=list)  # fan-chart sample paths

    # Provenance of the daily points. In "compounded_interpolation" mode each
    # model contributes one cumulative horizon-day return and the days between
    # are a compounded path to it — so `forecast_points` is one prediction plus
    # interpolation, not a per-day forecast. Consumers should not present the
    # intermediate points as independent daily predictions.
    path_type: str = Field(default="compounded_interpolation")
    per_step_predictions: bool = Field(default=False)
    model_output_count: int = Field(default=0)

    # Which ensemble members actually contributed. A forecast built from two of
    # three models is still a forecast, but the client has to be able to say so
    # rather than presenting it as the full ensemble.
    models_available: List[str] = Field(default_factory=list)
    models_unavailable: Dict[str, str] = Field(default_factory=dict)
    degraded: bool = Field(default=False)

    # Populated when nothing was servable and preparation was started. See the
    # note on PredictResponse.preparation.
    preparation: Optional[Dict] = None


# ── Pattern Detection Schemas ──────────────────────────────────
class KeyLevel(BaseModel):
    date: str
    price: float


class MultiTFPatternItem(BaseModel):
    pattern_name: str
    direction: str  # "bullish" | "bearish" | "neutral"
    status: str     # "forming" | "confirmed" | "broken"
    start_date: str
    end_date: str
    timeframe: str
    weight: int
    confidence: float

    # Path & levels
    key_levels: List[KeyLevel]
    trendlines: Optional[List[List[KeyLevel]]] = None
    # Actionable levels
    entry_price: Optional[float] = None
    neckline: Optional[float] = None
    breakout_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    strength_label: Optional[str] = None
    secondary_targets: List[float] = Field(default_factory=list)
    current_price: Optional[float] = None
    entry_distance_pct: Optional[float] = None
    setup_relevance_status: Optional[str] = None
    setup_relevance_reason: Optional[str] = None
    setup_relevance_ok: Optional[bool] = None


class SetupScoreComponents(BaseModel):
    ml_probability: float
    pattern_quality: float
    indicator_alignment: float
    trend_confirmation: Optional[float] = None
    volume_confirmation: float
    support_resistance_confirmation: Optional[float] = None
    risk_reward_score: Optional[float] = None
    conflict_penalty: Optional[float] = None
    low_risk_reward_penalty: Optional[float] = None
    composite_score: float


class BestTradeSetup(BaseModel):
    pattern_name: str
    timeframe: str
    direction: str
    pattern_status: str
    confidence_score: Optional[float] = None
    current_price: Optional[float] = None
    entry_distance_pct: Optional[float] = None
    entry_price: float
    stop_loss: float
    primary_target: float
    secondary_targets: List[float] = Field(default_factory=list)
    risk_reward_ratio: Optional[float] = None
    strength_label: str
    target_move_pct: Optional[float] = None
    action: str
    score_components: SetupScoreComponents


class ConfluenceSignal(BaseModel):
    pattern_name: str
    direction: str
    timeframes: List[str]
    total_weight: int


class ConfluenceResponse(BaseModel):
    symbol: str
    confluence_signals: List[ConfluenceSignal]


class SRLevel(BaseModel):
    price: float
    type: str  # "support" | "resistance"
    strength: str  # "strong" | "normal"
    confirmations: int
    sources: List[str]
    zone_low: float
    zone_high: float


class SupportResistanceResponse(BaseModel):
    symbol: str
    current_price: float
    levels: List[SRLevel]
    trendlines: List[Dict]
    dynamic_levels: List[Dict]


class BestSetupStatus(BaseModel):
    status: str
    setup_available: bool
    reason_code: str
    reason: str
    sufficient_data: bool
    has_detected_pattern: bool
    confidence_ok: bool
    levels_ok: bool
    price_relevance_ok: bool = True
    risk_reward_ok: bool
    no_conflicting_filters: bool
    candle_count: int
    min_candles: int
    min_confidence: float
    min_risk_reward: float
    candidate_pattern_name: Optional[str] = None
    candidate_confidence: Optional[float] = None
    candidate_risk_reward: Optional[float] = None
    candidate_strength_label: Optional[str] = None
    candidate_relevance_status: Optional[str] = None
    candidate_relevance_reason: Optional[str] = None
    candidate_entry_distance_pct: Optional[float] = None
    current_price: Optional[float] = None
    conflicting_pattern_names: List[str] = Field(default_factory=list)


class PatternResponse(BaseModel):
    symbol: str
    timeframe: str
    status: str
    best_setup_status: BestSetupStatus
    best_setup: Optional[BestTradeSetup] = None
    best_pattern: Optional[MultiTFPatternItem] = None
    patterns: List[MultiTFPatternItem]


# ── Backtest Schemas ───────────────────────────────────────────
class BacktestRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_INDEX_SYMBOL)
    start_date: str = Field(default="2022-01-01")
    end_date: str = Field(default="2024-12-31")
    initial_capital: float = Field(default=100000)
    strategy: Optional[Literal["ta_only", "ml_hybrid", "buy_hold"]] = Field(default=None)
    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.xgboost)
    primary_model: Optional[ModelTypeEnum] = Field(default=None)
    position_size: float = Field(default=0.1, ge=0.01, le=1.0)
    commission_rate: float = Field(default=0.0)
    slippage_rate: float = Field(default=0.001)
    include_market_benchmark: bool = Field(default=True)
    benchmark_symbol: str = Field(default=DEFAULT_INDEX_SYMBOL)
    validation_mode: ValidationModeEnum = Field(default=ValidationModeEnum.single_period)
    walk_forward_splits: int = Field(default=3, ge=2, le=10)
    walk_forward_gap: int = Field(default=5, ge=0, le=60)


class BacktestResponse(BaseModel):
    backtest_id: str
    summary: Dict[str, Any] = Field(default_factory=dict)
    price_series: List[Dict[str, Any]] = Field(default_factory=list)
    primary_run: Dict[str, Any] = Field(default_factory=dict)
    strategy_runs: List[Dict[str, Any]] = Field(default_factory=list)
    model_runs: List[Dict[str, Any]] = Field(default_factory=list)
    benchmarks: List[Dict[str, Any]] = Field(default_factory=list)
    validation: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    bh_curve: List[Dict[str, Any]] = Field(default_factory=list)
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    benchmark_notice: Optional[str] = None
    message: str


class BacktestEvidenceResponse(BaseModel):
    """
    The walk-forward record for one symbol, as the UI needs to read it.

    Distinct from :class:`BacktestResponse`, which is a trading simulation over
    one strategy. This is the out-of-sample scorecard the benchmark produces —
    purged folds, excess over base rate, the verdict against the random walk,
    and what the forecast is worth after costs.

    ``models`` is empty rather than a 404 when the benchmark has not been run
    for the symbol: "nobody has scored this yet" is an ordinary answer with a
    remedy, not a missing resource, and ``message`` carries the remedy.
    """

    symbol: str
    source: str
    models: List[Dict[str, Any]] = Field(default_factory=list)
    message: str


# ── Portfolio Schemas ──────────────────────────────────────────
class PortfolioOptimizeRequest(BaseModel):
    symbols: List[str] = Field(default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
    method: OptimizationMethod = Field(default=OptimizationMethod.max_sharpe)
    lookback_days: int = Field(default=252)
    risk_free_rate: float = Field(default=0.04)
    constraints: Optional[Dict] = None


class PortfolioOptimizeResponse(BaseModel):
    method: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    metrics: Dict


class EfficientFrontierResponse(BaseModel):
    points: List[Dict]
    optimal_portfolio: Dict
    current_portfolio: Optional[Dict] = None


# ── Export Schemas ─────────────────────────────────────────────
class ExportRequest(BaseModel):
    resource_type: str  # "prices", "predictions", "backtest", "portfolio"
    symbol: Optional[str] = None
    format: str = "csv"  # "csv" or "pdf"
    filters: Optional[Dict] = None
