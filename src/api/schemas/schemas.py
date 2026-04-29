"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Literal, Optional
from datetime import date, datetime
from enum import Enum

from src.defaults import DEFAULT_INDEX_SYMBOL


SUPPORTED_FORECAST_HORIZONS = [7, 15, 30, 60]


# ── Enums ──────────────────────────────────────────────────────
class DataSourceEnum(str, Enum):
    yfinance = "yfinance"
    alpha_vantage = "alpha_vantage"


class ModelTypeEnum(str, Enum):
    lstm = "lstm"
    xgboost = "xgboost"
    random_forest = "random_forest"


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


class HistoricalSignal(BaseModel):
    date: str
    type: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    predicted_return: Optional[float] = None
    probability_up: Optional[float] = None
    direction: Optional[str] = None


# ── Ensemble Prediction Schemas ────────────────────────────────

class EnsemblePredictRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_INDEX_SYMBOL)
    horizon: Literal[7, 15, 30, 60] = Field(default=30)


class EnsembleTrainRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_INDEX_SYMBOL)
    horizons: List[int] = Field(default_factory=lambda: SUPPORTED_FORECAST_HORIZONS.copy())
    lookback_days: int = Field(default=1825, ge=365, le=3650)
    model_types: List[str] = Field(default_factory=lambda: ["xgboost", "random_forest", "lstm"])

    @field_validator("horizons")
    @classmethod
    def validate_supported_horizons(cls, value: List[int]) -> List[int]:
        requested = sorted({int(item) for item in value})
        unsupported = [item for item in requested if item not in SUPPORTED_FORECAST_HORIZONS]
        if unsupported:
            raise ValueError(
                f"Unsupported forecast horizon(s): {unsupported}. "
                f"Supported horizons are {SUPPORTED_FORECAST_HORIZONS}."
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
    change_pct: float
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
    upper_90: float
    lower_90: float
    upper_95: Optional[float] = None
    lower_95: Optional[float] = None
    upper_68: Optional[float] = None
    lower_68: Optional[float] = None


class EnsemblePredictResponse(BaseModel):
    symbol: str
    current_price: float
    current_price_source: Optional[str] = None
    horizon: int
    ensemble: Optional[EnsembleSummary] = None
    weights: Dict[str, float] = Field(default_factory=dict)
    forecast_points: List[EnsembleForecastPoint] = Field(default_factory=list)
    status: str = Field(default="ok")
    model_available: bool = Field(default=True)
    message: Optional[str] = None


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
