/**
 * The chart panel: price, the winning model's forecast, and who won.
 *
 * This is the panel the Predictions tab puts at the top, and it owns three
 * things that have to agree with each other or the picture lies:
 *
 *   1. the forecast fetch, keyed on the same horizon the 7D/15D/30D/60D
 *      selector is on, so the line redrawn on a horizon change is the line for
 *      that horizon rather than the previous one resliced;
 *   2. the chart itself, which is `ForecastOverlayChart`;
 *   3. the attribution above it, which names the model behind every mark on
 *      the canvas.
 *
 * Two models, named
 * -----------------
 * The backend ranks price and direction separately and returns two winners.
 * Putting both names in the header is not decoration: the dashed line and the
 * arrow on the same chart are usually two different models' output, and a
 * reader who assumes one model produced both will misread every disagreement
 * between them as a bug. When they disagree, the panel says so in words.
 *
 * Which chart
 * -----------
 * The Advanced Chart embed cannot carry an overlay (the reasons are in
 * `ForecastOverlayChart`'s header), so the forecast view is our own canvas and
 * TradingView's is one click away. The toggle defaults to the forecast, because
 * this is the Predictions tab and the forecast is what it is for.
 */

import { useState } from "react";

import { C } from "../utils/data";
import { useBestModelForecast } from "../hooks/useMarketData";
import { tradingViewUrl } from "../utils/tradingview";
import ForecastOverlayChart from "./ForecastOverlayChart";
import TradingViewChart from "./TradingViewChart";

const PANEL = "#0F1623";
const SURFACE = "#161B22";
const ACCENT = "#6366F1";
const AMBER = "#F5C842";

const CHART_MODES = [
    { value: "forecast", label: "Forecast overlay" },
    { value: "tradingview", label: "TradingView" },
];

/** How each metric is written. The unit is part of the number's meaning. */
const METRIC_FORMAT = {
    mae: (value) => `$${value.toFixed(2)}`,
    rmse: (value) => `$${value.toFixed(2)}`,
    mape: (value) => `${value.toFixed(2)}%`,
    r2: (value) => value.toFixed(3),
    accuracy: (value) => `${(value * 100).toFixed(1)}%`,
    f1: (value) => value.toFixed(3),
    roc_auc: (value) => value.toFixed(3),
};

const METRIC_LABEL = {
    mae: "MAE",
    rmse: "RMSE",
    mape: "MAPE",
    r2: "R²",
    accuracy: "Accuracy",
    f1: "F1",
    roc_auc: "AUC",
};

const EVIDENCE_LABEL = {
    walk_forward_benchmark: "walk-forward benchmark",
    bundle_holdout: "chronological holdout",
    direction_walk_forward: "walk-forward evaluation",
};

function formatMetric(key, value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return null;
    return (METRIC_FORMAT[key] || ((v) => v.toFixed(3)))(number);
}

/**
 * The metrics the ranking actually ran on, in the order the brief lists them.
 *
 * `metrics_used` is often shorter than the full scorecard: the per-horizon
 * regression bundles record no R-squared, and ranking on a metric only some
 * candidates carry would decide the winner by who happened to be measured. So
 * the ones that counted are the ones shown.
 */
function scoreLine(model) {
    if (!model) return [];
    return (model.metrics_used || [])
        .map((key) => {
            const text = formatMetric(key, model.metrics?.[key]);
            return text ? { key, label: METRIC_LABEL[key] || key, text } : null;
        })
        .filter(Boolean);
}

function Chip({ children, color = C.textMid, background = "rgba(148,163,184,.10)", title }) {
    return (
        <span
            title={title}
            style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                padding: "3px 8px", borderRadius: 999, background,
                color, fontSize: 11, fontWeight: 800, whiteSpace: "nowrap",
            }}
        >
            {children}
        </span>
    );
}

/**
 * One winner's attribution row: what it is, what it won on, and who it beat.
 *
 * `wonCount` of `metrics_used` is the honest headline. A model that took the
 * mean rank but only one metric of four is a split decision, and saying "won 1
 * of 4" is what stops that reading as a clean sweep.
 */
function WinnerRow({ title, model, accent, extra }) {
    if (!model) {
        return (
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <span style={{ color: C.textDim, fontSize: 11, fontWeight: 800, minWidth: 62 }}>{title}</span>
                <span style={{ color: C.textDim, fontSize: 11 }}>no model qualified</span>
            </div>
        );
    }

    const scores = scoreLine(model);
    const wonCount = Object.entries(model.metric_winners || {})
        .filter(([key, winner]) => winner === model.model_type && (model.metrics_used || []).includes(key))
        .length;
    const total = (model.metrics_used || []).length;

    return (
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ color: C.textDim, fontSize: 11, fontWeight: 800, minWidth: 62 }}>{title}</span>
            <Chip color={accent} background={`${accent}1F`}>{model.label}</Chip>
            {scores.map((score) => (
                <span key={score.key} style={{ color: C.textMid, fontSize: 11, fontVariantNumeric: "tabular-nums" }}>
                    {score.label} <strong style={{ color: C.text }}>{score.text}</strong>
                </span>
            ))}
            {total > 0 && (
                <span
                    style={{ color: C.textDim, fontSize: 11 }}
                    title={`Ranked against ${model.n_candidates} model${model.n_candidates === 1 ? "" : "s"} on the ${EVIDENCE_LABEL[model.evidence] || model.evidence}.`}
                >
                    won {wonCount} of {total} vs {Math.max(model.n_candidates - 1, 0)} other
                    {model.n_candidates - 1 === 1 ? "" : "s"}
                </span>
            )}
            {extra}
        </div>
    );
}

function Note({ tone = "dim", children }) {
    const palette = {
        dim: { color: C.textDim, border: "transparent", background: "transparent" },
        amber: { color: C.text, border: `${AMBER}55`, background: "rgba(245,200,66,.08)" },
        red: { color: C.text, border: "rgba(244,63,94,.35)", background: "rgba(244,63,94,.08)" },
    }[tone];
    return (
        <div style={{
            padding: tone === "dim" ? 0 : "8px 12px",
            border: tone === "dim" ? "none" : `1px solid ${palette.border}`,
            background: palette.background,
            borderRadius: 8,
            color: palette.color,
            fontSize: 11,
            lineHeight: 1.6,
        }}>
            {children}
        </div>
    );
}

export default function BestModelChartPanel({ symbol, horizon, bars, apiConnected, readyVersion = 0 }) {
    const [mode, setMode] = useState("forecast");
    // The Advanced Chart is a third-party iframe that takes seconds to appear,
    // so it is mounted the first time it is asked for and then kept mounted and
    // hidden. Unmounting it on every toggle would re-download the widget each
    // time, and tearing it down mid-load is the race that used to throw inside
    // the embed script (see TradingViewChart). Our own canvas has neither
    // problem, so it is rendered conditionally.
    const [tradingViewMounted, setTradingViewMounted] = useState(false);
    const selectMode = (value) => {
        setMode(value);
        if (value === "tradingview") setTradingViewMounted(true);
    };

    // The horizon is part of the query key, so switching 7D/15D/30D/60D asks a
    // different question rather than refetching the same one - which is what
    // stops a slow 60D reply landing on top of the 7D view the user moved to.
    const query = useBestModelForecast(symbol, {
        horizon,
        readyVersion,
        enabled: apiConnected,
    });
    const data = query.data ?? null;
    const loading = query.isPending && query.isFetching;
    const error = query.error
        ? query.error.message || "The forecast overlay could not be loaded."
        : null;

    const priceModel = data?.price_model || null;
    const directionModel = data?.direction_model || null;
    const direction = data?.direction || null;
    const forecast = data?.forecast || [];
    const unavailable = data && data.status !== "ok" && data.status !== "partial";

    // The two winners can point opposite ways on the same bar: one was ranked on
    // level error and the other on the sign, so nothing forces them to agree.
    // That is information, and it is only information if it is stated.
    const trajectoryRises = forecast.length > 0
        && Number(forecast[forecast.length - 1].predicted) >= Number(data?.current_price ?? 0);
    const headsDisagree = Boolean(direction?.direction) && forecast.length > 0
        && (direction.direction === "UP") !== trajectoryRises;

    return (
        <div style={{ background: PANEL, border: `1px solid ${C.border}`, borderRadius: 8, padding: 14, display: "grid", gap: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                <div style={{ color: C.text, fontSize: 13, fontWeight: 900 }}>
                    {symbol} — {mode === "forecast" ? `${horizon}D forecast on price` : "live chart"}
                </div>
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                    <div style={{ display: "inline-flex", background: SURFACE, border: `1px solid ${C.border}`, borderRadius: 8, padding: 3, gap: 3 }}>
                        {CHART_MODES.map((option) => (
                            <button
                                key={option.value}
                                type="button"
                                onClick={() => selectMode(option.value)}
                                style={{
                                    background: mode === option.value ? ACCENT : "transparent",
                                    color: mode === option.value ? "#F8FAFC" : C.textMid,
                                    border: "none", borderRadius: 6, padding: "6px 11px",
                                    fontSize: 11, fontWeight: 800, cursor: "pointer",
                                }}
                            >
                                {option.label}
                            </button>
                        ))}
                    </div>
                    <a
                        href={tradingViewUrl(symbol)}
                        target="_blank"
                        rel="noreferrer noopener"
                        style={{ color: AMBER, fontSize: 11, fontWeight: 700 }}
                    >Open on TradingView ↗</a>
                </div>
            </div>

            {mode === "forecast" && (
                <div style={{ display: "grid", gap: 6 }}>
                    <WinnerRow
                        title="Price"
                        model={priceModel}
                        accent={ACCENT}
                        extra={priceModel && priceModel.scored_horizon !== horizon ? (
                            <Chip color={AMBER} background="rgba(245,200,66,.12)" title={`Scored and served at a ${priceModel.scored_horizon}-bar horizon.`}>
                                next bar only
                            </Chip>
                        ) : null}
                    />
                    <WinnerRow
                        title="Direction"
                        model={directionModel}
                        accent={direction?.direction === "UP" ? C.green : direction?.direction === "DOWN" ? C.red : C.textMid}
                        extra={direction?.direction ? (
                            <Chip
                                color={direction.tradeable === false ? C.textDim : direction.direction === "UP" ? C.green : C.red}
                                background={direction.tradeable === false ? "rgba(148,163,184,.10)" : direction.direction === "UP" ? "rgba(16,185,129,.12)" : "rgba(244,63,94,.12)"}
                            >
                                {direction.direction}
                                {Number.isFinite(Number(direction.probability_up))
                                    ? ` ${Math.round((direction.direction === "UP" ? direction.probability_up : 1 - direction.probability_up) * 100)}%`
                                    : ""}
                            </Chip>
                        ) : null}
                    />
                </div>
            )}

            {mode === "forecast" && (
                loading && !data ? (
                    <div style={{ height: 520, display: "grid", placeItems: "center", background: SURFACE, border: `1px solid ${C.border}`, borderRadius: 8, color: C.textDim, fontSize: 12 }}>
                        Ranking models and loading the forecast…
                    </div>
                ) : (
                    <ForecastOverlayChart
                        bars={bars}
                        forecast={forecast}
                        direction={direction}
                        horizon={horizon}
                        height={520}
                    />
                )
            )}

            {/* Kept in the tree once mounted, hidden rather than unmounted.
                `display` is set explicitly instead of using the `hidden`
                attribute so no stylesheet can override it. */}
            {tradingViewMounted && (
                <div style={{ display: mode === "tradingview" ? "block" : "none" }}>
                    <TradingViewChart symbol={symbol} interval="1d" height={520} />
                </div>
            )}

            {mode === "forecast" && (
                <div style={{ display: "grid", gap: 8 }}>
                    <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center", color: C.textDim, fontSize: 11 }}>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                            <span style={{ width: 18, height: 0, borderTop: `2px dashed ${ACCENT}` }} />
                            forecast path
                        </span>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                            <span style={{ width: 14, height: 10, background: "rgba(99,102,241,.22)", borderRadius: 2 }} />
                            68% / 95% interval
                        </span>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                            <span style={{ color: C.green, fontWeight: 900 }}>▲</span>
                            <span style={{ color: C.red, fontWeight: 900 }}>▼</span>
                            direction call, on the next bar
                        </span>
                    </div>

                    {error && <Note tone="red">{error}</Note>}

                    {!error && unavailable && (
                        <Note tone="amber">
                            <strong style={{ color: AMBER, marginRight: 8 }}>No forecast to draw</strong>
                            {data?.message
                                || `No model has cleared its out-of-sample gate for ${symbol} at a ${horizon}-day horizon.`}
                        </Note>
                    )}

                    {!error && data?.status === "partial" && (
                        <Note tone="amber">
                            <strong style={{ color: AMBER, marginRight: 8 }}>Partial overlay</strong>
                            {data.reason === "direction_model_unavailable"
                                ? "The trajectory is drawn; no direction model could produce a call, so there is no arrow."
                                : "A direction call is shown; no price model could produce a trajectory, so there is no line."}
                            {data.message ? ` ${data.message}` : ""}
                        </Note>
                    )}

                    {direction?.gate_reason && (
                        <Note tone="amber">
                            <strong style={{ color: AMBER, marginRight: 8 }}>Direction not tradeable</strong>
                            {direction.gate_reason} The arrow is drawn muted and should be read as the
                            model's output, not as a signal.
                        </Note>
                    )}

                    {headsDisagree && (
                        <Note tone="dim">
                            The two winners disagree on this bar: the price model has the level
                            {trajectoryRises ? " rising" : " falling"} over {horizon} days while the direction model
                            calls the next bar {direction.direction}. They were ranked on different
                            things — level error against sign accuracy — so nothing forces them to agree.
                        </Note>
                    )}

                    {forecast.length > 0 && data?.per_step_predictions === false && (
                        <Note tone="dim">
                            The model produced one number for the whole {horizon}-day window; the days
                            between today and the endpoint are a compounded path to it, not per-day
                            predictions. The band is the Monte Carlo interval around that path.
                        </Note>
                    )}

                    {direction?.message && <Note tone="dim">{direction.message}</Note>}
                </div>
            )}
        </div>
    );
}
