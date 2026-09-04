/**
 * Predictions — a candlestick chart, the forecast that continues it, and a box.
 *
 * This tab answers two questions and no others: what is the predicted next
 * price, and is the expected direction up or down. Everything that produces
 * those two answers — the OHLCV download, the price-action / momentum /
 * volatility / support-resistance / volume features, Kronos, Chronos-2,
 * TimesFM 2.5 and the aggregation across them — runs on the server and arrives
 * as one payload from GET /api/predict/forecast/{symbol}.
 *
 * That single request is the reason this file is short. The tab used to fan out
 * to an ensemble endpoint plus one call per model, then reconcile the replies
 * on screen: partial-ensemble notices, per-step provenance, weight bars,
 * classifier track records, evidence stacks, historical analogs. None of that
 * was decoration — it was the UI doing work the API had left undone. With the
 * pipeline behind one endpoint there is nothing to reconcile, so there is
 * nothing to explain.
 *
 * The evidence panels are not deleted from the codebase, only from here.
 * `DirectionAnalysisPanel` and `DirectionPanel` still exist and still work; they
 * belong on an analysis or admin surface, where a reader has asked for the
 * reasoning rather than for a price.
 */

import { useMemo, useState } from "react";

import { C } from "../utils/data";
import { useForecastHistory, useSimpleForecast } from "../hooks/useMarketData";
import ForecastOverlayChart from "../components/ForecastOverlayChart";

const PANEL = "#0F1623";
const SURFACE = "#161B22";

/**
 * Where "Current Price" came from, in words.
 *
 * Worth the line: an extended-hours quote is the usual reason the box's current
 * price is nowhere near the close the models read, and a reader who cannot see
 * which of the two they are looking at has no way to tell a moving number from
 * a settled one. `latest_close` means no quote was available at all and the two
 * prices are the same, so it says that rather than naming a session.
 */
const QUOTE_SOURCE = {
    regular_market: "market hours",
    pre_market: "pre-market",
    post_market: "after hours",
    latest_close: "same as prev close",
};

/**
 * How much history the chart draws. Not a forecast horizon: all three models
 * are built for one step (TimesFM compiles at max_horizon=1, Kronos samples a
 * single chunk), so the forecast is always the next bar and the box says so.
 * Offering a 30D button here would promise a number the models never produce.
 */
const RANGES = [
    { label: "3M", bars: 63 },
    { label: "6M", bars: 126 },
    { label: "1Y", bars: 252 },
];

/**
 * One request covers every range, and the buttons slice its result.
 *
 * Asking the server for a narrower window would be a different request, and the
 * server answers a forecast request by running Kronos — so switching 1Y to 3M
 * would pay for a transformer forward pass to draw fewer of the candles it
 * already had. The models read the whole download either way; only the drawing
 * changes, so only the drawing is redone.
 */
const HISTORY_BARS = RANGES[RANGES.length - 1].bars;

const CHART_HEIGHT = 460;

function formatPrice(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return `$${Number(value).toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatPct(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const number = Number(value);
    return `${number >= 0 ? "+" : ""}${number.toFixed(2)}%`;
}

const controlStyle = {
    background: SURFACE,
    color: C.text,
    border: `1px solid ${C.border}`,
    borderRadius: 8,
    padding: "8px 12px",
    fontSize: 13,
    fontWeight: 700,
    outline: "none",
    cursor: "pointer",
};

/** One label/value line in the forecast box. */
function Row({ label, children, emphasis }) {
    return (
        <div
            style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                gap: 20,
                padding: "13px 0",
                borderTop: `1px solid ${C.border}`,
            }}
        >
            <span style={{ color: C.textMid, fontSize: 13, fontWeight: 600 }}>{label}</span>
            <span
                style={{
                    color: C.text,
                    fontSize: emphasis ? 26 : 17,
                    fontWeight: emphasis ? 900 : 700,
                    fontVariantNumeric: "tabular-nums",
                    lineHeight: 1.15,
                }}
            >
                {children}
            </span>
        </div>
    );
}

/**
 * The forecast box: the point of the page.
 *
 * Every number here is measured against `anchor_price` — the close the models
 * actually read — and the box names it, because that close is routinely not
 * the price the reader is looking at. The server's download window ends at
 * today's date and yfinance treats that end as exclusive, so the frame always
 * stops at the previous session while "Current Price" is a live quote from this
 * one.
 *
 * On PLTR the gap was 4%: the models forecast 0.26% BELOW the bar they had
 * read, the box divided that forecast by the quote instead, and printed +3.8%
 * beside a DOWN arrow. So the anchor gets its own row, Expected Change is the
 * move away from it, and the quote sits under it as context rather than as a
 * reference for anything.
 *
 * `split` still marks the rows reading as contradictory, and `split_reason`
 * says why. The two causes need different sentences: writing the rarer one
 * (the models disagreeing with themselves) over the commoner one (the quote
 * having moved) is how this note came to describe something that was not
 * happening.
 */
function ForecastBox({ forecast }) {
    const up = forecast.direction === "UP";
    const directionColor = up ? C.green : C.red;

    // How far the quote has travelled from the bar the models read. The note at
    // the foot needs it, and it is the entire explanation of a box whose
    // Current Price and Forecast Price look like a rise beside a DOWN call.
    const anchor = Number(forecast.anchor_price);
    const quote = Number(forecast.current_price);
    const quoteGapPct =
        Number.isFinite(anchor) && Number.isFinite(quote) && anchor
            ? (quote / anchor - 1) * 100
            : null;

    return (
        <div
            style={{
                background: PANEL,
                border: `1px solid ${C.border}`,
                borderRadius: 10,
                padding: "20px 24px 6px",
            }}
        >
            <div
                style={{
                    color: C.textDim,
                    fontSize: 11,
                    fontWeight: 800,
                    letterSpacing: "0.14em",
                    textTransform: "uppercase",
                    paddingBottom: 16,
                }}
            >
                Next-timeframe forecast
            </div>

            <Row label={forecast.as_of ? `Prev Close · ${forecast.as_of}` : "Prev Close"}>
                {formatPrice(forecast.anchor_price)}
            </Row>
            <Row label="Current Price">
                {formatPrice(forecast.current_price)}
                {QUOTE_SOURCE[forecast.current_price_source] && (
                    <span style={{ color: C.textDim, fontSize: 12, fontWeight: 600, marginLeft: 8 }}>
                        {QUOTE_SOURCE[forecast.current_price_source]}
                    </span>
                )}
            </Row>
            <Row label="Forecast Price" emphasis>
                {formatPrice(forecast.forecast_price)}
            </Row>
            <Row label="Direction">
                <span style={{ color: directionColor, fontSize: 22, fontWeight: 900 }}>
                    {up ? "▲" : "▼"} {forecast.direction}
                </span>
            </Row>
            <Row label="Expected Change">
                <span style={{ color: Number(forecast.expected_change_pct) >= 0 ? C.green : C.red }}>
                    {formatPct(forecast.expected_change_pct)}
                </span>
                <span style={{ color: C.textDim, fontSize: 12, fontWeight: 600, marginLeft: 8 }}>
                    from prev close
                </span>
            </Row>
            <Row label="Forecast Horizon">
                <span style={{ color: C.textMid, fontWeight: 700, fontSize: 15 }}>
                    {forecast.horizon_label || "Next 1 Day"}
                    {forecast.forecast_date ? ` · ${forecast.forecast_date}` : ""}
                </span>
            </Row>

            {forecast.split && (
                <div
                    style={{
                        borderTop: `1px solid ${C.border}`,
                        padding: "12px 0",
                        color: C.textDim,
                        fontSize: 12,
                        lineHeight: 1.5,
                    }}
                >
                    {forecast.split_reason === "quote" ? (
                        <>
                            The models read the {forecast.as_of} close of{" "}
                            {formatPrice(forecast.anchor_price)} and forecast{" "}
                            {formatPrice(forecast.forecast_price)} from it — a move of{" "}
                            {formatPct(forecast.expected_change_pct)}. The live quote sits{" "}
                            {formatPct(quoteGapPct)} from that close, in a session no model saw,
                            so against the quote the same forecast reads{" "}
                            {formatPct(forecast.quote_change_pct)}. The call is the one on the
                            close.
                        </>
                    ) : (
                        <>
                            The models lean {forecast.direction} on probability while the forecast
                            price lands the other way against that same close. Both readings are
                            shown as they are.
                        </>
                    )}
                </div>
            )}
        </div>
    );
}

/**
 * The forecast box before the numbers exist.
 *
 * Same shell, same five rows, same heights as `ForecastBox` — so when the
 * models land, the values appear in place instead of the page growing a panel
 * and pushing the chart up. The chart above is already interactive at this
 * point; this is the only part still waiting.
 */
function ForecastBoxSkeleton({ symbol }) {
    return (
        <div
            style={{
                background: PANEL,
                border: `1px solid ${C.border}`,
                borderRadius: 10,
                padding: "20px 24px 6px",
            }}
        >
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 12,
                    paddingBottom: 16,
                }}
            >
                <span
                    style={{
                        color: C.textDim,
                        fontSize: 11,
                        fontWeight: 800,
                        letterSpacing: "0.14em",
                        textTransform: "uppercase",
                    }}
                >
                    Next-timeframe forecast
                </span>
                <span style={{ color: C.textDim, fontSize: 11 }}>
                    running the models for {symbol}…
                </span>
            </div>

            {[
                "Prev Close",
                "Current Price",
                "Forecast Price",
                "Direction",
                "Expected Change",
                "Forecast Horizon",
            ].map(
                (label, index) => (
                    <Row key={label} label={label} emphasis={index === 2}>
                        <span
                            className="skeleton-shimmer"
                            style={{
                                display: "inline-block",
                                width: index === 2 ? 132 : 96,
                                height: index === 2 ? 26 : 17,
                                borderRadius: 5,
                                background: "rgba(148,163,184,.14)",
                                verticalAlign: "middle",
                            }}
                        />
                    </Row>
                ),
            )}
            <div style={{ height: 6 }} />
        </div>
    );
}

function Notice({ tone = "dim", children }) {
    const palette =
        tone === "error"
            ? { color: C.red, border: "rgba(244,63,94,.35)", background: "rgba(244,63,94,.08)" }
            : { color: C.textMid, border: C.border, background: SURFACE };
    return (
        <div
            style={{
                padding: "14px 18px",
                border: `1px solid ${palette.border}`,
                background: palette.background,
                borderRadius: 10,
                color: palette.color,
                fontSize: 13,
                lineHeight: 1.55,
            }}
        >
            {children}
        </div>
    );
}

export default function PredictionsTab({
    selectedTicker,
    setSelectedTicker,
    watchlist = [],
    apiConnected,
    onBacktest,
}) {
    const [bars, setBars] = useState(RANGES[1].bars);

    // Two queries, because the two halves of this page cost three orders of
    // magnitude apart. The candles are a cached OHLCV download; the forecast is
    // Kronos sampling 128 transformer paths, and it is seconds. They used to be
    // one request, which meant the chart — ready almost immediately — sat blank
    // until the models finished, on every symbol switch.
    const historyQuery = useForecastHistory(selectedTicker, {
        days: HISTORY_BARS,
        enabled: apiConnected,
    });
    const query = useSimpleForecast(selectedTicker, { enabled: apiConnected });

    const data = query.data ?? null;
    // Only the forecast half is "loading" here; the chart has its own state and
    // must never be gated on this one.
    const loading = query.isPending;
    const error = query.error ? query.error.message || "The forecast could not be loaded." : null;

    const history = historyQuery.data?.bars ?? [];
    const historyLoading = historyQuery.isPending;
    const historyError = historyQuery.error
        ? historyQuery.error.message || "The price history could not be loaded."
        : null;
    const windowed = useMemo(() => history.slice(-bars), [history, bars]);
    const points = data?.forecast ?? [];
    const servable = data?.status === "ok" && points.length > 0;

    // The arrow on the chart carries the same UP/DOWN the box does, without the
    // probability behind it: the number is the model's, the tab's job is the
    // call. Memoised on the string so a re-render does not hand the chart a new
    // object and make it redraw its markers for an unchanged direction.
    const call = servable ? data.direction : null;
    const direction = useMemo(() => (call ? { direction: call } : null), [call]);

    const symbols = watchlist.includes(selectedTicker)
        ? watchlist
        : [selectedTicker, ...watchlist].filter(Boolean);

    if (!apiConnected) {
        return (
            <div style={{ padding: 48, color: C.textDim, textAlign: "center" }}>
                Connect to the API server to view the forecast.
            </div>
        );
    }

    return (
        <div style={{ display: "grid", gap: 16, paddingBottom: 36 }}>
            {/* ── Header: what am I looking at, and over what window ── */}
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "end",
                    gap: 16,
                    flexWrap: "wrap",
                }}
            >
                <div style={{ display: "grid", gap: 6 }}>
                    <div
                        style={{
                            color: C.textDim,
                            fontSize: 12,
                            fontWeight: 800,
                            letterSpacing: "0.08em",
                            textTransform: "uppercase",
                        }}
                    >
                        Predictions
                    </div>
                    <div style={{ color: C.text, fontSize: 26, fontWeight: 900, lineHeight: 1 }}>
                        {selectedTicker}
                    </div>
                </div>

                <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                    {typeof setSelectedTicker === "function" && symbols.length > 1 && (
                        <select
                            value={selectedTicker}
                            onChange={(event) => setSelectedTicker(event.target.value)}
                            style={controlStyle}
                            aria-label="Stock"
                        >
                            {symbols.map((symbol) => (
                                <option key={symbol} value={symbol}>
                                    {symbol}
                                </option>
                            ))}
                        </select>
                    )}
                    <div
                        style={{
                            display: "inline-flex",
                            background: SURFACE,
                            border: `1px solid ${C.border}`,
                            borderRadius: 8,
                            padding: 3,
                            gap: 3,
                        }}
                    >
                        {RANGES.map((range) => (
                            <button
                                key={range.label}
                                type="button"
                                onClick={() => setBars(range.bars)}
                                title={`Show ${range.label} of candles`}
                                style={{
                                    background: bars === range.bars ? C.amber : "transparent",
                                    color: bars === range.bars ? "#10131A" : C.textMid,
                                    border: "none",
                                    borderRadius: 6,
                                    padding: "7px 13px",
                                    fontSize: 12,
                                    fontWeight: 900,
                                    cursor: "pointer",
                                }}
                            >
                                {range.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* ── The chart: real candles, and the forecast continuing them ── */}
            <div
                style={{
                    background: PANEL,
                    border: `1px solid ${C.border}`,
                    borderRadius: 10,
                    padding: 12,
                }}
            >
                {historyLoading && !windowed.length ? (
                    <div
                        style={{
                            height: CHART_HEIGHT,
                            display: "grid",
                            placeItems: "center",
                            color: C.textDim,
                            fontSize: 13,
                        }}
                    >
                        Loading {selectedTicker}…
                    </div>
                ) : historyError && !windowed.length ? (
                    <div
                        style={{
                            height: CHART_HEIGHT,
                            display: "grid",
                            placeItems: "center",
                            color: C.red,
                            fontSize: 13,
                        }}
                    >
                        {historyError}
                    </div>
                ) : (
                    <ForecastOverlayChart
                        bars={windowed}
                        forecast={points}
                        direction={direction}
                        horizon={1}
                        height={CHART_HEIGHT}
                    />
                )}
            </div>

            {/* ── The answer ── */}
            {/* The one error state with no way out of itself. A forecast that
                failed on the network is not retried on window focus (that is off
                client-wide), and the query key only changes with the symbol or
                the range — so without this the tab stays broken until the user
                thinks to touch a control that has nothing to do with the error. */}
            {error && (
                <Notice tone="error">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                        <span>{error}</span>
                        <button
                            type="button"
                            onClick={() => query.refetch()}
                            disabled={query.isFetching}
                            style={{
                                background: "transparent",
                                border: `1px solid ${C.red}66`,
                                borderRadius: 6,
                                color: C.red,
                                padding: "5px 12px",
                                fontSize: 12,
                                fontWeight: 800,
                                cursor: query.isFetching ? "default" : "pointer",
                                opacity: query.isFetching ? 0.5 : 1,
                                whiteSpace: "nowrap",
                            }}
                        >
                            {query.isFetching ? "Retrying…" : "Try again"}
                        </button>
                    </div>
                </Notice>
            )}

            {!error && loading && <ForecastBoxSkeleton symbol={selectedTicker} />}

            {!error && data && !servable && (
                <Notice>{data.message || `No forecast is available for ${selectedTicker} right now.`}</Notice>
            )}

            {!error && servable && <ForecastBox forecast={data} />}

            {/* The one way out of this tab. A forecast is a claim about the next
                bar; this hands that claim — symbol, direction, the close it was
                made from — to the Backtest tab, which scores the same model
                against the bars it has already seen. Shown only once there is a
                forecast to carry, so it never appears as a dead control. */}
            {!error && servable && typeof onBacktest === "function" && (
                <button
                    type="button"
                    onClick={() =>
                        onBacktest({
                            symbol: selectedTicker,
                            direction: data.direction,
                            forecastPrice: data.forecast_price,
                            anchorPrice: data.anchor_price,
                            expectedChangePct: data.expected_change_pct,
                            asOf: data.as_of,
                            horizonLabel: data.horizon_label || "Next 1 Day",
                            models: data.models || [],
                        })
                    }
                    style={{
                        background: "transparent",
                        border: `1px solid ${C.amber}55`,
                        borderRadius: 10,
                        color: C.amber,
                        padding: "14px 18px",
                        fontSize: 13,
                        fontWeight: 800,
                        letterSpacing: "0.02em",
                        cursor: "pointer",
                        fontFamily: "inherit",
                    }}
                >
                    Backtest this prediction →
                </button>
            )}

            {/* ── Attribution, one line, because that is all it is worth ── */}
            {servable && data.models?.length > 0 && (
                <div style={{ color: C.textDim, fontSize: 12, textAlign: "center" }}>
                    Models: {data.models.join(" · ")}
                </div>
            )}
        </div>
    );
}
