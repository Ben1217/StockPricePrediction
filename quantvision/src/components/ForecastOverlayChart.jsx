/**
 * The price chart with the winning model's forecast drawn on it.
 *
 * Why this chart and not the Advanced Chart widget
 * ------------------------------------------------
 * `TradingViewChart` embeds TradingView's Advanced Chart, which is a
 * cross-origin iframe. There is no API on it to add a series, a shape or a
 * marker — the JSON config chooses a symbol, an interval and a list of
 * TradingView's own studies, and nothing else gets in. Anything painted over
 * the iframe from outside would be positioned in page pixels with no access to
 * the widget's price or time scales, so it would slide out of alignment the
 * first time anyone panned or zoomed. Superimposing our forecast on that widget
 * is not a thing that can be done, at any level of effort, without a licence
 * for TradingView's self-hosted Charting Library.
 *
 * So this pane is built on `lightweight-charts`, which is TradingView's own
 * open-source charting library and speaks the same visual language: same
 * candles, same crosshair, same time scale. What it adds is that we own the
 * canvas, so the forecast can be a real series on the same price scale as the
 * candles rather than a picture laid on top of one.
 *
 * The Advanced Chart is still one toggle away in the Predictions tab, because
 * it still wins on everything it was kept for — drawing tools, replay,
 * indicator search, comparison symbols. The division of labour in
 * `TradingViewChart`'s header comment is unchanged: TradingView visualises
 * price, our backend produces the analysis. This file is the one place the two
 * are drawn in the same coordinate space.
 *
 * What is on screen
 * -----------------
 *   candles          historical OHLCV from GET /api/data/prices
 *   dashed line      the price winner's trajectory, from the last real close
 *                    into future space on the right
 *   shaded band      its 95% and 68% intervals (see ForecastBand)
 *   line colour      green or red per segment, from that step's own direction
 *   arrow            the direction winner's call for the next bar, above or
 *                    below it, with P(up) in the label
 *
 * The arrow and the line come from two different models, which is not an
 * accident to be smoothed over: the backend ranks price and direction
 * separately because a model that tracks the level is often not the model that
 * calls the sign. The legend names both, and says so when they disagree.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
    CandlestickSeries,
    LineSeries,
    LineStyle,
    createChart,
    createSeriesMarkers,
} from "lightweight-charts";

import { C } from "../utils/data";
import { ForecastBand } from "./ForecastBand";

const COLORS = {
    up: "#10b981",
    down: "#f43f5e",
    flat: "#94a3b8",
    forecast: "#6366f1",
    grid: "rgba(42, 58, 92, 0.35)",
    panel: "#0F1623",
};

/** Trading days beyond the last forecast point, so the line is not flush right. */
const FUTURE_PADDING_BARS = 2;

/**
 * `lightweight-charts` keys its horizontal scale on `YYYY-MM-DD` for daily
 * bars, which is exactly what both APIs already return — so times pass through
 * untouched. Anything unparseable is dropped rather than coerced, because a bar
 * at the wrong date is worse than a missing one.
 */
function toBusinessDay(value) {
    const text = String(value || "").slice(0, 10);
    return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : null;
}

/**
 * A price, or NaN when the payload has no usable one.
 *
 * `Number(null)` and `Number("")` are both `0`, so a bare `Number()` turns a
 * missing bound into a real price of zero — which plots, silently, as a band
 * reaching the bottom of the pane. A missing number has to stay missing all the
 * way to the `Number.isFinite` check that drops it.
 */
function price(value) {
    if (value === null || value === undefined || value === "") return NaN;
    return Number(value);
}

function toCandles(bars) {
    if (!Array.isArray(bars)) return [];
    const rows = [];
    for (const bar of bars) {
        const time = toBusinessDay(bar?.date);
        if (!time) continue;
        const open = price(bar.open);
        const high = price(bar.high);
        const low = price(bar.low);
        const close = price(bar.close);
        if (![open, high, low, close].every(Number.isFinite)) continue;
        rows.push({ time, open, high, low, close });
    }
    // The series requires strictly ascending, de-duplicated times; the API sorts
    // already, but a repeated date from a provider hiccup would throw rather
    // than render, so the last write for a date wins.
    const byTime = new Map(rows.map((row) => [row.time, row]));
    return [...byTime.values()].sort((a, b) => (a.time < b.time ? -1 : 1));
}

/**
 * The forecast points that are actually drawable, in one place.
 *
 * Three things read the forecast — the line, the band and the arrows — and they
 * have to agree on which points exist. They will not if each filters for
 * itself: the API's first forecast bar is the next *business* day, which is the
 * same date as the last printed close whenever the price feed is a session
 * ahead of the model's last bar. Dropping that point from the line but not from
 * the markers would hang the direction arrow off the anchor rather than off the
 * bar it refers to.
 *
 * So the filter runs once and everything downstream takes its result.
 */
function alignForecast(forecast, anchor) {
    const points = [];
    for (const point of forecast || []) {
        const time = toBusinessDay(point?.date);
        const predicted = price(point?.predicted);
        if (!time || !Number.isFinite(predicted)) continue;
        // A forecast bar landing on or before the last historical bar would
        // collide with the anchor and break the ascending-time requirement.
        if (anchor && time <= anchor.time) continue;
        // `upper95`/`lower95` are this chart's slot names for the OUTER band and
        // are a drawing vocabulary, not a coverage claim — nothing on the pane
        // prints a percentage. Two payload shapes feed them: /predict/best sends
        // `upper95`, and /predict/forecast sends `upper_90`, whose bounds really
        // are the q0.05/q0.95 quantiles and so really are a 90% interval. Both
        // land in the same slot; neither is relabelled as the other.
        points.push({
            time,
            predicted,
            direction: point.direction,
            upper95: price(point.upper_90 ?? point.upper95),
            lower95: price(point.lower_90 ?? point.lower95),
            upper68: price(point.upper_68 ?? point.upper68),
            lower68: price(point.lower_68 ?? point.lower68),
        });
    }
    return points;
}

/**
 * The forecast as a line, anchored to the last real close.
 *
 * The anchor point matters: without it the dashed line starts one bar into the
 * future and reads as a second, unconnected series. With it, the forecast
 * visibly continues the candles.
 *
 * Each point carries its own colour, which `lightweight-charts` applies to the
 * segment ending at that point — so the line is green while the model has price
 * rising and red while it has price falling, without splitting it into separate
 * series that would each need their own legend entry.
 */
function toForecastLine(points, anchor) {
    const line = anchor
        ? [{ time: anchor.time, value: anchor.value, color: COLORS.forecast }]
        : [];
    for (const point of points) {
        line.push({
            time: point.time,
            value: point.predicted,
            color: COLORS[point.direction] || COLORS.forecast,
        });
    }
    return line;
}

function toBandPoints(points, anchor) {
    const band = anchor
        // The interval is zero wide at the last close, which is true: no model
        // is uncertain about a price that has already printed. Starting the
        // band there gives it the cone shape the uncertainty actually has.
        ? [{
            time: anchor.time,
            upper95: anchor.value,
            lower95: anchor.value,
            upper68: anchor.value,
            lower68: anchor.value,
        }]
        : [];
    for (const point of points) {
        const bounds = [point.upper95, point.lower95, point.upper68, point.lower68];
        // A next-bar model returns a point forecast with no interval, which
        // arrives as four copies of the same price — nothing is lost by drawing
        // that. A non-finite bound would poison the polygon, so it is dropped.
        if (!bounds.every(Number.isFinite)) continue;
        band.push({
            time: point.time,
            upper95: point.upper95,
            lower95: point.lower95,
            upper68: point.upper68,
            lower68: point.lower68,
        });
    }
    return band;
}

/**
 * The arrows.
 *
 * Two kinds, and keeping them distinct is the point. The first future bar gets
 * the *direction winner's* call, labelled with its probability — that is the
 * model that won accuracy/F1/AUC, and its call is the one the brief asks to see
 * on the chart. Every later bar gets a small marker only where the price
 * winner's trajectory turns, because an arrow on all sixty bars of a monotone
 * path is noise, not information.
 */
function toMarkers(points, direction, anchor) {
    const markers = [];
    if (!points.length) return markers;

    const target = points[points.length - 1];
    const rising = anchor ? target.predicted >= anchor.value : target.direction !== "down";
    const movePct =
        anchor && anchor.value
            ? ((target.predicted - anchor.value) / anchor.value) * 100
            : null;

    // The estimate, always labelled with the number it is making. This is the
    // foundation stack's own output and the chart is that stack's chart, so it
    // is drawn whether or not a scored call exists — a forecast line ending in
    // an unmarked point leaves the reader to eyeball a price off the axis.
    //
    // It says "est" because that is what it is. The scored call, when there is
    // one, is a different statement and gets its own marker below.
    const moveLabel = movePct === null
        ? `est $${target.predicted.toFixed(2)}`
        : `est $${target.predicted.toFixed(2)} (${movePct >= 0 ? "+" : ""}${movePct.toFixed(2)}%)`;
    markers.push({
        time: target.time,
        position: rising ? "aboveBar" : "belowBar",
        shape: "circle",
        color: rising ? COLORS.up : COLORS.down,
        text: moveLabel,
        size: 1,
    });

    // The scored direction, when the measured stack made one. Deliberately on
    // the opposite side of the bar from the estimate so the two labels cannot
    // overlap, and carrying its probability — which is the whole reason it is
    // a different kind of claim from the price above it.
    //
    // Absent on NEUTRAL, which is the common answer: the caller passes null
    // rather than a direction, and drawing an arrow anyway would put the one
    // call nobody has scored in the most prominent place on the page.
    if (direction?.direction) {
        const up = String(direction.direction).toUpperCase() === "UP";
        const probability = Number(direction.probability_up);
        // The label states the probability of the direction it is claiming, so
        // a DOWN arrow reads "DOWN 89%" rather than "DOWN 11%".
        const claimed = up ? probability : 1 - probability;
        const label = Number.isFinite(claimed)
            ? `${up ? "UP" : "DOWN"} ${Math.round(claimed * 100)}%`
            : up ? "UP" : "DOWN";
        markers.push({
            time: target.time,
            position: rising ? "belowBar" : "aboveBar",
            shape: up ? "arrowUp" : "arrowDown",
            color: direction.tradeable === false ? COLORS.flat : up ? COLORS.up : COLORS.down,
            text: label,
            size: 2,
        });
    }

    return markers;
}

function futurePadding(lastTime, bars = FUTURE_PADDING_BARS) {
    if (!lastTime) return [];
    const rows = [];
    const date = new Date(`${lastTime}T00:00:00Z`);
    let added = 0;
    while (added < bars) {
        date.setUTCDate(date.getUTCDate() + 1);
        const day = date.getUTCDay();
        if (day === 0 || day === 6) continue; // trading days only, like the forecast
        rows.push({ time: date.toISOString().slice(0, 10) });
        added += 1;
    }
    return rows;
}

export default function ForecastOverlayChart({
    bars,
    forecast,
    direction,
    horizon,
    height = 520,
}) {
    const containerRef = useRef(null);
    const chartRef = useRef(null);
    const candleSeriesRef = useRef(null);
    const forecastSeriesRef = useRef(null);
    const bandRef = useRef(null);
    const markersRef = useRef(null);
    const priceLinesRef = useRef([]);
    const [ready, setReady] = useState(false);

    const candles = useMemo(() => toCandles(bars), [bars]);
    // Only the bars inside the selected window. The 7D button must not draw a
    // 60-day line just because the backend was asked for the longest horizon
    // once and the response is still in cache.
    const windowed = useMemo(
        () => (Array.isArray(forecast) ? forecast.slice(0, Math.max(1, Number(horizon) || 0)) : []),
        [forecast, horizon],
    );

    /* Build the chart once, and tear it down on unmount. Data goes in through
       the effects below, so a new forecast never rebuilds the chart — which is
       what keeps the user's pan and zoom across a horizon change. */
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return undefined;

        const chart = createChart(container, {
            autoSize: true,
            layout: {
                background: { color: COLORS.panel },
                textColor: C.textMid,
                fontSize: 11,
                attributionLogo: false,
            },
            grid: {
                vertLines: { color: COLORS.grid },
                horzLines: { color: COLORS.grid },
            },
            rightPriceScale: { borderColor: C.border, scaleMargins: { top: 0.12, bottom: 0.12 } },
            timeScale: { borderColor: C.border, rightOffset: 4, fixLeftEdge: true },
            crosshair: { mode: 0 },
            localization: {
                priceFormatter: (value) => `$${Number(value).toFixed(2)}`,
            },
        });

        // `lastValueVisible` is off on BOTH series. Left on, the candles
        // printed the last close and the forecast line printed the estimate as
        // two anonymous numbers stacked on the same few pixels of the price
        // axis — $230.36 over $230.50 — with nothing saying which was which.
        // Named price lines replace them below, so each number carries its word.
        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: COLORS.up,
            downColor: COLORS.down,
            wickUpColor: COLORS.up,
            wickDownColor: COLORS.down,
            borderVisible: false,
            priceLineVisible: false,
            lastValueVisible: false,
        });

        const forecastSeries = chart.addSeries(LineSeries, {
            color: COLORS.forecast,
            lineWidth: 3,
            lineStyle: LineStyle.Dashed,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: true,
        });

        const band = new ForecastBand([]);
        forecastSeries.attachPrimitive(band);

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;
        forecastSeriesRef.current = forecastSeries;
        bandRef.current = band;
        markersRef.current = createSeriesMarkers(forecastSeries, []);
        setReady(true);

        return () => {
            setReady(false);
            markersRef.current = null;
            priceLinesRef.current = [];
            bandRef.current = null;
            forecastSeriesRef.current = null;
            candleSeriesRef.current = null;
            chartRef.current = null;
            chart.remove();
        };
    }, []);

    /* Candles. */
    useEffect(() => {
        if (!ready || !candleSeriesRef.current) return;
        candleSeriesRef.current.setData(candles);
    }, [ready, candles]);

    /* Forecast line, band and markers — one effect, because all three are
       derived from the same points and must never be a frame out of step. */
    useEffect(() => {
        if (!ready || !forecastSeriesRef.current) return;

        const lastCandle = candles[candles.length - 1];
        const anchor = lastCandle ? { time: lastCandle.time, value: lastCandle.close } : null;
        const points = alignForecast(windowed, anchor);
        const line = toForecastLine(points, anchor);

        const padded = line.length
            ? [...line, ...futurePadding(line[line.length - 1].time)]
            : [];
        forecastSeriesRef.current.setData(padded);
        bandRef.current?.setPoints(toBandPoints(points, anchor));
        markersRef.current?.setMarkers(toMarkers(points, direction, anchor));

        // ── Named price lines ────────────────────────────────────────────
        // Rebuilt every update, because a stale line would sit at a price from
        // the previous symbol. They replace the two unlabelled axis values.
        for (const line of priceLinesRef.current) {
            try { forecastSeriesRef.current.removePriceLine(line); } catch { /* series replaced */ }
        }
        priceLinesRef.current = [];

        if (anchor) {
            priceLinesRef.current.push(forecastSeriesRef.current.createPriceLine({
                price: anchor.value,
                color: COLORS.flat,
                lineWidth: 1,
                lineStyle: LineStyle.Dotted,
                axisLabelVisible: true,
                title: "prev close",
            }));
        }
        const target = points[points.length - 1];
        if (target) {
            const rising = anchor ? target.predicted >= anchor.value : true;
            priceLinesRef.current.push(forecastSeriesRef.current.createPriceLine({
                price: target.predicted,
                color: rising ? COLORS.up : COLORS.down,
                lineWidth: 2,
                lineStyle: LineStyle.Dashed,
                axisLabelVisible: true,
                title: "estimate",
            }));
        }

        // ── The visible window, set explicitly ───────────────────────────
        //
        // `fitContent()` used to do this, and what reached the screen did not
        // match what was asked for: the 6M button passes 126 sessions and the
        // pane rendered about thirteen, roughly a fortnight, with the right
        // half of the plot empty. Whatever the library was fitting, it was not
        // the data — so the range is now stated rather than inferred, from the
        // candles this component was actually handed.
        //
        // `setVisibleRange` takes the two ends and nothing else can widen or
        // narrow them, so the range buttons above the chart now govern the
        // pane directly.
        const timeScale = chartRef.current?.timeScale();
        const lastDrawn = padded.length ? padded[padded.length - 1].time : null;
        if (timeScale && candles.length) {
            try {
                timeScale.setVisibleRange({
                    from: candles[0].time,
                    to: lastDrawn || candles[candles.length - 1].time,
                });
            } catch {
                // A range the scale rejects (one bar, or a forecast that landed
                // before the last candle) is not worth failing the render for.
                timeScale.fitContent();
            }
        }
    }, [ready, candles, windowed, direction]);

    // What the pane is actually showing, in words. The chart has three visual
    // languages on it — candles, a dashed line, a shaded cone — and none of
    // them is self-evident to someone who has not been told. The session count
    // is here for a second reason: the range buttons above silently failed once,
    // drawing a fortnight when six months was selected, and a number on the
    // chart is what makes that visible instead of merely wrong.
    const shown = candles.length;
    const target = windowed.length ? windowed[windowed.length - 1] : null;

    return (
        <div style={{ position: "relative", width: "100%" }}>
            <div
                ref={containerRef}
                style={{ height, width: "100%", position: "relative" }}
            />

            <div
                style={{
                    position: "absolute",
                    top: 8,
                    left: 10,
                    right: 10,
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    gap: 12,
                    pointerEvents: "none",
                    flexWrap: "wrap",
                    fontFamily: "'DM Mono',monospace",
                }}
            >
                <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 10.5, color: C.textMid }}>
                    <LegendKey swatch={<span style={{ display: "inline-flex", gap: 2 }}>
                        <i style={{ width: 4, height: 11, background: COLORS.up, borderRadius: 1, display: "inline-block" }} />
                        <i style={{ width: 4, height: 11, background: COLORS.down, borderRadius: 1, display: "inline-block" }} />
                    </span>}>
                        actual sessions
                    </LegendKey>
                    <LegendKey swatch={
                        <i style={{
                            width: 18, height: 0, display: "inline-block",
                            borderTop: `2px dashed ${COLORS.forecast}`,
                        }} />
                    }>
                        next-session estimate
                    </LegendKey>
                    <LegendKey swatch={
                        <i style={{
                            width: 14, height: 11, display: "inline-block", borderRadius: 2,
                            background: "rgba(99,102,241,.22)", border: "1px solid rgba(129,140,248,.55)",
                        }} />
                    }>
                        68% / 90% range
                    </LegendKey>
                </div>

                <div style={{ fontSize: 10.5, color: C.textDim, textAlign: "right", whiteSpace: "nowrap" }}>
                    {shown} session{shown === 1 ? "" : "s"}
                    {target?.date ? ` · forecast ${String(target.date).slice(0, 10)}` : ""}
                </div>
            </div>
        </div>
    );
}

/** One legend entry: a swatch that looks like the thing, and its name. */
function LegendKey({ swatch, children }) {
    return (
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            {swatch}
            {children}
        </span>
    );
}
