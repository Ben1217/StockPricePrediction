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
        points.push({
            time,
            predicted,
            direction: point.direction,
            upper95: price(point.upper95),
            lower95: price(point.lower95),
            upper68: price(point.upper68),
            lower68: price(point.lower68),
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
function toMarkers(points, direction) {
    const markers = [];
    if (!points.length) return markers;

    if (direction?.direction) {
        // The route canonicalises this to UP/DOWN whichever model produced it,
        // so there is one vocabulary here rather than the Bullish/Bearish the
        // panels below the chart use for the same fact.
        const up = String(direction.direction).toUpperCase() === "UP";
        const probability = Number(direction.probability_up);
        // The label states the probability of the direction it is claiming, so
        // a DOWN arrow reads "DOWN 89%" rather than "DOWN 11%".
        const claimed = up ? probability : 1 - probability;
        const label = Number.isFinite(claimed)
            ? `${up ? "UP" : "DOWN"} ${Math.round(claimed * 100)}%`
            : up ? "UP" : "DOWN";
        markers.push({
            time: points[0].time,
            position: up ? "belowBar" : "aboveBar",
            shape: up ? "arrowUp" : "arrowDown",
            // A call the walk-forward run refused to ship is drawn muted, so it
            // cannot be read as the same kind of statement as a gated-through one.
            color: direction.tradeable === false ? COLORS.flat : up ? COLORS.up : COLORS.down,
            text: label,
            size: 2,
        });
    }

    let previous = null;
    for (const point of points) {
        if (point.direction === "flat") continue;
        if (previous && point.direction === previous) continue;
        // The first bar already carries the direction model's arrow; a second
        // one on the same bar would stack two claims on top of each other.
        if (point.time !== points[0].time) {
            const up = point.direction === "up";
            markers.push({
                time: point.time,
                position: up ? "belowBar" : "aboveBar",
                shape: up ? "arrowUp" : "arrowDown",
                color: up ? COLORS.up : COLORS.down,
                size: 1,
            });
        }
        previous = point.direction;
    }
    return markers;
}

/**
 * Empty bars past the last forecast point.
 *
 * `lightweight-charts` will not scroll past its last data point, so without
 * these the forecast ends hard against the right edge of the pane. Whitespace
 * data extends the time scale without drawing anything.
 */
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

        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: COLORS.up,
            downColor: COLORS.down,
            wickUpColor: COLORS.up,
            wickDownColor: COLORS.down,
            borderVisible: false,
            priceLineVisible: false,
        });

        const forecastSeries = chart.addSeries(LineSeries, {
            color: COLORS.forecast,
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            priceLineVisible: false,
            lastValueVisible: true,
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
        markersRef.current?.setMarkers(toMarkers(points, direction));

        // Fit once there is something to fit, so the forecast is in view rather
        // than off the right edge on first paint.
        if (padded.length) chartRef.current?.timeScale().fitContent();
    }, [ready, candles, windowed, direction]);

    return (
        <div
            ref={containerRef}
            style={{ height, width: "100%", position: "relative" }}
        />
    );
}
