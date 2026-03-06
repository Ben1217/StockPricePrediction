import React, { useEffect, useRef, useState } from "react";
import { createChart, CandlestickSeries, LineSeries, HistogramSeries, AreaSeries, createSeriesMarkers } from "lightweight-charts";
import { fetchPrices, fetchIndicators, fetchHistoricalSignals, fetchPatterns } from "../utils/api";
import { C } from "../utils/data";

function parseChartTime(dateStr) {
    if (!dateStr) return null;
    if (dateStr.length === 10) return dateStr; // YYYY-MM-DD
    return Math.floor(new Date(dateStr).getTime() / 1000);
}

// ── Shared chart options ───────────────────────────────────
const baseChartOpts = (interval) => ({
    layout: { background: { type: 'solid', color: C.bg1 }, textColor: C.textDim },
    grid: { vertLines: { color: C.border }, horzLines: { color: C.border } },
    rightPriceScale: { borderColor: C.border },
    timeScale: {
        borderColor: C.border,
        timeVisible: interval.includes("m") || interval === "1h" || interval === "4h",
        secondsVisible: false,
    },
});

// ── Confluence Panel ───────────────────────────────────────
function ConfluencePanel({ confluence }) {
    if (!confluence) return null;
    const { rsi_signal, rsi_value, macd_signal, pattern_signal, ml_direction, ml_confidence, overall, strength } = confluence;

    const overallColor = overall.includes("Buy") ? C.green : overall.includes("Sell") ? C.red : C.amber;
    const signalIcon = (sig) => {
        if (sig === "bullish" || sig === "bullish_cross" || sig === "oversold" || sig === "up") return { icon: "▲", color: C.green };
        if (sig === "bearish" || sig === "bearish_cross" || sig === "overbought" || sig === "down") return { icon: "▼", color: C.red };
        return { icon: "●", color: C.textDim };
    };

    const rows = [
        { label: "Pattern", signal: pattern_signal },
        { label: "RSI", signal: rsi_signal, extra: `${rsi_value}` },
        { label: "MACD", signal: macd_signal },
        { label: "ML Model", signal: ml_direction, extra: `${ml_confidence.toFixed(0)}%` },
    ];

    return (
        <div style={{
            position: "absolute", top: 8, right: 8, zIndex: 20,
            background: C.bg0 + "EE", border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "10px 14px", minWidth: 180, backdropFilter: "blur(8px)",
            fontFamily: "'DM Mono', monospace", fontSize: 10,
        }}>
            <div style={{ fontWeight: 800, color: overallColor, fontSize: 13, marginBottom: 6, textAlign: "center", letterSpacing: 1 }}>
                {overall.toUpperCase()}
            </div>
            <div style={{ width: "100%", height: 3, borderRadius: 2, background: C.border, marginBottom: 8 }}>
                <div style={{ width: `${Math.min(100, strength * 100)}%`, height: "100%", borderRadius: 2, background: overallColor, transition: "width 0.3s" }} />
            </div>
            {rows.map(r => {
                const s = signalIcon(r.signal);
                return (
                    <div key={r.label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "2px 0" }}>
                        <span style={{ color: C.textDim }}>{r.label}</span>
                        <span style={{ color: s.color }}>
                            {s.icon} {r.signal?.replace("_", " ")} {r.extra || ""}
                        </span>
                    </div>
                );
            })}
        </div>
    );
}

// ── Sub-panel Chart Creator ────────────────────────────────
function createSubChart(container, interval, height = 100) {
    if (!container) return null;
    const w = container.clientWidth || 900;
    return createChart(container, {
        width: w, height,
        ...baseChartOpts(interval),
        crosshair: { mode: 1 },
    });
}

// ── Main Component ─────────────────────────────────────────
export default function TradingViewDetail({ symbol, mode = "analysis", predictionData = null, onClose }) {
    const chartContainerRef = useRef(null);
    const rsiContainerRef = useRef(null);
    const macdContainerRef = useRef(null);
    const volContainerRef = useRef(null);

    const chartRef = useRef(null);
    const subChartsRef = useRef({});
    const seriesRefs = useRef({});

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [confluence, setConfluence] = useState(null);

    const [interval, setInterval] = useState("1d");
    const intervals = ["1m", "5m", "15m", "1h", "4h", "1d", "1wk"];

    // Indicator toggles — on-chart
    const [showSMA, setShowSMA] = useState(true);
    const [showEMA, setShowEMA] = useState(true);
    const [showBB, setShowBB] = useState(true);
    const [showVWAP, setShowVWAP] = useState(true);

    // Sub-panel toggles
    const [showRSI, setShowRSI] = useState(true);
    const [showMACD, setShowMACD] = useState(true);
    const [showVol, setShowVol] = useState(true);
    const [showATR, setShowATR] = useState(false);

    const stateData = useRef({ ohlc: [], indicators: [], signals: [], patterns: null });

    useEffect(() => {
        let cancelled = false;

        async function loadData() {
            setLoading(true); setError(null);
            try {
                let days = interval.includes("m") ? 5 : (interval === "1h" ? 20 : (interval === "4h" ? 60 : 120));
                const lookback = Math.max(days, 60);

                const [priceRes, indRes, patRes] = await Promise.all([
                    fetchPrices(symbol, "yfinance", days, interval),
                    fetchIndicators(symbol, days, interval),
                    fetchPatterns(symbol, interval, lookback),
                ].map(p => p.catch(e => null)));

                if (cancelled) return;
                if (!priceRes || !priceRes.bars) throw new Error("Failed to fetch price data");

                let sigRes = [];
                if (interval === "1d") {
                    try { sigRes = await fetchHistoricalSignals(symbol, 90, "xgboost"); } catch (e) { }
                }
                if (cancelled) return;

                stateData.current = {
                    ohlc: priceRes.bars,
                    indicators: indRes?.data || [],
                    signals: sigRes || [],
                    patterns: patRes || null,
                };

                if (patRes?.confluence) setConfluence(patRes.confluence);
                renderAll();
            } catch (err) {
                if (!cancelled) setError(err.message);
            } finally {
                if (!cancelled) setLoading(false);
            }
        }

        function cleanup() {
            Object.values(subChartsRef.current).forEach(c => { try { c.remove(); } catch (e) { } });
            subChartsRef.current = {};
            if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }
            seriesRefs.current = {};
        }

        function renderAll() {
            if (cancelled) return;
            cleanup();
            renderMainChart();
            renderSubPanels();
        }

        function renderMainChart() {
            if (!chartContainerRef.current) return;

            const w = chartContainerRef.current.clientWidth || 900;
            const h = chartContainerRef.current.clientHeight || 380;

            const chart = createChart(chartContainerRef.current, {
                width: w, height: h,
                ...baseChartOpts(interval),
                crosshair: { mode: 1 },
            });
            chartRef.current = chart;

            // ── Candlestick ────────────────────────────────
            const candleSeries = chart.addSeries(CandlestickSeries, {
                upColor: C.green, downColor: C.red,
                borderVisible: false,
                wickUpColor: C.green, wickDownColor: C.red,
            });
            seriesRefs.current.candle = candleSeries;

            const { ohlc, indicators, signals, patterns } = stateData.current;
            const ohlcData = ohlc.map(b => ({
                time: parseChartTime(b.date),
                open: b.open, high: b.high, low: b.low, close: b.close
            })).filter(b => b.time);
            candleSeries.setData(ohlcData);

            // ── Indicator Map ──────────────────────────────
            const indMap = {};
            indicators.forEach(i => indMap[parseChartTime(i.date)] = i);

            // ── On-Chart Overlays ──────────────────────────
            if (showSMA) {
                const s = chart.addSeries(LineSeries, { color: C.cyan, lineWidth: 1, lineStyle: 2 });
                s.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.SMA_20 })).filter(d => d.value != null));
            }
            if (showEMA) {
                // EMA 12
                const e12 = chart.addSeries(LineSeries, { color: "#4ade80", lineWidth: 1 });
                e12.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.EMA_12 })).filter(d => d.value != null));
                // EMA 26
                const e26 = chart.addSeries(LineSeries, { color: "#f472b6", lineWidth: 1 });
                e26.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.EMA_26 })).filter(d => d.value != null));
            }
            if (showBB) {
                const bbUp = chart.addSeries(LineSeries, { color: C.purple, lineWidth: 1, lineStyle: 3 });
                const bbMid = chart.addSeries(LineSeries, { color: C.purple + '88', lineWidth: 1, lineStyle: 2 });
                const bbLow = chart.addSeries(LineSeries, { color: C.purple, lineWidth: 1, lineStyle: 3 });
                bbUp.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.BB_High })).filter(d => d.value != null));
                bbMid.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.BB_Mid })).filter(d => d.value != null));
                bbLow.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.BB_Low })).filter(d => d.value != null));
            }
            // VWAP — intraday only
            const isIntraday = interval.includes("m") || interval === "1h" || interval === "4h";
            if (showVWAP && isIntraday) {
                const vwap = chart.addSeries(LineSeries, { color: "#f59e0b", lineWidth: 2, lineStyle: 0 });
                vwap.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.VWAP })).filter(d => d.value != null));
            }

            // ── ML Signal + Candlestick Pattern Markers ────
            const allMarkers = [];

            // ML signals
            if (signals && signals.length > 0) {
                signals.forEach(s => {
                    const isBuy = s.type === "BUY";
                    allMarkers.push({
                        time: parseChartTime(s.date),
                        position: isBuy ? 'belowBar' : 'aboveBar',
                        color: isBuy ? C.green : C.red,
                        shape: isBuy ? 'arrowUp' : 'arrowDown',
                        text: `${s.type} ${(s.confidence || 0).toFixed(0)}%`
                    });
                });
            }

            // Candlestick patterns
            if (patterns?.candlestick_patterns) {
                patterns.candlestick_patterns.forEach(p => {
                    const isBullish = p.direction === "bullish";
                    allMarkers.push({
                        time: parseChartTime(p.date),
                        position: isBullish ? 'belowBar' : 'aboveBar',
                        color: isBullish ? "#22d3ee" : "#fb923c",
                        shape: isBullish ? 'arrowUp' : 'arrowDown',
                        text: `${p.pattern_name} (${(p.confidence * 100).toFixed(0)}%)`
                    });
                });
            }

            // Filter markers to only those matching chart data, deduplicate by time+position, sort
            const validTimes = new Set(ohlcData.map(d => d.time));
            const uniqueMarkers = [];
            const seen = new Set();
            allMarkers.filter(m => m.time && validTimes.has(m.time)).forEach(m => {
                const key = `${m.time}-${m.position}`;
                if (!seen.has(key)) { seen.add(key); uniqueMarkers.push(m); }
            });
            uniqueMarkers.sort((a, b) => {
                const tA = typeof a.time === 'string' ? new Date(a.time).getTime() : a.time;
                const tB = typeof b.time === 'string' ? new Date(b.time).getTime() : b.time;
                return tA - tB;
            });
            if (uniqueMarkers.length > 0) {
                createSeriesMarkers(candleSeries, uniqueMarkers);
            }

            // ── Chart Pattern Trendlines ───────────────────
            if (patterns?.chart_patterns) {
                patterns.chart_patterns.forEach(cp => {
                    if (cp.key_levels && cp.key_levels.length >= 2) {
                        const isBullish = cp.pattern_name.includes("Bottom") || cp.pattern_name.includes("Inverse") ||
                            cp.pattern_name.includes("Bull") || cp.pattern_name.includes("Ascending") ||
                            cp.pattern_name.includes("Cup");
                        const lineColor = isBullish ? C.green + 'AA' : C.red + 'AA';

                        // Draw trendline through key levels
                        const trendSeries = chart.addSeries(LineSeries, {
                            color: lineColor, lineWidth: 2, lineStyle: 2,
                            lastValueVisible: false, priceLineVisible: false,
                        });
                        const points = cp.key_levels.map(kl => ({
                            time: parseChartTime(kl.date), value: kl.price
                        })).filter(p => p.time);
                        if (points.length >= 2) trendSeries.setData(points);

                        // Neckline
                        if (cp.neckline) {
                            candleSeries.createPriceLine({
                                price: cp.neckline,
                                color: C.amber + '88',
                                lineWidth: 1, lineStyle: 2,
                                axisLabelVisible: true,
                                title: cp.pattern_name,
                            });
                        }

                        // Breakout zone
                        if (cp.breakout_price && cp.status !== "broken") {
                            candleSeries.createPriceLine({
                                price: cp.breakout_price,
                                color: "#fbbf24",
                                lineWidth: 1, lineStyle: 3,
                                axisLabelVisible: true,
                                title: "Breakout Zone",
                            });
                        }

                        // Target price (confirmed only)
                        if (cp.target_price && cp.status === "confirmed") {
                            candleSeries.createPriceLine({
                                price: cp.target_price,
                                color: isBullish ? C.green + 'CC' : C.red + 'CC',
                                lineWidth: 1, lineStyle: 4,
                                axisLabelVisible: true,
                                title: `Target: $${cp.target_price.toFixed(2)}`,
                            });
                        }
                    }
                });
            }

            // ── Prediction Overlays ────────────────────────
            if (mode === "prediction" && predictionData?.forecasts) {
                const { forecasts } = predictionData;
                const predSeries = chart.addSeries(LineSeries, { color: C.amber, lineWidth: 2 });
                const u95 = chart.addSeries(LineSeries, { color: C.amber + '55', lineWidth: 1, lineStyle: 3 });
                const l95 = chart.addSeries(LineSeries, { color: C.amber + '55', lineWidth: 1, lineStyle: 3 });
                const u68 = chart.addSeries(LineSeries, { color: C.amber + '88', lineWidth: 1, lineStyle: 3 });
                const l68 = chart.addSeries(LineSeries, { color: C.amber + '88', lineWidth: 1, lineStyle: 3 });

                const dP = [], dU95 = [], dL95 = [], dU68 = [], dL68 = [];
                forecasts.forEach(f => {
                    const t = parseChartTime(f.date);
                    if (!t) return;
                    dP.push({ time: t, value: f.predicted });
                    dU95.push({ time: t, value: f.upper95 });
                    dL95.push({ time: t, value: f.lower95 });
                    dU68.push({ time: t, value: f.upper68 });
                    dL68.push({ time: t, value: f.lower68 });
                });
                const lastH = ohlcData[ohlcData.length - 1];
                if (lastH) {
                    [dP, dU95, dL95, dU68, dL68].forEach(arr => arr.unshift({ time: lastH.time, value: lastH.close }));
                }
                predSeries.setData(dP); u95.setData(dU95); l95.setData(dL95); u68.setData(dU68); l68.setData(dL68);
            }

            chart.timeScale().fitContent();
        }

        function renderSubPanels() {
            const { ohlc, indicators } = stateData.current;
            const indMap = {};
            indicators.forEach(i => indMap[parseChartTime(i.date)] = i);
            const ohlcData = ohlc.map(b => ({ time: parseChartTime(b.date), close: b.close, open: b.open, volume: b.volume })).filter(b => b.time);

            // ── RSI Sub-panel ──────────────────────────────
            if (showRSI && rsiContainerRef.current) {
                const rsiChart = createSubChart(rsiContainerRef.current, interval, 100);
                if (rsiChart) {
                    subChartsRef.current.rsi = rsiChart;
                    const rsiSeries = rsiChart.addSeries(LineSeries, { color: "#a78bfa", lineWidth: 1.5 });
                    const rsiData = ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.RSI })).filter(d => d.value != null);
                    rsiSeries.setData(rsiData);
                    // 70/30 lines
                    rsiSeries.createPriceLine({ price: 70, color: C.red + '88', lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: "" });
                    rsiSeries.createPriceLine({ price: 30, color: C.green + '88', lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: "" });
                    rsiSeries.createPriceLine({ price: 50, color: C.textDim + '44', lineWidth: 1, lineStyle: 3, axisLabelVisible: false, title: "" });
                    rsiChart.timeScale().fitContent();
                }
            }

            // ── MACD Sub-panel ─────────────────────────────
            if (showMACD && macdContainerRef.current) {
                const macdChart = createSubChart(macdContainerRef.current, interval, 100);
                if (macdChart) {
                    subChartsRef.current.macd = macdChart;
                    const macdLine = macdChart.addSeries(LineSeries, { color: "#60a5fa", lineWidth: 1.5 });
                    const sigLine = macdChart.addSeries(LineSeries, { color: "#f97316", lineWidth: 1 });
                    const histSeries = macdChart.addSeries(HistogramSeries, { priceFormat: { type: 'price' }, priceScaleId: '' });

                    macdLine.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.MACD })).filter(d => d.value != null));
                    sigLine.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.MACD_Signal })).filter(d => d.value != null));
                    histSeries.setData(ohlcData.map(b => {
                        const v = indMap[b.time]?.MACD_Histogram;
                        return v != null ? { time: b.time, value: v, color: v >= 0 ? C.green + '88' : C.red + '88' } : null;
                    }).filter(Boolean));
                    macdChart.priceScale('').applyOptions({ scaleMargins: { top: 0.1, bottom: 0.1 } });
                    macdChart.timeScale().fitContent();
                }
            }

            // ── Volume Sub-panel ───────────────────────────
            if (showVol && volContainerRef.current) {
                const volChart = createSubChart(volContainerRef.current, interval, 80);
                if (volChart) {
                    subChartsRef.current.vol = volChart;
                    const volSeries = volChart.addSeries(HistogramSeries, {
                        priceFormat: { type: 'volume' }, priceScaleId: '',
                    });
                    volSeries.setData(ohlcData.map(b => ({
                        time: b.time, value: b.volume,
                        color: b.close >= b.open ? C.green + '66' : C.red + '66'
                    })));
                    // Volume MA overlay
                    const volMA = volChart.addSeries(LineSeries, { color: C.amber, lineWidth: 1, priceScaleId: '' });
                    volMA.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.Volume_SMA_20 })).filter(d => d.value != null));
                    volChart.priceScale('').applyOptions({ scaleMargins: { top: 0.05, bottom: 0.0 } });
                    volChart.timeScale().fitContent();
                }
            }
        }

        loadData();

        const handleResize = () => {
            if (chartRef.current && chartContainerRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
            Object.entries(subChartsRef.current).forEach(([key, c]) => {
                const ref = key === 'rsi' ? rsiContainerRef : key === 'macd' ? macdContainerRef : volContainerRef;
                if (ref.current) c.applyOptions({ width: ref.current.clientWidth });
            });
        };
        window.addEventListener("resize", handleResize);

        return () => {
            cancelled = true;
            window.removeEventListener("resize", handleResize);
            cleanup();
        };
    }, [symbol, interval, showSMA, showEMA, showBB, showVWAP, showRSI, showMACD, showVol, showATR, mode, predictionData]);

    const isIntraday = interval.includes("m") || interval === "1h" || interval === "4h";

    // ── Toolbar toggle definitions ─────────────────────────
    const onChartToggles = [
        { label: "SMA", active: showSMA, set: setShowSMA },
        { label: "EMA", active: showEMA, set: setShowEMA },
        { label: "BB", active: showBB, set: setShowBB },
        ...(isIntraday ? [{ label: "VWAP", active: showVWAP, set: setShowVWAP }] : []),
    ];
    const subPanelToggles = [
        { label: "RSI", active: showRSI, set: setShowRSI },
        { label: "MACD", active: showMACD, set: setShowMACD },
        { label: "Vol", active: showVol, set: setShowVol },
        { label: "ATR", active: showATR, set: setShowATR },
    ];

    return (
        <div className="fade-up" style={{ display: "flex", flexDirection: "column", width: "100%" }}>
            {/* Toolbar */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, background: C.bg2, padding: "6px 14px", borderRadius: 8, border: `1px solid ${C.border}`, flexWrap: "wrap", gap: 8 }}>
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                    <div style={{ fontWeight: 800, color: C.text, fontSize: 16, marginRight: 6 }}>{symbol}</div>
                    <div style={{ display: "flex", background: C.bg0, padding: 2, borderRadius: 5 }}>
                        {intervals.map(inv => (
                            <button key={inv} onClick={() => setInterval(inv)} style={{
                                background: interval === inv ? C.amber : "transparent",
                                color: interval === inv ? "#000" : C.textMid,
                                border: "none", borderRadius: 3, padding: "3px 8px", fontSize: 10, fontWeight: 700, cursor: "pointer",
                            }}>{inv}</button>
                        ))}
                    </div>
                </div>
                <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
                    {/* On-chart indicators */}
                    {onChartToggles.map(ind => (
                        <button key={ind.label} onClick={() => ind.set(!ind.active)} style={{
                            background: ind.active ? C.bg3 : "transparent",
                            color: ind.active ? C.text : C.textDim,
                            border: `1px solid ${ind.active ? C.border : "transparent"}`,
                            borderRadius: 3, padding: "3px 7px", fontSize: 10, cursor: "pointer",
                        }}>{ind.label}</button>
                    ))}
                    <div style={{ width: 1, height: 16, background: C.border, margin: "0 2px" }} />
                    {/* Sub-panel toggles */}
                    {subPanelToggles.map(ind => (
                        <button key={ind.label} onClick={() => ind.set(!ind.active)} style={{
                            background: ind.active ? (C.bg3) : "transparent",
                            color: ind.active ? C.text : C.textDim,
                            border: `1px solid ${ind.active ? C.border : "transparent"}`,
                            borderRadius: 3, padding: "3px 7px", fontSize: 10, cursor: "pointer",
                        }}>{ind.label}</button>
                    ))}
                    <div style={{ width: 1, height: 16, background: C.border, margin: "0 2px" }} />
                    <button onClick={onClose} style={{
                        background: "transparent", border: "none", color: C.textDim, cursor: "pointer",
                        fontSize: 20, lineHeight: "20px", padding: "0 4px"
                    }}>×</button>
                </div>
            </div>

            {/* Main Chart Area */}
            <div style={{ position: "relative", height: 380, borderRadius: 8, overflow: "hidden", border: `1px solid ${C.border}` }}>
                {loading && (
                    <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: `${C.bg1}99`, zIndex: 10 }}>
                        <div style={{ color: C.textDim, fontSize: 12 }}>Loading chart data...</div>
                    </div>
                )}
                {error && (
                    <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: `${C.bg1}EE`, zIndex: 10 }}>
                        <div style={{ color: C.red, fontSize: 13, background: C.red + '22', padding: "8px 16px", borderRadius: 8 }}>⚠ Error: {error}</div>
                    </div>
                )}
                <ConfluencePanel confluence={confluence} />
                <div ref={chartContainerRef} style={{ width: "100%", height: "100%", background: C.bg1 }} />
            </div>

            {/* Sub-panels */}
            {showRSI && (
                <div style={{ marginTop: 4 }}>
                    <div style={{ fontSize: 9, color: C.textDim, padding: "2px 8px", fontFamily: "'DM Mono', monospace" }}>RSI (14)</div>
                    <div ref={rsiContainerRef} style={{ height: 100, borderRadius: 6, overflow: "hidden", border: `1px solid ${C.border}` }} />
                </div>
            )}
            {showMACD && (
                <div style={{ marginTop: 4 }}>
                    <div style={{ fontSize: 9, color: C.textDim, padding: "2px 8px", fontFamily: "'DM Mono', monospace" }}>MACD (12, 26, 9)</div>
                    <div ref={macdContainerRef} style={{ height: 100, borderRadius: 6, overflow: "hidden", border: `1px solid ${C.border}` }} />
                </div>
            )}
            {showVol && (
                <div style={{ marginTop: 4 }}>
                    <div style={{ fontSize: 9, color: C.textDim, padding: "2px 8px", fontFamily: "'DM Mono', monospace" }}>Volume</div>
                    <div ref={volContainerRef} style={{ height: 80, borderRadius: 6, overflow: "hidden", border: `1px solid ${C.border}` }} />
                </div>
            )}

            {/* Legend */}
            <div style={{ marginTop: 8, display: "flex", gap: 12, fontSize: 9, color: C.textDim, justifyContent: "center", flexWrap: "wrap" }}>
                <span><b style={{ color: C.green }}>↑</b> BUY Signal</span>
                <span><b style={{ color: C.red }}>↓</b> SELL Signal</span>
                <span style={{ color: "#22d3ee" }}>◆ Candlestick Pattern</span>
                {mode === "prediction" && (
                    <>
                        <span style={{ color: C.amber }}>— Predicted Path</span>
                        <span style={{ color: C.amber + '88' }}>- - 68% CI</span>
                        <span style={{ color: C.amber + '55' }}>··· 95% CI</span>
                    </>
                )}
                <span>Scroll to zoom · Drag to pan</span>
            </div>
        </div>
    );
}
