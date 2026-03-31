import React, { useEffect, useRef, useState } from "react";
import { createChart, CandlestickSeries, LineSeries, HistogramSeries, AreaSeries, createSeriesMarkers } from "lightweight-charts";
import { fetchPrices, fetchIndicators, fetchHistoricalSignals, fetchPatterns, fetchSupportResistance, fetchSentiment, fetchConfluence } from "../utils/api";
import { C } from "../utils/data";

function parseChartTime(dateStr) {
    if (!dateStr) return null;
    if (typeof dateStr === 'number') {
        return dateStr > 10000000000 ? Math.floor(dateStr / 1000) : Math.floor(dateStr);
    }
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return null;
    return Math.floor(d.getTime() / 1000);
}

function processChartData(data) {
    const seen = new Set();
    return data.filter(d => {
        if (!d.time || seen.has(d.time)) return false;
        seen.add(d.time);
        return true;
    }).sort((a, b) => a.time - b.time);
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

// ── Decision Panel (Right Sidebar) ─────────────────────────
function DecisionPanel({ sentiment, srSummary, loading }) {
    if (loading) return (
        <div style={{
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "16px", minWidth: 190, fontFamily: "'DM Mono', monospace", fontSize: 10,
            textAlign: "center", color: C.textDim,
        }}>
            <div style={{ fontSize: 16, marginBottom: 6, animation: "pulse 1.5s infinite" }}>⏳</div>
            Computing signals…
        </div>
    );

    if (!sentiment) return (
        <div style={{
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "16px", minWidth: 190, fontFamily: "'DM Mono', monospace", fontSize: 10,
            textAlign: "center", color: C.textDim,
        }}>
            No signal data
        </div>
    );

    const { entry_signal, confidence, confidence_label, trend, rsi_signal, volume_strength } = sentiment;
    const signalColor = entry_signal === "BULLISH" ? C.green : entry_signal === "BEARISH" ? C.red : C.amber;

    const sigIcon = (sig) => {
        if (sig === "bullish" || sig === "strong") return { icon: "▲", color: C.green };
        if (sig === "bearish" || sig === "weak") return { icon: "▼", color: C.red };
        return { icon: "●", color: C.textDim };
    };

    const trendS = sigIcon(trend);
    const rsiLabel = sentiment.details?.rsi != null
        ? (sentiment.details.rsi > 70 ? "Overbought" : sentiment.details.rsi < 30 ? "Oversold" : "Neutral")
        : "N/A";
    const rsiColor = sentiment.details?.rsi > 70 ? C.red : sentiment.details?.rsi < 30 ? C.green : C.textDim;
    const volS = sigIcon(volume_strength === "strong" ? "bullish" : volume_strength === "weak" ? "bearish" : "neutral");

    return (
        <div style={{
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "14px 16px", minWidth: 190, fontFamily: "'DM Mono', monospace", fontSize: 10,
            alignSelf: "flex-start",
        }}>
            {/* Final Signal — most prominent */}
            <div style={{
                fontWeight: 900, color: signalColor, fontSize: 16, textAlign: "center",
                letterSpacing: 1.5, marginBottom: 4,
            }}>
                {entry_signal === "WAIT" ? "NEUTRAL" : entry_signal}
            </div>

            {/* Confidence bar */}
            <div style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.textDim, marginBottom: 3 }}>
                    <span>Signal Strength</span>
                    <span style={{ color: signalColor, fontWeight: 700 }}>{confidence}%</span>
                </div>
                <div style={{ background: C.border, borderRadius: 4, height: 5, overflow: "hidden" }}>
                    <div style={{
                        width: `${confidence}%`, height: "100%", borderRadius: 4,
                        background: signalColor, transition: "width 0.4s ease",
                    }} />
                </div>
                <div style={{ fontSize: 8, color: C.textDim, textAlign: "center", marginTop: 3 }}>
                    {confidence_label}
                </div>
            </div>

            {/* Divider */}
            <div style={{ height: 1, background: C.border, margin: "8px 0" }} />

            {/* Indicator rows */}
            <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0" }}>
                <span style={{ color: C.textDim }}>Trend</span>
                <span style={{ color: trendS.color, fontWeight: 600 }}>{trendS.icon} {trend === "bullish" ? "Bullish" : trend === "bearish" ? "Bearish" : "Neutral"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0" }}>
                <span style={{ color: C.textDim }}>Momentum</span>
                <span style={{ color: rsiColor, fontWeight: 600 }}>
                    {rsiLabel === "Oversold" ? "▲" : rsiLabel === "Overbought" ? "▼" : "●"} {rsiLabel}
                </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0" }}>
                <span style={{ color: C.textDim }}>Volume</span>
                <span style={{ color: volS.color, fontWeight: 600 }}>{volS.icon} {volume_strength === "strong" ? "Strong" : "Weak"}</span>
            </div>

            {/* S&R */}
            {srSummary && (
                <>
                    <div style={{ height: 1, background: C.border, margin: "8px 0" }} />
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", alignItems: "center" }}>
                        <span style={{ color: C.textDim }}>Resistance</span>
                        <div style={{ textAlign: "right" }}>
                            <div style={{ color: C.red, fontWeight: 700 }}>${srSummary.resistance}</div>
                            {sentiment?.details?.dist_resistance !== undefined && srSummary.resistance !== "N/A" && (
                                <div style={{ fontSize: 8, color: C.textDim, marginTop: 2 }}>{sentiment.details.dist_resistance}% away</div>
                            )}
                        </div>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", alignItems: "center" }}>
                        <span style={{ color: C.textDim }}>Support</span>
                        <div style={{ textAlign: "right" }}>
                            <div style={{ color: C.cyan, fontWeight: 700 }}>${srSummary.support}</div>
                            {sentiment?.details?.dist_support !== undefined && srSummary.support !== "N/A" && (
                                <div style={{ fontSize: 8, color: C.textDim, marginTop: 2 }}>{sentiment.details.dist_support}% away</div>
                            )}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

// ── Pattern Catalogue (simplified for Pattern Mode) ────────
const PATTERN_CATALOGUE = [
    { category: "Chart Patterns", patterns: [
        "Head & Shoulders", "Double Bottom",
        "Bull Flag", "Symmetrical Triangle",
    ]},
];
const MAX_PATTERNS = 4;

// ── Pattern Dropdown Component ────────────────────────────
function PatternDropdown({ selected, onToggle, onClear, onClose }) {
    const dropRef = useRef(null);

    useEffect(() => {
        const handler = (e) => { if (dropRef.current && !dropRef.current.contains(e.target)) onClose(); };
        document.addEventListener("mousedown", handler);
        return () => document.removeEventListener("mousedown", handler);
    }, [onClose]);

    const atMax = selected.size >= MAX_PATTERNS;

    return (
        <div ref={dropRef} style={{
            position: "absolute", top: "100%", right: 0, marginTop: 4, zIndex: 100,
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 10,
            padding: "12px 14px", minWidth: 240, maxHeight: 380, overflowY: "auto",
            backdropFilter: "blur(12px)", boxShadow: "0 8px 32px rgba(0,0,0,.55)",
            fontFamily: "'DM Mono', monospace",
        }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: C.text, letterSpacing: 1 }}>SELECT PATTERNS</span>
                <button onClick={onClear} style={{
                    background: "transparent", border: `1px solid ${C.border}`, borderRadius: 4,
                    color: C.textDim, fontSize: 9, padding: "2px 8px", cursor: "pointer",
                    transition: "color .2s",
                }} onMouseEnter={e => e.target.style.color = C.red}
                    onMouseLeave={e => e.target.style.color = C.textDim}>Clear All</button>
            </div>
            <div style={{ fontSize: 9, color: atMax ? C.amber : C.textDim, marginBottom: 10, transition: "color .3s" }}>
                {selected.size}/{MAX_PATTERNS} selected {atMax && "— limit reached"}
            </div>
            {PATTERN_CATALOGUE.map(cat => (
                <div key={cat.category} style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 9, color: C.textDim, fontWeight: 700, marginBottom: 4, textTransform: "uppercase", letterSpacing: .8 }}>{cat.category}</div>
                    {cat.patterns.map(p => {
                        const isSelected = selected.has(p);
                        const disabled = atMax && !isSelected;
                        return (
                            <label key={p} title={disabled ? "Max 2 patterns allowed" : ""}
                                style={{
                                    display: "flex", alignItems: "center", gap: 8, padding: "4px 6px",
                                    borderRadius: 5, cursor: disabled ? "not-allowed" : "pointer",
                                    opacity: disabled ? 0.35 : 1,
                                    background: isSelected ? C.amber + "15" : "transparent",
                                    transition: "background .2s, opacity .2s",
                                }}
                                onMouseEnter={e => { if (!disabled) e.currentTarget.style.background = isSelected ? C.amber + "22" : C.bg2; }}
                                onMouseLeave={e => { e.currentTarget.style.background = isSelected ? C.amber + "15" : "transparent"; }}>
                                <input type="checkbox" checked={isSelected} disabled={disabled}
                                    onChange={() => { if (!disabled) onToggle(p); }}
                                    style={{ accentColor: C.amber, cursor: disabled ? "not-allowed" : "pointer", width: 13, height: 13 }} />
                                <span style={{ fontSize: 11, color: isSelected ? C.text : C.textMid }}>{p}</span>
                            </label>
                        );
                    })}
                </div>
            ))}
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

// ── Mode Toggle Button ─────────────────────────────────────
function ModeButton({ label, emoji, active, disabled, onClick }) {
    return (
        <button onClick={onClick} disabled={disabled} style={{
            background: active ? (emoji === "🟢" ? C.green + "22" : emoji === "🟣" ? "#a78bfa22" : C.red + "22") : "transparent",
            color: active ? (emoji === "🟢" ? C.green : emoji === "🟣" ? "#a78bfa" : C.red) : disabled ? C.textDim + "55" : C.textDim,
            border: `1px solid ${active ? (emoji === "🟢" ? C.green + "55" : emoji === "🟣" ? "#a78bfa55" : C.red + "55") : "transparent"}`,
            borderRadius: 6, padding: "4px 12px", fontSize: 10, fontWeight: 700, cursor: disabled ? "not-allowed" : "pointer",
            transition: "all .2s", display: "flex", alignItems: "center", gap: 5,
            opacity: disabled ? 0.4 : 1,
        }}>
            <span style={{ fontSize: 8 }}>{emoji}</span> {label}
        </button>
    );
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
    const [srSummary, setSrSummary] = useState(null);

    // Sentiment data for DecisionPanel
    const [sentiment, setSentiment] = useState(null);
    const [sentimentLoading, setSentimentLoading] = useState(false);

    const [interval, setInterval] = useState("1d");
    const intervals = ["1m", "1h", "1d", "1wk", "1mo"];

    // ── View Mode ──────────────────────────────────────────
    const [viewMode, setViewMode] = useState("indicator"); // "indicator" | "pattern" | "advanced"

    // ── User Level ─────────────────────────────────────────
    const [userLevel, setUserLevel] = useState(() => {
        try { return localStorage.getItem("qv_userLevel") || "beginner"; } catch { return "beginner"; }
    });
    useEffect(() => {
        localStorage.setItem("qv_userLevel", userLevel);
        // If beginner, force indicator mode
        if (userLevel === "beginner" && viewMode !== "indicator") setViewMode("indicator");
    }, [userLevel]);

    // ── Derive toggle states from viewMode ─────────────────
    const showSR = viewMode === "indicator" || viewMode === "advanced";
    const showSMA200 = viewMode === "indicator" || viewMode === "advanced";
    const showSMA = viewMode === "advanced";
    const showEMA = viewMode === "advanced";
    const showBB = viewMode === "advanced";
    const showVWAP = viewMode === "advanced";
    const showRSI = viewMode === "indicator" || viewMode === "advanced";
    const showMACD = viewMode === "advanced";
    const showVol = viewMode === "indicator" || viewMode === "advanced";
    const showATR = viewMode === "advanced";
    const showPatterns = viewMode === "pattern" || viewMode === "advanced";
    const showMLSignals = viewMode === "advanced";

    // Pattern selection layer
    const [selectedPatterns, setSelectedPatterns] = useState(() => {
        try {
            const saved = localStorage.getItem("qv_selectedPatterns");
            return saved ? new Set(JSON.parse(saved)) : new Set();
        } catch { return new Set(); }
    });
    const [patternDropdownOpen, setPatternDropdownOpen] = useState(false);

    useEffect(() => {
        localStorage.setItem("qv_selectedPatterns", JSON.stringify([...selectedPatterns]));
    }, [selectedPatterns]);

    const togglePattern = (name) => {
        setSelectedPatterns(prev => {
            const next = new Set(prev);
            if (next.has(name)) next.delete(name);
            else if (next.size < MAX_PATTERNS) next.add(name);
            return next;
        });
    };
    const clearPatterns = () => setSelectedPatterns(new Set());

    const stateData = useRef({ ohlc: [], indicators: [], signals: [], patterns: null, srData: null });

    // Fetch confluence independent of main timeframe API
    const [confluence, setConfluence] = useState([]);
    useEffect(() => {
        let active = true;
        fetchConfluence(symbol).then(d => {
            if (active && d?.confluence_signals) setConfluence(d.confluence_signals);
        }).catch(() => {});
        return () => { active = false; };
    }, [symbol]);

    // ── Fetch sentiment for DecisionPanel ──────────────────
    useEffect(() => {
        setSentimentLoading(true);
        import("../utils/api").then(api => {
            api.fetchSentiment(symbol).then(d => { setSentiment(d); setSentimentLoading(false); })
               .catch(() => { setSentiment(null); setSentimentLoading(false); });
        });
    }, [symbol]);

    useEffect(() => {
        let cancelled = false;

        async function loadData() {
            setLoading(true); setError(null);
            try {
                let days = interval.includes("m") ? 5 : (interval === "1h" ? 20 : (interval === "4h" ? 60 : 120));
                const lookback = Math.max(days, 60);

                const [priceRes, indRes, patRes, srRes] = await Promise.all([
                    fetchPrices(symbol, "yfinance", days, interval),
                    fetchIndicators(symbol, days, interval),
                    fetchPatterns(symbol, interval),
                    showSR ? fetchSupportResistance(symbol, interval, lookback) : Promise.resolve(null),
                ].map(p => p.catch(e => null)));

                if (cancelled) return;
                if (!priceRes || !priceRes.bars) throw new Error("Failed to fetch price data");

                let sigRes = [];
                if (interval === "1d" && showMLSignals) {
                    try { sigRes = await fetchHistoricalSignals(symbol, 90, "xgboost"); } catch (e) { }
                }
                if (cancelled) return;

                stateData.current = {
                    ohlc: priceRes.bars,
                    indicators: indRes?.data || [],
                    signals: sigRes || [],
                    patterns: patRes?.patterns || null,
                    srData: srRes || null,
                };

                if (srRes?.levels) {
                    const sup = srRes.levels.filter(l => l.type === "support").sort((a,b)=>b.price - a.price);
                    const res = srRes.levels.filter(l => l.type === "resistance").sort((a,b)=>a.price - b.price);
                    setSrSummary({
                         support: sup[0] ? sup[0].price.toFixed(2) : "N/A",
                         resistance: res[0] ? res[0].price.toFixed(2) : "N/A"
                    });
                } else {
                    setSrSummary(null);
                }

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

            const { ohlc, indicators, signals, patterns, srData } = stateData.current;
            const ohlcData = processChartData(ohlc.map(b => ({
                time: parseChartTime(b.date),
                open: b.open, high: b.high, low: b.low, close: b.close
            })));
            candleSeries.setData(ohlcData);

            // ── Indicator Map ──────────────────────────────
            const indMap = {};
            indicators.forEach(i => indMap[parseChartTime(i.date)] = i);

            // ── On-Chart Overlays (mode-dependent) ─────────
            // 200 SMA (Indicator + Advanced)
            if (showSMA200) {
                const sma200 = chart.addSeries(LineSeries, { color: "#f59e0b", lineWidth: 2, lineStyle: 0 });
                sma200.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.SMA_200 })).filter(d => d.value != null));
            }

            // SMA 20 (Advanced only)
            if (showSMA) {
                const s = chart.addSeries(LineSeries, { color: C.cyan, lineWidth: 1, lineStyle: 2 });
                s.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.SMA_20 })).filter(d => d.value != null));
            }

            // EMA 12/26 (Advanced only)
            if (showEMA) {
                const e12 = chart.addSeries(LineSeries, { color: "#4ade80", lineWidth: 1 });
                e12.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.EMA_12 })).filter(d => d.value != null));
                const e26 = chart.addSeries(LineSeries, { color: "#f472b6", lineWidth: 1 });
                e26.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.EMA_26 })).filter(d => d.value != null));
            }

            // Bollinger Bands (Advanced only)
            if (showBB) {
                const bbUp = chart.addSeries(LineSeries, { color: C.purple, lineWidth: 1, lineStyle: 3 });
                const bbMid = chart.addSeries(LineSeries, { color: C.purple + '88', lineWidth: 1, lineStyle: 2 });
                const bbLow = chart.addSeries(LineSeries, { color: C.purple, lineWidth: 1, lineStyle: 3 });
                bbUp.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.BB_High })).filter(d => d.value != null));
                bbMid.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.BB_Mid })).filter(d => d.value != null));
                bbLow.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.BB_Low })).filter(d => d.value != null));
            }

            // VWAP — intraday only (Advanced)
            const isIntraday = interval.includes("m") || interval === "1h" || interval === "4h";
            if (showVWAP && isIntraday) {
                const vwap = chart.addSeries(LineSeries, { color: "#f59e0b", lineWidth: 2, lineStyle: 0 });
                vwap.setData(ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.VWAP })).filter(d => d.value != null));
            }

            // ── Markers ────────────────────────────────────
            const allMarkers = [];

            // ML signals (Advanced only)
            if (showMLSignals && signals && signals.length > 0) {
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

            // Chart patterns (Markers)
            if (showPatterns && patterns && selectedPatterns.size > 0) {
                patterns.filter(p => selectedPatterns.has(p.pattern_name)).forEach(p => {
                    const isBullish = p.direction === "bullish";
                    const isNeutral = p.direction === "neutral";
                    
                    // Check if highly confluent
                    const isConfluent = confluence.some(c => c.pattern_name === p.pattern_name && c.direction === p.direction);

                    allMarkers.push({
                        time: parseChartTime(p.end_date),
                        position: isNeutral ? 'aboveBar' : isBullish ? 'belowBar' : 'aboveBar',
                        color: isNeutral ? C.textDim : isBullish ? "#22d3ee" : "#fb923c",
                        shape: isNeutral ? 'circle' : isBullish ? 'arrowUp' : 'arrowDown',
                        text: `${isConfluent ? "🔥 " : ""}[${p.timeframe}-${p.weight}] ${p.pattern_name}`
                    });
                });
            }

            // S&R dynamic MA layers removed

            // Filter, deduplicate, sort markers
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

            // ── Chart Pattern Trendlines & Draw ────────────────────
            if (showPatterns && patterns && selectedPatterns.size > 0) {
                patterns.filter(cp => selectedPatterns.has(cp.pattern_name)).forEach(cp => {
                    const isBullish = cp.direction === "bullish";
                    
                    // Draw continuous key_levels if it's head & shoulders or double bottom
                    if (["Head & Shoulders", "Double Bottom"].includes(cp.pattern_name) && cp.key_levels?.length >= 2) {
                        const lineColor = isBullish ? C.green + 'AA' : C.red + 'AA';
                        const trendSeries = chart.addSeries(LineSeries, {
                            color: lineColor, lineWidth: 2, lineStyle: 0,
                            lastValueVisible: false, priceLineVisible: false,
                        });
                        const points = processChartData(cp.key_levels.map(kl => ({
                            time: parseChartTime(kl.date), value: kl.price
                        })));
                        trendSeries.setData(points);
                    }
                    
                    // Draw trendlines channels if they exist
                    if (cp.trendlines && cp.trendlines.length > 0) {
                        cp.trendlines.forEach(tl => {
                            if (tl.length >= 2) {
                                const tlSeries = chart.addSeries(LineSeries, {
                                    color: C.textMid + '88', lineWidth: 1, lineStyle: 2,
                                    lastValueVisible: false, priceLineVisible: false,
                                });
                                const points = processChartData(tl.map(kl => ({
                                    time: parseChartTime(kl.date), value: kl.price
                                })));
                                tlSeries.setData(points);
                            }
                        });
                    }

                    // Only draw action levels if the pattern is confirmed, OR if it's not a symmetrical triangle
                    if (cp.status === "confirmed" || cp.pattern_name !== "Symmetrical Triangle") {
                        if (cp.neckline) {
                            candleSeries.createPriceLine({
                                price: cp.neckline, color: C.amber + '88', lineWidth: 1, lineStyle: 2,
                                axisLabelVisible: true, title: cp.pattern_name + " Neckline",
                            });
                        }

                        if (cp.breakout_price) {
                            candleSeries.createPriceLine({
                                price: cp.breakout_price, color: "#fbbf24", lineWidth: 1, lineStyle: 3,
                                axisLabelVisible: true, title: "Entry / Breakout",
                            });
                        }

                        if (cp.target_price) {
                            candleSeries.createPriceLine({
                                price: cp.target_price,
                                color: C.green + 'CC',
                                lineWidth: 1, lineStyle: 4, axisLabelVisible: true,
                                title: `Target: $${cp.target_price.toFixed(2)}`,
                            });
                        }

                        if (cp.stop_loss !== null && cp.stop_loss !== undefined) {
                            candleSeries.createPriceLine({
                                price: cp.stop_loss,
                                color: C.red + 'CC',
                                lineWidth: 1, lineStyle: 4, axisLabelVisible: true,
                                title: `Stop: $${cp.stop_loss.toFixed(2)}`,
                            });
                        }
                    } else if (cp.pattern_name === "Symmetrical Triangle" && cp.status !== "confirmed") {
                        // Unconfirmed Symmetrical triangle label (the trendlines are drawn above already)
                        // Awaiting breakout - no target/stop/entry
                        const labelSeries = chart.addSeries(LineSeries, { lastValueVisible: false, priceLineVisible: false });
                        labelSeries.setMarkers([{
                            time: parseChartTime(cp.end_date), position: 'inBar', color: C.textDim,
                            shape: 'circle', text: "Symmetrical Triangle — Awaiting Breakout"
                        }]);
                    }
                });
            }

            // ── S&R Horizontal Lines ────────────────────────
            if (showSR && srData?.levels && srData.levels.length > 0) {
                const supports = srData.levels.filter(l => l.type === "support").sort((a, b) => b.price - a.price);
                const resistances = srData.levels.filter(l => l.type === "resistance").sort((a, b) => a.price - b.price);
                const keyLevels = [...supports.slice(0, 1), ...resistances.slice(0, 1)];

                keyLevels.forEach(sr => {
                    const isSupp = sr.type === "support";
                    candleSeries.createPriceLine({
                        price: sr.price,
                        color: isSupp ? C.cyan : C.red,
                        lineWidth: 2, lineStyle: 0, axisLabelVisible: false,
                        title: `${isSupp ? "Key Support" : "Key Resistance"} — $${sr.price.toFixed(2)}`,
                    });
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
                predSeries.setData(processChartData(dP)); u95.setData(processChartData(dU95)); l95.setData(processChartData(dL95)); u68.setData(processChartData(dU68)); l68.setData(processChartData(dL68));
            }

            chart.timeScale().fitContent();
        }

        function renderSubPanels() {
            const { ohlc, indicators } = stateData.current;
            const indMap = {};
            indicators.forEach(i => indMap[parseChartTime(i.date)] = i);
            const ohlcData = processChartData(ohlc.map(b => ({ time: parseChartTime(b.date), close: b.close, open: b.open, volume: b.volume })));

            // ── RSI Sub-panel (Indicator + Advanced) ────────
            if (showRSI && rsiContainerRef.current) {
                const rsiChart = createSubChart(rsiContainerRef.current, interval, 100);
                if (rsiChart) {
                    subChartsRef.current.rsi = rsiChart;
                    const rsiSeries = rsiChart.addSeries(LineSeries, { color: "#a78bfa", lineWidth: 1.5 });
                    const rsiData = ohlcData.map(b => ({ time: b.time, value: indMap[b.time]?.RSI })).filter(d => d.value != null);
                    rsiSeries.setData(rsiData);
                    rsiSeries.createPriceLine({ price: 70, color: C.red + '88', lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: "" });
                    rsiSeries.createPriceLine({ price: 30, color: C.green + '88', lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: "" });
                    rsiSeries.createPriceLine({ price: 50, color: C.textDim + '44', lineWidth: 1, lineStyle: 3, axisLabelVisible: false, title: "" });
                    rsiChart.timeScale().fitContent();
                }
            }

            // ── MACD Sub-panel (Advanced only) ──────────────
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

            // ── Volume Sub-panel (Indicator + Advanced) ─────
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
    }, [symbol, interval, viewMode, mode, predictionData, selectedPatterns]);

    return (
        <div className="fade-up" style={{ display: "flex", flexDirection: "column", width: "100%" }}>
            {/* Chart + Sidebar Layout */}
            <div style={{ display: "flex", gap: 12, alignItems: "stretch" }}>
                {/* Left side: Toolbar + Charts */}
                <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
                    {/* Toolbar */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, background: C.bg2, padding: "6px 14px", borderRadius: 8, border: `1px solid ${C.border}`, flexWrap: "wrap", gap: 8 }}>
                        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                            <div style={{ fontWeight: 800, color: C.text, fontSize: 16, marginRight: 6 }}>{symbol}</div>
                            {/* Interval selector */}
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

                        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                            {/* Mode Toggle */}
                            <div style={{ display: "flex", gap: 4, background: C.bg0, padding: 3, borderRadius: 6 }}>
                                <ModeButton label="Indicator" emoji="🟢" active={viewMode === "indicator"} onClick={() => setViewMode("indicator")} />
                                <ModeButton label="Pattern" emoji="🟣" active={viewMode === "pattern"} disabled={userLevel === "beginner"} onClick={() => { if (userLevel !== "beginner") setViewMode("pattern"); }} />
                                <ModeButton label="Advanced" emoji="🔴" active={viewMode === "advanced"} disabled={userLevel === "beginner"} onClick={() => { if (userLevel !== "beginner") setViewMode("advanced"); }} />
                            </div>

                            {/* Pattern selector (Pattern/Advanced mode only) */}
                            {showPatterns && (
                                <>
                                    <div style={{ width: 1, height: 16, background: C.border, margin: "0 2px" }} />
                                    <div style={{ position: "relative" }}>
                                        <button onClick={() => setPatternDropdownOpen(v => !v)} style={{
                                            background: patternDropdownOpen ? C.amber + '33' : (selectedPatterns.size > 0 ? C.bg3 : "transparent"),
                                            color: selectedPatterns.size > 0 ? C.amber : C.textDim,
                                            border: `1px solid ${selectedPatterns.size > 0 ? C.amber + '55' : C.border}`,
                                            borderRadius: 3, padding: "3px 10px", fontSize: 10, cursor: "pointer",
                                            fontWeight: 700, display: "flex", alignItems: "center", gap: 5,
                                            transition: "all .2s",
                                        }}>
                                            <span>Patterns</span>
                                            <span style={{
                                                background: selectedPatterns.size > 0 ? C.amber : C.textDim,
                                                color: C.bg0, borderRadius: 8, padding: "0 5px", fontSize: 9, fontWeight: 800,
                                                minWidth: 18, textAlign: "center",
                                            }}>{selectedPatterns.size}/{MAX_PATTERNS}</span>
                                        </button>
                                        {patternDropdownOpen && (
                                            <PatternDropdown
                                                selected={selectedPatterns}
                                                onToggle={togglePattern}
                                                onClear={clearPatterns}
                                                onClose={() => setPatternDropdownOpen(false)}
                                            />
                                        )}
                                    </div>
                                </>
                            )}

                            {/* User Level Toggle */}
                            <div style={{ width: 1, height: 16, background: C.border, margin: "0 2px" }} />
                            <button onClick={() => setUserLevel(prev => prev === "beginner" ? "advanced" : "beginner")} style={{
                                background: userLevel === "advanced" ? C.amber + "15" : "transparent",
                                color: userLevel === "advanced" ? C.amber : C.textDim,
                                border: `1px solid ${userLevel === "advanced" ? C.amber + "44" : C.border}`,
                                borderRadius: 6, padding: "4px 10px", fontSize: 10, fontWeight: 700, cursor: "pointer",
                                transition: "all .2s", display: "flex", alignItems: "center", gap: 4,
                            }}>
                                {userLevel === "beginner" ? "👤 Beginner" : "⚡ Advanced"}
                            </button>

                            {/* Close */}
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
                </div>

                {/* Right side: Decision Panel */}
                <div style={{ width: 220, flexShrink: 0 }}>
                    <DecisionPanel sentiment={sentiment} srSummary={showSR ? srSummary : null} loading={sentimentLoading} />
                </div>
            </div>

            {/* Legend */}
            <div style={{ marginTop: 8, display: "flex", gap: 12, fontSize: 9, color: C.textDim, justifyContent: "center", flexWrap: "wrap" }}>
                {showSR && <span style={{ color: C.cyan }}>— Support</span>}
                {showSR && <span style={{ color: C.red }}>— Resistance</span>}
                {showSMA200 && <span style={{ color: "#f59e0b" }}>— 200 MA</span>}
                {showMLSignals && <span><b style={{ color: C.green }}>↑</b> BUY Signal</span>}
                {showMLSignals && <span><b style={{ color: C.red }}>↓</b> SELL Signal</span>}
                {showPatterns && selectedPatterns.size > 0 && <span style={{ color: "#22d3ee" }}>◆ Pattern ({[...selectedPatterns].join(", ")})</span>}
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
