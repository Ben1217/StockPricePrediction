import React, { useEffect, useRef, useState } from "react";
import { createChart, CandlestickSeries, LineSeries, HistogramSeries, AreaSeries, createSeriesMarkers } from "lightweight-charts";
import { fetchPrices, fetchIndicators, fetchHistoricalSignals, fetchPatterns, fetchSupportResistance, fetchConfluence } from "../utils/api";
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
function formatSetupPrice(value) {
    return value == null ? "—" : `$${Number(value).toFixed(2)}`;
}

function isActionableSetup(pattern) {
    return Boolean(
        pattern?.direction &&
        pattern.direction !== "neutral" &&
        pattern.entry_price != null &&
        pattern.target_price != null &&
        pattern.stop_loss != null &&
        pattern.setup_relevance_ok !== false
    );
}

function getSetupTone(pattern) {
    if (pattern?.direction === "bullish") return { color: C.green, label: "Bullish" };
    if (pattern?.direction === "bearish") return { color: C.red, label: "Bearish" };
    return { color: C.amber, label: "Neutral" };
}

function getSetupAction(pattern) {
    if (!pattern) return "No setup";
    if (pattern.direction === "bullish" && pattern.entry_price != null) {
        return `Buy above ${Number(pattern.entry_price).toFixed(2)}`;
    }
    if (pattern.direction === "bearish" && pattern.entry_price != null) {
        return `Sell below ${Number(pattern.entry_price).toFixed(2)}`;
    }
    return "Wait for breakout";
}

function getSetupRiskReward(pattern) {
    if (!isActionableSetup(pattern)) return null;
    const risk = Math.abs(Number(pattern.entry_price) - Number(pattern.stop_loss));
    const reward = Math.abs(Number(pattern.target_price) - Number(pattern.entry_price));
    if (!risk) return null;
    return reward / risk;
}

function getPatternConfidence(pattern) {
    const value = Number(pattern?.confidence ?? pattern?.confidence_score);
    return Number.isFinite(value) ? value : null;
}

function getPatternRiskReward(pattern) {
    const direct = Number(pattern?.risk_reward_ratio);
    if (Number.isFinite(direct)) return direct;
    return getSetupRiskReward(pattern);
}

function getPatternDirectionLabel(direction) {
    if (direction === "bullish") return "Bullish";
    if (direction === "bearish") return "Bearish";
    return "Neutral";
}

function getPatternRejectReason(pattern, setupStatus) {
    if (!pattern) return "No pattern detected";
    if (pattern.status === "broken" || pattern.pattern_status === "broken") return "Pattern already broken";
    if (!isActionableSetup(pattern)) return "Entry/stop/target not valid";

    const confidence = getPatternConfidence(pattern);
    const minConfidence = Number(setupStatus?.min_confidence ?? 70);
    if (confidence != null && confidence < minConfidence) return "Pattern confidence below threshold";

    const riskReward = getPatternRiskReward(pattern);
    const minRiskReward = Number(setupStatus?.min_risk_reward ?? 1.5);
    if (riskReward == null || riskReward < minRiskReward) return "Risk/reward too weak";

    if (Array.isArray(setupStatus?.conflicting_pattern_names) && setupStatus.conflicting_pattern_names.includes(pattern.pattern_name)) {
        return "Conflicting pattern signals";
    }

    if (setupStatus?.candidate_pattern_name === pattern.pattern_name && setupStatus?.reason && setupStatus.reason_code !== "VALID_SETUP") {
        return setupStatus.reason;
    }

    return "Lower-ranked than best setup";
}

function formatSetupPercent(value, signed = true) {
    if (value == null) return "--";
    const number = Number(value);
    const prefix = signed && number > 0 ? "+" : "";
    return `${prefix}${number.toFixed(1)}%`;
}

function getReasonTone(reasonCode) {
    if (reasonCode === "INSUFFICIENT_DATA") return C.amber;
    if (reasonCode === "SETUP_COMPLETED" || reasonCode === "SETUP_STALE") return C.amber;
    if (reasonCode === "VALID_SETUP") return C.green;
    return C.red;
}

function getStrengthTone(strengthLabel) {
    if (strengthLabel === "Strong") return C.green;
    if (strengthLabel === "Moderate") return C.amber;
    return C.red;
}

function formatIndicatorVolume(value) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    const volume = Number(value);
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M shares`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K shares`;
    return `${Math.round(volume)} shares`;
}

function getIndicatorActionTone(action) {
    if (action === "BUY") return { color: C.green, label: "BUY" };
    if (action === "SELL") return { color: C.red, label: "SELL" };
    return { color: C.amber, label: "WAIT" };
}

function getIndicatorRsiLabel(rsiValue) {
    if (rsiValue == null || Number.isNaN(Number(rsiValue))) return "Unavailable";
    const rsi = Number(rsiValue);
    if (rsi >= 70) return "Overbought";
    if (rsi <= 30) return "Oversold";
    return "Neutral";
}

function getIndicatorAtrLabel(atrValue, closeValue) {
    if (atrValue == null || closeValue == null || !Number(closeValue)) return "Volatility unavailable";
    const atrPct = (Number(atrValue) / Number(closeValue)) * 100;
    if (atrPct >= 4) return "High Volatility";
    if (atrPct >= 2) return "Moderate Volatility";
    return "Low Volatility";
}

function getPrimaryLevels(srData) {
    const levels = Array.isArray(srData?.levels) ? srData.levels : [];
    const supports = levels.filter(level => level.type === "support").sort((a, b) => b.price - a.price);
    const resistances = levels.filter(level => level.type === "resistance").sort((a, b) => a.price - b.price);
    return {
        support: supports[0]?.price ?? null,
        resistance: resistances[0]?.price ?? null,
    };
}

function buildIndicatorSummary({ indicators, prices, srData, interval }) {
    const latestIndicator = Array.isArray(indicators) && indicators.length > 0 ? indicators[indicators.length - 1] : null;
    const latestPrice = Array.isArray(prices) && prices.length > 0 ? prices[prices.length - 1] : null;
    const srLevels = getPrimaryLevels(srData);

    const close = latestPrice?.close ?? null;
    const rsi = latestIndicator?.RSI ?? null;
    const atr = latestIndicator?.ATR ?? null;
    const volume = latestPrice?.volume ?? null;
    const avgVolume = latestIndicator?.Volume_SMA_20 ?? null;
    const support = srLevels.support;
    const resistance = srLevels.resistance;

    let trend = null;
    if (close != null && latestIndicator?.SMA_200 != null) {
        trend = Number(close) >= Number(latestIndicator.SMA_200) ? "bullish" : "bearish";
    }

    let volumeStrength = null;
    if (volume != null && avgVolume != null) {
        volumeStrength = Number(volume) >= Number(avgVolume) ? "strong" : "weak";
    }

    const trendScore = trend === "bullish" ? 1 : trend === "bearish" ? -1 : 0;
    const rsiScore = rsi != null && rsi <= 30 ? 1 : rsi != null && rsi >= 70 ? -1 : 0;
    const volumeScore = volumeStrength === "strong" ? (trend === "bearish" ? -1 : 1) : 0;
    const score = trendScore + rsiScore + volumeScore;
    const action = score >= 2 ? "BUY" : score <= -2 ? "SELL" : "WAIT";
    const sentimentLabel = score >= 2 ? "Bullish" : score <= -2 ? "Bearish" : "Neutral";
    const confidence = Math.min(95, Math.max(35, 40 + (Math.abs(score) * 20)));

    return {
        mode: "INDICATOR",
        timeframe: interval,
        trend: trend || "neutral",
        rsi,
        volume,
        avgVolume,
        atr,
        support,
        resistance,
        action,
        confidence,
        sentimentLabel,
        volumeStrength: volumeStrength || "weak",
        close,
    };
}

function IndicatorSummaryPanel({ summary, loading, error }) {
    if (loading) return (
        <div style={{
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "16px", minWidth: 280, fontFamily: "'DM Mono', monospace", fontSize: 10,
            textAlign: "center", color: C.textDim,
        }}>
            <div style={{ fontSize: 16, marginBottom: 6, animation: "pulse 1.5s infinite" }}>...</div>
            Computing indicator summary...
        </div>
    );

    if (error) return (
        <div style={{
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "16px", minWidth: 280, fontFamily: "'DM Mono', monospace", fontSize: 10,
            alignSelf: "flex-start",
        }}>
            <div style={{
                color: C.textDim, fontSize: 9, letterSpacing: 1.4, marginBottom: 10,
                fontWeight: 800, textTransform: "uppercase",
            }}>
                Indicator Summary
            </div>
            <div style={{
                background: C.red + "14",
                border: `1px solid ${C.red}33`,
                borderRadius: 8,
                padding: "12px 14px",
                color: C.red,
                fontSize: 12,
                fontWeight: 700,
                lineHeight: 1.5,
            }}>
                Indicator summary unavailable: {error}
            </div>
        </div>
    );

    if (!summary) return (
        <div style={{
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "16px", minWidth: 280, fontFamily: "'DM Mono', monospace", fontSize: 10,
            alignSelf: "flex-start",
        }}>
            <div style={{
                color: C.textDim, fontSize: 9, letterSpacing: 1.4, marginBottom: 10,
                fontWeight: 800, textTransform: "uppercase",
            }}>
                Indicator Summary
            </div>
            <div style={{
                background: C.amber + "14",
                border: `1px solid ${C.amber}33`,
                borderRadius: 8,
                padding: "12px 14px",
                color: C.text,
                lineHeight: 1.5,
            }}>
                Indicator data is still loading.
            </div>
        </div>
    );

    const actionTone = getIndicatorActionTone(summary.action);
    const trendTone = summary.trend === "bullish" ? C.green : summary.trend === "bearish" ? C.red : C.amber;
    const rsiLabel = getIndicatorRsiLabel(summary.rsi);
    const atrLabel = getIndicatorAtrLabel(summary.atr, summary.close);
    const confidenceText = summary.confidence != null ? `${Number(summary.confidence).toFixed(0)}%` : "--";
    const supportText = summary.support != null ? `$${Number(summary.support).toFixed(2)}` : "--";
    const resistanceText = summary.resistance != null ? `$${Number(summary.resistance).toFixed(2)}` : "--";
    const trendLabel = summary.trend ? `${summary.trend.charAt(0).toUpperCase()}${summary.trend.slice(1)}` : "Neutral";

    return (
        <div style={{
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "16px", minWidth: 280, fontFamily: "'DM Mono', monospace", fontSize: 10,
            alignSelf: "flex-start",
        }}>
            <div style={{
                color: C.textDim, fontSize: 9, letterSpacing: 1.4, marginBottom: 10,
                fontWeight: 800, textTransform: "uppercase",
            }}>
                Indicator Summary
            </div>

            <div style={{
                background: actionTone.color + "14",
                border: `1px solid ${actionTone.color}33`,
                borderRadius: 10,
                padding: "14px 16px",
                marginBottom: 14,
            }}>
                <div style={{ color: C.textDim, fontSize: 9, textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 8 }}>
                    Action
                </div>
                <div style={{ color: actionTone.color, fontSize: 28, fontWeight: 900, lineHeight: 1, marginBottom: 8 }}>
                    {actionTone.label}
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, color: C.textMid, fontSize: 9 }}>
                    <span>{summary.sentimentLabel || "Indicator decision"}</span>
                    <span>{summary.timeframe}</span>
                </div>
            </div>

            <div style={{ display: "grid", gap: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>Trend</span>
                    <span style={{ color: trendTone, fontWeight: 700 }}>{trendLabel}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>RSI</span>
                    <span style={{ color: C.text, fontWeight: 700 }}>
                        {summary.rsi != null ? Number(summary.rsi).toFixed(1) : "--"} {"->"} {rsiLabel}
                    </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>Volume</span>
                    <span style={{ color: C.text, fontWeight: 700 }}>
                        {formatIndicatorVolume(summary.volume)} {"->"} {summary.volumeStrength === "strong" ? "Strong" : "Weak"}
                    </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>ATR</span>
                    <span style={{ color: C.text, fontWeight: 700 }}>
                        {summary.atr != null ? Number(summary.atr).toFixed(2) : "--"} {"->"} {atrLabel}
                    </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>Support</span>
                    <span style={{ color: C.cyan, fontWeight: 700 }}>{supportText}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>Resistance</span>
                    <span style={{ color: C.red, fontWeight: 700 }}>{resistanceText}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0" }}>
                    <span style={{ color: C.textDim }}>Confidence</span>
                    <span style={{ color: actionTone.color, fontWeight: 700 }}>{confidenceText}</span>
                </div>
            </div>
        </div>
    );
}

function DecisionCheckRow({ label, passed }) {
    const color = passed ? C.green : C.red;
    return (
        <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
            <span style={{ color: C.textDim }}>{label}</span>
            <span style={{ color, fontWeight: 700 }}>{passed ? "Pass" : "Fail"}</span>
        </div>
    );
}

function PatternDetailsSection({ patterns, bestSetup, setupStatus, advancedMode }) {
    const rows = (patterns || []).slice(0, advancedMode ? 10 : 6);
    const components = bestSetup?.score_components;
    if (!rows.length && !components) return null;

    return (
        <div style={{ marginTop: 12, display: "grid", gap: 10 }}>
            {components && (
                <div style={{ border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px", background: C.bg1 }}>
                    <div style={{ color: C.textDim, fontSize: 9, letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 8 }}>
                        Confidence Calculation
                    </div>
                    {[
                        ["Confidence", bestSetup.confidence_score != null ? `${Number(bestSetup.confidence_score).toFixed(0)}%` : "--"],
                        ["Risk / reward score", components.risk_reward_score != null ? Number(components.risk_reward_score).toFixed(2) : "--"],
                        ["Trend confirmation", (components.trend_confirmation ?? components.indicator_alignment) != null ? Number(components.trend_confirmation ?? components.indicator_alignment).toFixed(2) : "--"],
                        ["Volume confirmation", components.volume_confirmation != null ? Number(components.volume_confirmation).toFixed(2) : "--"],
                        ["S/R confirmation", components.support_resistance_confirmation != null ? Number(components.support_resistance_confirmation).toFixed(2) : "--"],
                        ["Conflict penalty", components.conflict_penalty != null ? Number(components.conflict_penalty).toFixed(2) : "--"],
                    ].map(([label, value]) => (
                        <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: 9 }}>
                            <span style={{ color: C.textDim }}>{label}</span>
                            <span style={{ color: C.text, fontWeight: 700 }}>{value}</span>
                        </div>
                    ))}
                </div>
            )}

            {rows.length > 0 && (
                <div style={{ border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px", background: C.bg1 }}>
                    <div style={{ color: C.textDim, fontSize: 9, letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 8 }}>
                        Detected Patterns
                    </div>
                    <div style={{ display: "grid", gap: 8 }}>
                        {rows.map((pattern, index) => {
                            const isBest = Boolean(bestSetup && pattern.pattern_name === bestSetup.pattern_name && pattern.direction === bestSetup.direction);
                            const confidence = getPatternConfidence(pattern);
                            const riskReward = getPatternRiskReward(pattern);
                            const tone = isBest ? getSetupTone(bestSetup).color : C.textDim;
                            return (
                                <div key={`${pattern.pattern_name}-${pattern.direction}-${pattern.end_date}-${index}`} style={{
                                    border: `1px solid ${isBest ? tone + "55" : C.border}`,
                                    borderRadius: 8,
                                    padding: "8px 9px",
                                    background: isBest ? tone + "10" : "transparent",
                                }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: 4 }}>
                                        <span style={{ color: isBest ? tone : C.text, fontWeight: 800 }}>{pattern.pattern_name}</span>
                                        <span style={{ color: isBest ? tone : C.amber, fontWeight: 800 }}>
                                            {isBest ? "Valid Setup" : "Rejected Setup"}
                                        </span>
                                    </div>
                                    <div style={{ color: C.textDim, fontSize: 9, lineHeight: 1.5 }}>
                                        {getPatternDirectionLabel(pattern.direction)}
                                        {confidence != null ? ` | ${confidence.toFixed(0)}%` : ""}
                                        {riskReward != null ? ` | R/R ${riskReward.toFixed(2)}` : ""}
                                    </div>
                                    {!isBest && (
                                        <div style={{ color: C.textMid, fontSize: 9, lineHeight: 1.45, marginTop: 3 }}>
                                            {getPatternRejectReason(pattern, setupStatus)}
                                        </div>
                                    )}
                                    {advancedMode && (
                                        <div style={{ color: C.textDim, fontSize: 9, lineHeight: 1.45, marginTop: 4 }}>
                                            Entry {formatSetupPrice(pattern.entry_price)} | Stop {formatSetupPrice(pattern.stop_loss)} | Target {formatSetupPrice(pattern.target_price)}
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}

function TradeSetupPanel({ bestSetup, setupStatus, alternativeCount, loading, patterns = [], showDetails = false, advancedMode = false }) {
    const [showTargets, setShowTargets] = useState(false);
    if (loading) return (
        <div style={{
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "16px", minWidth: 280, fontFamily: "'DM Mono', monospace", fontSize: 10,
            textAlign: "center", color: C.textDim,
        }}>
            <div style={{ fontSize: 16, marginBottom: 6, animation: "pulse 1.5s infinite" }}>⏳</div>
            Ranking setups…
        </div>
    );

    const checks = setupStatus ? [
        { label: "Pattern detected", passed: setupStatus.has_detected_pattern },
        { label: `Confidence >= ${Number(setupStatus.min_confidence || 0).toFixed(0)}%`, passed: setupStatus.confidence_ok },
        { label: "Valid entry / stop / target", passed: setupStatus.levels_ok },
        { label: "Relevant to current price", passed: setupStatus.price_relevance_ok !== false },
        { label: `Risk / reward >= ${Number(setupStatus.min_risk_reward || 0).toFixed(1)}`, passed: setupStatus.risk_reward_ok },
        { label: "No conflicting filters", passed: setupStatus.no_conflicting_filters },
        { label: `Sufficient candles (${setupStatus.candle_count || 0}/${setupStatus.min_candles || 0})`, passed: setupStatus.sufficient_data },
    ] : [];

    if (!bestSetup) {
        const reasonTone = getReasonTone(setupStatus?.reason_code);
        return (
            <div style={{
                background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
                padding: "16px", minWidth: 280, fontFamily: "'DM Mono', monospace", fontSize: 10,
                alignSelf: "flex-start",
            }}>
                <div style={{
                    color: C.textDim, fontSize: 9, letterSpacing: 1.4, marginBottom: 10,
                    fontWeight: 800, textTransform: "uppercase",
                }}>
                    Best Setup Status
                </div>

                <div style={{
                    background: reasonTone + "14",
                    border: `1px solid ${reasonTone}33`,
                    borderRadius: 8,
                    padding: "12px 14px",
                    marginBottom: 12,
                }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 8 }}>
                        <div style={{ color: C.textDim, fontSize: 9, textTransform: "uppercase", letterSpacing: 1.2 }}>
                            No Clear Setup
                        </div>
                        <div style={{
                            color: reasonTone,
                            background: reasonTone + "18",
                            border: `1px solid ${reasonTone}33`,
                            borderRadius: 999,
                            padding: "3px 8px",
                            fontSize: 9,
                            fontWeight: 800,
                        }}>
                            {setupStatus?.reason_code || "NO_PATTERN"}
                        </div>
                    </div>
                    <div style={{ color: reasonTone, fontSize: 14, fontWeight: 900, lineHeight: 1.35 }}>
                        No clear trade setup right now
                    </div>
                    <div style={{ color: C.textMid, fontSize: 10, lineHeight: 1.5, marginTop: 6 }}>
                        Reason: {setupStatus?.reason || "No pattern detected"}
                    </div>
                    {setupStatus?.candidate_pattern_name && (
                        <div style={{ color: C.textDim, fontSize: 9, marginTop: 8, lineHeight: 1.5 }}>
                            Candidate: {setupStatus.candidate_pattern_name}
                            {setupStatus.candidate_confidence != null ? ` • ${Number(setupStatus.candidate_confidence).toFixed(0)}%` : ""}
                            {setupStatus.candidate_risk_reward != null ? ` • R/R ${Number(setupStatus.candidate_risk_reward).toFixed(2)}` : ""}
                        </div>
                    )}
                    {setupStatus?.candidate_relevance_status && (
                        <div style={{ color: C.textDim, fontSize: 9, marginTop: 8, lineHeight: 1.5 }}>
                            Relevance: {setupStatus.candidate_relevance_status}
                            {setupStatus.candidate_entry_distance_pct != null ? `, entry ${Number(setupStatus.candidate_entry_distance_pct).toFixed(1)}% from current` : ""}
                        </div>
                    )}
                    {setupStatus?.current_price != null && (
                        <div style={{ color: C.textDim, fontSize: 9, marginTop: 8, lineHeight: 1.5 }}>
                            Current price: {formatSetupPrice(setupStatus.current_price)}
                        </div>
                    )}
                    {setupStatus?.conflicting_pattern_names?.length > 0 && (
                        <div style={{ color: C.textDim, fontSize: 9, marginTop: 8, lineHeight: 1.5 }}>
                            Conflicts: {setupStatus.conflicting_pattern_names.join(", ")}
                        </div>
                    )}
                </div>

                {checks.length > 0 && (
                    <div style={{ display: "grid", gap: 0 }}>
                        {checks.map(check => (
                            <DecisionCheckRow key={check.label} label={check.label} passed={check.passed} />
                        ))}
                    </div>
                )}
                {showDetails && (
                    <PatternDetailsSection
                        patterns={patterns}
                        bestSetup={bestSetup}
                        setupStatus={setupStatus}
                        advancedMode={advancedMode}
                    />
                )}
            </div>
        );
    }

    const tone = getSetupTone(bestSetup);
    const confidence = Number(bestSetup.confidence_score ?? 0);
    const riskReward = bestSetup.risk_reward_ratio != null ? Number(bestSetup.risk_reward_ratio) : null;
    const actionText = bestSetup.action;
    const statusLabel = bestSetup.pattern_status === "confirmed" ? "Confirmed" : bestSetup.pattern_status === "forming" ? "Forming" : "Broken";
    const strengthTone = getStrengthTone(bestSetup.strength_label);

    return (
        <div style={{
            background: C.bg0, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "16px", minWidth: 280, fontFamily: "'DM Mono', monospace", fontSize: 10,
            alignSelf: "flex-start",
        }}>
            <div style={{
                color: C.textDim, fontSize: 9, letterSpacing: 1.4, marginBottom: 10,
                fontWeight: 800, textTransform: "uppercase",
            }}>
                Best Setup Status
            </div>

            <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
                    <div>
                        <div style={{ color: tone.color, fontSize: 18, fontWeight: 900, lineHeight: 1.2 }}>
                            {bestSetup.pattern_name}
                        </div>
                        <div style={{ color: C.textDim, fontSize: 10, marginTop: 4 }}>
                            Valid Setup | {bestSetup.timeframe} | {tone.label}
                        </div>
                    </div>
                    <div style={{
                        background: tone.color + "1a", color: tone.color, border: `1px solid ${tone.color}44`,
                        borderRadius: 999, padding: "4px 10px", fontWeight: 800, fontSize: 10,
                        whiteSpace: "nowrap",
                    }}>
                        {confidence.toFixed(0)}%
                    </div>
                </div>
            </div>

            <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.textDim, marginBottom: 4 }}>
                    <span>Confidence</span>
                    <span style={{ color: tone.color, fontWeight: 700 }}>{confidence.toFixed(0)}%</span>
                </div>
                <div style={{ background: C.border, borderRadius: 4, height: 6, overflow: "hidden" }}>
                    <div style={{
                        width: `${Math.max(Math.min(confidence, 100), 0)}%`,
                        height: "100%",
                        borderRadius: 4,
                        background: tone.color,
                        transition: "width 0.4s ease",
                    }} />
                </div>
            </div>

            <div style={{ display: "grid", gap: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>Status</span>
                    <span style={{ color: tone.color, fontWeight: 700 }}>{statusLabel}</span>
                </div>
                {bestSetup.current_price != null && (
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                        <span style={{ color: C.textDim }}>Current</span>
                        <span style={{ color: C.text, fontWeight: 700 }}>{formatSetupPrice(bestSetup.current_price)}</span>
                    </div>
                )}
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>Direction</span>
                    <span style={{ color: tone.color, fontWeight: 700 }}>{tone.label}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>Entry</span>
                    <span style={{ color: C.text, fontWeight: 700 }}>{formatSetupPrice(bestSetup.entry_price)}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>Stop</span>
                    <span style={{ color: C.red, fontWeight: 700 }}>{formatSetupPrice(bestSetup.stop_loss)}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim }}>Target</span>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{ color: C.green, fontWeight: 700 }}>{formatSetupPrice(bestSetup.primary_target)}</span>
                        <span style={{ color: tone.color, fontSize: 9 }}>{formatSetupPercent(bestSetup.target_move_pct)}</span>
                    </div>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0" }}>
                    <span style={{ color: C.textDim }}>Risk / Reward</span>
                    <span style={{ color: C.text, fontWeight: 700 }}>{riskReward ? riskReward.toFixed(1) : "—"}</span>
                </div>
                {bestSetup.entry_distance_pct != null && (
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0" }}>
                        <span style={{ color: C.textDim }}>Entry Distance</span>
                        <span style={{ color: C.text, fontWeight: 700 }}>{formatSetupPercent(bestSetup.entry_distance_pct, false)}</span>
                    </div>
                )}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", marginTop: 10 }}>
                <span style={{ color: C.textDim }}>Strength</span>
                <span style={{ color: strengthTone, fontWeight: 800 }}>{bestSetup.strength_label}</span>
            </div>

            {showDetails && Array.isArray(bestSetup.secondary_targets) && bestSetup.secondary_targets.length > 0 && (
                <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 10,
                    marginTop: 10,
                    padding: "10px 12px",
                    border: `1px solid ${C.border}`,
                    borderRadius: 8,
                    flexWrap: "wrap",
                }}>
                    <div style={{ display: "flex", gap: 12, color: C.textMid, fontSize: 9, flexWrap: "wrap" }}>
                        {showTargets && bestSetup.secondary_targets[0] != null && <span>T2 {formatSetupPrice(bestSetup.secondary_targets[0])}</span>}
                        {showTargets && bestSetup.secondary_targets[1] != null && <span>T3 {formatSetupPrice(bestSetup.secondary_targets[1])}</span>}
                        {!showTargets && <span>Secondary targets hidden</span>}
                    </div>
                    <button onClick={() => setShowTargets(v => !v)} style={{
                        background: "transparent",
                        color: tone.color,
                        border: `1px solid ${tone.color}44`,
                        borderRadius: 999,
                        padding: "4px 10px",
                        fontSize: 9,
                        fontWeight: 800,
                        cursor: "pointer",
                    }}>
                        {showTargets ? "Collapse targets ^" : "Expand targets v"}
                    </button>
                </div>
            )}

            <div style={{
                marginTop: 14,
                background: tone.color + "14",
                border: `1px solid ${tone.color}33`,
                borderRadius: 8,
                padding: "12px 14px",
            }}>
                <div style={{ color: C.textDim, fontSize: 9, textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 6 }}>
                    Reason
                </div>
                <div style={{ color: tone.color, fontSize: 14, fontWeight: 900, lineHeight: 1.35 }}>
                    {setupStatus?.reason || "Best setup ready"}
                </div>
                <div style={{ color: C.textMid, fontSize: 9, marginTop: 6, lineHeight: 1.5 }}>
                    {actionText} | {statusLabel} | {tone.label}
                </div>
            </div>

            {alternativeCount > 0 && (
                <div style={{ marginTop: 12, color: C.textDim, fontSize: 9, lineHeight: 1.5 }}>
                    {alternativeCount} other pattern{alternativeCount === 1 ? "" : "s"} hidden by default for a cleaner decision view.
                </div>
            )}

            {showDetails && checks.length > 0 && (
                <div style={{ marginTop: 12 }}>
                    <div style={{ color: C.textDim, fontSize: 9, textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 4 }}>
                        Decision Checks
                    </div>
                    <div style={{ display: "grid" }}>
                        {checks.map(check => (
                            <DecisionCheckRow key={check.label} label={check.label} passed={check.passed} />
                        ))}
                    </div>
                </div>
            )}
            {showDetails && (
                <PatternDetailsSection
                    patterns={patterns}
                    bestSetup={bestSetup}
                    setupStatus={setupStatus}
                    advancedMode={advancedMode}
                />
            )}
        </div>
    );
}

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
function PatternScopeButton({ label, active, onClick }) {
    return (
        <button onClick={onClick} style={{
            background: active ? C.amber + "22" : "transparent",
            color: active ? C.amber : C.textDim,
            border: `1px solid ${active ? C.amber + "55" : C.border}`,
            borderRadius: 6, padding: "4px 10px", fontSize: 10, fontWeight: 700,
            cursor: "pointer", transition: "all .2s",
        }}>
            {label}
        </button>
    );
}

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

    const [interval, setInterval] = useState("1d");
    const intervals = ["1m", "1h", "1d", "1wk", "1mo"];

    // ── View Mode ──────────────────────────────────────────
    const [viewMode, setViewMode] = useState("pattern"); // "indicator" | "pattern" | "advanced"
    const [patternScope, setPatternScope] = useState("best");

    // ── User Level ─────────────────────────────────────────
    const [userLevel, setUserLevel] = useState(() => {
        try { return localStorage.getItem("qv_userLevel") || "beginner"; } catch { return "beginner"; }
    });
    useEffect(() => {
        localStorage.setItem("qv_userLevel", userLevel);
    }, [userLevel]);
    useEffect(() => {
        localStorage.setItem("qv_patternScope", patternScope);
    }, [patternScope]);

    // ── Derive toggle states from viewMode ─────────────────
    const isIndicatorMode = viewMode === "indicator";
    const isPatternMode = viewMode === "pattern" || viewMode === "advanced";
    const showSR = isIndicatorMode;
    const showSMA200 = isIndicatorMode;
    const showSMA = false;
    const showEMA = false;
    const showBB = false;
    const showVWAP = false;
    const showRSI = isIndicatorMode;
    const showMACD = false;
    const showVol = isIndicatorMode;
    const showATR = false;
    const showPatterns = isPatternMode;
    const showMLSignals = false;

    const stateData = useRef({
        ohlc: [],
        indicators: [],
        signals: [],
        patterns: [],
        bestPattern: null,
        bestSetup: null,
        bestSetupStatus: null,
        srData: null,
        indicatorSummary: null,
    });

    // Fetch confluence independent of main timeframe API
    const [confluence, setConfluence] = useState([]);
    useEffect(() => {
        if (!isPatternMode) {
            setConfluence([]);
            return;
        }
        let active = true;
        fetchConfluence(symbol).then(d => {
            if (active && d?.confluence_signals) setConfluence(d.confluence_signals);
        }).catch(() => {});
        return () => { active = false; };
    }, [symbol, isPatternMode]);

    // ── Fetch sentiment for DecisionPanel ──────────────────
    useEffect(() => {
        let cancelled = false;

        async function loadData() {
            setLoading(true); setError(null);
            try {
                const priceDays = { "1m": 5, "1h": 180, "1d": 420, "1wk": 2500, "1mo": 5600 }[interval] || 120;
                const indicatorDays = { "1m": 120, "1h": 240, "1d": 320, "1wk": 300, "1mo": 180 }[interval] || 120;
                const lookback = { "1m": 90, "1h": 365, "1d": 420, "1wk": 2500, "1mo": 5600 }[interval] || 180;
                const [priceRes, indRes, patRes, srRes] = await Promise.all([
                    fetchPrices(symbol, "yfinance", priceDays, interval),
                    fetchIndicators(symbol, indicatorDays, interval),
                    isPatternMode ? fetchPatterns(symbol, interval) : Promise.resolve(null),
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
                    patterns: patRes?.patterns || [],
                    bestPattern: patRes?.best_pattern || null,
                    bestSetup: patRes?.best_setup || null,
                    bestSetupStatus: patRes?.best_setup_status || null,
                    srData: srRes || null,
                    indicatorSummary: isIndicatorMode ? buildIndicatorSummary({
                        indicators: indRes?.data || [],
                        prices: priceRes?.bars || [],
                        srData: srRes,
                        interval,
                    }) : null,
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

            const { ohlc, indicators, signals, bestPattern, bestSetup, srData } = stateData.current;
            const visiblePatterns = showPatterns && bestSetup && bestPattern ? [bestPattern] : [];
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
            if (visiblePatterns.length > 0) {
                visiblePatterns.forEach(p => {
                    const isBullish = p.direction === "bullish";
                    const isNeutral = p.direction === "neutral";
                    
                    const isConfluent = false;

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
            if (visiblePatterns.length > 0) {
                visiblePatterns.forEach(cp => {
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

                    if (patternScope === "best" || patternScope === "all") {
                        if (cp.entry_price != null) {
                            candleSeries.createPriceLine({
                                price: cp.entry_price,
                                color: "#fbbf24", lineWidth: 2, lineStyle: 0,
                                axisLabelVisible: true, title: "Entry",
                            });
                        }

                        if (cp.target_price != null) {
                            candleSeries.createPriceLine({
                                price: cp.target_price,
                                color: C.green + 'CC',
                                lineWidth: 2, lineStyle: 0, axisLabelVisible: true,
                                title: "Target 1",
                            });
                        }

                        if (cp.stop_loss != null) {
                            candleSeries.createPriceLine({
                                price: cp.stop_loss,
                                color: C.red + 'CC',
                                lineWidth: 2, lineStyle: 0, axisLabelVisible: true,
                                title: "Stop Loss",
                            });
                        }
                    } else if (cp.status === "confirmed" || cp.pattern_name !== "Symmetrical Triangle") {
                        if (cp.entry_price != null) {
                            candleSeries.createPriceLine({
                                price: cp.entry_price,
                                color: "#fbbf24", lineWidth: 1, lineStyle: 3,
                                axisLabelVisible: true, title: `${cp.pattern_name} Entry`,
                            });
                        }

                        if (cp.target_price != null) {
                            candleSeries.createPriceLine({
                                price: cp.target_price,
                                color: C.green + 'CC',
                                lineWidth: 1, lineStyle: 4, axisLabelVisible: true,
                                title: `${cp.pattern_name} Target`,
                            });
                        }

                        if (cp.stop_loss != null) {
                            candleSeries.createPriceLine({
                                price: cp.stop_loss,
                                color: C.red + 'CC',
                                lineWidth: 1, lineStyle: 4, axisLabelVisible: true,
                                title: `${cp.pattern_name} Stop`,
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
    }, [symbol, interval, viewMode, mode, predictionData, patternScope, confluence]);

    const bestPattern = stateData.current.bestPattern;
    const bestSetup = stateData.current.bestSetup;
    const bestSetupStatus = stateData.current.bestSetupStatus;
    const indicatorSummary = stateData.current.indicatorSummary;
    const selectedPatterns = new Set(bestSetup ? [bestSetup.pattern_name] : []);
    const alternativeCount = Math.max((stateData.current.patterns?.length || 0) - (bestSetup ? 1 : 0), 0);

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
                                <ModeButton label="Pattern" emoji="🟣" active={viewMode === "pattern"} onClick={() => setViewMode("pattern")} />
                                <ModeButton label="Advanced" emoji="🔴" active={viewMode === "advanced"} onClick={() => setViewMode("advanced")} />
                            </div>

                            {showPatterns && (
                                <>
                                    <div style={{ width: 1, height: 16, background: C.border, margin: "0 2px" }} />
                                    <div style={{ display: "flex", gap: 4, background: C.bg0, padding: 3, borderRadius: 6 }}>
                                        <PatternScopeButton label="Best Setup Only" active={patternScope === "best"} onClick={() => setPatternScope("best")} />
                                        <PatternScopeButton label="Show All Patterns" active={patternScope === "all"} onClick={() => setPatternScope("all")} />
                                    </div>
                                </>
                            )}

                            {/* Pattern selector (Pattern/Advanced mode only) */}
                            {false && (
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

                {/* Right side: Mode-specific decision panel */}
                <div style={{ width: 300, flexShrink: 0 }}>
                    {isIndicatorMode ? (
                        <IndicatorSummaryPanel summary={indicatorSummary} loading={loading} error={error} />
                    ) : (
                        <TradeSetupPanel
                            bestSetup={bestSetup}
                            setupStatus={bestSetupStatus}
                            alternativeCount={alternativeCount}
                            loading={loading}
                            patterns={stateData.current.patterns || []}
                            showDetails={patternScope === "all" || viewMode === "advanced"}
                            advancedMode={viewMode === "advanced"}
                        />
                    )}
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
