import { useEffect, useState } from "react";
import { fetchPrices, fetchIndicators, fetchPatterns, fetchSupportResistance } from "../utils/api";
import { C } from "../utils/data";
import { CHART_TIMEFRAMES, tradingViewUrl } from "../utils/tradingview";
import TradingViewChart from "./TradingViewChart";

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

function buildIndicatorSummary({ indicators, prices, srData, timeframe }) {
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
        timeframe: timeframe,
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


/**
 * Chart detail view: TradingView's chart, our analysis panels beside it.
 *
 * This used to render its own candlesticks with lightweight-charts — candles,
 * volume, RSI and MACD panes, markers, trendlines, a prediction cone. All of
 * that was an imitation of the chart TradingView already ships, and it has been
 * removed rather than improved. What is left is the half that was ours to begin
 * with: pattern detection, trade-setup evaluation and the indicator summary,
 * every number of which comes from our backend.
 *
 * The mode toggle now switches which *analysis* is shown, not which series are
 * drawn — TradingView owns what is on the chart, and the studies it opens with
 * follow the mode so the visual and the panel are reading the same thing.
 */

// TradingView's own study IDs. Indicator mode opens the chart with the
// oscillators the summary panel talks about; pattern mode keeps it clean so
// structure is readable.
const INDICATOR_STUDIES = ["STD;Volume", "STD;RSI", "STD;MACD"];
const PATTERN_STUDIES = ["STD;Volume"];

export default function TradingViewDetail({ symbol, onClose }) {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [timeframe, setTimeframe] = useState("1d");

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

    const isIndicatorMode = viewMode === "indicator";
    const isPatternMode = viewMode === "pattern" || viewMode === "advanced";

    const [analysis, setAnalysis] = useState({
        patterns: [],
        bestSetup: null,
        bestSetupStatus: null,
        indicatorSummary: null,
    });

    // ── Analysis for the side panel ────────────────────────
    // Prices and indicators are still fetched — not to draw them, but because
    // the indicator summary and the setup evaluation are computed from them.
    useEffect(() => {
        let cancelled = false;

        async function loadData() {
            setLoading(true);
            setError(null);
            try {
                const priceDays = { "1m": 5, "15m": 30, "1h": 180, "1d": 420, "1wk": 2500, "1mo": 5600 }[timeframe] || 120;
                const indicatorDays = { "1m": 120, "15m": 120, "1h": 240, "1d": 320, "1wk": 300, "1mo": 180 }[timeframe] || 120;
                const lookback = { "1m": 90, "15m": 120, "1h": 365, "1d": 420, "1wk": 2500, "1mo": 5600 }[timeframe] || 180;

                const [priceRes, indRes, patRes, srRes] = await Promise.all([
                    fetchPrices(symbol, "yfinance", priceDays, timeframe),
                    fetchIndicators(symbol, indicatorDays, timeframe),
                    isPatternMode ? fetchPatterns(symbol, timeframe) : Promise.resolve(null),
                    isIndicatorMode ? fetchSupportResistance(symbol, timeframe, lookback) : Promise.resolve(null),
                ].map(p => p.catch(() => null)));

                if (cancelled) return;
                if (!priceRes || !priceRes.bars) throw new Error("Failed to fetch price data");

                setAnalysis({
                    patterns: patRes?.patterns || [],
                    bestSetup: patRes?.best_setup || null,
                    bestSetupStatus: patRes?.best_setup_status || null,
                    indicatorSummary: isIndicatorMode ? buildIndicatorSummary({
                        indicators: indRes?.data || [],
                        prices: priceRes?.bars || [],
                        srData: srRes,
                        timeframe,
                    }) : null,
                });
            } catch (err) {
                if (!cancelled) setError(err.message);
            } finally {
                if (!cancelled) setLoading(false);
            }
        }

        loadData();
        return () => { cancelled = true; };
    }, [symbol, timeframe, isPatternMode, isIndicatorMode]);

    const { patterns, bestSetup, bestSetupStatus, indicatorSummary } = analysis;
    const alternativeCount = Math.max((patterns?.length || 0) - (bestSetup ? 1 : 0), 0);

    return (
        <div className="fade-up" style={{ display: "flex", flexDirection: "column", width: "100%" }}>
            <div style={{ display: "flex", gap: 12, alignItems: "stretch", flexWrap: "wrap" }}>
                {/* Left: toolbar + the real TradingView chart */}
                <div style={{ flex: "1 1 560px", minWidth: 0, display: "flex", flexDirection: "column" }}>
                    <div style={{
                        display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8,
                        background: C.bg2, padding: "6px 14px", borderRadius: 8, border: `1px solid ${C.border}`,
                        flexWrap: "wrap", gap: 8,
                    }}>
                        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                            <div style={{ fontWeight: 800, color: C.text, fontSize: 16, marginRight: 6 }}>{symbol}</div>
                            <div style={{ display: "flex", background: C.bg0, padding: 2, borderRadius: 5 }}>
                                {CHART_TIMEFRAMES.map(inv => (
                                    <button key={inv} onClick={() => setTimeframe(inv)} style={{
                                        background: timeframe === inv ? C.amber : "transparent",
                                        color: timeframe === inv ? "#000" : C.textMid,
                                        border: "none", borderRadius: 3, padding: "3px 8px",
                                        fontSize: 10, fontWeight: 700, cursor: "pointer",
                                    }}>{inv}</button>
                                ))}
                            </div>
                        </div>

                        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                            <div style={{ display: "flex", gap: 4, background: C.bg0, padding: 3, borderRadius: 6 }}>
                                <ModeButton label="Indicator" emoji="🟢" active={viewMode === "indicator"} onClick={() => setViewMode("indicator")} />
                                <ModeButton label="Pattern" emoji="🟣" active={viewMode === "pattern"} onClick={() => setViewMode("pattern")} />
                                <ModeButton label="Advanced" emoji="🔴" active={viewMode === "advanced"} onClick={() => setViewMode("advanced")} />
                            </div>

                            {isPatternMode && (
                                <>
                                    <div style={{ width: 1, height: 16, background: C.border, margin: "0 2px" }} />
                                    <div style={{ display: "flex", gap: 4, background: C.bg0, padding: 3, borderRadius: 6 }}>
                                        <PatternScopeButton label="Best Setup Only" active={patternScope === "best"} onClick={() => setPatternScope("best")} />
                                        <PatternScopeButton label="Show All Patterns" active={patternScope === "all"} onClick={() => setPatternScope("all")} />
                                    </div>
                                </>
                            )}

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

                            <div style={{ width: 1, height: 16, background: C.border, margin: "0 2px" }} />
                            <button onClick={onClose} style={{
                                background: "transparent", border: "none", color: C.textDim, cursor: "pointer",
                                fontSize: 20, lineHeight: "20px", padding: "0 4px",
                            }}>×</button>
                        </div>
                    </div>

                    <div style={{ borderRadius: 8, overflow: "hidden", border: `1px solid ${C.border}` }}>
                        <TradingViewChart
                            symbol={symbol}
                            interval={timeframe}
                            height={520}
                            studies={isIndicatorMode ? INDICATOR_STUDIES : PATTERN_STUDIES}
                        />
                    </div>
                </div>

                {/* Right: the mode-specific panel, every number of it from our backend */}
                <div style={{ width: 300, flexShrink: 0 }}>
                    {isIndicatorMode ? (
                        <IndicatorSummaryPanel summary={indicatorSummary} loading={loading} error={error} />
                    ) : (
                        <TradeSetupPanel
                            bestSetup={bestSetup}
                            setupStatus={bestSetupStatus}
                            alternativeCount={alternativeCount}
                            loading={loading}
                            patterns={patterns || []}
                            showDetails={patternScope === "all" || viewMode === "advanced"}
                            advancedMode={viewMode === "advanced"}
                        />
                    )}
                </div>
            </div>

            <div style={{
                marginTop: 8, display: "flex", gap: 14, fontSize: 9, color: C.textDim,
                justifyContent: "center", flexWrap: "wrap",
            }}>
                <span>Chart by TradingView — drawing tools, indicators and replay are theirs</span>
                <a href={tradingViewUrl(symbol)} target="_blank" rel="noreferrer noopener" style={{ color: C.amber }}>
                    Open full chart ↗
                </a>
                <span>Patterns, levels and setups on the right are computed by our backend</span>
            </div>
        </div>
    );
}
