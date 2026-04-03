import { useState, useMemo, useEffect, useRef } from "react";
import {
    LineChart, Line, AreaChart, Area, ComposedChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    ReferenceLine
} from "recharts";
import { C } from "../utils/data";
import { fetchExtendedQuote, fetchSentiment } from "../utils/api";
import { ChartTooltip, StatCard, Section, Hint } from "../components/UIComponents";
import TradingViewDetail from "../components/TradingViewDetail";

// ── Session Panel ──────────────────────────────────────────────────────────────
const SESSION_META = {
    PRE_MARKET: { emoji: "🌅", label: "Pre-Market", color: "#60a5fa", time: "4:00–9:30 AM ET" },
    REGULAR: { emoji: "🔔", label: "Regular Mkt", color: "#34d399", time: "9:30 AM–4:00 PM ET" },
    POST_MARKET: { emoji: "🌙", label: "After-Hours", color: "#a78bfa", time: "4:00–8:00 PM ET" },
    MARKET_CLOSED: { emoji: "💤", label: "Market Closed", color: "#6b7280", time: "" },
};

function SessionRow({ emoji, label, color, timeLabel, price, change, changePct, time, active, unavailable, lowVol }) {
    const up = (change ?? 0) >= 0;
    const chgClr = unavailable ? C.textDim : up ? C.green : C.red;
    return (
        <div style={{
            display: "flex", alignItems: "center", gap: 12, padding: "10px 14px",
            background: active ? color + "15" : C.bg2,
            border: `1px solid ${active ? color + "55" : C.border}`,
            borderRadius: 8, marginBottom: 8, transition: "all .2s",
        }}>
            <span style={{ fontSize: 18 }}>{emoji}</span>
            <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ color: active ? color : C.textMid, fontWeight: 700, fontSize: 11, letterSpacing: 1 }}>
                        {label}
                    </span>
                    {active && (
                        <span style={{ background: color, color: "#000", borderRadius: 4, fontSize: 9, padding: "1px 5px", fontWeight: 800 }}>
                            ACTIVE
                        </span>
                    )}
                    {lowVol && (
                        <span style={{ background: "#f59e0b22", color: "#f59e0b", borderRadius: 4, fontSize: 9, padding: "1px 5px" }}>
                            ⚠ LOW VOL
                        </span>
                    )}
                </div>
                <div style={{ fontSize: 10, color: C.textDim, marginTop: 2 }}>
                    {time ? `As of ${time}` : timeLabel}
                </div>
            </div>
            <div style={{ textAlign: "right" }}>
                {unavailable ? (
                    <span style={{ color: C.textDim, fontSize: 11 }}>N/A — Session not started</span>
                ) : (
                    <>
                        <div style={{ color: C.text, fontWeight: 800, fontSize: 14, fontFamily: "'DM Mono',monospace" }}>
                            ${price?.toFixed(2) ?? "—"}
                        </div>
                        {change != null && (
                            <div style={{ color: chgClr, fontSize: 10 }}>
                                {up ? "▲" : "▼"} {Math.abs(change).toFixed(2)} ({changePct != null ? `${up ? "+" : ""}${changePct.toFixed(2)}%` : "—"})
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}

function MarketSessionPanel({ symbol, source }) {
    const [quote, setQuote] = useState(null);
    const [loading, setLoad] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        setLoad(true); setError(null);
        fetchExtendedQuote(symbol, source)
            .then(d => { setQuote(d); setLoad(false); })
            .catch(e => { setError(e.message); setLoad(false); });
    }, [symbol, source]);

    if (loading) return <div style={{ color: C.textDim, fontSize: 12, padding: "8px 0" }}>Loading session data…</div>;
    if (error) return <div style={{ color: C.red, fontSize: 12, padding: "8px 0" }}>⚠ {error}</div>;
    if (!quote) return null;

    const session = quote.session;
    const lowVol = quote.low_volume_warning;

    function fmtTime(iso) {
        if (!iso) return null;
        try {
            return new Date(iso).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", timeZone: "America/New_York", timeZoneName: "short" });
        } catch { return null; }
    }

    return (
        <Section title="MARKET SESSIONS">
            <SessionRow
                {...SESSION_META.PRE_MARKET}
                timeLabel={SESSION_META.PRE_MARKET.time}
                price={quote.pre.price}
                change={quote.pre.change}
                changePct={quote.pre.change_pct}
                time={fmtTime(quote.pre.time)}
                active={session === "PRE_MARKET"}
                unavailable={!quote.pre.available}
                lowVol={session === "PRE_MARKET" && lowVol}
            />
            <SessionRow
                {...SESSION_META.REGULAR}
                timeLabel={SESSION_META.REGULAR.time}
                price={quote.regular.price}
                change={quote.regular.price != null && quote.regular.prev_close != null
                    ? quote.regular.price - quote.regular.prev_close : null}
                changePct={quote.regular.price != null && quote.regular.prev_close != null
                    ? ((quote.regular.price - quote.regular.prev_close) / quote.regular.prev_close) * 100 : null}
                time={null}
                active={session === "REGULAR"}
                unavailable={!quote.regular.price}
                lowVol={false}
            />
            <SessionRow
                {...SESSION_META.POST_MARKET}
                timeLabel={SESSION_META.POST_MARKET.time}
                price={quote.post.price}
                change={quote.post.change}
                changePct={quote.post.change_pct}
                time={fmtTime(quote.post.time)}
                active={session === "POST_MARKET"}
                unavailable={!quote.post.available}
                lowVol={session === "POST_MARKET" && lowVol}
            />
            {session === "MARKET_CLOSED" && (
                <div style={{ textAlign: "center", color: C.textDim, fontSize: 11, padding: "6px 0 2px" }}>
                    💤 Market Closed — showing latest available prices
                </div>
            )}
        </Section>
    );
}

// ── Indicator Sentiment Panel ────────────────────────────────────────────────
const INDICATOR_META = {
    trend: { label: "200 MA (Trend)", icon: "📈", descriptions: { bullish: "Price above 200 MA", bearish: "Price below 200 MA", neutral: "Insufficient data" } },
    rsi: { label: "RSI 14 (Momentum)", icon: "⚡", descriptions: { bullish: "RSI crossed above 30", bearish: "RSI crossed below 70", neutral: "RSI in neutral zone" } },
    volume: { label: "Volume (Confirmation)", icon: "📊", descriptions: { bullish: "High volume bullish candle", bearish: "High volume bearish candle", neutral: "Volume below average" } },
};

function sentimentColor(score) {
    if (score >= 3) return C.green;
    if (score >= 2) return "#4ade80";
    if (score >= 1) return "#86efac";
    if (score <= -3) return C.red;
    if (score <= -2) return "#fb7185";
    if (score <= -1) return "#fb923c";
    return C.amber;
}

function entryBadge(entry) {
    const map = {
        BULLISH: { bg: C.green, emoji: "🟢", glow: C.green },
        BEARISH: { bg: C.red, emoji: "🔴", glow: C.red },
        WAIT: { bg: C.amber, emoji: "🟡", glow: C.amber },
    };
    const m = map[entry] || map.WAIT;
    return (
        <span style={{
            background: m.bg + "22", color: m.bg, borderRadius: 8,
            padding: "6px 16px", fontSize: 13, fontWeight: 800,
            border: `1px solid ${m.bg}55`, letterSpacing: 1,
            display: "inline-flex", alignItems: "center", gap: 6,
            boxShadow: `0 0 12px ${m.glow}22`,
        }}>
            <span>{m.emoji}</span> {entry}
        </span>
    );
}

function formatShareVolume(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    const volume = Number(value);
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M shares`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K shares`;
    return `${Math.round(volume)} shares`;
}

function getAtrLabel(atr, close) {
    if (atr == null || close == null || !close) return "Volatility unavailable";
    const atrPct = (Number(atr) / Number(close)) * 100;
    if (atrPct >= 4) return "High Volatility";
    if (atrPct >= 2) return "Moderate Volatility";
    return "Low Volatility";
}

function getActionTone(entrySignal) {
    if (entrySignal === "BULLISH") return { label: "BUY", color: C.green, glow: `${C.green}22` };
    if (entrySignal === "BEARISH") return { label: "SELL", color: C.red, glow: `${C.red}22` };
    return { label: "WAIT", color: C.amber, glow: `${C.amber}22` };
}

function SignalCard({ name, signal, score, detail1Label, detail1Value, detail2Label, detail2Value }) {
    const meta = INDICATOR_META[name];
    const clr = score > 0 ? C.green : score < 0 ? C.red : C.amber;
    const desc = meta.descriptions[signal] || "—";
    return (
        <div style={{
            background: C.bg2, border: `1px solid ${clr}33`,
            borderRadius: 12, padding: "16px 18px", transition: "all .25s",
            position: "relative", overflow: "hidden",
        }}>
            {/* Glow accent */}
            <div style={{
                position: "absolute", top: 0, left: 0, right: 0, height: 3,
                background: `linear-gradient(90deg, transparent, ${clr}, transparent)`,
                opacity: 0.6,
            }} />
            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 18 }}>{meta.icon}</span>
                    <span style={{ color: C.text, fontSize: 12, fontWeight: 700, fontFamily: "'Syne',sans-serif" }}>{meta.label}</span>
                </div>
                <span style={{
                    background: clr + "22", color: clr, borderRadius: 6,
                    padding: "3px 10px", fontSize: 11, fontWeight: 800,
                    fontFamily: "'DM Mono',monospace",
                    border: `1px solid ${clr}44`,
                }}>
                    {score > 0 ? "+1" : score < 0 ? "-1" : "0"}
                </span>
            </div>
            {/* Signal label */}
            <div style={{
                color: clr, fontSize: 15, fontWeight: 800, textTransform: "uppercase",
                letterSpacing: 1.2, marginBottom: 6,
            }}>
                {signal}
            </div>
            <div style={{ color: C.textMid, fontSize: 11, marginBottom: 10 }}>{desc}</div>
            {/* Detail values */}
            <div style={{ display: "flex", gap: 16, fontSize: 10, color: C.textDim }}>
                {detail1Label && (
                    <span>
                        <span style={{ color: C.textMid, fontWeight: 600 }}>{detail1Label}:</span>{" "}
                        <span style={{ color: C.text, fontFamily: "'DM Mono',monospace" }}>{detail1Value}</span>
                    </span>
                )}
                {detail2Label && (
                    <span>
                        <span style={{ color: C.textMid, fontWeight: 600 }}>{detail2Label}:</span>{" "}
                        <span style={{ color: C.text, fontFamily: "'DM Mono',monospace" }}>{detail2Value}</span>
                    </span>
                )}
            </div>
        </div>
    );
}

function IndicatorSentimentPanel({ data, latestBar, loading, error }) {
    if (loading) return (
        <div className="fade-up" style={{ marginTop: 24 }}>
            <div style={{
                background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 12,
                padding: "28px 0", textAlign: "center",
            }}>
                <div style={{ fontSize: 24, marginBottom: 8, animation: "pulse 1.5s infinite" }}>🧠</div>
                <div style={{ color: C.textDim, fontSize: 12 }}>Computing sentiment signals…</div>
            </div>
        </div>
    );
    if (error) return (
        <div className="fade-up" style={{ marginTop: 24 }}>
            <div style={{
                background: C.bg2, border: `1px solid ${C.red}33`, borderRadius: 12,
                padding: "20px 24px", textAlign: "center",
            }}>
                <div style={{ color: C.red, fontSize: 12 }}>⚠ Sentiment unavailable — {error}</div>
            </div>
        </div>
    );
    if (!data) return null;

    const score = data.score ?? 0;
    const clr = sentimentColor(score);
    // Map score [-3,+3] to [0,100] for gauge
    const gaugePct = ((score + 3) / 6) * 100;
    const det = data.details || {};
    const actionTone = getActionTone(data.entry_signal);
    const volumeText = formatShareVolume(det.volume);
    const avgVolumeText = formatShareVolume(det.avg_volume_20);
    const atrValue = latestBar?.atr;
    const atrLabel = getAtrLabel(atrValue, det.close ?? latestBar?.close);
    const supportText = det.support != null ? `$${Number(det.support).toFixed(2)}` : "—";
    const resistanceText = det.resistance != null ? `$${Number(det.resistance).toFixed(2)}` : "—";

    return (
        <div className="fade-up" style={{ marginTop: 24 }}>
            {/* Header */}
            <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                marginBottom: 14,
            }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 22 }}>🧠</span>
                    <span style={{
                        fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 15,
                        color: C.text, letterSpacing: 1, textTransform: "uppercase",
                    }}>Indicator Sentiment</span>
                </div>
                <span style={{
                    background: clr + "18", color: clr, borderRadius: 999,
                    padding: "6px 14px", fontSize: 11, fontWeight: 800,
                    border: `1px solid ${clr}44`, letterSpacing: 0.8,
                }}>
                    {data.confidence}% confidence
                </span>
            </div>

            {/* Score Card */}
            <div style={{
                background: `linear-gradient(135deg, ${C.bg2}, ${C.bg3})`,
                border: `1px solid ${clr}33`,
                borderRadius: 14, padding: "22px 28px", marginBottom: 16,
                position: "relative", overflow: "hidden",
            }}>
                {/* Background glow */}
                <div style={{
                    position: "absolute", top: -40, right: -40, width: 120, height: 120,
                    background: `radial-gradient(circle, ${clr}15, transparent)`,
                    borderRadius: "50%",
                }} />
                <div style={{
                    position: "relative",
                    background: actionTone.glow,
                    border: `1px solid ${actionTone.color}44`,
                    borderRadius: 16,
                    padding: "18px 20px",
                    marginBottom: 18,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 16,
                    flexWrap: "wrap",
                }}>
                    <div>
                        <div style={{
                            color: C.textDim, fontSize: 10, letterSpacing: 1.5, marginBottom: 8,
                            fontFamily: "'Syne',sans-serif",
                        }}>
                            ACTION SIGNAL
                        </div>
                        <div style={{
                            color: actionTone.color,
                            fontSize: 36,
                            fontWeight: 900,
                            lineHeight: 1,
                            letterSpacing: 1.8,
                            fontFamily: "'Syne',sans-serif",
                        }}>
                            {actionTone.label}
                        </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                        <div style={{ color: C.text, fontSize: 13, fontWeight: 800, marginBottom: 4 }}>
                            {data.sentiment}
                        </div>
                        <div style={{ color: C.textDim, fontSize: 10 }}>
                            Indicators provide context
                        </div>
                    </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", position: "relative" }}>
                    <div>
                        <div style={{
                            color: C.textDim, fontSize: 10, letterSpacing: 1.5, marginBottom: 8,
                            fontFamily: "'Syne',sans-serif",
                        }}>
                            SENTIMENT SCORE
                        </div>
                        <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
                            <span style={{
                                color: clr, fontSize: 42, fontWeight: 800,
                                fontFamily: "'DM Mono',monospace", lineHeight: 1,
                            }}>
                                {score >= 0 ? "+" : ""}{score}
                            </span>
                            <span style={{
                                background: clr + "22", color: clr, borderRadius: 8,
                                padding: "5px 14px", fontSize: 13, fontWeight: 700,
                                border: `1px solid ${clr}44`,
                            }}>
                                {data.sentiment}
                            </span>
                        </div>
                    </div>
                    {/* Gauge */}
                    <div style={{ width: 200 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.textDim, marginBottom: 5 }}>
                            <span>-3 Bearish</span><span>0</span><span>+3 Bullish</span>
                        </div>
                        <div style={{
                            background: C.bg1, borderRadius: 8, height: 10, position: "relative",
                            overflow: "hidden", border: `1px solid ${C.border}`,
                        }}>
                            <div style={{
                                position: "absolute", inset: 0,
                                background: `linear-gradient(90deg, ${C.red}55, ${C.red}33, ${C.amber}33, ${C.green}33, ${C.green}55)`,
                                borderRadius: 8,
                            }} />
                            {/* Center line */}
                            <div style={{
                                position: "absolute", left: "50%", top: 0, bottom: 0,
                                width: 1, background: C.textDim + "66",
                            }} />
                            {/* Needle */}
                            <div style={{
                                position: "absolute", top: -3, width: 6, height: 16,
                                background: clr, borderRadius: 3,
                                left: `calc(${gaugePct}% - 3px)`,
                                boxShadow: `0 0 8px ${clr}`,
                                transition: "left .6s ease",
                            }} />
                        </div>
                        {/* Tick marks */}
                        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 3, padding: "0 2px" }}>
                            {[-3, -2, -1, 0, 1, 2, 3].map(v => (
                                <div key={v} style={{
                                    width: 1, height: 4,
                                    background: v === score ? clr : C.textDim + "44",
                                }} />
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            <div style={{
                background: C.bg2,
                border: `1px solid ${C.border}`,
                borderRadius: 12,
                padding: "16px 18px",
                marginBottom: 16,
                display: "grid",
                gap: 10,
            }}>
                <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1.4, textTransform: "uppercase", fontFamily: "'Syne',sans-serif" }}>
                    Indicator Summary
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                    <span style={{ color: C.textDim }}>Trend</span>
                    <span style={{ color: data.trend === "bullish" ? C.green : data.trend === "bearish" ? C.red : C.amber, fontWeight: 700 }}>
                        {data.trend?.charAt(0).toUpperCase()}{data.trend?.slice(1)}
                    </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                    <span style={{ color: C.textDim }}>RSI</span>
                    <span style={{ color: C.text, fontWeight: 700 }}>
                        {det.rsi != null ? det.rsi.toFixed(1) : "—"} → {data.rsi_signal === "bullish" ? "Bullish" : data.rsi_signal === "bearish" ? "Bearish" : "Neutral"}
                    </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                    <span style={{ color: C.textDim }}>Volume</span>
                    <span style={{ color: C.text, fontWeight: 700 }}>
                        {volumeText} → {data.volume_strength === "strong" ? "Strong" : "Weak"}
                    </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                    <span style={{ color: C.textDim }}>ATR</span>
                    <span style={{ color: C.text, fontWeight: 700 }}>
                        {atrValue != null ? Number(atrValue).toFixed(2) : "—"} → {atrLabel}
                    </span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, paddingTop: 8, borderTop: `1px solid ${C.border}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                        <span style={{ color: C.textDim }}>Support</span>
                        <span style={{ color: C.cyan, fontWeight: 700 }}>{supportText}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                        <span style={{ color: C.textDim }}>Resistance</span>
                        <span style={{ color: C.red, fontWeight: 700 }}>{resistanceText}</span>
                    </div>
                </div>
            </div>

            {/* 3 Indicator Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 16 }}>
                <SignalCard
                    name="trend"
                    signal={data.trend}
                    score={data.trend_score}
                    detail1Label="Close" detail1Value={det.close != null ? `$${det.close.toFixed(2)}` : "—"}
                    detail2Label="200 MA" detail2Value={det.sma_200 != null ? `$${det.sma_200.toFixed(2)}` : "—"}
                />
                <SignalCard
                    name="rsi"
                    signal={data.rsi_signal}
                    score={data.rsi_score}
                    detail1Label="RSI" detail1Value={det.rsi != null ? det.rsi.toFixed(1) : "—"}
                    detail2Label="Prev RSI" detail2Value={det.rsi_prev != null ? det.rsi_prev.toFixed(1) : "—"}
                />
                <SignalCard
                    name="volume"
                    signal={data.volume_signal}
                    score={data.volume_score}
                    detail1Label="Volume" detail1Value={det.volume != null ? (det.volume / 1e6).toFixed(1) + "M" : "—"}
                    detail2Label="Avg 20d" detail2Value={det.avg_volume_20 != null ? (det.avg_volume_20 / 1e6).toFixed(1) + "M" : "—"}
                />
            </div>

            {/* Disclaimer */}
            <div style={{
                background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 8,
                padding: "10px 16px", fontSize: 10, color: C.textDim, lineHeight: 1.5,
                display: "flex", alignItems: "center", gap: 8,
            }}>
                <span style={{ fontSize: 14 }}>ℹ️</span>
                <span>Probability-based system using daily timeframe. Not predictive. Signals update on each trading day close.</span>
            </div>
        </div>
    );
}

// ── Analysis Tab ───────────────────────────────────────────────────────────────
export default function AnalysisTab({ selectedTicker, setSelectedTicker, priceData, indicatorData, dataSource, apiConnected }) {
    const [showDetails, setShowDetails] = useState(false);
    const [showSMA, setShowSMA] = useState(true);
    const [showEMA, setShowEMA] = useState(false);
    const [showVolume, setShowVolume] = useState(true);
    const [timeRange, setTimeRange] = useState(60);
    const [searchInput, setSearchInput] = useState("");
    const searchRef = useRef(null);

    // Sentiment state
    const [sentiment, setSentiment] = useState(null);
    const [sentimentLoading, setSentimentLoading] = useState(false);
    const [sentimentError, setSentimentError] = useState(null);

    useEffect(() => {
        if (!apiConnected || !selectedTicker) return;
        setSentimentLoading(true);
        setSentimentError(null);
        fetchSentiment(selectedTicker)
            .then(d => { setSentiment(d); setSentimentLoading(false); })
            .catch(e => { setSentimentError(e.message); setSentimentLoading(false); });
    }, [selectedTicker, apiConnected]);

    const handleQuickSearch = () => {
        const t = searchInput.trim().toUpperCase();
        if (t && setSelectedTicker) { setSelectedTicker(t); setSearchInput(""); }
    };

    const chartData = useMemo(() => {
        if (!priceData?.bars || !indicatorData?.data) return [];
        const prices = priceData.bars.slice(-timeRange);
        const indMap = {};
        (indicatorData.data || []).forEach(d => { indMap[d.date] = d; });
        return prices.map(bar => {
            const ind = indMap[bar.date] || {};
            return {
                date: bar.date.slice(5),
                close: bar.close, open: bar.open, high: bar.high, low: bar.low, volume: bar.volume,
                sma20: ind.SMA_20, ema12: ind.EMA_12, rsi: ind.RSI,
                macd: ind.MACD, macdSig: ind.MACD_Signal, macdHist: ind.MACD_Histogram,
                bbUpper: ind.BB_High, bbMid: ind.BB_Mid, bbLower: ind.BB_Low,
                atr: ind.ATR, obv: ind.OBV,
            };
        });
    }, [priceData, indicatorData, timeRange]);

    if (!apiConnected) {
        return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>🔌</div>
            <div style={{ fontSize: 14 }}>Connect to API server to view live data</div>
            <div style={{ fontSize: 11, marginTop: 8, color: C.textDim }}>Start the API: <code>python -m uvicorn src.api.main:app --port 8000</code></div>
        </div>;
    }

    if (!chartData.length) {
        return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>Loading {selectedTicker} data...</div>;
    }

    const last = chartData[chartData.length - 1];
    const prev = chartData[chartData.length - 2] || last;
    const change = last.close - prev.close;
    const changePct = (change / prev.close) * 100;

    if (showDetails) {
        return <TradingViewDetail symbol={selectedTicker} mode="analysis" onClose={() => setShowDetails(false)} />;
    }

    return (
        <div className="fade-up">
            {/* Quick ticker search bar */}
            <div style={{ display: "flex", gap: 8, marginBottom: 18, alignItems: "center" }}>
                <div style={{ position: "relative", flex: 1, maxWidth: 320 }}>
                    <span style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", fontSize: 12, color: C.textDim, pointerEvents: "none" }}>🔍</span>
                    <input
                        ref={searchRef}
                        placeholder="Search any ticker — e.g. TSM, BABA, JPM…"
                        value={searchInput}
                        onChange={e => setSearchInput(e.target.value)}
                        onKeyDown={e => { if (e.key === "Enter") handleQuickSearch(); }}
                        style={{
                            width: "100%", background: C.bg2, border: `1px solid ${C.border}`,
                            borderRadius: 8, color: C.text, padding: "8px 10px 8px 30px",
                            fontSize: 11, fontFamily: "'DM Mono',monospace", outline: "none",
                            boxSizing: "border-box",
                        }}
                    />
                </div>
                <button
                    onClick={handleQuickSearch}
                    disabled={!searchInput.trim()}
                    style={{
                        background: searchInput.trim() ? C.amber : C.bg2,
                        color: searchInput.trim() ? "#000" : C.textDim,
                        border: "none", borderRadius: 8, padding: "8px 16px",
                        cursor: searchInput.trim() ? "pointer" : "not-allowed",
                        fontWeight: 700, fontSize: 11, fontFamily: "'Syne',sans-serif",
                        transition: "all .15s",
                    }}
                >Analyse</button>
                <span style={{ fontSize: 10, color: C.textDim }}>( Loads any ticker without changing your watchlist )</span>
            </div>

            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text }}>{selectedTicker}</h1>
                    <span style={{
                        background: changePct >= 0 ? C.green + "22" : C.red + "22", color: changePct >= 0 ? C.green : C.red,
                        padding: "4px 10px", borderRadius: 6, fontSize: 12, fontWeight: 700
                    }}>
                        {changePct >= 0 ? "+" : ""}{changePct.toFixed(2)}%
                    </span>
                    <span style={{ background: C.amberDim, color: C.amber, padding: "4px 10px", borderRadius: 6, fontSize: 11 }}>
                        via {dataSource.replace("_", " ")}
                    </span>
                </div>
                <button
                    onClick={() => setShowDetails(true)}
                    style={{
                        background: C.bg3, color: C.text, border: `1px solid ${C.border}`,
                        borderRadius: 8, padding: "8px 16px", cursor: "pointer",
                        fontWeight: 700, fontSize: 12, fontFamily: "'Syne',sans-serif",
                        display: "flex", alignItems: "center", gap: 8, transition: "all .15s"
                    }}
                >
                    <span>📈</span> Chart Details
                </button>
            </div>
            {/* Last close */}
            <div style={{ fontSize: 11, color: C.textDim, marginBottom: 16 }}>
                Last close: <b style={{ color: C.text }}>${last.close.toFixed(2)}</b>
            </div>

            {/* Market Session Panel */}
            <MarketSessionPanel symbol={selectedTicker} source={dataSource} />

            {/* Stat cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 12, margin: "20px 0" }}>
                <StatCard label="PRICE" value={`$${last.close.toFixed(2)}`}
                    sub={`${change >= 0 ? "+" : ""}$${change.toFixed(2)} today`} color={change >= 0 ? C.green : C.red} />
                <StatCard label="SMA 20" value={last.sma20 ? `$${last.sma20.toFixed(2)}` : "—"}
                    sub={last.close > (last.sma20 || 0) ? "↑ Above SMA" : "↓ Below SMA"} color={C.cyan} />
                <StatCard label="RSI (14)" value={last.rsi ? last.rsi.toFixed(1) : "—"}
                    sub={last.rsi > 70 ? "Overbought" : last.rsi < 30 ? "Oversold" : "Neutral"}
                    color={last.rsi > 70 ? C.red : last.rsi < 30 ? C.green : C.amber} />
                <StatCard label="EMA 12" value={last.ema12 ? `$${last.ema12.toFixed(2)}` : "—"}
                    sub={last.close > (last.ema12 || 0) ? "↑ Above EMA" : "↓ Below EMA"} color={C.green} />
                <StatCard label="ATR" value={last.atr ? last.atr.toFixed(2) : "—"} sub="Volatility" color={C.purple} />
                <StatCard label="VOLUME" value={last.volume ? (last.volume >= 1e6 ? (last.volume / 1e6).toFixed(2) + "M" : last.volume.toLocaleString()) : "—"}
                    sub="Shares Traded" color={C.amber} />
            </div>

            {/* Time range + overlay toggles */}
            <div style={{ display: "flex", gap: 6, marginBottom: 12, justifyContent: "flex-end" }}>
                {[30, 60, 90, 120].map(d => (
                    <button key={d} onClick={() => setTimeRange(d)} style={{
                        background: timeRange === d ? C.amber : C.bg2, color: timeRange === d ? "#000" : C.textMid,
                        border: "none", borderRadius: 6, padding: "6px 14px", fontSize: 11, cursor: "pointer", fontWeight: 700,
                    }}>{d}D</button>
                ))}
            </div>
            <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
                {[
                    { label: "SMA 20", active: showSMA, set: setShowSMA },
                    { label: "EMA 12", active: showEMA, set: setShowEMA },
                    { label: "Volume", active: showVolume, set: setShowVolume },
                ].map(o => (
                    <button key={o.label} onClick={() => o.set(!o.active)} style={{
                        background: o.active ? C.amberDim : C.bg2, border: `1px solid ${o.active ? C.amber + "55" : C.border}`,
                        color: o.active ? C.amber : C.textMid, borderRadius: 20, padding: "5px 14px", fontSize: 11,
                        cursor: "pointer", transition: "all .2s",
                    }}>{o.label}</button>
                ))}
                <Hint text="Toggle indicators" />
            </div>

            {/* Price Chart */}
            <Section title={`PRICE CHART — ${selectedTicker}`}>
                <ResponsiveContainer width="100%" height={320}>
                    <ComposedChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 10 }} />
                        <YAxis tick={{ fill: C.textDim, fontSize: 10 }} domain={["auto", "auto"]}
                            tickFormatter={v => `$${v}`} />
                        <Tooltip content={<ChartTooltip />} />
                        {showSMA && <Line type="monotone" dataKey="sma20" stroke={C.cyan} strokeWidth={1.5} dot={false} strokeDasharray="6 3" />}
                        {showEMA && <Line type="monotone" dataKey="ema12" stroke={C.green} strokeWidth={1.5} dot={false} />}
                        <Line type="monotone" dataKey="close" stroke={C.amber} strokeWidth={2} dot={false} />
                        {showVolume && <Bar dataKey="volume" fill={C.amber + "22"} yAxisId="vol" />}
                        <YAxis yAxisId="vol" orientation="right" hide domain={[0, d => d * 5]} />
                    </ComposedChart>
                </ResponsiveContainer>
            </Section>

            {/* RSI & EMA 12 */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
                <Section title="RSI (14)">
                    <ResponsiveContainer width="100%" height={160}>
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                            <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 9 }} />
                            <YAxis domain={[0, 100]} tick={{ fill: C.textDim, fontSize: 9 }} />
                            <ReferenceLine y={70} stroke={C.red} strokeDasharray="3 3" label={{ value: "OB", fill: C.red, fontSize: 9 }} />
                            <ReferenceLine y={30} stroke={C.green} strokeDasharray="3 3" label={{ value: "OS", fill: C.green, fontSize: 9 }} />
                            <Line type="monotone" dataKey="rsi" stroke={C.amber} strokeWidth={1.5} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </Section>
                <Section title="EMA 12 vs Price">
                    <ResponsiveContainer width="100%" height={160}>
                        <ComposedChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                            <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 9 }} />
                            <YAxis tick={{ fill: C.textDim, fontSize: 9 }} domain={["auto", "auto"]} tickFormatter={v => `$${v}`} />
                            <Line type="monotone" dataKey="close" stroke={C.amber} strokeWidth={1} dot={false} strokeOpacity={0.4} />
                            <Line type="monotone" dataKey="ema12" stroke={C.green} strokeWidth={1.5} dot={false} />
                        </ComposedChart>
                    </ResponsiveContainer>
                </Section>
            </div>

            {/* Chart Usage Guide */}
            <div style={{
                background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 8,
                padding: "12px 16px", fontSize: 11, color: C.textDim, lineHeight: 1.5,
                marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16,
            }}>
                <div><strong style={{color: C.text}}>📉 Price Chart:</strong> Tracks overall trend. SMA indicates direction; volume confirms price moves.</div>
                <div><strong style={{color: C.text}}>⚡ RSI (14):</strong> Measures momentum. &gt;70 means Overbought (due to drop), &lt;30 means Oversold (due to bounce).</div>
                <div><strong style={{color: C.text}}>🎯 EMA 12 vs Price:</strong> Fast trend gauge. Price bouncing above the green line signals a strong short-term uptrend.</div>
            </div>

            {/* ── Indicator Sentiment Panel ────────────────────── */}
            <IndicatorSentimentPanel data={sentiment} latestBar={last} loading={sentimentLoading} error={sentimentError} />
        </div>
    );
}
