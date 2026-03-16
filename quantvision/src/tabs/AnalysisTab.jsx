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

// ── Sentiment Panel ────────────────────────────────────────────────────────────
const SIGNAL_LABELS = {
    "volume.vwap_z": { label: "VWAP Deviation", icon: "📊", category: "Volume" },
    "volume.cmf_20": { label: "Chaikin Money Flow", icon: "💰", category: "Volume" },
    "volume.obv_divergence": { label: "OBV Divergence", icon: "📉", category: "Volume" },
    "momentum.rsi_divergence": { label: "RSI Divergence", icon: "⚡", category: "Momentum" },
    "momentum.macd_exhaust": { label: "MACD Exhaustion", icon: "🔄", category: "Momentum" },
    "momentum.roc_z": { label: "ROC Z-Score", icon: "🚀", category: "Momentum" },
    "micro.ofi": { label: "Order Flow Imbalance", icon: "🔬", category: "Microstructure" },
    "vol.iv_rank": { label: "IV Rank", icon: "😱", category: "Volatility" },
    "vol.term_slope": { label: "VIX Term Structure", icon: "📐", category: "Volatility" },
    "options.pcr_5d": { label: "Put/Call Ratio", icon: "⚖️", category: "Options" },
    "breadth.mcclellan": { label: "McClellan Osc.", icon: "📡", category: "Breadth" },
    "positioning.cot_z": { label: "COT Net Position", icon: "🏛️", category: "Positioning" },
};

function scoreColor(score) {
    if (score > 0.3) return C.green;
    if (score > 0.1) return "#4ade80";
    if (score < -0.3) return C.red;
    if (score < -0.1) return "#fb923c";
    return C.amber;
}

function regimeBadge(regime) {
    const map = {
        risk_on: { bg: C.green + "22", color: C.green, label: "🟢 Risk-On (Contango)" },
        risk_off: { bg: C.red + "22", color: C.red, label: "🔴 Risk-Off (Backwardation)" },
        neutral: { bg: C.amber + "22", color: C.amber, label: "🟡 Neutral" },
    };
    const m = map[regime] || map.neutral;
    return (
        <span style={{
            background: m.bg, color: m.color, borderRadius: 6,
            padding: "4px 10px", fontSize: 10, fontWeight: 700,
            border: `1px solid ${m.color}33`,
        }}>{m.label}</span>
    );
}

function SentimentPanel({ data, loading, error }) {
    if (loading) return (
        <Section style={{ marginTop: 24 }}>
            <div style={{ color: C.textDim, fontSize: 12, padding: "20px 0", textAlign: "center" }}>
                ⏳ Computing sentiment signals…
            </div>
        </Section>
    );
    if (error) return (
        <Section style={{ marginTop: 24 }}>
            <div style={{ color: C.red, fontSize: 12, padding: "16px 0", textAlign: "center" }}>
                ⚠ Sentiment unavailable — {error}
            </div>
        </Section>
    );
    if (!data) return null;

    const sc = data.composite_score || 0;
    const clr = scoreColor(sc);
    const pct = ((sc + 1) / 2) * 100; // map [-1,+1] to [0,100] for the bar
    const activeSignals = (data.signals || []).filter(s => !s.is_stale);
    const staleSignals = (data.signals || []).filter(s => s.is_stale);

    return (
        <div className="fade-up" style={{ marginTop: 24 }}>
            {/* Header */}
            <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                marginBottom: 12,
            }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 20 }}>🧠</span>
                    <span style={{
                        fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 14,
                        color: C.text, letterSpacing: 1, textTransform: "uppercase",
                    }}>Technical Sentiment</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    {regimeBadge(data.regime)}
                    <span style={{
                        background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 6,
                        padding: "4px 10px", fontSize: 10, color: C.textMid,
                    }}>
                        {data.active_signals}/{data.total_signals} signals active
                    </span>
                </div>
            </div>

            {/* Composite Score Card */}
            <div style={{
                background: `linear-gradient(135deg, ${C.bg2}, ${C.bg3})`,
                border: `1px solid ${clr}33`,
                borderRadius: 12, padding: "20px 24px", marginBottom: 16,
            }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <div>
                        <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1.5, marginBottom: 6, fontFamily: "'Syne',sans-serif" }}>
                            COMPOSITE SCORE
                        </div>
                        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
                            <span style={{
                                color: clr, fontSize: 36, fontWeight: 800,
                                fontFamily: "'DM Mono',monospace", lineHeight: 1,
                            }}>
                                {sc >= 0 ? "+" : ""}{sc.toFixed(3)}
                            </span>
                            <span style={{
                                background: clr + "22", color: clr, borderRadius: 6,
                                padding: "4px 12px", fontSize: 12, fontWeight: 700,
                            }}>
                                {data.interpretation}
                            </span>
                        </div>
                    </div>
                    {/* Mini gauge bar */}
                    <div style={{ width: 180 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.textDim, marginBottom: 4 }}>
                            <span>Bearish</span><span>Neutral</span><span>Bullish</span>
                        </div>
                        <div style={{
                            background: C.bg1, borderRadius: 6, height: 8, position: "relative",
                            overflow: "hidden", border: `1px solid ${C.border}`,
                        }}>
                            {/* gradient background */}
                            <div style={{
                                position: "absolute", inset: 0,
                                background: `linear-gradient(90deg, ${C.red}55, ${C.amber}55, ${C.green}55)`,
                                borderRadius: 6,
                            }} />
                            {/* needle */}
                            <div style={{
                                position: "absolute", top: -2, width: 4, height: 12,
                                background: clr, borderRadius: 2,
                                left: `calc(${pct}% - 2px)`,
                                boxShadow: `0 0 6px ${clr}`,
                                transition: "left .5s ease",
                            }} />
                        </div>
                    </div>
                </div>
            </div>

            {/* Active Signals Grid */}
            {activeSignals.length > 0 && (
                <div style={{
                    display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10,
                    marginBottom: 16,
                }}>
                    {activeSignals.map(s => {
                        const meta = SIGNAL_LABELS[s.name] || { label: s.name, icon: "📌", category: "Other" };
                        const norm = s.normalised;
                        const barClr = norm > 0.1 ? C.green : norm < -0.1 ? C.red : C.amber;
                        const barW = Math.abs(norm) * 100;
                        return (
                            <div key={s.name} style={{
                                background: C.bg2, border: `1px solid ${C.border}`,
                                borderRadius: 10, padding: "12px 14px",
                                transition: "border-color .2s",
                            }}>
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                        <span style={{ fontSize: 14 }}>{meta.icon}</span>
                                        <span style={{ color: C.text, fontSize: 11, fontWeight: 700 }}>{meta.label}</span>
                                    </div>
                                    <span style={{
                                        color: C.textDim, fontSize: 9, background: C.bg3,
                                        borderRadius: 4, padding: "2px 6px",
                                    }}>{meta.category}</span>
                                </div>
                                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                                    <span style={{
                                        color: barClr, fontSize: 16, fontWeight: 800,
                                        fontFamily: "'DM Mono',monospace", minWidth: 50,
                                    }}>
                                        {norm >= 0 ? "+" : ""}{norm.toFixed(3)}
                                    </span>
                                    <div style={{
                                        flex: 1, height: 5, background: C.bg1, borderRadius: 3,
                                        overflow: "hidden", position: "relative",
                                    }}>
                                        <div style={{
                                            position: "absolute",
                                            left: norm >= 0 ? "50%" : `${50 - barW / 2}%`,
                                            width: `${barW / 2}%`,
                                            height: "100%", background: barClr, borderRadius: 3,
                                            transition: "all .4s ease",
                                        }} />
                                    </div>
                                </div>
                                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.textDim }}>
                                    <span>z: {s.z_score.toFixed(2)}</span>
                                    <span>raw: {s.value.toFixed(4)}</span>
                                    <span>conf: {(s.confidence * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Stale / Pending Signals */}
            {staleSignals.length > 0 && (
                <div style={{
                    background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 10,
                    padding: "12px 16px",
                }}>
                    <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1, marginBottom: 8, fontFamily: "'Syne',sans-serif" }}>
                        PENDING DATA FEEDS ({staleSignals.length})
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                        {staleSignals.map(s => {
                            const meta = SIGNAL_LABELS[s.name] || { label: s.name, icon: "📌" };
                            return (
                                <span key={s.name} style={{
                                    background: C.bg3, border: `1px solid ${C.border}`,
                                    borderRadius: 6, padding: "4px 10px", fontSize: 10,
                                    color: C.textDim, display: "flex", alignItems: "center", gap: 4,
                                }}>
                                    <span>{meta.icon}</span> {meta.label}
                                    <span style={{ color: C.amber, fontSize: 9 }}>⏳</span>
                                </span>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}

// ── Analysis Tab ───────────────────────────────────────────────────────────────
export default function AnalysisTab({ selectedTicker, setSelectedTicker, priceData, indicatorData, dataSource, apiConnected }) {
    const [showDetails, setShowDetails] = useState(false);
    const [showBB, setShowBB] = useState(true);
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
                <span style={{ fontSize: 10, color: C.textDim }}>( Loads any ticker without changing your watchlist )</span>
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
            {/* Market Session Panel */}
            <div style={{ fontSize: 11, color: C.textDim, marginBottom: 16 }}>
                Last close: <b style={{ color: C.text }}>${last.close.toFixed(2)}</b>
            </div>

            {/* Market Session Panel */}
            <MarketSessionPanel symbol={selectedTicker} source={dataSource} />

            {/* Stat cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 12, margin: "20px 0" }}>
                <StatCard label="PRICE" value={`$${last.close.toFixed(2)}`}
                    sub={`${change >= 0 ? "+" : ""}$${change.toFixed(2)} today`} color={change >= 0 ? C.green : C.red} />
                <StatCard label="SMA 20" value={last.sma20 ? `$${last.sma20.toFixed(2)}` : "—"}
                    sub={last.close > (last.sma20 || 0) ? "↑ Above SMA" : "↓ Below SMA"} color={C.cyan} />
                <StatCard label="RSI (14)" value={last.rsi ? last.rsi.toFixed(1) : "—"}
                    sub={last.rsi > 70 ? "Overbought" : last.rsi < 30 ? "Oversold" : "Neutral"}
                    color={last.rsi > 70 ? C.red : last.rsi < 30 ? C.green : C.amber} />
                <StatCard label="MACD" value={last.macd ? last.macd.toFixed(2) : "—"}
                    sub={last.macd > 0 ? "Bullish" : "Bearish"} color={last.macd > 0 ? C.green : C.red} />
                <StatCard label="ATR" value={last.atr ? last.atr.toFixed(2) : "—"} sub="Volatility" color={C.purple} />
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
                    { label: "Bollinger Bands", active: showBB, set: setShowBB },
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
                        {showBB && <Area type="monotone" dataKey="bbUpper" stroke="none" fill={C.purple + "15"} />}
                        {showBB && <Area type="monotone" dataKey="bbLower" stroke="none" fill={C.bg1} />}
                        {showBB && <Line type="monotone" dataKey="bbUpper" stroke={C.purple} strokeWidth={1} dot={false} strokeDasharray="4 4" />}
                        {showBB && <Line type="monotone" dataKey="bbLower" stroke={C.purple} strokeWidth={1} dot={false} strokeDasharray="4 4" />}
                        {showSMA && <Line type="monotone" dataKey="sma20" stroke={C.cyan} strokeWidth={1.5} dot={false} strokeDasharray="6 3" />}
                        {showEMA && <Line type="monotone" dataKey="ema12" stroke={C.green} strokeWidth={1.5} dot={false} />}
                        <Line type="monotone" dataKey="close" stroke={C.amber} strokeWidth={2} dot={false} />
                        {showVolume && <Bar dataKey="volume" fill={C.amber + "22"} yAxisId="vol" />}
                        <YAxis yAxisId="vol" orientation="right" hide domain={[0, d => d * 5]} />
                    </ComposedChart>
                </ResponsiveContainer>
            </Section>

            {/* RSI & MACD */}
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
                <Section title="MACD">
                    <ResponsiveContainer width="100%" height={160}>
                        <ComposedChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                            <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 9 }} />
                            <YAxis tick={{ fill: C.textDim, fontSize: 9 }} />
                            <ReferenceLine y={0} stroke={C.border} />
                            <Bar dataKey="macdHist" fill={C.amber + "55"}>
                                {chartData.map((d, i) => (
                                    <rect key={i} fill={(d.macdHist || 0) >= 0 ? C.green + "88" : C.red + "88"} />
                                ))}
                            </Bar>
                            <Line type="monotone" dataKey="macd" stroke={C.amber} strokeWidth={1.5} dot={false} />
                            <Line type="monotone" dataKey="macdSig" stroke={C.red} strokeWidth={1} dot={false} />
                        </ComposedChart>
                </ResponsiveContainer>
            </Section>
            </div>

            {/* ── Technical Sentiment Signal Panel ──────────────────── */}
            <SentimentPanel data={sentiment} loading={sentimentLoading} error={sentimentError} />
        </div >
    );
}
