import { useState, useMemo } from "react";
import {
    LineChart, Line, AreaChart, Area, ComposedChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    ReferenceLine
} from "recharts";
import { C } from "../utils/data";
import { ChartTooltip, StatCard, Section, Hint } from "../components/UIComponents";

export default function AnalysisTab({ selectedTicker, priceData, indicatorData, dataSource, apiConnected }) {
    const [showBB, setShowBB] = useState(true);
    const [showSMA, setShowSMA] = useState(true);
    const [showEMA, setShowEMA] = useState(false);
    const [showVolume, setShowVolume] = useState(true);
    const [timeRange, setTimeRange] = useState(60);

    // Build chart data from API response
    const chartData = useMemo(() => {
        if (!priceData?.bars || !indicatorData?.data) return [];
        const prices = priceData.bars.slice(-timeRange);
        const indMap = {};
        (indicatorData.data || []).forEach(d => { indMap[d.date] = d; });
        return prices.map(bar => {
            const ind = indMap[bar.date] || {};
            return {
                date: bar.date.slice(5), // MM-DD
                close: bar.close,
                open: bar.open,
                high: bar.high,
                low: bar.low,
                volume: bar.volume,
                sma20: ind.SMA_20,
                ema12: ind.EMA_12,
                rsi: ind.RSI,
                macd: ind.MACD,
                macdSig: ind.MACD_Signal,
                macdHist: ind.MACD_Histogram,
                bbUpper: ind.BB_High,
                bbMid: ind.BB_Mid,
                bbLower: ind.BB_Low,
                atr: ind.ATR,
                obv: ind.OBV,
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
    const fundamentals = { pe: "—", mktCap: "—", beta: "—" };

    return (
        <div className="fade-up">
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16 }}>
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
            <div style={{ fontSize: 11, color: C.textDim, marginBottom: 20 }}>
                Last close: <b style={{ color: C.text }}>${last.close.toFixed(2)}</b>
            </div>

            {/* Time range buttons */}
            <div style={{ display: "flex", gap: 6, marginBottom: 16, justifyContent: "flex-end" }}>
                {[30, 60, 90, 120].map(d => (
                    <button key={d} onClick={() => setTimeRange(d)} style={{
                        background: timeRange === d ? C.amber : C.bg2, color: timeRange === d ? "#000" : C.textMid,
                        border: "none", borderRadius: 6, padding: "6px 14px", fontSize: 11, cursor: "pointer", fontWeight: 700,
                    }}>{d}D</button>
                ))}
            </div>

            {/* Stat cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 12, marginBottom: 20 }}>
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

            {/* Overlay toggles */}
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
        </div>
    );
}
