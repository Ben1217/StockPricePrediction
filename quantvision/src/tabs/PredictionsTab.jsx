import { useState, useEffect } from "react";
import {
    LineChart, Line, AreaChart, Area, ComposedChart,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ReferenceLine, ReferenceArea, Legend
} from "recharts";
import { C } from "../utils/data";
import { fetchPredictions } from "../utils/api";
import { ChartTooltip, StatCard, Section } from "../components/UIComponents";
import TradingViewDetail from "../components/TradingViewDetail";

// Custom Legend for Forecast Charts
const ForecastLegend = ({ payload }) => {
    return (
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 16, fontSize: 11, marginBottom: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 12, height: 12, background: C.textDim, borderRadius: 2 }} />
                <span style={{ color: C.textDim }}>Historical</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 16, height: 2, background: C.amber }} />
                <span style={{ color: C.textMid }}>Predicted</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 12, height: 12, background: C.amber + "10", borderTop: `1px dashed ${C.amber}44`, borderBottom: `1px dashed ${C.amber}44` }} />
                <span style={{ color: C.textMid }}>95% CI</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 12, height: 12, background: C.amber + "18" }} />
                <span style={{ color: C.textMid }}>68% CI</span>
            </div>
        </div>
    );
};

export default function PredictionsTab({ selectedTicker, apiConnected, priceData }) {
    const [showDetails, setShowDetails] = useState(false);
    const [horizon, setHorizon] = useState(30);
    const [modelType, setModelType] = useState("xgboost");
    const [predData, setPredData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!apiConnected) return;
        setLoading(true); setError(null);
        fetchPredictions(selectedTicker, modelType, horizon)
            .then(d => { setPredData(d); setLoading(false); })
            .catch(e => { setError(e.message); setLoading(false); });
    }, [selectedTicker, modelType, horizon, apiConnected]);

    if (!apiConnected) {
        return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>🔌</div>
            <div>Connect to API server for predictions</div>
        </div>;
    }
    if (loading) return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>Loading predictions...</div>;

    const forecasts = predData?.forecasts || [];
    const currentPrice = predData?.current_price || 0;
    const lastPred = forecasts[forecasts.length - 1] || {};
    const predChange = lastPred.predicted ? ((lastPred.predicted - currentPrice) / currentPrice * 100) : 0;
    const modelInfo = predData?.model_info || {};

    // Build seamless chart dataset
    // 1. Take last 60 historical bars if available
    // 2. Append forecasts
    const historyBars = priceData?.bars?.slice(-60) || [];
    const chartData = [];

    let forecastBoundaryDate = null;

    // Process history
    historyBars.forEach(b => {
        chartData.push({
            date: b.date.slice(5), // MM-DD
            historical: b.close,
            predicted: null
        });
    });

    // To connect the lines, inject the last historical close into the first forecast properties
    if (historyBars.length > 0 && forecasts.length > 0) {
        const lastHist = historyBars[historyBars.length - 1];
        forecastBoundaryDate = lastHist.date.slice(5);
        chartData.push({
            date: forecastBoundaryDate,
            historical: lastHist.close,
            predicted: lastHist.close,
            upper95: lastHist.close,
            lower95: lastHist.close,
            upper68: lastHist.close,
            lower68: lastHist.close
        });
    }

    // Process forecast
    forecasts.forEach((f, i) => {
        const fd = f.date?.slice(5) || `D${i + 1}`;
        if (!forecastBoundaryDate && i === 0) forecastBoundaryDate = fd;
        chartData.push({
            date: fd,
            historical: null,
            predicted: f.predicted,
            upper95: f.upper95,
            lower95: f.lower95,
            upper68: f.upper68,
            lower68: f.lower68,
        });
    });

    if (showDetails) {
        return <TradingViewDetail symbol={selectedTicker} mode="prediction" predictionData={predData} onClose={() => setShowDetails(false)} />;
    }

    return (
        <div className="fade-up">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
                <div>
                    <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text }}>
                        🤖 AI Price Forecast — {selectedTicker}
                    </h1>
                    <div style={{ fontSize: 11, color: C.textDim, marginTop: 4 }}>
                        Model: {modelInfo.method || modelType} · Source: {modelInfo.source || "—"}
                    </div>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                    <select value={modelType} onChange={e => setModelType(e.target.value)} style={{
                        background: C.bg2, color: C.text, border: `1px solid ${C.border}`, borderRadius: 6,
                        padding: "6px 12px", fontSize: 11, fontFamily: "'DM Mono',monospace",
                    }}>
                        <option value="xgboost">XGBoost</option>
                        <option value="random_forest">Random Forest</option>
                        <option value="lstm">LSTM (PyTorch)</option>
                    </select>
                    {[7, 15, 30, 60].map(h => (
                        <button key={h} onClick={() => setHorizon(h)} style={{
                            background: horizon === h ? C.amber : C.bg2, color: horizon === h ? "#000" : C.textMid,
                            border: "none", borderRadius: 6, padding: "6px 14px", fontSize: 11, cursor: "pointer", fontWeight: 700,
                        }}>{h}D</button>
                    ))}

                    <button
                        onClick={() => setShowDetails(true)}
                        style={{
                            background: C.bg3, color: C.text, border: `1px solid ${C.border}`,
                            borderRadius: 6, padding: "6px 14px", cursor: "pointer", marginLeft: 16,
                            fontWeight: 700, fontSize: 11, fontFamily: "'Syne',sans-serif",
                            display: "flex", alignItems: "center", gap: 6, transition: "all .15s"
                        }}
                    >
                        <span>📈</span> Chart Details
                    </button>
                </div>
            </div>

            {error && <div style={{ background: C.red + "22", color: C.red, padding: 12, borderRadius: 8, marginBottom: 16, fontSize: 12 }}>{error}</div>}

            {/* Stats */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 }}>
                <StatCard label="CURRENT" value={`$${currentPrice.toFixed(2)}`} sub="Last close" color={C.text} />
                <StatCard label={`${horizon}D TARGET`} value={lastPred.predicted ? `$${lastPred.predicted.toFixed(2)}` : "—"}
                    sub={`${predChange >= 0 ? "+" : ""}${predChange.toFixed(2)}%`} color={predChange >= 0 ? C.green : C.red} />
                <StatCard label="UPPER 95%" value={lastPred.upper95 ? `$${lastPred.upper95.toFixed(2)}` : "—"} sub="Bull case" color={C.green} />
                <StatCard label="LOWER 95%" value={lastPred.lower95 ? `$${lastPred.lower95.toFixed(2)}` : "—"} sub="Bear case" color={C.red} />
            </div>

            {/* Forecast chart */}
            <Section title={`FORECAST — ${horizon} Day Horizon`}>
                <ResponsiveContainer width="100%" height={360}>
                    <ComposedChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 10 }} />
                        <YAxis tick={{ fill: C.textDim, fontSize: 10 }} domain={["auto", "auto"]} tickFormatter={v => `$${v}`} />
                        <Tooltip content={<ChartTooltip />} />
                        <Legend verticalAlign="top" height={36} content={<ForecastLegend />} />

                        {forecastBoundaryDate && (
                            <ReferenceLine x={forecastBoundaryDate} stroke={C.border} strokeDasharray="3 3"
                                label={{ value: "Forecast Start", fill: C.textDim, fontSize: 10, position: "insideTopLeft", dy: 10 }} />
                        )}
                        {forecastBoundaryDate && (
                            <ReferenceArea x1={forecastBoundaryDate} fill={C.amber} fillOpacity={0.03} />
                        )}

                        <Area type="monotone" dataKey="upper95" stroke="none" fill={C.amber + "10"} isAnimationActive={false} />
                        <Area type="monotone" dataKey="lower95" stroke="none" fill={C.bg1} isAnimationActive={false} />
                        <Area type="monotone" dataKey="upper68" stroke="none" fill={C.amber + "18"} isAnimationActive={false} />
                        <Area type="monotone" dataKey="lower68" stroke="none" fill={C.bg1} isAnimationActive={false} />

                        <Line type="monotone" dataKey="upper95" stroke={C.amber + "44"} strokeWidth={1} dot={false} strokeDasharray="4 4" isAnimationActive={false} />
                        <Line type="monotone" dataKey="lower95" stroke={C.amber + "44"} strokeWidth={1} dot={false} strokeDasharray="4 4" isAnimationActive={false} />
                        <Line type="monotone" dataKey="historical" stroke={C.textDim} strokeWidth={2} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="predicted" stroke={C.amber} strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    </ComposedChart>
                </ResponsiveContainer>
            </Section>

            {/* Forecast Details Table */}
            <Section title="FORECAST BREAKDOWN">
                <div style={{ maxHeight: 300, overflowY: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                        <thead>
                            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                {["Date", "Predicted", "Lower 95%", "Upper 95%", "Range"].map(h => (
                                    <th key={h} style={{ padding: "8px 12px", color: C.textDim, textAlign: "right", fontWeight: 400 }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {forecasts.map((f, i) => (
                                <tr key={i} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                    <td style={{ padding: "6px 12px", color: C.textMid }}>{f.date}</td>
                                    <td style={{ padding: "6px 12px", color: C.amber, textAlign: "right", fontWeight: 700 }}>${f.predicted.toFixed(2)}</td>
                                    <td style={{ padding: "6px 12px", color: C.red, textAlign: "right" }}>${f.lower95.toFixed(2)}</td>
                                    <td style={{ padding: "6px 12px", color: C.green, textAlign: "right" }}>${f.upper95.toFixed(2)}</td>
                                    <td style={{ padding: "6px 12px", color: C.textDim, textAlign: "right" }}>
                                        ${(f.upper95 - f.lower95).toFixed(2)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </Section>
        </div>
    );
}
