import { useState, useEffect } from "react";
import {
    LineChart, Line, AreaChart, Area, ComposedChart,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ReferenceLine, ReferenceArea, Legend
} from "recharts";
import { C } from "../utils/data";
import { fetchPredictions, triggerTraining } from "../utils/api";
import { ChartTooltip, StatCard, Section } from "../components/UIComponents";
import TradingViewDetail from "../components/TradingViewDetail";

// Palette of colours for scenario fan paths
const SCENARIO_COLORS = [
    "#e8a83855", "#e8a83845", "#e8a83835", "#e8a83830",
    "#e8a83828", "#e8a83822", "#e8a83820", "#e8a83818",
    "#e8a83815", "#e8a83812", "#e8a83810", "#e8a83808",
];

// Custom Legend for Forecast Charts
const ForecastLegend = () => {
    return (
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 16, fontSize: 11, marginBottom: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 12, height: 12, background: C.textDim, borderRadius: 2 }} />
                <span style={{ color: C.textDim }}>Historical</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 16, height: 2, background: C.amber }} />
                <span style={{ color: C.textMid }}>Predicted (Median)</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 12, height: 12, background: C.amber + "10", borderTop: `1px dashed ${C.amber}44`, borderBottom: `1px dashed ${C.amber}44` }} />
                <span style={{ color: C.textMid }}>90% CI</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 12, height: 12, background: C.amber + "18" }} />
                <span style={{ color: C.textMid }}>50% CI</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 12, height: 3, background: C.amber + "30", borderRadius: 2 }} />
                <span style={{ color: C.textMid }}>Scenarios</span>
            </div>
        </div>
    );
};

function formatChartDateLabel(date) {
    return typeof date === "string" ? date.slice(5) : date;
}

function formatMetaLabel(value) {
    if (!value) return "—";
    return String(value)
        .replaceAll("_", " ")
        .replaceAll("-", " ")
        .replace(/\b\w/g, m => m.toUpperCase());
}

export default function PredictionsTab({ selectedTicker, apiConnected, priceData }) {
    const [showDetails, setShowDetails] = useState(false);
    const [horizon, setHorizon] = useState(30);
    const [modelType, setModelType] = useState("xgboost");
    const [predData, setPredData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showScenarios, setShowScenarios] = useState(true);
    const [trainingState, setTrainingState] = useState({ loading: false, message: null, error: null });

    useEffect(() => {
        if (!apiConnected) return;
        setLoading(true); setError(null);
        setTrainingState(prev => ({ ...prev, error: null }));
        fetchPredictions(selectedTicker, modelType, horizon)
            .then(d => { setPredData(d); setLoading(false); })
            .catch(e => { setError(e.message); setLoading(false); });
    }, [selectedTicker, modelType, horizon, apiConnected]);

    async function handleTrainModel() {
        setTrainingState({ loading: true, message: null, error: null });
        try {
            const result = await triggerTraining({
                symbol: selectedTicker,
                model_type: modelType,
                horizons: [1, 7, 15, 30, 60],
            });
            setTrainingState({
                loading: false,
                message: result?.message || `Training started for ${selectedTicker} ${modelType}.`,
                error: null,
            });
        } catch (trainError) {
            setTrainingState({
                loading: false,
                message: null,
                error: trainError?.message || "Unable to start training.",
            });
        }
    }

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
    const scenarioPaths = predData?.scenario_paths || [];
    const hasForecasts = forecasts.length > 0;
    const canTrain = Boolean(predData?.can_train ?? modelInfo.can_train);
    const requestedModelLabel = formatMetaLabel(modelInfo.requested_model || modelType);
    const servingModelLabel = formatMetaLabel(modelInfo.serving_model || modelType);
    const forecastEngineLabel = formatMetaLabel(modelInfo.forecast_engine || modelInfo.method);
    const uncertaintyLabel = formatMetaLabel(modelInfo.uncertainty_method);
    const artifactSourceLabel = formatMetaLabel(modelInfo.artifact_source || modelInfo.source);
    const unavailableMessage = !error && !hasForecasts && modelInfo.status === "unavailable"
        ? (predData?.message || modelInfo.message || `No trained ${modelType} bundle found for ${selectedTicker}.`)
        : null;

    // Build seamless chart dataset
    // 1. Take last 60 historical bars if available
    // 2. Append forecasts with scenario path data merged in
    const historyBars = priceData?.bars?.slice(-60) || [];
    const chartData = [];

    let forecastBoundaryDate = null;

    // Process history
    historyBars.forEach((b, index) => {
        const isLastHistoryBar = index === historyBars.length - 1;
        const shouldBridgeForecast = isLastHistoryBar && forecasts.length > 0;
        const point = {
            date: b.date,
            historical: b.close,
            predicted: shouldBridgeForecast ? b.close : null,
            upper95: shouldBridgeForecast ? b.close : null,
            lower95: shouldBridgeForecast ? b.close : null,
            upper68: shouldBridgeForecast ? b.close : null,
            lower68: shouldBridgeForecast ? b.close : null,
        };
        // Scenario paths: set starting price at bridge point
        if (shouldBridgeForecast && scenarioPaths.length > 0) {
            scenarioPaths.forEach((_, si) => {
                point[`s${si}`] = b.close;
            });
        }
        chartData.push(point);
    });

    // Keep a boundary marker
    if (historyBars.length > 0 && forecasts.length > 0) {
        const lastHist = historyBars[historyBars.length - 1];
        forecastBoundaryDate = lastHist.date;
    }

    // Process forecast — merge scenario path values into chart data
    forecasts.forEach((f, i) => {
        const fd = f.date || `forecast-${i + 1}`;
        if (!forecastBoundaryDate && i === 0) forecastBoundaryDate = fd;
        const point = {
            date: fd,
            historical: null,
            predicted: f.predicted,
            upper95: f.upper95,
            lower95: f.lower95,
            upper68: f.upper68,
            lower68: f.lower68,
        };
        // Scenario paths: each path[si] has values [startPrice, step1, step2, ...]
        // so path index i+1 corresponds to step i (index 0 is the starting price)
        if (scenarioPaths.length > 0) {
            scenarioPaths.forEach((path, si) => {
                point[`s${si}`] = path[i + 1] != null ? path[i + 1] : null;
            });
        }
        chartData.push(point);
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
                        Requested: {requestedModelLabel} · Serving: {servingModelLabel} · Source: {artifactSourceLabel}
                        {modelInfo.n_scenarios ? ` · ${modelInfo.n_scenarios} scenarios` : ""}
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

                    {/* Scenario paths toggle */}
                    {scenarioPaths.length > 0 && (
                        <button
                            onClick={() => setShowScenarios(!showScenarios)}
                            style={{
                                background: showScenarios ? C.amber + "33" : C.bg2,
                                color: showScenarios ? C.amber : C.textMid,
                                border: `1px solid ${showScenarios ? C.amber + "66" : C.border}`,
                                borderRadius: 6, padding: "6px 12px", fontSize: 10, cursor: "pointer",
                                fontWeight: 600, fontFamily: "'DM Mono',monospace",
                            }}
                        >
                            {showScenarios ? "◉" : "○"} Scenarios
                        </button>
                    )}

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
            {unavailableMessage && (
                <div style={{
                    background: C.amber + "15",
                    border: `1px solid ${C.amber}44`,
                    color: C.amber,
                    padding: 12,
                    borderRadius: 8,
                    marginBottom: 16,
                    fontSize: 12,
                }}>
                    <div>{unavailableMessage}</div>
                    {canTrain && (
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
                            <button
                                onClick={handleTrainModel}
                                disabled={trainingState.loading}
                                style={{
                                    background: C.amber,
                                    color: "#000",
                                    border: "none",
                                    borderRadius: 6,
                                    padding: "7px 12px",
                                    fontSize: 11,
                                    cursor: trainingState.loading ? "default" : "pointer",
                                    fontWeight: 700,
                                }}
                            >
                                {trainingState.loading ? "Starting Training..." : `Train ${requestedModelLabel}`}
                            </button>
                            {trainingState.message && <span style={{ color: C.textMid }}>{trainingState.message}</span>}
                            {trainingState.error && <span style={{ color: C.red }}>{trainingState.error}</span>}
                        </div>
                    )}
                </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 }}>
                <StatCard
                    label="REQUESTED MODEL"
                    value={requestedModelLabel}
                    sub={`Serving ${servingModelLabel}`}
                    color={C.text}
                />
                <StatCard
                    label="FORECAST ENGINE"
                    value={forecastEngineLabel}
                    sub={uncertaintyLabel}
                    color={C.amber}
                />
                <StatCard
                    label="ARTIFACT SOURCE"
                    value={artifactSourceLabel}
                    sub={modelInfo.bundle_version || "No bundle version"}
                    color={C.textMid}
                />
                <StatCard
                    label="BUNDLE STATUS"
                    value={modelInfo.model_available === false ? "Missing" : "Ready"}
                    sub={modelInfo.trained_at ? `Trained ${String(modelInfo.trained_at).slice(0, 10)}` : "Awaiting bundle"}
                    color={modelInfo.model_available === false ? C.red : C.green}
                />
            </div>

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
                <ResponsiveContainer width="100%" height={400}>
                    <ComposedChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 10 }} tickFormatter={formatChartDateLabel} />
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

                        {/* Confidence band fills */}
                        <Area type="monotone" dataKey="upper95" stroke="none" fill={C.amber + "10"} isAnimationActive={false} />
                        <Area type="monotone" dataKey="lower95" stroke="none" fill={C.bg1} isAnimationActive={false} />
                        <Area type="monotone" dataKey="upper68" stroke="none" fill={C.amber + "18"} isAnimationActive={false} />
                        <Area type="monotone" dataKey="lower68" stroke="none" fill={C.bg1} isAnimationActive={false} />

                        {/* Scenario fan paths — thin semi-transparent lines */}
                        {showScenarios && scenarioPaths.map((_, si) => (
                            <Line
                                key={`scenario-${si}`}
                                type="monotone"
                                dataKey={`s${si}`}
                                stroke={SCENARIO_COLORS[si % SCENARIO_COLORS.length]}
                                strokeWidth={1}
                                dot={false}
                                isAnimationActive={false}
                                connectNulls={false}
                            />
                        ))}

                        {/* Confidence band outlines */}
                        <Line type="monotone" dataKey="upper95" stroke={C.amber + "44"} strokeWidth={1} dot={false} strokeDasharray="4 4" isAnimationActive={false} />
                        <Line type="monotone" dataKey="lower95" stroke={C.amber + "44"} strokeWidth={1} dot={false} strokeDasharray="4 4" isAnimationActive={false} />

                        {/* Main lines */}
                        <Line type="monotone" dataKey="historical" stroke={C.textDim} strokeWidth={2} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="predicted" stroke={C.amber} strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    </ComposedChart>
                </ResponsiveContainer>
                {!hasForecasts && (
                    <div style={{ marginTop: 10, color: C.textDim, fontSize: 11 }}>
                        No forecast points were returned for this ticker and model selection.
                    </div>
                )}
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
                            {!hasForecasts && (
                                <tr>
                                    <td colSpan={5} style={{ padding: "16px 12px", color: C.textDim, textAlign: "center" }}>
                                        No forecast rows returned by the API.
                                    </td>
                                </tr>
                            )}
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

