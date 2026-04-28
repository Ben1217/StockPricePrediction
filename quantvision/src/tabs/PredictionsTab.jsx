import { useEffect, useMemo, useRef, useState } from "react";
import {
    Area,
    CartesianGrid,
    ComposedChart,
    Line,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { C } from "../utils/data";
import { fetchEnsemblePrediction, getEnsembleTrainingStatus, triggerEnsembleTraining } from "../utils/api";

const HORIZONS = [7, 15, 30, 60];
const MODEL_OPTIONS = [
    { value: "all", label: "All Models" },
    { value: "ensemble", label: "Ensemble" },
    { value: "lstm", label: "LSTM" },
    { value: "xgboost", label: "XGBoost" },
    { value: "random_forest", label: "Random Forest" },
];

const COLORS = {
    historical: "#9CA3AF",
    ensemble: "#F5C842",
    lstm: "#60A5FA",
    xgboost: "#F59E0B",
    random_forest: "#34D399",
    band: "#6366F1",
    surface: "#161B22",
    panel: "#0F1623",
};

const WEIGHT_LABELS = {
    lstm: "PyTorch LSTM",
    xgboost: "XGBoost",
    random_forest: "Random Forest",
};

const MODEL_LABELS = {
    historical: "Historical",
    ensemble: "Ensemble",
    lstm: "LSTM",
    xgboost: "XGBoost",
    random_forest: "Random Forest",
};

function formatPrice(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)}%`;
}

function formatDateLabel(date) {
    if (typeof date !== "string" || date.length < 10) return date;
    return date.slice(5).replace("-", "/");
}

function MetricCard({ label, value, sub, color }) {
    return (
        <div style={{
            background: COLORS.surface,
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: "16px 18px",
            minHeight: 96,
            display: "grid",
            alignContent: "center",
            gap: 7,
        }}>
            <div style={{ fontSize: 11, color: C.textDim, fontWeight: 700, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                {label}
            </div>
            <div style={{ fontSize: 24, lineHeight: 1.1, color: color || C.text, fontWeight: 800 }}>
                {value}
            </div>
            {sub && <div style={{ fontSize: 12, color: C.textMid }}>{sub}</div>}
        </div>
    );
}

function TrainButton({ symbol, onComplete }) {
    const [state, setState] = useState({ loading: false, jobId: null, message: null, error: null });
    const pollRef = useRef(null);

    async function handleTrain() {
        setState({ loading: true, jobId: null, message: "Starting ensemble training...", error: null });
        try {
            const result = await triggerEnsembleTraining(symbol);
            const jobId = result?.job_id;
            setState((s) => ({ ...s, jobId, message: result?.message || "Training started." }));

            pollRef.current = setInterval(async () => {
                try {
                    const status = await getEnsembleTrainingStatus(jobId);
                    if (status.status === "completed") {
                        clearInterval(pollRef.current);
                        setState({ loading: false, jobId, message: "Training complete.", error: null });
                        setTimeout(() => onComplete?.(), 1200);
                    } else if (status.status === "failed") {
                        clearInterval(pollRef.current);
                        setState({ loading: false, jobId, message: null, error: status.error || "Training failed." });
                    }
                } catch {
                    // Keep polling; transient status fetches are common while the API is busy.
                }
            }, 3000);
        } catch (err) {
            setState({ loading: false, jobId: null, message: null, error: err?.message || "Training request failed." });
        }
    }

    useEffect(() => () => clearInterval(pollRef.current), []);

    return (
        <div style={{ display: "grid", gap: 8, justifyItems: "start" }}>
            <button
                type="button"
                onClick={handleTrain}
                disabled={state.loading}
                style={{
                    background: state.loading ? C.bg3 : COLORS.ensemble,
                    color: state.loading ? C.textDim : "#10131A",
                    border: "none",
                    borderRadius: 8,
                    padding: "9px 16px",
                    fontSize: 13,
                    fontWeight: 800,
                    cursor: state.loading ? "default" : "pointer",
                }}
            >
                {state.loading ? "Training in progress..." : `Train ensemble for ${symbol}`}
            </button>
            {state.message && <div style={{ fontSize: 12, color: C.textMid }}>{state.message}</div>}
            {state.error && <div style={{ fontSize: 12, color: C.red }}>{state.error}</div>}
        </div>
    );
}

function TooltipContent({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    const rows = payload
        .filter((item) => MODEL_LABELS[item.dataKey] && item.value !== null && item.value !== undefined)
        .map((item) => ({
            key: item.dataKey,
            label: MODEL_LABELS[item.dataKey],
            value: item.value,
            color: item.color,
        }));
    if (!rows.length) return null;

    return (
        <div style={{
            background: "#0B101A",
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: "10px 12px",
            boxShadow: "0 12px 28px rgba(0,0,0,.28)",
            minWidth: 170,
        }}>
            <div style={{ color: C.text, fontWeight: 700, fontSize: 12, marginBottom: 8 }}>{label}</div>
            <div style={{ display: "grid", gap: 5 }}>
                {rows.map((row) => (
                    <div key={row.key} style={{ display: "flex", justifyContent: "space-between", gap: 18, fontSize: 12 }}>
                        <span style={{ color: row.color }}>{row.label}</span>
                        <span style={{ color: C.text }}>{formatPrice(row.value)}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function ForecastChart({ priceData, forecastPoints, selectedModel }) {
    const { chartData, todayDate } = useMemo(() => {
        const history = (priceData?.bars || []).slice(-60);
        const rows = [];
        let boundary = null;

        history.forEach((bar, index) => {
            const isLast = index === history.length - 1;
            if (isLast) boundary = bar.date;
            rows.push({
                date: bar.date,
                historical: bar.close,
                ensemble: isLast && forecastPoints.length ? bar.close : null,
                lstm: isLast && forecastPoints.length ? bar.close : null,
                xgboost: isLast && forecastPoints.length ? bar.close : null,
                random_forest: isLast && forecastPoints.length ? bar.close : null,
                upper_90: isLast && forecastPoints.length ? bar.close : null,
                lower_90: isLast && forecastPoints.length ? bar.close : null,
            });
        });

        forecastPoints.forEach((point) => {
            rows.push({
                date: point.date,
                historical: null,
                ensemble: point.ensemble,
                lstm: point.lstm,
                xgboost: point.xgboost,
                random_forest: point.random_forest,
                upper_90: point.upper_90,
                lower_90: point.lower_90,
            });
        });

        return { chartData: rows, todayDate: boundary };
    }, [priceData, forecastPoints]);

    const show = (model) => selectedModel === "all" || selectedModel === model;

    return (
        <div style={{ background: COLORS.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 14, fontSize: 12, color: C.textMid }}>
                {[
                    ["historical", "Historical", null],
                    ["ensemble", "Ensemble", null],
                    ["lstm", "LSTM", "5 5"],
                    ["xgboost", "XGBoost", "5 5"],
                    ["random_forest", "Random Forest", "5 5"],
                ].map(([key, label, dash]) => (
                    <div key={key} style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
                        <span style={{ width: 18, height: 0, borderTop: `3px ${dash ? "dashed" : "solid"} ${COLORS[key]}` }} />
                        <span>{label}</span>
                    </div>
                ))}
            </div>
            <ResponsiveContainer width="100%" height={366}>
                <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                    <CartesianGrid stroke={C.border} strokeDasharray="3 3" opacity={0.28} vertical={false} />
                    <XAxis
                        dataKey="date"
                        tick={{ fill: C.textDim, fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={formatDateLabel}
                        minTickGap={22}
                    />
                    <YAxis
                        orientation="right"
                        tick={{ fill: C.textDim, fontSize: 11 }}
                        domain={["auto", "auto"]}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(value) => `$${Number(value).toFixed(0)}`}
                    />
                    <Tooltip content={<TooltipContent />} />
                    <Area
                        type="monotone"
                        dataKey="upper_90"
                        stroke="none"
                        fill={COLORS.band}
                        fillOpacity={0.09}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    <Area
                        type="monotone"
                        dataKey="lower_90"
                        stroke="none"
                        fill={COLORS.panel}
                        fillOpacity={1}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    {todayDate && (
                        <ReferenceLine
                            x={todayDate}
                            stroke={C.textDim}
                            strokeDasharray="4 4"
                            label={{ value: "Today", position: "top", fill: C.textMid, fontSize: 11 }}
                        />
                    )}
                    <Line type="monotone" dataKey="historical" stroke={COLORS.historical} strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line hide={!show("lstm")} type="monotone" dataKey="lstm" stroke={COLORS.lstm} strokeWidth={2} strokeDasharray="6 5" dot={false} connectNulls isAnimationActive={false} />
                    <Line hide={!show("xgboost")} type="monotone" dataKey="xgboost" stroke={COLORS.xgboost} strokeWidth={2} strokeDasharray="6 5" dot={false} connectNulls isAnimationActive={false} />
                    <Line hide={!show("random_forest")} type="monotone" dataKey="random_forest" stroke={COLORS.random_forest} strokeWidth={2} strokeDasharray="6 5" dot={false} connectNulls isAnimationActive={false} />
                    <Line hide={!show("ensemble")} type="monotone" dataKey="ensemble" stroke={COLORS.ensemble} strokeWidth={4} dot={false} connectNulls isAnimationActive={false} />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
}

function WeightsPanel({ weights, consensus, signal, reliability }) {
    const resolved = {
        lstm: Number(weights?.lstm ?? 0.4),
        xgboost: Number(weights?.xgboost ?? 0.35),
        random_forest: Number(weights?.random_forest ?? 0.25),
    };
    const signalColor = signal === "Bearish" ? C.red : signal === "Neutral" ? C.amber : C.green;

    return (
        <div style={{ background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
            <div style={{ fontSize: 13, fontWeight: 800, color: C.text, marginBottom: 16 }}>Ensemble Weights</div>
            <div style={{ display: "grid", gap: 13 }}>
                {["lstm", "xgboost", "random_forest"].map((key) => {
                    const pct = Math.round(resolved[key] * 100);
                    return (
                        <div key={key} style={{ display: "grid", gridTemplateColumns: "128px 1fr 44px", alignItems: "center", gap: 12 }}>
                            <div style={{ color: C.textMid, fontSize: 12, fontWeight: 700 }}>{WEIGHT_LABELS[key]}</div>
                            <div style={{ height: 10, background: "#0A0F18", borderRadius: 999, overflow: "hidden", border: "1px solid rgba(255,255,255,.05)" }}>
                                <div style={{ width: `${pct}%`, height: "100%", background: COLORS[key] }} />
                            </div>
                            <div style={{ color: C.text, fontSize: 12, fontWeight: 800, textAlign: "right" }}>{pct}%</div>
                        </div>
                    );
                })}
            </div>
            <div style={{ marginTop: 18, borderTop: `1px solid ${C.border}`, paddingTop: 14 }}>
                <div style={{ color: signalColor, fontSize: 13, fontWeight: 800 }}>
                    {consensus || `${reliability || "Medium"} reliability`}
                </div>
            </div>
        </div>
    );
}

function ForecastTable({ rows }) {
    const tableRows = (rows || []).slice(0, 7);
    return (
        <div style={{ background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18, overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 620, fontSize: 12 }}>
                <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                        <th style={{ color: C.textDim, textAlign: "left", padding: "0 10px 10px 0", fontWeight: 800 }}>DATE</th>
                        <th style={{ color: COLORS.ensemble, textAlign: "right", padding: "0 10px 10px", fontWeight: 800 }}>ENSEMBLE</th>
                        <th style={{ color: COLORS.lstm, textAlign: "right", padding: "0 10px 10px", fontWeight: 800 }}>LSTM</th>
                        <th style={{ color: COLORS.xgboost, textAlign: "right", padding: "0 10px 10px", fontWeight: 800 }}>XGBOOST</th>
                        <th style={{ color: COLORS.random_forest, textAlign: "right", padding: "0 0 10px 10px", fontWeight: 800 }}>RF</th>
                    </tr>
                </thead>
                <tbody>
                    {tableRows.map((row) => (
                        <tr key={row.date} style={{ borderBottom: "1px solid rgba(255,255,255,.055)" }}>
                            <td style={{ color: C.textMid, padding: "10px 10px 10px 0", fontVariantNumeric: "tabular-nums" }}>{row.date}</td>
                            <td style={{ color: COLORS.ensemble, textAlign: "right", padding: "10px", fontWeight: 800 }}>{formatPrice(row.ensemble)}</td>
                            <td style={{ color: COLORS.lstm, textAlign: "right", padding: "10px" }}>{formatPrice(row.lstm)}</td>
                            <td style={{ color: COLORS.xgboost, textAlign: "right", padding: "10px" }}>{formatPrice(row.xgboost)}</td>
                            <td style={{ color: COLORS.random_forest, textAlign: "right", padding: "10px 0 10px 10px" }}>{formatPrice(row.random_forest)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

export default function PredictionsTab({ selectedTicker, apiConnected, priceData }) {
    const [horizon, setHorizon] = useState(30);
    const [selectedModel, setSelectedModel] = useState("all");
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [refreshKey, setRefreshKey] = useState(0);

    useEffect(() => {
        if (!apiConnected) return;
        setLoading(true);
        setError(null);
        fetchEnsemblePrediction(selectedTicker, horizon)
            .then((payload) => {
                setData(payload);
                setLoading(false);
            })
            .catch((err) => {
                setError(err?.message || "Forecast request failed.");
                setLoading(false);
            });
    }, [selectedTicker, horizon, apiConnected, refreshKey]);

    if (!apiConnected) {
        return (
            <div style={{ padding: 48, color: C.textDim, textAlign: "center" }}>
                Connect to the API server to view predictions.
            </div>
        );
    }

    const currentPrice = data?.current_price ?? priceData?.bars?.[priceData.bars.length - 1]?.close;
    const ensemble = data?.ensemble;
    const modelUnavailable = data?.status === "unavailable" || data?.model_available === false;
    const bullish = Number(ensemble?.change_pct || 0) >= 0;
    const reliabilityColor = ensemble?.reliability === "Low" ? C.red : ensemble?.reliability === "Medium" ? C.amber : C.green;

    return (
        <div style={{ display: "grid", gap: 18, paddingBottom: 36 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap", alignItems: "end" }}>
                <div style={{ display: "grid", gap: 6 }}>
                    <div style={{ color: C.textDim, fontSize: 12, fontWeight: 800, letterSpacing: "0.06em", textTransform: "uppercase" }}>Predictions</div>
                    <div style={{ color: C.text, fontSize: 25, fontWeight: 900, lineHeight: 1 }}>{selectedTicker}</div>
                </div>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                    <select
                        value={selectedModel}
                        onChange={(event) => setSelectedModel(event.target.value)}
                        style={{
                            background: COLORS.surface,
                            color: C.text,
                            border: `1px solid ${C.border}`,
                            borderRadius: 8,
                            padding: "8px 10px",
                            fontSize: 13,
                            fontWeight: 700,
                            outline: "none",
                        }}
                    >
                        {MODEL_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                    </select>
                    <div style={{ display: "inline-flex", background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 3, gap: 3 }}>
                        {HORIZONS.map((days) => (
                            <button
                                key={days}
                                type="button"
                                onClick={() => setHorizon(days)}
                                style={{
                                    background: horizon === days ? COLORS.ensemble : "transparent",
                                    color: horizon === days ? "#10131A" : C.textMid,
                                    border: "none",
                                    borderRadius: 6,
                                    padding: "7px 12px",
                                    minWidth: 42,
                                    fontSize: 12,
                                    fontWeight: 900,
                                    cursor: "pointer",
                                }}
                            >
                                {days}D
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {loading && (
                <div style={{ padding: 46, color: C.textDim, background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, textAlign: "center" }}>
                    Loading forecast...
                </div>
            )}

            {!loading && error && (
                <div style={{ padding: 18, color: C.red, background: "rgba(244,63,94,.08)", border: "1px solid rgba(244,63,94,.35)", borderRadius: 8 }}>
                    {error}
                </div>
            )}

            {!loading && !error && modelUnavailable && (
                <div style={{ padding: 28, background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, display: "grid", gap: 16 }}>
                    <div>
                        <div style={{ color: C.text, fontSize: 18, fontWeight: 900, marginBottom: 6 }}>Ensemble models not trained</div>
                        <div style={{ color: C.textMid, fontSize: 13 }}>{data?.message || `Train the LSTM, XGBoost, and Random Forest ensemble for ${selectedTicker}.`}</div>
                    </div>
                    <TrainButton symbol={selectedTicker} onComplete={() => setRefreshKey((key) => key + 1)} />
                </div>
            )}

            {!loading && !error && !modelUnavailable && ensemble && (
                <>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
                        <MetricCard label="Current" value={formatPrice(currentPrice)} />
                        <MetricCard
                            label="Ensemble Forecast"
                            value={formatPrice(ensemble.target)}
                            sub={`${formatPct(ensemble.change_pct)} ${ensemble.signal || ""} | ${ensemble.reliability} reliability`}
                            color={bullish ? COLORS.ensemble : C.red}
                        />
                        <MetricCard label="Bull 90%" value={formatPrice(ensemble.upper_90)} color={C.green} />
                        <MetricCard label="Bear 90%" value={formatPrice(ensemble.lower_90)} color={C.red} />
                    </div>

                    <ForecastChart
                        priceData={priceData}
                        forecastPoints={(data.forecast_points || []).slice(0, horizon)}
                        selectedModel={selectedModel}
                    />

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 14, alignItems: "start" }}>
                        <WeightsPanel
                            weights={data.weights}
                            consensus={ensemble.consensus}
                            signal={ensemble.signal}
                            reliability={ensemble.reliability}
                        />
                        <div style={{ display: "grid", gap: 10 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
                                <div style={{ color: C.text, fontSize: 13, fontWeight: 900 }}>Forecast Table</div>
                                <div style={{ color: reliabilityColor, fontSize: 12, fontWeight: 800 }}>
                                    {ensemble.reliability} Reliability
                                </div>
                            </div>
                            <ForecastTable rows={data.forecast_points || []} />
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
