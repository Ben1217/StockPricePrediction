import { useState, useEffect } from "react";
import {
    ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis,
    PolarRadiusAxis, Legend
} from "recharts";
import { C } from "../utils/data";
import { optimizePortfolio, fetchFrontier } from "../utils/api";
import { StatCard, Section, ChartTooltip } from "../components/UIComponents";

export default function OptimizationTab({ apiConnected, notify }) {
    const [symbols, setSymbols] = useState("AAPL,MSFT,GOOGL,AMZN,NVDA");
    const [method, setMethod] = useState("max_sharpe");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [frontier, setFrontier] = useState(null);
    const [error, setError] = useState(null);

    const runOptimize = async () => {
        setLoading(true); setError(null);
        const symList = symbols.split(",").map(s => s.trim()).filter(Boolean);
        try {
            const [opt, front] = await Promise.all([
                optimizePortfolio({ symbols: symList, method }),
                fetchFrontier({ symbols: symList, method }).catch(() => null),
            ]);
            setResult(opt);
            setFrontier(front);
            notify?.("Optimization complete");
        } catch (e) { setError(e.message); }
        setLoading(false);
    };

    if (!apiConnected) return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>🔌 Connect to API</div>;

    const weights = result?.weights || {};
    const metrics = result?.metrics || {};
    const frontierPts = frontier?.points || [];
    const optimal = frontier?.optimal_portfolio || {};

    const weightData = Object.entries(weights).map(([sym, w]) => ({
        name: sym, weight: Math.round(w * 100),
    }));

    const radarData = [
        { metric: "Return", value: Math.min(100, ((result?.expected_return || 0) + 0.05) * 200) },
        { metric: "Sharpe", value: Math.min(100, (result?.sharpe_ratio || 0) * 40) },
        { metric: "Low Vol", value: Math.min(100, (1 - (result?.volatility || 0.2)) * 100) },
        { metric: "Diversif.", value: Math.min(100, Object.keys(weights).length * 20) },
        { metric: "Liquidity", value: 85 },
    ];

    const inputStyle = {
        background: C.bg2, color: C.text, border: `1px solid ${C.border}`, borderRadius: 6,
        padding: "8px 12px", fontSize: 12, fontFamily: "'DM Mono',monospace",
    };

    return (
        <div className="fade-up">
            <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text, marginBottom: 4 }}>
                ⚡ Portfolio Optimization
            </h1>
            <div style={{ fontSize: 11, color: C.textDim, marginBottom: 20 }}>
                Mean-Variance · Risk-Parity · Max Sharpe · Min Volatility
            </div>

            {/* Config */}
            <Section title="CONFIGURATION">
                <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
                    <div style={{ flex: 1 }}>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Symbols (comma-separated)</label>
                        <input value={symbols} onChange={e => setSymbols(e.target.value.toUpperCase())} style={{ ...inputStyle, width: "100%" }} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Method</label>
                        <select value={method} onChange={e => setMethod(e.target.value)} style={inputStyle}>
                            <option value="max_sharpe">Max Sharpe</option>
                            <option value="min_volatility">Min Volatility</option>
                            <option value="max_return">Max Return</option>
                        </select>
                    </div>
                    <button onClick={runOptimize} disabled={loading} style={{
                        background: loading ? C.bg2 : `linear-gradient(135deg, ${C.amber}, #f97316)`,
                        color: loading ? C.textDim : "#000", border: "none", borderRadius: 8,
                        padding: "10px 24px", fontSize: 13, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
                        fontFamily: "'Syne',sans-serif",
                    }}>{loading ? "⏳ Optimizing..." : "⚡ Optimize"}</button>
                </div>
            </Section>

            {error && <div style={{ background: C.red + "22", color: C.red, padding: 12, borderRadius: 8, marginTop: 16, fontSize: 12 }}>{error}</div>}

            {result && <>
                {/* Metrics */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginTop: 20, marginBottom: 20 }}>
                    <StatCard label="EXP. RETURN" value={`${((result.expected_return || 0) * 100).toFixed(1)}%`} sub="Annualized" color={C.green} />
                    <StatCard label="VOLATILITY" value={`${((result.volatility || 0) * 100).toFixed(1)}%`} sub="Annualized" color={C.red} />
                    <StatCard label="SHARPE" value={(result.sharpe_ratio || 0).toFixed(2)} sub="Risk-adjusted" color={C.cyan} />
                    <StatCard label="METHOD" value={method.replace("_", " ").toUpperCase()} sub="Optimization" color={C.amber} />
                </div>

                {/* Weights + Frontier */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                    <Section title="OPTIMAL WEIGHTS">
                        {weightData.map((d, i) => (
                            <div key={d.name} style={{ marginBottom: 12 }}>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                    <span style={{ color: C.amber, fontWeight: 700, fontSize: 12 }}>{d.name}</span>
                                    <span style={{ color: C.text, fontSize: 12 }}>{d.weight}%</span>
                                </div>
                                <div style={{ background: C.bg2, borderRadius: 4, height: 8, overflow: "hidden" }}>
                                    <div style={{
                                        width: `${d.weight}%`, height: "100%", background: C.cyan, borderRadius: 4,
                                        transition: "width .5s ease"
                                    }} />
                                </div>
                            </div>
                        ))}
                    </Section>

                    <Section title="EFFICIENT FRONTIER">
                        {frontierPts.length > 0 ? (
                            <ResponsiveContainer width="100%" height={250}>
                                <ScatterChart>
                                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                                    <XAxis dataKey="volatility" name="Risk" tick={{ fill: C.textDim, fontSize: 9 }}
                                        tickFormatter={v => `${(v * 100).toFixed(0)}%`} label={{ value: "Risk (σ)", fill: C.textDim, fontSize: 10, position: "bottom" }} />
                                    <YAxis dataKey="return" name="Return" tick={{ fill: C.textDim, fontSize: 9 }}
                                        tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                                    <Tooltip formatter={v => `${(v * 100).toFixed(2)}%`} />
                                    <Scatter data={frontierPts} fill={C.cyan} r={3} />
                                    {optimal.volatility && (
                                        <Scatter data={[optimal]} fill={C.amber} r={8} shape="triangle" />
                                    )}
                                </ScatterChart>
                            </ResponsiveContainer>
                        ) : <div style={{ color: C.textDim, padding: 40, textAlign: "center" }}>Run optimization to see frontier</div>}
                    </Section>
                </div>

                {/* Radar + Metrics table */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
                    <Section title="RISK PROFILE">
                        <ResponsiveContainer width="100%" height={250}>
                            <RadarChart data={radarData}>
                                <PolarGrid stroke={C.border} />
                                <PolarAngleAxis dataKey="metric" tick={{ fill: C.textDim, fontSize: 10 }} />
                                <PolarRadiusAxis tick={false} domain={[0, 100]} />
                                <Radar dataKey="value" stroke={C.cyan} fill={C.cyan} fillOpacity={0.2} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </Section>
                    <Section title="OPTIMIZATION METRICS">
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                            <thead><tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                <th style={{ padding: "8px 12px", color: C.textDim, textAlign: "left" }}>Metric</th>
                                <th style={{ padding: "8px 12px", color: C.textDim, textAlign: "right" }}>Value</th>
                            </tr></thead>
                            <tbody>
                                {Object.entries(metrics).map(([k, v]) => (
                                    <tr key={k} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                        <td style={{ padding: "6px 12px", color: C.textMid }}>{k.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}</td>
                                        <td style={{ padding: "6px 12px", color: C.amber, textAlign: "right", fontWeight: 700 }}>
                                            {typeof v === "number" ? v.toFixed(4) : String(v)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </Section>
                </div>
            </>}
        </div>
    );
}
