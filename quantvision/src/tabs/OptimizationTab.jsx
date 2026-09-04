import { useState } from "react";
import {
    ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer
} from "recharts";
import { C } from "../utils/data";
import { optimizePortfolio, fetchFrontier, fetchCorrelation } from "../utils/api";
import { StatCard, Section, Hint } from "../components/UIComponents";

/**
 * The four objectives the backend actually accepts (OptimizationMethod in
 * schemas.py). Hints describe what each one really solves for — `max_sharpe`
 * is a return-maximisation under a volatility cap, not a true Sharpe maximum.
 */
const METHODS = {
    max_sharpe: {
        label: "Max Sharpe",
        hint: "Aims for the best return per unit of risk. Solved as: highest expected return while keeping annual volatility under 20%.",
    },
    min_volatility: {
        label: "Min Volatility",
        hint: "The calmest mix — the smallest expected price swings, even if that means lower returns.",
    },
    max_return: {
        label: "Max Return",
        hint: "Chases the highest expected return and ignores risk. Usually concentrates into one or two names.",
    },
    risk_parity: {
        label: "Risk Parity",
        hint: "Every holding contributes the same amount of risk, so no single stock dominates the portfolio.",
    },
};

export default function OptimizationTab({ apiConnected, notify, watchlist = [] }) {
    // Seeded from the watchlist rather than a fixed list, so the tab explores
    // the symbols this user actually follows.
    const [symbols, setSymbols] = useState(() => (watchlist.length ? watchlist.join(",") : ""));
    const [method, setMethod] = useState("max_sharpe");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [frontier, setFrontier] = useState(null);
    const [correlation, setCorrelation] = useState(null);
    const [error, setError] = useState(null);

    const runOptimize = async () => {
        setLoading(true); setError(null);
        const symList = symbols.split(",").map(s => s.trim()).filter(Boolean);
        try {
            const [opt, front, corr] = await Promise.all([
                optimizePortfolio({ symbols: symList, method }),
                fetchFrontier({ symbols: symList, method }).catch(() => null),
                fetchCorrelation(symList).catch(() => null),
            ]);
            setResult(opt);
            setFrontier(front);
            setCorrelation(corr);
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
                Max Sharpe · Min Volatility · Max Return · Risk Parity
            </div>

            {/* Config */}
            <Section
                title="CONFIGURATION"
                hint="Pick the stocks you want to hold and how you want them weighted. The optimizer looks at the last 12 months of daily returns to decide the split."
            >
                <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
                    <div style={{ flex: 1 }}>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Symbols (comma-separated)</label>
                        <input value={symbols} onChange={e => setSymbols(e.target.value.toUpperCase())} style={{ ...inputStyle, width: "100%" }} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>
                            Method<Hint text={METHODS[method].hint} />
                        </label>
                        <select value={method} onChange={e => setMethod(e.target.value)} style={inputStyle}>
                            {Object.entries(METHODS).map(([key, m]) => (
                                <option key={key} value={key}>{m.label}</option>
                            ))}
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
                    <StatCard label="EXP. RETURN" value={`${((result.expected_return || 0) * 100).toFixed(1)}%`}
                        sub="Annualized" color={C.green}
                        hint="What this mix would have earned per year over the lookback window. A historical average, not a forecast." />
                    <StatCard label="VOLATILITY" value={`${((result.volatility || 0) * 100).toFixed(1)}%`}
                        sub="Annualized" color={C.red}
                        hint="How much the portfolio value swings in a typical year. Higher means a bumpier ride in both directions." />
                    <StatCard label="SHARPE" value={(result.sharpe_ratio || 0).toFixed(2)}
                        sub="Risk-adjusted" color={C.cyan}
                        hint="Return earned per unit of risk, above the risk-free rate. Above 1 is good, above 2 is excellent." />
                    {/* result.method, not the dropdown — shows what the server actually ran. */}
                    <StatCard label="METHOD" value={(METHODS[result.method] || METHODS[method]).label}
                        sub="Objective" color={C.amber}
                        hint={(METHODS[result.method] || METHODS[method]).hint} />
                </div>

                {/* Weights + Frontier */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                    <Section title="OPTIMAL WEIGHTS"
                        hint="How much of every $100 to put in each stock. Positions are capped at 40% and floored at 2% by default.">
                        {weightData.map((d) => (
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

                    <Section title="EFFICIENT FRONTIER"
                        hint="Every dot is one possible mix of these stocks. The curve is the best return available at each level of risk — the amber triangle is the pick with the best risk/return trade-off.">
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
                    {/* Replaces a radar chart whose five axes were rescalings
                        invented for the picture — including a "Liquidity" spoke
                        pinned at a constant 85 with no data behind it. This is
                        the measured thing that radar was gesturing at: how alike
                        the holdings are, which is the entire reason a split can
                        beat a concentrated bet. */}
                    <Section
                        title="HOW ALIKE ARE THESE STOCKS?"
                        hint="Correlation of daily returns over the last 90 days. 1.00 means two names moved in lockstep, 0 means they moved independently, and below 0 means they tended to move opposite ways. Diversification is exactly what you buy by holding names that do not move together."
                    >
                        {correlation?.matrix && correlation?.tickers?.length ? (
                            <>
                                <div style={{ color: C.textMid, fontSize: 12, marginBottom: 12 }}>
                                    Average pair:{" "}
                                    <span style={{ color: C.cyan, fontFamily: "'DM Mono',monospace", fontWeight: 700 }}>
                                        {Number(correlation.avg_correlation).toFixed(2)}
                                    </span>
                                    <span style={{ color: C.textDim }}>
                                        {" "}— lower is better diversified.
                                    </span>
                                </div>
                                <div style={{ overflowX: "auto" }}>
                                    <table style={{ borderCollapse: "collapse", fontSize: 11, fontFamily: "'DM Mono',monospace" }}>
                                        <thead>
                                            <tr>
                                                <th style={{ padding: "6px 8px" }} />
                                                {correlation.tickers.map(t => (
                                                    <th key={t} style={{ padding: "6px 8px", color: C.textDim, fontWeight: 400 }}>{t}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {correlation.tickers.map(row => (
                                                <tr key={row}>
                                                    <td style={{ padding: "6px 8px", color: C.amber, fontWeight: 700 }}>{row}</td>
                                                    {correlation.tickers.map(col => {
                                                        const value = Number(correlation.matrix[row]?.[col]);
                                                        const self = row === col;
                                                        // Tint by strength: strongly-paired names are the
                                                        // ones a reader should notice.
                                                        const tint = !Number.isFinite(value) || self
                                                            ? "transparent"
                                                            : value >= 0.8 ? `${C.red}33`
                                                                : value >= 0.5 ? `${C.amber}26`
                                                                    : value <= 0 ? `${C.green}22` : "transparent";
                                                        return (
                                                            <td key={col} style={{
                                                                padding: "6px 8px", textAlign: "right",
                                                                background: tint,
                                                                color: self ? C.textDim : C.text,
                                                            }}>
                                                                {Number.isFinite(value) ? value.toFixed(2) : "—"}
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                {correlation.high_corr_pairs?.length > 0 && (
                                    <div style={{ color: C.textMid, fontSize: 11.5, marginTop: 12, lineHeight: 1.6 }}>
                                        Moving closely together:{" "}
                                        {correlation.high_corr_pairs
                                            .map(p => `${p.ticker_a}/${p.ticker_b} (${Number(p.correlation).toFixed(2)})`)
                                            .join(", ")}
                                        . Holding both buys less protection than the position count suggests.
                                    </div>
                                )}
                            </>
                        ) : (
                            <div style={{ color: C.textDim, padding: 40, textAlign: "center", fontSize: 12 }}>
                                Run optimization to see how closely these move together
                            </div>
                        )}
                    </Section>
                    <Section title="OPTIMIZATION METRICS"
                        hint="How this mix would have behaved over the lookback window, had you held it the whole time.">
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
