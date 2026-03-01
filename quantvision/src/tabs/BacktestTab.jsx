import { useState } from "react";
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer
} from "recharts";
import { C } from "../utils/data";
import { runBacktest, getCSVExportURL, getPDFExportURL } from "../utils/api";
import { StatCard, Section, ChartTooltip } from "../components/UIComponents";

export default function BacktestTab({ selectedTicker, apiConnected, notify }) {
    const [symbol, setSymbol] = useState(selectedTicker);
    const [startDate, setStartDate] = useState("2022-01-01");
    const [endDate, setEndDate] = useState("2024-12-31");
    const [capital, setCapital] = useState(100000);
    const [modelType, setModelType] = useState("xgboost");
    const [posSize, setPosSize] = useState(0.1);
    const [running, setRunning] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleRun = async () => {
        setRunning(true); setError(null);
        try {
            const res = await runBacktest({
                symbol, start_date: startDate, end_date: endDate,
                initial_capital: capital, model_type: modelType, position_size: posSize,
            });
            setResult(res);
            notify?.(`Backtest complete: ${res.message}`);
        } catch (e) {
            setError(e.message);
        }
        setRunning(false);
    };

    if (!apiConnected) return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>🔌 Connect to API</div>;

    const metrics = result?.metrics || {};
    const equity = result?.equity_curve || [];
    const trades = result?.trades || [];
    const inputStyle = {
        background: C.bg2, color: C.text, border: `1px solid ${C.border}`, borderRadius: 6,
        padding: "8px 12px", fontSize: 12, fontFamily: "'DM Mono',monospace", width: "100%",
    };

    return (
        <div className="fade-up">
            <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text, marginBottom: 20 }}>
                🧪 Backtesting Engine
            </h1>

            {/* Controls */}
            <Section title="CONFIGURATION">
                <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 12, marginBottom: 16 }}>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Symbol</label>
                        <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Start Date</label>
                        <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>End Date</label>
                        <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Capital ($)</label>
                        <input type="number" value={capital} onChange={e => setCapital(+e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Model</label>
                        <select value={modelType} onChange={e => setModelType(e.target.value)} style={inputStyle}>
                            <option value="xgboost">XGBoost</option>
                            <option value="random_forest">Random Forest</option>
                            <option value="lstm">LSTM</option>
                        </select>
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Position %</label>
                        <input type="number" value={posSize} step={0.05} min={0.01} max={1}
                            onChange={e => setPosSize(+e.target.value)} style={inputStyle} />
                    </div>
                </div>
                <div style={{ display: "flex", gap: 12 }}>
                    <button onClick={handleRun} disabled={running} style={{
                        background: running ? C.bg2 : `linear-gradient(135deg, ${C.amber}, #f97316)`,
                        color: running ? C.textDim : "#000", border: "none", borderRadius: 8,
                        padding: "12px 32px", fontSize: 14, fontWeight: 700, cursor: running ? "not-allowed" : "pointer",
                        fontFamily: "'Syne',sans-serif",
                    }}>{running ? "⏳ Running..." : "▶ Run Backtest"}</button>
                    {result && <>
                        <a href={getCSVExportURL("backtest", symbol)} style={{
                            background: C.bg2, color: C.amber, border: `1px solid ${C.border}`, borderRadius: 8,
                            padding: "12px 20px", fontSize: 12, textDecoration: "none", display: "flex", alignItems: "center",
                        }}>📥 CSV</a>
                        <a href={getPDFExportURL("backtest", symbol)} style={{
                            background: C.bg2, color: C.amber, border: `1px solid ${C.border}`, borderRadius: 8,
                            padding: "12px 20px", fontSize: 12, textDecoration: "none", display: "flex", alignItems: "center",
                        }}>📄 PDF Report</a>
                    </>}
                </div>
            </Section>

            {error && <div style={{ background: C.red + "22", color: C.red, padding: 12, borderRadius: 8, marginTop: 16, fontSize: 12 }}>{error}</div>}

            {result && <>
                {/* Metrics */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 12, marginTop: 20 }}>
                    <StatCard label="TOTAL RETURN" value={`${((metrics.total_return || 0) * 100).toFixed(2)}%`}
                        sub={`$${(metrics.final_value || capital).toFixed(0)}`}
                        color={(metrics.total_return || 0) >= 0 ? C.green : C.red} />
                    <StatCard label="SHARPE RATIO" value={(metrics.sharpe_ratio || 0).toFixed(2)}
                        sub="Risk-adjusted" color={C.cyan} />
                    <StatCard label="MAX DRAWDOWN" value={`${((metrics.max_drawdown || 0) * 100).toFixed(1)}%`}
                        sub="Peak to trough" color={C.red} />
                    <StatCard label="TRADES" value={metrics.total_trades || 0}
                        sub={`Buy: ${metrics.buy_trades || 0} / Sell: ${metrics.sell_trades || 0}`} color={C.amber} />
                    <StatCard label="WIN RATE" value={metrics.win_rate ? `${(metrics.win_rate * 100).toFixed(1)}%` : "—"}
                        sub="Trade accuracy" color={C.green} />
                </div>

                {/* Equity curve */}
                <Section title="EQUITY CURVE" style={{ marginTop: 16 }}>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={equity}>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                            <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 9 }}
                                tickFormatter={d => d?.slice(5) || d} interval={Math.floor(equity.length / 8)} />
                            <YAxis tick={{ fill: C.textDim, fontSize: 9 }} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
                            <Tooltip content={<ChartTooltip />} />
                            <Line type="monotone" dataKey="value" stroke={C.amber} strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </Section>

                {/* Trade log */}
                <Section title={`TRADE LOG (${trades.length} trades)`} style={{ marginTop: 16 }}>
                    <div style={{ maxHeight: 300, overflowY: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                            <thead>
                                <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                    {["Date", "Type", "Qty", "Price", "Commission"].map(h => (
                                        <th key={h} style={{ padding: "8px 12px", color: C.textDim, textAlign: "left", fontWeight: 400 }}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {trades.slice(0, 100).map((t, i) => (
                                    <tr key={i} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                        <td style={{ padding: "6px 12px", color: C.textMid }}>{t.date}</td>
                                        <td style={{ padding: "6px 12px", color: t.type === "BUY" ? C.green : C.red, fontWeight: 700 }}>{t.type}</td>
                                        <td style={{ padding: "6px 12px", color: C.text }}>{t.quantity.toFixed(2)}</td>
                                        <td style={{ padding: "6px 12px", color: C.text }}>${t.price.toFixed(2)}</td>
                                        <td style={{ padding: "6px 12px", color: C.textDim }}>${t.commission.toFixed(2)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </Section>
            </>}
        </div>
    );
}
