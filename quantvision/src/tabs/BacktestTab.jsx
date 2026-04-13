import { useEffect, useState } from "react";
import {
    CartesianGrid,
    ComposedChart,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Scatter,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { C, DEFAULT_INDEX_SYMBOL } from "../utils/data";
import { runBacktest, getCSVExportURL, getPDFExportURL } from "../utils/api";
import { StatCard, Section } from "../components/UIComponents";

const EQUITY_COLORS = {
    hybrid_ml_ta: C.amber,
    technical_only: C.cyan,
    buy_and_hold: C.green,
    market: C.red,
};

function formatPct(value, digits = 2) {
    return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "-";
}

function formatNumber(value, digits = 2) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function formatCurrency(value) {
    return typeof value === "number" && Number.isFinite(value)
        ? `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`
        : "-";
}

function mergeEquitySeries(runs) {
    const rows = new Map();
    runs.forEach((run) => {
        if (run?.status !== "ok") return;
        (run.equity_curve || []).forEach((point) => {
            const current = rows.get(point.date) || { date: point.date };
            current[run.key] = point.value;
            rows.set(point.date, current);
        });
    });
    return Array.from(rows.values()).sort((a, b) => a.date.localeCompare(b.date));
}

function buildPriceChartData(priceSeries, markers) {
    const byDate = new Map();
    (markers || []).forEach((marker) => {
        byDate.set(marker.date, marker);
    });
    return (priceSeries || []).map((point) => {
        const marker = byDate.get(point.date);
        return {
            ...point,
            buyPrice: marker?.type === "BUY" ? marker.price : null,
            sellPrice: marker?.type === "SELL" ? marker.price : null,
            markerReason: marker?.reason || "",
            markerLabel: marker?.label || "",
        };
    });
}

function SimpleTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    return (
        <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, minWidth: 180 }}>
            <div style={{ color: C.text, fontWeight: 700, marginBottom: 6 }}>{label}</div>
            {payload.map((item, index) => (
                <div key={`${item.dataKey || item.name || "series"}-${index}`} style={{ color: item.color || C.textDim, fontSize: 11, marginBottom: 4 }}>
                    {item.name || item.dataKey}: {typeof item.value === "number" ? item.value.toFixed(2) : item.value}
                </div>
            ))}
            {payload[0]?.payload?.markerReason && (
                <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.45 }}>
                    {payload[0].payload.markerLabel}: {payload[0].payload.markerReason}
                </div>
            )}
        </div>
    );
}

function ComparisonTable({ title, rows }) {
    return (
        <Section title={title} style={{ marginTop: 16 }}>
            <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, minWidth: 760 }}>
                    <thead>
                        <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                            {["Run", "Status", "Return", "CAGR", "Sharpe", "Sortino", "Max DD", "Trades", "Message"].map((header) => (
                                <th key={header} style={{ padding: "8px 10px", color: C.textDim, textAlign: "left", fontWeight: 500 }}>
                                    {header}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((run) => (
                            <tr key={run.key} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                <td style={{ padding: "8px 10px", color: C.text }}>{run.label}</td>
                                <td style={{ padding: "8px 10px", color: run.status === "ok" ? C.green : C.red, fontWeight: 700 }}>{run.status}</td>
                                <td style={{ padding: "8px 10px", color: C.text }}>{formatPct(run.metrics?.total_return)}</td>
                                <td style={{ padding: "8px 10px", color: C.text }}>{formatPct(run.metrics?.cagr)}</td>
                                <td style={{ padding: "8px 10px", color: C.text }}>{formatNumber(run.metrics?.sharpe_ratio)}</td>
                                <td style={{ padding: "8px 10px", color: C.text }}>{formatNumber(run.metrics?.sortino_ratio)}</td>
                                <td style={{ padding: "8px 10px", color: C.text }}>{formatPct(run.metrics?.max_drawdown)}</td>
                                <td style={{ padding: "8px 10px", color: C.text }}>{run.metrics?.total_trades ?? "-"}</td>
                                <td style={{ padding: "8px 10px", color: C.textDim, maxWidth: 280 }}>{run.message || "-"}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </Section>
    );
}

export default function BacktestTab({ selectedTicker, apiConnected, notify }) {
    const [symbol, setSymbol] = useState(selectedTicker);
    const [startDate, setStartDate] = useState("2022-01-01");
    const [endDate, setEndDate] = useState("2024-12-31");
    const [capital, setCapital] = useState(100000);
    const [primaryModel, setPrimaryModel] = useState("xgboost");
    const [posSize, setPosSize] = useState(0.1);
    const [commission, setCommission] = useState(0);
    const [slippage, setSlippage] = useState(0.001);
    const [includeBenchmark, setIncludeBenchmark] = useState(true);
    const [benchmarkSymbol, setBenchmarkSymbol] = useState(DEFAULT_INDEX_SYMBOL);
    const [validationMode, setValidationMode] = useState("single_period");
    const [running, setRunning] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        setSymbol(selectedTicker);
    }, [selectedTicker]);

    const handleRun = async () => {
        setRunning(true);
        setError(null);
        try {
            const res = await runBacktest({
                symbol,
                start_date: startDate,
                end_date: endDate,
                initial_capital: capital,
                model_type: primaryModel,
                primary_model: primaryModel,
                position_size: posSize,
                commission_rate: commission,
                slippage_rate: slippage,
                include_market_benchmark: includeBenchmark,
                benchmark_symbol: benchmarkSymbol,
                validation_mode: validationMode,
            });
            setResult(res);
            notify?.(`Backtest complete: ${res.primary_run?.label || res.message}`);
        } catch (e) {
            setError(e.message);
        }
        setRunning(false);
    };

    if (!apiConnected) return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>Connect to API</div>;

    const metrics = result?.metrics || {};
    const primaryRun = result?.primary_run || {};
    const trades = result?.trades || [];
    const priceChartData = buildPriceChartData(result?.price_series || [], primaryRun?.markers || []);
    const marketBenchmark = (result?.benchmarks || []).find((run) => run.benchmark_type === "market");
    const equityRuns = [...(result?.strategy_runs || [])];
    if (marketBenchmark) {
        equityRuns.push({ ...marketBenchmark, key: "market" });
    }
    const equityChartData = mergeEquitySeries(equityRuns);

    const inputStyle = {
        background: C.bg2,
        color: C.text,
        border: `1px solid ${C.border}`,
        borderRadius: 6,
        padding: "8px 12px",
        fontSize: 12,
        fontFamily: "'DM Mono',monospace",
        width: "100%",
    };

    return (
        <div className="fade-up">
            <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text, marginBottom: 20 }}>
                Backtesting Validation
            </h1>

            <Section title="CONFIGURATION">
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 16 }}>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Symbol</label>
                        <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Start Date</label>
                        <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>End Date</label>
                        <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Capital ($)</label>
                        <input type="number" value={capital} onChange={(e) => setCapital(+e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Primary Model</label>
                        <select value={primaryModel} onChange={(e) => setPrimaryModel(e.target.value)} style={inputStyle}>
                            <option value="xgboost">XGBoost</option>
                            <option value="random_forest">Random Forest</option>
                            <option value="lstm">LSTM</option>
                        </select>
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Position %</label>
                        <input type="number" value={posSize} step={0.05} min={0.01} max={1} onChange={(e) => setPosSize(+e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Commission</label>
                        <input type="number" value={commission} step={0.0005} min={0} onChange={(e) => setCommission(+e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Slippage</label>
                        <input type="number" value={slippage} step={0.0005} min={0} onChange={(e) => setSlippage(+e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Validation</label>
                        <select value={validationMode} onChange={(e) => setValidationMode(e.target.value)} style={inputStyle}>
                            <option value="single_period">Single Period</option>
                            <option value="walk_forward">Walk Forward</option>
                        </select>
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Market Benchmark</label>
                        <label style={{ display: "flex", alignItems: "center", gap: 8, color: C.text, fontSize: 12 }}>
                            <input type="checkbox" checked={includeBenchmark} onChange={(e) => setIncludeBenchmark(e.target.checked)} />
                            Include benchmark
                        </label>
                    </div>
                    <div>
                        <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Benchmark Symbol</label>
                        <input value={benchmarkSymbol} disabled={!includeBenchmark} onChange={(e) => setBenchmarkSymbol(e.target.value.toUpperCase())} style={{ ...inputStyle, opacity: includeBenchmark ? 1 : 0.5 }} />
                    </div>
                </div>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                    <button onClick={handleRun} disabled={running} style={{
                        background: running ? C.bg2 : `linear-gradient(135deg, ${C.amber}, #f97316)`,
                        color: running ? C.textDim : "#000",
                        border: "none",
                        borderRadius: 8,
                        padding: "12px 28px",
                        fontSize: 14,
                        fontWeight: 700,
                        cursor: running ? "not-allowed" : "pointer",
                        fontFamily: "'Syne',sans-serif",
                    }}>{running ? "Running..." : "Run Comparison Backtest"}</button>
                    {result && <>
                        <a href={getCSVExportURL("backtest", symbol)} style={{
                            background: C.bg2,
                            color: C.amber,
                            border: `1px solid ${C.border}`,
                            borderRadius: 8,
                            padding: "12px 18px",
                            fontSize: 12,
                            textDecoration: "none",
                        }}>CSV</a>
                        <a href={getPDFExportURL("backtest", symbol)} style={{
                            background: C.bg2,
                            color: C.amber,
                            border: `1px solid ${C.border}`,
                            borderRadius: 8,
                            padding: "12px 18px",
                            fontSize: 12,
                            textDecoration: "none",
                        }}>PDF Report</a>
                    </>}
                </div>
            </Section>

            {error && <div style={{ background: `${C.red}22`, color: C.red, padding: 12, borderRadius: 8, marginTop: 16, fontSize: 12 }}>{error}</div>}

            {result && <>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginTop: 20 }}>
                    <StatCard label="PRIMARY RETURN" value={formatPct(metrics.total_return)} sub={formatCurrency(metrics.final_value)} color={(metrics.total_return || 0) >= 0 ? C.green : C.red} />
                    <StatCard label="CAGR" value={formatPct(metrics.cagr)} sub="Annualized growth" color={C.amber} />
                    <StatCard label="SHARPE" value={formatNumber(metrics.sharpe_ratio)} sub="Risk-adjusted" color={C.cyan} />
                    <StatCard label="SORTINO" value={formatNumber(metrics.sortino_ratio)} sub="Downside risk" color={C.green} />
                    <StatCard label="CALMAR" value={formatNumber(metrics.calmar_ratio)} sub="Return vs drawdown" color={C.amber} />
                    <StatCard label="PROFIT FACTOR" value={formatNumber(metrics.profit_factor)} sub={`Expectancy ${formatPct(metrics.expectancy)}`} color={C.text} />
                    <StatCard label="MAX DD" value={formatPct(metrics.max_drawdown)} sub={`Wins ${metrics.max_consecutive_wins || 0} / Losses ${metrics.max_consecutive_losses || 0}`} color={C.red} />
                    <StatCard label="TRADES" value={metrics.total_trades || 0} sub={`Avg win ${formatCurrency(metrics.average_win)} / Avg loss ${formatCurrency(metrics.average_loss)}`} color={C.amber} />
                </div>

                <ComparisonTable title="BENCHMARK COMPARISON" rows={result.benchmarks || []} />
                <ComparisonTable title="STRATEGY COMPARISON" rows={result.strategy_runs || []} />
                <ComparisonTable title="MODEL COMPARISON" rows={result.model_runs || []} />

                <Section title="EQUITY COMPARISON" style={{ marginTop: 16 }}>
                    <ResponsiveContainer width="100%" height={320}>
                        <LineChart data={equityChartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                            <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 9 }} tickFormatter={(d) => d?.slice(5) || d} />
                            <YAxis tick={{ fill: C.textDim, fontSize: 9 }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                            <Tooltip content={<SimpleTooltip />} />
                            <Legend />
                            {(result.strategy_runs || []).map((run) => (
                                run.status === "ok" && (
                                    <Line key={run.key} type="monotone" dataKey={run.key} name={run.label} stroke={EQUITY_COLORS[run.key] || C.text} strokeWidth={2} dot={false} />
                                )
                            ))}
                            {marketBenchmark?.status === "ok" && (
                                <Line type="monotone" dataKey="market" name={marketBenchmark.label} stroke={EQUITY_COLORS.market} strokeWidth={2} dot={false} strokeDasharray="6 3" />
                            )}
                        </LineChart>
                    </ResponsiveContainer>
                </Section>

                <Section title="PRIMARY PRICE CHART" style={{ marginTop: 16 }}>
                    <ResponsiveContainer width="100%" height={320}>
                        <ComposedChart data={priceChartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                            <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 9 }} tickFormatter={(d) => d?.slice(5) || d} />
                            <YAxis tick={{ fill: C.textDim, fontSize: 9 }} tickFormatter={(v) => `$${v.toFixed(0)}`} />
                            <Tooltip content={<SimpleTooltip />} />
                            <Legend />
                            <Line type="monotone" dataKey="close" name="Close" stroke={C.text} strokeWidth={2} dot={false} />
                            <Scatter name="Buy" dataKey="buyPrice" fill={C.green} />
                            <Scatter name="Sell" dataKey="sellPrice" fill={C.red} />
                        </ComposedChart>
                    </ResponsiveContainer>
                </Section>

                {result.validation?.mode === "walk_forward" && (
                    <Section title="WALK-FORWARD VALIDATION" style={{ marginTop: 16 }}>
                        <div style={{ color: result.validation.status === "ok" ? C.text : C.red, fontSize: 12, marginBottom: 12 }}>
                            {result.validation.status === "ok"
                                ? `Completed ${result.validation.n_splits} folds with gap ${result.validation.gap}.`
                                : result.validation.message}
                        </div>
                        {result.validation.status === "ok" && (
                            <div style={{ overflowX: "auto" }}>
                                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                                    <thead>
                                        <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                            {["Fold", "RMSE", "MAE", "R2", "Directional Accuracy", "Sharpe"].map((header) => (
                                                <th key={header} style={{ padding: "8px 10px", color: C.textDim, textAlign: "left", fontWeight: 500 }}>{header}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {(result.validation.folds || []).map((fold) => (
                                            <tr key={fold.fold} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                                <td style={{ padding: "8px 10px", color: C.text }}>{fold.fold}</td>
                                                <td style={{ padding: "8px 10px", color: C.text }}>{formatNumber(fold.rmse, 4)}</td>
                                                <td style={{ padding: "8px 10px", color: C.text }}>{formatNumber(fold.mae, 4)}</td>
                                                <td style={{ padding: "8px 10px", color: C.text }}>{formatNumber(fold.r2, 4)}</td>
                                                <td style={{ padding: "8px 10px", color: C.text }}>{formatNumber(fold.directional_accuracy, 4)}</td>
                                                <td style={{ padding: "8px 10px", color: C.text }}>{formatNumber(fold.sharpe_ratio, 4)}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </Section>
                )}

                <Section title={`TRADE LOG (${trades.length} trades)`} style={{ marginTop: 16 }}>
                    <div style={{ maxHeight: 360, overflowY: "auto", overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, minWidth: 860 }}>
                            <thead>
                                <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                    {["Date", "Type", "Qty", "Price", "Commission", "PnL", "Return", "Reason"].map((header) => (
                                        <th key={header} style={{ padding: "8px 10px", color: C.textDim, textAlign: "left", fontWeight: 500 }}>{header}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {trades.slice(0, 120).map((trade, index) => (
                                    <tr key={`${trade.date}-${trade.type}-${index}`} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                        <td style={{ padding: "8px 10px", color: C.textMid }}>{trade.date}</td>
                                        <td style={{ padding: "8px 10px", color: trade.type === "BUY" ? C.green : C.red, fontWeight: 700 }}>{trade.type}</td>
                                        <td style={{ padding: "8px 10px", color: C.text }}>{formatNumber(trade.quantity, 2)}</td>
                                        <td style={{ padding: "8px 10px", color: C.text }}>{formatCurrency(trade.price)}</td>
                                        <td style={{ padding: "8px 10px", color: C.text }}>{formatCurrency(trade.commission)}</td>
                                        <td style={{ padding: "8px 10px", color: trade.realized_pnl >= 0 ? C.green : C.red }}>{trade.realized_pnl == null ? "-" : formatCurrency(trade.realized_pnl)}</td>
                                        <td style={{ padding: "8px 10px", color: C.text }}>{trade.return_pct == null ? "-" : formatPct(trade.return_pct)}</td>
                                        <td style={{ padding: "8px 10px", color: C.textDim, maxWidth: 420 }}>{trade.reason || "-"}</td>
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
