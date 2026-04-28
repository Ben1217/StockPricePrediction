import { useEffect, useMemo, useState } from "react";
import {
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { C } from "../utils/data";
import { runBacktest } from "../utils/api";

const STRATEGY_OPTIONS = [
    { value: "ta_only", label: "Technical Analysis" },
    { value: "ml_hybrid", label: "Hybrid ML + Technical Analysis" },
    { value: "buy_hold", label: "Buy and Hold" },
];

function todayISO() {
    return new Date().toISOString().slice(0, 10);
}

function formatPct(value, digits = 2) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "-";
    return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

function formatNumber(value, digits = 2) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function formatCurrency(value) {
    return typeof value === "number" && Number.isFinite(value)
        ? `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`
        : "-";
}

function monthlyCurve(equityCurve = [], bhCurve = []) {
    const benchmarkByDate = new Map(bhCurve.map((point) => [point.date, point.value]));
    const rows = equityCurve
        .map((point) => ({
            date: point.date,
            month: point.date?.slice(0, 7),
            strategy: point.value,
            buyHold: benchmarkByDate.get(point.date),
        }))
        .filter((point) => point.date && point.strategy != null && point.buyHold != null);

    const sampledByMonth = new Map();
    rows.forEach((point, index) => {
        const isLast = index === rows.length - 1;
        sampledByMonth.set(point.month || point.date, point);
        if (isLast) sampledByMonth.set(point.date, point);
    });
    return Array.from(sampledByMonth.values());
}

function BacktestTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    return (
        <div className="tooltip-card">
            <div style={{ color: C.amber, marginBottom: 6 }}>{label}</div>
            {payload.map((item) => (
                <div key={item.dataKey} style={{ display: "flex", gap: 14, justifyContent: "space-between", color: item.color }}>
                    <span>{item.name}</span>
                    <span>{formatCurrency(item.value)}</span>
                </div>
            ))}
        </div>
    );
}

function Field({ label, children }) {
    return (
        <label style={{ display: "flex", flexDirection: "column", gap: 5, minWidth: 0 }}>
            <span style={{ color: C.textDim, fontSize: 10, letterSpacing: 1, textTransform: "uppercase" }}>{label}</span>
            {children}
        </label>
    );
}

function MetricCard({ label, value, sub, tone = "neutral" }) {
    const toneColor = tone === "green" ? C.green : tone === "red" ? C.red : tone === "cyan" ? C.cyan : C.text;
    return (
        <div style={{
            background: C.bg2,
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: "14px 16px",
            minWidth: 0,
        }}>
            <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1, textTransform: "uppercase", marginBottom: 6 }}>
                {label}
            </div>
            <div style={{ color: toneColor, fontSize: 22, fontWeight: 800, fontFamily: "'DM Mono',monospace", lineHeight: 1.1 }}>
                {value}
            </div>
            {sub && <div style={{ color: C.textMid, fontSize: 11, marginTop: 5 }}>{sub}</div>}
        </div>
    );
}

function TradeTable({ trades = [] }) {
    if (!trades.length) {
        return <div style={{ color: C.textDim, fontSize: 12, padding: "18px 0" }}>No trades were generated for this period.</div>;
    }

    return (
        <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, minWidth: 780 }}>
                <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                        {["Date", "Type", "Shares", "Price", "P&L", "Return", "Reason"].map((header) => (
                            <th key={header} style={{ padding: "8px 10px", color: C.textDim, textAlign: "left", fontWeight: 500 }}>
                                {header}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {trades.map((trade, index) => {
                        const pnl = trade.pnl;
                        const returnPct = trade.return_pct;
                        return (
                            <tr key={`${trade.date}-${trade.type}-${index}`} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                <td style={{ padding: "9px 10px", color: C.textMid }}>{trade.date}</td>
                                <td style={{ padding: "9px 10px" }}>
                                    <span style={{
                                        color: trade.type === "BUY" ? C.green : C.red,
                                        border: `1px solid ${(trade.type === "BUY" ? C.green : C.red)}55`,
                                        background: `${trade.type === "BUY" ? C.green : C.red}18`,
                                        borderRadius: 4,
                                        padding: "2px 8px",
                                        fontWeight: 700,
                                    }}>
                                        {trade.type}
                                    </span>
                                </td>
                                <td style={{ padding: "9px 10px", color: C.text }}>{formatNumber(trade.shares, 4)}</td>
                                <td style={{ padding: "9px 10px", color: C.text }}>{formatCurrency(trade.price)}</td>
                                <td style={{ padding: "9px 10px", color: pnl == null ? C.textDim : pnl >= 0 ? C.green : C.red }}>
                                    {pnl == null ? "-" : formatCurrency(pnl)}
                                </td>
                                <td style={{ padding: "9px 10px", color: returnPct == null ? C.textDim : returnPct >= 0 ? C.green : C.red }}>
                                    {returnPct == null ? "-" : formatPct(returnPct)}
                                </td>
                                <td style={{ padding: "9px 10px", color: C.textDim, maxWidth: 340, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={trade.reason}>
                                    {trade.reason || "-"}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default function BacktestTab({ selectedTicker, apiConnected, notify }) {
    const [form, setForm] = useState({
        symbol: selectedTicker || "MSFT",
        start_date: "2022-01-01",
        end_date: todayISO(),
        initial_capital: 100000,
        strategy: "ta_only",
    });
    const [status, setStatus] = useState("idle");
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    useEffect(() => {
        setForm((current) => ({ ...current, symbol: selectedTicker || current.symbol }));
    }, [selectedTicker]);

    const chartData = useMemo(
        () => monthlyCurve(result?.equity_curve || [], result?.bh_curve || []),
        [result]
    );

    const inputStyle = {
        background: C.bg2,
        color: C.text,
        border: `1px solid ${C.border}`,
        borderRadius: 6,
        padding: "9px 11px",
        fontSize: 12,
        fontFamily: "'DM Mono',monospace",
        width: "100%",
        outline: "none",
        minHeight: 36,
    };

    const canRun = apiConnected
        && status !== "running"
        && form.symbol.trim()
        && form.start_date
        && form.end_date
        && Number(form.initial_capital) > 0;
    const isRunning = status === "running";
    const metrics = result?.metrics || {};

    const handleRun = async () => {
        setStatus("running");
        setError("");
        try {
            const response = await runBacktest({
                symbol: form.symbol.trim().toUpperCase(),
                start_date: form.start_date,
                end_date: form.end_date,
                initial_capital: Number(form.initial_capital),
                strategy: form.strategy,
            });
            setResult(response);
            setStatus("done");
            notify?.(`Backtest complete for ${response.summary?.symbol || form.symbol.toUpperCase()}`);
        } catch (err) {
            setError(err.message || "Backtest failed");
            setStatus("error");
        }
    };

    if (!apiConnected) {
        return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>Connect to API</div>;
    }

    return (
        <div className="fade-up">
            <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text, marginBottom: 6 }}>
                Backtesting Validation
            </h1>
            <div style={{ color: C.textDim, fontSize: 12, marginBottom: 18 }}>
                Test a selected strategy against historical buy-and-hold performance.
            </div>

            <div style={{
                background: C.bg1,
                border: `1px solid ${C.border}`,
                borderRadius: 8,
                padding: 16,
                marginBottom: 18,
            }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, alignItems: "end" }}>
                    <Field label="Symbol">
                        <input
                            value={form.symbol}
                            onChange={(event) => setForm((current) => ({ ...current, symbol: event.target.value.toUpperCase() }))}
                            placeholder="MSFT"
                            style={inputStyle}
                        />
                    </Field>
                    <Field label="Start date">
                        <input
                            type="date"
                            value={form.start_date}
                            onChange={(event) => setForm((current) => ({ ...current, start_date: event.target.value }))}
                            style={inputStyle}
                        />
                    </Field>
                    <Field label="End date">
                        <input
                            type="date"
                            value={form.end_date}
                            onChange={(event) => setForm((current) => ({ ...current, end_date: event.target.value }))}
                            style={inputStyle}
                        />
                    </Field>
                    <Field label="Initial capital">
                        <input
                            type="number"
                            min="1"
                            value={form.initial_capital}
                            onChange={(event) => setForm((current) => ({ ...current, initial_capital: event.target.value }))}
                            style={inputStyle}
                        />
                    </Field>
                    <Field label="Strategy">
                        <select
                            value={form.strategy}
                            onChange={(event) => setForm((current) => ({ ...current, strategy: event.target.value }))}
                            style={inputStyle}
                        >
                            {STRATEGY_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                        </select>
                    </Field>
                    <button
                        onClick={handleRun}
                        disabled={isRunning || !canRun}
                        style={{
                            background: isRunning || !canRun ? C.bg2 : `linear-gradient(135deg, ${C.amber}, #f97316)`,
                            color: isRunning || !canRun ? C.textDim : "#000",
                            border: "none",
                            borderRadius: 8,
                            padding: "10px 18px",
                            minHeight: 38,
                            cursor: isRunning || !canRun ? "not-allowed" : "pointer",
                            fontWeight: 800,
                            fontSize: 13,
                            fontFamily: "'Syne',sans-serif",
                            whiteSpace: "nowrap",
                        }}
                    >
                        {isRunning ? "Running..." : "Run backtest"}
                    </button>
                </div>
            </div>

            {status === "error" && (
                <div style={{ background: `${C.red}18`, border: `1px solid ${C.red}55`, color: C.red, padding: 12, borderRadius: 8, marginBottom: 16, fontSize: 12 }}>
                    Backtest error: {error}
                </div>
            )}

            {isRunning && (
                <div style={{ color: C.textDim, textAlign: "center", padding: 36, fontSize: 12 }}>
                    Running backtest for {form.symbol.toUpperCase()}...
                </div>
            )}

            {status === "done" && result && (
                <>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.textDim, fontSize: 11, marginBottom: 14 }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: C.green, display: "inline-block" }} />
                        Backtest complete - {result.summary?.symbol || form.symbol.toUpperCase()} from {form.start_date} to {form.end_date}
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 16 }}>
                        <MetricCard label="Total return" value={formatPct(metrics.total_return)} sub={formatCurrency(metrics.final_value)} tone={(metrics.total_return || 0) >= 0 ? "green" : "red"} />
                        <MetricCard label="CAGR" value={formatPct(metrics.cagr)} tone={(metrics.cagr || 0) >= 0 ? "green" : "red"} />
                        <MetricCard label="Sharpe ratio" value={formatNumber(metrics.sharpe)} tone={(metrics.sharpe || 0) >= 0 ? "cyan" : "red"} />
                        <MetricCard label="Max drawdown" value={formatPct(metrics.max_drawdown)} tone="red" />
                        <MetricCard label="Win rate" value={typeof metrics.win_rate === "number" ? `${metrics.win_rate.toFixed(1)}%` : "-"} tone={(metrics.win_rate || 0) >= 50 ? "green" : "red"} />
                        <MetricCard label="Total trades" value={metrics.n_trades ?? 0} tone="neutral" />
                    </div>

                    <div style={{
                        background: C.amberLow,
                        border: `1px solid ${C.amber}44`,
                        borderRadius: 8,
                        color: C.textMid,
                        padding: "11px 14px",
                        fontSize: 12,
                        marginBottom: 16,
                        lineHeight: 1.5,
                    }}>
                        {result.benchmark_notice || (
                            <>Buy-and-hold returned {formatPct(metrics.bh_return)} over the same period; the selected strategy returned {formatPct(metrics.total_return)}.</>
                        )}
                    </div>

                    <div style={{
                        background: C.bg1,
                        border: `1px solid ${C.border}`,
                        borderRadius: 8,
                        padding: 16,
                        marginBottom: 16,
                    }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
                            <div style={{ color: C.text, fontSize: 14, fontWeight: 700, fontFamily: "'Syne',sans-serif" }}>
                                Equity Curve
                            </div>
                            <div style={{ display: "flex", gap: 14, color: C.textDim, fontSize: 11 }}>
                                <span><span style={{ display: "inline-block", width: 22, height: 2, background: C.green, marginRight: 6, verticalAlign: "middle" }} />Strategy</span>
                                <span><span style={{ display: "inline-block", width: 22, borderTop: `2px dashed ${C.textDim}`, marginRight: 6, verticalAlign: "middle" }} />Buy and hold</span>
                            </div>
                        </div>
                        <ResponsiveContainer width="100%" height={280}>
                            <LineChart data={chartData} margin={{ top: 6, right: 10, left: 0, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                                <XAxis dataKey="month" tick={{ fill: C.textDim, fontSize: 10 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                                <YAxis tick={{ fill: C.textDim, fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`} />
                                <Tooltip content={<BacktestTooltip />} />
                                <Line type="monotone" dataKey="strategy" name="Strategy" stroke={C.green} strokeWidth={2.5} dot={false} />
                                <Line type="monotone" dataKey="buyHold" name="Buy and hold" stroke={C.textDim} strokeWidth={2} strokeDasharray="6 4" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <div style={{
                        background: C.bg1,
                        border: `1px solid ${C.border}`,
                        borderRadius: 8,
                        padding: 16,
                    }}>
                        <div style={{ color: C.text, fontSize: 14, fontWeight: 700, fontFamily: "'Syne',sans-serif", marginBottom: 12 }}>
                            Trade History
                        </div>
                        <TradeTable trades={result.trades || []} />
                    </div>
                </>
            )}
        </div>
    );
}
