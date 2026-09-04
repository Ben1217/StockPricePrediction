/**
 * Backtesting — a prediction, and what happened next.
 *
 * The page used to be a standalone form: type a symbol, pick a strategy, get an
 * equity curve. Nothing on it knew a prediction had ever been made, and the only
 * thing it shared with the Predictions tab was the selected ticker.
 *
 * It is now organised around the prediction under test. The panel at the top
 * names the model, the stock and the horizon being scored; the Predictions tab
 * can hand one straight to it, in which case the run starts on arrival. Below
 * the run, the model's own dated calls are joined to the bars that followed
 * them, so "predicted" and "actual" sit on the same dates and the hit rate is
 * something the reader can check rather than take on faith.
 *
 * The accuracy panel is `/predict/historical-signals` joined client-side to the
 * `price_series` the backtest already returns; the walk-forward panel below it
 * reads `/backtest/evidence`, which serves the benchmark's own out-of-sample
 * record. Those are two different backtests — a trading simulation over one
 * period, and the scorecard every model was ranked on — and this tab is where
 * they finally sit together.
 *
 * Both are next-bar only, and that is a property of the models rather than of
 * this page: nothing in the system forecasts more than one step, so the horizon
 * control offers one and explains the rest.
 */

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
import { useBacktestEvidence, useHistoricalSignals } from "../hooks/useMarketData";
import { scoreSignals } from "../utils/scoring";

const STRATEGY_OPTIONS = [
    { value: "ml_hybrid", label: "Model + Technical Analysis" },
    { value: "ta_only", label: "Technical Analysis only" },
    { value: "buy_hold", label: "Buy and Hold" },
];

/** The three bundles the backtest engine can actually replay. */
const MODEL_OPTIONS = [
    { value: "xgboost", label: "XGBoost" },
    { value: "random_forest", label: "Random Forest" },
    { value: "lstm", label: "LSTM" },
];

/**
 * Horizons the engine can score. Only the next bar is real, and not because a
 * piece of plumbing is missing — the models themselves are one-step.
 *
 * `train_model_bundles` fits a single direction bundle at horizon 1 and records
 * the requested horizons as metadata describing what it may *serve*; the
 * unified and foundation models forecast one bar and label the answer rather
 * than extrapolating it. So a 30-day option here would need a multi-horizon
 * model to exist first, which is a modelling decision rather than a backtest
 * one. They are listed and disabled rather than hidden, because a reader is
 * better served by "not this, and here is why" than by a control that was never
 * there.
 */
const HORIZON_OPTIONS = [
    { value: 1, label: "Next day (1D)", enabled: true },
    { value: 7, label: "1 week (7D) — needs a multi-horizon model", enabled: false },
    { value: 30, label: "1 month (30D) — needs a multi-horizon model", enabled: false },
];

const PANEL = "#0F1623";

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

function Panel({ title, right, children }) {
    return (
        <div style={{
            background: C.bg1,
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: 16,
        }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
                <div style={{ color: C.text, fontSize: 14, fontWeight: 700, fontFamily: "'Syne',sans-serif" }}>{title}</div>
                {right}
            </div>
            {children}
        </div>
    );
}

/**
 * Predicted versus actual, one row per dated call.
 *
 * The strip is the whole record at a glance — one mark per call, green where the
 * model got the next day's direction right. The table under it is the last eight,
 * where a reader can check a single day against the price they remember.
 */
function AccuracyPanel({ scored, symbol, modelLabel, status, message }) {
    if (status !== "ready") {
        return (
            <Panel title="Predicted vs actual">
                <div style={{ color: status === "error" ? C.red : C.textDim, fontSize: 12, padding: "10px 0", lineHeight: 1.6 }}>
                    {message}
                </div>
            </Panel>
        );
    }

    const recent = scored.rows.slice(-8).reverse();
    const strip = scored.rows.slice(-90);

    return (
        <Panel
            title="Predicted vs actual"
            right={
                <div style={{ color: C.textDim, fontSize: 11 }}>
                    {modelLabel} on {symbol} · next-day direction
                </div>
            }
        >
            {/* Excess over base rate leads, because it is the only one of these
                numbers that can be read on its own. Hit rate follows with the
                bar it has to clear printed underneath it. */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 12 }}>
                <MetricCard
                    label="Excess over base rate"
                    value={`${scored.eobr >= 0 ? "+" : ""}${scored.eobr.toFixed(1)} pp`}
                    sub={scored.eobr >= 0 ? "better than always-majority" : "worse than always-majority"}
                    tone={scored.eobr >= 0 ? "green" : "red"}
                />
                <MetricCard
                    label="Direction hit rate"
                    value={`${scored.hitRate.toFixed(1)}%`}
                    sub={`vs ${scored.majority.toFixed(1)}% always-majority`}
                    tone={scored.eobr >= 0 ? "green" : "red"}
                />
                <MetricCard
                    label="Base rate"
                    value={`${scored.baseRate.toFixed(1)}%`}
                    sub="of scored bars closed up"
                    tone="neutral"
                />
                <MetricCard
                    label="Calls scored"
                    value={scored.total}
                    sub={`${scored.hits} correct · ${scored.total - scored.hits} missed`}
                    tone="neutral"
                />
            </div>

            <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.6, marginBottom: 16 }}>
                Excess over base rate is the hit rate minus what always calling the commoner
                direction would have scored on these same bars. At or below zero the model added
                nothing. It is not a significance test — a few hundred bars is a small sample for a
                gap of a point or two.
            </div>

            <div style={{ marginBottom: 6, color: C.textDim, fontSize: 10, letterSpacing: 1, textTransform: "uppercase" }}>
                Last {strip.length} calls — oldest to newest
            </div>
            <div style={{ display: "flex", gap: 2, alignItems: "flex-end", height: 34, marginBottom: 18, overflow: "hidden" }}>
                {strip.map((row) => (
                    <div
                        key={row.date}
                        title={`${row.date} · predicted ${row.predictedUp ? "UP" : "DOWN"} · actual ${formatPct(row.actualPct)}`}
                        style={{
                            flex: "1 1 0",
                            minWidth: 2,
                            height: row.hit ? 30 : 16,
                            borderRadius: 2,
                            background: row.hit ? C.green : C.red,
                            opacity: row.hit ? 0.85 : 0.7,
                        }}
                    />
                ))}
            </div>

            <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, minWidth: 520 }}>
                    <thead>
                        <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                            {["Date", "Predicted", "Confidence", "Actual next bar", "Result"].map((header) => (
                                <th key={header} style={{ padding: "8px 10px", color: C.textDim, textAlign: "left", fontWeight: 500 }}>
                                    {header}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {recent.map((row) => (
                            <tr key={row.date} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                <td style={{ padding: "9px 10px", color: C.textMid }}>{row.date}</td>
                                <td style={{ padding: "9px 10px", color: row.predictedUp ? C.green : C.red, fontWeight: 700 }}>
                                    {row.predictedUp ? "▲ UP" : "▼ DOWN"}
                                </td>
                                <td style={{ padding: "9px 10px", color: C.textDim }}>
                                    {row.probability == null ? "-" : `${(Math.abs(row.probability - 0.5) * 200).toFixed(0)}%`}
                                </td>
                                <td style={{ padding: "9px 10px", color: row.actualUp ? C.green : C.red }}>
                                    {formatPct(row.actualPct)}
                                </td>
                                <td style={{ padding: "9px 10px" }}>
                                    <span style={{
                                        color: row.hit ? C.green : C.red,
                                        border: `1px solid ${(row.hit ? C.green : C.red)}55`,
                                        background: `${row.hit ? C.green : C.red}18`,
                                        borderRadius: 4,
                                        padding: "2px 8px",
                                        fontWeight: 700,
                                    }}>
                                        {row.hit ? "HIT" : "MISS"}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </Panel>
    );
}

/** A fraction rendered as percentage points, or an em dash. */
function pp(value, digits = 2) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const points = Number(value) * 100;
    return `${points >= 0 ? "+" : ""}${points.toFixed(digits)} pp`;
}

function num(value, digits = 4) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return Number(value).toFixed(digits);
}

/**
 * The walk-forward record: what the benchmark already measured about this symbol.
 *
 * This is the other backtest. `/backtest/run` above simulates one strategy over
 * one period; this is the out-of-sample scorecard every model was scored on —
 * purged folds, excess over base rate, and the verdict against the random walk.
 * The two have coexisted without meeting, and this panel is the meeting.
 *
 * Every column tolerates null, because a record written before the null tests
 * existed has aggregates and nothing else. Blank is the honest rendering of
 * "not measured"; zero would be a claim.
 */
function EvidencePanel({ evidence, loading, error, symbol }) {
    if (loading) {
        return (
            <Panel title="Walk-forward record">
                <div style={{ color: C.textDim, fontSize: 12, padding: "10px 0" }}>
                    Loading the benchmark record for {symbol}…
                </div>
            </Panel>
        );
    }
    if (error) {
        return (
            <Panel title="Walk-forward record">
                <div style={{ color: C.red, fontSize: 12, padding: "10px 0" }}>
                    {error.message || "The benchmark record could not be loaded."}
                </div>
            </Panel>
        );
    }
    if (!evidence) return null;

    const models = evidence.models || [];

    return (
        <Panel
            title="Walk-forward record"
            right={<div style={{ color: C.textDim, fontSize: 11 }}>out-of-sample, purged folds</div>}
        >
            {models.length === 0 ? (
                <div style={{ color: C.textDim, fontSize: 12, padding: "10px 0", lineHeight: 1.6 }}>
                    {evidence.message}
                </div>
            ) : (
                <>
                    <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, minWidth: 720 }}>
                            <thead>
                                <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                    {["Model", "Folds", "Bars", "EOBR (fold mean)", "R² vs random walk", "Verdict", "Net of costs"].map((header) => (
                                        <th key={header} style={{ padding: "8px 10px", color: C.textDim, textAlign: "left", fontWeight: 500, whiteSpace: "nowrap" }}>
                                            {header}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {models.map((entry) => {
                                    const rw = entry.vs_random_walk;
                                    const beatsRandomWalk = rw && Number(rw.r2) > 0;
                                    const netReturn = entry.economics?.overlay?.rules?.long_flat?.total_return;
                                    const eobr = entry.direction?.eobr;
                                    return (
                                        <tr key={entry.model_type} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                            <td style={{ padding: "9px 10px", color: C.text, whiteSpace: "nowrap" }}>
                                                {entry.label || entry.model_type}
                                            </td>
                                            <td style={{ padding: "9px 10px", color: C.textDim }}>{entry.n_folds ?? "—"}</td>
                                            <td style={{ padding: "9px 10px", color: C.textDim }}>{entry.total_test_rows ?? "—"}</td>
                                            <td style={{ padding: "9px 10px", color: eobr == null ? C.textDim : eobr >= 0 ? C.green : C.red, whiteSpace: "nowrap" }}>
                                                {pp(eobr)}
                                                {/* Fold-to-fold spread. A mean smaller than its own
                                                    spread is not evidence of anything, and this is
                                                    the cheapest way to show that. */}
                                                {entry.direction?.eobr_std != null && (
                                                    <span style={{ color: C.textDim, marginLeft: 6 }}>
                                                        ± {(Number(entry.direction.eobr_std) * 100).toFixed(2)}
                                                    </span>
                                                )}
                                            </td>
                                            <td style={{ padding: "9px 10px", color: rw?.r2 == null ? C.textDim : beatsRandomWalk ? C.green : C.red }}>
                                                {num(rw?.r2)}
                                            </td>
                                            <td style={{ padding: "9px 10px", color: C.textDim, whiteSpace: "nowrap" }}>
                                                {rw?.verdict
                                                    ? `${rw.verdict === "baseline_better" ? "random walk" : "model"} · p ${num(rw.p_value, 4)}`
                                                    : "—"}
                                            </td>
                                            <td style={{ padding: "9px 10px", color: netReturn == null ? C.textDim : netReturn >= 0 ? C.green : C.red }}>
                                                {netReturn == null ? "—" : formatPct(netReturn * 100)}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                    <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.6, marginTop: 12 }}>
                        R² vs random walk below zero means the random walk forecast the price better.
                        EOBR is direction accuracy minus the always-majority rate. Net of costs is the
                        long-flat rule after per-side charges, over the same bars.
                        {evidence.message?.includes("predates") && ` ${evidence.message}`}
                    </div>
                </>
            )}
        </Panel>
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

export default function BacktestTab({
    selectedTicker,
    setSelectedTicker,
    watchlist = [],
    apiConnected,
    notify,
    request,
    onRequestConsumed,
}) {
    // The symbol is deliberately absent: `selectedTicker` is already the app's
    // one source of truth for it, and copying it into local state here is what
    // forced a prop-sync effect that could fight the user's own selection.
    const [form, setForm] = useState({
        start_date: "2022-01-01",
        end_date: todayISO(),
        initial_capital: 100000,
        strategy: "ml_hybrid",
        model_type: "xgboost",
        horizon: 1,
    });
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [status, setStatus] = useState("idle");
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");
    // The configuration the displayed result belongs to. The accuracy panel is
    // keyed off this rather than off `form`, so editing the form after a run
    // never scores one model's calls against another model's backtest.
    const [ranConfig, setRanConfig] = useState(null);

    /**
     * Run one backtest and record what it was.
     *
     * `carriedFrom` is the prediction this run came from, when the Predictions
     * tab started it. It rides along into `ranConfig` rather than living in its
     * own state so that everything describing the result on screen — symbol,
     * model, period, provenance — lands in a single update.
     */
    const runWith = async (config, carriedFrom = null) => {
        setStatus("running");
        setError("");
        const symbol = String(config.symbol || "").trim().toUpperCase();
        try {
            const response = await runBacktest({
                symbol,
                start_date: config.start_date,
                end_date: config.end_date,
                initial_capital: Number(config.initial_capital),
                strategy: config.strategy,
                model_type: config.model_type,
            });
            setResult(response);
            setRanConfig({ ...config, symbol, carriedFrom });
            setStatus("done");
            notify?.(`Backtest complete for ${response.summary?.symbol || symbol}`);
        } catch (err) {
            setResult(null);
            setRanConfig(null);
            setError(err.message || "Backtest failed");
            setStatus("error");
        }
    };

    // A prediction handed over from the Predictions tab runs on arrival — the
    // button there said "backtest this prediction", so making the user press a
    // second one here would be asking twice for the same thing.
    //
    // The request is consumed rather than remembered. This tab unmounts every
    // time the user leaves it, so a request left standing in App would re-run
    // the backtest on each return visit; clearing it at the source is what makes
    // the handover happen exactly once.
    useEffect(() => {
        if (!request || !apiConnected) return;
        // set-state-in-effect fires because `runWith` flips the panel to its
        // running state before awaiting. That is the request starting, not a
        // render cascade: an effect is the only place a prop change can begin
        // one, and a spinner that waits for the response is worse than none.
        // eslint-disable-next-line react-hooks/set-state-in-effect
        runWith({ ...form, symbol: request.symbol }, request);
        // Nothing else is stored here: the run records what it ran on when it
        // lands, and the request is cleared at its source so this fires once.
        onRequestConsumed?.();
        // Deliberately keyed on the handover alone. `form` is read, not tracked:
        // this body runs in the render where `request.at` changed, so the value
        // it closes over is the current one, and listing it here would re-run
        // the backtest on every keystroke in the advanced panel.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [request?.at, apiConnected]);

    // Independent of any run: this is what the benchmark already knows about the
    // symbol, so it loads on arrival rather than waiting for a backtest.
    const evidenceQuery = useBacktestEvidence(selectedTicker, { enabled: apiConnected });

    const signalsQuery = useHistoricalSignals(ranConfig?.symbol, {
        days: 365,
        modelType: ranConfig?.model_type,
        enabled: apiConnected && Boolean(ranConfig) && ranConfig?.strategy === "ml_hybrid",
    });

    const chartData = useMemo(
        () => monthlyCurve(result?.equity_curve || [], result?.bh_curve || []),
        [result]
    );

    const scored = useMemo(
        () => scoreSignals(signalsQuery.data, result?.price_series),
        [signalsQuery.data, result]
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

    const isRunning = status === "running";
    const canRun = apiConnected
        && !isRunning
        && String(selectedTicker || "").trim()
        && form.start_date
        && form.end_date
        && Number(form.initial_capital) > 0;
    const metrics = result?.metrics || {};

    const modelLabel = MODEL_OPTIONS.find((option) => option.value === ranConfig?.model_type)?.label
        || ranConfig?.model_type
        || "";

    // What the accuracy panel can say, and why. A 404 from historical-signals is
    // the ordinary answer for an untrained bundle, not a fault, so it reads as an
    // instruction rather than an error.
    let accuracyStatus = "empty";
    let accuracyMessage = "Run a backtest to score the model's calls against what the market did.";
    if (ranConfig && ranConfig.strategy !== "ml_hybrid") {
        accuracyMessage = "This run used technical signals only. Choose “Model + Technical Analysis” to score a model's predictions.";
    } else if (ranConfig && signalsQuery.isPending) {
        accuracyMessage = `Loading ${modelLabel} calls for ${ranConfig.symbol}…`;
    } else if (ranConfig && signalsQuery.error) {
        // 404 and 409 are one situation wearing two status codes: there is no
        // model that forecasts direction for this symbol. Only the cause and
        // the remedy differ, and neither is a fault — rendering them red as
        // "the request failed" sends a reader hunting a network problem they
        // do not have.
        const status = signalsQuery.error.status;
        if (status === 404) {
            accuracyStatus = "empty";
            accuracyMessage =
                `No ${modelLabel} model for ${ranConfig.symbol} yet. Training starts on its own ` +
                `when the symbol is selected; this panel fills in once it finishes.`;
        } else if (status === 409) {
            // Deliberately not "it will retrain itself". Readiness decides by
            // whether a bundle file is present and how old it is; it never
            // reads what the bundle was trained for. So a legacy price bundle
            // satisfies the check, auto-preparation skips the symbol, and this
            // state persists until someone forces a retrain.
            accuracyStatus = "empty";
            accuracyMessage =
                `The ${modelLabel} bundle for ${ranConfig.symbol} was trained to predict price, ` +
                `not direction, so its calls cannot be scored. This one will not fix itself: ` +
                `readiness checks that a bundle exists, not what it was trained for, so a forced ` +
                `retrain is needed.`;
        } else {
            accuracyStatus = "error";
            accuracyMessage = signalsQuery.error.message || "The model's past calls could not be loaded.";
        }
    } else if (ranConfig && !scored) {
        accuracyMessage = `${modelLabel} produced no dated calls inside ${ranConfig.start_date} to ${ranConfig.end_date}. Widen the date range to overlap the model's signal history.`;
    } else if (scored) {
        accuracyStatus = "ready";
    }

    const symbolOptions = watchlist.includes(selectedTicker)
        ? watchlist
        : [selectedTicker, ...watchlist].filter(Boolean);

    if (!apiConnected) {
        return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>Connect to API</div>;
    }

    return (
        <div className="fade-up" style={{ display: "grid", gap: 16, paddingBottom: 36 }}>
            <div>
                <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text, marginBottom: 6 }}>
                    Backtesting Validation
                </h1>
                <div style={{ color: C.textDim, fontSize: 12 }}>
                    Score a prediction against the bars it did not see, and compare it with buy-and-hold.
                </div>
            </div>

            {/* ── The prediction under test ───────────────────────── */}
            <div style={{
                background: PANEL,
                border: `1px solid ${C.border}`,
                borderLeft: `3px solid ${C.amber}`,
                borderRadius: 8,
                padding: 16,
            }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
                    <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1.4, textTransform: "uppercase", fontWeight: 700 }}>
                        Prediction under test
                    </div>
                    {ranConfig?.carriedFrom && (
                        <div style={{ color: C.amber, fontSize: 11 }}>
                            Carried over from Predictions · {ranConfig.carriedFrom.horizonLabel}
                            {ranConfig.carriedFrom.direction ? ` · called ${ranConfig.carriedFrom.direction}` : ""}
                        </div>
                    )}
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, alignItems: "end" }}>
                    <Field label="Stock">
                        <select
                            value={selectedTicker}
                            onChange={(event) => setSelectedTicker?.(event.target.value)}
                            style={inputStyle}
                        >
                            {symbolOptions.map((symbol) => (
                                <option key={symbol} value={symbol}>{symbol}</option>
                            ))}
                        </select>
                    </Field>
                    <Field label="Model">
                        <select
                            value={form.model_type}
                            onChange={(event) => setForm((current) => ({ ...current, model_type: event.target.value }))}
                            style={inputStyle}
                        >
                            {MODEL_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                        </select>
                    </Field>
                    <Field label="Horizon">
                        <select
                            value={form.horizon}
                            onChange={(event) => setForm((current) => ({ ...current, horizon: Number(event.target.value) }))}
                            style={inputStyle}
                        >
                            {HORIZON_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value} disabled={!option.enabled}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                    </Field>
                    <button
                        onClick={() => runWith({ ...form, symbol: selectedTicker })}
                        disabled={!canRun}
                        style={{
                            background: !canRun ? C.bg2 : `linear-gradient(135deg, ${C.amber}, #f97316)`,
                            color: !canRun ? C.textDim : "#000",
                            border: "none",
                            borderRadius: 8,
                            padding: "10px 18px",
                            minHeight: 38,
                            cursor: !canRun ? "not-allowed" : "pointer",
                            fontWeight: 800,
                            fontSize: 13,
                            fontFamily: "'Syne',sans-serif",
                            whiteSpace: "nowrap",
                        }}
                    >
                        {isRunning ? "Running..." : "Run backtest"}
                    </button>
                </div>

                <button
                    type="button"
                    onClick={() => setShowAdvanced((open) => !open)}
                    style={{
                        background: "transparent",
                        border: "none",
                        color: C.textDim,
                        fontSize: 11,
                        fontFamily: "'DM Mono',monospace",
                        cursor: "pointer",
                        padding: "12px 0 0",
                    }}
                >
                    {showAdvanced ? "▾" : "▸"} Period, capital and strategy
                </button>

                {showAdvanced && (
                    <div style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                        gap: 12,
                        marginTop: 12,
                        paddingTop: 14,
                        borderTop: `1px solid ${C.border}`,
                    }}>
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
                    </div>
                )}
            </div>

            <EvidencePanel
                evidence={evidenceQuery.data}
                loading={evidenceQuery.isPending}
                error={evidenceQuery.error}
                symbol={selectedTicker}
            />

            {status === "error" && (
                <div style={{ background: `${C.red}18`, border: `1px solid ${C.red}55`, color: C.red, padding: 12, borderRadius: 8, fontSize: 12 }}>
                    Backtest error: {error}
                </div>
            )}

            {isRunning && (
                <div style={{ color: C.textDim, textAlign: "center", padding: 36, fontSize: 12 }}>
                    Running backtest for {String(selectedTicker || "").toUpperCase()}...
                </div>
            )}

            {status === "done" && result && ranConfig && (
                <>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.textDim, fontSize: 11 }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: C.green, display: "inline-block" }} />
                        {result.summary?.symbol || ranConfig.symbol} · {ranConfig.start_date} to {ranConfig.end_date}
                        {ranConfig.strategy === "ml_hybrid" ? ` · ${modelLabel}` : ""}
                    </div>

                    {/* The engine falls back to technical signals when a bundle
                        will not load. Silence here would let a reader credit a
                        model for a run it took no part in. */}
                    {result.summary?.ml_status === "fallback_ta_only" && (
                        <div style={{ background: `${C.amber}14`, border: `1px solid ${C.amber}44`, color: C.textMid, padding: "11px 14px", borderRadius: 8, fontSize: 12, lineHeight: 1.5 }}>
                            No {modelLabel} predictions were available for this period, so the run used technical
                            signals alone. The returns below are not the model's.
                        </div>
                    )}

                    <AccuracyPanel
                        scored={scored}
                        symbol={ranConfig.symbol}
                        modelLabel={modelLabel}
                        status={accuracyStatus}
                        message={accuracyMessage}
                    />

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 }}>
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
                        lineHeight: 1.5,
                    }}>
                        {result.benchmark_notice || (
                            <>Buy-and-hold returned {formatPct(metrics.bh_return)} over the same period; the selected strategy returned {formatPct(metrics.total_return)}.</>
                        )}
                    </div>

                    <Panel
                        title="Equity Curve"
                        right={
                            <div style={{ display: "flex", gap: 14, color: C.textDim, fontSize: 11 }}>
                                <span><span style={{ display: "inline-block", width: 22, height: 2, background: C.green, marginRight: 6, verticalAlign: "middle" }} />Strategy</span>
                                <span><span style={{ display: "inline-block", width: 22, borderTop: `2px dashed ${C.textDim}`, marginRight: 6, verticalAlign: "middle" }} />Buy and hold</span>
                            </div>
                        }
                    >
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
                    </Panel>

                    <Panel title="Trade History">
                        <TradeTable trades={result.trades || []} />
                    </Panel>
                </>
            )}
        </div>
    );
}
