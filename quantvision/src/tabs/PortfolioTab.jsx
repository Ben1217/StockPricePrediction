/**
 * Portfolio — what you own, and whether the money is split well.
 *
 * The page used to open with four invented positions (AAPL 50 @ $170 and
 * friends) that looked exactly like a real holding until you read the source.
 * It also priced them one HTTP request at a time and left "daily change" as an
 * em dash. None of that survives here: holdings start empty, come from the
 * person using the page, and persist in the browser.
 *
 * The redesign is built around one question, because that question is what
 * portfolio optimization actually answers:
 *
 *     Keeping exactly the stocks you already own, is there a better way to
 *     divide your money between them?
 *
 * So the centrepiece is a comparison, not a recommendation. Your current split
 * and the optimizer's split are scored through the *same* endpoint over the
 * *same* window with one variable changed — the weights — and shown side by
 * side. A reader who has never heard the word "Sharpe" can still see that one
 * column swings less than the other for a similar return, and that is the whole
 * idea of optimization in one glance.
 *
 * Everything on this page is measured, never modelled. The returns, volatility
 * and correlations are what these stocks did over the lookback window. The tab
 * deliberately makes no forecast — forecasting lives on the Predictions tab,
 * and conflating the two is how a backtest becomes a promise.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";

import { C } from "../utils/data";
import {
    fetchCorrelation,
    fetchPortfolioMetrics,
    fetchRiskAlerts,
    optimizePortfolio,
} from "../utils/api";
import { useQuotes } from "../hooks/useMarketData";
import { StatCard, Section, Hint } from "../components/UIComponents";

const PIE_COLORS = [C.amber, C.cyan, C.green, C.purple, C.red, "#f97316", "#ec4899", "#8b5cf6"];
const HOLDINGS_KEY = "qv_holdings";

/**
 * The four objectives the backend accepts, in the words of what they solve for.
 * `max_sharpe` is a return maximisation under a volatility cap rather than a
 * true Sharpe maximum, and the hint says so instead of implying otherwise.
 */
const METHODS = {
    max_sharpe: {
        label: "Best risk/reward",
        technical: "max_sharpe",
        hint: "Aims for the most return per unit of risk. Solved as: the highest expected return while keeping annual volatility under 20%.",
    },
    min_volatility: {
        label: "Calmest ride",
        technical: "min_volatility",
        hint: "The smallest expected price swings, even if that means lower returns.",
    },
    max_return: {
        label: "Highest return",
        technical: "max_return",
        hint: "Chases the highest expected return and ignores risk entirely. Usually concentrates into one or two names.",
    },
    risk_parity: {
        label: "Equal risk share",
        technical: "risk_parity",
        hint: "Every holding contributes the same amount of risk, so no single stock dominates the portfolio's behaviour.",
    },
};

/* ── Persistence ─────────────────────────────────────────────── */

/**
 * Holdings live in the browser, not on the server.
 *
 * Every read and write is guarded: a private window, cleared site data or a
 * browser set to block storage all make these throw rather than return empty,
 * and a portfolio page that white-screens because of a storage setting is worse
 * than one that starts empty.
 */
function loadHoldings() {
    try {
        const raw = window.localStorage.getItem(HOLDINGS_KEY);
        const parsed = raw ? JSON.parse(raw) : null;
        if (!Array.isArray(parsed)) return [];
        return parsed
            .map((row) => ({
                ticker: String(row?.ticker || "").trim().toUpperCase(),
                shares: Number(row?.shares),
                avgCost: Number(row?.avgCost),
            }))
            .filter((row) => row.ticker && Number.isFinite(row.shares) && row.shares > 0
                && Number.isFinite(row.avgCost) && row.avgCost > 0);
    } catch {
        return [];
    }
}

function saveHoldings(holdings) {
    try {
        window.localStorage.setItem(HOLDINGS_KEY, JSON.stringify(holdings));
    } catch {
        // Storage unavailable. The page still works for this session.
    }
}

/* ── Formatting ──────────────────────────────────────────────── */

const money = (value) =>
    Number.isFinite(value)
        ? `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : "—";

const compactMoney = (value) =>
    Number.isFinite(value)
        ? `$${Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(1)}K` : value.toFixed(0)}`
        : "—";

const pct = (fraction, digits = 1) =>
    Number.isFinite(fraction) ? `${(fraction * 100).toFixed(digits)}%` : "—";

const signedPct = (fraction, digits = 1) =>
    Number.isFinite(fraction) ? `${fraction >= 0 ? "+" : ""}${(fraction * 100).toFixed(digits)}%` : "—";

/* ── Pieces ──────────────────────────────────────────────────── */

function WeightBar({ label, weight, color, delta }) {
    const width = Math.max(0, Math.min(100, weight * 100));
    return (
        <div style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                <span style={{ color: C.amber, fontWeight: 700, fontSize: 12 }}>{label}</span>
                <span style={{ color: C.text, fontSize: 12, fontFamily: "'DM Mono',monospace" }}>
                    {pct(weight)}
                    {delta !== undefined && Math.abs(delta) >= 0.005 && (
                        <span style={{ color: delta > 0 ? C.green : C.red, marginLeft: 8, fontSize: 11 }}>
                            {delta > 0 ? "▲" : "▼"} {Math.abs(delta * 100).toFixed(1)}
                        </span>
                    )}
                </span>
            </div>
            <div style={{ background: C.bg2, borderRadius: 4, height: 8, overflow: "hidden" }}>
                <div style={{ width: `${width}%`, height: "100%", background: color, borderRadius: 4 }} />
            </div>
        </div>
    );
}

/**
 * One measured quantity, current split beside optimized split.
 *
 * `better` decides which side is highlighted, and it is passed in rather than
 * inferred: higher is better for return and Sharpe, lower is better for
 * volatility and drawdown, and a component that guessed would eventually praise
 * the wrong column.
 */
function CompareRow({ label, plain, current, suggested, format, higherIsBetter, hint }) {
    const hasBoth = Number.isFinite(current) && Number.isFinite(suggested);
    const improved = hasBoth && (higherIsBetter ? suggested > current : suggested < current);
    const same = hasBoth && Math.abs(suggested - current) < 1e-6;

    return (
        <tr style={{ borderBottom: `1px solid ${C.border}22` }}>
            <td style={{ padding: "10px 12px" }}>
                <div style={{ color: C.text, fontSize: 12, display: "flex", alignItems: "center" }}>
                    {label}
                    {hint && <Hint text={hint} />}
                </div>
                <div style={{ color: C.textDim, fontSize: 10.5, marginTop: 2 }}>{plain}</div>
            </td>
            <td style={{ padding: "10px 12px", textAlign: "right", color: C.textMid, fontFamily: "'DM Mono',monospace", fontSize: 13 }}>
                {format(current)}
            </td>
            <td style={{
                padding: "10px 12px", textAlign: "right", fontFamily: "'DM Mono',monospace", fontSize: 13,
                color: same ? C.textMid : improved ? C.green : C.red, fontWeight: 700,
            }}>
                {format(suggested)}
            </td>
        </tr>
    );
}

function EmptyState({ children }) {
    return (
        <div style={{ color: C.textDim, fontSize: 12, padding: "18px 0", lineHeight: 1.7 }}>{children}</div>
    );
}

/* ── The tab ─────────────────────────────────────────────────── */

export default function PortfolioTab({ notify, apiConnected, setSelectedTicker, watchlist = [] }) {
    const [holdings, setHoldings] = useState(loadHoldings);
    const [draft, setDraft] = useState({ ticker: "", shares: "", avgCost: "" });
    const [method, setMethod] = useState("max_sharpe");
    const [optimizing, setOptimizing] = useState(false);
    const [optimizeError, setOptimizeError] = useState("");
    const [comparison, setComparison] = useState(null);

    useEffect(() => { saveHoldings(holdings); }, [holdings]);

    const tickers = useMemo(() => holdings.map((h) => h.ticker), [holdings]);

    // One batched request for every holding. The previous version issued one
    // per position in a loop, which is N round trips to draw one table.
    const quotesQuery = useQuotes(tickers, apiConnected && tickers.length > 0);
    // The `?? {}` default lives inside the memo on purpose: written outside, it
    // builds a fresh object on every render and the memo below never holds.
    const quotesData = quotesQuery.data;

    const priced = useMemo(() => holdings.map((holding) => {
        const quote = (quotesData ?? {})[holding.ticker];
        const live = Number(quote?.price);
        // Falling back to average cost would render an unpriced position as a
        // flat 0.00% gain, which reads as data rather than as its absence.
        const hasPrice = Number.isFinite(live) && live > 0;
        const price = hasPrice ? live : null;
        const value = hasPrice ? price * holding.shares : null;
        const cost = holding.avgCost * holding.shares;
        return {
            ...holding,
            price,
            hasPrice,
            value,
            cost,
            dayChange: Number.isFinite(Number(quote?.change)) ? Number(quote.change) / 100 : null,
            pnl: hasPrice ? value - cost : null,
            pnlPct: hasPrice && cost > 0 ? (value - cost) / cost : null,
        };
    }), [holdings, quotesData]);

    const totals = useMemo(() => {
        const pricedRows = priced.filter((row) => row.hasPrice);
        const value = pricedRows.reduce((sum, row) => sum + row.value, 0);
        const cost = pricedRows.reduce((sum, row) => sum + row.cost, 0);
        const dayMove = pricedRows.reduce(
            (sum, row) => sum + (Number.isFinite(row.dayChange) ? row.dayChange * row.value : 0),
            0,
        );
        return {
            value,
            cost,
            pnl: value - cost,
            pnlPct: cost > 0 ? (value - cost) / cost : null,
            dayPct: value > 0 ? dayMove / value : null,
            pricedCount: pricedRows.length,
        };
    }, [priced]);

    /** Current split, as fractions of portfolio value. The optimizer's input. */
    const currentWeights = useMemo(() => {
        if (!(totals.value > 0)) return null;
        const weights = {};
        priced.forEach((row) => {
            if (row.hasPrice) weights[row.ticker] = row.value / totals.value;
        });
        return Object.keys(weights).length ? weights : null;
    }, [priced, totals.value]);

    const canOptimize = apiConnected && currentWeights && Object.keys(currentWeights).length >= 2;

    const addPosition = useCallback(() => {
        const ticker = draft.ticker.trim().toUpperCase();
        const shares = Number(draft.shares);
        const avgCost = Number(draft.avgCost);
        if (!ticker) { notify?.("Enter a ticker"); return; }
        if (!Number.isFinite(shares) || shares <= 0) { notify?.("Shares must be greater than zero"); return; }
        if (!Number.isFinite(avgCost) || avgCost <= 0) { notify?.("Average cost must be greater than zero"); return; }
        if (holdings.some((h) => h.ticker === ticker)) { notify?.(`${ticker} is already in the portfolio`); return; }

        setHoldings((current) => [...current, { ticker, shares, avgCost }]);
        setDraft({ ticker: "", shares: "", avgCost: "" });
        setComparison(null);
        notify?.(`Added ${ticker}`);
    }, [draft, holdings, notify]);

    const removePosition = useCallback((ticker) => {
        setHoldings((current) => current.filter((row) => row.ticker !== ticker));
        setComparison(null);
        notify?.(`Removed ${ticker}`);
    }, [notify]);

    /**
     * Score both splits, then ask what is wrong with the current one.
     *
     * The two metric calls are the comparison: identical symbols, identical
     * lookback, identical maths, weights the only difference. Correlation and
     * alerts describe the portfolio as it stands today, so they are keyed to the
     * current weights rather than to the suggestion.
     */
    const runOptimization = useCallback(async () => {
        if (!currentWeights) return;
        setOptimizing(true);
        setOptimizeError("");
        const symbols = Object.keys(currentWeights);
        try {
            const optimized = await optimizePortfolio({ symbols, method });
            const suggestedWeights = optimized.weights || {};

            const [currentMetrics, suggestedMetrics, correlation, alerts] = await Promise.all([
                fetchPortfolioMetrics(symbols, { weights: currentWeights }),
                fetchPortfolioMetrics(symbols, { weights: suggestedWeights }),
                fetchCorrelation(symbols).catch(() => null),
                fetchRiskAlerts(symbols, currentWeights).catch(() => null),
            ]);

            setComparison({
                method: optimized.method || method,
                symbols,
                currentWeights,
                suggestedWeights,
                current: currentMetrics.metrics || {},
                suggested: suggestedMetrics.metrics || {},
                correlation,
                alerts,
            });
            notify?.("Comparison ready");
        } catch (err) {
            setComparison(null);
            setOptimizeError(err.message || "The optimization could not be run.");
        } finally {
            setOptimizing(false);
        }
    }, [currentWeights, method, notify]);

    if (!apiConnected) {
        return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>Connect to the API server to use the portfolio.</div>;
    }

    const inputStyle = {
        background: C.bg2, color: C.text, border: `1px solid ${C.border}`, borderRadius: 6,
        padding: "8px 12px", fontSize: 12, fontFamily: "'DM Mono',monospace",
    };

    const suggestions = watchlist.filter((symbol) => !tickers.includes(symbol));

    return (
        <div className="fade-up" style={{ display: "grid", gap: 16, paddingBottom: 36 }}>
            <div>
                <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text, marginBottom: 6 }}>
                    Portfolio
                </h1>
                <p style={{ color: C.textMid, fontSize: 13, lineHeight: 1.7, maxWidth: "72ch", margin: 0 }}>
                    Enter what you own. Then answer the one question portfolio optimization exists to
                    answer: <strong style={{ color: C.text }}>keeping exactly these stocks, is there a
                    better way to divide your money between them?</strong> The optimizer looks at how
                    each stock moved over the past year — how much it returned, how much it swung, and
                    how often the stocks moved together — and finds the split with the best trade-off.
                    It cannot tell you which stocks to own, and every number here is measured on
                    history rather than forecast.
                </p>
            </div>

            {/* ── Summary ─────────────────────────────────────── */}
            {holdings.length > 0 && (
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                    <StatCard
                        label="Market value" value={compactMoney(totals.value)}
                        sub={`${totals.pricedCount} of ${holdings.length} priced`} color={C.amber}
                        hint="What the holdings are worth at the latest quote."
                    />
                    <StatCard
                        label="Cost basis" value={compactMoney(totals.cost)}
                        sub="What you paid" color={C.textMid}
                        hint="Shares multiplied by the average cost you entered."
                    />
                    <StatCard
                        label="Open P&L" value={compactMoney(totals.pnl)}
                        sub={signedPct(totals.pnlPct, 2)} color={totals.pnl >= 0 ? C.green : C.red}
                        hint="Market value minus cost basis, across every priced position."
                    />
                    <StatCard
                        label="Today" value={signedPct(totals.dayPct, 2)}
                        sub="Value-weighted" color={Number(totals.dayPct) >= 0 ? C.green : C.red}
                        hint="Each position's move today, weighted by how much of the portfolio it is."
                    />
                </div>
            )}

            {/* ── Holdings ────────────────────────────────────── */}
            <Section
                title="Holdings"
                hint="Stored in this browser only. Click a row to make that ticker the app's selected symbol, so the Predictions and Backtest tabs follow it."
            >
                {holdings.length === 0 ? (
                    <EmptyState>
                        No positions yet. Add one below to begin — the optimizer needs at least two
                        holdings before it has a split to reason about.
                        {suggestions.length > 0 && (
                            <div style={{ marginTop: 10 }}>
                                From your watchlist:{" "}
                                {suggestions.map((symbol) => (
                                    <button
                                        key={symbol}
                                        type="button"
                                        onClick={() => setDraft((d) => ({ ...d, ticker: symbol }))}
                                        style={{
                                            background: "transparent", border: `1px solid ${C.border}`,
                                            color: C.amber, borderRadius: 5, padding: "3px 9px",
                                            marginRight: 6, fontSize: 11, cursor: "pointer",
                                            fontFamily: "'DM Mono',monospace",
                                        }}
                                    >
                                        {symbol}
                                    </button>
                                ))}
                            </div>
                        )}
                    </EmptyState>
                ) : (
                    <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, minWidth: 760 }}>
                            <thead>
                                <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                    {["Ticker", "Shares", "Avg cost", "Price", "Today", "Value", "P&L", "Weight", ""].map((header) => (
                                        <th key={header} style={{
                                            padding: "8px 12px", color: C.textDim, fontWeight: 400,
                                            textAlign: header === "Ticker" ? "left" : "right",
                                        }}>{header}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {priced.map((row) => (
                                    <tr
                                        key={row.ticker}
                                        onClick={() => setSelectedTicker?.(row.ticker)}
                                        style={{ borderBottom: `1px solid ${C.border}22`, cursor: "pointer" }}
                                    >
                                        <td style={{ padding: "8px 12px", color: C.amber, fontWeight: 700 }}>{row.ticker}</td>
                                        <td style={{ padding: "8px 12px", color: C.text, textAlign: "right" }}>{row.shares}</td>
                                        <td style={{ padding: "8px 12px", color: C.textMid, textAlign: "right" }}>{money(row.avgCost)}</td>
                                        <td style={{ padding: "8px 12px", color: row.hasPrice ? C.text : C.textDim, textAlign: "right" }}>
                                            {row.hasPrice ? money(row.price) : "no quote"}
                                        </td>
                                        <td style={{
                                            padding: "8px 12px", textAlign: "right",
                                            color: row.dayChange == null ? C.textDim : row.dayChange >= 0 ? C.green : C.red,
                                        }}>
                                            {row.dayChange == null ? "—" : signedPct(row.dayChange, 2)}
                                        </td>
                                        <td style={{ padding: "8px 12px", color: C.text, textAlign: "right", fontWeight: 700 }}>
                                            {row.hasPrice ? money(row.value) : "—"}
                                        </td>
                                        <td style={{
                                            padding: "8px 12px", textAlign: "right", fontWeight: 700,
                                            color: row.pnl == null ? C.textDim : row.pnl >= 0 ? C.green : C.red,
                                        }}>
                                            {row.pnl == null ? "—" : `${money(row.pnl)} (${signedPct(row.pnlPct, 1)})`}
                                        </td>
                                        <td style={{ padding: "8px 12px", color: C.textMid, textAlign: "right" }}>
                                            {currentWeights?.[row.ticker] != null ? pct(currentWeights[row.ticker]) : "—"}
                                        </td>
                                        <td style={{ padding: "8px 12px", textAlign: "right" }}>
                                            <button
                                                onClick={(event) => { event.stopPropagation(); removePosition(row.ticker); }}
                                                aria-label={`Remove ${row.ticker}`}
                                                style={{ background: "transparent", border: "none", color: C.red, cursor: "pointer", fontSize: 14 }}
                                            >
                                                ✕
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                <div style={{ display: "flex", gap: 8, alignItems: "flex-end", flexWrap: "wrap", marginTop: 16, paddingTop: 14, borderTop: `1px solid ${C.border}` }}>
                    <label style={{ display: "grid", gap: 4 }}>
                        <span style={{ fontSize: 10, color: C.textDim }}>Ticker</span>
                        <input
                            value={draft.ticker}
                            onChange={(e) => setDraft((d) => ({ ...d, ticker: e.target.value.toUpperCase() }))}
                            placeholder="AAPL"
                            style={{ ...inputStyle, width: 90 }}
                        />
                    </label>
                    <label style={{ display: "grid", gap: 4 }}>
                        <span style={{ fontSize: 10, color: C.textDim }}>Shares</span>
                        <input
                            type="number" min="0" step="any" value={draft.shares}
                            onChange={(e) => setDraft((d) => ({ ...d, shares: e.target.value }))}
                            placeholder="10"
                            style={{ ...inputStyle, width: 90 }}
                        />
                    </label>
                    <label style={{ display: "grid", gap: 4 }}>
                        <span style={{ fontSize: 10, color: C.textDim }}>Avg cost</span>
                        <input
                            type="number" min="0" step="any" value={draft.avgCost}
                            onChange={(e) => setDraft((d) => ({ ...d, avgCost: e.target.value }))}
                            placeholder="150.00"
                            style={{ ...inputStyle, width: 110 }}
                        />
                    </label>
                    <button
                        onClick={addPosition}
                        style={{
                            background: C.amber, color: "#000", border: "none", borderRadius: 6,
                            padding: "9px 18px", fontSize: 12, fontWeight: 700, cursor: "pointer",
                        }}
                    >
                        Add position
                    </button>
                </div>
            </Section>

            {/* ── The comparison ──────────────────────────────── */}
            <Section
                title="Current split vs optimized split"
                hint="Both columns are scored by the same endpoint, over the same window, with the weights as the only difference."
                right={
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 10, color: C.textDim, textTransform: "uppercase", letterSpacing: 1 }}>
                                Goal
                            </span>
                            <Hint text={METHODS[method].hint} />
                            <select value={method} onChange={(e) => setMethod(e.target.value)} style={inputStyle}>
                                {Object.entries(METHODS).map(([key, entry]) => (
                                    <option key={key} value={key}>{entry.label}</option>
                                ))}
                            </select>
                        </label>
                        <button
                            onClick={runOptimization}
                            disabled={!canOptimize || optimizing}
                            style={{
                                background: !canOptimize || optimizing ? C.bg2 : `linear-gradient(135deg, ${C.amber}, #f97316)`,
                                color: !canOptimize || optimizing ? C.textDim : "#000",
                                border: "none", borderRadius: 8, padding: "9px 18px", fontSize: 12,
                                fontWeight: 800, fontFamily: "'Syne',sans-serif",
                                cursor: !canOptimize || optimizing ? "not-allowed" : "pointer",
                            }}
                        >
                            {optimizing ? "Working…" : "Compare"}
                        </button>
                    </div>
                }
            >
                {!canOptimize && (
                    <EmptyState>
                        {holdings.length < 2
                            ? "Add at least two positions. With one holding there is no split to optimize — all the money is already in one place."
                            : "Waiting for live prices. Weights are derived from market value, so at least two positions need a quote before the comparison can run."}
                    </EmptyState>
                )}

                {optimizeError && (
                    <div style={{ background: `${C.red}18`, border: `1px solid ${C.red}55`, color: C.red, padding: 12, borderRadius: 8, fontSize: 12 }}>
                        {optimizeError}
                    </div>
                )}

                {canOptimize && !comparison && !optimizeError && (
                    <EmptyState>
                        Press <strong style={{ color: C.text }}>Compare</strong> to score your split
                        against the optimizer&rsquo;s over the last year of daily returns.
                    </EmptyState>
                )}

                {comparison && (
                    <div style={{ display: "grid", gap: 18 }}>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 18 }}>
                            <div>
                                <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1.4, textTransform: "uppercase", marginBottom: 12 }}>
                                    Your split now
                                </div>
                                {comparison.symbols.map((symbol, index) => (
                                    <WeightBar
                                        key={symbol}
                                        label={symbol}
                                        weight={comparison.currentWeights[symbol] ?? 0}
                                        color={PIE_COLORS[index % PIE_COLORS.length]}
                                    />
                                ))}
                            </div>
                            <div>
                                <div style={{ color: C.amber, fontSize: 10, letterSpacing: 1.4, textTransform: "uppercase", marginBottom: 12 }}>
                                    Optimizer&rsquo;s split · {METHODS[comparison.method]?.label || comparison.method}
                                    {METHODS[comparison.method] && (
                                        // The textbook name, kept visible so the plain-English
                                        // label above connects to the term a reader will meet
                                        // everywhere else.
                                        <span style={{ color: C.textDim, textTransform: "none", letterSpacing: 0, marginLeft: 6 }}>
                                            ({METHODS[comparison.method].technical})
                                        </span>
                                    )}
                                </div>
                                {comparison.symbols.map((symbol, index) => (
                                    <WeightBar
                                        key={symbol}
                                        label={symbol}
                                        weight={comparison.suggestedWeights[symbol] ?? 0}
                                        color={PIE_COLORS[index % PIE_COLORS.length]}
                                        delta={(comparison.suggestedWeights[symbol] ?? 0) - (comparison.currentWeights[symbol] ?? 0)}
                                    />
                                ))}
                            </div>
                        </div>

                        <div style={{ overflowX: "auto" }}>
                            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 520 }}>
                                <thead>
                                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                        <th style={{ padding: "8px 12px", color: C.textDim, textAlign: "left", fontWeight: 400, fontSize: 10, letterSpacing: 1, textTransform: "uppercase" }}>
                                            Measured over the last year
                                        </th>
                                        <th style={{ padding: "8px 12px", color: C.textDim, textAlign: "right", fontWeight: 400, fontSize: 10, letterSpacing: 1, textTransform: "uppercase" }}>
                                            Your split
                                        </th>
                                        <th style={{ padding: "8px 12px", color: C.amber, textAlign: "right", fontWeight: 400, fontSize: 10, letterSpacing: 1, textTransform: "uppercase" }}>
                                            Optimized
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <CompareRow
                                        label="Annual return" plain="What it would have grown by, per year"
                                        current={comparison.current.annual_return} suggested={comparison.suggested.annual_return}
                                        format={(v) => signedPct(v, 2)} higherIsBetter
                                        hint="The lookback window's return, annualised. History, not a forecast."
                                    />
                                    <CompareRow
                                        label="Volatility" plain="How much the value swings in a typical year"
                                        current={comparison.current.annual_volatility} suggested={comparison.suggested.annual_volatility}
                                        format={(v) => pct(v, 2)} higherIsBetter={false}
                                        hint="Annualised standard deviation. Lower means a smoother ride in both directions."
                                    />
                                    <CompareRow
                                        label="Sharpe ratio" plain="Return earned per unit of risk taken"
                                        current={comparison.current.sharpe_ratio} suggested={comparison.suggested.sharpe_ratio}
                                        format={(v) => (Number.isFinite(v) ? v.toFixed(2) : "—")} higherIsBetter
                                        hint="The single number optimization is usually judged on. Above 1 is good."
                                    />
                                    <CompareRow
                                        label="Max drawdown" plain="The worst peak-to-trough fall"
                                        current={comparison.current.max_drawdown} suggested={comparison.suggested.max_drawdown}
                                        format={(v) => pct(v, 2)} higherIsBetter
                                        hint="How far the portfolio fell from its high before recovering. Closer to zero is better."
                                    />
                                </tbody>
                            </table>
                        </div>

                        {comparison.correlation && (
                            <div style={{ background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 8, padding: "12px 14px", fontSize: 12, color: C.textMid, lineHeight: 1.6 }}>
                                <strong style={{ color: C.text }}>Why the weights move.</strong> Your holdings
                                have an average pairwise correlation of{" "}
                                <span style={{ color: C.cyan, fontFamily: "'DM Mono',monospace" }}>
                                    {Number(comparison.correlation.avg_correlation).toFixed(2)}
                                </span>
                                . Two stocks that move together are close to one position under two names, so
                                the optimizer leans toward the ones that move differently.
                                {comparison.correlation.high_corr_pairs?.length > 0 && (
                                    <> Closely-paired here:{" "}
                                        {comparison.correlation.high_corr_pairs
                                            .map((pair) => `${pair.ticker_a}/${pair.ticker_b} ${Number(pair.correlation).toFixed(2)}`)
                                            .join(", ")}.
                                    </>
                                )}
                            </div>
                        )}

                        {comparison.alerts?.alerts?.length > 0 && (
                            <div style={{ display: "grid", gap: 6 }}>
                                <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1.4, textTransform: "uppercase" }}>
                                    Risk checks on your current split
                                </div>
                                {comparison.alerts.alerts.map((alert, index) => (
                                    <div
                                        key={`${alert.alert_type}-${alert.ticker ?? index}`}
                                        style={{
                                            display: "flex", gap: 10, alignItems: "baseline",
                                            background: C.bg2, border: `1px solid ${C.border}`,
                                            borderLeft: `3px solid ${alert.severity === "CRITICAL" ? C.red : C.amber}`,
                                            borderRadius: 6, padding: "9px 12px", fontSize: 12, color: C.textMid,
                                        }}
                                    >
                                        <span style={{
                                            color: alert.severity === "CRITICAL" ? C.red : C.amber,
                                            fontSize: 9.5, letterSpacing: 1, fontFamily: "'DM Mono',monospace",
                                            textTransform: "uppercase", whiteSpace: "nowrap",
                                        }}>
                                            {alert.alert_type.replace(/_/g, " ")}
                                        </span>
                                        <span>{alert.message}</span>
                                    </div>
                                ))}
                            </div>
                        )}

                        <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.6 }}>
                            Both columns are measured on the same past year, so the optimized split is the
                            one that <em>would have</em> worked best over that window. Nothing here promises
                            it repeats — that is what the Backtest tab is for.
                        </div>
                    </div>
                )}
            </Section>

            {/* ── Allocation ──────────────────────────────────── */}
            {currentWeights && (
                <Section title="Allocation" hint="Your current split by market value.">
                    <ResponsiveContainer width="100%" height={220}>
                        <PieChart>
                            <Pie
                                data={priced.filter((row) => row.hasPrice).map((row) => ({ name: row.ticker, value: row.value }))}
                                cx="50%" cy="50%" innerRadius={55} outerRadius={85} dataKey="value"
                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                labelLine={false}
                            >
                                {priced.filter((row) => row.hasPrice).map((row, index) => (
                                    <Cell key={row.ticker} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip formatter={(value) => money(Number(value))} />
                        </PieChart>
                    </ResponsiveContainer>
                </Section>
            )}
        </div>
    );
}
