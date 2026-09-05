import { useState, useCallback, useEffect, useMemo } from "react";
import {
    PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
    LineChart, Line, XAxis, YAxis, CartesianGrid, ReferenceDot,
} from "recharts";
import { C } from "../utils/data";
import {
    optimizePortfolio, fetchFrontier, fetchCorrelation,
    fetchPortfolioMetrics, fetchSimpleForecast, fetchDirectionAnalysis,
} from "../utils/api";
import { eachLimited } from "../utils/concurrency";
import { StatCard, Section, Hint, Badge } from "../components/UIComponents";

/* ══════════════════════════════════════════════════════════════════════════
   This tab is the last stage of the project's pipeline:

     Prediction → Forecast price & direction → Technical analysis
                → Portfolio optimization → Recommended portfolio

   It is laid out as those stages, in order, so the reader can follow one
   thread from "what do the models think of this stock" to "so it gets 18% of
   the money". The old layout showed the optimizer's output only — four
   statistics, a weight list, a scatter plot and a raw metrics dump — with no
   trace of the prediction or technical-analysis work anywhere on the page.

   HONESTY NOTE, and it governs the whole design: the optimizer solves on 12
   months of historical covariance. It does NOT read the forecast. So the
   forecast and the evidence stack are presented as a forward-looking
   CROSS-CHECK on a backward-looking allocation — never as its cause. Where the
   two disagree, the page says so and offers to rebuild without the offenders,
   which is the one place a prediction genuinely changes the portfolio.
   ══════════════════════════════════════════════════════════════════════════ */

/* Identity palette for allocation slices. Validated for this dark surface
   (#0d1524) on all six checks — lightness band, chroma floor, CVD separation,
   normal-vision separation and contrast. Assigned in fixed order and never
   cycled: the watchlist caps at 8, which is exactly the number of slots.

   Kept separate from C.green / C.red on purpose. Those two carry status
   meaning everywhere else on the page (return vs risk, up vs down), and a
   stock tinted status-green would read as "this one is good". */
const SLICE = [
    "#d97706", "#0891b2", "#8b5cf6", "#db2777",
    "#2563eb", "#059669", "#ea580c", "#0d9488",
];

/* The four objectives the backend accepts (OptimizationMethod in schemas.py),
   led by the question a person actually has rather than the term of art. The
   old dropdown offered "Max Sharpe / Min Volatility / Max Return / Risk Parity"
   and left the reader to hover a `?` to find out what they were choosing. */
const GOALS = [
    {
        key: "max_sharpe",
        title: "Balanced",
        plain: "Best return for the risk taken",
        detail: "Solved as: the highest expected return while keeping annual volatility under 20%. The usual starting point.",
    },
    {
        key: "min_volatility",
        title: "Defensive",
        plain: "Smoothest ride, lower returns",
        detail: "Minimises expected price swings. Picks the calmest mix even when that costs return.",
    },
    {
        key: "max_return",
        title: "Aggressive",
        plain: "Chase return, ignore risk",
        detail: "Maximises expected return with no risk term at all. Usually concentrates into one or two names.",
    },
    {
        key: "risk_parity",
        title: "Equal risk",
        plain: "No single stock dominates",
        detail: "Sizes positions so every holding contributes the same amount of risk. A volatile stock gets a smaller slice.",
    },
];
const GOAL_BY_KEY = Object.fromEntries(GOALS.map((g) => [g.key, g]));

/* Fixed order so the seven dots mean the same thing in every card and the eye
   can compare one stock's strip against another's. */
const EVIDENCE_ORDER = [
    "trend", "momentum", "volume", "price_action",
    "support_resistance", "volatility", "historical_analogs",
];
const EVIDENCE_SHORT = {
    trend: "Trend",
    momentum: "Momentum",
    volume: "Volume",
    price_action: "Price action",
    support_resistance: "Support / resistance",
    volatility: "Volatility",
    historical_analogs: "Analogs",
};

/* Position limits the backend applies when the request carries no constraints
   (see the /optimize route). Shown to the reader because they explain most of
   the extreme weights on the page — a 40% slice is usually the cap binding,
   not the optimizer's enthusiasm. */
const MAX_POSITION = 0.4;
const MIN_POSITION = 0.02;

/* ── formatting ─────────────────────────────────────────────────────────── */
const pct = (v, digits = 1) =>
    Number.isFinite(Number(v)) ? `${(Number(v) * 100).toFixed(digits)}%` : "—";
const signedPct = (v, digits = 2) =>
    Number.isFinite(Number(v)) ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(digits)}%` : "—";
const money = (v) =>
    Number.isFinite(Number(v)) ? `$${Number(v).toFixed(2)}` : "—";
const num = (v, digits = 2) =>
    Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : "—";

const DIRECTION_COLOR = { UP: C.green, DOWN: C.red, NEUTRAL: C.textDim };

/* ══════════════════════════════════════════════════════════════════════════
   The rationale: why this stock got this weight.

   Assembled from four measured sources and nothing else — no invented
   scoring. Each clause names where it came from so a reader can check it.
   ══════════════════════════════════════════════════════════════════════════ */
function buildRationale({ weight, attribution, forecast, analysis, sharpeRank, total, isEqualFallback }) {
    const clauses = [];

    if (isEqualFallback) {
        clauses.push({
            kind: "neutral",
            text: "The solver could not hit its volatility target, so every stock is held equally. This split is a fallback, not an optimum.",
        });
        return clauses;
    }

    // 1. Is a constraint binding? This explains more extreme weights than anything else.
    if (weight >= MAX_POSITION - 1e-4) {
        clauses.push({
            kind: "cap",
            text: `Capped at ${pct(MAX_POSITION, 0)} — the optimizer wanted more here but the position limit stopped it.`,
        });
    } else if (weight <= MIN_POSITION + 1e-4) {
        clauses.push({
            kind: "floor",
            text: `Held at the ${pct(MIN_POSITION, 0)} floor — the optimizer would drop it entirely if allowed to.`,
        });
    }

    // 2. Historical risk/return — the thing the optimizer actually solved on.
    if (attribution) {
        const sh = Number(attribution.stock_sharpe);
        const ret = Number(attribution.stock_annual_return);
        const vol = Number(attribution.stock_annual_volatility);
        if (Number.isFinite(sh) && Number.isFinite(ret) && Number.isFinite(vol)) {
            const rank =
                sharpeRank === 1 ? "the best risk-adjusted return in this set"
                    : sharpeRank === total ? "the weakest risk-adjusted return in this set"
                        : `${sharpeRank} of ${total} on risk-adjusted return`;
            clauses.push({
                kind: "history",
                text: `Past 12 months: ${pct(ret)} return at ${pct(vol)} volatility — ${rank} (Sharpe ${num(sh)}).`,
            });
        }
    }

    // 3. The forward-looking cross-check. Explicitly not an input to the weight.
    if (forecast?.status === "ok") {
        const dir = String(forecast.direction || "").toUpperCase();
        clauses.push({
            kind: dir === "DOWN" ? "conflict" : "forecast",
            text: `Models forecast ${money(forecast.forecast_price)} next session (${signedPct(forecast.expected_change_pct)}, ${dir || "—"}).`,
        });
    }

    // 4. The single strongest piece of technical evidence, named.
    const top = (analysis?.evidence || [])
        .filter((r) => Number.isFinite(Number(r.contribution_pp)))
        .sort((a, b) => Math.abs(Number(b.contribution_pp)) - Math.abs(Number(a.contribution_pp)))[0];
    if (top) {
        clauses.push({
            kind: "evidence",
            text: `Strongest technical signal: ${EVIDENCE_SHORT[top.source] || top.label} — ${top.state}, leaning ${top.leans}.`,
        });
    }

    return clauses;
}

/* ── Step header ────────────────────────────────────────────────────────── */
function StepHeader({ n, title, subtitle, right }) {
    return (
        <div style={{
            display: "flex", alignItems: "center", gap: 12,
            marginBottom: 12, marginTop: 28,
        }}>
            <div style={{
                width: 26, height: 26, borderRadius: "50%", flexShrink: 0,
                background: C.amberDim, border: `1px solid ${C.amber}66`,
                color: C.amber, fontSize: 12, fontWeight: 700,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontFamily: "'Syne',sans-serif",
            }}>{n}</div>
            <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                    fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 15,
                    color: C.text, lineHeight: 1.2,
                }}>{title}</div>
                {subtitle && (
                    <div style={{ fontSize: 11, color: C.textDim, marginTop: 2 }}>{subtitle}</div>
                )}
            </div>
            {right}
        </div>
    );
}

/* ── Pipeline breadcrumb ────────────────────────────────────────────────── */
function Pipeline({ stage }) {
    const steps = ["Prediction", "Forecast & direction", "Technical analysis", "Optimization", "Portfolio"];
    return (
        <div style={{
            display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap",
            fontSize: 10, color: C.textDim, fontFamily: "'DM Mono',monospace",
        }}>
            {steps.map((s, i) => (
                <span key={s} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{
                        color: i <= stage ? C.amber : C.textDim,
                        borderBottom: i <= stage ? `1px solid ${C.amber}55` : "1px solid transparent",
                        paddingBottom: 1,
                    }}>{s}</span>
                    {i < steps.length - 1 && <span style={{ opacity: .5 }}>→</span>}
                </span>
            ))}
        </div>
    );
}

/* ── Evidence strip: seven categories, one dot each ─────────────────────── */
function EvidenceStrip({ analysis }) {
    const bySource = Object.fromEntries((analysis?.evidence || []).map((r) => [r.source, r]));
    return (
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            {EVIDENCE_ORDER.map((src) => {
                const row = bySource[src];
                const leans = row?.leans;
                const color = leans === "up" ? C.green : leans === "down" ? C.red : C.textDim;
                const strength = Math.min(1, Math.abs(Number(row?.contribution_pp) || 0) / 1.5);
                return (
                    <span
                        key={src}
                        title={row
                            ? `${EVIDENCE_SHORT[src]}: ${row.state} — leaning ${row.leans} (${signedPct(row.contribution_pp)}pp)`
                            : `${EVIDENCE_SHORT[src]}: no reading`}
                        style={{
                            width: 9, height: 9, borderRadius: "50%",
                            background: row ? color : "transparent",
                            border: `1px solid ${row ? color : C.border}`,
                            opacity: row ? 0.35 + strength * 0.65 : 1,
                            cursor: "help", flexShrink: 0,
                        }}
                    />
                );
            })}
        </div>
    );
}

/* ── One stock's forward-looking signal ─────────────────────────────────── */
function SignalCard({ symbol, signal, color, onOpen }) {
    const { forecast, analysis, loading, error } = signal || {};
    const ok = forecast?.status === "ok";
    const dir = String((analysis?.direction) || (forecast?.direction) || "").toUpperCase();
    const dirColor = DIRECTION_COLOR[dir] || C.textDim;

    return (
        <div style={{
            background: C.bg2, border: `1px solid ${C.border}`,
            borderLeft: `3px solid ${color}`, borderRadius: 10, padding: "12px 14px",
        }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 10 }}>
                <button
                    onClick={() => onOpen?.(symbol)}
                    title={`Open ${symbol} in the Analysis tab`}
                    style={{
                        background: "none", border: "none", padding: 0, cursor: "pointer",
                        color: C.text, fontSize: 14, fontWeight: 700,
                        fontFamily: "'Syne',sans-serif", letterSpacing: .5,
                    }}
                >{symbol}</button>
                {loading
                    ? <span style={{ fontSize: 10, color: C.textDim }}>analysing…</span>
                    : dir
                        ? <Badge color={dirColor}>{dir === "UP" ? "▲" : dir === "DOWN" ? "▼" : "●"} {dir}</Badge>
                        : null}
            </div>

            {loading && (
                <div style={{ height: 52, display: "flex", alignItems: "center", color: C.textDim, fontSize: 11 }}>
                    Running forecast and technical analysis…
                </div>
            )}

            {!loading && error && (
                <div style={{ fontSize: 11, color: C.textDim, lineHeight: 1.5 }}>
                    No model reading available. {error}
                </div>
            )}

            {!loading && !error && (
                <>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 8 }}>
                        <span style={{ fontSize: 12, color: C.textMid, fontFamily: "'DM Mono',monospace" }}>
                            {money(forecast?.current_price ?? forecast?.anchor_price)}
                        </span>
                        <span style={{ color: C.textDim, fontSize: 11 }}>→</span>
                        <span style={{
                            fontSize: 16, fontWeight: 700, color: ok ? C.text : C.textDim,
                            fontFamily: "'DM Mono',monospace",
                        }}>{ok ? money(forecast.forecast_price) : "—"}</span>
                        {ok && (
                            <span style={{
                                fontSize: 12, fontWeight: 700, fontFamily: "'DM Mono',monospace",
                                color: Number(forecast.expected_change_pct) >= 0 ? C.green : C.red,
                            }}>{signedPct(forecast.expected_change_pct)}</span>
                        )}
                    </div>

                    {!ok && forecast?.message && (
                        <div style={{ fontSize: 10.5, color: C.textDim, marginBottom: 8, lineHeight: 1.4 }}>
                            {forecast.message}
                        </div>
                    )}

                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                        <EvidenceStrip analysis={analysis} />
                        {analysis?.confidence?.label && (
                            <span style={{ fontSize: 10, color: C.textDim }}>
                                {analysis.confidence.label} confidence
                            </span>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}

/* ── One allocation row, expandable into its reasoning ──────────────────── */
function AllocationRow({ symbol, weight, color, rationale, open, onToggle }) {
    const percent = weight * 100;
    const conflict = rationale.some((c) => c.kind === "conflict");
    return (
        <div style={{ borderBottom: `1px solid ${C.border}44` }}>
            <button
                onClick={onToggle}
                style={{
                    width: "100%", background: "none", border: "none", cursor: "pointer",
                    padding: "10px 2px", textAlign: "left", display: "block",
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                    <span style={{
                        width: 9, height: 9, borderRadius: 2, background: color, flexShrink: 0,
                    }} />
                    <span style={{
                        color: C.text, fontWeight: 700, fontSize: 12.5,
                        fontFamily: "'Syne',sans-serif", flex: 1,
                    }}>{symbol}</span>
                    {conflict && (
                        <span title="The models forecast this stock down" style={{ fontSize: 11 }}>⚠️</span>
                    )}
                    <span style={{
                        color: C.text, fontSize: 13, fontWeight: 700,
                        fontFamily: "'DM Mono',monospace",
                    }}>{percent.toFixed(1)}%</span>
                    <span style={{ color: C.textDim, fontSize: 10, width: 10 }}>{open ? "▾" : "▸"}</span>
                </div>
                <div style={{ background: C.bg2, borderRadius: 4, height: 6, overflow: "hidden" }}>
                    <div style={{
                        width: `${Math.max(percent, 0.6)}%`, height: "100%", background: color,
                        borderRadius: 4, transition: "width .5s ease",
                    }} />
                </div>
            </button>

            {open && (
                <div style={{ padding: "2px 2px 12px 19px", display: "grid", gap: 6 }}>
                    {rationale.length === 0 && (
                        <div style={{ fontSize: 11, color: C.textDim }}>
                            No reasoning available — the supporting data did not load.
                        </div>
                    )}
                    {rationale.map((c, i) => (
                        <div key={i} style={{
                            fontSize: 11.5, lineHeight: 1.55,
                            color: c.kind === "conflict" ? C.red
                                : c.kind === "cap" || c.kind === "floor" ? C.amber
                                    : C.textMid,
                        }}>
                            <span style={{ opacity: .6, marginRight: 6 }}>
                                {c.kind === "history" ? "📊"
                                    : c.kind === "forecast" ? "🤖"
                                        : c.kind === "conflict" ? "⚠️"
                                            : c.kind === "evidence" ? "📐"
                                                : "•"}
                            </span>
                            {c.text}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

/* ── Diversification, without an N×N grid to decode ─────────────────────── */
function Diversification({ correlation }) {
    if (!correlation?.tickers?.length) {
        return <div style={{ color: C.textDim, fontSize: 11.5 }}>Not available.</div>;
    }
    const avg = Number(correlation.avg_correlation);
    // −1..1 mapped onto the meter; in practice equities sit in 0..1.
    const position = Math.max(0, Math.min(1, (avg + 1) / 2)) * 100;
    const verdict = avg >= 0.7 ? { text: "Poorly diversified", color: C.red }
        : avg >= 0.4 ? { text: "Moderately diversified", color: C.amber }
            : { text: "Well diversified", color: C.green };

    return (
        <div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 10 }}>
                <span style={{
                    fontSize: 22, fontWeight: 700, color: verdict.color,
                    fontFamily: "'DM Mono',monospace", lineHeight: 1,
                }}>{num(avg)}</span>
                <span style={{ fontSize: 12, color: verdict.color, fontWeight: 700 }}>{verdict.text}</span>
            </div>

            <div style={{ position: "relative", height: 6, background: C.bg2, borderRadius: 4, marginBottom: 6 }}>
                <div style={{
                    position: "absolute", inset: 0, borderRadius: 4,
                    background: `linear-gradient(90deg, ${C.green}55, ${C.amber}55, ${C.red}55)`,
                }} />
                <div style={{
                    position: "absolute", left: `${position}%`, top: -3, width: 2, height: 12,
                    background: C.text, transform: "translateX(-1px)", borderRadius: 1,
                }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9.5, color: C.textDim, marginBottom: 12 }}>
                <span>−1.0 move opposite</span><span>0 independent</span><span>+1.0 lockstep</span>
            </div>

            <div style={{ fontSize: 11.5, color: C.textMid, lineHeight: 1.6 }}>
                Average correlation of daily returns across the last 90 days. Holding names
                that move together buys less protection than the position count suggests.
            </div>

            {correlation.high_corr_pairs?.length > 0 && (
                <div style={{
                    marginTop: 12, padding: "10px 12px", borderRadius: 8,
                    background: `${C.red}14`, border: `1px solid ${C.red}33`,
                }}>
                    <div style={{ fontSize: 10, color: C.red, letterSpacing: 1, marginBottom: 6, fontFamily: "'Syne',sans-serif", fontWeight: 700 }}>
                        MOVING NEARLY AS ONE
                    </div>
                    {correlation.high_corr_pairs.slice(0, 4).map((p, i) => (
                        <div key={i} style={{ fontSize: 11.5, color: C.textMid, fontFamily: "'DM Mono',monospace" }}>
                            {p.ticker_a} / {p.ticker_b} — {num(p.correlation)}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

/* ── Collapsible for the details most readers never need ────────────────── */
function Advanced({ title, children }) {
    const [open, setOpen] = useState(false);
    return (
        <div style={{
            border: `1px solid ${C.border}`, borderRadius: 10,
            background: C.bg1, marginTop: 16, overflow: "hidden",
        }}>
            <button
                onClick={() => setOpen((v) => !v)}
                style={{
                    width: "100%", background: "none", border: "none", cursor: "pointer",
                    padding: "12px 16px", display: "flex", alignItems: "center",
                    justifyContent: "space-between", color: C.textMid,
                    fontFamily: "'Syne',sans-serif", fontSize: 10.5,
                    letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700,
                }}
            >
                <span>{title}</span>
                <span style={{ color: C.textDim }}>{open ? "▾ hide" : "▸ show"}</span>
            </button>
            {open && <div style={{ padding: "0 16px 16px" }}>{children}</div>}
        </div>
    );
}

/* ══════════════════════════════════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════════════════════════════════ */
export default function OptimizationTab({
    apiConnected, notify, watchlist = [], setSelectedTicker,
    request, onRequestConsumed,
}) {
    const [selected, setSelected] = useState(() => watchlist.slice(0, 8));
    const [extra, setExtra] = useState("");
    const [goal, setGoal] = useState("max_sharpe");

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);        // /optimize
    const [attribution, setAttribution] = useState(null); // /metrics?include_attribution
    const [correlation, setCorrelation] = useState(null);
    const [frontier, setFrontier] = useState(null);
    const [signals, setSignals] = useState({});        // symbol → {forecast, analysis, loading, error}
    const [builtWith, setBuiltWith] = useState([]);    // the symbol set behind `result`
    const [openRow, setOpenRow] = useState(null);
    // Symbols that arrived from the Heatmap. Kept so a stock that was never in
    // the watchlist still has a chip here to be de-selected from.
    const [handedOver, setHandedOver] = useState([]);

    const universe = useMemo(() => {
        const seen = new Set();
        return [...watchlist, ...selected, ...handedOver].filter((t) => {
            const k = String(t || "").toUpperCase();
            if (!k || seen.has(k)) return false;
            seen.add(k);
            return true;
        });
    }, [watchlist, selected, handedOver]);

    const toggle = (t) => {
        setSelected((cur) => cur.includes(t)
            ? cur.filter((s) => s !== t)
            : cur.length >= 8 ? cur : [...cur, t]);
    };

    const addExtra = () => {
        const t = extra.trim().toUpperCase();
        if (!t) return;
        if (selected.includes(t)) { setExtra(""); return; }
        if (selected.length >= 8) { notify?.("Eight stocks is the maximum."); return; }
        setSelected((cur) => [...cur, t]);
        setExtra("");
    };

    /** Fire the fast portfolio maths, then stream the per-stock model work in
     *  behind it. The optimizer answers in about a second; a forecast is
     *  several seconds per symbol, so waiting for all of them before drawing
     *  anything would leave the page blank for a minute on eight stocks. */
    const build = useCallback(async (symbolsIn, { silent = false } = {}) => {
        const symbols = symbolsIn.map((s) => String(s).trim().toUpperCase()).filter(Boolean);
        if (symbols.length < 2) {
            setError("Pick at least two stocks — a single holding has nothing to optimize.");
            return;
        }
        setLoading(true);
        setError(null);
        setOpenRow(null);

        let optimized;
        try {
            optimized = await optimizePortfolio({ symbols, method: goal });
            setResult(optimized);
            setBuiltWith(symbols);
        } catch (e) {
            setError(e.message || "Optimization failed.");
            setLoading(false);
            return;
        }

        // Everything below is supporting detail: a failure degrades the page
        // rather than replacing the portfolio the user asked for with an error.
        const weights = optimized?.weights || {};
        const [attr, corr, front] = await Promise.all([
            fetchPortfolioMetrics(symbols, { weights, includeAttribution: true }).catch(() => null),
            fetchCorrelation(symbols, 90).catch(() => null),
            fetchFrontier({ symbols, method: goal }).catch(() => null),
        ]);
        setAttribution(attr?.attribution || null);
        setCorrelation(corr);
        setFrontier(front);
        setLoading(false);
        if (!silent) notify?.("Portfolio built — loading model signals");

        setSignals(Object.fromEntries(symbols.map((s) => [s, { loading: true }])));
        await eachLimited(symbols, 3, async (sym) => {
            const [forecast, analysis] = await Promise.all([
                fetchSimpleForecast(sym).catch((e) => ({ status: "error", message: e.message })),
                fetchDirectionAnalysis(sym).catch(() => null),
            ]);
            setSignals((cur) => ({
                ...cur,
                [sym]: {
                    forecast,
                    analysis,
                    loading: false,
                    error: forecast?.status === "error" ? forecast.message : null,
                },
            }));
        });
    }, [goal, notify]);

    // A shortlist handed over from the Heatmap builds on arrival. The button
    // there said "Optimize these N", so asking the user to press Build here as
    // well would be asking twice for the same thing — the same reasoning that
    // makes a prediction run itself on the Backtest tab.
    //
    // The goal is deliberately left at whatever is selected (Balanced by
    // default). It is a separate decision from *which* stocks, the four goal
    // cards are directly above the result, and rebuilding is one click.
    //
    // The request is consumed rather than remembered: this tab unmounts when the
    // user leaves it, so a request left standing in App would re-run the
    // optimizer on every return visit.
    useEffect(() => {
        if (!request || !apiConnected) return;

        // One stock arriving from the Predictions tab. It is added to the
        // selection and nothing is built: a portfolio needs at least two
        // holdings, so auto-running here would only produce an error the user
        // did not ask for. The optimizer's own Build button stays the trigger.
        if (request.add) {
            const symbol = String(request.add).toUpperCase();
            // eslint-disable-next-line react-hooks/set-state-in-effect
            setHandedOver((cur) => (cur.includes(symbol) ? cur : [...cur, symbol]));
            setSelected((cur) => {
                if (cur.includes(symbol)) return cur;
                if (cur.length >= 8) {
                    notify?.(`Selection is full (8). Remove one to add ${symbol}.`);
                    return cur;
                }
                return [...cur, symbol];
            });
            notify?.(`${symbol} added to the optimizer — choose your goal and build`);
            onRequestConsumed?.();
            return;
        }

        const symbols = request.symbols || [];
        if (symbols.length < 2) return;
        // set-state-in-effect is expected here for the same reason as above:
        // `build` flips the panel to its running state before awaiting, which
        // is the handover starting rather than a render cascade.
        setHandedOver(symbols);
        setSelected(symbols.slice(0, 8));
        build(symbols.slice(0, 8), { silent: true });
        notify?.(`Building a portfolio from your ${symbols.length}-stock shortlist`);
        onRequestConsumed?.();
        // Keyed on the handover alone. `build` closes over the current goal in
        // the render where `request.at` changed, and listing it here would
        // re-run the optimizer every time the goal buttons re-created it.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [request?.at, apiConnected]);

    if (!apiConnected) {
        return (
            <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>
                🔌 Connect to the API to build a portfolio
            </div>
        );
    }

    /* ── Derived view data ──────────────────────────────────────────────── */
    const weights = result?.weights || {};
    const holdings = Object.entries(weights)
        .map(([symbol, w]) => ({ symbol, weight: Number(w) || 0 }))
        .sort((a, b) => b.weight - a.weight);

    const colorOf = Object.fromEntries(builtWith.map((s, i) => [s, SLICE[i % SLICE.length]]));

    // An exactly-equal split is the optimizer's documented fallback for an
    // infeasible or non-converging problem, not a solution. Saying so beats
    // presenting it as one.
    //
    // The tolerance is 1e-3, not machine epsilon: the backend rounds weights to
    // four decimals, so a genuine 1/3 arrives as 0.3333 and misses exact
    // equality by 3.3e-5. A stricter test silently never fired — which is how a
    // fallback split reached the screen labelled as an optimum.
    const isEqualFallback = holdings.length > 2 &&
        holdings.every((h) => Math.abs(h.weight - 1 / holdings.length) < 1e-3);

    const byStock = attribution?.by_stock || {};
    const sharpeOrder = [...holdings]
        .sort((a, b) => (Number(byStock[b.symbol]?.stock_sharpe) || -Infinity) -
            (Number(byStock[a.symbol]?.stock_sharpe) || -Infinity))
        .map((h) => h.symbol);

    const rationaleOf = (symbol, weight) => buildRationale({
        weight,
        attribution: byStock[symbol],
        forecast: signals[symbol]?.forecast,
        analysis: signals[symbol]?.analysis,
        sharpeRank: sharpeOrder.indexOf(symbol) + 1,
        total: holdings.length,
        isEqualFallback,
    });

    // Where the forward-looking work disagrees with the backward-looking split.
    const bearish = holdings.filter(({ symbol }) => {
        const f = signals[symbol]?.forecast;
        const a = signals[symbol]?.analysis;
        const dir = String(a?.direction || f?.direction || "").toUpperCase();
        return dir === "DOWN";
    });
    const bearishWeight = bearish.reduce((s, h) => s + h.weight, 0);
    const signalsDone = builtWith.length > 0 &&
        builtWith.every((s) => signals[s] && !signals[s].loading);

    const frontierPoints = (frontier?.points || [])
        .filter((p) => Number.isFinite(p.volatility) && Number.isFinite(p.return))
        .sort((a, b) => a.volatility - b.volatility);

    const donutData = holdings.map((h) => ({ name: h.symbol, value: +(h.weight * 100).toFixed(2) }));

    const inputStyle = {
        background: C.bg2, color: C.text, border: `1px solid ${C.border}`,
        borderRadius: 6, padding: "8px 12px", fontSize: 12,
        fontFamily: "'DM Mono',monospace",
    };

    const stage = result ? (signalsDone ? 4 : 3) : 0;

    return (
        <div className="fade-up">
            {/* ── Header ─────────────────────────────────────────────────── */}
            <h1 style={{
                fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28,
                color: C.text, marginBottom: 6,
            }}>⚡ Build a Portfolio</h1>
            <div style={{ fontSize: 12, color: C.textMid, marginBottom: 10, maxWidth: 760, lineHeight: 1.6 }}>
                Turns a shortlist of stocks into a recommended split of your money — and shows
                what the prediction models and technical analysis say about each one, so you can
                see why every holding is sized the way it is.
            </div>
            <Pipeline stage={stage} />

            {/* ══ STEP 1 ═══════════════════════════════════════════════════ */}
            <StepHeader
                n={1}
                title="Choose your stocks and your goal"
                subtitle="Up to eight. Tap a ticker to include or exclude it."
            />
            <Section>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
                    {universe.map((t) => {
                        const on = selected.includes(t);
                        return (
                            <button
                                key={t}
                                onClick={() => toggle(t)}
                                style={{
                                    background: on ? C.amberDim : C.bg2,
                                    border: `1px solid ${on ? C.amber : C.border}`,
                                    color: on ? C.amber : C.textDim,
                                    borderRadius: 20, padding: "6px 14px", cursor: "pointer",
                                    fontFamily: "'DM Mono',monospace", fontSize: 12,
                                    fontWeight: on ? 700 : 400, transition: "all .15s",
                                }}
                            >{on ? "✓ " : ""}{t}</button>
                        );
                    })}
                    {universe.length === 0 && (
                        <span style={{ color: C.textDim, fontSize: 12 }}>
                            Your watchlist is empty — add a ticker below.
                        </span>
                    )}
                </div>

                <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 18 }}>
                    <input
                        value={extra}
                        onChange={(e) => setExtra(e.target.value.toUpperCase())}
                        onKeyDown={(e) => { if (e.key === "Enter") addExtra(); }}
                        placeholder="Add another ticker…"
                        style={{ ...inputStyle, width: 190 }}
                    />
                    <button
                        onClick={addExtra}
                        style={{
                            ...inputStyle, cursor: "pointer", color: C.textMid,
                            background: C.bg3, border: `1px solid ${C.border}`,
                        }}
                    >+ Add</button>
                    <span style={{ fontSize: 11, color: C.textDim, marginLeft: "auto" }}>
                        {selected.length}/8 selected
                    </span>
                </div>

                {/* Goal, in plain language */}
                <div style={{
                    fontSize: 10, color: C.textDim, letterSpacing: 1.5,
                    textTransform: "uppercase", fontFamily: "'Syne',sans-serif",
                    fontWeight: 700, marginBottom: 10,
                }}>
                    What should the optimizer aim for?
                </div>
                <div style={{
                    display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))",
                    gap: 10, marginBottom: 18,
                }}>
                    {GOALS.map((g) => {
                        const on = goal === g.key;
                        return (
                            <button
                                key={g.key}
                                onClick={() => setGoal(g.key)}
                                title={g.detail}
                                style={{
                                    textAlign: "left", cursor: "pointer", borderRadius: 8,
                                    padding: "12px 14px",
                                    background: on ? C.amberLow : C.bg2,
                                    border: `1px solid ${on ? C.amber : C.border}`,
                                    transition: "all .15s",
                                }}
                            >
                                <div style={{
                                    fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 13,
                                    color: on ? C.amber : C.text, marginBottom: 3,
                                }}>{g.title}</div>
                                <div style={{ fontSize: 11, color: C.textMid, lineHeight: 1.45 }}>{g.plain}</div>
                            </button>
                        );
                    })}
                </div>

                <button
                    onClick={() => build(selected)}
                    disabled={loading || selected.length < 2}
                    style={{
                        background: loading || selected.length < 2
                            ? C.bg2 : `linear-gradient(135deg, ${C.amber}, #f97316)`,
                        color: loading || selected.length < 2 ? C.textDim : "#000",
                        border: "none", borderRadius: 8, padding: "12px 28px",
                        fontSize: 13, fontWeight: 700, fontFamily: "'Syne',sans-serif",
                        cursor: loading || selected.length < 2 ? "not-allowed" : "pointer",
                    }}
                >
                    {loading ? "⏳ Building…" : "⚡ Build my portfolio"}
                </button>
                {selected.length < 2 && (
                    <span style={{ fontSize: 11, color: C.textDim, marginLeft: 12 }}>
                        Pick at least two stocks.
                    </span>
                )}
            </Section>

            {error && (
                <div style={{
                    background: `${C.red}22`, color: C.red, padding: 12,
                    borderRadius: 8, marginTop: 16, fontSize: 12,
                }}>{error}</div>
            )}

            {result && (
                <>
                    {/* ══ STEP 2 ═══════════════════════════════════════════ */}
                    <StepHeader
                        n={2}
                        title="What the models say about each stock"
                        subtitle="Next-session forecast from the prediction models, plus the technical evidence behind the direction call."
                        right={
                            <span style={{ fontSize: 10, color: C.textDim, display: "flex", alignItems: "center" }}>
                                seven signals per stock
                                <Hint text="Each dot is one technical-analysis category: trend, momentum, volume, price action, support/resistance, volatility and historical analogs. Green leans up, red leans down; the stronger the effect on the direction call, the more solid the dot. Hover a dot for its reading." />
                            </span>
                        }
                    />
                    <div style={{
                        display: "grid", gap: 12,
                        gridTemplateColumns: "repeat(auto-fill,minmax(255px,1fr))",
                    }}>
                        {holdings.map(({ symbol }) => (
                            <SignalCard
                                key={symbol}
                                symbol={symbol}
                                signal={signals[symbol]}
                                color={colorOf[symbol] || C.border}
                                onOpen={setSelectedTicker}
                            />
                        ))}
                    </div>

                    {/* ══ STEP 3 ═══════════════════════════════════════════ */}
                    <StepHeader
                        n={3}
                        title="Your recommended portfolio"
                        subtitle={`${GOAL_BY_KEY[result.method]?.title || GOAL_BY_KEY[goal].title} · ${holdings.length} holdings · sized on the last 12 months of daily returns.`}
                    />

                    {isEqualFallback && (
                        <div style={{
                            background: `${C.amber}18`, border: `1px solid ${C.amber}44`,
                            color: C.amber, padding: "12px 14px", borderRadius: 8,
                            fontSize: 11.5, marginBottom: 14, lineHeight: 1.6,
                        }}>
                            <strong>This is an equal-weight fallback, not an optimum.</strong>{" "}
                            {result.method === "max_sharpe"
                                ? `The Balanced goal requires annual volatility under 20%, and no mix of these
                                   stocks reaches it — this set runs at ${pct(result.volatility)}. With no
                                   feasible answer the solver split the money evenly.`
                                : `The solver could not converge on these stocks, so it split the money evenly.`}
                            {" "}Try the <em>Defensive</em> goal, or swap in a calmer stock.
                        </div>
                    )}

                    {/* Where prediction and allocation disagree — the one place the
                        forecast is offered a say in the portfolio. */}
                    {signalsDone && bearish.length > 0 && (
                        <div style={{
                            background: `${C.red}14`, border: `1px solid ${C.red}38`,
                            borderRadius: 8, padding: "12px 14px", marginBottom: 14,
                            display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap",
                        }}>
                            <div style={{ flex: 1, minWidth: 260, fontSize: 11.5, color: C.textMid, lineHeight: 1.6 }}>
                                <strong style={{ color: C.red }}>
                                    The models forecast {bearish.map((b) => b.symbol).join(", ")} down.
                                </strong>{" "}
                                That is {pct(bearishWeight)} of this portfolio. The optimizer never saw those
                                forecasts — it sizes positions on past returns only — so this is yours to act on.
                            </div>
                            <button
                                onClick={() => {
                                    const keep = holdings
                                        .filter((h) => !bearish.some((b) => b.symbol === h.symbol))
                                        .map((h) => h.symbol);
                                    if (keep.length < 2) { notify?.("Too few stocks would be left."); return; }
                                    setSelected(keep);
                                    build(keep);
                                }}
                                style={{
                                    background: "transparent", border: `1px solid ${C.red}66`,
                                    color: C.red, borderRadius: 6, padding: "8px 14px",
                                    fontSize: 11.5, cursor: "pointer", fontWeight: 700,
                                    fontFamily: "'Syne',sans-serif", whiteSpace: "nowrap",
                                }}
                            >Rebuild without them</button>
                        </div>
                    )}

                    {/* Headline numbers */}
                    <div style={{
                        display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))",
                        gap: 12, marginBottom: 16,
                    }}>
                        <StatCard
                            label="Expected return" value={pct(result.expected_return)}
                            sub="per year" color={C.green}
                            hint="What this exact mix would have earned per year over the last 12 months. A historical average — not a forecast of what it will earn." />
                        <StatCard
                            label="Risk" value={pct(result.volatility)}
                            sub="volatility per year" color={C.red}
                            hint="How much the portfolio's value swings in a typical year. Higher means a bumpier ride in both directions." />
                        <StatCard
                            label="Return per unit of risk" value={num(result.sharpe_ratio)}
                            sub="Sharpe ratio" color={C.cyan}
                            hint="Return earned above the risk-free rate for each unit of risk taken. Above 1 is good, above 2 is excellent." />
                        <StatCard
                            label="Diversification"
                            value={correlation ? num(correlation.avg_correlation) : "—"}
                            sub="average correlation" color={C.purple}
                            hint="How closely these stocks move together, 0 to 1. Lower is better diversified — holdings that move as one give you less protection than the position count suggests." />
                    </div>

                    {/* Allocation: donut + reasoned list */}
                    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,320px) minmax(0,1fr)", gap: 16 }}>
                        <Section title="The split" hint="How much of every $100 goes into each stock.">
                            <ResponsiveContainer width="100%" height={230}>
                                <PieChart>
                                    <Pie
                                        data={donutData}
                                        dataKey="value"
                                        nameKey="name"
                                        innerRadius={58}
                                        outerRadius={92}
                                        paddingAngle={2}
                                        stroke={C.bg1}
                                        strokeWidth={2}
                                        /* Direct labels on the slices big enough to hold one.
                                           They are also the secondary encoding that lets the
                                           one adjacent hue pair in the CVD floor band stay
                                           distinguishable without relying on colour. */
                                        label={({ name, value }) => (value >= 8 ? name : "")}
                                        labelLine={false}
                                        isAnimationActive={false}
                                    >
                                        {donutData.map((d) => (
                                            <Cell key={d.name} fill={colorOf[d.name] || C.border} />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        formatter={(v, n) => [`${Number(v).toFixed(1)}%`, n]}
                                        contentStyle={{
                                            background: C.bg3, border: `1px solid ${C.border}`,
                                            borderRadius: 6, fontSize: 11,
                                            fontFamily: "'DM Mono',monospace", color: C.text,
                                        }}
                                    />
                                </PieChart>
                            </ResponsiveContainer>
                            <div style={{ fontSize: 10.5, color: C.textDim, textAlign: "center", lineHeight: 1.5 }}>
                                Positions are capped at {pct(MAX_POSITION, 0)} and floored at {pct(MIN_POSITION, 0)}.
                            </div>
                        </Section>

                        <Section
                            title="Why each stock got its share"
                            hint="Tap any holding to see the reasoning: what its past risk and return were, what the models forecast next, and which technical signal mattered most."
                        >
                            {holdings.map(({ symbol, weight }) => (
                                <AllocationRow
                                    key={symbol}
                                    symbol={symbol}
                                    weight={weight}
                                    color={colorOf[symbol] || C.border}
                                    rationale={rationaleOf(symbol, weight)}
                                    open={openRow === symbol}
                                    onToggle={() => setOpenRow((cur) => (cur === symbol ? null : symbol))}
                                />
                            ))}
                            <div style={{
                                fontSize: 10.5, color: C.textDim, marginTop: 12,
                                lineHeight: 1.6, paddingTop: 10, borderTop: `1px solid ${C.border}44`,
                            }}>
                                The weights come from 12 months of price history. The forecast and technical
                                signals beside them are a forward-looking cross-check — they did not influence
                                the sizing.
                            </div>
                        </Section>
                    </div>

                    {/* Diversification */}
                    <div style={{ marginTop: 16 }}>
                        <Section
                            title="How well spread is the risk?"
                            hint="Two stocks that move in lockstep are close to one position wearing two names. This is what diversification actually buys you."
                        >
                            <Diversification correlation={correlation} />
                        </Section>
                    </div>

                    {/* ══ Advanced ═════════════════════════════════════════ */}
                    <Advanced title="Advanced — efficient frontier, correlation matrix, full metrics">
                        {/* Frontier. The old version scattered 50 unconnected dots and
                            marked the frontier's own best-Sharpe point with a triangle —
                            a different portfolio from the one in the weights list, never
                            reconciled. Here the frontier is a curve and the marker is
                            THIS portfolio, so the gap between them is visible: it sits
                            inside the curve because the position caps hold it there. */}
                        <div style={{ fontSize: 11.5, color: C.textMid, margin: "14px 0 10px", lineHeight: 1.6 }}>
                            Every point on the curve is the best return available at that level of risk,
                            using these stocks. The marker is your portfolio. It sits inside the curve when
                            the {pct(MAX_POSITION, 0)} position cap prevents the mathematically optimal split.
                        </div>
                        {frontierPoints.length > 0 ? (
                            <ResponsiveContainer width="100%" height={280}>
                                <LineChart data={frontierPoints} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                                    <XAxis
                                        dataKey="volatility" type="number"
                                        domain={["dataMin", "dataMax"]}
                                        tick={{ fill: C.textDim, fontSize: 10 }}
                                        tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                                        label={{ value: "Risk (annual volatility)", fill: C.textDim, fontSize: 10, position: "insideBottom", offset: -14 }}
                                    />
                                    <YAxis
                                        dataKey="return" type="number"
                                        domain={["auto", "auto"]}
                                        tick={{ fill: C.textDim, fontSize: 10 }}
                                        tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                                        label={{ value: "Expected return", fill: C.textDim, fontSize: 10, angle: -90, position: "insideLeft" }}
                                    />
                                    <Tooltip
                                        formatter={(v, n) => [`${(Number(v) * 100).toFixed(2)}%`, n === "return" ? "Return" : n]}
                                        labelFormatter={(v) => `Risk ${(Number(v) * 100).toFixed(1)}%`}
                                        contentStyle={{
                                            background: C.bg3, border: `1px solid ${C.border}`,
                                            borderRadius: 6, fontSize: 11,
                                            fontFamily: "'DM Mono',monospace", color: C.text,
                                        }}
                                    />
                                    <Line
                                        type="monotone" dataKey="return" name="return"
                                        stroke={C.cyan} strokeWidth={2} dot={false}
                                        isAnimationActive={false}
                                    />
                                    {Number.isFinite(result.volatility) && Number.isFinite(result.expected_return) && (
                                        <ReferenceDot
                                            x={result.volatility} y={result.expected_return}
                                            r={7} fill={C.amber} stroke={C.bg1} strokeWidth={2}
                                            label={{ value: "Your portfolio", fill: C.amber, fontSize: 10, position: "top" }}
                                        />
                                    )}
                                </LineChart>
                            </ResponsiveContainer>
                        ) : (
                            <div style={{ color: C.textDim, padding: 30, textAlign: "center", fontSize: 12 }}>
                                The frontier could not be computed for this set.
                            </div>
                        )}

                        {/* Full correlation matrix, for readers who want every pair. */}
                        {correlation?.tickers?.length > 0 && (
                            <>
                                <div style={{
                                    fontFamily: "'Syne',sans-serif", fontSize: 10, letterSpacing: 1.5,
                                    textTransform: "uppercase", color: C.textMid, fontWeight: 700,
                                    marginTop: 24, marginBottom: 10,
                                }}>Correlation matrix — every pair</div>
                                <div style={{ overflowX: "auto" }}>
                                    <table style={{ borderCollapse: "collapse", fontSize: 11, fontFamily: "'DM Mono',monospace" }}>
                                        <thead>
                                            <tr>
                                                <th style={{ padding: "6px 8px" }} />
                                                {correlation.tickers.map((t) => (
                                                    <th key={t} style={{ padding: "6px 8px", color: C.textDim, fontWeight: 400 }}>{t}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {correlation.tickers.map((row) => (
                                                <tr key={row}>
                                                    <td style={{ padding: "6px 8px", color: C.textMid, fontWeight: 700 }}>{row}</td>
                                                    {correlation.tickers.map((col) => {
                                                        const value = Number(correlation.matrix?.[row]?.[col]);
                                                        const self = row === col;
                                                        const tint = !Number.isFinite(value) || self ? "transparent"
                                                            : value >= 0.8 ? `${C.red}33`
                                                                : value >= 0.5 ? `${C.amber}26`
                                                                    : value <= 0 ? `${C.green}22` : "transparent";
                                                        return (
                                                            <td key={col} style={{
                                                                padding: "6px 8px", textAlign: "right",
                                                                background: tint, color: self ? C.textDim : C.text,
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
                            </>
                        )}

                        {/* Per-stock history, the numbers the optimizer actually solved on. */}
                        {holdings.length > 0 && Object.keys(byStock).length > 0 && (
                            <>
                                <div style={{
                                    fontFamily: "'Syne',sans-serif", fontSize: 10, letterSpacing: 1.5,
                                    textTransform: "uppercase", color: C.textMid, fontWeight: 700,
                                    marginTop: 24, marginBottom: 10,
                                }}>Per-stock history — what the optimizer solved on</div>
                                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                                    <thead>
                                        <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                            {["Stock", "Weight", "Annual return", "Volatility", "Sharpe", "Contribution"].map((h, i) => (
                                                <th key={h} style={{
                                                    padding: "8px 10px", color: C.textDim,
                                                    textAlign: i === 0 ? "left" : "right", fontWeight: 400,
                                                }}>{h}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {holdings.map(({ symbol, weight }) => {
                                            const a = byStock[symbol] || {};
                                            return (
                                                <tr key={symbol} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                                    <td style={{ padding: "7px 10px", color: C.text, fontWeight: 700 }}>{symbol}</td>
                                                    <td style={{ padding: "7px 10px", textAlign: "right", color: C.textMid, fontFamily: "'DM Mono',monospace" }}>{pct(weight)}</td>
                                                    <td style={{ padding: "7px 10px", textAlign: "right", fontFamily: "'DM Mono',monospace", color: Number(a.stock_annual_return) >= 0 ? C.green : C.red }}>{pct(a.stock_annual_return)}</td>
                                                    <td style={{ padding: "7px 10px", textAlign: "right", color: C.textMid, fontFamily: "'DM Mono',monospace" }}>{pct(a.stock_annual_volatility)}</td>
                                                    <td style={{ padding: "7px 10px", textAlign: "right", color: C.text, fontFamily: "'DM Mono',monospace" }}>{num(a.stock_sharpe)}</td>
                                                    <td style={{ padding: "7px 10px", textAlign: "right", color: C.textMid, fontFamily: "'DM Mono',monospace" }}>{pct(a.contribution_to_portfolio)}</td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </>
                        )}

                        {/* The raw metrics that used to be a top-level panel. Kept, but
                            behind a disclosure and with the four already on the page above
                            left out rather than printed twice. */}
                        {result.metrics && (
                            <>
                                <div style={{
                                    fontFamily: "'Syne',sans-serif", fontSize: 10, letterSpacing: 1.5,
                                    textTransform: "uppercase", color: C.textMid, fontWeight: 700,
                                    marginTop: 24, marginBottom: 10,
                                }}>Full risk statistics</div>
                                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                                    <tbody>
                                        {Object.entries(result.metrics)
                                            .filter(([k]) => !["annual_return", "annual_volatility", "sharpe_ratio"].includes(k))
                                            .map(([k, v]) => (
                                                <tr key={k} style={{ borderBottom: `1px solid ${C.border}22` }}>
                                                    <td style={{ padding: "6px 10px", color: C.textMid }}>
                                                        {k.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                                                    </td>
                                                    <td style={{
                                                        padding: "6px 10px", color: C.text, textAlign: "right",
                                                        fontFamily: "'DM Mono',monospace",
                                                    }}>
                                                        {v == null ? "—" : typeof v === "number" ? v.toFixed(4) : String(v)}
                                                    </td>
                                                </tr>
                                            ))}
                                    </tbody>
                                </table>
                            </>
                        )}
                    </Advanced>

                    <div style={{
                        fontSize: 10.5, color: C.textDim, marginTop: 20,
                        textAlign: "center", lineHeight: 1.6,
                    }}>
                        Historical performance and model forecasts are not guarantees. Not financial advice.
                    </div>
                </>
            )}
        </div>
    );
}
