/**
 * Signal Heatmap — stage three of the project's pipeline.
 *
 *     Stock data → technical analysis + prediction → THIS → shortlist → optimizer
 *
 * This tab used to be a generic S&P 500 sector treemap: ~80 companies hardcoded
 * with stale prices, market caps and 52-week ranges; tiles sized by that
 * hardcoded market cap; colour carrying one day of price change; and a "signal"
 * badge invented on the spot from |change| >= 2% plus above-average volume. None
 * of it touched a model, none of it touched the watchlist, and its only exit set
 * the global ticker without going anywhere. It was a stock-market heatmap that
 * happened to be installed in a machine-learning dashboard.
 *
 * What it is now: a screen over the symbols the user actually holds, coloured by
 * what this project's own models say about them, ending in a shortlist handed to
 * the optimizer.
 *
 * WHERE EVERY NUMBER COMES FROM — nothing is scored in this file:
 *
 *   /data/quotes                      price, today's %                1 request
 *   /portfolio/metrics + attribution  12m return, volatility, Sharpe  1 request
 *   /portfolio/correlation            who moves with whom             1 request
 *   /direction/{symbol}/analysis      direction, P(up), confidence,
 *                                     seven evidence categories,
 *                                     price action, expected range    1 per symbol
 *   /predict/forecast/{symbol}        next-bar price                  on demand only
 *
 * The first three are batched and land in about a second. The fourth is roughly
 * a second of feature building and model fitting per symbol, cached server-side
 * against the last printed bar, so it is streamed in at bounded concurrency and
 * the grid fills in visibly instead of waiting on the slowest name. The fifth is
 * several seconds of transformer sampling, so it is never fetched for a grid —
 * only for the one symbol whose drawer is open, and only when asked for.
 *
 * HONESTY, which governs the whole design:
 *
 *   - NEUTRAL is shown as NEUTRAL with `neutral_reason` beside it. The colour
 *     still carries the probability, because the probability is real, but a
 *     hatch overlay and a badge say the model declined to call a direction from
 *     it. Two facts, two channels, neither overwriting the other.
 *   - Backward-looking numbers (return, volatility, Sharpe, today's move) and
 *     forward-looking ones (direction, probability, evidence) are separated on
 *     every card and in the table. The optimizer solves on the former and does
 *     not read the latter; presenting them as one blob would imply a causation
 *     the system does not have.
 *   - A symbol with no reading renders as "no reading", not as a neutral grey
 *     tile that looks like a measured coin-flip.
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { C, SP500_LIST } from "../utils/data";
import { fetchDirectionAnalysis, fetchSimpleForecast } from "../utils/api";
import { eachLimited } from "../utils/concurrency";
import { useQuotes, usePortfolioMetrics, useCorrelation } from "../hooks/useMarketData";
import { Hint, Section } from "../components/UIComponents";

/* ═══════════════════════════════════════════════════════════════════════════
   COLOUR SCALES

   Three encodings, one on screen at a time, each with its own legend directly
   beneath the grid. The old tab had one scale permanently on ("today's %
   change") and a legend that claimed +/-4% while the code clamped at +/-3%.

   The ramps are not eyeballed. Each was checked for lightness monotonicity, an
   adjacent lightness delta of at least 0.06, and a single hue; the worst
   contrast between the label ink (C.text) and any tile fill in any ramp is
   5.09:1, above the 4.5:1 floor for body text. The diverging ramps are two hues
   around a neutral slate midpoint that reads as "nothing"; the sequential ramp
   is one hue, light to dark.

   Violet carries risk deliberately. Red already means "predicted down" on the
   signal scale, and a volatile stock painted red would read as a sell the moment
   the reader switched modes.
   ═══════════════════════════════════════════════════════════════════════════ */

const RAMP = {
    down: ["#1e293b", "#3c1f30", "#65203a", "#8d1b40", "#b01344"],
    up: ["#1e293b", "#1a3a3c", "#154a43", "#0a5d4a", "#026e4f"],
    risk: ["#1e2438", "#2f2e56", "#423776", "#564194", "#6b4cb3"],
};

/** A tile the models could not read. Distinct from the neutral midpoint on
 *  purpose: "no answer" and "a measured coin-flip" are not the same fact. */
const NO_READING = "#101725";

const clamp01 = (t) => (t < 0 ? 0 : t > 1 ? 1 : t);

function lerpHex(a, b, t) {
    const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
    const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
    return `#${pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, "0")).join("")}`;
}

/** Position `t` in [0,1] along a multi-stop ramp, interpolating between stops. */
function rampAt(stops, t) {
    const x = clamp01(t) * (stops.length - 1);
    const i = Math.min(Math.floor(x), stops.length - 2);
    return lerpHex(stops[i], stops[i + 1], x - i);
}

/**
 * The three colour modes.
 *
 * `domain` is stated in the legend rather than fitted to what happens to be on
 * screen. A scale that rescaled itself to the current filter would repaint every
 * surviving tile when one stock was filtered out, and two screenshots of the
 * same stock would disagree about its colour. Values outside the domain clamp,
 * and the number is printed on the tile, so clamping never hides one.
 */
const MODES = {
    signal: {
        label: "Signal",
        title: "Predicted direction",
        kind: "diverging",
        center: 0.5,
        halfSpan: 0.10,
        axis: "P(up) next session",
        legend: ["40% · leans down", "50% · coin-flip", "60% · leans up"],
        hint: "The blended probability that tomorrow closes above today. Seven categories of "
            + "technical evidence read off this stock's own bars, combined with the stored "
            + "classifier, each weighted by its measured out-of-sample Brier skill — a source "
            + "that never beat the base rate gets no vote.",
        value: (r) => r.probabilityUp,
        format: (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—"),
        better: "desc",
    },
    momentum: {
        label: "Momentum",
        title: "Realised 12-month return",
        kind: "diverging",
        center: 0,
        halfSpan: 0.40,
        axis: "annualised return, past 252 sessions",
        legend: ["−40%", "0%", "+40%"],
        hint: "What the stock actually returned, annualised over the past 252 sessions. History, "
            + "not a forecast — and the kind of number the optimizer genuinely solves on.",
        value: (r) => r.annualReturn,
        format: (v) => (Number.isFinite(v) ? `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%` : "—"),
        better: "desc",
    },
    risk: {
        label: "Risk",
        title: "Realised volatility",
        kind: "sequential",
        domain: [0.15, 0.60],
        axis: "annualised volatility, past 252 sessions",
        legend: ["15% · calm", "", "60% · volatile"],
        hint: "Annualised standard deviation of daily returns over the past 252 sessions. Violet "
            + "rather than red on purpose: red already means 'predicted down' on the signal scale.",
        value: (r) => r.annualVol,
        format: (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—"),
        better: "asc",
    },
};

const MODE_KEYS = Object.keys(MODES);

/**
 * The fill for one row under one mode, plus the raw value behind it.
 *
 * The `noEdge` branch is not a nicety. When neither the evidence stack nor the
 * classifier clears a positive out-of-sample Brier skill, the server falls back
 * to the stock's historical base rate and reports that as the probability. On a
 * real symbol that came back as 53.4% — which, coloured by probability alone,
 * painted a green tile reading "the models like this one" about a stock whose
 * models had scored *negative* skill. The number is real and stays on the tile;
 * the colour is the midpoint, because a base rate is not a signal about this
 * stock. Only the signal scale is affected: the other two are measured history,
 * which is true whether or not a model found an edge in it.
 */
function tileFill(mode, row) {
    const spec = MODES[mode];
    const value = spec.value(row);
    if (!Number.isFinite(value)) return { fill: NO_READING, value: null };
    if (spec.kind === "sequential") {
        const [lo, hi] = spec.domain;
        return { fill: rampAt(RAMP.risk, (value - lo) / (hi - lo)), value };
    }
    if (mode === "signal" && row.noEdge) return { fill: RAMP.up[0], value };
    const t = (value - spec.center) / spec.halfSpan;
    return { fill: t < 0 ? rampAt(RAMP.down, -t) : rampAt(RAMP.up, t), value };
}

/* ═══════════════════════════════════════════════════════════════════════════
   FORMATTING
   Two units share this page and must never be confused: the API returns today's
   move as a percentage number (1.24 meaning 1.24%) and every annualised figure
   as a fraction (0.32 meaning 32%).
   ═══════════════════════════════════════════════════════════════════════════ */

const money = (v) => (Number.isFinite(Number(v)) ? `$${Number(v).toFixed(2)}` : "—");
/** For values that are already a percentage. */
const signedPct = (v, d = 2) =>
    (Number.isFinite(Number(v)) ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(d)}%` : "—");
/** For values that are a fraction of one. */
const fracPct = (v, d = 1) =>
    (Number.isFinite(Number(v)) ? `${(Number(v) * 100).toFixed(d)}%` : "—");
const signedFracPct = (v, d = 1) =>
    (Number.isFinite(Number(v)) ? `${Number(v) >= 0 ? "+" : ""}${(Number(v) * 100).toFixed(d)}%` : "—");
const num = (v, d = 2) => (Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "—");

/**
 * How far a level sits from the price, in words.
 *
 * The server reports both distances as positive-when-unbroken: support is
 * (close - support) / support, resistance is (resistance - close) / close. A
 * signed percentage therefore says nothing a reader can use — the sign is the
 * same for a level 5% below and one 5% above. These name the side instead, and
 * say so explicitly when a level has already been taken out.
 */
const gapBelow = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "";
    return n >= 0 ? ` · ${n.toFixed(1)}% below` : ` · broken, ${Math.abs(n).toFixed(1)}% above`;
};
const gapAbove = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "";
    return n >= 0 ? ` · ${n.toFixed(1)}% above` : ` · broken, ${Math.abs(n).toFixed(1)}% below`;
};

const DIRECTION_COLOR = { UP: C.green, DOWN: C.red, NEUTRAL: C.textMid };
const DIRECTION_GLYPH = { UP: "▲", DOWN: "▼", NEUTRAL: "●" };

/* Fixed order so the seven dots mean the same thing on every card and the eye
   can compare one stock's strip against another's. Same order and labels as the
   Optimize tab's strip — they read the same endpoint, and two orderings of the
   same seven categories would be two vocabularies for one thing. */
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

const SECTOR_OF = Object.fromEntries(SP500_LIST.map((s) => [s.ticker, s.sector]));

/** `^GSPC` and friends are benchmarks, not holdings. They stay on the grid as
 *  context but are never offered to the optimizer, which allocates money. */
const isBenchmark = (symbol) => String(symbol || "").startsWith("^");

/** The optimizer takes at most eight names, so a shortlist may not exceed that. */
const SHORTLIST_MAX = 8;
const UNIVERSE_CAP = 16;
const EXTRA_KEY = "qv_heatmap_extra";

function loadExtra() {
    try {
        const saved = JSON.parse(localStorage.getItem(EXTRA_KEY));
        if (Array.isArray(saved)) {
            return [...new Set(saved.map((s) => String(s || "").trim().toUpperCase()).filter(Boolean))];
        }
    } catch { /* ignore */ }
    return [];
}

function saveExtra(list) {
    try { localStorage.setItem(EXTRA_KEY, JSON.stringify(list)); } catch { /* ignore */ }
}

/* ═══════════════════════════════════════════════════════════════════════════
   THE PER-SYMBOL MODEL READ

   One request per symbol, three at a time. The limit is not arbitrary: the
   route is roughly a second of pandas and model fitting on a decade of bars,
   and the FastAPI threadpool serving it is also serving the rest of the page.
   Sixteen at once buries it; sixteen in series is a blank grid for a quarter of
   a minute. Three keeps the tiles filling in visibly.
   ═══════════════════════════════════════════════════════════════════════════ */

function useDirectionGrid(symbols, enabled) {
    const [reads, setReads] = useState({});
    const [refreshing, setRefreshing] = useState(false);
    // Guards against a stale run writing over a newer one: the universe can
    // change (a ticker added, one removed) while a fan-out is mid-flight, and
    // the results of the abandoned run must not land.
    const runRef = useRef(0);
    const key = symbols.join(",");

    // Every write happens after an await, which is what keeps the mount effect
    // below free of a synchronous state update. Rows already held are left
    // alone until their replacement lands, so the grid dims in place rather
    // than collapsing to a skeleton and shoving the page around.
    const load = useCallback(async ({ refresh = false } = {}) => {
        const list = key ? key.split(",") : [];
        if (!enabled || list.length === 0) return;
        const run = ++runRef.current;

        await eachLimited(list, 3, async (symbol) => {
            let next;
            try {
                next = await fetchDirectionAnalysis(symbol, { refresh });
            } catch (error) {
                next = {
                    status: "error",
                    // The server's own explanation, not a sentence invented here:
                    // "no bars for XYZ" and "too little history" need different
                    // answers from the reader.
                    message: error?.detail || error?.message || "The analysis could not be loaded.",
                };
            }
            if (runRef.current !== run) return;
            setReads((cur) => ({ ...cur, [symbol]: next }));
        });
    }, [key, enabled]);

    useEffect(() => { load(); }, [load]);

    // Derived, not stored. A symbol with no entry has not answered yet, which
    // is the same question "is a run in flight" was being asked to answer — and
    // deriving it is what removes the state update from the effect above. The
    // one case it cannot see is a manual refresh, where every symbol already
    // has an entry, so that button owns its own flag.
    const refresh = useCallback(async () => {
        setRefreshing(true);
        try { await load({ refresh: true }); } finally { setRefreshing(false); }
    }, [load]);

    const running = refreshing || symbols.some((symbol) => !reads[symbol]);
    return { reads, running, refresh };
}

/* ═══════════════════════════════════════════════════════════════════════════
   SMALL PARTS
   ═══════════════════════════════════════════════════════════════════════════ */

/** The seven evidence categories, one dot each, in a fixed order. */
function EvidenceStrip({ evidence, dot = 8 }) {
    const bySource = Object.fromEntries((evidence || []).map((r) => [r.source, r]));
    return (
        <div style={{ display: "flex", gap: 3, alignItems: "center" }} aria-hidden="true">
            {EVIDENCE_ORDER.map((source) => {
                const row = bySource[source];
                const color = row?.leans === "up" ? C.green : row?.leans === "down" ? C.red : C.textDim;
                // Scaled against 1.5pp, which is roughly the largest single
                // category contribution the stack produces on a daily horizon.
                const strength = Math.min(1, Math.abs(Number(row?.contribution_pp) || 0) / 1.5);
                return (
                    <span
                        key={source}
                        style={{
                            width: dot, height: dot, borderRadius: "50%", flexShrink: 0,
                            background: row ? color : "transparent",
                            border: `1px solid ${row ? color : "rgba(148,163,184,.35)"}`,
                            opacity: row ? 0.35 + strength * 0.65 : 1,
                        }}
                    />
                );
            })}
        </div>
    );
}

/**
 * Conviction as a bar that grows out of the centre.
 *
 * The centre line is the coin-flip, not zero — which is the whole point. A 55%
 * probability drawn as a bar filled 55% of the way looks like a strong reading;
 * drawn as a small offset from the middle it looks like what it is.
 */
function ConvictionBar({ probabilityUp, muted }) {
    const p = Number(probabilityUp);
    const hasReading = Number.isFinite(p);
    // Same +/-10pp domain as the signal colour scale, so bar length and tile
    // colour are two views of one number rather than two different claims.
    const t = hasReading ? Math.max(-1, Math.min(1, (p - 0.5) / 0.10)) : 0;
    const color = !hasReading ? C.textDim : t >= 0 ? C.green : C.red;
    return (
        <div
            style={{
                position: "relative", height: 4, borderRadius: 2,
                background: "rgba(226,232,240,.13)", overflow: "hidden",
                opacity: muted ? 0.5 : 1,
            }}
        >
            <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "rgba(226,232,240,.4)" }} />
            {hasReading && (
                <div
                    style={{
                        position: "absolute", top: 0, bottom: 0, borderRadius: 2, background: color,
                        left: t >= 0 ? "50%" : `${50 + t * 50}%`,
                        width: `${Math.abs(t) * 50}%`,
                    }}
                />
            )}
        </div>
    );
}

/** UP / DOWN / NEUTRAL as glyph + word + colour — three channels, so the call
 *  never rests on hue alone. */
function DirectionBadge({ direction, small }) {
    const dir = String(direction || "").toUpperCase();
    const color = DIRECTION_COLOR[dir] || C.textDim;
    if (!dir) return null;
    return (
        <span style={{
            display: "inline-flex", alignItems: "center", gap: 4,
            background: `${color}1f`, border: `1px solid ${color}55`, color,
            borderRadius: 999, padding: small ? "1px 6px" : "3px 9px",
            fontSize: small ? 9.5 : 11, fontWeight: 700, lineHeight: 1.5, whiteSpace: "nowrap",
        }}>
            <span aria-hidden="true">{DIRECTION_GLYPH[dir]}</span>{dir}
        </span>
    );
}

/** The continuous gradient bar under the grid. A continuous encoding takes a
 *  continuous legend; discrete swatches would imply buckets that do not exist. */
function ScaleLegend({ mode }) {
    const spec = MODES[mode];
    const stops = spec.kind === "sequential"
        ? RAMP.risk
        : [...[...RAMP.down].reverse(), ...RAMP.up.slice(1)];
    return (
        <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
            <div style={{ flex: "1 1 260px", minWidth: 220, maxWidth: 420 }}>
                <div style={{
                    height: 9, borderRadius: 5,
                    border: "1px solid rgba(226,232,240,.14)",
                    background: `linear-gradient(90deg, ${stops.join(", ")})`,
                }} />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 5, fontSize: 9.5, color: C.textMid }}>
                    {spec.legend.map((label, i) => (
                        <span key={i} style={{ flex: 1, textAlign: i === 0 ? "left" : i === 2 ? "right" : "center" }}>{label}</span>
                    ))}
                </div>
            </div>
            <div style={{ fontSize: 10, color: C.textMid, lineHeight: 1.6, flex: "1 1 220px", minWidth: 200 }}>
                <span style={{ color: C.text, fontWeight: 700 }}>Colour = {spec.axis}.</span>{" "}
                Every tile is the same size — nothing on this grid encodes company size.
                <div style={{ marginTop: 4, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                    <span style={{
                        width: 22, height: 11, borderRadius: 3, flexShrink: 0,
                        background: "#1e293b",
                        backgroundImage: "repeating-linear-gradient(135deg, rgba(226,232,240,.22) 0 4px, transparent 4px 9px)",
                        border: "1px solid rgba(226,232,240,.16)",
                    }} />
                    <span>hatched = model declined to call a direction</span>
                    <span style={{
                        width: 22, height: 11, borderRadius: 3, flexShrink: 0, marginLeft: 6,
                        background: NO_READING, border: "1px solid rgba(226,232,240,.16)",
                    }} />
                    <span>no reading available</span>
                    {mode === "signal" && (
                        <>
                            <span style={{
                                width: 22, height: 11, borderRadius: 3, flexShrink: 0, marginLeft: 6,
                                background: RAMP.up[0], border: "1px solid rgba(226,232,240,.16)",
                            }} />
                            <span>midpoint = measured, but no edge over the base rate</span>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   ONE TILE
   The whole card is the hit target and a real <button>, so the grid is
   keyboard-navigable and every tile answers to Enter as well as to a click.
   ═══════════════════════════════════════════════════════════════════════════ */

const StockCell = memo(function StockCell({ row, mode, selected, shortlisted, onOpen, onToggle, onHover, onLeave }) {
    const { fill, value } = tileFill(mode, row);
    const spec = MODES[mode];
    const dir = row.direction;
    const unread = row.status !== "ok";

    return (
        <div style={{ position: "relative" }}>
            <button
                type="button"
                className={`hm-cell${dir === "NEUTRAL" ? " hm-neutral" : ""}${row.pending ? " hm-pending" : ""}`}
                data-selected={selected ? "true" : "false"}
                onClick={() => onOpen(row.symbol)}
                onMouseEnter={(e) => onHover({ row, rect: e.currentTarget.getBoundingClientRect() })}
                onMouseLeave={onLeave}
                onFocus={(e) => onHover({ row, rect: e.currentTarget.getBoundingClientRect() })}
                onBlur={onLeave}
                aria-label={`${row.symbol}, ${spec.title} ${spec.format(value)}, ${dir ? `predicted ${dir}` : "no model reading"}. Open details.`}
                style={{
                    width: "100%",
                    background: fill,
                    border: "1px solid rgba(226,232,240,.10)",
                    color: C.text,
                    display: "block",
                }}
            >
                {/* ── identity ─────────────────────────────────────────── */}
                <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 6 }}>
                    <span style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 15, letterSpacing: .4 }}>
                        {row.symbol}
                    </span>
                    <span style={{ fontSize: 10.5, color: "rgba(226,232,240,.62)", fontVariantNumeric: "tabular-nums" }}>
                        {money(row.price)}
                    </span>
                </div>
                <div style={{
                    fontSize: 9.5, color: "rgba(226,232,240,.5)", marginTop: 1,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                    {row.sector || (isBenchmark(row.symbol) ? "Benchmark index" : "—")}
                </div>

                {/* ── the number this grid is currently coloured by ────── */}
                <div style={{ marginTop: 9, display: "flex", alignItems: "baseline", gap: 6 }}>
                    <span style={{ fontSize: 22, fontWeight: 800, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
                        {spec.format(value)}
                    </span>
                    <span style={{ fontSize: 9, color: "rgba(226,232,240,.55)", lineHeight: 1.2 }}>
                        {mode === "signal" ? "P(up)" : mode === "momentum" ? "12m" : "vol"}
                    </span>
                </div>

                {/* ── forward-looking: the call and its conviction ─────── */}
                <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 6, minHeight: 20 }}>
                    {row.pending && !dir
                        ? <span style={{ fontSize: 9.5, color: "rgba(226,232,240,.55)" }}>analysing…</span>
                        : dir
                            ? <><DirectionBadge direction={dir} small />
                                <span style={{ fontSize: 9, color: "rgba(226,232,240,.55)" }}>
                                    {row.noEdge ? "no measured edge" : `${row.confidenceLabel} conf.`}
                                </span></>
                            : <span style={{ fontSize: 9.5, color: "rgba(226,232,240,.5)" }}>no model reading</span>}
                </div>
                <div style={{ marginTop: 6 }}>
                    <ConvictionBar probabilityUp={row.probabilityUp} muted={unread} />
                </div>
                <div style={{ marginTop: 7 }}>
                    <EvidenceStrip evidence={row.evidence} />
                </div>

                {/* ── backward-looking, kept visibly apart from the above ─ */}
                <div style={{
                    marginTop: 8, paddingTop: 6, borderTop: "1px solid rgba(226,232,240,.11)",
                    display: "flex", justifyContent: "space-between", gap: 6,
                    fontSize: 9.5, color: "rgba(226,232,240,.62)", fontVariantNumeric: "tabular-nums",
                }}>
                    <span>today {signedPct(row.changePct)}</span>
                    <span>vol {fracPct(row.annualVol, 0)}</span>
                    <span>SR {num(row.sharpe, 2)}</span>
                </div>
            </button>

            {/* Shortlist toggle. Its own control rather than a second meaning
                for the tile: opening a stock to read it and committing it to a
                portfolio are different intents, and one click should not do both. */}
            {!isBenchmark(row.symbol) && (
                <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); onToggle(row.symbol); }}
                    title={shortlisted ? `Remove ${row.symbol} from the shortlist` : `Add ${row.symbol} to the shortlist`}
                    aria-pressed={shortlisted}
                    style={{
                        position: "absolute", top: 8, right: 8, width: 19, height: 19,
                        borderRadius: 5, cursor: "pointer", padding: 0, lineHeight: 1,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 11, fontWeight: 800, zIndex: 4,
                        background: shortlisted ? C.amber : "rgba(8,12,20,.45)",
                        border: `1px solid ${shortlisted ? C.amber : "rgba(226,232,240,.3)"}`,
                        color: shortlisted ? "#0b1220" : "rgba(226,232,240,.75)",
                    }}
                >{shortlisted ? "✓" : "+"}</button>
            )}
        </div>
    );
});

/* ═══════════════════════════════════════════════════════════════════════════
   HOVER CARD
   Enhances, never gates: everything here is also on the tile, in the drawer and
   in the table view.

   Anchored to the cell rather than to the pointer. Following the cursor meant a
   state update on every mousemove — which re-rendered the whole grid — and it
   left keyboard users with nothing, since a focus event has no cursor to follow.
   Anchored, it opens once per cell, and Tab shows exactly what hover shows.
   Flips to the left of the cell when it would otherwise run off the right edge;
   the old tooltip was placed at `mouse.x + 16` with no bound and disappeared off
   screen on the last column.
   ═══════════════════════════════════════════════════════════════════════════ */

function HoverCard({ row, rect }) {
    if (!row || !rect) return null;
    const W = 260;
    const vw = window.innerWidth || 1200;
    const vh = window.innerHeight || 800;
    const left = rect.right + 10 + W <= vw - 8
        ? rect.right + 10
        : Math.max(8, rect.left - W - 10);
    const top = Math.min(Math.max(8, rect.top), vh - 230);
    const top3 = [...(row.evidence || [])]
        .filter((r) => Number.isFinite(Number(r.contribution_pp)))
        .sort((a, b) => Math.abs(b.contribution_pp) - Math.abs(a.contribution_pp))
        .slice(0, 3);

    return (
        <div
            role="tooltip"
            style={{
                position: "fixed", left, top, width: W, zIndex: 400, pointerEvents: "none",
                background: "rgba(8,12,20,.97)", border: `1px solid ${C.border}`, borderRadius: 10,
                padding: "11px 13px", boxShadow: "0 10px 34px rgba(0,0,0,.6)",
                fontFamily: "'DM Mono',monospace", fontSize: 11, color: C.textMid,
            }}
        >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 13, color: C.text }}>{row.symbol}</span>
                <DirectionBadge direction={row.direction} small />
            </div>

            {row.status === "ok" ? (
                <>
                    {[
                        ["P(up) tomorrow", fracPct(row.probabilityUp), C.text],
                        ["Confidence", row.confidenceLabel || "—", C.text],
                        ["12m return", signedFracPct(row.annualReturn), Number(row.annualReturn) >= 0 ? C.green : C.red],
                        ["Volatility", fracPct(row.annualVol), C.text],
                        ["Sharpe", num(row.sharpe), C.text],
                    ].map(([k, v, color]) => (
                        <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 12, marginBottom: 3 }}>
                            <span>{k}</span>
                            <span style={{ color, fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>{v}</span>
                        </div>
                    ))}
                    {top3.length > 0 && (
                        <div style={{ marginTop: 8, paddingTop: 7, borderTop: `1px solid ${C.border}` }}>
                            <div style={{ fontSize: 9, letterSpacing: 1, textTransform: "uppercase", color: C.textDim, marginBottom: 4 }}>
                                Strongest evidence
                            </div>
                            {top3.map((r) => (
                                <div key={r.source} style={{ display: "flex", justifyContent: "space-between", gap: 10, marginBottom: 2 }}>
                                    <span>{EVIDENCE_SHORT[r.source] || r.label}</span>
                                    <span style={{ color: r.leans === "up" ? C.green : r.leans === "down" ? C.red : C.textDim }}>
                                        {r.state}
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </>
            ) : (
                <div style={{ lineHeight: 1.55 }}>{row.message || (row.pending ? "Running the analysis…" : "No model reading.")}</div>
            )}

            <div style={{ marginTop: 8, fontSize: 9.5, color: C.textDim, textAlign: "center" }}>
                Click for the full evidence →
            </div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   TABLE VIEW
   The grid's twin, not a lesser alternative: identical data, sortable, and the
   one view that carries every value as text rather than as colour.
   ═══════════════════════════════════════════════════════════════════════════ */

const COLUMNS = [
    { key: "symbol", label: "Stock", dir: "asc" },
    { key: "direction", label: "Call", dir: "desc" },
    { key: "signal", label: "P(up)", dir: "desc" },
    { key: "confidence", label: "Conf.", dir: "desc" },
    { key: "today", label: "Today", dir: "desc" },
    { key: "momentum", label: "12m ret", dir: "desc" },
    { key: "risk", label: "Volatility", dir: "asc" },
    { key: "sharpe", label: "Sharpe", dir: "desc" },
    { key: "corr", label: "Max corr", dir: "asc" },
];

function SignalTable({ rows, sortKey, sortDir, onSort, onOpen, shortlist, onToggle }) {
    return (
        <div style={{ overflowX: "auto", maxHeight: 560, overflowY: "auto" }}>
            <table className="hm-table">
                <caption style={{ captionSide: "top", textAlign: "left", padding: "0 0 10px", fontSize: 11, color: C.textMid }}>
                    The same stocks as the grid, as text. Backward-looking columns
                    (Today, 12m, Volatility, Sharpe, Max corr) are measured history;
                    Call, P(up) and Confidence are the models&rsquo; forward-looking read.
                    Click a heading to sort, a row to open it.
                </caption>
                <thead>
                    <tr>
                        <th scope="col" style={{ cursor: "default", width: 34 }} />
                        {COLUMNS.map((col) => (
                            <th
                                key={col.key}
                                scope="col"
                                onClick={() => onSort(col.key, col.dir)}
                                aria-sort={sortKey === col.key ? (sortDir === "asc" ? "ascending" : "descending") : "none"}
                                style={{ color: sortKey === col.key ? C.amber : undefined }}
                            >
                                {col.label}{sortKey === col.key ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row) => {
                        const on = shortlist.includes(row.symbol);
                        return (
                            <tr
                                key={row.symbol}
                                tabIndex={0}
                                onClick={() => onOpen(row.symbol)}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(row.symbol); }
                                }}
                                className={row.pending ? "hm-pending" : undefined}
                            >
                                <td onClick={(e) => e.stopPropagation()}>
                                    {!isBenchmark(row.symbol) && (
                                        <input
                                            type="checkbox"
                                            checked={on}
                                            onChange={() => onToggle(row.symbol)}
                                            aria-label={`Shortlist ${row.symbol}`}
                                            style={{ accentColor: C.amber, cursor: "pointer" }}
                                        />
                                    )}
                                </td>
                                <td style={{ color: C.text, fontWeight: 700 }}>
                                    {row.symbol}
                                    <span style={{ color: C.textDim, marginLeft: 6, fontSize: 10 }}>{row.sector || ""}</span>
                                </td>
                                <td style={{ color: DIRECTION_COLOR[row.direction] || C.textDim, fontWeight: 700 }}>
                                    {row.direction ? `${DIRECTION_GLYPH[row.direction]} ${row.direction}` : (row.pending ? "…" : "—")}
                                </td>
                                <td style={{ color: C.text }}>{fracPct(row.probabilityUp)}</td>
                                <td style={{ color: C.textMid }}>{row.confidenceLabel || "—"}</td>
                                <td style={{ color: Number(row.changePct) >= 0 ? C.green : C.red }}>{signedPct(row.changePct)}</td>
                                <td style={{ color: Number(row.annualReturn) >= 0 ? C.green : C.red }}>{signedFracPct(row.annualReturn)}</td>
                                <td style={{ color: C.text }}>{fracPct(row.annualVol)}</td>
                                <td style={{ color: C.text }}>{num(row.sharpe)}</td>
                                <td style={{ color: Math.abs(Number(row.maxCorr?.value)) >= 0.8 ? C.amber : C.textMid }}>
                                    {row.maxCorr ? `${num(row.maxCorr.value)} ${row.maxCorr.symbol}` : "—"}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   DETAIL DRAWER
   Replaces the old floating tooltip as the place detail lives. A tooltip that
   has to hold seven evidence rows, a price-action reading and an expected range
   is a panel wearing the wrong clothes.
   ═══════════════════════════════════════════════════════════════════════════ */

function DrawerRow({ label, children, tone }) {
    return (
        <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 14,
            padding: "8px 0", borderBottom: `1px solid ${C.border}`,
        }}>
            <span style={{ color: C.textMid, fontSize: 11 }}>{label}</span>
            <span style={{
                color: tone || C.text, fontSize: 12.5, fontWeight: 700,
                fontVariantNumeric: "tabular-nums", textAlign: "right",
            }}>{children}</span>
        </div>
    );
}

function DetailDrawer({ row, onClose, onOpenTab, shortlisted, onToggle }) {
    // Reset when the drawer moves to another stock is handled by the `key` at
    // the call site, which remounts this component per symbol. A forecast left
    // standing would otherwise be attributed to the wrong stock, and an effect
    // that clears it runs a render later — long enough to paint one symbol's
    // price under another symbol's name.
    const [forecast, setForecast] = useState(null);
    const [forecastState, setForecastState] = useState("idle");

    useEffect(() => {
        const onKey = (e) => { if (e.key === "Escape") onClose(); };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onClose]);

    const runForecast = async () => {
        setForecastState("loading");
        try {
            const data = await fetchSimpleForecast(row.symbol);
            setForecast(data);
            setForecastState("done");
        } catch (error) {
            setForecast({ status: "error", message: error?.detail || error?.message || "Forecast failed." });
            setForecastState("done");
        }
    };

    if (!row) return null;
    const dirColor = DIRECTION_COLOR[row.direction] || C.textDim;
    const pa = row.priceAction;
    const range = row.expectedRange;

    const btn = (bg, fg, border) => ({
        background: bg, color: fg, border: `1px solid ${border}`, borderRadius: 8,
        padding: "9px 12px", fontSize: 11.5, fontWeight: 700, cursor: "pointer",
        fontFamily: "'DM Mono',monospace", flex: "1 1 auto", whiteSpace: "nowrap",
    });

    return (
        <>
            <div
                onClick={onClose}
                style={{ position: "fixed", inset: 0, zIndex: 450, background: "rgba(4,8,16,.6)", backdropFilter: "blur(2px)" }}
            />
            <aside
                className="hm-drawer"
                role="dialog"
                aria-label={`${row.symbol} model detail`}
                style={{
                    position: "fixed", top: 0, right: 0, bottom: 0, zIndex: 460,
                    width: "min(420px, 94vw)", overflowY: "auto",
                    background: C.bg1, borderLeft: `1px solid ${C.border}`,
                    padding: "18px 20px 28px", fontFamily: "'DM Mono',monospace",
                    boxShadow: "-16px 0 48px rgba(0,0,0,.55)",
                }}
            >
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 4 }}>
                    <div>
                        <div style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 24, color: C.text, lineHeight: 1.1 }}>
                            {row.symbol}
                        </div>
                        <div style={{ fontSize: 10.5, color: C.textDim, marginTop: 3 }}>
                            {row.sector || (isBenchmark(row.symbol) ? "Benchmark index" : "—")}
                            {row.asOf ? ` · bars to ${row.asOf}` : ""}
                        </div>
                    </div>
                    <button
                        type="button" onClick={onClose} aria-label="Close"
                        style={{
                            background: "transparent", border: `1px solid ${C.border}`, borderRadius: 7,
                            color: C.textMid, width: 28, height: 28, cursor: "pointer", fontSize: 13, flexShrink: 0,
                        }}
                    >✕</button>
                </div>

                {row.status !== "ok" ? (
                    <div style={{
                        marginTop: 16, padding: "14px 16px", borderRadius: 9,
                        border: `1px solid ${C.border}`, background: C.bg2,
                        color: C.textMid, fontSize: 12, lineHeight: 1.6,
                    }}>
                        {row.pending ? "Running the direction analysis…" : (row.message || "No model reading is available for this symbol.")}
                    </div>
                ) : (
                    <>
                        {/* ── The call ──────────────────────────────────── */}
                        <div style={{
                            marginTop: 14, padding: "14px 16px", borderRadius: 10,
                            background: C.bg2, border: `1px solid ${dirColor}44`, borderLeft: `3px solid ${dirColor}`,
                        }}>
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                                <span style={{ color: dirColor, fontSize: 20, fontWeight: 900 }}>
                                    {DIRECTION_GLYPH[row.direction]} {row.direction}
                                </span>
                                <span style={{ fontSize: 22, fontWeight: 800, color: C.text, fontVariantNumeric: "tabular-nums" }}>
                                    {fracPct(row.probabilityUp)}
                                </span>
                            </div>
                            <div style={{ marginTop: 8 }}>
                                <ConvictionBar probabilityUp={row.probabilityUp} />
                            </div>
                            <div style={{ fontSize: 10.5, color: C.textMid, marginTop: 8, lineHeight: 1.55 }}>
                                Probability that the next session closes up, against a base rate of{" "}
                                {fracPct(row.baseRate)}. Confidence:{" "}
                                <span style={{ color: C.text, fontWeight: 700 }}>{row.confidenceLabel}</span>.
                            </div>
                            {row.confidenceBasis && (
                                <div style={{ fontSize: 10, color: C.textDim, marginTop: 5, lineHeight: 1.5 }}>
                                    {row.confidenceBasis}
                                </div>
                            )}
                            {/* Why a direction was withheld. A NEUTRAL with no reason
                                beside it reads as a model with no opinion rather than
                                one that has an opinion it may not act on. */}
                            {row.neutralReason && (
                                <div style={{
                                    marginTop: 10, paddingTop: 9, borderTop: `1px solid ${C.border}`,
                                    fontSize: 10.5, color: C.textMid, lineHeight: 1.6,
                                }}>
                                    <span style={{ color: C.amber, fontWeight: 700 }}>No call: </span>
                                    {row.neutralReason}
                                </div>
                            )}
                        </div>

                        {/* ── Evidence ──────────────────────────────────── */}
                        <div style={{ marginTop: 20, marginBottom: 8, display: "flex", alignItems: "center" }}>
                            <span style={{
                                fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 10,
                                letterSpacing: 1.5, textTransform: "uppercase", color: C.textMid,
                            }}>Technical evidence</span>
                            <Hint text="Seven categories read off this stock's own daily bars. The percentage-point figure is the counterfactual: the probability with that category in the sum, minus the probability without it." />
                        </div>
                        {(row.evidence || []).map((r) => {
                            const color = r.leans === "up" ? C.green : r.leans === "down" ? C.red : C.textDim;
                            return (
                                <div key={r.source} style={{ padding: "8px 0", borderBottom: `1px solid ${C.border}` }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                                        <span style={{ color: C.text, fontSize: 11.5, fontWeight: 700 }}>
                                            {EVIDENCE_SHORT[r.source] || r.label}
                                        </span>
                                        <span style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                                            <span style={{ color, fontSize: 11 }}>{r.state}</span>
                                            <span style={{
                                                color: C.textMid, fontSize: 10.5, fontVariantNumeric: "tabular-nums",
                                                minWidth: 52, textAlign: "right",
                                            }}>{signedPct(r.contribution_pp, 2).replace("%", "pp")}</span>
                                        </span>
                                    </div>
                                    {r.detail && (
                                        <div style={{ fontSize: 10, color: C.textDim, marginTop: 3, lineHeight: 1.5 }}>{r.detail}</div>
                                    )}
                                </div>
                            );
                        })}

                        {/* ── Price action ──────────────────────────────── */}
                        {pa?.available && (
                            <>
                                <div style={{
                                    marginTop: 20, marginBottom: 6,
                                    fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 10,
                                    letterSpacing: 1.5, textTransform: "uppercase", color: C.textMid,
                                }}>Price action</div>
                                {/* Two readings, not one: `label` is the composite
                                    price-action score, `structure_label` is the swing
                                    structure alone. They routinely disagree — a
                                    bullish composite over no clear structure — and
                                    showing one under the other's tone said something
                                    neither of them did. */}
                                <DrawerRow label="Overall reading" tone={pa.label === "bullish" ? C.green : pa.label === "bearish" ? C.red : C.text}>
                                    {pa.label ? pa.label[0].toUpperCase() + pa.label.slice(1) : "—"}
                                </DrawerRow>
                                <DrawerRow label="Swing structure">
                                    {pa.structure_label || "—"}
                                </DrawerRow>
                                <DrawerRow label={`Support (${pa.levels?.window ?? "—"}d low)`}>
                                    {money(pa.levels?.support)}
                                    <span style={{ color: C.textDim, fontWeight: 400, fontSize: 10.5 }}>
                                        {gapBelow(pa.levels?.support_distance_pct)}
                                    </span>
                                </DrawerRow>
                                <DrawerRow label={`Resistance (${pa.levels?.window ?? "—"}d high)`}>
                                    {money(pa.levels?.resistance)}
                                    <span style={{ color: C.textDim, fontWeight: 400, fontSize: 10.5 }}>
                                        {gapAbove(pa.levels?.resistance_distance_pct)}
                                    </span>
                                </DrawerRow>
                                {pa.events?.length > 0 && (
                                    <div style={{ padding: "9px 0", fontSize: 10.5, color: C.textMid, lineHeight: 1.65 }}>
                                        {pa.events.map((e) => (
                                            <div key={e} style={{ display: "flex", gap: 7 }}>
                                                <span style={{ color: C.amber }}>•</span><span>{e}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </>
                        )}

                        {/* ── Expected range ────────────────────────────── */}
                        {range?.available && (
                            <>
                                <div style={{
                                    marginTop: 18, marginBottom: 6, display: "flex", alignItems: "center",
                                }}>
                                    <span style={{
                                        fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 10,
                                        letterSpacing: 1.5, textTransform: "uppercase", color: C.textMid,
                                    }}>Expected range</span>
                                    <Hint text="Not a forecast path. The 10th-to-90th percentile of what actually followed the most similar historical setups for this stock, applied to the last close." />
                                </div>
                                <div style={{
                                    display: "flex", justifyContent: "space-between", gap: 8,
                                    padding: "12px 14px", borderRadius: 9, background: C.bg2, border: `1px solid ${C.border}`,
                                }}>
                                    {[["10th", range.price_low, C.red], ["median", range.price_median, C.text], ["90th", range.price_high, C.green]].map(([k, v, color]) => (
                                        <div key={k} style={{ textAlign: "center", flex: 1 }}>
                                            <div style={{ fontSize: 9, color: C.textDim, letterSpacing: .6 }}>{k}</div>
                                            <div style={{ fontSize: 13, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>{money(v)}</div>
                                        </div>
                                    ))}
                                </div>
                                <div style={{ fontSize: 10, color: C.textDim, marginTop: 5, lineHeight: 1.5 }}>
                                    From {range.n_samples} matched historical setups.
                                </div>
                            </>
                        )}

                        {/* ── Measured history — kept apart from the forecast ── */}
                        <div style={{
                            marginTop: 20, marginBottom: 4, display: "flex", alignItems: "center",
                        }}>
                            <span style={{
                                fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 10,
                                letterSpacing: 1.5, textTransform: "uppercase", color: C.textMid,
                            }}>Measured history</span>
                            <Hint text="Realised over the past 252 sessions. This is what the optimizer solves on — it does not read the direction call above." />
                        </div>
                        <DrawerRow label="Price · today">
                            {money(row.price)}
                            <span style={{ color: Number(row.changePct) >= 0 ? C.green : C.red, marginLeft: 8 }}>
                                {signedPct(row.changePct)}
                            </span>
                        </DrawerRow>
                        <DrawerRow label="12-month return" tone={Number(row.annualReturn) >= 0 ? C.green : C.red}>
                            {signedFracPct(row.annualReturn)}
                        </DrawerRow>
                        <DrawerRow label="Annualised volatility">{fracPct(row.annualVol)}</DrawerRow>
                        <DrawerRow label="Sharpe ratio">{num(row.sharpe)}</DrawerRow>
                        {row.maxCorr && (
                            <DrawerRow
                                label="Strongest co-movement"
                                tone={Math.abs(row.maxCorr.value) >= 0.8 ? C.amber : C.text}
                            >
                                {row.maxCorr.symbol} · {num(row.maxCorr.value)}
                            </DrawerRow>
                        )}
                        {/* Ranked by magnitude, so the pair can be strongly inverse as
                            well as strongly aligned. The two are not the same fact and
                            do not get the same sentence. */}
                        {Math.abs(row.maxCorr?.value ?? 0) >= 0.8 && (
                            <div style={{ fontSize: 10.5, color: C.amber, marginTop: 7, lineHeight: 1.6 }}>
                                {row.maxCorr.value > 0
                                    ? "These two move almost as one position. Holding both buys far less "
                                    + "diversification than the position count suggests."
                                    : "These two move almost exactly opposite. Held together they largely "
                                    + "cancel — which is a hedge, not a return."}
                            </div>
                        )}

                        {/* ── The expensive call, only when asked for ───── */}
                        <div style={{ marginTop: 20 }}>
                            {forecastState === "idle" && (
                                <button type="button" onClick={runForecast} style={btn("transparent", C.amber, `${C.amber}55`)}>
                                    Run the price forecast for {row.symbol} →
                                </button>
                            )}
                            {forecastState === "loading" && (
                                <div style={{ fontSize: 11, color: C.textMid, textAlign: "center", padding: "10px 0" }}>
                                    Sampling the forecast models… this takes a few seconds.
                                </div>
                            )}
                            {forecastState === "done" && forecast && (
                                forecast.status === "ok" ? (
                                    <div style={{ padding: "12px 14px", borderRadius: 9, background: C.bg2, border: `1px solid ${C.border}` }}>
                                        <div style={{ fontSize: 9.5, letterSpacing: 1.4, textTransform: "uppercase", color: C.textDim, marginBottom: 8 }}>
                                            {forecast.horizon_label || "Next 1 day"}
                                        </div>
                                        <div style={{ display: "flex", alignItems: "baseline", gap: 9, flexWrap: "wrap" }}>
                                            <span style={{ color: C.textMid, fontSize: 12 }}>{money(forecast.anchor_price)}</span>
                                            <span style={{ color: C.textDim }}>→</span>
                                            <span style={{ fontSize: 19, fontWeight: 800, color: C.text }}>{money(forecast.forecast_price)}</span>
                                            <span style={{
                                                fontSize: 12, fontWeight: 700,
                                                color: Number(forecast.expected_change_pct) >= 0 ? C.green : C.red,
                                            }}>{signedPct(forecast.expected_change_pct)}</span>
                                        </div>
                                        <div style={{ fontSize: 10, color: C.textDim, marginTop: 7, lineHeight: 1.5 }}>
                                            Measured from the {forecast.as_of} close the models actually read.
                                            {forecast.models?.length ? ` Models: ${forecast.models.join(" · ")}.` : ""}
                                        </div>
                                    </div>
                                ) : (
                                    <div style={{ fontSize: 11, color: C.textMid, lineHeight: 1.6 }}>
                                        {forecast.message || "No forecast is available for this symbol."}
                                    </div>
                                )
                            )}
                        </div>
                    </>
                )}

                {/* ── Where this stock goes next ─────────────────────── */}
                <div style={{ display: "flex", gap: 8, marginTop: 22, flexWrap: "wrap" }}>
                    {!isBenchmark(row.symbol) && (
                        <button
                            type="button"
                            onClick={() => onToggle(row.symbol)}
                            style={btn(shortlisted ? C.amber : "transparent", shortlisted ? "#0b1220" : C.amber, `${C.amber}66`)}
                        >
                            {shortlisted ? "✓ On shortlist" : "+ Add to shortlist"}
                        </button>
                    )}
                    <button type="button" onClick={() => onOpenTab(row.symbol, "predictions")} style={btn(C.bg2, C.text, C.border)}>
                        Predictions
                    </button>
                    <button type="button" onClick={() => onOpenTab(row.symbol, "analysis")} style={btn(C.bg2, C.text, C.border)}>
                        Charts
                    </button>
                </div>
            </aside>
        </>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   CONTROLS
   ═══════════════════════════════════════════════════════════════════════════ */

function Segmented({ options, value, onChange, ariaLabel }) {
    return (
        <div
            role="group"
            aria-label={ariaLabel}
            style={{
                display: "inline-flex", background: C.bg2, border: `1px solid ${C.border}`,
                borderRadius: 8, padding: 3, gap: 3,
            }}
        >
            {options.map((opt) => {
                const on = value === opt.key;
                return (
                    <button
                        key={opt.key}
                        type="button"
                        onClick={() => onChange(opt.key)}
                        title={opt.title}
                        aria-pressed={on}
                        style={{
                            background: on ? C.amber : "transparent",
                            color: on ? "#0b1220" : C.textMid,
                            border: "none", borderRadius: 6, padding: "6px 12px",
                            fontSize: 11, fontWeight: on ? 800 : 600, cursor: "pointer",
                            fontFamily: "'DM Mono',monospace", whiteSpace: "nowrap",
                        }}
                    >{opt.label}</button>
                );
            })}
        </div>
    );
}

/** The pipeline this tab sits inside, with this stage lit. Same idiom as the
 *  Optimize tab, so the two read as one journey rather than two features. */
function Pipeline() {
    const steps = ["Stock data", "Technical analysis + prediction", "Heatmap", "Shortlist", "Optimize"];
    return (
        <div style={{
            display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap",
            fontSize: 10, color: C.textDim, fontFamily: "'DM Mono',monospace",
        }}>
            {steps.map((s, i) => (
                <span key={s} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{
                        color: i === 2 ? C.amber : i < 2 ? C.textMid : C.textDim,
                        fontWeight: i === 2 ? 700 : 400,
                        borderBottom: i === 2 ? `1px solid ${C.amber}` : "1px solid transparent",
                        paddingBottom: 1,
                    }}>{s}</span>
                    {i < steps.length - 1 && <span style={{ opacity: .5 }}>→</span>}
                </span>
            ))}
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   THE TAB
   ═══════════════════════════════════════════════════════════════════════════ */

export default function HeatmapTab({ apiConnected, watchlist = [], onOpenTicker, onOptimize, notify }) {
    const [mode, setMode] = useState("signal");
    const [view, setView] = useState("grid");
    const [filter, setFilter] = useState("all");
    const [search, setSearch] = useState("");
    // "auto" follows the active colour mode, so the grid is ordered by whatever
    // it is currently coloured by. Any explicit choice pins it.
    const [sortKey, setSortKey] = useState("auto");
    const [sortDir, setSortDir] = useState("desc");
    const [extra, setExtra] = useState(loadExtra);
    const [extraInput, setExtraInput] = useState("");
    const [shortlist, setShortlist] = useState([]);
    const [openSymbol, setOpenSymbol] = useState(null);
    const [hover, setHover] = useState(null);

    useEffect(() => { saveExtra(extra); }, [extra]);

    // Stable identities: the tiles are memoised, and an inline arrow here would
    // hand every one of them a new prop on every render and defeat that.
    const clearHover = useCallback(() => setHover(null), []);

    /* ── Universe ───────────────────────────────────────────────────────── */
    const universe = useMemo(() => {
        const seen = new Set();
        const out = [];
        for (const raw of [...watchlist, ...extra]) {
            const t = String(raw || "").trim().toUpperCase();
            if (!t || seen.has(t)) continue;
            seen.add(t);
            out.push(t);
            if (out.length >= UNIVERSE_CAP) break;
        }
        return out;
    }, [watchlist, extra]);

    /* ── The three batched reads, plus the streamed per-symbol one ──────── */
    const quotesQuery = useQuotes(universe, apiConnected);
    const metricsQuery = usePortfolioMetrics(universe, { enabled: apiConnected });
    const correlationQuery = useCorrelation(universe, { enabled: apiConnected });
    const { reads, running, refresh } = useDirectionGrid(universe, apiConnected);

    /* ── One row per symbol, assembled from all four sources ────────────── */
    // The three payloads are unwrapped inside the memo rather than above it.
    // `data ?? {}` is a fresh object on every render, so as dependencies they
    // would invalidate this memo on every pointer move over the grid.
    const quotesData = quotesQuery.data;
    const metricsData = metricsQuery.data;
    const correlationData = correlationQuery.data;

    const rows = useMemo(() => {
        const quotes = quotesData ?? {};
        const byStock = metricsData?.attribution?.by_stock ?? {};
        const corrMatrix = correlationData?.matrix ?? null;

        return universe.map((symbol) => {
            const quote = quotes[symbol] || {};
            const attribution = byStock[symbol] || {};
            // No entry means the fan-out has not reached this symbol yet.
            const read = reads[symbol];
            const pending = !read;

            // The strongest correlation this stock has with anything else on the
            // grid. Self-correlation is 1 by definition and is excluded.
            let maxCorr = null;
            const rowCorr = corrMatrix?.[symbol];
            if (rowCorr) {
                for (const [other, value] of Object.entries(rowCorr)) {
                    if (other === symbol || !Number.isFinite(Number(value))) continue;
                    if (!maxCorr || Math.abs(value) > Math.abs(maxCorr.value)) {
                        maxCorr = { symbol: other, value: Number(value) };
                    }
                }
            }

            return {
                symbol,
                sector: SECTOR_OF[symbol] || null,
                // Backward-looking.
                price: quote.price ?? read?.last_close ?? null,
                changePct: quote.change ?? null,
                annualReturn: Number.isFinite(Number(attribution.stock_annual_return)) ? Number(attribution.stock_annual_return) : null,
                annualVol: Number.isFinite(Number(attribution.stock_annual_volatility)) ? Number(attribution.stock_annual_volatility) : null,
                sharpe: Number.isFinite(Number(attribution.stock_sharpe)) ? Number(attribution.stock_sharpe) : null,
                maxCorr,
                // Forward-looking.
                status: read?.status ?? "pending",
                pending,
                message: read?.message ?? null,
                direction: read?.direction ?? null,
                probabilityUp: Number.isFinite(Number(read?.probability_up)) ? Number(read.probability_up) : null,
                baseRate: read?.base_rate ?? null,
                confidenceLabel: read?.confidence?.label ?? null,
                confidenceBasis: read?.confidence?.basis ?? null,
                // Both blend weights at zero means the answer fell back to the
                // base rate: measured, and measured to have no edge. Distinct
                // from "no reading", which is the absence of a measurement.
                noEdge: read?.status === "ok"
                    && Number(read?.blend?.evidence_stack?.weight ?? 0) === 0
                    && Number(read?.blend?.classifier?.weight ?? 0) === 0,
                neutralReason: read?.neutral_reason ?? null,
                evidence: read?.evidence ?? null,
                priceAction: read?.price_action ?? null,
                expectedRange: read?.expected_range ?? null,
                asOf: read?.as_of ?? null,
            };
        });
    }, [universe, quotesData, metricsData, correlationData, reads]);

    /* ── Filter, search, sort ───────────────────────────────────────────── */
    const visible = useMemo(() => {
        const q = search.trim().toUpperCase();
        let list = rows.filter((r) => {
            if (q && !r.symbol.includes(q) && !(r.sector || "").toUpperCase().includes(q)) return false;
            if (filter === "up") return r.direction === "UP";
            if (filter === "down") return r.direction === "DOWN";
            if (filter === "edge") return r.status === "ok" && !r.noEdge;
            return true;
        });

        const key = sortKey === "auto" ? mode : sortKey;
        const dir = sortKey === "auto" ? (MODES[mode].better === "asc" ? "asc" : "desc") : sortDir;
        const pick = {
            symbol: (r) => r.symbol,
            direction: (r) => (r.direction === "UP" ? 2 : r.direction === "NEUTRAL" ? 1 : r.direction === "DOWN" ? 0 : -1),
            signal: (r) => r.probabilityUp,
            confidence: (r) => ({ High: 3, Moderate: 2, Low: 1 }[r.confidenceLabel] ?? 0),
            today: (r) => r.changePct,
            momentum: (r) => r.annualReturn,
            risk: (r) => r.annualVol,
            sharpe: (r) => r.sharpe,
            corr: (r) => (r.maxCorr ? Math.abs(r.maxCorr.value) : null),
        }[key] || ((r) => r.probabilityUp);

        list = [...list].sort((a, b) => {
            const va = pick(a);
            const vb = pick(b);
            // A missing value is never "the best" — an unread stock must not sort
            // to the top of a list the reader is about to pick a portfolio from.
            if (va == null && vb == null) return a.symbol < b.symbol ? -1 : 1;
            if (va == null) return 1;
            if (vb == null) return -1;
            if (typeof va === "string") return dir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
            return dir === "asc" ? va - vb : vb - va;
        });
        return list;
    }, [rows, search, filter, sortKey, sortDir, mode]);

    /* ── Shortlist ──────────────────────────────────────────────────────── */
    const toggleShortlist = useCallback((symbol) => {
        setShortlist((cur) => {
            if (cur.includes(symbol)) return cur.filter((s) => s !== symbol);
            if (cur.length >= SHORTLIST_MAX) {
                notify?.(`The optimizer takes at most ${SHORTLIST_MAX} stocks.`);
                return cur;
            }
            return [...cur, symbol];
        });
    }, [notify]);

    /* Two rules, tried in order, and the notification names the one that fired.
       The first is the strict read; the second exists because the direction gate
       is strict enough that a whole watchlist can sit at NEUTRAL, and a button
       that then silently selects nothing is not a useful control. What it must
       never do is invent a ranking — both rules are thresholds on numbers the
       server already computed, and the reader is told which was applied. */
    const autoPick = useCallback(() => {
        const eligible = rows.filter((r) => !isBenchmark(r.symbol) && r.status === "ok");
        const byProbability = (a, b) => (b.probabilityUp ?? 0) - (a.probabilityUp ?? 0);

        const called = eligible
            .filter((r) => r.direction === "UP" && r.confidenceLabel && r.confidenceLabel !== "Low")
            .sort(byProbability);
        // The models measured an edge here, and that edge points up: P(up) above
        // this stock's own base rate, not above a flat 50%.
        const leaning = eligible
            .filter((r) => !r.noEdge && Number(r.probabilityUp) > Number(r.baseRate ?? 0.5))
            .sort(byProbability);

        const [picked, rule] = called.length
            ? [called, "predicted UP above Low confidence"]
            : leaning.length
                ? [leaning, "no UP call was made anywhere, so these are the stocks where the models "
                    + "measured an edge and it points above their own base rate"]
                : [[], null];

        const picks = picked.slice(0, SHORTLIST_MAX).map((r) => r.symbol);
        setShortlist(picks);
        notify?.(picks.length
            ? `${picks.length} shortlisted — ${rule}.`
            : "No stock currently clears either rule: none has an UP call, and none has a measured edge pointing up.");
    }, [rows, notify]);

    const addExtra = () => {
        const t = extraInput.trim().toUpperCase();
        if (!t) return;
        if (universe.includes(t)) { setExtraInput(""); return; }
        if (universe.length >= UNIVERSE_CAP) { notify?.(`The heatmap holds ${UNIVERSE_CAP} stocks at most.`); return; }
        setExtra((cur) => [...cur, t]);
        setExtraInput("");
    };

    const handleSort = (key, defaultDir) => {
        if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
        else { setSortKey(key); setSortDir(defaultDir); }
    };

    const openRow = visible.find((r) => r.symbol === openSymbol)
        || rows.find((r) => r.symbol === openSymbol)
        || null;

    const readCount = rows.filter((r) => r.status === "ok").length;
    const upCount = rows.filter((r) => r.direction === "UP").length;
    const edgeCount = rows.filter((r) => r.status === "ok" && !r.noEdge).length;
    const downCount = rows.filter((r) => r.direction === "DOWN").length;
    const neutralCount = rows.filter((r) => r.direction === "NEUTRAL").length;

    const inputStyle = {
        background: C.bg2, color: C.text, border: `1px solid ${C.border}`,
        borderRadius: 8, padding: "7px 11px", fontSize: 11.5,
        fontFamily: "'DM Mono',monospace", outline: "none",
    };

    if (!apiConnected) {
        return (
            <div style={{ textAlign: "center", padding: 60, color: C.textDim, fontSize: 13 }}>
                🔌 Connect to the API server to screen your stocks.
            </div>
        );
    }

    return (
        <div className="fade-up" style={{ fontFamily: "'DM Mono',monospace", paddingBottom: 40 }}>
            {/* ── Header ─────────────────────────────────────────────────── */}
            <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text, marginBottom: 6 }}>
                🌡️ Signal Heatmap
            </h1>
            <div style={{ fontSize: 12, color: C.textMid, marginBottom: 10, maxWidth: 780, lineHeight: 1.65 }}>
                Every stock you follow on one screen, coloured by what the prediction models and
                technical analysis say about it — so the strong and the weak separate at a glance,
                and the ones worth allocating to go straight to the optimizer.
            </div>
            <Pipeline />

            {/* ── Universe ───────────────────────────────────────────────── */}
            <div style={{ marginTop: 18 }}>
                <Section
                    title="Stocks on the grid"
                    hint={`Your watchlist, plus anything you add here. Capped at ${UNIVERSE_CAP} — beyond that the per-symbol model runs stop being worth the wait.`}
                    right={
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ fontSize: 10, color: C.textDim }}>
                                {readCount}/{universe.length} analysed
                            </span>
                            <button
                                type="button"
                                onClick={refresh}
                                disabled={running}
                                title="Recompute every reading instead of serving the server's cache"
                                style={{
                                    background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 7,
                                    color: C.textMid, padding: "5px 11px", fontSize: 10.5,
                                    cursor: running ? "not-allowed" : "pointer", opacity: running ? .5 : 1,
                                    fontFamily: "'DM Mono',monospace",
                                }}
                            >{running ? "analysing…" : "↻ Refresh"}</button>
                        </div>
                    }
                >
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 7, alignItems: "center" }}>
                        {universe.map((t) => {
                            const isExtra = extra.includes(t) && !watchlist.includes(t);
                            return (
                                <span
                                    key={t}
                                    style={{
                                        display: "inline-flex", alignItems: "center", gap: 6,
                                        background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 20,
                                        padding: "5px 10px", fontSize: 11, color: C.textMid,
                                    }}
                                >
                                    <span style={{ color: C.text }}>{t}</span>
                                    {isBenchmark(t) && <span style={{ fontSize: 9, color: C.textDim }}>index</span>}
                                    {isExtra && (
                                        <button
                                            type="button"
                                            onClick={() => setExtra((cur) => cur.filter((s) => s !== t))}
                                            aria-label={`Remove ${t} from the heatmap`}
                                            style={{
                                                background: "none", border: "none", color: C.textDim,
                                                cursor: "pointer", fontSize: 11, padding: 0, lineHeight: 1,
                                            }}
                                        >✕</button>
                                    )}
                                </span>
                            );
                        })}
                        <input
                            value={extraInput}
                            onChange={(e) => setExtraInput(e.target.value.toUpperCase())}
                            onKeyDown={(e) => { if (e.key === "Enter") addExtra(); }}
                            placeholder="Add a ticker…"
                            aria-label="Add a ticker to the heatmap"
                            style={{ ...inputStyle, width: 132 }}
                        />
                        <button type="button" onClick={addExtra} style={{ ...inputStyle, cursor: "pointer", color: C.textMid, background: C.bg3 }}>
                            + Add
                        </button>
                    </div>
                    <div style={{ fontSize: 10, color: C.textDim, marginTop: 9, lineHeight: 1.6 }}>
                        Tickers added here stay on the heatmap only — they do not join your watchlist.
                        {universe.length === 1 && (
                            <span style={{ color: C.amber }}>
                                {" "}Return, volatility and correlation need at least two stocks to
                                compare, so the Momentum and Risk views are empty until you add one.
                            </span>
                        )}
                    </div>
                </Section>
            </div>

            {/* ── One control row, above everything it scopes ─────────────── */}
            <div style={{
                display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
                margin: "16px 0 12px",
            }}>
                <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                    <span style={{ fontSize: 9.5, letterSpacing: 1.3, textTransform: "uppercase", color: C.textDim, fontFamily: "'Syne',sans-serif", fontWeight: 700 }}>
                        Colour by
                    </span>
                    <Segmented
                        ariaLabel="What the tile colour means"
                        value={mode}
                        onChange={setMode}
                        options={MODE_KEYS.map((k) => ({ key: k, label: MODES[k].label, title: MODES[k].title }))}
                    />
                    <Hint text={MODES[mode].hint} />
                </div>

                <Segmented
                    ariaLabel="Filter by the model's call"
                    value={filter}
                    onChange={setFilter}
                    options={[
                        { key: "all", label: `All ${rows.length}` },
                        { key: "up", label: `▲ ${upCount}`, title: "Predicted UP" },
                        { key: "down", label: `▼ ${downCount}`, title: "Predicted DOWN" },
                        { key: "edge", label: `Edge ${edgeCount}`, title: "Stocks where at least one source cleared a positive out-of-sample Brier skill — the models have measured, rather than assumed, something about them" },
                    ]}
                />

                <input
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Filter by ticker or sector…"
                    aria-label="Filter by ticker or sector"
                    style={{ ...inputStyle, flex: "1 1 160px", minWidth: 140, maxWidth: 240 }}
                />

                <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
                    {sortKey !== "auto" && (
                        <button
                            type="button"
                            onClick={() => { setSortKey("auto"); setSortDir("desc"); }}
                            style={{ ...inputStyle, cursor: "pointer", color: C.amber, background: "transparent", fontSize: 10.5 }}
                        >sorted by {sortKey} ✕</button>
                    )}
                    <Segmented
                        ariaLabel="Grid or table"
                        value={view}
                        onChange={setView}
                        options={[
                            { key: "grid", label: "Grid", title: "Colour-coded tiles" },
                            { key: "table", label: "Table", title: "The same data as sortable text" },
                        ]}
                    />
                </div>
            </div>

            {/* ── Failures that would otherwise show as silently missing data ── */}
            {(metricsQuery.error || correlationQuery.error) && (
                <div
                    role="alert"
                    style={{
                        background: `${C.red}14`, border: `1px solid ${C.red}44`, borderRadius: 8,
                        padding: "9px 14px", marginBottom: 12, fontSize: 11, color: C.red,
                        display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap",
                    }}
                >
                    <span>
                        The 12-month return, volatility and correlation figures could not be loaded
                        {metricsQuery.error ? ` — ${metricsQuery.error.message}` : ""}. The model
                        readings below are unaffected.
                    </span>
                    <button
                        type="button"
                        onClick={() => { metricsQuery.refetch(); correlationQuery.refetch(); }}
                        style={{
                            background: "transparent", border: `1px solid ${C.red}66`, borderRadius: 5,
                            color: C.red, padding: "3px 10px", fontSize: 10, cursor: "pointer",
                            fontFamily: "'DM Mono',monospace",
                        }}
                    >Retry</button>
                </div>
            )}

            {/* ── The grid, or its table twin ─────────────────────────────── */}
            <Section style={{ padding: 14 }}>
                {universe.length === 0 ? (
                    <div style={{ padding: "48px 0", textAlign: "center", color: C.textDim, fontSize: 12.5 }}>
                        No stocks to screen. Add a ticker above, or to your watchlist.
                    </div>
                ) : visible.length === 0 ? (
                    <div style={{ padding: "48px 0", textAlign: "center", color: C.textDim, fontSize: 12.5 }}>
                        No stock matches this filter.
                        {filter !== "all" && (
                            <button
                                type="button"
                                onClick={() => setFilter("all")}
                                style={{ background: "none", border: "none", color: C.amber, cursor: "pointer", marginLeft: 6, fontSize: 12.5, fontFamily: "inherit" }}
                            >Show all</button>
                        )}
                    </div>
                ) : view === "grid" ? (
                    <>
                        <div className="hm-grid">
                            {visible.map((row) => (
                                <StockCell
                                    key={row.symbol}
                                    row={row}
                                    mode={mode}
                                    selected={openSymbol === row.symbol}
                                    shortlisted={shortlist.includes(row.symbol)}
                                    onOpen={setOpenSymbol}
                                    onToggle={toggleShortlist}
                                    onHover={setHover}
                                    onLeave={clearHover}
                                />
                            ))}
                        </div>
                        <div style={{ marginTop: 16, paddingTop: 13, borderTop: `1px solid ${C.border}` }}>
                            <ScaleLegend mode={mode} />
                        </div>
                    </>
                ) : (
                    <SignalTable
                        rows={visible}
                        sortKey={sortKey === "auto" ? mode : sortKey}
                        sortDir={sortKey === "auto" ? (MODES[mode].better === "asc" ? "asc" : "desc") : sortDir}
                        onSort={handleSort}
                        onOpen={setOpenSymbol}
                        shortlist={shortlist}
                        onToggle={toggleShortlist}
                    />
                )}
            </Section>

            {/* ── What the grid currently says, in one sentence ───────────── */}
            <div style={{ marginTop: 12, fontSize: 11, color: C.textMid, lineHeight: 1.7 }}>
                Of {rows.length} stocks, the models call{" "}
                <span style={{ color: C.green, fontWeight: 700 }}>{upCount} up</span>,{" "}
                <span style={{ color: C.red, fontWeight: 700 }}>{downCount} down</span>, and withhold a
                direction on <span style={{ color: C.text, fontWeight: 700 }}>{neutralCount}</span>.
                {" "}A withheld call is a real answer: it means no source on that stock has
                demonstrated an edge over its own base rate, or today&rsquo;s conviction sits in the
                bottom third of the model&rsquo;s historical range.
                {correlationData?.high_corr_pairs?.length > 0 && (
                    <> {" "}<span style={{ color: C.amber }}>
                        {correlationData.high_corr_pairs.length} pair
                        {correlationData.high_corr_pairs.length === 1 ? "" : "s"} move almost as one
                        ({correlationData.high_corr_pairs.slice(0, 3).map((p) => `${p.ticker_a}/${p.ticker_b}`).join(", ")})
                        — holding both buys less diversification than the count suggests.
                    </span></>
                )}
            </div>

            {/* ── The exit: a shortlist, handed to the optimizer ──────────── */}
            <div style={{
                position: "sticky", bottom: 14, zIndex: 40, marginTop: 18,
                background: C.bg1, border: `1px solid ${shortlist.length ? `${C.amber}66` : C.border}`,
                borderRadius: 12, padding: "12px 16px",
                display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap",
                boxShadow: "0 10px 30px rgba(4,8,16,.55)",
            }}>
                <div style={{ flex: "1 1 220px", minWidth: 200 }}>
                    <div style={{
                        fontSize: 9.5, letterSpacing: 1.3, textTransform: "uppercase",
                        color: C.textDim, fontFamily: "'Syne',sans-serif", fontWeight: 700, marginBottom: 4,
                    }}>Shortlist for the optimizer</div>
                    <div style={{ fontSize: 11.5, color: shortlist.length ? C.text : C.textDim }}>
                        {shortlist.length
                            ? `${shortlist.join(" · ")} — ${shortlist.length}/${SHORTLIST_MAX}`
                            : "Pick the stocks worth allocating to, using + on a tile."}
                    </div>
                </div>

                <button
                    type="button"
                    onClick={autoPick}
                    title={"Selects stocks the models call UP above Low confidence, strongest first. "
                        + "If no UP call exists anywhere, falls back to stocks with a measured edge "
                        + "pointing above their own base rate — and says which rule it used."}
                    style={{
                        ...inputStyle, cursor: "pointer", color: C.textMid, background: C.bg2,
                        padding: "9px 13px", fontWeight: 700,
                    }}
                >Auto-pick strong</button>

                {shortlist.length > 0 && (
                    <button
                        type="button"
                        onClick={() => setShortlist([])}
                        style={{ ...inputStyle, cursor: "pointer", color: C.textDim, background: "transparent", padding: "9px 13px" }}
                    >Clear</button>
                )}

                <button
                    type="button"
                    disabled={shortlist.length < 2}
                    onClick={() => onOptimize?.(shortlist)}
                    title={shortlist.length < 2
                        ? "The optimizer needs at least two stocks — a single holding has nothing to optimize"
                        : `Build a portfolio from ${shortlist.join(", ")}`}
                    style={{
                        background: shortlist.length >= 2 ? `linear-gradient(135deg, ${C.amber}, #f97316)` : C.bg2,
                        border: shortlist.length >= 2 ? "none" : `1px solid ${C.border}`,
                        color: shortlist.length >= 2 ? "#0b1220" : C.textDim,
                        borderRadius: 9, padding: "11px 18px", fontSize: 12, fontWeight: 800,
                        cursor: shortlist.length >= 2 ? "pointer" : "not-allowed",
                        fontFamily: "'Syne',sans-serif", whiteSpace: "nowrap",
                    }}
                >
                    Optimize {shortlist.length >= 2 ? `these ${shortlist.length}` : "selected"} →
                </button>
            </div>

            <div style={{ marginTop: 10, fontSize: 10, color: C.textDim, lineHeight: 1.65, maxWidth: 780 }}>
                The optimizer solves on 12 months of historical covariance — it does not read the
                direction calls above. Choosing what goes into it is the one place a prediction on
                this page changes a portfolio, which is why the shortlist is yours to set.
            </div>

            {hover && view === "grid" && <HoverCard row={hover.row} rect={hover.rect} />}

            {openRow && (
                <DetailDrawer
                    key={openRow.symbol}
                    row={openRow}
                    onClose={() => setOpenSymbol(null)}
                    onOpenTab={(symbol, tab) => { setOpenSymbol(null); onOpenTicker?.(symbol, tab); }}
                    shortlisted={shortlist.includes(openRow.symbol)}
                    onToggle={toggleShortlist}
                />
            )}
        </div>
    );
}
