/**
 * Predictions — what the models say about the next session, and what that is worth.
 *
 * The page answers one question in a fixed order, and the order is the design:
 *
 *     Is there a call?        the measured direction, or an honest refusal
 *     What is the number?     the next-bar price estimate and its interval
 *     What does it look like? candles with the forecast continuing them
 *     Why?                    the seven evidence categories, with weights
 *     Where are the levels?   support and resistance around the estimate
 *     Can I trust it?         the out-of-sample record, on demand
 *
 * Two models, and they are not interchangeable
 * --------------------------------------------
 * `GET /direction/{sym}/analysis` is the **measured** one. Seven evidence
 * categories are scored off this symbol's own bars, a logistic stack is fitted
 * on them, and the result is blended with the stored classifier in log-odds
 * weighted by each source's out-of-sample Brier skill. A source that scored no
 * better than the base rate gets weight zero; when both do, the answer is the
 * base rate and the direction is NEUTRAL. It knows its own accuracy and the
 * confidence interval around it.
 *
 * `GET /predict/forecast/{sym}` is the **unmeasured** one. Kronos, Chronos-2 and
 * TimesFM 2.5 produce a next-bar price, a 90% interval and a probability. It has
 * never been walk-forwarded — `probability_is_calibrated` is hardcoded false in
 * the payload — so nothing has checked whether its 0.7 comes up more often than
 * its 0.6.
 *
 * They disagree constantly, and that is not a bug to smooth over: across AAPL,
 * MSFT, NVDA, JPM, XOM, KO, TSLA and PLTR the measured stack returned NEUTRAL
 * for every one while the foundation stack returned a confident UP or DOWN for
 * every one. So the measured call leads and the price sits under it, labelled.
 * Showing the foundation arrow as *the* answer — which this tab used to do —
 * presents the only direction nobody has scored as the one to act on.
 *
 * What is deliberately NOT here
 * -----------------------------
 * No "confidence" derived from the foundation probability. No signal strength
 * bar invented from the size of the expected move. No aggregate score across
 * the two models. Every number on this page is one the backend computed and
 * named; where a number has not been validated the page says so beside it
 * rather than in a footnote.
 */

import { useMemo, useState } from "react";

import { C } from "../utils/data";
import {
    useDirectionAnalysis,
    useForecastHistory,
    useSimpleForecast,
    useSupportResistance,
} from "../hooks/useMarketData";
import ForecastOverlayChart from "../components/ForecastOverlayChart";
import { Badge, Hint, Section, StatCard } from "../components/UIComponents";

/**
 * Where "Current Price" came from, in words.
 *
 * An extended-hours quote is the usual reason the current price is nowhere near
 * the close the models read, and a reader who cannot see which of the two they
 * are looking at has no way to tell a moving number from a settled one.
 * `latest_close` means no quote was available at all.
 */
const QUOTE_SOURCE = {
    regular_market: "market hours",
    pre_market: "pre-market",
    post_market: "after hours",
    latest_close: "same as prev close",
};

/**
 * How much history the chart draws. Not a forecast horizon: all three
 * foundation members are built for one step, so the forecast is always the next
 * bar. A 30D button here would promise a number the models never produce.
 */
const RANGES = [
    { label: "3M", bars: 63 },
    { label: "6M", bars: 126 },
    { label: "1Y", bars: 252 },
];

const HISTORY_BARS = RANGES[RANGES.length - 1].bars;
const CHART_HEIGHT = 420;

/** Category order for the evidence table: as the backend lists them, by weight. */
const EVIDENCE_HINT =
    "Each category is scored from this stock's own bars, then weighted by how well " +
    "that category has actually predicted this stock's next day. A category can read " +
    "bullish and still pull the answer down if its history here says it should.";

function formatPrice(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return `$${Number(value).toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatPct(value, digits = 2) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const number = Number(value);
    return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}%`;
}

function formatProbability(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return `${(Number(value) * 100).toFixed(1)}%`;
}

/**
 * The sentence to show for a failed request.
 *
 * Prefers `ApiError.detail` — what the server actually said — over `message`,
 * which wraps it in the HTTP status. A 404 on an unknown ticker and a 422 on a
 * too-short history both have something useful to say to whoever picked the
 * stock; neither is improved by the number in front of it.
 */
function errorText(error, fallback) {
    if (!error) return null;
    return error.detail || error.message || fallback;
}

const controlStyle = {
    background: C.bg2,
    color: C.text,
    border: `1px solid ${C.border}`,
    borderRadius: 8,
    padding: "8px 12px",
    fontSize: 13,
    fontWeight: 700,
    fontFamily: "'DM Mono',monospace",
    outline: "none",
    cursor: "pointer",
};

/* ═══════════════════════════════════════════════════════════════════════════
   THE VERDICT — the measured call, or the measured refusal
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * The headline. Reads the direction the backend actually scored.
 *
 * NEUTRAL is rendered as a first-class answer with its reason, not as a missing
 * result. It is what the stack returns when nothing has beaten its own base
 * rate out of sample, and on this dataset it is the common case — so a layout
 * that treats it as an empty state would be empty most of the time, for the
 * most informative thing the system has to say.
 */
function VerdictCard({ analysis, loading, error, symbol }) {
    if (loading) {
        return (
            <Section title="Direction call">
                <div style={{ color: C.textDim, fontSize: 13, padding: "20px 0" }}>
                    Running the walk-forward for {symbol}…
                </div>
            </Section>
        );
    }

    if (error) {
        return (
            <Section title="Direction call">
                <div style={{ color: C.red, fontSize: 13 }}>{error}</div>
            </Section>
        );
    }

    if (!analysis) return null;

    if (analysis.status !== "ok") {
        return (
            <Section title="Direction call">
                <div style={{ color: C.textMid, fontSize: 13, lineHeight: 1.6 }}>
                    {analysis.message || `No measured direction is available for ${symbol}.`}
                </div>
            </Section>
        );
    }

    const direction = String(analysis.direction || "").toUpperCase();
    const neutral = direction === "NEUTRAL";
    const up = direction === "UP";
    const tone = neutral ? C.textMid : up ? C.green : C.red;

    const stack = analysis.blend?.evidence_stack || {};
    const ci = Array.isArray(stack.accuracy_ci) ? stack.accuracy_ci : null;
    const confidence = analysis.confidence || {};

    return (
        <Section
            title="Direction call"
            hint="The only direction on this page that has been scored out of sample."
            right={<Badge color={neutral ? C.textDim : tone}>{neutral ? "NO CALL" : "CALL MADE"}</Badge>}
        >
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
                <div style={{ flex: "1 1 260px", minWidth: 240 }}>
                    <div
                        style={{
                            color: tone,
                            fontFamily: "'Syne',sans-serif",
                            fontWeight: 800,
                            fontSize: neutral ? 26 : 34,
                            lineHeight: 1.1,
                            letterSpacing: "-.01em",
                        }}
                    >
                        {neutral ? "No directional edge" : `${up ? "▲" : "▼"} ${direction}`}
                    </div>

                    <div style={{ color: C.textMid, fontSize: 12.5, lineHeight: 1.6, marginTop: 10 }}>
                        {neutral ? (
                            <>The model declines to call {symbol} for the next session — {analysis.neutral_reason}.</>
                        ) : (
                            <>
                                P(up) {formatProbability(analysis.probability_up)} · confidence{" "}
                                <strong style={{ color: C.text }}>{confidence.label}</strong>
                                {confidence.basis ? ` (${confidence.basis})` : ""}
                            </>
                        )}
                    </div>
                </div>

                {/* The record behind the call, which is what makes it a call
                    rather than an opinion. Shown next to the verdict rather
                    than buried, because for this dataset it is the reason the
                    verdict is usually NEUTRAL. */}
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", flex: "1 1 320px" }}>
                    <StatCard
                        label="Measured accuracy"
                        value={stack.accuracy != null ? `${(stack.accuracy * 100).toFixed(1)}%` : "—"}
                        sub={ci ? `95% CI ${(ci[0] * 100).toFixed(1)}–${(ci[1] * 100).toFixed(1)}%` : "no interval"}
                        color={C.cyan}
                        hint="Out-of-sample accuracy of the evidence stack on this symbol's own history."
                    />
                    <StatCard
                        label="Base rate"
                        value={analysis.base_rate != null ? formatProbability(analysis.base_rate) : "—"}
                        sub={stack.n_test_rows ? `${stack.n_test_rows.toLocaleString()} test days` : "—"}
                        color={C.purple}
                        hint="How often this stock rose, unconditionally. A model must beat this to be worth anything."
                    />
                </div>
            </div>
        </Section>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   THE PRICE — foundation stack, labelled for what it is
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * The next-bar price estimate, its interval, and the caveat that belongs to it.
 *
 * Every figure is measured against `anchor_price` — the close the models read —
 * and the panel names it, because that close is routinely not the price the
 * reader is looking at. On PLTR the gap was 4%: the models forecast 0.26% BELOW
 * the bar they read, and dividing by the live quote instead printed +3.8%
 * beside a DOWN arrow.
 */
function ForecastPanel({ forecast }) {
    const point = forecast.forecast?.[0] || null;
    const anchor = Number(forecast.anchor_price);
    const quote = Number(forecast.current_price);
    const quoteGapPct =
        Number.isFinite(anchor) && Number.isFinite(quote) && anchor
            ? (quote / anchor - 1) * 100
            : null;

    const rises = Number(forecast.expected_change_pct) >= 0;

    return (
        <Section
            title="Next-session price estimate"
            hint="Kronos + Chronos-2 + TimesFM 2.5, combined by inverse variance. A price, not a recommendation."
            right={<Badge color={C.amber}>NOT YET VALIDATED</Badge>}
        >
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <StatCard
                    label={`Prev close · ${forecast.as_of || "—"}`}
                    value={formatPrice(forecast.anchor_price)}
                    sub="the bar the models read"
                    color={C.textDim}
                />
                <StatCard
                    label="Current price"
                    value={formatPrice(forecast.current_price)}
                    sub={QUOTE_SOURCE[forecast.current_price_source] || "—"}
                    color={C.textMid}
                />
                <StatCard
                    label="Estimate"
                    value={formatPrice(forecast.forecast_price)}
                    sub={`${formatPct(forecast.expected_change_pct)} from prev close`}
                    positive={rises}
                    color={C.amber}
                />
                <StatCard
                    label="90% range"
                    value={point ? `${formatPrice(point.lower_90)} – ${formatPrice(point.upper_90)}` : "—"}
                    sub={point ? `68%: ${formatPrice(point.lower_68)} – ${formatPrice(point.upper_68)}` : "no interval"}
                    color={C.cyan}
                    hint="Where the combined models put 90% (and 68%) of the probability for the next close."
                />
            </div>

            {/* The honesty line. Permanent, because the condition it describes
                is permanent until somebody walk-forwards this stack. */}
            <div
                style={{
                    marginTop: 14,
                    paddingTop: 12,
                    borderTop: `1px solid ${C.border}`,
                    color: C.textDim,
                    fontSize: 11.5,
                    lineHeight: 1.6,
                }}
            >
                These three models have <strong style={{ color: C.textMid }}>not been backtested</strong> on{" "}
                {forecast.symbol}. They report P(up){" "}
                {formatProbability(forecast.probability_up)} for the next bar, but nothing has yet checked
                whether that number is reliable — so it is shown as the model's own output, not as a
                confidence. The scored call is the one above.
                {forecast.split && forecast.split_reason === "quote" && (
                    <>
                        {" "}The live quote sits {formatPct(quoteGapPct)} from the close the models read, in a
                        session none of them saw; against that quote the same estimate reads{" "}
                        {formatPct(forecast.quote_change_pct)}.
                    </>
                )}
                {forecast.split && forecast.split_reason === "heads" && (
                    <>
                        {" "}The combined probability and the combined price point in opposite directions
                        against the same close. Both are shown as they are.
                    </>
                )}
                {forecast.thin_history && (
                    <>
                        {" "}
                        <span style={{ color: C.amber }}>
                            {forecast.symbol} has only {forecast.history_days} trading days on record, so this
                            estimate comes from {(forecast.models || []).length} of the 3 models.
                        </span>
                    </>
                )}
            </div>
        </Section>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   WHY — the seven evidence categories
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * One row per evidence category, with the contribution the fit actually gave it.
 *
 * `contribution_pp` decomposes the evidence stack's own probability, not the
 * blended one — the backend says so in `evidence_note`, and this panel repeats
 * it, because a reader who adds these up against the headline number and finds
 * they do not reconcile will conclude the page is broken rather than that they
 * are two different quantities.
 */
function EvidencePanel({ analysis }) {
    const rows = analysis?.evidence || [];
    if (!rows.length) return null;

    const widest = Math.max(...rows.map((r) => Math.abs(Number(r.contribution_pp) || 0)), 0.01);

    return (
        <Section
            title="Why — the evidence behind the call"
            hint={EVIDENCE_HINT}
            right={<span style={{ color: C.textDim, fontSize: 10.5 }}>{rows.length} categories</span>}
        >
            <div style={{ display: "flex", flexDirection: "column" }}>
                {rows.map((row) => {
                    const pp = Number(row.contribution_pp) || 0;
                    const positive = pp > 0;
                    const width = (Math.abs(pp) / widest) * 50; // half-width each side of centre
                    return (
                        <div
                            key={row.source}
                            style={{
                                display: "grid",
                                gridTemplateColumns: "150px 1fr 150px 72px",
                                gap: 12,
                                alignItems: "center",
                                padding: "9px 0",
                                borderBottom: `1px solid ${C.border}66`,
                            }}
                        >
                            <span style={{ color: C.text, fontSize: 12, fontWeight: 500 }}>{row.label}</span>

                            {/* Diverging bar from a centre line: the sign is the
                                information, so it is encoded as a side rather
                                than only as a colour. */}
                            <div style={{ position: "relative", height: 10, background: C.bg2, borderRadius: 2 }}>
                                <div
                                    style={{
                                        position: "absolute",
                                        left: "50%",
                                        top: 0,
                                        bottom: 0,
                                        width: 1,
                                        background: C.border,
                                    }}
                                />
                                <div
                                    style={{
                                        position: "absolute",
                                        top: 1,
                                        bottom: 1,
                                        [positive ? "left" : "right"]: "50%",
                                        width: `${width}%`,
                                        background: positive ? C.green : C.red,
                                        opacity: 0.75,
                                        borderRadius: 2,
                                    }}
                                />
                            </div>

                            <span style={{ color: C.textMid, fontSize: 11.5 }} title={row.detail || ""}>
                                {row.state}
                            </span>
                            <span
                                style={{
                                    color: positive ? C.green : pp < 0 ? C.red : C.textDim,
                                    fontSize: 12,
                                    textAlign: "right",
                                    fontFamily: "'DM Mono',monospace",
                                }}
                            >
                                {pp >= 0 ? "+" : ""}
                                {pp.toFixed(2)}pp
                            </span>
                        </div>
                    );
                })}
            </div>

            <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.6, marginTop: 12 }}>
                {analysis.evidence_note}
            </div>
        </Section>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   LEVELS — support and resistance around the estimate
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * The pivot levels, and where the forecast sits relative to them.
 *
 * The same endpoint the Technical Analysis tab draws, so a level quoted here is
 * the level the user already saw on that chart. Often only one side exists — a
 * stock mid-range with no confirmed pivot above it has a support and no
 * resistance — so this renders what came back rather than assuming a pair.
 */
function LevelsPanel({ levels, forecastPrice, loading, error }) {
    if (loading) {
        return (
            <Section title="Support & resistance">
                <div style={{ color: C.textDim, fontSize: 12.5 }}>Finding levels…</div>
            </Section>
        );
    }
    if (error) {
        return (
            <Section title="Support & resistance">
                <div style={{ color: C.textMid, fontSize: 12.5 }}>{error}</div>
            </Section>
        );
    }

    const rows = Array.isArray(levels) ? levels : [];
    if (!rows.length) {
        return (
            <Section title="Support & resistance">
                <div style={{ color: C.textMid, fontSize: 12.5, lineHeight: 1.6 }}>
                    No pivot level cleared the confirmation threshold in the last 180 sessions. That is a
                    reading, not a gap in the data — this stock has no level the algorithm considers
                    confirmed right now.
                </div>
            </Section>
        );
    }

    return (
        <Section
            title="Support & resistance"
            hint="Confirmed pivot zones from the last 180 sessions, and where the estimate falls against them."
        >
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {rows.map((level) => {
                    const price = Number(level.price);
                    const isSupport = String(level.type).toLowerCase() === "support";
                    const gapPct =
                        Number.isFinite(forecastPrice) && Number.isFinite(price) && price
                            ? ((forecastPrice - price) / price) * 100
                            : null;
                    return (
                        <StatCard
                            key={`${level.type}-${level.price}`}
                            label={isSupport ? "Support" : "Resistance"}
                            value={formatPrice(price)}
                            sub={
                                gapPct === null
                                    ? `${level.confirmations} touches`
                                    : `estimate is ${formatPct(gapPct, 1)} ${gapPct >= 0 ? "above" : "below"} · ${level.confirmations} touches`
                            }
                            color={isSupport ? C.green : C.red}
                        />
                    );
                })}
            </div>
        </Section>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MODEL STATUS — collapsed, because it is a check rather than a read
   ═══════════════════════════════════════════════════════════════════════════ */

function StatusRow({ label, children }) {
    return (
        <div
            style={{
                display: "grid",
                gridTemplateColumns: "180px 1fr",
                gap: 14,
                padding: "7px 0",
                borderBottom: `1px solid ${C.border}44`,
                fontSize: 11.5,
                alignItems: "baseline",
            }}
        >
            <span style={{ color: C.textDim, textTransform: "uppercase", letterSpacing: 1, fontSize: 10 }}>
                {label}
            </span>
            <span style={{ color: C.textMid, lineHeight: 1.6 }}>{children}</span>
        </div>
    );
}

/**
 * The audit trail: what ran, over what data, and how well it has scored.
 *
 * Collapsed by default. It is the answer to "can I trust this", which is a
 * question asked once rather than on every glance — but it is on the same page
 * as the number, because a track record kept on a different screen from the
 * claim it qualifies is not really disclosed.
 */
function ModelStatusPanel({ forecast, analysis }) {
    const [open, setOpen] = useState(false);
    const stack = analysis?.blend?.evidence_stack || {};
    const classifier = analysis?.blend?.classifier || {};

    return (
        <Section
            title="Model status & track record"
            right={
                <button
                    type="button"
                    onClick={() => setOpen((v) => !v)}
                    style={{ ...controlStyle, padding: "5px 12px", fontSize: 11 }}
                >
                    {open ? "Hide" : "Show"}
                </button>
            }
        >
            {!open ? (
                <div style={{ color: C.textDim, fontSize: 12, lineHeight: 1.6 }}>
                    Data through {forecast?.as_of || "—"} ·{" "}
                    {(forecast?.models || []).length} forecast models ·{" "}
                    {analysis?.status === "ok"
                        ? `direction stack scored on ${(stack.n_test_rows || 0).toLocaleString()} out-of-sample days`
                        : "direction stack unavailable"}
                </div>
            ) : (
                <div>
                    <StatusRow label="Data through">
                        {forecast?.as_of || "—"} · {forecast?.history_days ?? "—"} trading days downloaded
                        {forecast?.thin_history && (
                            <span style={{ color: C.amber }}> · short history</span>
                        )}
                    </StatusRow>
                    <StatusRow label="Forecast models">
                        {(forecast?.models || []).join(" · ") || "—"}
                    </StatusRow>
                    <StatusRow label="Forecast P(up)">
                        {formatProbability(forecast?.probability_up)} —{" "}
                        {forecast?.probability_is_calibrated ? (
                            "calibrated"
                        ) : (
                            <span style={{ color: C.amber }}>not calibrated; no walk-forward has been run</span>
                        )}
                    </StatusRow>
                    <StatusRow label="Evidence stack">
                        Brier skill {stack.brier_skill_score != null ? stack.brier_skill_score.toFixed(4) : "—"} ·
                        accuracy {stack.accuracy != null ? `${(stack.accuracy * 100).toFixed(1)}%` : "—"} ·
                        weight in blend {stack.weight != null ? stack.weight.toFixed(3) : "—"}
                    </StatusRow>
                    <StatusRow label="Classifier">
                        {classifier.model || "—"} ·{" "}
                        {classifier.tradeable === true
                            ? "cleared its ship criteria"
                            : classifier.tradeable === false
                                ? "did not clear its ship criteria"
                                : "no walk-forward report"}
                    </StatusRow>
                    {classifier.gate_reason && (
                        <StatusRow label="Gate reason">{classifier.gate_reason}</StatusRow>
                    )}
                    <StatusRow label="Blend rule">{analysis?.blend?.note || "—"}</StatusRow>
                    <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.6, marginTop: 12 }}>
                        A negative Brier skill means the model's probabilities scored worse than always
                        predicting the historical base rate. Weight zero in the blend is the consequence,
                        not a configuration choice.
                    </div>
                </div>
            )}
        </Section>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   TAB
   ═══════════════════════════════════════════════════════════════════════════ */

function Notice({ tone = "dim", children }) {
    const palette =
        tone === "error"
            ? { color: C.red, border: "rgba(244,63,94,.35)", background: "rgba(244,63,94,.08)" }
            : { color: C.textMid, border: C.border, background: C.bg2 };
    return (
        <div
            style={{
                padding: "14px 18px",
                border: `1px solid ${palette.border}`,
                background: palette.background,
                borderRadius: 10,
                color: palette.color,
                fontSize: 13,
                lineHeight: 1.55,
            }}
        >
            {children}
        </div>
    );
}

export default function PredictionsTab({
    selectedTicker,
    setSelectedTicker,
    watchlist = [],
    apiConnected,
    onBacktest,
    onOptimize,
}) {
    const [bars, setBars] = useState(RANGES[1].bars);

    // Four requests, deliberately separate, because they cost orders of
    // magnitude apart and each section is worth showing the moment it lands.
    // The candles are a cached download (~0.08s); the forecast is seconds of
    // transformer sampling; the direction analysis is a walk-forward. Bundling
    // them would hold the chart behind the slowest of the three.
    const historyQuery = useForecastHistory(selectedTicker, {
        days: HISTORY_BARS,
        enabled: apiConnected,
    });
    const query = useSimpleForecast(selectedTicker, { enabled: apiConnected });
    const analysisQuery = useDirectionAnalysis(selectedTicker, { enabled: apiConnected });
    const levelsQuery = useSupportResistance(selectedTicker, { enabled: apiConnected });

    const data = query.data ?? null;
    const analysis = analysisQuery.data ?? null;
    const loading = query.isPending;
    const error = errorText(query.error, "The forecast could not be loaded.");

    const history = historyQuery.data?.bars ?? [];
    const historyLoading = historyQuery.isPending;
    const historyError = errorText(historyQuery.error, "The price history could not be loaded.");
    const windowed = useMemo(() => history.slice(-bars), [history, bars]);
    const points = data?.forecast ?? [];
    const servable = data?.status === "ok" && points.length > 0;

    // The arrow on the chart carries the SCORED call when there is one, and the
    // probability that belongs to it. When the measured stack returns NEUTRAL
    // the chart gets no arrow at all — drawing the foundation direction there
    // instead would put the unscored call in the most prominent place on the
    // page, which is the thing this layout exists to stop.
    const measured = analysis?.status === "ok" ? analysis : null;
    const scored = measured ? String(measured.direction || "").toUpperCase() : null;
    const scoredProbability = measured ? measured.probability_up : null;
    const direction = useMemo(
        () =>
            scored && scored !== "NEUTRAL"
                ? { direction: scored, probability_up: scoredProbability }
                : null,
        [scored, scoredProbability]
    );

    const symbols = watchlist.includes(selectedTicker)
        ? watchlist
        : [selectedTicker, ...watchlist].filter(Boolean);

    if (!apiConnected) {
        return (
            <div style={{ padding: 48, color: C.textDim, textAlign: "center" }}>
                Connect to the API server to view the forecast.
            </div>
        );
    }

    return (
        <div style={{ display: "grid", gap: 16, paddingBottom: 36 }}>
            {/* ── Header ── */}
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "end",
                    gap: 16,
                    flexWrap: "wrap",
                }}
            >
                <div style={{ display: "grid", gap: 6 }}>
                    <div
                        style={{
                            color: C.textDim,
                            fontSize: 10,
                            fontWeight: 700,
                            letterSpacing: 1.5,
                            textTransform: "uppercase",
                            fontFamily: "'Syne',sans-serif",
                        }}
                    >
                        Next-session prediction
                    </div>
                    <div
                        style={{
                            color: C.text,
                            fontSize: 26,
                            fontWeight: 800,
                            lineHeight: 1,
                            fontFamily: "'Syne',sans-serif",
                        }}
                    >
                        {selectedTicker}
                        {data?.as_of && (
                            <span style={{ color: C.textDim, fontSize: 12, fontWeight: 400, marginLeft: 10 }}>
                                as of {data.as_of}
                            </span>
                        )}
                    </div>
                </div>

                <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                    {typeof setSelectedTicker === "function" && symbols.length > 1 && (
                        <select
                            value={selectedTicker}
                            onChange={(event) => setSelectedTicker(event.target.value)}
                            style={controlStyle}
                            aria-label="Stock"
                        >
                            {symbols.map((symbol) => (
                                <option key={symbol} value={symbol}>
                                    {symbol}
                                </option>
                            ))}
                        </select>
                    )}
                    <div
                        style={{
                            display: "inline-flex",
                            background: C.bg2,
                            border: `1px solid ${C.border}`,
                            borderRadius: 8,
                            padding: 3,
                            gap: 3,
                        }}
                    >
                        {RANGES.map((range) => (
                            <button
                                key={range.label}
                                type="button"
                                onClick={() => setBars(range.bars)}
                                title={`Show ${range.label} of candles`}
                                style={{
                                    background: bars === range.bars ? C.amber : "transparent",
                                    color: bars === range.bars ? "#10131A" : C.textMid,
                                    border: "none",
                                    borderRadius: 6,
                                    padding: "7px 13px",
                                    fontSize: 12,
                                    fontWeight: 800,
                                    cursor: "pointer",
                                    fontFamily: "'DM Mono',monospace",
                                }}
                            >
                                {range.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* ── 1. The scored call, first ── */}
            <VerdictCard
                analysis={analysis}
                loading={analysisQuery.isPending}
                error={errorText(analysisQuery.error, null)}
                symbol={selectedTicker}
            />

            {/* ── 2. The number ── */}
            {error && (
                <Notice tone="error">
                    <div
                        style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            gap: 16,
                            flexWrap: "wrap",
                        }}
                    >
                        <span>{error}</span>
                        <button
                            type="button"
                            onClick={() => query.refetch()}
                            disabled={query.isFetching}
                            style={{
                                background: "transparent",
                                border: `1px solid ${C.red}66`,
                                borderRadius: 6,
                                color: C.red,
                                padding: "5px 12px",
                                fontSize: 12,
                                fontWeight: 800,
                                cursor: query.isFetching ? "default" : "pointer",
                                opacity: query.isFetching ? 0.5 : 1,
                                whiteSpace: "nowrap",
                            }}
                        >
                            {query.isFetching ? "Retrying…" : "Try again"}
                        </button>
                    </div>
                </Notice>
            )}

            {!error && loading && (
                <Section title="Next-session price estimate">
                    <div style={{ color: C.textDim, fontSize: 13, padding: "20px 0" }}>
                        Running the forecast models for {selectedTicker}…
                    </div>
                </Section>
            )}

            {!error && data && !servable && (
                <Notice>{data.message || `No forecast is available for ${selectedTicker} right now.`}</Notice>
            )}

            {!error && servable && <ForecastPanel forecast={data} />}

            {/* ── 3. The picture ── */}
            <Section title={`Price & forecast — ${selectedTicker}`}>
                {historyLoading && !windowed.length ? (
                    <div
                        style={{
                            height: CHART_HEIGHT,
                            display: "grid",
                            placeItems: "center",
                            color: C.textDim,
                            fontSize: 13,
                        }}
                    >
                        Loading {selectedTicker}…
                    </div>
                ) : historyError && !windowed.length ? (
                    <div
                        style={{
                            height: CHART_HEIGHT,
                            display: "grid",
                            placeItems: "center",
                            color: C.red,
                            fontSize: 13,
                        }}
                    >
                        {historyError}
                    </div>
                ) : (
                    <ForecastOverlayChart
                        bars={windowed}
                        forecast={points}
                        direction={direction}
                        horizon={1}
                        height={CHART_HEIGHT}
                    />
                )}
            </Section>

            {/* ── 4. Why ── */}
            {analysis?.status === "ok" && <EvidencePanel analysis={analysis} />}

            {/* ── 5. Levels ── */}
            <LevelsPanel
                levels={levelsQuery.data?.levels}
                forecastPrice={Number(data?.forecast_price)}
                loading={levelsQuery.isPending}
                error={errorText(levelsQuery.error, null)}
            />

            {/* ── 6. The audit trail ── */}
            <ModelStatusPanel forecast={data} analysis={analysis} />

            {/* ── 7. Where this goes next ──
                Two exits, matching the two things a reader can do with a
                prediction: check it against history, or size it in a portfolio.
                Both are shown only once there is something to carry. */}
            {!error && servable && (
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    {typeof onBacktest === "function" && (
                        <button
                            type="button"
                            onClick={() =>
                                onBacktest({
                                    symbol: selectedTicker,
                                    direction: scored && scored !== "NEUTRAL" ? scored : data.direction,
                                    forecastPrice: data.forecast_price,
                                    anchorPrice: data.anchor_price,
                                    expectedChangePct: data.expected_change_pct,
                                    asOf: data.as_of,
                                    horizonLabel: data.horizon_label || "Next 1 Day",
                                    models: data.models || [],
                                })
                            }
                            style={{
                                flex: "1 1 240px",
                                background: "transparent",
                                border: `1px solid ${C.amber}55`,
                                borderRadius: 10,
                                color: C.amber,
                                padding: "14px 18px",
                                fontSize: 13,
                                fontWeight: 800,
                                cursor: "pointer",
                                fontFamily: "'Syne',sans-serif",
                            }}
                        >
                            Backtest this prediction →
                        </button>
                    )}
                    {typeof onOptimize === "function" && (
                        <button
                            type="button"
                            onClick={() => onOptimize(selectedTicker)}
                            title="Add this stock to the portfolio optimizer's selection"
                            style={{
                                flex: "1 1 240px",
                                background: "transparent",
                                border: `1px solid ${C.cyan}55`,
                                borderRadius: 10,
                                color: C.cyan,
                                padding: "14px 18px",
                                fontSize: 13,
                                fontWeight: 800,
                                cursor: "pointer",
                                fontFamily: "'Syne',sans-serif",
                            }}
                        >
                            Add {selectedTicker} to portfolio →
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}
