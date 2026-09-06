/**
 * Predictions — what the models say about the next bar, and what that is worth.
 *
 * The page answers one question in a fixed order, and the order is the design:
 *
 *     Which stock, which bar? the symbol and the DAY / WEEK / MONTH timeframe
 *     Is there a call?        the measured direction, or an honest refusal
 *     What is the number?     the next-bar price estimate and its interval
 *     What does it look like? candles with the forecast continuing them
 *     Why?                    the seven evidence categories, with weights
 *     Where are the levels?   support and resistance around the estimate
 *     Can I trust it?         the out-of-sample record, on demand
 *
 * The timeframe is a horizon, not a zoom
 * --------------------------------------
 * This control used to read 3M / 6M / 1Y and changed only how many daily
 * candles were drawn; the forecast underneath it was the same number under all
 * three. It now reads DAY / WEEK / MONTH and it changes the question. The
 * server aggregates its daily download into weekly or monthly candles and runs
 * the identical one-step pipeline on those, so "the next bar" becomes the next
 * week or the next month — a real horizon change with no extrapolation
 * anywhere, because a one-step model on weekly bars is a one-week model.
 *
 * All four requests on this page take the timeframe, and that is a correctness
 * requirement rather than tidiness: candles, forecast, direction call and
 * support/resistance must describe the same bar, or the page shows a daily
 * support level under a monthly estimate and reads as a level just broken.
 * The timeframe is likewise part of every query key, because a weekly and a
 * daily forecast for the same symbol are different numbers that share an
 * `as_of` on any Friday.
 *
 * The wording comes from the server's own `bar_noun` and `horizon_label`
 * rather than from the button that was pressed, so a caption can never
 * describe a different bar from the numbers beside it.
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
 * its 0.6. That is true on every timeframe; changing the bar size does not
 * validate anything, and the NOT YET VALIDATED badge stays on all three.
 *
 * One asymmetry the timeframe introduces: the stored classifier that joins the
 * measured stack's blend was fitted and scored against a next-*day* label, so
 * on WEEK and MONTH the server leaves it out and says so in `classifier_note`.
 * Those answers rest on the evidence stack alone — which runs its own
 * walk-forward on the bars it was handed, so they are still measured.
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
import { money, signedPctPoints } from "../utils/format";

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
 * The three timeframes, and everything on the page that depends on which is
 * chosen.
 *
 * This selector used to read 3M / 6M / 1Y and it only ever changed how many
 * daily candles were drawn — the forecast, the direction call and the levels
 * were the same numbers under all three. It is now the horizon control, and the
 * horizon change is real: `key` goes to the server, which aggregates its daily
 * download to weekly or monthly candles and runs the one-step models on those.
 * The next bar is then a week or a month, so "next bar" means what the button
 * says.
 *
 * Everything else here follows from that and exists so no panel can be left
 * describing a different bar size from its neighbour:
 *
 *   `noun`/`nouns`   what one candle is called, everywhere it is named in prose
 *   `chartBars`      how much history to draw — a comparable span in each case
 *                    (six months, two years, ten years), not a comparable count
 *   `interval`       how the support/resistance route spells this bar size, so
 *                    the levels are drawn from the same candles as the chart
 *   `srLookback`     that route's window, in BARS of this timeframe, inside the
 *                    limits it clamps to. The detector reads the last 100
 *                    candles of whatever it is given, so this only has to be
 *                    comfortably above that; the panel captions itself from the
 *                    `bars_analysed` the response reports back.
 *
 * The server is the authority on all of it and echoes `timeframe`, `bar_noun`
 * and `horizon_label` in every response; this table is what the client needs
 * *before* the first response lands.
 */
const TIMEFRAMES = [
    {
        key: "day",
        label: "Day",
        noun: "session",
        nouns: "sessions",
        adjective: "daily",
        headline: "Next-session prediction",
        chartBars: 126,
        interval: "1d",
        srLookback: 180,
        describes: "Daily candles. The models forecast tomorrow's close.",
    },
    {
        key: "week",
        label: "Week",
        noun: "week",
        nouns: "weeks",
        adjective: "weekly",
        headline: "Next-week prediction",
        chartBars: 104,
        interval: "1wk",
        srLookback: 260,
        describes: "Weekly candles. The models forecast next week's close.",
    },
    {
        key: "month",
        label: "Month",
        noun: "month",
        nouns: "months",
        adjective: "monthly",
        headline: "Next-month prediction",
        chartBars: 120,
        interval: "1mo",
        srLookback: 240,
        describes: "Monthly candles. The models forecast next month's close.",
    },
];

const DEFAULT_TIMEFRAME = TIMEFRAMES[0];

function timeframeByKey(key) {
    return TIMEFRAMES.find((entry) => entry.key === key) || DEFAULT_TIMEFRAME;
}

const CHART_HEIGHT = 420;

/** Category order for the evidence table: as the backend lists them, by weight. */
const EVIDENCE_HINT =
    "Each category is scored from this stock's own bars, then weighted by how well " +
    "that category has actually predicted this stock's next day. A category can read " +
    "bullish and still pull the answer down if its history here says it should.";

// `formatPct` takes a value that is ALREADY a percentage — see src/utils/format.js.
const formatPrice = money;
const formatPct = signedPctPoints;

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
   THE TIMEFRAME — the one control that changes the answer
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * DAY / WEEK / MONTH, as one segmented control.
 *
 * Deliberately the most prominent control on the page. It is not a view option:
 * pressing it re-runs three transformers, a seven-category evidence stack and a
 * walk-forward on a different set of candles, and every number below changes
 * meaning with it. The control it replaced (3M / 6M / 1Y) changed nothing but
 * the width of the chart, and sat in the corner accordingly — which is the
 * wrong place for a switch that decides what is being predicted.
 *
 * `aria-pressed` rather than a radio group: three buttons where one is active
 * is what this is, and a screen reader is told which by the same attribute the
 * highlight is drawn from, so the two cannot disagree.
 */
function TimeframeSelector({ value, onChange, busy }) {
    return (
        <div style={{ display: "grid", gap: 7 }}>
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
                Timeframe
            </div>
            <div
                role="group"
                aria-label="Prediction timeframe"
                style={{
                    display: "inline-flex",
                    background: C.bg2,
                    border: `1px solid ${C.border}`,
                    borderRadius: 10,
                    padding: 4,
                    gap: 4,
                }}
            >
                {TIMEFRAMES.map((entry) => {
                    const active = entry.key === value;
                    return (
                        <button
                            key={entry.key}
                            type="button"
                            aria-pressed={active}
                            onClick={() => onChange(entry.key)}
                            title={entry.describes}
                            style={{
                                background: active ? C.amber : "transparent",
                                color: active ? "#10131A" : C.textMid,
                                border: "none",
                                borderRadius: 7,
                                padding: "9px 20px",
                                fontSize: 12.5,
                                fontWeight: 800,
                                letterSpacing: 1.2,
                                textTransform: "uppercase",
                                cursor: "pointer",
                                fontFamily: "'DM Mono',monospace",
                                // The one moving part: a press starts seconds of
                                // model time, and a control that looks inert
                                // until the first panel lands gets pressed twice.
                                opacity: busy && !active ? 0.55 : 1,
                                transition: "background 120ms ease, color 120ms ease",
                            }}
                        >
                            {entry.label}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}

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
function VerdictCard({ analysis, loading, error, symbol, nouns }) {
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
            hint={`The only direction on this page that has been scored out of sample — measured on ${nouns.plural}, for the next ${nouns.noun}.`}
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
                            <>
                                The model declines to call {symbol} for the next {nouns.noun} —{" "}
                                {analysis.neutral_reason}.
                            </>
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
                        hint={`Out-of-sample accuracy of the evidence stack at predicting this symbol's next ${nouns.noun}.`}
                    />
                    <StatCard
                        label="Base rate"
                        value={analysis.base_rate != null ? formatProbability(analysis.base_rate) : "—"}
                        sub={stack.n_test_rows ? `${stack.n_test_rows.toLocaleString()} test ${nouns.plural}` : "—"}
                        color={C.purple}
                        hint={`How often this stock rose over one ${nouns.noun}, unconditionally. A model must beat this to be worth anything.`}
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
function ForecastPanel({ forecast, nouns }) {
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
            title={`Next-${nouns.noun} price estimate`}
            hint={
                `Kronos + Chronos-2 + TimesFM 2.5, combined by inverse variance and run on ` +
                `${nouns.adjective} candles. A price, not a recommendation.`
            }
            right={<Badge color={C.amber}>NOT YET VALIDATED</Badge>}
        >
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <StatCard
                    label={`Last ${nouns.noun} close · ${forecast.as_of || "—"}`}
                    value={formatPrice(forecast.anchor_price)}
                    sub={
                        forecast.last_bar_complete === false
                            ? `the forming ${nouns.noun}, so far`
                            : `the ${nouns.noun} the models read`
                    }
                    color={C.textDim}
                />
                <StatCard
                    label="Current price"
                    value={formatPrice(forecast.current_price)}
                    sub={QUOTE_SOURCE[forecast.current_price_source] || "—"}
                    color={C.textMid}
                />
                <StatCard
                    label={forecast.horizon_label ? `Estimate · ${forecast.horizon_label}` : "Estimate"}
                    value={formatPrice(forecast.forecast_price)}
                    sub={
                        `${formatPct(forecast.expected_change_pct)} from last close` +
                        (forecast.forecast_date ? ` · ${forecast.forecast_date}` : "")
                    }
                    positive={rises}
                    color={C.amber}
                />
                <StatCard
                    label="90% range"
                    value={point ? `${formatPrice(point.lower_90)} – ${formatPrice(point.upper_90)}` : "—"}
                    sub={point ? `68%: ${formatPrice(point.lower_68)} – ${formatPrice(point.upper_68)}` : "no interval"}
                    color={C.cyan}
                    hint={`Where the combined models put 90% (and 68%) of the probability for the next ${nouns.noun}'s close.`}
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
                {formatProbability(forecast.probability_up)} for the next {nouns.noun}, but nothing has yet
                checked whether that number is reliable — so it is shown as the model's own output, not as a
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
                            {forecast.symbol} has only {forecast.bars_available ?? forecast.history_days}{" "}
                            {nouns.plural} on record
                            {forecast.bars_available && forecast.history_days
                                ? ` (${forecast.history_days.toLocaleString()} trading days)`
                                : ""}
                            , short of the 128 Kronos needs for its context — so this estimate comes from{" "}
                            {(forecast.models || []).length} of the 3 models.
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
/**
 * The confirmed pivot zones around the estimate, on the selected timeframe.
 *
 * `analysed` is the server's own `bars_analysed`, not the lookback that was
 * requested: the detector reads the last 100 candles however much history is
 * downloaded, so captioning this panel from the request would advertise a
 * twenty-year monthly window for an eight-year read. Falling back to the
 * requested figure only matters before the first response lands.
 */
function LevelsPanel({ levels, forecastPrice, loading, error, nouns, analysed }) {
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
                    No pivot level cleared the confirmation threshold in the last {analysed} {nouns.plural}.
                    That is a reading, not a gap in the data — this stock has no level the algorithm
                    considers confirmed on this timeframe right now.
                </div>
            </Section>
        );
    }

    return (
        <Section
            title="Support & resistance"
            hint={`Confirmed pivot zones from the last ${analysed} ${nouns.plural}, and where the estimate falls against them.`}
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
function ModelStatusPanel({ forecast, analysis, nouns, selectedTicker }) {
    const [open, setOpen] = useState(false);
    const stack = analysis?.blend?.evidence_stack || {};
    const stackMeta = analysis?.stack?.meta || {};
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
                        ? `direction stack scored on ${(stack.n_test_rows || 0).toLocaleString()} out-of-sample ${nouns.plural}`
                        : "direction stack unavailable"}
                </div>
            ) : (
                <div>
                    <StatusRow label="Timeframe">
                        {forecast?.timeframe_label || "—"} candles · the models forecast the next{" "}
                        {nouns.noun}
                        {forecast?.forecast_date ? `, ending ${forecast.forecast_date}` : ""}
                        {forecast?.last_bar_complete === false && (
                            <span style={{ color: C.amber }}>
                                {" "}· anchored on a {nouns.noun} that is still forming
                            </span>
                        )}
                    </StatusRow>
                    <StatusRow label="Data through">
                        {forecast?.as_of || "—"} · {forecast?.bars_available ?? "—"} {nouns.plural} aggregated
                        from {forecast?.history_days?.toLocaleString() ?? "—"} trading days
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
                        {stackMeta.scale_window != null && (
                            <> · scores scaled against {stackMeta.scale_window} {nouns.plural}</>
                        )}
                    </StatusRow>
                    {/* A difference in what the trend category *is* — six inputs
                        or five — so it is stated rather than left for a reader
                        to discover by comparing two timeframes. Routinely
                        dropped on the monthly frame and on a recent listing,
                        where 200 bars of history do not exist to average. */}
                    {stackMeta.long_trend_leg === false && (
                        <StatusRow label="Trend category">
                            Built from five inputs, not six: {selectedTicker} has too few{" "}
                            {nouns.plural} for a 200-{nouns.noun} moving average to be read on
                            enough of them.
                        </StatusRow>
                    )}
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
    const [timeframeKey, setTimeframeKey] = useState(DEFAULT_TIMEFRAME.key);
    const timeframe = timeframeByKey(timeframeKey);

    // Four requests, deliberately separate, because they cost orders of
    // magnitude apart and each section is worth showing the moment it lands.
    // The candles are a cached download (~0.08s); the forecast is seconds of
    // transformer sampling; the direction analysis is a walk-forward. Bundling
    // them would hold the chart behind the slowest of the three.
    //
    // All four take the timeframe, and they have to: the candles, the forecast,
    // the direction call and the levels must all describe the same bar. Leaving
    // one on daily would put a daily support level under a monthly estimate and
    // read as a level the estimate had just broken.
    const historyQuery = useForecastHistory(selectedTicker, {
        bars: timeframe.chartBars,
        timeframe: timeframe.key,
        enabled: apiConnected,
    });
    const query = useSimpleForecast(selectedTicker, {
        timeframe: timeframe.key,
        enabled: apiConnected,
    });
    const analysisQuery = useDirectionAnalysis(selectedTicker, {
        timeframe: timeframe.key,
        enabled: apiConnected,
    });
    const levelsQuery = useSupportResistance(selectedTicker, {
        interval: timeframe.interval,
        lookback: timeframe.srLookback,
        enabled: apiConnected,
    });

    const data = query.data ?? null;
    const analysis = analysisQuery.data ?? null;
    const loading = query.isPending;
    const error = errorText(query.error, "The forecast could not be loaded.");

    const history = historyQuery.data?.bars ?? [];
    const historyLoading = historyQuery.isPending;
    const historyError = errorText(historyQuery.error, "The price history could not be loaded.");
    const points = data?.forecast ?? [];
    const servable = data?.status === "ok" && points.length > 0;

    // The wording comes from the payload, not from the button.
    //
    // Each timeframe is its own query key, so a switch leaves `data` undefined
    // until the new response lands — the panels show their loading states and
    // the local table fills the gap. Reading the noun off the payload rather
    // than off `timeframeKey` therefore does not change what is on screen
    // today; it guarantees that if it ever does, the caption is the one that
    // belongs to the numbers underneath it. The server is the authority on
    // what bar it answered about, and this is the client agreeing to that.
    const served = timeframeByKey(data?.timeframe || historyQuery.data?.timeframe || timeframe.key);
    const nouns = {
        noun: data?.bar_noun || historyQuery.data?.bar_noun || timeframe.noun,
        plural: data?.bar_noun_plural || historyQuery.data?.bar_noun_plural || timeframe.nouns,
        // "daily"/"weekly"/"monthly" is not derivable from the noun -- the
        // daily one is "session" -- so it is looked up rather than suffixed.
        adjective: served.adjective,
    };
    const switching =
        query.isFetching || historyQuery.isFetching || analysisQuery.isFetching;

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
                        {timeframe.headline}
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

                {/* Stock, then timeframe: the order of the flow the page
                    describes — pick the instrument, pick the bar, read the
                    forecast for that bar. */}
                <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap" }}>
                    {typeof setSelectedTicker === "function" && symbols.length > 1 && (
                        <div style={{ display: "grid", gap: 7 }}>
                            <label
                                htmlFor="prediction-symbol"
                                style={{
                                    color: C.textDim,
                                    fontSize: 10,
                                    fontWeight: 700,
                                    letterSpacing: 1.5,
                                    textTransform: "uppercase",
                                    fontFamily: "'Syne',sans-serif",
                                }}
                            >
                                Stock
                            </label>
                            <select
                                id="prediction-symbol"
                                value={selectedTicker}
                                onChange={(event) => setSelectedTicker(event.target.value)}
                                style={{ ...controlStyle, padding: "10px 12px" }}
                            >
                                {symbols.map((symbol) => (
                                    <option key={symbol} value={symbol}>
                                        {symbol}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}
                    <TimeframeSelector
                        value={timeframe.key}
                        onChange={setTimeframeKey}
                        busy={switching}
                    />
                </div>
            </div>

            {/* What the selection means, in one line, above every panel that
                depends on it. The panels each say which bar they are about, but
                a reader who has just pressed MONTH deserves to be told once,
                plainly, rather than inferring it from four separate captions. */}
            <div
                style={{
                    color: C.textDim,
                    fontSize: 11.5,
                    lineHeight: 1.6,
                    marginTop: -6,
                }}
            >
                {timeframe.describes}
                {data?.status === "ok" && data.last_bar_complete === false && (
                    <span style={{ color: C.amber }}>
                        {" "}The current {nouns.noun} is still forming
                        {data.last_bar_sessions
                            ? ` (${data.last_bar_sessions} session${data.last_bar_sessions === 1 ? "" : "s"} so far)`
                            : ""}
                        , so the estimate is for the {nouns.noun} after it
                        {data.forecast_date ? `, ending ${data.forecast_date}` : ""}.
                    </span>
                )}
            </div>

            {/* ── 1. The scored call, first ── */}
            <VerdictCard
                analysis={analysis}
                loading={analysisQuery.isPending}
                error={errorText(analysisQuery.error, null)}
                symbol={selectedTicker}
                nouns={nouns}
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
                <Section title={`Next-${timeframe.noun} price estimate`}>
                    <div style={{ color: C.textDim, fontSize: 13, padding: "20px 0" }}>
                        Running the forecast models for {selectedTicker} on {timeframe.adjective} candles…
                    </div>
                </Section>
            )}

            {!error && data && !servable && (
                <Notice>
                    {data.message ||
                        `No ${timeframe.adjective} forecast is available for ${selectedTicker} right now.`}
                </Notice>
            )}

            {!error && servable && <ForecastPanel forecast={data} nouns={nouns} />}

            {/* ── 3. The picture ── */}
            <Section title={`Price & forecast — ${selectedTicker} · ${timeframe.label.toUpperCase()}`}>
                {historyLoading && !history.length ? (
                    <div
                        style={{
                            height: CHART_HEIGHT,
                            display: "grid",
                            placeItems: "center",
                            color: C.textDim,
                            fontSize: 13,
                        }}
                    >
                        Loading {timeframe.adjective} candles for {selectedTicker}…
                    </div>
                ) : historyError && !history.length ? (
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
                        bars={history}
                        forecast={points}
                        direction={direction}
                        horizon={1}
                        height={CHART_HEIGHT}
                        barNoun={nouns.noun}
                        barNounPlural={nouns.plural}
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
                nouns={nouns}
                analysed={levelsQuery.data?.bars_analysed ?? timeframe.srLookback}
            />

            {/* ── 6. The audit trail ── */}
            <ModelStatusPanel
                forecast={data}
                analysis={analysis}
                nouns={nouns}
                selectedTicker={selectedTicker}
            />

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
                                    // The server's own label, so a monthly
                                    // estimate cannot be carried into the
                                    // Backtest tab as a next-day call.
                                    horizonLabel: data.horizon_label || "Next 1 Day",
                                    timeframe: data.timeframe || timeframe.key,
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
