/**
 * AI Direction Analysis — the panel that sits under the TradingView chart.
 *
 * TradingView draws the market. This draws what our backend concluded from it,
 * and nothing here computes anything: every number, label and sentence comes
 * from `GET /api/direction/{symbol}/analysis`. That boundary is the point of the
 * redesign, so it is worth stating plainly — a threshold invented in this file
 * would be a model decision hidden in a stylesheet.
 *
 * What it deliberately does *not* draw is a forecast path. The old panel drew a
 * line from today's price to a predicted one, which implies a candle-by-candle
 * claim no daily classifier can support. What is drawn instead is a probability
 * split, a confidence label, the range past analogs actually produced, and the
 * evidence that moved the answer — ranked by how much it moved it.
 *
 * The three states this renders, in order of how often they should appear:
 *
 *   NEUTRAL   the model has no measurable edge, or today's conviction is in the
 *             bottom third of its own range. The arrow is grey and the reason is
 *             printed. This is the honest answer far more often than a dashboard
 *             usually admits, and it is not treated as a failure to render.
 *   UP/DOWN   with the probability split and the confidence label beside it.
 *   preparing the classifier's walk-forward run is under way; the evidence stack
 *             already has an answer and shows it.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { fetchDirectionAnalysis } from "../utils/api";
import { C } from "../utils/data";
import { Badge, Hint, Section } from "./UIComponents";

const pct = (value, digits = 0) =>
    typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
const money = (value) =>
    typeof value === "number" && Number.isFinite(value)
        ? `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : "—";
const signedPp = (value) =>
    typeof value === "number" && Number.isFinite(value)
        ? `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(1)}pp`
        : "—";

const DIRECTION_STYLE = {
    UP: { colour: C.green, arrow: "↑", word: "UP" },
    DOWN: { colour: C.red, arrow: "↓", word: "DOWN" },
    NEUTRAL: { colour: C.textMid, arrow: "→", word: "NEUTRAL" },
};

const CONFIDENCE_COLOUR = { High: C.green, Moderate: C.amber, Low: C.textDim };

/* ─── Probability split ──────────────────────────────────────── */
/**
 * One bar, two shares. Not a gauge with a needle: a needle at 68% on a 0-100
 * dial reads as "68% of the way to certain", where the number actually means
 * "up a bit more often than down". Two adjacent shares of one bar say that.
 */
function ProbabilitySplit({ up, down, direction }) {
    const upPercent = Math.max(0, Math.min(100, (up ?? 0.5) * 100));
    const decided = direction !== "NEUTRAL";

    return (
        <div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 6 }}>
                <span style={{ color: decided ? C.green : C.textMid, fontWeight: 700 }}>
                    UP {pct(up, 1)}
                </span>
                <span style={{ color: decided ? C.red : C.textMid, fontWeight: 700 }}>
                    DOWN {pct(down, 1)}
                </span>
            </div>
            <div style={{
                display: "flex", height: 12, borderRadius: 6, overflow: "hidden",
                background: C.bg3, border: `1px solid ${C.border}`,
            }}>
                <div style={{
                    width: `${upPercent}%`,
                    background: decided ? C.green : C.textDim,
                    opacity: decided ? 0.85 : 0.4,
                    transition: "width .35s ease",
                }} />
                <div style={{
                    width: `${100 - upPercent}%`,
                    background: decided ? C.red : C.textDim,
                    opacity: decided ? 0.85 : 0.25,
                    transition: "width .35s ease",
                }} />
            </div>
            {/* The 50% mark, so a 51/49 split is visibly a coin flip rather than
                a bar that happens to be slightly longer on one side. */}
            <div style={{ position: "relative", height: 12, marginTop: 2 }}>
                <div style={{
                    position: "absolute", left: "50%", top: -16, width: 1, height: 16,
                    background: C.text, opacity: 0.5,
                }} />
                <span style={{ position: "absolute", left: "50%", transform: "translateX(-50%)", fontSize: 9, color: C.textDim }}>
                    50%
                </span>
            </div>
        </div>
    );
}

/* ─── One evidence row ───────────────────────────────────────── */
function EvidenceRow({ row, maxContribution }) {
    const leansUp = row.contribution_pp > 0;
    const colour = row.contribution_pp === 0 ? C.textDim : leansUp ? C.green : C.red;
    // Bars are scaled against the largest contribution on screen, so the ranking
    // stays readable when every category is worth a fraction of a point.
    const width = maxContribution > 0
        ? Math.min(100, (Math.abs(row.contribution_pp) / maxContribution) * 100)
        : 0;

    return (
        <div style={{ padding: "8px 0", borderBottom: `1px solid ${C.border}` }}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10 }}>
                <span style={{ color: C.textMid, fontSize: 11, minWidth: 138 }}>{row.label}</span>
                <span style={{ color: C.text, fontSize: 11, flex: 1 }}>{row.state}</span>
                <span style={{ color: colour, fontSize: 11, fontWeight: 700, minWidth: 58, textAlign: "right" }}>
                    {signedPp(row.contribution_pp)}
                </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 5 }}>
                <div style={{ flex: 1, height: 4, background: C.bg3, borderRadius: 2, overflow: "hidden", display: "flex", justifyContent: leansUp ? "flex-start" : "flex-end" }}>
                    <div style={{ width: `${width}%`, height: "100%", background: colour, opacity: 0.75 }} />
                </div>
            </div>
            {row.detail && (
                <div style={{ color: C.textDim, fontSize: 10, marginTop: 4, lineHeight: 1.5 }}>{row.detail}</div>
            )}
        </div>
    );
}

/* ─── Multi-horizon strip ────────────────────────────────────── */
function HorizonStrip({ horizons }) {
    if (!horizons) return null;
    const rows = [
        ["short", "Short-term"],
        ["medium", "Medium-term"],
        ["long", "Long-term"],
    ];
    const agreementColour = {
        aligned: C.green,
        "partly aligned": C.amber,
        conflicting: C.red,
        undecided: C.textDim,
    }[horizons.agreement] || C.textDim;

    return (
        <div>
            <div style={{ display: "grid", gap: 6 }}>
                {rows.map(([key, label]) => {
                    const read = horizons[key];
                    if (!read?.available) return null;
                    const style = DIRECTION_STYLE[read.direction] || DIRECTION_STYLE.NEUTRAL;
                    return (
                        <div key={key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                            <span style={{ color: C.textMid, fontSize: 11 }}>{label}</span>
                            <span style={{ color: C.textDim, fontSize: 10, flex: 1 }}>
                                {read.description} · {read.window} bars
                            </span>
                            <span style={{ color: style.colour, fontSize: 11, fontWeight: 700 }}>
                                {style.arrow} {style.word}
                            </span>
                        </div>
                    );
                })}
            </div>
            <div style={{ marginTop: 10, paddingTop: 8, borderTop: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", fontSize: 10 }}>
                <span style={{ color: C.textDim }}>Agreement</span>
                <span style={{ color: agreementColour, fontWeight: 700, textTransform: "capitalize" }}>
                    {horizons.agreement}
                </span>
            </div>
        </div>
    );
}

/* ─── Expected range ─────────────────────────────────────────── */
/**
 * A spread, drawn as a spread.
 *
 * The last close is a dot; the range is the band it sits in. There is
 * deliberately no line leaving the dot: the model has a claim about where price
 * ends up, not about the path it takes to get there, and drawing one would be
 * the exact fiction this panel replaced.
 */
function ExpectedRange({ range, lastClose }) {
    if (!range?.available) {
        return (
            <div style={{ color: C.textDim, fontSize: 11 }}>
                {range?.reason || "No comparable historical sample for a range."}
            </div>
        );
    }

    const { price_low: low, price_high: high, price_median: median } = range;
    const span = high - low;
    const position = (value) => (span > 0 ? Math.max(0, Math.min(100, ((value - low) / span) * 100)) : 50);

    return (
        <div>
            <div style={{ position: "relative", height: 46, marginTop: 6 }}>
                <div style={{
                    position: "absolute", top: 18, left: 0, right: 0, height: 10,
                    background: `linear-gradient(90deg, ${C.red}33, ${C.bg3}, ${C.green}33)`,
                    borderRadius: 5, border: `1px solid ${C.border}`,
                }} />
                {Number.isFinite(median) && (
                    <div title={`Median outcome ${money(median)}`} style={{
                        position: "absolute", top: 14, left: `${position(median)}%`,
                        width: 2, height: 18, background: C.textMid,
                    }} />
                )}
                {Number.isFinite(lastClose) && (
                    <div title={`Last close ${money(lastClose)}`} style={{
                        position: "absolute", top: 17, left: `${position(lastClose)}%`,
                        width: 12, height: 12, marginLeft: -6, borderRadius: "50%",
                        background: C.amber, border: `2px solid ${C.bg1}`,
                    }} />
                )}
                <span style={{ position: "absolute", top: 32, left: 0, fontSize: 10, color: C.red }}>{money(low)}</span>
                <span style={{ position: "absolute", top: 32, right: 0, fontSize: 10, color: C.green }}>{money(high)}</span>
                <span style={{ position: "absolute", top: 0, left: 0, fontSize: 10, color: C.textDim }}>10th pct</span>
                <span style={{ position: "absolute", top: 0, right: 0, fontSize: 10, color: C.textDim }}>90th pct</span>
            </div>
            <div style={{ color: C.textDim, fontSize: 10, marginTop: 8, lineHeight: 1.5 }}>
                {range.basis}, over {range.n_samples} matched setups. The dot is the last close;
                there is no forecast path here because the model does not have one.
            </div>
        </div>
    );
}

/* ─── Provenance ─────────────────────────────────────────────── */
/** How the two sources were weighted, and what each has actually scored. */
function BlendNote({ blend, evaluation }) {
    if (!blend) return null;
    const stack = blend.evidence_stack || {};
    const classifier = blend.classifier || {};

    const row = (label, weight, detail) => (
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, padding: "3px 0" }}>
            <span style={{ color: C.textMid, fontSize: 10, minWidth: 118 }}>{label}</span>
            <span style={{ color: C.textDim, fontSize: 10, flex: 1 }}>{detail}</span>
            <span style={{ color: weight > 0 ? C.amber : C.textDim, fontSize: 10, fontWeight: 700 }}>
                weight {typeof weight === "number" ? weight.toFixed(3) : "—"}
            </span>
        </div>
    );

    return (
        <div style={{ background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px" }}>
            {row(
                "Evidence stack",
                stack.weight,
                stack.n_test_rows
                    ? `P(up) ${pct(stack.probability_up, 1)} · ${pct(stack.accuracy, 1)} accurate over ${stack.n_test_rows} out-of-sample days` +
                      (stack.accuracy_ci ? ` (95% CI ${pct(stack.accuracy_ci[0], 1)}–${pct(stack.accuracy_ci[1], 1)})` : "")
                    : `P(up) ${pct(stack.probability_up, 1)} · ${evaluation?.reason || "no walk-forward record yet"}`,
            )}
            {row(
                classifier.model ? `Classifier (${classifier.model})` : "Classifier",
                classifier.weight,
                classifier.probability_up != null
                    ? `P(up) ${pct(classifier.probability_up, 1)}${classifier.tradeable === false ? ` · gated: ${classifier.gate_reason}` : ""}`
                    : "not in the blend",
            )}
            <div style={{ color: C.textDim, fontSize: 10, marginTop: 8, paddingTop: 8, borderTop: `1px solid ${C.border}`, lineHeight: 1.5 }}>
                {blend.note}
            </div>
        </div>
    );
}

/* ─── Panel ──────────────────────────────────────────────────── */
export default function DirectionAnalysisPanel({ symbol, apiConnected = true, model = "logistic" }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    // Requests are not aborted, they are ignored: switching tickers quickly must
    // not let a slow answer for the previous one land in the current panel.
    const latestRequest = useRef(0);

    const load = useCallback(async ({ refresh = false } = {}) => {
        if (!symbol || !apiConnected) return;
        const token = latestRequest.current + 1;
        latestRequest.current = token;
        setLoading(true);
        setError(null);
        try {
            const payload = await fetchDirectionAnalysis(symbol, { model, refresh });
            if (latestRequest.current === token) setData(payload);
        } catch (err) {
            if (latestRequest.current === token) {
                setData(null);
                setError(err?.message || "Analysis request failed.");
            }
        } finally {
            if (latestRequest.current === token) setLoading(false);
        }
    }, [symbol, apiConnected, model]);

    // `load` is memoised on exactly the inputs that change the answer, so this
    // fires on a ticker or model change and on nothing else.
    useEffect(() => { load(); }, [load]);

    const title = `AI DIRECTION ANALYSIS — ${String(symbol || "").toUpperCase()}`;

    if (!apiConnected) {
        return (
            <Section title={title}>
                <div style={{ color: C.textDim, fontSize: 12 }}>
                    Connect to the API server to run the direction analysis.
                </div>
            </Section>
        );
    }

    if (loading && !data) {
        return (
            <Section title={title}>
                <div style={{ color: C.textDim, fontSize: 12, padding: "18px 0", textAlign: "center" }}>
                    Reading the chart: indicators, price action, historical analogs, models…
                </div>
            </Section>
        );
    }

    if (error) {
        return (
            <Section title={title}>
                <div style={{ color: C.red, fontSize: 12, display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
                    <span>{error}</span>
                    <button
                        onClick={() => load({ refresh: true })}
                        style={{
                            background: "transparent", border: `1px solid ${C.red}66`, borderRadius: 4,
                            color: C.red, fontSize: 10, padding: "3px 10px", cursor: "pointer",
                            fontFamily: "'DM Mono',monospace",
                        }}
                    >Retry</button>
                </div>
            </Section>
        );
    }

    if (!data) return null;

    if (data.status === "unavailable") {
        return (
            <Section title={title}>
                <div style={{ color: C.textMid, fontSize: 12, lineHeight: 1.6 }}>
                    {data.message}
                    <div style={{ color: C.textDim, fontSize: 11, marginTop: 6 }}>
                        {data.data?.clean_rows} sessions available
                        {data.data?.first_bar ? ` from ${data.data.first_bar}` : ""}.
                    </div>
                </div>
            </Section>
        );
    }

    const style = DIRECTION_STYLE[data.direction] || DIRECTION_STYLE.NEUTRAL;
    const confidenceColour = CONFIDENCE_COLOUR[data.confidence?.label] || C.textDim;
    const maxContribution = Math.max(
        ...(data.evidence || []).map((row) => Math.abs(row.contribution_pp || 0)),
        0,
    );

    return (
        <Section
            title={title}
            hint="Direction, probability and confidence computed by our backend from this symbol's own history. TradingView draws the chart; it does not produce any number on this panel."
            right={
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ color: C.textDim, fontSize: 10 }}>
                        as of {data.as_of}{data.cached ? " · cached" : ""}
                    </span>
                    <button
                        onClick={() => load({ refresh: true })}
                        disabled={loading}
                        style={{
                            background: "transparent", border: `1px solid ${C.border}`, borderRadius: 4,
                            color: C.textMid, fontSize: 10, padding: "3px 10px",
                            cursor: loading ? "wait" : "pointer", fontFamily: "'DM Mono',monospace",
                        }}
                    >{loading ? "…" : "Recompute"}</button>
                </div>
            }
        >
            {/* Headline: direction, split, confidence */}
            <div style={{ display: "grid", gridTemplateColumns: "minmax(180px, 240px) 1fr", gap: 20, alignItems: "center" }}>
                <div>
                    <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1.5 }}>DIRECTION</div>
                    <div style={{
                        color: style.colour, fontSize: 40, fontWeight: 800, lineHeight: 1.1,
                        fontFamily: "'Syne',sans-serif",
                    }}>
                        {style.arrow} {style.word}
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
                        <span style={{ color: C.textDim, fontSize: 10 }}>Confidence</span>
                        <Badge color={confidenceColour}>{data.confidence?.label || "—"}</Badge>
                    </div>
                    <div style={{ color: C.textDim, fontSize: 10, marginTop: 6, lineHeight: 1.5 }}>
                        {data.confidence?.basis}
                    </div>
                </div>

                <div>
                    <ProbabilitySplit up={data.probability_up} down={data.probability_down} direction={data.direction} />
                    {data.neutral_reason && (
                        <div style={{
                            marginTop: 14, padding: "8px 12px", borderRadius: 6,
                            background: C.amberLow, border: `1px solid ${C.amber}33`,
                            color: C.textMid, fontSize: 11, lineHeight: 1.6,
                        }}>
                            <b style={{ color: C.amber }}>No directional call.</b> {data.neutral_reason}.
                            The split above is still the model's best read; it is simply not one to act on.
                        </div>
                    )}
                    {data.base_rate != null && (
                        <div style={{ color: C.textDim, fontSize: 10, marginTop: 8 }}>
                            This symbol has risen on {pct(data.base_rate, 1)} of its sessions — the number the
                            probability above has to beat to mean anything.
                        </div>
                    )}
                </div>
            </div>

            {/* Evidence + horizons */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 18, marginTop: 20 }}>
                <div>
                    <div style={{ display: "flex", alignItems: "center", color: C.textMid, fontSize: 10, letterSpacing: 1.5, marginBottom: 4 }}>
                        EVIDENCE
                        <Hint text="Each category's effect on the answer, in percentage points: the probability with that evidence minus the probability without it. Ranked by size, not by category." />
                    </div>
                    {(data.evidence || []).map((row) => (
                        <EvidenceRow key={row.source} row={row} maxContribution={maxContribution} />
                    ))}
                    {data.evidence_note && (
                        <div style={{ color: C.textDim, fontSize: 10, marginTop: 8, lineHeight: 1.5 }}>
                            {data.evidence_note}.
                        </div>
                    )}
                </div>

                <div style={{ display: "grid", gap: 18, alignContent: "start" }}>
                    <div>
                        <div style={{ display: "flex", alignItems: "center", color: C.textMid, fontSize: 10, letterSpacing: 1.5, marginBottom: 10 }}>
                            TIME HORIZONS
                            <Hint text="The same chart read at three lookbacks. A direction is only called when the read clears half a standard deviation of its own history, so a quiet symbol is not read as trending on a move a volatile one would ignore." />
                        </div>
                        <HorizonStrip horizons={data.horizons} />
                    </div>

                    <div>
                        <div style={{ display: "flex", alignItems: "center", color: C.textMid, fontSize: 10, letterSpacing: 1.5 }}>
                            PRICE ACTION
                            <Hint text="Swing structure and the events at the levels, read off the candles. Higher highs with higher lows is bullish structure; lower highs with lower lows is bearish." />
                        </div>
                        {data.price_action?.available ? (
                            <div style={{ marginTop: 8 }}>
                                <div style={{ color: C.text, fontSize: 12 }}>
                                    {data.price_action.structure_label}
                                </div>
                                {(data.price_action.events || []).length > 0 && (
                                    <ul style={{ margin: "6px 0 0", paddingLeft: 16, color: C.textMid, fontSize: 11, lineHeight: 1.7 }}>
                                        {data.price_action.events.map((event) => <li key={event}>{event}</li>)}
                                    </ul>
                                )}
                                <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 10, color: C.textDim }}>
                                    <span>Support <b style={{ color: C.green }}>{money(data.price_action.levels?.support)}</b></span>
                                    <span>Resistance <b style={{ color: C.red }}>{money(data.price_action.levels?.resistance)}</b></span>
                                </div>
                            </div>
                        ) : (
                            <div style={{ color: C.textDim, fontSize: 11, marginTop: 6 }}>
                                {data.price_action?.reason || "No structure read available."}
                            </div>
                        )}
                    </div>

                    <div>
                        <div style={{ display: "flex", alignItems: "center", color: C.textMid, fontSize: 10, letterSpacing: 1.5 }}>
                            EXPECTED RANGE
                            <Hint text="Where the most similar historical setups ended up, priced off the last close. A spread, not a path — the model has no claim about the route." />
                        </div>
                        <ExpectedRange range={data.expected_range} lastClose={data.last_close} />
                    </div>
                </div>
            </div>

            {/* Reasoning */}
            {(data.reasoning || []).length > 0 && (
                <div style={{ marginTop: 18 }}>
                    <div style={{ color: C.textMid, fontSize: 10, letterSpacing: 1.5, marginBottom: 8 }}>REASONING</div>
                    <ul style={{ margin: 0, paddingLeft: 18, color: C.textMid, fontSize: 11, lineHeight: 1.9 }}>
                        {data.reasoning.map((line) => <li key={line}>{line}</li>)}
                    </ul>
                </div>
            )}

            {/* How the answer was assembled */}
            <div style={{ marginTop: 18 }}>
                <div style={{ color: C.textMid, fontSize: 10, letterSpacing: 1.5, marginBottom: 8 }}>HOW THIS WAS COMBINED</div>
                <BlendNote blend={data.blend} evaluation={data.evaluation} />
                {data.classifier_note && (
                    <div style={{ color: C.textDim, fontSize: 10, marginTop: 8, lineHeight: 1.5 }}>
                        {data.classifier_note}
                        {data.preparation?.status === "running" || data.preparation?.status === "queued"
                            ? " — a walk-forward run is under way; it will join the blend once it lands."
                            : ""}
                    </div>
                )}
            </div>
        </Section>
    );
}
