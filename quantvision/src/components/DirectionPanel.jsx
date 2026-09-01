/**
 * Next-day direction panel.
 *
 * Replaces the 30-day interpolated forecast line with three things that can
 * actually be read:
 *
 *   1. A P(up tomorrow) gauge, shown *inside* the evaluation that says whether
 *      it means anything. When the walk-forward verdict is "do not ship", the
 *      dial is drawn grey and struck through with the reason. A confident-looking
 *      needle over a model that loses to a coin is the exact failure the whole
 *      pipeline exists to remove, so the gate is not a footnote here.
 *   2. A rolling 60-day hit-rate strip, drawn against the rolling base rate in
 *      the same window. Hit rate alone is unreadable: 54% looks like skill until
 *      you see the market rose 55% of the time over those same sessions.
 *   3. An equity curve against buy & hold on the same bars and the same cost
 *      model, because "made money" is only interesting relative to not trading.
 *
 * Every accuracy number rendered here carries its confidence interval. The
 * server supplies the interval; this component never prints the point estimate
 * on its own.
 */

import { useCallback, useEffect, useState } from "react";
import {
    Area,
    CartesianGrid,
    ComposedChart,
    Line,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

import { ApiError, fetchDirection } from "../utils/api";
import { C } from "../utils/data";
import { Badge, Section } from "./UIComponents";
import ModelPreparation from "./ModelPreparation";

const pct = (value, digits = 1) =>
    typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
const num = (value, digits = 3) =>
    typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
const money = (value) =>
    typeof value === "number" && Number.isFinite(value)
        ? `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : "—";
const signedPct = (value, digits = 2) =>
    typeof value === "number" && Number.isFinite(value)
        ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`
        : "—";

/* ─── P(up tomorrow) gauge ───────────────────────────────────── */
/**
 * A 180° arc from 0.40 to 0.60. The range is deliberately narrow: daily
 * direction probabilities from any honest model live within a few points of the
 * base rate, and a 0-to-1 dial would render every prediction as a needle
 * pinned to the middle, implying a precision the model does not have.
 */
function ProbabilityGauge({ probability, baseRate, tradeable }) {
    const LOW = 0.4;
    const HIGH = 0.6;
    const clamp = (v) => Math.min(Math.max(v, LOW), HIGH);
    const toAngle = (v) => 180 - ((clamp(v) - LOW) / (HIGH - LOW)) * 180;

    const radius = 78;
    const cx = 100;
    const cy = 96;
    const point = (angle, r = radius) => [
        cx + r * Math.cos((angle * Math.PI) / 180),
        cy - r * Math.sin((angle * Math.PI) / 180),
    ];

    const [startX, startY] = point(180);
    const [endX, endY] = point(0);
    const needleAngle = toAngle(probability);
    const [needleX, needleY] = point(needleAngle, radius - 10);
    const [baseX, baseY] = point(toAngle(baseRate), radius);
    const [baseInnerX, baseInnerY] = point(toAngle(baseRate), radius - 16);

    const active = tradeable ? C.green : C.textDim;
    const isUp = probability >= 0.5;

    return (
        <svg viewBox="0 0 200 122" style={{ width: "100%", maxWidth: 260 }} role="img"
             aria-label={`Probability of an up move tomorrow: ${pct(probability)}`}>
            {/* Track */}
            <path
                d={`M ${startX} ${startY} A ${radius} ${radius} 0 0 1 ${endX} ${endY}`}
                fill="none" stroke={C.bg3} strokeWidth={14} strokeLinecap="round"
            />
            {/* Down half / up half, so the 0.50 midpoint is visible without a label */}
            <path
                d={`M ${startX} ${startY} A ${radius} ${radius} 0 0 1 ${point(90)[0]} ${point(90)[1]}`}
                fill="none" stroke={tradeable ? C.red : C.bg3} strokeWidth={14}
                strokeLinecap="round" opacity={tradeable ? 0.35 : 0.6}
            />
            <path
                d={`M ${point(90)[0]} ${point(90)[1]} A ${radius} ${radius} 0 0 1 ${endX} ${endY}`}
                fill="none" stroke={tradeable ? C.green : C.bg3} strokeWidth={14}
                strokeLinecap="round" opacity={tradeable ? 0.35 : 0.6}
            />
            {/* The training base rate: the number the model has to beat */}
            <line x1={baseX} y1={baseY} x2={baseInnerX} y2={baseInnerY}
                  stroke={C.amber} strokeWidth={2} />
            {/* Needle */}
            <line x1={cx} y1={cy} x2={needleX} y2={needleY}
                  stroke={active} strokeWidth={3} strokeLinecap="round" />
            <circle cx={cx} cy={cy} r={5} fill={active} />

            <text x={cx} y={62} textAnchor="middle" fill={active}
                  style={{ fontSize: 26, fontFamily: "'DM Mono',monospace", fontWeight: 700 }}>
                {pct(probability)}
            </text>
            <text x={cx} y={78} textAnchor="middle" fill={C.textDim}
                  style={{ fontSize: 9, letterSpacing: 1.4, fontFamily: "'Syne',sans-serif" }}>
                P(UP TOMORROW) · {isUp ? "UP" : "DOWN"}
            </text>
            <text x={12} y={116} fill={C.textDim} style={{ fontSize: 9, fontFamily: "'DM Mono',monospace" }}>
                {pct(LOW, 0)}
            </text>
            <text x={188} y={116} textAnchor="end" fill={C.textDim}
                  style={{ fontSize: 9, fontFamily: "'DM Mono',monospace" }}>
                {pct(HIGH, 0)}
            </text>
        </svg>
    );
}

/* ─── Tooltip shared by both charts ──────────────────────────── */
function DirectionTooltip({ active, payload, label, format }) {
    if (!active || !payload?.length) return null;
    return (
        <div className="tooltip-card">
            <div style={{ color: C.amber, marginBottom: 4, fontSize: 10, letterSpacing: 1 }}>{label}</div>
            {payload.map((entry, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", gap: 16, color: entry.color }}>
                    <span style={{ color: C.textMid }}>{entry.name}</span>
                    <span>{format(entry.value)}</span>
                </div>
            ))}
        </div>
    );
}

/* ─── Verdict banner ─────────────────────────────────────────── */
function VerdictBanner({ verdict, evaluation }) {
    if (!verdict) return null;
    const ship = Boolean(verdict.ship);
    const colour = ship ? C.green : C.amber;
    const labels = {
        beats_best_baseline_accuracy: "beats best baseline",
        accuracy_edge_is_significant: "edge is significant",
        positive_probability_skill: "positive probability skill",
        beats_buy_and_hold_after_costs: "beats buy & hold after costs",
        survives_the_charged_cost: "survives the charged cost",
        passes_leakage_check: "passes leakage check",
    };

    return (
        <div style={{
            border: `1px solid ${colour}55`, background: `${colour}12`, borderRadius: 8,
            padding: "12px 14px", marginBottom: 14,
        }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 8 }}>
                <Badge color={colour}>{ship ? "SHIP" : "DO NOT SHIP"}</Badge>
                <span style={{ color: C.textMid, fontSize: 12 }}>
                    {evaluation?.n_test_days} out-of-sample days over {evaluation?.n_folds} walk-forward folds
                </span>
                {evaluation?.leakage_check_passed === false && (
                    <Badge color={C.red}>LEAKAGE CHECK FAILED — every number below is void</Badge>
                )}
            </div>
            <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
                {Object.entries(verdict.criteria || {}).map(([key, passed]) => (
                    <span key={key} style={{
                        color: passed ? C.green : C.textDim, fontSize: 11,
                        fontFamily: "'DM Mono',monospace",
                    }}>
                        {passed ? "✓" : "✗"} {labels[key] || key}
                    </span>
                ))}
            </div>
        </div>
    );
}

/* ─── Baseline comparison table ──────────────────────────────── */
function BaselineTable({ evaluation }) {
    const rows = [
        {
            name: "model",
            accuracy: evaluation.accuracy,
            ci: evaluation.accuracy_ci,
            balanced: evaluation.balanced_accuracy,
            mcc: evaluation.mcc,
            highlight: true,
        },
        ...Object.entries(evaluation.baselines || {}).map(([name, metrics]) => ({
            name,
            accuracy: metrics.accuracy,
            ci: metrics.accuracy_ci,
            balanced: metrics.balanced_accuracy,
            mcc: metrics.mcc,
            best: name === evaluation.best_baseline,
        })),
    ];

    const cell = { padding: "6px 8px", fontFamily: "'DM Mono',monospace", fontSize: 12 };
    const head = { ...cell, color: C.textDim, fontSize: 10, letterSpacing: 1, textTransform: "uppercase" };

    return (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
                <tr>
                    <th style={{ ...head, textAlign: "left" }}>predictor</th>
                    <th style={{ ...head, textAlign: "right" }}>accuracy</th>
                    <th style={{ ...head, textAlign: "right" }}>95% CI</th>
                    <th style={{ ...head, textAlign: "right" }}>balanced</th>
                    <th style={{ ...head, textAlign: "right" }}>MCC</th>
                </tr>
            </thead>
            <tbody>
                {rows.map((row) => (
                    <tr key={row.name} style={{ borderTop: `1px solid ${C.border}` }}>
                        <td style={{ ...cell, color: row.highlight ? C.cyan : C.textMid }}>
                            {row.name}{row.best ? " *" : ""}
                        </td>
                        <td style={{ ...cell, textAlign: "right", color: row.highlight ? C.text : C.textMid }}>
                            {pct(row.accuracy, 2)}
                        </td>
                        <td style={{ ...cell, textAlign: "right", color: C.textDim }}>
                            {row.ci ? `${pct(row.ci[0], 1)}–${pct(row.ci[1], 1)}` : "—"}
                        </td>
                        <td style={{ ...cell, textAlign: "right", color: C.textMid }}>{pct(row.balanced, 2)}</td>
                        <td style={{ ...cell, textAlign: "right", color: row.mcc > 0 ? C.green : C.textDim }}>
                            {num(row.mcc)}
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

/* ─── Panel ──────────────────────────────────────────────────── */
export default function DirectionPanel({ symbol, model = "logistic", modelPrep }) {
    const [state, setState] = useState({ status: "idle", data: null, error: null });

    // The walk-forward evaluation is one of the artifacts App's preparation run
    // produces, so this refetches when that run finishes rather than polling on
    // its own or asking the user to go and generate one.
    const readyVersion = modelPrep?.readyVersion ?? 0;

    const load = useCallback(async (signal) => {
        setState((prev) => ({ ...prev, status: "loading", error: null }));
        try {
            const data = await fetchDirection(symbol, model, true);
            if (signal?.aborted) return;
            // The server answers 200 with status "preparing" or "unavailable"
            // when no evaluation exists yet — it has started the run, or said
            // why it cannot. Either way the gauge stays absent, which is the
            // invariant: a probability is never shown without the measured
            // accuracy that says whether it means anything.
            setState({
                status: data?.evaluation ? "ready" : "absent",
                data,
                error: data?.message || null,
            });
        } catch (err) {
            if (signal?.aborted) return;
            setState({
                // A 404 no longer means "no report" — the route stopped raising
                // one for that — so it is treated as the transport error it now is.
                status: err instanceof ApiError && err.status === 404 ? "absent" : "error",
                data: null,
                error: err.message,
            });
        }
    }, [symbol, model]);

    useEffect(() => {
        const controller = new AbortController();
        load(controller.signal);
        return () => controller.abort();
    }, [load, readyVersion]);

    if (state.status === "loading" || state.status === "idle") {
        return <Section title="Next-day direction"><div style={{ color: C.textDim, fontSize: 12 }}>Loading…</div></Section>;
    }

    if (state.status === "absent") {
        return (
            <Section title="Next-day direction">
                {modelPrep ? (
                    <ModelPreparation preparation={modelPrep} context="direction evaluation" />
                ) : (
                    <div style={{ color: C.textMid, fontSize: 12, lineHeight: 1.7 }}>
                        {state.error || `Preparing the walk-forward evaluation for ${symbol}.`}
                    </div>
                )}
            </Section>
        );
    }

    if (state.status === "error") {
        return (
            <Section title="Next-day direction">
                <div style={{ color: C.red, fontSize: 12 }}>{state.error}</div>
            </Section>
        );
    }

    const { evaluation, verdict, backtest, next_session: nextSession } = state.data;
    const rollingHitRate = state.data.rolling_hit_rate || [];
    const equityCurve = state.data.equity_curve || [];
    const edge = evaluation.edge_vs_best_baseline || {};
    const strategy = backtest?.strategy || {};
    const benchmark = backtest?.benchmark || {};
    const breakeven = backtest?.breakeven || {};
    const tradeable = Boolean(nextSession?.available && nextSession?.tradeable);

    return (
        <Section
            title={`Next-day direction · ${symbol}`}
            hint="Predicts the SIGN of tomorrow's move, not a price. Unlike the 30-day forecast path, every point here is a separate prediction scored against the outcome that actually happened."
            right={
                <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                    {evaluation.n_features && (
                        <Badge color={C.textDim}>{evaluation.n_features} features</Badge>
                    )}
                    <Badge color={C.cyan}>{model}</Badge>
                </span>
            }
        >
            <VerdictBanner verdict={verdict} evaluation={evaluation} />

            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start" }}>
                {/* Gauge, never shown without its caveat */}
                <div style={{ flex: "0 0 260px", minWidth: 230 }}>
                    {nextSession?.available ? (
                        <>
                            <ProbabilityGauge
                                probability={nextSession.probability_up}
                                baseRate={nextSession.train_base_rate}
                                tradeable={tradeable}
                            />
                            <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.6, marginTop: 6 }}>
                                <div>
                                    <span style={{ color: C.amber }}>│</span> training base rate{" "}
                                    {pct(nextSession.train_base_rate, 1)} · model is{" "}
                                    {nextSession.edge_over_base_rate_pp >= 0 ? "+" : ""}
                                    {num(nextSession.edge_over_base_rate_pp, 2)}pp from it
                                </div>
                                <div style={{ marginTop: 4 }}>as of close {nextSession.as_of}</div>
                            </div>
                            {nextSession.price_forecast && (
                                <div style={{
                                    marginTop: 10, padding: "10px 12px", borderRadius: 6,
                                    border: `1px solid ${C.border}`, background: C.bg2,
                                }}>
                                    <div style={{
                                        color: C.textDim, fontSize: 10, letterSpacing: 1.4,
                                        textTransform: "uppercase", marginBottom: 6,
                                    }}>
                                        Tomorrow's close &mdash; forecast range
                                    </div>
                                    <div style={{
                                        display: "flex", justifyContent: "space-between",
                                        fontFamily: "'DM Mono',monospace", fontSize: 12,
                                    }}>
                                        <span style={{ color: C.red }}>
                                            {money(nextSession.price_forecast.price_lo_5)}
                                        </span>
                                        <span style={{ color: C.text, fontWeight: 700 }}>
                                            {money(nextSession.price_forecast.price_median)}
                                        </span>
                                        <span style={{ color: C.green }}>
                                            {money(nextSession.price_forecast.price_hi_95)}
                                        </span>
                                    </div>
                                    <div style={{
                                        display: "flex", justifyContent: "space-between",
                                        color: C.textDim, fontSize: 9, marginTop: 2,
                                    }}>
                                        <span>5th</span><span>median</span><span>95th</span>
                                    </div>
                                    {/* A range means nothing without the share of days it
                                        actually contained the outcome, measured out of sample. */}
                                    {evaluation.price_band && (
                                        <div style={{ color: C.textDim, fontSize: 11, marginTop: 8, lineHeight: 1.5 }}>
                                            This band held {pct(evaluation.price_band.coverage, 1)} of
                                            actual closes over {evaluation.price_band.n} out-of-sample
                                            days, against a {pct(evaluation.price_band.nominal_coverage, 0)} claim.
                                        </div>
                                    )}
                                </div>
                            )}

                            {!tradeable && (
                                <div style={{
                                    marginTop: 10, padding: "8px 10px", borderRadius: 6,
                                    border: `1px solid ${C.amber}44`, background: C.amberLow,
                                    color: C.textMid, fontSize: 11, lineHeight: 1.6,
                                }}>
                                    {nextSession.gate_reason || "Not validated for trading."}
                                    <div style={{ marginTop: 6, color: C.textDim }}>{nextSession.caveat}</div>
                                </div>
                            )}
                        </>
                    ) : (
                        <div style={{ color: C.textDim, fontSize: 12 }}>
                            Live gauge unavailable{nextSession?.error ? `: ${nextSession.error}` : "."}
                        </div>
                    )}
                </div>

                {/* Baselines and the significance of the gap */}
                <div style={{ flex: "1 1 380px", minWidth: 320 }}>
                    <BaselineTable evaluation={evaluation} />
                    <div style={{ color: C.textMid, fontSize: 11, marginTop: 10, lineHeight: 1.7 }}>
                        Model vs best baseline (<span style={{ color: C.text }}>{evaluation.best_baseline}</span>):{" "}
                        <span style={{ color: edge.edge_pp > 0 ? C.green : C.red }}>
                            {edge.edge_pp >= 0 ? "+" : ""}{num(edge.edge_pp, 2)}pp
                        </span>{" "}
                        ± {num(edge.standard_error_pp, 2)}pp, one-sided p = {num(edge.p_value_one_sided, 4)} —{" "}
                        <strong style={{ color: edge.significant ? C.green : C.amber }}>
                            {edge.significant ? "significant" : "not significant"}
                        </strong>
                        {edge.n_required ? ` (an edge this size needs ~${edge.n_required} test days)` : ""}.
                        <div style={{ marginTop: 6, color: C.textDim }}>
                            AUC {num(evaluation.roc_auc)} · Brier {num(evaluation.brier_score)} ·
                            Brier skill vs constant classifier{" "}
                            <span style={{ color: evaluation.brier_skill_score > 0 ? C.green : C.red }}>
                                {evaluation.brier_skill_score >= 0 ? "+" : ""}{num(evaluation.brier_skill_score, 4)}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Rolling hit rate against the rolling base rate */}
            {rollingHitRate.length > 0 && (
                <div style={{ marginTop: 20 }}>
                    <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1.4, textTransform: "uppercase", marginBottom: 8 }}>
                        Rolling {rollingHitRate[0]?.window}-day hit rate vs base rate
                    </div>
                    <ResponsiveContainer width="100%" height={150}>
                        <ComposedChart data={rollingHitRate} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                            <CartesianGrid stroke={C.border} strokeDasharray="2 4" vertical={false} />
                            <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 10 }} minTickGap={48} />
                            <YAxis
                                domain={[0.3, 0.7]} tick={{ fill: C.textDim, fontSize: 10 }} width={44}
                                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                            />
                            <Tooltip content={<DirectionTooltip format={(v) => pct(v, 2)} />} />
                            {/* A coin. Anything hugging this line is a coin. */}
                            <ReferenceLine y={0.5} stroke={C.textDim} strokeDasharray="4 4" />
                            <Line
                                type="monotone" dataKey="base_rate" name="base rate"
                                stroke={C.amber} strokeWidth={1.5} strokeDasharray="4 3" dot={false}
                            />
                            <Line
                                type="monotone" dataKey="hit_rate" name="hit rate"
                                stroke={C.cyan} strokeWidth={2} dot={false}
                            />
                        </ComposedChart>
                    </ResponsiveContainer>
                    <div style={{ color: C.textDim, fontSize: 11, marginTop: 4, lineHeight: 1.6 }}>
                        The amber line is the share of up days in the same window. Beating 50% is not skill
                        if the amber line is above it too — the gap between the two is the only part that is.
                    </div>
                </div>
            )}

            {/* Equity curve vs buy & hold */}
            {equityCurve.length > 0 && (
                <div style={{ marginTop: 20 }}>
                    <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1.4, textTransform: "uppercase", marginBottom: 8 }}>
                        Equity vs buy &amp; hold · {breakeven.cost_charged_bps} bps round trip per active day
                    </div>
                    <ResponsiveContainer width="100%" height={190}>
                        <ComposedChart data={equityCurve} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="directionEquityFill" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor={C.cyan} stopOpacity={0.28} />
                                    <stop offset="100%" stopColor={C.cyan} stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid stroke={C.border} strokeDasharray="2 4" vertical={false} />
                            <XAxis dataKey="date" tick={{ fill: C.textDim, fontSize: 10 }} minTickGap={48} />
                            <YAxis
                                tick={{ fill: C.textDim, fontSize: 10 }} width={52}
                                domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)}
                            />
                            <Tooltip content={<DirectionTooltip format={(v) => num(v, 4)} />} />
                            <ReferenceLine y={1} stroke={C.textDim} strokeDasharray="4 4" />
                            <Area
                                type="monotone" dataKey="strategy" name="strategy"
                                stroke={C.cyan} strokeWidth={2} fill="url(#directionEquityFill)"
                            />
                            <Line
                                type="monotone" dataKey="benchmark" name="buy & hold"
                                stroke={C.purple} strokeWidth={1.6} strokeDasharray="5 4" dot={false}
                            />
                        </ComposedChart>
                    </ResponsiveContainer>

                    <div style={{ display: "flex", gap: 22, flexWrap: "wrap", marginTop: 10, fontSize: 11, fontFamily: "'DM Mono',monospace" }}>
                        {[
                            ["total", signedPct(strategy.total_return), signedPct(benchmark.total_return)],
                            ["CAGR", signedPct(strategy.cagr), signedPct(benchmark.cagr)],
                            ["Sharpe", num(strategy.sharpe, 2), num(benchmark.sharpe, 2)],
                            ["max DD", signedPct(strategy.max_drawdown), signedPct(benchmark.max_drawdown)],
                            ["time in mkt", `${num(strategy.time_in_market_pct, 1)}%`, "100%"],
                        ].map(([label, left, right]) => (
                            <div key={label}>
                                <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1, textTransform: "uppercase" }}>{label}</div>
                                <div style={{ color: C.cyan }}>{left}</div>
                                <div style={{ color: C.purple }}>{right}</div>
                            </div>
                        ))}
                    </div>

                    <div style={{ color: C.textMid, fontSize: 11, marginTop: 12, lineHeight: 1.7 }}>
                        Execution: signal at the close of <em>t</em>, entry at the open of <em>t+1</em>, exit at
                        that day's close — the overnight gap is not captured, because a decision made at the
                        close cannot be filled at it.{" "}
                        <strong style={{ color: C.amber }}>
                            Breakeven cost {num(breakeven.breakeven_cost_bps_positive, 2)} bps
                        </strong>{" "}
                        — above that round-trip cost the edge is gone
                        {typeof breakeven.mean_gross_return_per_trade_bps === "number"
                            ? ` (gross edge ${num(breakeven.mean_gross_return_per_trade_bps, 2)} bps per trade)`
                            : ""}.
                    </div>
                </div>
            )}
        </Section>
    );
}
