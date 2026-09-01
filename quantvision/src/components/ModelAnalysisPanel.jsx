/**
 * Model output on the Analysis tab, from the same pipeline the Predictions tab reads.
 *
 * The point is that there is exactly one pipeline. This panel does not train, does
 * not pick models, and does not have its own notion of what is available: it waits
 * on App's preparation run and then reads the two endpoints the Predictions tab
 * reads — the ensemble forecast and the walk-forward direction report. If the two
 * tabs ever disagree, it is because the server changed between requests, not
 * because they consulted different machinery.
 *
 * Numbers are shown with what qualifies them: accuracy with its confidence
 * interval, a forecast with its band, a probability with the verdict that says
 * whether it is tradeable. A bare point estimate is not rendered here.
 */

import { useEffect, useState } from "react";

import { fetchDirection, fetchEnsemblePrediction } from "../utils/api";
import { C } from "../utils/data";
import { Badge, Section, StatCard } from "./UIComponents";
import ModelPreparation from "./ModelPreparation";

/** Horizon the Analysis summary reports. Matches the Predictions tab default. */
const SUMMARY_HORIZON = 30;

const pct = (value, digits = 1) =>
    typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
const money = (value) =>
    typeof value === "number" && Number.isFinite(value)
        ? `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : "—";
const signed = (value, digits = 2) =>
    typeof value === "number" && Number.isFinite(value)
        ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`
        : "—";

export default function ModelAnalysisPanel({ symbol, modelPrep, apiConnected }) {
    const [state, setState] = useState({ loading: true, forecast: null, direction: null, error: null });
    const readyVersion = modelPrep?.readyVersion ?? 0;
    // See PredictionsTab: the readiness check is too brief to be worth hiding
    // already-loaded output for.
    const preparing = modelPrep?.status === "preparing";

    useEffect(() => {
        if (!apiConnected || !symbol) return undefined;
        let cancelled = false;
        setState((prev) => ({ ...prev, loading: true, error: null }));

        // allSettled: the direction evaluation and the price ensemble fail
        // independently, and one being absent is no reason to hide the other.
        Promise.allSettled([
            fetchEnsemblePrediction(symbol, SUMMARY_HORIZON),
            fetchDirection(symbol, "logistic", true),
        ]).then(([forecastResult, directionResult]) => {
            if (cancelled) return;
            setState({
                loading: false,
                forecast: forecastResult.status === "fulfilled" ? forecastResult.value : null,
                direction: directionResult.status === "fulfilled" ? directionResult.value : null,
                error: forecastResult.status === "rejected" && directionResult.status === "rejected"
                    ? forecastResult.reason?.message || "Model output is unavailable."
                    : null,
            });
        });

        return () => { cancelled = true; };
    }, [symbol, apiConnected, readyVersion]);

    if (preparing) {
        return (
            <Section title="Model analysis" hint="Produced by the same trained models the Predictions tab serves.">
                <ModelPreparation preparation={modelPrep} context="analysis" />
            </Section>
        );
    }

    if (state.loading) {
        return (
            <Section title="Model analysis">
                <div style={{ color: C.textDim, fontSize: 12 }}>Loading model output…</div>
            </Section>
        );
    }

    const forecast = state.forecast;
    const direction = state.direction;
    const summary = forecast?.ensemble;
    const evaluation = direction?.evaluation;
    const nextSession = direction?.next_session;
    const hasForecast = forecast?.status === "ok" && summary;
    const hasDirection = Boolean(evaluation);

    if (!hasForecast && !hasDirection) {
        return (
            <Section title="Model analysis">
                <ModelPreparation preparation={modelPrep} context="model analysis" />
            </Section>
        );
    }

    const tradeable = Boolean(nextSession?.available && nextSession?.tradeable);
    const weights = Object.entries(forecast?.weights || {});

    return (
        <Section
            title="Model analysis"
            hint="The same trained bundles and walk-forward evaluation the Predictions tab serves — not a separate model."
            right={
                <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                    {forecast?.degraded && <Badge color={C.amber}>partial ensemble</Badge>}
                    {hasDirection && (
                        <Badge color={direction.verdict?.ship ? C.green : C.textDim}>
                            {direction.verdict?.ship ? "ship" : "do not ship"}
                        </Badge>
                    )}
                </span>
            }
        >
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))", gap: 12 }}>
                {hasForecast && (
                    <>
                        <StatCard
                            label={`${SUMMARY_HORIZON}D FORECAST`}
                            value={money(summary.target)}
                            sub={`${signed(summary.change_pct)} · ${summary.signal || "—"}`}
                            color={Number(summary.change_pct) >= 0 ? C.green : C.red}
                        />
                        <StatCard
                            label="95% BAND"
                            value={`${money(summary.lower_95)} – ${money(summary.upper_95)}`}
                            sub={summary.reliability ? `${summary.reliability} reliability` : null}
                            color={C.cyan}
                        />
                    </>
                )}
                {hasDirection && (
                    <>
                        <StatCard
                            label="OUT-OF-SAMPLE ACCURACY"
                            value={pct(evaluation.accuracy)}
                            sub={`95% CI ${pct(evaluation.accuracy_ci?.[0])}–${pct(evaluation.accuracy_ci?.[1])} over ${evaluation.n_test_days} days`}
                            color={C.amber}
                        />
                        <StatCard
                            label="P(UP TOMORROW)"
                            value={nextSession?.available ? pct(nextSession.probability_up) : "—"}
                            sub={tradeable ? "Cleared its ship criteria" : "Not tradeable — see the verdict"}
                            color={tradeable ? C.green : C.textDim}
                        />
                    </>
                )}
            </div>

            {hasDirection && !tradeable && (
                <div style={{
                    marginTop: 12, padding: "10px 12px", borderRadius: 6,
                    background: C.bg2, border: `1px solid ${C.border}`,
                    color: C.textMid, fontSize: 11.5, lineHeight: 1.7,
                }}>
                    {nextSession?.gate_reason
                        || direction.verdict?.summary
                        || "The walk-forward run did not clear its ship criteria, so the probability is shown without being presented as actionable."}
                </div>
            )}

            {weights.length > 0 && (
                <div style={{ marginTop: 12, display: "flex", gap: 14, flexWrap: "wrap", fontSize: 11, color: C.textDim }}>
                    <span>Ensemble weights:</span>
                    {weights.map(([model, weight]) => (
                        <span key={model} style={{ fontFamily: "'DM Mono',monospace", color: C.textMid }}>
                            {model.replace("_", " ")} {pct(weight, 0)}
                        </span>
                    ))}
                </div>
            )}

            {forecast?.message && (
                <div style={{ marginTop: 10, color: C.textDim, fontSize: 11, lineHeight: 1.6 }}>
                    {forecast.message}
                </div>
            )}
        </Section>
    );
}
