import { useEffect, useMemo, useState } from "react";
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
import { C } from "../utils/data";
import { fetchEnsemblePrediction, fetchPredictions } from "../utils/api";
import DirectionPanel from "../components/DirectionPanel";
import ModelPreparation from "../components/ModelPreparation";

const HORIZONS = [7, 15, 30, 60];

/** Multi-horizon regression bundles. These draw the forecast fan. */
const LEGACY_MODEL_KEYS = ["xgboost", "random_forest", "lstm"];

/**
 * Unified next-timeframe models: one price and one P(up) for the next bar.
 *
 * Fetched only when one is actually selected, never as part of the "All
 * Models" sweep. Kronos runs a transformer forward pass per request, so
 * firing it on every ticker change would stall the tab for everyone who
 * never opened it.
 */
const UNIFIED_MODEL_KEYS = ["unified_xgboost", "unified_random_forest", "unified_lstm", "unified_kronos"];

const MODEL_KEYS = [...LEGACY_MODEL_KEYS, ...UNIFIED_MODEL_KEYS];

const isUnifiedModel = (modelType) => UNIFIED_MODEL_KEYS.includes(modelType);
const MODEL_OPTIONS = [
    { value: "all", label: "All Models" },
    { value: "ensemble", label: "Ensemble" },
    { value: "lstm", label: "LSTM" },
    { value: "xgboost", label: "XGBoost" },
    { value: "random_forest", label: "Random Forest" },
    { value: "unified_xgboost", label: "Unified XGBoost" },
    { value: "unified_random_forest", label: "Unified Random Forest" },
    { value: "unified_lstm", label: "Unified LSTM" },
    { value: "unified_kronos", label: "Unified Kronos" },
];

const COLORS = {
    historical: "#9CA3AF",
    ensemble: "#F5C842",
    lstm: "#60A5FA",
    xgboost: "#F59E0B",
    random_forest: "#34D399",
    unified_xgboost: "#FB923C",
    unified_random_forest: "#4ADE80",
    unified_lstm: "#818CF8",
    unified_kronos: "#E879F9",
    band: "#6366F1",
    scenario: "#7C8AA5",
    surface: "#161B22",
    panel: "#0F1623",
};

const WEIGHT_LABELS = {
    lstm: "PyTorch LSTM",
    xgboost: "XGBoost",
    random_forest: "Random Forest",
};

const MODEL_LABELS = {
    historical: "Historical",
    prediction: "Prediction",
    ensemble: "Ensemble",
    lstm: "LSTM",
    xgboost: "XGBoost",
    random_forest: "Random Forest",
    unified_xgboost: "Unified XGBoost",
    unified_random_forest: "Unified Random Forest",
    unified_lstm: "Unified LSTM",
    unified_kronos: "Unified Kronos",
};

function formatPrice(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)}%`;
}

function formatDateLabel(date) {
    if (typeof date !== "string" || date.length < 10) return date;
    return date.slice(5).replace("-", "/");
}

function toFiniteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function firstFiniteNumber(...values) {
    for (const value of values) {
        const number = toFiniteNumber(value);
        if (number !== null) return number;
    }
    return null;
}

function modelLabel(model) {
    return MODEL_LABELS[model] || MODEL_LABELS.ensemble;
}

function getFinalForecastPoint(points) {
    return Array.isArray(points) && points.length ? points[points.length - 1] : null;
}

function normalizeSingleForecast(payload, modelType) {
    const points = (payload?.forecasts || []).map((point) => ({
        date: point.date,
        predicted: firstFiniteNumber(point.predicted, point.prediction),
        lower_95: point.lower95,
        upper_95: point.upper95,
        lower_68: point.lower68,
        upper_68: point.upper68,
        [modelType]: firstFiniteNumber(point.predicted, point.prediction),
    }));
    return {
        ...payload,
        forecast_points: points,
    };
}

function normalizeEnsemblePoint(point) {
    const prediction = firstFiniteNumber(point.prediction, point.predicted, point.ensemble);
    return {
        ...point,
        ensemble: prediction,
        prediction,
        predicted: prediction,
        lower_95: point.lower_95 ?? point.lower_90,
        upper_95: point.upper_95 ?? point.upper_90,
        lower_68: point.lower_68 ?? point.lower_90,
        upper_68: point.upper_68 ?? point.upper_90,
    };
}

function buildFallbackEnsemble(models) {
    const available = MODEL_KEYS
        .map((key) => [key, models?.[key]])
        .filter(([, payload]) => payload?.status === "ok" && Array.isArray(payload.forecast_points) && payload.forecast_points.length);
    if (!available.length) return null;

    const first = available[0][1];
    const points = first.forecast_points.map((point, index) => {
        const values = available
            .map(([key, payload]) => [key, firstFiniteNumber(
                payload.forecast_points[index]?.prediction,
                payload.forecast_points[index]?.predicted,
                payload.forecast_points[index]?.[key],
            )])
            .filter(([, value]) => value !== null);
        const predicted = values.reduce((sum, [, value]) => sum + Number(value), 0) / Math.max(values.length, 1);
        // Averaging point forecasts gives a centre but no interval. Reusing the
        // centre as both bounds drew a zero-width band and printed the forecast
        // price into the Upper 95% and Lower 95% cards as though it were an
        // interval, which is why all three read the same number. Null is the
        // honest answer: this path has no calibrated uncertainty.
        const bounds = available.reduce(
            (acc, [, payload]) => {
                const p = payload.forecast_points[index];
                acc.lower_95.push(toFiniteNumber(p?.lower_95));
                acc.upper_95.push(toFiniteNumber(p?.upper_95));
                acc.lower_68.push(toFiniteNumber(p?.lower_68));
                acc.upper_68.push(toFiniteNumber(p?.upper_68));
                return acc;
            },
            { lower_95: [], upper_95: [], lower_68: [], upper_68: [] },
        );
        const widest = (list, pick) => {
            const finite = list.filter((v) => v !== null);
            return finite.length ? pick(...finite) : null;
        };
        const row = {
            date: point.date,
            ensemble: predicted,
            prediction: predicted,
            predicted,
            lower_95: widest(bounds.lower_95, Math.min),
            upper_95: widest(bounds.upper_95, Math.max),
            lower_68: widest(bounds.lower_68, Math.min),
            upper_68: widest(bounds.upper_68, Math.max),
        };
        values.forEach(([key, value]) => {
            row[key] = Number(value);
        });
        return row;
    });
    const currentPrice = first.current_price;
    const finalPoint = getFinalForecastPoint(points);
    const changePct = finalPoint && currentPrice
        ? ((finalPoint.predicted - currentPrice) / currentPrice) * 100
        : 0;

    return {
        status: "ok",
        model_available: true,
        current_price: currentPrice,
        current_price_source: first.current_price_source,
        forecast_points: points,
        ensemble: {
            target: finalPoint?.predicted,
            change_pct: changePct,
            upper_95: finalPoint?.upper_95,
            lower_95: finalPoint?.lower_95,
            upper_68: finalPoint?.upper_68,
            lower_68: finalPoint?.lower_68,
            upper_90: finalPoint?.upper_95,
            lower_90: finalPoint?.lower_95,
            signal: changePct >= 0 ? "Bullish" : "Bearish",
            reliability: available.length === MODEL_KEYS.length ? "Medium" : "Low",
            consensus: `${available.length} model${available.length === 1 ? "" : "s"} available`,
        },
        weights: available.reduce((acc, [key]) => {
            acc[key] = 1 / available.length;
            return acc;
        }, {}),
    };
}

function resolveDisplayData(data, selectedModel) {
    if (!data) return {};
    const ensemblePayload = data.ensemblePayload?.status === "ok"
        ? {
            ...data.ensemblePayload,
            forecast_points: (data.ensemblePayload.forecast_points || []).map(normalizeEnsemblePoint),
        }
        : buildFallbackEnsemble(data.models);

    if (MODEL_KEYS.includes(selectedModel)) {
        const payload = data.models?.[selectedModel];
        const points = payload?.forecast_points || [];
        const finalPoint = getFinalForecastPoint(points);
        const changePct = finalPoint && payload?.current_price
            ? ((finalPoint.predicted - payload.current_price) / payload.current_price) * 100
            : payload?.expected_change_pct;
        return {
            payload,
            points,
            currentPrice: payload?.current_price,
            currentPriceSource: payload?.current_price_source,
            target: finalPoint?.predicted ?? payload?.target_price ?? payload?.predicted_price,
            changePct,
            lower95: finalPoint?.lower_95 ?? payload?.lower95,
            upper95: finalPoint?.upper_95 ?? payload?.upper95,
            lower68: finalPoint?.lower_68 ?? payload?.lower68,
            upper68: finalPoint?.upper_68 ?? payload?.upper68,
            signal: payload?.signal || (Number(changePct) >= 0 ? "Bullish" : "Bearish"),
            reliability: payload?.status === "ok" ? "Model" : "Unavailable",
            tableLabel: modelLabel(selectedModel),
            chartModel: selectedModel,
            unavailable: payload?.status !== "ok",
            message: payload?.message || payload?.model_info?.message,
            // The single-model route carries provenance on model_info.
            pathType: payload?.model_info?.path_type,
            perStepPredictions: payload?.model_info?.per_step_predictions,
            modelOutputCount: payload?.model_info?.model_output_count,
            scenarioPaths: payload?.scenario_paths,
            probabilityUp: payload?.probability_up,
            probabilityDown: payload?.probability_down,
            headsNote: payload?.model_info?.heads_note,
        };
    }

    const finalPoint = getFinalForecastPoint(ensemblePayload?.forecast_points);
    const summary = ensemblePayload?.ensemble;
    return {
        payload: ensemblePayload,
        points: ensemblePayload?.forecast_points || [],
        currentPrice: ensemblePayload?.current_price,
        currentPriceSource: ensemblePayload?.current_price_source,
        target: summary?.target ?? finalPoint?.predicted,
        changePct: summary?.change_pct,
        lower95: summary?.lower_95 ?? summary?.lower_90 ?? finalPoint?.lower_95,
        upper95: summary?.upper_95 ?? summary?.upper_90 ?? finalPoint?.upper_95,
        lower68: summary?.lower_68 ?? finalPoint?.lower_68,
        upper68: summary?.upper_68 ?? finalPoint?.upper_68,
        signal: summary?.signal,
        reliability: summary?.reliability,
        consensus: summary?.consensus,
        tableLabel: selectedModel === "ensemble" ? "Ensemble" : "Forecast",
        chartModel: selectedModel === "all" ? "all" : "ensemble",
        unavailable: !ensemblePayload || ensemblePayload.status !== "ok",
        message: ensemblePayload?.message,
        pathType: ensemblePayload?.path_type,
        perStepPredictions: ensemblePayload?.per_step_predictions,
        modelOutputCount: ensemblePayload?.model_output_count,
        scenarioPaths: ensemblePayload?.scenario_paths,
        degraded: ensemblePayload?.degraded,
        modelsAvailable: ensemblePayload?.models_available,
        modelsUnavailable: ensemblePayload?.models_unavailable,
    };
}

/**
 * Say where the daily points came from.
 *
 * In the default mode each model emits a single cumulative horizon-day return
 * and the days in between are compounded toward it. Those points are not daily
 * predictions, and a chart that draws them like one invites the reader to trust
 * detail the model never produced.
 */
function ForecastProvenanceNote({ pathType, perStepPredictions, modelOutputCount, horizon, scenarioCount }) {
    if (!pathType) return null;
    // per_step_predictions is authoritative; path_type is the readable fallback.
    const perStep = perStepPredictions ?? pathType === "recursive_per_step";
    const interpolated = !perStep;
    return (
        <div
            style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 12px",
                borderRadius: 8,
                border: `1px solid ${interpolated ? C.amber || "#8a6d3b" : C.border}`,
                background: "rgba(255,255,255,0.03)",
                color: C.sub,
                fontSize: 12,
                lineHeight: 1.5,
            }}
        >
            <span style={{ fontWeight: 900, color: interpolated ? C.amber || "#d6a13a" : C.green }}>
                {interpolated ? "Projected path" : "Per-step forecast"}
            </span>
            <span>
                {interpolated
                    ? `Each model produced exactly one number — the ${horizon}-day return${
                          modelOutputCount ? ` (${modelOutputCount} model outputs total)` : ""
                      }. The dashed line is not a forecast of the days it passes through: it is
                       P(t) = P(0) x (1 + r) ^ (t/${horizon}), the compounding that reaches that
                       endpoint. Being monotone by construction, it cannot show a direction change,
                       and a turn in it would be arithmetic, not a prediction. Read the endpoint and
                       the band; ignore the shape between them.`
                    : `Every point is a separate model prediction${
                          modelOutputCount ? ` (${modelOutputCount} model outputs)` : ""
                      }. Later steps are conditioned on earlier predicted bars, so error compounds.`}
                {scenarioCount > 0 && (
                    <>
                        {" "}
                        The centre line is an average outcome and is smooth by
                        construction — the {scenarioCount} simulated paths behind it show the
                        day-to-day volatility any real price path would carry.
                    </>
                )}
            </span>
        </div>
    );
}

function MetricCard({ label, value, sub, color }) {
    return (
        <div style={{
            background: COLORS.surface,
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: "16px 18px",
            minHeight: 96,
            display: "grid",
            alignContent: "center",
            gap: 7,
        }}>
            <div style={{ fontSize: 11, color: C.textDim, fontWeight: 700, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                {label}
            </div>
            <div style={{ fontSize: 24, lineHeight: 1.1, color: color || C.text, fontWeight: 800 }}>
                {value}
            </div>
            {sub && <div style={{ fontSize: 12, color: C.textMid }}>{sub}</div>}
        </div>
    );
}

function TooltipContent({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    const rows = payload
        .filter((item) => MODEL_LABELS[item.dataKey] && item.value !== null && item.value !== undefined)
        .map((item) => ({
            key: item.dataKey,
            label: MODEL_LABELS[item.dataKey],
            value: item.value,
            color: item.color,
        }));
    if (!rows.length) return null;

    return (
        <div style={{
            background: "#0B101A",
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: "10px 12px",
            boxShadow: "0 12px 28px rgba(0,0,0,.28)",
            minWidth: 170,
        }}>
            <div style={{ color: C.text, fontWeight: 700, fontSize: 12, marginBottom: 8 }}>{label}</div>
            <div style={{ display: "grid", gap: 5 }}>
                {rows.map((row) => (
                    <div key={row.key} style={{ display: "flex", justifyContent: "space-between", gap: 18, fontSize: 12 }}>
                        <span style={{ color: row.color }}>{row.label}</span>
                        <span style={{ color: C.text }}>{formatPrice(row.value)}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function ForecastChart({ priceData, forecastPoints, selectedModel, scenarioPaths, showScenarios, onToggleScenarios, interpolated }) {
    const { chartData, todayDate, yDomain, scenarioKeys } = useMemo(() => {
        const history = (priceData?.bars || []).slice(-60);
        const future = (forecastPoints || [])
            .filter((point) => point?.date)
            .slice()
            .sort((a, b) => String(a.date).localeCompare(String(b.date)));
        const rows = [];
        let boundary = null;

        // Monte Carlo paths, one series per scenario. Each path is [today, ...steps],
        // so element 0 pins to the last historical bar and element i+1 to forecast
        // point i — that alignment is what makes the fan start at today's price
        // instead of floating free of the history.
        const paths = showScenarios && Array.isArray(scenarioPaths) ? scenarioPaths : [];
        const keys = paths.map((_, index) => `scenario_${index}`);

        history.forEach((bar, index) => {
            const isLast = index === history.length - 1;
            if (isLast) boundary = bar.date;
            const row = {
                date: bar.date,
                historical: bar.close,
                ensemble: isLast && forecastPoints.length ? bar.close : null,
                lstm: isLast && forecastPoints.length ? bar.close : null,
                xgboost: isLast && forecastPoints.length ? bar.close : null,
                random_forest: isLast && forecastPoints.length ? bar.close : null,
                upper_90: isLast && forecastPoints.length ? bar.close : null,
                lower_90: isLast && forecastPoints.length ? bar.close : null,
                upper_95: isLast && forecastPoints.length ? bar.close : null,
                lower_95: isLast && forecastPoints.length ? bar.close : null,
                upper_68: isLast && forecastPoints.length ? bar.close : null,
                lower_68: isLast && forecastPoints.length ? bar.close : null,
            };
            keys.forEach((key, pathIndex) => {
                row[key] = isLast ? toFiniteNumber(paths[pathIndex]?.[0]) ?? bar.close : null;
            });
            rows.push(row);
        });

        future.forEach((point, stepIndex) => {
            const prediction = firstFiniteNumber(point.prediction, point.predicted, point.ensemble);
            const row = {
                date: point.date,
                historical: null,
                ensemble: prediction,
                lstm: firstFiniteNumber(point.lstm, selectedModel === "lstm" ? prediction : null),
                xgboost: firstFiniteNumber(point.xgboost, selectedModel === "xgboost" ? prediction : null),
                random_forest: firstFiniteNumber(point.random_forest, selectedModel === "random_forest" ? prediction : null),
                upper_90: point.upper_90 ?? point.upper_95,
                lower_90: point.lower_90 ?? point.lower_95,
                upper_95: point.upper_95 ?? point.upper_90,
                lower_95: point.lower_95 ?? point.lower_90,
                upper_68: point.upper_68,
                lower_68: point.lower_68,
            };
            keys.forEach((key, pathIndex) => {
                row[key] = toFiniteNumber(paths[pathIndex]?.[stepIndex + 1]);
            });
            rows.push(row);
        });

        const visibleKeys = selectedModel === "all"
            ? ["historical", "ensemble", "lstm", "xgboost", "random_forest"]
            : ["historical", selectedModel];
        // Scenario paths are deliberately included in the domain: clipping the fan
        // would understate exactly the spread it exists to show.
        const domainKeys = [...visibleKeys, ...keys];
        const lineValues = rows.flatMap((row) => domainKeys.map((key) => toFiniteNumber(row[key]))).filter((value) => value !== null);
        const min = Math.min(...lineValues);
        const max = Math.max(...lineValues);
        const span = Number.isFinite(max - min) ? max - min : 0;
        const pad = Math.max(span * 0.08, Math.abs(max || min || 0) * 0.01, 1);
        const domain = lineValues.length
            ? [Math.max(0, Number((min - pad).toFixed(2))), Number((max + pad).toFixed(2))]
            : ["auto", "auto"];

        return { chartData: rows, todayDate: boundary, yDomain: domain, scenarioKeys: keys };
    }, [priceData, forecastPoints, selectedModel, scenarioPaths, showScenarios]);

    const show = (model) => selectedModel === "all" || selectedModel === model;
    const scenarioCount = Array.isArray(scenarioPaths) ? scenarioPaths.length : 0;

    return (
        <div style={{ background: COLORS.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 14, fontSize: 12, color: C.textMid, alignItems: "center" }}>
                {[
                    ["historical", "Historical", null],
                    ["ensemble", "Ensemble", null],
                    ["lstm", "LSTM", "5 5"],
                    ["xgboost", "XGBoost", "5 5"],
                    ["random_forest", "Random Forest", "5 5"],
                ].map(([key, label, dash]) => (
                    <div key={key} style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
                        <span style={{ width: 18, height: 0, borderTop: `3px ${dash ? "dashed" : "solid"} ${COLORS[key]}` }} />
                        <span>{label}</span>
                    </div>
                ))}
                {scenarioCount > 0 && (
                    <button
                        type="button"
                        onClick={onToggleScenarios}
                        title="Monte Carlo paths simulated by resampling this stock's own recent daily moves. They show the volatility a forecast has to live inside; the centre line is the average outcome, not a path the market is expected to trace."
                        style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 7,
                            marginLeft: "auto",
                            background: "transparent",
                            border: `1px solid ${C.border}`,
                            borderRadius: 6,
                            padding: "4px 9px",
                            color: showScenarios ? C.text : C.textDim,
                            fontSize: 12,
                            cursor: "pointer",
                        }}
                    >
                        <span style={{
                            width: 18,
                            height: 0,
                            borderTop: `2px solid ${COLORS.scenario}`,
                            opacity: showScenarios ? 0.9 : 0.3,
                        }} />
                        <span>{scenarioCount} simulated paths</span>
                    </button>
                )}
            </div>
            <ResponsiveContainer width="100%" height={366}>
                <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                    <CartesianGrid stroke={C.border} strokeDasharray="3 3" opacity={0.28} vertical={false} />
                    <XAxis
                        dataKey="date"
                        tick={{ fill: C.textDim, fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={formatDateLabel}
                        minTickGap={22}
                    />
                    <YAxis
                        orientation="right"
                        tick={{ fill: C.textDim, fontSize: 11 }}
                        domain={yDomain}
                        allowDataOverflow
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(value) => `$${Number(value).toFixed(0)}`}
                    />
                    <Tooltip content={<TooltipContent />} />
                    <Area
                        type="monotone"
                        dataKey="upper_95"
                        stroke="none"
                        fill={COLORS.band}
                        fillOpacity={0.06}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    <Area
                        type="monotone"
                        dataKey="lower_95"
                        stroke="none"
                        fill={COLORS.panel}
                        fillOpacity={1}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    <Area
                        type="monotone"
                        dataKey="upper_68"
                        stroke="none"
                        fill={COLORS.band}
                        fillOpacity={0.14}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    <Area
                        type="monotone"
                        dataKey="lower_68"
                        stroke="none"
                        fill={COLORS.panel}
                        fillOpacity={1}
                        connectNulls
                        isAnimationActive={false}
                        tooltipType="none"
                    />
                    {todayDate && (
                        <ReferenceLine
                            x={todayDate}
                            stroke={C.textDim}
                            strokeDasharray="4 4"
                            label={{ value: "Today", position: "top", fill: C.textMid, fontSize: 11 }}
                        />
                    )}
                    {/* Simulated paths first, so the forecast lines draw over them.
                        `type="linear"` keeps each daily step as its own segment —
                        monotone smoothing would round off the very step-to-step
                        movement these paths exist to show. */}
                    {scenarioKeys.map((key) => (
                        <Line
                            key={key}
                            type="linear"
                            dataKey={key}
                            stroke={COLORS.scenario}
                            strokeWidth={1}
                            strokeOpacity={0.32}
                            dot={false}
                            connectNulls
                            isAnimationActive={false}
                            legendType="none"
                            tooltipType="none"
                        />
                    ))}
                    <Line type="monotone" dataKey="historical" stroke={COLORS.historical} strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line hide={!show("lstm")} type="monotone" dataKey="lstm" stroke={COLORS.lstm} strokeWidth={2} strokeDasharray="6 5" dot={false} connectNulls isAnimationActive={false} />
                    <Line hide={!show("xgboost")} type="monotone" dataKey="xgboost" stroke={COLORS.xgboost} strokeWidth={2} strokeDasharray="6 5" dot={false} connectNulls isAnimationActive={false} />
                    <Line hide={!show("random_forest")} type="monotone" dataKey="random_forest" stroke={COLORS.random_forest} strokeWidth={2} strokeDasharray="6 5" dot={false} connectNulls isAnimationActive={false} />
                    {/* Dashed whenever the path is interpolated. A solid, heavy
                        line reads as a series of forecasts; when each model
                        emitted one endpoint and the days between it are
                        P_0*(1+r)^(t/H), that reading is wrong — the curve is
                        monotone by construction and cannot show a turn. */}
                    <Line
                        hide={!show("ensemble")}
                        type="monotone"
                        dataKey="ensemble"
                        stroke={COLORS.ensemble}
                        strokeWidth={interpolated ? 3 : 4}
                        strokeDasharray={interpolated ? "7 5" : undefined}
                        dot={false}
                        connectNulls
                        isAnimationActive={false}
                    />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
}

function WeightsPanel({ weights, consensus, signal, reliability }) {
    // A model missing from `weights` did not contribute — it was excluded from a
    // partial ensemble. Falling back to its nominal share showed an excluded LSTM
    // at 40% alongside two renormalised members, so the bars summed to 140% and
    // credited a model that never ran. Absent means zero.
    const hasWeights = weights && Object.keys(weights).length > 0;
    const resolved = {
        lstm: Number(hasWeights ? weights.lstm ?? 0 : 0.4),
        xgboost: Number(hasWeights ? weights.xgboost ?? 0 : 0.35),
        random_forest: Number(hasWeights ? weights.random_forest ?? 0 : 0.25),
    };
    const signalColor = signal === "Bearish" ? C.red : signal === "Neutral" ? C.amber : C.green;

    return (
        <div style={{ background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18 }}>
            <div style={{ fontSize: 13, fontWeight: 800, color: C.text, marginBottom: 16 }}>Ensemble Weights</div>
            <div style={{ display: "grid", gap: 13 }}>
                {["lstm", "xgboost", "random_forest"].map((key) => {
                    const pct = Math.round(resolved[key] * 100);
                    const excluded = pct === 0;
                    return (
                        <div key={key} style={{ display: "grid", gridTemplateColumns: "128px 1fr 66px", alignItems: "center", gap: 12 }}>
                            <div style={{ color: excluded ? C.textDim : C.textMid, fontSize: 12, fontWeight: 700 }}>
                                {WEIGHT_LABELS[key]}
                            </div>
                            <div style={{ height: 10, background: "#0A0F18", borderRadius: 999, overflow: "hidden", border: "1px solid rgba(255,255,255,.05)" }}>
                                <div style={{ width: `${pct}%`, height: "100%", background: COLORS[key] }} />
                            </div>
                            <div style={{ color: excluded ? C.textDim : C.text, fontSize: 12, fontWeight: 800, textAlign: "right" }}>
                                {excluded ? "excl." : `${pct}%`}
                            </div>
                        </div>
                    );
                })}
            </div>
            <div style={{ marginTop: 18, borderTop: `1px solid ${C.border}`, paddingTop: 14 }}>
                <div style={{ color: signalColor, fontSize: 13, fontWeight: 800 }}>
                    {consensus || `${reliability || "Medium"} reliability`}
                </div>
            </div>
        </div>
    );
}

function ForecastTable({ rows, modelKey, label }) {
    const tableRows = (rows || []).slice(0, 7);
    return (
        <div style={{ background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 18, overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 620, fontSize: 12 }}>
                <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                        <th style={{ color: C.textDim, textAlign: "left", padding: "0 10px 10px 0", fontWeight: 800 }}>DATE</th>
                        <th style={{ color: COLORS[modelKey] || COLORS.ensemble, textAlign: "right", padding: "0 10px 10px", fontWeight: 800 }}>{label || "PREDICTED PRICE"}</th>
                        <th style={{ color: C.red, textAlign: "right", padding: "0 10px 10px", fontWeight: 800 }}>LOWER 95%</th>
                        <th style={{ color: C.green, textAlign: "right", padding: "0 10px 10px", fontWeight: 800 }}>UPPER 95%</th>
                        <th style={{ color: C.textDim, textAlign: "right", padding: "0 0 10px 10px", fontWeight: 800 }}>RANGE</th>
                    </tr>
                </thead>
                <tbody>
                    {tableRows.map((row) => {
                        const predicted = row.predicted ?? row.ensemble ?? row[modelKey];
                        const lower = row.lower_95 ?? row.lower_90;
                        const upper = row.upper_95 ?? row.upper_90;
                        const range = Number.isFinite(Number(upper)) && Number.isFinite(Number(lower)) ? Number(upper) - Number(lower) : null;
                        return (
                            <tr key={row.date} style={{ borderBottom: "1px solid rgba(255,255,255,.055)" }}>
                                <td style={{ color: C.textMid, padding: "10px 10px 10px 0", fontVariantNumeric: "tabular-nums" }}>{row.date}</td>
                                <td style={{ color: COLORS[modelKey] || COLORS.ensemble, textAlign: "right", padding: "10px", fontWeight: 800 }}>{formatPrice(predicted)}</td>
                                <td style={{ color: C.red, textAlign: "right", padding: "10px" }}>{formatPrice(lower)}</td>
                                <td style={{ color: C.green, textAlign: "right", padding: "10px" }}>{formatPrice(upper)}</td>
                                <td style={{ color: C.textMid, textAlign: "right", padding: "10px 0 10px 10px" }}>{formatPrice(range)}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default function PredictionsTab({ selectedTicker, apiConnected, priceData, modelPrep }) {
    const [horizon, setHorizon] = useState(30);
    const [selectedModel, setSelectedModel] = useState("all");
    const [showScenarios, setShowScenarios] = useState(true);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Preparation is App's, shared with the Analysis tab. `readyVersion` ticks
    // when a training run finishes, which is what re-runs the fetch below —
    // there is no button, and nothing here decides what gets trained.
    const readyVersion = modelPrep?.readyVersion ?? 0;
    // Only a live training run suppresses the normal loading state. The brief
    // readiness check on every ticker change must not flash a panel over a
    // forecast that is about to render perfectly well.
    const preparing = modelPrep?.status === "preparing";

    // Only the models the current view actually shows. A unified model is a
    // single next-day number that the fan chart has no room for, so selecting
    // one asks for that model alone; everything else asks for the fan.
    const activeModelKeys = useMemo(
        () => (isUnifiedModel(selectedModel) ? [selectedModel] : LEGACY_MODEL_KEYS),
        [selectedModel],
    );

    useEffect(() => {
        if (!apiConnected) return;
        setLoading(true);
        setError(null);
        Promise.allSettled([
            fetchEnsemblePrediction(selectedTicker, horizon),
            ...activeModelKeys.map((modelType) =>
                fetchPredictions(selectedTicker, modelType, isUnifiedModel(modelType) ? 1 : horizon),
            ),
        ])
            .then((results) => {
                const [ensembleResult, ...modelResults] = results;
                const models = {};
                const errors = {};
                activeModelKeys.forEach((modelType, index) => {
                    const result = modelResults[index];
                    if (result.status === "fulfilled") {
                        models[modelType] = normalizeSingleForecast(result.value, modelType);
                    } else {
                        models[modelType] = {
                            status: "unavailable",
                            model_available: false,
                            message: result.reason?.message || "This model produced no forecast for the current request.",
                            forecasts: [],
                            forecast_points: [],
                        };
                        errors[modelType] = models[modelType].message;
                    }
                });
                setData({
                    ensemblePayload: ensembleResult.status === "fulfilled" ? ensembleResult.value : null,
                    models,
                    errors,
                });
                setLoading(false);
            })
            .catch((err) => {
                setError(err?.message || "Forecast request failed.");
                setLoading(false);
            });
    }, [selectedTicker, horizon, apiConnected, readyVersion, activeModelKeys]);

    if (!apiConnected) {
        return (
            <div style={{ padding: 48, color: C.textDim, textAlign: "center" }}>
                Connect to the API server to view predictions.
            </div>
        );
    }

    const display = resolveDisplayData(data, selectedModel);
    const currentPrice = display.currentPrice ?? priceData?.bars?.[priceData.bars.length - 1]?.close;
    const modelUnavailable = !loading && !error && display.unavailable;
    const bullish = Number(display.changePct || 0) >= 0;
    const reliabilityColor = display.reliability === "Low" ? C.red : display.reliability === "Medium" ? C.amber : C.green;
    // A unified model answers for the next bar whatever the horizon selector says.
    const horizonLocked = isUnifiedModel(selectedModel);

    return (
        <div style={{ display: "grid", gap: 18, paddingBottom: 36 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap", alignItems: "end" }}>
                <div style={{ display: "grid", gap: 6 }}>
                    <div style={{ color: C.textDim, fontSize: 12, fontWeight: 800, letterSpacing: "0.06em", textTransform: "uppercase" }}>Predictions</div>
                    <div style={{ color: C.text, fontSize: 25, fontWeight: 900, lineHeight: 1 }}>{selectedTicker}</div>
                </div>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                    <select
                        value={selectedModel}
                        onChange={(event) => setSelectedModel(event.target.value)}
                        style={{
                            background: COLORS.surface,
                            color: C.text,
                            border: `1px solid ${C.border}`,
                            borderRadius: 8,
                            padding: "8px 10px",
                            fontSize: 13,
                            fontWeight: 700,
                            outline: "none",
                        }}
                    >
                        {MODEL_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                    </select>
                    <div style={{ display: "inline-flex", background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 3, gap: 3 }}>
                        {HORIZONS.map((days) => (
                            <button
                                key={days}
                                type="button"
                                disabled={horizonLocked}
                                title={horizonLocked ? "Unified models forecast the next bar only" : undefined}
                                onClick={() => setHorizon(days)}
                                style={{
                                    background: horizon === days ? COLORS.ensemble : "transparent",
                                    color: horizon === days ? "#10131A" : C.textMid,
                                    border: "none",
                                    borderRadius: 6,
                                    padding: "7px 12px",
                                    minWidth: 42,
                                    fontSize: 12,
                                    fontWeight: 900,
                                    cursor: horizonLocked ? "not-allowed" : "pointer",
                                    opacity: horizonLocked ? 0.5 : 1,
                                }}
                            >
                                {days}D
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Next-day direction sits above the multi-day forecast on purpose.
                It is the part with a measured out-of-sample track record and a
                costed backtest; the price cone below it is a single scalar per
                model with the intermediate days interpolated. Reading order
                should follow evidence, not horizon length. */}
            <DirectionPanel symbol={selectedTicker} modelPrep={modelPrep} />

            {/* One panel covers every reason there is nothing to draw, and the
                server decides which: training under way with its stages, a real
                failure with a retry, or a model that trained and did not clear
                its out-of-sample baseline. None of them is a button asking the
                user to start a training run.

                It renders while preparation is live even if a partial forecast
                is also on screen — a degraded ensemble filling in its missing
                member should say so while it happens. */}
            {(preparing || (!loading && !error && modelUnavailable)) && (
                <ModelPreparation preparation={modelPrep} context="forecast" />
            )}

            {!preparing && loading && (
                <div style={{ padding: 46, color: C.textDim, background: COLORS.surface, border: `1px solid ${C.border}`, borderRadius: 8, textAlign: "center" }}>
                    Loading forecast...
                </div>
            )}

            {!preparing && !loading && error && (
                <div style={{ padding: 18, color: C.red, background: "rgba(244,63,94,.08)", border: "1px solid rgba(244,63,94,.35)", borderRadius: 8 }}>
                    {error}
                </div>
            )}

            {!loading && !error && !modelUnavailable && display.payload && (
                <>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
                        <MetricCard
                            label="Current"
                            value={formatPrice(currentPrice)}
                            sub={display.currentPriceSource ? display.currentPriceSource.replace("_", " ") : null}
                        />
                        <MetricCard
                            label={`${modelLabel(display.chartModel)} ${horizonLocked ? "Next Bar" : "Forecast"}`}
                            value={formatPrice(display.target)}
                            sub={`${formatPct(display.changePct)} ${display.signal || ""} | ${display.reliability || "Model"}`}
                            color={bullish ? COLORS.ensemble : C.red}
                        />
                        {Number.isFinite(Number(display.probabilityUp)) ? (
                            <MetricCard
                                label="Direction"
                                value={display.probabilityUp >= 0.5 ? "UP" : "DOWN"}
                                sub={`Up ${(display.probabilityUp * 100).toFixed(1)}% | Down ${((display.probabilityDown ?? 1 - display.probabilityUp) * 100).toFixed(1)}%`}
                                color={display.probabilityUp >= 0.5 ? C.green : C.red}
                            />
                        ) : (
                            <MetricCard label="Upper 95%" value={formatPrice(display.upper95)} color={C.green} />
                        )}
                        <MetricCard
                            label={horizonLocked ? "Forecast Horizon" : "Lower 95%"}
                            value={horizonLocked ? "Next 1 Bar" : formatPrice(display.lower95)}
                            color={horizonLocked ? COLORS.band : C.red}
                        />
                    </div>

                    <ForecastChart
                        priceData={priceData}
                        forecastPoints={(display.points || []).slice(0, horizon)}
                        selectedModel={display.chartModel}
                        scenarioPaths={display.scenarioPaths}
                        showScenarios={showScenarios}
                        onToggleScenarios={() => setShowScenarios((on) => !on)}
                        interpolated={
                            display.perStepPredictions === false ||
                            (display.perStepPredictions == null &&
                                display.pathType !== "recursive_per_step")
                        }
                    />

                    <ForecastProvenanceNote
                        pathType={display.pathType}
                        perStepPredictions={display.perStepPredictions}
                        modelOutputCount={display.modelOutputCount}
                        horizon={horizon}
                        scenarioCount={showScenarios ? (display.scenarioPaths?.length ?? 0) : 0}
                    />

                    {display.headsNote && (
                        <div style={{
                            padding: "8px 12px",
                            borderRadius: 8,
                            background: "rgba(245, 200, 66, 0.08)",
                            border: `1px solid ${COLORS.ensemble}44`,
                            color: C.sub,
                            fontSize: 12,
                        }}>
                            {display.headsNote}
                        </div>
                    )}

                    {display.degraded && display.modelsUnavailable && (
                        <div style={{
                            padding: "8px 12px",
                            borderRadius: 8,
                            border: `1px solid ${C.amber || "#8a6d3b"}`,
                            background: "rgba(255,255,255,0.03)",
                            color: C.sub,
                            fontSize: 12,
                            lineHeight: 1.5,
                        }}>
                            <span style={{ fontWeight: 900, color: C.amber || "#d6a13a", marginRight: 8 }}>
                                Partial ensemble
                            </span>
                            Served by {(display.modelsAvailable || []).map((m) => MODEL_LABELS[m] || m).join(", ")}.{" "}
                            {Object.entries(display.modelsUnavailable)
                                .map(([m, why]) => `${MODEL_LABELS[m] || m} excluded — ${why}`)
                                .join("; ")}.
                        </div>
                    )}

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 14, alignItems: "start" }}>
                        <WeightsPanel
                            weights={display.payload?.weights}
                            consensus={display.consensus}
                            signal={display.signal}
                            reliability={display.reliability}
                        />
                        <div style={{ display: "grid", gap: 10 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
                                <div style={{ color: C.text, fontSize: 13, fontWeight: 900 }}>Forecast Table</div>
                                <div style={{ color: reliabilityColor, fontSize: 12, fontWeight: 800 }}>
                                    {display.reliability || "Model"}
                                </div>
                            </div>
                            <ForecastTable
                                rows={display.points || []}
                                modelKey={display.chartModel === "all" ? "ensemble" : display.chartModel}
                                label={`${display.tableLabel || "Predicted"} Price`}
                            />
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
