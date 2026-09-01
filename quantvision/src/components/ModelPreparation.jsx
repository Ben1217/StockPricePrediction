/**
 * What the user sees while the backend gets a symbol's models ready.
 *
 * Three states, and the distinction between them is the whole point:
 *
 *   preparing   work is running — show the stages, not an error
 *   error       preparation actually failed — show the server's reason and a retry
 *   settled     preparation finished and some models still cannot serve — show
 *               *why*, which here is a measurement rather than a missing file
 *
 * The third case is the one worth being careful about. A bundle that trained and
 * then failed its out-of-sample skill gate is not an unfinished setup step, and
 * offering a "train" button for it would invite the user to run the same fit for
 * the same verdict. So this component never renders a training button: preparation
 * is the server's job, and retry exists only for genuine failures.
 */

import { C } from "../utils/data";

const STATE_MARKS = {
    done: { glyph: "✓", color: C.green },
    skipped: { glyph: "✓", color: C.textDim },
    running: { glyph: "●", color: C.amber },
    failed: { glyph: "✕", color: C.red },
    pending: { glyph: "○", color: C.textDim },
};

function StageRow({ stage }) {
    const mark = STATE_MARKS[stage.state] || STATE_MARKS.pending;
    const active = stage.state === "running";
    return (
        <li style={{
            display: "flex", alignItems: "baseline", gap: 10,
            padding: "5px 0", listStyle: "none",
        }}>
            <span style={{
                color: mark.color, fontFamily: "'DM Mono',monospace", fontSize: 13, width: 14,
                animation: active ? "pulse 1.2s ease-in-out infinite" : undefined,
            }}>
                {mark.glyph}
            </span>
            <span style={{
                color: active ? C.text : stage.state === "pending" ? C.textDim : C.textMid,
                fontSize: 12.5, fontWeight: active ? 700 : 500,
            }}>
                {stage.label}
            </span>
            {stage.detail && (
                <span style={{ color: C.textDim, fontSize: 11, fontFamily: "'DM Mono',monospace" }}>
                    {stage.detail}
                </span>
            )}
        </li>
    );
}

function ProgressBar({ value }) {
    return (
        <div style={{
            height: 4, borderRadius: 2, background: C.bg3, overflow: "hidden", marginTop: 4,
        }}>
            <div style={{
                height: "100%", width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%`,
                background: C.amber, transition: "width .4s ease",
            }} />
        </div>
    );
}

function Frame({ children, tone = C.border }) {
    return (
        <div style={{
            padding: "18px 20px", background: C.bg1, border: `1px solid ${tone}`,
            borderRadius: 8, display: "grid", gap: 12,
        }}>
            {children}
        </div>
    );
}

/**
 * @param {object}   preparation  the useModelPreparation() return value
 * @param {string}   context      what the caller was trying to show ("forecast")
 */
export default function ModelPreparation({ preparation, context = "models" }) {
    const { status, symbol, stages, progress, error, warnings, readiness, job, retry } = preparation || {};

    if (status === "checking" || status === "idle") {
        return (
            <Frame>
                <div style={{ color: C.textMid, fontSize: 13 }}>
                    Checking {symbol} models…
                </div>
            </Frame>
        );
    }

    if (status === "preparing") {
        const stalled = job?.stalledReport;
        return (
            <Frame tone={`${C.amber}55`}>
                <div>
                    <div style={{ color: C.text, fontSize: 16, fontWeight: 800, marginBottom: 4 }}>
                        Preparing {symbol} model{context === "models" ? "s" : ` ${context}`}…
                    </div>
                    <div style={{ color: C.textMid, fontSize: 12 }}>
                        Training runs on the server. This view updates on its own when it finishes.
                    </div>
                    <ProgressBar value={progress} />
                </div>
                <ul style={{ margin: 0, padding: 0 }}>
                    {stages.map((stage) => <StageRow key={stage.name} stage={stage} />)}
                </ul>
                {stalled && (
                    <div style={{ color: C.textDim, fontSize: 11.5, lineHeight: 1.6 }}>
                        Still training after 45 minutes. The job is unaffected by this page —
                        reload later to pick up the result.
                    </div>
                )}
            </Frame>
        );
    }

    if (status === "error") {
        return (
            <Frame tone={`${C.red}55`}>
                <div>
                    <div style={{ color: C.text, fontSize: 16, fontWeight: 800, marginBottom: 4 }}>
                        Unable to prepare {symbol} models
                    </div>
                    <div style={{ color: C.red, fontSize: 12.5, lineHeight: 1.6 }}>
                        {error || "Preparation failed for an unknown reason."}
                    </div>
                </div>
                {stages.length > 0 && (
                    <ul style={{ margin: 0, padding: 0 }}>
                        {stages.map((stage) => <StageRow key={stage.name} stage={stage} />)}
                    </ul>
                )}
                <button
                    type="button"
                    onClick={retry}
                    style={{
                        justifySelf: "start", background: C.amber, color: "#10131A", border: "none",
                        borderRadius: 6, padding: "8px 16px", fontSize: 12.5, fontWeight: 800,
                        cursor: "pointer",
                    }}
                >
                    Retry
                </button>
            </Frame>
        );
    }

    // Preparation is not running, and the caller still has nothing to render.
    // Three different situations hide behind that, and telling them apart is the
    // difference between an honest result and a silent dead end.
    const blocked = readiness?.blocked || [];
    const autoDisabled = readiness?.auto_prepare === false;
    const workRemains = Boolean(readiness?.needs_training);
    // Work is outstanding but nothing is running: either the server's cooldown
    // is holding a repeat attempt back, or automatic preparation is switched off.
    // Both are worth a Retry, which forces past the cooldown.
    const stalled = workRemains && !autoDisabled;

    return (
        <Frame tone={stalled ? `${C.amber}44` : C.border}>
            <div>
                <div style={{ color: C.text, fontSize: 16, fontWeight: 800, marginBottom: 4 }}>
                    No {context} for {symbol}
                </div>
                <div style={{ color: C.textMid, fontSize: 12.5, lineHeight: 1.7 }}>
                    {autoDisabled
                        ? `Automatic preparation is switched off on this server, and ${symbol} still needs models trained.`
                        : readiness?.summary
                            || "The models for this symbol have been trained; none of them cleared their out-of-sample baseline."}
                </div>
            </div>

            {blocked.length > 0 && (
                <ul style={{ margin: 0, paddingLeft: 18, color: C.textDim, fontSize: 11.5, lineHeight: 1.8 }}>
                    {blocked.slice(0, 6).map((item) => (
                        <li key={item.key}>{item.detail || item.key}</li>
                    ))}
                </ul>
            )}

            {warnings?.length > 0 && (
                <div style={{ color: C.amber, fontSize: 11.5, lineHeight: 1.7 }}>
                    {warnings.slice(0, 3).map((warning, index) => <div key={index}>{warning}</div>)}
                </div>
            )}

            {stalled ? (
                <>
                    <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.6 }}>
                        The last run finished without closing every gap, so the server is
                        holding off on repeating it. Retry starts a fresh attempt now.
                    </div>
                    <button
                        type="button"
                        onClick={retry}
                        style={{
                            justifySelf: "start", background: C.amber, color: "#10131A", border: "none",
                            borderRadius: 6, padding: "8px 16px", fontSize: 12.5, fontWeight: 800,
                            cursor: "pointer",
                        }}
                    >
                        Retry
                    </button>
                </>
            ) : (
                <div style={{ color: C.textDim, fontSize: 11, lineHeight: 1.6 }}>
                    {autoDisabled
                        ? "An operator can start one with POST /api/models/" + symbol + "/prepare."
                        : "Withholding a forecast a model could not justify is the intended outcome here, "
                          + "not a setup step you still have to run — retraining the same history would "
                          + "reproduce the same verdict."}
                </div>
            )}
        </Frame>
    );
}
