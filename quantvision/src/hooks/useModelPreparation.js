/**
 * One symbol's model readiness, shared by every tab that renders model output.
 *
 * The rule this hook exists to enforce: **the frontend never trains anything and
 * never decides what to train.** It asks the server what a symbol can serve, asks
 * it to prepare the symbol when the answer is "not everything", and polls until
 * that finishes. Which bundles are missing, whether a walk-forward run is needed,
 * and whether retraining would even help are all server-side judgements.
 *
 * It lives in App so Predictions and Analysis observe the same preparation rather
 * than each starting their own. `readyVersion` is the signal panels depend on:
 * it increments when preparation completes, which is their cue to refetch.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, fetchModelReadiness, getPreparationStatus, prepareModels } from "../utils/api";

/** How often to ask a running job how it is going. */
const POLL_INTERVAL_MS = 2500;

/**
 * Stop polling after this long. Training keeps running on the server — this is a
 * limit on how long the page reports on it, not on the job. A walk-forward run
 * plus fifteen bundles can take a while on a cold symbol, so the ceiling is
 * generous; past it the UI says the work continues and to come back.
 */
const POLL_TIMEOUT_MS = 45 * 60 * 1000;

const IDLE = {
    status: "idle",
    readiness: null,
    job: null,
    error: null,
};

/** Job states that mean work is still happening. */
const ACTIVE_JOB_STATES = new Set(["queued", "running"]);

/** Stage states that mean that stage will not produce anything more. */
const SETTLED_STAGE_STATES = new Set(["done", "skipped", "failed"]);

const countSettledStages = (job) =>
    (job?.stages || []).filter((stage) => SETTLED_STAGE_STATES.has(stage.state)).length;

export function isPreparing(state) {
    return state?.status === "preparing";
}

/**
 * Everything the UI needs to decide between "show results", "show progress", and
 * "show an error with a retry".
 *
 * Returns:
 *   status       "idle" | "checking" | "ready" | "preparing" | "error"
 *   readiness    the server's component-by-component verdict
 *   job          the preparation job, while one is attached
 *   stages       job stages, ready to render as a checklist
 *   readyVersion increments when preparation finishes — refetch on this
 *   retry()      force a fresh attempt, bypassing the server's cooldown
 */
export function useModelPreparation(symbol, { enabled = true } = {}) {
    const [state, setState] = useState(IDLE);
    const [readyVersion, setReadyVersion] = useState(0);

    // Guards against a slow response for a previous ticker landing after the
    // user has moved on and overwriting the current one.
    const requestRef = useRef(0);
    const timerRef = useRef(null);
    // How many stages had finished at the last refresh signal. The server runs
    // the cheap direction evaluation before the long bundle training, so its
    // result is servable minutes before the job is — waiting for completion
    // would leave the finished panel showing a progress bar the whole time.
    const settledStagesRef = useRef(0);

    const stopPolling = useCallback(() => {
        clearTimeout(timerRef.current);
        timerRef.current = null;
    }, []);

    /** Poll one job to completion, then bump readyVersion so panels refetch. */
    const watchJob = useCallback((jobId, token, startedAt) => {
        stopPolling();
        timerRef.current = setTimeout(async () => {
            if (token !== requestRef.current) return;
            try {
                const job = await getPreparationStatus(jobId);
                if (token !== requestRef.current) return;

                if (ACTIVE_JOB_STATES.has(job.status)) {
                    // A stage finishing means new artifacts are on disk. Signal a
                    // refetch now rather than at the end of the job, so the panel
                    // whose data just landed stops showing progress.
                    const settled = countSettledStages(job);
                    if (settled > settledStagesRef.current) {
                        settledStagesRef.current = settled;
                        setReadyVersion((version) => version + 1);
                    }

                    if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
                        setState((prev) => ({
                            ...prev,
                            status: "preparing",
                            job: { ...job, stalledReport: true },
                        }));
                        return;
                    }
                    setState((prev) => ({ ...prev, status: "preparing", job, error: null }));
                    watchJob(jobId, token, startedAt);
                    return;
                }

                if (job.status === "failed") {
                    setState((prev) => ({ ...prev, status: "error", job, error: job.error }));
                    return;
                }

                // Completed. The job carries the readiness it measured after
                // training, so the panels get the post-training truth without a
                // second round trip.
                setState({
                    status: "ready",
                    readiness: job.readiness ?? null,
                    job,
                    error: null,
                });
                setReadyVersion((version) => version + 1);
            } catch (err) {
                if (token !== requestRef.current) return;
                // A job that aged out of the tracker is not a failure to report —
                // re-check readiness and let the answer speak for itself.
                if (err instanceof ApiError && err.status === 404) {
                    setReadyVersion((version) => version + 1);
                    setState((prev) => ({ ...prev, status: "ready", job: null }));
                    return;
                }
                setState((prev) => ({ ...prev, status: "error", error: err.message }));
            }
        }, POLL_INTERVAL_MS);
    }, [stopPolling]);

    const check = useCallback(async ({ force = false } = {}) => {
        const token = ++requestRef.current;
        stopPolling();
        settledStagesRef.current = 0;
        setState((prev) => ({ ...prev, status: "checking", error: null }));

        try {
            // A forced attempt skips straight to the POST: the user has read an
            // error and asked again, and a readiness check in between would only
            // report the same gap they already saw.
            const payload = force
                ? await prepareModels(symbol, { force: true })
                : await fetchModelReadiness(symbol);
            if (token !== requestRef.current) return;

            const readiness = payload.readiness ?? null;
            let job = payload.preparation ?? null;

            // Ask the server to close the gap. It decides whether anything can
            // actually be done — a symbol whose models trained and failed their
            // skill gate comes back with no job, and that is a final answer.
            if (!force && readiness?.needs_training) {
                const started = await prepareModels(symbol);
                if (token !== requestRef.current) return;
                job = started.preparation ?? null;
            }

            if (job && ACTIVE_JOB_STATES.has(job.status)) {
                setState({ status: "preparing", readiness, job, error: null });
                watchJob(job.job_id, token, Date.now());
                return;
            }

            if (job && job.status === "failed") {
                setState({ status: "error", readiness, job, error: job.error });
                return;
            }

            setState({ status: "ready", readiness, job, error: null });
        } catch (err) {
            if (token !== requestRef.current) return;
            setState({ status: "error", readiness: null, job: null, error: err.message });
        }
    }, [symbol, stopPolling, watchJob]);

    useEffect(() => {
        if (!enabled || !symbol) {
            setState(IDLE);
            return undefined;
        }
        check();
        return () => {
            // Invalidate in-flight work for the symbol being left behind.
            requestRef.current += 1;
            stopPolling();
        };
    }, [symbol, enabled, check, stopPolling]);

    const retry = useCallback(() => check({ force: true }), [check]);

    return {
        ...state,
        symbol,
        stages: state.job?.stages ?? [],
        progress: state.job?.progress ?? 0,
        warnings: state.job?.warnings ?? [],
        readyVersion,
        retry,
    };
}

export default useModelPreparation;
