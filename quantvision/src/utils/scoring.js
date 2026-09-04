/**
 * Scoring the model's dated calls against what the market did next.
 *
 * Lives outside the tab because it is the number a reader trusts, and a
 * pure function can be checked without mounting React. It was extracted
 * after a direction-vocabulary bug survived a build, a lint pass and a
 * manual review inside the component.
 */

/**
 * Score the model's dated calls against the bar that followed each one.
 *
 * `signals` are what the model said on a given date; `priceSeries` is the run's
 * own closes. A call is only counted when both its own bar and the next one are
 * in the series — a signal on the last bar has no outcome yet, and dropping it
 * is the difference between a hit rate and a guess.
 */
export function scoreSignals(signals, priceSeries) {
    if (!Array.isArray(signals) || !signals.length) return null;
    if (!Array.isArray(priceSeries) || priceSeries.length < 2) return null;

    const closeByDate = new Map();
    const dates = [];
    priceSeries.forEach((point) => {
        const close = Number(point?.close);
        if (!point?.date || !Number.isFinite(close)) return;
        closeByDate.set(point.date, close);
        dates.push(point.date);
    });
    // ISO dates sort lexicographically, so this is the calendar order.
    dates.sort();

    const nextDate = new Map();
    for (let index = 0; index < dates.length - 1; index += 1) {
        nextDate.set(dates[index], dates[index + 1]);
    }

    const rows = [];
    signals.forEach((signal) => {
        const close = closeByDate.get(signal?.date);
        const following = nextDate.get(signal?.date);
        if (close == null || !following) return;
        const nextClose = closeByDate.get(following);
        if (nextClose == null || close === 0) return;

        const probability = Number(signal.probability_up);
        const label = String(signal.direction || "").trim().toUpperCase();

        // `probability_up` leads because every other field on this signal was
        // derived from it, and it is a number rather than a vocabulary.
        //
        // The label is worth spelling out: this endpoint renders direction as
        // "Bullish"/"Bearish" (direction_from_probability), while the forecast
        // endpoint renders it as "UP"/"DOWN". An earlier version of this file
        // compared only against "UP", which is never equal to "Bullish" — so
        // every call was silently scored as a down-call and the hit rate below
        // measured the opposite of what it claimed.
        const predictedUp = Number.isFinite(probability)
            ? probability >= 0.5
            : label === "UP" || label === "BULLISH"
                ? true
                : label === "DOWN" || label === "BEARISH"
                    ? false
                    : signal.type === "BUY";

        const actualPct = ((nextClose - close) / close) * 100;
        const actualUp = nextClose >= close;

        rows.push({
            date: signal.date,
            predictedUp,
            actualUp,
            actualPct,
            hit: predictedUp === actualUp,
            probability: Number.isFinite(probability) ? probability : null,
        });
    });

    if (!rows.length) return null;
    rows.sort((a, b) => (a.date < b.date ? -1 : 1));
    const hits = rows.filter((row) => row.hit).length;
    const hitRate = (hits / rows.length) * 100;

    // What the market actually did over the same bars, and what a model that
    // simply always called the commoner direction would have scored. A hit rate
    // shown without this is unreadable: on a window that rose 56% of the time,
    // 52% correct is a losing record, not a winning one.
    const upDays = rows.filter((row) => row.actualUp).length;
    const baseRate = (upDays / rows.length) * 100;
    const majority = Math.max(baseRate, 100 - baseRate);

    return {
        rows,
        hits,
        total: rows.length,
        hitRate,
        baseRate,
        majority,
        eobr: hitRate - majority,
    };
}
