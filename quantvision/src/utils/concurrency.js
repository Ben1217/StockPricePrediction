/**
 * Bounded-concurrency fan-out.
 *
 * Lives here because two tabs need the same guarantee for the same reason: a
 * per-symbol model call is roughly a second of server-side pandas and model
 * fitting, so firing sixteen at once buries a threadpool that is also serving
 * the rest of the page, and firing them serially is a minute of blank screen.
 *
 * `fn` is called for its side effects — the caller reports each result as it
 * lands, which is what lets a grid fill in visibly rather than appearing all at
 * once when the slowest symbol finishes. A rejection is swallowed per item on
 * purpose: one dead ticker must not abort the other fifteen, and the caller
 * records the failure in its own state.
 */
export async function eachLimited(items, limit, fn) {
    let cursor = 0;
    const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
        while (cursor < items.length) {
            const item = items[cursor++];
            try { await fn(item); } catch { /* per-item failure is reported in caller state */ }
        }
    });
    await Promise.all(workers);
}
