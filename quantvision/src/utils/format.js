/**
 * Number formatting shared by every tab and panel.
 *
 * These lived as private consts in eight files, and they had already diverged in
 * three ways that a reader could not see from the call site:
 *
 *   1. **Unit.** `signedPct` meant "this value is already a percentage" in the
 *      Heatmap, Optimization, Predictions and Backtest tabs, and "this value is a
 *      fraction of one, multiply it" in the Portfolio tab and the direction
 *      panels. Same name, same signature, answers a hundred times apart. The
 *      names below say which they take — `pct`/`signedPct` scale a fraction,
 *      `pctPoints`/`signedPctPoints` do not — so the unit is a decision the
 *      caller makes visibly rather than one they inherit from a file.
 *   2. **Coercion.** Half the copies guarded with `Number.isFinite(Number(v))`,
 *      which accepts the string "12.5"; the other half with `typeof v ===
 *      "number"`, which renders it as a dash. The API sends numbers, so the
 *      coercing form is the one kept — it is the strictly more forgiving of the
 *      two and never turns a displayable value into a dash.
 *   3. **Grouping.** `money` was `$1234.00` in two files and `$1,234.00` in five.
 *      Both are here, named for what they do, because a dense heatmap tile and a
 *      portfolio holdings row genuinely want different things.
 *
 * Every function answers `DASH` for anything it cannot render — null, undefined,
 * NaN, Infinity, a non-numeric string — so a caller never has to pre-check.
 */

/** Shown in place of a value that is missing or not finite. */
export const DASH = "—";

/** Finite number, or null. The one coercion rule the rest of this module shares. */
function finite(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

/** `1234.5` → `"$1,234.50"`. Grouped; for prose, tables and holdings rows. */
export function money(value) {
    const number = finite(value);
    return number === null
        ? DASH
        : `$${number.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/** `1234.5` → `"$1234.50"`. Ungrouped; for narrow cells where a comma costs a character. */
export function moneyPlain(value) {
    const number = finite(value);
    return number === null ? DASH : `$${number.toFixed(2)}`;
}

/** `1234.5` → `"$1.2K"`. For axis ticks and chips. */
export function moneyCompact(value) {
    const number = finite(value);
    if (number === null) return DASH;
    return `$${Math.abs(number) >= 1000 ? `${(number / 1000).toFixed(1)}K` : number.toFixed(0)}`;
}

/** `0.0734` → `"7.3%"`. Takes a **fraction of one**. */
export function pct(value, digits = 1) {
    const number = finite(value);
    return number === null ? DASH : `${(number * 100).toFixed(digits)}%`;
}

/** `0.0734` → `"+7.34%"`. Takes a **fraction of one**. */
export function signedPct(value, digits = 2) {
    const number = finite(value);
    return number === null ? DASH : `${number >= 0 ? "+" : ""}${(number * 100).toFixed(digits)}%`;
}

/** `7.34` → `"7.3%"`. Takes a value that is **already a percentage**. */
export function pctPoints(value, digits = 1) {
    const number = finite(value);
    return number === null ? DASH : `${number.toFixed(digits)}%`;
}

/** `7.34` → `"+7.34%"`. Takes a value that is **already a percentage**. */
export function signedPctPoints(value, digits = 2) {
    const number = finite(value);
    return number === null ? DASH : `${number >= 0 ? "+" : ""}${number.toFixed(digits)}%`;
}

/** `1.2345` → `"1.23"`. */
export function num(value, digits = 2) {
    const number = finite(value);
    return number === null ? DASH : number.toFixed(digits);
}
