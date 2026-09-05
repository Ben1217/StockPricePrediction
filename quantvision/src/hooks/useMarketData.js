/**
 * Shared market-data queries.
 *
 * Every tab reads through these hooks so the same symbol is fetched once and
 * served from cache everywhere else — concurrent identical requests are
 * de-duplicated by TanStack Query rather than hitting the API N times.
 */
import { useQuery } from "@tanstack/react-query";
import {
    ApiError,
    fetchBestModelForecast,
    fetchHealth,
    fetchIndicators,
    fetchPrices,
    fetchQuotes,
    fetchDataSources,
    fetchBacktestEvidence,
    fetchForecastHistory,
    fetchHistoricalSignals,
    fetchSimpleForecast,
    fetchPortfolioMetrics,
    fetchCorrelation,
    fetchSP500,
    fetchDirectionAnalysis,
    fetchSupportResistance,
} from "../utils/api";

/** Query keys live in one place so cache invalidation stays greppable. */
export const qk = {
    health: ["health"],
    sources: ["data", "sources"],
    sp500: ["data", "sp500"],
    quotes: (symbols) => ["data", "quotes", [...symbols].sort().join(",")],
    prices: (symbol, source, days, interval) => ["data", "prices", symbol, source, days, interval],
    indicators: (symbol, days, interval) => ["data", "indicators", symbol, days, interval],
    bestForecast: (symbol, horizon, readyVersion) => ["predict", "best", symbol, horizon, readyVersion],
    forecastHistory: (symbol, days) => ["predict", "history", symbol, days],
    simpleForecast: (symbol) => ["predict", "forecast", symbol],
    directionAnalysis: (symbol, model) => ["direction", "analysis", symbol, model],
    supportResistance: (symbol, interval, lookback) =>
        ["patterns", "support-resistance", symbol, interval, lookback],
    historicalSignals: (symbol, days, modelType) => ["predict", "historical-signals", symbol, days, modelType],
    backtestEvidence: (symbol, model) => ["backtest", "evidence", symbol, model ?? "all"],
    portfolioMetrics: (symbols, lookback, attribution) =>
        ["portfolio", "metrics", [...symbols].sort().join(","), lookback, attribution],
    correlation: (symbols, lookbackDays) =>
        ["portfolio", "correlation", [...symbols].sort().join(","), lookbackDays],
};

export function useApiHealth() {
    const query = useQuery({
        queryKey: qk.health,
        queryFn: ({ signal }) => fetchHealth({ signal }),
        retry: 0,
        staleTime: 30_000,
    });
    return { ...query, connected: query.isSuccess };
}

/**
 * The S&P 500 constituent list, mapped to the shape the stock picker renders.
 *
 * Cached for the session: index membership changes a handful of times a year,
 * and the server re-scrapes it on a 24h cache of its own, so re-requesting it
 * on every mount of the search modal buys nothing.
 *
 * Failure is not surfaced. The only caller has the bundled full-index snapshot
 * to fall back on, so an unreachable API costs freshness rather than the
 * picker — which is why this never retries into an error state the UI would
 * then have to explain.
 */
export function useSP500(enabled = true) {
    return useQuery({
        queryKey: qk.sp500,
        queryFn: async () => {
            const rows = await fetchSP500();
            return rows
                .filter((row) => row?.symbol)
                .map((row) => ({
                    ticker: row.symbol,
                    name: row.company || row.symbol,
                    sector: row.sector || "",
                }));
        },
        enabled,
        staleTime: Infinity,
        gcTime: Infinity,
        retry: 1,
    });
}

export function useDataSources(enabled = true) {
    return useQuery({
        queryKey: qk.sources,
        queryFn: () => fetchDataSources(),
        enabled,
        staleTime: 10 * 60_000,
    });
}

/** One batched request for a list of symbols. */
export function useQuotes(symbols, enabled = true) {
    const list = Array.isArray(symbols) ? symbols : [];
    return useQuery({
        queryKey: qk.quotes(list),
        queryFn: ({ signal }) => fetchQuotes(list, { signal }),
        enabled: enabled && list.length > 0,
        // Quotes go stale faster than the reference data above.
        staleTime: 30_000,
    });
}

export function usePrices(symbol, { source = "yfinance", days = 120, interval = "1d", enabled = true } = {}) {
    return useQuery({
        queryKey: qk.prices(symbol, source, days, interval),
        queryFn: () => fetchPrices(symbol, source, days, interval),
        enabled: enabled && Boolean(symbol),
    });
}

export function useIndicators(symbol, { days = 120, interval = "1d", enabled = true } = {}) {
    return useQuery({
        queryKey: qk.indicators(symbol, days, interval),
        queryFn: () => fetchIndicators(symbol, days, interval),
        enabled: enabled && Boolean(symbol),
    });
}

/**
 * The best performing model's forecast for the chart overlay.
 *
 * The horizon is part of the key, so switching the 7D/15D/30D/60D selector is a
 * different query rather than a refetch that briefly leaves the previous
 * horizon's line on screen. `readyVersion` is in there too: it ticks when a
 * preparation run finishes, and a symbol that had nothing servable a minute ago
 * may now have a winner.
 *
 * The response is a ranking plus a model run, so it is worth a longer stale
 * time than a quote: the bundles it reads only change when training does, and
 * daily bars change once a day.
 */
export function useBestModelForecast(symbol, { horizon = 30, readyVersion = 0, enabled = true } = {}) {
    return useQuery({
        queryKey: qk.bestForecast(symbol, horizon, readyVersion),
        queryFn: () => fetchBestModelForecast(symbol, horizon),
        enabled: enabled && Boolean(symbol),
        staleTime: 5 * 60_000,
        retry: 0,
    });
}

/**
 * The forecast for one symbol. The candles are `useForecastHistory`.
 *
 * The symbol alone is the key: the forecast is always the next bar, so no
 * control on the tab varies it. The chart range in particular must not — it
 * would re-run three models to redraw candles the client already holds.
 *
 * Retries transport failures only, and nothing the server answered.
 * -----------------------------------------------------------------
 * An `ApiError` means the request reached the server and the server replied.
 * Those replies are deterministic here — an unservable symbol is a 200 with
 * `status: "unavailable"`, and a 4xx (unknown ticker, too little history) comes
 * out the same way every time — so repeating one buys nothing and costs a
 * second run of three transformer models.
 *
 * A rejection that is *not* an `ApiError` never got an HTTP response at all:
 * `fetch` rejects with a TypeError when the network moves under it
 * (ERR_NETWORK_CHANGED on a VPN or Wi-Fi switch), when the machine is offline,
 * or when the API server is not up. Nothing was computed, so retrying is free,
 * and not retrying is what leaves the tab stuck: `refetchOnWindowFocus` is off
 * globally, so a latched error survives until the symbol or range changes.
 *
 * This is deliberately narrower than the client-wide default in main.jsx, which
 * retries any non-4xx twice — including 5xx, which for this endpoint would mean
 * paying for the model run again to get the same failure.
 */
const reachedTheServer = (error) => error instanceof ApiError;

/**
 * The candles, on their own query so the chart renders without the forecast.
 *
 * Kept generous and constant (`days`) so the range buttons slice a frame that
 * is already in the cache instead of issuing a request per range.
 */
export function useForecastHistory(symbol, { days = 252, enabled = true } = {}) {
    return useQuery({
        queryKey: qk.forecastHistory(symbol, days),
        queryFn: () => fetchForecastHistory(symbol, days),
        enabled: enabled && Boolean(symbol),
        staleTime: 5 * 60_000,
        retry: (failureCount, error) => !reachedTheServer(error) && failureCount < 2,
    });
}

export function useSimpleForecast(symbol, { enabled = true } = {}) {
    return useQuery({
        queryKey: qk.simpleForecast(symbol),
        queryFn: () => fetchSimpleForecast(symbol),
        enabled: enabled && Boolean(symbol),
        staleTime: 5 * 60_000,
        retry: (failureCount, error) => !reachedTheServer(error) && failureCount < 2,
        retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
    });
}

/**
 * The model's recent dated calls — what it said, day by day, before the outcome
 * was known.
 *
 * This is the series the Backtest tab scores itself against: each signal is
 * joined to the bar after it, so "predicted" and "actual" sit on the same dates.
 * A 404 is the ordinary answer for a symbol whose bundle has not been trained
 * yet, so it is never retried — the panel reads the status and says so.
 */
/**
 * The measured direction call: seven evidence categories, a fitted weight each,
 * and the out-of-sample record that decides whether any of it is trusted.
 *
 * This is the only direction on the page that has been scored. The foundation
 * forecast from `useSimpleForecast` produces a probability too, but nothing has
 * ever walk-forwarded it, so the two are not interchangeable and a caller must
 * not present them as one number.
 *
 * NEUTRAL is a normal, frequent answer here — it is what the stack returns when
 * neither it nor the classifier beat their own base rate out of sample — and
 * `neutral_reason` carries the measurement behind it. Treating that as an error
 * state would hide the most honest thing the system says.
 *
 * A 200 with `status: "unavailable"` means too little history for a
 * walk-forward, which is also an answer rather than a failure.
 *
 * Not retried against the server: the walk-forward is deterministic on a given
 * last bar and costs seconds, so a repeat asks for the identical number twice.
 */
export function useDirectionAnalysis(symbol, { model = "logistic", enabled = true } = {}) {
    return useQuery({
        queryKey: qk.directionAnalysis(symbol, model),
        queryFn: () => fetchDirectionAnalysis(symbol, { model }),
        enabled: enabled && Boolean(symbol),
        staleTime: 10 * 60_000,
        retry: (failureCount, error) => !reachedTheServer(error) && failureCount < 2,
    });
}

/**
 * Pivot-derived support and resistance for one symbol.
 *
 * Deliberately the same endpoint the Technical Analysis tab draws, so a level
 * quoted beside a forecast is the level the user already saw on the chart.
 * Computing a second set here would put two different numbers for "support"
 * in one app.
 *
 * Often returns fewer than two levels — a symbol mid-range with no confirmed
 * pivot above it has a support and no resistance — so callers render what is
 * there rather than assuming a pair.
 */
export function useSupportResistance(symbol, { interval = "1d", lookback = 180, enabled = true } = {}) {
    return useQuery({
        queryKey: qk.supportResistance(symbol, interval, lookback),
        queryFn: () => fetchSupportResistance(symbol, interval, lookback),
        enabled: enabled && Boolean(symbol),
        staleTime: 10 * 60_000,
        retry: (failureCount, error) => !reachedTheServer(error) && failureCount < 2,
    });
}

export function useHistoricalSignals(symbol, { days = 365, modelType = "xgboost", enabled = true } = {}) {
    return useQuery({
        queryKey: qk.historicalSignals(symbol, days, modelType),
        queryFn: () => fetchHistoricalSignals(symbol, days, modelType),
        enabled: enabled && Boolean(symbol) && Boolean(modelType),
        staleTime: 5 * 60_000,
        retry: (failureCount, error) => !reachedTheServer(error) && failureCount < 2,
    });
}

/**
 * The benchmark's walk-forward record for one symbol.
 *
 * Read-only and cheap — it is a JSON file on the server, not a model run — so
 * it is safe to fetch alongside a backtest rather than behind it. An empty
 * `models` array is a normal answer, and the payload's own `message` explains
 * which case it is.
 */
export function useBacktestEvidence(symbol, { model, enabled = true } = {}) {
    return useQuery({
        queryKey: qk.backtestEvidence(symbol, model),
        queryFn: () => fetchBacktestEvidence(symbol, model),
        enabled: enabled && Boolean(symbol),
        staleTime: 5 * 60_000,
        retry: (failureCount, error) => !reachedTheServer(error) && failureCount < 2,
    });
}

/**
 * Per-stock realised return, volatility and Sharpe for a whole list, in one call.
 *
 * The heatmap's entire backward-looking half comes from here. `include_attribution`
 * is what makes it per-stock rather than per-portfolio: without it the response is
 * one blended set of numbers for the basket, which says nothing about which member
 * of the basket is carrying the risk.
 *
 * The weights are left unset deliberately. The server then assumes an equal split,
 * and equal weights are the right assumption for a screen — every stock is being
 * compared on its own record, before any allocation exists to weight it by. The
 * `weight` field that comes back is therefore 1/n and carries no information; the
 * heatmap reads `stock_annual_return`, `stock_annual_volatility` and `stock_sharpe`
 * and ignores it.
 *
 * Keyed on the sorted symbol list, so reordering the universe re-reads the cache
 * instead of re-downloading a year of bars for the same set.
 */
export function usePortfolioMetrics(symbols, { lookback = 252, includeAttribution = true, enabled = true } = {}) {
    const list = Array.isArray(symbols) ? symbols : [];
    return useQuery({
        queryKey: qk.portfolioMetrics(list, lookback, includeAttribution),
        queryFn: () => fetchPortfolioMetrics(list, { lookback, includeAttribution }),
        // One symbol has no covariance to compute and the route needs a basket.
        enabled: enabled && list.length >= 2,
        staleTime: 10 * 60_000,
        retry: (failureCount, error) => !reachedTheServer(error) && failureCount < 2,
    });
}

/**
 * Pairwise correlation across the list — the diversification read.
 *
 * Separate from the metrics call above because it answers a different question and
 * takes a different window: 90 days for how the holdings move together *now*,
 * against 252 for what they have returned. Merging them would force one window on
 * both and make one of the two answers wrong.
 */
export function useCorrelation(symbols, { lookbackDays = 90, enabled = true } = {}) {
    const list = Array.isArray(symbols) ? symbols : [];
    return useQuery({
        queryKey: qk.correlation(list, lookbackDays),
        queryFn: () => fetchCorrelation(list, lookbackDays),
        enabled: enabled && list.length >= 2,
        staleTime: 10 * 60_000,
        retry: (failureCount, error) => !reachedTheServer(error) && failureCount < 2,
    });
}
