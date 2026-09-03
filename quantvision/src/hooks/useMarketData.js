/**
 * Shared market-data queries.
 *
 * Every tab reads through these hooks so the same symbol is fetched once and
 * served from cache everywhere else — concurrent identical requests are
 * de-duplicated by TanStack Query rather than hitting the API N times.
 */
import { useQuery } from "@tanstack/react-query";
import {
    fetchBestModelForecast,
    fetchHealth,
    fetchIndicators,
    fetchPrices,
    fetchQuotes,
    fetchDataSources,
} from "../utils/api";

/** Query keys live in one place so cache invalidation stays greppable. */
export const qk = {
    health: ["health"],
    sources: ["data", "sources"],
    quotes: (symbols) => ["data", "quotes", [...symbols].sort().join(",")],
    prices: (symbol, source, days, interval) => ["data", "prices", symbol, source, days, interval],
    indicators: (symbol, days, interval) => ["data", "indicators", symbol, days, interval],
    bestForecast: (symbol, horizon, readyVersion) => ["predict", "best", symbol, horizon, readyVersion],
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
