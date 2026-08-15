/**
 * Shared market-data queries.
 *
 * Every tab reads through these hooks so the same symbol is fetched once and
 * served from cache everywhere else — concurrent identical requests are
 * de-duplicated by TanStack Query rather than hitting the API N times.
 */
import { useQuery } from "@tanstack/react-query";
import {
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
