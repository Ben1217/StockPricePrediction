import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.jsx'
import { ApiError, loadRequestLimits } from './utils/api'

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            // Market data is fetched from a rate-limited upstream, so serve it from
            // cache aggressively and refetch on an explicit action instead of on focus.
            staleTime: 60_000,
            gcTime: 5 * 60_000,
            refetchOnWindowFocus: false,
            retry: (failureCount, error) => {
                // 4xx are deterministic — retrying only doubles latency.
                if (error instanceof ApiError && error.status < 500) return false
                return failureCount < 2
            },
        },
    },
})

// Adopt the server's per-interval request windows, replacing the bundled fallback
// table in utils/api.js. Deliberately not awaited: the app renders immediately and
// the first few requests clamp with the bundled numbers, which is exactly what they
// did before this endpoint existed. It never rejects, so there is nothing to catch.
loadRequestLimits()

createRoot(document.getElementById('root')).render(
    <StrictMode>
        <QueryClientProvider client={queryClient}>
            <App />
        </QueryClientProvider>
    </StrictMode>,
)
