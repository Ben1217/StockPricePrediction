/**
 * The real TradingView Advanced Chart, embedded.
 *
 * This replaces the hand-drawn candlestick chart the app used to render with
 * lightweight-charts. That chart was an imitation of this one: it redrew
 * candles, volume, RSI and MACD panes, its own crosshair and its own timeframe
 * buttons, and every one of those was a worse version of something TradingView
 * already ships. Drawing tools, replay, comparison symbols, indicator search
 * and the symbol's own exchange data came with none of it.
 *
 * So the division of labour is now explicit and holds everywhere in the app:
 *
 *     TradingView  ->  price visualisation. Candles, volume, timeframes, studies.
 *     Our backend  ->  the analysis. Indicators, price action, patterns, models,
 *                      probability, direction, evidence.
 *
 * Nothing in this file computes anything about the market. It mounts a widget
 * and points it at a symbol.
 *
 * Mechanics
 * ---------
 * The official embed is a `<script>` whose *text content* is the JSON config —
 * not a module with an API, and not something React can render declaratively.
 * So the widget is (re)built imperatively whenever its inputs change: clear the
 * container, append a fresh configured script, let TradingView populate it.
 * Clearing first matters — appending a second script leaves two charts stacked
 * in the same box.
 */

import { useEffect, useRef, useState } from "react";

import { C } from "../utils/data";
import { toTradingViewInterval, toTradingViewSymbol, tradingViewUrl } from "../utils/tradingview";

const EMBED_SRC = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";

/**
 * How long to wait before assuming the embed is not coming.
 *
 * The script is third-party: an offline machine, a blocked CDN or a corporate
 * proxy all fail silently, leaving an empty box that looks like our bug. After
 * this the component says what happened and offers the chart on tradingview.com
 * instead.
 */
const LOAD_TIMEOUT_MS = 12000;

/** Studies the chart opens with. These are TradingView's, drawn by TradingView. */
const DEFAULT_STUDIES = ["STD;Volume", "STD;RSI"];

export default function TradingViewChart({
    symbol,
    interval = "1d",
    height = 520,
    studies = DEFAULT_STUDIES,
    hideSideToolbar = false,
    withDateRanges = true,
}) {
    const containerRef = useRef(null);
    const [failed, setFailed] = useState(false);

    const tvSymbol = toTradingViewSymbol(symbol);
    const tvInterval = toTradingViewInterval(interval);
    // Studies arrive as an array literal at most call sites, so the reference is
    // new on every parent render. The effect keys on the *content* instead and
    // rebuilds the list from it, which stops the widget tearing itself down and
    // remounting each time the parent re-renders for an unrelated reason.
    const studyKey = studies.join(",");

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return undefined;

        setFailed(false);
        container.innerHTML = "";

        const widgetHost = document.createElement("div");
        widgetHost.className = "tradingview-widget-container__widget";
        widgetHost.style.height = "100%";
        widgetHost.style.width = "100%";
        container.appendChild(widgetHost);

        const script = document.createElement("script");
        script.src = EMBED_SRC;
        script.type = "text/javascript";
        script.async = true;
        script.innerHTML = JSON.stringify({
            autosize: true,
            symbol: tvSymbol,
            interval: tvInterval,
            timezone: "America/New_York",
            theme: "dark",
            style: "1",                 // candlesticks
            locale: "en",
            enable_publishing: false,
            // The ticker is chosen in our watchlist bar; letting the widget
            // change it too would leave the chart showing one symbol while
            // every panel beneath it analysed another.
            allow_symbol_change: false,
            withdateranges: withDateRanges,
            hide_side_toolbar: hideSideToolbar,
            hide_legend: false,
            details: false,
            calendar: false,
            studies: studyKey ? studyKey.split(",") : [],
            support_host: "https://www.tradingview.com",
        });
        script.onerror = () => setFailed(true);
        container.appendChild(script);

        // The embed reports nothing on success, so "did it render" is answered
        // by looking for the iframe it injects.
        const timer = window.setTimeout(() => {
            if (!container.querySelector("iframe")) setFailed(true);
        }, LOAD_TIMEOUT_MS);

        return () => {
            window.clearTimeout(timer);
            container.innerHTML = "";
        };
    }, [tvSymbol, tvInterval, studyKey, hideSideToolbar, withDateRanges]);

    return (
        <div style={{ position: "relative", height, width: "100%" }}>
            <div
                ref={containerRef}
                className="tradingview-widget-container"
                style={{ height: "100%", width: "100%" }}
            />
            {failed && (
                <div
                    role="alert"
                    style={{
                        position: "absolute", inset: 0, display: "flex", gap: 10,
                        flexDirection: "column", alignItems: "center", justifyContent: "center",
                        background: C.bg1, border: `1px solid ${C.border}`, borderRadius: 8,
                        color: C.textMid, fontSize: 12, textAlign: "center", padding: 24,
                    }}
                >
                    <span style={{ fontSize: 22 }}>📉</span>
                    <span>TradingView's chart could not be loaded for {String(symbol || "").toUpperCase()}.</span>
                    <span style={{ color: C.textDim, fontSize: 11, maxWidth: 420, lineHeight: 1.6 }}>
                        The widget is served from tradingview.com — check the network connection
                        or any extension blocking third-party embeds. The analysis below is
                        computed by our own backend and is unaffected.
                    </span>
                    <a
                        href={tradingViewUrl(symbol)}
                        target="_blank"
                        rel="noreferrer noopener"
                        style={{ color: C.amber, fontSize: 11 }}
                    >
                        Open {tvSymbol} on tradingview.com ↗
                    </a>
                </div>
            )}
        </div>
    );
}
