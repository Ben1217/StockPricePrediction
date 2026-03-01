import { useState, useEffect, useCallback } from "react";
import "./index.css";
import { fetchPrices, fetchIndicators, fetchPredictions, fetchDataSources } from "./utils/api";
import { C, TICKERS } from "./utils/data";
import { Tab } from "./components/UIComponents";
import AnalysisTab from "./tabs/AnalysisTab";
import PredictionsTab from "./tabs/PredictionsTab";
import PortfolioTab from "./tabs/PortfolioTab";
import OptimizationTab from "./tabs/OptimizationTab";
import BacktestTab from "./tabs/BacktestTab";

export default function App() {
    const [activeTab, setActiveTab] = useState("analysis");
    const [selectedTicker, setSelectedTicker] = useState("AAPL");
    const [dataSource, setDataSource] = useState("yfinance");
    const [availableSources, setAvailableSources] = useState(["yfinance"]);
    const [notification, setNotif] = useState(null);
    const [loading, setLoading] = useState(false);

    // Live data state
    const [priceData, setPriceData] = useState(null);
    const [indicatorData, setIndicatorData] = useState(null);
    const [tickerQuotes, setTickerQuotes] = useState({});
    const [apiConnected, setApiConnected] = useState(false);

    // Check API connection & sources on mount
    useEffect(() => {
        fetch("http://localhost:8000/health")
            .then(r => r.json())
            .then(() => {
                setApiConnected(true);
                fetchDataSources().then(s => setAvailableSources(s.sources || ["yfinance"])).catch(() => { });
            })
            .catch(() => setApiConnected(false));
    }, []);

    // Fetch prices for all tickers (for ticker bar)
    useEffect(() => {
        if (!apiConnected) return;
        const fetchQuotes = async () => {
            const quotes = {};
            for (const t of TICKERS) {
                try {
                    const resp = await fetchPrices(t, "yfinance", 5);
                    if (resp.bars && resp.bars.length >= 2) {
                        const last = resp.bars[resp.bars.length - 1];
                        const prev = resp.bars[resp.bars.length - 2];
                        quotes[t] = {
                            price: last.close,
                            change: ((last.close - prev.close) / prev.close) * 100,
                        };
                    }
                } catch {
                    /* skip */
                }
            }
            setTickerQuotes(quotes);
        };
        fetchQuotes();
    }, [apiConnected]);

    // Fetch data when ticker or source changes
    useEffect(() => {
        if (!apiConnected) return;
        setLoading(true);
        Promise.all([
            fetchPrices(selectedTicker, dataSource, 120).catch(() => null),
            fetchIndicators(selectedTicker, 120).catch(() => null),
        ]).then(([prices, indicators]) => {
            setPriceData(prices);
            setIndicatorData(indicators);
            setLoading(false);
        });
    }, [selectedTicker, dataSource, apiConnected]);

    const notify = (msg) => { setNotif(msg); setTimeout(() => setNotif(null), 3000); };

    return (
        <div style={{ minHeight: "100vh", background: C.bg0, color: C.text, fontFamily: "'DM Mono',monospace" }}>
            {notification && (
                <div style={{
                    position: "fixed", top: 16, right: 16, zIndex: 999,
                    background: C.bg1, border: `1px solid ${C.amber}`, borderRadius: 8, padding: "10px 18px",
                    color: C.amber, fontSize: 12, fontFamily: "'DM Mono',monospace",
                    boxShadow: "0 4px 24px rgba(251,191,36,.25)", animation: "fadeUp .3s ease",
                }}>{notification}</div>
            )}

            {/* Header */}
            <div style={{
                background: C.bg1, borderBottom: `1px solid ${C.border}`, padding: "0 28px",
                display: "flex", alignItems: "center", justifyContent: "space-between",
                position: "sticky", top: 0, zIndex: 100, backdropFilter: "blur(12px)",
            }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <div style={{
                        background: `linear-gradient(135deg, ${C.amber}, #f97316)`,
                        borderRadius: 8, width: 32, height: 32, display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 16, fontWeight: 800, color: "#000", fontFamily: "'Syne',sans-serif",
                    }}>Q</div>
                    <div>
                        <div style={{ fontSize: 15, fontWeight: 700, fontFamily: "'Syne',sans-serif", color: C.amber, lineHeight: 1 }}>QuantVision</div>
                        <div style={{ fontSize: 9, color: C.textDim, letterSpacing: 2 }}>AI PORTFOLIO INTELLIGENCE</div>
                    </div>
                </div>
                <div style={{ display: "flex", gap: 0 }}>
                    {[
                        { key: "analysis", label: "📈  Analysis" },
                        { key: "predictions", label: "🤖  Predictions" },
                        { key: "portfolio", label: "💼  Portfolio" },
                        { key: "backtest", label: "🧪  Backtest" },
                        { key: "optimization", label: "⚡  Optimize" },
                    ].map(t => (
                        <Tab key={t.key} active={activeTab === t.key} onClick={() => setActiveTab(t.key)}>{t.label}</Tab>
                    ))}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div className={apiConnected ? "live-dot" : ""} style={{ width: 7, height: 7, borderRadius: "50%", background: apiConnected ? C.green : C.red }} />
                    <span style={{ fontSize: 10, color: C.textDim, letterSpacing: 1 }}>{apiConnected ? "LIVE" : "OFFLINE"}</span>
                    {/* Data source selector */}
                    <select value={dataSource} onChange={e => setDataSource(e.target.value)} style={{
                        background: C.bg2, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4,
                        padding: "4px 8px", fontSize: 10, fontFamily: "'DM Mono',monospace", cursor: "pointer",
                    }}>
                        {availableSources.map(s => <option key={s} value={s}>{s.replace("_", " ").toUpperCase()}</option>)}
                    </select>
                </div>
            </div>

            {/* Ticker bar */}
            <div style={{
                background: C.bg1, borderBottom: `1px solid ${C.border}`, padding: "0 28px",
                display: "flex", alignItems: "center", gap: 4, overflowX: "auto",
            }}>
                {TICKERS.map(t => {
                    const q = tickerQuotes[t] || {};
                    return (
                        <button key={t} onClick={() => setSelectedTicker(t)} style={{
                            background: selectedTicker === t ? C.amberDim : "transparent",
                            border: `1px solid ${selectedTicker === t ? C.amber + "55" : "transparent"}`,
                            borderRadius: 6, padding: "8px 14px", cursor: "pointer",
                            display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 2, minWidth: 80,
                        }}>
                            <span style={{ color: selectedTicker === t ? C.amber : C.text, fontSize: 12, fontWeight: 700, fontFamily: "'Syne',sans-serif" }}>{t}</span>
                            <span style={{ color: C.textMid, fontSize: 10 }}>{q.price ? `$${q.price.toFixed(2)}` : "—"}</span>
                            <span style={{ color: (q.change || 0) >= 0 ? C.green : C.red, fontSize: 9 }}>
                                {q.change ? `${q.change >= 0 ? "+" : ""}${q.change.toFixed(2)}%` : ""}
                            </span>
                        </button>
                    );
                })}
            </div>

            {/* Loading bar */}
            {loading && (
                <div style={{
                    height: 2, background: `linear-gradient(90deg, transparent, ${C.amber}, transparent)`,
                    animation: "pulse 1s ease-in-out infinite"
                }} />
            )}

            {/* Main content */}
            <div style={{ padding: "24px 28px", maxWidth: 1400, margin: "0 auto" }}>
                {activeTab === "analysis" && (
                    <AnalysisTab
                        selectedTicker={selectedTicker}
                        priceData={priceData}
                        indicatorData={indicatorData}
                        dataSource={dataSource}
                        apiConnected={apiConnected}
                    />
                )}
                {activeTab === "predictions" && (
                    <PredictionsTab
                        selectedTicker={selectedTicker}
                        apiConnected={apiConnected}
                        priceData={priceData}
                    />
                )}
                {activeTab === "portfolio" && (
                    <PortfolioTab
                        notify={notify}
                        apiConnected={apiConnected}
                        setSelectedTicker={setSelectedTicker}
                        setActiveTab={setActiveTab}
                    />
                )}
                {activeTab === "backtest" && (
                    <BacktestTab
                        selectedTicker={selectedTicker}
                        apiConnected={apiConnected}
                        notify={notify}
                    />
                )}
                {activeTab === "optimization" && (
                    <OptimizationTab
                        apiConnected={apiConnected}
                        notify={notify}
                    />
                )}
            </div>

            {/* Footer */}
            <div style={{
                borderTop: `1px solid ${C.border}`, padding: "14px 28px",
                display: "flex", justifyContent: "space-between", alignItems: "center",
                fontSize: 10, color: C.textDim,
            }}>
                <div style={{ display: "flex", gap: 20 }}>
                    <span>Data: {dataSource.replace("_", " ")} {apiConnected ? "· Connected" : "· Offline"}</span>
                    <span>Models: PyTorch LSTM + XGBoost + RF ensemble</span>
                    <span>⚠️ Not financial advice</span>
                </div>
                <div style={{ color: C.amber }}>QuantVision v3.0.0</div>
            </div>
        </div>
    );
}
