import { useState, useEffect, useCallback, useRef } from "react";
import "./index.css";
import { fetchPrices, fetchIndicators, fetchDataSources } from "./utils/api";
import { C, TICKERS, SP500_LIST } from "./utils/data";
import { Tab } from "./components/UIComponents";
import AnalysisTab from "./tabs/AnalysisTab";
import PredictionsTab from "./tabs/PredictionsTab";
import PortfolioTab from "./tabs/PortfolioTab";
import OptimizationTab from "./tabs/OptimizationTab";
import BacktestTab from "./tabs/BacktestTab";
import HeatmapTab from "./tabs/HeatmapTab";

/* ─── Sector colour map for screener badges ──────────────────── */
const SECTOR_COLORS = {
    "Technology": "#6366f1",
    "Communication": "#0ea5e9",
    "Consumer Discretionary": "#f59e0b",
    "Consumer Staples": "#10b981",
    "Health Care": "#ec4899",
    "Financials": "#8b5cf6",
    "Industrials": "#64748b",
    "Energy": "#f97316",
    "Materials": "#84cc16",
    "Real Estate": "#06b6d4",
    "Utilities": "#a78bfa",
};

const ALL_SECTORS = ["All", ...Object.keys(SECTOR_COLORS)];

/* ─── Helpers ────────────────────────────────────────────────── */
function loadWatchlist() {
    try {
        const saved = JSON.parse(localStorage.getItem("qv_watchlist"));
        if (Array.isArray(saved) && saved.length > 0) return saved;
    } catch { /* ignore */ }
    return [...TICKERS];
}

function saveWatchlist(list) {
    try { localStorage.setItem("qv_watchlist", JSON.stringify(list)); } catch { /* ignore */ }
}

/* ═══════════════════════════════════════════════════════════════
   STOCK SEARCH MODAL
═══════════════════════════════════════════════════════════════ */
function StockSearchModal({ watchlist, onAdd, onClose }) {
    const [tab, setTab] = useState("search");           // "search" | "screener"
    const [query, setQuery] = useState("");
    const [searching, setSearching] = useState(false);
    const [searchError, setSearchError] = useState("");
    const [sectorFilter, setSectorFilter] = useState("All");
    const inputRef = useRef(null);

    // Focus input on open
    useEffect(() => { inputRef.current?.focus(); }, []);

    // Close on Escape
    useEffect(() => {
        const handler = (e) => { if (e.key === "Escape") onClose(); };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [onClose]);

    const isFull = watchlist.length >= 8;

    /* Search: validate ticker via API, then add */
    const handleSearch = async () => {
        const t = query.trim().toUpperCase();
        if (!t) return;
        if (watchlist.includes(t)) { setSearchError(`${t} is already in your watchlist.`); return; }
        if (isFull) { setSearchError("Watchlist is full (max 8). Remove a ticker first."); return; }
        setSearching(true);
        setSearchError("");
        try {
            const resp = await fetchPrices(t, "yfinance", 5);
            if (resp?.bars?.length > 0) {
                onAdd(t);
                onClose();
            } else {
                setSearchError(`Could not find data for "${t}". Check the ticker symbol.`);
            }
        } catch {
            setSearchError(`Could not fetch "${t}". Check the ticker symbol.`);
        } finally {
            setSearching(false);
        }
    };

    /* Screener filtered list */
    const screenerList = SP500_LIST.filter(s => {
        const q = query.toLowerCase();
        const sectorMatch = sectorFilter === "All" || s.sector === sectorFilter;
        const textMatch = !q || s.ticker.toLowerCase().includes(q) || s.name.toLowerCase().includes(q);
        return sectorMatch && textMatch;
    });

    const btnStyle = (active) => ({
        background: active ? C.amberDim : "transparent",
        border: `1px solid ${active ? C.amber + "66" : C.border}`,
        borderRadius: 20, color: active ? C.amber : C.textMid,
        padding: "5px 16px", cursor: "pointer", fontSize: 11,
        fontFamily: "'DM Mono',monospace", transition: "all .15s",
    });

    return (
        <div
            onClick={onClose}
            style={{
                position: "fixed", inset: 0, zIndex: 500,
                background: "rgba(4,8,16,.85)", backdropFilter: "blur(8px)",
                display: "flex", alignItems: "center", justifyContent: "center",
            }}
        >
            <div
                onClick={e => e.stopPropagation()}
                style={{
                    background: C.bg1, border: `1px solid ${C.border}`,
                    borderRadius: 14, width: 540, maxWidth: "95vw", maxHeight: "80vh",
                    display: "flex", flexDirection: "column",
                    boxShadow: "0 24px 64px rgba(0,0,0,.7)",
                    fontFamily: "'DM Mono',monospace",
                }}
            >
                {/* Modal header */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 20px 12px", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 14, color: C.text }}>
                        ＋ Add Stock to Watchlist
                    </span>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{ fontSize: 10, color: C.textDim }}>{watchlist.length}/8 slots used</span>
                        <button onClick={onClose} style={{ background: "none", border: "none", color: C.textDim, cursor: "pointer", fontSize: 16, lineHeight: 1 }}>✕</button>
                    </div>
                </div>

                {/* Tab row */}
                <div style={{ display: "flex", gap: 6, padding: "12px 20px 0" }}>
                    <button style={btnStyle(tab === "search")} onClick={() => setTab("search")}>🔍 Search Any Ticker</button>
                    <button style={btnStyle(tab === "screener")} onClick={() => setTab("screener")}>📊 S&P 500 Screener</button>
                </div>

                {/* Search bar (shared between tabs) */}
                <div style={{ padding: "12px 20px 0", display: "flex", gap: 8 }}>
                    <input
                        ref={inputRef}
                        placeholder={tab === "search" ? "Enter any ticker — e.g. TSM, BABA, JPM…" : "Filter by ticker or company name…"}
                        value={query}
                        onChange={e => { setQuery(e.target.value); setSearchError(""); }}
                        onKeyDown={e => { if (e.key === "Enter" && tab === "search") handleSearch(); }}
                        style={{
                            flex: 1, background: C.bg2, border: `1px solid ${C.border}`,
                            borderRadius: 8, color: C.text, padding: "8px 12px",
                            fontSize: 12, fontFamily: "'DM Mono',monospace", outline: "none",
                        }}
                    />
                    {tab === "search" && (
                        <button
                            onClick={handleSearch}
                            disabled={searching || isFull}
                            style={{
                                background: C.amber, color: "#000", border: "none",
                                borderRadius: 8, padding: "8px 16px", cursor: searching || isFull ? "not-allowed" : "pointer",
                                fontWeight: 700, fontSize: 12, fontFamily: "'Syne',sans-serif",
                                opacity: searching || isFull ? 0.5 : 1,
                            }}
                        >
                            {searching ? "…" : "Search"}
                        </button>
                    )}
                </div>

                {/* Sector filter (screener only) */}
                {tab === "screener" && (
                    <div style={{ padding: "8px 20px 0", display: "flex", gap: 4, flexWrap: "wrap" }}>
                        {ALL_SECTORS.map(s => (
                            <button
                                key={s}
                                onClick={() => setSectorFilter(s)}
                                style={{
                                    background: sectorFilter === s ? (SECTOR_COLORS[s] || C.amber) + "33" : "transparent",
                                    border: `1px solid ${sectorFilter === s ? (SECTOR_COLORS[s] || C.amber) + "88" : C.border}`,
                                    borderRadius: 20, color: sectorFilter === s ? (SECTOR_COLORS[s] || C.amber) : C.textDim,
                                    padding: "3px 10px", cursor: "pointer", fontSize: 10,
                                    fontFamily: "'DM Mono',monospace",
                                }}
                            >{s}</button>
                        ))}
                    </div>
                )}

                {/* Error message */}
                {searchError && (
                    <div style={{ margin: "8px 20px 0", padding: "8px 12px", background: C.red + "18", border: `1px solid ${C.red}55`, borderRadius: 6, color: C.red, fontSize: 11 }}>
                        ⚠ {searchError}
                    </div>
                )}

                {/* Full warning */}
                {isFull && (
                    <div style={{ margin: "8px 20px 0", padding: "8px 12px", background: C.amber + "15", border: `1px solid ${C.amber}44`, borderRadius: 6, color: C.amber, fontSize: 11 }}>
                        Watchlist is full (8/8). Hover a ticker chip and click × to remove one first.
                    </div>
                )}

                {/* Body */}
                <div style={{ flex: 1, overflowY: "auto", padding: "10px 20px 20px" }}>
                    {tab === "search" ? (
                        <div style={{ color: C.textDim, fontSize: 11, paddingTop: 12, lineHeight: 1.8 }}>
                            Type any valid stock ticker above and press <b style={{ color: C.amber }}>Search</b> or <b style={{ color: C.amber }}>Enter</b>.
                            <br />Works with US and global stocks — NYSE, NASDAQ, TSX, LSE, etc.
                            <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 4 }}>
                                {["TSM", "BABA", "NVO", "ASML", "SHOP", "PLTR", "ARM", "COIN", "SNOW", "ABNB"].map(t => (
                                    <button
                                        key={t}
                                        onClick={() => { setQuery(t); inputRef.current?.focus(); }}
                                        style={{
                                            background: C.bg2, border: `1px solid ${C.border}`,
                                            borderRadius: 20, color: C.textMid, padding: "3px 10px",
                                            cursor: "pointer", fontSize: 10, fontFamily: "'DM Mono',monospace",
                                        }}
                                    >{t}</button>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6 }}>
                            {screenerList.length === 0 && (
                                <div style={{ color: C.textDim, fontSize: 11, textAlign: "center", padding: "24px 0" }}>No stocks match your filter.</div>
                            )}
                            {screenerList.map(s => {
                                const inList = watchlist.includes(s.ticker);
                                const sColor = SECTOR_COLORS[s.sector] || C.amber;
                                return (
                                    <div
                                        key={s.ticker}
                                        onClick={() => { if (!inList && !isFull) { onAdd(s.ticker); onClose(); } }}
                                        style={{
                                            display: "flex", alignItems: "center", justifyContent: "space-between",
                                            padding: "8px 12px", borderRadius: 8,
                                            background: inList ? C.bg2 : "transparent",
                                            border: `1px solid ${inList ? C.border : "transparent"}`,
                                            cursor: inList || isFull ? "not-allowed" : "pointer",
                                            opacity: inList || isFull ? 0.55 : 1,
                                            transition: "background .12s",
                                        }}
                                        onMouseEnter={e => { if (!inList && !isFull) e.currentTarget.style.background = C.bg2; }}
                                        onMouseLeave={e => { if (!inList && !isFull) e.currentTarget.style.background = "transparent"; }}
                                    >
                                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                            <span style={{ color: C.amber, fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 12, minWidth: 50 }}>{s.ticker}</span>
                                            <span style={{ color: C.textMid, fontSize: 11 }}>{s.name}</span>
                                        </div>
                                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                            <span style={{
                                                background: sColor + "22", border: `1px solid ${sColor}55`,
                                                color: sColor, borderRadius: 20, padding: "2px 8px", fontSize: 9, fontWeight: 600,
                                            }}>{s.sector}</span>
                                            {inList && <span style={{ color: C.textDim, fontSize: 9 }}>● In watchlist</span>}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════
   CHAT WIDGET — Draggable AI Agent Interface
   Drag the header (open) or the 🤖 button (closed) to reposition.
═══════════════════════════════════════════════════════════════ */
function ChatWidget({ apiConnected }) {
    const PANEL_W = 400, PANEL_H = 520, BTN_SIZE = 56;

    const [open, setOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: "assistant", text: "Hi! I'm QuantVision AI. Ask me anything about stocks, predictions, or your portfolio." },
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    // ── Drag state ──────────────────────────────────────────────
    const [pos, setPos] = useState({ x: window.innerWidth - BTN_SIZE - 24, y: window.innerHeight - BTN_SIZE - 24 });
    const dragRef = useRef({ dragging: false, startX: 0, startY: 0, origX: 0, origY: 0, moved: false });

    const clamp = (x, y, w, h) => ({
        x: Math.max(0, Math.min(x, window.innerWidth - w)),
        y: Math.max(0, Math.min(y, window.innerHeight - h)),
    });

    const onDragStart = (e) => {
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        dragRef.current = { dragging: true, startX: clientX, startY: clientY, origX: pos.x, origY: pos.y, moved: false };
        e.preventDefault();
    };

    useEffect(() => {
        const onMove = (e) => {
            const d = dragRef.current;
            if (!d.dragging) return;
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            const dx = clientX - d.startX, dy = clientY - d.startY;
            if (Math.abs(dx) > 3 || Math.abs(dy) > 3) d.moved = true;
            const w = open ? PANEL_W : BTN_SIZE, h = open ? PANEL_H : BTN_SIZE;
            setPos(clamp(d.origX + dx, d.origY + dy, w, h));
        };
        const onEnd = () => { dragRef.current.dragging = false; };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onEnd);
        window.addEventListener("touchmove", onMove, { passive: false });
        window.addEventListener("touchend", onEnd);
        return () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onEnd);
            window.removeEventListener("touchmove", onMove);
            window.removeEventListener("touchend", onEnd);
        };
    }, [open, pos]);

    // Re-clamp on window resize
    useEffect(() => {
        const onResize = () => {
            const w = open ? PANEL_W : BTN_SIZE, h = open ? PANEL_H : BTN_SIZE;
            setPos(prev => clamp(prev.x, prev.y, w, h));
        };
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, [open]);

    // ── Auto-scroll messages ────────────────────────────────────
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // ── Send message ────────────────────────────────────────────
    const sendMessage = async () => {
        const q = input.trim();
        if (!q || loading) return;
        setMessages(prev => [...prev, { role: "user", text: q }]);
        setInput("");
        setLoading(true);
        try {
            const res = await fetch("http://localhost:8000/api/agent/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: q }),
            });
            const data = await res.json();
            setMessages(prev => [...prev, { role: "assistant", text: data.answer || data.detail || "No response received." }]);
        } catch (e) {
            setMessages(prev => [...prev, { role: "assistant", text: `⚠ Error: ${e.message}. Make sure the backend is running.` }]);
        } finally {
            setLoading(false);
        }
    };

    // ── Closed state: draggable floating button ─────────────────
    if (!open) {
        return (
            <button
                id="chat-widget-toggle"
                onMouseDown={onDragStart}
                onTouchStart={onDragStart}
                onClick={() => { if (!dragRef.current.moved) setOpen(true); }}
                style={{
                    position: "fixed", left: pos.x, top: pos.y, zIndex: 900,
                    width: BTN_SIZE, height: BTN_SIZE, borderRadius: "50%",
                    background: `linear-gradient(135deg, ${C.amber}, #f97316)`,
                    border: "none", cursor: "grab",
                    boxShadow: "0 4px 24px rgba(251,191,36,.4)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 24, color: "#000", fontWeight: 800,
                    transition: "box-shadow .2s",
                    userSelect: "none", touchAction: "none",
                }}
                onMouseEnter={e => { e.currentTarget.style.boxShadow = "0 6px 32px rgba(251,191,36,.6)"; }}
                onMouseLeave={e => { e.currentTarget.style.boxShadow = "0 4px 24px rgba(251,191,36,.4)"; }}
                title="Ask QuantVision AI — drag to reposition"
            >🤖</button>
        );
    }

    // ── Open state: draggable panel ─────────────────────────────
    return (
        <div style={{
            position: "fixed", left: pos.x, top: pos.y, zIndex: 900,
            width: PANEL_W, maxWidth: "calc(100vw - 16px)", height: PANEL_H, maxHeight: "calc(100vh - 16px)",
            background: C.bg1, border: `1px solid ${C.border}`, borderRadius: 16,
            display: "flex", flexDirection: "column",
            boxShadow: "0 16px 64px rgba(0,0,0,.6)",
            fontFamily: "'DM Mono',monospace",
            userSelect: "none",
        }}>
            {/* Header — drag handle */}
            <div
                onMouseDown={onDragStart}
                onTouchStart={onDragStart}
                style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "14px 16px", borderBottom: `1px solid ${C.border}`,
                    background: `linear-gradient(135deg, ${C.bg1}, ${C.bg2})`,
                    borderRadius: "16px 16px 0 0",
                    cursor: "grab", touchAction: "none",
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: 10, pointerEvents: "none" }}>
                    <div style={{
                        background: `linear-gradient(135deg, ${C.amber}, #f97316)`,
                        borderRadius: 8, width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 14, fontWeight: 800, color: "#000",
                    }}>AI</div>
                    <div>
                        <div style={{ fontSize: 12, fontWeight: 700, fontFamily: "'Syne',sans-serif", color: C.amber, lineHeight: 1 }}>QuantVision AI</div>
                        <div style={{ fontSize: 9, color: apiConnected ? C.green : C.red, letterSpacing: 1 }}>
                            {apiConnected ? "● ONLINE" : "● OFFLINE"}
                        </div>
                    </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 9, color: C.textDim, pointerEvents: "none" }}>⠿ drag</span>
                    <button
                        onClick={() => setOpen(false)}
                        style={{ background: "none", border: "none", color: C.textDim, cursor: "pointer", fontSize: 16, lineHeight: 1, padding: 4, pointerEvents: "auto" }}
                    >✕</button>
                </div>
            </div>

            {/* Messages */}
            <div style={{
                flex: 1, overflowY: "auto", padding: "12px 14px",
                display: "flex", flexDirection: "column", gap: 10,
                userSelect: "text",
            }}>
                {messages.map((m, i) => (
                    <div key={i} style={{
                        alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                        maxWidth: "85%",
                        padding: "10px 14px",
                        borderRadius: m.role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                        background: m.role === "user"
                            ? `linear-gradient(135deg, ${C.amber}22, ${C.amber}11)`
                            : C.bg2,
                        border: `1px solid ${m.role === "user" ? C.amber + "33" : C.border}`,
                        color: C.text, fontSize: 12, lineHeight: 1.6,
                        whiteSpace: "pre-wrap", wordBreak: "break-word",
                    }}>
                        {m.text}
                    </div>
                ))}
                {loading && (
                    <div style={{
                        alignSelf: "flex-start", padding: "10px 14px",
                        borderRadius: "14px 14px 14px 4px",
                        background: C.bg2, border: `1px solid ${C.border}`,
                        color: C.amber, fontSize: 12,
                    }}>
                        <span style={{ animation: "pulse 1s ease-in-out infinite" }}>Thinking…</span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div style={{
                padding: "10px 14px", borderTop: `1px solid ${C.border}`,
                display: "flex", gap: 8, userSelect: "text",
            }}>
                <input
                    id="chat-widget-input"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                    placeholder="Ask about stocks, signals, portfolio…"
                    disabled={loading}
                    style={{
                        flex: 1, background: C.bg0, border: `1px solid ${C.border}`,
                        borderRadius: 10, color: C.text, padding: "10px 14px",
                        fontSize: 12, fontFamily: "'DM Mono',monospace", outline: "none",
                        opacity: loading ? 0.6 : 1,
                    }}
                />
                <button
                    id="chat-widget-send"
                    onClick={sendMessage}
                    disabled={loading || !input.trim()}
                    style={{
                        background: `linear-gradient(135deg, ${C.amber}, #f97316)`,
                        border: "none", borderRadius: 10, padding: "10px 16px",
                        cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                        color: "#000", fontWeight: 700, fontSize: 12,
                        fontFamily: "'Syne',sans-serif",
                        opacity: loading || !input.trim() ? 0.5 : 1,
                        transition: "opacity .15s",
                    }}
                >Send</button>
            </div>

            {/* Quick prompts */}
            <div style={{
                padding: "6px 14px 12px", display: "flex", gap: 4, flexWrap: "wrap",
            }}>
                {["Strongest buy signal?", "Predict AAPL", "Rebalance portfolio?"].map(q => (
                    <button
                        key={q}
                        onClick={() => { setInput(q); }}
                        style={{
                            background: C.bg2, border: `1px solid ${C.border}`,
                            borderRadius: 20, color: C.textDim, padding: "3px 10px",
                            cursor: "pointer", fontSize: 9, fontFamily: "'DM Mono',monospace",
                            transition: "border-color .15s, color .15s",
                        }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = C.amber; e.currentTarget.style.color = C.amber; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.color = C.textDim; }}
                    >{q}</button>
                ))}
            </div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════
   APP
═══════════════════════════════════════════════════════════════ */
export default function App() {
    const [activeTab, setActiveTab] = useState("analysis");
    const [watchlist, setWatchlist] = useState(loadWatchlist);
    const [selectedTicker, setSelectedTicker] = useState(() => loadWatchlist()[0]);
    const [dataSource, setDataSource] = useState("yfinance");
    const [availableSources, setAvailableSources] = useState(["yfinance"]);
    const [notification, setNotif] = useState(null);
    const [loading, setLoading] = useState(false);
    const [showModal, setShowModal] = useState(false);
    const [hoveredChip, setHoveredChip] = useState(null); // ticker string

    // Live data state
    const [priceData, setPriceData] = useState(null);
    const [indicatorData, setIndicatorData] = useState(null);
    const [tickerQuotes, setTickerQuotes] = useState({});
    const [apiConnected, setApiConnected] = useState(false);

    // Persist watchlist on change
    useEffect(() => { saveWatchlist(watchlist); }, [watchlist]);

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

    // Fetch prices for all watchlist tickers (ticker bar quotes)
    useEffect(() => {
        if (!apiConnected) return;
        const fetchQuotes = async () => {
            const quotes = {};
            for (const t of watchlist) {
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
                } catch { /* skip */ }
            }
            setTickerQuotes(quotes);
        };
        fetchQuotes();
    }, [apiConnected, watchlist]);

    // Fetch price & indicator data when selected ticker changes
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

    /* Watchlist mutations */
    const addTicker = useCallback((ticker) => {
        if (watchlist.includes(ticker) || watchlist.length >= 8) return;
        setWatchlist(prev => [...prev, ticker]);
        setSelectedTicker(ticker);
        setActiveTab("analysis");
        notify(`${ticker} added to watchlist`);
    }, [watchlist]);

    const removeTicker = useCallback((ticker) => {
        if (watchlist.length <= 1) { notify("You need at least 1 ticker in your watchlist."); return; }
        setWatchlist(prev => {
            const next = prev.filter(t => t !== ticker);
            // If the removed ticker was selected, switch to first remaining
            if (selectedTicker === ticker) setSelectedTicker(next[0]);
            return next;
        });
    }, [watchlist, selectedTicker]);

    return (
        <div style={{ minHeight: "100vh", background: C.bg0, color: C.text, fontFamily: "'DM Mono',monospace" }}>
            {/* Notification toast */}
            {notification && (
                <div style={{
                    position: "fixed", top: 16, right: 16, zIndex: 999,
                    background: C.bg1, border: `1px solid ${C.amber}`, borderRadius: 8, padding: "10px 18px",
                    color: C.amber, fontSize: 12, fontFamily: "'DM Mono',monospace",
                    boxShadow: "0 4px 24px rgba(251,191,36,.25)", animation: "fadeUp .3s ease",
                }}>{notification}</div>
            )}

            {/* Stock Add Modal */}
            {showModal && (
                <StockSearchModal
                    watchlist={watchlist}
                    onAdd={addTicker}
                    onClose={() => setShowModal(false)}
                />
            )}

            {/* ── Header ──────────────────────────────────────────── */}
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
                        { key: "heatmap", label: "🌡️  Heatmap" },
                    ].map(t => (
                        <Tab key={t.key} active={activeTab === t.key} onClick={() => setActiveTab(t.key)}>{t.label}</Tab>
                    ))}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div className={apiConnected ? "live-dot" : ""} style={{ width: 7, height: 7, borderRadius: "50%", background: apiConnected ? C.green : C.red }} />
                    <span style={{ fontSize: 10, color: C.textDim, letterSpacing: 1 }}>{apiConnected ? "LIVE" : "OFFLINE"}</span>
                    <select value={dataSource} onChange={e => setDataSource(e.target.value)} style={{
                        background: C.bg2, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4,
                        padding: "4px 8px", fontSize: 10, fontFamily: "'DM Mono',monospace", cursor: "pointer",
                    }}>
                        {availableSources.map(s => <option key={s} value={s}>{s.replace("_", " ").toUpperCase()}</option>)}
                    </select>
                </div>
            </div>

            {/* ── Ticker / Watchlist bar ───────────────────────────── */}
            <div style={{
                background: C.bg1, borderBottom: `1px solid ${C.border}`, padding: "0 28px",
                display: "flex", alignItems: "center", gap: 2, overflowX: "auto", flexShrink: 0,
            }}>
                {watchlist.map(t => {
                    const q = tickerQuotes[t] || {};
                    const active = selectedTicker === t;
                    const hovered = hoveredChip === t;
                    return (
                        <div
                            key={t}
                            style={{ position: "relative" }}
                            onMouseEnter={() => setHoveredChip(t)}
                            onMouseLeave={() => setHoveredChip(null)}
                        >
                            <button
                                onClick={() => { setSelectedTicker(t); setActiveTab("analysis"); }}
                                style={{
                                    background: active ? C.amberDim : hovered ? C.bg2 : "transparent",
                                    border: `1px solid ${active ? C.amber + "55" : "transparent"}`,
                                    borderRadius: 6, padding: "8px 14px", cursor: "pointer",
                                    display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 2, minWidth: 80,
                                    transition: "background .15s",
                                }}
                            >
                                <span style={{ color: active ? C.amber : C.text, fontSize: 12, fontWeight: 700, fontFamily: "'Syne',sans-serif" }}>{t}</span>
                                <span style={{ color: C.textMid, fontSize: 10 }}>{q.price ? `$${q.price.toFixed(2)}` : "—"}</span>
                                <span style={{ color: (q.change || 0) >= 0 ? C.green : C.red, fontSize: 9 }}>
                                    {q.change != null ? `${q.change >= 0 ? "+" : ""}${q.change.toFixed(2)}%` : ""}
                                </span>
                            </button>

                            {/* × delete button — appears on hover */}
                            {hovered && (
                                <button
                                    onClick={e => { e.stopPropagation(); removeTicker(t); }}
                                    title={`Remove ${t}`}
                                    style={{
                                        position: "absolute", top: 3, right: 3,
                                        width: 16, height: 16, borderRadius: "50%",
                                        background: C.red + "dd", border: "none",
                                        color: "#fff", fontSize: 9, fontWeight: 800,
                                        cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                                        lineHeight: 1, padding: 0, zIndex: 10,
                                        animation: "fadeUp .12s ease both",
                                    }}
                                >✕</button>
                            )}
                        </div>
                    );
                })}

                {/* + Add slot — visible when < 8 */}
                {watchlist.length < 8 && (
                    <button
                        onClick={() => setShowModal(true)}
                        title="Add a stock to your watchlist"
                        style={{
                            background: "transparent",
                            border: `1.5px dashed ${C.border}`,
                            borderRadius: 6, padding: "8px 14px", cursor: "pointer",
                            color: C.textDim, fontSize: 11, display: "flex", alignItems: "center",
                            gap: 4, minWidth: 68, transition: "border-color .15s, color .15s",
                            fontFamily: "'DM Mono',monospace",
                        }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = C.amber; e.currentTarget.style.color = C.amber; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.color = C.textDim; }}
                    >
                        ＋ Add
                    </button>
                )}
            </div>

            {/* Loading bar */}
            {loading && (
                <div style={{
                    height: 2, background: `linear-gradient(90deg, transparent, ${C.amber}, transparent)`,
                    animation: "pulse 1s ease-in-out infinite"
                }} />
            )}

            {/* ── Main content ─────────────────────────────────────── */}
            <div style={{ padding: "24px 28px", maxWidth: 1400, margin: "0 auto" }}>
                {activeTab === "analysis" && (
                    <AnalysisTab
                        selectedTicker={selectedTicker}
                        setSelectedTicker={(t) => {
                            setSelectedTicker(t.toUpperCase().trim());
                        }}
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
                {activeTab === "heatmap" && (
                    <HeatmapTab apiConnected={apiConnected} />
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

            {/* AI Chat Widget */}
            <ChatWidget apiConnected={apiConnected} />
        </div>
    );
}
