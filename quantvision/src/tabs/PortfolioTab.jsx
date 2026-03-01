import { useState, useEffect } from "react";
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, PieChart, Pie, Cell
} from "recharts";
import { C, TICKERS } from "../utils/data";
import { fetchPrices, getCSVExportURL } from "../utils/api";
import { StatCard, Section, ChartTooltip } from "../components/UIComponents";

const PIE_COLORS = [C.amber, C.cyan, C.green, C.purple, C.red, "#f97316", "#ec4899", "#8b5cf6"];

export default function PortfolioTab({ notify, apiConnected, setSelectedTicker, setActiveTab }) {
    const [portfolio, setPortfolio] = useState([
        { ticker: "AAPL", shares: 50, avgCost: 170 },
        { ticker: "MSFT", shares: 30, avgCost: 350 },
        { ticker: "NVDA", shares: 20, avgCost: 780 },
        { ticker: "AMZN", shares: 15, avgCost: 175 },
    ]);
    const [liveData, setLiveData] = useState({});
    const [addTicker, setAddTicker] = useState("GOOGL");
    const [addShares, setAddShares] = useState(10);
    const [addCost, setAddCost] = useState(150);

    // Fetch live prices for portfolio holdings
    useEffect(() => {
        if (!apiConnected) return;
        const fetchHoldings = async () => {
            const data = {};
            for (const h of portfolio) {
                try {
                    const resp = await fetchPrices(h.ticker, "yfinance", 5);
                    if (resp.bars?.length) {
                        const last = resp.bars[resp.bars.length - 1];
                        data[h.ticker] = last.close;
                    }
                } catch { /* skip */ }
            }
            setLiveData(data);
        };
        fetchHoldings();
    }, [portfolio, apiConnected]);

    const holdings = portfolio.map(h => {
        const curr = liveData[h.ticker] || h.avgCost;
        const value = curr * h.shares;
        const cost = h.avgCost * h.shares;
        return { ...h, curr, value, cost, pnl: value - cost, pnlPct: ((value - cost) / cost) * 100 };
    });
    const totalValue = holdings.reduce((s, h) => s + h.value, 0);
    const totalCost = holdings.reduce((s, h) => s + h.cost, 0);
    const totalPnL = totalValue - totalCost;

    const addPosition = () => {
        if (!addTicker || addShares <= 0) return;
        const exists = portfolio.find(p => p.ticker === addTicker);
        if (exists) { notify?.("Ticker already in portfolio"); return; }
        setPortfolio([...portfolio, { ticker: addTicker, shares: addShares, avgCost: addCost }]);
        notify?.(`Added ${addTicker}`);
    };

    const removePosition = (ticker) => {
        setPortfolio(portfolio.filter(p => p.ticker !== ticker));
        notify?.(`Removed ${ticker}`);
    };

    if (!apiConnected) return <div style={{ textAlign: "center", padding: 60, color: C.textDim }}>🔌 Connect to API</div>;

    const inputStyle = {
        background: C.bg2, color: C.text, border: `1px solid ${C.border}`, borderRadius: 6,
        padding: "8px 12px", fontSize: 12, fontFamily: "'DM Mono',monospace",
    };

    return (
        <div className="fade-up">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 28, color: C.text }}>💼 Portfolio Manager</h1>
                <a href={getCSVExportURL("prices", portfolio[0]?.ticker)} style={{
                    background: C.bg2, color: C.amber, border: `1px solid ${C.border}`, borderRadius: 8,
                    padding: "8px 16px", fontSize: 11, textDecoration: "none",
                }}>📥 Export CSV</a>
            </div>

            {/* Summary stats */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 }}>
                <StatCard label="TOTAL VALUE" value={`$${(totalValue / 1000).toFixed(1)}K`} sub={`${holdings.length} positions`} color={C.amber} />
                <StatCard label="TOTAL COST" value={`$${(totalCost / 1000).toFixed(1)}K`} sub="Cost basis" color={C.textMid} />
                <StatCard label="TOTAL P&L" value={`$${totalPnL >= 0 ? "+" : ""}${totalPnL.toFixed(0)}`}
                    sub={`${((totalPnL / totalCost) * 100).toFixed(2)}%`} color={totalPnL >= 0 ? C.green : C.red} />
                <StatCard label="DAILY CHG" value="—" sub="Live" color={C.cyan} />
            </div>

            {/* Holdings table */}
            <Section title="HOLDINGS">
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead>
                        <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                            {["Ticker", "Shares", "Avg Cost", "Current", "Value", "P&L", "Weight", ""].map(h => (
                                <th key={h} style={{ padding: "8px 12px", color: C.textDim, textAlign: "right", fontWeight: 400 }}>{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {holdings.map(h => (
                            <tr key={h.ticker} style={{ borderBottom: `1px solid ${C.border}22`, cursor: "pointer" }}
                                onClick={() => { setSelectedTicker?.(h.ticker); setActiveTab?.("analysis"); }}>
                                <td style={{ padding: "8px 12px", color: C.amber, fontWeight: 700, textAlign: "left" }}>{h.ticker}</td>
                                <td style={{ padding: "8px 12px", color: C.text, textAlign: "right" }}>{h.shares}</td>
                                <td style={{ padding: "8px 12px", color: C.textMid, textAlign: "right" }}>${h.avgCost.toFixed(2)}</td>
                                <td style={{ padding: "8px 12px", color: C.text, textAlign: "right" }}>${h.curr.toFixed(2)}</td>
                                <td style={{ padding: "8px 12px", color: C.text, textAlign: "right", fontWeight: 700 }}>${h.value.toFixed(0)}</td>
                                <td style={{ padding: "8px 12px", color: h.pnl >= 0 ? C.green : C.red, textAlign: "right", fontWeight: 700 }}>
                                    {h.pnl >= 0 ? "+" : ""}${h.pnl.toFixed(0)} ({h.pnlPct.toFixed(1)}%)
                                </td>
                                <td style={{ padding: "8px 12px", color: C.textMid, textAlign: "right" }}>
                                    {totalValue > 0 ? `${((h.value / totalValue) * 100).toFixed(1)}%` : "—"}
                                </td>
                                <td style={{ padding: "8px 12px", textAlign: "right" }}>
                                    <button onClick={ev => { ev.stopPropagation(); removePosition(h.ticker); }} style={{
                                        background: "transparent", border: "none", color: C.red, cursor: "pointer", fontSize: 14,
                                    }}>✕</button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </Section>

            {/* Add position + Allocation chart */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
                <Section title="ADD POSITION">
                    <div style={{ display: "flex", gap: 8, alignItems: "flex-end", flexWrap: "wrap" }}>
                        <div>
                            <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Ticker</label>
                            <input value={addTicker} onChange={e => setAddTicker(e.target.value.toUpperCase())} style={{ ...inputStyle, width: 80 }} />
                        </div>
                        <div>
                            <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Shares</label>
                            <input type="number" value={addShares} onChange={e => setAddShares(+e.target.value)} style={{ ...inputStyle, width: 80 }} />
                        </div>
                        <div>
                            <label style={{ fontSize: 10, color: C.textDim, display: "block", marginBottom: 4 }}>Avg Cost</label>
                            <input type="number" value={addCost} onChange={e => setAddCost(+e.target.value)} style={{ ...inputStyle, width: 100 }} />
                        </div>
                        <button onClick={addPosition} style={{
                            background: C.amber, color: "#000", border: "none", borderRadius: 6, padding: "8px 16px",
                            fontSize: 12, fontWeight: 700, cursor: "pointer",
                        }}>+ Add</button>
                    </div>
                </Section>
                <Section title="ALLOCATION">
                    <ResponsiveContainer width="100%" height={200}>
                        <PieChart>
                            <Pie data={holdings.map(h => ({ name: h.ticker, value: h.value }))}
                                cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value"
                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                labelLine={false}>
                                {holdings.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                            </Pie>
                            <Tooltip />
                        </PieChart>
                    </ResponsiveContainer>
                </Section>
            </div>
        </div>
    );
}
