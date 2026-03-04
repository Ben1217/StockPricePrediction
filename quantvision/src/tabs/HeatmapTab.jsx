import { useState, useEffect, useRef, useCallback } from "react";
import { C } from "../utils/data";
import { fetchPrices } from "../utils/api";

/* ═══════════════════════════════════════════════════════════════
   SECTOR & COMPANY STATIC DATA  (11 GICS sectors, ~80 companies)
═══════════════════════════════════════════════════════════════ */
const SECTOR_DATA = [
    {
        id: "tech", name: "Information Technology", icon: "💻", indexWeight: 29.3,
        companies: [
            { ticker: "AAPL", name: "Apple Inc.", logo: "apple.com", marketCapB: 2870, price: 182.63, chg: 1.24, vol: 58234000, w52hi: 199.62, w52lo: 124.17 },
            { ticker: "MSFT", name: "Microsoft Corp.", logo: "microsoft.com", marketCapB: 2810, price: 378.85, chg: 0.87, vol: 22100000, w52hi: 430.82, w52lo: 275.37 },
            { ticker: "NVDA", name: "NVIDIA Corp.", logo: "nvidia.com", marketCapB: 2140, price: 875.43, chg: 3.21, vol: 43500000, w52hi: 974.00, w52lo: 402.95 },
            { ticker: "AVGO", name: "Broadcom Inc.", logo: "broadcom.com", marketCapB: 640, price: 168.23, chg: 0.54, vol: 8200000, w52hi: 185.16, w52lo: 103.42 },
            { ticker: "ORCL", name: "Oracle Corp.", logo: "oracle.com", marketCapB: 390, price: 142.18, chg: -0.33, vol: 9800000, w52hi: 153.83, w52lo: 100.46 },
            { ticker: "ADBE", name: "Adobe Inc.", logo: "adobe.com", marketCapB: 200, price: 446.31, chg: -1.12, vol: 3900000, w52hi: 570.42, w52lo: 419.19 },
            { ticker: "AMD", name: "Advanced Micro Devices", logo: "amd.com", marketCapB: 195, price: 120.15, chg: 2.78, vol: 47600000, w52hi: 207.00, w52lo: 99.13 },
            { ticker: "INTC", name: "Intel Corp.", logo: "intel.com", marketCapB: 92, price: 21.34, chg: -2.56, vol: 89200000, w52hi: 51.28, w52lo: 18.51 },
        ],
    },
    {
        id: "health", name: "Health Care", icon: "🏥", indexWeight: 12.4,
        companies: [
            { ticker: "LLY", name: "Eli Lilly & Co.", logo: "lilly.com", marketCapB: 740, price: 788.21, chg: 2.13, vol: 3600000, w52hi: 972.53, w52lo: 522.88 },
            { ticker: "UNH", name: "UnitedHealth Group", logo: "unitedhealthgroup.com", marketCapB: 480, price: 524.17, chg: -0.87, vol: 3100000, w52hi: 592.45, w52lo: 466.37 },
            { ticker: "JNJ", name: "Johnson & Johnson", logo: "jnj.com", marketCapB: 375, price: 155.72, chg: 0.24, vol: 9200000, w52hi: 168.38, w52lo: 143.13 },
            { ticker: "MRK", name: "Merck & Co.", logo: "merck.com", marketCapB: 310, price: 122.45, chg: -1.05, vol: 8700000, w52hi: 133.91, w52lo: 100.67 },
            { ticker: "ABT", name: "Abbott Laboratories", logo: "abbott.com", marketCapB: 195, price: 112.66, chg: 0.61, vol: 6300000, w52hi: 119.85, w52lo: 96.84 },
            { ticker: "TMO", name: "Thermo Fisher Scientific", logo: "thermofisher.com", marketCapB: 189, price: 489.34, chg: -0.42, vol: 2100000, w52hi: 612.71, w52lo: 466.17 },
            { ticker: "AMGN", name: "Amgen Inc.", logo: "amgen.com", marketCapB: 159, price: 270.18, chg: 1.35, vol: 3500000, w52hi: 328.89, w52lo: 246.11 },
        ],
    },
    {
        id: "finance", name: "Financials", icon: "🏦", indexWeight: 12.9,
        companies: [
            { ticker: "BRK.B", name: "Berkshire Hathaway B", logo: "berkshirehathaway.com", marketCapB: 875, price: 388.24, chg: 0.44, vol: 4500000, w52hi: 421.80, w52lo: 339.22 },
            { ticker: "JPM", name: "JPMorgan Chase", logo: "jpmorganchase.com", marketCapB: 570, price: 199.84, chg: 1.17, vol: 10800000, w52hi: 220.82, w52lo: 141.61 },
            { ticker: "V", name: "Visa Inc.", logo: "visa.com", marketCapB: 540, price: 271.93, chg: 0.72, vol: 5900000, w52hi: 290.96, w52lo: 227.83 },
            { ticker: "MA", name: "Mastercard Inc.", logo: "mastercard.com", marketCapB: 420, price: 452.37, chg: 0.58, vol: 3200000, w52hi: 484.42, w52lo: 373.04 },
            { ticker: "BAC", name: "Bank of America", logo: "bankofamerica.com", marketCapB: 300, price: 38.21, chg: -0.53, vol: 32100000, w52hi: 44.44, w52lo: 24.85 },
            { ticker: "GS", name: "Goldman Sachs", logo: "goldmansachs.com", marketCapB: 178, price: 476.22, chg: 1.83, vol: 2400000, w52hi: 521.36, w52lo: 288.76 },
            { ticker: "MS", name: "Morgan Stanley", logo: "morganstanley.com", marketCapB: 167, price: 100.43, chg: 0.96, vol: 7700000, w52hi: 107.89, w52lo: 66.76 },
        ],
    },
    {
        id: "consumer", name: "Consumer Discretionary", icon: "🛍️", indexWeight: 10.1,
        companies: [
            { ticker: "AMZN", name: "Amazon.com Inc.", logo: "amazon.com", marketCapB: 1940, price: 185.21, chg: 2.04, vol: 35800000, w52hi: 201.20, w52lo: 118.35 },
            { ticker: "TSLA", name: "Tesla Inc.", logo: "tesla.com", marketCapB: 780, price: 245.33, chg: -3.84, vol: 108500000, w52hi: 299.29, w52lo: 138.80 },
            { ticker: "HD", name: "Home Depot", logo: "homedepot.com", marketCapB: 355, price: 340.87, chg: -0.29, vol: 3600000, w52hi: 396.45, w52lo: 274.26 },
            { ticker: "MCD", name: "McDonald's Corp.", logo: "mcdonalds.com", marketCapB: 214, price: 291.48, chg: 0.12, vol: 3100000, w52hi: 317.46, w52lo: 243.47 },
            { ticker: "NKE", name: "Nike Inc.", logo: "nike.com", marketCapB: 100, price: 65.24, chg: -1.76, vol: 12900000, w52hi: 113.22, w52lo: 56.21 },
            { ticker: "SBUX", name: "Starbucks Corp.", logo: "starbucks.com", marketCapB: 85, price: 74.44, chg: -0.88, vol: 9400000, w52hi: 103.62, w52lo: 68.11 },
        ],
    },
    {
        id: "comms", name: "Communication Services", icon: "📡", indexWeight: 8.7,
        companies: [
            { ticker: "GOOGL", name: "Alphabet Inc. (A)", logo: "google.com", marketCapB: 1910, price: 155.64, chg: 1.33, vol: 23200000, w52hi: 191.75, w52lo: 120.21 },
            { ticker: "META", name: "Meta Platforms", logo: "meta.com", marketCapB: 1280, price: 505.33, chg: 1.77, vol: 16400000, w52hi: 589.39, w52lo: 274.38 },
            { ticker: "NFLX", name: "Netflix Inc.", logo: "netflix.com", marketCapB: 268, price: 620.48, chg: 0.54, vol: 5200000, w52hi: 741.25, w52lo: 344.73 },
            { ticker: "DIS", name: "Walt Disney Co.", logo: "disney.com", marketCapB: 195, price: 107.22, chg: -0.66, vol: 11800000, w52hi: 123.74, w52lo: 78.73 },
            { ticker: "T", name: "AT&T Inc.", logo: "att.com", marketCapB: 116, price: 16.43, chg: 0.18, vol: 42000000, w52hi: 22.01, w52lo: 13.94 },
        ],
    },
    {
        id: "industrial", name: "Industrials", icon: "⚙️", indexWeight: 8.4,
        companies: [
            { ticker: "GE", name: "GE Aerospace", logo: "ge.com", marketCapB: 195, price: 179.28, chg: 2.41, vol: 8700000, w52hi: 197.20, w52lo: 103.55 },
            { ticker: "CAT", name: "Caterpillar Inc.", logo: "cat.com", marketCapB: 177, price: 340.56, chg: 0.77, vol: 2900000, w52hi: 418.01, w52lo: 232.98 },
            { ticker: "HON", name: "Honeywell Intl.", logo: "honeywell.com", marketCapB: 128, price: 201.34, chg: -0.22, vol: 3400000, w52hi: 228.89, w52lo: 183.90 },
            { ticker: "UPS", name: "United Parcel Service", logo: "ups.com", marketCapB: 100, price: 115.42, chg: -1.43, vol: 5200000, w52hi: 162.76, w52lo: 107.64 },
            { ticker: "BA", name: "Boeing Co.", logo: "boeing.com", marketCapB: 92, price: 155.88, chg: -2.17, vol: 11300000, w52hi: 267.54, w52lo: 137.04 },
            { ticker: "RTX", name: "RTX Corp.", logo: "rtx.com", marketCapB: 148, price: 117.43, chg: 1.06, vol: 6400000, w52hi: 123.00, w52lo: 78.98 },
        ],
    },
    {
        id: "staples", name: "Consumer Staples", icon: "🛒", indexWeight: 6.1,
        companies: [
            { ticker: "PG", name: "Procter & Gamble", logo: "pg.com", marketCapB: 362, price: 152.32, chg: 0.31, vol: 6100000, w52hi: 171.64, w52lo: 139.62 },
            { ticker: "KO", name: "Coca-Cola Co.", logo: "coca-cola.com", marketCapB: 265, price: 61.24, chg: 0.14, vol: 14600000, w52hi: 67.20, w52lo: 51.55 },
            { ticker: "PEP", name: "PepsiCo Inc.", logo: "pepsico.com", marketCapB: 210, price: 151.67, chg: -0.41, vol: 5300000, w52hi: 182.70, w52lo: 148.30 },
            { ticker: "COST", name: "Costco Wholesale", logo: "costco.com", marketCapB: 385, price: 867.44, chg: 0.88, vol: 2300000, w52hi: 1009.27, w52lo: 663.80 },
            { ticker: "WMT", name: "Walmart Inc.", logo: "walmart.com", marketCapB: 790, price: 98.42, chg: 1.02, vol: 16200000, w52hi: 105.31, w52lo: 56.80 },
        ],
    },
    {
        id: "energy", name: "Energy", icon: "⚡", indexWeight: 3.7,
        companies: [
            { ticker: "XOM", name: "Exxon Mobil Corp.", logo: "exxonmobil.com", marketCapB: 487, price: 113.77, chg: 0.63, vol: 15800000, w52hi: 124.38, w52lo: 95.77 },
            { ticker: "CVX", name: "Chevron Corp.", logo: "chevron.com", marketCapB: 266, price: 145.21, chg: 0.27, vol: 8400000, w52hi: 167.11, w52lo: 132.63 },
            { ticker: "SLB", name: "Schlumberger Ltd.", logo: "slb.com", marketCapB: 56, price: 40.14, chg: -0.93, vol: 12300000, w52hi: 57.72, w52lo: 37.63 },
            { ticker: "COP", name: "ConocoPhillips", logo: "conocophillips.com", marketCapB: 128, price: 104.56, chg: 1.24, vol: 7100000, w52hi: 134.36, w52lo: 91.27 },
        ],
    },
    {
        id: "realestate", name: "Real Estate", icon: "🏢", indexWeight: 2.4,
        companies: [
            { ticker: "PLD", name: "Prologis Inc.", logo: "prologis.com", marketCapB: 94, price: 115.34, chg: -0.48, vol: 4300000, w52hi: 141.48, w52lo: 100.96 },
            { ticker: "AMT", name: "American Tower", logo: "americantower.com", marketCapB: 85, price: 183.22, chg: 0.72, vol: 2900000, w52hi: 229.72, w52lo: 163.05 },
            { ticker: "EQIX", name: "Equinix Inc.", logo: "equinix.com", marketCapB: 78, price: 832.51, chg: -0.17, vol: 780000, w52hi: 920.28, w52lo: 710.47 },
            { ticker: "SPG", name: "Simon Property Group", logo: "simon.com", marketCapB: 54, price: 162.44, chg: 0.39, vol: 2100000, w52hi: 177.97, w52lo: 107.32 },
        ],
    },
    {
        id: "utilities", name: "Utilities", icon: "💡", indexWeight: 2.3,
        companies: [
            { ticker: "NEE", name: "NextEra Energy", logo: "nexteraenergy.com", marketCapB: 136, price: 68.92, chg: 0.58, vol: 9400000, w52hi: 86.10, w52lo: 51.83 },
            { ticker: "DUK", name: "Duke Energy Corp.", logo: "duke-energy.com", marketCapB: 78, price: 106.34, chg: 0.22, vol: 3800000, w52hi: 115.84, w52lo: 89.32 },
            { ticker: "SO", name: "Southern Co.", logo: "southernco.com", marketCapB: 67, price: 68.21, chg: -0.14, vol: 5200000, w52hi: 91.45, w52lo: 63.73 },
            { ticker: "AEP", name: "American Electric Power", logo: "aep.com", marketCapB: 48, price: 97.44, chg: 0.41, vol: 3100000, w52hi: 105.67, w52lo: 73.28 },
        ],
    },
    {
        id: "materials", name: "Materials", icon: "⛏️", indexWeight: 2.7,
        companies: [
            { ticker: "LIN", name: "Linde plc", logo: "linde.com", marketCapB: 193, price: 443.28, chg: 0.82, vol: 1900000, w52hi: 481.68, w52lo: 367.19 },
            { ticker: "SHW", name: "Sherwin-Williams Co.", logo: "sherwin-williams.com", marketCapB: 92, price: 348.54, chg: -0.27, vol: 1600000, w52hi: 387.02, w52lo: 268.97 },
            { ticker: "ECL", name: "Ecolab Inc.", logo: "ecolab.com", marketCapB: 58, price: 201.33, chg: 0.44, vol: 2200000, w52hi: 241.41, w52lo: 171.22 },
            { ticker: "FCX", name: "Freeport-McMoRan Inc.", logo: "fcx.com", marketCapB: 61, price: 41.36, chg: -1.88, vol: 19400000, w52hi: 55.21, w52lo: 33.14 },
        ],
    },
];

/* ═══════════════════════════════════════════════════════════════
   SQUARIFIED TREEMAP ALGORITHM
═══════════════════════════════════════════════════════════════ */
function squarify(items, rect) {
    if (!items.length) return [];
    const total = items.reduce((s, it) => s + it._value, 0);
    const area = rect.w * rect.h;
    const results = [];

    function worst(row, w) {
        const s = row.reduce((a, b) => a + b._value, 0);
        const sNorm = s / total * area;
        const maxV = Math.max(...row.map(r => r._value / total * area));
        const minV = Math.min(...row.map(r => r._value / total * area));
        return Math.max((w * w * maxV) / (sNorm * sNorm), (sNorm * sNorm) / (w * w * minV));
    }

    function layoutRow(row, r) {
        const rowSum = row.reduce((a, b) => a + b._value, 0) / total * area;
        const isHoriz = r.w >= r.h;
        let x = r.x, y = r.y;
        const fixedDim = isHoriz ? rowSum / r.h : rowSum / r.w;
        row.forEach(it => {
            const cellArea = it._value / total * area;
            const varDim = fixedDim > 0 ? cellArea / fixedDim : 0;
            if (isHoriz) {
                results.push({ ...it, x, y, w: fixedDim, h: varDim });
                y += varDim;
            } else {
                results.push({ ...it, x, y, w: varDim, h: fixedDim });
                x += varDim;
            }
        });
        return isHoriz
            ? { x: r.x + fixedDim, y: r.y, w: r.w - fixedDim, h: r.h }
            : { x: r.x, y: r.y + fixedDim, w: r.w, h: r.h - fixedDim };
    }

    let remaining = [...items].sort((a, b) => b._value - a._value);
    let r = { ...rect };

    while (remaining.length) {
        const shortSide = Math.min(r.w, r.h);
        let row = [remaining[0]];
        let i = 1;
        while (i < remaining.length) {
            const newRow = [...row, remaining[i]];
            if (row.length > 1 && worst(newRow, shortSide) > worst(row, shortSide)) break;
            row = newRow;
            i++;
        }
        r = layoutRow(row, r);
        remaining = remaining.slice(row.length);
    }
    return results;
}

/* ═══════════════════════════════════════════════════════════════
   COLOUR HELPERS
═══════════════════════════════════════════════════════════════ */
function lerp(a, b, t) { return a + (b - a) * t; }

function chgToColor(chg) {
    const clamp = Math.max(-3, Math.min(3, chg));
    if (clamp < 0) {
        const t = -clamp / 3;
        return `rgb(${Math.round(lerp(23, 127, t))},${Math.round(lerp(38, 29, t))},${Math.round(lerp(61, 29, t))})`;
    }
    const t = clamp / 3;
    return `rgb(${Math.round(lerp(23, 4, t))},${Math.round(lerp(38, 78, t))},${Math.round(lerp(61, 59, t))})`;
}

function chgToText(chg) { return chg >= 0 ? C.green : C.red; }

function fmtChg(chg) { return `${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%`; }

function fmtVol(v) {
    if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
    return `${(v / 1e3).toFixed(0)}K`;
}

function fmtCap(b) {
    if (b >= 1000) return `$${(b / 1000).toFixed(2)}T`;
    return `$${b.toFixed(0)}B`;
}

/* ═══════════════════════════════════════════════════════════════
   LOGO CACHE  (module-level Map + localStorage persistence)
═══════════════════════════════════════════════════════════════ */
const LOGO_CACHE = new Map();
const LOGO_FAILED = new Set();
const LS_KEY = "qv_logo_cache_v1";

// Restore localStorage hits into the in-memory Map on first load
try {
    const stored = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
    Object.entries(stored).forEach(([domain, state]) => {
        if (state === "failed") LOGO_FAILED.add(domain);
        else LOGO_CACHE.set(domain, state);
    });
} catch { /* ignore */ }

function persistCache() {
    try {
        const obj = {};
        LOGO_CACHE.forEach((v, k) => { obj[k] = v; });
        LOGO_FAILED.forEach(k => { obj[k] = "failed"; });
        localStorage.setItem(LS_KEY, JSON.stringify(obj));
    } catch { /* ignore */ }
}

function logoUrl(domain) {
    return `https://logo.clearbit.com/${domain}?size=64`;
}

// Initials fallback — first letter of each word, max 2 chars
function getInitials(name) {
    return name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase();
}

// Deterministic pastel hue from ticker string
function tickerHue(ticker) {
    let h = 0;
    for (let i = 0; i < ticker.length; i++) h = (h * 31 + ticker.charCodeAt(i)) % 360;
    return h;
}

/* ─── CompanyLogo component ─────────────────────────────────── */
function CompanyLogo({ ticker, name, domain, size = 20 }) {
    const [imgState, setImgState] = useState(() => {
        if (!domain || LOGO_FAILED.has(domain)) return "failed";
        if (LOGO_CACHE.has(domain)) return "ok";
        return "loading";
    });

    const src = domain ? logoUrl(domain) : null;
    const hue = tickerHue(ticker);
    const initials = getInitials(name);

    const onLoad = () => {
        LOGO_CACHE.set(domain, src);
        persistCache();
        setImgState("ok");
    };

    const onError = () => {
        LOGO_FAILED.add(domain);
        persistCache();
        setImgState("failed");
    };

    if (imgState === "failed" || !src) {
        // Initials placeholder
        return (
            <div
                aria-label={`${name} logo`}
                role="img"
                style={{
                    width: size, height: size, borderRadius: 4, flexShrink: 0,
                    background: `hsl(${hue},55%,25%)`,
                    border: `1px solid hsl(${hue},55%,40%)`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: size * 0.38, fontWeight: 700, color: `hsl(${hue},80%,80%)`,
                    fontFamily: "'Syne',sans-serif", userSelect: "none",
                }}
            >{initials}</div>
        );
    }

    return (
        <>
            {/* img is always mounted; show initials placeholder until onLoad */}
            {imgState === "loading" && (
                <div
                    aria-hidden="true"
                    style={{
                        width: size, height: size, borderRadius: 4, flexShrink: 0,
                        background: `hsl(${hue},55%,20%)`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: size * 0.38, color: `hsl(${hue},80%,70%)`,
                        fontFamily: "'Syne',sans-serif",
                    }}
                >{initials}</div>
            )}
            <img
                src={src}
                alt={`${name} logo`}
                loading="lazy"
                onLoad={onLoad}
                onError={onError}
                style={{
                    width: size, height: size, borderRadius: 4, flexShrink: 0,
                    objectFit: "contain",
                    display: imgState === "ok" ? "block" : "none",
                }}
            />
        </>
    );
}

/* ═══════════════════════════════════════════════════════════════
   SPARKLINE (pure SVG, 7-point)
═══════════════════════════════════════════════════════════════ */
function Sparkline({ data, color, width = 72, height = 24 }) {
    if (!data || data.length < 2) return null;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const pts = data.map((v, i) => {
        const x = (i / (data.length - 1)) * width;
        const y = height - ((v - min) / range) * (height - 2) - 1;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    return (
        <svg width={width} height={height} style={{ display: "block" }}>
            <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
        </svg>
    );
}

/* ═══════════════════════════════════════════════════════════════
   SECTOR TOOLTIP
═══════════════════════════════════════════════════════════════ */
function SectorTooltip({ sector, mousePos, containerRef }) {
    if (!sector) return null;
    const avgChg = sector.companies.reduce((s, c) => s + c.chg, 0) / sector.companies.length;
    const best = [...sector.companies].sort((a, b) => b.chg - a.chg)[0];
    const worst = [...sector.companies].sort((a, b) => a.chg - b.chg)[0];
    const sparkData = sector.companies.map((c, i) => c.price * (1 + Math.sin(i * 0.8) * 0.03));
    const borderCol = avgChg >= 0 ? C.green : C.red;
    return (
        <div style={{
            position: "fixed",
            left: mousePos.x + 16,
            top: mousePos.y - 20,
            zIndex: 1000,
            background: "rgba(8,12,20,.95)",
            border: `1px solid ${borderCol}55`,
            borderRadius: 10,
            padding: "12px 16px",
            backdropFilter: "blur(14px)",
            minWidth: 210,
            maxWidth: 260,
            boxShadow: `0 8px 32px rgba(0,0,0,.55), 0 0 0 1px ${borderCol}22`,
            pointerEvents: "none",
            fontFamily: "'DM Mono',monospace",
            fontSize: 11,
        }}>
            <div style={{ fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 13, color: C.text, marginBottom: 8 }}>
                {sector.icon}&nbsp;{sector.name}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ color: C.textDim }}>Avg change</span>
                <span style={{ color: chgToText(avgChg), fontWeight: 700 }}>{fmtChg(avgChg)}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ color: C.textDim }}>Companies</span>
                <span style={{ color: C.text }}>{sector.companies.length}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                <span style={{ color: C.textDim }}>Index weight</span>
                <span style={{ color: C.text }}>{sector.indexWeight.toFixed(1)}%</span>
            </div>
            <div style={{ height: 1, background: `${C.border}`, marginBottom: 8 }} />
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ color: C.green, fontSize: 10 }}>▲ Best</span>
                <span style={{ color: C.green }}>{best.ticker} {fmtChg(best.chg)}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                <span style={{ color: C.red, fontSize: 10 }}>▼ Worst</span>
                <span style={{ color: C.red }}>{worst.ticker} {fmtChg(worst.chg)}</span>
            </div>
            <Sparkline data={sparkData} color={borderCol} />
            <div style={{ marginTop: 8, color: C.textDim, fontSize: 10, textAlign: "center" }}>Click to drill in →</div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════
   COMPANY TOOLTIP
═══════════════════════════════════════════════════════════════ */
function CompanyTooltip({ company, mousePos }) {
    if (!company) return null;
    const borderCol = company.chg >= 0 ? C.green : C.red;
    const sparkData = [company.w52lo, company.price * 0.97, company.price * 0.99, company.price, company.price * 1.01, company.price * 0.98, company.price];
    return (
        <div style={{
            position: "fixed",
            left: mousePos.x + 16,
            top: mousePos.y - 20,
            zIndex: 1000,
            background: "rgba(8,12,20,.95)",
            border: `1px solid ${borderCol}55`,
            borderRadius: 10,
            padding: "12px 16px",
            backdropFilter: "blur(14px)",
            minWidth: 200,
            maxWidth: 250,
            boxShadow: `0 8px 32px rgba(0,0,0,.55), 0 0 0 1px ${borderCol}22`,
            pointerEvents: "none",
            fontFamily: "'DM Mono',monospace",
            fontSize: 11,
        }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 13, color: C.amber }}>{company.ticker}</span>
                <span style={{ color: chgToText(company.chg), fontWeight: 700 }}>{fmtChg(company.chg)}</span>
            </div>
            <div style={{ color: C.textMid, fontSize: 10, marginBottom: 8 }}>{company.name}</div>
            <div style={{ height: 1, background: C.border, marginBottom: 8 }} />
            {[
                ["Price", `$${company.price.toFixed(2)}`],
                ["Volume", fmtVol(company.vol)],
                ["Market Cap", fmtCap(company.marketCapB)],
                ["52W High", `$${company.w52hi.toFixed(2)}`],
                ["52W Low", `$${company.w52lo.toFixed(2)}`],
            ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ color: C.textDim }}>{k}</span>
                    <span style={{ color: C.text }}>{v}</span>
                </div>
            ))}
            <div style={{ marginTop: 8 }}>
                <Sparkline data={sparkData} color={borderCol} />
            </div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════
   TREEMAP CELL
═══════════════════════════════════════════════════════════════ */
const GAP = 2;

function SectorCell({ cell, onClick, onHover, onLeave, onMouseMove }) {
    const [hovered, setHovered] = useState(false);
    const bigMover = Math.abs(cell.chg) >= 2.5;
    const w = cell.w - GAP;
    const h = cell.h - GAP;

    return (
        <div
            onClick={() => onClick(cell)}
            onMouseEnter={() => { setHovered(true); onHover(cell); }}
            onMouseLeave={() => { setHovered(false); onLeave(); }}
            onMouseMove={onMouseMove}
            style={{
                position: "absolute",
                left: cell.x,
                top: cell.y,
                width: w,
                height: h,
                background: chgToColor(cell.chg),
                borderRadius: 6,
                overflow: "hidden",
                cursor: "pointer",
                transition: "filter .15s, transform .15s",
                transform: hovered ? "scale(1.025)" : "scale(1)",
                filter: hovered ? "brightness(1.25)" : "brightness(1)",
                border: `1px solid rgba(255,255,255,.07)`,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                padding: "6px 4px",
                userSelect: "none",
                zIndex: hovered ? 10 : 1,
            }}
        >
            {/* Light reflection gradient */}
            <div style={{
                position: "absolute", inset: 0, borderRadius: 6,
                background: "linear-gradient(135deg, rgba(255,255,255,.09) 0%, transparent 50%)",
                pointerEvents: "none",
            }} />

            {/* Shimmer for big movers */}
            {bigMover && (
                <div style={{
                    position: "absolute", inset: 0, borderRadius: 6,
                    background: "linear-gradient(105deg, transparent 40%, rgba(255,255,255,.08) 50%, transparent 60%)",
                    backgroundSize: "200% 100%",
                    animation: "shimmer 2.2s linear infinite",
                    pointerEvents: "none",
                }} />
            )}

            {h > 40 && (
                <>
                    {/* Logo row: logo + sector icon/name side by side when wide enough */}
                    <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 3 }}>
                        {cell.icon && <span style={{ fontSize: Math.min(16, h * 0.15), lineHeight: 1 }}>{cell.icon}</span>}
                    </div>
                    <div style={{
                        fontSize: Math.min(11, Math.max(8, w * 0.1)),
                        fontFamily: "'Syne',sans-serif",
                        fontWeight: 700,
                        color: "#fff",
                        textAlign: "center",
                        lineHeight: 1.2,
                        maxWidth: "90%",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: h > 60 ? "normal" : "nowrap",
                    }}>
                        {w > 120 ? cell.name : cell.name.split(" ")[0]}
                    </div>
                    <div style={{ fontSize: Math.min(12, Math.max(9, w * 0.09)), color: chgToText(cell.chg), fontWeight: 700, marginTop: 2 }}>
                        {fmtChg(cell.chg)}
                    </div>
                    {h > 90 && w > 100 && (
                        <div style={{ marginTop: 4, display: "flex", gap: 4, flexWrap: "wrap", justifyContent: "center" }}>
                            {cell.topTickers?.slice(0, 3).map(t => (
                                <span key={t} style={{ fontSize: 8, color: "rgba(255,255,255,.55)", background: "rgba(0,0,0,.25)", borderRadius: 3, padding: "1px 4px" }}>{t}</span>
                            ))}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

function CompanyCell({ cell, onHover, onLeave, onMouseMove }) {
    const [hovered, setHovered] = useState(false);
    const bigMover = Math.abs(cell.chg) >= 2.5;
    const w = cell.w - GAP;
    const h = cell.h - GAP;

    return (
        <div
            onMouseEnter={() => { setHovered(true); onHover(cell); }}
            onMouseLeave={() => { setHovered(false); onLeave(); }}
            onMouseMove={onMouseMove}
            style={{
                position: "absolute",
                left: cell.x,
                top: cell.y,
                width: w,
                height: h,
                background: chgToColor(cell.chg),
                borderRadius: 5,
                overflow: "hidden",
                cursor: "default",
                transition: "filter .15s, transform .15s",
                transform: hovered ? "scale(1.03)" : "scale(1)",
                filter: hovered ? "brightness(1.3)" : "brightness(1)",
                border: "1px solid rgba(255,255,255,.07)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                padding: "4px 3px",
                userSelect: "none",
                zIndex: hovered ? 10 : 1,
            }}
        >
            <div style={{
                position: "absolute", inset: 0, borderRadius: 5,
                background: "linear-gradient(135deg, rgba(255,255,255,.09) 0%, transparent 50%)",
                pointerEvents: "none",
            }} />
            {bigMover && (
                <div style={{
                    position: "absolute", inset: 0, borderRadius: 5,
                    background: "linear-gradient(105deg, transparent 40%, rgba(255,255,255,.08) 50%, transparent 60%)",
                    backgroundSize: "200% 100%",
                    animation: "shimmer 2.2s linear infinite",
                    pointerEvents: "none",
                }} />
            )}

            {h > 20 && (
                <>
                    {/* Logo — centered on tiny tiles, top-left badge on large */}
                    {w > 50 && h > 30 ? (
                        <div style={{
                            position: "absolute",
                            top: 4, left: 5,
                            opacity: 0.92,
                        }}>
                            <CompanyLogo
                                ticker={cell.ticker}
                                name={cell.name}
                                domain={cell.logo}
                                size={Math.min(22, Math.max(14, Math.min(w, h) * 0.22))}
                            />
                        </div>
                    ) : h > 20 && w > 28 && (
                        <div style={{ marginBottom: 1 }}>
                            <CompanyLogo ticker={cell.ticker} name={cell.name} domain={cell.logo} size={14} />
                        </div>
                    )}

                    <div style={{ fontSize: Math.min(11, Math.max(8, Math.min(w, h) * 0.18)), color: C.amber, fontFamily: "'Syne',sans-serif", fontWeight: 700, lineHeight: 1 }}>
                        {cell.ticker}
                    </div>
                    {h > 36 && (
                        <div style={{ fontSize: Math.min(9, w * 0.085), color: "rgba(255,255,255,.6)", marginTop: 1, textAlign: "center", maxWidth: "92%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {cell.name.length > 14 ? cell.name.slice(0, 12) + "…" : cell.name}
                        </div>
                    )}
                    <div style={{ fontSize: Math.min(10, Math.max(8, w * 0.09)), color: chgToText(cell.chg), fontWeight: 700, marginTop: 2 }}>
                        {fmtChg(cell.chg)}
                    </div>
                    {h > 60 && w > 80 && (
                        <>
                            <div style={{ fontSize: 8, color: "rgba(255,255,255,.5)", marginTop: 2 }}>${cell.price.toFixed(2)}</div>
                            <div style={{ fontSize: 8, color: "rgba(255,255,255,.4)" }}>{fmtCap(cell.marketCapB)}</div>
                        </>
                    )}
                </>
            )}
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN HEATMAP TAB
═══════════════════════════════════════════════════════════════ */
export default function HeatmapTab({ apiConnected }) {
    // ── State ──────────────────────────────────────────────────────
    const [sectors, setSectors] = useState(SECTOR_DATA);
    const [drillSector, setDrillSector] = useState(null); // null = sector view
    const [viewMode, setViewMode] = useState("sectors"); // "sectors" | "allstocks"
    const [sortMode, setSortMode] = useState("cap"); // "cap" | "gainers" | "losers"
    const [search, setSearch] = useState("");
    const [status, setStatus] = useState("mock"); // "mock" | "fetching" | "live" | "error"
    const [liveTime, setLiveTime] = useState(null);
    const [fetchError, setFetchError] = useState(null);
    const [hoveredSector, setHoveredSector] = useState(null);
    const [hoveredCompany, setHoveredCompany] = useState(null);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

    const containerRef = useRef(null);
    const [containerSize, setContainerSize] = useState({ w: 900, h: 520 });
    const fetchingRef = useRef(false);

    // ── Resize observer ─────────────────────────────────────────────
    useEffect(() => {
        if (!containerRef.current) return;
        const ro = new ResizeObserver(entries => {
            const { width, height } = entries[0].contentRect;
            setContainerSize({ w: Math.floor(width), h: Math.floor(height) });
        });
        ro.observe(containerRef.current);
        return () => ro.disconnect();
    }, []);

    // ── Live data fetch ─────────────────────────────────────────────
    const fetchLiveData = useCallback(async () => {
        if (fetchingRef.current || !apiConnected) return;
        fetchingRef.current = true;
        setStatus("fetching");
        setFetchError(null);

        const allTickers = SECTOR_DATA.flatMap(s => s.companies.map(c => c.ticker));

        try {
            const results = await Promise.allSettled(
                allTickers.map(ticker =>
                    fetchPrices(ticker.replace(".", "-"), "yfinance", 5).then(r => {
                        if (r?.bars?.length >= 2) {
                            const last = r.bars[r.bars.length - 1];
                            const prev = r.bars[r.bars.length - 2];
                            return {
                                ticker,
                                price: last.close,
                                chg: ((last.close - prev.close) / prev.close) * 100,
                                vol: last.volume || 0,
                            };
                        }
                        return null;
                    })
                )
            );

            const liveMap = {};
            results.forEach(res => {
                if (res.status === "fulfilled" && res.value) {
                    liveMap[res.value.ticker] = res.value;
                }
            });

            setSectors(prev => prev.map(sector => ({
                ...sector,
                companies: sector.companies.map(company => {
                    const live = liveMap[company.ticker];
                    if (!live) return company;
                    return { ...company, price: live.price, chg: live.chg, vol: live.vol };
                }),
            })));

            const time = new Date().toLocaleTimeString();
            setLiveTime(time);
            setStatus("live");
        } catch (err) {
            setFetchError(err.message || "Fetch failed");
            setStatus("error");
        } finally {
            fetchingRef.current = false;
        }
    }, [apiConnected]);

    // Auto-fetch on mount
    useEffect(() => {
        fetchLiveData();
    }, [fetchLiveData]);

    // Preload logos for top-30 companies by market cap
    useEffect(() => {
        const top30 = SECTOR_DATA
            .flatMap(s => s.companies)
            .sort((a, b) => b.marketCapB - a.marketCapB)
            .slice(0, 30);
        top30.forEach(c => {
            if (!c.logo || LOGO_FAILED.has(c.logo) || LOGO_CACHE.has(c.logo)) return;
            const img = new Image();
            img.src = logoUrl(c.logo);
            img.onload = () => { LOGO_CACHE.set(c.logo, img.src); persistCache(); };
            img.onerror = () => { LOGO_FAILED.add(c.logo); persistCache(); };
        });
    }, []);

    // ── Derived data ────────────────────────────────────────────────
    const allCompanies = sectors.flatMap(s => s.companies.map(c => ({ ...c, sectorName: s.name, sectorIcon: s.icon })));

    const advancing = allCompanies.filter(c => c.chg >= 0).length;
    const declining = allCompanies.filter(c => c.chg < 0).length;

    const sectorsWithMeta = sectors.map(s => ({
        ...s,
        chg: s.companies.reduce((sum, c) => sum + c.chg, 0) / s.companies.length,
        topTickers: [...s.companies].sort((a, b) => b.marketCapB - a.marketCapB).slice(0, 3).map(c => c.ticker),
    }));

    // ── Filter by search ────────────────────────────────────────────
    const q = search.toLowerCase().trim();

    const filteredSectors = sectorsWithMeta.map(s => ({
        ...s,
        companies: s.companies.filter(c =>
            !q || c.ticker.toLowerCase().includes(q) || c.name.toLowerCase().includes(q) || s.name.toLowerCase().includes(q)
        ),
    })).filter(s => !q || s.companies.length > 0);

    // ── Sort companies in drill view ────────────────────────────────
    function getSortedCompanies(companies) {
        const sorted = [...companies];
        if (sortMode === "cap") sorted.sort((a, b) => b.marketCapB - a.marketCapB);
        else if (sortMode === "gainers") sorted.sort((a, b) => b.chg - a.chg);
        else if (sortMode === "losers") sorted.sort((a, b) => a.chg - b.chg);
        return sorted;
    }

    // ── Treemap inputs ──────────────────────────────────────────────
    let treemapItems = [];
    if (viewMode === "sectors") {
        treemapItems = filteredSectors.map(s => ({ ...s, _value: s.indexWeight }));
    } else if (viewMode === "allstocks") {
        const companies = q
            ? allCompanies.filter(c => c.ticker.toLowerCase().includes(q) || c.name.toLowerCase().includes(q))
            : allCompanies;
        treemapItems = getSortedCompanies(companies).map(c => ({ ...c, _value: c.marketCapB }));
    } else if (drillSector) {
        const drillData = filteredSectors.find(s => s.id === drillSector.id);
        const companies = drillData ? getSortedCompanies(drillData.companies) : [];
        treemapItems = companies.map(c => ({ ...c, _value: c.marketCapB }));
    }

    const cells = containerSize.w > 0 && containerSize.h > 0 && treemapItems.length > 0
        ? squarify(treemapItems, { x: 0, y: 0, w: containerSize.w, h: containerSize.h })
        : [];

    // ── Handlers ────────────────────────────────────────────────────
    const handleMouseMove = (e) => setMousePos({ x: e.clientX, y: e.clientY });

    const handleSectorClick = (cell) => {
        setDrillSector(cell);
        setViewMode("drill");
        setSearch("");
        setSortMode("cap");
    };

    const drillIntoSector = (sectorId) => {
        const s = sectorsWithMeta.find(x => x.id === sectorId);
        if (s) { setDrillSector(s); setViewMode("drill"); }
    };

    const goBack = () => { setDrillSector(null); setViewMode("sectors"); setSearch(""); };

    // ── Status badge ────────────────────────────────────────────────
    const StatusBadge = () => {
        if (status === "fetching") return (
            <span style={{ fontSize: 10, color: C.amber, display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ animation: "pulse 1s ease-in-out infinite", display: "inline-block" }}>⟳</span> FETCHING LIVE DATA…
            </span>
        );
        if (status === "live") return (
            <span style={{ fontSize: 10, color: C.green, display: "flex", alignItems: "center", gap: 4 }}>
                <span className="live-dot" style={{ width: 6, height: 6, borderRadius: "50%", background: C.green, display: "inline-block" }} />
                LIVE · {liveTime}
            </span>
        );
        if (status === "error") return (
            <span style={{ fontSize: 10, color: C.red }}>⚠ FETCH FAILED</span>
        );
        return <span style={{ fontSize: 10, color: C.textDim }}>◌ MOCK DATA</span>;
    };

    /* ── RENDER ──────────────────────────────────────────────────── */
    return (
        <div style={{ fontFamily: "'DM Mono',monospace", animation: "fadeUp .35s ease both" }}>

            {/* ── Advancing / Declining Bar ────── */}
            <div style={{ marginBottom: 6 }}>
                <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", marginBottom: 4 }}>
                    <div style={{ flex: advancing, background: C.green, opacity: 0.85, transition: "flex .6s ease" }} />
                    <div style={{ flex: declining, background: C.red, opacity: 0.85, transition: "flex .6s ease" }} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: C.textDim }}>
                    <span style={{ color: C.green }}>▲ {advancing} advancing</span>
                    <span style={{ color: C.textDim, fontSize: 9 }}>S&amp;P 500 Components</span>
                    <span style={{ color: C.red }}>{declining} declining ▼</span>
                </div>
            </div>

            {/* ── Sector Strip ─────────────────── */}
            <div style={{ display: "flex", height: 10, borderRadius: 4, overflow: "hidden", marginBottom: 10, gap: 1 }}>
                {sectorsWithMeta.map(s => (
                    <div
                        key={s.id}
                        title={s.name}
                        onClick={() => drillIntoSector(s.id)}
                        style={{
                            flex: s.indexWeight,
                            background: chgToColor(s.chg),
                            cursor: "pointer",
                            transition: "filter .15s",
                            borderRadius: 2,
                            minWidth: 4,
                        }}
                        onMouseEnter={e => { e.currentTarget.style.filter = "brightness(1.5)"; }}
                        onMouseLeave={e => { e.currentTarget.style.filter = "brightness(1)"; }}
                    />
                ))}
            </div>

            {/* ── Controls Row ─────────────────── */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                {/* Back button */}
                {viewMode === "drill" && (
                    <button onClick={goBack} style={{
                        background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 6,
                        color: C.amber, padding: "5px 12px", cursor: "pointer", fontSize: 11,
                        fontFamily: "'DM Mono',monospace", display: "flex", alignItems: "center", gap: 4,
                    }}>← All Sectors</button>
                )}

                {/* Drill-view sort pills */}
                {viewMode === "drill" && (
                    <div style={{ display: "flex", gap: 4 }}>
                        {[["cap", "Market Cap"], ["gainers", "Top Gainers ▲"], ["losers", "Top Losers ▼"]].map(([k, label]) => (
                            <button key={k} onClick={() => setSortMode(k)} style={{
                                background: sortMode === k ? C.amberDim : C.bg2,
                                border: `1px solid ${sortMode === k ? C.amber + "66" : C.border}`,
                                borderRadius: 20, color: sortMode === k ? C.amber : C.textMid,
                                padding: "4px 10px", cursor: "pointer", fontSize: 10,
                                fontFamily: "'DM Mono',monospace",
                            }}>{label}</button>
                        ))}
                    </div>
                )}

                {/* View toggle */}
                {viewMode !== "drill" && (
                    <div style={{ display: "flex", gap: 4 }}>
                        {[["sectors", "Sectors"], ["allstocks", "All Stocks"]].map(([k, label]) => (
                            <button key={k} onClick={() => { setViewMode(k); setSearch(""); }} style={{
                                background: viewMode === k ? C.amberDim : C.bg2,
                                border: `1px solid ${viewMode === k ? C.amber + "66" : C.border}`,
                                borderRadius: 20, color: viewMode === k ? C.amber : C.textMid,
                                padding: "4px 12px", cursor: "pointer", fontSize: 10,
                                fontFamily: "'DM Mono',monospace",
                            }}>{label}</button>
                        ))}
                    </div>
                )}

                {/* Search */}
                <div style={{ position: "relative", flexGrow: 1, maxWidth: 220 }}>
                    <span style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", color: C.textDim, fontSize: 11 }}>🔍</span>
                    <input
                        placeholder="Filter ticker / company…"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        style={{
                            background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 6,
                            color: C.text, padding: "5px 10px 5px 26px", fontSize: 11,
                            fontFamily: "'DM Mono',monospace", width: "100%", outline: "none",
                        }}
                    />
                </div>

                <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
                    <StatusBadge />
                    <button
                        onClick={fetchLiveData}
                        disabled={status === "fetching"}
                        style={{
                            background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 6,
                            color: C.textMid, padding: "5px 10px", cursor: status === "fetching" ? "not-allowed" : "pointer",
                            fontSize: 11, fontFamily: "'DM Mono',monospace",
                            opacity: status === "fetching" ? 0.5 : 1,
                        }}
                    >
                        {status === "fetching" ? "…" : "↻ Refresh"}
                    </button>
                </div>
            </div>

            {/* ── Drill header ─────────────────── */}
            {viewMode === "drill" && drillSector && (
                <div style={{ marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 20 }}>{drillSector.icon}</span>
                    <span style={{ fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 15, color: C.text }}>{drillSector.name}</span>
                    <span style={{ fontSize: 11, color: chgToText(drillSector.chg), fontWeight: 700 }}>{fmtChg(drillSector.chg)}</span>
                    <span style={{ fontSize: 10, color: C.textDim, marginLeft: 4 }}>{drillSector.indexWeight?.toFixed(1)}% of index</span>
                </div>
            )}

            {/* ── Error Banner ─────────────────── */}
            {fetchError && (
                <div style={{
                    background: `${C.red}18`, border: `1px solid ${C.red}44`, borderRadius: 6,
                    padding: "8px 14px", marginBottom: 10, display: "flex", justifyContent: "space-between", alignItems: "center",
                    fontSize: 11, color: C.red,
                }}>
                    <span>⚠ {fetchError} — Showing last known data.</span>
                    <button onClick={fetchLiveData} style={{
                        background: C.red + "22", border: `1px solid ${C.red}66`, borderRadius: 4,
                        color: C.red, padding: "3px 10px", cursor: "pointer", fontSize: 10, fontFamily: "'DM Mono',monospace",
                    }}>Retry</button>
                </div>
            )}

            {/* ── Treemap Container ─────────────── */}
            <div
                ref={containerRef}
                onMouseMove={handleMouseMove}
                style={{
                    position: "relative",
                    width: "100%",
                    height: 490,
                    borderRadius: 10,
                    overflow: "hidden",
                    border: `1px solid ${C.border}`,
                    background: C.bg1,
                }}
            >
                {/* Loading overlay */}
                {status === "fetching" && (
                    <div style={{
                        position: "absolute", inset: 0, zIndex: 50,
                        background: "rgba(8,12,20,.72)", backdropFilter: "blur(6px)",
                        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16,
                    }}>
                        <div style={{
                            width: 40, height: 40, borderRadius: "50%",
                            border: `3px solid ${C.border}`, borderTopColor: C.amber,
                            animation: "spin 0.8s linear infinite",
                        }} />
                        <div style={{ fontSize: 12, color: C.textMid }}>Fetching live data for {allCompanies.length} tickers…</div>
                    </div>
                )}

                {/* Cells */}
                {cells.map((cell, i) => (
                    viewMode === "sectors"
                        ? <SectorCell key={cell.id || i} cell={cell} onClick={handleSectorClick} onHover={setHoveredSector} onLeave={() => setHoveredSector(null)} onMouseMove={handleMouseMove} />
                        : <CompanyCell key={cell.ticker || i} cell={cell} onHover={setHoveredCompany} onLeave={() => setHoveredCompany(null)} onMouseMove={handleMouseMove} />
                ))}

                {cells.length === 0 && status !== "fetching" && (
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: C.textDim, fontSize: 13 }}>
                        No data matches your filter.
                    </div>
                )}
            </div>

            {/* ── Sector Chips ─────────────────── */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 12 }}>
                {sectorsWithMeta.map(s => (
                    <button
                        key={s.id}
                        onClick={() => drillIntoSector(s.id)}
                        style={{
                            background: chgToColor(s.chg),
                            border: `1px solid rgba(255,255,255,.12)`,
                            borderRadius: 20, color: "#fff",
                            padding: "4px 12px", cursor: "pointer",
                            fontSize: 11, fontFamily: "'DM Mono',monospace",
                            display: "flex", alignItems: "center", gap: 5,
                            transition: "filter .15s",
                        }}
                        onMouseEnter={e => e.currentTarget.style.filter = "brightness(1.35)"}
                        onMouseLeave={e => e.currentTarget.style.filter = "brightness(1)"}
                    >
                        {s.icon} {s.name.split(" ")[0]}
                        <span style={{ color: chgToText(s.chg), fontWeight: 700, marginLeft: 2 }}>{fmtChg(s.chg)}</span>
                    </button>
                ))}
            </div>

            {/* ── Colour Legend ─────────────────── */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 14, fontSize: 10, color: C.textDim }}>
                <span style={{ color: C.red }}>-4%</span>
                <div style={{
                    flex: 1, height: 8, borderRadius: 4, maxWidth: 200,
                    background: "linear-gradient(90deg, #7f1d1d, #17263d, #064e3b)",
                }} />
                <span style={{ color: C.green }}>+4%</span>
                <span style={{ marginLeft: 16, color: C.textDim }}>Box size = market cap / index weight</span>
            </div>

            {/* Tooltips */}
            {hoveredSector && viewMode === "sectors" && (
                <SectorTooltip sector={hoveredSector} mousePos={mousePos} />
            )}
            {hoveredCompany && viewMode !== "sectors" && (
                <CompanyTooltip company={hoveredCompany} mousePos={mousePos} />
            )}
        </div>
    );
}
