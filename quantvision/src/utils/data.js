/* ─── Seed-based pseudo random (deterministic) ─────────────────── */
function seededRand(seed) {
    let s = seed;
    return () => {
        s = (s * 16807 + 0) % 2147483647;
        return (s - 1) / 2147483646;
    };
}

/* ─── Generate price series ──────────────────────────────────── */
export function genPriceSeries(ticker, days = 120) {
    const bases = { AAPL: 182, MSFT: 378, GOOGL: 155, AMZN: 185, NVDA: 875, TSLA: 245, META: 505, NFLX: 620 };
    const vols = { AAPL: .012, MSFT: .011, GOOGL: .013, AMZN: .015, NVDA: .022, TSLA: .028, META: .016, NFLX: .017 };
    const drifts = { AAPL: .0003, MSFT: .0004, GOOGL: .0002, AMZN: .0003, NVDA: .0008, TSLA: -.0001, META: .0005, NFLX: .0004 };
    const rand = seededRand(ticker.charCodeAt(0) * 31 + ticker.charCodeAt(1));
    let price = bases[ticker] || 150;
    const vol = vols[ticker] || .015;
    const drift = drifts[ticker] || .0002;
    const series = [];
    const now = new Date();
    for (let i = days; i >= 0; i--) {
        const d = new Date(now); d.setDate(d.getDate() - i);
        if (d.getDay() === 0 || d.getDay() === 6) continue;
        const r = (rand() - .48) * vol * 2;
        price = price * (1 + drift + r);
        const open = price * (1 + (rand() - .5) * .005);
        const high = Math.max(open, price) * (1 + rand() * .006);
        const low = Math.min(open, price) * (1 - rand() * .006);
        const vol2 = Math.floor(rand() * 8e7 + 2e7);
        series.push({
            date: d.toISOString().slice(0, 10),
            open: +open.toFixed(2), high: +high.toFixed(2),
            low: +low.toFixed(2), close: +price.toFixed(2), volume: vol2,
        });
    }
    return series;
}

/* ─── Technical indicators ───────────────────────────────────── */
export function sma(data, n) {
    return data.map((_, i) => {
        if (i < n - 1) return null;
        return +(data.slice(i - n + 1, i + 1).reduce((s, d) => s + d.close, 0) / n).toFixed(2);
    });
}
export function ema(data, n) {
    const k = 2 / (n + 1);
    const out = [];
    let e = data[0].close;
    data.forEach((d, i) => {
        if (i === 0) { out.push(+e.toFixed(2)); return; }
        e = d.close * k + e * (1 - k);
        out.push(+e.toFixed(2));
    });
    return out;
}
export function rsi(data, n = 14) {
    const out = Array(n).fill(null);
    let avgG = 0, avgL = 0;
    for (let i = 1; i <= n; i++) {
        const diff = data[i].close - data[i - 1].close;
        if (diff > 0) avgG += diff; else avgL -= diff;
    }
    avgG /= n; avgL /= n;
    out.push(+(100 - 100 / (1 + avgG / avgL)).toFixed(2));
    for (let i = n + 1; i < data.length; i++) {
        const diff = data[i].close - data[i - 1].close;
        const g = diff > 0 ? diff : 0;
        const l = diff < 0 ? -diff : 0;
        avgG = (avgG * (n - 1) + g) / n;
        avgL = (avgL * (n - 1) + l) / n;
        out.push(+(100 - 100 / (1 + avgG / (avgL || .0001))).toFixed(2));
    }
    return out;
}
export function macd(data) {
    const e12 = ema(data, 12);
    const e26 = ema(data, 26);
    const macdLine = e12.map((v, i) => v && e26[i] ? +(v - e26[i]).toFixed(2) : null);
    const signalData = macdLine.map((v) => ({ close: v || 0 }));
    const signal = ema(signalData, 9);
    const hist = macdLine.map((v, i) => v !== null ? +(v - signal[i]).toFixed(2) : null);
    return { macdLine, signal, hist };
}
export function bollingerBands(data, n = 20, k = 2) {
    const smaArr = sma(data, n);
    return data.map((_, i) => {
        if (smaArr[i] === null) return { mid: null, upper: null, lower: null };
        const slice = data.slice(i - n + 1, i + 1).map(d => d.close);
        const mean = smaArr[i];
        const std = Math.sqrt(slice.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
        return { mid: +mean.toFixed(2), upper: +(mean + k * std).toFixed(2), lower: +(mean - k * std).toFixed(2) };
    });
}

/* ─── Generate predictions ───────────────────────────────────── */
export function genPredictions(lastPrice, days = 30, ticker = "AAPL") {
    const rand = seededRand(ticker.charCodeAt(0) * 97 + days);
    const drifts = { AAPL: .0004, MSFT: .0005, NVDA: .001, TSLA: -.0002, GOOGL: .0003 };
    const drift = drifts[ticker] || .0003;
    const vol = .012;
    let p = lastPrice;
    const preds = [];
    const now = new Date();
    for (let i = 1; i <= days; i++) {
        const d = new Date(now); d.setDate(d.getDate() + i);
        if (d.getDay() === 0) d.setDate(d.getDate() + 1);
        if (d.getDay() === 6) d.setDate(d.getDate() + 2);
        p = p * (1 + drift + (rand() - .47) * vol);
        const conf = vol * Math.sqrt(i) * 1.64;
        preds.push({
            date: d.toISOString().slice(0, 10),
            predicted: +p.toFixed(2),
            upper95: +(p * (1 + conf)).toFixed(2), lower95: +(p * (1 - conf)).toFixed(2),
            upper68: +(p * (1 + conf * .6)).toFixed(2), lower68: +(p * (1 - conf * .6)).toFixed(2),
        });
    }
    return preds;
}

/* ─── Portfolio optimization (mock mean-variance) ───────────── */
export function optimizePortfolio(tickers, method = "sharpe") {
    const rand = seededRand(tickers.join("").charCodeAt(0) * 7 + method.length);
    const weights = tickers.map(() => rand());
    const sum = weights.reduce((a, b) => a + b, 0);
    return weights.map(w => +(w / sum * 100).toFixed(1));
}

/* ─── Constants ─────────────────────────────────────────────── */
export const FUNDAMENTALS = {
    AAPL: { pe: 28.4, eps: 6.42, mktCap: "2.87T", beta: .95, div: .96, sector: "Technology" },
    MSFT: { pe: 35.1, eps: 10.76, mktCap: "2.81T", beta: .9, div: 2.72, sector: "Technology" },
    GOOGL: { pe: 24.8, eps: 6.22, mktCap: "1.91T", beta: 1.05, div: 0, sector: "Technology" },
    AMZN: { pe: 44.2, eps: 4.19, mktCap: "1.94T", beta: 1.15, div: 0, sector: "Consumer" },
    NVDA: { pe: 68.3, eps: 12.8, mktCap: "2.14T", beta: 1.65, div: .16, sector: "Technology" },
    TSLA: { pe: 52.1, eps: 4.7, mktCap: "780B", beta: 2.1, div: 0, sector: "Automotive" },
    META: { pe: 26.9, eps: 18.8, mktCap: "1.28T", beta: 1.22, div: 2.0, sector: "Technology" },
    NFLX: { pe: 43.6, eps: 14.2, mktCap: "268B", beta: 1.35, div: 0, sector: "Media" },
};

export const TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX"];

export const C = {
    bg0: "#080c14", bg1: "#0d1524", bg2: "#111d2e", bg3: "#17263d",
    border: "rgba(42,58,92,.7)", amber: "#fbbf24", amberDim: "rgba(251,191,36,.15)",
    amberLow: "rgba(251,191,36,.07)", cyan: "#22d3ee", green: "#10b981",
    red: "#f43f5e", purple: "#a78bfa", text: "#e2e8f0", textDim: "#64748b", textMid: "#94a3b8",
};
