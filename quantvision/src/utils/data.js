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
    const bases = { "^GSPC": 5200, AAPL: 182, MSFT: 378, GOOGL: 155, AMZN: 185, NVDA: 875, TSLA: 245, META: 505, NFLX: 620 };
    const vols = { "^GSPC": .008, AAPL: .012, MSFT: .011, GOOGL: .013, AMZN: .015, NVDA: .022, TSLA: .028, META: .016, NFLX: .017 };
    const drifts = { "^GSPC": .00025, AAPL: .0003, MSFT: .0004, GOOGL: .0002, AMZN: .0003, NVDA: .0008, TSLA: -.0001, META: .0005, NFLX: .0004 };
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
    "^GSPC": { pe: 0, eps: 0, mktCap: "Index", beta: 1.0, div: 0, sector: "Index" },
    AAPL: { pe: 28.4, eps: 6.42, mktCap: "2.87T", beta: .95, div: .96, sector: "Technology" },
    MSFT: { pe: 35.1, eps: 10.76, mktCap: "2.81T", beta: .9, div: 2.72, sector: "Technology" },
    GOOGL: { pe: 24.8, eps: 6.22, mktCap: "1.91T", beta: 1.05, div: 0, sector: "Technology" },
    AMZN: { pe: 44.2, eps: 4.19, mktCap: "1.94T", beta: 1.15, div: 0, sector: "Consumer" },
    NVDA: { pe: 68.3, eps: 12.8, mktCap: "2.14T", beta: 1.65, div: .16, sector: "Technology" },
    TSLA: { pe: 52.1, eps: 4.7, mktCap: "780B", beta: 2.1, div: 0, sector: "Automotive" },
    META: { pe: 26.9, eps: 18.8, mktCap: "1.28T", beta: 1.22, div: 2.0, sector: "Technology" },
    NFLX: { pe: 43.6, eps: 14.2, mktCap: "268B", beta: 1.35, div: 0, sector: "Media" },
};

export const DEFAULT_INDEX_SYMBOL = "^GSPC";
export const LEGACY_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX"];
export const TICKERS = [DEFAULT_INDEX_SYMBOL, "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"];

/**
 * The S&P 500 constituents, as a bundled fallback.
 *
 * The live list comes from `GET /api/data/sp500`, which scrapes the index and
 * therefore tracks additions and removals; this snapshot is what the stock
 * picker shows when the API is unreachable. It is the FULL index, not a
 * selection from it: the previous 130-name hand-maintained list silently
 * capped the picker at a quarter of the universe, so 388 constituents -- every
 * utility, most of the industrials, and every recent addition -- had no way
 * into the app at all, while 14 names in it (ASML, TSM, BABA, SHOP, NVO, ARM,
 * SPOT, MELI, SE, SNOW and others) are not S&P 500 members and were listed as
 * though they were.
 *
 * Sectors are the GICS names the index publishes, which is what
 * `/api/data/sp500` returns -- so the live list and this fallback key into the
 * same palette rather than needing a translation on one of the two paths.
 *
 * Tickers are in the dash spelling the data provider indexes (BRK-B, not
 * BRK.B). The API accepts either -- `ticker_variants` in the OHLCV layer falls
 * back from one to the other -- but the dash form resolves on the first
 * request.
 *
 * Regenerate from the API rather than editing by hand:
 *     curl localhost:8000/api/data/sp500
 */
export const SP500_LIST = [
    { ticker: "A", name: "Agilent Technologies", sector: "Health Care" },
    { ticker: "AAPL", name: "Apple Inc.", sector: "Information Technology" },
    { ticker: "ABBV", name: "AbbVie", sector: "Health Care" },
    { ticker: "ABNB", name: "Airbnb", sector: "Consumer Discretionary" },
    { ticker: "ABT", name: "Abbott Laboratories", sector: "Health Care" },
    { ticker: "ACGL", name: "Arch Capital Group", sector: "Financials" },
    { ticker: "ACN", name: "Accenture", sector: "Information Technology" },
    { ticker: "ADBE", name: "Adobe Inc.", sector: "Information Technology" },
    { ticker: "ADI", name: "Analog Devices", sector: "Information Technology" },
    { ticker: "ADM", name: "Archer Daniels Midland", sector: "Consumer Staples" },
    { ticker: "ADP", name: "Automatic Data Processing", sector: "Industrials" },
    { ticker: "ADSK", name: "Autodesk", sector: "Information Technology" },
    { ticker: "AEE", name: "Ameren", sector: "Utilities" },
    { ticker: "AEP", name: "American Electric Power", sector: "Utilities" },
    { ticker: "AES", name: "AES Corporation", sector: "Utilities" },
    { ticker: "AFL", name: "Aflac", sector: "Financials" },
    { ticker: "AIG", name: "American International Group", sector: "Financials" },
    { ticker: "AIZ", name: "Assurant", sector: "Financials" },
    { ticker: "AJG", name: "Arthur J. Gallagher & Co.", sector: "Financials" },
    { ticker: "AKAM", name: "Akamai Technologies", sector: "Information Technology" },
    { ticker: "ALB", name: "Albemarle Corporation", sector: "Materials" },
    { ticker: "ALGN", name: "Align Technology", sector: "Health Care" },
    { ticker: "ALL", name: "Allstate", sector: "Financials" },
    { ticker: "ALLE", name: "Allegion", sector: "Industrials" },
    { ticker: "AMAT", name: "Applied Materials", sector: "Information Technology" },
    { ticker: "AMCR", name: "Amcor", sector: "Materials" },
    { ticker: "AMD", name: "Advanced Micro Devices", sector: "Information Technology" },
    { ticker: "AME", name: "Ametek", sector: "Industrials" },
    { ticker: "AMGN", name: "Amgen", sector: "Health Care" },
    { ticker: "AMP", name: "Ameriprise Financial", sector: "Financials" },
    { ticker: "AMT", name: "American Tower", sector: "Real Estate" },
    { ticker: "AMZN", name: "Amazon", sector: "Consumer Discretionary" },
    { ticker: "ANET", name: "Arista Networks", sector: "Information Technology" },
    { ticker: "AON", name: "Aon plc", sector: "Financials" },
    { ticker: "AOS", name: "A. O. Smith", sector: "Industrials" },
    { ticker: "APA", name: "APA Corporation", sector: "Energy" },
    { ticker: "APD", name: "Air Products", sector: "Materials" },
    { ticker: "APH", name: "Amphenol", sector: "Information Technology" },
    { ticker: "APO", name: "Apollo Global Management", sector: "Financials" },
    { ticker: "APP", name: "AppLovin", sector: "Communication Services" },
    { ticker: "APTV", name: "Aptiv", sector: "Consumer Discretionary" },
    { ticker: "ARE", name: "Alexandria Real Estate Equities", sector: "Real Estate" },
    { ticker: "ARES", name: "Ares Management", sector: "Financials" },
    { ticker: "ATO", name: "Atmos Energy", sector: "Utilities" },
    { ticker: "AVGO", name: "Broadcom", sector: "Information Technology" },
    { ticker: "AVY", name: "Avery Dennison", sector: "Materials" },
    { ticker: "AWK", name: "American Water Works", sector: "Utilities" },
    { ticker: "AXON", name: "Axon Enterprise", sector: "Industrials" },
    { ticker: "AXP", name: "American Express", sector: "Financials" },
    { ticker: "AZO", name: "AutoZone", sector: "Consumer Discretionary" },
    { ticker: "BA", name: "Boeing", sector: "Industrials" },
    { ticker: "BAC", name: "Bank of America", sector: "Financials" },
    { ticker: "BALL", name: "Ball Corporation", sector: "Materials" },
    { ticker: "BAX", name: "Baxter International", sector: "Health Care" },
    { ticker: "BBY", name: "Best Buy", sector: "Consumer Discretionary" },
    { ticker: "BDX", name: "Becton Dickinson", sector: "Health Care" },
    { ticker: "BEN", name: "Franklin Resources", sector: "Financials" },
    { ticker: "BF-B", name: "Brown–Forman", sector: "Consumer Staples" },
    { ticker: "BG", name: "Bunge Global", sector: "Consumer Staples" },
    { ticker: "BIIB", name: "Biogen", sector: "Health Care" },
    { ticker: "BKNG", name: "Booking Holdings", sector: "Consumer Discretionary" },
    { ticker: "BKR", name: "Baker Hughes", sector: "Energy" },
    { ticker: "BLDR", name: "Builders FirstSource", sector: "Industrials" },
    { ticker: "BLK", name: "BlackRock", sector: "Financials" },
    { ticker: "BMY", name: "Bristol Myers Squibb", sector: "Health Care" },
    { ticker: "BNY", name: "BNY Mellon", sector: "Financials" },
    { ticker: "BR", name: "Broadridge Financial Solutions", sector: "Industrials" },
    { ticker: "BRK-B", name: "Berkshire Hathaway", sector: "Financials" },
    { ticker: "BRO", name: "Brown & Brown", sector: "Financials" },
    { ticker: "BSX", name: "Boston Scientific", sector: "Health Care" },
    { ticker: "BX", name: "Blackstone Inc.", sector: "Financials" },
    { ticker: "BXP", name: "BXP, Inc.", sector: "Real Estate" },
    { ticker: "C", name: "Citigroup", sector: "Financials" },
    { ticker: "CAH", name: "Cardinal Health", sector: "Health Care" },
    { ticker: "CARR", name: "Carrier Global", sector: "Industrials" },
    { ticker: "CASY", name: "Casey's", sector: "Consumer Staples" },
    { ticker: "CAT", name: "Caterpillar Inc.", sector: "Industrials" },
    { ticker: "CB", name: "Chubb Limited", sector: "Financials" },
    { ticker: "CBOE", name: "Cboe Global Markets", sector: "Financials" },
    { ticker: "CBRE", name: "CBRE Group", sector: "Real Estate" },
    { ticker: "CCI", name: "Crown Castle", sector: "Real Estate" },
    { ticker: "CCL", name: "Carnival Corporation", sector: "Consumer Discretionary" },
    { ticker: "CDNS", name: "Cadence Design Systems", sector: "Information Technology" },
    { ticker: "CDW", name: "CDW Corporation", sector: "Information Technology" },
    { ticker: "CEG", name: "Constellation Energy", sector: "Utilities" },
    { ticker: "CF", name: "CF Industries", sector: "Materials" },
    { ticker: "CFG", name: "Citizens Financial Group", sector: "Financials" },
    { ticker: "CHD", name: "Church & Dwight", sector: "Consumer Staples" },
    { ticker: "CHRW", name: "C.H. Robinson", sector: "Industrials" },
    { ticker: "CHTR", name: "Charter Communications", sector: "Communication Services" },
    { ticker: "CI", name: "Cigna", sector: "Health Care" },
    { ticker: "CIEN", name: "Ciena", sector: "Information Technology" },
    { ticker: "CINF", name: "Cincinnati Financial", sector: "Financials" },
    { ticker: "CL", name: "Colgate-Palmolive", sector: "Consumer Staples" },
    { ticker: "CLX", name: "Clorox", sector: "Consumer Staples" },
    { ticker: "CMCSA", name: "Comcast", sector: "Communication Services" },
    { ticker: "CME", name: "CME Group", sector: "Financials" },
    { ticker: "CMG", name: "Chipotle Mexican Grill", sector: "Consumer Discretionary" },
    { ticker: "CMI", name: "Cummins", sector: "Industrials" },
    { ticker: "CMS", name: "CMS Energy", sector: "Utilities" },
    { ticker: "CNC", name: "Centene Corporation", sector: "Health Care" },
    { ticker: "CNP", name: "CenterPoint Energy", sector: "Utilities" },
    { ticker: "COF", name: "Capital One", sector: "Financials" },
    { ticker: "COHR", name: "Coherent Corp.", sector: "Information Technology" },
    { ticker: "COIN", name: "Coinbase", sector: "Financials" },
    { ticker: "COO", name: "Cooper Companies (The)", sector: "Health Care" },
    { ticker: "COP", name: "ConocoPhillips", sector: "Energy" },
    { ticker: "COR", name: "Cencora", sector: "Health Care" },
    { ticker: "COST", name: "Costco", sector: "Consumer Staples" },
    { ticker: "CPAY", name: "Corpay", sector: "Financials" },
    { ticker: "CPRT", name: "Copart", sector: "Industrials" },
    { ticker: "CPT", name: "Camden Property Trust", sector: "Real Estate" },
    { ticker: "CRH", name: "CRH plc", sector: "Materials" },
    { ticker: "CRL", name: "Charles River Laboratories", sector: "Health Care" },
    { ticker: "CRM", name: "Salesforce", sector: "Information Technology" },
    { ticker: "CRWD", name: "CrowdStrike", sector: "Information Technology" },
    { ticker: "CSCO", name: "Cisco", sector: "Information Technology" },
    { ticker: "CSGP", name: "CoStar Group", sector: "Real Estate" },
    { ticker: "CSX", name: "CSX Corporation", sector: "Industrials" },
    { ticker: "CTAS", name: "Cintas", sector: "Industrials" },
    { ticker: "CTSH", name: "Cognizant", sector: "Information Technology" },
    { ticker: "CTVA", name: "Corteva", sector: "Materials" },
    { ticker: "CVNA", name: "Carvana", sector: "Consumer Discretionary" },
    { ticker: "CVS", name: "CVS Health", sector: "Health Care" },
    { ticker: "CVX", name: "Chevron Corporation", sector: "Energy" },
    { ticker: "D", name: "Dominion Energy", sector: "Utilities" },
    { ticker: "DAL", name: "Delta Air Lines", sector: "Industrials" },
    { ticker: "DASH", name: "DoorDash", sector: "Consumer Discretionary" },
    { ticker: "DD", name: "DuPont", sector: "Industrials" },
    { ticker: "DDOG", name: "Datadog", sector: "Information Technology" },
    { ticker: "DE", name: "Deere & Company", sector: "Industrials" },
    { ticker: "DECK", name: "Deckers Brands", sector: "Consumer Discretionary" },
    { ticker: "DELL", name: "Dell Technologies", sector: "Information Technology" },
    { ticker: "DG", name: "Dollar General", sector: "Consumer Staples" },
    { ticker: "DGX", name: "Quest Diagnostics", sector: "Health Care" },
    { ticker: "DHI", name: "D. R. Horton", sector: "Consumer Discretionary" },
    { ticker: "DHR", name: "Danaher Corporation", sector: "Health Care" },
    { ticker: "DIS", name: "Walt Disney Company (The)", sector: "Communication Services" },
    { ticker: "DLR", name: "Digital Realty", sector: "Real Estate" },
    { ticker: "DLTR", name: "Dollar Tree", sector: "Consumer Staples" },
    { ticker: "DOC", name: "Healthpeak Properties", sector: "Real Estate" },
    { ticker: "DOV", name: "Dover Corporation", sector: "Industrials" },
    { ticker: "DOW", name: "Dow Inc.", sector: "Materials" },
    { ticker: "DPZ", name: "Domino's", sector: "Consumer Discretionary" },
    { ticker: "DRI", name: "Darden Restaurants", sector: "Consumer Discretionary" },
    { ticker: "DTE", name: "DTE Energy", sector: "Utilities" },
    { ticker: "DUK", name: "Duke Energy", sector: "Utilities" },
    { ticker: "DVA", name: "DaVita", sector: "Health Care" },
    { ticker: "DVN", name: "Devon Energy", sector: "Energy" },
    { ticker: "DXCM", name: "Dexcom", sector: "Health Care" },
    { ticker: "EBAY", name: "eBay Inc.", sector: "Consumer Discretionary" },
    { ticker: "ECHO", name: "EchoStar", sector: "Communication Services" },
    { ticker: "ECL", name: "Ecolab", sector: "Materials" },
    { ticker: "ED", name: "Consolidated Edison", sector: "Utilities" },
    { ticker: "EFX", name: "Equifax", sector: "Industrials" },
    { ticker: "EG", name: "Everest Group", sector: "Financials" },
    { ticker: "EIX", name: "Edison International", sector: "Utilities" },
    { ticker: "EL", name: "Estée Lauder Companies (The)", sector: "Consumer Staples" },
    { ticker: "ELV", name: "Elevance Health", sector: "Health Care" },
    { ticker: "EME", name: "Emcor", sector: "Industrials" },
    { ticker: "EMR", name: "Emerson Electric", sector: "Industrials" },
    { ticker: "EOG", name: "EOG Resources", sector: "Energy" },
    { ticker: "EQIX", name: "Equinix", sector: "Real Estate" },
    { ticker: "EQT", name: "EQT Corporation", sector: "Energy" },
    { ticker: "ERIE", name: "Erie Indemnity", sector: "Financials" },
    { ticker: "ES", name: "Eversource Energy", sector: "Utilities" },
    { ticker: "ESS", name: "Essex Property Trust", sector: "Real Estate" },
    { ticker: "ETN", name: "Eaton Corporation", sector: "Industrials" },
    { ticker: "ETR", name: "Entergy", sector: "Utilities" },
    { ticker: "EVRG", name: "Evergy", sector: "Utilities" },
    { ticker: "EW", name: "Edwards Lifesciences", sector: "Health Care" },
    { ticker: "EXC", name: "Exelon", sector: "Utilities" },
    { ticker: "EXE", name: "Expand Energy", sector: "Energy" },
    { ticker: "EXPD", name: "Expeditors International", sector: "Industrials" },
    { ticker: "EXPE", name: "Expedia Group", sector: "Consumer Discretionary" },
    { ticker: "EXR", name: "Extra Space Storage", sector: "Real Estate" },
    { ticker: "F", name: "Ford Motor Company", sector: "Consumer Discretionary" },
    { ticker: "FANG", name: "Diamondback Energy", sector: "Energy" },
    { ticker: "FAST", name: "Fastenal", sector: "Industrials" },
    { ticker: "FCX", name: "Freeport-McMoRan", sector: "Materials" },
    { ticker: "FDS", name: "FactSet", sector: "Financials" },
    { ticker: "FDX", name: "FedEx", sector: "Industrials" },
    { ticker: "FDXF", name: "FedEx Freight", sector: "Industrials" },
    { ticker: "FE", name: "FirstEnergy", sector: "Utilities" },
    { ticker: "FERG", name: "Ferguson Enterprises", sector: "Industrials" },
    { ticker: "FFIV", name: "F5, Inc.", sector: "Information Technology" },
    { ticker: "FICO", name: "Fair Isaac", sector: "Information Technology" },
    { ticker: "FIS", name: "Fidelity National Information Services", sector: "Financials" },
    { ticker: "FISV", name: "Fiserv", sector: "Financials" },
    { ticker: "FITB", name: "Fifth Third Bancorp", sector: "Financials" },
    { ticker: "FIX", name: "Comfort Systems USA", sector: "Industrials" },
    { ticker: "FLEX", name: "Flex Ltd.", sector: "Information Technology" },
    { ticker: "FOX", name: "Fox Corporation (Class B)", sector: "Communication Services" },
    { ticker: "FOXA", name: "Fox Corporation (Class A)", sector: "Communication Services" },
    { ticker: "FRT", name: "Federal Realty Investment Trust", sector: "Real Estate" },
    { ticker: "FSLR", name: "First Solar", sector: "Information Technology" },
    { ticker: "FTNT", name: "Fortinet", sector: "Information Technology" },
    { ticker: "FTV", name: "Fortive", sector: "Industrials" },
    { ticker: "GD", name: "General Dynamics", sector: "Industrials" },
    { ticker: "GDDY", name: "GoDaddy", sector: "Information Technology" },
    { ticker: "GE", name: "GE Aerospace", sector: "Industrials" },
    { ticker: "GEHC", name: "GE HealthCare", sector: "Health Care" },
    { ticker: "GEN", name: "Gen Digital", sector: "Information Technology" },
    { ticker: "GEV", name: "GE Vernova", sector: "Industrials" },
    { ticker: "GILD", name: "Gilead Sciences", sector: "Health Care" },
    { ticker: "GIS", name: "General Mills", sector: "Consumer Staples" },
    { ticker: "GL", name: "Globe Life", sector: "Financials" },
    { ticker: "GLW", name: "Corning Inc.", sector: "Information Technology" },
    { ticker: "GM", name: "General Motors", sector: "Consumer Discretionary" },
    { ticker: "GNRC", name: "Generac", sector: "Industrials" },
    { ticker: "GOOG", name: "Alphabet Inc. (Class C)", sector: "Communication Services" },
    { ticker: "GOOGL", name: "Alphabet Inc. (Class A)", sector: "Communication Services" },
    { ticker: "GPC", name: "Genuine Parts Company", sector: "Consumer Discretionary" },
    { ticker: "GPN", name: "Global Payments", sector: "Financials" },
    { ticker: "GRMN", name: "Garmin", sector: "Consumer Discretionary" },
    { ticker: "GS", name: "Goldman Sachs", sector: "Financials" },
    { ticker: "GWW", name: "W. W. Grainger", sector: "Industrials" },
    { ticker: "HAL", name: "Halliburton", sector: "Energy" },
    { ticker: "HAS", name: "Hasbro", sector: "Consumer Discretionary" },
    { ticker: "HBAN", name: "Huntington Bancshares", sector: "Financials" },
    { ticker: "HCA", name: "HCA Healthcare", sector: "Health Care" },
    { ticker: "HD", name: "Home Depot (The)", sector: "Consumer Discretionary" },
    { ticker: "HIG", name: "Hartford (The)", sector: "Financials" },
    { ticker: "HII", name: "Huntington Ingalls Industries", sector: "Industrials" },
    { ticker: "HLT", name: "Hilton Worldwide", sector: "Consumer Discretionary" },
    { ticker: "HON", name: "Honeywell Technologies", sector: "Industrials" },
    { ticker: "HONA", name: "Honeywell Aerospace", sector: "Industrials" },
    { ticker: "HOOD", name: "Robinhood Markets", sector: "Financials" },
    { ticker: "HPE", name: "Hewlett Packard Enterprise", sector: "Information Technology" },
    { ticker: "HPQ", name: "HP Inc.", sector: "Information Technology" },
    { ticker: "HRL", name: "Hormel Foods", sector: "Consumer Staples" },
    { ticker: "HSIC", name: "Henry Schein", sector: "Health Care" },
    { ticker: "HST", name: "Host Hotels & Resorts", sector: "Real Estate" },
    { ticker: "HSY", name: "Hershey Company (The)", sector: "Consumer Staples" },
    { ticker: "HUBB", name: "Hubbell Incorporated", sector: "Industrials" },
    { ticker: "HUM", name: "Humana", sector: "Health Care" },
    { ticker: "HWM", name: "Howmet Aerospace", sector: "Industrials" },
    { ticker: "IBKR", name: "Interactive Brokers", sector: "Financials" },
    { ticker: "IBM", name: "IBM", sector: "Information Technology" },
    { ticker: "ICE", name: "Intercontinental Exchange", sector: "Financials" },
    { ticker: "IDXX", name: "Idexx Laboratories", sector: "Health Care" },
    { ticker: "IEX", name: "IDEX Corporation", sector: "Industrials" },
    { ticker: "IFF", name: "International Flavors & Fragrances", sector: "Materials" },
    { ticker: "INCY", name: "Incyte", sector: "Health Care" },
    { ticker: "INTC", name: "Intel", sector: "Information Technology" },
    { ticker: "INTU", name: "Intuit", sector: "Information Technology" },
    { ticker: "INVH", name: "Invitation Homes", sector: "Real Estate" },
    { ticker: "IP", name: "International Paper", sector: "Materials" },
    { ticker: "IQV", name: "IQVIA", sector: "Health Care" },
    { ticker: "IR", name: "Ingersoll Rand", sector: "Industrials" },
    { ticker: "IRM", name: "Iron Mountain", sector: "Real Estate" },
    { ticker: "ISRG", name: "Intuitive Surgical", sector: "Health Care" },
    { ticker: "IT", name: "Gartner", sector: "Information Technology" },
    { ticker: "ITW", name: "Illinois Tool Works", sector: "Industrials" },
    { ticker: "IVZ", name: "Invesco", sector: "Financials" },
    { ticker: "J", name: "Jacobs Solutions", sector: "Industrials" },
    { ticker: "JBHT", name: "J.B. Hunt", sector: "Industrials" },
    { ticker: "JBL", name: "Jabil", sector: "Information Technology" },
    { ticker: "JCI", name: "Johnson Controls", sector: "Industrials" },
    { ticker: "JKHY", name: "Jack Henry & Associates", sector: "Financials" },
    { ticker: "JNJ", name: "Johnson & Johnson", sector: "Health Care" },
    { ticker: "JPM", name: "JPMorgan Chase", sector: "Financials" },
    { ticker: "KDP", name: "Keurig Dr Pepper", sector: "Consumer Staples" },
    { ticker: "KEY", name: "KeyCorp", sector: "Financials" },
    { ticker: "KEYS", name: "Keysight Technologies", sector: "Information Technology" },
    { ticker: "KHC", name: "Kraft Heinz", sector: "Consumer Staples" },
    { ticker: "KIM", name: "Kimco Realty", sector: "Real Estate" },
    { ticker: "KKR", name: "KKR & Co.", sector: "Financials" },
    { ticker: "KLAC", name: "KLA Corporation", sector: "Information Technology" },
    { ticker: "KMB", name: "Kimberly-Clark", sector: "Consumer Staples" },
    { ticker: "KMI", name: "Kinder Morgan", sector: "Energy" },
    { ticker: "KO", name: "Coca-Cola Company (The)", sector: "Consumer Staples" },
    { ticker: "KR", name: "Kroger", sector: "Consumer Staples" },
    { ticker: "KVUE", name: "Kenvue", sector: "Consumer Staples" },
    { ticker: "L", name: "Loews Corporation", sector: "Financials" },
    { ticker: "LDOS", name: "Leidos", sector: "Industrials" },
    { ticker: "LEN", name: "Lennar", sector: "Consumer Discretionary" },
    { ticker: "LH", name: "Labcorp", sector: "Health Care" },
    { ticker: "LHX", name: "L3Harris", sector: "Industrials" },
    { ticker: "LII", name: "Lennox International", sector: "Industrials" },
    { ticker: "LIN", name: "Linde plc", sector: "Materials" },
    { ticker: "LITE", name: "Lumentum", sector: "Information Technology" },
    { ticker: "LLY", name: "Lilly (Eli)", sector: "Health Care" },
    { ticker: "LMT", name: "Lockheed Martin", sector: "Industrials" },
    { ticker: "LNT", name: "Alliant Energy", sector: "Utilities" },
    { ticker: "LOW", name: "Lowe's", sector: "Consumer Discretionary" },
    { ticker: "LRCX", name: "Lam Research", sector: "Information Technology" },
    { ticker: "LULU", name: "Lululemon Athletica", sector: "Consumer Discretionary" },
    { ticker: "LUV", name: "Southwest Airlines", sector: "Industrials" },
    { ticker: "LVS", name: "Las Vegas Sands", sector: "Consumer Discretionary" },
    { ticker: "LYB", name: "LyondellBasell", sector: "Materials" },
    { ticker: "LYV", name: "Live Nation Entertainment", sector: "Communication Services" },
    { ticker: "MA", name: "Mastercard", sector: "Financials" },
    { ticker: "MAA", name: "Mid-America Apartment Communities", sector: "Real Estate" },
    { ticker: "MAR", name: "Marriott International", sector: "Consumer Discretionary" },
    { ticker: "MAS", name: "Masco", sector: "Industrials" },
    { ticker: "MCD", name: "McDonald's", sector: "Consumer Discretionary" },
    { ticker: "MCHP", name: "Microchip Technology", sector: "Information Technology" },
    { ticker: "MCK", name: "McKesson Corporation", sector: "Health Care" },
    { ticker: "MCO", name: "Moody's Corporation", sector: "Financials" },
    { ticker: "MDLZ", name: "Mondelez International", sector: "Consumer Staples" },
    { ticker: "MDT", name: "Medtronic", sector: "Health Care" },
    { ticker: "MET", name: "MetLife", sector: "Financials" },
    { ticker: "META", name: "Meta Platforms", sector: "Communication Services" },
    { ticker: "MGM", name: "MGM Resorts", sector: "Consumer Discretionary" },
    { ticker: "MKC", name: "McCormick & Company", sector: "Consumer Staples" },
    { ticker: "MLM", name: "Martin Marietta Materials", sector: "Materials" },
    { ticker: "MMM", name: "3M", sector: "Industrials" },
    { ticker: "MNST", name: "Monster Beverage", sector: "Consumer Staples" },
    { ticker: "MO", name: "Altria", sector: "Consumer Staples" },
    { ticker: "MOS", name: "Mosaic Company (The)", sector: "Materials" },
    { ticker: "MPC", name: "Marathon Petroleum", sector: "Energy" },
    { ticker: "MPWR", name: "Monolithic Power Systems", sector: "Information Technology" },
    { ticker: "MRK", name: "Merck & Co.", sector: "Health Care" },
    { ticker: "MRNA", name: "Moderna", sector: "Health Care" },
    { ticker: "MRSH", name: "Marsh McLennan", sector: "Financials" },
    { ticker: "MRVL", name: "Marvell Technology", sector: "Information Technology" },
    { ticker: "MS", name: "Morgan Stanley", sector: "Financials" },
    { ticker: "MSCI", name: "MSCI", sector: "Financials" },
    { ticker: "MSFT", name: "Microsoft", sector: "Information Technology" },
    { ticker: "MSI", name: "Motorola Solutions", sector: "Information Technology" },
    { ticker: "MTB", name: "M&T Bank", sector: "Financials" },
    { ticker: "MTD", name: "Mettler Toledo", sector: "Health Care" },
    { ticker: "MU", name: "Micron Technology", sector: "Information Technology" },
    { ticker: "NCLH", name: "Norwegian Cruise Line Holdings", sector: "Consumer Discretionary" },
    { ticker: "NDAQ", name: "Nasdaq, Inc.", sector: "Financials" },
    { ticker: "NDSN", name: "Nordson Corporation", sector: "Industrials" },
    { ticker: "NEE", name: "NextEra Energy", sector: "Utilities" },
    { ticker: "NEM", name: "Newmont", sector: "Materials" },
    { ticker: "NFLX", name: "Netflix", sector: "Communication Services" },
    { ticker: "NI", name: "NiSource", sector: "Utilities" },
    { ticker: "NKE", name: "Nike, Inc.", sector: "Consumer Discretionary" },
    { ticker: "NOC", name: "Northrop Grumman", sector: "Industrials" },
    { ticker: "NOW", name: "ServiceNow", sector: "Information Technology" },
    { ticker: "NRG", name: "NRG Energy", sector: "Utilities" },
    { ticker: "NSC", name: "Norfolk Southern", sector: "Industrials" },
    { ticker: "NTAP", name: "NetApp", sector: "Information Technology" },
    { ticker: "NTRS", name: "Northern Trust", sector: "Financials" },
    { ticker: "NUE", name: "Nucor", sector: "Materials" },
    { ticker: "NVDA", name: "Nvidia", sector: "Information Technology" },
    { ticker: "NVR", name: "NVR, Inc.", sector: "Consumer Discretionary" },
    { ticker: "NWS", name: "News Corp (Class B)", sector: "Communication Services" },
    { ticker: "NWSA", name: "News Corp (Class A)", sector: "Communication Services" },
    { ticker: "NXPI", name: "NXP Semiconductors", sector: "Information Technology" },
    { ticker: "O", name: "Realty Income", sector: "Real Estate" },
    { ticker: "ODFL", name: "Old Dominion", sector: "Industrials" },
    { ticker: "OKE", name: "Oneok", sector: "Energy" },
    { ticker: "OMC", name: "Omnicom Group", sector: "Communication Services" },
    { ticker: "ON", name: "ON Semiconductor", sector: "Information Technology" },
    { ticker: "ORCL", name: "Oracle Corporation", sector: "Information Technology" },
    { ticker: "ORLY", name: "O'Reilly Automotive", sector: "Consumer Discretionary" },
    { ticker: "OTIS", name: "Otis Worldwide", sector: "Industrials" },
    { ticker: "OXY", name: "Occidental Petroleum", sector: "Energy" },
    { ticker: "PANW", name: "Palo Alto Networks", sector: "Information Technology" },
    { ticker: "PAYX", name: "Paychex", sector: "Industrials" },
    { ticker: "PCAR", name: "Paccar", sector: "Industrials" },
    { ticker: "PCG", name: "PG&E Corporation", sector: "Utilities" },
    { ticker: "PEG", name: "Public Service Enterprise Group", sector: "Utilities" },
    { ticker: "PEP", name: "PepsiCo", sector: "Consumer Staples" },
    { ticker: "PFE", name: "Pfizer", sector: "Health Care" },
    { ticker: "PFG", name: "Principal Financial Group", sector: "Financials" },
    { ticker: "PG", name: "Procter & Gamble", sector: "Consumer Staples" },
    { ticker: "PGR", name: "Progressive Corporation", sector: "Financials" },
    { ticker: "PH", name: "Parker Hannifin", sector: "Industrials" },
    { ticker: "PHM", name: "PulteGroup", sector: "Consumer Discretionary" },
    { ticker: "PKG", name: "Packaging Corporation of America", sector: "Materials" },
    { ticker: "PLD", name: "Prologis", sector: "Real Estate" },
    { ticker: "PLTR", name: "Palantir Technologies", sector: "Information Technology" },
    { ticker: "PM", name: "Philip Morris International", sector: "Consumer Staples" },
    { ticker: "PNC", name: "PNC Financial Services", sector: "Financials" },
    { ticker: "PNR", name: "Pentair", sector: "Industrials" },
    { ticker: "PNW", name: "Pinnacle West Capital", sector: "Utilities" },
    { ticker: "PODD", name: "Insulet Corporation", sector: "Health Care" },
    { ticker: "PPG", name: "PPG Industries", sector: "Materials" },
    { ticker: "PPL", name: "PPL Corporation", sector: "Utilities" },
    { ticker: "PRU", name: "Prudential Financial", sector: "Financials" },
    { ticker: "PSA", name: "Public Storage", sector: "Real Estate" },
    { ticker: "PSKY", name: "Paramount Skydance Corporation", sector: "Communication Services" },
    { ticker: "PSX", name: "Phillips 66", sector: "Energy" },
    { ticker: "PTC", name: "PTC Inc.", sector: "Information Technology" },
    { ticker: "PWR", name: "Quanta Services", sector: "Industrials" },
    { ticker: "PYPL", name: "PayPal", sector: "Financials" },
    { ticker: "Q", name: "Qnity Electronics", sector: "Information Technology" },
    { ticker: "QCOM", name: "Qualcomm", sector: "Information Technology" },
    { ticker: "RCL", name: "Royal Caribbean Group", sector: "Consumer Discretionary" },
    { ticker: "RDDT", name: "Reddit", sector: "Communication Services" },
    { ticker: "REG", name: "Regency Centers", sector: "Real Estate" },
    { ticker: "REGN", name: "Regeneron Pharmaceuticals", sector: "Health Care" },
    { ticker: "RF", name: "Regions Financial Corporation", sector: "Financials" },
    { ticker: "RJF", name: "Raymond James Financial", sector: "Financials" },
    { ticker: "RL", name: "Ralph Lauren Corporation", sector: "Consumer Discretionary" },
    { ticker: "RMD", name: "ResMed|", sector: "Health Care" },
    { ticker: "ROK", name: "Rockwell Automation", sector: "Industrials" },
    { ticker: "ROL", name: "Rollins, Inc.", sector: "Industrials" },
    { ticker: "ROP", name: "Roper Technologies", sector: "Information Technology" },
    { ticker: "ROST", name: "Ross Stores", sector: "Consumer Discretionary" },
    { ticker: "RSG", name: "Republic Services", sector: "Industrials" },
    { ticker: "RTX", name: "RTX Corporation", sector: "Industrials" },
    { ticker: "RVTY", name: "Revvity", sector: "Health Care" },
    { ticker: "SBAC", name: "SBA Communications", sector: "Real Estate" },
    { ticker: "SBUX", name: "Starbucks", sector: "Consumer Discretionary" },
    { ticker: "SCHW", name: "Charles Schwab Corporation", sector: "Financials" },
    { ticker: "SHW", name: "Sherwin-Williams", sector: "Materials" },
    { ticker: "SJM", name: "J.M. Smucker Company (The)", sector: "Consumer Staples" },
    { ticker: "SLB", name: "Schlumberger", sector: "Energy" },
    { ticker: "SMCI", name: "Supermicro", sector: "Information Technology" },
    { ticker: "SNA", name: "Snap-on", sector: "Industrials" },
    { ticker: "SNDK", name: "Sandisk", sector: "Information Technology" },
    { ticker: "SNPS", name: "Synopsys", sector: "Information Technology" },
    { ticker: "SO", name: "Southern Company", sector: "Utilities" },
    { ticker: "SOLV", name: "Solventum", sector: "Health Care" },
    { ticker: "SPG", name: "Simon Property Group", sector: "Real Estate" },
    { ticker: "SPGI", name: "S&P Global", sector: "Financials" },
    { ticker: "SRE", name: "Sempra", sector: "Utilities" },
    { ticker: "STE", name: "Steris", sector: "Health Care" },
    { ticker: "STLD", name: "Steel Dynamics", sector: "Materials" },
    { ticker: "STT", name: "State Street Corporation", sector: "Financials" },
    { ticker: "STX", name: "Seagate Technology", sector: "Information Technology" },
    { ticker: "STZ", name: "Constellation Brands", sector: "Consumer Staples" },
    { ticker: "SW", name: "Smurfit Westrock", sector: "Materials" },
    { ticker: "SWK", name: "Stanley Black & Decker", sector: "Industrials" },
    { ticker: "SWKS", name: "Skyworks Solutions", sector: "Information Technology" },
    { ticker: "SYF", name: "Synchrony Financial", sector: "Financials" },
    { ticker: "SYK", name: "Stryker Corporation", sector: "Health Care" },
    { ticker: "SYY", name: "Sysco", sector: "Consumer Staples" },
    { ticker: "T", name: "AT&T", sector: "Communication Services" },
    { ticker: "TAP", name: "Molson Coors Beverage Company", sector: "Consumer Staples" },
    { ticker: "TDG", name: "TransDigm Group", sector: "Industrials" },
    { ticker: "TDY", name: "Teledyne Technologies", sector: "Information Technology" },
    { ticker: "TECH", name: "Bio-Techne", sector: "Health Care" },
    { ticker: "TEL", name: "TE Connectivity", sector: "Information Technology" },
    { ticker: "TER", name: "Teradyne", sector: "Information Technology" },
    { ticker: "TFC", name: "Truist Financial", sector: "Financials" },
    { ticker: "TGT", name: "Target Corporation", sector: "Consumer Staples" },
    { ticker: "TJX", name: "TJX Companies", sector: "Consumer Discretionary" },
    { ticker: "TKO", name: "TKO Group Holdings", sector: "Communication Services" },
    { ticker: "TMO", name: "Thermo Fisher Scientific", sector: "Health Care" },
    { ticker: "TMUS", name: "T-Mobile US", sector: "Communication Services" },
    { ticker: "TPL", name: "Texas Pacific Land Corporation", sector: "Energy" },
    { ticker: "TPR", name: "Tapestry, Inc.", sector: "Consumer Discretionary" },
    { ticker: "TRGP", name: "Targa Resources", sector: "Energy" },
    { ticker: "TRMB", name: "Trimble Inc.", sector: "Information Technology" },
    { ticker: "TROW", name: "T. Rowe Price", sector: "Financials" },
    { ticker: "TRV", name: "Travelers Companies (The)", sector: "Financials" },
    { ticker: "TSCO", name: "Tractor Supply", sector: "Consumer Discretionary" },
    { ticker: "TSLA", name: "Tesla, Inc.", sector: "Consumer Discretionary" },
    { ticker: "TSN", name: "Tyson Foods", sector: "Consumer Staples" },
    { ticker: "TT", name: "Trane Technologies", sector: "Industrials" },
    { ticker: "TTD", name: "Trade Desk (The)", sector: "Communication Services" },
    { ticker: "TTWO", name: "Take-Two Interactive", sector: "Communication Services" },
    { ticker: "TXN", name: "Texas Instruments", sector: "Information Technology" },
    { ticker: "TXT", name: "Textron", sector: "Industrials" },
    { ticker: "TYL", name: "Tyler Technologies", sector: "Information Technology" },
    { ticker: "UAL", name: "United Airlines Holdings", sector: "Industrials" },
    { ticker: "UBER", name: "Uber", sector: "Industrials" },
    { ticker: "UDR", name: "UDR, Inc.", sector: "Real Estate" },
    { ticker: "UHS", name: "Universal Health Services", sector: "Health Care" },
    { ticker: "ULTA", name: "Ulta Beauty", sector: "Consumer Discretionary" },
    { ticker: "UNH", name: "UnitedHealth Group", sector: "Health Care" },
    { ticker: "UNP", name: "Union Pacific Corporation", sector: "Industrials" },
    { ticker: "UPS", name: "United Parcel Service", sector: "Industrials" },
    { ticker: "URI", name: "United Rentals", sector: "Industrials" },
    { ticker: "USB", name: "U.S. Bancorp", sector: "Financials" },
    { ticker: "V", name: "Visa Inc.", sector: "Financials" },
    { ticker: "VEEV", name: "Veeva Systems", sector: "Health Care" },
    { ticker: "VICI", name: "Vici Properties", sector: "Real Estate" },
    { ticker: "VLO", name: "Valero Energy", sector: "Energy" },
    { ticker: "VLTO", name: "Veralto", sector: "Industrials" },
    { ticker: "VMC", name: "Vulcan Materials Company", sector: "Materials" },
    { ticker: "VMRK", name: "Vivmark Residential", sector: "Real Estate" },
    { ticker: "VRSK", name: "Verisk Analytics", sector: "Industrials" },
    { ticker: "VRSN", name: "Verisign", sector: "Information Technology" },
    { ticker: "VRT", name: "Vertiv", sector: "Industrials" },
    { ticker: "VRTX", name: "Vertex Pharmaceuticals", sector: "Health Care" },
    { ticker: "VST", name: "Vistra Corp.", sector: "Utilities" },
    { ticker: "VTR", name: "Ventas", sector: "Real Estate" },
    { ticker: "VTRS", name: "Viatris", sector: "Health Care" },
    { ticker: "VZ", name: "Verizon", sector: "Communication Services" },
    { ticker: "WAB", name: "Wabtec", sector: "Industrials" },
    { ticker: "WAT", name: "Waters Corporation", sector: "Health Care" },
    { ticker: "WBD", name: "Warner Bros. Discovery", sector: "Communication Services" },
    { ticker: "WDAY", name: "Workday, Inc.", sector: "Information Technology" },
    { ticker: "WDC", name: "Western Digital", sector: "Information Technology" },
    { ticker: "WEC", name: "WEC Energy Group", sector: "Utilities" },
    { ticker: "WELL", name: "Welltower", sector: "Real Estate" },
    { ticker: "WFC", name: "Wells Fargo", sector: "Financials" },
    { ticker: "WM", name: "Waste Management", sector: "Industrials" },
    { ticker: "WMB", name: "Williams Companies", sector: "Energy" },
    { ticker: "WMT", name: "Walmart", sector: "Consumer Staples" },
    { ticker: "WRB", name: "W. R. Berkley Corporation", sector: "Financials" },
    { ticker: "WSM", name: "Williams-Sonoma, Inc.", sector: "Consumer Discretionary" },
    { ticker: "WST", name: "West Pharmaceutical Services", sector: "Health Care" },
    { ticker: "WTW", name: "Willis Towers Watson", sector: "Financials" },
    { ticker: "WY", name: "Weyerhaeuser", sector: "Real Estate" },
    { ticker: "WYNN", name: "Wynn Resorts", sector: "Consumer Discretionary" },
    { ticker: "XEL", name: "Xcel Energy", sector: "Utilities" },
    { ticker: "XOM", name: "ExxonMobil", sector: "Energy" },
    { ticker: "XYL", name: "Xylem Inc.", sector: "Industrials" },
    { ticker: "XYZ", name: "Block, Inc.", sector: "Financials" },
    { ticker: "YUM", name: "Yum! Brands", sector: "Consumer Discretionary" },
    { ticker: "ZBH", name: "Zimmer Biomet", sector: "Health Care" },
    { ticker: "ZBRA", name: "Zebra Technologies", sector: "Information Technology" },
    { ticker: "ZTS", name: "Zoetis", sector: "Health Care" },
];

export const C = {
    bg0: "#080c14", bg1: "#0d1524", bg2: "#111d2e", bg3: "#17263d",
    border: "rgba(42,58,92,.7)", amber: "#fbbf24", amberDim: "rgba(251,191,36,.15)",
    amberLow: "rgba(251,191,36,.07)", cyan: "#22d3ee", green: "#10b981",
    red: "#f43f5e", purple: "#a78bfa", text: "#e2e8f0", textDim: "#64748b", textMid: "#94a3b8",
};
