/**
 * The shaded uncertainty band behind the forecast line.
 *
 * lightweight-charts draws series, and a band is not one: it is the region
 * between two of them. There is no built-in series type for that, and the two
 * obvious workarounds are both wrong — an area series fills to the bottom of
 * the pane rather than to another line, and a pair of dotted bound lines draws
 * the edges of an interval without drawing the interval.
 *
 * So this is a series primitive: a plugin the library hands a canvas and a
 * coordinate converter, which paints the polygon between the upper and lower
 * bounds underneath everything else. It is attached to the forecast line
 * series, so it inherits that series' price scale and cannot drift out of
 * alignment with the line it belongs to.
 *
 * Two intervals are painted, the 95% and the 68%, the inner one darker. That is
 * deliberate: a single flat band reads as a boundary the price will stay
 * inside, and nested bands read as what they are — a density, thickest where
 * the outcome is most likely.
 *
 * @see https://tradingview.github.io/lightweight-charts/docs/plugins/series-primitives
 */

/**
 * Paints one filled region between an upper and a lower series of points.
 *
 * Held apart from the primitive because the 95% and 68% intervals are the same
 * drawing operation at different opacities, and because a renderer that only
 * knows about coordinates is testable without a chart.
 */
class BandRenderer {
    constructor(regions) {
        this._regions = regions;
    }

    draw(target) {
        target.useMediaCoordinateSpace(({ context }) => {
            for (const region of this._regions) {
                const { upper, lower, fill } = region;
                // Two points make a line, not an area. One is what a horizon of
                // a single bar produces, and filling it would paint nothing
                // while still costing a path.
                if (upper.length < 2 || lower.length < 2) continue;

                context.beginPath();
                context.moveTo(upper[0].x, upper[0].y);
                for (let i = 1; i < upper.length; i += 1) {
                    context.lineTo(upper[i].x, upper[i].y);
                }
                // Back along the lower bound to close the polygon.
                for (let i = lower.length - 1; i >= 0; i -= 1) {
                    context.lineTo(lower[i].x, lower[i].y);
                }
                context.closePath();
                context.fillStyle = fill;
                context.fill();
            }
        });
    }
}

/**
 * A pane view is the library's unit of "something to draw in the chart area".
 *
 * It recomputes screen coordinates from the data on every update — pan, zoom
 * and resize all invalidate them — and hands the renderer a plain list of
 * points, so nothing downstream needs a reference to the chart.
 */
class BandPaneView {
    constructor(source) {
        this._source = source;
        this._regions = [];
    }

    update() {
        const series = this._source.series;
        const timeScale = this._source.chart?.timeScale();
        const points = this._source.points;
        if (!series || !timeScale || !points?.length) {
            this._regions = [];
            return;
        }

        const project = (time, price) => {
            const x = timeScale.timeToCoordinate(time);
            const y = series.priceToCoordinate(price);
            // A bar scrolled off the visible range converts to null. Dropping
            // those rather than coercing them to 0 keeps the polygon from
            // collapsing onto the top of the pane while the user pans.
            return x === null || y === null ? null : { x, y };
        };

        const build = (upperKey, lowerKey, fill) => {
            const upper = [];
            const lower = [];
            for (const point of points) {
                const top = project(point.time, point[upperKey]);
                const bottom = project(point.time, point[lowerKey]);
                if (top && bottom) {
                    upper.push(top);
                    lower.push(bottom);
                }
            }
            return { upper, lower, fill };
        };

        this._regions = [
            build("upper95", "lower95", this._source.options.fill95),
            build("upper68", "lower68", this._source.options.fill68),
        ];
    }

    /**
     * Behind the candles and the forecast line.
     *
     * The band is context for the line, so anything drawn on top of it stays
     * readable; painted at 'normal' it would wash out the very series it exists
     * to qualify.
     */
    zOrder() {
        return "bottom";
    }

    renderer() {
        return new BandRenderer(this._regions);
    }
}

const DEFAULT_OPTIONS = {
    fill95: "rgba(99, 102, 241, 0.13)",
    fill68: "rgba(99, 102, 241, 0.20)",
};

/**
 * The primitive itself. Attach to a line series with `series.attachPrimitive`.
 *
 * `points` are `{ time, upper95, lower95, upper68, lower68 }` in the same time
 * format the series was given. Call `setPoints` to replace them; the chart
 * redraws itself through the `requestUpdate` callback the library supplies, so
 * callers never touch the canvas.
 */
export class ForecastBand {
    constructor(points = [], options = {}) {
        this.points = points;
        this.options = { ...DEFAULT_OPTIONS, ...options };
        this.chart = null;
        this.series = null;
        this._requestUpdate = null;
        this._paneView = new BandPaneView(this);
    }

    attached({ chart, series, requestUpdate }) {
        this.chart = chart;
        this.series = series;
        this._requestUpdate = requestUpdate;
    }

    detached() {
        this.chart = null;
        this.series = null;
        this._requestUpdate = null;
    }

    setPoints(points) {
        this.points = points || [];
        this._requestUpdate?.();
    }

    updateAllViews() {
        this._paneView.update();
    }

    paneViews() {
        return [this._paneView];
    }

    /**
     * Widen the price scale to fit the band.
     *
     * Without this the chart autoscales to the series data alone and the top of
     * the 95% interval is clipped off the pane — the band would silently be
     * drawn only as far as the candles happened to reach.
     */
    autoscaleInfo() {
        if (!this.points.length) return null;
        let minValue = Infinity;
        let maxValue = -Infinity;
        for (const point of this.points) {
            minValue = Math.min(minValue, point.lower95);
            maxValue = Math.max(maxValue, point.upper95);
        }
        if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) return null;
        return { priceRange: { minValue, maxValue } };
    }
}
