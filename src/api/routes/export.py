"""
Export API routes — CSV and PDF export for audit logs.
"""

import io
import csv
import numpy as np
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter()


def _format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _comparison_rows(runs):
    rows = [["Label", "Status", "Total Return", "CAGR", "Sharpe", "Sortino", "Max Drawdown", "Trades"]]
    for run in runs:
        metrics = run.get("metrics", {})
        rows.append([
            run.get("label", ""),
            run.get("status", ""),
            _format_metric_value(metrics.get("total_return", "")),
            _format_metric_value(metrics.get("cagr", "")),
            _format_metric_value(metrics.get("sharpe_ratio", "")),
            _format_metric_value(metrics.get("sortino_ratio", "")),
            _format_metric_value(metrics.get("max_drawdown", "")),
            str(metrics.get("total_trades", "")),
        ])
    return rows


@router.get("/csv/{resource}")
async def export_csv(
    resource: str,
    symbol: str = Query("SPY"),
):
    """Export data as CSV. Resources: prices, predictions, backtest, portfolio."""
    if resource == "prices":
        import yfinance as yf
        import pandas as pd
        from datetime import timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise HTTPException(404, f"No data for {symbol}")
        output = io.StringIO()
        df.to_csv(output)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={symbol}_prices_{datetime.now():%Y%m%d}.csv"},
        )

    elif resource == "backtest":
        from src.api.routes.backtest import _backtest_results
        if not _backtest_results:
            raise HTTPException(404, "No backtest results available. Run a backtest first.")
        latest_id = list(_backtest_results.keys())[-1]
        data = _backtest_results[latest_id]
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Summary"])
        for key, value in data.get("summary", {}).items():
            writer.writerow([key, value])

        writer.writerow([])
        writer.writerow(["Strategy Comparison"])
        writer.writerows(_comparison_rows(data.get("strategy_runs", [])))

        writer.writerow([])
        writer.writerow(["Model Comparison"])
        writer.writerows(_comparison_rows(data.get("model_runs", [])))

        writer.writerow([])
        writer.writerow(["Benchmarks"])
        writer.writerows(_comparison_rows(data.get("benchmarks", [])))

        writer.writerow([])
        writer.writerow(["Primary Trade Log"])
        writer.writerow(["date", "symbol", "type", "quantity", "price", "commission", "reason", "realized_pnl", "return_pct"])
        for t in data.get("trades", []):
            writer.writerow([
                t.get("date"),
                t.get("symbol"),
                t.get("type"),
                t.get("quantity"),
                t.get("price"),
                t.get("commission"),
                t.get("reason"),
                t.get("realized_pnl"),
                t.get("return_pct"),
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=backtest_{latest_id}_{datetime.now():%Y%m%d}.csv"},
        )

    elif resource == "predictions":
        from src.api.routes.predict import predict
        from src.api.schemas.schemas import PredictRequest
        req = PredictRequest(symbol=symbol, horizon=30)
        resp = await predict(req)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["date", "predicted", "lower68", "upper68", "lower95", "upper95"])
        for f in resp.forecasts:
            writer.writerow([f.date, f.predicted, f.lower68, f.upper68, f.lower95, f.upper95])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={symbol}_predictions_{datetime.now():%Y%m%d}.csv"},
        )

    else:
        raise HTTPException(400, f"Unknown resource: {resource}. Options: prices, predictions, backtest")


@router.get("/pdf/{resource}")
async def export_pdf(
    resource: str,
    symbol: str = Query("SPY"),
):
    """Export data as PDF report."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
    except ImportError:
        raise HTTPException(500, "reportlab not installed")

    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"QuantVision — {resource.capitalize()} Report", styles["Title"]))
    elements.append(Paragraph(f"Symbol: {symbol} | Generated: {datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]))
    elements.append(Spacer(1, 24))

    if resource == "backtest":
        from src.api.routes.backtest import _backtest_results
        if not _backtest_results:
            raise HTTPException(404, "No backtest results available")
        latest_id = list(_backtest_results.keys())[-1]
        data = _backtest_results[latest_id]
        metrics = data.get("metrics", {})

        elements.append(Paragraph("Performance Metrics", styles["Heading2"]))
        table_data = [["Metric", "Value"]]
        for k, v in metrics.items():
            table_data.append([k.replace("_", " ").title(), _format_metric_value(v)])
        t = Table(table_data, colWidths=[200, 150])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Strategy Comparison", styles["Heading2"]))
        strategy_table = Table(_comparison_rows(data.get("strategy_runs", [])), colWidths=[110, 55, 55, 45, 45, 45, 55, 40])
        strategy_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(strategy_table)
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Model Comparison", styles["Heading2"]))
        model_table = Table(_comparison_rows(data.get("model_runs", [])), colWidths=[110, 55, 55, 45, 45, 45, 55, 40])
        model_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(model_table)
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Benchmark Comparison", styles["Heading2"]))
        benchmark_table = Table(_comparison_rows(data.get("benchmarks", [])), colWidths=[110, 55, 55, 45, 45, 45, 55, 40])
        benchmark_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(benchmark_table)
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Primary Trade Log", styles["Heading2"]))
        trade_data = [["Date", "Type", "Qty", "Price", "Commission", "PnL"]]
        for tr in data.get("trades", [])[:50]:
            trade_data.append([
                tr.get("date"),
                tr.get("type"),
                str(tr.get("quantity", "")),
                f"${float(tr.get('price', 0)):.2f}" if tr.get("price") is not None else "",
                f"${float(tr.get('commission', 0)):.2f}" if tr.get("commission") is not None else "",
                f"${float(tr.get('realized_pnl', 0)):.2f}" if tr.get("realized_pnl") is not None else "",
            ])
        trade_table = Table(trade_data, colWidths=[85, 45, 55, 60, 60, 60])
        trade_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(trade_table)
    else:
        elements.append(Paragraph(f"Report for {resource} — {symbol}", styles["Normal"]))

    elements.append(Spacer(1, 24))
    elements.append(Paragraph("⚠️ Not financial advice — for informational use only", styles["Normal"]))

    doc.build(elements)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={resource}_{symbol}_{datetime.now():%Y%m%d}.pdf"},
    )
