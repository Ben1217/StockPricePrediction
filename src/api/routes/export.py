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
        writer.writerow(["date", "symbol", "type", "quantity", "price", "commission"])
        for t in data["trades"]:
            writer.writerow([t["date"], t["symbol"], t["type"], t["quantity"], t["price"], t["commission"]])
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
        metrics = data["metrics"]

        elements.append(Paragraph("Performance Metrics", styles["Heading2"]))
        table_data = [["Metric", "Value"]]
        for k, v in metrics.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            table_data.append([k.replace("_", " ").title(), val])
        t = Table(table_data, colWidths=[200, 150])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Trade Log", styles["Heading2"]))
        trade_data = [["Date", "Type", "Qty", "Price", "Commission"]]
        for tr in data["trades"][:50]:  # Limit to 50
            trade_data.append([
                tr["date"], tr["type"], f"{tr['quantity']:.2f}",
                f"${tr['price']:.2f}", f"${tr['commission']:.2f}",
            ])
        t2 = Table(trade_data, colWidths=[100, 60, 70, 80, 80])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(t2)
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
