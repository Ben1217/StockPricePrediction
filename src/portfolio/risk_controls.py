"""
Risk Controls & Alerts — guardrail logic for portfolio risk management.

Checks position concentration, sector limits, portfolio stop-loss,
Sharpe ratio, drawdown, and correlation-based diversification.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional


@dataclass
class RiskAlert:
    """Structured risk alert with severity and context."""

    alert_type: Literal[
        "CONCENTRATION",
        "SECTOR_LIMIT",
        "PORTFOLIO_STOP",
        "HIGH_CORRELATION",
        "LOW_SHARPE",
        "HIGH_DRAWDOWN",
    ]
    severity: Literal["WARNING", "CRITICAL"]
    message: str
    value: float
    threshold: float
    ticker: Optional[str] = None


def check_risk_limits(
    current_weights: Dict[str, float],
    sector_weights: Dict[str, float],
    portfolio_metrics: Dict[str, float],
    correlation_result: Dict,
    config: Optional[Dict] = None,
) -> List[Dict]:
    """
    Run all risk checks and return a list of alert dicts.

    Parameters
    ----------
    current_weights : dict
        {ticker: current_weight}
    sector_weights : dict
        {sector: total_weight}
    portfolio_metrics : dict
        Output of calculate_portfolio_metrics()
    correlation_result : dict
        Output of calculate_correlation_matrix()
    config : dict, optional
        Override thresholds: max_position, max_sector, portfolio_stop_loss,
        min_sharpe, max_drawdown, high_corr_threshold

    Returns
    -------
    list[dict]
        List of alert dicts sorted by severity (CRITICAL first)
    """
    cfg = config or {}
    alerts: List[Dict] = []

    max_pos = cfg.get("max_position", 0.15)
    max_sector = cfg.get("max_sector", 0.35)
    stop_loss = cfg.get("portfolio_stop_loss", -0.15)
    min_sharpe = cfg.get("min_sharpe", 0.5)
    max_dd = cfg.get("max_drawdown", 0.20)

    # 1. Position concentration check
    for ticker, w in current_weights.items():
        if w > max_pos:
            alerts.append(
                asdict(
                    RiskAlert(
                        alert_type="CONCENTRATION",
                        severity="WARNING",
                        message=f"{ticker} weight {w:.1%} exceeds max {max_pos:.1%}",
                        value=w,
                        threshold=max_pos,
                        ticker=ticker,
                    )
                )
            )

    # 2. Sector concentration check
    for sector, w in sector_weights.items():
        if w > max_sector:
            alerts.append(
                asdict(
                    RiskAlert(
                        alert_type="SECTOR_LIMIT",
                        severity="WARNING",
                        message=f"{sector} sector {w:.1%} exceeds limit {max_sector:.1%}",
                        value=w,
                        threshold=max_sector,
                    )
                )
            )

    # 3. Portfolio-level stop loss
    total_return = portfolio_metrics.get("total_return", 0)
    if total_return < stop_loss:
        alerts.append(
            asdict(
                RiskAlert(
                    alert_type="PORTFOLIO_STOP",
                    severity="CRITICAL",
                    message=f"Portfolio return {total_return:.1%} below stop loss {stop_loss:.1%}",
                    value=total_return,
                    threshold=stop_loss,
                )
            )
        )

    # 4. Low Sharpe warning
    sharpe = portfolio_metrics.get("sharpe_ratio", 0)
    if sharpe < min_sharpe:
        alerts.append(
            asdict(
                RiskAlert(
                    alert_type="LOW_SHARPE",
                    severity="WARNING",
                    message=f"Sharpe ratio {sharpe:.2f} below minimum {min_sharpe}",
                    value=sharpe,
                    threshold=min_sharpe,
                )
            )
        )

    # 5. Max drawdown breach
    dd = abs(portfolio_metrics.get("max_drawdown", 0))
    if dd > max_dd:
        alerts.append(
            asdict(
                RiskAlert(
                    alert_type="HIGH_DRAWDOWN",
                    severity="CRITICAL",
                    message=f"Max drawdown {dd:.1%} exceeds limit {max_dd:.1%}",
                    value=dd,
                    threshold=max_dd,
                )
            )
        )

    # 6. High correlation pairs
    for pair in correlation_result.get("high_corr_pairs", []):
        alerts.append(
            asdict(
                RiskAlert(
                    alert_type="HIGH_CORRELATION",
                    severity="WARNING",
                    message=(
                        f"{pair['ticker_a']}/{pair['ticker_b']} correlation "
                        f"{pair['correlation']:.2f} — poor diversification"
                    ),
                    value=pair["correlation"],
                    threshold=0.80,
                )
            )
        )

    # Sort: CRITICAL first, then WARNING
    alerts.sort(key=lambda a: 0 if a["severity"] == "CRITICAL" else 1)
    return alerts
