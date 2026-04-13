"""
Backtest Engine
Simple backtesting framework for strategy evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ..portfolio.performance_metrics import calculate_portfolio_metrics
from ..utils.logger import get_logger
from ..utils.config_loader import get_config_value

logger = get_logger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    date: pd.Timestamp
    symbol: str
    trade_type: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    reason: str = ""
    confidence: float = 0.0
    predicted_return: Optional[float] = None
    probability_up: Optional[float] = None
    patterns: List[str] = field(default_factory=list)
    strategy: Optional[str] = None
    model_type: Optional[str] = None
    entry_date: Optional[pd.Timestamp] = None
    holding_days: Optional[int] = None
    realized_pnl: Optional[float] = None
    return_pct: Optional[float] = None


class BacktestEngine:
    """Simple backtesting engine for strategy evaluation"""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.0,
        slippage_rate: float = 0.001
    ):
        """
        Initialize backtest engine

        Parameters
        ----------
        initial_capital : float
            Starting capital
        commission_rate : float
            Commission per trade (as fraction)
        slippage_rate : float
            Slippage per trade (as fraction)
        """
        config = get_config_value('backtesting', {})

        self.initial_capital = config.get('initial_capital', initial_capital)
        self.commission_rate = config.get('costs', {}).get('commission', commission_rate)
        self.slippage_rate = config.get('costs', {}).get('slippage', slippage_rate)

        self.reset()

    def reset(self):
        """Reset backtest state"""
        self.cash = self.initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.position_cost_basis: Dict[str, float] = {}
        self.position_entry_dates: Dict[str, pd.Timestamp] = {}
        self.position_metadata: Dict[str, Dict] = {}
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.portfolio_values: List[float] = []
        self.dates: List[pd.Timestamp] = []

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        position_size: float = 0.1,
        signal_details: Optional[pd.DataFrame] = None,
        strategy_name: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> Dict:
        """
        Run backtest

        Parameters
        ----------
        prices : pandas.DataFrame
            Price data with datetime index and columns for each symbol
        signals : pandas.DataFrame
            Trading signals (1 = buy, -1 = sell, 0 = hold)
        position_size : float
            Fraction of portfolio per position

        Returns
        -------
        dict
            Backtest results
        """
        self.reset()

        for date in prices.index:
            current_prices = prices.loc[date]

            # Calculate portfolio value
            portfolio_value = self.cash
            for symbol, qty in self.positions.items():
                if symbol in current_prices.index:
                    portfolio_value += qty * current_prices[symbol]

            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)

            # Process signals
            if date in signals.index:
                for symbol in signals.columns:
                    signal = signals.loc[date, symbol]
                    price = current_prices.get(symbol, None)
                    detail = None

                    if (
                        signal_details is not None
                        and date in signal_details.index
                        and symbol in signal_details.columns
                    ):
                        detail = signal_details.loc[date, symbol]

                    if price is None or pd.isna(signal):
                        continue

                    if signal == 1:  # Buy signal
                        self._execute_buy(
                            date,
                            symbol,
                            price,
                            portfolio_value * position_size,
                            detail=detail,
                            strategy_name=strategy_name,
                            model_type=model_type,
                        )
                    elif signal == -1:  # Sell signal
                        self._execute_sell(
                            date,
                            symbol,
                            price,
                            detail=detail,
                            strategy_name=strategy_name,
                            model_type=model_type,
                        )

        return self._generate_report()

    def _execute_buy(
        self,
        date: pd.Timestamp,
        symbol: str,
        price: float,
        amount: float,
        detail: Optional[Dict] = None,
        strategy_name: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        """Execute buy order"""
        if self.positions.get(symbol, 0) > 0:
            return

        # Apply slippage
        execution_price = price * (1 + self.slippage_rate)

        # Calculate quantity
        commission = amount * self.commission_rate
        quantity = (amount - commission) / execution_price

        if quantity > 0 and self.cash >= amount:
            self.cash -= amount
            self.positions[symbol] = quantity
            self.position_cost_basis[symbol] = amount
            self.position_entry_dates[symbol] = date
            self.position_metadata[symbol] = detail or {}

            self.trades.append(Trade(
                date=date,
                symbol=symbol,
                trade_type='BUY',
                quantity=quantity,
                price=execution_price,
                commission=commission,
                slippage=price * self.slippage_rate * quantity,
                reason=(detail or {}).get('reason', ''),
                confidence=float((detail or {}).get('confidence', 0.0) or 0.0),
                predicted_return=(detail or {}).get('predicted_return'),
                probability_up=(detail or {}).get('probability_up'),
                patterns=list((detail or {}).get('patterns', []) or []),
                strategy=strategy_name,
                model_type=model_type,
            ))

    def _execute_sell(
        self,
        date: pd.Timestamp,
        symbol: str,
        price: float,
        detail: Optional[Dict] = None,
        strategy_name: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        """Execute sell order (sell all)"""
        quantity = self.positions.get(symbol, 0)

        if quantity > 0:
            # Apply slippage
            execution_price = price * (1 - self.slippage_rate)
            proceeds = quantity * execution_price
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission
            cost_basis = self.position_cost_basis.get(symbol, 0.0)
            realized_pnl = net_proceeds - cost_basis
            return_pct = (realized_pnl / cost_basis) if cost_basis > 0 else 0.0
            entry_date = self.position_entry_dates.get(symbol)
            holding_days = None
            if entry_date is not None:
                holding_days = int((date - entry_date).days)

            self.cash += net_proceeds
            del self.positions[symbol]
            self.position_cost_basis.pop(symbol, None)
            self.position_entry_dates.pop(symbol, None)
            entry_metadata = self.position_metadata.pop(symbol, {})

            sell_trade = Trade(
                date=date,
                symbol=symbol,
                trade_type='SELL',
                quantity=quantity,
                price=execution_price,
                commission=commission,
                slippage=price * self.slippage_rate * quantity,
                reason=(detail or {}).get('reason', ''),
                confidence=float((detail or {}).get('confidence', 0.0) or 0.0),
                predicted_return=(detail or {}).get('predicted_return'),
                probability_up=(detail or {}).get('probability_up'),
                patterns=list((detail or {}).get('patterns', []) or []),
                strategy=strategy_name,
                model_type=model_type,
                entry_date=entry_date,
                holding_days=holding_days,
                realized_pnl=realized_pnl,
                return_pct=return_pct,
            )
            self.trades.append(sell_trade)
            self.closed_trades.append(sell_trade)

    def _generate_report(self) -> Dict:
        """Generate backtest report"""
        if not self.portfolio_values:
            return {}

        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().dropna()

        metrics = calculate_portfolio_metrics(returns)

        # Additional backtest-specific metrics
        metrics['initial_capital'] = self.initial_capital
        metrics['final_value'] = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        metrics['total_trades'] = len(self.trades)
        metrics['closed_trades'] = len(self.closed_trades)
        metrics['total_commission'] = sum(t.commission for t in self.trades)
        metrics['total_slippage'] = sum(t.slippage for t in self.trades)

        # Trade analysis
        buy_trades = [t for t in self.trades if t.trade_type == 'BUY']
        sell_trades = [t for t in self.trades if t.trade_type == 'SELL']
        metrics['buy_trades'] = len(buy_trades)
        metrics['sell_trades'] = len(sell_trades)
        metrics.update(self._calculate_trade_metrics())

        if len(portfolio_series) > 1:
            span_days = max(1, (portfolio_series.index[-1] - portfolio_series.index[0]).days)
            years = span_days / 365.25
            metrics['cagr'] = (
                (metrics['final_value'] / self.initial_capital) ** (1 / years) - 1
                if years > 0 and self.initial_capital > 0
                else 0.0
            )
        else:
            metrics['cagr'] = 0.0

        logger.info(f"Backtest complete: {metrics['total_return']:.2%} return, "
                   f"{metrics['total_trades']} trades")

        return {
            'metrics': metrics,
            'portfolio_values': portfolio_series,
            'trades': self.trades,
            'closed_trades': self.closed_trades,
        }

    def _calculate_trade_metrics(self) -> Dict[str, float]:
        """Calculate closed-trade performance metrics."""
        if not self.closed_trades:
            return {
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'expectancy': 0.0,
                'expectancy_value': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
            }

        pnl_values = pd.Series([float(t.realized_pnl or 0.0) for t in self.closed_trades])
        return_values = pd.Series([float(t.return_pct or 0.0) for t in self.closed_trades])
        wins = pnl_values[pnl_values > 0]
        losses = pnl_values[pnl_values < 0]

        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = float(losses.sum()) if not losses.empty else 0.0
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else gross_profit if gross_profit > 0 else 0.0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        for pnl in pnl_values:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
            else:
                current_wins = 0
                current_losses = 0
            max_wins = max(max_wins, current_wins)
            max_losses = max(max_losses, current_losses)

        return {
            'profit_factor': float(profit_factor) if np.isfinite(profit_factor) else gross_profit,
            'average_win': float(wins.mean()) if not wins.empty else 0.0,
            'average_loss': float(losses.mean()) if not losses.empty else 0.0,
            'expectancy': float(return_values.mean()) if not return_values.empty else 0.0,
            'expectancy_value': float(pnl_values.mean()) if not pnl_values.empty else 0.0,
            'max_consecutive_wins': int(max_wins),
            'max_consecutive_losses': int(max_losses),
        }
