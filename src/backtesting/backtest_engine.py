"""
Backtest Engine
Simple backtesting framework for strategy evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

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
        self.trades: List[Trade] = []
        self.portfolio_values: List[float] = []
        self.dates: List[pd.Timestamp] = []

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        position_size: float = 0.1
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

                    if price is None or pd.isna(signal):
                        continue

                    if signal == 1:  # Buy signal
                        self._execute_buy(date, symbol, price, portfolio_value * position_size)
                    elif signal == -1:  # Sell signal
                        self._execute_sell(date, symbol, price)

        return self._generate_report()

    def _execute_buy(
        self,
        date: pd.Timestamp,
        symbol: str,
        price: float,
        amount: float
    ):
        """Execute buy order"""
        # Apply slippage
        execution_price = price * (1 + self.slippage_rate)

        # Calculate quantity
        commission = amount * self.commission_rate
        quantity = (amount - commission) / execution_price

        if quantity > 0 and self.cash >= amount:
            self.cash -= amount
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity

            self.trades.append(Trade(
                date=date,
                symbol=symbol,
                trade_type='BUY',
                quantity=quantity,
                price=execution_price,
                commission=commission,
                slippage=price * self.slippage_rate * quantity
            ))

    def _execute_sell(
        self,
        date: pd.Timestamp,
        symbol: str,
        price: float
    ):
        """Execute sell order (sell all)"""
        quantity = self.positions.get(symbol, 0)

        if quantity > 0:
            # Apply slippage
            execution_price = price * (1 - self.slippage_rate)
            proceeds = quantity * execution_price
            commission = proceeds * self.commission_rate

            self.cash += proceeds - commission
            del self.positions[symbol]

            self.trades.append(Trade(
                date=date,
                symbol=symbol,
                trade_type='SELL',
                quantity=quantity,
                price=execution_price,
                commission=commission,
                slippage=price * self.slippage_rate * quantity
            ))

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
        metrics['total_commission'] = sum(t.commission for t in self.trades)
        metrics['total_slippage'] = sum(t.slippage for t in self.trades)

        # Trade analysis
        buy_trades = [t for t in self.trades if t.trade_type == 'BUY']
        sell_trades = [t for t in self.trades if t.trade_type == 'SELL']
        metrics['buy_trades'] = len(buy_trades)
        metrics['sell_trades'] = len(sell_trades)

        logger.info(f"Backtest complete: {metrics['total_return']:.2%} return, "
                   f"{metrics['total_trades']} trades")

        return {
            'metrics': metrics,
            'portfolio_values': portfolio_series,
            'trades': self.trades,
        }
