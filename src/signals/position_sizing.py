"""
Position Sizing Calculator Module

Calculates optimal position sizes based on:
- Risk per trade (fixed dollar or percentage)
- Signal confidence
- Portfolio risk limits
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskProfile:
    """User's risk management profile."""
    account_balance: float = 30000.0
    risk_per_trade: float = 100.0  # Dollar amount
    risk_method: Literal['fixed_dollar', 'percentage'] = 'fixed_dollar'
    max_position_size: float = 5000.0
    max_portfolio_risk: float = 0.05  # 5%
    max_positions: int = 10
    confidence_scaling: bool = True


class PositionSizeCalculator:
    """
    Calculate position sizes based on user's risk profile.
    
    Uses the fixed dollar risk method from day trading principles:
    Shares = Risk Amount / Risk Per Share
    
    where Risk Per Share = Entry Price - Stop Loss
    """
    
    def __init__(self, risk_profile: Optional[RiskProfile] = None):
        """
        Initialize calculator.
        
        Parameters
        ----------
        risk_profile : RiskProfile, optional
            User's risk settings. Uses defaults if not provided.
        """
        self.profile = risk_profile or RiskProfile()
    
    def calculate_shares(
        self, 
        entry_price: float, 
        stop_loss: float,
        signal_confidence: float = 80.0,
        current_portfolio_risk: float = 0.0
    ) -> Dict:
        """
        Calculate optimal position size.
        
        Parameters
        ----------
        entry_price : float
            Planned entry price
        stop_loss : float
            Stop loss price
        signal_confidence : float
            Signal confidence (0-100)
        current_portfolio_risk : float
            Current total portfolio risk as decimal
        
        Returns
        -------
        dict
            shares, dollar_risk, position_value, warnings, etc.
        """
        warnings = []
        
        # Validate inputs
        if entry_price <= 0 or stop_loss <= 0:
            return {
                'shares': 0,
                'error': 'Invalid entry or stop loss price',
                'warnings': ['Entry and stop loss must be positive values']
            }
        
        # Step 1: Determine base risk amount
        if self.profile.risk_method == 'fixed_dollar':
            base_risk = self.profile.risk_per_trade
        else:  # percentage
            base_risk = self.profile.account_balance * self.profile.risk_per_trade
        
        # Step 2: Adjust for confidence (if enabled)
        if self.profile.confidence_scaling:
            adjusted_risk = self._adjust_for_confidence(base_risk, signal_confidence)
        else:
            adjusted_risk = base_risk
        
        # Step 3: Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return {
                'shares': 0,
                'error': 'Entry price equals stop loss - invalid setup',
                'warnings': ['Stop loss and entry must be different']
            }
        
        # Step 4: Calculate shares
        raw_shares = adjusted_risk / risk_per_share
        shares = int(raw_shares)  # Round down for safety
        
        # Step 5: Check minimum shares
        if shares < 1:
            warnings.append(
                f"⚠️ Position too small (< 1 share). "
                f"Consider widening stop loss or increasing risk amount."
            )
            shares = 0
        
        # Step 6: Apply position size limit
        position_value = shares * entry_price
        
        if position_value > self.profile.max_position_size:
            old_shares = shares
            shares = int(self.profile.max_position_size / entry_price)
            position_value = shares * entry_price
            warnings.append(
                f"⚠️ Position limited from {old_shares} to {shares} shares "
                f"(max ${self.profile.max_position_size:,.0f})"
            )
        
        # Step 7: Check portfolio risk limit
        actual_risk = shares * risk_per_share
        new_portfolio_risk = current_portfolio_risk + (actual_risk / self.profile.account_balance)
        
        if new_portfolio_risk > self.profile.max_portfolio_risk:
            # Calculate maximum additional risk allowed
            max_additional = (self.profile.max_portfolio_risk - current_portfolio_risk) 
            max_additional *= self.profile.account_balance
            
            if max_additional > 0:
                shares = int(max_additional / risk_per_share)
                position_value = shares * entry_price
                actual_risk = shares * risk_per_share
                warnings.append(
                    f"⚠️ Position reduced to maintain portfolio risk limit "
                    f"({self.profile.max_portfolio_risk:.0%})"
                )
            else:
                shares = 0
                position_value = 0
                actual_risk = 0
                warnings.append(
                    f"❌ Cannot add position - portfolio risk limit reached "
                    f"({current_portfolio_risk:.1%} / {self.profile.max_portfolio_risk:.0%})"
                )
        
        # Calculate metrics
        confidence_multiplier = adjusted_risk / base_risk if base_risk > 0 else 1.0
        
        # Target calculation (2.5:1 reward/risk)
        is_long = entry_price > stop_loss
        target_price = entry_price + (risk_per_share * 2.5) if is_long else entry_price - (risk_per_share * 2.5)
        potential_profit = shares * risk_per_share * 2.5
        
        return {
            'shares': shares,
            'position_value': position_value,
            'dollar_risk': actual_risk,
            'risk_per_share': risk_per_share,
            'risk_percentage': (actual_risk / self.profile.account_balance) * 100 if shares > 0 else 0,
            'confidence_multiplier': confidence_multiplier,
            'target_price': target_price,
            'potential_profit': potential_profit,
            'risk_reward_ratio': 2.5,
            'warnings': warnings
        }
    
    def _adjust_for_confidence(self, base_risk: float, confidence: float) -> float:
        """
        Scale position size based on signal confidence.
        
        Scaling:
        - 90-100% → 100% of base risk
        - 80-90%  → 100% of base risk
        - 70-80%  → 75% of base risk
        - 60-70%  → 50% of base risk
        - <60%    → 25% of base risk (or skip)
        """
        if confidence >= 80:
            return base_risk * 1.0
        elif confidence >= 70:
            return base_risk * 0.75
        elif confidence >= 60:
            return base_risk * 0.5
        else:
            return base_risk * 0.25
    
    def get_summary(self) -> Dict:
        """Get current risk profile summary."""
        return {
            'account_balance': self.profile.account_balance,
            'risk_per_trade': self.profile.risk_per_trade,
            'risk_method': self.profile.risk_method,
            'max_position_size': self.profile.max_position_size,
            'max_portfolio_risk': self.profile.max_portfolio_risk,
            'confidence_scaling': self.profile.confidence_scaling
        }


def get_default_risk_profile() -> RiskProfile:
    """Get default risk profile for new users."""
    return RiskProfile(
        account_balance=30000.0,
        risk_per_trade=100.0,
        risk_method='fixed_dollar',
        max_position_size=5000.0,
        max_portfolio_risk=0.05,
        max_positions=10,
        confidence_scaling=True
    )
