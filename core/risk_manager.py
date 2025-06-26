import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
from config.settings import RISK_LEVELS

logger = logging.getLogger(__name__)

class RiskManager:
    """Advanced risk management system with dynamic adjustment"""
    
    def __init__(self, initial_capital: float, risk_level: str):
        self.initial_capital = initial_capital
        self.risk_level = risk_level
        self.risk_params = RISK_LEVELS[risk_level]
        
        # Dynamic risk tracking
        self.daily_loss_limit = initial_capital * self.risk_params['max_daily_loss']
        self.position_limit = self.risk_params['max_position_size']
        self.correlation_limit = self.risk_params['max_correlation']
        
        # Risk metrics tracking
        self.daily_losses = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.risk_adjusted_returns = []
        self.position_correlations = {}
        
        # Emergency controls
        self.emergency_stop = False
        self.risk_alerts = []
        
        logger.info(f"Risk Manager initialized: {risk_level} level")
    
    def can_trade(self, pair: str, signal: Dict[str, Any]) -> bool:
        """Determine if a trade can be executed based on risk parameters"""
        try:
            # Emergency stop check
            if self.emergency_stop:
                logger.warning(f"Trading blocked: Emergency stop active")
                return False
            
            # Daily loss limit check
            if self.daily_losses >= self.daily_loss_limit:
                logger.warning(f"Trading blocked: Daily loss limit reached")
                return False
            
            # Drawdown limit check
            if self.current_drawdown >= self.risk_params['max_drawdown']:
                logger.warning(f"Trading blocked: Maximum drawdown reached")
                return False
            
            # Signal quality check
            if signal.get('confidence', 0) < self.risk_params['min_signal_confidence']:
                logger.info(f"Trade blocked: Signal confidence too low for {pair}")
                return False
            
            # Volatility check
            if signal.get('volatility', 0) > self.risk_params['max_volatility']:
                logger.info(f"Trade blocked: Volatility too high for {pair}")
                return False
            
            # Correlation check
            if self._check_correlation_risk(pair):
                logger.info(f"Trade blocked: Correlation risk too high for {pair}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in can_trade check: {e}")
            return False
    
    def calculate_position_size(self, pair: str, signal: Dict[str, Any], available_balance: float) -> float:
        """Calculate optimal position size based on risk parameters"""
        try:
            # Base position size as percentage of capital
            base_size = available_balance * self.risk_params['position_size_pct']
            
            # Adjust based on signal confidence
            confidence_multiplier = signal.get('confidence', 0.5)
            adjusted_size = base_size * confidence_multiplier
            
            # Adjust based on volatility (higher volatility = smaller position)
            volatility = signal.get('volatility', 0.02)
            volatility_multiplier = max(0.5, 1 - (volatility * 10))
            adjusted_size *= volatility_multiplier
            
            # Adjust based on current drawdown
            drawdown_multiplier = max(0.3, 1 - (self.current_drawdown / self.risk_params['max_drawdown']))
            adjusted_size *= drawdown_multiplier
            
            # Apply maximum position size limit
            max_position_value = self.initial_capital * self.position_limit
            if adjusted_size > max_position_value:
                adjusted_size = max_position_value
            
            # Convert to token amount
            price = signal.get('price', 1)
            position_tokens = adjusted_size / price
            
            logger.info(f"Position size calculated for {pair}: {position_tokens:.6f} tokens (${adjusted_size:.2f})")
            
            return position_tokens
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def assess_portfolio_risk(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        try:
            if not positions:
                return {'risk_level': 'LOW', 'risky_positions': []}
            
            total_exposure = 0
            position_risks = []
            
            for pair, position in positions.items():
                # Calculate position risk
                unrealized_pnl_pct = position.get('unrealized_pnl_pct', 0)
                position_value = position.get('size', 0) * position.get('current_price', 0)
                
                total_exposure += position_value
                
                # Individual position risk assessment
                risk_score = 0
                if unrealized_pnl_pct < -10:  # >10% loss
                    risk_score += 3
                elif unrealized_pnl_pct < -5:  # >5% loss
                    risk_score += 2
                elif unrealized_pnl_pct < -2:  # >2% loss
                    risk_score += 1
                
                position_risks.append({
                    'pair': pair,
                    'risk_score': risk_score,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'position_value': position_value
                })
            
            # Sort by risk score
            position_risks.sort(key=lambda x: x['risk_score'], reverse=True)
            
            # Overall portfolio risk assessment
            portfolio_exposure_pct = total_exposure / self.initial_capital
            high_risk_positions = [p for p in position_risks if p['risk_score'] >= 2]
            
            # Determine risk level
            if (portfolio_exposure_pct > 0.8 or 
                len(high_risk_positions) > 3 or 
                self.current_drawdown > self.risk_params['max_drawdown'] * 0.8):
                risk_level = 'CRITICAL'
            elif (portfolio_exposure_pct > 0.6 or 
                  len(high_risk_positions) > 1 or 
                  self.current_drawdown > self.risk_params['max_drawdown'] * 0.5):
                risk_level = 'HIGH'
            elif portfolio_exposure_pct > 0.4 or len(high_risk_positions) > 0:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            return {
                'risk_level': risk_level,
                'portfolio_exposure_pct': portfolio_exposure_pct,
                'risky_positions': high_risk_positions,
                'total_positions': len(positions),
                'current_drawdown': self.current_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return {'risk_level': 'UNKNOWN', 'risky_positions': []}
    
    def update_risk_metrics(self, portfolio_value: float, daily_pnl: float):
        """Update risk metrics based on portfolio performance"""
        try:
            # Update daily losses
            if daily_pnl < 0:
                self.daily_losses = abs(daily_pnl)
            else:
                self.daily_losses = 0  # Reset if profitable
            
            # Update drawdown
            peak_value = max(self.initial_capital, portfolio_value)
            current_drawdown = (peak_value - portfolio_value) / peak_value
            self.current_drawdown = current_drawdown
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Check for emergency stop conditions
            if (self.current_drawdown >= self.risk_params['max_drawdown'] or
                self.daily_losses >= self.daily_loss_limit):
                self.emergency_stop = True
                logger.critical("🚨 EMERGENCY STOP ACTIVATED")
            
            # Calculate risk-adjusted returns (Sharpe-like ratio)
            returns = (portfolio_value - self.initial_capital) / self.initial_capital
            self.risk_adjusted_returns.append(returns)
            
            # Keep only recent returns for calculation
            if len(self.risk_adjusted_returns) > 100:
                self.risk_adjusted_returns = self.risk_adjusted_returns[-100:]
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def _check_correlation_risk(self, pair: str) -> bool:
        """Check if adding this position would create too much correlation risk"""
        try:
            # For crypto pairs, check if we already have too much exposure to similar assets
            base_asset = pair.split('/')[0]  # e.g., 'BTC' from 'BTC/USDT'
            
            # Count similar positions
            similar_exposure = 0
            for existing_pair in self.position_correlations:
                if existing_pair.startswith(base_asset):
                    similar_exposure += 1
            
            # Limit exposure to same base asset
            return similar_exposure >= 2
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return False
    
    def adjust_parameters(self, performance_metrics: Dict[str, Any]):
        """Dynamically adjust risk parameters based on performance"""
        try:
            win_rate = performance_metrics.get('win_rate', 50)
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            
            # Increase position size if performing well
            if win_rate > 70 and sharpe_ratio > 1.5:
                self.risk_params['position_size_pct'] = min(
                    self.risk_params['position_size_pct'] * 1.1,
                    RISK_LEVELS[self.risk_level]['position_size_pct'] * 1.5
                )
                logger.info("Risk parameters adjusted: Increased position sizing")
            
            # Decrease position size if performing poorly
            elif win_rate < 40 or sharpe_ratio < 0:
                self.risk_params['position_size_pct'] = max(
                    self.risk_params['position_size_pct'] * 0.9,
                    RISK_LEVELS[self.risk_level]['position_size_pct'] * 0.5
                )
                logger.info("Risk parameters adjusted: Decreased position sizing")
            
            # Reset emergency stop if conditions improve
            if (self.current_drawdown < self.risk_params['max_drawdown'] * 0.5 and
                self.daily_losses < self.daily_loss_limit * 0.5 and
                self.emergency_stop):
                self.emergency_stop = False
                logger.info("Emergency stop deactivated - conditions improved")
            
        except Exception as e:
            logger.error(f"Error adjusting risk parameters: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        return {
            'risk_level': self.risk_level,
            'emergency_stop': self.emergency_stop,
            'daily_losses': self.daily_losses,
            'daily_loss_limit': self.daily_loss_limit,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_limit': self.risk_params['max_drawdown'],
            'position_size_pct': self.risk_params['position_size_pct'],
            'risk_alerts': self.risk_alerts[-10:] if self.risk_alerts else []
        }
