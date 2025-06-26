"""
Options Strategies Engine per Delta-Neutral e Multi-Leg Spreads
Implementa straddle, strangle, butterfly, iron condor per catturare volatilità
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class StrategyType(Enum):
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    BUTTERFLY = "butterfly"
    IRON_CONDOR = "iron_condor"
    DELTA_HEDGE = "delta_hedge"

@dataclass
class OptionContract:
    symbol: str
    strike: float
    expiry: datetime
    option_type: OptionType
    premium: float
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_vol: float
    underlying_price: float

@dataclass
class OptionsPosition:
    contracts: List[Tuple[OptionContract, int]]  # (contract, quantity)
    strategy_type: StrategyType
    entry_time: datetime
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    delta: float
    gamma: float
    theta: float
    vega: float

class BlackScholesCalculator:
    """Calcolatore Black-Scholes per opzioni"""
    
    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: OptionType) -> float:
        """Calcola prezzo opzione usando Black-Scholes"""
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == OptionType.CALL:
            price = S * BlackScholesCalculator._norm_cdf(d1) - K * np.exp(-r * T) * BlackScholesCalculator._norm_cdf(d2)
        else:
            price = K * np.exp(-r * T) * BlackScholesCalculator._norm_cdf(-d2) - S * BlackScholesCalculator._norm_cdf(-d1)
        
        return max(0, price)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: OptionType) -> Dict[str, float]:
        """Calcola greeks per opzione"""
        if T <= 0:
            return {
                'delta': 1.0 if (option_type == OptionType.CALL and S > K) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = BlackScholesCalculator._norm_cdf(d1)
        else:
            delta = -BlackScholesCalculator._norm_cdf(-d1)
        
        # Gamma (stesso per call e put)
        gamma = BlackScholesCalculator._norm_pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        common_term = -S * BlackScholesCalculator._norm_pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == OptionType.CALL:
            theta = common_term - r * K * np.exp(-r * T) * BlackScholesCalculator._norm_cdf(d2)
        else:
            theta = common_term + r * K * np.exp(-r * T) * BlackScholesCalculator._norm_cdf(-d2)
        theta = theta / 365  # Per day
        
        # Vega (stesso per call e put)
        vega = S * BlackScholesCalculator._norm_pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Cumulative distribution function normale standard"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Probability density function normale standard"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

class ImpliedVolatilityCalculator:
    """Calcolatore volatilità implicita"""
    
    @staticmethod
    def calculate_iv(market_price: float, S: float, K: float, T: float, 
                    r: float, option_type: OptionType) -> float:
        """Calcola volatilità implicita usando Newton-Raphson"""
        if T <= 0:
            return 0.0
        
        # Initial guess
        sigma = 0.3
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            price = BlackScholesCalculator.calculate_option_price(S, K, T, r, sigma, option_type)
            greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, option_type)
            
            diff = price - market_price
            vega = greeks['vega'] * 100  # Convert back to absolute vega
            
            if abs(diff) < tolerance or vega < 1e-10:
                break
                
            sigma = sigma - diff / vega
            sigma = max(0.01, min(5.0, sigma))  # Keep within reasonable bounds
        
        return sigma

class OptionsStrategiesEngine:
    """Motore principale per strategie options"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.active_positions = {}
        self.volatility_targets = {
            'low': 0.15,    # 15% target IV for low vol strategies
            'medium': 0.25, # 25% target IV
            'high': 0.40    # 40% target IV for high vol strategies
        }
        
    def create_long_straddle(self, underlying_price: float, strike: float, 
                           expiry_days: int, implied_vol: float) -> OptionsPosition:
        """Crea long straddle (long call + long put stesso strike)"""
        expiry = datetime.now() + timedelta(days=expiry_days)
        T = expiry_days / 365.0
        
        # Create call and put contracts
        call = self._create_option_contract(
            underlying_price, strike, T, implied_vol, OptionType.CALL
        )
        put = self._create_option_contract(
            underlying_price, strike, T, implied_vol, OptionType.PUT
        )
        
        # Position: Long 1 call, Long 1 put
        contracts = [(call, 1), (put, 1)]
        
        # Calculate P&L characteristics
        total_premium = call.premium + put.premium
        max_loss = total_premium
        max_profit = float('inf')  # Unlimited upside
        breakeven_up = strike + total_premium
        breakeven_down = strike - total_premium
        
        # Combined greeks
        total_delta = call.delta + put.delta  # Should be close to 0
        total_gamma = call.gamma + put.gamma
        total_theta = call.theta + put.theta  # Negative (time decay)
        total_vega = call.vega + put.vega      # Positive (volatility exposure)
        
        return OptionsPosition(
            contracts=contracts,
            strategy_type=StrategyType.LONG_STRADDLE,
            entry_time=datetime.now(),
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_down, breakeven_up],
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega
        )
    
    def create_long_strangle(self, underlying_price: float, call_strike: float, 
                           put_strike: float, expiry_days: int, implied_vol: float) -> OptionsPosition:
        """Crea long strangle (OTM call + OTM put)"""
        expiry = datetime.now() + timedelta(days=expiry_days)
        T = expiry_days / 365.0
        
        call = self._create_option_contract(
            underlying_price, call_strike, T, implied_vol, OptionType.CALL
        )
        put = self._create_option_contract(
            underlying_price, put_strike, T, implied_vol, OptionType.PUT
        )
        
        contracts = [(call, 1), (put, 1)]
        
        total_premium = call.premium + put.premium
        max_loss = total_premium
        max_profit = float('inf')
        breakeven_up = call_strike + total_premium
        breakeven_down = put_strike - total_premium
        
        return OptionsPosition(
            contracts=contracts,
            strategy_type=StrategyType.LONG_STRANGLE,
            entry_time=datetime.now(),
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_down, breakeven_up],
            delta=call.delta + put.delta,
            gamma=call.gamma + put.gamma,
            theta=call.theta + put.theta,
            vega=call.vega + put.vega
        )
    
    def create_butterfly_spread(self, underlying_price: float, lower_strike: float,
                              middle_strike: float, upper_strike: float,
                              expiry_days: int, implied_vol: float) -> OptionsPosition:
        """Crea butterfly spread (neutral strategy)"""
        expiry = datetime.now() + timedelta(days=expiry_days)
        T = expiry_days / 365.0
        
        # Long 1 lower strike call
        call_lower = self._create_option_contract(
            underlying_price, lower_strike, T, implied_vol, OptionType.CALL
        )
        
        # Short 2 middle strike calls
        call_middle = self._create_option_contract(
            underlying_price, middle_strike, T, implied_vol, OptionType.CALL
        )
        
        # Long 1 upper strike call
        call_upper = self._create_option_contract(
            underlying_price, upper_strike, T, implied_vol, OptionType.CALL
        )
        
        contracts = [
            (call_lower, 1),   # Long
            (call_middle, -2), # Short
            (call_upper, 1)    # Long
        ]
        
        # Net premium (should be debit)
        net_premium = call_lower.premium - 2 * call_middle.premium + call_upper.premium
        max_loss = abs(net_premium)
        max_profit = (middle_strike - lower_strike) - abs(net_premium)
        
        return OptionsPosition(
            contracts=contracts,
            strategy_type=StrategyType.BUTTERFLY,
            entry_time=datetime.now(),
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[
                lower_strike + abs(net_premium),
                upper_strike - abs(net_premium)
            ],
            delta=call_lower.delta - 2 * call_middle.delta + call_upper.delta,
            gamma=call_lower.gamma - 2 * call_middle.gamma + call_upper.gamma,
            theta=call_lower.theta - 2 * call_middle.theta + call_upper.theta,
            vega=call_lower.vega - 2 * call_middle.vega + call_upper.vega
        )
    
    def create_iron_condor(self, underlying_price: float, put_strike_low: float,
                         put_strike_high: float, call_strike_low: float,
                         call_strike_high: float, expiry_days: int, 
                         implied_vol: float) -> OptionsPosition:
        """Crea iron condor (range-bound strategy)"""
        expiry = datetime.now() + timedelta(days=expiry_days)
        T = expiry_days / 365.0
        
        # Short put spread + Short call spread
        put_low = self._create_option_contract(
            underlying_price, put_strike_low, T, implied_vol, OptionType.PUT
        )
        put_high = self._create_option_contract(
            underlying_price, put_strike_high, T, implied_vol, OptionType.PUT
        )
        call_low = self._create_option_contract(
            underlying_price, call_strike_low, T, implied_vol, OptionType.CALL
        )
        call_high = self._create_option_contract(
            underlying_price, call_strike_high, T, implied_vol, OptionType.CALL
        )
        
        contracts = [
            (put_low, 1),    # Long put (protection)
            (put_high, -1),  # Short put
            (call_low, -1),  # Short call
            (call_high, 1)   # Long call (protection)
        ]
        
        # Net credit received
        net_credit = -put_low.premium + put_high.premium - call_low.premium + call_high.premium
        max_profit = net_credit
        max_loss = (put_strike_high - put_strike_low) - net_credit
        
        return OptionsPosition(
            contracts=contracts,
            strategy_type=StrategyType.IRON_CONDOR,
            entry_time=datetime.now(),
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[
                put_strike_high - net_credit,
                call_strike_low + net_credit
            ],
            delta=put_low.delta - put_high.delta - call_low.delta + call_high.delta,
            gamma=put_low.gamma - put_high.gamma - call_low.gamma + call_high.gamma,
            theta=put_low.theta - put_high.theta - call_low.theta + call_high.theta,
            vega=put_low.vega - put_high.vega - call_low.vega + call_high.vega
        )
    
    def _create_option_contract(self, underlying_price: float, strike: float, 
                              time_to_expiry: float, implied_vol: float,
                              option_type: OptionType) -> OptionContract:
        """Crea contratto opzione con calcolo greeks"""
        
        premium = BlackScholesCalculator.calculate_option_price(
            underlying_price, strike, time_to_expiry, self.risk_free_rate, 
            implied_vol, option_type
        )
        
        greeks = BlackScholesCalculator.calculate_greeks(
            underlying_price, strike, time_to_expiry, self.risk_free_rate,
            implied_vol, option_type
        )
        
        return OptionContract(
            symbol=f"OPT_{strike}_{option_type.value}",
            strike=strike,
            expiry=datetime.now() + timedelta(days=int(time_to_expiry * 365)),
            option_type=option_type,
            premium=premium,
            delta=greeks['delta'],
            gamma=greeks['gamma'],
            theta=greeks['theta'],
            vega=greeks['vega'],
            implied_vol=implied_vol,
            underlying_price=underlying_price
        )
    
    def analyze_volatility_opportunity(self, current_iv: float, historical_vol: float,
                                     underlying_price: float) -> Dict[str, Any]:
        """Analizza opportunità basate su volatilità"""
        
        # IV Percentile calculation (simplified)
        iv_percentile = min(100, max(0, (current_iv - 0.1) / 0.3 * 100))
        
        # Volatility regime classification
        vol_regime = "low" if current_iv < 0.20 else "high" if current_iv > 0.35 else "medium"
        
        # Strategy recommendations
        recommendations = []
        
        if current_iv > historical_vol * 1.5:  # IV molto alta
            recommendations.extend([
                {
                    'strategy': 'short_straddle',
                    'reason': 'High IV - sell premium',
                    'confidence': 0.8,
                    'risk': 'high'
                },
                {
                    'strategy': 'iron_condor',
                    'reason': 'High IV - range bound expectation',
                    'confidence': 0.7,
                    'risk': 'medium'
                }
            ])
        elif current_iv < historical_vol * 0.7:  # IV molto bassa
            recommendations.extend([
                {
                    'strategy': 'long_straddle',
                    'reason': 'Low IV - buy cheap options',
                    'confidence': 0.75,
                    'risk': 'medium'
                },
                {
                    'strategy': 'long_strangle',
                    'reason': 'Low IV - cheap volatility play',
                    'confidence': 0.6,
                    'risk': 'medium'
                }
            ])
        else:  # IV neutrale
            recommendations.append({
                'strategy': 'butterfly',
                'reason': 'Neutral IV - profit from time decay',
                'confidence': 0.6,
                'risk': 'low'
            })
        
        return {
            'current_iv': current_iv,
            'historical_vol': historical_vol,
            'iv_percentile': iv_percentile,
            'volatility_regime': vol_regime,
            'iv_vs_hv_ratio': current_iv / historical_vol if historical_vol > 0 else 1.0,
            'recommendations': recommendations,
            'underlying_price': underlying_price
        }
    
    def execute_delta_neutral_strategy(self, position: OptionsPosition, 
                                     underlying_price: float, hedge_ratio: float = 1.0) -> Dict[str, Any]:
        """Esegue strategia delta-neutral con hedge dinamico"""
        
        current_delta = position.delta
        hedge_size = -current_delta * hedge_ratio  # Opposite delta to neutralize
        
        position_id = f"delta_neutral_{datetime.now().timestamp()}"
        
        self.active_positions[position_id] = {
            'options_position': position,
            'hedge_size': hedge_size,
            'underlying_price': underlying_price,
            'entry_time': datetime.now(),
            'target_delta': 0.0,
            'delta_tolerance': 0.1,  # ±0.1 delta tolerance
            'last_rebalance': datetime.now()
        }
        
        logger.info(f"Executed delta-neutral strategy: {position.strategy_type.value}, hedge size: {hedge_size:.4f}")
        
        return {
            'success': True,
            'position_id': position_id,
            'strategy_type': position.strategy_type.value,
            'initial_delta': current_delta,
            'hedge_size': hedge_size,
            'max_profit': position.max_profit,
            'max_loss': position.max_loss,
            'vega_exposure': position.vega,
            'theta_decay': position.theta
        }
    
    def rebalance_delta_hedge(self, position_id: str, new_underlying_price: float) -> Dict[str, Any]:
        """Ribilancia hedge delta-neutral"""
        
        if position_id not in self.active_positions:
            return {'success': False, 'reason': 'position_not_found'}
        
        position_data = self.active_positions[position_id]
        options_pos = position_data['options_position']
        
        # Recalculate option deltas with new underlying price
        # (In real implementation, this would fetch current option prices)
        
        # For now, estimate new delta based on gamma
        price_change = new_underlying_price - position_data['underlying_price']
        estimated_new_delta = options_pos.delta + (options_pos.gamma * price_change)
        
        current_hedge = position_data['hedge_size']
        new_hedge_size = -estimated_new_delta
        hedge_adjustment = new_hedge_size - current_hedge
        
        # Only rebalance if delta is outside tolerance
        if abs(estimated_new_delta) > position_data['delta_tolerance']:
            position_data['hedge_size'] = new_hedge_size
            position_data['underlying_price'] = new_underlying_price
            position_data['last_rebalance'] = datetime.now()
            
            logger.info(f"Rebalanced delta hedge for {position_id}: adjustment {hedge_adjustment:.4f}")
            
            return {
                'success': True,
                'rebalanced': True,
                'old_delta': estimated_new_delta - (options_pos.gamma * price_change),
                'new_delta': estimated_new_delta,
                'hedge_adjustment': hedge_adjustment,
                'new_hedge_size': new_hedge_size
            }
        
        return {
            'success': True,
            'rebalanced': False,
            'current_delta': estimated_new_delta,
            'reason': 'within_tolerance'
        }
    
    def get_options_dashboard(self) -> Dict[str, Any]:
        """Dashboard per strategie options"""
        
        total_vega = sum(
            pos['options_position'].vega 
            for pos in self.active_positions.values()
        )
        
        total_theta = sum(
            pos['options_position'].theta 
            for pos in self.active_positions.values()
        )
        
        strategy_breakdown = {}
        for pos in self.active_positions.values():
            strategy = pos['options_position'].strategy_type.value
            if strategy not in strategy_breakdown:
                strategy_breakdown[strategy] = 0
            strategy_breakdown[strategy] += 1
        
        return {
            'active_strategies': len(self.active_positions),
            'total_vega_exposure': total_vega,
            'total_theta_decay': total_theta,
            'strategy_breakdown': strategy_breakdown,
            'volatility_targets': self.volatility_targets,
            'risk_free_rate': self.risk_free_rate,
            'positions_summary': [
                {
                    'id': pid,
                    'strategy': pos['options_position'].strategy_type.value,
                    'vega': pos['options_position'].vega,
                    'theta': pos['options_position'].theta,
                    'current_delta': pos['options_position'].delta,
                    'hedge_size': pos['hedge_size']
                }
                for pid, pos in self.active_positions.items()
            ]
        }