"""
Dynamic Leverage Engine con ATR, Margin Management e De-leverage automatico
Implementa leverage adattivo 1-10x basato su volatilità e drawdown storico
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ATRCalculator:
    """Calcolatore di Average True Range per volatilità"""
    
    def __init__(self, period: int = 14):
        self.period = period
        
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calcola ATR per determinare volatilità"""
        if len(data) < self.period:
            return pd.Series([0.01] * len(data), index=data.index)
            
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        
        return atr.fillna(0.01) if hasattr(atr, 'fillna') else pd.Series([0.01] * len(data), index=data.index)

class DrawdownTracker:
    """Tracker di drawdown per adattamento leverage"""
    
    def __init__(self, lookback_periods: int = 30):
        self.lookback_periods = lookback_periods
        self.equity_curve = []
        self.drawdown_history = []
        
    def update_equity(self, equity_value: float):
        """Aggiorna valore equity per calcolo drawdown"""
        self.equity_curve.append(equity_value)
        
        # Mantieni solo gli ultimi N periodi
        if len(self.equity_curve) > self.lookback_periods:
            self.equity_curve.pop(0)
            
        self._calculate_current_drawdown()
    
    def _calculate_current_drawdown(self):
        """Calcola drawdown corrente"""
        if len(self.equity_curve) < 2:
            self.drawdown_history.append(0.0)
            return
            
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        self.drawdown_history.append(drawdown)
        
        # Mantieni solo storia recente
        if len(self.drawdown_history) > self.lookback_periods:
            self.drawdown_history.pop(0)
    
    def get_max_drawdown(self) -> float:
        """Ottiene max drawdown nel periodo"""
        return max(self.drawdown_history) if self.drawdown_history else 0.0
    
    def get_avg_drawdown(self) -> float:
        """Ottiene drawdown medio"""
        return np.mean(self.drawdown_history) if self.drawdown_history else 0.0

class MarginManager:
    """Gestore margin con isolated/cross mode"""
    
    def __init__(self, initial_balance: float):
        self.total_balance = initial_balance
        self.isolated_positions = {}  # symbol -> margin_info
        self.cross_margin_used = 0.0
        self.margin_ratio_threshold = 0.8  # Soglia per de-leverage
        
    def add_isolated_position(self, symbol: str, position_size: float, 
                            leverage: float, entry_price: float) -> bool:
        """Aggiunge posizione isolated margin"""
        required_margin = (position_size * entry_price) / leverage
        
        if required_margin > self.get_available_balance():
            logger.warning(f"Insufficient balance for isolated position {symbol}")
            return False
            
        self.isolated_positions[symbol] = {
            'size': position_size,
            'leverage': leverage,
            'entry_price': entry_price,
            'margin_used': required_margin,
            'unrealized_pnl': 0.0,
            'liquidation_price': self._calculate_liquidation_price(
                entry_price, leverage, 'long'  # Assumiamo long per ora
            )
        }
        
        return True
    
    def _calculate_liquidation_price(self, entry_price: float, 
                                   leverage: float, side: str) -> float:
        """Calcola prezzo di liquidazione"""
        maintenance_margin_rate = 0.01  # 1%
        
        if side == 'long':
            return entry_price * (1 - (1/leverage) + maintenance_margin_rate)
        else:
            return entry_price * (1 + (1/leverage) - maintenance_margin_rate)
    
    def update_position_pnl(self, symbol: str, current_price: float):
        """Aggiorna PnL posizione"""
        if symbol not in self.isolated_positions:
            return
            
        pos = self.isolated_positions[symbol]
        pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['size']
        
        # Calcola margin ratio
        margin_value = pos['margin_used'] + pos['unrealized_pnl']
        position_value = pos['size'] * current_price
        margin_ratio = margin_value / position_value if position_value > 0 else 0
        
        # Trigger de-leverage se necessario
        if margin_ratio > self.margin_ratio_threshold:
            self._trigger_deleverage(symbol, current_price)
    
    def _trigger_deleverage(self, symbol: str, current_price: float):
        """Trigger de-leverage automatico"""
        logger.warning(f"Triggering de-leverage for {symbol} at price {current_price}")
        
        pos = self.isolated_positions[symbol]
        # Riduci leverage del 50%
        new_leverage = max(1.0, pos['leverage'] * 0.5)
        
        # Calcola nuova size per mantenere stesso margin
        new_size = (pos['margin_used'] * new_leverage) / current_price
        
        pos['leverage'] = new_leverage
        pos['size'] = new_size
        
        logger.info(f"De-leveraged {symbol}: New leverage {new_leverage:.2f}x, New size {new_size:.6f}")
    
    def get_available_balance(self) -> float:
        """Calcola balance disponibile"""
        used_margin = sum(pos['margin_used'] for pos in self.isolated_positions.values())
        return self.total_balance - used_margin - self.cross_margin_used
    
    def get_total_unrealized_pnl(self) -> float:
        """Calcola PnL totale non realizzato"""
        return sum(pos['unrealized_pnl'] for pos in self.isolated_positions.values())

class DynamicLeverageEngine:
    """Motore principale per leverage dinamico"""
    
    def __init__(self, initial_balance: float):
        self.atr_calculator = ATRCalculator()
        self.drawdown_tracker = DrawdownTracker()
        self.margin_manager = MarginManager(initial_balance)
        
        # Parametri di configurazione
        self.min_leverage = 1.0
        self.max_leverage = 10.0
        self.base_leverage = 3.0
        
        # Soglie ATR per adattamento leverage
        self.low_volatility_threshold = 0.02    # 2%
        self.high_volatility_threshold = 0.08   # 8%
        
        # Soglie drawdown
        self.max_drawdown_threshold = 0.15      # 15%
        
    def calculate_optimal_leverage(self, symbol: str, market_data: pd.DataFrame,
                                 current_price: float) -> Dict[str, Any]:
        """Calcola leverage ottimale basato su ATR e drawdown"""
        
        # Calcola ATR
        atr_series = self.atr_calculator.calculate_atr(market_data)
        current_atr = atr_series.iloc[-1] if len(atr_series) > 0 else 0.02
        atr_percentage = current_atr / current_price
        
        # Calcola leverage basato su volatilità
        if atr_percentage <= self.low_volatility_threshold:
            volatility_leverage = self.max_leverage * 0.8  # Meno aggressivo anche in bassa vol
        elif atr_percentage >= self.high_volatility_threshold:
            volatility_leverage = self.min_leverage
        else:
            # Interpolazione lineare
            vol_range = self.high_volatility_threshold - self.low_volatility_threshold
            vol_position = (atr_percentage - self.low_volatility_threshold) / vol_range
            volatility_leverage = self.max_leverage * 0.8 - (vol_position * (self.max_leverage * 0.8 - self.min_leverage))
        
        # Adattamento basato su drawdown
        max_dd = self.drawdown_tracker.get_max_drawdown()
        drawdown_multiplier = 1.0
        
        if max_dd > self.max_drawdown_threshold:
            drawdown_multiplier = 0.5  # Riduci leverage del 50%
        elif max_dd > self.max_drawdown_threshold * 0.5:
            drawdown_multiplier = 0.75  # Riduci del 25%
        
        # Leverage finale
        optimal_leverage = max(
            self.min_leverage,
            min(self.max_leverage, volatility_leverage * drawdown_multiplier)
        )
        
        return {
            'optimal_leverage': optimal_leverage,
            'current_atr': current_atr,
            'atr_percentage': atr_percentage,
            'max_drawdown': max_dd,
            'volatility_regime': self._classify_volatility_regime(atr_percentage),
            'drawdown_multiplier': drawdown_multiplier,
            'recommended_action': self._get_leverage_recommendation(optimal_leverage)
        }
    
    def _classify_volatility_regime(self, atr_percentage: float) -> str:
        """Classifica regime di volatilità"""
        if atr_percentage <= self.low_volatility_threshold:
            return "low_volatility"
        elif atr_percentage >= self.high_volatility_threshold:
            return "high_volatility" 
        else:
            return "medium_volatility"
    
    def _get_leverage_recommendation(self, leverage: float) -> str:
        """Genera raccomandazione basata su leverage"""
        if leverage >= 7:
            return "aggressive_bullish"
        elif leverage >= 5:
            return "moderate_bullish"
        elif leverage >= 3:
            return "neutral"
        elif leverage >= 2:
            return "conservative"
        else:
            return "defensive"
    
    def execute_leveraged_position(self, symbol: str, side: str, size: float,
                                 price: float, leverage: float) -> Dict[str, Any]:
        """Esegue posizione con leverage specificato"""
        
        # Verifica disponibilità margin
        success = self.margin_manager.add_isolated_position(
            symbol, size, leverage, price
        )
        
        if not success:
            return {
                'success': False,
                'reason': 'insufficient_margin',
                'available_balance': self.margin_manager.get_available_balance()
            }
        
        # Calcola trailing stop basato su leverage
        trailing_stop_distance = self._calculate_trailing_stop(leverage, price)
        
        return {
            'success': True,
            'symbol': symbol,
            'side': side,
            'size': size,
            'leverage': leverage,
            'entry_price': price,
            'trailing_stop_distance': trailing_stop_distance,
            'liquidation_price': self.margin_manager.isolated_positions[symbol]['liquidation_price'],
            'margin_used': self.margin_manager.isolated_positions[symbol]['margin_used']
        }
    
    def _calculate_trailing_stop(self, leverage: float, price: float) -> float:
        """Calcola distanza trailing stop basata su leverage"""
        # Stop loss più stretto per leverage alto
        base_stop_pct = 0.02  # 2%
        leverage_adjusted_stop = base_stop_pct * (10 / leverage)  # Inverso del leverage
        
        return price * leverage_adjusted_stop
    
    def update_positions(self, market_prices: Dict[str, float]):
        """Aggiorna tutte le posizioni con prezzi correnti"""
        for symbol, price in market_prices.items():
            if symbol in self.margin_manager.isolated_positions:
                self.margin_manager.update_position_pnl(symbol, price)
    
    def get_leverage_dashboard(self) -> Dict[str, Any]:
        """Dashboard stato leverage engine"""
        total_pnl = self.margin_manager.get_total_unrealized_pnl()
        available_balance = self.margin_manager.get_available_balance()
        
        positions_summary = []
        for symbol, pos in self.margin_manager.isolated_positions.items():
            positions_summary.append({
                'symbol': symbol,
                'size': pos['size'],
                'leverage': pos['leverage'],
                'entry_price': pos['entry_price'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'liquidation_price': pos['liquidation_price'],
                'margin_used': pos['margin_used']
            })
        
        return {
            'total_balance': self.margin_manager.total_balance,
            'available_balance': available_balance,
            'total_unrealized_pnl': total_pnl,
            'max_drawdown': self.drawdown_tracker.get_max_drawdown(),
            'active_positions': len(self.margin_manager.isolated_positions),
            'positions_summary': positions_summary,
            'leverage_range': f"{self.min_leverage}x - {self.max_leverage}x"
        }