"""
Perpetual Futures Arbitrage Engine con Funding Rates
Esegue arbitraggio spot-futures e inter-exchange su contratti perpetui
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FundingRate:
    exchange: str
    symbol: str
    rate: float
    next_funding_time: datetime
    predicted_rate: float
    
@dataclass
class PerpetualContract:
    exchange: str
    symbol: str
    mark_price: float
    index_price: float
    funding_rate: float
    open_interest: float
    volume_24h: float
    basis_points: float  # (mark_price - index_price) / index_price * 10000

class FundingRatePredictor:
    """Predittore di funding rates basato su pattern storici"""
    
    def __init__(self):
        self.funding_history = {}  # exchange_symbol -> list of rates
        self.pattern_memory = 24  # Ore di memoria per pattern
        
    def add_funding_data(self, exchange: str, symbol: str, rate: float, timestamp: datetime):
        """Aggiunge dato storico funding rate"""
        key = f"{exchange}_{symbol}"
        if key not in self.funding_history:
            self.funding_history[key] = []
            
        self.funding_history[key].append({
            'rate': rate,
            'timestamp': timestamp,
            'hour': timestamp.hour
        })
        
        # Mantieni solo ultimi dati
        if len(self.funding_history[key]) > self.pattern_memory * 3:  # 3 giorni di storia
            self.funding_history[key] = self.funding_history[key][-self.pattern_memory * 3:]
    
    def predict_next_funding(self, exchange: str, symbol: str) -> float:
        """Predice prossimo funding rate"""
        key = f"{exchange}_{symbol}"
        if key not in self.funding_history or len(self.funding_history[key]) < 8:
            return 0.0001  # Default 0.01%
            
        data = self.funding_history[key]
        recent_rates = [d['rate'] for d in data[-8:]]  # Ultimi 8 funding
        
        # Media mobile con peso maggiore ai recenti
        weights = np.exp(np.linspace(0, 1, len(recent_rates)))
        weighted_avg = np.average(recent_rates, weights=weights)
        
        # Aggiungi pattern temporale
        current_hour = datetime.now().hour
        hour_pattern = self._get_hourly_pattern(data, current_hour)
        
        predicted = (weighted_avg * 0.7) + (hour_pattern * 0.3)
        
        # Clamp tra valori ragionevoli
        return max(-0.01, min(0.01, predicted))  # ±1%
    
    def _get_hourly_pattern(self, data: List[Dict], target_hour: int) -> float:
        """Ottiene pattern funding per ora specifica"""
        hour_rates = [d['rate'] for d in data if d['hour'] == target_hour]
        return np.mean(hour_rates) if hour_rates else 0.0001

class PerpetualArbitrageEngine:
    """Motore principale per arbitraggio perpetual futures"""
    
    def __init__(self):
        self.funding_predictor = FundingRatePredictor()
        self.exchanges = ['binance', 'bybit', 'okx', 'deribit']
        self.contracts = {}  # symbol -> {exchange -> PerpetualContract}
        self.active_positions = {}
        
        # Parametri di configurazione
        self.min_profit_threshold = 0.0025  # 0.25% minimo profit
        self.max_position_size = 10000  # USDT max per position
        self.funding_rate_threshold = 0.001  # 0.1% soglia funding interessante
        
    def update_contract_data(self, contracts: List[PerpetualContract]):
        """Aggiorna dati contratti perpetui"""
        for contract in contracts:
            symbol = contract.symbol
            if symbol not in self.contracts:
                self.contracts[symbol] = {}
            
            self.contracts[symbol][contract.exchange] = contract
            
            # Aggiorna predictor
            self.funding_predictor.add_funding_data(
                contract.exchange, symbol, contract.funding_rate, datetime.now()
            )
    
    def find_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """Trova opportunità di arbitraggio"""
        opportunities = []
        
        for symbol in self.contracts:
            # Spot-Future arbitrage
            spot_future_opps = self._find_spot_future_arbitrage(symbol)
            opportunities.extend(spot_future_opps)
            
            # Inter-exchange arbitrage
            inter_exchange_opps = self._find_inter_exchange_arbitrage(symbol)
            opportunities.extend(inter_exchange_opps)
            
            # Funding rate arbitrage
            funding_opps = self._find_funding_rate_arbitrage(symbol)
            opportunities.extend(funding_opps)
        
        # Ordina per profittabilità
        opportunities.sort(key=lambda x: x['expected_profit'], reverse=True)
        
        return opportunities[:10]  # Top 10 opportunità
    
    def _find_spot_future_arbitrage(self, symbol: str) -> List[Dict[str, Any]]:
        """Trova arbitraggio spot-future"""
        opportunities = []
        
        if symbol not in self.contracts:
            return opportunities
            
        for exchange, contract in self.contracts[symbol].items():
            # Calcola spread spot-future
            basis_pct = contract.basis_points / 10000
            
            if abs(basis_pct) > self.min_profit_threshold:
                # Determina direzione
                if basis_pct > 0:  # Future premium
                    action = "sell_future_buy_spot"
                    profit_potential = basis_pct
                else:  # Future discount
                    action = "buy_future_sell_spot"
                    profit_potential = abs(basis_pct)
                
                # Calcola funding cost/benefit
                predicted_funding = self.funding_predictor.predict_next_funding(exchange, symbol)
                funding_hours_to_expiry = 8  # Assumiamo 8h all'equilibrio
                total_funding_cost = predicted_funding * funding_hours_to_expiry
                
                net_profit = profit_potential - abs(total_funding_cost)
                
                if net_profit > self.min_profit_threshold:
                    opportunities.append({
                        'type': 'spot_future_arbitrage',
                        'symbol': symbol,
                        'exchange': exchange,
                        'action': action,
                        'basis_points': contract.basis_points,
                        'expected_profit': net_profit,
                        'funding_cost': total_funding_cost,
                        'mark_price': contract.mark_price,
                        'index_price': contract.index_price,
                        'confidence': self._calculate_confidence(contract, net_profit)
                    })
        
        return opportunities
    
    def _find_inter_exchange_arbitrage(self, symbol: str) -> List[Dict[str, Any]]:
        """Trova arbitraggio inter-exchange"""
        opportunities = []
        
        if symbol not in self.contracts or len(self.contracts[symbol]) < 2:
            return opportunities
            
        exchanges = list(self.contracts[symbol].keys())
        
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex1, ex2 = exchanges[i], exchanges[j]
                contract1 = self.contracts[symbol][ex1]
                contract2 = self.contracts[symbol][ex2]
                
                price_diff = contract1.mark_price - contract2.mark_price
                price_diff_pct = abs(price_diff) / max(contract1.mark_price, contract2.mark_price)
                
                if price_diff_pct > self.min_profit_threshold:
                    # Determina direzione
                    if price_diff > 0:
                        buy_exchange = ex2
                        sell_exchange = ex1
                        cheaper_contract = contract2
                        expensive_contract = contract1
                    else:
                        buy_exchange = ex1
                        sell_exchange = ex2
                        cheaper_contract = contract1
                        expensive_contract = contract2
                    
                    # Considera funding rates di entrambi
                    buy_funding = self.funding_predictor.predict_next_funding(buy_exchange, symbol)
                    sell_funding = self.funding_predictor.predict_next_funding(sell_exchange, symbol)
                    funding_differential = sell_funding - buy_funding  # Net funding benefit
                    
                    # Calcola profitto netto
                    gross_profit = price_diff_pct
                    estimated_fees = 0.0008  # 0.08% total fees (0.04% per side)
                    net_profit = gross_profit - estimated_fees + (funding_differential * 4)  # 4 funding periods
                    
                    if net_profit > self.min_profit_threshold:
                        opportunities.append({
                            'type': 'inter_exchange_arbitrage',
                            'symbol': symbol,
                            'buy_exchange': buy_exchange,
                            'sell_exchange': sell_exchange,
                            'price_difference': abs(price_diff),
                            'price_diff_pct': price_diff_pct,
                            'expected_profit': net_profit,
                            'funding_differential': funding_differential,
                            'buy_price': cheaper_contract.mark_price,
                            'sell_price': expensive_contract.mark_price,
                            'confidence': self._calculate_inter_exchange_confidence(
                                cheaper_contract, expensive_contract, net_profit
                            )
                        })
        
        return opportunities
    
    def _find_funding_rate_arbitrage(self, symbol: str) -> List[Dict[str, Any]]:
        """Trova opportunità pure funding rate arbitrage"""
        opportunities = []
        
        if symbol not in self.contracts:
            return opportunities
            
        for exchange, contract in self.contracts[symbol].items():
            current_funding = contract.funding_rate
            predicted_funding = self.funding_predictor.predict_next_funding(exchange, symbol)
            
            # Se funding rate è molto alto/basso, c'è opportunità
            if abs(current_funding) > self.funding_rate_threshold:
                
                if current_funding > self.funding_rate_threshold:
                    # Funding positivo alto -> short futures per ricevere funding
                    action = "short_to_receive_funding"
                    profit_estimate = current_funding * 3  # 3 funding periods
                    side = "short"
                elif current_funding < -self.funding_rate_threshold:
                    # Funding negativo alto -> long futures per ricevere funding
                    action = "long_to_receive_funding"
                    profit_estimate = abs(current_funding) * 3
                    side = "long"
                else:
                    continue
                
                # Hedge risk con spot
                hedge_cost = 0.0002  # 0.02% spread cost per hedge
                net_profit = profit_estimate - hedge_cost
                
                if net_profit > self.min_profit_threshold * 0.5:  # Soglia più bassa per funding arb
                    opportunities.append({
                        'type': 'funding_rate_arbitrage',
                        'symbol': symbol,
                        'exchange': exchange,
                        'action': action,
                        'side': side,
                        'current_funding': current_funding,
                        'predicted_funding': predicted_funding,
                        'expected_profit': net_profit,
                        'hedge_required': True,
                        'confidence': self._calculate_funding_confidence(current_funding, predicted_funding)
                    })
        
        return opportunities
    
    def _calculate_confidence(self, contract: PerpetualContract, profit: float) -> float:
        """Calcola confidence score per opportunità"""
        base_confidence = 0.5
        
        # Volume boost
        if contract.volume_24h > 1000000:  # >1M volume
            base_confidence += 0.2
        
        # Open Interest boost
        if contract.open_interest > 500000:  # >500K OI
            base_confidence += 0.15
        
        # Profit margin boost
        if profit > 0.005:  # >0.5%
            base_confidence += 0.15
        
        return min(1.0, base_confidence)
    
    def _calculate_inter_exchange_confidence(self, contract1: PerpetualContract, 
                                          contract2: PerpetualContract, profit: float) -> float:
        """Calcola confidence per arbitraggio inter-exchange"""
        min_volume = min(contract1.volume_24h, contract2.volume_24h)
        min_oi = min(contract1.open_interest, contract2.open_interest)
        
        base_confidence = 0.4
        
        if min_volume > 500000:
            base_confidence += 0.25
        if min_oi > 300000:
            base_confidence += 0.2
        if profit > 0.01:  # >1%
            base_confidence += 0.15
        
        return min(1.0, base_confidence)
    
    def _calculate_funding_confidence(self, current: float, predicted: float) -> float:
        """Calcola confidence per funding rate arbitrage"""
        base_confidence = 0.6
        
        # Consistency boost se current e predicted sono simili
        if abs(current - predicted) < abs(current) * 0.3:
            base_confidence += 0.2
        
        # Extreme funding boost
        if abs(current) > 0.005:  # >0.5%
            base_confidence += 0.2
        
        return min(1.0, base_confidence)
    
    def execute_arbitrage(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue strategia di arbitraggio"""
        
        if opportunity['type'] == 'spot_future_arbitrage':
            return self._execute_spot_future_arbitrage(opportunity)
        elif opportunity['type'] == 'inter_exchange_arbitrage':
            return self._execute_inter_exchange_arbitrage(opportunity)
        elif opportunity['type'] == 'funding_rate_arbitrage':
            return self._execute_funding_rate_arbitrage(opportunity)
        
        return {'success': False, 'reason': 'unknown_opportunity_type'}
    
    def _execute_spot_future_arbitrage(self, opp: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue arbitraggio spot-future"""
        position_size = min(self.max_position_size, 5000)  # Conservative size
        
        position_id = f"spot_future_{opp['symbol']}_{datetime.now().timestamp()}"
        
        if opp['action'] == 'sell_future_buy_spot':
            # Sell future, buy spot
            future_side = 'short'
            spot_side = 'long'
        else:
            # Buy future, sell spot  
            future_side = 'long'
            spot_side = 'short'
        
        self.active_positions[position_id] = {
            'type': 'spot_future_arbitrage',
            'symbol': opp['symbol'],
            'exchange': opp['exchange'],
            'future_side': future_side,
            'spot_side': spot_side,
            'size': position_size,
            'entry_time': datetime.now(),
            'expected_profit': opp['expected_profit'],
            'status': 'active'
        }
        
        logger.info(f"Executed spot-future arbitrage for {opp['symbol']}: {future_side} future, {spot_side} spot")
        
        return {
            'success': True,
            'position_id': position_id,
            'type': 'spot_future_arbitrage',
            'size': position_size,
            'expected_profit_pct': opp['expected_profit'] * 100
        }
    
    def _execute_inter_exchange_arbitrage(self, opp: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue arbitraggio inter-exchange"""
        position_size = min(self.max_position_size, 7500)
        
        position_id = f"inter_exchange_{opp['symbol']}_{datetime.now().timestamp()}"
        
        self.active_positions[position_id] = {
            'type': 'inter_exchange_arbitrage',
            'symbol': opp['symbol'],
            'buy_exchange': opp['buy_exchange'],
            'sell_exchange': opp['sell_exchange'],
            'size': position_size,
            'buy_price': opp['buy_price'],
            'sell_price': opp['sell_price'],
            'entry_time': datetime.now(),
            'expected_profit': opp['expected_profit'],
            'status': 'active'
        }
        
        logger.info(f"Executed inter-exchange arbitrage for {opp['symbol']}: Buy on {opp['buy_exchange']}, Sell on {opp['sell_exchange']}")
        
        return {
            'success': True,
            'position_id': position_id,
            'type': 'inter_exchange_arbitrage',
            'size': position_size,
            'price_spread': opp['price_difference'],
            'expected_profit_pct': opp['expected_profit'] * 100
        }
    
    def _execute_funding_rate_arbitrage(self, opp: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue funding rate arbitrage"""
        position_size = min(self.max_position_size, 15000)  # Larger size for funding arb
        
        position_id = f"funding_arb_{opp['symbol']}_{datetime.now().timestamp()}"
        
        self.active_positions[position_id] = {
            'type': 'funding_rate_arbitrage',
            'symbol': opp['symbol'],
            'exchange': opp['exchange'],
            'side': opp['side'],
            'size': position_size,
            'entry_time': datetime.now(),
            'current_funding': opp['current_funding'],
            'expected_profit': opp['expected_profit'],
            'hedge_required': opp['hedge_required'],
            'status': 'active'
        }
        
        logger.info(f"Executed funding rate arbitrage for {opp['symbol']}: {opp['side']} position, funding rate {opp['current_funding']:.4f}")
        
        return {
            'success': True,
            'position_id': position_id,
            'type': 'funding_rate_arbitrage',
            'size': position_size,
            'funding_rate': opp['current_funding'],
            'expected_profit_pct': opp['expected_profit'] * 100
        }
    
    def get_arbitrage_dashboard(self) -> Dict[str, Any]:
        """Dashboard per perpetual arbitrage"""
        opportunities = self.find_arbitrage_opportunities()
        
        total_expected_profit = sum(pos['expected_profit'] for pos in self.active_positions.values())
        
        return {
            'active_opportunities': len(opportunities),
            'best_opportunity': opportunities[0] if opportunities else None,
            'active_positions': len(self.active_positions),
            'total_expected_profit': total_expected_profit,
            'opportunities_by_type': {
                'spot_future': len([o for o in opportunities if o['type'] == 'spot_future_arbitrage']),
                'inter_exchange': len([o for o in opportunities if o['type'] == 'inter_exchange_arbitrage']),
                'funding_rate': len([o for o in opportunities if o['type'] == 'funding_rate_arbitrage'])
            },
            'top_opportunities': opportunities[:5],
            'position_summary': [
                {
                    'id': pid,
                    'type': pos['type'],
                    'symbol': pos['symbol'],
                    'expected_profit': pos['expected_profit'],
                    'status': pos['status']
                }
                for pid, pos in self.active_positions.items()
            ]
        }