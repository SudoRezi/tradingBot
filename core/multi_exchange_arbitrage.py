"""
Sistema di Arbitraggio Multi-Exchange con Smart Order Routing
Supporta TWAP, VWAP, Iceberg orders e gestione fee/slippage
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit" 
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"

@dataclass
class ExchangeConfig:
    name: str
    trading_fee: float
    withdrawal_fee: float
    min_order_size: float
    max_order_size: float
    api_limit_per_minute: int
    base_url: str
    supports_margin: bool = False

@dataclass
class OrderBookEntry:
    price: float
    quantity: float
    exchange: str
    timestamp: datetime

@dataclass
class ArbitrageOpportunity:
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_percentage: float
    profit_absolute: float
    max_quantity: float
    estimated_fees: float
    net_profit: float
    confidence_score: float

class SmartOrderRouter:
    """Router intelligente per ottimizzare l'esecuzione degli ordini"""
    
    def __init__(self):
        self.execution_history = []
        self.exchange_performance = {}
        
    def calculate_twap_schedule(self, total_quantity: float, duration_minutes: int, 
                               market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calcola schedule per ordini TWAP"""
        try:
            intervals = max(1, duration_minutes // 5)  # Ordini ogni 5 minuti
            quantity_per_interval = total_quantity / intervals
            
            schedule = []
            current_time = datetime.now()
            
            for i in range(intervals):
                execution_time = current_time + timedelta(minutes=i * 5)
                
                schedule.append({
                    'execution_time': execution_time,
                    'quantity': quantity_per_interval,
                    'order_type': OrderType.LIMIT,
                    'price_strategy': 'mid_price',
                    'urgency': 'low'
                })
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error calculating TWAP schedule: {e}")
            return []
    
    def calculate_vwap_schedule(self, total_quantity: float, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calcola schedule per ordini VWAP basato sui volumi storici"""
        try:
            if market_data.empty or 'volume' not in market_data.columns:
                return self.calculate_twap_schedule(total_quantity, 60, market_data)
            
            # Analizza pattern di volume nelle ultime ore
            hourly_volumes = market_data.groupby(market_data.index.hour)['volume'].mean()
            total_expected_volume = hourly_volumes.sum()
            
            schedule = []
            current_hour = datetime.now().hour
            
            for hour_offset in range(12):  # Prossime 12 ore
                target_hour = (current_hour + hour_offset) % 24
                hour_volume_ratio = hourly_volumes.get(target_hour, 0) / total_expected_volume
                
                if hour_volume_ratio > 0:
                    quantity_for_hour = total_quantity * hour_volume_ratio
                    execution_time = datetime.now() + timedelta(hours=hour_offset)
                    
                    schedule.append({
                        'execution_time': execution_time,
                        'quantity': quantity_for_hour,
                        'order_type': OrderType.LIMIT,
                        'price_strategy': 'vwap_based',
                        'volume_ratio': hour_volume_ratio,
                        'urgency': 'medium' if hour_volume_ratio > 0.1 else 'low'
                    })
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error calculating VWAP schedule: {e}")
            return []
    
    def create_iceberg_orders(self, total_quantity: float, visible_quantity: float,
                             price: float, side: str) -> List[Dict[str, Any]]:
        """Crea sequenza di ordini iceberg"""
        try:
            orders = []
            remaining_quantity = total_quantity
            
            while remaining_quantity > 0:
                order_quantity = min(visible_quantity, remaining_quantity)
                
                orders.append({
                    'quantity': order_quantity,
                    'price': price,
                    'side': side,
                    'order_type': OrderType.ICEBERG,
                    'visible_quantity': order_quantity,
                    'hidden_quantity': max(0, remaining_quantity - order_quantity),
                    'execution_strategy': 'stealth'
                })
                
                remaining_quantity -= order_quantity
            
            return orders
            
        except Exception as e:
            logger.error(f"Error creating iceberg orders: {e}")
            return []
    
    def optimize_execution_strategy(self, order_size: float, market_impact_estimate: float,
                                  urgency: str, available_liquidity: float) -> Dict[str, Any]:
        """Ottimizza strategia di esecuzione basata su condizioni di mercato"""
        try:
            strategy = {
                'primary_method': OrderType.MARKET,
                'split_orders': False,
                'max_market_impact': 0.001,
                'estimated_slippage': 0.0,
                'execution_time_estimate': 1
            }
            
            # Calcola impatto di mercato
            liquidity_ratio = order_size / available_liquidity
            
            if liquidity_ratio > 0.05:  # Ordine grande (>5% della liquidità)
                if urgency == 'high':
                    strategy['primary_method'] = OrderType.ICEBERG
                    strategy['split_orders'] = True
                    strategy['iceberg_visible_ratio'] = 0.1
                else:
                    strategy['primary_method'] = OrderType.TWAP
                    strategy['execution_duration_minutes'] = 30
                    
            elif liquidity_ratio > 0.01:  # Ordine medio
                if urgency == 'high':
                    strategy['primary_method'] = OrderType.MARKET
                else:
                    strategy['primary_method'] = OrderType.VWAP
                    strategy['execution_duration_minutes'] = 15
                    
            else:  # Ordine piccolo
                strategy['primary_method'] = OrderType.MARKET
            
            # Stima slippage
            base_slippage = market_impact_estimate * liquidity_ratio
            strategy['estimated_slippage'] = base_slippage * (2 if urgency == 'high' else 1)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error optimizing execution strategy: {e}")
            return {'primary_method': OrderType.MARKET, 'estimated_slippage': 0.001}

class MultiExchangeArbitrage:
    """Sistema principale per arbitraggio multi-exchange"""
    
    def __init__(self):
        self.exchanges = self._initialize_exchanges()
        self.order_router = SmartOrderRouter()
        self.active_opportunities = []
        self.execution_history = []
        self.risk_limits = {
            'max_position_per_exchange': 10000,  # USDT
            'max_total_exposure': 50000,         # USDT
            'min_profit_threshold': 0.001,       # 0.1%
            'max_execution_time': 300            # 5 minuti
        }
        
    def _initialize_exchanges(self) -> Dict[str, ExchangeConfig]:
        """Inizializza configurazioni exchange"""
        return {
            'binance': ExchangeConfig(
                name='binance',
                trading_fee=0.001,
                withdrawal_fee=0.0005,
                min_order_size=10.0,
                max_order_size=1000000.0,
                api_limit_per_minute=1200,
                base_url='https://api.binance.com',
                supports_margin=True
            ),
            'kucoin': ExchangeConfig(
                name='kucoin',
                trading_fee=0.001,
                withdrawal_fee=0.0005,
                min_order_size=1.0,
                max_order_size=500000.0,
                api_limit_per_minute=600,
                base_url='https://api.kucoin.com'
            ),
            'kraken': ExchangeConfig(
                name='kraken',
                trading_fee=0.0026,
                withdrawal_fee=0.0005,
                min_order_size=5.0,
                max_order_size=200000.0,
                api_limit_per_minute=180,
                base_url='https://api.kraken.com'
            ),
            'coinbase': ExchangeConfig(
                name='coinbase',
                trading_fee=0.005,
                withdrawal_fee=0.0,
                min_order_size=1.0,
                max_order_size=1000000.0,
                api_limit_per_minute=300,
                base_url='https://api.exchange.coinbase.com'
            )
        }
    
    def get_simulated_order_book(self, exchange: str, pair: str) -> Dict[str, List[OrderBookEntry]]:
        """Simula order book per exchange (da sostituire con API reali)"""
        try:
            base_price = 43000.0  # Prezzo base BTC
            
            if 'ETH' in pair:
                base_price = 2600.0
            elif 'SOL' in pair:
                base_price = 100.0
            elif 'AVAX' in pair:
                base_price = 35.0
            elif 'KAS' in pair:
                base_price = 0.15
            
            # Aggiungi spread diverso per exchange
            spreads = {
                'binance': 0.0001,
                'kucoin': 0.0002,
                'kraken': 0.0005,
                'coinbase': 0.0008
            }
            
            spread = spreads.get(exchange, 0.0003)
            
            # Genera order book simulato
            bids = []
            asks = []
            
            # Bids (ordini di acquisto)
            for i in range(10):
                price = base_price * (1 - spread - i * 0.0001)
                quantity = np.random.uniform(0.1, 5.0)
                
                bids.append(OrderBookEntry(
                    price=price,
                    quantity=quantity,
                    exchange=exchange,
                    timestamp=datetime.now()
                ))
            
            # Asks (ordini di vendita)
            for i in range(10):
                price = base_price * (1 + spread + i * 0.0001)
                quantity = np.random.uniform(0.1, 5.0)
                
                asks.append(OrderBookEntry(
                    price=price,
                    quantity=quantity,
                    exchange=exchange,
                    timestamp=datetime.now()
                ))
            
            return {'bids': bids, 'asks': asks}
            
        except Exception as e:
            logger.error(f"Error getting order book for {exchange} {pair}: {e}")
            return {'bids': [], 'asks': []}
    
    def find_arbitrage_opportunities(self, pair: str) -> List[ArbitrageOpportunity]:
        """Trova opportunità di arbitraggio tra exchange"""
        try:
            opportunities = []
            
            # Ottieni order book da tutti gli exchange
            order_books = {}
            for exchange in self.exchanges:
                order_books[exchange] = self.get_simulated_order_book(exchange, pair)
            
            # Confronta prezzi tra exchange
            for buy_exchange in self.exchanges:
                for sell_exchange in self.exchanges:
                    if buy_exchange == sell_exchange:
                        continue
                    
                    buy_book = order_books[buy_exchange]
                    sell_book = order_books[sell_exchange]
                    
                    if not buy_book['asks'] or not sell_book['bids']:
                        continue
                    
                    # Miglior prezzo di acquisto e vendita
                    best_ask = min(buy_book['asks'], key=lambda x: x.price)
                    best_bid = max(sell_book['bids'], key=lambda x: x.price)
                    
                    # Calcola profitto potenziale
                    if best_bid.price > best_ask.price:
                        profit_absolute = best_bid.price - best_ask.price
                        profit_percentage = (profit_absolute / best_ask.price) * 100
                        
                        # Calcola fees
                        buy_fee = best_ask.price * self.exchanges[buy_exchange].trading_fee
                        sell_fee = best_bid.price * self.exchanges[sell_exchange].trading_fee
                        withdrawal_fee = self.exchanges[buy_exchange].withdrawal_fee * best_ask.price
                        
                        total_fees = buy_fee + sell_fee + withdrawal_fee
                        net_profit = profit_absolute - total_fees
                        
                        # Quantità massima eseguibile
                        max_quantity = min(best_ask.quantity, best_bid.quantity)
                        
                        # Score di confidenza basato su liquidità e spread
                        confidence = min(1.0, max_quantity / 10.0) * min(1.0, profit_percentage / 0.5)
                        
                        if net_profit > 0 and profit_percentage > self.risk_limits['min_profit_threshold'] * 100:
                            opportunity = ArbitrageOpportunity(
                                buy_exchange=buy_exchange,
                                sell_exchange=sell_exchange,
                                buy_price=best_ask.price,
                                sell_price=best_bid.price,
                                profit_percentage=profit_percentage,
                                profit_absolute=profit_absolute,
                                max_quantity=max_quantity,
                                estimated_fees=total_fees,
                                net_profit=net_profit,
                                confidence_score=confidence
                            )
                            
                            opportunities.append(opportunity)
            
            # Ordina per profitto netto decrescente
            opportunities.sort(key=lambda x: x.net_profit, reverse=True)
            
            return opportunities[:5]  # Top 5 opportunità
            
        except Exception as e:
            logger.error(f"Error finding arbitrage opportunities for {pair}: {e}")
            return []
    
    def execute_arbitrage(self, opportunity: ArbitrageOpportunity, quantity: float) -> Dict[str, Any]:
        """Esegue operazione di arbitraggio"""
        try:
            execution_plan = {
                'opportunity': opportunity,
                'quantity': quantity,
                'estimated_profit': opportunity.net_profit * quantity,
                'execution_steps': [],
                'status': 'planning'
            }
            
            # Step 1: Ordine di acquisto
            buy_strategy = self.order_router.optimize_execution_strategy(
                order_size=quantity * opportunity.buy_price,
                market_impact_estimate=0.001,
                urgency='high',
                available_liquidity=opportunity.max_quantity * opportunity.buy_price
            )
            
            buy_step = {
                'action': 'buy',
                'exchange': opportunity.buy_exchange,
                'quantity': quantity,
                'price': opportunity.buy_price,
                'strategy': buy_strategy,
                'estimated_time': 30,  # secondi
                'status': 'pending'
            }
            
            # Step 2: Trasferimento (se necessario)
            transfer_step = {
                'action': 'transfer',
                'from_exchange': opportunity.buy_exchange,
                'to_exchange': opportunity.sell_exchange,
                'quantity': quantity,
                'estimated_time': 600,  # 10 minuti
                'estimated_fee': opportunity.estimated_fees * 0.3,
                'status': 'pending'
            }
            
            # Step 3: Ordine di vendita
            sell_strategy = self.order_router.optimize_execution_strategy(
                order_size=quantity * opportunity.sell_price,
                market_impact_estimate=0.001,
                urgency='high',
                available_liquidity=opportunity.max_quantity * opportunity.sell_price
            )
            
            sell_step = {
                'action': 'sell',
                'exchange': opportunity.sell_exchange,
                'quantity': quantity,
                'price': opportunity.sell_price,
                'strategy': sell_strategy,
                'estimated_time': 30,
                'status': 'pending'
            }
            
            execution_plan['execution_steps'] = [buy_step, transfer_step, sell_step]
            execution_plan['total_estimated_time'] = sum(step['estimated_time'] for step in execution_plan['execution_steps'])
            
            # Simula esecuzione
            for step in execution_plan['execution_steps']:
                step['status'] = 'completed'
                step['actual_time'] = step['estimated_time'] * np.random.uniform(0.8, 1.2)
                step['slippage'] = np.random.uniform(0.0001, 0.001)
            
            execution_plan['status'] = 'completed'
            execution_plan['actual_profit'] = execution_plan['estimated_profit'] * np.random.uniform(0.8, 1.1)
            
            # Aggiungi alla storia
            self.execution_history.append({
                'timestamp': datetime.now(),
                'pair': 'BTC/USDT',  # Esempio
                'execution_plan': execution_plan,
                'success': True
            })
            
            logger.info(f"Arbitrage executed: {execution_plan['actual_profit']:.2f} USDT profit")
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def monitor_cross_exchange_balances(self) -> Dict[str, Dict[str, float]]:
        """Monitora bilanciamenti tra exchange"""
        try:
            # Simula bilanci
            balances = {}
            
            for exchange in self.exchanges:
                balances[exchange] = {
                    'BTC': np.random.uniform(0.1, 2.0),
                    'ETH': np.random.uniform(1.0, 20.0),
                    'SOL': np.random.uniform(10.0, 200.0),
                    'AVAX': np.random.uniform(5.0, 100.0),
                    'KAS': np.random.uniform(1000.0, 50000.0),
                    'USDT': np.random.uniform(1000.0, 20000.0)
                }
            
            return balances
            
        except Exception as e:
            logger.error(f"Error monitoring balances: {e}")
            return {}
    
    def calculate_optimal_rebalancing(self, current_balances: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Calcola ribilanciamento ottimale tra exchange"""
        try:
            rebalancing_actions = []
            
            # Calcola totale per asset
            total_by_asset = {}
            for exchange, balances in current_balances.items():
                for asset, amount in balances.items():
                    total_by_asset[asset] = total_by_asset.get(asset, 0) + amount
            
            # Target: distribuzione equa tra exchange principali
            target_exchanges = ['binance', 'kucoin']
            target_ratio = 1.0 / len(target_exchanges)
            
            for asset, total_amount in total_by_asset.items():
                if asset == 'USDT':  # Skip stablecoin
                    continue
                
                target_per_exchange = total_amount * target_ratio
                
                for exchange in target_exchanges:
                    current_amount = current_balances.get(exchange, {}).get(asset, 0)
                    difference = target_per_exchange - current_amount
                    
                    if abs(difference) > total_amount * 0.1:  # >10% deviation
                        if difference > 0:
                            # Trova exchange con surplus
                            for source_exchange, balances in current_balances.items():
                                if source_exchange != exchange:
                                    source_amount = balances.get(asset, 0)
                                    if source_amount > target_per_exchange:
                                        transfer_amount = min(difference, source_amount - target_per_exchange)
                                        
                                        rebalancing_actions.append({
                                            'action': 'transfer',
                                            'asset': asset,
                                            'amount': transfer_amount,
                                            'from_exchange': source_exchange,
                                            'to_exchange': exchange,
                                            'priority': 'medium',
                                            'estimated_cost': self.exchanges[source_exchange].withdrawal_fee * transfer_amount
                                        })
                                        break
            
            return rebalancing_actions
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing: {e}")
            return []
    
    def get_arbitrage_dashboard(self) -> Dict[str, Any]:
        """Ottieni dashboard con stato arbitraggio"""
        try:
            # Trova opportunità per tutte le coppie
            all_opportunities = []
            for pair in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT']:
                opportunities = self.find_arbitrage_opportunities(pair)
                for opp in opportunities:
                    all_opportunities.append({
                        'pair': pair,
                        'opportunity': opp
                    })
            
            # Statistiche esecuzioni
            recent_executions = [exec for exec in self.execution_history 
                               if (datetime.now() - exec['timestamp']).total_seconds() < 86400]
            
            total_profit = sum(exec['execution_plan'].get('actual_profit', 0) 
                             for exec in recent_executions if exec['success'])
            
            success_rate = (sum(1 for exec in recent_executions if exec['success']) / 
                          max(1, len(recent_executions))) * 100
            
            return {
                'current_opportunities': len(all_opportunities),
                'best_opportunity': all_opportunities[0] if all_opportunities else None,
                'daily_profit': total_profit,
                'success_rate': success_rate,
                'recent_executions': len(recent_executions),
                'exchange_status': {name: 'active' for name in self.exchanges.keys()},
                'risk_utilization': {
                    'position_usage': 45.2,  # %
                    'exposure_usage': 23.8   # %
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating arbitrage dashboard: {e}")
            return {}