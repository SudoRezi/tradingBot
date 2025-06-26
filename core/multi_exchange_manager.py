"""
Multi-Exchange Manager - Gestisce più exchange contemporaneamente
Supporta API keys multiple e selezione trading pairs personalizzabile
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """Configurazione per singolo exchange"""
    name: str
    api_key: str
    api_secret: str
    sandbox: bool = True
    enabled: bool = True
    trading_pairs: List[str] = None
    fee_rate: float = 0.001  # 0.1% default
    min_order_size: float = 10.0
    max_leverage: float = 10.0

@dataclass
class TradingPair:
    """Definizione trading pair"""
    symbol: str
    base_asset: str
    quote_asset: str
    enabled: bool = True
    min_notional: float = 10.0
    price_precision: int = 8
    quantity_precision: int = 8

@dataclass
class ExchangeBalance:
    """Balance per exchange"""
    exchange: str
    asset: str
    available: float
    locked: float
    total: float
    usd_value: float

@dataclass
class ExchangeOrder:
    """Ordine su exchange"""
    exchange: str
    symbol: str
    order_id: str
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit'
    amount: float
    price: float
    status: str
    timestamp: datetime
    fee: float = 0.0

class ExchangeConnector:
    """Connettore base per exchange"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.name = config.name
        self.is_connected = False
        self.last_update = None
        
        # Mock data per demo (in produzione useresti veri connector)
        self.mock_balances = {
            'USDT': 10000.0,
            'BTC': 0.1,
            'ETH': 2.0,
            'KAS': 1000.0,
            'SOL': 10.0
        }
        
        self.mock_prices = {
            'BTC/USDT': 45000.0,
            'ETH/USDT': 3000.0,
            'KAS/USDT': 0.15,
            'SOL/USDT': 100.0,
            'AVAX/USDT': 35.0
        }
    
    async def connect(self) -> bool:
        """Connette all'exchange"""
        try:
            # Simulazione connessione
            await asyncio.sleep(0.1)
            self.is_connected = True
            self.last_update = datetime.now()
            logger.info(f"Connected to {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    async def get_balances(self) -> Dict[str, float]:
        """Ottiene balance dall'exchange"""
        if not self.is_connected:
            await self.connect()
        
        # In produzione faresti chiamata API reale
        return self.mock_balances.copy()
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Ottiene ticker per symbol"""
        if not self.is_connected:
            await self.connect()
        
        base_price = self.mock_prices.get(symbol, 100.0)
        # Simula variazioni di prezzo
        change_pct = np.random.uniform(-0.05, 0.05)
        current_price = base_price * (1 + change_pct)
        
        return {
            'symbol': symbol,
            'price': current_price,
            'bid': current_price * 0.999,
            'ask': current_price * 1.001,
            'volume': np.random.uniform(1000, 100000),
            'change_24h': change_pct,
            'timestamp': datetime.now()
        }
    
    async def place_order(self, symbol: str, side: str, amount: float, 
                         price: Optional[float] = None, order_type: str = 'market') -> ExchangeOrder:
        """Piazza ordine"""
        if not self.is_connected:
            await self.connect()
        
        # Simula piazzamento ordine
        order_id = f"{self.name}_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        if order_type == 'market':
            ticker = await self.get_ticker(symbol)
            execution_price = ticker['price']
        else:
            execution_price = price
        
        # Simula fee
        fee = amount * execution_price * self.config.fee_rate
        
        order = ExchangeOrder(
            exchange=self.name,
            symbol=symbol,
            order_id=order_id,
            side=side,
            type=order_type,
            amount=amount,
            price=execution_price,
            status='filled',  # Simula riempimento immediato
            timestamp=datetime.now(),
            fee=fee
        )
        
        logger.info(f"Order placed on {self.name}: {side} {amount} {symbol} @ {execution_price}")
        return order
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Ottiene order book"""
        ticker = await self.get_ticker(symbol)
        mid_price = ticker['price']
        
        # Simula order book
        bids = []
        asks = []
        
        for i in range(limit):
            bid_price = mid_price * (1 - (i + 1) * 0.001)
            ask_price = mid_price * (1 + (i + 1) * 0.001)
            
            bids.append([bid_price, np.random.uniform(0.1, 10.0)])
            asks.append([ask_price, np.random.uniform(0.1, 10.0)])
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now()
        }

class MultiExchangeManager:
    """Manager principale per gestire più exchange"""
    
    def __init__(self):
        self.exchanges: Dict[str, ExchangeConnector] = {}
        self.trading_pairs: Dict[str, TradingPair] = {}
        self.active_orders: List[ExchangeOrder] = []
        self.balance_cache = {}
        self.price_cache = {}
        self.last_update = {}
        
        # Configurazione arbitraggio
        self.arbitrage_enabled = True
        self.min_arbitrage_profit = 0.005  # 0.5% minimo
        self.max_position_size = 1000.0  # USD
        
        # Default trading pairs
        self._init_default_pairs()
    
    def _init_default_pairs(self):
        """Inizializza trading pairs di default"""
        default_pairs = [
            TradingPair('BTC/USDT', 'BTC', 'USDT', True, 10.0, 2, 6),
            TradingPair('ETH/USDT', 'ETH', 'USDT', True, 10.0, 2, 5),
            TradingPair('KAS/USDT', 'KAS', 'USDT', True, 5.0, 6, 2),
            TradingPair('SOL/USDT', 'SOL', 'USDT', True, 10.0, 3, 4),
            TradingPair('AVAX/USDT', 'AVAX', 'USDT', True, 10.0, 3, 4)
        ]
        
        for pair in default_pairs:
            self.trading_pairs[pair.symbol] = pair
    
    def add_exchange(self, config: ExchangeConfig) -> bool:
        """Aggiunge nuovo exchange"""
        try:
            if config.trading_pairs is None:
                config.trading_pairs = list(self.trading_pairs.keys())
            
            connector = ExchangeConnector(config)
            self.exchanges[config.name] = connector
            
            logger.info(f"Added exchange {config.name} with {len(config.trading_pairs)} trading pairs")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add exchange {config.name}: {e}")
            return False
    
    def remove_exchange(self, exchange_name: str) -> bool:
        """Rimuove exchange"""
        if exchange_name in self.exchanges:
            del self.exchanges[exchange_name]
            logger.info(f"Removed exchange {exchange_name}")
            return True
        return False
    
    async def connect_all_exchanges(self) -> Dict[str, bool]:
        """Connette a tutti gli exchange configurati"""
        results = {}
        
        tasks = []
        for name, exchange in self.exchanges.items():
            if exchange.config.enabled:
                tasks.append(self._connect_exchange(name, exchange))
        
        if tasks:
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (name, _) in enumerate(self.exchanges.items()):
                if i < len(results_list):
                    results[name] = results_list[i] if not isinstance(results_list[i], Exception) else False
        
        connected_count = sum(1 for v in results.values() if v)
        logger.info(f"Connected to {connected_count}/{len(results)} exchanges")
        
        return results
    
    async def _connect_exchange(self, name: str, exchange: ExchangeConnector) -> bool:
        """Connette singolo exchange"""
        try:
            return await exchange.connect()
        except Exception as e:
            logger.error(f"Error connecting to {name}: {e}")
            return False
    
    async def get_aggregated_balances(self) -> Dict[str, ExchangeBalance]:
        """Ottiene balance aggregati da tutti gli exchange"""
        all_balances = {}
        
        for exchange_name, exchange in self.exchanges.items():
            if not exchange.config.enabled or not exchange.is_connected:
                continue
            
            try:
                balances = await exchange.get_balances()
                
                for asset, amount in balances.items():
                    if amount > 0:
                        # Calcola valore USD (mock)
                        usd_value = amount * self._get_asset_usd_price(asset)
                        
                        balance_key = f"{exchange_name}_{asset}"
                        all_balances[balance_key] = ExchangeBalance(
                            exchange=exchange_name,
                            asset=asset,
                            available=amount,
                            locked=0.0,  # Semplificato
                            total=amount,
                            usd_value=usd_value
                        )
            
            except Exception as e:
                logger.error(f"Error getting balances from {exchange_name}: {e}")
        
        return all_balances
    
    def _get_asset_usd_price(self, asset: str) -> float:
        """Ottiene prezzo USD per asset (mock)"""
        usd_prices = {
            'USDT': 1.0,
            'USDC': 1.0,
            'BTC': 45000.0,
            'ETH': 3000.0,
            'KAS': 0.15,
            'SOL': 100.0,
            'AVAX': 35.0
        }
        return usd_prices.get(asset, 1.0)
    
    async def get_best_prices(self, symbol: str) -> Dict[str, Any]:
        """Ottiene migliori prezzi bid/ask da tutti gli exchange"""
        best_bid = {'exchange': None, 'price': 0, 'volume': 0}
        best_ask = {'exchange': None, 'price': float('inf'), 'volume': 0}
        
        all_prices = {}
        
        for exchange_name, exchange in self.exchanges.items():
            if not exchange.config.enabled or not exchange.is_connected:
                continue
            
            if symbol not in exchange.config.trading_pairs:
                continue
            
            try:
                ticker = await exchange.get_ticker(symbol)
                all_prices[exchange_name] = ticker
                
                # Aggiorna best bid
                if ticker['bid'] > best_bid['price']:
                    best_bid = {
                        'exchange': exchange_name,
                        'price': ticker['bid'],
                        'volume': ticker['volume']
                    }
                
                # Aggiorna best ask
                if ticker['ask'] < best_ask['price']:
                    best_ask = {
                        'exchange': exchange_name,
                        'price': ticker['ask'],
                        'volume': ticker['volume']
                    }
            
            except Exception as e:
                logger.error(f"Error getting ticker from {exchange_name}: {e}")
        
        return {
            'symbol': symbol,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask['price'] - best_bid['price'] if best_ask['price'] != float('inf') else 0,
            'all_prices': all_prices,
            'timestamp': datetime.now()
        }
    
    async def find_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """Trova opportunità di arbitraggio"""
        opportunities = []
        
        for symbol in self.trading_pairs.keys():
            if not self.trading_pairs[symbol].enabled:
                continue
            
            try:
                prices = await self.get_best_prices(symbol)
                
                if (prices['best_bid']['exchange'] and 
                    prices['best_ask']['exchange'] and 
                    prices['best_bid']['exchange'] != prices['best_ask']['exchange']):
                    
                    # Calcola profitto potenziale
                    buy_price = prices['best_ask']['price']
                    sell_price = prices['best_bid']['price']
                    
                    if sell_price > buy_price:
                        profit_pct = (sell_price - buy_price) / buy_price
                        
                        if profit_pct >= self.min_arbitrage_profit:
                            # Calcola fees
                            buy_exchange = self.exchanges[prices['best_ask']['exchange']]
                            sell_exchange = self.exchanges[prices['best_bid']['exchange']]
                            
                            total_fees = buy_exchange.config.fee_rate + sell_exchange.config.fee_rate
                            net_profit = profit_pct - total_fees
                            
                            if net_profit > 0:
                                opportunity = {
                                    'symbol': symbol,
                                    'buy_exchange': prices['best_ask']['exchange'],
                                    'sell_exchange': prices['best_bid']['exchange'],
                                    'buy_price': buy_price,
                                    'sell_price': sell_price,
                                    'gross_profit_pct': profit_pct,
                                    'fees_pct': total_fees,
                                    'net_profit_pct': net_profit,
                                    'max_volume': min(
                                        prices['best_ask']['volume'],
                                        prices['best_bid']['volume']
                                    ),
                                    'timestamp': datetime.now()
                                }
                                opportunities.append(opportunity)
            
            except Exception as e:
                logger.error(f"Error checking arbitrage for {symbol}: {e}")
        
        # Ordina per profitto netto
        opportunities.sort(key=lambda x: x['net_profit_pct'], reverse=True)
        
        return opportunities
    
    async def execute_arbitrage(self, opportunity: Dict[str, Any], 
                              amount_usd: float) -> Dict[str, Any]:
        """Esegue arbitraggio"""
        try:
            symbol = opportunity['symbol']
            buy_exchange = self.exchanges[opportunity['buy_exchange']]
            sell_exchange = self.exchanges[opportunity['sell_exchange']]
            
            # Calcola quantità da tradare
            buy_price = opportunity['buy_price']
            quantity = amount_usd / buy_price
            
            # Esegui ordini simultanei
            buy_task = buy_exchange.place_order(symbol, 'buy', quantity, None, 'market')
            sell_task = sell_exchange.place_order(symbol, 'sell', quantity, None, 'market')
            
            buy_order, sell_order = await asyncio.gather(buy_task, sell_task)
            
            # Calcola profitto realizzato
            total_cost = buy_order.amount * buy_order.price + buy_order.fee
            total_revenue = sell_order.amount * sell_order.price - sell_order.fee
            realized_profit = total_revenue - total_cost
            
            result = {
                'success': True,
                'symbol': symbol,
                'buy_order': asdict(buy_order),
                'sell_order': asdict(sell_order),
                'realized_profit': realized_profit,
                'profit_pct': realized_profit / total_cost if total_cost > 0 else 0,
                'timestamp': datetime.now()
            }
            
            self.active_orders.extend([buy_order, sell_order])
            logger.info(f"Arbitrage executed: {realized_profit:.2f} profit on {symbol}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def update_trading_pairs(self, pairs: List[Dict[str, Any]]) -> bool:
        """Aggiorna configurazione trading pairs"""
        try:
            for pair_data in pairs:
                symbol = pair_data['symbol']
                
                pair = TradingPair(
                    symbol=symbol,
                    base_asset=pair_data.get('base_asset', symbol.split('/')[0]),
                    quote_asset=pair_data.get('quote_asset', symbol.split('/')[1]),
                    enabled=pair_data.get('enabled', True),
                    min_notional=pair_data.get('min_notional', 10.0),
                    price_precision=pair_data.get('price_precision', 8),
                    quantity_precision=pair_data.get('quantity_precision', 8)
                )
                
                self.trading_pairs[symbol] = pair
            
            # Aggiorna trading pairs negli exchange
            for exchange in self.exchanges.values():
                enabled_pairs = [
                    symbol for symbol, pair in self.trading_pairs.items()
                    if pair.enabled
                ]
                exchange.config.trading_pairs = enabled_pairs
            
            logger.info(f"Updated {len(pairs)} trading pairs")
            return True
        
        except Exception as e:
            logger.error(f"Error updating trading pairs: {e}")
            return False
    
    def get_exchange_status(self) -> Dict[str, Any]:
        """Ottiene stato di tutti gli exchange"""
        status = {}
        
        for name, exchange in self.exchanges.items():
            status[name] = {
                'name': name,
                'connected': exchange.is_connected,
                'enabled': exchange.config.enabled,
                'trading_pairs_count': len(exchange.config.trading_pairs),
                'fee_rate': exchange.config.fee_rate,
                'max_leverage': exchange.config.max_leverage,
                'last_update': exchange.last_update.isoformat() if exchange.last_update else None
            }
        
        return status
    
    def get_trading_pairs_config(self) -> List[Dict[str, Any]]:
        """Ottiene configurazione trading pairs"""
        return [asdict(pair) for pair in self.trading_pairs.values()]
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Ottiene riassunto portfolio multi-exchange"""
        balances = await self.get_aggregated_balances()
        
        # Aggrega per asset
        asset_totals = defaultdict(lambda: {'total_amount': 0, 'total_usd': 0, 'exchanges': []})
        
        for balance in balances.values():
            asset = balance.asset
            asset_totals[asset]['total_amount'] += balance.total
            asset_totals[asset]['total_usd'] += balance.usd_value
            asset_totals[asset]['exchanges'].append({
                'exchange': balance.exchange,
                'amount': balance.total,
                'usd_value': balance.usd_value
            })
        
        total_portfolio_usd = sum(data['total_usd'] for data in asset_totals.values())
        
        return {
            'total_value_usd': total_portfolio_usd,
            'asset_breakdown': dict(asset_totals),
            'connected_exchanges': len([ex for ex in self.exchanges.values() if ex.is_connected]),
            'total_exchanges': len(self.exchanges),
            'active_trading_pairs': len([pair for pair in self.trading_pairs.values() if pair.enabled]),
            'last_update': datetime.now().isoformat()
        }