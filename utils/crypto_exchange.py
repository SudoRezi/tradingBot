import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import random
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class CryptoExchange:
    """Simulated crypto exchange for demonstration - Replace with real exchange API"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Simulated market data
        self.current_prices = {
            'KAS/USDT': 0.15,
            'BTC/USDT': 43000.0,
            'ETH/USDT': 2600.0
        }
        
        # Price volatility for realistic simulation
        self.volatility = {
            'KAS/USDT': 0.05,  # 5% volatility
            'BTC/USDT': 0.03,  # 3% volatility
            'ETH/USDT': 0.04   # 4% volatility
        }
        
        # For generating realistic OHLCV data
        self.price_history = {}
        self._initialize_price_history()
        
        logger.info("Crypto Exchange API initialized (Simulation Mode)")
    
    def _initialize_price_history(self):
        """Initialize price history for simulation"""
        try:
            for pair in self.current_prices:
                base_price = self.current_prices[pair]
                history = []
                
                # Generate 500 historical candles
                for i in range(500):
                    timestamp = datetime.now() - timedelta(minutes=500-i)
                    
                    # Generate realistic OHLCV data
                    if i == 0:
                        open_price = base_price
                    else:
                        open_price = history[-1]['close']
                    
                    # Add some randomness
                    change = np.random.normal(0, self.volatility[pair] * base_price / 24)  # Hourly volatility
                    close_price = max(open_price + change, base_price * 0.1)  # Prevent negative prices
                    
                    # Generate high and low
                    high_low_range = abs(change) * random.uniform(1, 3)
                    high_price = max(open_price, close_price) + random.uniform(0, high_low_range)
                    low_price = min(open_price, close_price) - random.uniform(0, high_low_range)
                    
                    # Generate volume
                    volume = random.uniform(1000, 10000) * (1 + abs(change/base_price) * 10)
                    
                    candle = {
                        'timestamp': timestamp,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume
                    }
                    
                    history.append(candle)
                
                self.price_history[pair] = history
                
        except Exception as e:
            logger.error(f"Error initializing price history: {e}")
    
    def get_ohlcv(self, pair: str, timeframe: str = '1m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a trading pair"""
        try:
            if pair not in self.price_history:
                logger.warning(f"No data available for {pair}")
                return None
            
            # Update price history with new candle
            self._update_price_data(pair)
            
            # Get requested number of candles
            recent_data = self.price_history[pair][-limit:]
            
            # Convert to DataFrame
            data = []
            for candle in recent_data:
                data.append({
                    'timestamp': candle['timestamp'],
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': candle['volume']
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {pair}: {e}")
            return None
    
    def _update_price_data(self, pair: str):
        """Update price data with new candle"""
        try:
            if pair not in self.price_history:
                return
            
            last_candle = self.price_history[pair][-1]
            now = datetime.now()
            
            # Only add new candle if enough time has passed
            if (now - last_candle['timestamp']).seconds < 30:  # 30 seconds for demo
                return
            
            # Generate new candle
            base_price = self.current_prices[pair]
            open_price = last_candle['close']
            
            # Market trend simulation (slight upward bias for demo)
            trend = random.uniform(-0.5, 1.0)  # Slight bullish bias
            change = np.random.normal(trend * 0.001, self.volatility[pair] * base_price / 1440)  # Per minute volatility
            
            close_price = max(open_price + change, base_price * 0.1)
            
            # Generate high and low
            high_low_range = abs(change) * random.uniform(1, 2)
            high_price = max(open_price, close_price) + random.uniform(0, high_low_range)
            low_price = min(open_price, close_price) - random.uniform(0, high_low_range)
            
            # Generate volume
            volume = random.uniform(1000, 5000) * (1 + abs(change/base_price) * 20)
            
            new_candle = {
                'timestamp': now,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
            # Add new candle and maintain history size
            self.price_history[pair].append(new_candle)
            if len(self.price_history[pair]) > 1000:
                self.price_history[pair] = self.price_history[pair][-1000:]
            
            # Update current price
            self.current_prices[pair] = close_price
            
        except Exception as e:
            logger.error(f"Error updating price data for {pair}: {e}")
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a trading pair"""
        try:
            self._update_price_data(pair)
            return self.current_prices.get(pair)
        except Exception as e:
            logger.error(f"Error getting current price for {pair}: {e}")
            return None
    
    def get_order_book(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get order book for a trading pair"""
        try:
            current_price = self.get_current_price(pair)
            if current_price is None:
                return None
            
            # Generate simulated order book
            bids = []
            asks = []
            
            # Generate bids (buy orders)
            for i in range(10):
                price = current_price * (1 - (i + 1) * 0.001)  # 0.1% increments down
                size = random.uniform(0.1, 10.0)
                bids.append([price, size])
            
            # Generate asks (sell orders)
            for i in range(10):
                price = current_price * (1 + (i + 1) * 0.001)  # 0.1% increments up
                size = random.uniform(0.1, 10.0)
                asks.append([price, size])
            
            return {
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting order book for {pair}: {e}")
            return None
    
    def get_recent_trades(self, pair: str, limit: int = 50) -> Optional[pd.DataFrame]:
        """Get recent trades for a trading pair"""
        try:
            current_price = self.get_current_price(pair)
            if current_price is None:
                return None
            
            trades = []
            for i in range(limit):
                timestamp = datetime.now() - timedelta(seconds=i*10)
                price = current_price * random.uniform(0.999, 1.001)  # Small price variation
                size = random.uniform(0.1, 5.0)
                side = random.choice(['buy', 'sell'])
                
                trades.append({
                    'timestamp': timestamp,
                    'price': price,
                    'size': size,
                    'side': side
                })
            
            df = pd.DataFrame(trades)
            return df.sort_values('timestamp', ascending=False)
            
        except Exception as e:
            logger.error(f"Error getting recent trades for {pair}: {e}")
            return None
    
    def place_order(self, pair: str, side: str, order_type: str, 
                   amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Place a trading order (simulated)"""
        try:
            # Simulate order placement
            order_id = f"order_{int(time.time())}_{random.randint(1000, 9999)}"
            current_price = self.get_current_price(pair)
            
            if price is None:
                price = current_price
            
            # Simulate order execution (always successful in demo)
            order = {
                'id': order_id,
                'pair': pair,
                'side': side.lower(),
                'type': order_type.lower(),
                'amount': amount,
                'price': price,
                'status': 'filled',
                'timestamp': datetime.now(),
                'filled': amount,
                'cost': amount * price
            }
            
            logger.info(f"Order placed (simulated): {side} {amount} {pair} @ ${price:.6f}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {
                'id': None,
                'status': 'error',
                'error': str(e)
            }
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance (simulated)"""
        try:
            # Return simulated balance
            return {
                'USDT': 10000.0,  # Starting with 10k USDT
                'KAS': 0.0,
                'BTC': 0.0,
                'ETH': 0.0
            }
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}
    
    def get_trading_fees(self) -> Dict[str, float]:
        """Get trading fees"""
        return {
            'maker': 0.001,  # 0.1%
            'taker': 0.001   # 0.1%
        }
