import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import random
from typing import Dict, Any, Optional
from utils.crypto_exchange import CryptoExchange

logger = logging.getLogger(__name__)

class DataManager:
    """Manages market data fetching and caching"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = CryptoExchange(api_key, api_secret)
        
        # Data cache
        self.data_cache = {}
        self.cache_duration = 30  # seconds
        self.last_update = {}
        
        logger.info("Data Manager initialized")
    
    def get_market_data(self, pair: str, timeframe: str = '1m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market data for a trading pair"""
        try:
            # Check cache first
            cache_key = f"{pair}_{timeframe}_{limit}"
            
            if (cache_key in self.data_cache and 
                cache_key in self.last_update and
                (datetime.now() - self.last_update[cache_key]).seconds < self.cache_duration):
                return self.data_cache[cache_key]
            
            # Fetch new data
            data = self.exchange.get_ohlcv(pair, timeframe, limit)
            
            if data is not None and not data.empty:
                # Cache the data
                self.data_cache[cache_key] = data
                self.last_update[cache_key] = datetime.now()
                
                logger.debug(f"Market data updated for {pair}: {len(data)} candles")
                return data
            else:
                logger.warning(f"No market data received for {pair}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting market data for {pair}: {e}")
            return None
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a trading pair"""
        try:
            return self.exchange.get_current_price(pair)
        except Exception as e:
            logger.error(f"Error getting current price for {pair}: {e}")
            return None
    
    def get_order_book(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get order book for a trading pair"""
        try:
            return self.exchange.get_order_book(pair)
        except Exception as e:
            logger.error(f"Error getting order book for {pair}: {e}")
            return None
    
    def get_recent_trades(self, pair: str, limit: int = 50) -> Optional[pd.DataFrame]:
        """Get recent trades for a trading pair"""
        try:
            return self.exchange.get_recent_trades(pair, limit)
        except Exception as e:
            logger.error(f"Error getting recent trades for {pair}: {e}")
            return None
    
    def clear_cache(self):
        """Clear data cache"""
        try:
            self.data_cache.clear()
            self.last_update.clear()
            logger.info("Data cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            'cached_pairs': list(self.data_cache.keys()),
            'cache_size': len(self.data_cache),
            'last_updates': {k: v.isoformat() for k, v in self.last_update.items()}
        }
