#!/usr/bin/env python3
"""
Real-Time Data Feeds Manager
Sistema avanzato per feed dati in tempo reale multi-exchange
"""

import asyncio
import websockets
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import sqlite3
import logging

@dataclass
class MarketData:
    """Struttura dati di mercato"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    exchange: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None

class RealTimeDataManager:
    """Manager per feed dati real-time multi-exchange"""
    
    def __init__(self, db_path="data/realtime_feeds.db"):
        self.db_path = db_path
        self.active_connections = {}
        self.subscribers = {}
        self.data_cache = {}
        self.running = False
        
        # Setup database
        self._init_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _init_database(self):
        """Inizializza database per cache dati"""
        
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    bid REAL,
                    ask REAL,
                    high_24h REAL,
                    low_24h REAL,
                    change_24h REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON market_data(symbol, timestamp)
            """)
    
    def subscribe_to_symbol(self, symbol: str, exchange: str, callback: Callable):
        """Sottoscrivi aggiornamenti per simbolo"""
        
        key = f"{exchange}:{symbol}"
        
        if key not in self.subscribers:
            self.subscribers[key] = []
        
        self.subscribers[key].append(callback)
        
        # Avvia connessione se non esiste
        if key not in self.active_connections:
            self._start_connection(symbol, exchange)
    
    def unsubscribe_from_symbol(self, symbol: str, exchange: str, callback: Callable):
        """Cancella sottoscrizione"""
        
        key = f"{exchange}:{symbol}"
        
        if key in self.subscribers and callback in self.subscribers[key]:
            self.subscribers[key].remove(callback)
            
            # Se nessun subscriber, chiudi connessione
            if not self.subscribers[key]:
                self._stop_connection(symbol, exchange)
    
    def _start_connection(self, symbol: str, exchange: str):
        """Avvia connessione WebSocket per exchange"""
        
        key = f"{exchange}:{symbol}"
        
        if exchange.lower() == "binance":
            thread = threading.Thread(
                target=self._binance_websocket,
                args=(symbol, key),
                daemon=True
            )
        elif exchange.lower() == "coinbase":
            thread = threading.Thread(
                target=self._coinbase_websocket,
                args=(symbol, key),
                daemon=True
            )
        elif exchange.lower() == "kraken":
            thread = threading.Thread(
                target=self._kraken_websocket,
                args=(symbol, key),
                daemon=True
            )
        else:
            # Fallback con dati simulati
            thread = threading.Thread(
                target=self._simulated_feed,
                args=(symbol, exchange, key),
                daemon=True
            )
        
        thread.start()
        self.active_connections[key] = thread
    
    def _stop_connection(self, symbol: str, exchange: str):
        """Ferma connessione WebSocket"""
        
        key = f"{exchange}:{symbol}"
        
        if key in self.active_connections:
            # WebSocket threads si fermano automaticamente quando non ci sono subscribers
            del self.active_connections[key]
    
    async def _binance_websocket(self, symbol: str, key: str):
        """WebSocket Binance"""
        
        # Converte simbolo in formato Binance
        binance_symbol = symbol.replace("/", "").lower()
        uri = f"wss://stream.binance.com:9443/ws/{binance_symbol}@ticker"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.logger.info(f"Connected to Binance WebSocket for {symbol}")
                
                while key in self.subscribers and self.subscribers[key]:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        data = json.loads(message)
                        
                        market_data = MarketData(
                            symbol=symbol,
                            price=float(data.get("c", 0)),  # Current price
                            volume=float(data.get("v", 0)), # Volume
                            timestamp=datetime.now(),
                            exchange="Binance",
                            bid=float(data.get("b", 0)),    # Best bid
                            ask=float(data.get("a", 0)),    # Best ask
                            high_24h=float(data.get("h", 0)),
                            low_24h=float(data.get("l", 0)),
                            change_24h=float(data.get("P", 0))
                        )
                        
                        self._notify_subscribers(key, market_data)
                        self._cache_data(market_data)
                        
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout on Binance WebSocket for {symbol}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error in Binance WebSocket: {e}")
                        break
                        
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance WebSocket: {e}")
            # Fallback a dati simulati
            await self._simulated_feed(symbol, "Binance", key)
    
    def _binance_websocket(self, symbol: str, key: str):
        """Wrapper sincrono per Binance WebSocket"""
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._binance_websocket_async(symbol, key))
        except Exception as e:
            self.logger.error(f"Binance WebSocket error: {e}")
            self._simulated_feed(symbol, "Binance", key)
    
    async def _binance_websocket_async(self, symbol: str, key: str):
        """WebSocket Binance asincrono"""
        
        binance_symbol = symbol.replace("/", "").lower()
        uri = f"wss://stream.binance.com:9443/ws/{binance_symbol}@ticker"
        
        try:
            async with websockets.connect(uri) as websocket:
                while key in self.subscribers and self.subscribers[key]:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    market_data = MarketData(
                        symbol=symbol,
                        price=float(data.get("c", 0)),
                        volume=float(data.get("v", 0)),
                        timestamp=datetime.now(),
                        exchange="Binance",
                        bid=float(data.get("b", 0)),
                        ask=float(data.get("a", 0)),
                        high_24h=float(data.get("h", 0)),
                        low_24h=float(data.get("l", 0)),
                        change_24h=float(data.get("P", 0))
                    )
                    
                    self._notify_subscribers(key, market_data)
                    self._cache_data(market_data)
                    
        except Exception as e:
            self.logger.error(f"Binance WebSocket error: {e}")
    
    def _coinbase_websocket(self, symbol: str, key: str):
        """WebSocket Coinbase Pro"""
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._coinbase_websocket_async(symbol, key))
        except Exception as e:
            self.logger.error(f"Coinbase WebSocket error: {e}")
            self._simulated_feed(symbol, "Coinbase", key)
    
    async def _coinbase_websocket_async(self, symbol: str, key: str):
        """WebSocket Coinbase asincrono"""
        
        uri = "wss://ws-feed.exchange.coinbase.com"
        
        # Messaggio di sottoscrizione
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [symbol],
            "channels": ["ticker"]
        }
        
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps(subscribe_msg))
                
                while key in self.subscribers and self.subscribers[key]:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data.get("type") == "ticker":
                        market_data = MarketData(
                            symbol=symbol,
                            price=float(data.get("price", 0)),
                            volume=float(data.get("volume_24h", 0)),
                            timestamp=datetime.now(),
                            exchange="Coinbase",
                            bid=float(data.get("best_bid", 0)),
                            ask=float(data.get("best_ask", 0)),
                            high_24h=float(data.get("high_24h", 0)),
                            low_24h=float(data.get("low_24h", 0))
                        )
                        
                        self._notify_subscribers(key, market_data)
                        self._cache_data(market_data)
                        
        except Exception as e:
            self.logger.error(f"Coinbase WebSocket error: {e}")
    
    def _kraken_websocket(self, symbol: str, key: str):
        """WebSocket Kraken"""
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._kraken_websocket_async(symbol, key))
        except Exception as e:
            self.logger.error(f"Kraken WebSocket error: {e}")
            self._simulated_feed(symbol, "Kraken", key)
    
    async def _kraken_websocket_async(self, symbol: str, key: str):
        """WebSocket Kraken asincrono"""
        
        uri = "wss://ws.kraken.com"
        
        # Converte simbolo per Kraken
        kraken_symbol = symbol.replace("/", "")
        
        subscribe_msg = {
            "event": "subscribe",
            "pair": [kraken_symbol],
            "subscription": {"name": "ticker"}
        }
        
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps(subscribe_msg))
                
                while key in self.subscribers and self.subscribers[key]:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if isinstance(data, list) and len(data) > 1:
                        ticker_data = data[1]
                        
                        if isinstance(ticker_data, dict):
                            market_data = MarketData(
                                symbol=symbol,
                                price=float(ticker_data.get("c", [0])[0]),
                                volume=float(ticker_data.get("v", [0])[0]),
                                timestamp=datetime.now(),
                                exchange="Kraken",
                                bid=float(ticker_data.get("b", [0])[0]),
                                ask=float(ticker_data.get("a", [0])[0]),
                                high_24h=float(ticker_data.get("h", [0])[0]),
                                low_24h=float(ticker_data.get("l", [0])[0])
                            )
                            
                            self._notify_subscribers(key, market_data)
                            self._cache_data(market_data)
                            
        except Exception as e:
            self.logger.error(f"Kraken WebSocket error: {e}")
    
    def _simulated_feed(self, symbol: str, exchange: str, key: str):
        """Feed dati simulati per testing"""
        
        import random
        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        
        while key in self.subscribers and self.subscribers[key]:
            try:
                # Simula movimento prezzo realistico
                price_change = random.uniform(-0.02, 0.02)  # ±2%
                current_price = base_price * (1 + price_change)
                
                volume = random.uniform(100, 10000)
                
                market_data = MarketData(
                    symbol=symbol,
                    price=current_price,
                    volume=volume,
                    timestamp=datetime.now(),
                    exchange=exchange,
                    bid=current_price * 0.999,
                    ask=current_price * 1.001,
                    high_24h=current_price * 1.05,
                    low_24h=current_price * 0.95,
                    change_24h=random.uniform(-5, 5)
                )
                
                self._notify_subscribers(key, market_data)
                self._cache_data(market_data)
                
                time.sleep(1)  # Aggiorna ogni secondo
                
            except Exception as e:
                self.logger.error(f"Simulated feed error: {e}")
                break
    
    def _notify_subscribers(self, key: str, market_data: MarketData):
        """Notifica tutti i subscribers"""
        
        if key in self.subscribers:
            for callback in self.subscribers[key]:
                try:
                    callback(market_data)
                except Exception as e:
                    self.logger.error(f"Subscriber callback error: {e}")
    
    def _cache_data(self, market_data: MarketData):
        """Cache dati nel database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO market_data 
                    (symbol, price, volume, timestamp, exchange, bid, ask, high_24h, low_24h, change_24h)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_data.symbol,
                    market_data.price,
                    market_data.volume,
                    market_data.timestamp.isoformat(),
                    market_data.exchange,
                    market_data.bid,
                    market_data.ask,
                    market_data.high_24h,
                    market_data.low_24h,
                    market_data.change_24h
                ))
                
                # Mantieni solo ultimi 1000 record per simbolo
                conn.execute("""
                    DELETE FROM market_data 
                    WHERE symbol = ? AND id NOT IN (
                        SELECT id FROM market_data 
                        WHERE symbol = ? 
                        ORDER BY created_at DESC 
                        LIMIT 1000
                    )
                """, (market_data.symbol, market_data.symbol))
                
        except Exception as e:
            self.logger.error(f"Cache error: {e}")
    
    def get_latest_data(self, symbol: str, exchange: str = None) -> Optional[MarketData]:
        """Ottieni ultimi dati per simbolo"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if exchange:
                    cursor = conn.execute("""
                        SELECT * FROM market_data 
                        WHERE symbol = ? AND exchange = ?
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """, (symbol, exchange))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM market_data 
                        WHERE symbol = ?
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """, (symbol,))
                
                row = cursor.fetchone()
                
                if row:
                    return MarketData(
                        symbol=row[1],
                        price=row[2],
                        volume=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        exchange=row[5],
                        bid=row[6],
                        ask=row[7],
                        high_24h=row[8],
                        low_24h=row[9],
                        change_24h=row[10]
                    )
                    
        except Exception as e:
            self.logger.error(f"Get latest data error: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, hours: int = 24) -> List[MarketData]:
        """Ottieni dati storici"""
        
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND created_at >= ?
                    ORDER BY created_at ASC
                """, (symbol, cutoff_time.isoformat()))
                
                data = []
                for row in cursor.fetchall():
                    data.append(MarketData(
                        symbol=row[1],
                        price=row[2],
                        volume=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        exchange=row[5],
                        bid=row[6],
                        ask=row[7],
                        high_24h=row[8],
                        low_24h=row[9],
                        change_24h=row[10]
                    ))
                
                return data
                
        except Exception as e:
            self.logger.error(f"Get historical data error: {e}")
            return []
    
    def stop_all_feeds(self):
        """Ferma tutti i feed"""
        
        self.subscribers.clear()
        self.active_connections.clear()
        self.logger.info("All real-time feeds stopped")

# Singleton instance
realtime_data_manager = RealTimeDataManager()

def get_realtime_manager() -> RealTimeDataManager:
    """Ottieni istanza singleton del manager"""
    return realtime_data_manager

# Test del sistema
def test_realtime_feeds():
    """Test sistema feed real-time"""
    
    manager = get_realtime_manager()
    
    def print_data(data: MarketData):
        print(f"{data.exchange} - {data.symbol}: ${data.price:.2f} (Vol: {data.volume:.0f})")
    
    # Test sottoscrizione
    manager.subscribe_to_symbol("BTC/USDT", "Binance", print_data)
    manager.subscribe_to_symbol("ETH/USDT", "Binance", print_data)
    
    print("Real-time feeds started. Press Ctrl+C to stop...")
    
    try:
        time.sleep(30)  # Test per 30 secondi
    except KeyboardInterrupt:
        pass
    
    manager.stop_all_feeds()
    print("Test completed")

if __name__ == "__main__":
    test_realtime_feeds()