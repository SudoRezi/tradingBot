#!/usr/bin/env python3
"""
Advanced Order System
Sistema avanzato per ordini complessi e gestione rischio
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
import time

class OrderType(Enum):
    """Tipi di ordine"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    BRACKET = "bracket"

class OrderStatus(Enum):
    """Status ordine"""
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"

class OrderSide(Enum):
    """Lato ordine"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Ordine di trading avanzato"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trigger_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    
    # Parametri avanzati
    iceberg_visible_qty: Optional[float] = None
    trailing_amount: Optional[float] = None
    trailing_percent: Optional[float] = None
    parent_order_id: Optional[str] = None
    child_orders: List[str] = None
    
    # Parametri TWAP/VWAP
    execution_timeframe: Optional[int] = None  # minuti
    slice_count: Optional[int] = None
    slice_interval: Optional[int] = None  # secondi
    
    # Metadati
    exchange: str = "simulator"
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.child_orders is None:
            self.child_orders = []

class AdvancedOrderSystem:
    """Sistema avanzato gestione ordini"""
    
    def __init__(self, db_path="data/orders.db"):
        self.db_path = db_path
        self.active_orders = {}
        self.order_monitors = {}
        self.running = True
        
        # Setup database
        self._init_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Avvia monitor ordini
        self._start_order_monitor()
    
    def _init_database(self):
        """Inizializza database ordini"""
        
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    stop_price REAL,
                    trigger_price REAL,
                    time_in_force TEXT DEFAULT 'GTC',
                    status TEXT NOT NULL,
                    filled_quantity REAL DEFAULT 0.0,
                    avg_fill_price REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    exchange TEXT DEFAULT 'simulator',
                    portfolio_id TEXT,
                    strategy_id TEXT,
                    notes TEXT,
                    advanced_params TEXT  -- JSON per parametri avanzati
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    exchange_fill_id TEXT,
                    fees REAL DEFAULT 0.0,
                    FOREIGN KEY (order_id) REFERENCES orders (id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_status 
                ON orders(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_symbol 
                ON orders(symbol)
            """)
    
    def create_market_order(self, symbol: str, side: OrderSide, quantity: float, **kwargs) -> str:
        """Crea ordine market"""
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            **kwargs
        )
        
        return self._submit_order(order)
    
    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float, price: float, **kwargs) -> str:
        """Crea ordine limit"""
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            **kwargs
        )
        
        return self._submit_order(order)
    
    def create_stop_loss_order(self, symbol: str, side: OrderSide, quantity: float, stop_price: float, **kwargs) -> str:
        """Crea ordine stop loss"""
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LOSS,
            quantity=quantity,
            stop_price=stop_price,
            **kwargs
        )
        
        return self._submit_order(order)
    
    def create_trailing_stop_order(self, symbol: str, side: OrderSide, quantity: float, 
                                 trailing_amount: float = None, trailing_percent: float = None, **kwargs) -> str:
        """Crea ordine trailing stop"""
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.TRAILING_STOP,
            quantity=quantity,
            trailing_amount=trailing_amount,
            trailing_percent=trailing_percent,
            **kwargs
        )
        
        return self._submit_order(order)
    
    def create_bracket_order(self, symbol: str, side: OrderSide, quantity: float, entry_price: float,
                           take_profit_price: float, stop_loss_price: float, **kwargs) -> str:
        """Crea ordine bracket (entry + take profit + stop loss)"""
        
        # Ordine principale
        main_order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=entry_price,
            **kwargs
        )
        
        # Ordine take profit
        tp_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        take_profit_order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=tp_side,
            order_type=OrderType.TAKE_PROFIT,
            quantity=quantity,
            trigger_price=take_profit_price,
            parent_order_id=main_order.id,
            **kwargs
        )
        
        # Ordine stop loss
        stop_loss_order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=tp_side,
            order_type=OrderType.STOP_LOSS,
            quantity=quantity,
            stop_price=stop_loss_price,
            parent_order_id=main_order.id,
            **kwargs
        )
        
        # Collega ordini
        main_order.child_orders = [take_profit_order.id, stop_loss_order.id]
        
        # Submetti ordini
        main_order_id = self._submit_order(main_order)
        self._submit_order(take_profit_order)
        self._submit_order(stop_loss_order)
        
        return main_order_id
    
    def create_iceberg_order(self, symbol: str, side: OrderSide, quantity: float, price: float,
                           visible_quantity: float, **kwargs) -> str:
        """Crea ordine iceberg"""
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.ICEBERG,
            quantity=quantity,
            price=price,
            iceberg_visible_qty=visible_quantity,
            **kwargs
        )
        
        return self._submit_order(order)
    
    def create_twap_order(self, symbol: str, side: OrderSide, quantity: float, 
                         timeframe_minutes: int, slice_count: int = None, **kwargs) -> str:
        """Crea ordine TWAP (Time-Weighted Average Price)"""
        
        if slice_count is None:
            slice_count = max(5, timeframe_minutes // 5)  # Un slice ogni 5 minuti max
        
        slice_interval = (timeframe_minutes * 60) // slice_count  # secondi
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.TWAP,
            quantity=quantity,
            execution_timeframe=timeframe_minutes,
            slice_count=slice_count,
            slice_interval=slice_interval,
            **kwargs
        )
        
        return self._submit_order(order)
    
    def create_vwap_order(self, symbol: str, side: OrderSide, quantity: float, 
                         timeframe_minutes: int, **kwargs) -> str:
        """Crea ordine VWAP (Volume-Weighted Average Price)"""
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.VWAP,
            quantity=quantity,
            execution_timeframe=timeframe_minutes,
            **kwargs
        )
        
        return self._submit_order(order)
    
    def _submit_order(self, order: Order) -> str:
        """Submetti ordine al sistema"""
        
        try:
            # Salva in database
            self._save_order_to_db(order)
            
            # Aggiungi a ordini attivi se non market
            if order.order_type != OrderType.MARKET:
                self.active_orders[order.id] = order
            
            # Per ordini market, esegui immediatamente
            if order.order_type == OrderType.MARKET:
                self._execute_market_order(order)
            
            self.logger.info(f"Order submitted: {order.id} - {order.symbol} {order.side.value} {order.quantity}")
            
            return order.id
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            raise
    
    def _save_order_to_db(self, order: Order):
        """Salva ordine nel database"""
        
        # Parametri avanzati come JSON
        advanced_params = {
            "iceberg_visible_qty": order.iceberg_visible_qty,
            "trailing_amount": order.trailing_amount,
            "trailing_percent": order.trailing_percent,
            "parent_order_id": order.parent_order_id,
            "child_orders": order.child_orders,
            "execution_timeframe": order.execution_timeframe,
            "slice_count": order.slice_count,
            "slice_interval": order.slice_interval
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders 
                (id, symbol, side, order_type, quantity, price, stop_price, trigger_price,
                 time_in_force, status, filled_quantity, avg_fill_price, created_at, updated_at,
                 expires_at, exchange, portfolio_id, strategy_id, notes, advanced_params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.id, order.symbol, order.side.value, order.order_type.value,
                order.quantity, order.price, order.stop_price, order.trigger_price,
                order.time_in_force, order.status.value, order.filled_quantity,
                order.avg_fill_price, order.created_at.isoformat(), order.updated_at.isoformat(),
                order.expires_at.isoformat() if order.expires_at else None,
                order.exchange, order.portfolio_id, order.strategy_id, order.notes,
                json.dumps(advanced_params)
            ))
    
    def _execute_market_order(self, order: Order):
        """Esegui ordine market immediatamente"""
        
        # Simula esecuzione con prezzo di mercato
        from real_time_data_feeds import get_realtime_manager
        
        realtime_manager = get_realtime_manager()
        market_data = realtime_manager.get_latest_data(order.symbol, order.exchange)
        
        if market_data:
            fill_price = market_data.ask if order.side == OrderSide.BUY else market_data.bid
        else:
            # Fallback con prezzo simulato
            fill_price = 50000 if "BTC" in order.symbol else 3000 if "ETH" in order.symbol else 100
        
        # Registra fill
        self._record_fill(order.id, order.quantity, fill_price)
        
        # Aggiorna status ordine
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.updated_at = datetime.now()
        
        self._save_order_to_db(order)
        
        self.logger.info(f"Market order executed: {order.id} @ ${fill_price:.2f}")
    
    def _record_fill(self, order_id: str, quantity: float, price: float, fees: float = 0.0):
        """Registra fill di un ordine"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO order_fills 
                (order_id, quantity, price, timestamp, fees)
                VALUES (?, ?, ?, ?, ?)
            """, (order_id, quantity, price, datetime.now().isoformat(), fees))
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancella ordine"""
        
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                
                self._save_order_to_db(order)
                del self.active_orders[order_id]
                
                # Se è un bracket order, cancella anche gli ordini figli
                if order.child_orders:
                    for child_id in order.child_orders:
                        self.cancel_order(child_id)
                
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Ottieni ordine per ID"""
        
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Cerca nel database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM orders WHERE id = ?
                """, (order_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_order(row)
                    
        except Exception as e:
            self.logger.error(f"Error getting order: {e}")
        
        return None
    
    def get_orders_by_symbol(self, symbol: str, status: OrderStatus = None) -> List[Order]:
        """Ottieni ordini per simbolo"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if status:
                    cursor = conn.execute("""
                        SELECT * FROM orders WHERE symbol = ? AND status = ?
                        ORDER BY created_at DESC
                    """, (symbol, status.value))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM orders WHERE symbol = ?
                        ORDER BY created_at DESC
                    """, (symbol,))
                
                orders = []
                for row in cursor.fetchall():
                    orders.append(self._row_to_order(row))
                
                return orders
                
        except Exception as e:
            self.logger.error(f"Error getting orders by symbol: {e}")
            return []
    
    def get_active_orders(self) -> List[Order]:
        """Ottieni tutti gli ordini attivi"""
        
        return list(self.active_orders.values())
    
    def _row_to_order(self, row) -> Order:
        """Converte row database in oggetto Order"""
        
        advanced_params = json.loads(row[19]) if row[19] else {}
        
        return Order(
            id=row[0],
            symbol=row[1],
            side=OrderSide(row[2]),
            order_type=OrderType(row[3]),
            quantity=row[4],
            price=row[5],
            stop_price=row[6],
            trigger_price=row[7],
            time_in_force=row[8],
            status=OrderStatus(row[9]),
            filled_quantity=row[10],
            avg_fill_price=row[11],
            created_at=datetime.fromisoformat(row[12]),
            updated_at=datetime.fromisoformat(row[13]),
            expires_at=datetime.fromisoformat(row[14]) if row[14] else None,
            exchange=row[15],
            portfolio_id=row[16],
            strategy_id=row[17],
            notes=row[18],
            **advanced_params
        )
    
    def _generate_order_id(self) -> str:
        """Genera ID ordine unico"""
        
        import uuid
        return f"ORD_{uuid.uuid4().hex[:8].upper()}"
    
    def _start_order_monitor(self):
        """Avvia monitor ordini in background"""
        
        def monitor_loop():
            while self.running:
                try:
                    self._check_pending_orders()
                    time.sleep(1)  # Check ogni secondo
                except Exception as e:
                    self.logger.error(f"Order monitor error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _check_pending_orders(self):
        """Controlla ordini pending per triggers"""
        
        from real_time_data_feeds import get_realtime_manager
        
        realtime_manager = get_realtime_manager()
        
        for order_id, order in list(self.active_orders.items()):
            try:
                if order.status != OrderStatus.ACTIVE:
                    continue
                
                # Ottieni prezzo corrente
                market_data = realtime_manager.get_latest_data(order.symbol, order.exchange)
                
                if not market_data:
                    continue
                
                current_price = market_data.price
                
                # Check trigger conditions
                triggered = False
                
                if order.order_type == OrderType.STOP_LOSS:
                    if order.side == OrderSide.SELL and current_price <= order.stop_price:
                        triggered = True
                    elif order.side == OrderSide.BUY and current_price >= order.stop_price:
                        triggered = True
                
                elif order.order_type == OrderType.TAKE_PROFIT:
                    if order.side == OrderSide.SELL and current_price >= order.trigger_price:
                        triggered = True
                    elif order.side == OrderSide.BUY and current_price <= order.trigger_price:
                        triggered = True
                
                elif order.order_type == OrderType.TRAILING_STOP:
                    self._update_trailing_stop(order, current_price)
                
                elif order.order_type in [OrderType.TWAP, OrderType.VWAP]:
                    self._process_algorithmic_order(order, market_data)
                
                if triggered:
                    self._trigger_order(order, current_price)
                    
            except Exception as e:
                self.logger.error(f"Error checking order {order_id}: {e}")
    
    def _trigger_order(self, order: Order, trigger_price: float):
        """Triggera ordine"""
        
        # Esegui come market order
        fill_price = trigger_price
        
        self._record_fill(order.id, order.quantity, fill_price)
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.updated_at = datetime.now()
        
        self._save_order_to_db(order)
        
        # Rimuovi da ordini attivi
        if order.id in self.active_orders:
            del self.active_orders[order.id]
        
        self.logger.info(f"Order triggered: {order.id} @ ${fill_price:.2f}")
        
        # Se è ordine bracket, cancella altri ordini del gruppo
        if order.parent_order_id or order.child_orders:
            self._handle_bracket_order_fill(order)
    
    def _handle_bracket_order_fill(self, filled_order: Order):
        """Gestisce fill di ordine bracket"""
        
        # Se è un ordine figlio, cancella gli altri figli
        if filled_order.parent_order_id:
            parent_order = self.get_order(filled_order.parent_order_id)
            if parent_order and parent_order.child_orders:
                for child_id in parent_order.child_orders:
                    if child_id != filled_order.id:
                        self.cancel_order(child_id)
    
    def _update_trailing_stop(self, order: Order, current_price: float):
        """Aggiorna trailing stop"""
        
        if not hasattr(order, '_highest_price'):
            order._highest_price = current_price
            order._lowest_price = current_price
        
        if order.side == OrderSide.SELL:
            # Per vendita, traccia il prezzo più alto
            if current_price > order._highest_price:
                order._highest_price = current_price
                
                # Aggiorna stop price
                if order.trailing_percent:
                    new_stop = current_price * (1 - order.trailing_percent / 100)
                else:
                    new_stop = current_price - order.trailing_amount
                
                if new_stop > order.stop_price:
                    order.stop_price = new_stop
                    self._save_order_to_db(order)
        
        else:  # BUY
            # Per acquisto, traccia il prezzo più basso
            if current_price < order._lowest_price:
                order._lowest_price = current_price
                
                # Aggiorna stop price
                if order.trailing_percent:
                    new_stop = current_price * (1 + order.trailing_percent / 100)
                else:
                    new_stop = current_price + order.trailing_amount
                
                if new_stop < order.stop_price:
                    order.stop_price = new_stop
                    self._save_order_to_db(order)
    
    def _process_algorithmic_order(self, order: Order, market_data):
        """Processa ordini algoritmici (TWAP/VWAP)"""
        
        # Implementazione semplificata - in produzione sarebbe più complessa
        if not hasattr(order, '_last_slice_time'):
            order._last_slice_time = order.created_at
            order._slices_executed = 0
        
        now = datetime.now()
        time_since_last_slice = (now - order._last_slice_time).total_seconds()
        
        if time_since_last_slice >= order.slice_interval:
            # Esegui slice
            slice_quantity = order.quantity / order.slice_count
            remaining_quantity = order.quantity - order.filled_quantity
            
            if remaining_quantity > 0:
                execute_quantity = min(slice_quantity, remaining_quantity)
                
                # Esegui slice come market order
                fill_price = market_data.ask if order.side == OrderSide.BUY else market_data.bid
                
                self._record_fill(order.id, execute_quantity, fill_price)
                
                order.filled_quantity += execute_quantity
                order._slices_executed += 1
                order._last_slice_time = now
                
                # Calcola prezzo medio
                if order.avg_fill_price:
                    total_filled_value = order.avg_fill_price * (order.filled_quantity - execute_quantity)
                    total_filled_value += fill_price * execute_quantity
                    order.avg_fill_price = total_filled_value / order.filled_quantity
                else:
                    order.avg_fill_price = fill_price
                
                # Check se completato
                if order.filled_quantity >= order.quantity or order._slices_executed >= order.slice_count:
                    order.status = OrderStatus.FILLED
                    if order.id in self.active_orders:
                        del self.active_orders[order.id]
                
                order.updated_at = now
                self._save_order_to_db(order)
                
                self.logger.info(f"TWAP/VWAP slice executed: {order.id} - {execute_quantity} @ ${fill_price:.2f}")
    
    def get_order_statistics(self) -> Dict:
        """Ottieni statistiche ordini"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        status,
                        COUNT(*) as count,
                        SUM(quantity) as total_quantity,
                        AVG(avg_fill_price) as avg_price
                    FROM orders 
                    WHERE avg_fill_price IS NOT NULL
                    GROUP BY status
                """)
                
                stats = {}
                for row in cursor.fetchall():
                    stats[row[0]] = {
                        "count": row[1],
                        "total_quantity": row[2],
                        "avg_price": row[3]
                    }
                
                # Aggiungi statistiche generali
                cursor = conn.execute("SELECT COUNT(*) FROM orders")
                total_orders = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM orders WHERE status = 'filled'")
                filled_orders = cursor.fetchone()[0]
                
                stats["summary"] = {
                    "total_orders": total_orders,
                    "filled_orders": filled_orders,
                    "fill_rate": filled_orders / total_orders if total_orders > 0 else 0,
                    "active_orders": len(self.active_orders)
                }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting order statistics: {e}")
            return {}
    
    def stop(self):
        """Ferma sistema ordini"""
        
        self.running = False
        self.logger.info("Advanced order system stopped")

# Singleton instance
advanced_order_system = AdvancedOrderSystem()

def get_order_system() -> AdvancedOrderSystem:
    """Ottieni istanza singleton del sistema ordini"""
    return advanced_order_system

# Test del sistema
def test_order_system():
    """Test sistema ordini avanzato"""
    
    order_system = get_order_system()
    
    # Test ordine market
    market_order_id = order_system.create_market_order("BTC/USDT", OrderSide.BUY, 0.001)
    print(f"Market order created: {market_order_id}")
    
    # Test ordine limit
    limit_order_id = order_system.create_limit_order("ETH/USDT", OrderSide.BUY, 1.0, 3000.0)
    print(f"Limit order created: {limit_order_id}")
    
    # Test ordine stop loss
    stop_order_id = order_system.create_stop_loss_order("BTC/USDT", OrderSide.SELL, 0.001, 45000.0)
    print(f"Stop loss order created: {stop_order_id}")
    
    # Test ordine bracket
    bracket_order_id = order_system.create_bracket_order(
        "ETH/USDT", OrderSide.BUY, 1.0, 2900.0, 3200.0, 2800.0
    )
    print(f"Bracket order created: {bracket_order_id}")
    
    # Test TWAP
    twap_order_id = order_system.create_twap_order("BTC/USDT", OrderSide.BUY, 0.01, 30)
    print(f"TWAP order created: {twap_order_id}")
    
    # Statistiche
    stats = order_system.get_order_statistics()
    print("Order statistics:", stats)
    
    time.sleep(5)
    
    # Cancella ordini test
    order_system.cancel_order(limit_order_id)
    order_system.cancel_order(stop_order_id)
    
    print("Test completed")

if __name__ == "__main__":
    test_order_system()