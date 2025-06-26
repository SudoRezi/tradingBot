"""
Speed Optimization Engine - Massima Velocità Trading
Ottimizzazioni per trading ad alta frequenza e reattività istantanea
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
import psutil
import gc
from collections import deque
import weakref
import cProfile
import functools

@dataclass
class PerformanceMetrics:
    latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    network_latency_ms: float

class HighSpeedDataProcessor:
    """Processore dati ottimizzato per velocità massima"""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.price_buffer = deque(maxlen=buffer_size)
        self.volume_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        
        # Pre-allocate arrays for calculations
        self.calc_array = np.zeros(buffer_size)
        self.temp_array = np.zeros(buffer_size)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=psutil.cpu_count())
        
    def add_data_point(self, price: float, volume: float, timestamp: float):
        """Aggiunge punto dati con velocità ottimizzata"""
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        self.timestamp_buffer.append(timestamp)
    
    @functools.lru_cache(maxsize=1000)
    def calculate_indicators_cached(self, prices_hash: int, period: int) -> Dict[str, float]:
        """Calcola indicatori con caching per velocità"""
        if len(self.price_buffer) < period:
            return {}
        
        # Converti a numpy array per calcoli vettorizzati
        prices = np.array(list(self.price_buffer)[-period:])
        volumes = np.array(list(self.volume_buffer)[-period:])
        
        # Calcoli vettorizzati ultra-veloci
        sma = np.mean(prices)
        ema = self._calculate_ema_fast(prices)
        rsi = self._calculate_rsi_fast(prices)
        vwap = np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else sma
        
        return {
            'sma': sma,
            'ema': ema,
            'rsi': rsi,
            'vwap': vwap,
            'volatility': np.std(prices),
            'momentum': (prices[-1] - prices[0]) / prices[0] * 100
        }
    
    def _calculate_ema_fast(self, prices: np.ndarray, alpha: float = 0.1) -> float:
        """EMA ottimizzata per velocità"""
        if len(prices) == 0:
            return 0.0
        
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _calculate_rsi_fast(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI ottimizzato per velocità"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class FastOrderExecutor:
    """Esecutore ordini ottimizzato per latenza minima"""
    
    def __init__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=0.5),  # 500ms timeout
            connector=aiohttp.TCPConnector(
                limit=100,
                limit_per_host=50,
                keepalive_timeout=300,
                enable_cleanup_closed=True
            )
        )
        
        # Pre-compiled order templates
        self.order_templates = {
            'market_buy': self._create_market_buy_template,
            'market_sell': self._create_market_sell_template,
            'limit_buy': self._create_limit_buy_template,
            'limit_sell': self._create_limit_sell_template
        }
        
        # Order queue for batch processing
        self.order_queue = asyncio.Queue(maxsize=1000)
        self.is_processing = False
        
    def _create_market_buy_template(self, symbol: str, quantity: float) -> Dict[str, Any]:
        """Template pre-compilato per ordini market buy"""
        return {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'MARKET',
            'quantity': quantity,
            'timestamp': int(time.time() * 1000)
        }
    
    def _create_market_sell_template(self, symbol: str, quantity: float) -> Dict[str, Any]:
        """Template pre-compilato per ordini market sell"""
        return {
            'symbol': symbol,
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': quantity,
            'timestamp': int(time.time() * 1000)
        }
    
    def _create_limit_buy_template(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Template pre-compilato per ordini limit buy"""
        return {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'LIMIT',
            'quantity': quantity,
            'price': price,
            'timeInForce': 'GTC',
            'timestamp': int(time.time() * 1000)
        }
    
    def _create_limit_sell_template(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Template pre-compilato per ordini limit sell"""
        return {
            'symbol': symbol,
            'side': 'SELL',
            'type': 'LIMIT',
            'quantity': quantity,
            'price': price,
            'timeInForce': 'GTC',
            'timestamp': int(time.time() * 1000)
        }
    
    async def execute_order_fast(self, order_type: str, symbol: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Esegue ordine con latenza minima"""
        start_time = time.perf_counter()
        
        # Crea ordine da template pre-compilato
        if order_type in ['market_buy', 'market_sell']:
            order_data = self.order_templates[order_type](symbol, quantity)
        else:
            order_data = self.order_templates[order_type](symbol, quantity, price)
        
        # Simula esecuzione ultra-veloce
        execution_time = time.perf_counter() - start_time
        
        return {
            'order_id': f"fast_{int(time.time() * 1000000)}",
            'status': 'FILLED',
            'execution_time_ms': execution_time * 1000,
            'order_data': order_data,
            'timestamp': time.time()
        }
    
    async def batch_execute_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Esegue batch di ordini in parallelo"""
        tasks = []
        for order in orders:
            task = self.execute_order_fast(
                order['type'],
                order['symbol'],
                order['quantity'],
                order.get('price')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

class MemoryOptimizer:
    """Ottimizzatore memoria per performance massime"""
    
    def __init__(self):
        self.memory_pools = {}
        self.object_cache = weakref.WeakValueDictionary()
        
    def create_memory_pool(self, name: str, size: int, item_size: int):
        """Crea pool di memoria pre-allocata"""
        self.memory_pools[name] = {
            'data': bytearray(size * item_size),
            'size': size,
            'item_size': item_size,
            'used': 0,
            'free_list': list(range(size))
        }
    
    def allocate_from_pool(self, pool_name: str) -> Optional[memoryview]:
        """Alloca memoria dal pool pre-allocato"""
        pool = self.memory_pools.get(pool_name)
        if not pool or not pool['free_list']:
            return None
        
        index = pool['free_list'].pop()
        start = index * pool['item_size']
        end = start + pool['item_size']
        pool['used'] += 1
        
        return memoryview(pool['data'][start:end])
    
    def deallocate_to_pool(self, pool_name: str, index: int):
        """Rilascia memoria al pool"""
        pool = self.memory_pools.get(pool_name)
        if pool:
            pool['free_list'].append(index)
            pool['used'] -= 1
    
    def optimize_garbage_collection(self):
        """Ottimizza garbage collection per trading"""
        # Disabilita GC durante trading critico
        gc.disable()
        
        # Trigger manual GC quando safe
        def safe_gc_trigger():
            gc.collect()
            gc.enable()
        
        return safe_gc_trigger

class NetworkOptimizer:
    """Ottimizzatore rete per latenza minima"""
    
    def __init__(self):
        self.connection_pools = {}
        self.dns_cache = {}
        
    async def create_optimized_session(self, exchange_url: str) -> aiohttp.ClientSession:
        """Crea sessione HTTP ottimizzata"""
        
        # Configurazione ottimizzata per latenza
        connector = aiohttp.TCPConnector(
            limit=200,                    # Max connections
            limit_per_host=100,           # Max per host
            ttl_dns_cache=300,           # DNS cache TTL
            use_dns_cache=True,          # Enable DNS cache
            keepalive_timeout=300,       # Keep connections alive
            enable_cleanup_closed=True,  # Cleanup closed connections
            force_close=False,           # Reuse connections
            local_addr=None              # Let system choose optimal interface
        )
        
        # Timeout ottimizzati per trading
        timeout = aiohttp.ClientTimeout(
            total=1.0,         # Total timeout
            connect=0.3,       # Connection timeout
            sock_read=0.5,     # Socket read timeout
            sock_connect=0.2   # Socket connect timeout
        )
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'HighSpeedTradingBot/1.0',
                'Connection': 'keep-alive',
                'Accept-Encoding': 'gzip, deflate'
            }
        )
        
        return session
    
    def optimize_tcp_settings(self):
        """Restituisce impostazioni TCP ottimali"""
        return {
            'tcp_settings': {
                'TCP_NODELAY': 'Disable Nagle algorithm for lower latency',
                'SO_REUSEADDR': 'Allow rapid connection reuse',
                'SO_KEEPALIVE': 'Enable keepalive packets',
                'TCP_USER_TIMEOUT': 'Set custom timeout for failed connections'
            },
            'buffer_sizes': {
                'SO_RCVBUF': 'Increase receive buffer size',
                'SO_SNDBUF': 'Increase send buffer size'
            },
            'priority_settings': {
                'SO_PRIORITY': 'Set high priority for trading packets',
                'IP_TOS': 'Type of Service for low latency'
            }
        }

class CPUOptimizer:
    """Ottimizzatore CPU per performance massime"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.trading_cores = list(range(self.cpu_count // 2))  # Prima metà CPU per trading
        self.analysis_cores = list(range(self.cpu_count // 2, self.cpu_count))  # Seconda metà per analisi
    
    def set_cpu_affinity(self, process_type: str = 'trading'):
        """Imposta affinità CPU per processo"""
        import os
        
        if process_type == 'trading':
            cores = self.trading_cores
        else:
            cores = self.analysis_cores
        
        try:
            os.sched_setaffinity(0, cores)
            return f"CPU affinity set to cores: {cores}"
        except:
            return "CPU affinity setting not supported on this system"
    
    def get_cpu_optimization_settings(self):
        """Restituisce impostazioni ottimizzazione CPU"""
        return {
            'process_priority': {
                'trading_process': 'HIGH_PRIORITY_CLASS',
                'analysis_process': 'NORMAL_PRIORITY_CLASS',
                'monitoring_process': 'BELOW_NORMAL_PRIORITY_CLASS'
            },
            'thread_affinity': {
                'order_execution': self.trading_cores[:2],
                'market_data': self.trading_cores[2:4] if len(self.trading_cores) > 2 else self.trading_cores,
                'strategy_calculation': self.analysis_cores[:2],
                'data_analysis': self.analysis_cores[2:] if len(self.analysis_cores) > 2 else self.analysis_cores
            },
            'performance_counters': {
                'cache_misses': 'Monitor L1/L2/L3 cache performance',
                'context_switches': 'Minimize context switching overhead',
                'interrupts': 'Monitor interrupt handling efficiency'
            }
        }

class PerformanceMonitor:
    """Monitor performance real-time"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.start_time = time.time()
        
    def measure_execution_time(self, func: Callable) -> Callable:
        """Decorator per misurare tempo esecuzione"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            
            execution_time = (end - start) * 1000  # Convert to milliseconds
            self.metrics_history.append({
                'function': func.__name__,
                'execution_time_ms': execution_time,
                'timestamp': time.time()
            })
            
            return result
        return wrapper
    
    async def measure_async_execution(self, coro):
        """Misura tempo esecuzione funzioni async"""
        start = time.perf_counter()
        result = await coro
        end = time.perf_counter()
        
        return result, (end - start) * 1000
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche performance"""
        if not self.metrics_history:
            return {}
        
        execution_times = [m['execution_time_ms'] for m in self.metrics_history]
        
        return {
            'avg_execution_time_ms': np.mean(execution_times),
            'min_execution_time_ms': np.min(execution_times),
            'max_execution_time_ms': np.max(execution_times),
            'p95_execution_time_ms': np.percentile(execution_times, 95),
            'p99_execution_time_ms': np.percentile(execution_times, 99),
            'total_operations': len(self.metrics_history),
            'operations_per_second': len(self.metrics_history) / (time.time() - self.start_time)
        }

class SpeedOptimizationEngine:
    """Engine principale per ottimizzazioni velocità"""
    
    def __init__(self):
        self.data_processor = HighSpeedDataProcessor()
        self.order_executor = FastOrderExecutor()
        self.memory_optimizer = MemoryOptimizer()
        self.network_optimizer = NetworkOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        # Inizializza ottimizzazioni
        self._initialize_optimizations()
    
    def _initialize_optimizations(self):
        """Inizializza tutte le ottimizzazioni"""
        
        # Crea memory pools
        self.memory_optimizer.create_memory_pool('price_data', 10000, 64)
        self.memory_optimizer.create_memory_pool('order_data', 1000, 256)
        
        # Imposta CPU affinity
        cpu_result = self.cpu_optimizer.set_cpu_affinity('trading')
        
        # Ottimizza garbage collection
        self.gc_trigger = self.memory_optimizer.optimize_garbage_collection()
    
    @performance_monitor.measure_execution_time
    def process_market_data_fast(self, price: float, volume: float) -> Dict[str, float]:
        """Processa dati mercato con velocità ottimizzata"""
        timestamp = time.time()
        self.data_processor.add_data_point(price, volume, timestamp)
        
        # Hash per caching
        prices_hash = hash(tuple(list(self.data_processor.price_buffer)[-20:]))
        
        return self.data_processor.calculate_indicators_cached(prices_hash, 20)
    
    async def execute_strategy_fast(self, strategy_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue strategia con latenza minima"""
        execution_results = []
        
        for signal in strategy_signals.get('signals', []):
            result = await self.order_executor.execute_order_fast(
                signal['type'],
                signal['symbol'],
                signal['quantity'],
                signal.get('price')
            )
            execution_results.append(result)
        
        return {
            'total_orders': len(execution_results),
            'avg_execution_time': np.mean([r['execution_time_ms'] for r in execution_results]),
            'results': execution_results
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Genera report ottimizzazioni"""
        performance_stats = self.performance_monitor.get_performance_stats()
        cpu_settings = self.cpu_optimizer.get_cpu_optimization_settings()
        tcp_settings = self.network_optimizer.optimize_tcp_settings()
        
        return {
            'performance_metrics': performance_stats,
            'cpu_optimizations': cpu_settings,
            'network_optimizations': tcp_settings,
            'memory_pools': {
                name: {
                    'size': pool['size'],
                    'used': pool['used'],
                    'utilization': pool['used'] / pool['size'] * 100
                }
                for name, pool in self.memory_optimizer.memory_pools.items()
            },
            'optimization_status': 'ACTIVE',
            'recommendations': self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Genera raccomandazioni ottimizzazione"""
        recommendations = []
        
        # Analizza utilizzo memoria
        for name, pool in self.memory_optimizer.memory_pools.items():
            utilization = pool['used'] / pool['size'] * 100
            if utilization > 80:
                recommendations.append(f"Consider increasing {name} pool size")
            elif utilization < 20:
                recommendations.append(f"Consider reducing {name} pool size")
        
        # Analizza performance
        stats = self.performance_monitor.get_performance_stats()
        if stats and stats.get('avg_execution_time_ms', 0) > 10:
            recommendations.append("Average execution time high - consider CPU optimization")
        
        if stats and stats.get('operations_per_second', 0) < 1000:
            recommendations.append("Low throughput - consider parallelization")
        
        return recommendations

# Esempio utilizzo
async def test_speed_optimizations():
    """Test ottimizzazioni velocità"""
    engine = SpeedOptimizationEngine()
    
    # Test processing veloce
    indicators = engine.process_market_data_fast(50000.0, 1.5)
    print(f"Indicators calculated: {indicators}")
    
    # Test esecuzione strategia
    strategy_signals = {
        'signals': [
            {'type': 'market_buy', 'symbol': 'BTCUSDT', 'quantity': 0.001},
            {'type': 'limit_sell', 'symbol': 'BTCUSDT', 'quantity': 0.001, 'price': 51000.0}
        ]
    }
    
    execution_result = await engine.execute_strategy_fast(strategy_signals)
    print(f"Strategy executed: {execution_result}")
    
    # Report ottimizzazioni
    report = engine.get_optimization_report()
    print(f"Optimization report: {report}")

if __name__ == "__main__":
    asyncio.run(test_speed_optimizations())