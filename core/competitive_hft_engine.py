"""
Competitive High-Frequency Trading Engine
Sistema di trading ad alta frequenza per competere con altri bot in tempo reale
"""

import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger('competitive_hft')

@dataclass
class MarketMicrostructure:
    """Analisi microstruttura del mercato per identificare bot competitors"""
    symbol: str
    bid_ask_spread: float
    order_book_depth: Dict[str, float]
    trade_velocity: float
    bot_signatures: List[str]
    liquidity_zones: List[Dict[str, Any]]
    price_momentum: float
    volume_imbalance: float

@dataclass
class BotBehaviorProfile:
    """Profilo comportamentale di bot competitor identificato"""
    bot_id: str
    trading_pattern: str  # "scalping", "arbitrage", "momentum", "market_making"
    reaction_time_ms: float
    position_sizes: List[float]
    entry_signals: List[str]
    exit_patterns: List[str]
    aggression_level: float  # 0-1
    success_rate: float
    last_seen: datetime

class OrderBookAnalyzer:
    """Analizza order book per identificare pattern di bot competitors"""
    
    def __init__(self):
        self.historical_snapshots = []
        self.bot_signatures = {}
        self.competitive_opportunities = []
    
    def analyze_orderbook_dynamics(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza dinamiche dell'order book per identificare bot activity"""
        
        # Simula analisi order book real-time
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        # Identifica pattern tipici di bot
        bot_patterns = self._detect_bot_patterns(bids, asks)
        
        # Calcola metriche di competitività
        competitiveness_score = self._calculate_competitiveness(bids, asks)
        
        # Identifica opportunità di front-running etico
        front_run_opportunities = self._identify_front_running_opportunities(bids, asks)
        
        return {
            'bot_activity_detected': len(bot_patterns) > 0,
            'detected_patterns': bot_patterns,
            'competitiveness_score': competitiveness_score,
            'front_run_opportunities': front_run_opportunities,
            'optimal_entry_price': self._calculate_optimal_entry(bids, asks),
            'speed_advantage_window_ms': self._estimate_speed_window(),
            'recommended_action': self._generate_competitive_action(bot_patterns)
        }
    
    def _detect_bot_patterns(self, bids: List, asks: List) -> List[Dict[str, Any]]:
        """Identifica pattern tipici di trading bot"""
        patterns = []
        
        # Pattern 1: Ordini a prezzi round number (tipico di bot)
        round_number_orders = []
        for bid in bids[:10]:  # Top 10 bids
            price = float(bid[0])
            if price == round(price, 2):  # Prezzo "pulito"
                round_number_orders.append({
                    'type': 'round_number_bid',
                    'price': price,
                    'size': float(bid[1]),
                    'bot_probability': 0.7
                })
        
        # Pattern 2: Ordini identici ripetuti (market making bot)
        size_frequency = {}
        for ask in asks[:10]:
            size = float(ask[1])
            size_frequency[size] = size_frequency.get(size, 0) + 1
        
        repeated_sizes = [size for size, freq in size_frequency.items() if freq > 2]
        if repeated_sizes:
            patterns.append({
                'type': 'market_making_pattern',
                'repeated_sizes': repeated_sizes,
                'bot_probability': 0.8
            })
        
        # Pattern 3: Spread molto stretto (HFT bot)
        if bids and asks:
            spread = float(asks[0][0]) - float(bids[0][0])
            spread_percentage = (spread / float(bids[0][0])) * 100
            
            if spread_percentage < 0.05:  # Spread < 0.05%
                patterns.append({
                    'type': 'hft_tight_spread',
                    'spread_percentage': spread_percentage,
                    'bot_probability': 0.9
                })
        
        return patterns
    
    def _calculate_competitiveness(self, bids: List, asks: List) -> float:
        """Calcola score di competitività del mercato (0-1)"""
        if not bids or not asks:
            return 0.0
        
        # Fattori di competitività
        spread = float(asks[0][0]) - float(bids[0][0])
        spread_score = max(0, 1 - (spread / float(bids[0][0]) * 1000))  # Spread più stretto = più competitivo
        
        # Volume concentration nei top levels
        total_bid_volume = sum(float(bid[1]) for bid in bids[:5])
        top_bid_volume = float(bids[0][1]) if bids else 0
        volume_concentration = top_bid_volume / total_bid_volume if total_bid_volume > 0 else 0
        
        # Order book depth
        depth_score = min(1.0, len(bids) / 20)  # Più ordini = più competitivo
        
        competitiveness = (spread_score * 0.4 + volume_concentration * 0.3 + depth_score * 0.3)
        return min(1.0, competitiveness)
    
    def _identify_front_running_opportunities(self, bids: List, asks: List) -> List[Dict[str, Any]]:
        """Identifica opportunità di anticipazione etica (non front-running illegale)"""
        opportunities = []
        
        if not bids or not asks:
            return opportunities
        
        # Opportunità 1: Large order detection nei livelli profondi
        for i, bid in enumerate(bids[1:6], 1):  # Livelli 2-6
            size = float(bid[1])
            if size > 10000:  # Ordine grande
                opportunities.append({
                    'type': 'large_bid_anticipation',
                    'level': i,
                    'price': float(bid[0]),
                    'size': size,
                    'action': 'consider_buy_before_level',
                    'time_sensitivity': 'high'
                })
        
        # Opportunità 2: Support/Resistance level approach
        if len(bids) >= 3 and len(asks) >= 3:
            bid_prices = [float(bid[0]) for bid in bids[:3]]
            ask_prices = [float(ask[0]) for ask in asks[:3]]
            
            # Se il prezzo si avvicina a un livello di supporto forte
            support_strength = sum(float(bid[1]) for bid in bids[:3])
            if support_strength > 5000:  # Support forte
                opportunities.append({
                    'type': 'support_level_bounce',
                    'support_price': bid_prices[0],
                    'strength': support_strength,
                    'action': 'prepare_buy_on_bounce',
                    'time_sensitivity': 'medium'
                })
        
        return opportunities
    
    def _calculate_optimal_entry(self, bids: List, asks: List) -> Optional[float]:
        """Calcola prezzo di entrata ottimale per competere con altri bot"""
        if not bids or not asks:
            return None
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        
        # Per competere, ci posizioniamo appena meglio del miglior bid/ask
        tick_size = 0.01  # Minimum price increment
        
        # Per buy order: slightly above best bid
        optimal_buy = best_bid + tick_size
        
        # Per sell order: slightly below best ask  
        optimal_sell = best_ask - tick_size
        
        # Verifica che ci sia ancora profitto dopo il miglioramento
        if (optimal_sell - optimal_buy) > (tick_size * 2):
            return {'buy': optimal_buy, 'sell': optimal_sell}
        else:
            return None
    
    def _estimate_speed_window(self) -> float:
        """Stima finestra temporale per vantaggio velocità (ms)"""
        # Simula latenza network e execution
        base_latency = 50  # 50ms base latency
        competition_factor = np.random.uniform(0.5, 1.5)  # Fattore competizione
        
        return base_latency * competition_factor
    
    def _generate_competitive_action(self, bot_patterns: List[Dict]) -> str:
        """Genera azione competitiva basata sui pattern rilevati"""
        if not bot_patterns:
            return "monitor"
        
        # Se rilevati market maker bot
        if any(p['type'] == 'market_making_pattern' for p in bot_patterns):
            return "aggressive_take_liquidity"
        
        # Se rilevati HFT bot
        if any(p['type'] == 'hft_tight_spread' for p in bot_patterns):
            return "speed_competition_mode"
        
        # Default: monitoring attivo
        return "active_monitoring"

class SpeedOptimizer:
    """Ottimizza velocità di esecuzione per competere con altri bot"""
    
    def __init__(self):
        self.execution_times = []
        self.network_latency = []
        self.optimization_settings = {
            'order_batching': True,
            'async_execution': True,
            'predictive_positioning': True,
            'micro_second_timing': True
        }
    
    def optimize_execution_speed(self, trade_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Ottimizza velocità di esecuzione per battere la competizione"""
        
        # Calcola timing ottimale
        optimal_timing = self._calculate_optimal_timing(trade_signal)
        
        # Prepara ordini per esecuzione rapida
        optimized_orders = self._prepare_speed_optimized_orders(trade_signal)
        
        # Configura parametri per bassa latenza
        speed_config = self._configure_low_latency_settings()
        
        return {
            'execution_timing': optimal_timing,
            'optimized_orders': optimized_orders,
            'speed_configuration': speed_config,
            'estimated_execution_time_ms': self._estimate_execution_time(),
            'competitive_advantage_ms': self._calculate_speed_advantage()
        }
    
    def _calculate_optimal_timing(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Calcola timing ottimale per l'esecuzione"""
        
        # Timing basato su volatilità del mercato
        volatility = signal.get('volatility', 0.02)
        urgency_multiplier = 1 + volatility  # Più volatilità = più urgenza
        
        # Timing basato su competition level
        competition_level = signal.get('competition_level', 0.5)
        competition_urgency = 1 + (competition_level * 0.5)
        
        optimal_delay_ms = max(10, 100 / (urgency_multiplier * competition_urgency))
        
        return {
            'optimal_delay_ms': optimal_delay_ms,
            'execution_window_ms': optimal_delay_ms * 2,
            'retry_interval_ms': optimal_delay_ms / 2,
            'max_retries': 3
        }
    
    def _prepare_speed_optimized_orders(self, signal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepara ordini ottimizzati per velocità"""
        orders = []
        
        # Ordine principale
        main_order = {
            'type': 'limit',
            'side': signal.get('action', 'buy'),
            'symbol': signal.get('symbol', 'BTCUSDT'),
            'quantity': signal.get('quantity', 0.001),
            'price': signal.get('price'),
            'time_in_force': 'IOC',  # Immediate or Cancel per velocità
            'priority': 'high'
        }
        orders.append(main_order)
        
        # Ordini di backup per garantire esecuzione
        backup_prices = self._generate_backup_prices(signal.get('price'), signal.get('action'))
        for i, backup_price in enumerate(backup_prices):
            backup_order = main_order.copy()
            backup_order['price'] = backup_price
            backup_order['priority'] = 'medium'
            backup_order['delay_ms'] = (i + 1) * 50  # Delay incrementale
            orders.append(backup_order)
        
        return orders
    
    def _generate_backup_prices(self, main_price: float, action: str) -> List[float]:
        """Genera prezzi di backup per garantire esecuzione"""
        if not main_price:
            return []
        
        tick_size = 0.01
        backup_prices = []
        
        if action == 'buy':
            # Per buy: prezzi leggermente più alti come backup
            for i in range(1, 4):
                backup_prices.append(main_price + (tick_size * i))
        else:
            # Per sell: prezzi leggermente più bassi come backup
            for i in range(1, 4):
                backup_prices.append(main_price - (tick_size * i))
        
        return backup_prices
    
    def _configure_low_latency_settings(self) -> Dict[str, Any]:
        """Configura impostazioni per bassa latenza"""
        return {
            'connection_pooling': True,
            'keep_alive': True,
            'tcp_nodelay': True,
            'compression': False,  # Disabilita compressione per velocità
            'parallel_connections': 3,
            'timeout_ms': 500,
            'retry_backoff': 'exponential'
        }
    
    def _estimate_execution_time(self) -> float:
        """Stima tempo di esecuzione in millisecondi"""
        base_time = 25  # 25ms base
        network_overhead = np.random.uniform(10, 30)
        processing_time = np.random.uniform(5, 15)
        
        return base_time + network_overhead + processing_time
    
    def _calculate_speed_advantage(self) -> float:
        """Calcola vantaggio di velocità vs competizione"""
        our_speed = self._estimate_execution_time()
        competitor_avg_speed = 150  # 150ms velocità media competitor
        
        advantage_ms = max(0, competitor_avg_speed - our_speed)
        return advantage_ms

class CompetitiveHFTEngine:
    """Motore principale per High-Frequency Trading competitivo"""
    
    def __init__(self):
        self.orderbook_analyzer = OrderBookAnalyzer()
        self.speed_optimizer = SpeedOptimizer()
        self.bot_profiles = {}
        self.competitive_trades = []
        self.is_active = False
        self.monitoring_thread = None
        
        logger.info("🚀 Competitive HFT Engine inizializzato")
    
    def start_competitive_trading(self, symbols: List[str] = None):
        """Avvia trading competitivo ad alta frequenza"""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
        
        self.is_active = True
        self.symbols = symbols
        
        # Avvia monitoring real-time
        self.monitoring_thread = threading.Thread(
            target=self._competitive_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"⚡ Competitive HFT attivo su {len(symbols)} simboli")
    
    def stop_competitive_trading(self):
        """Ferma trading competitivo"""
        self.is_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("🛑 Competitive HFT fermato")
    
    def _competitive_monitoring_loop(self):
        """Loop principale di monitoring competitivo"""
        while self.is_active:
            try:
                for symbol in self.symbols:
                    self._analyze_symbol_competition(symbol)
                
                time.sleep(0.1)  # 100ms cycle per alta frequenza
                
            except Exception as e:
                logger.error(f"Errore nel monitoring competitivo: {e}")
                time.sleep(1)
    
    def _analyze_symbol_competition(self, symbol: str):
        """Analizza competizione per un simbolo specifico"""
        
        # Simula order book real-time
        orderbook = self._get_simulated_orderbook(symbol)
        
        # Analizza dinamiche competitive
        analysis = self.orderbook_analyzer.analyze_orderbook_dynamics(orderbook)
        
        if analysis['bot_activity_detected']:
            # Genera strategia competitiva
            competitive_strategy = self._generate_competitive_strategy(symbol, analysis)
            
            # Esegue trade se opportunità identificata
            if competitive_strategy['execute']:
                self._execute_competitive_trade(symbol, competitive_strategy)
    
    def _get_simulated_orderbook(self, symbol: str) -> Dict[str, Any]:
        """Simula order book real-time (sostituire con API reale)"""
        
        # Simula order book con pattern realistici di bot
        base_price = 50000 if 'BTC' in symbol else 3000
        
        bids = []
        asks = []
        
        # Genera bids con pattern di bot
        for i in range(20):
            price = base_price - (i * 0.5)
            
            # Simula ordini tipici di bot
            if i == 0:
                size = np.random.uniform(0.1, 0.5)  # Best bid normale
            elif i in [2, 4, 6]:  # Pattern bot ogni 2 livelli
                size = 1.0  # Ordini identici (market maker bot)
            elif price == round(price):  # Round numbers
                size = np.random.uniform(0.8, 1.2)  # Bot pattern
            else:
                size = np.random.uniform(0.1, 2.0)
            
            bids.append([price, size])
        
        # Genera asks simili
        for i in range(20):
            price = base_price + (i * 0.5)
            
            if i == 0:
                size = np.random.uniform(0.1, 0.5)
            elif i in [2, 4, 6]:
                size = 1.0  # Market maker pattern
            elif price == round(price):
                size = np.random.uniform(0.8, 1.2)
            else:
                size = np.random.uniform(0.1, 2.0)
            
            asks.append([price, size])
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_competitive_strategy(self, symbol: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera strategia per competere con bot rilevati"""
        
        strategy = {
            'execute': False,
            'action': None,
            'price': None,
            'quantity': 0.001,
            'reasoning': [],
            'confidence': 0.0
        }
        
        # Analizza pattern rilevati
        patterns = analysis.get('detected_patterns', [])
        opportunities = analysis.get('front_run_opportunities', [])
        competitiveness = analysis.get('competitiveness_score', 0)
        
        # Strategia basata su market making bot
        if any(p['type'] == 'market_making_pattern' for p in patterns):
            strategy['action'] = 'take_liquidity'
            strategy['execute'] = True
            strategy['confidence'] = 0.7
            strategy['reasoning'].append("Market maker bot detected - taking liquidity")
        
        # Strategia basata su HFT bot
        if any(p['type'] == 'hft_tight_spread' for p in patterns):
            if competitiveness > 0.8:
                strategy['action'] = 'speed_competition'
                strategy['execute'] = True
                strategy['confidence'] = 0.8
                strategy['reasoning'].append("HFT competition - speed advantage required")
        
        # Strategia basata su opportunità
        if opportunities:
            large_orders = [op for op in opportunities if op['type'] == 'large_bid_anticipation']
            if large_orders:
                strategy['action'] = 'anticipate_large_order'
                strategy['execute'] = True
                strategy['confidence'] = 0.9
                strategy['reasoning'].append("Large order anticipation opportunity")
        
        # Calcola prezzo ottimale se dobbiamo eseguire
        if strategy['execute'] and analysis.get('optimal_entry_price'):
            optimal_prices = analysis['optimal_entry_price']
            if strategy['action'] in ['take_liquidity', 'anticipate_large_order']:
                strategy['price'] = optimal_prices.get('buy')
                strategy['action'] = 'buy'
            else:
                strategy['price'] = optimal_prices.get('sell') 
                strategy['action'] = 'sell'
        
        return strategy
    
    def _execute_competitive_trade(self, symbol: str, strategy: Dict[str, Any]):
        """Esegue trade competitivo ottimizzato per velocità"""
        
        trade_signal = {
            'symbol': symbol,
            'action': strategy['action'],
            'price': strategy['price'],
            'quantity': strategy['quantity'],
            'volatility': 0.02,
            'competition_level': 0.8
        }
        
        # Ottimizza per velocità
        speed_optimization = self.speed_optimizer.optimize_execution_speed(trade_signal)
        
        # Simula esecuzione trade
        execution_result = self._simulate_trade_execution(
            symbol, 
            strategy, 
            speed_optimization
        )
        
        # Registra risultato
        competitive_trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'strategy': strategy,
            'execution_result': execution_result,
            'speed_optimization': speed_optimization
        }
        
        self.competitive_trades.append(competitive_trade)
        
        logger.info(f"⚡ Competitive trade eseguito: {symbol} - {strategy['action']} - {execution_result['status']}")
    
    def _simulate_trade_execution(self, symbol: str, strategy: Dict[str, Any], 
                                speed_opt: Dict[str, Any]) -> Dict[str, Any]:
        """Simula esecuzione trade (sostituire con API exchange reale)"""
        
        # Simula successo basato su velocità e strategia
        execution_time = speed_opt['estimated_execution_time_ms']
        speed_advantage = speed_opt['competitive_advantage_ms']
        
        # Probabilità successo basata su vantaggio velocità
        success_probability = min(0.95, 0.6 + (speed_advantage / 200))
        
        success = np.random.random() < success_probability
        
        if success:
            return {
                'status': 'filled',
                'executed_price': strategy['price'],
                'executed_quantity': strategy['quantity'],
                'execution_time_ms': execution_time,
                'competitive_advantage_ms': speed_advantage,
                'fees': strategy['quantity'] * strategy['price'] * 0.001
            }
        else:
            return {
                'status': 'rejected',
                'reason': 'outpaced_by_competitor',
                'execution_time_ms': execution_time,
                'competitive_disadvantage_ms': abs(speed_advantage - 100)
            }
    
    def get_competitive_dashboard(self) -> Dict[str, Any]:
        """Dashboard per monitoring trading competitivo"""
        
        recent_trades = self.competitive_trades[-20:] if self.competitive_trades else []
        
        # Statistiche performance
        successful_trades = [t for t in recent_trades 
                           if t['execution_result']['status'] == 'filled']
        
        total_trades = len(recent_trades)
        success_rate = len(successful_trades) / total_trades if total_trades > 0 else 0
        
        # Analisi velocità
        execution_times = [t['speed_optimization']['estimated_execution_time_ms'] 
                          for t in recent_trades]
        avg_execution_time = np.mean(execution_times) if execution_times else 0
        
        # Analisi profitti
        profits = []
        for trade in successful_trades:
            # Simula profitto basato su spread capture
            profit = trade['strategy']['quantity'] * trade['strategy']['price'] * 0.0005
            profits.append(profit)
        
        total_profit = sum(profits) if profits else 0
        
        return {
            'competitive_hft_status': {
                'is_active': self.is_active,
                'monitored_symbols': len(getattr(self, 'symbols', [])),
                'active_since': datetime.now().isoformat()
            },
            'performance_metrics': {
                'total_trades': total_trades,
                'successful_trades': len(successful_trades),
                'success_rate_percent': success_rate * 100,
                'avg_execution_time_ms': avg_execution_time,
                'total_profit_usdt': total_profit
            },
            'recent_activity': [
                {
                    'timestamp': trade['timestamp'].isoformat(),
                    'symbol': trade['symbol'],
                    'action': trade['strategy']['action'],
                    'status': trade['execution_result']['status'],
                    'reasoning': trade['strategy']['reasoning']
                }
                for trade in recent_trades[-5:]
            ],
            'bot_detection': {
                'market_makers_detected': len([p for trade in recent_trades 
                                             for p in trade['strategy'].get('patterns', [])
                                             if p.get('type') == 'market_making_pattern']),
                'hft_bots_detected': len([p for trade in recent_trades 
                                        for p in trade['strategy'].get('patterns', [])
                                        if p.get('type') == 'hft_tight_spread']),
                'competition_level': 'HIGH' if success_rate < 0.7 else 'MEDIUM' if success_rate < 0.85 else 'LOW'
            },
            'speed_analytics': {
                'our_avg_speed_ms': avg_execution_time,
                'competitor_avg_speed_ms': 150,
                'speed_advantage_ms': max(0, 150 - avg_execution_time),
                'speed_rank': 'COMPETITIVE' if avg_execution_time < 100 else 'NEEDS_OPTIMIZATION'
            }
        }