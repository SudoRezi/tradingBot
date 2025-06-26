import threading
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from apscheduler.schedulers.background import BackgroundScheduler
from .market_analyzer import MarketAnalyzer
from .strategy_engine import StrategyEngine
from .risk_manager import RiskManager
from .portfolio_manager import PortfolioManager
from .data_manager import DataManager
from .notification_system import NotificationSystem
from .advanced_ml_engine import AdvancedMLEngine
from .multi_exchange_arbitrage import MultiExchangeArbitrage
from .volatility_garch_engine import VolatilityEngine
from .stress_test_monte_carlo import StressTestEngine
from .alternative_data_engine import AlternativeDataEngine
from .mlops_pipeline import MLOpsPipeline
from .dynamic_leverage_engine import DynamicLeverageEngine
from .perpetual_futures_arbitrage import PerpetualArbitrageEngine
from .options_strategies_engine import OptionsStrategiesEngine
from .portfolio_diversification_engine import PortfolioDiversificationEngine
from .tax_reporting_engine import TaxReportingEngine, TaxEvent, TaxMethod, TaxableTransaction
from config.settings import TRADING_PAIRS, ANALYSIS_INTERVAL

logger = logging.getLogger(__name__)

class AITrader:
    """Main AI Trading Engine - Operates 24/7 Autonomously"""
    
    def __init__(self, api_key, api_secret, initial_capital, risk_level):
        self.api_key = api_key
        self.api_secret = api_secret
        self.initial_capital = initial_capital
        self.risk_level = risk_level
        self.is_running = False
        
        # Initialize core components
        self.data_manager = DataManager(api_key, api_secret)
        self.market_analyzer = MarketAnalyzer()
        self.strategy_engine = StrategyEngine(risk_level)
        self.risk_manager = RiskManager(initial_capital, risk_level)
        self.portfolio_manager = PortfolioManager(initial_capital)
        self.notification_system = NotificationSystem()
        
        # Initialize advanced components
        self.advanced_ml_engine = AdvancedMLEngine()
        self.arbitrage_engine = MultiExchangeArbitrage()
        self.volatility_engine = VolatilityEngine()
        self.stress_test_engine = StressTestEngine()
        self.alternative_data_engine = AlternativeDataEngine()
        self.mlops_pipeline = MLOpsPipeline()
        
        # Initialize institutional-grade engines
        self.leverage_engine = DynamicLeverageEngine(initial_capital)
        self.futures_arbitrage_engine = PerpetualArbitrageEngine()
        self.options_engine = OptionsStrategiesEngine()
        self.diversification_engine = PortfolioDiversificationEngine()
        self.tax_engine = TaxReportingEngine(TaxMethod.FIFO)
        
        # Initialize scheduler for 24/7 operations
        self.scheduler = BackgroundScheduler()
        self._setup_trading_schedule()
        
        logger.info(f"AI Trader initialized with {initial_capital} USDT capital")
    
    def _setup_trading_schedule(self):
        """Setup automated trading schedule"""
        # Main trading analysis every 30 seconds
        self.scheduler.add_job(
            self._execute_trading_cycle,
            'interval',
            seconds=ANALYSIS_INTERVAL,
            id='main_trading_cycle'
        )
        
        # Portfolio rebalancing every 5 minutes
        self.scheduler.add_job(
            self._rebalance_portfolio,
            'interval',
            minutes=5,
            id='portfolio_rebalance'
        )
        
        # Risk assessment every minute
        self.scheduler.add_job(
            self._assess_risks,
            'interval',
            minutes=1,
            id='risk_assessment'
        )
        
        # Performance optimization every hour
        self.scheduler.add_job(
            self._optimize_strategies,
            'interval',
            hours=1,
            id='strategy_optimization'
        )
    
    def start(self):
        """Start the AI trading system"""
        if self.is_running:
            logger.warning("AI Trader is already running")
            return
        
        try:
            self.is_running = True
            self.scheduler.start()
            logger.info("🚀 AI Trader started - Operating 24/7")
            
            # Send startup notification
            self.notification_system.send_notification(
                "🚀 AI Trader Started",
                f"Your AI is now trading autonomously with {self.initial_capital} USDT"
            )
            
        except Exception as e:
            logger.error(f"Failed to start AI Trader: {e}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop the AI trading system"""
        if not self.is_running:
            return
        
        try:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("⏸️ AI Trader stopped")
            
        except Exception as e:
            logger.error(f"Error stopping AI Trader: {e}")
    
    def _execute_trading_cycle(self):
        """Main trading cycle - executed automatically"""
        try:
            logger.info("Executing trading cycle...")
            
            # Analyze all trading pairs simultaneously
            market_data = {}
            signals = {}
            
            for pair in TRADING_PAIRS:
                # Get latest market data
                data = self.data_manager.get_market_data(pair)
                if data is None:
                    continue
                
                market_data[pair] = data
                
                # Analyze market conditions
                analysis = self.market_analyzer.analyze_market(data, pair)
                
                # Generate trading signals from traditional strategies
                traditional_signal = self.strategy_engine.generate_signal(analysis, pair)
                
                # Get advanced ML signal
                ml_signal = self.advanced_ml_engine.get_ml_signal(data.values)
                
                # Get alternative data signal
                alt_data_signal = self.alternative_data_engine.get_comprehensive_sentiment(pair.split('/')[0])
                
                # Get volatility-adjusted position sizing
                vol_recommendation = self.volatility_engine.get_position_sizing_recommendation(
                    pair, self.portfolio_manager.get_total_portfolio_value(), 
                    data['close'].iloc[-1], 'BUY'
                )
                
                # Combine all signals
                combined_signal = self._combine_advanced_signals(
                    traditional_signal, ml_signal, alt_data_signal, vol_recommendation
                )
                
                signals[pair] = combined_signal
                
                # Execute trades if signal is strong enough
                if combined_signal and combined_signal['confidence'] > 0.7:
                    self._execute_advanced_trade(pair, combined_signal)
            
            # Update portfolio performance
            self.portfolio_manager.update_positions(market_data)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _execute_trade(self, pair, signal):
        """Execute trade based on AI signal"""
        try:
            # Risk assessment
            if not self.risk_manager.can_trade(pair, signal):
                logger.info(f"Risk manager blocked trade for {pair}")
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                pair, signal, self.portfolio_manager.get_available_balance()
            )
            
            if position_size <= 0:
                return
            
            # Execute the trade
            success = self.portfolio_manager.execute_trade(
                pair, signal['action'], position_size, signal['price']
            )
            
            if success:
                logger.info(f"✅ Trade executed: {signal['action']} {position_size} {pair} @ {signal['price']}")
                
                # Send notification for significant trades
                if position_size * signal['price'] > self.initial_capital * 0.05:  # >5% of capital
                    self.notification_system.send_notification(
                        f"💰 Large Trade Executed",
                        f"{signal['action']} {position_size:.6f} {pair} @ ${signal['price']:.6f}"
                    )
            
        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}")
    
    def _combine_advanced_signals(self, traditional_signal, ml_signal, alt_data_signal, vol_recommendation):
        """Combina segnali da tutte le fonti avanzate"""
        try:
            if not traditional_signal and not ml_signal:
                return None
            
            # Pesi per diverse fonti
            weights = {
                'traditional': 0.3,
                'ml': 0.4,
                'alternative_data': 0.2,
                'volatility': 0.1
            }
            
            # Estrai segnali
            trad_action = traditional_signal.get('action', 'HOLD') if traditional_signal else 'HOLD'
            trad_strength = traditional_signal.get('strength', 0) if traditional_signal else 0
            
            ml_action = ml_signal.get('action', 'HOLD') if ml_signal else 'HOLD'
            ml_strength = ml_signal.get('strength', 0) if ml_signal else 0
            
            alt_direction = alt_data_signal.get('signal_direction', 'NEUTRAL') if alt_data_signal else 'NEUTRAL'
            alt_strength = alt_data_signal.get('signal_strength', 0) if alt_data_signal else 0
            
            # Converti alternative data in azione
            alt_action = 'BUY' if alt_direction == 'BULLISH' else 'SELL' if alt_direction == 'BEARISH' else 'HOLD'
            
            # Combina azioni
            actions = [trad_action, ml_action, alt_action]
            buy_votes = actions.count('BUY')
            sell_votes = actions.count('SELL')
            
            if buy_votes > sell_votes:
                final_action = 'BUY'
            elif sell_votes > buy_votes:
                final_action = 'SELL'
            else:
                final_action = 'HOLD'
            
            # Combina strength
            total_strength = (
                trad_strength * weights['traditional'] +
                ml_strength * weights['ml'] +
                alt_strength * weights['alternative_data']
            )
            
            # Confidence basata su consenso
            consensus = max(buy_votes, sell_votes) / len(actions)
            confidence = consensus * total_strength
            
            # Usa volatility per position sizing
            position_size = vol_recommendation.get('recommended_position_size', 0) if vol_recommendation else 0
            
            return {
                'action': final_action,
                'strength': total_strength,
                'confidence': confidence,
                'position_size': position_size,
                'price': traditional_signal.get('price', 0) if traditional_signal else 0,
                'source': 'advanced_ensemble',
                'component_signals': {
                    'traditional': {'action': trad_action, 'strength': trad_strength},
                    'ml': {'action': ml_action, 'strength': ml_strength},
                    'alternative_data': {'action': alt_action, 'strength': alt_strength}
                }
            }
            
        except Exception as e:
            logger.error(f"Error combining advanced signals: {e}")
            return traditional_signal
    
    def _execute_advanced_trade(self, pair: str, signal: Dict[str, Any]):
        """Esegue trade con logica avanzata"""
        try:
            # Risk assessment con tutti i sistemi
            if not self.risk_manager.can_trade(pair, signal):
                logger.info(f"Risk manager blocked advanced trade for {pair}")
                return
            
            # Usa position size da volatility engine se disponibile
            position_size = signal.get('position_size', 0)
            if position_size <= 0:
                position_size = self.risk_manager.calculate_position_size(
                    pair, signal, self.portfolio_manager.get_available_balance()
                )
            
            if position_size <= 0:
                return
            
            # Controlla opportunità arbitraggio
            arbitrage_opportunities = self.arbitrage_engine.find_arbitrage_opportunities(pair)
            if arbitrage_opportunities:
                best_opportunity = arbitrage_opportunities[0]
                if best_opportunity.net_profit > 50:  # Soglia minima profitto
                    logger.info(f"Arbitrage opportunity found for {pair}: ${best_opportunity.net_profit:.2f}")
                    self.arbitrage_engine.execute_arbitrage(best_opportunity, position_size)
                    return
            
            # Esegue trade normale con position size ottimizzato
            success = self.portfolio_manager.execute_trade(
                pair, signal['action'], position_size, signal['price']
            )
            
            if success:
                logger.info(f"✅ Advanced trade executed: {signal['action']} {position_size} {pair} @ {signal['price']}")
                
                # Log per MLOps monitoring
                if hasattr(self, 'mlops_pipeline'):
                    features = np.array([signal['strength'], signal['confidence'], position_size])
                    prediction = 1 if signal['action'] == 'BUY' else -1 if signal['action'] == 'SELL' else 0
                    self.mlops_pipeline.log_prediction_and_monitor(
                        'advanced_ensemble', features, prediction, signal['confidence']
                    )
                
                # Notifica per trade significativi
                if position_size * signal['price'] > self.initial_capital * 0.03:  # >3% del capitale
                    self.notification_system.send_notification(
                        f"🤖 Advanced AI Trade Executed",
                        f"{signal['action']} {position_size:.6f} {pair} @ ${signal['price']:.6f}\n"
                        f"Sources: {', '.join(signal['component_signals'].keys())}\n"
                        f"Confidence: {signal['confidence']:.1%}"
                    )
            
        except Exception as e:
            logger.error(f"Error executing advanced trade for {pair}: {e}")
    
    def _rebalance_portfolio(self):
        """Automatically rebalance portfolio"""
        try:
            logger.info("Rebalancing portfolio...")
            
            # Check if rebalancing is needed
            if self.portfolio_manager.needs_rebalancing():
                rebalance_actions = self.portfolio_manager.calculate_rebalancing()
                
                for action in rebalance_actions:
                    self.portfolio_manager.execute_trade(
                        action['pair'], action['side'], action['size'], action['price']
                    )
                
                logger.info(f"Portfolio rebalanced: {len(rebalance_actions)} actions")
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
    
    def _assess_risks(self):
        """Assess and manage risks automatically"""
        try:
            # Check global portfolio risk
            portfolio_risk = self.risk_manager.assess_portfolio_risk(
                self.portfolio_manager.positions
            )
            
            # Emergency stop if risk is too high
            if portfolio_risk['risk_level'] == 'CRITICAL':
                logger.warning("🚨 CRITICAL RISK DETECTED - Reducing positions")
                
                # Close riskiest positions
                risky_positions = portfolio_risk['risky_positions']
                for position in risky_positions[:3]:  # Close top 3 riskiest
                    self.portfolio_manager.close_position(position['pair'])
                
                # Send emergency notification
                self.notification_system.send_notification(
                    "🚨 Risk Management Alert",
                    "AI detected high risk and automatically reduced positions"
                )
            
            # Update stop losses
            self.portfolio_manager.update_stop_losses()
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
    
    def _optimize_strategies(self):
        """Automatically optimize trading strategies based on performance"""
        try:
            logger.info("Optimizing trading strategies...")
            
            # Get performance metrics
            performance = self.portfolio_manager.get_performance_metrics()
            
            # Let strategy engine learn from results
            self.strategy_engine.optimize_from_performance(performance)
            
            # Adjust risk parameters if needed
            self.risk_manager.adjust_parameters(performance)
            
            logger.info("Strategy optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing strategies: {e}")
    
    def get_status(self):
        """Get current AI trader status"""
        return {
            'is_running': self.is_running,
            'active_pairs': TRADING_PAIRS,
            'portfolio_value': self.portfolio_manager.get_total_portfolio_value(),
            'active_positions': len(self.portfolio_manager.positions),
            'total_trades': self.portfolio_manager.total_trades,
            'win_rate': self.portfolio_manager.get_win_rate(),
            'uptime': datetime.now() - self.portfolio_manager.start_time if self.is_running else None
        }
