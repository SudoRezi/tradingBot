#!/usr/bin/env python3
"""
Autonomous AI Trader - Completely Self-Learning Trading AI
AI che fa trading autonomamente e impara dalle proprie decisioni
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Dict, List, Any, Optional
import logging
import asyncio

class AutonomousAITrader:
    """AI Trading completamente autonoma che impara dalle proprie decisioni"""
    
    def __init__(self, ai_memory_path="autonomous_ai_memory"):
        self.memory_path = Path(ai_memory_path)
        self.memory_path.mkdir(exist_ok=True)
        
        # Database per storing decisioni AI e performance
        self.db_path = self.memory_path / "autonomous_ai.db"
        self.init_database()
        
        # Modelli AI autonomi
        self.ai_models = {}
        self.learning_data = {
            'decision_patterns': [],
            'market_correlations': [],
            'performance_history': [],
            'strategy_evolution': {},
            'risk_adaptation': {}
        }
        
        # Configurazione autonoma
        self.autonomous_config = {
            'learning_rate': 0.01,
            'exploration_rate': 0.1,
            'min_confidence_threshold': 0.6,
            'max_position_size': 0.1,  # 10% max per trade
            'stop_loss_threshold': 0.02,  # 2% stop loss
            'take_profit_threshold': 0.05,  # 5% take profit
            'rebalance_frequency': 3600,  # 1 ora
            'learning_window': 100,  # Ultimi 100 trades per apprendere
        }
        
        self.load_existing_ai_memory()
        
    def init_database(self):
        """Inizializza database per memoria AI autonoma"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabella decisioni AI con outcomes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                action TEXT,
                price REAL,
                quantity REAL,
                ai_reasoning TEXT,
                market_analysis TEXT,
                confidence REAL,
                expected_outcome REAL,
                actual_outcome REAL,
                profit_loss REAL,
                model_version TEXT,
                execution_time REAL
            )
        ''')
        
        # Tabella strategie AI apprese
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                conditions TEXT,
                success_rate REAL,
                avg_profit REAL,
                risk_score REAL,
                usage_count INTEGER,
                last_updated DATETIME,
                performance_score REAL
            )
        ''')
        
        # Tabella correlazioni di mercato scoperte dall'AI
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_pair TEXT,
                correlation_strength REAL,
                time_lag INTEGER,
                discovered_date DATETIME,
                validation_count INTEGER,
                reliability_score REAL
            )
        ''')
        
        # Tabella evoluzione AI
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                performance_metrics TEXT,
                learning_achievements TEXT,
                timestamp DATETIME,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def autonomous_market_analysis(self, symbols: List[str]) -> Dict:
        """Analisi di mercato autonoma dell'AI"""
        
        market_analysis = {}
        
        for symbol in symbols:
            # Simula analisi tecnica avanzata
            analysis = {
                'technical_score': np.random.uniform(0.3, 0.9),
                'volume_analysis': np.random.uniform(0.2, 0.8),
                'momentum_indicators': np.random.uniform(0.1, 0.95),
                'volatility_assessment': np.random.uniform(0.15, 0.85),
                'trend_strength': np.random.uniform(0.25, 0.9),
                'support_resistance': np.random.uniform(0.3, 0.8)
            }
            
            # Calcola score composito
            composite_score = np.mean(list(analysis.values()))
            analysis['composite_score'] = composite_score
            
            # Determina azione AI
            if composite_score > 0.7:
                action = 'BUY'
                confidence = min(0.9, composite_score + 0.1)
            elif composite_score < 0.4:
                action = 'SELL'
                confidence = min(0.9, (1 - composite_score) + 0.1)
            else:
                action = 'HOLD'
                confidence = 0.5
                
            analysis['ai_action'] = action
            analysis['ai_confidence'] = confidence
            
            market_analysis[symbol] = analysis
            
        return market_analysis
        
    async def make_autonomous_decision(self, symbol: str, market_data: Dict) -> Dict:
        """AI prende decisione autonoma basata su analisi"""
        
        # Recupera strategie apprese dall'AI
        learned_strategies = self.get_learned_strategies(symbol)
        
        # Analizza condizioni attuali
        current_conditions = self.analyze_current_conditions(market_data)
        
        # Applica modelli AI per decisione
        ai_decision = self.apply_ai_models(symbol, current_conditions, learned_strategies)
        
        # Valuta rischio autonomamente
        risk_assessment = self.autonomous_risk_assessment(ai_decision, market_data)
        
        # Decisione finale AI
        final_decision = {
            'symbol': symbol,
            'action': ai_decision['action'],
            'confidence': ai_decision['confidence'],
            'position_size': risk_assessment['position_size'],
            'stop_loss': risk_assessment['stop_loss'],
            'take_profit': risk_assessment['take_profit'],
            'ai_reasoning': ai_decision['reasoning'],
            'risk_score': risk_assessment['risk_score'],
            'expected_outcome': ai_decision['expected_return'],
            'timestamp': datetime.now()
        }
        
        # Registra decisione per apprendimento futuro
        self.record_ai_decision(final_decision)
        
        return final_decision
        
    def get_learned_strategies(self, symbol: str) -> List[Dict]:
        """Recupera strategie che l'AI ha imparato per questo symbol"""
        
        conn = sqlite3.connect(self.db_path)
        
        strategies = pd.read_sql_query('''
            SELECT * FROM ai_strategies 
            WHERE strategy_name LIKE ? AND success_rate > 0.5
            ORDER BY performance_score DESC
            LIMIT 10
        ''', conn, params=[f"%{symbol}%"])
        
        conn.close()
        
        return strategies.to_dict('records') if not strategies.empty else []
        
    def analyze_current_conditions(self, market_data: Dict) -> Dict:
        """Analizza condizioni attuali di mercato"""
        
        conditions = {
            'rsi': market_data.get('rsi', 50),
            'macd_signal': market_data.get('macd_signal', 'neutral'),
            'volume_ratio': market_data.get('volume_ratio', 1.0),
            'price_change_24h': market_data.get('price_change_24h', 0),
            'volatility': market_data.get('volatility', 0.02),
            'market_sentiment': market_data.get('sentiment', 'neutral')
        }
        
        # AI classifica le condizioni
        if conditions['rsi'] < 30 and conditions['price_change_24h'] < -0.05:
            conditions['market_state'] = 'oversold_opportunity'
        elif conditions['rsi'] > 70 and conditions['price_change_24h'] > 0.05:
            conditions['market_state'] = 'overbought_risk'
        elif abs(conditions['price_change_24h']) < 0.02:
            conditions['market_state'] = 'consolidation'
        else:
            conditions['market_state'] = 'trending'
            
        return conditions
        
    def apply_ai_models(self, symbol: str, conditions: Dict, strategies: List[Dict]) -> Dict:
        """Applica modelli AI per generare decisione"""
        
        # Simula ensemble di modelli AI
        model_predictions = []
        
        # Modello 1: Technical Pattern Recognition
        tech_score = self.technical_pattern_ai(conditions)
        model_predictions.append({
            'model': 'technical_ai',
            'action': 'BUY' if tech_score > 0.6 else 'SELL' if tech_score < 0.4 else 'HOLD',
            'confidence': abs(tech_score - 0.5) * 2,
            'weight': 0.3
        })
        
        # Modello 2: Strategy Learning AI
        strategy_score = self.strategy_learning_ai(strategies, conditions)
        model_predictions.append({
            'model': 'strategy_ai',
            'action': 'BUY' if strategy_score > 0.6 else 'SELL' if strategy_score < 0.4 else 'HOLD',
            'confidence': min(0.9, strategy_score),
            'weight': 0.4
        })
        
        # Modello 3: Market Regime AI
        regime_score = self.market_regime_ai(conditions)
        model_predictions.append({
            'model': 'regime_ai',
            'action': 'BUY' if regime_score > 0.65 else 'SELL' if regime_score < 0.35 else 'HOLD',
            'confidence': regime_score,
            'weight': 0.3
        })
        
        # Combina predizioni con ensemble
        final_decision = self.ensemble_decision(model_predictions)
        
        return final_decision
        
    def technical_pattern_ai(self, conditions: Dict) -> float:
        """AI per riconoscimento pattern tecnici"""
        
        score = 0.5  # Base neutral
        
        # RSI analysis
        rsi = conditions.get('rsi', 50)
        if rsi < 30:
            score += 0.2  # Oversold bullish
        elif rsi > 70:
            score -= 0.2  # Overbought bearish
            
        # Volume analysis
        volume_ratio = conditions.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score += 0.1  # High volume confirmation
        elif volume_ratio < 0.5:
            score -= 0.05  # Low volume weakness
            
        # Price momentum
        price_change = conditions.get('price_change_24h', 0)
        if price_change > 0.03:
            score += 0.15  # Strong momentum
        elif price_change < -0.03:
            score -= 0.15  # Weak momentum
            
        return max(0, min(1, score))
        
    def strategy_learning_ai(self, strategies: List[Dict], conditions: Dict) -> float:
        """AI che applica strategie apprese"""
        
        if not strategies:
            return 0.5  # Neutral se no strategie
            
        total_score = 0
        total_weight = 0
        
        for strategy in strategies:
            # Valuta se condizioni attuali corrispondono alla strategia
            strategy_conditions = json.loads(strategy.get('conditions', '{}'))
            
            similarity = self.calculate_conditions_similarity(conditions, strategy_conditions)
            
            if similarity > 0.7:  # Se condizioni simili
                weight = strategy['success_rate'] * strategy['performance_score']
                score = strategy['success_rate']
                
                total_score += score * weight
                total_weight += weight
                
        return total_score / total_weight if total_weight > 0 else 0.5
        
    def market_regime_ai(self, conditions: Dict) -> float:
        """AI per identificazione regime di mercato"""
        
        market_state = conditions.get('market_state', 'neutral')
        volatility = conditions.get('volatility', 0.02)
        
        if market_state == 'oversold_opportunity':
            return 0.8  # Strong buy signal
        elif market_state == 'overbought_risk':
            return 0.2  # Strong sell signal
        elif market_state == 'trending':
            return 0.7 if conditions.get('price_change_24h', 0) > 0 else 0.3
        else:  # consolidation
            return 0.5  # Neutral
            
    def ensemble_decision(self, predictions: List[Dict]) -> Dict:
        """Combina predizioni multiple in decisione finale"""
        
        action_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_confidence = 0
        total_weight = 0
        
        for pred in predictions:
            weight = pred['weight'] * pred['confidence']
            action_scores[pred['action']] += weight
            total_confidence += pred['confidence'] * pred['weight']
            total_weight += pred['weight']
            
        # Decisione finale
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.5
        
        # Expected return basato su historical performance
        expected_return = self.calculate_expected_return(final_action, final_confidence)
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'expected_return': expected_return,
            'reasoning': f"Ensemble decision: {final_action} with {final_confidence:.2%} confidence",
            'model_scores': action_scores
        }
        
    def autonomous_risk_assessment(self, decision: Dict, market_data: Dict) -> Dict:
        """AI valuta autonomamente il rischio"""
        
        base_position_size = self.autonomous_config['max_position_size']
        
        # Aggiusta size basato su confidence
        confidence_factor = decision['confidence']
        position_size = base_position_size * confidence_factor
        
        # Aggiusta basato su volatilità
        volatility = market_data.get('volatility', 0.02)
        volatility_factor = 1 / (1 + volatility * 10)  # Riduci size con alta volatilità
        position_size *= volatility_factor
        
        # Stop loss e take profit dinamici
        if decision['action'] == 'BUY':
            stop_loss = -self.autonomous_config['stop_loss_threshold'] * (1 + volatility)
            take_profit = self.autonomous_config['take_profit_threshold'] * (1 + confidence_factor * 0.5)
        elif decision['action'] == 'SELL':
            stop_loss = self.autonomous_config['stop_loss_threshold'] * (1 + volatility)
            take_profit = -self.autonomous_config['take_profit_threshold'] * (1 + confidence_factor * 0.5)
        else:  # HOLD
            stop_loss = 0
            take_profit = 0
            position_size = 0
            
        risk_score = volatility * (1 - confidence_factor) * position_size
        
        return {
            'position_size': min(position_size, self.autonomous_config['max_position_size']),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_score': risk_score
        }
        
    def record_ai_decision(self, decision: Dict):
        """Registra decisione AI per apprendimento futuro"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ai_decisions (
                timestamp, symbol, action, confidence, 
                ai_reasoning, expected_outcome, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision['timestamp'],
            decision['symbol'],
            decision['action'],
            decision['confidence'],
            decision['ai_reasoning'],
            decision['expected_outcome'],
            'autonomous_v1.0'
        ))
        
        conn.commit()
        conn.close()
        
    def update_ai_performance(self, decision_id: int, actual_outcome: float, profit_loss: float):
        """Aggiorna performance AI per apprendimento"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Aggiorna outcome nella decisione
        cursor.execute('''
            UPDATE ai_decisions 
            SET actual_outcome = ?, profit_loss = ?
            WHERE id = ?
        ''', (actual_outcome, profit_loss, decision_id))
        
        # Trigger apprendimento AI
        self.ai_learning_update(decision_id, actual_outcome, profit_loss)
        
        conn.commit()
        conn.close()
        
    def ai_learning_update(self, decision_id: int, outcome: float, profit: float):
        """AI impara dalle proprie decisioni"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Recupera decisione
        decision = pd.read_sql_query(
            "SELECT * FROM ai_decisions WHERE id = ?", 
            conn, 
            params=[decision_id]
        )
        
        if not decision.empty:
            dec = decision.iloc[0]
            
            # Analizza cosa ha funzionato o no
            success = outcome > 0
            strategy_name = f"{dec['action']}_{dec['symbol']}_pattern"
            
            # Aggiorna o crea strategia
            self.update_ai_strategy(strategy_name, success, profit, dec)
            
            # Aggiorna configurazione autonoma se necessario
            if profit < -0.05:  # Grande perdita
                self.autonomous_config['max_position_size'] *= 0.95  # Riduci risk
            elif profit > 0.05:  # Grande guadagno
                self.autonomous_config['max_position_size'] *= 1.02  # Aumenta leggermente
                
        conn.close()
        
    def update_ai_strategy(self, strategy_name: str, success: bool, profit: float, decision_data):
        """Aggiorna strategia AI basata su risultati"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cerca strategia esistente
        existing = cursor.execute(
            "SELECT * FROM ai_strategies WHERE strategy_name = ?", 
            [strategy_name]
        ).fetchone()
        
        if existing:
            # Aggiorna strategia esistente
            old_success_rate = existing[2]
            old_avg_profit = existing[3]
            old_usage_count = existing[5]
            
            new_usage_count = old_usage_count + 1
            new_success_rate = ((old_success_rate * old_usage_count) + (1 if success else 0)) / new_usage_count
            new_avg_profit = ((old_avg_profit * old_usage_count) + profit) / new_usage_count
            
            # Performance score composito
            performance_score = new_success_rate * (1 + max(0, new_avg_profit))
            
            cursor.execute('''
                UPDATE ai_strategies 
                SET success_rate = ?, avg_profit = ?, usage_count = ?, 
                    last_updated = ?, performance_score = ?
                WHERE strategy_name = ?
            ''', (new_success_rate, new_avg_profit, new_usage_count, 
                  datetime.now(), performance_score, strategy_name))
        else:
            # Crea nuova strategia
            cursor.execute('''
                INSERT INTO ai_strategies (
                    strategy_name, success_rate, avg_profit, usage_count,
                    last_updated, performance_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (strategy_name, 1.0 if success else 0.0, profit, 1,
                  datetime.now(), 1.0 if success else 0.0))
                  
        conn.commit()
        conn.close()
        
    def calculate_conditions_similarity(self, current: Dict, historical: Dict) -> float:
        """Calcola similarità tra condizioni di mercato"""
        
        if not historical:
            return 0.0
            
        similarities = []
        
        for key in current:
            if key in historical:
                if isinstance(current[key], (int, float)) and isinstance(historical[key], (int, float)):
                    # Similarità numerica
                    diff = abs(current[key] - historical[key])
                    max_val = max(abs(current[key]), abs(historical[key]), 1)
                    similarity = 1 - (diff / max_val)
                    similarities.append(max(0, similarity))
                elif current[key] == historical[key]:
                    # Similarità categorica
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
                    
        return np.mean(similarities) if similarities else 0.0
        
    def calculate_expected_return(self, action: str, confidence: float) -> float:
        """Calcola expected return basato su historical performance"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Recupera performance storica per questa azione
        historical_returns = pd.read_sql_query('''
            SELECT actual_outcome FROM ai_decisions 
            WHERE action = ? AND actual_outcome IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 50
        ''', conn, params=[action])
        
        conn.close()
        
        if historical_returns.empty:
            # Default expected returns
            default_returns = {'BUY': 0.02, 'SELL': 0.01, 'HOLD': 0.0}
            return default_returns.get(action, 0.0) * confidence
        else:
            avg_return = historical_returns['actual_outcome'].mean()
            return avg_return * confidence
            
    def get_ai_performance_stats(self) -> Dict:
        """Ottieni statistiche performance AI"""
        
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Statistiche generali
        total_decisions = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM ai_decisions", conn
        ).iloc[0]['count']
        
        completed_decisions = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM ai_decisions WHERE actual_outcome IS NOT NULL", conn
        ).iloc[0]['count']
        
        stats['total_decisions'] = total_decisions
        stats['completed_decisions'] = completed_decisions
        
        if completed_decisions > 0:
            # Performance metrics
            perf_query = '''
                SELECT 
                    AVG(CASE WHEN actual_outcome > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(profit_loss) as avg_profit,
                    SUM(profit_loss) as total_profit,
                    AVG(confidence) as avg_confidence
                FROM ai_decisions 
                WHERE actual_outcome IS NOT NULL
            '''
            perf_stats = pd.read_sql_query(perf_query, conn).iloc[0]
            stats.update(perf_stats.to_dict())
            
            # Top strategies
            top_strategies = pd.read_sql_query('''
                SELECT strategy_name, success_rate, avg_profit, usage_count, performance_score
                FROM ai_strategies 
                ORDER BY performance_score DESC 
                LIMIT 5
            ''', conn)
            
            stats['top_strategies'] = top_strategies.to_dict('records')
            
        conn.close()
        
        return stats
        
    def save_ai_memory_backup(self, backup_path: str = None) -> str:
        """Salva backup completo memoria AI autonoma"""
        
        if backup_path is None:
            backup_path = f"autonomous_ai_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
        import shutil
        shutil.make_archive(
            backup_path.replace('.zip', ''),
            'zip',
            self.memory_path
        )
        
        return backup_path
        
    def load_ai_memory_backup(self, backup_path: str):
        """Carica backup memoria AI"""
        
        import zipfile
        import shutil
        
        # Backup current memory
        current_backup = self.save_ai_memory_backup("current_ai_backup.zip")
        
        try:
            # Clear current memory
            shutil.rmtree(self.memory_path)
            self.memory_path.mkdir(exist_ok=True)
            
            # Extract backup
            with zipfile.ZipFile(backup_path, 'r') as zip_ref:
                zip_ref.extractall(self.memory_path)
                
            # Reload memory
            self.load_existing_ai_memory()
            
            return True
            
        except Exception as e:
            # Restore current memory if failed
            self.load_ai_memory_backup(current_backup)
            raise e
            
    def load_existing_ai_memory(self):
        """Carica memoria AI esistente"""
        
        if self.db_path.exists():
            stats = self.get_ai_performance_stats()
            
            # Aggiorna configurazione basata su performance storica
            if stats.get('win_rate', 0) < 0.4:
                # Performance scarsa, riduci risk
                self.autonomous_config['max_position_size'] *= 0.8
                self.autonomous_config['min_confidence_threshold'] *= 1.1
            elif stats.get('win_rate', 0) > 0.7:
                # Performance buona, aumenta leggermente risk
                self.autonomous_config['max_position_size'] *= 1.05
                
async def main():
    """Test sistema AI autonomo"""
    
    ai_trader = AutonomousAITrader()
    
    # Simula analisi e decisione autonoma
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    print("🤖 Autonomous AI Trader - Making Independent Decisions")
    print("=" * 60)
    
    market_analysis = await ai_trader.autonomous_market_analysis(symbols)
    
    for symbol, analysis in market_analysis.items():
        print(f"\n{symbol}:")
        print(f"  AI Action: {analysis['ai_action']}")
        print(f"  Confidence: {analysis['ai_confidence']:.2%}")
        print(f"  Composite Score: {analysis['composite_score']:.3f}")
        
        # Fai decisione autonoma dettagliata
        market_data = {
            'rsi': analysis['technical_score'] * 100,
            'volatility': 0.02,
            'volume_ratio': 1.2,
            'price_change_24h': (analysis['composite_score'] - 0.5) * 0.1
        }
        
        decision = await ai_trader.make_autonomous_decision(symbol, market_data)
        
        print(f"  Final Decision: {decision['action']}")
        print(f"  Position Size: {decision['position_size']:.2%}")
        print(f"  Stop Loss: {decision['stop_loss']:.2%}")
        print(f"  Take Profit: {decision['take_profit']:.2%}")
        print(f"  AI Reasoning: {decision['ai_reasoning']}")
        
    # Mostra statistiche AI
    stats = ai_trader.get_ai_performance_stats()
    print(f"\n📊 AI Performance Stats:")
    print(f"  Total Decisions: {stats['total_decisions']}")
    print(f"  Completed: {stats['completed_decisions']}")
    
    if stats.get('win_rate'):
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Avg Profit: ${stats['avg_profit']:.2f}")
        print(f"  Total Profit: ${stats['total_profit']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())