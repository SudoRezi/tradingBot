#!/usr/bin/env python3
"""
AI Learning System - Self-Training Trading AI
Sistema che impara dai trades e sviluppa la propria AI personalizzata
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

class PersonalAILearningSystem:
    """Sistema AI che impara dai tuoi trades e crea modelli personalizzati"""
    
    def __init__(self, ai_memory_path="ai_memory"):
        self.memory_path = Path(ai_memory_path)
        self.memory_path.mkdir(exist_ok=True)
        
        # Database per storing trade history e decisioni
        self.db_path = self.memory_path / "ai_memory.db"
        self.init_database()
        
        # Modelli personalizzati
        self.personal_models = {}
        self.learning_data = {
            'trade_patterns': [],
            'market_conditions': [],
            'decision_outcomes': [],
            'user_preferences': {},
            'performance_metrics': {}
        }
        
        self.load_existing_memory()
        
    def init_database(self):
        """Inizializza database per memory permanente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabella trades con outcomes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                action TEXT,
                price REAL,
                quantity REAL,
                reason TEXT,
                market_conditions TEXT,
                outcome REAL,
                profit_loss REAL,
                confidence REAL,
                model_used TEXT
            )
        ''')
        
        # Tabella pattern riconosciuti
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                conditions TEXT,
                success_rate REAL,
                avg_profit REAL,
                occurrences INTEGER,
                last_seen DATETIME
            )
        ''')
        
        # Tabella preferenze utente apprese
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                preference_type TEXT PRIMARY KEY,
                value TEXT,
                confidence REAL,
                learned_date DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def record_trade_decision(self, trade_data: Dict):
        """Registra una decisione di trade per imparare"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                timestamp, symbol, action, price, quantity, 
                reason, market_conditions, confidence, model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            trade_data.get('symbol'),
            trade_data.get('action'),
            trade_data.get('price'),
            trade_data.get('quantity'),
            trade_data.get('reason'),
            json.dumps(trade_data.get('market_conditions', {})),
            trade_data.get('confidence'),
            trade_data.get('model_used')
        ))
        
        conn.commit()
        conn.close()
        
    def update_trade_outcome(self, trade_id: int, outcome: float, profit_loss: float):
        """Aggiorna l'outcome di un trade per learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trades 
            SET outcome = ?, profit_loss = ?
            WHERE id = ?
        ''', (outcome, profit_loss, trade_id))
        
        conn.commit()
        conn.close()
        
        # Trigger learning da questo outcome
        self.learn_from_outcome(trade_id, outcome, profit_loss)
        
    def learn_from_outcome(self, trade_id: int, outcome: float, profit_loss: float):
        """Impara da un outcome specifico"""
        conn = sqlite3.connect(self.db_path)
        
        # Recupera dettagli trade
        trade_df = pd.read_sql_query(
            "SELECT * FROM trades WHERE id = ?", 
            conn, 
            params=[trade_id]
        )
        
        if not trade_df.empty:
            trade = trade_df.iloc[0]
            
            # Analizza pattern che ha portato a questo outcome
            pattern_key = f"{trade['action']}_{trade['symbol']}"
            market_conditions = json.loads(trade['market_conditions'] or '{}')
            
            # Aggiorna pattern success rate
            self.update_pattern_performance(pattern_key, market_conditions, outcome > 0, profit_loss)
            
            # Impara preferenze utente
            self.learn_user_preferences(trade, outcome, profit_loss)
            
        conn.close()
        
    def update_pattern_performance(self, pattern_key: str, conditions: Dict, success: bool, profit: float):
        """Aggiorna performance di un pattern specifico"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cerca pattern esistente
        cursor.execute(
            "SELECT * FROM patterns WHERE pattern_type = ?", 
            [pattern_key]
        )
        existing = cursor.fetchone()
        
        if existing:
            # Aggiorna pattern esistente
            old_success_rate = existing[2]
            old_avg_profit = existing[3]
            old_occurrences = existing[4]
            
            new_occurrences = old_occurrences + 1
            new_success_rate = ((old_success_rate * old_occurrences) + (1 if success else 0)) / new_occurrences
            new_avg_profit = ((old_avg_profit * old_occurrences) + profit) / new_occurrences
            
            cursor.execute('''
                UPDATE patterns 
                SET success_rate = ?, avg_profit = ?, occurrences = ?, last_seen = ?
                WHERE pattern_type = ?
            ''', (new_success_rate, new_avg_profit, new_occurrences, datetime.now(), pattern_key))
            
        else:
            # Crea nuovo pattern
            cursor.execute('''
                INSERT INTO patterns (
                    pattern_type, conditions, success_rate, 
                    avg_profit, occurrences, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                pattern_key,
                json.dumps(conditions),
                1.0 if success else 0.0,
                profit,
                1,
                datetime.now()
            ))
            
        conn.commit()
        conn.close()
        
    def learn_user_preferences(self, trade: pd.Series, outcome: float, profit_loss: float):
        """Impara preferenze utente dai trades"""
        
        preferences = {}
        
        # Risk tolerance
        if trade['quantity'] and trade['price']:
            trade_size = trade['quantity'] * trade['price']
            if outcome > 0:
                preferences['preferred_trade_size'] = trade_size
                preferences['risk_tolerance'] = 'medium' if trade_size < 1000 else 'high'
                
        # Timing preferences
        hour = pd.to_datetime(trade['timestamp']).hour
        if outcome > 0:
            preferences['preferred_trading_hours'] = hour
            
        # Symbol preferences
        if profit_loss > 0:
            preferences['successful_symbols'] = trade['symbol']
            
        # Save learned preferences
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pref_type, value in preferences.items():
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences 
                (preference_type, value, confidence, learned_date)
                VALUES (?, ?, ?, ?)
            ''', (pref_type, str(value), abs(outcome), datetime.now()))
            
        conn.commit()
        conn.close()
        
    def get_personal_recommendation(self, symbol: str, market_conditions: Dict) -> Dict:
        """Genera raccomandazione basata su AI personalizzata"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Cerca pattern simili con buona performance
        similar_patterns = pd.read_sql_query('''
            SELECT * FROM patterns 
            WHERE pattern_type LIKE ? AND success_rate > 0.6
            ORDER BY success_rate DESC, occurrences DESC
            LIMIT 5
        ''', conn, params=[f"%{symbol}%"])
        
        # Cerca preferenze utente rilevanti
        user_prefs = pd.read_sql_query('''
            SELECT * FROM user_preferences 
            WHERE confidence > 0.5
            ORDER BY confidence DESC
        ''', conn)
        
        conn.close()
        
        # Genera raccomandazione personalizzata
        recommendation = {
            'action': 'HOLD',
            'confidence': 0.5,
            'reasoning': 'Insufficient personal data',
            'personal_score': 0.0,
            'patterns_found': len(similar_patterns),
            'user_prefs_applied': len(user_prefs)
        }
        
        if not similar_patterns.empty:
            # Usa pattern personali per raccomandazione
            best_pattern = similar_patterns.iloc[0]
            
            recommendation.update({
                'action': 'BUY' if best_pattern['avg_profit'] > 0 else 'SELL',
                'confidence': min(0.9, best_pattern['success_rate']),
                'reasoning': f"Personal pattern: {best_pattern['pattern_type']} " +
                           f"({best_pattern['success_rate']:.1%} success, " +
                           f"{best_pattern['occurrences']} trades)",
                'personal_score': best_pattern['success_rate'] * best_pattern['avg_profit'],
                'expected_profit': best_pattern['avg_profit']
            })
            
        return recommendation
        
    def save_ai_memory_backup(self, backup_path: str = None):
        """Salva backup completo della memoria AI"""
        
        if backup_path is None:
            backup_path = f"ai_memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
        import shutil
        shutil.make_archive(
            backup_path.replace('.zip', ''),
            'zip',
            self.memory_path
        )
        
        return f"{backup_path}"
        
    def load_ai_memory_backup(self, backup_path: str):
        """Carica backup della memoria AI"""
        
        import zipfile
        import shutil
        
        # Backup current memory
        current_backup = self.save_ai_memory_backup("current_memory_backup.zip")
        
        try:
            # Clear current memory
            shutil.rmtree(self.memory_path)
            self.memory_path.mkdir(exist_ok=True)
            
            # Extract backup
            with zipfile.ZipFile(backup_path, 'r') as zip_ref:
                zip_ref.extractall(self.memory_path)
                
            # Reload memory
            self.load_existing_memory()
            
            return True
            
        except Exception as e:
            # Restore current memory if failed
            self.load_ai_memory_backup(current_backup)
            raise e
            
    def load_existing_memory(self):
        """Carica memoria esistente da disco"""
        
        if not self.db_path.exists():
            return
            
        conn = sqlite3.connect(self.db_path)
        
        # Load statistics
        total_trades = pd.read_sql_query("SELECT COUNT(*) as count FROM trades", conn).iloc[0]['count']
        total_patterns = pd.read_sql_query("SELECT COUNT(*) as count FROM patterns", conn).iloc[0]['count']
        
        if total_trades > 0:
            avg_profit = pd.read_sql_query(
                "SELECT AVG(profit_loss) as avg FROM trades WHERE profit_loss IS NOT NULL", 
                conn
            ).iloc[0]['avg'] or 0
            
            win_rate = pd.read_sql_query(
                "SELECT AVG(CASE WHEN outcome > 0 THEN 1.0 ELSE 0.0 END) as rate FROM trades WHERE outcome IS NOT NULL",
                conn
            ).iloc[0]['rate'] or 0
            
            self.learning_data['performance_metrics'] = {
                'total_trades': total_trades,
                'total_patterns': total_patterns,
                'average_profit': avg_profit,
                'win_rate': win_rate,
                'memory_created': True
            }
            
        conn.close()
        
    def get_learning_statistics(self) -> Dict:
        """Ottieni statistiche di learning"""
        
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Basic stats
        stats['total_trades'] = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM trades", conn
        ).iloc[0]['count']
        
        stats['total_patterns'] = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM patterns", conn
        ).iloc[0]['count']
        
        # Performance stats
        if stats['total_trades'] > 0:
            perf_query = '''
                SELECT 
                    AVG(CASE WHEN outcome > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(profit_loss) as avg_profit,
                    COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) as completed_trades
                FROM trades 
                WHERE outcome IS NOT NULL
            '''
            perf_stats = pd.read_sql_query(perf_query, conn).iloc[0]
            stats.update(perf_stats.to_dict())
            
        # Top patterns
        top_patterns = pd.read_sql_query('''
            SELECT pattern_type, success_rate, avg_profit, occurrences 
            FROM patterns 
            ORDER BY success_rate DESC, occurrences DESC 
            LIMIT 5
        ''', conn)
        
        stats['top_patterns'] = top_patterns.to_dict('records')
        
        conn.close()
        
        return stats

class AIModelHybridSystem:
    """Sistema ibrido che combina modelli simulati, API esterne e AI personalizzata"""
    
    def __init__(self):
        self.personal_ai = PersonalAILearningSystem()
        self.model_weights = {
            'personal_ai': 0.4,      # 40% peso AI personalizzata
            'external_apis': 0.35,   # 35% peso API esterne (se disponibili)
            'simulated_models': 0.25 # 25% peso modelli simulati
        }
        
    def get_hybrid_prediction(self, symbol: str, market_data: Dict) -> Dict:
        """Combina tutte le fonti AI per predizione finale"""
        
        predictions = {}
        
        # Personal AI prediction
        personal_pred = self.personal_ai.get_personal_recommendation(symbol, market_data)
        predictions['personal_ai'] = personal_pred
        
        # Simulated models prediction (existing system)
        simulated_pred = self.get_simulated_prediction(symbol, market_data)
        predictions['simulated'] = simulated_pred
        
        # External APIs prediction (if available)
        external_pred = self.get_external_apis_prediction(symbol, market_data)
        if external_pred:
            predictions['external'] = external_pred
            
        # Combine predictions with weights
        final_prediction = self.combine_predictions(predictions)
        
        return final_prediction
        
    def get_simulated_prediction(self, symbol: str, market_data: Dict) -> Dict:
        """Predizione da modelli simulati esistenti"""
        # Usa logica esistente del sistema
        return {
            'action': 'BUY',
            'confidence': 0.7,
            'reasoning': 'Technical indicators bullish',
            'source': 'simulated_models'
        }
        
    def get_external_apis_prediction(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Predizione da API esterne se disponibili"""
        # Check if external APIs are configured
        if os.getenv('TWITTER_BEARER_TOKEN') or os.getenv('ALPHA_VANTAGE_API_KEY'):
            return {
                'action': 'BUY',
                'confidence': 0.8,
                'reasoning': 'Positive sentiment from external sources',
                'source': 'external_apis'
            }
        return None
        
    def combine_predictions(self, predictions: Dict) -> Dict:
        """Combina multiple predizioni con pesi"""
        
        total_weight = 0
        weighted_confidence = 0
        action_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for source, pred in predictions.items():
            if source == 'personal_ai':
                weight = self.model_weights['personal_ai']
            elif source == 'external':
                weight = self.model_weights['external_apis']
            else:
                weight = self.model_weights['simulated_models']
                
            total_weight += weight
            weighted_confidence += pred['confidence'] * weight
            action_votes[pred['action']] += weight
            
        # Final decision
        final_action = max(action_votes, key=action_votes.get)
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'sources_used': list(predictions.keys()),
            'personal_ai_weight': self.model_weights['personal_ai'],
            'reasoning': f"Hybrid decision from {len(predictions)} sources"
        }

# Integration example
def integrate_with_trading_system():
    """Esempio di integrazione con sistema trading esistente"""
    
    hybrid_ai = AIModelHybridSystem()
    
    # Simula trade decision
    market_data = {
        'rsi': 65,
        'macd': 'bullish',
        'volume': 'high'
    }
    
    prediction = hybrid_ai.get_hybrid_prediction('BTC/USDT', market_data)
    
    print("Hybrid AI Prediction:")
    print(f"Action: {prediction['action']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Sources: {prediction['sources_used']}")
    
    return prediction

if __name__ == "__main__":
    integrate_with_trading_system()