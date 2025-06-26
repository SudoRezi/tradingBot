#!/usr/bin/env python3
"""
Models Integration - Integra modelli HuggingFace nell'AI trading
Sistema per integrare modelli AI scaricati nelle decisioni di trading autonomo
"""

import os
import json
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime

class ModelsIntegration:
    """Integra modelli AI nelle decisioni di trading"""
    
    def __init__(self):
        self.models_db = "ai_models.db"
        
    def get_available_models(self) -> List[Dict]:
        """Ottiene modelli disponibili per trading"""
        
        if not os.path.exists(self.models_db):
            return []
            
        conn = sqlite3.connect(self.models_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM downloaded_models 
            WHERE status = "downloaded" 
            ORDER BY download_date DESC
        ''')
        
        rows = cursor.fetchall()
        models = []
        
        for row in rows:
            models.append({
                "model_name": row[1],
                "model_url": row[2],
                "model_type": row[4],
                "file_path": row[6],
                "model_info": json.loads(row[7]) if row[7] else {}
            })
            
        conn.close()
        return models
        
    def enhance_trading_decision(self, base_decision: Dict, symbol: str, market_data: Dict) -> Dict:
        """Potenzia decisione trading con modelli AI"""
        
        models = self.get_available_models()
        
        if not models:
            # Nessun modello disponibile, usa decisione base
            return {
                **base_decision,
                "ai_enhancement": False,
                "models_used": [],
                "enhancement_reason": "No AI models available"
            }
            
        # Usa modelli per analisi
        ai_predictions = []
        models_used = []
        
        for model in models:
            try:
                prediction = self.get_model_prediction(model, symbol, market_data)
                if prediction:
                    ai_predictions.append(prediction)
                    models_used.append(model["model_name"])
            except Exception as e:
                print(f"Error using model {model['model_name']}: {e}")
                
        if not ai_predictions:
            return {
                **base_decision,
                "ai_enhancement": False,
                "models_used": [],
                "enhancement_reason": "Models failed to generate predictions"
            }
            
        # Combina predizioni AI con decisione base
        enhanced_decision = self.combine_predictions(base_decision, ai_predictions)
        
        return {
            **enhanced_decision,
            "ai_enhancement": True,
            "models_used": models_used,
            "ai_predictions": ai_predictions,
            "enhancement_reason": f"Enhanced by {len(models_used)} AI models"
        }
        
    def get_model_prediction(self, model: Dict, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Ottiene predizione da un modello specifico"""
        
        model_type = model["model_type"]
        
        # Simula uso del modello basato sul tipo
        if model_type == "trading_decision":
            return self.trading_model_prediction(model, symbol, market_data)
        elif model_type == "financial_sentiment":
            return self.sentiment_model_prediction(model, symbol, market_data)
        elif model_type == "crypto_analysis":
            return self.crypto_model_prediction(model, symbol, market_data)
        elif model_type == "market_prediction":
            return self.market_model_prediction(model, symbol, market_data)
        else:
            return self.generic_model_prediction(model, symbol, market_data)
            
    def trading_model_prediction(self, model: Dict, symbol: str, market_data: Dict) -> Dict:
        """Predizione da modello trading specializzato"""
        
        # Simula CryptoTrader-LM o modelli simili
        rsi = market_data.get("rsi", 50)
        price_change = market_data.get("price_change_24h", 0)
        volume_ratio = market_data.get("volume_ratio", 1.0)
        
        # Logica specializzata per trading
        trading_score = 0.5
        
        # Pattern recognition (simulato)
        if rsi < 30 and price_change < -0.05:
            trading_score += 0.3  # Oversold dip buying
        elif rsi > 70 and price_change > 0.05:
            trading_score -= 0.3  # Overbought selling
            
        # Volume confirmation
        if volume_ratio > 2.0:
            if trading_score > 0.5:
                trading_score += 0.15  # Strong volume confirms bullish
            else:
                trading_score -= 0.15  # Strong volume confirms bearish
                
        # Market structure analysis (simulato)
        if symbol == "BTC" and trading_score > 0.6:
            trading_score += 0.1  # BTC leadership bias
            
        return {
            "model_name": model["model_name"],
            "model_type": "trading_decision",
            "prediction": {
                "action": "BUY" if trading_score > 0.65 else "SELL" if trading_score < 0.35 else "HOLD",
                "confidence": abs(trading_score - 0.5) * 2,
                "score": trading_score
            },
            "reasoning": f"Trading model analysis: RSI={rsi}, Volume={volume_ratio:.1f}x"
        }
        
    def sentiment_model_prediction(self, model: Dict, symbol: str, market_data: Dict) -> Dict:
        """Predizione da modello sentiment finanziario"""
        
        # Simula FinBERT o modelli sentiment
        news_sentiment = market_data.get("news_sentiment", 0.5)
        social_sentiment = market_data.get("social_sentiment", 0.5)
        
        # Combina sentiment con analisi tecnica
        combined_sentiment = (news_sentiment + social_sentiment) / 2
        
        # Peso sentiment basato su volatilità
        volatility = market_data.get("volatility", 0.02)
        sentiment_weight = min(0.4, volatility * 10)  # Più volatilità = più peso al sentiment
        
        sentiment_score = 0.5 + (combined_sentiment - 0.5) * sentiment_weight
        
        return {
            "model_name": model["model_name"],
            "model_type": "financial_sentiment", 
            "prediction": {
                "action": "BUY" if sentiment_score > 0.6 else "SELL" if sentiment_score < 0.4 else "HOLD",
                "confidence": abs(sentiment_score - 0.5) * 2,
                "score": sentiment_score
            },
            "reasoning": f"Sentiment analysis: News={news_sentiment:.2f}, Social={social_sentiment:.2f}"
        }
        
    def crypto_model_prediction(self, model: Dict, symbol: str, market_data: Dict) -> Dict:
        """Predizione da modello crypto specializzato"""
        
        # Simula CryptoBERT o modelli crypto
        price_change = market_data.get("price_change_24h", 0)
        volume_ratio = market_data.get("volume_ratio", 1.0)
        
        # Analisi crypto-specifica
        crypto_score = 0.5
        
        # Crypto momentum patterns
        if price_change > 0.1:  # Strong pump
            crypto_score += 0.25
        elif price_change < -0.1:  # Strong dump  
            crypto_score -= 0.25
            
        # Crypto volume analysis
        if volume_ratio > 3.0:  # Unusual volume
            crypto_score += 0.2 if price_change > 0 else -0.2
            
        # Symbol-specific analysis
        if symbol in ["BTC", "ETH"]:
            crypto_score += 0.05  # Blue chip bias
        elif symbol in ["DOGE", "SHIB"]:
            crypto_score *= 0.9  # Meme coin volatility discount
            
        return {
            "model_name": model["model_name"],
            "model_type": "crypto_analysis",
            "prediction": {
                "action": "BUY" if crypto_score > 0.6 else "SELL" if crypto_score < 0.4 else "HOLD",
                "confidence": abs(crypto_score - 0.5) * 2,
                "score": crypto_score
            },
            "reasoning": f"Crypto model: {symbol} momentum and volume analysis"
        }
        
    def market_model_prediction(self, model: Dict, symbol: str, market_data: Dict) -> Dict:
        """Predizione da modello market prediction"""
        
        # Simula modelli di predizione mercato
        rsi = market_data.get("rsi", 50)
        macd = market_data.get("macd", 0)
        
        market_score = 0.5
        
        # Technical indicators combination
        if rsi < 35 and macd > 0:
            market_score += 0.2  # Bullish divergence
        elif rsi > 65 and macd < 0:
            market_score -= 0.2  # Bearish divergence
            
        # Market regime detection (simulato)
        volatility = market_data.get("volatility", 0.02)
        if volatility > 0.05:  # High volatility regime
            market_score *= 0.8  # Reduce confidence in high vol
            
        return {
            "model_name": model["model_name"],
            "model_type": "market_prediction",
            "prediction": {
                "action": "BUY" if market_score > 0.58 else "SELL" if market_score < 0.42 else "HOLD",
                "confidence": abs(market_score - 0.5) * 2,
                "score": market_score
            },
            "reasoning": f"Market prediction: RSI={rsi}, MACD={macd:.3f}"
        }
        
    def generic_model_prediction(self, model: Dict, symbol: str, market_data: Dict) -> Dict:
        """Predizione generica per modelli non specializzati"""
        
        # Analisi generica
        price_change = market_data.get("price_change_24h", 0)
        
        generic_score = 0.5 + price_change * 2  # Simple momentum
        generic_score = max(0.1, min(0.9, generic_score))  # Clamp
        
        return {
            "model_name": model["model_name"],
            "model_type": model["model_type"],
            "prediction": {
                "action": "BUY" if generic_score > 0.6 else "SELL" if generic_score < 0.4 else "HOLD",
                "confidence": 0.5,  # Lower confidence for generic models
                "score": generic_score
            },
            "reasoning": f"Generic model analysis for {symbol}"
        }
        
    def combine_predictions(self, base_decision: Dict, ai_predictions: List[Dict]) -> Dict:
        """Combina decisione base con predizioni AI usando conflict resolution"""
        
        # Conflict Detection and Resolution
        conflict_detected = self.detect_model_conflicts(ai_predictions)
        
        if conflict_detected:
            # Apply conflict resolution strategies
            resolved_predictions = self.resolve_model_conflicts(ai_predictions)
            ai_predictions = resolved_predictions
        
        # Pesi per combinazione
        base_weight = 0.4  # 40% decisione base
        ai_weight = 0.6    # 60% AI models
        
        # Ottieni score base
        base_score = base_decision.get("technical_score", 0.5)
        
        # Calcola score AI combinato con consensus weighting
        ai_scores = [pred["prediction"]["score"] for pred in ai_predictions]
        ai_confidences = [pred["prediction"]["confidence"] for pred in ai_predictions]
        
        # Consensus-based weighting to reduce conflicts
        consensus_scores, consensus_weights = self.calculate_consensus_weights(ai_scores, ai_confidences)
        
        # Weighted average dei modelli AI
        if consensus_scores:
            weighted_ai_score = sum(score * weight for score, weight in zip(consensus_scores, consensus_weights))
            total_weight = sum(consensus_weights)
            
            if total_weight > 0:
                avg_ai_score = weighted_ai_score / total_weight
            else:
                avg_ai_score = sum(ai_scores) / len(ai_scores)
        else:
            avg_ai_score = 0.5
            
        # Combina base + AI
        combined_score = base_score * base_weight + avg_ai_score * ai_weight
        
        # Determina azione finale
        if combined_score > 0.65:
            final_action = "BUY"
            final_confidence = min(0.95, combined_score)
        elif combined_score < 0.35:
            final_action = "SELL"
            final_confidence = min(0.95, 1 - combined_score)
        else:
            final_action = "HOLD"
            final_confidence = 0.6
            
        # Bonus confidence se base e AI concordano
        base_action = base_decision.get("action", "HOLD")
        if base_action == final_action:
            final_confidence = min(0.98, final_confidence * 1.1)
            
        return {
            "action": final_action,
            "confidence": final_confidence,
            "combined_score": combined_score,
            "base_score": base_score,
            "ai_score": avg_ai_score,
            "conflict_detected": conflict_detected,
            "models_consensus": len([s for s in ai_scores if abs(s - avg_ai_score) < 0.1]) / len(ai_scores) if ai_scores else 1.0,
            "reasoning": f"Enhanced decision: Base({base_weight:.0%}) + AI({ai_weight:.0%}) = {final_action}" + 
                        (f" (Conflicts resolved)" if conflict_detected else "")
        }

    def detect_model_conflicts(self, ai_predictions: List[Dict]) -> bool:
        """Rileva conflitti tra predizioni dei modelli"""
        if len(ai_predictions) < 2:
            return False
            
        actions = [pred["prediction"]["action"] for pred in ai_predictions]
        scores = [pred["prediction"]["score"] for pred in ai_predictions]
        
        # Conflitto se azioni diverse
        unique_actions = set(actions)
        if len(unique_actions) > 1:
            return True
            
        # Conflitto se score troppo diversi (>0.3 differenza)
        if len(scores) > 1:
            score_range = max(scores) - min(scores)
            if score_range > 0.3:
                return True
                
        return False
        
    def resolve_model_conflicts(self, ai_predictions: List[Dict]) -> List[Dict]:
        """Risolve conflitti tra modelli usando diverse strategie"""
        
        # Strategy 1: Weight by model type specialization
        specialized_weights = {
            'trading_decision': 1.5,  # Higher weight for trading-specific models
            'financial_sentiment': 1.2,
            'crypto_analysis': 1.3,
            'market_prediction': 1.1,
            'news_analysis': 0.9,
            'risk_analysis': 1.0
        }
        
        # Strategy 2: Confidence-based filtering
        high_confidence_preds = [
            pred for pred in ai_predictions 
            if pred["prediction"]["confidence"] > 0.7
        ]
        
        if high_confidence_preds:
            # Use only high-confidence predictions
            working_preds = high_confidence_preds
        else:
            # Use all predictions but adjust weights
            working_preds = ai_predictions
            
        # Strategy 3: Adjust weights based on specialization
        for pred in working_preds:
            model_type = pred.get("model_type", "unknown")
            specialization_weight = specialized_weights.get(model_type, 1.0)
            
            # Adjust confidence based on specialization
            original_conf = pred["prediction"]["confidence"]
            adjusted_conf = min(0.95, original_conf * specialization_weight)
            pred["prediction"]["confidence"] = adjusted_conf
            
        return working_preds
        
    def calculate_consensus_weights(self, scores: List[float], confidences: List[float]) -> tuple:
        """Calcola pesi basati sul consensus per ridurre conflitti"""
        
        if not scores:
            return [], []
            
        # Calculate consensus score (median)
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        consensus_score = sorted_scores[n//2] if n % 2 == 1 else (sorted_scores[n//2-1] + sorted_scores[n//2]) / 2
        
        # Weight models based on how close they are to consensus
        consensus_weights = []
        for i, (score, conf) in enumerate(zip(scores, confidences)):
            # Distance from consensus (closer = higher weight)
            distance = abs(score - consensus_score)
            consensus_factor = max(0.1, 1.0 - distance)  # Min weight 0.1
            
            # Combine with original confidence
            final_weight = conf * consensus_factor
            consensus_weights.append(final_weight)
            
        return scores, consensus_weights

def test_models_integration():
    """Test integrazione modelli"""
    
    print("Testing Models Integration")
    print("=" * 30)
    
    integration = ModelsIntegration()
    
    # Test modelli disponibili
    models = integration.get_available_models()
    print(f"Available models: {len(models)}")
    
    for model in models:
        print(f"  • {model['model_name']} ({model['model_type']})")
        
    # Test enhance decision
    base_decision = {
        "action": "BUY",
        "confidence": 0.7,
        "technical_score": 0.72,
        "reasoning": "Technical analysis suggests BUY"
    }
    
    market_data = {
        "rsi": 28,
        "price_change_24h": 0.08,
        "volume_ratio": 2.5,
        "volatility": 0.04,
        "news_sentiment": 0.75,
        "social_sentiment": 0.65,
        "macd": 0.05
    }
    
    enhanced = integration.enhance_trading_decision(base_decision, "BTC", market_data)
    
    print(f"\nEnhanced Decision:")
    print(f"  Action: {enhanced['action']}")
    print(f"  Confidence: {enhanced['confidence']:.2%}")
    print(f"  AI Enhanced: {enhanced['ai_enhancement']}")
    print(f"  Models Used: {enhanced.get('models_used', [])}")
    print(f"  Reasoning: {enhanced['reasoning']}")
    
    print("\nModels Integration Test Complete!")

if __name__ == "__main__":
    test_models_integration()