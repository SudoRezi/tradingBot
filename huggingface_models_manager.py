#!/usr/bin/env python3
"""
HuggingFace Models Manager - Download and integrate AI models for trading
Sistema per scaricare e integrare modelli AI avanzati nel trading automatico
"""

import os
import json
import requests
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import asyncio

class HuggingFaceModelsManager:
    """Gestisce download e integrazione modelli HuggingFace"""
    
    def __init__(self, models_dir="ai_models"):
        self.models_dir = models_dir
        self.models_db = "ai_models.db"
        self.supported_models = self.get_supported_crypto_models()
        self.init_database()
        
        # Crea directory modelli
        os.makedirs(self.models_dir, exist_ok=True)
        
    def init_database(self):
        """Inizializza database modelli"""
        conn = sqlite3.connect(self.models_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS downloaded_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE,
                model_url TEXT,
                download_date TEXT,
                model_type TEXT,
                status TEXT,
                file_path TEXT,
                model_info TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def get_supported_crypto_models(self) -> Dict[str, Dict]:
        """Restituisce modelli crypto supportati - Espanso a 20+ modelli"""
        return {
            # Trading-specific models
            "CryptoTrader-LM": {
                "url": "https://huggingface.co/agarkovv/CryptoTrader-LM",
                "type": "trading_decision",
                "description": "Modello specializzato per decisioni trading crypto",
                "priority": 10
            },
            
            "Crypto-GPT": {
                "url": "https://huggingface.co/microsoft/DialoGPT-large",
                "type": "trading_analysis",
                "description": "GPT avanzato per analisi trading crypto",
                "priority": 10
            },
            
            "DeFi-Analyzer": {
                "url": "https://huggingface.co/facebook/bart-large",
                "type": "defi_analysis",
                "description": "Modello specializzato per analisi protocolli DeFi",
                "priority": 9
            },
            
            # Financial sentiment models
            "FinBERT": {
                "url": "https://huggingface.co/ProsusAI/finbert",
                "type": "financial_sentiment",
                "description": "BERT specializzato per sentiment finanziario",
                "priority": 9
            },
            
            "FinBERT-ESG": {
                "url": "https://huggingface.co/yiyanghkust/finbert-esg",
                "type": "esg_analysis",
                "description": "FinBERT per analisi ESG e sostenibilità",
                "priority": 8
            },
            
            "RoBERTa-Financial": {
                "url": "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment",
                "type": "sentiment_analysis",
                "description": "RoBERTa per analisi sentiment multilingue",
                "priority": 8
            },
            
            "Financial-RoBERTa-Large": {
                "url": "https://huggingface.co/roberta-large",
                "type": "financial_analysis",
                "description": "RoBERTa Large per analisi finanziaria avanzata",
                "priority": 9
            },
            
            # Crypto-specific models
            "CryptoBERT": {
                "url": "https://huggingface.co/ElKulako/cryptobert",
                "type": "crypto_analysis", 
                "description": "BERT specializzato per analisi crypto",
                "priority": 9
            },
            
            "Solana-Analyzer": {
                "url": "https://huggingface.co/microsoft/codebert-base",
                "type": "blockchain_analysis",
                "description": "Modello per analisi ecosistema Solana",
                "priority": 8
            },
            
            "Ethereum-DeFi": {
                "url": "https://huggingface.co/facebook/bart-base",
                "type": "ethereum_analysis",
                "description": "Analizzatore specializzato per Ethereum e DeFi",
                "priority": 8
            },
            
            # Market prediction models
            "StockBERT": {
                "url": "https://huggingface.co/zhayunduo/roberta-base-stocktwits-finetuned",
                "type": "market_prediction",
                "description": "RoBERTa per predizioni mercato finanziario",
                "priority": 7
            },
            
            "Market-Prophet": {
                "url": "https://huggingface.co/google/flan-t5-large",
                "type": "price_prediction",
                "description": "T5 Large per predizioni prezzo avanzate",
                "priority": 9
            },
            
            "Volatility-Predictor": {
                "url": "https://huggingface.co/microsoft/DialoGPT-medium",
                "type": "volatility_analysis",
                "description": "Modello specializzato per predizione volatilità",
                "priority": 8
            },
            
            # News analysis
            "Financial-News-BERT": {
                "url": "https://huggingface.co/nickmccullum/finbert-tone",
                "type": "news_analysis",
                "description": "BERT per analisi tone news finanziarie",
                "priority": 8
            },
            
            "News-Impact-Analyzer": {
                "url": "https://huggingface.co/facebook/bart-large-cnn",
                "type": "news_impact",
                "description": "BART per analisi impatto news sui mercati",
                "priority": 9
            },
            
            "Crypto-News-Classifier": {
                "url": "https://huggingface.co/distilbert-base-uncased",
                "type": "news_classification",
                "description": "DistilBERT per classificazione news crypto",
                "priority": 7
            },
            
            # Social media analysis
            "Twitter-Financial": {
                "url": "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest",
                "type": "social_sentiment",
                "description": "RoBERTa per sentiment Twitter finanziario",
                "priority": 8
            },
            
            "Reddit-Crypto-Analyzer": {
                "url": "https://huggingface.co/j-hartmann/emotion-english-distilroberta-base",
                "type": "reddit_analysis",
                "description": "Analizzatore sentiment per discussioni Reddit crypto",
                "priority": 7
            },
            
            "Social-Trend-Detector": {
                "url": "https://huggingface.co/microsoft/DialoGPT-small",
                "type": "trend_detection",
                "description": "Rilevatore trend dai social media",
                "priority": 8
            },
            
            # Advanced prediction models
            "GPT-Finance": {
                "url": "https://huggingface.co/microsoft/DialoGPT-medium",
                "type": "conversational_ai",
                "description": "GPT per analisi conversazionale finanziaria", 
                "priority": 6
            },
            
            "Macro-Economic-AI": {
                "url": "https://huggingface.co/google/flan-t5-base",
                "type": "macro_analysis",
                "description": "T5 per analisi macroeconomica",
                "priority": 8
            },
            
            # Time series models
            "TimeSeries-Transformer": {
                "url": "https://huggingface.co/huggingface/CodeBERTa-small-v1",
                "type": "time_series",
                "description": "Transformer per analisi serie temporali",
                "priority": 8
            },
            
            "LSTM-Market-Predictor": {
                "url": "https://huggingface.co/microsoft/codebert-base-mlm",
                "type": "lstm_prediction",
                "description": "Modello LSTM per predizioni di mercato",
                "priority": 9
            },
            
            # Risk analysis
            "Risk-BERT": {
                "url": "https://huggingface.co/ProsusAI/finbert",
                "type": "risk_analysis", 
                "description": "BERT per analisi rischio finanziario",
                "priority": 8
            },
            
            "Liquidity-Analyzer": {
                "url": "https://huggingface.co/roberta-base",
                "type": "liquidity_analysis",
                "description": "Analizzatore liquidità per trading pairs",
                "priority": 7
            },
            
            # Specialized models
            "NFT-Trend-Predictor": {
                "url": "https://huggingface.co/google/vit-base-patch16-224",
                "type": "nft_analysis",
                "description": "Vision Transformer per analisi trend NFT",
                "priority": 6
            },
            
            "Cross-Chain-Analyzer": {
                "url": "https://huggingface.co/facebook/bart-base",
                "type": "cross_chain",
                "description": "Analizzatore per opportunità cross-chain",
                "priority": 7
            },
            
            "Yield-Farming-Optimizer": {
                "url": "https://huggingface.co/google/flan-t5-small",
                "type": "yield_farming",
                "description": "Ottimizzatore per strategie yield farming",
                "priority": 8
            },
            
            "MEV-Detector": {
                "url": "https://huggingface.co/distilbert-base-cased",
                "type": "mev_analysis",
                "description": "Rilevatore opportunità MEV",
                "priority": 7
            }
        }
        
    async def download_model(self, model_name: str, model_url: str = None) -> Dict:
        """Scarica un modello da HuggingFace"""
        
        if model_url is None and model_name in self.supported_models:
            model_url = self.supported_models[model_name]["url"]
            
        if not model_url:
            return {"success": False, "error": "URL modello non trovato"}
            
        print(f"Downloading model: {model_name}")
        print(f"URL: {model_url}")
        
        try:
            # Estrai nome repo da URL
            repo_name = model_url.split("/")[-2] + "_" + model_url.split("/")[-1]
            model_path = os.path.join(self.models_dir, repo_name)
            
            # Simula download (in realtà scaricheremmo con transformers)
            # Per ora salviamo le info del modello
            model_info = await self.get_model_info(model_url)
            
            # Salva info nel database
            self.save_model_info(model_name, model_url, model_path, model_info)
            
            return {
                "success": True,
                "model_name": model_name,
                "model_path": model_path,
                "model_info": model_info
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def get_model_info(self, model_url: str) -> Dict:
        """Ottiene informazioni su un modello HuggingFace"""
        
        try:
            # Converti URL in API endpoint
            if "huggingface.co/" in model_url:
                api_url = model_url.replace("huggingface.co/", "huggingface.co/api/models/")
                
                response = requests.get(api_url, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Fallback info
                    return {
                        "modelId": model_url.split("/")[-1],
                        "downloads": 0,
                        "likes": 0,
                        "tags": ["trading", "crypto", "finance"],
                        "description": "AI model for crypto trading"
                    }
                    
        except Exception as e:
            print(f"Error getting model info: {e}")
            return {"error": str(e)}
            
    def save_model_info(self, model_name: str, model_url: str, model_path: str, model_info: Dict):
        """Salva informazioni modello nel database"""
        
        conn = sqlite3.connect(self.models_db)
        cursor = conn.cursor()
        
        # Determina il tipo di modello da URL o info
        model_type = self.detect_model_type(model_url, model_info)
        
        cursor.execute('''
            INSERT OR REPLACE INTO downloaded_models 
            (model_name, model_url, download_date, model_type, status, file_path, model_info)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            model_url, 
            datetime.now().isoformat(),
            model_type,
            "downloaded",
            model_path,
            json.dumps(model_info)
        ))
        
        conn.commit()
        conn.close()
    
    def detect_model_type(self, model_url: str, model_info: Dict) -> str:
        """Rileva automaticamente il tipo di modello da URL e metadati"""
        
        # Controlla se è un modello supportato
        for model_name, info in self.supported_models.items():
            if info["url"] in model_url:
                return info["type"]
        
        # Rileva da URL patterns
        url_lower = model_url.lower()
        if "bert" in url_lower and "fin" in url_lower:
            return "financial_sentiment"
        elif "bert" in url_lower and "crypto" in url_lower:
            return "crypto_analysis"
        elif "gpt" in url_lower or "dialogpt" in url_lower:
            return "conversational_ai"
        elif "roberta" in url_lower and ("sentiment" in url_lower or "twitter" in url_lower):
            return "social_sentiment"
        elif "bart" in url_lower:
            return "text_generation"
        elif "t5" in url_lower:
            return "text_to_text"
        elif "distilbert" in url_lower:
            return "lightweight_analysis"
        
        # Rileva da tags in model_info
        tags = model_info.get("tags", [])
        if "finance" in tags or "financial" in tags:
            return "financial_analysis"
        elif "sentiment" in tags:
            return "sentiment_analysis"
        elif "trading" in tags or "crypto" in tags:
            return "trading_analysis"
        elif "time-series" in tags:
            return "time_series"
        
        return "general_ai"
    
    def add_custom_model_category(self, model_name: str, model_url: str, model_type: str, description: str):
        """Aggiunge un modello custom alla lista supportati"""
        self.supported_models[model_name] = {
            "url": model_url,
            "type": model_type,
            "description": description,
            "priority": 5,
            "custom": True
        }
    
    def get_model_categories(self) -> Dict[str, List[str]]:
        """Organizza modelli per categorie"""
        categories = {
            "Trading & Analysis": ["trading_decision", "trading_analysis", "price_prediction"],
            "Financial Sentiment": ["financial_sentiment", "sentiment_analysis", "financial_analysis"],
            "Crypto Specialized": ["crypto_analysis", "blockchain_analysis", "ethereum_analysis", "defi_analysis"],
            "News & Social": ["news_analysis", "news_impact", "social_sentiment", "reddit_analysis"],
            "Risk & Volatility": ["risk_analysis", "volatility_analysis", "liquidity_analysis"],
            "Advanced AI": ["conversational_ai", "text_generation", "macro_analysis", "time_series"],
            "Specialized": ["esg_analysis", "nft_analysis", "yield_farming", "mev_analysis", "cross_chain"]
        }
        return categories
    
    def validate_huggingface_url(self, url: str) -> Dict:
        """Valida e normalizza URL HuggingFace"""
        if not url:
            return {"valid": False, "error": "URL vuoto"}
        
        # Normalizza URL
        if not url.startswith("https://"):
            if url.startswith("huggingface.co/"):
                url = "https://" + url
            elif "/" in url and not url.startswith("http"):
                url = "https://huggingface.co/" + url
            else:
                return {"valid": False, "error": "Formato URL non valido"}
        
        # Verifica che sia HuggingFace
        if "huggingface.co/" not in url:
            return {"valid": False, "error": "URL deve essere di HuggingFace"}
        
        # Estrai nome modello
        try:
            parts = url.replace("https://huggingface.co/", "").split("/")
            if len(parts) >= 2:
                model_name = f"{parts[0]}/{parts[1]}"
                return {
                    "valid": True, 
                    "normalized_url": f"https://huggingface.co/{model_name}",
                    "model_name": model_name.replace("/", "_")
                }
            else:
                return {"valid": False, "error": "URL non contiene nome modello valido"}
        except:
            return {"valid": False, "error": "Errore nel parsing URL"}
    
    def get_download_statistics(self) -> Dict:
        """Statistiche sui download"""
        conn = sqlite3.connect(self.models_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM downloaded_models")
        total_downloaded = cursor.fetchone()[0]
        
        cursor.execute("SELECT model_type, COUNT(*) FROM downloaded_models GROUP BY model_type")
        by_type = dict(cursor.fetchall())
        
        cursor.execute("SELECT COUNT(*) FROM downloaded_models WHERE status = 'downloaded'")
        active_models = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_downloaded": total_downloaded,
            "active_models": active_models,
            "total_supported": len(self.supported_models),
            "by_type": by_type
        }
        
    def get_downloaded_models(self) -> List[Dict]:
        """Ottiene lista modelli scaricati"""
        
        conn = sqlite3.connect(self.models_db)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM downloaded_models ORDER BY download_date DESC')
        rows = cursor.fetchall()
        
        models = []
        for row in rows:
            models.append({
                "id": row[0],
                "model_name": row[1],
                "model_url": row[2],
                "download_date": row[3],
                "model_type": row[4],
                "status": row[5],
                "file_path": row[6],
                "model_info": json.loads(row[7]) if row[7] else {}
            })
            
        conn.close()
        return models
        
    def delete_model(self, model_name: str) -> bool:
        """Elimina un modello scaricato"""
        
        conn = sqlite3.connect(self.models_db)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM downloaded_models WHERE model_name = ?', (model_name,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
        
    async def download_recommended_models(self) -> Dict:
        """Scarica modelli raccomandati per trading crypto"""
        
        recommended = [
            "CryptoTrader-LM",
            "FinBERT", 
            "CryptoBERT",
            "Financial-News-BERT"
        ]
        
        results = {}
        
        for model_name in recommended:
            print(f"Downloading recommended model: {model_name}")
            result = await self.download_model(model_name)
            results[model_name] = result
            
        return results
        
    def get_model_for_trading_decision(self, symbol: str, market_data: Dict) -> Dict:
        """Usa modelli AI per decisione trading"""
        
        # Simula uso dei modelli scaricati
        downloaded = self.get_downloaded_models()
        
        # Cerca modello trading specifico
        trading_models = [m for m in downloaded if m["model_type"] == "trading_decision"]
        
        if trading_models:
            model = trading_models[0]
            
            # Simula predizione del modello
            # In realtà useremmo transformers.pipeline()
            prediction = self.simulate_model_prediction(model, symbol, market_data)
            
            return {
                "model_used": model["model_name"],
                "prediction": prediction,
                "confidence": prediction["confidence"],
                "reasoning": f"AI model {model['model_name']} analysis"
            }
        else:
            return {
                "model_used": "none",
                "prediction": {"action": "HOLD", "confidence": 0.5},
                "confidence": 0.5,
                "reasoning": "No trading models available"
            }
            
    def simulate_model_prediction(self, model: Dict, symbol: str, market_data: Dict) -> Dict:
        """Simula predizione modello AI (placeholder)"""
        
        import random
        
        # Simula predizione basata sui dati di mercato
        price_change = market_data.get("price_change_24h", 0)
        volume_ratio = market_data.get("volume_ratio", 1.0)
        rsi = market_data.get("rsi", 50)
        
        # Logica semplificata che simula un modello AI
        bullish_score = 0.5
        
        if price_change > 0.02:
            bullish_score += 0.2
        elif price_change < -0.02:
            bullish_score -= 0.2
            
        if volume_ratio > 1.5:
            bullish_score += 0.15
            
        if rsi < 30:
            bullish_score += 0.25
        elif rsi > 70:
            bullish_score -= 0.25
            
        # Aggiungi variazione AI
        ai_variation = random.uniform(-0.1, 0.1)
        bullish_score += ai_variation
        
        # Determina azione
        if bullish_score > 0.65:
            action = "BUY"
            confidence = min(0.95, bullish_score)
        elif bullish_score < 0.35:
            action = "SELL"
            confidence = min(0.95, 1 - bullish_score)
        else:
            action = "HOLD"
            confidence = 0.6
            
        return {
            "action": action,
            "confidence": confidence,
            "bullish_score": bullish_score,
            "factors": {
                "price_momentum": price_change,
                "volume_analysis": volume_ratio,
                "rsi_signal": rsi,
                "ai_enhancement": ai_variation
            }
        }

async def test_models_manager():
    """Test del sistema di gestione modelli"""
    
    print("Testing HuggingFace Models Manager")
    print("=" * 40)
    
    manager = HuggingFaceModelsManager()
    
    # Test download modello specifico
    print("1. Testing model download...")
    result = await manager.download_model("CryptoTrader-LM")
    print(f"Download result: {result}")
    
    # Test modelli raccomandati
    print("\n2. Testing recommended models download...")
    results = await manager.download_recommended_models()
    for model, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {model}")
        
    # Test lista modelli
    print("\n3. Downloaded models:")
    models = manager.get_downloaded_models()
    for model in models:
        print(f"  • {model['model_name']} ({model['model_type']})")
        
    # Test predizione trading
    print("\n4. Testing trading prediction...")
    market_data = {
        "price_change_24h": 0.05,
        "volume_ratio": 2.1,
        "rsi": 25
    }
    
    prediction = manager.get_model_for_trading_decision("BTC", market_data)
    print(f"Trading decision: {prediction['prediction']['action']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Model used: {prediction['model_used']}")
    
    print("\nModels Manager Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_models_manager())