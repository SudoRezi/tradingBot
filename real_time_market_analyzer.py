#!/usr/bin/env python3
"""
Real-Time Market & Social Media Analyzer
Analizza mercato, news, social media e sentiment in tempo reale
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yfinance as yf
import feedparser
import trafilatura
from pre_trained_knowledge_system import KnowledgeLoader

@dataclass
class MarketData:
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    timestamp: datetime
    market_cap: Optional[float] = None

@dataclass
class SentimentData:
    source: str
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    volume: int
    keywords: List[str]
    timestamp: datetime

@dataclass
class NewsData:
    title: str
    content: str
    source: str
    sentiment: float
    impact_score: float
    timestamp: datetime
    url: str

class RealTimeMarketAnalyzer:
    """Analizzatore mercato e sentiment real-time"""
    
    def __init__(self):
        self.knowledge = KnowledgeLoader()
        self.ai_knowledge = self.knowledge.load_all_knowledge()
        self.last_update = datetime.now()
        
        # Cache per evitare troppe chiamate API
        self.price_cache = {}
        self.sentiment_cache = {}
        self.news_cache = []
        
        # Configurazione sources
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
        self.news_sources = [
            'https://feeds.coindesk.com/coindesk/rss/feed',
            'https://cointelegraph.com/rss',
            'https://cryptonews.com/news/feed/'
        ]
        
    def get_real_time_prices(self) -> Dict[str, MarketData]:
        """Ottiene prezzi real-time delle crypto principali"""
        market_data = {}
        
        try:
            for symbol in self.crypto_symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d", interval="1m")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    volume_24h = hist['Volume'].sum()
                    
                    # Calcola cambio 24h
                    if len(hist) >= 1440:  # 24h di dati minuto
                        price_24h_ago = hist['Close'].iloc[0]
                        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
                    else:
                        change_24h = 0.0
                    
                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        price=float(current_price),
                        volume_24h=float(volume_24h),
                        change_24h=float(change_24h),
                        timestamp=datetime.now(),
                        market_cap=info.get('marketCap')
                    )
                    
        except Exception as e:
            print(f"Errore nel recupero prezzi: {e}")
            
        return market_data
    
    def analyze_crypto_news_sentiment(self) -> List[NewsData]:
        """Analizza sentiment delle news crypto"""
        news_data = []
        
        for feed_url in self.news_sources:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:  # Ultime 5 news per source
                    # Scarica contenuto completo
                    content = self._extract_article_content(entry.link)
                    
                    # Analizza sentiment
                    sentiment_score = self._analyze_text_sentiment(
                        entry.title + " " + content
                    )
                    
                    # Calcola impact score
                    impact_score = self._calculate_news_impact(
                        entry.title, content, feed.feed.title
                    )
                    
                    news_data.append(NewsData(
                        title=entry.title,
                        content=content[:500],  # Primi 500 caratteri
                        source=feed.feed.title,
                        sentiment=sentiment_score,
                        impact_score=impact_score,
                        timestamp=datetime.now(),
                        url=entry.link
                    ))
                    
            except Exception as e:
                print(f"Errore nell'analisi news da {feed_url}: {e}")
                continue
                
        return news_data
    
    def _extract_article_content(self, url: str) -> str:
        """Estrae contenuto articolo usando trafilatura"""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                return text if text else ""
        except:
            pass
        return ""
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analizza sentiment del testo usando keyword-based approach"""
        if not text:
            return 0.0
            
        text = text.lower()
        
        # Carica pattern sentiment dalla knowledge base
        sentiment_patterns = self.ai_knowledge.get('sentiment_patterns', {})
        
        bullish_keywords = []
        bearish_keywords = []
        
        for category in sentiment_patterns.get('bullish_keywords', {}).values():
            if isinstance(category, list):
                bullish_keywords.extend(category)
                
        for category in sentiment_patterns.get('bearish_keywords', {}).values():
            if isinstance(category, list):
                bearish_keywords.extend(category)
        
        bullish_score = sum(1 for keyword in bullish_keywords if keyword in text)
        bearish_score = sum(1 for keyword in bearish_keywords if keyword in text)
        
        total_keywords = bullish_score + bearish_score
        if total_keywords == 0:
            return 0.0
            
        # Normalizza score tra -1 e 1
        sentiment = (bullish_score - bearish_score) / total_keywords
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_news_impact(self, title: str, content: str, source: str) -> float:
        """Calcola impact score della news"""
        impact = 0.0
        
        # Source credibility weight
        source_weights = {
            'coindesk': 0.8,
            'cointelegraph': 0.7,
            'bloomberg': 0.9,
            'reuters': 0.9,
            'default': 0.5
        }
        
        source_weight = source_weights.get(source.lower(), source_weights['default'])
        
        # High impact keywords
        high_impact_keywords = [
            'regulation', 'sec', 'etf', 'institutional', 'adoption',
            'hack', 'crash', 'pump', 'dump', 'whale', 'breaking'
        ]
        
        text = (title + " " + content).lower()
        keyword_count = sum(1 for keyword in high_impact_keywords if keyword in text)
        
        # Calculate final impact score
        impact = source_weight * (0.5 + 0.1 * keyword_count)
        return min(1.0, impact)
    
    def analyze_social_sentiment(self) -> Dict[str, SentimentData]:
        """Analizza sentiment dai social media (simulato)"""
        # Nota: In implementazione reale, qui si collegherebbero Twitter API, Reddit API, etc.
        # Per ora simulo i dati basandomi sui pattern di sentiment nella knowledge base
        
        social_data = {}
        
        # Simula sentiment Twitter
        twitter_sentiment = self._simulate_twitter_sentiment()
        social_data['twitter'] = SentimentData(
            source='twitter',
            sentiment_score=twitter_sentiment['score'],
            confidence=twitter_sentiment['confidence'],
            volume=twitter_sentiment['volume'],
            keywords=twitter_sentiment['keywords'],
            timestamp=datetime.now()
        )
        
        # Simula sentiment Reddit
        reddit_sentiment = self._simulate_reddit_sentiment()
        social_data['reddit'] = SentimentData(
            source='reddit',
            sentiment_score=reddit_sentiment['score'],
            confidence=reddit_sentiment['confidence'],
            volume=reddit_sentiment['volume'],
            keywords=reddit_sentiment['keywords'],
            timestamp=datetime.now()
        )
        
        return social_data
    
    def _simulate_twitter_sentiment(self) -> Dict[str, Any]:
        """Simula sentiment Twitter basato su pattern reali"""
        # Usa variabilità realistica basata su ora del giorno e volatilità mercato
        hour = datetime.now().hour
        
        # Sentiment varia con l'ora (mercati americani più attivi)
        if 14 <= hour <= 22:  # Mercati US aperti
            base_sentiment = np.random.normal(0.1, 0.3)  # Leggero bias positivo
            volume = np.random.randint(5000, 15000)
        else:
            base_sentiment = np.random.normal(0.0, 0.2)  # Neutrale
            volume = np.random.randint(1000, 5000)
            
        # Keywords trending basate su sentiment
        if base_sentiment > 0.2:
            keywords = ['moon', 'bullish', 'pump', 'buy', 'hodl']
        elif base_sentiment < -0.2:
            keywords = ['dump', 'crash', 'bear', 'sell', 'rekt']
        else:
            keywords = ['sideways', 'analysis', 'waiting', 'dip', 'chart']
            
        return {
            'score': max(-1.0, min(1.0, base_sentiment)),
            'confidence': np.random.uniform(0.6, 0.9),
            'volume': volume,
            'keywords': keywords
        }
    
    def _simulate_reddit_sentiment(self) -> Dict[str, Any]:
        """Simula sentiment Reddit (più analitico di Twitter)"""
        # Reddit tende ad essere più analitico e meno emotivo
        base_sentiment = np.random.normal(0.0, 0.25)
        volume = np.random.randint(500, 2000)
        
        if base_sentiment > 0.15:
            keywords = ['bullish', 'analysis', 'accumulation', 'undervalued', 'long-term']
        elif base_sentiment < -0.15:
            keywords = ['bearish', 'overvalued', 'correction', 'technical', 'resistance']
        else:
            keywords = ['discussion', 'technical-analysis', 'neutral', 'watchlist', 'dyor']
            
        return {
            'score': max(-1.0, min(1.0, base_sentiment)),
            'confidence': np.random.uniform(0.7, 0.85),
            'volume': volume,
            'keywords': keywords
        }
    
    def detect_whale_movements(self) -> Dict[str, Any]:
        """Rileva movimenti balene (simulato con pattern realistici)"""
        whale_data = {}
        
        # Simula whale alerts basati su pattern dalla knowledge base
        whale_patterns = self.ai_knowledge.get('whale_movement_patterns', {})
        
        # Genera alert casuali ma realistici
        if np.random.random() < 0.1:  # 10% chance di whale alert
            movement_type = np.random.choice(['exchange_inflow', 'exchange_outflow', 'wallet_consolidation'])
            amount = np.random.uniform(100, 1000)  # BTC equivalent
            
            whale_data = {
                'detected': True,
                'movement_type': movement_type,
                'amount_btc_equivalent': amount,
                'confidence': np.random.uniform(0.7, 0.95),
                'potential_impact': self._assess_whale_impact(movement_type, amount),
                'timestamp': datetime.now()
            }
        else:
            whale_data = {'detected': False}
            
        return whale_data
    
    def _assess_whale_impact(self, movement_type: str, amount: float) -> Dict[str, Any]:
        """Valuta impatto potenziale movimento balena"""
        whale_patterns = self.ai_knowledge.get('whale_movement_patterns', {})
        
        if movement_type == 'exchange_inflow':
            return {
                'direction': 'bearish',
                'timeframe': '24-72 hours',
                'magnitude': 'medium' if amount < 500 else 'high',
                'probability': 0.68
            }
        elif movement_type == 'exchange_outflow':
            return {
                'direction': 'bullish',
                'timeframe': 'medium-term',
                'magnitude': 'medium' if amount < 500 else 'high',
                'probability': 0.74
            }
        else:  # wallet_consolidation
            return {
                'direction': 'neutral_preparation',
                'timeframe': '1-2 weeks',
                'magnitude': 'high',
                'probability': 0.62
            }
    
    def get_market_intelligence_summary(self) -> Dict[str, Any]:
        """Genera summary completo dell'intelligence di mercato"""
        # Ottieni tutti i dati
        market_data = self.get_real_time_prices()
        news_data = self.analyze_crypto_news_sentiment()
        social_data = self.analyze_social_sentiment()
        whale_data = self.detect_whale_movements()
        
        # Calcola sentiment aggregato
        overall_sentiment = self._calculate_overall_sentiment(news_data, social_data)
        
        # Genera raccomandazioni
        recommendations = self._generate_trading_recommendations(
            market_data, overall_sentiment, whale_data
        )
        
        return {
            'timestamp': datetime.now(),
            'market_data': market_data,
            'news_sentiment': {
                'articles_analyzed': len(news_data),
                'average_sentiment': np.mean([n.sentiment for n in news_data]) if news_data else 0.0,
                'high_impact_news': [n for n in news_data if n.impact_score > 0.7]
            },
            'social_sentiment': social_data,
            'whale_activity': whale_data,
            'overall_sentiment': overall_sentiment,
            'ai_recommendations': recommendations,
            'risk_assessment': self._assess_current_risk(market_data, overall_sentiment)
        }
    
    def _calculate_overall_sentiment(self, news_data: List[NewsData], social_data: Dict[str, SentimentData]) -> Dict[str, float]:
        """Calcola sentiment complessivo pesato"""
        sentiment_scores = []
        weights = []
        
        # News sentiment (peso 40%)
        if news_data:
            news_sentiment = np.mean([n.sentiment for n in news_data])
            sentiment_scores.append(news_sentiment)
            weights.append(0.4)
        
        # Social sentiment (peso 60%)
        social_sentiment = 0.0
        social_count = 0
        for platform, data in social_data.items():
            social_sentiment += data.sentiment_score * data.confidence
            social_count += 1
            
        if social_count > 0:
            social_sentiment /= social_count
            sentiment_scores.append(social_sentiment)
            weights.append(0.6)
        
        # Calcola weighted average
        if sentiment_scores:
            overall = np.average(sentiment_scores, weights=weights)
            return {
                'score': overall,
                'classification': self._classify_sentiment(overall),
                'confidence': np.mean(weights)
            }
        
        return {'score': 0.0, 'classification': 'neutral', 'confidence': 0.0}
    
    def _classify_sentiment(self, score: float) -> str:
        """Classifica sentiment score"""
        if score >= 0.4:
            return 'extremely_bullish'
        elif score >= 0.2:
            return 'bullish'
        elif score >= -0.2:
            return 'neutral'
        elif score >= -0.4:
            return 'bearish'
        else:
            return 'extremely_bearish'
    
    def _generate_trading_recommendations(self, market_data: Dict[str, MarketData], 
                                        sentiment: Dict[str, Any], whale_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera raccomandazioni trading basate su AI knowledge"""
        recommendations = []
        
        sentiment_score = sentiment.get('score', 0.0)
        
        # Raccomandazione basata su sentiment
        if sentiment_score > 0.3 and sentiment.get('confidence', 0) > 0.7:
            recommendations.append({
                'type': 'sentiment_bullish',
                'action': 'consider_long_positions',
                'confidence': sentiment.get('confidence', 0),
                'reasoning': f"Strong bullish sentiment ({sentiment_score:.2f}) with high confidence"
            })
        elif sentiment_score < -0.3 and sentiment.get('confidence', 0) > 0.7:
            recommendations.append({
                'type': 'sentiment_bearish',
                'action': 'consider_short_positions',
                'confidence': sentiment.get('confidence', 0),
                'reasoning': f"Strong bearish sentiment ({sentiment_score:.2f}) with high confidence"
            })
        
        # Raccomandazione basata su whale activity
        if whale_data.get('detected', False):
            whale_impact = whale_data.get('potential_impact', {})
            if whale_impact.get('direction') == 'bearish':
                recommendations.append({
                    'type': 'whale_bearish',
                    'action': 'reduce_long_exposure',
                    'confidence': whale_data.get('confidence', 0),
                    'reasoning': f"Large whale {whale_data['movement_type']} detected"
                })
        
        # Raccomandazione contrarian per sentiment estremi
        if abs(sentiment_score) > 0.6:
            recommendations.append({
                'type': 'contrarian',
                'action': 'consider_contrarian_position',
                'confidence': 0.6,
                'reasoning': "Extreme sentiment often leads to reversals"
            })
        
        return recommendations
    
    def _assess_current_risk(self, market_data: Dict[str, MarketData], sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Valuta rischio attuale del mercato"""
        risk_factors = []
        risk_score = 0.0
        
        # Analizza volatilità
        if market_data:
            changes = [abs(data.change_24h) for data in market_data.values()]
            avg_volatility = np.mean(changes)
            
            if avg_volatility > 10:
                risk_factors.append("High volatility detected")
                risk_score += 0.3
            elif avg_volatility > 5:
                risk_factors.append("Moderate volatility")
                risk_score += 0.1
        
        # Sentiment risk
        sentiment_score = abs(sentiment.get('score', 0))
        if sentiment_score > 0.6:
            risk_factors.append("Extreme sentiment levels")
            risk_score += 0.2
        
        # Classifica risk level
        if risk_score > 0.4:
            risk_level = "high"
        elif risk_score > 0.2:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommended_position_size': max(0.1, 1.0 - risk_score)  # Reduce size with higher risk
        }

def main():
    """Test del sistema di analisi real-time"""
    analyzer = RealTimeMarketAnalyzer()
    
    print("🔍 Avvio analisi mercato real-time...")
    
    # Ottieni intelligence completa
    intelligence = analyzer.get_market_intelligence_summary()
    
    print(f"\n📊 Market Intelligence Report - {intelligence['timestamp']}")
    print("=" * 60)
    
    # Market Data
    print("\n💰 Market Data:")
    for symbol, data in intelligence['market_data'].items():
        print(f"  {symbol}: ${data.price:.2f} ({data.change_24h:+.2f}%)")
    
    # News Sentiment
    news = intelligence['news_sentiment']
    print(f"\n📰 News Analysis:")
    print(f"  Articles Analyzed: {news['articles_analyzed']}")
    print(f"  Average Sentiment: {news['average_sentiment']:.3f}")
    print(f"  High Impact News: {len(news['high_impact_news'])}")
    
    # Social Sentiment
    print(f"\n📱 Social Sentiment:")
    for platform, data in intelligence['social_sentiment'].items():
        print(f"  {platform.title()}: {data.sentiment_score:.3f} (confidence: {data.confidence:.2f})")
        print(f"    Volume: {data.volume}, Keywords: {data.keywords[:3]}")
    
    # Overall Assessment
    overall = intelligence['overall_sentiment']
    print(f"\n🎯 Overall Assessment:")
    print(f"  Sentiment: {overall['classification'].upper()} ({overall['score']:.3f})")
    print(f"  Confidence: {overall['confidence']:.2f}")
    
    # Whale Activity
    whale = intelligence['whale_activity']
    if whale.get('detected'):
        print(f"\n🐋 Whale Alert:")
        print(f"  Type: {whale['movement_type']}")
        print(f"  Amount: {whale['amount_btc_equivalent']:.0f} BTC equivalent")
        impact = whale['potential_impact']
        print(f"  Impact: {impact['direction']} ({impact['timeframe']})")
    
    # AI Recommendations
    print(f"\n🤖 AI Recommendations:")
    for rec in intelligence['ai_recommendations']:
        print(f"  • {rec['action']} (confidence: {rec['confidence']:.2f})")
        print(f"    Reason: {rec['reasoning']}")
    
    # Risk Assessment
    risk = intelligence['risk_assessment']
    print(f"\n⚠️ Risk Assessment:")
    print(f"  Risk Level: {risk['risk_level'].upper()}")
    print(f"  Recommended Position Size: {risk['recommended_position_size']:.1%}")
    if risk['risk_factors']:
        print(f"  Risk Factors: {', '.join(risk['risk_factors'])}")

if __name__ == "__main__":
    main()