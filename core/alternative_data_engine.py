"""
Motore di Alternative Data per News, Social Sentiment e On-Chain Metrics
Integra fonti esterne per migliorare le decisioni di trading
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import requests
import json
import re
from bs4 import BeautifulSoup
import yfinance as yf
from urllib.parse import quote_plus
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """Analizzatore di notizie e sentiment"""
    
    def __init__(self):
        self.news_sources = {
            'cointelegraph': 'https://cointelegraph.com',
            'coindesk': 'https://www.coindesk.com',
            'decrypt': 'https://decrypt.co',
            'bitcoinist': 'https://bitcoinist.com'
        }
        self.sentiment_keywords = {
            'bullish': ['bull', 'bullish', 'surge', 'rally', 'pump', 'moon', 'growth', 'adoption', 'breakthrough'],
            'bearish': ['bear', 'bearish', 'crash', 'dump', 'decline', 'sell-off', 'correction', 'drop'],
            'neutral': ['stable', 'consolidation', 'sideways', 'range', 'analysis', 'update']
        }
        
    def fetch_crypto_news(self, asset: str = 'bitcoin', hours_back: int = 24) -> List[Dict[str, Any]]:
        """Recupera notizie recenti per un asset"""
        try:
            news_articles = []
            
            # Simula recupero notizie (in produzione usare API reali)
            simulated_news = [
                {
                    'title': f'{asset.upper()} shows strong momentum amid institutional adoption',
                    'summary': f'Major institutions continue to invest in {asset}, driving positive market sentiment',
                    'source': 'CoinDesk',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'url': 'https://example.com/news1',
                    'sentiment_raw': 0.8
                },
                {
                    'title': f'Technical analysis suggests {asset.upper()} consolidation phase',
                    'summary': f'{asset} price action indicates potential breakout or breakdown in coming days',
                    'source': 'CoinTelegraph',
                    'timestamp': datetime.now() - timedelta(hours=6),
                    'url': 'https://example.com/news2',
                    'sentiment_raw': 0.1
                },
                {
                    'title': f'Regulatory clarity boosts {asset.upper()} market confidence',
                    'summary': f'Recent regulatory developments provide clearer framework for {asset} trading',
                    'source': 'Decrypt',
                    'timestamp': datetime.now() - timedelta(hours=12),
                    'url': 'https://example.com/news3',
                    'sentiment_raw': 0.6
                }
            ]
            
            for article in simulated_news:
                sentiment_score = self._analyze_sentiment(article['title'] + ' ' + article['summary'])
                
                news_articles.append({
                    'title': article['title'],
                    'summary': article['summary'],
                    'source': article['source'],
                    'timestamp': article['timestamp'],
                    'url': article['url'],
                    'sentiment_score': sentiment_score,
                    'relevance_score': self._calculate_relevance(article['title'], asset),
                    'impact_weight': self._calculate_impact_weight(article['source'], article['timestamp'])
                })
            
            return news_articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {asset}: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analizza sentiment del testo (-1 a +1)"""
        try:
            text_lower = text.lower()
            
            bullish_count = sum(text_lower.count(word) for word in self.sentiment_keywords['bullish'])
            bearish_count = sum(text_lower.count(word) for word in self.sentiment_keywords['bearish'])
            neutral_count = sum(text_lower.count(word) for word in self.sentiment_keywords['neutral'])
            
            total_words = len(text.split())
            
            if total_words == 0:
                return 0.0
            
            # Calcola score normalizzato
            bullish_weight = bullish_count / total_words * 10
            bearish_weight = bearish_count / total_words * 10
            
            sentiment = (bullish_weight - bearish_weight) / max(1, bullish_weight + bearish_weight + neutral_count/total_words*2)
            
            return np.clip(sentiment, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _calculate_relevance(self, title: str, asset: str) -> float:
        """Calcola rilevanza dell'articolo per l'asset"""
        try:
            title_lower = title.lower()
            asset_lower = asset.lower()
            
            # Punteggio base se l'asset è menzionato
            relevance = 0.5 if asset_lower in title_lower else 0.1
            
            # Aumenta se è nel titolo
            if asset_lower in title_lower[:50]:  # Primi 50 caratteri
                relevance += 0.3
            
            # Parole chiave che aumentano rilevanza
            high_impact_words = ['price', 'trading', 'market', 'analysis', 'prediction']
            for word in high_impact_words:
                if word in title_lower:
                    relevance += 0.1
            
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.5
    
    def _calculate_impact_weight(self, source: str, timestamp: datetime) -> float:
        """Calcola peso dell'impatto basato su fonte e freschezza"""
        try:
            # Peso per fonte
            source_weights = {
                'coindesk': 0.9,
                'cointelegraph': 0.8,
                'decrypt': 0.7,
                'bitcoinist': 0.6
            }
            
            source_weight = source_weights.get(source.lower(), 0.5)
            
            # Peso per freschezza (decay esponenziale)
            hours_ago = (datetime.now() - timestamp).total_seconds() / 3600
            freshness_weight = np.exp(-hours_ago / 12)  # Half-life di 12 ore
            
            return source_weight * freshness_weight
            
        except Exception as e:
            logger.error(f"Error calculating impact weight: {e}")
            return 0.5
    
    def aggregate_news_sentiment(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggrega sentiment da multiple notizie"""
        try:
            if not news_articles:
                return {
                    'overall_sentiment': 0.0,
                    'confidence': 0.0,
                    'article_count': 0,
                    'sentiment_distribution': {'bullish': 0, 'neutral': 0, 'bearish': 0}
                }
            
            # Calcola sentiment pesato
            weighted_sentiments = []
            total_weight = 0
            
            for article in news_articles:
                weight = article['impact_weight'] * article['relevance_score']
                weighted_sentiments.append(article['sentiment_score'] * weight)
                total_weight += weight
            
            if total_weight > 0:
                overall_sentiment = sum(weighted_sentiments) / total_weight
            else:
                overall_sentiment = 0.0
            
            # Distribuzione sentiment
            bullish_count = sum(1 for a in news_articles if a['sentiment_score'] > 0.2)
            bearish_count = sum(1 for a in news_articles if a['sentiment_score'] < -0.2)
            neutral_count = len(news_articles) - bullish_count - bearish_count
            
            # Confidence basata su numero articoli e coerenza
            sentiment_std = np.std([a['sentiment_score'] for a in news_articles])
            confidence = min(1.0, len(news_articles) / 10) * (1 - sentiment_std)
            
            return {
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'article_count': len(news_articles),
                'sentiment_distribution': {
                    'bullish': bullish_count,
                    'neutral': neutral_count,
                    'bearish': bearish_count
                },
                'weighted_average': overall_sentiment,
                'sentiment_strength': abs(overall_sentiment)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating news sentiment: {e}")
            return {'overall_sentiment': 0.0, 'confidence': 0.0}

class SocialSentimentAnalyzer:
    """Analizzatore di sentiment dai social media"""
    
    def __init__(self):
        self.social_platforms = ['twitter', 'reddit', 'telegram']
        self.influence_weights = {
            'high': 1.0,    # Account verificati, grandi follower
            'medium': 0.6,  # Account medi
            'low': 0.3      # Account piccoli
        }
    
    def get_social_sentiment(self, asset: str, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Ottiene sentiment dai social media"""
        try:
            # Simula dati social (in produzione usare API Twitter, Reddit, etc.)
            simulated_posts = [
                {
                    'platform': 'twitter',
                    'content': f'{asset.upper()} looking bullish! Great fundamentals and technical setup 🚀',
                    'author_influence': 'high',
                    'engagement': 1250,
                    'timestamp': datetime.now() - timedelta(hours=1),
                    'sentiment_raw': 0.8
                },
                {
                    'platform': 'reddit',
                    'content': f'Concerned about {asset} price action. Might see correction soon.',
                    'author_influence': 'medium',
                    'engagement': 45,
                    'timestamp': datetime.now() - timedelta(hours=3),
                    'sentiment_raw': -0.4
                },
                {
                    'platform': 'twitter',
                    'content': f'{asset} consolidating nicely. Waiting for clear direction.',
                    'author_influence': 'medium',
                    'engagement': 230,
                    'timestamp': datetime.now() - timedelta(hours=5),
                    'sentiment_raw': 0.1
                }
            ]
            
            processed_posts = []
            for post in simulated_posts:
                sentiment = self._analyze_social_sentiment(post['content'])
                influence_weight = self.influence_weights[post['author_influence']]
                engagement_weight = min(1.0, post['engagement'] / 1000)  # Normalizza engagement
                
                processed_posts.append({
                    'platform': post['platform'],
                    'sentiment_score': sentiment,
                    'influence_weight': influence_weight,
                    'engagement_weight': engagement_weight,
                    'timestamp': post['timestamp'],
                    'total_weight': influence_weight * engagement_weight
                })
            
            return self._aggregate_social_sentiment(processed_posts)
            
        except Exception as e:
            logger.error(f"Error getting social sentiment for {asset}: {e}")
            return {'overall_sentiment': 0.0, 'confidence': 0.0}
    
    def _analyze_social_sentiment(self, text: str) -> float:
        """Analizza sentiment di un post social"""
        try:
            # Rimuovi emoji e caratteri speciali
            clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
            
            # Parole chiave social-specific
            bullish_social = ['moon', 'lambo', 'hodl', 'diamond hands', 'to the moon', 'bullish', 'pump']
            bearish_social = ['dump', 'crash', 'rekt', 'paper hands', 'bearish', 'sell']
            
            bullish_count = sum(clean_text.count(word) for word in bullish_social)
            bearish_count = sum(clean_text.count(word) for word in bearish_social)
            
            if bullish_count + bearish_count == 0:
                return 0.0
            
            sentiment = (bullish_count - bearish_count) / (bullish_count + bearish_count)
            return np.clip(sentiment, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return 0.0
    
    def _aggregate_social_sentiment(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggrega sentiment da post social"""
        try:
            if not posts:
                return {'overall_sentiment': 0.0, 'confidence': 0.0}
            
            # Sentiment pesato
            weighted_sentiment = sum(post['sentiment_score'] * post['total_weight'] for post in posts)
            total_weight = sum(post['total_weight'] for post in posts)
            
            overall_sentiment = weighted_sentiment / max(total_weight, 0.1)
            
            # Confidence basata su volume e diversità
            platform_diversity = len(set(post['platform'] for post in posts)) / len(self.social_platforms)
            volume_score = min(1.0, len(posts) / 50)  # Normalizza a 50 post
            
            confidence = (platform_diversity + volume_score) / 2
            
            return {
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'post_count': len(posts),
                'platform_breakdown': self._get_platform_breakdown(posts),
                'sentiment_strength': abs(overall_sentiment)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating social sentiment: {e}")
            return {'overall_sentiment': 0.0, 'confidence': 0.0}
    
    def _get_platform_breakdown(self, posts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Breakdown per piattaforma"""
        breakdown = {}
        
        for platform in self.social_platforms:
            platform_posts = [p for p in posts if p['platform'] == platform]
            
            if platform_posts:
                avg_sentiment = np.mean([p['sentiment_score'] for p in platform_posts])
                post_count = len(platform_posts)
            else:
                avg_sentiment = 0.0
                post_count = 0
            
            breakdown[platform] = {
                'sentiment': avg_sentiment,
                'post_count': post_count
            }
        
        return breakdown

class OnChainAnalyzer:
    """Analizzatore di metriche on-chain"""
    
    def __init__(self):
        self.on_chain_metrics = [
            'active_addresses',
            'transaction_volume',
            'network_hash_rate',
            'exchange_inflows',
            'exchange_outflows',
            'whale_movements',
            'long_term_holder_supply'
        ]
    
    def get_on_chain_metrics(self, asset: str) -> Dict[str, Any]:
        """Ottiene metriche on-chain"""
        try:
            # Simula dati on-chain (in produzione usare API Glassnode, CryptoQuant, etc.)
            current_time = datetime.now()
            
            simulated_metrics = {
                'active_addresses': {
                    'current': 150000,
                    'change_24h': 0.05,  # +5%
                    'trend': 'increasing',
                    'significance': 'medium'
                },
                'transaction_volume': {
                    'current': 2500000000,  # $2.5B
                    'change_24h': -0.02,    # -2%
                    'trend': 'decreasing',
                    'significance': 'low'
                },
                'exchange_inflows': {
                    'current': 1200,  # BTC
                    'change_24h': 0.15,   # +15%
                    'trend': 'increasing',
                    'significance': 'high'  # Potenziale selling pressure
                },
                'exchange_outflows': {
                    'current': 800,   # BTC
                    'change_24h': -0.08,  # -8%
                    'trend': 'decreasing',
                    'significance': 'medium'
                },
                'whale_movements': {
                    'large_transactions_24h': 45,
                    'net_flow': -400,  # Net outflow dai whale
                    'significance': 'high'
                },
                'long_term_holder_supply': {
                    'percentage': 65.2,
                    'change_30d': 0.8,   # +0.8%
                    'trend': 'increasing',
                    'significance': 'medium'
                }
            }
            
            # Calcola score aggregato
            aggregate_score = self._calculate_on_chain_score(simulated_metrics)
            
            return {
                'metrics': simulated_metrics,
                'aggregate_score': aggregate_score,
                'timestamp': current_time,
                'interpretation': self._interpret_on_chain_data(simulated_metrics),
                'signal_strength': abs(aggregate_score)
            }
            
        except Exception as e:
            logger.error(f"Error getting on-chain metrics for {asset}: {e}")
            return {}
    
    def _calculate_on_chain_score(self, metrics: Dict[str, Any]) -> float:
        """Calcola score aggregato dalle metriche on-chain"""
        try:
            score_components = []
            
            # Active addresses (positivo se cresce)
            aa_change = metrics['active_addresses']['change_24h']
            score_components.append(aa_change * 0.2)
            
            # Exchange flows (inflows negativi, outflows positivi)
            inflow_change = metrics['exchange_inflows']['change_24h']
            outflow_change = metrics['exchange_outflows']['change_24h']
            flow_score = (outflow_change - inflow_change) * 0.3
            score_components.append(flow_score)
            
            # Whale movements (outflow positivo)
            whale_flow = metrics['whale_movements']['net_flow']
            whale_score = np.clip(whale_flow / 1000, -0.5, 0.5)  # Normalizza
            score_components.append(whale_score)
            
            # Long-term holders (positivo se cresce)
            lth_change = metrics['long_term_holder_supply']['change_30d'] / 100
            score_components.append(lth_change * 0.3)
            
            aggregate_score = sum(score_components)
            return np.clip(aggregate_score, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating on-chain score: {e}")
            return 0.0
    
    def _interpret_on_chain_data(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Interpreta i dati on-chain"""
        interpretations = {}
        
        # Exchange flows
        inflows = metrics['exchange_inflows']['change_24h']
        outflows = metrics['exchange_outflows']['change_24h']
        
        if inflows > 0.1:
            interpretations['exchange_flows'] = 'High exchange inflows - potential selling pressure'
        elif outflows > 0.1:
            interpretations['exchange_flows'] = 'High exchange outflows - potential accumulation'
        else:
            interpretations['exchange_flows'] = 'Balanced exchange flows'
        
        # Whale activity
        whale_flow = metrics['whale_movements']['net_flow']
        if whale_flow < -200:
            interpretations['whale_activity'] = 'Whales accumulating - bullish signal'
        elif whale_flow > 200:
            interpretations['whale_activity'] = 'Whales distributing - bearish signal'
        else:
            interpretations['whale_activity'] = 'Neutral whale activity'
        
        # Network activity
        aa_trend = metrics['active_addresses']['trend']
        interpretations['network_activity'] = f'Active addresses {aa_trend} - network usage trend'
        
        return interpretations

class AlternativeDataEngine:
    """Motore principale per alternative data"""
    
    def __init__(self):
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialSentimentAnalyzer()
        self.onchain_analyzer = OnChainAnalyzer()
        self.data_cache = {}
        self.last_update = {}
        
    def get_comprehensive_sentiment(self, asset: str) -> Dict[str, Any]:
        """Ottiene sentiment comprensivo da tutte le fonti"""
        try:
            # Cache key
            cache_key = f"{asset}_sentiment"
            current_time = datetime.now()
            
            # Controlla cache (5 minuti)
            if (cache_key in self.data_cache and 
                cache_key in self.last_update and
                (current_time - self.last_update[cache_key]).total_seconds() < 300):
                return self.data_cache[cache_key]
            
            # Raccolta dati
            news_data = self.news_analyzer.fetch_crypto_news(asset)
            news_sentiment = self.news_analyzer.aggregate_news_sentiment(news_data)
            
            social_sentiment = self.social_analyzer.get_social_sentiment(asset)
            
            onchain_data = self.onchain_analyzer.get_on_chain_metrics(asset)
            
            # Combina tutti i segnali
            combined_analysis = self._combine_alternative_signals(
                news_sentiment, social_sentiment, onchain_data
            )
            
            # Cache risultato
            self.data_cache[cache_key] = combined_analysis
            self.last_update[cache_key] = current_time
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error getting comprehensive sentiment for {asset}: {e}")
            return {}
    
    def _combine_alternative_signals(self, news: Dict[str, Any], social: Dict[str, Any],
                                   onchain: Dict[str, Any]) -> Dict[str, Any]:
        """Combina segnali da diverse fonti alternative"""
        try:
            # Pesi per diverse fonti
            weights = {
                'news': 0.4,
                'social': 0.3,
                'onchain': 0.3
            }
            
            # Estrai sentiment scores
            news_sentiment = news.get('overall_sentiment', 0.0)
            social_sentiment = social.get('overall_sentiment', 0.0)
            onchain_sentiment = onchain.get('aggregate_score', 0.0)
            
            # Confidence scores
            news_confidence = news.get('confidence', 0.0)
            social_confidence = social.get('confidence', 0.0)
            onchain_confidence = 0.8  # On-chain data generally reliable
            
            # Weighted sentiment
            weighted_sentiment = (
                news_sentiment * weights['news'] * news_confidence +
                social_sentiment * weights['social'] * social_confidence +
                onchain_sentiment * weights['onchain'] * onchain_confidence
            )
            
            total_weight = (
                weights['news'] * news_confidence +
                weights['social'] * social_confidence +
                weights['onchain'] * onchain_confidence
            )
            
            if total_weight > 0:
                combined_sentiment = weighted_sentiment / total_weight
            else:
                combined_sentiment = 0.0
            
            # Overall confidence
            overall_confidence = (news_confidence + social_confidence + onchain_confidence) / 3
            
            # Signal classification
            signal_strength = abs(combined_sentiment)
            if signal_strength > 0.6:
                signal_classification = 'STRONG'
            elif signal_strength > 0.3:
                signal_classification = 'MODERATE'
            else:
                signal_classification = 'WEAK'
            
            signal_direction = 'BULLISH' if combined_sentiment > 0.1 else 'BEARISH' if combined_sentiment < -0.1 else 'NEUTRAL'
            
            return {
                'combined_sentiment': combined_sentiment,
                'overall_confidence': overall_confidence,
                'signal_classification': signal_classification,
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'component_analysis': {
                    'news': {
                        'sentiment': news_sentiment,
                        'confidence': news_confidence,
                        'article_count': news.get('article_count', 0)
                    },
                    'social': {
                        'sentiment': social_sentiment,
                        'confidence': social_confidence,
                        'post_count': social.get('post_count', 0)
                    },
                    'onchain': {
                        'sentiment': onchain_sentiment,
                        'confidence': onchain_confidence,
                        'interpretation': onchain.get('interpretation', {})
                    }
                },
                'trading_recommendation': self._generate_trading_recommendation(
                    combined_sentiment, overall_confidence, signal_classification
                ),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error combining alternative signals: {e}")
            return {}
    
    def _generate_trading_recommendation(self, sentiment: float, confidence: float,
                                       classification: str) -> Dict[str, Any]:
        """Genera raccomandazione di trading basata sui dati alternativi"""
        try:
            # Base recommendation
            if sentiment > 0.3 and confidence > 0.6:
                action = 'BUY'
                strength = min(1.0, sentiment * confidence)
            elif sentiment < -0.3 and confidence > 0.6:
                action = 'SELL'
                strength = min(1.0, abs(sentiment) * confidence)
            else:
                action = 'HOLD'
                strength = 0.0
            
            # Risk adjustment
            if classification == 'WEAK':
                strength *= 0.5
            elif classification == 'STRONG':
                strength *= 1.2
            
            strength = min(1.0, strength)
            
            return {
                'action': action,
                'strength': strength,
                'confidence': confidence,
                'reasoning': f'{classification} {action.lower()} signal based on alternative data analysis',
                'risk_level': 'LOW' if strength < 0.3 else 'MEDIUM' if strength < 0.7 else 'HIGH'
            }
            
        except Exception as e:
            logger.error(f"Error generating trading recommendation: {e}")
            return {'action': 'HOLD', 'strength': 0.0}
    
    def get_alternative_data_dashboard(self) -> Dict[str, Any]:
        """Dashboard per dati alternativi"""
        try:
            return {
                'news_analyzer_status': 'active',
                'social_analyzer_status': 'active',
                'onchain_analyzer_status': 'active',
                'last_updates': {k: v.isoformat() for k, v in self.last_update.items()},
                'cache_size': len(self.data_cache),
                'supported_assets': ['BTC', 'ETH', 'SOL', 'AVAX', 'KAS'],
                'data_sources': {
                    'news': list(self.news_analyzer.news_sources.keys()),
                    'social': self.social_analyzer.social_platforms,
                    'onchain': self.onchain_analyzer.on_chain_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating alternative data dashboard: {e}")
            return {}