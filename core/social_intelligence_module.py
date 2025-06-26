"""
Social Intelligence Module con NLP Avanzato
Collezione dati real-time da X, Reddit, Telegram, Discord + Sentiment Analysis
"""

import asyncio
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from urllib.parse import urljoin
import feedparser

# Mock NLP imports (in produzione useresti le librerie reali)
try:
    # from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from sklearn.cluster import KMeans
    # from textblob import TextBlob
    pass
except ImportError:
    pass

logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    id: str
    platform: str
    content: str
    author: str
    timestamp: datetime
    engagement: int  # likes + retweets + comments
    author_followers: int
    author_verified: bool
    mentions: List[str]  # crypto symbols mentioned
    urls: List[str]
    hashtags: List[str]
    sentiment_score: float = 0.0
    influence_score: float = 0.0
    processed: bool = False

@dataclass
class TrendingTopic:
    topic: str
    mentions: int
    sentiment: float
    momentum: float  # rate of change
    related_tokens: List[str]
    key_posts: List[str]
    confidence: float

@dataclass
class SentimentAlert:
    token: str
    alert_type: str  # 'spike', 'trend_change', 'whale_correlation'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    data: Dict[str, Any]
    timestamp: datetime

class CryptoMentionDetector:
    """Rileva menzioni di criptovalute nei testi"""
    
    def __init__(self):
        # Database di token crypto con variazioni comuni
        self.crypto_patterns = {
            'BTC': ['bitcoin', 'btc', '$btc', '#btc', '#bitcoin'],
            'ETH': ['ethereum', 'eth', '$eth', '#eth', '#ethereum', 'ether'],
            'KAS': ['kaspa', 'kas', '$kas', '#kas', '#kaspa'],
            'SOL': ['solana', 'sol', '$sol', '#sol', '#solana'],
            'AVAX': ['avalanche', 'avax', '$avax', '#avax', '#avalanche'],
            'BNB': ['binance', 'bnb', '$bnb', '#bnb'],
            'ADA': ['cardano', 'ada', '$ada', '#ada', '#cardano'],
            'DOGE': ['dogecoin', 'doge', '$doge', '#doge', '#dogecoin'],
            'MATIC': ['polygon', 'matic', '$matic', '#matic', '#polygon'],
            'DOT': ['polkadot', 'dot', '$dot', '#dot', '#polkadot']
        }
        
        # Compila pattern regex per performance
        self.compiled_patterns = {}
        for token, patterns in self.crypto_patterns.items():
            pattern = r'\b(?:' + '|'.join(re.escape(p) for p in patterns) + r')\b'
            self.compiled_patterns[token] = re.compile(pattern, re.IGNORECASE)
    
    def extract_mentions(self, text: str) -> List[str]:
        """Estrae menzioni di crypto dal testo"""
        mentions = []
        text_lower = text.lower()
        
        for token, pattern in self.compiled_patterns.items():
            if pattern.search(text_lower):
                mentions.append(token)
        
        return list(set(mentions))  # Rimuovi duplicati

class MockNLPProcessor:
    """Processore NLP simulato (in produzione useresti BERT/RoBERTa)"""
    
    def __init__(self):
        # Parole chiave per sentiment
        self.positive_words = [
            'moon', 'bullish', 'pump', 'rocket', 'gains', 'profit', 'buy', 'hold',
            'diamond', 'hands', 'to the moon', 'breakout', 'rally', 'surge'
        ]
        self.negative_words = [
            'dump', 'crash', 'bearish', 'sell', 'loss', 'drop', 'fall', 'panic',
            'bear', 'correction', 'dip', 'red', 'blood', 'liquidation'
        ]
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analizza sentiment del testo (-1 a +1)"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Calcola score normalizzato
        total_words = len(text.split())
        if total_words == 0:
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        sentiment = positive_ratio - negative_ratio
        confidence = min(1.0, (positive_count + negative_count) / max(1, total_words * 0.1))
        
        # Normalizza sentiment tra -1 e 1
        sentiment = max(-1.0, min(1.0, sentiment * 10))
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_signals': positive_count,
            'negative_signals': negative_count
        }
    
    def extract_topics(self, texts: List[str], num_topics: int = 5) -> List[Dict[str, Any]]:
        """Estrae topic principali dai testi (simulato)"""
        # In produzione useresti BERTopic o LDA
        word_freq = Counter()
        
        for text in texts:
            words = text.lower().split()
            # Filtra parole comuni
            filtered_words = [w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'are', 'but']]
            word_freq.update(filtered_words)
        
        # Crea topic basati su parole frequenti
        topics = []
        most_common = word_freq.most_common(num_topics * 3)
        
        for i in range(min(num_topics, len(most_common))):
            topic_words = most_common[i*3:(i+1)*3] if i*3 < len(most_common) else most_common[i:]
            topic = {
                'id': i,
                'words': [word for word, _ in topic_words],
                'coherence': 0.7 + (0.3 * (num_topics - i) / num_topics),  # Mock coherence
                'size': sum(count for _, count in topic_words)
            }
            topics.append(topic)
        
        return topics

class SocialDataCollector:
    """Collettore dati da piattaforme social"""
    
    def __init__(self):
        self.crypto_detector = CryptoMentionDetector()
        self.nlp_processor = MockNLPProcessor()
        
        # Rate limiting
        self.last_request_time = {}
        self.request_delays = {
            'twitter': 1.0,  # 1 secondo tra richieste
            'reddit': 2.0,   # 2 secondi
            'telegram': 0.5, # 0.5 secondi
            'rss': 5.0       # 5 secondi
        }
        
        # URLs e endpoints (mock)
        self.endpoints = {
            'reddit': 'https://www.reddit.com/r/cryptocurrency/new.json',
            'rss_feeds': [
                'https://cointelegraph.com/rss',
                'https://cryptonews.com/news/feed/',
                'https://decrypt.co/feed'
            ]
        }
    
    def _rate_limit(self, platform: str):
        """Implementa rate limiting"""
        now = time.time()
        if platform in self.last_request_time:
            elapsed = now - self.last_request_time[platform]
            delay = self.request_delays.get(platform, 1.0)
            if elapsed < delay:
                time.sleep(delay - elapsed)
        
        self.last_request_time[platform] = time.time()
    
    async def collect_twitter_data(self, keywords: List[str], limit: int = 100) -> List[SocialPost]:
        """Raccoglie dati da X/Twitter (simulato)"""
        self._rate_limit('twitter')
        
        # Simulazione dati Twitter
        posts = []
        for i in range(min(limit, 20)):  # Limita per demo
            content = self._generate_mock_tweet(keywords)
            mentions = self.crypto_detector.extract_mentions(content)
            
            if mentions:  # Solo post con menzioni crypto
                post = SocialPost(
                    id=f"tweet_{i}_{int(time.time())}",
                    platform="twitter",
                    content=content,
                    author=f"user_{i}",
                    timestamp=datetime.now() - timedelta(minutes=i*5),
                    engagement=np.random.randint(1, 1000),
                    author_followers=np.random.randint(100, 50000),
                    author_verified=np.random.choice([True, False], p=[0.1, 0.9]),
                    mentions=mentions,
                    urls=[],
                    hashtags=re.findall(r'#\w+', content)
                )
                posts.append(post)
        
        logger.info(f"Collected {len(posts)} Twitter posts with crypto mentions")
        return posts
    
    async def collect_reddit_data(self, subreddits: List[str], limit: int = 50) -> List[SocialPost]:
        """Raccoglie dati da Reddit"""
        self._rate_limit('reddit')
        
        posts = []
        try:
            # In produzione useresti PRAW (Python Reddit API Wrapper)
            # Per ora simuliamo
            for i in range(min(limit, 15)):
                content = self._generate_mock_reddit_post()
                mentions = self.crypto_detector.extract_mentions(content)
                
                if mentions:
                    post = SocialPost(
                        id=f"reddit_{i}_{int(time.time())}",
                        platform="reddit",
                        content=content,
                        author=f"redditor_{i}",
                        timestamp=datetime.now() - timedelta(hours=i),
                        engagement=np.random.randint(1, 500),
                        author_followers=np.random.randint(10, 10000),
                        author_verified=False,
                        mentions=mentions,
                        urls=[],
                        hashtags=[]
                    )
                    posts.append(post)
        
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
        
        logger.info(f"Collected {len(posts)} Reddit posts with crypto mentions")
        return posts
    
    async def collect_rss_feeds(self) -> List[SocialPost]:
        """Raccoglie dati da feed RSS crypto"""
        self._rate_limit('rss')
        
        posts = []
        for feed_url in self.endpoints['rss_feeds']:
            try:
                # In produzione faresti richieste HTTP reali
                # feed = feedparser.parse(feed_url)
                
                # Simulazione
                for i in range(5):
                    content = self._generate_mock_news_article()
                    mentions = self.crypto_detector.extract_mentions(content)
                    
                    if mentions:
                        post = SocialPost(
                            id=f"rss_{hashlib.md5(feed_url.encode()).hexdigest()[:8]}_{i}",
                            platform="rss",
                            content=content,
                            author="crypto_news",
                            timestamp=datetime.now() - timedelta(hours=i*2),
                            engagement=np.random.randint(10, 200),
                            author_followers=10000,  # News outlets have many followers
                            author_verified=True,
                            mentions=mentions,
                            urls=[feed_url],
                            hashtags=[]
                        )
                        posts.append(post)
            
            except Exception as e:
                logger.error(f"Error parsing RSS feed {feed_url}: {e}")
        
        logger.info(f"Collected {len(posts)} RSS posts with crypto mentions")
        return posts
    
    def _generate_mock_tweet(self, keywords: List[str]) -> str:
        """Genera tweet simulato per demo"""
        templates = [
            "Just bought more {token}! This is going to moon 🚀 #crypto #hodl",
            "{token} is looking bullish today. Technical analysis shows breakout incoming",
            "Market dump but I'm still bullish on {token}. Diamond hands 💎",
            "{token} price action is crazy today. What are your thoughts?",
            "Technical analysis on {token}: Support at $X, resistance at $Y",
            "{token} whale just moved 1000 coins. Something big coming?",
            "DeFi summer continues with {token} leading the charge",
            "{token} fundamentals are solid despite market volatility"
        ]
        
        template = np.random.choice(templates)
        token = np.random.choice(['BTC', 'ETH', 'KAS', 'SOL', 'AVAX'])
        return template.format(token=token)
    
    def _generate_mock_reddit_post(self) -> str:
        """Genera post Reddit simulato"""
        templates = [
            "Daily discussion: What's your take on {token} today?",
            "Technical Analysis: {token} forming ascending triangle pattern",
            "News: Major exchange lists {token} for trading",
            "Discussion: Is {token} undervalued at current prices?",
            "Update: {token} network upgrade successful, price surge expected",
            "Warning: Unusual whale activity detected in {token}",
            "Research: {token} adoption metrics show strong growth"
        ]
        
        template = np.random.choice(templates)
        token = np.random.choice(['Bitcoin', 'Ethereum', 'Kaspa', 'Solana', 'Avalanche'])
        return template.format(token=token)
    
    def _generate_mock_news_article(self) -> str:
        """Genera articolo news simulato"""
        templates = [
            "{token} Price Analysis: Bulls Target ${price} After Bullish Breakout",
            "Institutional Adoption: Major Bank Announces {token} Custody Services",
            "Market Update: {token} Leads Crypto Rally With 15% Gains",
            "Breaking: {token} Network Processes Record Transaction Volume",
            "Regulatory News: Government Approves {token} ETF Application",
            "DeFi Update: {token} Total Value Locked Reaches New High",
            "Mining News: {token} Hash Rate Hits All-Time High"
        ]
        
        template = np.random.choice(templates)
        token = np.random.choice(['Bitcoin', 'Ethereum', 'Kaspa', 'Solana'])
        price = np.random.randint(100, 100000)
        return template.format(token=token, price=price)

class InfluencerTracker:
    """Traccia influencer e account verificati"""
    
    def __init__(self):
        # Database influencer crypto (mock)
        self.verified_influencers = {
            'elonmusk': {'followers': 100000000, 'influence_score': 0.9, 'category': 'celebrity'},
            'michael_saylor': {'followers': 1000000, 'influence_score': 0.8, 'category': 'institutional'},
            'cz_binance': {'followers': 5000000, 'influence_score': 0.85, 'category': 'exchange'},
            'vitalikbuterin': {'followers': 2000000, 'influence_score': 0.9, 'category': 'developer'},
            'cryptocurrency': {'followers': 3000000, 'influence_score': 0.7, 'category': 'community'}
        }
    
    def calculate_influence_score(self, post: SocialPost) -> float:
        """Calcola score di influenza del post"""
        base_score = 0.1
        
        # Boost per follower count
        if post.author_followers > 100000:
            base_score += 0.3
        elif post.author_followers > 10000:
            base_score += 0.2
        elif post.author_followers > 1000:
            base_score += 0.1
        
        # Boost per account verificato
        if post.author_verified:
            base_score += 0.2
        
        # Boost per engagement
        engagement_ratio = post.engagement / max(1, post.author_followers)
        base_score += min(0.3, engagement_ratio * 100)
        
        # Boost per influencer conosciuti
        if post.author in self.verified_influencers:
            influencer_data = self.verified_influencers[post.author]
            base_score += influencer_data['influence_score'] * 0.5
        
        return min(1.0, base_score)

class SocialIntelligenceModule:
    """Modulo principale di Social Intelligence"""
    
    def __init__(self):
        self.data_collector = SocialDataCollector()
        self.influencer_tracker = InfluencerTracker()
        self.nlp_processor = MockNLPProcessor()
        
        # Storage
        self.social_posts = []
        self.processed_posts = {}  # platform -> List[SocialPost]
        self.sentiment_history = defaultdict(list)  # token -> List[sentiment_data]
        self.trending_topics = []
        self.alerts = []
        
        # Configurazione
        self.monitoring_tokens = ['BTC', 'ETH', 'KAS', 'SOL', 'AVAX']
        self.alert_thresholds = {
            'sentiment_spike': 0.3,  # 30% change in sentiment
            'volume_spike': 2.0,     # 2x normal volume
            'trend_emergence': 0.7   # 70% confidence for new trend
        }
        
        # Background task control
        self.is_running = False
        self.collection_interval = 300  # 5 minuti
    
    async def start_monitoring(self):
        """Avvia monitoraggio continuo"""
        self.is_running = True
        logger.info("Starting social intelligence monitoring...")
        
        while self.is_running:
            try:
                await self._collection_cycle()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def stop_monitoring(self):
        """Ferma monitoraggio"""
        self.is_running = False
        logger.info("Stopped social intelligence monitoring")
    
    async def _collection_cycle(self):
        """Ciclo di collezione dati"""
        logger.info("Starting social data collection cycle...")
        
        # Raccolta dati parallela
        tasks = [
            self.data_collector.collect_twitter_data(self.monitoring_tokens),
            self.data_collector.collect_reddit_data(['cryptocurrency', 'cryptomoonshots']),
            self.data_collector.collect_rss_feeds()
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            new_posts = []
            for result in results:
                if isinstance(result, list):
                    new_posts.extend(result)
                else:
                    logger.error(f"Collection error: {result}")
            
            # Processa nuovi post
            processed = await self._process_posts(new_posts)
            self.social_posts.extend(processed)
            
            # Analizza trend e genera alert
            await self._analyze_trends()
            await self._check_alerts()
            
            # Cleanup vecchi dati
            self._cleanup_old_data()
            
            logger.info(f"Collection cycle completed. Processed {len(processed)} new posts")
            
        except Exception as e:
            logger.error(f"Error in collection cycle: {e}")
    
    async def _process_posts(self, posts: List[SocialPost]) -> List[SocialPost]:
        """Processa post con NLP e sentiment analysis"""
        processed = []
        
        for post in posts:
            try:
                # Sentiment analysis
                sentiment_data = self.nlp_processor.analyze_sentiment(post.content)
                post.sentiment_score = sentiment_data['sentiment']
                
                # Influence score
                post.influence_score = self.influencer_tracker.calculate_influence_score(post)
                
                # Salva dati sentiment per token menzionati
                for token in post.mentions:
                    self.sentiment_history[token].append({
                        'timestamp': post.timestamp,
                        'sentiment': post.sentiment_score,
                        'influence': post.influence_score,
                        'engagement': post.engagement,
                        'platform': post.platform
                    })
                
                post.processed = True
                processed.append(post)
                
            except Exception as e:
                logger.error(f"Error processing post {post.id}: {e}")
        
        return processed
    
    async def _analyze_trends(self):
        """Analizza trend emergenti"""
        # Raccoglie testi recenti (ultimo ora)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_texts = [
            post.content for post in self.social_posts 
            if post.timestamp > recent_cutoff and post.processed
        ]
        
        if len(recent_texts) < 10:
            return
        
        # Estrae topic
        topics = self.nlp_processor.extract_topics(recent_texts, num_topics=3)
        
        # Converte in TrendingTopic objects
        self.trending_topics = []
        for topic in topics:
            # Calcola mentions e sentiment per topic
            topic_posts = [
                post for post in self.social_posts 
                if post.timestamp > recent_cutoff and 
                any(word in post.content.lower() for word in topic['words'])
            ]
            
            if topic_posts:
                avg_sentiment = np.mean([post.sentiment_score for post in topic_posts])
                mentions = len(topic_posts)
                
                # Calcola momentum (rate of change)
                momentum = self._calculate_momentum(topic['words'])
                
                trending = TrendingTopic(
                    topic=' '.join(topic['words'][:3]),
                    mentions=mentions,
                    sentiment=avg_sentiment,
                    momentum=momentum,
                    related_tokens=list(set(
                        token for post in topic_posts for token in post.mentions
                    )),
                    key_posts=[post.id for post in topic_posts[:5]],
                    confidence=topic['coherence']
                )
                
                self.trending_topics.append(trending)
    
    def _calculate_momentum(self, topic_words: List[str]) -> float:
        """Calcola momentum per un topic"""
        # Confronta volume nelle ultime 2 ore vs 2 ore precedenti
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=2)
        previous_cutoff = now - timedelta(hours=4)
        
        recent_count = len([
            post for post in self.social_posts
            if post.timestamp > recent_cutoff and
            any(word in post.content.lower() for word in topic_words)
        ])
        
        previous_count = len([
            post for post in self.social_posts
            if previous_cutoff < post.timestamp <= recent_cutoff and
            any(word in post.content.lower() for word in topic_words)
        ])
        
        if previous_count == 0:
            return 1.0 if recent_count > 0 else 0.0
        
        return recent_count / previous_count
    
    async def _check_alerts(self):
        """Controlla condizioni per alert"""
        for token in self.monitoring_tokens:
            await self._check_token_alerts(token)
    
    async def _check_token_alerts(self, token: str):
        """Controlla alert per singolo token"""
        if token not in self.sentiment_history:
            return
        
        recent_data = [
            entry for entry in self.sentiment_history[token]
            if entry['timestamp'] > datetime.now() - timedelta(hours=1)
        ]
        
        if len(recent_data) < 5:
            return
        
        # Check sentiment spike
        current_sentiment = np.mean([entry['sentiment'] for entry in recent_data[-5:]])
        historical_sentiment = np.mean([
            entry['sentiment'] for entry in self.sentiment_history[token]
            if entry['timestamp'] < datetime.now() - timedelta(hours=2)
        ]) if len(self.sentiment_history[token]) > 5 else 0
        
        sentiment_change = abs(current_sentiment - historical_sentiment)
        
        if sentiment_change > self.alert_thresholds['sentiment_spike']:
            alert = SentimentAlert(
                token=token,
                alert_type='sentiment_spike',
                severity='high' if sentiment_change > 0.5 else 'medium',
                message=f"{token} sentiment spike detected: {sentiment_change:.2f} change",
                data={
                    'current_sentiment': current_sentiment,
                    'historical_sentiment': historical_sentiment,
                    'change': sentiment_change,
                    'sample_size': len(recent_data)
                },
                timestamp=datetime.now()
            )
            self.alerts.append(alert)
            logger.warning(f"Sentiment alert generated for {token}: {alert.message}")
    
    def _cleanup_old_data(self):
        """Rimuove dati vecchi per gestione memoria"""
        cutoff = datetime.now() - timedelta(days=7)
        
        # Pulisci post vecchi
        self.social_posts = [
            post for post in self.social_posts 
            if post.timestamp > cutoff
        ]
        
        # Pulisci sentiment history
        for token in self.sentiment_history:
            self.sentiment_history[token] = [
                entry for entry in self.sentiment_history[token]
                if entry['timestamp'] > cutoff
            ]
        
        # Pulisci alert vecchi
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff
        ]
    
    def get_social_dashboard(self) -> Dict[str, Any]:
        """Dashboard per social intelligence"""
        now = datetime.now()
        
        # Statistiche generali
        total_posts = len(self.social_posts)
        recent_posts = len([
            post for post in self.social_posts 
            if post.timestamp > now - timedelta(hours=24)
        ])
        
        # Sentiment per token
        token_sentiment = {}
        for token in self.monitoring_tokens:
            if token in self.sentiment_history:
                recent_sentiment = [
                    entry['sentiment'] for entry in self.sentiment_history[token]
                    if entry['timestamp'] > now - timedelta(hours=24)
                ]
                token_sentiment[token] = {
                    'avg_sentiment': np.mean(recent_sentiment) if recent_sentiment else 0,
                    'mentions_24h': len(recent_sentiment),
                    'trend': 'up' if len(recent_sentiment) > 0 and np.mean(recent_sentiment) > 0 else 'down'
                }
        
        # Platform breakdown
        platform_stats = {}
        for post in self.social_posts:
            if post.timestamp > now - timedelta(hours=24):
                platform = post.platform
                if platform not in platform_stats:
                    platform_stats[platform] = {'count': 0, 'avg_sentiment': 0}
                platform_stats[platform]['count'] += 1
                platform_stats[platform]['avg_sentiment'] += post.sentiment_score
        
        for platform in platform_stats:
            if platform_stats[platform]['count'] > 0:
                platform_stats[platform]['avg_sentiment'] /= platform_stats[platform]['count']
        
        return {
            'status': 'active' if self.is_running else 'stopped',
            'total_posts_collected': total_posts,
            'posts_24h': recent_posts,
            'monitored_tokens': self.monitoring_tokens,
            'token_sentiment': token_sentiment,
            'trending_topics': [asdict(topic) for topic in self.trending_topics],
            'recent_alerts': [asdict(alert) for alert in self.alerts[-10:]],
            'platform_stats': platform_stats,
            'collection_interval': self.collection_interval,
            'last_collection': now.isoformat() if self.social_posts else None
        }
    
    def get_token_analysis(self, token: str, hours: int = 24) -> Dict[str, Any]:
        """Analisi dettagliata per singolo token"""
        if token not in self.sentiment_history:
            return {'error': f'No data available for {token}'}
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_data = [
            entry for entry in self.sentiment_history[token]
            if entry['timestamp'] > cutoff
        ]
        
        if not recent_data:
            return {'error': f'No recent data for {token}'}
        
        # Calcola metriche
        sentiments = [entry['sentiment'] for entry in recent_data]
        influences = [entry['influence'] for entry in recent_data]
        engagements = [entry['engagement'] for entry in recent_data]
        
        # Trend analysis
        if len(sentiments) >= 2:
            trend = 'increasing' if sentiments[-1] > sentiments[0] else 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'token': token,
            'period_hours': hours,
            'total_mentions': len(recent_data),
            'avg_sentiment': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'max_sentiment': np.max(sentiments),
            'min_sentiment': np.min(sentiments),
            'avg_influence': np.mean(influences),
            'total_engagement': sum(engagements),
            'avg_engagement': np.mean(engagements),
            'trend': trend,
            'platform_breakdown': {
                platform: len([e for e in recent_data if e['platform'] == platform])
                for platform in set(entry['platform'] for entry in recent_data)
            }
        }