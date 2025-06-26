#!/usr/bin/env python3
"""
Lightweight AI Models - Real AI without heavy dependencies
Implementa modelli AI reali senza richiedere TensorFlow/PyTorch pesanti
"""

import os
import json
import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Optional
import re

class LightweightSentimentAnalyzer:
    """Sentiment analyzer leggero ma efficace per crypto"""
    
    def __init__(self):
        self.crypto_lexicon = self.build_crypto_lexicon()
        
    def build_crypto_lexicon(self) -> Dict[str, float]:
        """Costruisce lessico sentiment specifico per crypto"""
        return {
            # Bullish terms
            'moon': 0.9, 'bullish': 0.8, 'pump': 0.7, 'hodl': 0.6, 'buy': 0.7,
            'bull': 0.8, 'green': 0.6, 'rocket': 0.9, 'diamond': 0.8, 'hands': 0.6,
            'breakout': 0.7, 'rally': 0.8, 'surge': 0.8, 'gain': 0.7, 'profit': 0.8,
            'support': 0.6, 'bounce': 0.7, 'accumulate': 0.6, 'strong': 0.7,
            
            # Bearish terms  
            'dump': -0.8, 'crash': -0.9, 'bear': -0.8, 'bearish': -0.8, 'sell': -0.7,
            'red': -0.6, 'drop': -0.7, 'fall': -0.7, 'decline': -0.7, 'loss': -0.8,
            'resistance': -0.6, 'reject': -0.7, 'weak': -0.7, 'fud': -0.8,
            'panic': -0.9, 'rekt': -0.9, 'liquidation': -0.8, 'breakdown': -0.8,
            
            # Neutral/context
            'hold': 0.1, 'stable': 0.0, 'sideways': 0.0, 'consolidation': 0.0,
            'volume': 0.0, 'analysis': 0.0, 'technical': 0.0, 'chart': 0.0
        }
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analizza sentiment di un testo"""
        if not text:
            return {'sentiment': 0.5, 'confidence': 0.0, 'word_count': 0}
            
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        sentiment_scores = []
        matched_words = []
        
        for word in words:
            if word in self.crypto_lexicon:
                sentiment_scores.append(self.crypto_lexicon[word])
                matched_words.append(word)
                
        if not sentiment_scores:
            return {'sentiment': 0.5, 'confidence': 0.0, 'word_count': len(words)}
            
        # Calculate weighted sentiment
        avg_sentiment = np.mean(sentiment_scores)
        
        # Normalize to 0-1 scale
        normalized_sentiment = (avg_sentiment + 1) / 2
        
        # Confidence based on number of matches and text length
        confidence = min(1.0, len(matched_words) / max(1, len(words) * 0.1))
        
        return {
            'sentiment': max(0.0, min(1.0, normalized_sentiment)),
            'confidence': confidence,
            'word_count': len(words),
            'matched_words': matched_words,
            'raw_score': avg_sentiment
        }

class LightweightMarketIntelligence:
    """Intelligence di mercato leggera ma potente"""
    
    def __init__(self):
        self.sentiment_analyzer = LightweightSentimentAnalyzer()
        self.load_api_keys()
        
    def load_api_keys(self):
        """Carica API keys"""
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                        
    async def collect_twitter_intelligence(self, symbol: str) -> Dict:
        """Raccoglie intelligence da Twitter"""
        try:
            import tweepy
            
            token = os.getenv('TWITTER_BEARER_TOKEN')
            if not token:
                return self._empty_intelligence('twitter_unavailable')
                
            client = tweepy.Client(bearer_token=token)
            
            query = f"${symbol} OR #{symbol} OR {symbol} lang:en -is:retweet"
            tweets = client.search_recent_tweets(
                query=query,
                max_results=50,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return self._empty_intelligence('no_tweets')
                
            sentiments = []
            engagement_weights = []
            
            for tweet in tweets.data:
                analysis = self.sentiment_analyzer.analyze_text(tweet.text)
                
                if analysis['confidence'] > 0.1:  # Only use tweets with some sentiment
                    sentiments.append(analysis['sentiment'])
                    
                    # Weight by engagement
                    metrics = tweet.public_metrics
                    engagement = (
                        metrics['retweet_count'] * 3 +
                        metrics['like_count'] * 2 + 
                        metrics['reply_count'] * 5 +
                        metrics['quote_count'] * 4
                    )
                    engagement_weights.append(max(1, engagement))
                    
            if not sentiments:
                return self._empty_intelligence('no_sentiment_data')
                
            # Calculate weighted average sentiment
            weighted_sentiment = np.average(sentiments, weights=engagement_weights)
            confidence = min(1.0, len(sentiments) / 25)  # Max confidence with 25+ tweets
            
            return {
                'sentiment_score': weighted_sentiment,
                'confidence': confidence,
                'tweets_analyzed': len(sentiments),
                'total_tweets': len(tweets.data),
                'avg_engagement': np.mean(engagement_weights),
                'source': 'twitter_real',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._empty_intelligence(f'twitter_error: {str(e)[:50]}')
            
    async def collect_reddit_intelligence(self, symbol: str) -> Dict:
        """Raccoglie intelligence da Reddit"""
        try:
            import praw
            
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                return self._empty_intelligence('reddit_unavailable')
                
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent='CryptoAI/2.0'
            )
            
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets']
            all_sentiments = []
            all_weights = []
            posts_analyzed = 0
            
            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    
                    for post in subreddit.hot(limit=15):
                        content = post.title + ' ' + (post.selftext or '')
                        
                        if symbol.lower() in content.lower():
                            analysis = self.sentiment_analyzer.analyze_text(content)
                            
                            if analysis['confidence'] > 0.1:
                                all_sentiments.append(analysis['sentiment'])
                                
                                # Weight by Reddit score and comments
                                weight = max(1, post.score) + post.num_comments * 2
                                all_weights.append(weight)
                                posts_analyzed += 1
                                
                except Exception:
                    continue
                    
            if not all_sentiments:
                return self._empty_intelligence('no_reddit_data')
                
            weighted_sentiment = np.average(all_sentiments, weights=all_weights)
            confidence = min(1.0, posts_analyzed / 10)  # Max confidence with 10+ posts
            
            return {
                'sentiment_score': weighted_sentiment,
                'confidence': confidence,
                'posts_analyzed': posts_analyzed,
                'avg_score': np.mean(all_weights),
                'source': 'reddit_real',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._empty_intelligence(f'reddit_error: {str(e)[:50]}')
            
    async def collect_news_intelligence(self, symbol: str) -> Dict:
        """Raccoglie intelligence da news"""
        
        # Try NewsAPI first
        news_data = await self._collect_newsapi_data(symbol)
        if news_data['confidence'] > 0:
            return news_data
            
        # Try Alpha Vantage news as backup
        alpha_data = await self._collect_alpha_vantage_news(symbol)
        return alpha_data
        
    async def _collect_newsapi_data(self, symbol: str) -> Dict:
        """Raccoglie da NewsAPI"""
        try:
            api_key = os.getenv('NEWSAPI_KEY')
            if not api_key:
                return self._empty_intelligence('newsapi_unavailable')
                
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{symbol} cryptocurrency',
                'sortBy': 'publishedAt',
                'pageSize': 30,
                'apiKey': api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                return self._empty_intelligence(f'newsapi_http_{response.status_code}')
                
            data = response.json()
            articles = data.get('articles', [])
            
            if not articles:
                return self._empty_intelligence('no_news_articles')
                
            sentiments = []
            
            for article in articles[:20]:
                title = article.get('title', '')
                description = article.get('description', '')
                content = title + ' ' + (description or '')
                
                analysis = self.sentiment_analyzer.analyze_text(content)
                
                if analysis['confidence'] > 0.05:
                    sentiments.append(analysis['sentiment'])
                    
            if not sentiments:
                return self._empty_intelligence('no_news_sentiment')
                
            avg_sentiment = np.mean(sentiments)
            confidence = min(1.0, len(sentiments) / 15)
            
            return {
                'sentiment_score': avg_sentiment,
                'confidence': confidence,
                'articles_analyzed': len(sentiments),
                'total_articles': len(articles),
                'source': 'newsapi_real',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._empty_intelligence(f'newsapi_error: {str(e)[:50]}')
            
    async def _collect_alpha_vantage_news(self, symbol: str) -> Dict:
        """Raccoglie news da Alpha Vantage"""
        try:
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                return self._empty_intelligence('alpha_vantage_unavailable')
                
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': f'{symbol}-USD',
                'limit': 25,
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                return self._empty_intelligence(f'alpha_vantage_http_{response.status_code}')
                
            data = response.json()
            
            if 'feed' not in data:
                return self._empty_intelligence('no_alpha_vantage_feed')
                
            articles = data['feed']
            if not articles:
                return self._empty_intelligence('no_alpha_vantage_articles')
                
            sentiments = []
            
            for article in articles[:15]:
                # Use Alpha Vantage sentiment if available
                if 'overall_sentiment_score' in article:
                    # Convert from -1,1 to 0,1 scale
                    av_sentiment = float(article['overall_sentiment_score'])
                    normalized_sentiment = (av_sentiment + 1) / 2
                    sentiments.append(normalized_sentiment)
                else:
                    # Fallback to our analysis
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    content = title + ' ' + summary
                    
                    analysis = self.sentiment_analyzer.analyze_text(content)
                    if analysis['confidence'] > 0.05:
                        sentiments.append(analysis['sentiment'])
                        
            if not sentiments:
                return self._empty_intelligence('no_alpha_vantage_sentiment')
                
            avg_sentiment = np.mean(sentiments)
            confidence = min(1.0, len(sentiments) / 10)
            
            return {
                'sentiment_score': avg_sentiment,
                'confidence': confidence,
                'articles_analyzed': len(sentiments),
                'total_articles': len(articles),
                'source': 'alpha_vantage_news',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._empty_intelligence(f'alpha_vantage_error: {str(e)[:50]}')
            
    def _empty_intelligence(self, reason: str) -> Dict:
        """Ritorna intelligence vuota con motivo"""
        return {
            'sentiment_score': 0.5,
            'confidence': 0.0,
            'source': reason,
            'timestamp': datetime.now().isoformat()
        }
        
    async def get_comprehensive_intelligence(self, symbol: str) -> Dict:
        """Raccoglie intelligence completa da tutte le fonti"""
        
        # Collect from all sources
        twitter_intel = await self.collect_twitter_intelligence(symbol)
        reddit_intel = await self.collect_reddit_intelligence(symbol)
        news_intel = await self.collect_news_intelligence(symbol)
        
        # Combine all intelligence
        all_sources = {
            'twitter': twitter_intel,
            'reddit': reddit_intel,
            'news': news_intel
        }
        
        # Calculate weighted composite sentiment
        total_sentiment = 0
        total_weight = 0
        active_sources = []
        
        for source_name, intel in all_sources.items():
            confidence = intel['confidence']
            
            if confidence > 0.1:  # Only use sources with reasonable confidence
                weight = confidence
                total_sentiment += intel['sentiment_score'] * weight
                total_weight += weight
                active_sources.append(source_name)
                
        if total_weight > 0:
            composite_sentiment = total_sentiment / total_weight
            overall_confidence = min(1.0, total_weight / 2)  # Max confidence when multiple sources agree
        else:
            composite_sentiment = 0.5  # Neutral default
            overall_confidence = 0.0
            
        # Determine market signal
        if composite_sentiment > 0.65 and overall_confidence > 0.4:
            market_signal = 'BULLISH'
            signal_strength = composite_sentiment
        elif composite_sentiment < 0.35 and overall_confidence > 0.4:
            market_signal = 'BEARISH'
            signal_strength = 1 - composite_sentiment
        else:
            market_signal = 'NEUTRAL'
            signal_strength = 0.5
            
        return {
            'symbol': symbol,
            'composite_sentiment': composite_sentiment,
            'market_signal': market_signal,
            'signal_strength': signal_strength,
            'overall_confidence': overall_confidence,
            'active_sources': active_sources,
            'source_breakdown': all_sources,
            'intelligence_quality': len(active_sources) / 3,  # Quality score 0-1
            'timestamp': datetime.now().isoformat()
        }

async def test_lightweight_ai():
    """Test del sistema AI leggero"""
    
    print("Testing Lightweight Real AI System")
    print("=" * 40)
    
    intel_system = LightweightMarketIntelligence()
    
    symbols = ['BTC', 'ETH']
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        
        intelligence = await intel_system.get_comprehensive_intelligence(symbol)
        
        print(f"Market Signal: {intelligence['market_signal']}")
        print(f"Sentiment Score: {intelligence['composite_sentiment']:.3f}")
        print(f"Confidence: {intelligence['overall_confidence']:.2%}")
        print(f"Active Sources: {intelligence['active_sources']}")
        print(f"Intelligence Quality: {intelligence['intelligence_quality']:.2%}")
        
        for source, data in intelligence['source_breakdown'].items():
            conf = data['confidence']
            if conf > 0:
                print(f"  {source}: {data['sentiment_score']:.3f} ({conf:.2%} confidence)")
                
    print(f"\nLightweight AI Test Complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_lightweight_ai())