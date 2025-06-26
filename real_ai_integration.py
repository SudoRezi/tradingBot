#!/usr/bin/env python3
"""
Real AI Integration - Connects APIs to Autonomous Trading AI
Integra vere API con il sistema AI autonomo per potenziarlo
"""

import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio

class RealAIDataCollector:
    """Raccoglie dati reali per potenziare l'AI autonoma"""
    
    def __init__(self):
        self.load_api_keys()
        self.data_cache = {}
        
    def load_api_keys(self):
        """Carica API keys dalle variabili ambiente"""
        # Load from .env file
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                        
        self.api_keys = {
            'huggingface': os.getenv('HUGGINGFACE_TOKEN'),
            'twitter': os.getenv('TWITTER_BEARER_TOKEN'),
            'reddit_id': os.getenv('REDDIT_CLIENT_ID'),
            'reddit_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'newsapi': os.getenv('NEWSAPI_KEY')
        }
        
    async def collect_twitter_sentiment(self, crypto_symbol: str) -> Dict:
        """Raccoglie sentiment da Twitter per crypto specifico"""
        
        if not self.api_keys['twitter']:
            return {'sentiment_score': 0.5, 'source': 'unavailable', 'confidence': 0.0}
            
        try:
            import tweepy
            
            client = tweepy.Client(bearer_token=self.api_keys['twitter'])
            
            # Search recent tweets
            query = f"${crypto_symbol} OR #{crypto_symbol} OR {crypto_symbol} -is:retweet lang:en"
            tweets = client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return {'sentiment_score': 0.5, 'source': 'no_data', 'confidence': 0.0}
                
            # Analizza sentiment (simulato - in futuro con BERT reale)
            positive_words = ['moon', 'bullish', 'pump', 'buy', 'bull', 'green', 'up', 'rocket', 'hodl']
            negative_words = ['dump', 'bearish', 'sell', 'bear', 'red', 'down', 'crash', 'rekt']
            
            total_sentiment = 0
            tweet_count = 0
            
            for tweet in tweets.data:
                text = tweet.text.lower()
                
                positive_score = sum(1 for word in positive_words if word in text)
                negative_score = sum(1 for word in negative_words if word in text)
                
                if positive_score + negative_score > 0:
                    sentiment = positive_score / (positive_score + negative_score)
                    total_sentiment += sentiment
                    tweet_count += 1
                    
            if tweet_count > 0:
                avg_sentiment = total_sentiment / tweet_count
                confidence = min(1.0, tweet_count / 50)  # Max confidence with 50+ tweets
                
                return {
                    'sentiment_score': avg_sentiment,
                    'tweets_analyzed': tweet_count,
                    'source': 'twitter_real',
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'sentiment_score': 0.5, 'source': 'insufficient_data', 'confidence': 0.0}
                
        except Exception as e:
            print(f"Twitter API error: {e}")
            return {'sentiment_score': 0.5, 'source': 'error', 'confidence': 0.0}
            
    async def collect_reddit_sentiment(self, crypto_symbol: str) -> Dict:
        """Raccoglie sentiment da Reddit crypto communities"""
        
        if not self.api_keys['reddit_id'] or not self.api_keys['reddit_secret']:
            return {'sentiment_score': 0.5, 'source': 'unavailable', 'confidence': 0.0}
            
        try:
            import praw
            
            reddit = praw.Reddit(
                client_id=self.api_keys['reddit_id'],
                client_secret=self.api_keys['reddit_secret'],
                user_agent='CryptoAI/1.0'
            )
            
            # Search in crypto subreddits
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets']
            all_posts = []
            
            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    
                    # Get hot posts mentioning the crypto
                    for post in subreddit.hot(limit=20):
                        if crypto_symbol.lower() in post.title.lower() or crypto_symbol.lower() in post.selftext.lower():
                            all_posts.append({
                                'title': post.title,
                                'text': post.selftext,
                                'score': post.score,
                                'comments': post.num_comments
                            })
                except:
                    continue
                    
            if not all_posts:
                return {'sentiment_score': 0.5, 'source': 'no_posts', 'confidence': 0.0}
                
            # Analizza sentiment dei posts
            positive_words = ['bullish', 'moon', 'buy', 'hodl', 'pump', 'green', 'profit']
            negative_words = ['bearish', 'dump', 'sell', 'crash', 'red', 'loss', 'rekt']
            
            total_sentiment = 0
            weighted_total = 0
            
            for post in all_posts:
                text = (post['title'] + ' ' + post['text']).lower()
                
                positive_score = sum(1 for word in positive_words if word in text)
                negative_score = sum(1 for word in negative_words if word in text)
                
                if positive_score + negative_score > 0:
                    sentiment = positive_score / (positive_score + negative_score)
                    weight = max(1, post['score'])  # Use Reddit score as weight
                    
                    total_sentiment += sentiment * weight
                    weighted_total += weight
                    
            if weighted_total > 0:
                avg_sentiment = total_sentiment / weighted_total
                confidence = min(1.0, len(all_posts) / 10)  # Max confidence with 10+ posts
                
                return {
                    'sentiment_score': avg_sentiment,
                    'posts_analyzed': len(all_posts),
                    'source': 'reddit_real',
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'sentiment_score': 0.5, 'source': 'insufficient_data', 'confidence': 0.0}
                
        except Exception as e:
            print(f"Reddit API error: {e}")
            return {'sentiment_score': 0.5, 'source': 'error', 'confidence': 0.0}
            
    async def collect_news_sentiment(self, crypto_symbol: str) -> Dict:
        """Raccoglie sentiment da news finanziarie"""
        
        sentiment_data = {}
        
        # Alpha Vantage News
        if self.api_keys['alpha_vantage']:
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': f'{crypto_symbol}-USD',
                    'limit': 20,
                    'apikey': self.api_keys['alpha_vantage']
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'feed' in data:
                        articles = data['feed']
                        
                        if articles:
                            total_sentiment = 0
                            for article in articles[:10]:  # First 10 articles
                                if 'overall_sentiment_score' in article:
                                    # Alpha Vantage sentiment score (-1 to 1)
                                    score = float(article['overall_sentiment_score'])
                                    # Convert to 0-1 scale
                                    normalized_score = (score + 1) / 2
                                    total_sentiment += normalized_score
                                    
                            avg_sentiment = total_sentiment / len(articles[:10])
                            sentiment_data['alpha_vantage'] = {
                                'sentiment_score': avg_sentiment,
                                'articles_count': len(articles[:10]),
                                'confidence': 0.8
                            }
                            
            except Exception as e:
                print(f"Alpha Vantage error: {e}")
                
        # NewsAPI
        if self.api_keys['newsapi']:
            try:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': f'{crypto_symbol} cryptocurrency',
                    'sortBy': 'publishedAt',
                    'pageSize': 20,
                    'apiKey': self.api_keys['newsapi']
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('articles'):
                        articles = data['articles']
                        
                        # Simple sentiment analysis
                        positive_words = ['surge', 'bull', 'gain', 'rise', 'up', 'positive', 'growth']
                        negative_words = ['crash', 'bear', 'fall', 'down', 'negative', 'decline', 'drop']
                        
                        total_sentiment = 0
                        analyzed_count = 0
                        
                        for article in articles[:15]:
                            title = article.get('title', '').lower()
                            description = article.get('description', '').lower()
                            text = title + ' ' + description
                            
                            positive_score = sum(1 for word in positive_words if word in text)
                            negative_score = sum(1 for word in negative_words if word in text)
                            
                            if positive_score + negative_score > 0:
                                sentiment = positive_score / (positive_score + negative_score)
                                total_sentiment += sentiment
                                analyzed_count += 1
                                
                        if analyzed_count > 0:
                            avg_sentiment = total_sentiment / analyzed_count
                            sentiment_data['newsapi'] = {
                                'sentiment_score': avg_sentiment,
                                'articles_count': analyzed_count,
                                'confidence': 0.7
                            }
                            
            except Exception as e:
                print(f"NewsAPI error: {e}")
                
        # Combine all news sentiment
        if sentiment_data:
            combined_sentiment = 0
            total_weight = 0
            
            for source, data in sentiment_data.items():
                weight = data['confidence']
                combined_sentiment += data['sentiment_score'] * weight
                total_weight += weight
                
            final_sentiment = combined_sentiment / total_weight if total_weight > 0 else 0.5
            
            return {
                'sentiment_score': final_sentiment,
                'sources': list(sentiment_data.keys()),
                'source': 'news_real',
                'confidence': min(1.0, total_weight),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {'sentiment_score': 0.5, 'source': 'unavailable', 'confidence': 0.0}
            
    async def collect_market_intelligence(self, crypto_symbol: str) -> Dict:
        """Raccoglie intelligence completa per potenziare AI"""
        
        print(f"Collecting real market intelligence for {crypto_symbol}...")
        
        # Collect from all sources in parallel
        tasks = [
            self.collect_twitter_sentiment(crypto_symbol),
            self.collect_reddit_sentiment(crypto_symbol),
            self.collect_news_sentiment(crypto_symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        twitter_data = results[0] if not isinstance(results[0], Exception) else {'sentiment_score': 0.5, 'confidence': 0.0}
        reddit_data = results[1] if not isinstance(results[1], Exception) else {'sentiment_score': 0.5, 'confidence': 0.0}
        news_data = results[2] if not isinstance(results[2], Exception) else {'sentiment_score': 0.5, 'confidence': 0.0}
        
        # Combine all intelligence
        all_sources = {
            'twitter': twitter_data,
            'reddit': reddit_data,
            'news': news_data
        }
        
        # Calculate weighted average sentiment
        total_sentiment = 0
        total_weight = 0
        active_sources = []
        
        for source, data in all_sources.items():
            if data['confidence'] > 0:
                weight = data['confidence']
                total_sentiment += data['sentiment_score'] * weight
                total_weight += weight
                active_sources.append(source)
                
        if total_weight > 0:
            final_sentiment = total_sentiment / total_weight
            overall_confidence = min(1.0, total_weight / len(all_sources))
        else:
            final_sentiment = 0.5  # Neutral
            overall_confidence = 0.0
            
        # Determine market signal
        if final_sentiment > 0.65 and overall_confidence > 0.5:
            market_signal = 'BULLISH'
            signal_strength = final_sentiment
        elif final_sentiment < 0.35 and overall_confidence > 0.5:
            market_signal = 'BEARISH'
            signal_strength = 1 - final_sentiment
        else:
            market_signal = 'NEUTRAL'
            signal_strength = 0.5
            
        intelligence = {
            'symbol': crypto_symbol,
            'overall_sentiment': final_sentiment,
            'market_signal': market_signal,
            'signal_strength': signal_strength,
            'confidence': overall_confidence,
            'active_sources': active_sources,
            'source_breakdown': all_sources,
            'timestamp': datetime.now().isoformat(),
            'intelligence_score': final_sentiment * overall_confidence
        }
        
        print(f"Intelligence collected: {market_signal} signal ({signal_strength:.2f}) from {len(active_sources)} sources")
        
        return intelligence

class EnhancedAutonomousAI:
    """AI Autonoma potenziata con dati reali"""
    
    def __init__(self):
        self.data_collector = RealAIDataCollector()
        self.intelligence_cache = {}
        
    async def make_enhanced_decision(self, symbol: str, market_data: Dict) -> Dict:
        """Decisione AI potenziata con dati reali"""
        
        # Collect real market intelligence
        intelligence = await self.data_collector.collect_market_intelligence(symbol)
        
        # Base technical decision (from existing system)
        base_decision = self.get_technical_decision(market_data)
        
        # Enhance with real intelligence
        enhanced_decision = self.combine_intelligence(base_decision, intelligence)
        
        return enhanced_decision
        
    def get_technical_decision(self, market_data: Dict) -> Dict:
        """Decisione tecnica base (sistema esistente)"""
        
        rsi = market_data.get('rsi', 50)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        price_change = market_data.get('price_change_24h', 0)
        
        # Technical analysis scoring
        technical_score = 0.5
        
        if rsi < 30:
            technical_score += 0.2  # Oversold
        elif rsi > 70:
            technical_score -= 0.2  # Overbought
            
        if volume_ratio > 1.5:
            technical_score += 0.1  # High volume
            
        if price_change > 0.03:
            technical_score += 0.15  # Strong momentum
        elif price_change < -0.03:
            technical_score -= 0.15
            
        # Determine action
        if technical_score > 0.65:
            action = 'BUY'
            confidence = min(0.9, technical_score)
        elif technical_score < 0.35:
            action = 'SELL'  
            confidence = min(0.9, 1 - technical_score)
        else:
            action = 'HOLD'
            confidence = 0.5
            
        return {
            'action': action,
            'confidence': confidence,
            'technical_score': technical_score,
            'reasoning': f'Technical analysis: {action} ({confidence:.2%} confidence)'
        }
        
    def combine_intelligence(self, technical_decision: Dict, intelligence: Dict) -> Dict:
        """Combina analisi tecnica con intelligence reale"""
        
        # Weights
        technical_weight = 0.6
        intelligence_weight = 0.4
        
        # Get intelligence signal
        intelligence_score = intelligence['overall_sentiment']
        intelligence_confidence = intelligence['confidence']
        
        # Combine scores
        if intelligence_confidence > 0.3:  # Only use if reliable
            combined_score = (
                technical_decision['technical_score'] * technical_weight +
                intelligence_score * intelligence_weight
            )
            
            # Adjust confidence based on agreement
            technical_signal = 'BUY' if technical_decision['technical_score'] > 0.6 else 'SELL' if technical_decision['technical_score'] < 0.4 else 'NEUTRAL'
            intelligence_signal = intelligence['market_signal']
            
            if technical_signal == intelligence_signal or intelligence_signal == 'NEUTRAL':
                confidence_bonus = 0.1  # Signals agree
            else:
                confidence_bonus = -0.1  # Signals conflict
                
            final_confidence = min(0.95, technical_decision['confidence'] + confidence_bonus)
            
        else:
            # Low intelligence confidence, use mainly technical
            combined_score = technical_decision['technical_score']
            final_confidence = technical_decision['confidence'] * 0.9  # Slight penalty for lack of intelligence
            
        # Final decision
        if combined_score > 0.65:
            final_action = 'BUY'
        elif combined_score < 0.35:
            final_action = 'SELL'
        else:
            final_action = 'HOLD'
            
        enhanced_reasoning = (
            f"Enhanced AI Decision: {final_action} "
            f"(Technical: {technical_decision['technical_score']:.2f}, "
            f"Intelligence: {intelligence_score:.2f} from {len(intelligence['active_sources'])} sources)"
        )
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'combined_score': combined_score,
            'technical_component': technical_decision['technical_score'],
            'intelligence_component': intelligence_score,
            'intelligence_sources': intelligence['active_sources'],
            'market_signal': intelligence['market_signal'],
            'reasoning': enhanced_reasoning,
            'intelligence_data': intelligence,
            'enhancement_used': intelligence_confidence > 0.3
        }

async def test_real_ai_integration():
    """Test integrazione AI reale"""
    
    print("🚀 Testing Real AI Integration with Live APIs")
    print("=" * 50)
    
    enhanced_ai = EnhancedAutonomousAI()
    
    # Test symbols
    symbols = ['BTC', 'ETH', 'SOL']
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}...")
        
        # Simulated market data
        market_data = {
            'rsi': np.random.uniform(20, 80),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'price_change_24h': np.random.uniform(-0.1, 0.1),
            'volatility': np.random.uniform(0.01, 0.05)
        }
        
        # Get enhanced AI decision
        decision = await enhanced_ai.make_enhanced_decision(symbol, market_data)
        
        print(f"  🤖 AI Decision: {decision['action']}")
        print(f"  📈 Confidence: {decision['confidence']:.2%}")
        print(f"  🧠 Intelligence Sources: {decision['intelligence_sources']}")
        print(f"  📊 Market Signal: {decision['market_signal']}")
        print(f"  ⚡ Enhancement Used: {decision['enhancement_used']}")
        print(f"  💭 Reasoning: {decision['reasoning']}")
        
    print(f"\n✅ Real AI Integration Test Completed!")

if __name__ == "__main__":
    asyncio.run(test_real_ai_integration())