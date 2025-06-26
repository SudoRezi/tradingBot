# Guida Completa per Ottenere API Keys - Real AI Trading

## 1. HUGGINGFACE_TOKEN (GRATIS)

### Procedura:
1. Vai su https://huggingface.co/
2. Crea account gratuito
3. Vai su **Settings** → **Access Tokens**
4. Clicca **New token**
5. Nome: "AI Trading Bot"
6. Tipo: **Read** (sufficiente per scaricare modelli)
7. Copia il token generato

**Costo**: GRATIS
**Limiti**: Nessuno per download modelli pubblici
**Necessario per**: Scaricare BERT, FinBERT, modelli pre-addestrati

---

## 2. TWITTER_BEARER_TOKEN (GRATIS con limiti)

### Procedura:
1. Vai su https://developer.twitter.com/
2. Clicca **Sign up** 
3. Crea account sviluppatore (chiede motivo utilizzo)
4. Scrivi: "Academic research on cryptocurrency sentiment analysis"
5. Verifica email e completa profilo
6. Crea **New Project** → "Crypto Sentiment Analysis"
7. Genera **Bearer Token**

**Costo**: GRATIS (Essential Plan)
**Limiti**: 500,000 tweet/mese (sufficiente)
**Upgrade**: $100/mese per più dati

---

## 3. REDDIT_CLIENT_ID/SECRET (GRATIS)

### Procedura:
1. Vai su https://www.reddit.com/prefs/apps
2. Login con account Reddit
3. Clicca **Create App**
4. Nome: "Crypto Sentiment Bot"
5. Tipo: **Script**
6. Description: "Cryptocurrency sentiment analysis"
7. Redirect URI: `http://localhost:8080`
8. Clicca **Create app**
9. Copia **Client ID** (sotto il nome app)
10. Copia **Secret** (campo "secret")

**Costo**: GRATIS
**Limiti**: 60 richieste/minuto (ok per sentiment)
**User Agent**: Usa "CryptoBot/1.0 by YourUsername"

---

## 4. ALPHA_VANTAGE_API_KEY (GRATIS)

### Procedura:
1. Vai su https://www.alphavantage.co/support/#api-key
2. Inserisci email
3. Clicca **GET FREE API KEY**
4. Verifica email
5. Copia API key dalla email

**Costo**: GRATIS
**Limiti**: 500 richieste/giorno (sufficiente per news)
**Upgrade**: $50/mese per più richieste

---

## 5. NEWSAPI_KEY (GRATIS con limiti)

### Procedura:
1. Vai su https://newsapi.org/register
2. Crea account gratuito
3. Verifica email
4. Dashboard → copia **API Key**

**Costo**: GRATIS (Developer Plan)
**Limiti**: 1,000 richieste/giorno
**Upgrade**: $449/mese per uso commerciale

---

## 6. BINANCE_API_KEY (Per dati real-time)

### Procedura:
1. Crea account su Binance.com
2. Completa verifica identità
3. Vai su **API Management**
4. Crea **New API Key**
5. Nome: "Trading Bot"
6. **Abilita solo "Read Info"** (NO trading per sicurezza)
7. Copia API Key e Secret

**Costo**: GRATIS
**Requisiti**: Account verificato
**Permessi**: Solo lettura dati mercato

---

## Setup Automatico delle API Keys

### Script Setup
```python
# Salva questo come setup_api_keys.py
import os

def setup_environment_variables():
    """Setup delle variabili ambiente per le API keys"""
    
    api_keys = {
        'HUGGINGFACE_TOKEN': '',
        'TWITTER_BEARER_TOKEN': '',
        'REDDIT_CLIENT_ID': '',
        'REDDIT_CLIENT_SECRET': '',
        'ALPHA_VANTAGE_API_KEY': '',
        'NEWSAPI_KEY': '',
        'BINANCE_API_KEY': '',
        'BINANCE_SECRET_KEY': ''
    }
    
    print("🔑 Setup API Keys per Real AI Trading System")
    print("=" * 50)
    
    for key_name, _ in api_keys.items():
        current_value = os.getenv(key_name, '')
        if current_value:
            print(f"✅ {key_name}: Already set")
        else:
            value = input(f"Inserisci {key_name}: ").strip()
            if value:
                # Salva in .env file
                with open('.env', 'a') as f:
                    f.write(f"{key_name}={value}\n")
                print(f"✅ {key_name}: Saved")
            else:
                print(f"⚠️ {key_name}: Skipped")
    
    print("\n🎉 Setup completato!")
    print("Riavvia l'applicazione per usare le nuove API keys")

if __name__ == "__main__":
    setup_environment_variables()
```

### Test delle API Keys
```python
# Salva questo come test_api_keys.py
import os
import requests
import tweepy
import praw
from transformers import pipeline

def test_all_apis():
    """Test tutte le API keys configurate"""
    
    results = {}
    
    # Test HuggingFace
    try:
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            results['HuggingFace'] = "✅ Working"
        else:
            results['HuggingFace'] = "❌ No token"
    except Exception as e:
        results['HuggingFace'] = f"❌ Error: {e}"
    
    # Test Twitter
    try:
        twitter_token = os.getenv('TWITTER_BEARER_TOKEN')
        if twitter_token:
            client = tweepy.Client(bearer_token=twitter_token)
            tweets = client.search_recent_tweets(query="bitcoin", max_results=10)
            results['Twitter'] = f"✅ Working - Found {len(tweets.data)} tweets"
        else:
            results['Twitter'] = "❌ No token"
    except Exception as e:
        results['Twitter'] = f"❌ Error: {e}"
    
    # Test Reddit
    try:
        reddit_id = os.getenv('REDDIT_CLIENT_ID')
        reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
        if reddit_id and reddit_secret:
            reddit = praw.Reddit(
                client_id=reddit_id,
                client_secret=reddit_secret,
                user_agent="CryptoBot/1.0"
            )
            subreddit = reddit.subreddit("cryptocurrency")
            results['Reddit'] = "✅ Working"
        else:
            results['Reddit'] = "❌ No credentials"
    except Exception as e:
        results['Reddit'] = f"❌ Error: {e}"
    
    # Test Alpha Vantage
    try:
        av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if av_key:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={av_key}"
            response = requests.get(url)
            if response.status_code == 200:
                results['Alpha Vantage'] = "✅ Working"
            else:
                results['Alpha Vantage'] = f"❌ HTTP {response.status_code}"
        else:
            results['Alpha Vantage'] = "❌ No key"
    except Exception as e:
        results['Alpha Vantage'] = f"❌ Error: {e}"
    
    # Test NewsAPI
    try:
        news_key = os.getenv('NEWSAPI_KEY')
        if news_key:
            url = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={news_key}"
            response = requests.get(url)
            if response.status_code == 200:
                results['NewsAPI'] = "✅ Working"
            else:
                results['NewsAPI'] = f"❌ HTTP {response.status_code}"
        else:
            results['NewsAPI'] = "❌ No key"
    except Exception as e:
        results['NewsAPI'] = f"❌ Error: {e}"
    
    print("\n🧪 API Keys Test Results:")
    print("=" * 30)
    for service, status in results.items():
        print(f"{service}: {status}")
    
    return results

if __name__ == "__main__":
    test_all_apis()
```

## Priorità di Implementazione

### Phase 1 (Essenziali - GRATIS)
1. **HUGGINGFACE_TOKEN** - Per modelli AI base
2. **ALPHA_VANTAGE_API_KEY** - Per news sentiment
3. **REDDIT_CLIENT_ID/SECRET** - Per community sentiment

### Phase 2 (Avanzate)
4. **TWITTER_BEARER_TOKEN** - Per social sentiment real-time
5. **NEWSAPI_KEY** - Per feed news aggiuntivi
6. **BINANCE_API_KEY** - Per dati mercato real-time

## Limiti Gratuiti vs Necessità

### Sufficiente per AI Trading
- **HuggingFace**: Illimitato per download modelli
- **Alpha Vantage**: 500 req/giorno = ~20 news checks/ora
- **Reddit**: 60 req/min = sentiment ogni minuto
- **Twitter**: 500K tweet/mese = ~650 tweet/ora
- **NewsAPI**: 1000 req/giorno = ~40 news checks/ora

### Quando Upgradefare
- **Trading con +$10K**: Considera upgrade per più dati
- **Multiple asset tracking**: Potrebbero servire più richieste
- **High frequency sentiment**: Twitter Pro plan

## Costi Totali

### Setup Gratuito
- **Costo**: $0/mese
- **Capacità**: Sentiment ogni 1-5 minuti per BTC/ETH
- **Dati**: Sufficiente per algoritmic trading base

### Setup Professionale  
- **Costo**: ~$150/mese (Twitter Pro + Alpha Vantage Pro)
- **Capacità**: Sentiment real-time, multiple asset
- **Dati**: Livello semi-istituzionale

Inizia con il setup gratuito - è più che sufficiente per validare il sistema AI!