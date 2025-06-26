#!/usr/bin/env python3
"""
Setup API Keys for Real AI Trading System
Guida interattiva per configurare tutte le API keys necessarie
"""

import os
import requests
import json
from pathlib import Path

class APIKeySetup:
    def __init__(self):
        self.env_file = Path('.env')
        self.api_keys = {}
        
    def welcome_message(self):
        print("🔑 Setup API Keys per Real AI Trading System")
        print("=" * 50)
        print("Questa guida ti aiuterà a configurare tutte le API keys necessarie")
        print("per attivare i veri modelli AI invece delle simulazioni.\n")
        
    def setup_huggingface(self):
        print("1. HUGGINGFACE TOKEN (GRATIS)")
        print("-" * 30)
        print("Necessario per scaricare modelli BERT, FinBERT, etc.")
        print("👉 Vai su: https://huggingface.co/settings/tokens")
        print("👉 Crea account gratuito se necessario")
        print("👉 Clicca 'New token' → tipo 'Read' → copia il token")
        
        token = input("\nInserisci HUGGINGFACE_TOKEN (o Enter per saltare): ").strip()
        if token:
            self.api_keys['HUGGINGFACE_TOKEN'] = token
            print("✅ HuggingFace token salvato")
        else:
            print("⚠️ HuggingFace token saltato")
        print()
        
    def setup_twitter(self):
        print("2. TWITTER BEARER TOKEN (GRATIS con limiti)")
        print("-" * 40)
        print("Per analisi sentiment dai tweet crypto")
        print("👉 Vai su: https://developer.twitter.com/")
        print("👉 Crea account sviluppatore (chiede motivo)")
        print("👉 Scrivi: 'Academic research on cryptocurrency sentiment'")
        print("👉 Crea progetto → genera Bearer Token")
        
        token = input("\nInserisci TWITTER_BEARER_TOKEN (o Enter per saltare): ").strip()
        if token:
            self.api_keys['TWITTER_BEARER_TOKEN'] = token
            print("✅ Twitter token salvato")
            
            # Test token
            try:
                headers = {'Authorization': f'Bearer {token}'}
                response = requests.get(
                    'https://api.twitter.com/2/tweets/search/recent?query=bitcoin&max_results=10',
                    headers=headers
                )
                if response.status_code == 200:
                    print("✅ Twitter token funziona!")
                else:
                    print(f"⚠️ Twitter token errore: {response.status_code}")
            except:
                print("⚠️ Non riesco a testare Twitter token")
        else:
            print("⚠️ Twitter token saltato")
        print()
        
    def setup_reddit(self):
        print("3. REDDIT API (GRATIS)")
        print("-" * 20)
        print("Per sentiment da community Reddit crypto")
        print("👉 Vai su: https://www.reddit.com/prefs/apps")
        print("👉 Login → 'Create App' → tipo 'Script'")
        print("👉 Redirect URI: http://localhost:8080")
        print("👉 Copia Client ID (sotto nome app) e Secret")
        
        client_id = input("\nInserisci REDDIT_CLIENT_ID (o Enter per saltare): ").strip()
        if client_id:
            client_secret = input("Inserisci REDDIT_CLIENT_SECRET: ").strip()
            if client_secret:
                self.api_keys['REDDIT_CLIENT_ID'] = client_id
                self.api_keys['REDDIT_CLIENT_SECRET'] = client_secret
                print("✅ Reddit credentials salvate")
            else:
                print("⚠️ Secret mancante, Reddit saltato")
        else:
            print("⚠️ Reddit saltato")
        print()
        
    def setup_alpha_vantage(self):
        print("4. ALPHA VANTAGE API (GRATIS)")
        print("-" * 30)
        print("Per news finanziarie e sentiment")
        print("👉 Vai su: https://www.alphavantage.co/support/#api-key")
        print("👉 Inserisci email → 'GET FREE API KEY'")
        print("👉 Controlla email per il key")
        
        api_key = input("\nInserisci ALPHA_VANTAGE_API_KEY (o Enter per saltare): ").strip()
        if api_key:
            self.api_keys['ALPHA_VANTAGE_API_KEY'] = api_key
            print("✅ Alpha Vantage key salvato")
            
            # Test key
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={api_key}"
                response = requests.get(url)
                if 'Time Series' in response.text:
                    print("✅ Alpha Vantage key funziona!")
                else:
                    print("⚠️ Alpha Vantage key potrebbe essere invalido")
            except:
                print("⚠️ Non riesco a testare Alpha Vantage key")
        else:
            print("⚠️ Alpha Vantage saltato")
        print()
        
    def setup_newsapi(self):
        print("5. NEWS API (GRATIS con limiti)")
        print("-" * 30)
        print("Per feed news real-time")
        print("👉 Vai su: https://newsapi.org/register")
        print("👉 Crea account gratuito")
        print("👉 Dashboard → copia API Key")
        
        api_key = input("\nInserisci NEWSAPI_KEY (o Enter per saltare): ").strip()
        if api_key:
            self.api_keys['NEWSAPI_KEY'] = api_key
            print("✅ NewsAPI key salvato")
            
            # Test key
            try:
                url = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={api_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    print("✅ NewsAPI key funziona!")
                else:
                    print(f"⚠️ NewsAPI errore: {response.status_code}")
            except:
                print("⚠️ Non riesco a testare NewsAPI key")
        else:
            print("⚠️ NewsAPI saltato")
        print()
        
    def save_env_file(self):
        """Salva tutte le API keys nel file .env"""
        
        if not self.api_keys:
            print("❌ Nessuna API key configurata")
            return False
            
        # Leggi .env esistente se presente
        existing_env = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        existing_env[key] = value
                        
        # Aggiorna con nuove keys
        existing_env.update(self.api_keys)
        
        # Salva tutto
        with open(self.env_file, 'w') as f:
            f.write("# AI Trading Bot - API Keys\n")
            f.write("# Generated by setup_api_keys.py\n\n")
            
            for key, value in existing_env.items():
                f.write(f"{key}={value}\n")
                
        print(f"💾 API keys salvate in {self.env_file}")
        return True
        
    def show_summary(self):
        """Mostra riepilogo delle API keys configurate"""
        
        print("\n📊 RIEPILOGO CONFIGURAZIONE")
        print("=" * 40)
        
        total_keys = len(self.api_keys)
        
        if total_keys == 0:
            print("❌ Nessuna API key configurata")
            print("Il sistema userà solo algoritmi simulati")
            return
            
        print(f"✅ {total_keys} API keys configurate:")
        
        for key in self.api_keys:
            service = key.replace('_TOKEN', '').replace('_API_KEY', '').replace('_', ' ')
            print(f"  • {service}")
            
        print(f"\n🎯 PROSSIMI PASSI:")
        print("1. Riavvia l'applicazione AI trading")
        print("2. I modelli AI reali si attiveranno automaticamente")
        print("3. Controlla i logs per confermare il download dei modelli")
        
        if total_keys >= 3:
            print("\n🎉 Configurazione OTTIMA per AI trading reale!")
        elif total_keys >= 2:
            print("\n👍 Configurazione BUONA per iniziare")
        else:
            print("\n⚠️ Configurazione MINIMA - considera di aggiungere più APIs")
            
    def run_setup(self):
        """Esegue setup completo"""
        
        self.welcome_message()
        
        # Setup ogni API
        self.setup_huggingface()
        self.setup_alpha_vantage()  # Più facile, mettiamo prima
        self.setup_reddit()
        self.setup_newsapi()
        self.setup_twitter()  # Più complesso, mettiamo dopo
        
        # Salva e riassumi
        if self.save_env_file():
            self.show_summary()
        
        return len(self.api_keys)

def main():
    setup = APIKeySetup()
    keys_configured = setup.run_setup()
    
    print(f"\n🏁 Setup completato con {keys_configured} API keys")
    
    if keys_configured > 0:
        print("\nPer attivare i modelli AI reali:")
        print("python real_ai_models_downloader.py")
    
    return setup

if __name__ == "__main__":
    main()