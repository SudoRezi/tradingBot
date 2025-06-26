# API Requirements per Trading Bot

## Cosa Serve Realmente

### Per Exchange Centralizati (CEX)
**SI, bastano solo le API Keys per far funzionare il programma!**

#### Binance
- ✅ **API Key** 
- ✅ **Secret Key**
- ❌ Passphrase NON necessaria
- **Permessi richiesti:** Spot Trading, Futures Trading (opzionale), Read Info

#### Bybit  
- ✅ **API Key**
- ✅ **Secret Key** 
- ❌ Passphrase NON necessaria
- **Permessi richiesti:** Trade, Read

#### Coinbase Pro/Advanced
- ✅ **API Key**
- ✅ **Secret Key**
- ❌ Passphrase NON necessaria (rimossa dai nuovi account)
- **Permessi richiesti:** Trade, View

#### Kraken
- ✅ **API Key**
- ✅ **Private Key** (chiamano così il secret)
- ❌ Passphrase NON necessaria
- **Permessi richiesti:** Query Funds, Create & Modify Orders

#### OKX
- ✅ **API Key**
- ✅ **Secret Key**
- ❌ Passphrase NON necessaria per trading normale
- **Permessi richiesti:** Trade, Read

#### KuCoin
- ✅ **API Key**
- ✅ **Secret Key**
- ❌ Passphrase NON necessaria
- **Permessi richiesti:** General, Trade

### Per Exchange Decentralizzati (DEX)
#### Uniswap, PancakeSwap, 1inch
- ✅ **Wallet Address** (public)
- ✅ **Private Key** (per firmare transazioni)
- ✅ **RPC URL** (nodo Ethereum/BSC)
- ❌ API Keys NON necessarie

## Configurazione Corretta API

### Step per Ottenere API Keys:

1. **Registrati sull'exchange**
2. **Vai in Account/API Management**
3. **Crea nuova API Key**
4. **Abilita permessi:**
   - ✅ Read/Query
   - ✅ Spot Trading
   - ✅ Futures Trading (se vuoi usare leverage)
   - ❌ Withdraw (NON abilitare per sicurezza)
5. **Copia API Key e Secret**
6. **Inserisci nel bot**

### Sicurezza API:
- **IP Whitelist:** Consigliato limitare a IP specifici
- **Withdraw Permission:** MAI abilitare
- **Testnet:** Usa sempre prima per test
- **Backup Keys:** Salva in modo sicuro

### Perché Non Serve la Passphrase:
- La passphrase era richiesta solo per alcuni exchange più vecchi
- Le API moderne usano solo Key + Secret + firma HMAC
- Gli exchange hanno rimosso questo requisito per semplificare
- Il bot usa autenticazione standard REST API

## Conferma: SI, le API bastano!

Il trading bot funziona completamente con sole API Keys. Non servono:
- ❌ Passphrase
- ❌ Accesso web browser  
- ❌ 2FA codes
- ❌ App mobile
- ❌ Login manuale

Le API forniscono:
- ✅ Accesso a dati mercato real-time
- ✅ Piazzamento ordini automatico
- ✅ Gestione portfolio
- ✅ Storico transazioni
- ✅ Bilanciamento automatico

**Il sistema è completamente autonomo una volta configurate le API Keys.**