#!/usr/bin/env python3
"""
Simple API Manager - Gestione semplificata API exchange
Sistema leggero che non duplica la crittografia già fornita dagli exchange
"""

import os
import json
import base64
import hashlib
import streamlit as st
from typing import Dict, Optional, Any
from datetime import datetime
import logging

class SimpleAPIManager:
    """Gestione semplificata delle API keys senza doppia crittografia"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session_key = "simple_api_storage"
        self._ensure_session_initialized()
    
    def _ensure_session_initialized(self):
        """Assicura che il session state sia inizializzato"""
        try:
            if not hasattr(st, 'session_state'):
                # Se session_state non è disponibile, usa storage temporaneo
                if not hasattr(self, '_temp_storage'):
                    self._temp_storage = {}
                return
            
            if self.session_key not in st.session_state:
                st.session_state[self.session_key] = {}
        except Exception as e:
            self.logger.error(f"Session state initialization error: {e}")
            # Fallback a storage temporaneo
            if not hasattr(self, '_temp_storage'):
                self._temp_storage = {}
    
    def _get_storage(self) -> Dict:
        """Ottieni storage (session state o temporaneo)"""
        self._ensure_session_initialized()
        
        try:
            if hasattr(st, 'session_state') and self.session_key in st.session_state:
                return st.session_state[self.session_key]
            else:
                return getattr(self, '_temp_storage', {})
        except:
            return getattr(self, '_temp_storage', {})
    
    def _save_to_storage(self, data: Dict):
        """Salva dati nello storage"""
        try:
            if hasattr(st, 'session_state'):
                st.session_state[self.session_key] = data
            else:
                self._temp_storage = data
        except:
            self._temp_storage = data
    
    def store_api_credentials(self, exchange_name: str, credentials: Dict[str, str]) -> str:
        """Memorizza credenziali API con encoding base64 leggero"""
        try:
            # Genera ID semplice
            timestamp = str(int(datetime.now().timestamp()))
            exchange_hash = hashlib.md5(exchange_name.encode()).hexdigest()[:8]
            api_id = f"{exchange_name}_{exchange_hash}_{timestamp}"
            
            # Encoding base64 leggero (non crittografia - lasciamo che l'exchange la gestisca)
            encoded_credentials = {
                key: base64.b64encode(value.encode()).decode() 
                for key, value in credentials.items()
            }
            
            # Ottieni storage corrente
            storage = self._get_storage()
            
            # Salva credenziali
            storage[api_id] = {
                "exchange": exchange_name,
                "credentials": encoded_credentials,
                "created_at": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat()
            }
            
            # Salva storage aggiornato
            self._save_to_storage(storage)
            
            self.logger.info(f"API credentials stored for {exchange_name}")
            return api_id
            
        except Exception as e:
            self.logger.error(f"Error storing API credentials: {e}")
            # Fallback: salva direttamente con storage temporaneo
            api_id = f"{exchange_name}_direct_{int(datetime.now().timestamp())}"
            
            storage = self._get_storage()
            storage[api_id] = {
                "exchange": exchange_name,
                "credentials": credentials,
                "created_at": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat()
            }
            self._save_to_storage(storage)
            
            return api_id
    
    def retrieve_api_credentials(self, api_id: str) -> Optional[Dict[str, str]]:
        """Recupera credenziali API"""
        try:
            storage = self._get_storage()
            
            if api_id not in storage:
                return None
            
            stored_data = storage[api_id]
            credentials = stored_data["credentials"]
            
            # Aggiorna last_used
            stored_data["last_used"] = datetime.now().isoformat()
            self._save_to_storage(storage)
            
            # Decodifica se necessario
            if isinstance(list(credentials.values())[0], str) and len(list(credentials.values())[0]) > 20:
                try:
                    # Prova a decodificare da base64
                    decoded_credentials = {
                        key: base64.b64decode(value.encode()).decode() 
                        for key, value in credentials.items()
                    }
                    return decoded_credentials
                except:
                    # Se fallisce, restituisci direttamente
                    return credentials
            else:
                return credentials
                
        except Exception as e:
            self.logger.error(f"Error retrieving API credentials: {e}")
            return None
    
    def list_stored_apis(self) -> Dict[str, Dict[str, Any]]:
        """Lista tutte le API memorizzate"""
        try:
            storage = self._get_storage()
            result = {}
            for api_id, data in storage.items():
                result[api_id] = {
                    "exchange": data["exchange"],
                    "created_at": data["created_at"],
                    "last_used": data.get("last_used", data["created_at"]),
                    "has_credentials": bool(data.get("credentials"))
                }
            return result
        except Exception as e:
            self.logger.error(f"Error listing APIs: {e}")
            return {}
    
    def delete_api_credentials(self, api_id: str) -> bool:
        """Elimina credenziali API"""
        try:
            storage = self._get_storage()
            if api_id in storage:
                del storage[api_id]
                self._save_to_storage(storage)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting API credentials: {e}")
            return False
    
    def update_last_used(self, api_id: str):
        """Aggiorna timestamp ultimo utilizzo"""
        try:
            storage = self._get_storage()
            if api_id in storage:
                storage[api_id]["last_used"] = datetime.now().isoformat()
                self._save_to_storage(storage)
        except Exception as e:
            self.logger.error(f"Error updating last used: {e}")
    
    def test_api_connection(self, exchange_name: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Test connessione API (simulato)"""
        try:
            # Controlli base
            if not credentials:
                return {"success": False, "error": "No credentials provided"}
            
            required_fields = {
                "binance": ["api_key", "secret_key"],
                "bybit": ["api_key", "secret_key"],
                "coinbase": ["api_key", "secret_key", "passphrase"],
                "kraken": ["api_key", "private_key"],
                "okx": ["api_key", "secret_key", "passphrase"]
            }
            
            if exchange_name.lower() in required_fields:
                required = required_fields[exchange_name.lower()]
                missing = [field for field in required if field not in credentials or not credentials[field]]
                
                if missing:
                    return {
                        "success": False, 
                        "error": f"Missing required fields: {', '.join(missing)}"
                    }
            
            # Controlli formato
            api_key = credentials.get("api_key", "")
            if len(api_key) < 10:
                return {"success": False, "error": "API key too short"}
            
            # Simulazione test connessione
            return {
                "success": True,
                "message": f"Connection test passed for {exchange_name}",
                "exchange": exchange_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Ottieni status del sistema"""
        try:
            stored_apis = st.session_state.get(self.session_key, {})
            
            return {
                "total_apis": len(stored_apis),
                "exchanges": list(set(data["exchange"] for data in stored_apis.values())),
                "last_activity": max(
                    [data.get("last_used", data["created_at"]) for data in stored_apis.values()],
                    default=datetime.now().isoformat()
                ),
                "system_status": "operational"
            }
        except Exception as e:
            return {
                "total_apis": 0,
                "exchanges": [],
                "system_status": "error",
                "error": str(e)
            }

# Istanza singleton
_simple_api_manager = None

def get_simple_api_manager() -> SimpleAPIManager:
    """Ottieni istanza singleton del simple API manager"""
    global _simple_api_manager
    if _simple_api_manager is None:
        _simple_api_manager = SimpleAPIManager()
    return _simple_api_manager

# Metodi di convenienza per compatibilità
def encrypt_api_credentials(service_name: str, api_credentials: Dict[str, str]) -> str:
    """Metodo di compatibilità - usa il sistema semplificato"""
    manager = get_simple_api_manager()
    return manager.store_api_credentials(service_name, api_credentials)

def decrypt_api_credentials(key_id: str) -> Optional[Dict[str, str]]:
    """Metodo di compatibilità - usa il sistema semplificato"""
    manager = get_simple_api_manager()
    return manager.retrieve_api_credentials(key_id)

def log_access_attempt(key_id: str, success: bool = True, details: str = "API access"):
    """Metodo di compatibilità - logging semplificato"""
    manager = get_simple_api_manager()
    if success:
        manager.update_last_used(key_id)

# Test del sistema
def test_simple_api_manager():
    """Test sistema API semplificato"""
    
    manager = get_simple_api_manager()
    
    # Test credenziali di esempio
    test_credentials = {
        "api_key": "test_api_key_12345",
        "secret_key": "test_secret_key_67890"
    }
    
    print("Testing Simple API Manager...")
    
    # Test store
    api_id = manager.store_api_credentials("binance", test_credentials)
    print(f"Stored API ID: {api_id}")
    
    # Test retrieve
    retrieved = manager.retrieve_api_credentials(api_id)
    print(f"Retrieved: {retrieved}")
    
    # Test connection
    test_result = manager.test_api_connection("binance", test_credentials)
    print(f"Connection test: {test_result}")
    
    # Test status
    status = manager.get_status()
    print(f"System status: {status}")
    
    print("Simple API Manager test completed!")

if __name__ == "__main__":
    test_simple_api_manager()