"""
ArcticDB Data Manager - Storage Performante per Dati Crypto
Sistema di gestione dati ad alta velocità per tick data e analytics
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class ArcticDataManager:
    """Gestione performante dati crypto con fallback su SQLite"""
    
    def __init__(self, data_path: str = "data/arctic_crypto.db"):
        self.data_path = data_path
        self.arctic_available = False
        self.connection = None
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Inizializza storage con ArcticDB o fallback SQLite"""
        try:
            # Prova ArcticDB
            import arcticdb as adb
            self.arctic_lib = adb.Arctic(f"lmdb://{os.path.dirname(self.data_path)}/arctic")
            self.library = self.arctic_lib.get_library('crypto_data', create_if_missing=True)
            self.arctic_available = True
            print("✅ ArcticDB inizializzato con successo")
            
        except ImportError:
            # Fallback su SQLite ad alta performance
            self._initialize_sqlite_storage()
            print("⚡ Fallback SQLite ad alta performance attivato")
    
    def _initialize_sqlite_storage(self):
        """Inizializza SQLite ottimizzato per performance"""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        self.connection = sqlite3.connect(self.data_path, check_same_thread=False)
        
        # Ottimizzazioni SQLite per performance
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA cache_size=10000")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        
        # Crea tabelle ottimizzate
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS crypto_ohlcv (
                symbol TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS crypto_tick_data (
                symbol TEXT,
                timestamp INTEGER,
                price REAL,
                volume REAL,
                side TEXT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS market_intelligence (
                symbol TEXT,
                timestamp INTEGER,
                data_type TEXT,
                intelligence_data TEXT,
                confidence REAL,
                PRIMARY KEY (symbol, timestamp, data_type)
            )
        """)
        
        # Indici per performance
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON crypto_ohlcv(symbol, timestamp)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_tick_symbol_time ON crypto_tick_data(symbol, timestamp)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_intel_symbol_time ON market_intelligence(symbol, timestamp)")
        
        self.connection.commit()
    
    def store_ohlcv_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Memorizza dati OHLCV"""
        try:
            if self.arctic_available:
                return self._store_arctic_ohlcv(symbol, data)
            else:
                return self._store_sqlite_ohlcv(symbol, data)
        except Exception as e:
            print(f"Errore storage OHLCV {symbol}: {e}")
            return False
    
    def _store_arctic_ohlcv(self, symbol: str, data: pd.DataFrame) -> bool:
        """Memorizza con ArcticDB"""
        try:
            # Prepara dati per ArcticDB
            data_prepared = data.copy()
            if not isinstance(data_prepared.index, pd.DatetimeIndex):
                data_prepared.index = pd.to_datetime(data_prepared.index)
            
            # Write to ArcticDB
            self.library.write(f"ohlcv_{symbol}", data_prepared)
            return True
            
        except Exception as e:
            print(f"Errore ArcticDB per {symbol}: {e}")
            return self._store_sqlite_ohlcv(symbol, data)
    
    def _store_sqlite_ohlcv(self, symbol: str, data: pd.DataFrame) -> bool:
        """Memorizza con SQLite ottimizzato"""
        try:
            # Prepara dati
            data_prepared = data.copy()
            data_prepared['symbol'] = symbol
            data_prepared['timestamp'] = data_prepared.index.astype(np.int64) // 10**9
            
            # Inserimento batch ottimizzato
            data_prepared.to_sql('crypto_ohlcv', self.connection, if_exists='append', index=False, method='multi')
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"Errore SQLite per {symbol}: {e}")
            return False
    
    def get_ohlcv_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Recupera dati OHLCV"""
        try:
            if self.arctic_available:
                return self._get_arctic_ohlcv(symbol, start_date, end_date)
            else:
                return self._get_sqlite_ohlcv(symbol, start_date, end_date)
        except Exception as e:
            print(f"Errore recupero OHLCV {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_arctic_ohlcv(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Recupera da ArcticDB"""
        try:
            if start_date and end_date:
                data = self.library.read(f"ohlcv_{symbol}", date_range=(start_date, end_date))
            else:
                data = self.library.read(f"ohlcv_{symbol}")
            return data.data
            
        except Exception as e:
            print(f"Errore lettura ArcticDB {symbol}: {e}")
            return self._get_sqlite_ohlcv(symbol, start_date, end_date)
    
    def _get_sqlite_ohlcv(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Recupera da SQLite"""
        try:
            query = "SELECT * FROM crypto_ohlcv WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(int(start_date.timestamp()))
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(int(end_date.timestamp()))
            
            query += " ORDER BY timestamp"
            
            data = pd.read_sql_query(query, self.connection, params=params)
            
            if not data.empty:
                data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
                data.set_index('datetime', inplace=True)
                data.drop(['symbol', 'timestamp'], axis=1, inplace=True)
            
            return data
            
        except Exception as e:
            print(f"Errore SQLite query {symbol}: {e}")
            return pd.DataFrame()
    
    def store_market_intelligence(self, symbol: str, intelligence_type: str, data: Dict, confidence: float) -> bool:
        """Memorizza intelligence di mercato"""
        try:
            if not self.connection:
                return False
            
            timestamp = int(datetime.now().timestamp())
            intelligence_json = json.dumps(data)
            
            self.connection.execute("""
                INSERT OR REPLACE INTO market_intelligence 
                (symbol, timestamp, data_type, intelligence_data, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, timestamp, intelligence_type, intelligence_json, confidence))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"Errore storage intelligence {symbol}: {e}")
            return False
    
    def get_market_intelligence(self, symbol: str, intelligence_type: str = None, hours_back: int = 24) -> List[Dict]:
        """Recupera intelligence di mercato"""
        try:
            cutoff_time = int((datetime.now() - timedelta(hours=hours_back)).timestamp())
            
            if intelligence_type:
                query = """
                    SELECT * FROM market_intelligence 
                    WHERE symbol = ? AND data_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """
                params = [symbol, intelligence_type, cutoff_time]
            else:
                query = """
                    SELECT * FROM market_intelligence 
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """
                params = [symbol, cutoff_time]
            
            cursor = self.connection.execute(query, params)
            results = []
            
            for row in cursor.fetchall():
                try:
                    intelligence_data = json.loads(row[3])
                    results.append({
                        'symbol': row[0],
                        'timestamp': datetime.fromtimestamp(row[1]),
                        'data_type': row[2],
                        'data': intelligence_data,
                        'confidence': row[4]
                    })
                except json.JSONDecodeError:
                    continue
            
            return results
            
        except Exception as e:
            print(f"Errore recupero intelligence {symbol}: {e}")
            return []
    
    def get_available_symbols(self) -> List[str]:
        """Ottieni simboli disponibili"""
        try:
            if self.arctic_available:
                # Lista simboli da ArcticDB
                symbols = []
                for key in self.library.list_symbols():
                    if key.startswith('ohlcv_'):
                        symbols.append(key.replace('ohlcv_', ''))
                return symbols
            else:
                # Lista simboli da SQLite
                cursor = self.connection.execute("SELECT DISTINCT symbol FROM crypto_ohlcv")
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"Errore lista simboli: {e}")
            return []
    
    def get_storage_stats(self) -> Dict:
        """Statistiche storage"""
        try:
            stats = {
                'storage_type': 'ArcticDB' if self.arctic_available else 'SQLite',
                'symbols_count': len(self.get_available_symbols()),
                'storage_path': self.data_path
            }
            
            if not self.arctic_available and self.connection:
                # Statistiche SQLite
                cursor = self.connection.execute("SELECT COUNT(*) FROM crypto_ohlcv")
                stats['ohlcv_records'] = cursor.fetchone()[0]
                
                cursor = self.connection.execute("SELECT COUNT(*) FROM market_intelligence")
                stats['intelligence_records'] = cursor.fetchone()[0]
                
                # Dimensione database
                if os.path.exists(self.data_path):
                    stats['db_size_mb'] = os.path.getsize(self.data_path) / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            print(f"Errore stats storage: {e}")
            return {'error': str(e)}
    
    def optimize_storage(self) -> bool:
        """Ottimizza storage"""
        try:
            if not self.arctic_available and self.connection:
                # Ottimizzazioni SQLite
                self.connection.execute("VACUUM")
                self.connection.execute("ANALYZE")
                self.connection.commit()
                return True
            return True
            
        except Exception as e:
            print(f"Errore ottimizzazione storage: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """Pulizia dati vecchi"""
        try:
            cutoff_time = int((datetime.now() - timedelta(days=days_to_keep)).timestamp())
            
            if not self.arctic_available and self.connection:
                # Pulizia SQLite
                self.connection.execute("DELETE FROM market_intelligence WHERE timestamp < ?", (cutoff_time,))
                deleted = self.connection.total_changes
                self.connection.commit()
                print(f"🧹 Puliti {deleted} record di intelligence vecchi")
                return True
            
            return True
            
        except Exception as e:
            print(f"Errore pulizia dati: {e}")
            return False
    
    def close(self):
        """Chiudi connessioni"""
        try:
            if self.connection:
                self.connection.close()
        except Exception as e:
            print(f"Errore chiusura storage: {e}")

def get_arctic_manager() -> ArcticDataManager:
    """Ottieni istanza singleton del data manager"""
    if not hasattr(get_arctic_manager, '_instance'):
        get_arctic_manager._instance = ArcticDataManager()
    return get_arctic_manager._instance