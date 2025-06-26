#!/usr/bin/env python3
"""
Security & Encryption System
Sistema avanzato di sicurezza e crittografia per proteggere API keys e dati sensibili
"""

import os
import json
import sqlite3
import hashlib
import base64
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import logging
from dataclasses import dataclass

@dataclass
class SecurityEvent:
    """Evento di sicurezza"""
    timestamp: datetime
    event_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    source: str
    description: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    additional_data: Optional[Dict] = None

class SecurityEncryptionSystem:
    """Sistema sicurezza e crittografia avanzato"""
    
    def __init__(self, master_key_file=".master_key", security_db="data/security.db"):
        self.master_key_file = master_key_file
        self.security_db = security_db
        self.master_key = None
        self.session_keys = {}
        
        # Setup database
        self._init_security_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inizializza master key
        self._initialize_master_key()
    
    def _init_security_database(self):
        """Inizializza database sicurezza"""
        
        os.makedirs(os.path.dirname(self.security_db), exist_ok=True)
        
        with sqlite3.connect(self.security_db) as conn:
            # Tabella credenziali crittografate
            conn.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT UNIQUE NOT NULL,
                    encrypted_data TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
            """)
            
            # Tabella eventi di sicurezza
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT NOT NULL,
                    user_agent TEXT,
                    ip_address TEXT,
                    additional_data TEXT
                )
            """)
            
            # Tabella sessioni sicure
            conn.execute("""
                CREATE TABLE IF NOT EXISTS secure_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    encrypted_data TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
            """)
            
            # Tabella tentativi di accesso
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    service_name TEXT,
                    ip_address TEXT,
                    user_agent TEXT
                )
            """)
            
            # Indici per performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_access_attempts_timestamp ON access_attempts(timestamp)")
    
    def _initialize_master_key(self):
        """Inizializza o carica master key"""
        
        if os.path.exists(self.master_key_file):
            # Carica master key esistente
            try:
                with open(self.master_key_file, 'rb') as f:
                    self.master_key = f.read()
                
                self._log_security_event(
                    "MASTER_KEY_LOADED", "LOW", "SecuritySystem",
                    "Master key loaded successfully"
                )
                
            except Exception as e:
                self.logger.error(f"Error loading master key: {e}")
                self._generate_new_master_key()
        else:
            # Genera nuova master key
            self._generate_new_master_key()
    
    def _generate_new_master_key(self):
        """Genera nuova master key"""
        
        # Genera master key di 256 bit
        self.master_key = secrets.token_bytes(32)
        
        # Salva master key con permessi restrittivi
        with open(self.master_key_file, 'wb') as f:
            f.write(self.master_key)
        
        # Imposta permessi restrittivi (solo owner read/write)
        os.chmod(self.master_key_file, 0o600)
        
        self._log_security_event(
            "MASTER_KEY_GENERATED", "MEDIUM", "SecuritySystem",
            "New master key generated and saved"
        )
    
    def encrypt_api_credentials(self, service_name: str, credentials: Dict[str, str]) -> bool:
        """Crittografa credenziali API"""
        
        try:
            # Serializza credenziali
            credentials_json = json.dumps(credentials)
            credentials_bytes = credentials_json.encode('utf-8')
            
            # Genera salt unico
            salt = secrets.token_bytes(16)
            
            # Deriva chiave di crittografia dal master key + salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            
            encryption_key = kdf.derive(self.master_key)
            
            # Genera IV
            iv = secrets.token_bytes(16)
            
            # Crittografa con AES-256-CBC
            cipher = Cipher(
                algorithms.AES(encryption_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            
            # Padding PKCS7
            padding_length = 16 - (len(credentials_bytes) % 16)
            padded_data = credentials_bytes + bytes([padding_length] * padding_length)
            
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combina IV + encrypted data
            combined_data = iv + encrypted_data
            encoded_data = base64.b64encode(combined_data).decode('utf-8')
            encoded_salt = base64.b64encode(salt).decode('utf-8')
            
            # Salva nel database
            now = datetime.now().isoformat()
            
            with sqlite3.connect(self.security_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO encrypted_credentials 
                    (service_name, encrypted_data, salt, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (service_name, encoded_data, encoded_salt, now, now))
            
            self._log_security_event(
                "CREDENTIALS_ENCRYPTED", "LOW", "SecuritySystem",
                f"Credentials encrypted for service: {service_name}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error encrypting credentials: {e}")
            
            self._log_security_event(
                "ENCRYPTION_ERROR", "HIGH", "SecuritySystem",
                f"Failed to encrypt credentials for {service_name}: {str(e)}"
            )
            
            return False
    
    def decrypt_api_credentials(self, service_name: str) -> Optional[Dict[str, str]]:
        """Decrittografa credenziali API"""
        
        try:
            # Recupera dati crittografati
            with sqlite3.connect(self.security_db) as conn:
                cursor = conn.execute("""
                    SELECT encrypted_data, salt, access_count 
                    FROM encrypted_credentials 
                    WHERE service_name = ?
                """, (service_name,))
                
                row = cursor.fetchone()
                
                if not row:
                    self._log_security_event(
                        "CREDENTIALS_NOT_FOUND", "MEDIUM", "SecuritySystem",
                        f"Credentials not found for service: {service_name}"
                    )
                    return None
                
                encrypted_data_b64, salt_b64, access_count = row
                
                # Aggiorna contatore accessi
                conn.execute("""
                    UPDATE encrypted_credentials 
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE service_name = ?
                """, (datetime.now().isoformat(), service_name))
            
            # Decodifica dati
            encrypted_data = base64.b64decode(encrypted_data_b64)
            salt = base64.b64decode(salt_b64)
            
            # Estrai IV e dati crittografati
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]
            
            # Deriva chiave di decrittografia
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            
            decryption_key = kdf.derive(self.master_key)
            
            # Decrittografa con AES-256-CBC
            cipher = Cipher(
                algorithms.AES(decryption_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Rimuovi padding PKCS7
            padding_length = padded_data[-1]
            credentials_bytes = padded_data[:-padding_length]
            
            # Deserializza credenziali
            credentials_json = credentials_bytes.decode('utf-8')
            credentials = json.loads(credentials_json)
            
            self._log_security_event(
                "CREDENTIALS_DECRYPTED", "LOW", "SecuritySystem",
                f"Credentials decrypted for service: {service_name}"
            )
            
            return credentials
            
        except Exception as e:
            self.logger.error(f"Error decrypting credentials: {e}")
            
            self._log_security_event(
                "DECRYPTION_ERROR", "HIGH", "SecuritySystem",
                f"Failed to decrypt credentials for {service_name}: {str(e)}"
            )
            
            return None
    
    def create_secure_session(self, session_data: Dict = None, expires_hours: int = 24) -> str:
        """Crea sessione sicura temporanea"""
        
        try:
            # Genera session ID sicuro
            session_id = secrets.token_urlsafe(32)
            
            # Calcola scadenza
            expires_at = datetime.now() + timedelta(hours=expires_hours)
            
            # Crittografa dati sessione se forniti
            encrypted_session_data = None
            if session_data:
                session_json = json.dumps(session_data)
                encrypted_session_data = self._encrypt_session_data(session_json)
            
            # Salva sessione
            with sqlite3.connect(self.security_db) as conn:
                conn.execute("""
                    INSERT INTO secure_sessions 
                    (session_id, created_at, expires_at, encrypted_data)
                    VALUES (?, ?, ?, ?)
                """, (
                    session_id,
                    datetime.now().isoformat(),
                    expires_at.isoformat(),
                    encrypted_session_data
                ))
            
            self._log_security_event(
                "SESSION_CREATED", "LOW", "SecuritySystem",
                f"Secure session created: {session_id[:8]}..."
            )
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error creating secure session: {e}")
            return None
    
    def get_secure_session(self, session_id: str) -> Optional[Dict]:
        """Recupera dati sessione sicura"""
        
        try:
            with sqlite3.connect(self.security_db) as conn:
                cursor = conn.execute("""
                    SELECT encrypted_data, expires_at, access_count
                    FROM secure_sessions 
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                
                if not row:
                    self._log_security_event(
                        "SESSION_NOT_FOUND", "MEDIUM", "SecuritySystem",
                        f"Session not found: {session_id[:8]}..."
                    )
                    return None
                
                encrypted_data, expires_at_str, access_count = row
                
                # Check scadenza
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    # Sessione scaduta, rimuovi
                    conn.execute("DELETE FROM secure_sessions WHERE session_id = ?", (session_id,))
                    
                    self._log_security_event(
                        "SESSION_EXPIRED", "LOW", "SecuritySystem",
                        f"Session expired and removed: {session_id[:8]}..."
                    )
                    return None
                
                # Aggiorna contatore accessi
                conn.execute("""
                    UPDATE secure_sessions 
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE session_id = ?
                """, (datetime.now().isoformat(), session_id))
                
                # Decrittografa dati se presenti
                session_data = None
                if encrypted_data:
                    decrypted_json = self._decrypt_session_data(encrypted_data)
                    if decrypted_json:
                        session_data = json.loads(decrypted_json)
                
                return {
                    "session_id": session_id,
                    "data": session_data,
                    "expires_at": expires_at,
                    "access_count": access_count + 1
                }
                
        except Exception as e:
            self.logger.error(f"Error getting secure session: {e}")
            return None
    
    def _encrypt_session_data(self, data: str) -> str:
        """Crittografa dati di sessione"""
        
        data_bytes = data.encode('utf-8')
        
        # Usa master key direttamente per sessioni temporanee
        iv = secrets.token_bytes(16)
        
        cipher = Cipher(
            algorithms.AES(self.master_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        
        # Padding
        padding_length = 16 - (len(data_bytes) % 16)
        padded_data = data_bytes + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combina IV + encrypted data
        combined_data = iv + encrypted_data
        return base64.b64encode(combined_data).decode('utf-8')
    
    def _decrypt_session_data(self, encrypted_data_b64: str) -> Optional[str]:
        """Decrittografa dati di sessione"""
        
        try:
            encrypted_data = base64.b64decode(encrypted_data_b64)
            
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]
            
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Rimuovi padding
            padding_length = padded_data[-1]
            data_bytes = padded_data[:-padding_length]
            
            return data_bytes.decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error decrypting session data: {e}")
            return None
    
    def _log_security_event(self, event_type: str, severity: str, source: str, 
                          description: str, additional_data: Dict = None):
        """Log evento di sicurezza"""
        
        try:
            with sqlite3.connect(self.security_db) as conn:
                conn.execute("""
                    INSERT INTO security_events 
                    (timestamp, event_type, severity, source, description, additional_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    event_type,
                    severity,
                    source,
                    description,
                    json.dumps(additional_data) if additional_data else None
                ))
                
        except Exception as e:
            self.logger.error(f"Error logging security event: {e}")
    
    def log_access_attempt(self, source: str, success: bool, service_name: str = None,
                          ip_address: str = None, user_agent: str = None):
        """Log tentativo di accesso"""
        
        try:
            with sqlite3.connect(self.security_db) as conn:
                conn.execute("""
                    INSERT INTO access_attempts 
                    (timestamp, source, success, service_name, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    source,
                    success,
                    service_name,
                    ip_address,
                    user_agent
                ))
                
            # Log evento di sicurezza per tentativi falliti
            if not success:
                self._log_security_event(
                    "ACCESS_DENIED", "MEDIUM", source,
                    f"Failed access attempt for {service_name or 'unknown service'}",
                    {"ip_address": ip_address, "user_agent": user_agent}
                )
                
        except Exception as e:
            self.logger.error(f"Error logging access attempt: {e}")
    
    def get_security_report(self, hours: int = 24) -> Dict:
        """Genera report di sicurezza"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.security_db) as conn:
                # Eventi di sicurezza per severità
                cursor = conn.execute("""
                    SELECT severity, COUNT(*) 
                    FROM security_events 
                    WHERE timestamp >= ?
                    GROUP BY severity
                """, (cutoff_time.isoformat(),))
                
                events_by_severity = dict(cursor.fetchall())
                
                # Tentativi di accesso
                cursor = conn.execute("""
                    SELECT success, COUNT(*) 
                    FROM access_attempts 
                    WHERE timestamp >= ?
                    GROUP BY success
                """, (cutoff_time.isoformat(),))
                
                access_attempts = dict(cursor.fetchall())
                
                # Credenziali più accessate
                cursor = conn.execute("""
                    SELECT service_name, access_count, last_accessed
                    FROM encrypted_credentials 
                    ORDER BY access_count DESC 
                    LIMIT 10
                """)
                
                top_accessed_credentials = [
                    {"service": row[0], "access_count": row[1], "last_accessed": row[2]}
                    for row in cursor.fetchall()
                ]
                
                # Sessioni attive
                cursor = conn.execute("""
                    SELECT COUNT(*) 
                    FROM secure_sessions 
                    WHERE expires_at > ?
                """, (datetime.now().isoformat(),))
                
                active_sessions = cursor.fetchone()[0]
                
                return {
                    "report_period_hours": hours,
                    "events_by_severity": events_by_severity,
                    "access_attempts": {
                        "successful": access_attempts.get(1, 0),
                        "failed": access_attempts.get(0, 0)
                    },
                    "top_accessed_credentials": top_accessed_credentials,
                    "active_sessions": active_sessions,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error generating security report: {e}")
            return {}
    
    def cleanup_expired_sessions(self):
        """Pulisce sessioni scadute"""
        
        try:
            now = datetime.now().isoformat()
            
            with sqlite3.connect(self.security_db) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM secure_sessions WHERE expires_at <= ?
                """, (now,))
                
                expired_count = cursor.fetchone()[0]
                
                if expired_count > 0:
                    conn.execute("DELETE FROM secure_sessions WHERE expires_at <= ?", (now,))
                    
                    self._log_security_event(
                        "SESSIONS_CLEANUP", "LOW", "SecuritySystem",
                        f"Cleaned up {expired_count} expired sessions"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up expired sessions: {e}")
    
    def rotate_master_key(self) -> bool:
        """Ruota master key (operazione critica)"""
        
        try:
            # Backup della master key attuale
            old_master_key = self.master_key
            
            # Genera nuova master key
            new_master_key = secrets.token_bytes(32)
            
            # Re-critta tutte le credenziali con la nuova chiave
            with sqlite3.connect(self.security_db) as conn:
                cursor = conn.execute("SELECT service_name FROM encrypted_credentials")
                services = [row[0] for row in cursor.fetchall()]
            
            # Decrittografa con vecchia chiave e ri-critta con nuova
            credentials_backup = {}
            for service_name in services:
                credentials = self.decrypt_api_credentials(service_name)
                if credentials:
                    credentials_backup[service_name] = credentials
            
            # Aggiorna master key
            self.master_key = new_master_key
            
            # Re-critta tutte le credenziali
            for service_name, credentials in credentials_backup.items():
                self.encrypt_api_credentials(service_name, credentials)
            
            # Salva nuova master key
            with open(self.master_key_file, 'wb') as f:
                f.write(new_master_key)
            
            self._log_security_event(
                "MASTER_KEY_ROTATED", "CRITICAL", "SecuritySystem",
                f"Master key rotated successfully. {len(credentials_backup)} credentials re-encrypted."
            )
            
            return True
            
        except Exception as e:
            # Ripristina master key precedente in caso di errore
            self.master_key = old_master_key
            
            self.logger.error(f"Error rotating master key: {e}")
            
            self._log_security_event(
                "MASTER_KEY_ROTATION_FAILED", "CRITICAL", "SecuritySystem",
                f"Master key rotation failed: {str(e)}"
            )
            
            return False
    
    def export_encrypted_backup(self, backup_password: str) -> str:
        """Esporta backup crittografato di tutte le credenziali"""
        
        try:
            # Raccoglie tutte le credenziali
            backup_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "credentials": {}
            }
            
            with sqlite3.connect(self.security_db) as conn:
                cursor = conn.execute("SELECT service_name FROM encrypted_credentials")
                services = [row[0] for row in cursor.fetchall()]
            
            for service_name in services:
                credentials = self.decrypt_api_credentials(service_name)
                if credentials:
                    backup_data["credentials"][service_name] = credentials
            
            # Serializza backup
            backup_json = json.dumps(backup_data)
            backup_bytes = backup_json.encode('utf-8')
            
            # Crittografa backup con password utente
            password_bytes = backup_password.encode('utf-8')
            salt = secrets.token_bytes(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            
            backup_key = kdf.derive(password_bytes)
            
            iv = secrets.token_bytes(16)
            cipher = Cipher(algorithms.AES(backup_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Padding
            padding_length = 16 - (len(backup_bytes) % 16)
            padded_data = backup_bytes + bytes([padding_length] * padding_length)
            
            encrypted_backup = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combina salt + IV + encrypted data
            final_backup = salt + iv + encrypted_backup
            backup_b64 = base64.b64encode(final_backup).decode('utf-8')
            
            self._log_security_event(
                "BACKUP_EXPORTED", "MEDIUM", "SecuritySystem",
                f"Encrypted backup exported with {len(backup_data['credentials'])} credentials"
            )
            
            return backup_b64
            
        except Exception as e:
            self.logger.error(f"Error exporting encrypted backup: {e}")
            return None

# Singleton instance
security_system = SecurityEncryptionSystem()

def get_security_system() -> SecurityEncryptionSystem:
    """Ottieni istanza singleton del sistema sicurezza"""
    return security_system

# Test del sistema
def test_security_system():
    """Test sistema sicurezza"""
    
    sec_system = get_security_system()
    
    # Test crittografia credenziali
    test_credentials = {
        "api_key": "test_api_key_12345",
        "secret_key": "test_secret_key_67890",
        "passphrase": "test_passphrase"
    }
    
    print("Testing credential encryption...")
    success = sec_system.encrypt_api_credentials("test_exchange", test_credentials)
    print(f"Encryption success: {success}")
    
    print("Testing credential decryption...")
    decrypted = sec_system.decrypt_api_credentials("test_exchange")
    print(f"Decryption success: {decrypted is not None}")
    print(f"Credentials match: {decrypted == test_credentials}")
    
    # Test sessioni sicure
    print("Testing secure sessions...")
    session_data = {"user_id": "test_user", "permissions": ["read", "write"]}
    session_id = sec_system.create_secure_session(session_data)
    print(f"Session created: {session_id}")
    
    retrieved_session = sec_system.get_secure_session(session_id)
    print(f"Session retrieved: {retrieved_session is not None}")
    
    # Test report sicurezza
    print("Generating security report...")
    report = sec_system.get_security_report()
    print(f"Security report: {report}")
    
    print("Security system test completed")

if __name__ == "__main__":
    test_security_system()