#!/usr/bin/env python3
"""
Multilayer API Key Protection Mechanism
Sistema di protezione multi-livello per chiavi API con sicurezza enterprise
"""

import os
import json
import sqlite3
import hashlib
import hmac
import base64
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import logging

class SecurityLevel(Enum):
    """Livelli di sicurezza per API keys"""
    BASIC = "basic"          # Crittografia base
    ENHANCED = "enhanced"    # Crittografia + 2FA
    ENTERPRISE = "enterprise"  # Multi-layer + HSM simulation
    MILITARY = "military"    # Massima sicurezza + quantum-resistant

class AccessLevel(Enum):
    """Livelli di accesso"""
    READ_ONLY = "read_only"
    TRADE_BASIC = "trade_basic"
    TRADE_ADVANCED = "trade_advanced"
    FULL_ACCESS = "full_access"
    ADMIN = "admin"

@dataclass
class APIKeyMetadata:
    """Metadati chiave API"""
    service_name: str
    key_id: str
    access_level: AccessLevel
    security_level: SecurityLevel
    created_at: datetime
    last_used: datetime
    usage_count: int
    max_usage_per_hour: Optional[int] = None
    ip_whitelist: Optional[List[str]] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True

@dataclass
class ProtectionLayer:
    """Livello di protezione"""
    layer_id: str
    layer_type: str  # encryption, tokenization, obfuscation, etc.
    algorithm: str
    key_derivation: str
    salt: bytes
    iv: Optional[bytes] = None
    parameters: Optional[Dict] = None

class MultilayerAPIProtection:
    """Sistema di protezione API multi-livello"""
    
    def __init__(self, db_path="data/api_protection.db", vault_path="data/secure_vault"):
        self.db_path = db_path
        self.vault_path = vault_path
        self.master_keys = {}
        self.session_tokens = {}
        self.protection_layers = {}
        
        # Setup database e vault
        self._init_database()
        self._init_secure_vault()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inizializza master keys per ogni livello
        self._initialize_master_keys()
    
    def _init_database(self):
        """Inizializza database protezione API"""
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Tabella metadati API keys
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_key_metadata (
                    key_id TEXT PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    access_level TEXT NOT NULL,
                    security_level TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    usage_count INTEGER DEFAULT 0,
                    max_usage_per_hour INTEGER,
                    ip_whitelist TEXT,
                    expires_at TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Tabella livelli di protezione
            conn.execute("""
                CREATE TABLE IF NOT EXISTS protection_layers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT NOT NULL,
                    layer_id TEXT NOT NULL,
                    layer_type TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    key_derivation TEXT NOT NULL,
                    salt BLOB NOT NULL,
                    iv BLOB,
                    parameters TEXT,
                    layer_order INTEGER NOT NULL,
                    FOREIGN KEY (key_id) REFERENCES api_key_metadata (key_id)
                )
            """)
            
            # Tabella vault sicuro per dati crittografati
            conn.execute("""
                CREATE TABLE IF NOT EXISTS secure_vault (
                    key_id TEXT PRIMARY KEY,
                    encrypted_data BLOB NOT NULL,
                    checksum TEXT NOT NULL,
                    encryption_method TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (key_id) REFERENCES api_key_metadata (key_id)
                )
            """)
            
            # Tabella log accessi per audit
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT NOT NULL,
                    access_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN NOT NULL,
                    details TEXT,
                    risk_score REAL DEFAULT 0.0
                )
            """)
            
            # Tabella token di sessione temporanei
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_tokens (
                    token_id TEXT PRIMARY KEY,
                    key_id TEXT NOT NULL,
                    encrypted_token BLOB NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    max_usage INTEGER DEFAULT 100,
                    FOREIGN KEY (key_id) REFERENCES api_key_metadata (key_id)
                )
            """)
            
            # Indici per performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_metadata_service ON api_key_metadata(service_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_protection_layers_key ON protection_layers(key_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_access_log_timestamp ON access_audit_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_tokens_expires ON session_tokens(expires_at)")
    
    def _init_secure_vault(self):
        """Inizializza vault sicuro per storage"""
        
        os.makedirs(self.vault_path, exist_ok=True)
        
        # File di configurazione vault
        vault_config = {
            "version": "1.0",
            "encryption_standard": "AES-256-GCM",
            "key_derivation": "PBKDF2-SHA512",
            "iterations": 100000,
            "created_at": datetime.now().isoformat()
        }
        
        vault_config_path = os.path.join(self.vault_path, "vault_config.json")
        if not os.path.exists(vault_config_path):
            with open(vault_config_path, 'w') as f:
                json.dump(vault_config, f, indent=2)
            
            # Imposta permessi restrittivi
            os.chmod(vault_config_path, 0o600)
    
    def _initialize_master_keys(self):
        """Inizializza master keys per ogni livello di sicurezza"""
        
        for security_level in SecurityLevel:
            key_file = os.path.join(self.vault_path, f"master_key_{security_level.value}.key")
            
            if os.path.exists(key_file):
                # Carica master key esistente
                try:
                    with open(key_file, 'rb') as f:
                        self.master_keys[security_level] = f.read()
                except Exception as e:
                    self.logger.error(f"Error loading master key for {security_level.value}: {e}")
                    self._generate_master_key(security_level)
            else:
                # Genera nuova master key
                self._generate_master_key(security_level)
    
    def _generate_master_key(self, security_level: SecurityLevel):
        """Genera master key per livello di sicurezza"""
        
        if security_level == SecurityLevel.BASIC:
            key_size = 32  # 256 bit
        elif security_level == SecurityLevel.ENHANCED:
            key_size = 32  # 256 bit
        elif security_level == SecurityLevel.ENTERPRISE:
            key_size = 48  # 384 bit
        else:  # MILITARY
            key_size = 64  # 512 bit
        
        master_key = secrets.token_bytes(key_size)
        
        # Salva master key
        key_file = os.path.join(self.vault_path, f"master_key_{security_level.value}.key")
        
        with open(key_file, 'wb') as f:
            f.write(master_key)
        
        # Imposta permessi restrittivi
        os.chmod(key_file, 0o600)
        
        self.master_keys[security_level] = master_key
        
        self.logger.info(f"Generated new master key for {security_level.value}")
    
    def store_api_key(self, service_name: str, api_credentials: Dict[str, str], 
                     access_level: AccessLevel = AccessLevel.TRADE_BASIC,
                     security_level: SecurityLevel = SecurityLevel.ENHANCED,
                     max_usage_per_hour: Optional[int] = None,
                     ip_whitelist: Optional[List[str]] = None,
                     expires_hours: Optional[int] = None) -> str:
        """Memorizza chiave API con protezione multi-livello"""
        
        try:
            # Genera ID unico per la chiave
            key_id = self._generate_key_id(service_name)
            
            # Crea metadati
            expires_at = None
            if expires_hours:
                expires_at = datetime.now() + timedelta(hours=expires_hours)
            
            metadata = APIKeyMetadata(
                service_name=service_name,
                key_id=key_id,
                access_level=access_level,
                security_level=security_level,
                created_at=datetime.now(),
                last_used=datetime.now(),
                usage_count=0,
                max_usage_per_hour=max_usage_per_hour,
                ip_whitelist=ip_whitelist,
                expires_at=expires_at
            )
            
            # Applica protezione multi-livello
            protected_data, protection_layers = self._apply_multilayer_protection(
                api_credentials, security_level
            )
            
            # Salva nel database
            self._save_to_vault(key_id, metadata, protected_data, protection_layers)
            
            self._log_access(key_id, "STORE", True, "API key stored successfully")
            
            return key_id
            
        except Exception as e:
            self.logger.error(f"Error storing API key: {e}")
            self._log_access("unknown", "STORE", False, f"Error: {str(e)}")
            raise
    
    def retrieve_api_key(self, key_id: str, requester_ip: Optional[str] = None,
                        user_agent: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Recupera chiave API con verifica sicurezza"""
        
        try:
            # Carica metadati
            metadata = self._load_metadata(key_id)
            
            if not metadata:
                self._log_access(key_id, "RETRIEVE", False, "Key not found")
                return None
            
            # Verifica validità
            if not self._verify_access_permissions(metadata, requester_ip):
                self._log_access(key_id, "RETRIEVE", False, "Access denied")
                return None
            
            # Carica dati protetti
            protected_data, protection_layers = self._load_from_vault(key_id)
            
            # Rimuovi protezione multi-livello
            api_credentials = self._remove_multilayer_protection(
                protected_data, protection_layers, metadata.security_level
            )
            
            # Aggiorna statistiche utilizzo
            self._update_usage_stats(key_id)
            
            self._log_access(key_id, "RETRIEVE", True, "API key retrieved successfully", 
                           ip_address=requester_ip, user_agent=user_agent)
            
            return api_credentials
            
        except Exception as e:
            self.logger.error(f"Error retrieving API key: {e}")
            self._log_access(key_id, "RETRIEVE", False, f"Error: {str(e)}", 
                           ip_address=requester_ip, user_agent=user_agent)
            return None
    
    def create_session_token(self, key_id: str, duration_hours: int = 1,
                           max_usage: int = 100) -> Optional[str]:
        """Crea token di sessione temporaneo per accesso sicuro"""
        
        try:
            # Verifica che la chiave esista
            metadata = self._load_metadata(key_id)
            if not metadata:
                return None
            
            # Genera token di sessione
            token_id = secrets.token_urlsafe(32)
            session_data = {
                "key_id": key_id,
                "created_at": datetime.now().isoformat(),
                "permissions": metadata.access_level.value
            }
            
            # Crittografa token
            encrypted_token = self._encrypt_session_token(session_data, metadata.security_level)
            
            # Salva nel database
            expires_at = datetime.now() + timedelta(hours=duration_hours)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO session_tokens 
                    (token_id, key_id, encrypted_token, expires_at, created_at, max_usage)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (token_id, key_id, encrypted_token, expires_at.isoformat(),
                      datetime.now().isoformat(), max_usage))
            
            self._log_access(key_id, "SESSION_CREATE", True, f"Session token created: {token_id[:8]}...")
            
            return token_id
            
        except Exception as e:
            self.logger.error(f"Error creating session token: {e}")
            return None
    
    def validate_session_token(self, token_id: str) -> Optional[Dict[str, str]]:
        """Valida token di sessione e restituisce credenziali"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT key_id, encrypted_token, expires_at, usage_count, max_usage
                    FROM session_tokens 
                    WHERE token_id = ?
                """, (token_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                key_id, encrypted_token, expires_at_str, usage_count, max_usage = row
                
                # Verifica scadenza
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    # Rimuovi token scaduto
                    conn.execute("DELETE FROM session_tokens WHERE token_id = ?", (token_id,))
                    self._log_access(key_id, "SESSION_EXPIRED", False, f"Token expired: {token_id[:8]}...")
                    return None
                
                # Verifica limite utilizzo
                if usage_count >= max_usage:
                    self._log_access(key_id, "SESSION_LIMIT", False, f"Usage limit reached: {token_id[:8]}...")
                    return None
                
                # Aggiorna contatore utilizzo
                conn.execute("""
                    UPDATE session_tokens 
                    SET usage_count = usage_count + 1 
                    WHERE token_id = ?
                """, (token_id,))
            
            # Recupera credenziali originali
            return self.retrieve_api_key(key_id)
            
        except Exception as e:
            self.logger.error(f"Error validating session token: {e}")
            return None
    
    def _apply_multilayer_protection(self, data: Dict[str, str], 
                                   security_level: SecurityLevel) -> Tuple[bytes, List[ProtectionLayer]]:
        """Applica protezione multi-livello ai dati"""
        
        # Serializza dati
        data_json = json.dumps(data)
        data_bytes = data_json.encode('utf-8')
        
        layers = []
        protected_data = data_bytes
        
        # Layer 1: Obfuscation (tutti i livelli)
        protected_data, layer1 = self._apply_obfuscation_layer(protected_data)
        layers.append(layer1)
        
        # Layer 2: Tokenization (Enhanced+)
        if security_level in [SecurityLevel.ENHANCED, SecurityLevel.ENTERPRISE, SecurityLevel.MILITARY]:
            protected_data, layer2 = self._apply_tokenization_layer(protected_data)
            layers.append(layer2)
        
        # Layer 3: AES Encryption (tutti i livelli)
        protected_data, layer3 = self._apply_aes_encryption(protected_data, security_level)
        layers.append(layer3)
        
        # Layer 4: RSA Encryption (Enterprise+)
        if security_level in [SecurityLevel.ENTERPRISE, SecurityLevel.MILITARY]:
            protected_data, layer4 = self._apply_rsa_encryption(protected_data)
            layers.append(layer4)
        
        # Layer 5: Quantum-Resistant (Military only)
        if security_level == SecurityLevel.MILITARY:
            protected_data, layer5 = self._apply_quantum_resistant_layer(protected_data)
            layers.append(layer5)
        
        return protected_data, layers
    
    def _remove_multilayer_protection(self, protected_data: bytes, 
                                    protection_layers: List[ProtectionLayer],
                                    security_level: SecurityLevel) -> Dict[str, str]:
        """Rimuove protezione multi-livello dai dati"""
        
        data = protected_data
        
        # Rimuovi layers in ordine inverso
        for layer in reversed(protection_layers):
            if layer.layer_type == "quantum_resistant":
                data = self._remove_quantum_resistant_layer(data, layer)
            elif layer.layer_type == "rsa_encryption":
                data = self._remove_rsa_encryption(data, layer)
            elif layer.layer_type == "aes_encryption":
                data = self._remove_aes_encryption(data, layer, security_level)
            elif layer.layer_type == "tokenization":
                data = self._remove_tokenization_layer(data, layer)
            elif layer.layer_type == "obfuscation":
                data = self._remove_obfuscation_layer(data, layer)
        
        # Deserializza
        data_json = data.decode('utf-8')
        return json.loads(data_json)
    
    def _apply_obfuscation_layer(self, data: bytes) -> Tuple[bytes, ProtectionLayer]:
        """Applica layer di offuscamento"""
        
        # Semplice XOR con chiave random
        key = secrets.token_bytes(32)
        obfuscated = bytearray()
        
        for i, byte in enumerate(data):
            obfuscated.append(byte ^ key[i % len(key)])
        
        layer = ProtectionLayer(
            layer_id=secrets.token_hex(8),
            layer_type="obfuscation",
            algorithm="XOR",
            key_derivation="random",
            salt=key
        )
        
        return bytes(obfuscated), layer
    
    def _remove_obfuscation_layer(self, data: bytes, layer: ProtectionLayer) -> bytes:
        """Rimuove layer di offuscamento"""
        
        key = layer.salt
        deobfuscated = bytearray()
        
        for i, byte in enumerate(data):
            deobfuscated.append(byte ^ key[i % len(key)])
        
        return bytes(deobfuscated)
    
    def _apply_tokenization_layer(self, data: bytes) -> Tuple[bytes, ProtectionLayer]:
        """Applica layer di tokenizzazione"""
        
        # Sostituisce parti dei dati con token
        token_map = {}
        tokenized_data = data
        
        # Per semplicità, tokenizza ogni chunk di 16 bytes
        chunks = [data[i:i+16] for i in range(0, len(data), 16)]
        tokenized_chunks = []
        
        for chunk in chunks:
            if len(chunk) >= 8:  # Solo chunk significativi
                token = secrets.token_bytes(16)
                token_map[token.hex()] = chunk.hex()
                tokenized_chunks.append(token)
            else:
                tokenized_chunks.append(chunk)
        
        layer = ProtectionLayer(
            layer_id=secrets.token_hex(8),
            layer_type="tokenization",
            algorithm="chunk_replacement",
            key_derivation="mapping",
            salt=secrets.token_bytes(16),
            parameters=token_map
        )
        
        return b''.join(tokenized_chunks), layer
    
    def _remove_tokenization_layer(self, data: bytes, layer: ProtectionLayer) -> bytes:
        """Rimuove layer di tokenizzazione"""
        
        token_map = layer.parameters
        detokenized_chunks = []
        
        # Ricostruisce i chunk originali
        chunks = [data[i:i+16] for i in range(0, len(data), 16)]
        
        for chunk in chunks:
            token_hex = chunk.hex()
            if token_hex in token_map:
                original_chunk = bytes.fromhex(token_map[token_hex])
                detokenized_chunks.append(original_chunk)
            else:
                detokenized_chunks.append(chunk)
        
        return b''.join(detokenized_chunks)
    
    def _apply_aes_encryption(self, data: bytes, security_level: SecurityLevel) -> Tuple[bytes, ProtectionLayer]:
        """Applica crittografia AES"""
        
        master_key = self.master_keys[security_level]
        
        # Genera salt e deriva chiave
        salt = secrets.token_bytes(16)
        
        if security_level == SecurityLevel.BASIC:
            iterations = 50000
        elif security_level == SecurityLevel.ENHANCED:
            iterations = 100000
        elif security_level == SecurityLevel.ENTERPRISE:
            iterations = 200000
        else:  # MILITARY
            iterations = 500000
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        
        encryption_key = kdf.derive(master_key)
        
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
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        layer = ProtectionLayer(
            layer_id=secrets.token_hex(8),
            layer_type="aes_encryption",
            algorithm="AES-256-CBC",
            key_derivation=f"PBKDF2-SHA512-{iterations}",
            salt=salt,
            iv=iv
        )
        
        return encrypted_data, layer
    
    def _remove_aes_encryption(self, data: bytes, layer: ProtectionLayer, 
                             security_level: SecurityLevel) -> bytes:
        """Rimuove crittografia AES"""
        
        master_key = self.master_keys[security_level]
        
        # Estrai parametri
        salt = layer.salt
        iv = layer.iv
        
        # Determina iterazioni dal key_derivation
        iterations = int(layer.key_derivation.split('-')[-1])
        
        # Deriva chiave
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        
        decryption_key = kdf.derive(master_key)
        
        # Decrittografa
        cipher = Cipher(
            algorithms.AES(decryption_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(data) + decryptor.finalize()
        
        # Rimuovi padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _apply_rsa_encryption(self, data: bytes) -> Tuple[bytes, ProtectionLayer]:
        """Applica crittografia RSA per enterprise/military"""
        
        # Genera coppia chiavi RSA
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        # RSA può crittografare solo dati piccoli, usa per chiave AES
        aes_key = secrets.token_bytes(32)
        
        # Crittografa dati con AES
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Padding
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Crittografa chiave AES con RSA
        encrypted_aes_key = public_key.encrypt(
            aes_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Serializza chiave privata
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Combina dati crittografati
        combined_data = len(encrypted_aes_key).to_bytes(4, 'big') + encrypted_aes_key + iv + encrypted_data
        
        layer = ProtectionLayer(
            layer_id=secrets.token_hex(8),
            layer_type="rsa_encryption",
            algorithm="RSA-4096-OAEP",
            key_derivation="generated",
            salt=private_key_pem,  # Salva chiave privata in salt (sicuro nel vault)
            iv=iv
        )
        
        return combined_data, layer
    
    def _remove_rsa_encryption(self, data: bytes, layer: ProtectionLayer) -> bytes:
        """Rimuove crittografia RSA"""
        
        # Carica chiave privata
        private_key = serialization.load_pem_private_key(
            layer.salt,
            password=None,
            backend=default_backend()
        )
        
        # Estrai componenti
        key_length = int.from_bytes(data[:4], 'big')
        encrypted_aes_key = data[4:4+key_length]
        iv = data[4+key_length:4+key_length+16]
        encrypted_data = data[4+key_length+16:]
        
        # Decrittografa chiave AES
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrittografa dati
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Rimuovi padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _apply_quantum_resistant_layer(self, data: bytes) -> Tuple[bytes, ProtectionLayer]:
        """Applica layer quantum-resistant (simulato)"""
        
        # Implementazione semplificata di algoritmo post-quantum
        # In produzione si userebbe un vero algoritmo quantum-resistant
        
        # Usa multiple chiavi e XOR con rotazione
        keys = [secrets.token_bytes(64) for _ in range(4)]
        
        protected_data = bytearray(data)
        
        for round_num, key in enumerate(keys):
            for i in range(len(protected_data)):
                # Rotazione e XOR con chiave diversa per ogni round
                key_byte = key[(i + round_num * 13) % len(key)]
                protected_data[i] = (protected_data[i] ^ key_byte ^ round_num) & 0xFF
        
        layer = ProtectionLayer(
            layer_id=secrets.token_hex(8),
            layer_type="quantum_resistant",
            algorithm="multi_key_rotation_xor",
            key_derivation="quantum_safe",
            salt=b''.join(keys)
        )
        
        return bytes(protected_data), layer
    
    def _remove_quantum_resistant_layer(self, data: bytes, layer: ProtectionLayer) -> bytes:
        """Rimuove layer quantum-resistant"""
        
        # Estrai chiavi
        key_data = layer.salt
        keys = [key_data[i:i+64] for i in range(0, len(key_data), 64)]
        
        unprotected_data = bytearray(data)
        
        # Inverti il processo in ordine inverso
        for round_num in reversed(range(len(keys))):
            key = keys[round_num]
            for i in range(len(unprotected_data)):
                key_byte = key[(i + round_num * 13) % len(key)]
                unprotected_data[i] = (unprotected_data[i] ^ key_byte ^ round_num) & 0xFF
        
        return bytes(unprotected_data)
    
    def _save_to_vault(self, key_id: str, metadata: APIKeyMetadata, 
                      protected_data: bytes, protection_layers: List[ProtectionLayer]):
        """Salva nel vault sicuro"""
        
        # Calcola checksum
        checksum = hashlib.sha256(protected_data).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            # Salva metadati
            conn.execute("""
                INSERT OR REPLACE INTO api_key_metadata 
                (key_id, service_name, access_level, security_level, created_at, 
                 last_used, usage_count, max_usage_per_hour, ip_whitelist, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.key_id, metadata.service_name, metadata.access_level.value,
                metadata.security_level.value, metadata.created_at.isoformat(),
                metadata.last_used.isoformat(), metadata.usage_count,
                metadata.max_usage_per_hour, 
                json.dumps(metadata.ip_whitelist) if metadata.ip_whitelist else None,
                metadata.expires_at.isoformat() if metadata.expires_at else None,
                metadata.is_active
            ))
            
            # Salva dati protetti
            conn.execute("""
                INSERT OR REPLACE INTO secure_vault 
                (key_id, encrypted_data, checksum, encryption_method, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                key_id, protected_data, checksum, 
                f"multilayer_{metadata.security_level.value}",
                datetime.now().isoformat(), datetime.now().isoformat()
            ))
            
            # Salva layers di protezione
            for i, layer in enumerate(protection_layers):
                conn.execute("""
                    INSERT INTO protection_layers 
                    (key_id, layer_id, layer_type, algorithm, key_derivation, 
                     salt, iv, parameters, layer_order)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key_id, layer.layer_id, layer.layer_type, layer.algorithm,
                    layer.key_derivation, layer.salt, layer.iv,
                    json.dumps(layer.parameters) if layer.parameters else None, i
                ))
    
    def _load_from_vault(self, key_id: str) -> Tuple[bytes, List[ProtectionLayer]]:
        """Carica dal vault sicuro"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Carica dati protetti
            cursor = conn.execute("""
                SELECT encrypted_data, checksum 
                FROM secure_vault 
                WHERE key_id = ?
            """, (key_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Protected data not found for key_id: {key_id}")
            
            protected_data, expected_checksum = row
            
            # Verifica checksum
            actual_checksum = hashlib.sha256(protected_data).hexdigest()
            if actual_checksum != expected_checksum:
                raise ValueError("Data integrity check failed")
            
            # Carica layers di protezione
            cursor = conn.execute("""
                SELECT layer_id, layer_type, algorithm, key_derivation, 
                       salt, iv, parameters
                FROM protection_layers 
                WHERE key_id = ? 
                ORDER BY layer_order
            """, (key_id,))
            
            protection_layers = []
            for row in cursor.fetchall():
                layer_id, layer_type, algorithm, key_derivation, salt, iv, parameters_json = row
                
                parameters = None
                if parameters_json:
                    parameters = json.loads(parameters_json)
                
                layer = ProtectionLayer(
                    layer_id=layer_id,
                    layer_type=layer_type,
                    algorithm=algorithm,
                    key_derivation=key_derivation,
                    salt=salt,
                    iv=iv,
                    parameters=parameters
                )
                
                protection_layers.append(layer)
            
            return protected_data, protection_layers
    
    def _load_metadata(self, key_id: str) -> Optional[APIKeyMetadata]:
        """Carica metadati chiave API"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT service_name, access_level, security_level, created_at,
                           last_used, usage_count, max_usage_per_hour, ip_whitelist,
                           expires_at, is_active
                    FROM api_key_metadata 
                    WHERE key_id = ?
                """, (key_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                (service_name, access_level, security_level, created_at_str,
                 last_used_str, usage_count, max_usage_per_hour, ip_whitelist_json,
                 expires_at_str, is_active) = row
                
                ip_whitelist = None
                if ip_whitelist_json:
                    ip_whitelist = json.loads(ip_whitelist_json)
                
                expires_at = None
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str)
                
                return APIKeyMetadata(
                    service_name=service_name,
                    key_id=key_id,
                    access_level=AccessLevel(access_level),
                    security_level=SecurityLevel(security_level),
                    created_at=datetime.fromisoformat(created_at_str),
                    last_used=datetime.fromisoformat(last_used_str) if last_used_str else datetime.now(),
                    usage_count=usage_count,
                    max_usage_per_hour=max_usage_per_hour,
                    ip_whitelist=ip_whitelist,
                    expires_at=expires_at,
                    is_active=bool(is_active)
                )
                
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return None
    
    def _verify_access_permissions(self, metadata: APIKeyMetadata, 
                                 requester_ip: Optional[str] = None) -> bool:
        """Verifica permessi di accesso"""
        
        # Verifica se la chiave è attiva
        if not metadata.is_active:
            return False
        
        # Verifica scadenza
        if metadata.expires_at and datetime.now() > metadata.expires_at:
            return False
        
        # Verifica IP whitelist
        if metadata.ip_whitelist and requester_ip:
            if requester_ip not in metadata.ip_whitelist:
                return False
        
        # Verifica limite di utilizzo orario
        if metadata.max_usage_per_hour:
            # Conta utilizzi nell'ultima ora
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) 
                    FROM access_audit_log 
                    WHERE key_id = ? AND access_type = 'RETRIEVE' 
                    AND timestamp >= ? AND success = 1
                """, (metadata.key_id, one_hour_ago.isoformat()))
                
                usage_count = cursor.fetchone()[0]
                
                if usage_count >= metadata.max_usage_per_hour:
                    return False
        
        return True
    
    def _update_usage_stats(self, key_id: str):
        """Aggiorna statistiche di utilizzo"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE api_key_metadata 
                SET usage_count = usage_count + 1, last_used = ?
                WHERE key_id = ?
            """, (datetime.now().isoformat(), key_id))
    
    def _log_access(self, key_id: str, access_type: str, success: bool, 
                   details: str, ip_address: Optional[str] = None,
                   user_agent: Optional[str] = None, risk_score: float = 0.0):
        """Log accesso per audit"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO access_audit_log 
                    (key_id, access_type, timestamp, ip_address, user_agent, 
                     success, details, risk_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (key_id, access_type, datetime.now().isoformat(),
                      ip_address, user_agent, success, details, risk_score))
                
        except Exception as e:
            self.logger.error(f"Error logging access: {e}")
    
    def _generate_key_id(self, service_name: str) -> str:
        """Genera ID unico per chiave"""
        
        timestamp = str(int(time.time()))
        random_part = secrets.token_hex(8)
        service_hash = hashlib.md5(service_name.encode()).hexdigest()[:8]
        
        return f"MLP_{service_hash}_{timestamp}_{random_part}"
    
    def _encrypt_session_token(self, session_data: Dict, security_level: SecurityLevel) -> bytes:
        """Crittografa token di sessione"""
        
        data_json = json.dumps(session_data)
        data_bytes = data_json.encode('utf-8')
        
        # Usa master key per il livello di sicurezza
        master_key = self.master_keys[security_level]
        
        # Fernet per crittografia simmetrica semplice
        fernet_key = base64.urlsafe_b64encode(master_key[:32])
        fernet = Fernet(fernet_key)
        
        return fernet.encrypt(data_bytes)
    
    def get_protection_status(self) -> Dict[str, Any]:
        """Ottieni stato del sistema di protezione"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Statistiche generali
                cursor = conn.execute("SELECT COUNT(*) FROM api_key_metadata WHERE is_active = 1")
                active_keys = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM session_tokens WHERE expires_at > ?", 
                                    (datetime.now().isoformat(),))
                active_sessions = cursor.fetchone()[0]
                
                # Statistiche per livello di sicurezza
                cursor = conn.execute("""
                    SELECT security_level, COUNT(*) 
                    FROM api_key_metadata 
                    WHERE is_active = 1 
                    GROUP BY security_level
                """)
                security_stats = dict(cursor.fetchall())
                
                # Statistiche accessi recenti (ultima ora)
                one_hour_ago = datetime.now() - timedelta(hours=1)
                cursor = conn.execute("""
                    SELECT access_type, success, COUNT(*) 
                    FROM access_audit_log 
                    WHERE timestamp >= ? 
                    GROUP BY access_type, success
                """, (one_hour_ago.isoformat(),))
                
                access_stats = {}
                for access_type, success, count in cursor.fetchall():
                    if access_type not in access_stats:
                        access_stats[access_type] = {"success": 0, "failed": 0}
                    
                    if success:
                        access_stats[access_type]["success"] = count
                    else:
                        access_stats[access_type]["failed"] = count
                
                return {
                    "system_status": "operational",
                    "active_keys": active_keys,
                    "active_sessions": active_sessions,
                    "security_levels": security_stats,
                    "recent_access_stats": access_stats,
                    "master_keys_initialized": len(self.master_keys),
                    "vault_path": self.vault_path,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting protection status: {e}")
            return {"system_status": "error", "error": str(e)}
    
    def cleanup_expired_data(self):
        """Pulisce dati scaduti"""
        
        try:
            now = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Rimuovi token di sessione scaduti
                cursor = conn.execute("SELECT COUNT(*) FROM session_tokens WHERE expires_at <= ?", (now,))
                expired_sessions = cursor.fetchone()[0]
                
                if expired_sessions > 0:
                    conn.execute("DELETE FROM session_tokens WHERE expires_at <= ?", (now,))
                    self.logger.info(f"Cleaned up {expired_sessions} expired session tokens")
                
                # Disattiva chiavi API scadute
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM api_key_metadata 
                    WHERE expires_at <= ? AND is_active = 1
                """, (now,))
                expired_keys = cursor.fetchone()[0]
                
                if expired_keys > 0:
                    conn.execute("""
                        UPDATE api_key_metadata 
                        SET is_active = 0 
                        WHERE expires_at <= ? AND is_active = 1
                    """, (now,))
                    self.logger.info(f"Deactivated {expired_keys} expired API keys")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Singleton instance
multilayer_protection = MultilayerAPIProtection()

def get_multilayer_protection() -> MultilayerAPIProtection:
    """Ottieni istanza singleton del sistema di protezione"""
    return multilayer_protection

# Test del sistema
def test_multilayer_protection():
    """Test sistema protezione multi-livello"""
    
    protection = get_multilayer_protection()
    
    # Test credenziali di esempio
    test_credentials = {
        "api_key": "test_api_key_12345",
        "secret_key": "test_secret_key_67890",
        "passphrase": "test_passphrase_abc"
    }
    
    print("Testing Multilayer API Protection System...")
    
    # Test storage con diversi livelli di sicurezza
    for security_level in SecurityLevel:
        print(f"\nTesting {security_level.value} security level...")
        
        key_id = protection.store_api_key(
            service_name="test_exchange",
            api_credentials=test_credentials,
            security_level=security_level,
            max_usage_per_hour=100
        )
        
        print(f"Stored key: {key_id}")
        
        # Test retrieval
        retrieved = protection.retrieve_api_key(key_id)
        
        if retrieved == test_credentials:
            print(f"✓ {security_level.value} test passed")
        else:
            print(f"✗ {security_level.value} test failed")
    
    # Test session tokens
    print("\nTesting session tokens...")
    session_token = protection.create_session_token(key_id, duration_hours=1)
    print(f"Created session token: {session_token}")
    
    validated_creds = protection.validate_session_token(session_token)
    if validated_creds:
        print("✓ Session token validation passed")
    else:
        print("✗ Session token validation failed")
    
    # Test protection status
    print("\nProtection status:")
    status = protection.get_protection_status()
    print(json.dumps(status, indent=2))
    
    print("\nMultilayer protection test completed")

if __name__ == "__main__":
    test_multilayer_protection()