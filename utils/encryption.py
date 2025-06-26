import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class Encryption:
    """Handles encryption and decryption of sensitive data"""
    
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher_suite = Fernet(self.key)
    
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create new one"""
        try:
            key_file = '.encryption_key'
            
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    return f.read()
            else:
                # Create new key
                password = os.getenv('ENCRYPTION_PASSWORD', 'default_crypto_ai_trader_key').encode()
                salt = os.urandom(16)
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password))
                
                # Save key
                with open(key_file, 'wb') as f:
                    f.write(key)
                
                return key
        except Exception as e:
            logger.error(f"Error getting/creating encryption key: {e}")
            # Fallback to simple key
            return Fernet.generate_key()
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return data  # Return original if encryption fails
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return encrypted_data  # Return original if decryption fails

# Global encryption instance
_encryption = Encryption()

def encrypt_api_key(api_key: str) -> str:
    """Encrypt API key"""
    return _encryption.encrypt(api_key)

def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt API key"""
    return _encryption.decrypt(encrypted_key)
