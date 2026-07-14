"""Encryption utilities for sensitive metadata."""

import json
from typing import Any

from cryptography.fernet import Fernet

from privatelens.config import settings


class MetadataEncryptor:
    """Encrypt and decrypt sensitive metadata."""

    def __init__(self, key: str | None = None):
        self._key = key or settings.encryption_key
        self._fernet = None
        if self._key:
            self._fernet = Fernet(self._key.encode() if isinstance(self._key, str) else self._key)

    def is_configured(self) -> bool:
        """Check if encryption is configured."""
        return self._fernet is not None

    def encrypt(self, data: dict[str, Any]) -> bytes | None:
        """Encrypt a dictionary to bytes."""
        if not self._fernet:
            return None
        try:
            plaintext = json.dumps(data).encode("utf-8")
            return self._fernet.encrypt(plaintext)
        except Exception:
            return None

    def decrypt(self, ciphertext: bytes) -> dict[str, Any] | None:
        """Decrypt bytes to a dictionary."""
        if not self._fernet:
            return None
        try:
            plaintext = self._fernet.decrypt(ciphertext)
            return json.loads(plaintext.decode("utf-8"))
        except Exception:
            return None

    def generate_key(self) -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode("utf-8")


def ensure_encryption_key() -> str:
    """Ensure an encryption key exists, generating one if needed."""
    if settings.encryption_key:
        return settings.encryption_key
    key = Fernet.generate_key().decode("utf-8")
    # Note: In production, this should be saved securely
    return key
