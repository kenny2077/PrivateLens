"""Privacy guard and local-only mode enforcement."""

from pathlib import Path
from urllib.parse import urlparse

from privatelens.config import settings
from privatelens.utils.time import utcnow

LOCAL_SERVICE_HOSTS = {"localhost", "127.0.0.1", "::1", "ollama"}


def is_local_service_url(url: str) -> bool:
    """Return whether a URL targets an approved local runtime service."""
    return urlparse(url).hostname in LOCAL_SERVICE_HOSTS


class PrivacyGuard:
    """Enforce local-only mode and privacy settings."""

    def __init__(self):
        self.local_only = settings.local_only
        self._outbound_calls = []

    def verify_local_only(self) -> bool:
        """Check guarded service configuration without probing the network."""
        return not self.local_only or self.is_local_only()

    def log_outbound(self, url: str, purpose: str) -> None:
        """Log any outbound network call."""
        blocked = self.local_only and not self._is_local_url(url)
        self._outbound_calls.append(
            {
                "url": url,
                "purpose": purpose,
                "timestamp": utcnow(),
                "blocked": blocked,
            }
        )
        if blocked:
            raise PrivacyError(f"Blocked outbound {purpose} call in local-only mode: {url}")

    def get_outbound_log(self) -> list[dict]:
        """Get log of all outbound calls."""
        return self._outbound_calls

    def is_sensitive_operation(self, operation: str) -> bool:
        """Check if operation involves sensitive data."""
        sensitive_ops = ["face_embedding", "ocr", "vlm_caption", "index_upload"]
        return operation in sensitive_ops

    def require_local(self, operation: str) -> None:
        """Require that an operation runs locally only."""
        if self.local_only and not self.is_local_only():
            raise PrivacyError(f"Operation '{operation}' requires local-only mode")

    def is_local_only(self) -> bool:
        """Check if all processing is local."""
        return is_local_service_url(settings.ollama_url)

    def _is_local_url(self, url: str) -> bool:
        """Check if a URL targets the local machine."""
        return is_local_service_url(url)


class PrivacyError(Exception):
    """Raised when a privacy constraint is violated."""

    pass
