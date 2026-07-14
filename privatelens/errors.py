"""Project-level exceptions for PrivateLens."""


class PrivateLensError(Exception):
    """Base class for expected PrivateLens failures."""


class ExtractionError(PrivateLensError):
    """Raised when media feature extraction fails."""


class SearchError(PrivateLensError):
    """Raised when search cannot complete."""


class StorageError(PrivateLensError):
    """Raised when local index storage fails."""
