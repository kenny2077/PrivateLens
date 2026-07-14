"""Time helpers for the existing naive-UTC SQLite schema."""

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return current UTC without tzinfo for DateTime(timezone=False) columns."""
    return datetime.now(UTC).replace(tzinfo=None)
