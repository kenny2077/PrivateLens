"""Register image formats supported by PrivateLens."""

from functools import cache

from pillow_heif import register_heif_opener


SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".heif", ".tiff"}
)


@cache
def register_image_formats() -> None:
    """Register Pillow's HEIF/HEIC opener once per process."""
    register_heif_opener()
