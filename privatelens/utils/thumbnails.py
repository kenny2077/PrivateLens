"""Utility functions for PrivateLens."""

import hashlib
from pathlib import Path
from PIL import Image

from privatelens.utils.image_formats import register_image_formats


register_image_formats()


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_thumbnail(image_path: Path, asset_id: int, size: int = 256) -> Path:
    """Generate a thumbnail for an image."""
    from privatelens.config import settings

    thumb_dir = settings.resolved_thumbnail_dir
    thumb_dir.mkdir(parents=True, exist_ok=True)

    thumb_path = thumb_dir / f"{asset_id}.jpg"

    if thumb_path.exists():
        return thumb_path

    try:
        with Image.open(image_path) as source_image:
            source_image.thumbnail((size, size), Image.Resampling.LANCZOS)
            thumbnail = source_image.convert("RGB")
            thumbnail.save(thumb_path, "JPEG", quality=85)
    except Exception:
        # Create a placeholder
        placeholder = Image.new("RGB", (size, size), color=(50, 50, 50))
        placeholder.save(thumb_path, "JPEG")

    return thumb_path
