"""File system utilities."""

import os
from pathlib import Path

from privatelens.utils.image_formats import SUPPORTED_IMAGE_EXTENSIONS


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(folder: Path, recursive: bool = True) -> list[Path]:
    """Get all image files in a folder.

    Args:
        folder: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of image file paths
    """
    if recursive:
        return [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    else:
        return [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]


def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    return path.stat().st_size


def get_modified_time(path: Path) -> float:
    """Get file modification time as timestamp."""
    return path.stat().st_mtime
