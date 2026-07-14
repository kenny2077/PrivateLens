"""EXIF metadata extraction."""

import hashlib
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import Base

from privatelens.utils.image_formats import register_image_formats


logger = logging.getLogger(__name__)

register_image_formats()


class ExifExtractor:
    """Extract EXIF metadata from images."""

    def extract(self, image_path: Path) -> dict[str, Any]:
        """Extract all metadata from an image file."""
        result: dict[str, Any] = {
            "sha256": "",
            "phash": None,
            "width": None,
            "height": None,
            "file_size": None,
            "modified_at": None,
            "datetime": None,
            "make": None,
            "model": None,
            "gps_lat": None,
            "gps_lng": None,
            "valid": False,
            "error": None,
        }

        try:
            stat = image_path.stat()
            result["sha256"] = self._sha256(image_path)
            result["file_size"] = stat.st_size
            result["modified_at"] = datetime.fromtimestamp(stat.st_mtime)
            with Image.open(image_path) as img:
                result["width"] = img.width
                result["height"] = img.height
                result["phash"] = self._phash(img)

                try:
                    exif = img.getexif()
                    if exif:
                        result["datetime"] = self._parse_datetime(exif)
                        result["make"] = self._get_tag(exif, "Make")
                        result["model"] = self._get_tag(exif, "Model")
                        lat, lng = self._extract_gps(exif)
                        result["gps_lat"] = lat
                        result["gps_lng"] = lng
                except (OSError, TypeError, ValueError) as exc:
                    logger.debug("Ignoring malformed EXIF in %s: %s", image_path, exc)
            result["valid"] = True
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Unreadable image %s: %s", image_path, exc)

        return result

    def _sha256(self, path: Path) -> str:
        """Compute SHA256 hash of file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _phash(self, img: Image.Image) -> str | None:
        """Compute perceptual hash."""
        try:
            import imagehash

            return str(imagehash.phash(img))
        except ImportError:
            return None

    def _get_tag(self, exif: Mapping[int, Any], name: str) -> str | None:
        """Get EXIF tag by name."""
        tag_id = getattr(Base, name, None)
        if tag_id and tag_id in exif:
            value = exif[tag_id]
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="ignore").strip("\x00")
                except Exception:
                    return None
            return str(value).strip() if value else None
        return None

    def _parse_datetime(self, exif: Mapping[int, Any]) -> datetime | None:
        """Parse DateTimeOriginal from EXIF."""
        dt_str = self._get_tag(exif, "DateTimeOriginal") or self._get_tag(exif, "DateTime")
        if dt_str:
            try:
                return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        return None

    def _extract_gps(self, exif: Mapping[int, Any]) -> tuple[float | None, float | None]:
        """Extract GPS coordinates from EXIF."""
        gps_info = exif.get(Base.GPSInfo)
        if not gps_info:
            return None, None

        def _convert(value):
            d, m, s = value
            return d + m / 60.0 + s / 3600.0

        try:
            lat = _convert(gps_info[2])
            if gps_info[1] == "S":
                lat = -lat
            lng = _convert(gps_info[4])
            if gps_info[3] == "W":
                lng = -lng
            return lat, lng
        except (KeyError, TypeError, ZeroDivisionError):
            return None, None
