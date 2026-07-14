"""Apple Photos export importer."""

import json
import logging
import plistlib
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from privatelens.db.schema import get_engine, Asset
from privatelens.utils.image_formats import SUPPORTED_IMAGE_EXTENSIONS
from privatelens.utils.time import utcnow


logger = logging.getLogger(__name__)


class ApplePhotosImporter:
    """Import Apple Photos export folders into PrivateLens."""

    def __init__(self):
        self.engine = get_engine()

    def import_folder(self, export_path: Path) -> int:
        """Import an Apple Photos export folder.

        Apple Photos export typically contains:
        - Original photos in folders by date
        - Optional: metadata JSON files
        - Optional: .plist files with album info

        Args:
            export_path: Path to exported Apple Photos folder

        Returns:
            Number of photos imported
        """
        count = 0
        media_extensions = SUPPORTED_IMAGE_EXTENSIONS | {".mov", ".mp4"}

        with Session(self.engine) as session:
            for photo_path in export_path.rglob("*"):
                if photo_path.suffix.lower() not in media_extensions:
                    continue

                # Check if already exists
                existing = session.query(Asset).filter_by(path=str(photo_path)).first()
                if existing:
                    continue

                # Extract date from folder structure (e.g., 2024/03/15/IMG_1234.jpg)
                exif_datetime = self._parse_date_from_path(photo_path)

                # Try to find sidecar metadata
                metadata = self._read_sidecar(photo_path)

                asset = Asset(
                    path=str(photo_path),
                    sha256="",  # Will be computed during indexing
                    exif_datetime=exif_datetime or metadata.get("date"),
                    exif_make=metadata.get("make"),
                    exif_model=metadata.get("model"),
                    gps_lat=metadata.get("latitude"),
                    gps_lng=metadata.get("longitude"),
                    media_type="image",
                    last_seen_at=utcnow(),
                )
                session.add(asset)
                count += 1

                if count % 100 == 0:
                    session.commit()

            session.commit()

        logger.info("Imported %d photos from Apple Photos export", count)
        return count

    def _parse_date_from_path(self, path: Path) -> datetime | None:
        """Try to extract date from folder path structure."""
        # Look for patterns like 2024/03/15 or 2024-03-15
        parts = path.parts
        for i in range(len(parts) - 1):
            try:
                # Try YYYY/MM/DD pattern
                if i + 2 < len(parts):
                    year = int(parts[i])
                    month = int(parts[i + 1])
                    day = int(parts[i + 2])
                    if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                        return datetime(year, month, day)
            except ValueError:
                continue
        return None

    def _read_sidecar(self, photo_path: Path) -> dict[str, Any]:
        """Read sidecar metadata file if present."""
        # Check for .json sidecar (Google Takeout style, sometimes used)
        json_path = photo_path.with_suffix(photo_path.suffix + ".json")
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                return {
                    "date": datetime.fromtimestamp(
                        data.get("photoTakenTime", {}).get("timestamp", 0)
                    )
                    if data.get("photoTakenTime")
                    else None,
                    "make": data.get("photoTakenTime", {}).get("make"),
                    "model": data.get("photoTakenTime", {}).get("model"),
                    "latitude": data.get("geoData", {}).get("latitude"),
                    "longitude": data.get("geoData", {}).get("longitude"),
                }
            except Exception:
                pass

        # Check for .plist sidecar
        plist_path = photo_path.with_suffix(".plist")
        if plist_path.exists():
            try:
                with open(plist_path, "rb") as f:
                    data = plistlib.load(f)
                return {
                    "date": data.get("Created"),
                    "make": data.get("Make"),
                    "model": data.get("Model"),
                }
            except Exception:
                pass

        return {}
