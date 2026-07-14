"""Google Takeout importer."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from privatelens.db.schema import get_engine, Asset
from privatelens.utils.time import utcnow


logger = logging.getLogger(__name__)


class GoogleTakeoutImporter:
    """Import Google Takeout photo archives."""

    def __init__(self):
        self.engine = get_engine()

    def import_folder(self, takeout_path: Path) -> int:
        """Import Google Takeout folder.

        Google Takeout structure:
        - Photos/ folder with images
        - Each image has a .json sidecar with metadata

        Args:
            takeout_path: Path to extracted Takeout folder

        Returns:
            Number of photos imported
        """
        count = 0
        photos_dir = takeout_path / "Takeout" / "Google Photos"

        if not photos_dir.exists():
            # Try alternative structure
            photos_dir = takeout_path

        with Session(self.engine) as session:
            for album_dir in photos_dir.iterdir():
                if not album_dir.is_dir():
                    continue

                for json_path in album_dir.glob("*.json"):
                    # Skip metadata.json files
                    if json_path.name == "metadata.json":
                        continue

                    # Find corresponding photo
                    photo_name = json_path.stem
                    photo_path = None
                    for ext in [".jpg", ".jpeg", ".png", ".mp4", ".gif", ".webp"]:
                        candidate = album_dir / (photo_name + ext)
                        if candidate.exists():
                            photo_path = candidate
                            break

                    if not photo_path:
                        continue

                    # Check if already exists
                    existing = session.query(Asset).filter_by(path=str(photo_path)).first()
                    if existing:
                        continue

                    # Read metadata
                    metadata = self._read_json(json_path)

                    asset = Asset(
                        path=str(photo_path),
                        sha256="",
                        exif_datetime=metadata.get("date"),
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

        logger.info("Imported %d photos from Google Takeout", count)
        return count

    def _read_json(self, json_path: Path) -> dict[str, Any]:
        """Read Google Takeout JSON metadata."""
        try:
            with open(json_path) as f:
                data = json.load(f)

            timestamp = data.get("photoTakenTime", {}).get("timestamp")
            date = datetime.fromtimestamp(int(timestamp)) if timestamp else None

            geo = data.get("geoData", {})

            return {
                "date": date,
                "make": data.get("photoTakenTime", {}).get("make"),
                "model": data.get("photoTakenTime", {}).get("model"),
                "latitude": geo.get("latitude") if geo.get("latitude") != 0.0 else None,
                "longitude": geo.get("longitude") if geo.get("longitude") != 0.0 else None,
            }
        except Exception:
            return {}
