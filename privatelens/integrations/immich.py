"""Immich sidecar connector - reads Immich DB and builds enhanced index."""

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from privatelens.config import settings
from privatelens.db.schema import get_engine, Asset, ImageEmbedding, OcrBlock, Caption, Face, Person
from privatelens.utils.time import utcnow


logger = logging.getLogger(__name__)


class ImmichConnector:
    """Read Immich PostgreSQL database and sync to PrivateLens index."""

    def __init__(self, immich_db_url: str | None = None):
        """Initialize with Immich database URL.

        Args:
            immich_db_url: PostgreSQL connection string, e.g.
                postgresql://immich:password@localhost:5432/immich
        """
        self.immich_db_url = immich_db_url or settings.immich_db_url
        self.immich_engine: Engine | None = None
        self.privatelens_engine = get_engine()

    def _connect(self) -> bool:
        """Connect to Immich database."""
        if not self.immich_db_url:
            logger.warning("No Immich database URL configured")
            return False

        try:
            engine = create_engine(self.immich_db_url)
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.immich_engine = engine
            return True
        except Exception as e:
            logger.warning("Failed to connect to Immich: %s", e)
            return False

    def sync_assets(self) -> int:
        """Sync Immich assets to PrivateLens.

        Returns:
            Number of assets synced
        """
        if not self._connect():
            return 0
        immich_engine = self.immich_engine
        if immich_engine is None:
            return 0

        count = 0
        with Session(immich_engine) as immich_session:
            with Session(self.privatelens_engine) as pl_session:
                # Query Immich assets
                result = immich_session.execute(
                    text("""
                    SELECT 
                        a.id, a."originalPath", a."fileCreatedAt", 
                        a."fileModifiedAt", a."originalFileName",
                        a.checksum, a.width, a.height, a.type,
                        e."dateTimeOriginal", e."make", e.model,
                        e."latitude", e."longitude", e."city", e."country"
                    FROM assets a
                    LEFT JOIN exif e ON a.id = e."assetId"
                    WHERE a."deletedAt" IS NULL
                    LIMIT 10000
                """)
                )

                for row in result:
                    # Check if already exists
                    existing = pl_session.query(Asset).filter_by(path=row.originalPath).first()

                    if existing:
                        existing.last_seen_at = utcnow()
                        continue

                    asset = Asset(
                        path=row.originalPath,
                        sha256=row.checksum.hex() if row.checksum else "",
                        width=row.width,
                        height=row.height,
                        media_type=row.type.lower() if row.type else "unknown",
                        modified_at=row.fileModifiedAt,
                        exif_datetime=row.dateTimeOriginal,
                        exif_make=row.make,
                        exif_model=row.model,
                        gps_lat=row.latitude,
                        gps_lng=row.longitude,
                        last_seen_at=utcnow(),
                    )
                    pl_session.add(asset)
                    count += 1

                    if count % 100 == 0:
                        pl_session.commit()

                pl_session.commit()

        logger.info("Synced %d assets from Immich", count)
        return count

    def sync_faces(self) -> int:
        """Sync Immich face data to PrivateLens.

        Returns:
            Number of faces synced
        """
        if not self._connect():
            return 0
        immich_engine = self.immich_engine
        if immich_engine is None:
            return 0

        count = 0
        with Session(immich_engine) as immich_session:
            with Session(self.privatelens_engine) as pl_session:
                # Query Immich people and faces
                result = immich_session.execute(
                    text("""
                    SELECT 
                        p.id, p.name, p."birthDate",
                        af.id as face_id, af."assetId",
                        af."boundingBoxX1", af."boundingBoxY1",
                        af."boundingBoxX2", af."boundingBoxY2",
                        fs.embedding
                    FROM person p
                    JOIN asset_faces af ON p.id = af."personId"
                    LEFT JOIN face_search fs ON af.id = fs."faceId"
                    WHERE p."deletedAt" IS NULL
                    LIMIT 10000
                """)
                )

                for row in result:
                    # Find or create person
                    person = pl_session.query(Person).filter_by(display_name=row.name).first()

                    if not person:
                        person = Person(
                            display_name=row.name,
                            user_labeled=True,
                        )
                        pl_session.add(person)
                        pl_session.flush()

                    # Find asset
                    asset = (
                        pl_session.query(Asset)
                        .filter_by(
                            path=row.assetId  # This would need mapping
                        )
                        .first()
                    )

                    if asset:
                        face = Face(
                            asset_id=asset.id,
                            bbox=json.dumps(
                                {
                                    "x1": row.boundingBoxX1,
                                    "y1": row.boundingBoxY1,
                                    "x2": row.boundingBoxX2,
                                    "y2": row.boundingBoxY2,
                                }
                            ),
                            embedding=row.embedding.tobytes() if row.embedding else None,
                            cluster_id=person.id,
                            confidence=0.9,
                        )
                        pl_session.add(face)
                        count += 1

                        if count % 100 == 0:
                            pl_session.commit()

                pl_session.commit()

        logger.info("Synced %d faces from Immich", count)
        return count

    def full_sync(self) -> dict:
        """Perform full sync from Immich to PrivateLens.

        Returns:
            Dict with sync statistics
        """
        return {
            "assets": self.sync_assets(),
            "faces": self.sync_faces(),
        }
