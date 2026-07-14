"""Search queries and database access patterns."""

from dataclasses import dataclass
from datetime import datetime
import re

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from privatelens.db.schema import (
    get_engine,
    Asset,
    OcrBlock,
    Caption,
    Face,
    Person,
    Detection,
    ImageEmbedding,
    SearchEvent,
)


@dataclass(frozen=True)
class AssetRecord:
    id: int
    path: str
    thumbnail_path: str | None
    media_type: str
    is_sensitive: bool
    created_at: datetime
    exif_datetime: datetime | None


@dataclass(frozen=True)
class FaceRecord:
    id: int
    asset_id: int
    cluster_id: int | None
    confidence: float | None


@dataclass(frozen=True)
class OcrBlockRecord:
    id: int
    asset_id: int
    text: str
    confidence: float | None


@dataclass(frozen=True)
class CaptionRecord:
    id: int
    asset_id: int
    caption: str
    confidence: float | None


@dataclass(frozen=True)
class DetectionRecord:
    id: int
    asset_id: int
    label: str
    confidence: float | None
    source_model: str | None


@dataclass(frozen=True)
class PersonRecord:
    id: int
    display_name: str | None


class SearchQueries:
    """Database queries for the search engine."""

    def __init__(self):
        self.engine = get_engine()

    def asset_count(self) -> int:
        """Count assets in the index."""
        with Session(self.engine) as session:
            return session.query(Asset).count()

    def get_asset(self, asset_id: int) -> AssetRecord | None:
        """Get asset by ID."""
        with Session(self.engine) as session:
            asset = session.query(Asset).filter_by(id=asset_id).first()
            return self._asset_record(asset) if asset else None

    def get_assets_by_ids(self, asset_ids: list[int]) -> dict[int, AssetRecord]:
        """Get multiple assets by ID."""
        with Session(self.engine) as session:
            assets = session.query(Asset).filter(Asset.id.in_(asset_ids)).all()
            return {a.id: self._asset_record(a) for a in assets}

    def _escape_fts5(self, query: str) -> str:
        """Escape special characters for SQLite FTS5 MATCH.

        Each whitespace-delimited term is quoted and joined with AND. This keeps
        FTS operators inert while tolerating OCR engines that drop spaces between
        adjacent words.
        """
        terms = [term.replace('"', "") for term in query.split()]
        quoted_terms = [f'"{term}"' for term in terms if term]
        return " AND ".join(quoted_terms) or '""'

    def ocr_fts_search(self, query: str, limit: int = 200) -> list[tuple[int, str, float]]:
        """Search OCR text using FTS5.

        Returns:
            List of (asset_id, text, rank) tuples
        """
        safe_query = self._escape_fts5(query)
        with Session(self.engine) as session:
            results = session.execute(
                text("""
                    SELECT o.asset_id, o.text, rank
                    FROM ocr_fts f
                    JOIN ocr_blocks o ON f.rowid = o.id
                    WHERE ocr_fts MATCH :query
                    ORDER BY rank
                    LIMIT :limit
                """),
                {"query": safe_query, "limit": limit},
            ).fetchall()
            return [(r.asset_id, r.text, r.rank) for r in results]

    def caption_fts_search(self, query: str, limit: int = 200) -> list[tuple[int, str, float]]:
        """Search captions using FTS5."""
        safe_query = self._escape_fts5(query)
        with Session(self.engine) as session:
            results = session.execute(
                text("""
                    SELECT c.asset_id, c.caption, rank
                    FROM captions_fts f
                    JOIN captions c ON f.rowid = c.id
                    WHERE captions_fts MATCH :query
                    ORDER BY rank
                    LIMIT :limit
                """),
                {"query": safe_query, "limit": limit},
            ).fetchall()
            return [(r.asset_id, r.caption, r.rank) for r in results]

    def path_fts_search(self, query: str, limit: int = 200) -> list[tuple[int, str, float]]:
        """Search asset paths using FTS5."""
        safe_query = self._escape_fts5(query)
        with Session(self.engine) as session:
            results = session.execute(
                text("""
                    SELECT a.id, a.path, rank
                    FROM assets_fts f
                    JOIN assets a ON f.rowid = a.id
                    WHERE assets_fts MATCH :query
                    ORDER BY rank
                    LIMIT :limit
                """),
                {"query": safe_query, "limit": limit},
            ).fetchall()
            return [(r.id, r.path, r.rank) for r in results]

    def get_embeddings_for_assets(self, asset_ids: list[int]) -> dict[int, bytes]:
        """Get CLIP embeddings for specific assets."""
        with Session(self.engine) as session:
            embeddings = (
                session.query(ImageEmbedding).filter(ImageEmbedding.asset_id.in_(asset_ids)).all()
            )
            return {e.asset_id: e.vector for e in embeddings}

    def get_faces_for_asset(self, asset_id: int) -> list[FaceRecord]:
        """Get faces for an asset."""
        with Session(self.engine) as session:
            faces = session.query(Face).filter_by(asset_id=asset_id).all()
            return [self._face_record(face) for face in faces]

    def get_ocr_for_asset(self, asset_id: int) -> list[OcrBlockRecord]:
        """Get OCR blocks for an asset."""
        with Session(self.engine) as session:
            blocks = session.query(OcrBlock).filter_by(asset_id=asset_id).all()
            return [
                OcrBlockRecord(
                    id=block.id,
                    asset_id=block.asset_id,
                    text=block.text,
                    confidence=block.confidence,
                )
                for block in blocks
            ]

    def get_captions_for_asset(self, asset_id: int) -> list[CaptionRecord]:
        """Get captions for an asset."""
        with Session(self.engine) as session:
            captions = session.query(Caption).filter_by(asset_id=asset_id).all()
            return [
                CaptionRecord(
                    id=caption.id,
                    asset_id=caption.asset_id,
                    caption=caption.caption,
                    confidence=caption.confidence,
                )
                for caption in captions
            ]

    def get_detections_for_asset(self, asset_id: int) -> list[DetectionRecord]:
        """Get detections for an asset."""
        with Session(self.engine) as session:
            detections = session.query(Detection).filter_by(asset_id=asset_id).all()
            return [
                DetectionRecord(
                    id=detection.id,
                    asset_id=detection.asset_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    source_model=detection.source_model,
                )
                for detection in detections
            ]

    def detection_label_search(
        self,
        labels: list[str],
        query: str | None = None,
        limit: int = 200,
    ) -> list[tuple[int, str, float | None]]:
        """Search detections by label.

        Returns:
            List of (asset_id, label, confidence) tuples
        """
        raw_terms = labels or ([query] if query else [])
        terms = list(
            dict.fromkeys(
                normalized
                for term in raw_terms
                if (normalized := re.sub(r"[^a-z0-9]+", "_", term.casefold()).strip("_"))
            )
        )
        if not terms:
            return []

        with Session(self.engine) as session:
            results = (
                session.query(Detection.asset_id, Detection.label, Detection.confidence)
                .filter(func.lower(Detection.label).in_(terms))
                .order_by(Detection.confidence.is_(None), Detection.confidence.desc())
                .limit(limit)
                .all()
            )
            return [(r.asset_id, r.label, r.confidence) for r in results]

    def get_person_by_name(self, name: str) -> PersonRecord | None:
        """Find person by display name (case-insensitive)."""
        with Session(self.engine) as session:
            person = session.query(Person).filter(Person.display_name.ilike(f"%{name}%")).first()
            return PersonRecord(id=person.id, display_name=person.display_name) if person else None

    def get_faces_by_person(self, person_id: int, limit: int = 200) -> list[FaceRecord]:
        """Get faces belonging to a person."""
        with Session(self.engine) as session:
            faces = session.query(Face).filter_by(cluster_id=person_id).limit(limit).all()
            return [self._face_record(face) for face in faces]

    def face_count_search(
        self,
        face_count: int,
        limit: int = 200,
    ) -> list[tuple[int, int]]:
        """Find assets with exactly the requested number of detected faces."""
        with Session(self.engine) as session:
            results = (
                session.query(Face.asset_id, func.count(Face.id).label("face_count"))
                .group_by(Face.asset_id)
                .having(func.count(Face.id) == face_count)
                .limit(limit)
                .all()
            )
            return [(r.asset_id, r.face_count) for r in results]

    def asset_face_count(self, asset_id: int) -> int:
        """Count detected faces for one asset."""
        with Session(self.engine) as session:
            return session.query(Face).filter_by(asset_id=asset_id).count()

    def has_face_data(self) -> bool:
        """Return whether optional face extraction has produced any rows."""
        with Session(self.engine) as session:
            return session.query(Face.id).first() is not None

    def positive_feedback_counts(self, asset_ids: list[int]) -> dict[int, int]:
        """Count explicit positive clicks for the supplied candidate assets."""
        if not asset_ids:
            return {}

        with Session(self.engine) as session:
            rows = (
                session.query(
                    SearchEvent.result_clicked,
                    func.count(SearchEvent.id).label("positive_count"),
                )
                .filter(
                    SearchEvent.result_clicked.in_(asset_ids),
                    SearchEvent.feedback > 0,
                )
                .group_by(SearchEvent.result_clicked)
                .all()
            )
            return {int(row.result_clicked): int(row.positive_count) for row in rows}

    def _asset_record(self, asset: Asset) -> AssetRecord:
        return AssetRecord(
            id=asset.id,
            path=asset.path,
            thumbnail_path=asset.thumbnail_path,
            media_type=asset.media_type,
            is_sensitive=asset.is_sensitive,
            created_at=asset.created_at,
            exif_datetime=asset.exif_datetime,
        )

    def _face_record(self, face: Face) -> FaceRecord:
        return FaceRecord(
            id=face.id,
            asset_id=face.asset_id,
            cluster_id=face.cluster_id,
            confidence=face.confidence,
        )
