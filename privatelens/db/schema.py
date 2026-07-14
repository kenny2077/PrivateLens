"""SQLAlchemy models for PrivateLens SQLite database."""

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from privatelens.config import settings
from privatelens.utils.time import utcnow


logger = logging.getLogger(__name__)
VECTOR_DIMENSIONS = 512
VECTOR_BYTES = VECTOR_DIMENSIONS * 4
VECTOR_SCHEMA_VERSION = 2
SQLITE_BUSY_TIMEOUT_MS = 30_000

_engine: Engine | None = None
_engine_path: Path | None = None
_engine_lock = RLock()


class Base(DeclarativeBase):
    pass


class Asset(Base):
    """Core photo/video asset table."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    phash: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    media_type: Mapped[str] = mapped_column(
        String(20),
        default="unknown",
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exif_datetime: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exif_make: Mapped[str | None] = mapped_column(String, nullable=True)
    exif_model: Mapped[str | None] = mapped_column(String, nullable=True)
    gps_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    sensitive_type: Mapped[str | None] = mapped_column(String, nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String, nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    captions: Mapped[list["Caption"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )
    ocr_blocks: Mapped[list["OcrBlock"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )
    faces: Mapped[list["Face"]] = relationship(back_populates="asset", cascade="all, delete-orphan")
    detections: Mapped[list["Detection"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )
    sensitive_item: Mapped["SensitiveItem | None"] = relationship(
        back_populates="asset", uselist=False, cascade="all, delete-orphan"
    )
    embedding: Mapped["ImageEmbedding | None"] = relationship(
        back_populates="asset", uselist=False, cascade="all, delete-orphan"
    )


class ImageEmbedding(Base):
    """CLIP image embeddings stored as blobs."""

    __tablename__ = "image_embeddings"

    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String, default="openclip:ViT-B-32-quickgelu:openai")
    vector: Mapped[bytes] = mapped_column(nullable=False)  # serialized float32 array
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    asset: Mapped["Asset"] = relationship(back_populates="embedding")


class Caption(Base):
    """VLM-generated captions."""

    __tablename__ = "captions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model: Mapped[str] = mapped_column(String, nullable=False)
    caption: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_auto_generated: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    asset: Mapped["Asset"] = relationship(back_populates="captions")


class OcrBlock(Base):
    """OCR text blocks from images."""

    __tablename__ = "ocr_blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    bbox: Mapped[str | None] = mapped_column(String, nullable=True)  # JSON
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    page: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    asset: Mapped["Asset"] = relationship(back_populates="ocr_blocks")


class Face(Base):
    """Detected faces in photos."""

    __tablename__ = "faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bbox: Mapped[str] = mapped_column(String, nullable=False)  # JSON
    embedding: Mapped[bytes | None] = mapped_column(nullable=True)
    cluster_id: Mapped[int | None] = mapped_column(
        ForeignKey("people.id", ondelete="SET NULL"), nullable=True, index=True
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    asset: Mapped["Asset"] = relationship(back_populates="faces")
    person: Mapped["Person | None"] = relationship(back_populates="faces")


class Person(Base):
    """People clusters from face recognition."""

    __tablename__ = "people"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    user_labeled: Mapped[bool] = mapped_column(Boolean, default=False)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    thumbnail_asset_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    faces: Mapped[list["Face"]] = relationship(back_populates="person")


class Detection(Base):
    """Object/document detections."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String, nullable=False, index=True)
    bbox: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_model: Mapped[str | None] = mapped_column(String, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    asset: Mapped["Asset"] = relationship(back_populates="detections")


class SensitiveItem(Base):
    """Sensitive classifications with optional encrypted provenance metadata."""

    __tablename__ = "sensitive_items"

    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    type: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    encrypted_metadata: Mapped[bytes | None] = mapped_column(nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    asset: Mapped["Asset"] = relationship(back_populates="sensitive_item")


class SearchEvent(Base):
    """Search events for feedback loop."""

    __tablename__ = "search_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str | None] = mapped_column(String, nullable=True)
    results_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result_clicked: Mapped[int | None] = mapped_column(
        ForeignKey("assets.id", ondelete="SET NULL"), nullable=True
    )
    time_to_result_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)  # -1, 0, 1
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SearchRecipe(Base):
    """Built-in search recipes."""

    __tablename__ = "search_recipes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    query_plan: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


def load_sqlite_vec(dbapi_connection: Any) -> bool:
    """Load sqlite-vec into a DB-API connection when Python permits it."""
    if not hasattr(dbapi_connection, "load_extension") or not hasattr(
        dbapi_connection, "enable_load_extension"
    ):
        return False

    try:
        import sqlite_vec
    except ImportError:
        return False

    dbapi_connection.enable_load_extension(True)
    try:
        sqlite_vec.load(dbapi_connection)
    finally:
        dbapi_connection.enable_load_extension(False)
    return True


def _create_engine(db_path: Path) -> Engine:
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False, "timeout": SQLITE_BUSY_TIMEOUT_MS / 1000},
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite_connection(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        finally:
            cursor.close()
        try:
            load_sqlite_vec(dbapi_connection)
        except Exception as exc:
            logger.debug("sqlite-vec connection setup failed: %s", exc)

    return engine


def init_db() -> Engine:
    """Initialize SQLite database with all tables and FTS5 indexes."""
    from sqlalchemy import text

    db_path = settings.resolved_db_path.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode = WAL")
        conn.exec_driver_sql("PRAGMA synchronous = NORMAL")
        conn.commit()
    Base.metadata.create_all(engine)

    # Create FTS5 virtual tables for text search
    # Using trigram tokenizer for better multilingual support (CJK, etc.)
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
                text, content='ocr_blocks', content_rowid='id',
                tokenize='trigram'
            )
        """)
        )
        conn.execute(
            text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS captions_fts USING fts5(
                caption, content='captions', content_rowid='id',
                tokenize='trigram'
            )
        """)
        )
        conn.execute(
            text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS assets_fts USING fts5(
                path, content='assets', content_rowid='id',
                tokenize='trigram'
            )
        """)
        )
        conn.commit()

    # Create triggers to keep FTS5 indexes in sync
    with engine.connect() as conn:
        # ocr_blocks triggers
        conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS ocr_blocks_ai AFTER INSERT ON ocr_blocks BEGIN
                INSERT INTO ocr_fts(rowid, text) VALUES (new.id, new.text);
            END
        """)
        )
        conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS ocr_blocks_ad AFTER DELETE ON ocr_blocks BEGIN
                INSERT INTO ocr_fts(ocr_fts, rowid, text) VALUES ('delete', old.id, old.text);
            END
        """)
        )
        conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS ocr_blocks_au AFTER UPDATE ON ocr_blocks BEGIN
                INSERT INTO ocr_fts(ocr_fts, rowid, text) VALUES ('delete', old.id, old.text);
                INSERT INTO ocr_fts(rowid, text) VALUES (new.id, new.text);
            END
        """)
        )

        # captions triggers
        conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS captions_ai AFTER INSERT ON captions BEGIN
                INSERT INTO captions_fts(rowid, caption) VALUES (new.id, new.caption);
            END
        """)
        )
        conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS captions_ad AFTER DELETE ON captions BEGIN
                INSERT INTO captions_fts(captions_fts, rowid, caption) VALUES ('delete', old.id, old.caption);
            END
        """)
        )
        conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS captions_au AFTER UPDATE ON captions BEGIN
                INSERT INTO captions_fts(captions_fts, rowid, caption) VALUES ('delete', old.id, old.caption);
                INSERT INTO captions_fts(rowid, caption) VALUES (new.id, new.caption);
            END
        """)
        )

        # assets triggers
        conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS assets_ai AFTER INSERT ON assets BEGIN
                INSERT INTO assets_fts(rowid, path) VALUES (new.id, new.path);
            END
        """)
        )
        conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS assets_ad AFTER DELETE ON assets BEGIN
                INSERT INTO assets_fts(assets_fts, rowid, path) VALUES ('delete', old.id, old.path);
            END
        """)
        )
        conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS assets_au AFTER UPDATE ON assets BEGIN
                INSERT INTO assets_fts(assets_fts, rowid, path) VALUES ('delete', old.id, old.path);
                INSERT INTO assets_fts(rowid, path) VALUES (new.id, new.path);
            END
        """)
        )

        conn.commit()

    # Create and synchronize native sqlite-vec indexes when extension loading is supported.
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT vec_version()"))
            conn.execute(
                text(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_image_embeddings USING vec0(
                    asset_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{VECTOR_DIMENSIONS}] distance_metric=cosine
                )
            """)
            )
            conn.execute(
                text(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_faces USING vec0(
                    face_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{VECTOR_DIMENSIONS}] distance_metric=cosine
                )
            """)
            )
            schema_version = conn.exec_driver_sql("PRAGMA user_version").scalar_one()
            if schema_version < VECTOR_SCHEMA_VERSION:
                conn.execute(text("DROP TRIGGER IF EXISTS image_embeddings_vec_au"))
                conn.execute(text("DROP TRIGGER IF EXISTS faces_vec_au"))
            conn.execute(
                text(f"""
                CREATE TRIGGER IF NOT EXISTS image_embeddings_vec_ai
                AFTER INSERT ON image_embeddings
                WHEN length(new.vector) = {VECTOR_BYTES}
                BEGIN
                    INSERT OR REPLACE INTO vec_image_embeddings(asset_id, embedding)
                    VALUES (new.asset_id, new.vector);
                END
            """)
            )
            conn.execute(
                text(f"""
                CREATE TRIGGER IF NOT EXISTS image_embeddings_vec_au
                AFTER UPDATE OF vector ON image_embeddings
                BEGIN
                    DELETE FROM vec_image_embeddings WHERE asset_id = old.asset_id;
                    INSERT INTO vec_image_embeddings(asset_id, embedding)
                    SELECT new.asset_id, new.vector
                    WHERE length(new.vector) = {VECTOR_BYTES};
                END
            """)
            )
            conn.execute(
                text("""
                CREATE TRIGGER IF NOT EXISTS image_embeddings_vec_ad
                AFTER DELETE ON image_embeddings
                BEGIN
                    DELETE FROM vec_image_embeddings WHERE asset_id = old.asset_id;
                END
            """)
            )
            conn.execute(
                text(f"""
                CREATE TRIGGER IF NOT EXISTS faces_vec_ai
                AFTER INSERT ON faces
                WHEN new.embedding IS NOT NULL AND length(new.embedding) = {VECTOR_BYTES}
                BEGIN
                    INSERT OR REPLACE INTO vec_faces(face_id, embedding)
                    VALUES (new.id, new.embedding);
                END
            """)
            )
            conn.execute(
                text(f"""
                CREATE TRIGGER IF NOT EXISTS faces_vec_au
                AFTER UPDATE OF embedding ON faces
                BEGIN
                    DELETE FROM vec_faces WHERE face_id = old.id;
                    INSERT INTO vec_faces(face_id, embedding)
                    SELECT new.id, new.embedding
                    WHERE new.embedding IS NOT NULL AND length(new.embedding) = {VECTOR_BYTES};
                END
            """)
            )
            conn.execute(
                text("""
                CREATE TRIGGER IF NOT EXISTS faces_vec_ad
                AFTER DELETE ON faces
                BEGIN
                    DELETE FROM vec_faces WHERE face_id = old.id;
                END
            """)
            )
            if schema_version < VECTOR_SCHEMA_VERSION:
                conn.execute(text("DELETE FROM vec_image_embeddings"))
                conn.execute(text("DELETE FROM vec_faces"))
                conn.execute(
                    text(f"""
                    INSERT INTO vec_image_embeddings(asset_id, embedding)
                    SELECT asset_id, vector FROM image_embeddings
                    WHERE length(vector) = {VECTOR_BYTES}
                """)
                )
                conn.execute(
                    text(f"""
                    INSERT INTO vec_faces(face_id, embedding)
                    SELECT id, embedding FROM faces
                    WHERE embedding IS NOT NULL AND length(embedding) = {VECTOR_BYTES}
                """)
                )
                conn.exec_driver_sql(f"PRAGMA user_version = {VECTOR_SCHEMA_VERSION}")
            conn.commit()
    except Exception as exc:
        logger.info("sqlite-vec unavailable; using BLOB fallback: %s", exc)

    return engine


def get_engine() -> Engine:
    """Return the process engine, replacing it only when the DB path changes."""
    global _engine, _engine_path

    db_path = settings.resolved_db_path.expanduser().resolve()
    with _engine_lock:
        if _engine is None or _engine_path != db_path:
            if _engine is not None:
                _engine.dispose()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _engine = _create_engine(db_path)
            _engine_path = db_path
        return _engine


def reset_engine() -> None:
    """Dispose the current engine, primarily for controlled process teardown."""
    global _engine, _engine_path

    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _engine_path = None
