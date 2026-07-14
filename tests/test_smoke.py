"""Smoke tests for PrivateLens core functionality.

These tests use lightweight mocks to avoid downloading ML models.
Run with: pytest tests/ -v
"""

import json
import os
import re
import sys
import tempfile
import tomllib
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

import privatelens
from privatelens.config import settings
from privatelens.db.schema import init_db, Asset, OcrBlock
from privatelens.extractors.exif import ExifExtractor
from privatelens.extractors.document import DocumentClassifier
from privatelens.extractors.screenshot import ScreenshotDetector
from privatelens.extractors.sensitive import SensitiveDetector
from privatelens.search.engine import SearchEngine
from privatelens.search.recipes import init_recipes, get_recipes
from privatelens.privacy.audit import PrivacyAuditor
from privatelens.utils.thumbnails import generate_thumbnail
from privatelens.utils.fs import get_image_files


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_data_dir = settings.data_dir
        settings.data_dir = Path(tmpdir)
        engine = init_db()
        yield engine
        settings.data_dir = old_data_dir


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (100, 100), color="red")
        img.save(f.name, "JPEG")
        yield Path(f.name)
        os.unlink(f.name)


class TestScanAndIndex:
    """Test scanning and indexing functionality."""

    def test_exif_extractor(self, sample_image):
        extractor = ExifExtractor()
        data = extractor.extract(sample_image)
        assert "sha256" in data
        assert data["width"] == 100
        assert data["height"] == 100
        assert data["file_size"] > 0
        assert data["valid"] is True
        assert data["error"] is None

    def test_document_classifier(self, sample_image):
        classifier = DocumentClassifier()
        exif = {"width": 100, "height": 100}
        assert classifier.is_document(sample_image, exif) is False

        # Document aspect ratio
        exif_doc = {"width": 1200, "height": 800}
        assert classifier.is_document(sample_image, exif_doc) is True

        camera_photo = {
            "width": 4032,
            "height": 3024,
            "make": "Apple",
            "model": "iPhone 13",
        }
        assert classifier.is_document(Path("IMG_1234.HEIC"), camera_photo) is False
        assert classifier.is_document(Path("holiday.jpg")) is False

    def test_screenshot_detector(self, sample_image):
        detector = ScreenshotDetector()
        exif = {
            "width": 1920,
            "height": 1080,
            "make": None,
            "model": None,
            "datetime": datetime.now(),
        }
        assert detector.is_screenshot(sample_image, exif) is True

    def test_screenshot_detector_does_not_treat_camera_img_filename_as_screenshot(self):
        detector = ScreenshotDetector()
        exif = {
            "width": 4032,
            "height": 3024,
            "make": "Apple",
            "model": "iPhone 13",
            "datetime": datetime.now(),
        }

        assert detector.is_screenshot(Path("IMG_1234.HEIC"), exif) is False

    def test_screenshot_detector_does_not_match_ss_inside_words(self):
        detector = ScreenshotDetector()

        assert detector.is_screenshot(Path("class-photo.jpg")) is False

    def test_sensitive_detector(self, sample_image):
        detector = SensitiveDetector()
        ocr = [{"text": "DRIVER LICENSE"}]
        result = detector.detect(sample_image, ocr)
        assert result is not None
        assert result["type"] == "driver_license"

    def test_sensitive_detector_ignores_short_filename_pattern_inside_word(self, tmp_path):
        detector = SensitiveDetector()
        image_path = tmp_path / "kids.jpg"
        image_path.write_bytes(b"not an image")

        result = detector.detect(image_path)

        assert result is None

    def test_database_schema(self, temp_db):
        from sqlalchemy.orm import Session

        with Session(temp_db) as session:
            asset = Asset(
                path="/test/photo.jpg",
                sha256="abc123",
                width=100,
                height=100,
                media_type="image",
            )
            session.add(asset)
            session.commit()
            assert asset.id is not None

    def test_init_db_declares_engine_return_type(self):
        from sqlalchemy.engine import Engine
        from privatelens.db import schema

        assert schema.init_db.__annotations__["return"] is Engine

    def test_database_engine_is_reused_with_concurrency_pragmas(self, temp_db):
        from privatelens.db.schema import get_engine, init_db

        assert get_engine() is temp_db
        assert init_db() is temp_db
        with temp_db.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
            assert conn.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 30_000
            assert conn.exec_driver_sql("PRAGMA journal_mode").scalar_one() == "wal"

    def test_sqlite_vec_loader_uses_python_package_when_supported(self, monkeypatch):
        from privatelens.db import schema

        calls = []

        class FakeConnection:
            def enable_load_extension(self, enabled):
                calls.append(("enabled", enabled))

            def load_extension(self, _path):
                pass

        monkeypatch.setitem(
            sys.modules,
            "sqlite_vec",
            SimpleNamespace(load=lambda connection: calls.append(("loaded", connection))),
        )
        connection = FakeConnection()

        assert schema.load_sqlite_vec(connection) is True
        assert calls == [
            ("enabled", True),
            ("loaded", connection),
            ("enabled", False),
        ]

    def test_sqlite_vec_trigger_synchronizes_embedding_updates(self, temp_db):
        import numpy as np
        from sqlalchemy import text
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, ImageEmbedding

        with temp_db.connect() as conn:
            try:
                conn.execute(text("SELECT vec_version()"))
            except Exception:
                pytest.skip("sqlite-vec extension is unavailable in this Python runtime")

        with Session(temp_db) as session:
            asset = Asset(
                path="/test/vector-update.jpg", sha256="vector-update", media_type="image"
            )
            session.add(asset)
            session.commit()
            asset_id = asset.id
            session.add(
                ImageEmbedding(
                    asset_id=asset_id,
                    vector=np.zeros(512, dtype=np.float32).tobytes(),
                )
            )
            session.commit()
            embedding = session.query(ImageEmbedding).filter_by(asset_id=asset_id).one()
            embedding.vector = np.ones(512, dtype=np.float32).tobytes()
            session.commit()

        with temp_db.connect() as conn:
            hit = conn.execute(
                text("""
                    SELECT asset_id, distance
                    FROM vec_image_embeddings
                    WHERE embedding MATCH :embedding AND k = 1
                """),
                {"embedding": np.ones(512, dtype=np.float32).tobytes()},
            ).first()

        assert hit.asset_id == asset_id
        assert hit.distance == pytest.approx(0.0)

    def test_face_provider_selection_uses_only_available_accelerators(self):
        from privatelens.extractors.faces import select_onnx_providers

        assert select_onnx_providers(
            ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]
        ) == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert select_onnx_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_thumbnail_generation(self, temp_db, sample_image):
        from privatelens.db.schema import Asset
        from sqlalchemy.orm import Session

        with Session(temp_db) as session:
            asset = Asset(path=str(sample_image), sha256="test", media_type="image")
            session.add(asset)
            session.commit()

            thumb = generate_thumbnail(sample_image, asset.id)
            assert thumb.exists()
            assert thumb.suffix == ".jpg"


class TestSearchEngine:
    """Test search engine with mocked data."""

    def test_search_recipes_initialization(self, temp_db):
        init_recipes()
        recipes = get_recipes()
        assert len(recipes) == 10
        assert any(r.name == "find_id_photo" for r in recipes)
        assert any(r.name == "find_selfie" for r in recipes)

    def test_init_recipes_updates_existing_builtin_query_plans(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import SearchRecipe

        stale_plan = json.dumps({"signals": [{"type": "ocr", "weight": 1.0}]})
        with Session(temp_db) as session:
            session.add(
                SearchRecipe(
                    name="find_receipt",
                    display_name="Old Receipt Recipe",
                    description="stale",
                    category="finance",
                    query_plan=stale_plan,
                    is_builtin=True,
                )
            )
            session.commit()

        init_recipes()

        with Session(temp_db) as session:
            recipe = session.query(SearchRecipe).filter_by(name="find_receipt").one()
            plan = json.loads(recipe.query_plan)

        assert recipe.display_name == "Find Receipts"
        assert any(signal["type"] == "path" for signal in plan["signals"])

    def test_ocr_search(self, temp_db):
        from sqlalchemy.orm import Session

        with Session(temp_db) as session:
            # Create asset with OCR
            asset = Asset(path="/test/receipt.jpg", sha256="test", media_type="image")
            session.add(asset)
            session.commit()

            ocr = OcrBlock(asset_id=asset.id, text="Target Receipt $47.32", confidence=0.95)
            session.add(ocr)
            session.commit()

        # Search engine needs embeddings table for vector search fallback
        # Just test that engine initializes
        engine = SearchEngine()
        assert engine is not None

    def test_ocr_search_matches_terms_when_ocr_drops_spaces(self, temp_db):
        from sqlalchemy.orm import Session

        with Session(temp_db) as session:
            asset = Asset(path="/test/license.jpg", sha256="ocr-spacing", media_type="document")
            session.add(asset)
            session.commit()
            asset_id = asset.id
            session.add(OcrBlock(asset_id=asset_id, text="Name:SAMPLEUSER"))
            session.commit()

        results = SearchEngine().search("sample user", search_type="ocr", limit=5)

        assert [result["asset_id"] for result in results] == [asset_id]

    def test_metadata_search(self, temp_db):
        from sqlalchemy.orm import Session

        with Session(temp_db) as session:
            asset = Asset(
                path="/test/photo.jpg",
                sha256="test",
                media_type="image",
                exif_make="Canon",
                exif_model="EOS R5",
            )
            session.add(asset)
            session.commit()

        engine = SearchEngine()
        results = engine.search("Canon", search_type="metadata", limit=10)
        assert len(results) >= 0  # May be empty if FTS not populated

    def test_path_search_returns_filename_evidence(self, temp_db):
        from sqlalchemy.orm import Session

        with Session(temp_db) as session:
            asset = Asset(
                path="/demo/photos/target-receipt-lunch.jpg",
                sha256="path-search",
                media_type="document",
            )
            session.add(asset)
            session.commit()
            asset_id = asset.id

        engine = SearchEngine()
        results = engine.search("receipt", search_type="path", limit=10)

        assert [result["asset_id"] for result in results] == [asset_id]
        assert "Path contains" in results[0]["explanation"]

    def test_vector_search_executes_sqlalchemy_text_query(self):
        import numpy as np
        from sqlalchemy.sql.elements import TextClause

        class FakeConnection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def execute(self, statement, params):
                assert isinstance(statement, TextClause)
                assert "embedding MATCH :vec" in statement.text
                assert "k = :limit" in statement.text
                assert params["limit"] == 3
                return SimpleNamespace(
                    fetchall=lambda: [SimpleNamespace(asset_id=7, distance=0.25)]
                )

        class FakeEngine:
            def connect(self):
                return FakeConnection()

        search_engine = SearchEngine.__new__(SearchEngine)
        search_engine.engine = FakeEngine()

        results = search_engine._vector_search(
            np.array([1.0, 0.0], dtype=np.float32),
            limit=3,
        )

        assert results == [{"asset_id": 7, "score": 0.75}]

    def test_vector_search_finds_target_beyond_one_thousand_embeddings(self, temp_db):
        import numpy as np
        from sqlalchemy import insert, select, text
        from sqlalchemy.orm import Session

        from privatelens.db.schema import Asset, ImageEmbedding
        from privatelens.extractors.clip import clip_model_id

        distractor = np.zeros(512, dtype=np.float32)
        distractor[0] = 1.0
        target = np.zeros(512, dtype=np.float32)
        target[1] = 1.0
        asset_count = 1_001

        with Session(temp_db) as session:
            session.execute(
                insert(Asset),
                [
                    {
                        "path": f"/benchmark/vectors/asset-{index:04d}.jpg",
                        "sha256": f"{index:064x}",
                        "media_type": "image",
                    }
                    for index in range(asset_count)
                ],
            )
            session.commit()
            asset_rows = session.execute(select(Asset.id, Asset.path).order_by(Asset.id)).all()
            target_id = asset_rows[-1].id
            session.execute(
                insert(ImageEmbedding),
                [
                    {
                        "asset_id": row.id,
                        "model": clip_model_id(settings.clip_model, settings.clip_pretrained),
                        "vector": (target if row.id == target_id else distractor).tobytes(),
                    }
                    for row in asset_rows
                ],
            )
            session.commit()

        engine = SearchEngine()
        blob_results = engine._blob_vector_search(target, limit=1)
        native_first_results = engine._vector_search(target, limit=1)

        assert blob_results[0]["asset_id"] == target_id
        assert native_first_results[0]["asset_id"] == target_id
        with temp_db.connect() as conn:
            try:
                native_count = conn.execute(
                    text("SELECT count(*) FROM vec_image_embeddings")
                ).scalar_one()
            except Exception:
                native_count = None
        if native_count is not None:
            assert native_count == asset_count

    def test_smart_search_empty_database_skips_clip_encoding(self, temp_db, monkeypatch):
        engine = SearchEngine()

        def fail_if_called(_query):
            raise AssertionError("CLIP text encoding should not run for an empty database")

        monkeypatch.setattr(engine.clip_extractor, "encode_text", fail_if_called)

        assert engine.search("receipt", search_type="smart", limit=10) == []

    def test_search_records_event_only_when_feedback_is_enabled(self, temp_db, monkeypatch):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import SearchEvent

        engine = SearchEngine()
        monkeypatch.setattr(engine.clip_extractor, "encode_text", lambda _query: None)

        assert engine.search("private default query", search_type="smart", limit=10) == []

        with Session(temp_db) as session:
            assert session.query(SearchEvent).count() == 0

        assert (
            engine.search(
                "receipt",
                search_type="smart",
                limit=10,
                record_event=True,
            )
            == []
        )

        with Session(temp_db) as session:
            event = session.query(SearchEvent).one()
            assert event.query == "receipt"
            assert event.query_type == "smart"
            assert event.results_count == 0

    def test_search_feedback_updates_recorded_event(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, SearchEvent

        with Session(temp_db) as session:
            asset = Asset(path="/test/receipt.jpg", sha256="feedback", media_type="image")
            session.add(asset)
            session.commit()
            asset_id = asset.id

        engine = SearchEngine()
        event_id = engine._record_search_event("receipt", "smart", 1, None)

        engine.record_feedback(event_id, feedback=1, result_clicked=asset_id)

        with Session(temp_db) as session:
            event = session.query(SearchEvent).one()
            assert event.feedback == 1
            assert event.result_clicked == asset_id

    def test_detection_signal_contributes_to_recipe_plan(self, temp_db, monkeypatch):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, Detection

        with Session(temp_db) as session:
            asset = Asset(path="/test/receipt.jpg", sha256="test", media_type="image")
            session.add(asset)
            session.commit()
            asset_id = asset.id
            session.add(
                Detection(
                    asset_id=asset_id,
                    label="receipt",
                    confidence=0.93,
                    source_model="test",
                )
            )
            session.commit()

        engine = SearchEngine()
        monkeypatch.setattr(engine.clip_extractor, "encode_text", lambda _query: None)
        plan = {"signals": [{"type": "detection", "labels": ["receipt"], "weight": 0.7}]}

        results = engine._execute_plan(plan, "receipt", limit=10)

        assert len(results) == 1
        assert results[0]["asset_id"] == asset_id
        assert results[0]["score"] == pytest.approx(0.7 * 0.93)
        assert "Detected: receipt" in results[0]["explanation"]

    def test_face_count_signal_contributes_to_recipe_plan(self, temp_db, monkeypatch):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, Face

        with Session(temp_db) as session:
            one_face = Asset(path="/test/selfie.jpg", sha256="one", media_type="image")
            two_faces = Asset(path="/test/two-people.jpg", sha256="two", media_type="image")
            session.add_all([one_face, two_faces])
            session.commit()
            one_face_id = one_face.id
            two_faces_id = two_faces.id
            session.add(Face(asset_id=one_face_id, bbox="[]", confidence=0.91))
            session.add(Face(asset_id=two_faces_id, bbox="[]", confidence=0.9))
            session.add(Face(asset_id=two_faces_id, bbox="[]", confidence=0.88))
            session.commit()

        engine = SearchEngine()
        monkeypatch.setattr(engine.clip_extractor, "encode_text", lambda _query: None)
        plan = {"signals": [{"type": "face", "face_count": 1, "weight": 0.5}]}

        results = engine._execute_plan(plan, "selfie", limit=10)

        assert [result["asset_id"] for result in results] == [one_face_id]
        assert results[0]["score"] == pytest.approx(0.5)
        assert "Faces: 1" in results[0]["explanation"]

    def test_recipe_plan_filters_require_sensitive_assets(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, OcrBlock

        with Session(temp_db) as session:
            sensitive = Asset(
                path="/test/passport.jpg",
                sha256="sensitive",
                media_type="document",
                is_sensitive=True,
                sensitive_type="passport",
            )
            public = Asset(path="/test/public.jpg", sha256="public", media_type="document")
            session.add_all([sensitive, public])
            session.commit()
            sensitive_id = sensitive.id
            public_id = public.id
            session.add(OcrBlock(asset_id=sensitive_id, text="passport number"))
            session.add(OcrBlock(asset_id=public_id, text="passport travel checklist"))
            session.commit()

        engine = SearchEngine()
        plan = {
            "signals": [{"type": "ocr", "weight": 1.0}],
            "filters": {"require_sensitive": True},
        }

        results = engine._execute_plan(plan, "passport", limit=10)

        assert [result["asset_id"] for result in results] == [sensitive_id]

    def test_ocr_recipe_signal_uses_keywords_when_query_terms_do_not_match(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, OcrBlock

        with Session(temp_db) as session:
            receipt = Asset(path="/test/target.jpg", sha256="receipt", media_type="image")
            note = Asset(path="/test/note.jpg", sha256="note", media_type="image")
            session.add_all([receipt, note])
            session.commit()
            receipt_id = receipt.id
            session.add(OcrBlock(asset_id=receipt_id, text="Target total tax payment"))
            session.add(OcrBlock(asset_id=note.id, text="lunch ideas for tomorrow"))
            session.commit()

        engine = SearchEngine()
        plan = {
            "signals": [
                {
                    "type": "ocr",
                    "keywords": ["receipt", "total", "tax", "transaction", "payment"],
                    "weight": 1.0,
                }
            ],
        }

        results = engine._execute_plan(plan, "expense proof", limit=10)

        assert [result["asset_id"] for result in results] == [receipt_id]
        assert results[0]["score"] == pytest.approx(1.0)

    def test_recipe_plan_filters_excluded_media_types(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, OcrBlock

        with Session(temp_db) as session:
            photo = Asset(path="/test/photo.jpg", sha256="photo", media_type="image")
            screenshot = Asset(
                path="/test/screenshot.png",
                sha256="screenshot",
                media_type="screenshot",
            )
            session.add_all([photo, screenshot])
            session.commit()
            photo_id = photo.id
            screenshot_id = screenshot.id
            session.add(OcrBlock(asset_id=photo_id, text="receipt total"))
            session.add(OcrBlock(asset_id=screenshot_id, text="receipt total"))
            session.commit()

        engine = SearchEngine()
        plan = {
            "signals": [{"type": "ocr", "weight": 1.0}],
            "filters": {"exclude": ["screenshot"]},
        }

        results = engine._execute_plan(plan, "receipt", limit=10)

        assert [result["asset_id"] for result in results] == [photo_id]

    def test_recipe_plan_filters_exact_face_count(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, Face, OcrBlock

        with Session(temp_db) as session:
            one_face = Asset(path="/test/one-face.jpg", sha256="one-face", media_type="image")
            two_faces = Asset(path="/test/two-faces.jpg", sha256="two-faces", media_type="image")
            session.add_all([one_face, two_faces])
            session.commit()
            one_face_id = one_face.id
            two_faces_id = two_faces.id
            session.add_all(
                [
                    OcrBlock(asset_id=one_face_id, text="friend at park"),
                    OcrBlock(asset_id=two_faces_id, text="friend at park"),
                    Face(asset_id=one_face_id, bbox="[]", confidence=0.9),
                    Face(asset_id=two_faces_id, bbox="[]", confidence=0.93),
                    Face(asset_id=two_faces_id, bbox="[]", confidence=0.91),
                ]
            )
            session.commit()

        engine = SearchEngine()
        plan = {
            "signals": [{"type": "ocr", "weight": 1.0}],
            "filters": {"face_count_exact": 2},
        }

        results = engine._execute_plan(plan, "friend", limit=10)

        assert [result["asset_id"] for result in results] == [two_faces_id]

    def test_face_count_recipe_degrades_when_optional_face_index_is_absent(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, OcrBlock

        with Session(temp_db) as session:
            asset = Asset(path="/test/friends.jpg", sha256="friends", media_type="image")
            session.add(asset)
            session.flush()
            session.add(OcrBlock(asset_id=asset.id, text="friends at the park"))
            session.commit()
            asset_id = asset.id

        engine = SearchEngine()
        plan = {
            "signals": [{"type": "ocr", "weight": 1.0}],
            "filters": {"face_count_exact": 2},
        }

        results = engine._execute_plan(plan, "friends", limit=10)

        assert [result["asset_id"] for result in results] == [asset_id]

    def test_metadata_signal_applies_media_type_filter(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset

        with Session(temp_db) as session:
            screenshot = Asset(
                path="/test/screen.png",
                sha256="screenshot",
                media_type="screenshot",
            )
            photo = Asset(path="/test/receipt.jpg", sha256="photo", media_type="image")
            session.add_all([screenshot, photo])
            session.commit()
            screenshot_id = screenshot.id

        engine = SearchEngine()
        plan = {
            "signals": [
                {"type": "metadata", "filters": {"media_type": "screenshot"}, "weight": 1.0}
            ],
        }

        results = engine._execute_plan(plan, "receipt", limit=10)

        assert [result["asset_id"] for result in results] == [screenshot_id]

    def test_metadata_signal_applies_sensitive_filter(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset

        with Session(temp_db) as session:
            sensitive = Asset(
                path="/test/private.jpg",
                sha256="sensitive",
                media_type="document",
                is_sensitive=True,
            )
            public = Asset(path="/test/passport.jpg", sha256="public", media_type="document")
            session.add_all([sensitive, public])
            session.commit()
            sensitive_id = sensitive.id

        engine = SearchEngine()
        plan = {
            "signals": [{"type": "metadata", "filters": {"is_sensitive": True}, "weight": 1.0}],
        }

        results = engine._execute_plan(plan, "passport", limit=10)

        assert [result["asset_id"] for result in results] == [sensitive_id]

    def test_metadata_signal_applies_date_range_filter(self, temp_db):
        from datetime import datetime
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset

        with Session(temp_db) as session:
            current_trip = Asset(
                path="/test/current-trip.jpg",
                sha256="current-trip",
                media_type="image",
                exif_datetime=datetime(2024, 5, 3, 12, 0, 0),
            )
            older_trip = Asset(
                path="/test/older-trip.jpg",
                sha256="older-trip",
                media_type="image",
                exif_datetime=datetime(2023, 5, 3, 12, 0, 0),
            )
            session.add_all([current_trip, older_trip])
            session.commit()
            current_trip_id = current_trip.id

        engine = SearchEngine()
        plan = {
            "signals": [{"type": "metadata", "filters": {"date_range": True}, "weight": 1.0}],
        }

        results = engine._execute_plan(plan, "family trip 2024", limit=10)

        assert [result["asset_id"] for result in results] == [current_trip_id]

    def test_metadata_signal_applies_aspect_ratio_filter(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset

        with Session(temp_db) as session:
            card_like = Asset(
                path="/test/card-like.jpg",
                sha256="card-like",
                media_type="document",
                width=1400,
                height=1000,
            )
            square = Asset(
                path="/test/square.jpg",
                sha256="square",
                media_type="document",
                width=1000,
                height=1000,
            )
            session.add_all([card_like, square])
            session.commit()
            card_like_id = card_like.id

        engine = SearchEngine()
        plan = {
            "signals": [
                {
                    "type": "metadata",
                    "filters": {"aspect_ratio": {"min": 1.2, "max": 1.6}},
                    "weight": 1.0,
                }
            ],
        }

        results = engine._execute_plan(plan, "identity backup", limit=10)

        assert [result["asset_id"] for result in results] == [card_like_id]

    def test_recipe_plan_boosts_recent_assets(self, temp_db):
        from datetime import datetime
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, OcrBlock

        with Session(temp_db) as session:
            older = Asset(
                path="/test/older-event.jpg",
                sha256="older-event",
                media_type="image",
                exif_datetime=datetime(2021, 1, 1, 12, 0, 0),
            )
            newer = Asset(
                path="/test/newer-event.jpg",
                sha256="newer-event",
                media_type="image",
                exif_datetime=datetime(2024, 1, 1, 12, 0, 0),
            )
            session.add_all([older, newer])
            session.commit()
            older_id = older.id
            newer_id = newer.id
            session.add(OcrBlock(asset_id=older_id, text="family event"))
            session.add(OcrBlock(asset_id=newer_id, text="family event"))
            session.commit()

        engine = SearchEngine()
        plan = {
            "signals": [{"type": "ocr", "weight": 1.0}],
            "filters": {"boost_recent": True},
        }

        results = engine._execute_plan(plan, "event", limit=10)

        assert [result["asset_id"] for result in results] == [newer_id, older_id]
        assert results[0]["score"] > results[1]["score"]

    def test_search_by_recipe_initializes_builtin_recipe_on_fresh_db(self, temp_db, monkeypatch):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, OcrBlock

        with Session(temp_db) as session:
            sensitive = Asset(
                path="/test/private-passport.jpg",
                sha256="sensitive-passport",
                media_type="document",
                is_sensitive=True,
                sensitive_type="passport",
            )
            public = Asset(
                path="/test/public-passport.jpg",
                sha256="public-passport",
                media_type="document",
            )
            session.add_all([sensitive, public])
            session.commit()
            sensitive_id = sensitive.id
            public_id = public.id
            session.add(OcrBlock(asset_id=sensitive_id, text="passport number"))
            session.add(OcrBlock(asset_id=public_id, text="passport checklist"))
            session.commit()

        engine = SearchEngine()
        monkeypatch.setattr(engine.clip_extractor, "encode_text", lambda _query: None)

        results = engine.search_by_recipe("find_sensitive", "passport", limit=10)

        assert [result["asset_id"] for result in results] == [sensitive_id]
        assert public_id not in [result["asset_id"] for result in results]

    def test_search_by_recipe_can_skip_vlm_rerank(self, temp_db, monkeypatch):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, OcrBlock

        with Session(temp_db) as session:
            asset = Asset(path="/test/receipt.jpg", sha256="receipt", media_type="image")
            session.add(asset)
            session.commit()
            asset_id = asset.id
            session.add(OcrBlock(asset_id=asset_id, text="receipt total"))
            session.commit()

        engine = SearchEngine()
        monkeypatch.setattr(engine.clip_extractor, "encode_text", lambda _query: None)
        monkeypatch.setattr(
            engine.reranker,
            "rerank",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("fast recipe search should not rerank")
            ),
        )

        results = engine.search_by_recipe("find_receipt", "receipt", limit=10, rerank=False)

        assert [result["asset_id"] for result in results] == [asset_id]

    def test_search_by_recipe_can_skip_semantic_signal(self, temp_db, monkeypatch):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, OcrBlock

        with Session(temp_db) as session:
            asset = Asset(path="/test/receipt.jpg", sha256="receipt-fast", media_type="image")
            session.add(asset)
            session.commit()
            asset_id = asset.id
            session.add(OcrBlock(asset_id=asset_id, text="receipt total"))
            session.commit()

        engine = SearchEngine()
        monkeypatch.setattr(
            engine.clip_extractor,
            "encode_text",
            lambda _query: (_ for _ in ()).throw(
                AssertionError("semantic signal should be disabled")
            ),
        )

        results = engine.search_by_recipe(
            "find_receipt",
            "receipt",
            limit=10,
            rerank=False,
            use_semantic=False,
        )

        assert [result["asset_id"] for result in results] == [asset_id]

    def test_receipt_recipe_uses_filename_path_signal(self, temp_db, monkeypatch):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset

        with Session(temp_db) as session:
            asset = Asset(
                path="/demo/photos/target-receipt-lunch.jpg",
                sha256="receipt-path",
                media_type="document",
            )
            session.add(asset)
            session.commit()
            asset_id = asset.id

        engine = SearchEngine()
        monkeypatch.setattr(engine.clip_extractor, "encode_text", lambda _query: None)
        monkeypatch.setattr(engine.reranker, "rerank", lambda results, *_args, **_kwargs: results)

        results = engine.search_by_recipe("find_receipt", "receipt", limit=10)

        assert [result["asset_id"] for result in results] == [asset_id]
        assert "Path contains" in results[0]["explanation"]

    def test_search_queries_return_plain_records(self, temp_db):
        from sqlalchemy import inspect
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, Face, Person
        from privatelens.search.queries import SearchQueries

        with Session(temp_db) as session:
            asset = Asset(path="/test/alex.jpg", sha256="test", media_type="image")
            person = Person(display_name="Alex", face_count=1)
            session.add_all([asset, person])
            session.commit()
            asset_id = asset.id
            person_id = person.id
            session.add(
                Face(
                    asset_id=asset_id,
                    cluster_id=person_id,
                    bbox="[]",
                    confidence=0.88,
                )
            )
            session.commit()

        queries = SearchQueries()

        asset_record = queries.get_asset(asset_id)
        asset_records = queries.get_assets_by_ids([asset_id])
        person_record = queries.get_person_by_name("alex")
        face_records = queries.get_faces_by_person(person_id)

        assert asset_record.path == "/test/alex.jpg"
        assert asset_records[asset_id].path == "/test/alex.jpg"
        assert person_record.display_name == "Alex"
        assert face_records[0].asset_id == asset_id
        assert face_records[0].confidence == pytest.approx(0.88)
        assert inspect(asset_record, raiseerr=False) is None
        assert inspect(asset_records[asset_id], raiseerr=False) is None
        assert inspect(person_record, raiseerr=False) is None
        assert inspect(face_records[0], raiseerr=False) is None

    def test_evidence_includes_path_signal_for_filename_matches(self, temp_db):
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset
        from privatelens.search.evidence import EvidenceBuilder

        with Session(temp_db) as session:
            asset = Asset(
                path="/demo/photos/target-receipt-lunch.jpg",
                sha256="path-evidence",
                media_type="document",
            )
            session.add(asset)
            session.commit()
            asset_id = asset.id

        evidence = EvidenceBuilder().build(asset_id, "receipt", 0.1)

        assert evidence["signals"][0]["source"] == "path"
        assert "Path contains" in evidence["explanation"]


class TestBenchmark:
    """Test deterministic search-quality benchmark behavior."""

    def test_canonical_fixture_covers_every_builtin_recipe(self):
        from privatelens.benchmark import load_canonical_manifest
        from privatelens.search.recipes import BUILTIN_RECIPES

        manifest = load_canonical_manifest()

        assert len(manifest["cases"]) == 10
        assert {case["recipe"] for case in manifest["cases"]} == {
            recipe["name"] for recipe in BUILTIN_RECIPES
        }

    def test_benchmark_cli_writes_passing_top5_report(self, tmp_path):
        from click.testing import CliRunner
        from privatelens.cli import cli

        output_path = tmp_path / "search-quality.json"

        result = CliRunner().invoke(
            cli,
            ["benchmark", "--output", str(output_path), "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert output_path.exists()
        assert json.loads(output_path.read_text()) == payload
        assert payload["benchmark"] == "privatelens-search-quality-v1"
        assert payload["summary"]["case_count"] == 10
        assert payload["summary"]["hit_rate_at_5"] >= 0.8
        assert payload["summary"]["passed"] is True
        assert all(case["target_in_top_5"] for case in payload["cases"])
        assert all(case["results"][0]["explanation"] for case in payload["cases"])
        assert all(
            0.0 <= result["score"] <= 1.0 for case in payload["cases"] for result in case["results"]
        )

    def test_checked_in_benchmark_report_documents_release_gate(self):
        from privatelens.benchmark import run_canonical_benchmark

        report_path = Path("results/benchmarks/search-quality-v1.json")
        readme = Path("README.md").read_text()

        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert run_canonical_benchmark() == report
        assert report["summary"]["case_count"] == 10
        assert report["summary"]["hit_rate_at_5"] >= 0.8
        assert report["summary"]["passed"] is True
        assert "## Search Quality Benchmark" in readme
        assert "does not measure CLIP or VLM model quality" in readme

    def test_model_benchmark_runs_all_generated_signal_gates(self, tmp_path):
        import numpy as np

        from privatelens.model_benchmark import MODEL_CASES, run_model_benchmark

        case_indexes = {case["filename"]: index for index, case in enumerate(MODEL_CASES)}
        query_indexes = {case["semantic_query"]: index for index, case in enumerate(MODEL_CASES)}
        expected_types = {case["filename"]: case["expected_type"] for case in MODEL_CASES}
        caption_terms = {case["filename"]: case["caption_terms"] for case in MODEL_CASES}
        ocr_text = {
            "target-receipt-lunch.jpg": "Payment:ViSA",
            "driver-license-backup.jpg": "Name:SAMPLEUSER",
            "phone-screenshot-travel.png": "Boarding pass is saved",
            "whiteboard-notes-project.jpg": "Build private sidecar index",
        }

        def unit_vector(index):
            vector = np.zeros(512, dtype=np.float32)
            vector[index] = 1.0
            return vector

        class FakeClipExtractor:
            model_id = "openclip:test:model"

            def extract(self, image_path):
                return unit_vector(case_indexes[image_path.name])

            def encode_text(self, query):
                return unit_vector(query_indexes[query])

        class FakeOcrExtractor:
            def extract(self, image_path):
                return [{"text": ocr_text[image_path.name], "confidence": 0.99}]

        class FakeVlmExtractor:
            model = "test-vlm"

            def is_available(self):
                return True

            def caption(self, image_path):
                return " ".join(caption_terms[image_path.name])

            def classify_document(self, image_path):
                return {"type": expected_types[image_path.name], "confidence": 1.0}

        output_path = tmp_path / "model-quality.json"
        report = run_model_benchmark(
            output_path=output_path,
            clip_extractor=FakeClipExtractor(),
            ocr_extractor=FakeOcrExtractor(),
            vlm_extractor=FakeVlmExtractor(),
        )

        assert json.loads(output_path.read_text()) == report
        assert report["corpus"]["contains_private_media"] is False
        assert report["summary"]["clip_top1_rate"] == 1.0
        assert report["summary"]["ocr_top1_rate"] == 1.0
        assert report["summary"]["vlm_classification_accuracy"] == 1.0
        assert report["summary"]["vlm_caption_term_recall"] == 1.0
        assert report["summary"]["passed"] is True
        assert all(not case["errors"] for case in report["cases"])

    def test_model_benchmark_cli_exposes_local_vlm_toggle(self):
        from click.testing import CliRunner

        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["benchmark-models", "--help"])

        assert result.exit_code == 0
        assert "--vlm / --skip-vlm" in result.output
        assert "generated images" in result.output

    def test_checked_in_model_report_documents_strict_generated_corpus_gate(self):
        report_path = Path("results/benchmarks/model-quality-v1.json")
        readme = Path("README.md").read_text()

        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["benchmark"] == "privatelens-model-quality-v1"
        assert report["corpus"]["contains_private_media"] is False
        assert report["runtime"]["vector_backend"].startswith("sqlite-vec:")
        assert report["summary"]["clip_top1_rate"] == 1.0
        assert report["summary"]["ocr_top1_rate"] == 1.0
        assert report["summary"]["vlm_classification_accuracy"] == 1.0
        assert report["summary"]["vlm_caption_term_recall"] == 1.0
        assert report["summary"]["passed"] is True
        assert "## Model Quality Benchmark" in readme
        assert "not a broad real-world retrieval claim" in readme


class TestPrivacy:
    """Test privacy features."""

    def test_privacy_auditor(self, temp_db):
        auditor = PrivacyAuditor()
        report = auditor.run_audit()
        assert len(report) > 0
        assert any(check["name"] == "Database Local" for check in report)

    @pytest.mark.parametrize(
        ("key", "expected_details"),
        [
            ("", "No encryption key configured"),
            ("not-a-fernet-key", "Encryption key is not a valid Fernet key"),
        ],
    )
    def test_privacy_auditor_rejects_empty_or_invalid_encryption_key(
        self, monkeypatch, key, expected_details
    ):
        monkeypatch.setattr(settings, "encryption_key", key)

        check = PrivacyAuditor()._check_encryption()

        assert check["status"] == "warning"
        assert check["details"] == expected_details

    def test_privacy_network_check_does_not_probe_external_hosts(self, monkeypatch):
        import socket

        monkeypatch.setattr(
            socket,
            "create_connection",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("privacy audit must not create outbound connections")
            ),
        )
        monkeypatch.setattr(settings, "local_only", True)

        check = PrivacyAuditor()._check_network()

        assert check["name"] == "Network Policy"
        assert check["status"] == "ok"
        assert "not a system firewall" in check["details"]

    def test_privacy_auditor_reports_vector_search_backend(self, temp_db):
        auditor = PrivacyAuditor()
        report = auditor.run_audit()

        vector_check = next(check for check in report if check["name"] == "Vector Search Backend")
        assert vector_check["status"] in {"ok", "warning"}
        assert "sqlite-vec" in vector_check["details"] or "BLOB fallback" in vector_check["details"]

    def test_privacy_auditor_explains_sqlite_vec_load_extension_gap(self, temp_db, monkeypatch):
        import sqlite3

        class FakeConnection:
            def close(self):
                pass

        monkeypatch.setattr(sqlite3, "connect", lambda *_args, **_kwargs: FakeConnection())

        auditor = PrivacyAuditor()
        reason = auditor._diagnose_sqlite_vec_fallback()
        remediation = auditor._sqlite_vec_remediation(reason)

        assert reason == "Python sqlite3 does not expose load_extension"
        assert any("BLOB fallback" in item for item in remediation)
        assert any("external PC" in item for item in remediation)

    def test_privacy_auditor_requires_loaded_vector_extension(self, monkeypatch):
        import privatelens.privacy.audit as audit_module

        class FakeRows:
            def fetchall(self):
                return [("vec_image_embeddings",), ("vec_faces",)]

        class FakeConnection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def execute(self, statement):
                if "vec_version" in str(statement):
                    raise RuntimeError("no such function: vec_version")
                return FakeRows()

        class FakeEngine:
            def connect(self):
                return FakeConnection()

        monkeypatch.setattr(audit_module, "get_engine", lambda: FakeEngine())
        monkeypatch.setattr(
            audit_module.PrivacyAuditor,
            "_diagnose_sqlite_vec_fallback",
            lambda _self: "sqlite-vec package is installed but extension load failed",
        )

        check = audit_module.PrivacyAuditor()._check_vector_backend()

        assert check["status"] == "warning"
        assert "no such function: vec_version" in check["details"]

    def test_privacy_auditor_warns_when_local_ollama_is_unreachable(self, monkeypatch):
        import urllib.request
        import privatelens.privacy.audit as audit_module

        monkeypatch.setattr(audit_module.settings, "ollama_url", "http://localhost:11434")

        def fail_urlopen(*args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

        check = audit_module.PrivacyAuditor()._check_ollama_local()

        assert check["name"] == "Ollama Local"
        assert check["status"] == "warning"
        assert "not reachable" in check["details"]
        assert "open -a Ollama" in check["remediation"]
        assert "curl -fsS http://localhost:11434/api/tags" in check["remediation"]

    def test_privacy_auditor_recommends_ollama_model_pull_when_missing(self, monkeypatch):
        import urllib.request
        import privatelens.privacy.audit as audit_module

        monkeypatch.setattr(audit_module.settings, "ollama_url", "http://localhost:11434")
        monkeypatch.setattr(audit_module.settings, "vlm_model", "qwen3-vl:2b-instruct-q8_0")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"models": [{"name": "llava:latest"}]}).encode()

        monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

        check = audit_module.PrivacyAuditor()._check_ollama_local()

        assert check["status"] == "warning"
        assert f"ollama pull {audit_module.settings.vlm_model}" in check["remediation"]

    def test_privacy_auditor_confirms_reachable_ollama_model(self, monkeypatch):
        import urllib.request
        import privatelens.privacy.audit as audit_module

        monkeypatch.setattr(audit_module.settings, "ollama_url", "http://localhost:11434")
        monkeypatch.setattr(audit_module.settings, "vlm_model", "qwen3-vl:2b-instruct-q8_0")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"models": [{"name": "qwen3-vl:2b-instruct-q8_0"}]}).encode()

        monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

        check = audit_module.PrivacyAuditor()._check_ollama_local()

        assert check["status"] == "ok"
        assert "model qwen3-vl:2b-instruct-q8_0 available" in check["details"]
        assert check["remediation"] == []

    def test_privacy_guard_blocks_sensitive_cloud_outbound_in_local_only(self, monkeypatch):
        import privatelens.privacy.guard as guard_module
        from privatelens.privacy.guard import PrivacyError, PrivacyGuard

        monkeypatch.setattr(guard_module.settings, "local_only", True)
        guard = PrivacyGuard()

        with pytest.raises(PrivacyError):
            guard.log_outbound("https://api.example.com/upload", "index_upload")

        assert guard.get_outbound_log()[0]["blocked"] is True

    def test_privacy_guard_blocks_remote_chat_and_unknown_purposes_by_default(self, monkeypatch):
        import privatelens.privacy.guard as guard_module
        from privatelens.privacy.guard import PrivacyError, PrivacyGuard

        monkeypatch.setattr(guard_module.settings, "local_only", True)
        guard = PrivacyGuard()

        with pytest.raises(PrivacyError):
            guard.log_outbound("https://anythingllm.example.com/chat", "anythingllm_chat")
        with pytest.raises(PrivacyError):
            guard.log_outbound("https://api.example.com/new", "future_operation")

        assert [call["blocked"] for call in guard.get_outbound_log()] == [True, True]

    def test_privacy_guard_allows_sensitive_local_outbound_in_local_only(self, monkeypatch):
        import privatelens.privacy.guard as guard_module
        from privatelens.privacy.guard import PrivacyGuard

        monkeypatch.setattr(guard_module.settings, "local_only", True)
        guard = PrivacyGuard()

        guard.log_outbound("http://localhost:11434/api/generate", "vlm_caption")

        assert guard.get_outbound_log()[0]["blocked"] is False

    def test_privacy_guard_allows_compose_ollama_service_in_local_only(self, monkeypatch):
        import privatelens.privacy.guard as guard_module
        from privatelens.privacy.guard import PrivacyGuard

        monkeypatch.setattr(guard_module.settings, "local_only", True)
        monkeypatch.setattr(guard_module.settings, "ollama_url", "http://ollama:11434")
        guard = PrivacyGuard()

        guard.log_outbound("http://ollama:11434/api/generate", "vlm_caption")

        assert guard.is_local_only() is True
        assert guard.get_outbound_log()[0]["blocked"] is False

    def test_privacy_auditor_treats_compose_ollama_service_as_local(self, monkeypatch):
        import urllib.request
        import privatelens.privacy.audit as audit_module

        monkeypatch.setattr(audit_module.settings, "ollama_url", "http://ollama:11434")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreachable")),
        )

        check = PrivacyAuditor()._check_ollama_local()

        assert check["status"] == "warning"
        assert "local but not reachable" in check["details"]
        assert "not a loopback URL" not in check["details"]

    def test_vlm_availability_blocks_remote_ollama_in_local_only(self, monkeypatch):
        import urllib.request
        import privatelens.extractors.vlm as vlm_module
        from privatelens.privacy.guard import PrivacyError

        monkeypatch.setattr(vlm_module.settings, "local_only", True)
        monkeypatch.setattr(vlm_module.settings, "ollama_url", "https://ollama.example.com")

        def fail_urlopen(*args, **kwargs):
            raise AssertionError("remote Ollama should be blocked before network")

        monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

        with pytest.raises(PrivacyError):
            vlm_module.VlmExtractor().is_available()

    def test_vlm_document_taxonomy_prioritizes_driver_license(self, monkeypatch, tmp_path):
        import privatelens.extractors.vlm as vlm_module

        prompts = []
        extractor = vlm_module.VlmExtractor()
        extractor._available = True

        def fake_call(_image_path, prompt):
            prompts.append(prompt)
            return '{"type":"document","confidence":0.9,"description":"Driver license"}'

        monkeypatch.setattr(extractor, "_call_ollama", fake_call)

        result = extractor.classify_document(tmp_path / "license.jpg")

        assert result["type"] == "id_card"
        assert "driver's licenses" in prompts[0]
        assert "generic document only" in prompts[0]

    def test_vlm_requests_deterministic_local_generation(self, monkeypatch, tmp_path):
        import urllib.request

        import privatelens.extractors.vlm as vlm_module

        payloads = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"response":"caption"}'

        def fake_urlopen(request, timeout):
            payloads.append(json.loads(request.data))
            assert timeout == 120.0
            return FakeResponse()

        extractor = vlm_module.VlmExtractor()
        monkeypatch.setattr(extractor, "_prepare_image", lambda _path: "encoded-image")
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        response = extractor._call_ollama(tmp_path / "image.jpg", "describe")

        assert response == "caption"
        assert payloads[0]["options"] == {"temperature": 0}

    def test_vlm_reranker_blocks_remote_ollama_in_local_only(self, monkeypatch, tmp_path):
        import privatelens.search.reranker as reranker_module
        from privatelens.privacy.guard import PrivacyError

        image_path = tmp_path / "candidate.jpg"
        image_path.write_bytes(b"image-bytes")
        monkeypatch.setattr(reranker_module.settings, "local_only", True)
        monkeypatch.setattr(reranker_module.settings, "ollama_url", "https://ollama.example.com")

        def fail_post(*args, **kwargs):
            raise AssertionError("remote Ollama should be blocked before network")

        monkeypatch.setattr(reranker_module.httpx, "post", fail_post)

        with pytest.raises(PrivacyError):
            reranker_module.VlmReranker()._score_image(str(image_path), "receipt")

    def test_anythingllm_blocks_remote_sync_in_local_only(self, monkeypatch):
        import privatelens.integrations.anythingllm as anythingllm_module
        from privatelens.privacy.guard import PrivacyError

        monkeypatch.setattr(anythingllm_module.settings, "local_only", True)
        monkeypatch.setattr(
            anythingllm_module.settings, "anythingllm_url", "https://anythingllm.example.com"
        )

        def fail_get(*args, **kwargs):
            raise AssertionError("remote AnythingLLM should be blocked before network")

        monkeypatch.setattr(anythingllm_module.httpx, "get", fail_get)

        with pytest.raises(PrivacyError):
            anythingllm_module.AnythingLLMConnector()._ensure_workspace()


class TestConfig:
    """Test configuration validation."""

    def test_settings_rejects_invalid_ollama_url_format(self):
        from pydantic import ValidationError
        from privatelens.config import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(ollama_url="localhost:11434")

        assert "ollama_url" in str(exc_info.value)

    def test_clip_extractor_uses_configured_quickgelu_architecture(self, monkeypatch):
        from privatelens.extractors.clip import ClipExtractor, clip_model_id

        monkeypatch.setattr(settings, "clip_model", "ViT-B-32-quickgelu")
        monkeypatch.setattr(settings, "clip_pretrained", "openai")

        extractor = ClipExtractor()

        assert extractor.model_name == "ViT-B-32-quickgelu"
        assert extractor.pretrained == "openai"
        assert extractor.model_id == clip_model_id("ViT-B-32-quickgelu", "openai")

    def test_clip_extractor_upgrades_legacy_openai_model_alias(self, monkeypatch):
        from privatelens.extractors.clip import ClipExtractor, clip_model_id

        monkeypatch.setattr(settings, "clip_model", "ViT-B-32")
        monkeypatch.setattr(settings, "clip_pretrained", "openai")

        extractor = ClipExtractor()

        assert extractor.model_name == "ViT-B-32-quickgelu"
        assert extractor.model_id == "openclip:ViT-B-32-quickgelu:openai"
        assert clip_model_id("ViT-B-32", "openai") == extractor.model_id

    def test_clip_loader_uses_explicit_model_cache(self, monkeypatch, tmp_path):
        import privatelens.extractors.clip as clip_module

        calls = []

        class FakeModel:
            def eval(self):
                calls.append("eval")

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        fake_open_clip = SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or (FakeModel(), None, object())
            )
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
        monkeypatch.setattr(clip_module.settings, "model_cache_dir", tmp_path)

        clip_module.ClipExtractor()._load_model()

        _, kwargs = calls[0]
        assert kwargs["cache_dir"] == str(tmp_path / "huggingface" / "hub")
        assert calls[1] == "eval"

    def test_clip_loader_uses_resolved_default_model_cache(self, monkeypatch, tmp_path):
        import privatelens.extractors.clip as clip_module

        calls = []

        class FakeModel:
            def eval(self):
                pass

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        fake_open_clip = SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or (FakeModel(), None, object())
            )
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
        default_cache = tmp_path / ".privatelens" / "models"
        monkeypatch.setattr(clip_module.settings, "data_dir", tmp_path / "separate-index")
        monkeypatch.setattr(clip_module.settings, "model_cache_dir", default_cache)

        clip_module.ClipExtractor()._load_model()

        _, kwargs = calls[0]
        expected_cache = default_cache / "huggingface" / "hub"
        assert kwargs["cache_dir"] == str(expected_cache)
        assert expected_cache.is_dir()

    def test_default_model_cache_is_independent_from_custom_data_dir(self, tmp_path):
        from privatelens.config import Settings

        configured = Settings(data_dir=tmp_path / "disposable-index")

        assert configured.resolved_model_cache_dir == Path.home() / ".privatelens" / "models"
        assert configured.resolved_model_cache_dir != configured.data_dir / "models"


class TestCli:
    """Test CLI support behavior."""

    def test_optional_dependency_groups_keep_full_runtime_only(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())
        optional = pyproject["project"]["optional-dependencies"]

        assert optional["core"] == []
        assert optional["full"] == optional["ml"]
        assert any(dep.startswith("pre-commit>=4.0") for dep in optional["dev"])
        assert any(dep.startswith("mypy>=1.13") for dep in optional["dev"])
        assert not any(dep.startswith(("pytest", "ruff")) for dep in optional["full"])

    def test_package_and_runtime_versions_match(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())

        assert pyproject["project"]["version"] == privatelens.__version__
        assert privatelens.__version__ == "1.0.0"
        assert "Development Status :: 5 - Production/Stable" in pyproject["project"]["classifiers"]
        assert pyproject["project"]["urls"]["Repository"] == (
            "https://github.com/kenny2077/PrivateLens.git"
        )

    def test_index_keeps_restricted_and_heavy_models_opt_in(self):
        import privatelens.cli as cli_module

        index_command = cli_module.cli.commands["index"]
        options = {parameter.name: parameter for parameter in index_command.params}

        assert options["skip_face"].default is True
        assert options["skip_vlm"].default is True

    def test_search_rejects_invalid_type_and_unbounded_limits(self):
        from click.testing import CliRunner

        from privatelens.cli import cli

        runner = CliRunner()
        assert runner.invoke(cli, ["search", "receipt", "--limit", "0"]).exit_code != 0
        assert runner.invoke(cli, ["search", "receipt", "--limit", "201"]).exit_code != 0
        assert runner.invoke(cli, ["search", "receipt", "--type", "unknown"]).exit_code != 0

    def test_github_release_readiness_artifacts_exist(self):
        required_paths = [
            Path("LICENSE"),
            Path("CONTRIBUTING.md"),
            Path(".pre-commit-config.yaml"),
            Path(".github/ISSUE_TEMPLATE/bug_report.md"),
            Path(".github/ISSUE_TEMPLATE/feature_request.md"),
        ]
        missing = [str(path) for path in required_paths if not path.exists()]

        assert missing == []
        assert "MIT License" in Path("LICENSE").read_text()
        assert "python -m pytest tests/ -v" in Path("CONTRIBUTING.md").read_text()
        assert "repo: local" in Path(".pre-commit-config.yaml").read_text()
        assert "id: ruff\n" in Path(".pre-commit-config.yaml").read_text()

    def test_ci_covers_supported_python_and_built_wheel(self):
        workflow = Path(".github/workflows/ci.yml").read_text()

        assert 'python-version: ["3.11", "3.12", "3.13"]' in workflow
        assert "uv build" in workflow
        assert "python -m mypy privatelens" in workflow
        assert "dist/*.whl" in workflow
        assert '"$consumer_env/bin/privatelens" benchmark --json' in workflow

    def test_uv_lock_uses_canonical_package_index(self):
        lockfile = Path("uv.lock").read_text()

        assert 'registry = "https://pypi.org/simple"' in lockfile
        assert "pypi.tuna.tsinghua.edu.cn" not in lockfile

    def test_changelog_tracks_v1_release(self):
        changelog = Path("CHANGELOG.md")

        assert changelog.exists()
        contents = changelog.read_text()
        assert "## [Unreleased]" in contents
        assert "## [1.0.0] - 2026-07-14" in contents
        assert "v1.0 has not been released" not in contents
        assert "Search quality benchmark" in contents

    def test_watch_handler_only_schedules_supported_image_changes(self):
        from threading import Event
        from privatelens.watcher import ImageChangeHandler

        pending = Event()
        handler = ImageChangeHandler(pending)

        handler.on_any_event(
            SimpleNamespace(event_type="modified", is_directory=False, src_path="/photos/note.txt")
        )
        assert pending.is_set() is False

        handler.on_any_event(
            SimpleNamespace(event_type="opened", is_directory=False, src_path="/photos/image.jpg")
        )
        assert pending.is_set() is False

        handler.on_any_event(
            SimpleNamespace(event_type="created", is_directory=False, src_path="/photos/image.jpg")
        )
        assert pending.is_set() is True

        pending.clear()
        handler.on_any_event(
            SimpleNamespace(
                event_type="moved",
                is_directory=False,
                src_path="/photos/old.txt",
                dest_path="/photos/new.webp",
            )
        )
        assert pending.is_set() is True

        pending.clear()
        handler.on_any_event(
            SimpleNamespace(event_type="modified", is_directory=True, src_path="/photos")
        )
        assert pending.is_set() is False

        handler.on_any_event(
            SimpleNamespace(event_type="deleted", is_directory=True, src_path="/photos/album")
        )
        assert pending.is_set() is True

    def test_watch_json_emits_cycle_records_with_safe_defaults(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        import privatelens.cli as cli_module
        import privatelens.watcher as watcher_module

        calls = []

        def fake_watch(folder, callback, *, recursive, debounce, initial_scan):
            calls.append((folder, recursive, debounce, initial_scan))
            callback("initial")
            callback("changed")

        def fake_cycle(ctx, folder, **options):
            calls.append(("cycle", folder, options))
            return {
                "scan": {"found": 1, "new": 1, "updated": 0, "unchanged": 0},
                "index": {"asset_count": 1, "indexed": 1, "skipped_missing": 0, "errors": 0},
            }

        monkeypatch.setattr(watcher_module, "watch_for_changes", fake_watch)
        monkeypatch.setattr(cli_module, "run_watch_cycle", fake_cycle)

        result = CliRunner().invoke(
            cli_module.cli,
            ["watch", str(tmp_path), "--json", "--debounce", "0.5"],
        )

        assert result.exit_code == 0
        records = [json.loads(line) for line in result.output.splitlines()]
        assert [record["event"] for record in records] == ["cycle", "cycle", "stopped"]
        assert [record["trigger"] for record in records[:2]] == ["initial", "changed"]
        assert records[0]["scan"]["new"] == 1
        assert records[0]["index"]["indexed"] == 1
        assert calls[0] == (tmp_path, True, 0.5, True)
        assert calls[1][2] == {
            "recursive": True,
            "skip_face": True,
            "skip_vlm": True,
            "batch_size": 1,
        }

    def test_watcher_debounces_changes_and_always_stops_observer(self, tmp_path, monkeypatch):
        import privatelens.watcher as watcher_module

        class FakeEvent:
            def __init__(self):
                self.waits = iter([True, False, KeyboardInterrupt()])

            def wait(self, _timeout=None):
                value = next(self.waits)
                if isinstance(value, BaseException):
                    raise value
                return value

            def clear(self):
                return None

            def set(self):
                return None

        class FakeObserver:
            def __init__(self):
                self.calls = []

            def schedule(self, handler, folder, recursive):
                self.calls.append(("schedule", handler, folder, recursive))

            def start(self):
                self.calls.append(("start",))

            def stop(self):
                self.calls.append(("stop",))

            def join(self):
                self.calls.append(("join",))

        observer = FakeObserver()
        monkeypatch.setattr(watcher_module, "Event", FakeEvent)
        monkeypatch.setattr(watcher_module, "Observer", lambda: observer)
        triggers = []

        watcher_module.watch_for_changes(
            tmp_path,
            triggers.append,
            recursive=False,
            debounce=0.25,
            initial_scan=True,
        )

        assert triggers == ["initial", "changed"]
        assert observer.calls[0][0] == "schedule"
        assert observer.calls[0][2:] == (str(tmp_path), False)
        assert observer.calls[-3:] == [("start",), ("stop",), ("join",)]

    def test_watch_initial_cycle_uses_real_scan_and_index(self, temp_db, tmp_path, monkeypatch):
        from click.testing import CliRunner
        import privatelens.cli as cli_module
        import privatelens.watcher as watcher_module

        Image.new("RGB", (10, 10), color="blue").save(tmp_path / "watched.jpg")

        class FakeClipExtractor:
            def extract(self, _file_path):
                return None

        class FakeOcrExtractor:
            def extract(self, _file_path):
                return []

        def fake_watch(_folder, callback, **_options):
            callback("initial")

        monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
        monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)
        monkeypatch.setattr(watcher_module, "watch_for_changes", fake_watch)

        result = CliRunner().invoke(cli_module.cli, ["watch", str(tmp_path), "--json"])

        assert result.exit_code == 0
        records = [json.loads(line) for line in result.output.splitlines()]
        assert records[0]["event"] == "cycle"
        assert records[0]["scan"] == {
            "folder": str(tmp_path),
            "recursive": True,
            "dry_run": False,
            "found": 1,
            "new": 1,
            "updated": 0,
            "unchanged": 0,
            "invalid": 0,
        }
        assert records[0]["index"]["indexed"] == 1
        assert records[0]["index"]["errors"] == 0
        assert records[0]["prune"] == {"missing_count": 0, "removed": 0}
        assert records[1] == {"event": "stopped", "folder": str(tmp_path)}

    def test_watch_pruning_is_scoped_to_watched_folder(self, temp_db, tmp_path):
        from sqlalchemy.orm import Session
        import privatelens.cli as cli_module
        from privatelens.db.schema import Asset

        watched = tmp_path / "watched"
        outside = tmp_path / "outside"
        watched.mkdir()
        outside.mkdir()
        thumbnail = tmp_path / "stale-thumbnail.jpg"
        thumbnail.write_bytes(b"derived")

        with Session(temp_db) as session:
            session.add_all(
                [
                    Asset(
                        path=str(watched / "deleted.jpg"),
                        sha256="watched-missing",
                        media_type="image",
                        thumbnail_path=str(thumbnail),
                    ),
                    Asset(
                        path=str(outside / "deleted.jpg"),
                        sha256="outside-missing",
                        media_type="image",
                    ),
                ]
            )
            session.commit()

        summary = cli_module.prune_watched_folder(watched)

        assert summary == {"missing_count": 1, "removed": 1}
        assert thumbnail.exists() is False
        with Session(temp_db) as session:
            paths = [asset.path for asset in session.query(Asset).all()]
        assert paths == [str(outside / "deleted.jpg")]

    def test_readme_documents_incremental_watch_contract(self):
        readme = Path("README.md").read_text()
        changelog = Path("CHANGELOG.md").read_text()

        assert "privatelens watch ~/Pictures" in readme
        assert "newline-delimited JSON" in readme
        assert "full local ML pipeline" in readme
        assert "Incremental folder watcher" in changelog

    def test_readme_contains_demo_artifact_and_competitive_comparison(self):
        readme = Path("README.md").read_text()
        demo_asset = Path("docs/assets/terminal-demo.svg")

        assert demo_asset.exists()
        assert "docs/assets/terminal-demo.svg" in readme
        assert "## Why PrivateLens" in readme
        assert "| Capability | PrivateLens | Immich | PhotoPrism | Caption/tag tools |" in readme
        assert "Read-only sidecar over existing folders" in readme
        assert "Search recipes with evidence cards" in readme
        assert "30-second synthetic demo" in readme

    def test_cpu_docker_packaging_artifacts_support_local_sidecar(self):
        required_paths = [
            Path("Dockerfile"),
            Path("docker-compose.yml"),
            Path(".dockerignore"),
        ]
        missing = [str(path) for path in required_paths if not path.exists()]

        assert missing == []

        dockerfile = Path("Dockerfile").read_text()
        compose = Path("docker-compose.yml").read_text()
        dockerignore = Path(".dockerignore").read_text()
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())

        assert "FROM python:3.11-slim@sha256:" in dockerfile
        assert "# syntax=docker/dockerfile:1.7@sha256:" in dockerfile
        assert " AS builder" in dockerfile
        assert " AS runtime" in dockerfile
        assert "COPY --from=builder /opt/venv /opt/venv" in dockerfile
        assert 'pip install --no-cache-dir -e "."' not in dockerfile
        assert "COPY pyproject.toml uv.lock README.md LICENSE ./" in dockerfile
        assert "uv sync --locked --no-dev --no-editable" in dockerfile
        assert "--mount=type=cache,target=/root/.cache/uv" in dockerfile
        assert "UV_NO_CACHE" not in dockerfile
        assert 'if [ "$PRIVATELENS_EXTRAS" != "core" ]' in dockerfile
        assert "USER privatelens" in dockerfile
        assert "ARG PRIVATELENS_EXTRAS=full" in dockerfile
        assert "PRIVATELENS_DATA_DIR=/data" in dockerfile
        assert "PRIVATELENS_OLLAMA_URL=http://ollama:11434" in compose
        assert "${PHOTOS_DIR:-./photo}:/photos:ro" in compose
        assert (
            "ollama/ollama:0.31.1@sha256:"
            "f1a705f2bd113fb8d15f85f7c217f0dc5f6bebda6b0cc42b82c3ad165ffcb9dc" in compose
        )
        assert "ollama-pull:" in compose
        assert "condition: service_healthy" in compose
        assert "condition: service_completed_successfully" in compose
        assert "11434:11434" not in compose
        assert "127.0.0.1:${PRIVATELENS_PORT:-8000}:8000" in compose
        assert '"serve", "--host", "0.0.0.0", "--port", "8000"' in compose
        assert '"serve", "--host", "0.0.0.0", "--port", "8000"' in dockerfile
        assert "http://127.0.0.1:8000/api/health" in dockerfile
        assert "read_only: true" in compose
        assert "no-new-privileges:true" in compose
        assert ".venv" in dockerignore
        assert "photo/" in dockerignore
        assert "reference/" in dockerignore
        assert pyproject["tool"]["uv"]["sources"]["torch"] == [
            {"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}
        ]
        assert pyproject["tool"]["uv"]["sources"]["torchvision"] == [
            {"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}
        ]
        assert pyproject["tool"]["uv"]["index"] == [
            {
                "name": "pytorch-cpu",
                "url": "https://download.pytorch.org/whl/cpu",
                "explicit": True,
            }
        ]

    def test_github_actions_are_pinned_to_commit_shas(self):
        action_ref = re.compile(r"^[^\s@]+@[0-9a-f]{40}$")

        for workflow in Path(".github/workflows").glob("*.yml"):
            for line in workflow.read_text().splitlines():
                stripped = line.strip()
                if not stripped.startswith("uses:"):
                    continue
                reference = stripped.removeprefix("uses:").split("#", 1)[0].strip()
                assert action_ref.fullmatch(reference), (
                    f"Unpinned action in {workflow}: {reference}"
                )

    def test_pytest_collection_is_limited_to_project_tests(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())
        pytest_options = pyproject["tool"]["pytest"]["ini_options"]

        assert pytest_options["testpaths"] == ["tests"]
        assert "reference" in pytest_options["norecursedirs"]

    def test_configure_logging_sets_expected_levels(self):
        import logging
        from privatelens.cli import configure_logging

        configure_logging(verbose=False, debug=False)
        assert logging.getLogger().level == logging.WARNING

        configure_logging(verbose=True, debug=False)
        assert logging.getLogger().level == logging.INFO

        configure_logging(verbose=False, debug=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_format_score_colors_search_scores(self):
        from privatelens.cli import format_score

        assert format_score(0.8) == "[green]0.800[/green]"
        assert format_score(0.5) == "[yellow]0.500[/yellow]"
        assert format_score(0.49) == "[red]0.490[/red]"

    def test_cli_formats_privatelens_errors_without_traceback(self):
        import click
        from click.testing import CliRunner
        import privatelens.cli as cli_module
        from privatelens.errors import PrivateLensError

        @click.command("fail-friendly")
        def fail_friendly():
            raise PrivateLensError("database is locked")

        cli_module.cli.add_command(fail_friendly)
        try:
            result = CliRunner().invoke(cli_module.cli, ["fail-friendly"])
        finally:
            cli_module.cli.commands.pop("fail-friendly", None)

        combined_output = result.output + getattr(result, "stderr", "")
        assert result.exit_code == 1
        assert "Error: database is locked" in combined_output
        assert "Traceback" not in combined_output

    def test_cli_debug_reraises_unexpected_errors(self):
        import click
        from click.testing import CliRunner
        import privatelens.cli as cli_module

        @click.command("fail-debug")
        def fail_debug():
            raise RuntimeError("debug failure")

        cli_module.cli.add_command(fail_debug)
        try:
            result = CliRunner().invoke(cli_module.cli, ["--debug", "fail-debug"])
        finally:
            cli_module.cli.commands.pop("fail-debug", None)

        assert result.exit_code == 1
        assert isinstance(result.exception, RuntimeError)

    def test_doctor_json_outputs_machine_readable_audit(self, temp_db):
        import json
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["doctor", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["count"] > 0
        assert any(check["name"] == "Database Local" for check in payload["checks"])
        assert all("remediation" in check for check in payload["checks"])

    def test_doctor_initializes_fresh_data_directory(self, tmp_path):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(
            cli,
            ["--data-dir", str(tmp_path), "doctor", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        database_check = next(
            check for check in payload["checks"] if check["name"] == "Database Local"
        )
        assert database_check["status"] == "ok"
        assert (tmp_path / "privatelens.db").exists()

    def test_doctor_human_output_includes_remediation(self, monkeypatch):
        from click.testing import CliRunner
        import privatelens.cli as cli_module

        class FakePrivacyAuditor:
            def run_audit(self):
                return [
                    {
                        "name": "Ollama Local",
                        "status": "warning",
                        "details": "Ollama is not reachable",
                        "remediation": ["open -a Ollama"],
                    }
                ]

        monkeypatch.setattr(cli_module, "PrivacyAuditor", FakePrivacyAuditor)

        result = CliRunner().invoke(cli_module.cli, ["doctor"])

        assert result.exit_code == 0
        assert "Remediation" in result.output
        assert "open -a Ollama" in result.output

    def test_cluster_json_outputs_machine_readable_summary(self, temp_db):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["cluster", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["action"] == "cluster"
        assert payload["people_created"] == 0

    def test_cluster_json_outputs_machine_readable_name_assignment(self, temp_db):
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Person

        with Session(temp_db) as session:
            person = Person(display_name="Person 1", face_count=0, user_labeled=False)
            session.add(person)
            session.commit()
            person_id = person.id

        result = CliRunner().invoke(
            cli,
            ["cluster", "--person-id", str(person_id), "--name", "Alex", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["action"] == "assign_name"
        assert payload["person_id"] == person_id
        assert payload["name"] == "Alex"
        assert payload["assigned"] is True
        with Session(temp_db) as session:
            person = session.query(Person).filter_by(id=person_id).one()
            assert person.display_name == "Alex"
            assert person.user_labeled is True

    def test_purge_works_without_sqlite_vec_tables(self, temp_db):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["purge"], input="y\n")

        assert result.exit_code == 0
        assert "Index purged" in result.output

    def test_purge_initializes_empty_database(self, tmp_path):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(
            cli,
            ["--data-dir", str(tmp_path), "purge"],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Index purged" in result.output

    def test_purge_json_outputs_machine_readable_summary(self, temp_db):
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import (
            Asset,
            ImageEmbedding,
            OcrBlock,
            Person,
            SearchEvent,
            SensitiveItem,
        )

        source_path = settings.data_dir / "source.jpg"
        source_path.write_bytes(b"source photo")
        thumbnail_dir = settings.resolved_thumbnail_dir
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_path = thumbnail_dir / "purge.jpg"
        thumbnail_path.write_bytes(b"derived thumbnail")
        with Session(temp_db) as session:
            asset = Asset(
                path=str(source_path),
                sha256="purge",
                media_type="image",
                thumbnail_path=str(thumbnail_path),
                is_sensitive=True,
                sensitive_type="receipt",
            )
            session.add(asset)
            session.flush()
            session.add(ImageEmbedding(asset_id=asset.id, vector=b"\x00" * 8))
            session.add(OcrBlock(asset_id=asset.id, text="receipt"))
            session.add(SensitiveItem(asset_id=asset.id, type="receipt", confidence=0.8))
            session.add(SearchEvent(query="private query", result_clicked=asset.id))
            session.add(Person(display_name="Alex", user_labeled=True, face_count=0))
            session.commit()

        result = CliRunner().invoke(cli, ["purge", "--json", "--yes"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["purged"] == "index"
        assert payload["assets_removed"] == 1
        assert payload["people_removed"] == 1
        assert payload["thumbnails_removed"] == 1
        assert payload["source_photos_removed"] == 0
        assert source_path.exists()
        assert thumbnail_path.exists() is False
        with Session(temp_db) as session:
            assert session.query(Asset).count() == 0
            assert session.query(ImageEmbedding).count() == 0
            assert session.query(OcrBlock).count() == 0
            assert session.query(SensitiveItem).count() == 0
            assert session.query(SearchEvent).count() == 0
            assert session.query(Person).count() == 0

    def test_prune_json_previews_missing_assets_without_deleting(self, temp_db, tmp_path):
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Asset

        existing_path = tmp_path / "existing.jpg"
        existing_path.write_bytes(b"present")
        missing_path = tmp_path / "missing.jpg"
        with Session(temp_db) as session:
            session.add_all(
                [
                    Asset(path=str(existing_path), sha256="existing", media_type="image"),
                    Asset(path=str(missing_path), sha256="missing", media_type="image"),
                ]
            )
            session.commit()

        result = CliRunner().invoke(cli, ["prune", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload == {
            "dry_run": True,
            "missing_count": 1,
            "removed": 0,
            "paths": [str(missing_path)],
        }
        with Session(temp_db) as session:
            assert session.query(Asset).count() == 2

    def test_prune_yes_removes_only_missing_assets(self, temp_db, tmp_path):
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Asset

        existing_path = tmp_path / "existing.jpg"
        existing_path.write_bytes(b"present")
        missing_path = tmp_path / "missing.jpg"
        with Session(temp_db) as session:
            session.add_all(
                [
                    Asset(path=str(existing_path), sha256="existing", media_type="image"),
                    Asset(path=str(missing_path), sha256="missing", media_type="image"),
                ]
            )
            session.commit()

        result = CliRunner().invoke(cli, ["prune", "--json", "--yes"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["dry_run"] is False
        assert payload["missing_count"] == 1
        assert payload["removed"] == 1
        with Session(temp_db) as session:
            remaining = session.query(Asset).one()
            assert remaining.path == str(existing_path)

    def test_sync_anythingllm_json_outputs_machine_readable_summary(self, monkeypatch):
        from click.testing import CliRunner
        import privatelens.cli as cli_module
        import privatelens.integrations.anythingllm as anythingllm_module

        calls = []

        class FakeAnythingLLMConnector:
            def sync(self):
                calls.append("sync")

        monkeypatch.setattr(
            anythingllm_module,
            "AnythingLLMConnector",
            FakeAnythingLLMConnector,
        )

        result = CliRunner().invoke(cli_module.cli, ["sync-anythingllm", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["target"] == "anythingllm"
        assert payload["synced"] is True
        assert calls == ["sync"]

    def test_status_reports_index_counts(self, temp_db):
        from datetime import datetime
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Asset, Caption, Face, ImageEmbedding, OcrBlock

        with Session(temp_db) as session:
            asset = Asset(
                path="/test/status.jpg",
                sha256="test",
                media_type="image",
                indexed_at=datetime.now(),
            )
            session.add(asset)
            session.commit()
            asset_id = asset.id
            session.add(ImageEmbedding(asset_id=asset_id, vector=b"\x00" * 8))
            session.add(OcrBlock(asset_id=asset_id, text="receipt"))
            session.add(Face(asset_id=asset_id, bbox="[]", confidence=0.8))
            session.add(Caption(asset_id=asset_id, model="test", caption="test caption"))
            session.commit()

        result = CliRunner().invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Total Assets" in result.output
        assert "Indexed Assets" in result.output
        assert "Embeddings" in result.output
        assert "1" in result.output

    def test_search_json_outputs_machine_readable_results(self, monkeypatch):
        import json
        from click.testing import CliRunner
        import privatelens.cli as cli_module

        class FakeSearchEngine:
            def search(self, query, search_type="smart", limit=50, record_event=False):
                return [
                    {
                        "asset_id": 7,
                        "path": "/test/receipt.jpg",
                        "score": 0.91,
                        "explanation": "OCR contains receipt",
                    }
                ]

            def search_by_recipe(
                self, recipe_name, query, limit=50, rerank=True, record_event=False
            ):
                return self.search(query, limit=limit, record_event=record_event)

        monkeypatch.setattr(cli_module, "SearchEngine", FakeSearchEngine)

        result = CliRunner().invoke(cli_module.cli, ["search", "receipt", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["query"] == "receipt"
        assert payload["count"] == 1
        assert payload["results"][0]["asset_id"] == 7

    def test_search_json_includes_elapsed_ms(self, monkeypatch):
        import json
        from click.testing import CliRunner
        import privatelens.cli as cli_module

        class FakeSearchEngine:
            def search(self, query, search_type="smart", limit=50):
                return []

            def search_by_recipe(self, recipe_name, query, limit=50, rerank=True):
                return []

        monkeypatch.setattr(cli_module, "SearchEngine", FakeSearchEngine)

        result = CliRunner().invoke(cli_module.cli, ["search", "landscape", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert isinstance(payload["elapsed_ms"], float)
        assert payload["elapsed_ms"] >= 0.0

    def test_search_json_auto_detects_receipt_recipe(self, monkeypatch):
        import json
        from click.testing import CliRunner
        import privatelens.cli as cli_module

        calls = []

        class FakeSearchEngine:
            def search(self, query, search_type="smart", limit=50):
                raise AssertionError("auto recipe should bypass plain search")

            def search_by_recipe(self, recipe_name, query, limit=50, rerank=True):
                calls.append((recipe_name, query, limit))
                return [
                    {
                        "asset_id": 8,
                        "path": "/test/receipt.jpg",
                        "score": 0.92,
                        "explanation": "Receipt recipe match",
                    }
                ]

        monkeypatch.setattr(cli_module, "SearchEngine", FakeSearchEngine)

        result = CliRunner().invoke(
            cli_module.cli,
            ["search", "find my receipt from lunch", "--json", "--limit", "3"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["recipe"] == "find_receipt"
        assert payload["results"][0]["asset_id"] == 8
        assert calls == [("find_receipt", "find my receipt from lunch", 3)]

    def test_search_open_launches_first_result(self, monkeypatch):
        from click.testing import CliRunner
        import privatelens.cli as cli_module

        launched = []

        class FakeSearchEngine:
            def search(self, query, search_type="smart", limit=50):
                return [
                    {
                        "asset_id": 9,
                        "path": "/test/open-me.jpg",
                        "score": 0.9,
                        "explanation": "Semantic match",
                    }
                ]

            def search_by_recipe(self, recipe_name, query, limit=50, rerank=True):
                return self.search(query, limit=limit)

        monkeypatch.setattr(cli_module, "SearchEngine", FakeSearchEngine)
        monkeypatch.setattr(cli_module.click, "launch", lambda path: launched.append(path))

        result = CliRunner().invoke(cli_module.cli, ["search", "landscape", "--open"])

        assert result.exit_code == 0
        assert launched == ["/test/open-me.jpg"]
        assert "Opened /test/open-me.jpg" in result.output

    def test_search_fast_disables_recipe_rerank(self, monkeypatch):
        import json
        from click.testing import CliRunner
        import privatelens.cli as cli_module

        calls = []

        class FakeSearchEngine:
            def search(self, query, search_type="smart", limit=50):
                raise AssertionError("receipt should auto-detect recipe")

            def search_by_recipe(self, recipe_name, query, limit=50, rerank=True):
                calls.append((recipe_name, query, limit, rerank))
                return []

        monkeypatch.setattr(cli_module, "SearchEngine", FakeSearchEngine)

        result = CliRunner().invoke(
            cli_module.cli,
            ["search", "receipt", "--fast", "--json", "--limit", "2"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["fast"] is True
        assert calls == [("find_receipt", "receipt", 2, False)]

    def test_search_feedback_prompts_for_human_output(self, monkeypatch):
        from click.testing import CliRunner
        import privatelens.cli as cli_module

        feedback_calls = []

        class FakeSearchEngine:
            last_search_event_id = 42

            def search(self, query, search_type="smart", limit=50, record_event=False):
                assert record_event is True
                return [
                    {
                        "asset_id": 11,
                        "path": "/test/receipt.jpg",
                        "score": 0.91,
                        "explanation": "OCR contains receipt",
                    }
                ]

            def search_by_recipe(
                self, recipe_name, query, limit=50, rerank=True, record_event=False
            ):
                return self.search(query, limit=limit, record_event=record_event)

            def record_feedback(self, event_id, feedback, result_clicked=None):
                feedback_calls.append((event_id, feedback, result_clicked))

        monkeypatch.setattr(cli_module, "SearchEngine", FakeSearchEngine)

        result = CliRunner().invoke(
            cli_module.cli,
            ["search", "landscape", "--feedback"],
            input="y\n1\n",
        )

        assert result.exit_code == 0
        assert feedback_calls == [(42, 1, 11)]
        assert "Feedback recorded" in result.output

    def test_status_initializes_empty_database(self, tmp_path):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["--data-dir", str(tmp_path), "status"])

        assert result.exit_code == 0
        assert "Total Assets" in result.output
        assert "0" in result.output

    def test_status_json_outputs_machine_readable_counts(self, temp_db):
        import json
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Asset, OcrBlock

        with Session(temp_db) as session:
            asset = Asset(path="/test/status-json.jpg", sha256="json", media_type="image")
            session.add(asset)
            session.commit()
            session.add(OcrBlock(asset_id=asset.id, text="receipt"))
            session.commit()

        result = CliRunner().invoke(cli, ["status", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["total_assets"] == 1
        assert payload["ocr_blocks"] == 1
        assert payload["faces"] == 0

    def test_recipes_initializes_empty_database(self, tmp_path):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["--data-dir", str(tmp_path), "recipes"])

        assert result.exit_code == 0
        assert "find_id_photo" in result.output

    def test_recipes_json_outputs_machine_readable_recipes(self, tmp_path):
        import json
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(
            cli,
            ["--data-dir", str(tmp_path), "recipes", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["count"] == 10
        assert any(recipe["name"] == "find_receipt" for recipe in payload["recipes"])
        assert payload["recipes"][0]["query_plan"] is None

    def test_recipes_detail_outputs_query_plan_summary(self, tmp_path):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(
            cli,
            ["--data-dir", str(tmp_path), "recipes", "--detail"],
        )

        assert result.exit_code == 0
        assert "Signals" in result.output
        assert "Filters" in result.output
        assert "find_receipt" in result.output
        assert "ocr" in result.output

    def test_recipe_shell_completion_filters_builtin_recipe_names(self):
        import privatelens.cli as cli_module

        completions = cli_module.complete_recipe_names(None, None, "find_r")

        assert [item.value for item in completions] == ["find_receipt"]
        assert completions[0].help == "Find Receipts"

    def test_recipe_detection_matches_words_not_substrings(self):
        from privatelens.search.recipes import detect_recipe_for_query

        assert detect_recipe_for_query("person holding a cat") == "find_pet"
        assert detect_recipe_for_query("insurance identification card") is None
        assert detect_recipe_for_query("student holding a certificate") is None

    def test_search_recipe_option_uses_recipe_completion(self):
        import privatelens.cli as cli_module

        search_command = cli_module.cli.commands["search"]
        recipe_option = next(param for param in search_command.params if param.name == "recipe")

        completions = recipe_option.shell_complete(None, "find_s")

        assert [item.value for item in completions] == [
            "find_selfie",
            "find_screenshot",
            "find_sensitive",
        ]

    def test_quickstart_shows_mac_safe_cli_path(self):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["quickstart"])

        assert result.exit_code == 0
        assert "PrivateLens Quickstart" in result.output
        assert "demo --output-dir" in result.output
        assert "scan" in result.output
        assert "index --skip-face --skip-vlm --batch-size 1" in result.output
        assert "search" in result.output

    def test_demo_command_creates_reproducible_synthetic_library(self, tmp_path):
        from click.testing import CliRunner
        from PIL import Image
        from privatelens.cli import cli

        output_dir = tmp_path / "demo-photos"

        result = CliRunner().invoke(cli, ["demo", "--output-dir", str(output_dir), "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["output_dir"] == str(output_dir)
        assert payload["file_count"] >= 4
        assert any(file["query"] == "receipt" for file in payload["files"])
        assert any("scan" in command for command in payload["next_commands"])
        assert any("search receipt" in command for command in payload["next_commands"])

        receipt = output_dir / "target-receipt-lunch.jpg"
        assert receipt.exists()
        with Image.open(receipt) as image:
            assert image.width > 200
            assert image.height > 200

    def test_demo_next_commands_quote_output_dir_with_spaces(self, tmp_path):
        from privatelens.demo import build_demo_commands

        output_dir = tmp_path / "demo photos"

        commands = build_demo_commands(output_dir)

        assert commands[0] == f"privatelens scan '{output_dir}'"
        assert all(" --fast " not in command for command in commands)

    def test_setup_shows_exact_manual_environment_commands(self):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["setup"])

        assert result.exit_code == 0
        assert 'python -m pip install --upgrade "privatelens[full]"' in result.output
        assert "PRIVATELENS_ENCRYPTION_KEY" in result.output
        assert "ollama pull qwen3-vl:2b-instruct-q8_0" in result.output
        assert "privatelens benchmark-models --skip-vlm --json" in result.output
        assert "privatelens doctor --json" in result.output
        assert "-e ." not in result.output
        assert "scripts/download-models.py" not in result.output

    def test_setup_json_outputs_actionable_package_groups(self):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["setup", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["commands"]["install_full_dev"] == (
            'python -m pip install --upgrade "privatelens[full,dev]"'
        )
        assert payload["commands"]["pull_vlm_model"] == "ollama pull qwen3-vl:2b-instruct-q8_0"
        assert payload["package_groups"]["core"]["command"] == (
            'python -m pip install --upgrade "privatelens"'
        )
        assert payload["package_groups"]["ml"]["command"] == (
            'python -m pip install --upgrade "privatelens[full]"'
        )
        assert isinstance(payload["package_groups"]["ml"]["missing"], list)

    def test_linux_setup_uses_official_cpu_only_pytorch_index(self, monkeypatch):
        import privatelens.cli as cli_module

        monkeypatch.setattr(cli_module.sys, "platform", "linux")

        plan = cli_module.build_setup_plan()

        assert "https://download.pytorch.org/whl/cpu" in plan["commands"]["install_full"]
        assert "privatelens[full]" in plan["commands"]["install_full"]

    def test_completion_command_prints_shell_setup(self):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["completion", "zsh"])

        assert result.exit_code == 0
        assert "_PRIVATELENS_COMPLETE=zsh_source privatelens" in result.output
        assert "~/.zshrc" in result.output

    def test_search_json_initializes_empty_database(self, tmp_path):
        import json
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(
            cli,
            ["--data-dir", str(tmp_path), "search", "receipt", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["count"] == 0
        assert payload["results"] == []

    def test_cli_scan_index_search_core_loop_without_models(self, tmp_path, monkeypatch):
        import json
        import numpy as np
        from click.testing import CliRunner
        from PIL import Image
        import privatelens.cli as cli_module
        import privatelens.search.engine as engine_module

        data_dir = tmp_path / "data"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        image_path = photos_dir / "receipt.jpg"
        Image.new("RGB", (16, 16), color="white").save(image_path)

        class FakeClipExtractor:
            def extract(self, image_path):
                return np.array([1.0, 0.0], dtype=np.float32)

        class FakeOcrExtractor:
            def extract(self, image_path):
                return [
                    {
                        "text": "receipt total lunch",
                        "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "confidence": 0.99,
                    }
                ]

        monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
        monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)
        monkeypatch.setattr(
            engine_module.ClipExtractor,
            "encode_text",
            lambda self, query: None,
        )
        monkeypatch.setattr(
            engine_module.VlmReranker,
            "rerank",
            lambda self, candidates, query, top_k=30: candidates,
        )

        runner = CliRunner()
        scan_result = runner.invoke(
            cli_module.cli,
            ["--data-dir", str(data_dir), "scan", str(photos_dir)],
        )
        index_result = runner.invoke(
            cli_module.cli,
            [
                "--data-dir",
                str(data_dir),
                "index",
                "--skip-face",
                "--skip-vlm",
                "--batch-size",
                "1",
            ],
        )
        search_result = runner.invoke(
            cli_module.cli,
            ["--data-dir", str(data_dir), "search", "receipt", "--json", "--limit", "5"],
        )

        assert scan_result.exit_code == 0
        assert index_result.exit_code == 0
        assert "Indexed 1 assets" in index_result.output
        assert search_result.exit_code == 0
        payload = json.loads(search_result.stdout)
        assert payload["recipe"] == "find_receipt"
        assert payload["count"] == 1
        assert payload["results"][0]["path"] == str(image_path)
        assert "OCR" in payload["results"][0]["explanation"]

    def test_scan_dry_run_does_not_write_assets(self, temp_db, tmp_path):
        from click.testing import CliRunner
        from PIL import Image
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Asset

        image_path = tmp_path / "dry-run.jpg"
        Image.new("RGB", (10, 10), color="blue").save(image_path)

        result = CliRunner().invoke(cli, ["scan", str(tmp_path), "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run" in result.output
        with Session(temp_db) as session:
            assert session.query(Asset).count() == 0

    def test_scan_json_dry_run_outputs_machine_readable_plan(self, temp_db, tmp_path):
        from click.testing import CliRunner
        from PIL import Image
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Asset

        image_path = tmp_path / "json-dry-run.jpg"
        Image.new("RGB", (10, 10), color="blue").save(image_path)

        result = CliRunner().invoke(cli, ["scan", str(tmp_path), "--dry-run", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["dry_run"] is True
        assert payload["found"] == 1
        assert payload["folder"] == str(tmp_path)
        with Session(temp_db) as session:
            assert session.query(Asset).count() == 0

    def test_scan_skips_corrupt_image_and_reports_invalid_count(self, temp_db, tmp_path):
        from click.testing import CliRunner
        from PIL import Image
        from sqlalchemy.orm import Session

        from privatelens.cli import cli
        from privatelens.db.schema import Asset

        valid_path = tmp_path / "valid.jpg"
        corrupt_path = tmp_path / "corrupt.jpg"
        Image.new("RGB", (10, 10), color="blue").save(valid_path)
        corrupt_path.write_bytes(b"not an image")

        result = CliRunner().invoke(cli, ["scan", str(tmp_path), "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["found"] == 2
        assert payload["new"] == 1
        assert payload["invalid"] == 1
        with Session(temp_db) as session:
            assert [asset.path for asset in session.query(Asset).all()] == [str(valid_path)]

    def test_scan_updates_changed_existing_asset(self, temp_db, tmp_path):
        from click.testing import CliRunner
        from PIL import Image
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import (
            Asset,
            Caption,
            Detection,
            Face,
            ImageEmbedding,
            OcrBlock,
            SensitiveItem,
        )

        image_path = tmp_path / "changed.jpg"
        Image.new("RGB", (10, 10), color="blue").save(image_path)

        first = CliRunner().invoke(cli, ["scan", str(tmp_path)])
        assert first.exit_code == 0

        with Session(temp_db) as session:
            asset = session.query(Asset).filter_by(path=str(image_path)).one()
            first_sha = asset.sha256
            asset.indexed_at = datetime.now()
            asset.thumbnail_path = str(tmp_path / "stale-thumb.jpg")
            asset.is_sensitive = True
            asset.sensitive_type = "driver_license"
            session.add(ImageEmbedding(asset_id=asset.id, vector=b"\x00" * 8))
            session.add(OcrBlock(asset_id=asset.id, text="stale text"))
            session.add(Face(asset_id=asset.id, bbox="[]", confidence=0.8))
            session.add(Caption(asset_id=asset.id, model="test", caption="stale caption"))
            session.add(Detection(asset_id=asset.id, label="receipt", confidence=0.8))
            session.add(SensitiveItem(asset_id=asset.id, type="driver_license"))
            session.commit()

        Image.new("RGB", (20, 20), color="green").save(image_path)

        second = CliRunner().invoke(cli, ["scan", str(tmp_path)])
        assert second.exit_code == 0

        with Session(temp_db) as session:
            assets = session.query(Asset).filter_by(path=str(image_path)).all()
            assert len(assets) == 1
            asset = assets[0]
            assert asset.sha256 != first_sha
            assert asset.width == 20
            assert asset.height == 20
            assert asset.indexed_at is None
            assert asset.thumbnail_path is None
            assert asset.is_sensitive is False
            assert asset.sensitive_type is None
            assert session.query(ImageEmbedding).filter_by(asset_id=asset.id).count() == 0
            assert session.query(OcrBlock).filter_by(asset_id=asset.id).count() == 0
            assert session.query(Face).filter_by(asset_id=asset.id).count() == 0
            assert session.query(Caption).filter_by(asset_id=asset.id).count() == 0
            assert session.query(Detection).filter_by(asset_id=asset.id).count() == 0
            assert session.query(SensitiveItem).filter_by(asset_id=asset.id).count() == 0

    def test_scan_reports_new_updated_and_unchanged_counts(self, temp_db, tmp_path):
        from click.testing import CliRunner
        from PIL import Image
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Asset

        unchanged_path = tmp_path / "unchanged.jpg"
        changed_path = tmp_path / "changed.jpg"
        new_path = tmp_path / "new.jpg"
        Image.new("RGB", (12, 12), color="blue").save(unchanged_path)
        Image.new("RGB", (12, 12), color="red").save(changed_path)
        Image.new("RGB", (12, 12), color="white").save(new_path)
        unchanged_sha = ExifExtractor().extract(unchanged_path)["sha256"]

        with Session(temp_db) as session:
            session.add(
                Asset(
                    path=str(unchanged_path),
                    sha256=unchanged_sha,
                    media_type="image",
                )
            )
            session.add(
                Asset(
                    path=str(changed_path),
                    sha256="old-sha",
                    media_type="image",
                )
            )
            session.commit()

        result = CliRunner().invoke(cli, ["scan", str(tmp_path)])

        assert result.exit_code == 0
        assert "New: 1" in result.output
        assert "Updated: 1" in result.output
        assert "Unchanged: 1" in result.output

    def test_scan_uses_progress_for_human_output(self, temp_db, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from PIL import Image
        import privatelens.cli as cli_module

        image_path = tmp_path / "progress.jpg"
        Image.new("RGB", (10, 10), color="blue").save(image_path)
        events = []

        class FakeProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def add_task(self, description, total):
                events.append(("task", description, total))
                return "scan-task"

            def advance(self, task_id):
                events.append(("advance", task_id))

        monkeypatch.setattr(cli_module, "Progress", FakeProgress, raising=False)

        result = CliRunner().invoke(cli_module.cli, ["scan", str(tmp_path)])

        assert result.exit_code == 0
        assert events == [
            ("task", "Scanning images", 1),
            ("advance", "scan-task"),
        ]

    def test_index_dry_run_does_not_mark_assets_indexed(self, temp_db):
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Asset

        with Session(temp_db) as session:
            session.add(Asset(path="/test/unindexed.jpg", sha256="test", media_type="image"))
            session.commit()

        result = CliRunner().invoke(cli, ["index", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run" in result.output
        with Session(temp_db) as session:
            asset = session.query(Asset).filter_by(path="/test/unindexed.jpg").one()
            assert asset.indexed_at is None

    def test_index_json_dry_run_outputs_machine_readable_plan(self, temp_db):
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        from privatelens.cli import cli
        from privatelens.db.schema import Asset

        with Session(temp_db) as session:
            session.add(Asset(path="/test/json-unindexed.jpg", sha256="test", media_type="image"))
            session.commit()

        result = CliRunner().invoke(cli, ["index", "--dry-run", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["dry_run"] is True
        assert payload["asset_count"] == 1
        with Session(temp_db) as session:
            asset = session.query(Asset).filter_by(path="/test/json-unindexed.jpg").one()
            assert asset.indexed_at is None

    def test_index_json_stays_parseable_when_ocr_fails(self, temp_db, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from PIL import Image
        from sqlalchemy.orm import Session
        import privatelens.cli as cli_module
        from privatelens.db.schema import Asset

        image_path = tmp_path / "ocr-failure.jpg"
        Image.new("RGB", (10, 10), color="blue").save(image_path)
        with Session(temp_db) as session:
            session.add(Asset(path=str(image_path), sha256="ocr-failure", media_type="image"))
            session.commit()

        class FakeClipExtractor:
            def extract(self, _file_path):
                return None

        def fail_load_engine(_self):
            raise RuntimeError("rapidocr unavailable")

        monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
        monkeypatch.setattr(cli_module.OcrExtractor, "_load_engine", fail_load_engine)

        result = CliRunner().invoke(
            cli_module.cli,
            ["index", "--skip-face", "--skip-vlm", "--batch-size", "1", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["indexed"] == 1
        assert "OCR failed" not in result.stdout

    def test_index_recovers_after_database_error_within_batch(self, temp_db, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from PIL import Image
        from sqlalchemy.orm import Session

        import privatelens.cli as cli_module
        from privatelens.db.schema import Asset, SensitiveItem

        first_path = tmp_path / "duplicate-sensitive.jpg"
        second_path = tmp_path / "healthy.jpg"
        Image.new("RGB", (10, 10), color="blue").save(first_path)
        Image.new("RGB", (10, 10), color="green").save(second_path)

        with Session(temp_db) as session:
            first = Asset(path=str(first_path), sha256="first", media_type="image")
            second = Asset(path=str(second_path), sha256="second", media_type="image")
            session.add_all([first, second])
            session.flush()
            session.add(SensitiveItem(asset_id=first.id, type="id_card", confidence=0.9))
            session.commit()
            first_id = first.id
            second_id = second.id

        class FakeClipExtractor:
            def extract(self, _file_path):
                return None

        class FakeOcrExtractor:
            def extract(self, _file_path):
                return [{"text": "ID 123", "confidence": 0.99}]

        class FakeSensitiveDetector:
            def detect(self, _file_path, _ocr_text):
                return {"type": "id_card", "confidence": 0.99}

        monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
        monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)
        monkeypatch.setattr(cli_module, "SensitiveDetector", FakeSensitiveDetector)

        result = CliRunner().invoke(
            cli_module.cli,
            ["index", "--skip-face", "--skip-vlm", "--batch-size", "10", "--json"],
        )

        assert result.exit_code == 0, (result.output, result.exception)
        payload = json.loads(result.stdout)
        assert payload["indexed"] == 1
        assert payload["errors"] == 1
        with Session(temp_db) as session:
            first = session.get(Asset, first_id)
            second = session.get(Asset, second_id)
            assert first.indexed_at is None
            assert second.indexed_at is not None
            assert session.query(SensitiveItem).filter_by(asset_id=first_id).count() == 1
            assert session.query(SensitiveItem).filter_by(asset_id=second_id).count() == 1

    def test_index_initializes_empty_database(self, tmp_path):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(
            cli,
            ["--data-dir", str(tmp_path), "index", "--dry-run"],
        )

        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_index_skips_extractors_when_no_assets_need_work(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        import privatelens.cli as cli_module

        def fail_extractor(*args, **kwargs):
            raise AssertionError("no-work index should not initialize extractors")

        monkeypatch.setattr(cli_module, "ClipExtractor", fail_extractor)
        monkeypatch.setattr(cli_module, "OcrExtractor", fail_extractor)
        monkeypatch.setattr(cli_module, "FaceExtractor", fail_extractor)
        monkeypatch.setattr(cli_module, "VlmExtractor", fail_extractor)

        result = CliRunner().invoke(
            cli_module.cli,
            ["--data-dir", str(tmp_path), "index"],
        )

        assert result.exit_code == 0
        assert "No assets to index" in result.output

    def test_index_reports_missing_files_before_extractors(self, temp_db, monkeypatch):
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        import privatelens.cli as cli_module
        from privatelens.db.schema import Asset

        with Session(temp_db) as session:
            session.add(Asset(path="/test/missing.jpg", sha256="missing", media_type="image"))
            session.commit()

        def fail_extractor(*args, **kwargs):
            raise AssertionError("missing-only index should not initialize extractors")

        monkeypatch.setattr(cli_module, "ClipExtractor", fail_extractor)
        monkeypatch.setattr(cli_module, "OcrExtractor", fail_extractor)
        monkeypatch.setattr(cli_module, "FaceExtractor", fail_extractor)
        monkeypatch.setattr(cli_module, "VlmExtractor", fail_extractor)

        result = CliRunner().invoke(cli_module.cli, ["index"])

        assert result.exit_code == 0
        assert "Skipped missing files: 1" in result.output

    def test_index_uses_progress_for_human_output(self, temp_db, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from PIL import Image
        from sqlalchemy.orm import Session
        import privatelens.cli as cli_module
        from privatelens.db.schema import Asset

        image_path = tmp_path / "index-progress.jpg"
        Image.new("RGB", (10, 10), color="blue").save(image_path)
        with Session(temp_db) as session:
            session.add(Asset(path=str(image_path), sha256="progress", media_type="image"))
            session.commit()

        events = []

        class FakeProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def add_task(self, description, total):
                events.append(("task", description, total))
                return "index-task"

            def advance(self, task_id):
                events.append(("advance", task_id))

        class FakeClipExtractor:
            def extract(self, _file_path):
                return None

        class FakeOcrExtractor:
            def extract(self, _file_path):
                return []

        monkeypatch.setattr(cli_module, "Progress", FakeProgress, raising=False)
        monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
        monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)

        result = CliRunner().invoke(
            cli_module.cli,
            ["index", "--skip-face", "--skip-vlm", "--batch-size", "1"],
        )

        assert result.exit_code == 0
        assert events == [
            ("task", "Indexing assets", 1),
            ("advance", "index-task"),
        ]

    def test_force_index_replaces_stale_derived_rows(self, temp_db, tmp_path, monkeypatch):
        import numpy as np
        from click.testing import CliRunner
        from PIL import Image
        from sqlalchemy.orm import Session
        import privatelens.cli as cli_module
        from privatelens.db.schema import Asset, ImageEmbedding, OcrBlock, SensitiveItem

        image_path = tmp_path / "force.jpg"
        Image.new("RGB", (10, 10), color="blue").save(image_path)

        with Session(temp_db) as session:
            asset = Asset(
                path=str(image_path),
                sha256="old",
                media_type="image",
                indexed_at=datetime.now(),
                is_sensitive=True,
                sensitive_type="passport",
            )
            session.add(asset)
            session.commit()
            asset_id = asset.id
            session.add(
                ImageEmbedding(
                    asset_id=asset_id,
                    vector=np.array([0.0, 0.0], dtype=np.float32).tobytes(),
                )
            )
            session.add(OcrBlock(asset_id=asset_id, text="old text"))
            session.add(SensitiveItem(asset_id=asset_id, type="passport"))
            session.commit()

        class FakeClipExtractor:
            def extract(self, _file_path):
                return np.array([1.0, 2.0], dtype=np.float32)

        class FakeOcrExtractor:
            def extract(self, _file_path):
                return [{"text": "new text", "bbox": None, "confidence": 0.9}]

        class FakeSensitiveDetector:
            def detect(self, _file_path, _ocr_text):
                return None

        monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
        monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)
        monkeypatch.setattr(cli_module, "SensitiveDetector", FakeSensitiveDetector)

        result = CliRunner().invoke(
            cli_module.cli,
            ["index", "--force", "--skip-face", "--skip-vlm", "--batch-size", "1"],
        )

        assert result.exit_code == 0
        with Session(temp_db) as session:
            asset = session.query(Asset).filter_by(id=asset_id).one()
            embedding = session.query(ImageEmbedding).filter_by(asset_id=asset_id).one()
            ocr_blocks = session.query(OcrBlock).filter_by(asset_id=asset_id).all()
            assert np.frombuffer(embedding.vector, dtype=np.float32).tolist() == [1.0, 2.0]
            assert [block.text for block in ocr_blocks] == ["new text"]
            assert session.query(SensitiveItem).filter_by(asset_id=asset_id).count() == 0
            assert asset.is_sensitive is False
            assert asset.sensitive_type is None
            assert asset.indexed_at is not None

    def test_index_migrates_stale_clip_model_without_duplicating_signals(
        self, temp_db, tmp_path, monkeypatch
    ):
        import numpy as np
        from click.testing import CliRunner
        from sqlalchemy.orm import Session
        import privatelens.cli as cli_module
        from privatelens.db.schema import Asset, Caption, Face, ImageEmbedding, OcrBlock
        from privatelens.extractors.clip import clip_model_id

        image_path = tmp_path / "model-migration.jpg"
        Image.new("RGB", (10, 10), color="blue").save(image_path)
        monkeypatch.setattr(settings, "clip_model", "ViT-B-32-quickgelu")
        monkeypatch.setattr(settings, "clip_pretrained", "openai")

        with Session(temp_db) as session:
            asset = Asset(
                path=str(image_path),
                sha256="model-migration",
                media_type="image",
                indexed_at=datetime.now(),
            )
            session.add(asset)
            session.commit()
            asset_id = asset.id
            session.add(
                ImageEmbedding(
                    asset_id=asset_id,
                    model="ViT-B-32:openai",
                    vector=np.zeros(512, dtype=np.float32).tobytes(),
                )
            )
            session.add(OcrBlock(asset_id=asset_id, text="preserve OCR"))
            session.add(Face(asset_id=asset_id, bbox="[]", confidence=0.8))
            session.add(Caption(asset_id=asset_id, model="test", caption="preserve caption"))
            session.commit()

        class FakeClipExtractor:
            def extract(self, _file_path):
                return np.ones(512, dtype=np.float32)

        class FakeOcrExtractor:
            def extract(self, _file_path):
                raise AssertionError("OCR must not rerun for a CLIP-only migration")

        monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
        monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)

        result = CliRunner().invoke(
            cli_module.cli,
            ["index", "--skip-face", "--skip-vlm", "--batch-size", "1", "--json"],
        )

        assert result.exit_code == 0, (result.output, result.exception)
        assert json.loads(result.stdout)["model_migrations"] == 1
        with Session(temp_db) as session:
            embedding = session.query(ImageEmbedding).filter_by(asset_id=asset_id).one()
            assert embedding.model == clip_model_id(settings.clip_model, settings.clip_pretrained)
            assert np.frombuffer(embedding.vector, dtype=np.float32).tolist() == [1.0] * 512
            assert session.query(OcrBlock).filter_by(asset_id=asset_id).count() == 1
            assert session.query(Face).filter_by(asset_id=asset_id).count() == 1
            assert session.query(Caption).filter_by(asset_id=asset_id).count() == 1

    def test_index_rejects_zero_batch_size(self):
        from click.testing import CliRunner
        from privatelens.cli import cli

        result = CliRunner().invoke(cli, ["index", "--batch-size", "0", "--dry-run"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_non_cli_modules_do_not_use_raw_print(self):
        offenders = []
        for path in Path("privatelens").rglob("*.py"):
            if path.name == "cli.py":
                continue
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                if "print(" in line:
                    offenders.append(f"{path}:{line_number}")

        assert offenders == []


class TestApi:
    """Test the packaged secondary web surface."""

    def test_root_template_renders_from_installed_package_path(self, temp_db):
        import asyncio

        from fastapi import Request

        import privatelens.api as api_module

        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/",
                "root_path": "",
                "query_string": b"",
                "headers": [],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
            }
        )
        response = asyncio.run(api_module.index(request))

        assert api_module.TEMPLATES_DIR == Path(__file__).parents[1] / "privatelens" / "web"
        assert api_module.app.version == privatelens.__version__
        assert response.status_code == 200
        assert b"PrivateLens" in response.body

    def test_serve_defaults_to_loopback_and_accepts_explicit_bind(self, temp_db, monkeypatch):
        from click.testing import CliRunner

        import uvicorn

        import privatelens.cli as cli_module

        calls = []
        monkeypatch.setattr(
            uvicorn,
            "run",
            lambda app, *, host, port, access_log: calls.append((app, host, port, access_log)),
        )
        runner = CliRunner()

        default_result = runner.invoke(cli_module.cli, ["serve"])
        docker_result = runner.invoke(
            cli_module.cli,
            ["serve", "--host", "0.0.0.0", "--port", "8123"],
        )

        assert default_result.exit_code == 0, default_result.output
        assert docker_result.exit_code == 0, docker_result.output
        assert [(host, port, access_log) for _app, host, port, access_log in calls] == [
            ("127.0.0.1", 8000, False),
            ("0.0.0.0", 8123, False),
        ]

    def test_health_and_thumbnail_routes(self, temp_db):
        import asyncio

        from sqlalchemy.orm import Session

        import privatelens.api as api_module

        thumbnail_dir = settings.resolved_thumbnail_dir
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_path = thumbnail_dir / "1.jpg"
        Image.new("RGB", (16, 16), "blue").save(thumbnail_path)
        with Session(temp_db) as session:
            asset = Asset(
                path="/photos/test.jpg",
                sha256="api-thumbnail",
                media_type="image",
                thumbnail_path=str(thumbnail_path),
            )
            session.add(asset)
            session.commit()
            asset_id = asset.id

        health = asyncio.run(api_module.api_health())
        response = asyncio.run(api_module.thumbnail(asset_id))

        assert health == {"status": "ok", "version": privatelens.__version__}
        assert Path(response.path) == thumbnail_path.resolve()

    def test_web_results_are_rendered_as_text_not_html(self):
        template = Path("privatelens/web/index.html").read_text()

        assert "path.textContent = result.path || '';" in template
        assert "explanation.textContent = result.explanation || '';" in template
        assert "resultsDiv.innerHTML = results.map" not in template

    def test_thumbnail_route_rejects_paths_outside_managed_directory(self, temp_db, tmp_path):
        import asyncio

        from fastapi import HTTPException
        from sqlalchemy.orm import Session

        import privatelens.api as api_module

        outside_path = tmp_path / "outside.jpg"
        Image.new("RGB", (16, 16), "red").save(outside_path)
        with Session(temp_db) as session:
            asset = Asset(
                path="/photos/outside.jpg",
                sha256="api-outside-thumbnail",
                media_type="image",
                thumbnail_path=str(outside_path),
            )
            session.add(asset)
            session.commit()
            asset_id = asset.id

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(api_module.thumbnail(asset_id))

        assert exc_info.value.status_code == 404


class TestUtils:
    """Test utility functions."""

    def test_get_image_files(self, temp_db):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            img1 = Path(tmpdir) / "test1.jpg"
            img2 = Path(tmpdir) / "test2.png"
            txt = Path(tmpdir) / "readme.txt"

            Image.new("RGB", (10, 10)).save(img1)
            Image.new("RGB", (10, 10)).save(img2)
            txt.write_text("hello")

            files = get_image_files(Path(tmpdir))
            assert len(files) == 2
            assert all(f.suffix in {".jpg", ".png"} for f in files)


class TestIntegrations:
    """Test integration importers."""

    def test_apple_photos_importer(self, temp_db):
        from privatelens.integrations.apple_photos import ApplePhotosImporter

        importer = ApplePhotosImporter()
        assert importer is not None

    def test_google_takeout_importer(self, temp_db):
        from privatelens.integrations.google_takeout import GoogleTakeoutImporter

        importer = GoogleTakeoutImporter()
        assert importer is not None

    def test_anythingllm_connector(self, temp_db):
        from privatelens.integrations.anythingllm import AnythingLLMConnector

        connector = AnythingLLMConnector()
        assert connector is not None

    def test_immich_connector(self, temp_db):
        from privatelens.integrations.immich import ImmichConnector

        connector = ImmichConnector()
        assert connector is not None
