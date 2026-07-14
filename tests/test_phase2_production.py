"""Focused production tests for Phase 2 detection and feedback behavior."""

import json
from datetime import datetime

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image
from sqlalchemy.orm import Session

import privatelens.cli as cli_module
from privatelens.config import settings
from privatelens.db.schema import (
    Asset,
    Caption,
    Detection,
    Face,
    ImageEmbedding,
    OcrBlock,
    SearchEvent,
    SensitiveItem,
    init_db,
    reset_engine,
)
from privatelens.privacy.audit import PrivacyAuditor
from privatelens.privacy.encrypt import MetadataEncryptor
from privatelens.search.engine import SearchEngine
from privatelens.search.queries import SearchQueries


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data_dir)
    monkeypatch.setattr(settings, "db_path", None)
    monkeypatch.setattr(settings, "thumbnail_dir", None)
    monkeypatch.setattr(settings, "model_cache_dir", None)
    reset_engine()
    engine = init_db()
    yield engine, data_dir
    reset_engine()


def test_detection_taxonomy_is_canonical_and_deduplicated():
    from privatelens.extractors.detections import derive_detections

    detections = derive_detections(
        media_type="screenshot",
        ocr_blocks=[{"text": "RECEIPT TOTAL"}],
        document_classification={"type": "receipt", "confidence": 0.85},
        sensitive_detection={
            "type": "driver_license",
            "confidence": 0.88,
            "source": "ocr",
        },
        vlm_classification={
            "type": "id_card",
            "confidence": 0.95,
            "description": "driver license",
        },
        vlm_caption="A receipt beside a whiteboard, a dog, and a car dashboard.",
        vlm_model="test-vlm",
    )

    by_label = {detection.label: detection for detection in detections}
    assert set(by_label) == {
        "screenshot",
        "text",
        "receipt",
        "document",
        "id_card",
        "whiteboard",
        "pet",
        "car",
    }
    assert len(detections) == len(by_label)
    assert by_label["id_card"].source_model == "vlm:test-vlm"
    assert by_label["id_card"].confidence == 0.95


def test_detection_taxonomy_rejects_non_finite_confidence_and_generic_photo_document():
    from privatelens.extractors.detections import derive_detections

    generic_detections = derive_detections(
        media_type="image",
        ocr_blocks=[{"text": "WELCOME", "confidence": 0.9}],
        document_classification={"type": "document", "confidence": float("nan")},
    )
    generic_by_label = {detection.label: detection for detection in generic_detections}
    assert "document" not in generic_by_label
    assert generic_by_label["text"].confidence == pytest.approx(0.9)

    vlm_detections = derive_detections(
        media_type="image",
        vlm_classification={"type": "receipt", "confidence": float("nan")},
        vlm_model="test-vlm",
    )
    by_label = {detection.label: detection for detection in vlm_detections}
    assert by_label["receipt"].confidence == 0.0
    assert np.isfinite(by_label["receipt"].confidence)

    document_detections = derive_detections(
        media_type="document",
        document_classification={"type": "document", "confidence": 0.5},
    )
    assert "document" in {detection.label for detection in document_detections}


def test_production_index_writes_deduplicated_detection_rows(isolated_db, tmp_path, monkeypatch):
    engine, data_dir = isolated_db
    image_path = tmp_path / "receipt.jpg"
    Image.new("RGB", (400, 600), "white").save(image_path)

    with Session(engine) as session:
        asset = Asset(
            path=str(image_path),
            sha256="1" * 64,
            media_type="document",
        )
        session.add(asset)
        session.flush()
        session.add_all(
            [
                Detection(asset_id=asset.id, label="stale", source_model="old"),
                Detection(asset_id=asset.id, label="receipt", source_model="old"),
            ]
        )
        session.commit()
        asset_id = asset.id

    class FakeClipExtractor:
        def extract(self, _path):
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1.0
            return vector

    class FakeOcrExtractor:
        def extract(self, _path):
            return [{"text": "RECEIPT TOTAL TAX PAYMENT", "confidence": 0.96}]

    class FakeVlmExtractor:
        model = "test-vlm"

        def caption(self, _path):
            return "A receipt on a whiteboard beside a dog and a car dashboard."

        def classify_document(self, _path):
            return {"type": "receipt", "confidence": 0.92, "description": "receipt"}

    monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
    monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)
    monkeypatch.setattr(cli_module, "VlmExtractor", FakeVlmExtractor)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "--data-dir",
            str(data_dir),
            "index",
            "--force",
            "--skip-face",
            "--with-vlm",
            "--batch-size",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["indexed"] == 1
    with Session(engine) as session:
        rows = session.query(Detection).filter_by(asset_id=asset_id).all()

    by_label = {row.label: row for row in rows}
    assert "stale" not in by_label
    assert set(by_label) >= {"document", "text", "receipt", "whiteboard", "pet", "car"}
    assert len(rows) == len(by_label)
    assert all(row.source_model for row in rows)
    assert all(0.0 <= (row.confidence or 0.0) <= 1.0 for row in rows)


@pytest.mark.parametrize("with_key", [True, False])
def test_index_encrypts_sensitive_classification_payload_when_configured(
    isolated_db, tmp_path, monkeypatch, with_key
):
    engine, data_dir = isolated_db
    image_path = tmp_path / "receipt.jpg"
    Image.new("RGB", (120, 80), "white").save(image_path)
    key = MetadataEncryptor().generate_key() if with_key else None
    monkeypatch.setattr(settings, "encryption_key", key)

    with Session(engine) as session:
        session.add(Asset(path=str(image_path), sha256="e" * 64, media_type="document"))
        session.commit()

    class FakeClipExtractor:
        def extract(self, _path):
            return np.ones(512, dtype=np.float32)

    class FakeOcrExtractor:
        def extract(self, _path):
            return [{"text": "RECEIPT TOTAL", "confidence": 0.95}]

    monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
    monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "--data-dir",
            str(data_dir),
            "index",
            "--skip-face",
            "--skip-vlm",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    with Session(engine) as session:
        sensitive = session.query(SensitiveItem).one()
        encrypted_metadata = sensitive.encrypted_metadata

    if key is None:
        assert encrypted_metadata is None
    else:
        assert encrypted_metadata is not None
        assert MetadataEncryptor(key).decrypt(encrypted_metadata) == {
            "type": "receipt",
            "confidence": 0.7,
            "source": "filename",
        }


def test_privacy_audit_reports_partial_sensitive_encryption(isolated_db, monkeypatch):
    engine, _data_dir = isolated_db
    key = MetadataEncryptor().generate_key()
    monkeypatch.setattr(settings, "encryption_key", key)
    encrypted = MetadataEncryptor(key).encrypt({"source": "test"})
    with Session(engine) as session:
        first = Asset(path="/photos/first.jpg", sha256="f" * 64, media_type="document")
        second = Asset(path="/photos/second.jpg", sha256="a" * 64, media_type="document")
        session.add_all([first, second])
        session.flush()
        session.add_all(
            [
                SensitiveItem(
                    asset_id=first.id,
                    type="receipt",
                    confidence=0.8,
                    encrypted_metadata=encrypted,
                ),
                SensitiveItem(asset_id=second.id, type="id_card", confidence=0.8),
            ]
        )
        session.commit()

    check = PrivacyAuditor()._check_sensitive_encryption()

    assert check["status"] == "warning"
    assert "1/2 sensitive classification payloads encrypted" in check["details"]
    assert "1/2 decryptable" in check["details"]
    assert "index --force" in check["remediation"][0]


def test_privacy_audit_rejects_ciphertext_from_a_different_key(isolated_db, monkeypatch):
    engine, _data_dir = isolated_db
    original_key = MetadataEncryptor().generate_key()
    wrong_key = MetadataEncryptor().generate_key()
    ciphertext = MetadataEncryptor(original_key).encrypt({"source": "test"})
    monkeypatch.setattr(settings, "encryption_key", wrong_key)
    with Session(engine) as session:
        asset = Asset(path="/photos/wrong-key.jpg", sha256="c" * 64, media_type="document")
        session.add(asset)
        session.flush()
        session.add(
            SensitiveItem(
                asset_id=asset.id,
                type="receipt",
                confidence=0.8,
                encrypted_metadata=ciphertext,
            )
        )
        session.commit()

    check = PrivacyAuditor()._check_sensitive_encryption()

    assert check["status"] == "warning"
    assert "1/1 sensitive classification payloads encrypted" in check["details"]
    assert "0/1 decryptable" in check["details"]


def test_index_respects_disabled_sensitive_scan(isolated_db, tmp_path, monkeypatch):
    engine, data_dir = isolated_db
    image_path = tmp_path / "receipt.jpg"
    Image.new("RGB", (120, 80), "white").save(image_path)
    monkeypatch.setattr(settings, "sensitive_scan", False)

    with Session(engine) as session:
        session.add(Asset(path=str(image_path), sha256="b" * 64, media_type="document"))
        session.commit()

    class FakeClipExtractor:
        def extract(self, _path):
            return np.ones(512, dtype=np.float32)

    class FakeOcrExtractor:
        def extract(self, _path):
            return [{"text": "RECEIPT TOTAL", "confidence": 0.95}]

    monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
    monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "--data-dir",
            str(data_dir),
            "index",
            "--skip-face",
            "--skip-vlm",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    with Session(engine) as session:
        asset = session.query(Asset).one()
        assert session.query(SensitiveItem).count() == 0
        assert asset.is_sensitive is False
        assert asset.sensitive_type is None


@pytest.mark.parametrize("component_option", ["--only-face", "--only-vlm"])
def test_component_only_pass_does_not_suppress_later_base_indexing(
    isolated_db, tmp_path, monkeypatch, component_option
):
    engine, data_dir = isolated_db
    image_path = tmp_path / "receipt.jpg"
    Image.new("RGB", (120, 80), "white").save(image_path)

    with Session(engine) as session:
        asset = Asset(path=str(image_path), sha256="2" * 64, media_type="image")
        session.add(asset)
        session.commit()
        asset_id = asset.id

    class FakeFaceExtractor:
        def __init__(self, _model_name):
            pass

        def extract(self, _path):
            return [{"bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "confidence": 0.9}]

    class FakeClipExtractor:
        def extract(self, _path):
            return np.ones(512, dtype=np.float32)

    class FakeOcrExtractor:
        def extract(self, _path):
            return [{"text": "RECEIPT TOTAL", "confidence": 0.95}]

    class FakeVlmExtractor:
        model = "test-vlm"

        def caption(self, _path):
            return "A receipt."

        def classify_document(self, _path):
            return {"type": "receipt", "confidence": 0.9}

    monkeypatch.setattr(cli_module, "FaceExtractor", FakeFaceExtractor)
    monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
    monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)
    monkeypatch.setattr(cli_module, "VlmExtractor", FakeVlmExtractor)

    runner = CliRunner()
    component_result = runner.invoke(
        cli_module.cli,
        ["--data-dir", str(data_dir), "index", component_option, "--json"],
    )
    assert component_result.exit_code == 0, component_result.output

    with Session(engine) as session:
        stored_asset = session.get(Asset, asset_id)
        assert stored_asset is not None
        assert stored_asset.indexed_at is None
        if component_option == "--only-face":
            assert session.query(Face).filter_by(asset_id=asset_id).count() == 1
        else:
            assert session.query(Caption).filter_by(asset_id=asset_id).count() == 1

    base_result = runner.invoke(
        cli_module.cli,
        [
            "--data-dir",
            str(data_dir),
            "index",
            "--skip-face",
            "--skip-vlm",
            "--json",
        ],
    )
    assert base_result.exit_code == 0, base_result.output

    with Session(engine) as session:
        stored_asset = session.get(Asset, asset_id)
        assert stored_asset is not None
        assert stored_asset.indexed_at is not None
        assert session.query(ImageEmbedding).filter_by(asset_id=asset_id).count() == 1
        assert session.query(OcrBlock).filter_by(asset_id=asset_id).count() == 1
        labels = {row.label for row in session.query(Detection).filter_by(asset_id=asset_id).all()}
        assert {"text", "receipt"} <= labels


def test_unavailable_forced_vlm_preserves_caption_and_vlm_detections(
    isolated_db, tmp_path, monkeypatch
):
    engine, data_dir = isolated_db
    image_path = tmp_path / "credential.jpg"
    Image.new("RGB", (120, 80), "white").save(image_path)

    with Session(engine) as session:
        asset = Asset(
            path=str(image_path),
            sha256="3" * 64,
            media_type="image",
            indexed_at=datetime(2026, 1, 1),
        )
        session.add(asset)
        session.flush()
        session.add(
            Caption(
                asset_id=asset.id,
                model="test-vlm",
                caption="An official credential.",
                confidence=0.8,
            )
        )
        session.add(
            Detection(
                asset_id=asset.id,
                label="id_card",
                confidence=0.95,
                source_model="vlm:test-vlm",
            )
        )
        session.commit()
        asset_id = asset.id

    class UnavailableVlmExtractor:
        model = "test-vlm"

        def caption(self, _path):
            return None

        def classify_document(self, _path):
            return None

    monkeypatch.setattr(cli_module, "VlmExtractor", UnavailableVlmExtractor)

    result = CliRunner().invoke(
        cli_module.cli,
        ["--data-dir", str(data_dir), "index", "--force", "--only-vlm", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["errors"] == 0
    with Session(engine) as session:
        captions = session.query(Caption).filter_by(asset_id=asset_id).all()
        detections = session.query(Detection).filter_by(asset_id=asset_id).all()
    assert [(caption.caption, caption.model) for caption in captions] == [
        ("An official credential.", "test-vlm")
    ]
    assert [(row.label, row.source_model) for row in detections] == [("id_card", "vlm:test-vlm")]


def test_detection_search_uses_exact_canonical_labels(isolated_db):
    engine, _data_dir = isolated_db
    with Session(engine) as session:
        car = Asset(path="/photos/car.jpg", sha256="4" * 64, media_type="image")
        id_card = Asset(path="/photos/id.jpg", sha256="5" * 64, media_type="image")
        session.add_all([car, id_card])
        session.flush()
        session.add_all(
            [
                Detection(asset_id=car.id, label="car", confidence=0.9, source_model="test"),
                Detection(
                    asset_id=id_card.id,
                    label="id_card",
                    confidence=0.95,
                    source_model="test",
                ),
            ]
        )
        session.commit()
        car_id = car.id

    assert SearchQueries().detection_label_search(["car"]) == [(car_id, "car", 0.9)]


def test_failed_asset_does_not_increment_model_migration_count(isolated_db, tmp_path, monkeypatch):
    engine, data_dir = isolated_db
    image_path = tmp_path / "broken-ocr.jpg"
    Image.new("RGB", (120, 80), "white").save(image_path)

    with Session(engine) as session:
        asset = Asset(path=str(image_path), sha256="6" * 64, media_type="image")
        session.add(asset)
        session.flush()
        session.add(
            ImageEmbedding(asset_id=asset.id, model="legacy:model", vector=np.ones(512).tobytes())
        )
        session.commit()
        asset_id = asset.id

    class FakeClipExtractor:
        def extract(self, _path):
            return np.ones(512, dtype=np.float32)

    class InvalidOcrExtractor:
        def extract(self, _path):
            return [{"confidence": 0.9}]

    monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
    monkeypatch.setattr(cli_module, "OcrExtractor", InvalidOcrExtractor)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "--data-dir",
            str(data_dir),
            "index",
            "--skip-face",
            "--skip-vlm",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["errors"] == 1
    assert payload["model_migrations"] == 0
    with Session(engine) as session:
        embedding = session.get(ImageEmbedding, asset_id)
        assert embedding is not None
        assert embedding.model == "legacy:model"


def test_positive_feedback_reorders_near_ties_for_smart_and_recipe_search(isolated_db, monkeypatch):
    engine, _data_dir = isolated_db
    with Session(engine) as session:
        assets = [
            Asset(path=f"/photos/{name}.jpg", sha256=str(index) * 64, media_type="image")
            for index, name in enumerate(["leader", "positive", "negative"], start=1)
        ]
        session.add_all(assets)
        session.flush()
        leader_id, positive_id, negative_id = [asset.id for asset in assets]
        session.add_all(
            [
                SearchEvent(
                    query="previous",
                    query_type="smart",
                    results_count=3,
                    result_clicked=positive_id,
                    feedback=1,
                ),
                SearchEvent(
                    query="previous",
                    query_type="smart",
                    results_count=3,
                    result_clicked=negative_id,
                    feedback=-1,
                ),
            ]
        )
        session.commit()

    search_engine = SearchEngine()
    monkeypatch.setattr(
        search_engine.clip_extractor,
        "encode_text",
        lambda _query: np.ones(512, dtype=np.float32),
    )
    monkeypatch.setattr(
        search_engine,
        "_vector_search",
        lambda _embedding, limit: [
            {"asset_id": leader_id, "score": 0.60},
            {"asset_id": positive_id, "score": 0.58},
            {"asset_id": negative_id, "score": 0.58},
        ][:limit],
    )
    monkeypatch.setattr(search_engine.queries, "ocr_fts_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(search_engine.queries, "caption_fts_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(search_engine.queries, "path_fts_search", lambda *_args, **_kwargs: [])

    smart_ids = [result["asset_id"] for result in search_engine._smart_search("query", 3)]
    recipe_ids = [
        result["asset_id"]
        for result in search_engine._execute_plan(
            {"signals": [{"type": "semantic", "weight": 1.0}]},
            "query",
            3,
        )
    ]

    assert smart_ids == [positive_id, leader_id, negative_id]
    assert recipe_ids == [positive_id, leader_id, negative_id]

    monkeypatch.setattr(
        search_engine.queries,
        "positive_feedback_counts",
        lambda _asset_ids: {positive_id: 100},
    )
    capped_candidates = {leader_id: 0.60, positive_id: 0.54}
    search_engine._apply_positive_feedback_boost(capped_candidates)
    assert capped_candidates[positive_id] == pytest.approx(0.59)
    assert capped_candidates[leader_id] > capped_candidates[positive_id]


@pytest.mark.parametrize(
    ("answer", "expected_feedback"),
    [
        ("y\n2\n", (42, 1, 22)),
        ("n\n", (42, -1, None)),
    ],
)
def test_cli_feedback_records_only_an_explicitly_selected_positive_result(
    monkeypatch, answer, expected_feedback
):
    feedback_calls = []

    class FakeSearchEngine:
        last_search_event_id = 42

        def search(self, _query, search_type="smart", limit=50, record_event=False):
            assert record_event is True
            return [
                {
                    "asset_id": 11,
                    "path": "/photos/first.jpg",
                    "score": 0.8,
                    "explanation": "first",
                },
                {
                    "asset_id": 22,
                    "path": "/photos/second.jpg",
                    "score": 0.7,
                    "explanation": "second",
                },
            ][:limit]

        def search_by_recipe(self, recipe_name, query, limit=50, rerank=True, record_event=False):
            return self.search(query, limit=limit, record_event=record_event)

        def record_feedback(self, event_id, feedback, result_clicked=None):
            feedback_calls.append((event_id, feedback, result_clicked))

    monkeypatch.setattr(cli_module, "SearchEngine", FakeSearchEngine)
    monkeypatch.setattr(cli_module, "detect_recipe_for_query", lambda _query: None)

    result = CliRunner().invoke(
        cli_module.cli,
        ["search", "landscape", "--feedback"],
        input=answer,
    )

    assert result.exit_code == 0, result.output
    assert feedback_calls == [expected_feedback]
