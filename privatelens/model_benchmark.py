"""Model-dependent quality benchmark over the generated demo corpus."""

import gc
import json
import platform
import sqlite3
import tempfile
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from privatelens.config import settings
from privatelens.db.schema import (
    VECTOR_DIMENSIONS,
    Asset,
    ImageEmbedding,
    OcrBlock,
    init_db,
)
from privatelens.demo import create_demo_library
from privatelens.extractors.clip import ClipExtractor, clip_model_id
from privatelens.extractors.ocr import OcrExtractor
from privatelens.extractors.vlm import VlmExtractor
from privatelens.search.engine import SearchEngine


MODEL_CASES: list[dict[str, Any]] = [
    {
        "name": "receipt",
        "filename": "target-receipt-lunch.jpg",
        "semantic_query": "a photo of a store receipt",
        "ocr_query": "payment visa",
        "expected_type": "receipt",
        "caption_terms": ["receipt", "total"],
    },
    {
        "name": "driver-license",
        "filename": "driver-license-backup.jpg",
        "semantic_query": "a driver's license identity card",
        "ocr_query": "sample user",
        "expected_type": "id_card",
        "caption_terms": ["driver", "license"],
    },
    {
        "name": "travel-screenshot",
        "filename": "phone-screenshot-travel.png",
        "semantic_query": "a mobile phone chat screenshot",
        "ocr_query": "boarding pass",
        "expected_type": "screenshot",
        "caption_terms": ["screenshot", "boarding"],
    },
    {
        "name": "whiteboard-notes",
        "filename": "whiteboard-notes-project.jpg",
        "semantic_query": "project planning notes on a whiteboard",
        "ocr_query": "sidecar index",
        "expected_type": "document",
        "caption_terms": ["project", "sidecar"],
    },
]

MINIMUM_CLIP_TOP1_RATE = 1.0
MINIMUM_OCR_TOP1_RATE = 1.0
MINIMUM_VLM_CLASSIFICATION_ACCURACY = 1.0
MINIMUM_VLM_CAPTION_TERM_RECALL = 1.0


def run_model_benchmark(
    output_path: Path | None = None,
    *,
    include_vlm: bool = True,
    clip_extractor: Any | None = None,
    ocr_extractor: Any | None = None,
    vlm_extractor: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run real model extraction and retrieval over generated, non-private images."""
    old_data_dir = settings.data_dir
    old_db_path = settings.db_path
    started_at = perf_counter()

    try:
        with tempfile.TemporaryDirectory(prefix="privatelens-model-benchmark-") as tmpdir:
            root = Path(tmpdir)
            settings.data_dir = root / "index"
            settings.db_path = None
            image_dir = root / "images"
            create_demo_library(image_dir, force=True)
            report = _run_model_cases(
                image_dir,
                include_vlm=include_vlm,
                clip_extractor=clip_extractor,
                ocr_extractor=ocr_extractor,
                vlm_extractor=vlm_extractor,
                progress=progress,
            )
    finally:
        settings.data_dir = old_data_dir
        settings.db_path = old_db_path

    report["timings_ms"]["total"] = _elapsed_ms(started_at)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _run_model_cases(
    image_dir: Path,
    *,
    include_vlm: bool,
    clip_extractor: Any | None,
    ocr_extractor: Any | None,
    vlm_extractor: Any | None,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    errors: dict[str, dict[str, str]] = {case["filename"]: {} for case in MODEL_CASES}

    _notify(progress, "Extracting CLIP image and text embeddings...")
    clip_started = perf_counter()
    clip_runtime = clip_extractor or ClipExtractor()
    clip_id = getattr(
        clip_runtime,
        "model_id",
        clip_model_id(settings.clip_model, settings.clip_pretrained),
    )
    image_vectors: dict[str, np.ndarray | None] = {}
    query_vectors: dict[str, np.ndarray | None] = {}
    for case in MODEL_CASES:
        filename = case["filename"]
        image_vectors[filename] = _validated_vector(
            clip_runtime.extract(image_dir / filename),
            errors[filename],
            "image_embedding",
        )
        query_vectors[filename] = _validated_vector(
            clip_runtime.encode_text(case["semantic_query"]),
            errors[filename],
            "text_embedding",
        )
    timings["clip"] = _elapsed_ms(clip_started)

    clip_extractor = None
    del clip_runtime
    gc.collect()

    _notify(progress, "Extracting OCR and building the temporary search index...")
    ocr_started = perf_counter()
    ocr_runtime = ocr_extractor or OcrExtractor()
    ocr_blocks: dict[str, list[dict[str, Any]]] = {}
    for case in MODEL_CASES:
        filename = case["filename"]
        blocks = ocr_runtime.extract(image_dir / filename) or []
        ocr_blocks[filename] = blocks
        if not blocks:
            errors[filename]["ocr"] = "no OCR text extracted"
    timings["ocr"] = _elapsed_ms(ocr_started)

    db_engine, asset_ids = _seed_model_index(image_vectors, ocr_blocks, clip_id)
    search_engine = SearchEngine()
    vector_backend = _vector_backend(db_engine)

    _notify(progress, "Measuring CLIP and OCR retrieval quality...")
    retrieval_started = perf_counter()
    cases: list[dict[str, Any]] = []
    id_to_filename = {asset_id: filename for filename, asset_id in asset_ids.items()}
    for case in MODEL_CASES:
        filename = case["filename"]
        target_id = asset_ids[filename]
        query_vector = query_vectors[filename]
        vector_results = (
            search_engine._vector_search(query_vector, limit=len(MODEL_CASES))
            if query_vector is not None
            else []
        )
        ocr_results = search_engine.search(
            case["ocr_query"],
            search_type="ocr",
            limit=len(MODEL_CASES),
        )
        clip_rank = _rank_for_asset(vector_results, target_id)
        ocr_rank = _rank_for_asset(ocr_results, target_id)
        cases.append(
            {
                "name": case["name"],
                "filename": filename,
                "clip": {
                    "query": case["semantic_query"],
                    "first_relevant_rank": clip_rank,
                    "top_1": clip_rank == 1,
                    "ranked_filenames": [
                        id_to_filename[result["asset_id"]]
                        for result in vector_results
                        if result["asset_id"] in id_to_filename
                    ],
                },
                "ocr": {
                    "query": case["ocr_query"],
                    "first_relevant_rank": ocr_rank,
                    "top_1": ocr_rank == 1,
                    "ranked_filenames": [
                        id_to_filename[result["asset_id"]]
                        for result in ocr_results
                        if result["asset_id"] in id_to_filename
                    ],
                    "text": " ".join(
                        str(block.get("text", "")) for block in ocr_blocks[filename]
                    ).strip(),
                },
                "vlm": None,
                "errors": errors[filename],
            }
        )
    timings["retrieval"] = _elapsed_ms(retrieval_started)

    search_engine.engine.dispose()
    search_engine.queries.engine.dispose()
    db_engine.dispose()

    vlm_available = None
    vlm_model = settings.vlm_model if include_vlm else None
    if include_vlm:
        _notify(progress, "Measuring local VLM caption and classification quality...")
        vlm_started = perf_counter()
        vlm_runtime = vlm_extractor or VlmExtractor()
        vlm_model = getattr(vlm_runtime, "model", settings.vlm_model)
        vlm_available = bool(vlm_runtime.is_available())
        for case, case_report in zip(MODEL_CASES, cases, strict=True):
            filename = case["filename"]
            caption = vlm_runtime.caption(image_dir / filename) if vlm_available else None
            classification = (
                vlm_runtime.classify_document(image_dir / filename) if vlm_available else None
            )
            predicted_type = (
                str(classification.get("type", "")).strip().lower()
                if isinstance(classification, dict)
                else None
            )
            matched_terms = [
                term for term in case["caption_terms"] if term.lower() in (caption or "").lower()
            ]
            caption_recall = len(matched_terms) / len(case["caption_terms"])
            case_report["vlm"] = {
                "expected_type": case["expected_type"],
                "predicted_type": predicted_type,
                "classification_correct": predicted_type == case["expected_type"],
                "caption_terms": case["caption_terms"],
                "matched_caption_terms": matched_terms,
                "caption_term_recall": round(caption_recall, 4),
                "caption": caption,
            }
            if not caption:
                case_report["errors"]["vlm_caption"] = "no caption returned"
            if not classification:
                case_report["errors"]["vlm_classification"] = "no classification returned"
        timings["vlm"] = _elapsed_ms(vlm_started)

    summary = _summarize(cases, include_vlm)
    return {
        "benchmark": "privatelens-model-quality-v1",
        "version": 1,
        "mode": "generated-inspectable-corpus",
        "corpus": {
            "source": "privatelens.demo",
            "asset_count": len(MODEL_CASES),
            "contains_private_media": False,
        },
        "models": {
            "clip": clip_id,
            "ocr": f"rapidocr-onnxruntime:{_package_version('rapidocr-onnxruntime')}",
            "vlm": vlm_model,
        },
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "sqlite": sqlite3.sqlite_version,
            "sqlite_vec": _package_version("sqlite-vec"),
            "vector_backend": vector_backend,
            "vlm_available": vlm_available,
        },
        "summary": summary,
        "timings_ms": timings,
        "cases": cases,
    }


def _seed_model_index(
    image_vectors: dict[str, np.ndarray | None],
    ocr_blocks: dict[str, list[dict[str, Any]]],
    clip_id: str,
):
    engine = init_db()
    asset_ids: dict[str, int] = {}
    with Session(engine) as session:
        for index, case in enumerate(MODEL_CASES, 1):
            filename = case["filename"]
            asset = Asset(
                path=filename,
                sha256=f"model-benchmark-{index:02d}",
                media_type="image",
            )
            session.add(asset)
            session.flush()
            asset_ids[filename] = asset.id

            vector = image_vectors[filename]
            if vector is not None:
                session.add(
                    ImageEmbedding(
                        asset_id=asset.id,
                        model=clip_id,
                        vector=vector.tobytes(),
                    )
                )
            for block in ocr_blocks[filename]:
                text_value = str(block.get("text", "")).strip()
                if text_value:
                    session.add(
                        OcrBlock(
                            asset_id=asset.id,
                            text=text_value,
                            confidence=block.get("confidence"),
                        )
                    )
        session.commit()
    return engine, asset_ids


def _validated_vector(
    vector: Any,
    errors: dict[str, str],
    field: str,
) -> np.ndarray | None:
    if vector is None:
        errors[field] = "no vector returned"
        return None
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size != VECTOR_DIMENSIONS:
        errors[field] = f"expected {VECTOR_DIMENSIONS} dimensions, received {array.size}"
        return None
    if not np.isfinite(array).all():
        errors[field] = "vector contains non-finite values"
        return None
    return array


def _summarize(cases: list[dict[str, Any]], include_vlm: bool) -> dict[str, Any]:
    case_count = len(cases)
    clip_top1_rate = sum(case["clip"]["top_1"] for case in cases) / case_count
    ocr_top1_rate = sum(case["ocr"]["top_1"] for case in cases) / case_count
    vlm_accuracy = None
    caption_recall = None
    vlm_passed = True
    if include_vlm:
        vlm_accuracy = sum(case["vlm"]["classification_correct"] for case in cases) / case_count
        caption_recall = sum(case["vlm"]["caption_term_recall"] for case in cases) / case_count
        vlm_passed = (
            vlm_accuracy >= MINIMUM_VLM_CLASSIFICATION_ACCURACY
            and caption_recall >= MINIMUM_VLM_CAPTION_TERM_RECALL
        )

    passed = (
        clip_top1_rate >= MINIMUM_CLIP_TOP1_RATE
        and ocr_top1_rate >= MINIMUM_OCR_TOP1_RATE
        and vlm_passed
    )
    return {
        "case_count": case_count,
        "clip_top1_rate": round(clip_top1_rate, 4),
        "ocr_top1_rate": round(ocr_top1_rate, 4),
        "vlm_classification_accuracy": (
            round(vlm_accuracy, 4) if vlm_accuracy is not None else None
        ),
        "vlm_caption_term_recall": (
            round(caption_recall, 4) if caption_recall is not None else None
        ),
        "minimum_clip_top1_rate": MINIMUM_CLIP_TOP1_RATE,
        "minimum_ocr_top1_rate": MINIMUM_OCR_TOP1_RATE,
        "minimum_vlm_classification_accuracy": (
            MINIMUM_VLM_CLASSIFICATION_ACCURACY if include_vlm else None
        ),
        "minimum_vlm_caption_term_recall": (
            MINIMUM_VLM_CAPTION_TERM_RECALL if include_vlm else None
        ),
        "vlm_required": include_vlm,
        "passed": passed,
    }


def _rank_for_asset(results: list[dict[str, Any]], asset_id: int) -> int | None:
    return next(
        (rank for rank, result in enumerate(results, 1) if result["asset_id"] == asset_id),
        None,
    )


def _vector_backend(engine) -> str:
    try:
        with engine.connect() as conn:
            vec_version = conn.execute(text("SELECT vec_version()")).scalar_one()
        return f"sqlite-vec:{vec_version}"
    except Exception:
        return "blob-fallback"


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _notify(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
