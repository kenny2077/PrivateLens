"""Deterministic structured-search quality benchmark."""

import json
import tempfile
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from privatelens.config import settings
from privatelens.db.schema import Asset, Detection, Face, OcrBlock, init_db
from privatelens.search.engine import SearchEngine


def load_canonical_manifest() -> dict[str, Any]:
    """Load the packaged ten-case benchmark fixture."""
    resource = files("privatelens.benchmarks").joinpath("search-quality-v1.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def run_canonical_benchmark(output_path: Path | None = None) -> dict[str, Any]:
    """Seed a temporary index, run all canonical cases, and return metrics."""
    manifest = load_canonical_manifest()
    old_data_dir = settings.data_dir
    old_db_path = settings.db_path

    try:
        with tempfile.TemporaryDirectory(prefix="privatelens-benchmark-") as tmpdir:
            settings.data_dir = Path(tmpdir)
            settings.db_path = None
            _seed_assets(manifest["assets"])
            report = _run_cases(manifest)
    finally:
        settings.data_dir = old_data_dir
        settings.db_path = old_db_path

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _seed_assets(asset_specs: list[dict[str, Any]]) -> None:
    engine = init_db()
    with Session(engine) as session:
        for index, spec in enumerate(asset_specs, 1):
            captured_at = datetime.fromisoformat(spec["exif_datetime"])
            asset = Asset(
                path=spec["path"],
                sha256=f"benchmark-{index:02d}",
                media_type=spec["media_type"],
                width=spec["width"],
                height=spec["height"],
                is_sensitive=spec["is_sensitive"],
                exif_datetime=captured_at,
                created_at=captured_at,
            )
            session.add(asset)
            session.flush()

            for text_value in spec["ocr"]:
                session.add(OcrBlock(asset_id=asset.id, text=text_value, confidence=0.99))
            for detection in spec["detections"]:
                session.add(
                    Detection(
                        asset_id=asset.id,
                        label=detection["label"],
                        confidence=detection["confidence"],
                        source_model="benchmark-fixture",
                    )
                )
            for _ in range(spec["faces"]):
                session.add(Face(asset_id=asset.id, bbox="[]", confidence=0.99))
        session.commit()


def _run_cases(manifest: dict[str, Any]) -> dict[str, Any]:
    top_k = int(manifest["top_k"])
    engine = SearchEngine()
    case_reports = []

    for case in manifest["cases"]:
        results = engine.search_by_recipe(
            case["recipe"],
            case["query"],
            limit=top_k,
            rerank=False,
            use_semantic=False,
        )
        case_reports.append(_score_case(case, results, top_k))

    case_count = len(case_reports)
    hit_rate = sum(case["target_in_top_5"] for case in case_reports) / case_count
    summary = {
        "case_count": case_count,
        "top_k": top_k,
        "hit_rate_at_5": round(hit_rate, 4),
        "mean_precision_at_5": round(
            sum(case["precision_at_5"] for case in case_reports) / case_count,
            4,
        ),
        "mean_recall_at_5": round(
            sum(case["recall_at_5"] for case in case_reports) / case_count,
            4,
        ),
        "mean_reciprocal_rank": round(
            sum(case["reciprocal_rank"] for case in case_reports) / case_count,
            4,
        ),
        "minimum_hit_rate_at_5": manifest["minimum_hit_rate_at_5"],
        "passed": hit_rate >= manifest["minimum_hit_rate_at_5"],
    }
    return {
        "benchmark": manifest["benchmark"],
        "version": manifest["version"],
        "mode": "structured-signals-no-clip-no-vlm",
        "summary": summary,
        "cases": case_reports,
    }


def _score_case(
    case: dict[str, Any],
    results: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    relevant = set(case["relevant_paths"])
    result_paths = [result["path"] for result in results]
    hits = sum(path in relevant for path in result_paths)
    first_rank = next(
        (rank for rank, path in enumerate(result_paths, 1) if path in relevant),
        None,
    )
    return {
        "name": case["name"],
        "recipe": case["recipe"],
        "query": case["query"],
        "relevant_paths": sorted(relevant),
        "target_in_top_5": first_rank is not None,
        "first_relevant_rank": first_rank,
        "precision_at_5": round(hits / top_k, 4),
        "recall_at_5": round(hits / len(relevant), 4),
        "reciprocal_rank": round(1 / first_rank, 4) if first_rank else 0.0,
        "results": [
            {
                "rank": rank,
                "path": result["path"],
                "score": round(float(result["score"]), 6),
                "explanation": result.get("explanation", ""),
                "signals": [signal["source"] for signal in result.get("signals", [])],
            }
            for rank, result in enumerate(results, 1)
        ],
    }
