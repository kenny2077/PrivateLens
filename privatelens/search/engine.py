"""Hybrid search engine combining multiple signals."""

import json
import re
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

import numpy as np

from privatelens.config import settings
from privatelens.db.schema import init_db
from privatelens.extractors.clip import ClipExtractor
from privatelens.search.queries import SearchQueries
from privatelens.search.evidence import EvidenceBuilder
from privatelens.search.reranker import VlmReranker


POSITIVE_FEEDBACK_STEP = 0.03
MAX_POSITIVE_FEEDBACK_BOOST = 0.05


class SearchEngine:
    """Multi-signal hybrid search engine."""

    def __init__(self):
        self.clip_extractor = ClipExtractor()
        self.engine = init_db()
        self.queries = SearchQueries()
        self.evidence = EvidenceBuilder()
        self.reranker = VlmReranker()
        self.last_search_event_id: int | None = None

    def search(
        self,
        query: str,
        search_type: str = "smart",
        limit: int = 50,
        record_event: bool = False,
    ) -> list[dict[str, Any]]:
        """Search photos using specified strategy."""
        self.last_search_event_id = None
        started_at = perf_counter()
        if search_type == "smart":
            results = self._smart_search(query, limit)
        elif search_type == "ocr":
            results = self._ocr_search(query, limit)
        elif search_type == "face":
            results = self._face_search(query, limit)
        elif search_type == "metadata":
            results = self._metadata_search(query, limit)
        elif search_type == "path":
            results = self._path_search(query, limit)
        else:
            results = self._smart_search(query, limit)

        if record_event:
            self.last_search_event_id = self._record_search_event(
                query,
                search_type,
                len(results),
                started_at,
            )
        return results

    def _smart_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Hybrid search: CLIP semantic + OCR text + metadata."""
        if self.queries.asset_count() == 0:
            return []

        candidates: dict[int, float] = {}

        # Signal 1: CLIP semantic search (BLOB fallback if sqlite-vec unavailable)
        query_embedding = self.clip_extractor.encode_text(query)
        if query_embedding is not None:
            vec_results = self._vector_search(query_embedding, limit=limit * 3)
            for r in vec_results:
                candidates[r["asset_id"]] = candidates.get(r["asset_id"], 0.0) + r["score"] * 0.4

        # Signal 2: OCR text search (FTS5)
        ocr_results = self.queries.ocr_fts_search(query, limit=limit * 3)
        for asset_id, text, rank in ocr_results:
            candidates[asset_id] = candidates.get(asset_id, 0.0) + 0.3

        # Signal 3: Caption text search (FTS5)
        caption_results = self.queries.caption_fts_search(query, limit=limit * 3)
        for asset_id, caption, rank in caption_results:
            candidates[asset_id] = candidates.get(asset_id, 0.0) + 0.2

        # Signal 4: Filename/path search
        path_results = self.queries.path_fts_search(query, limit=limit * 3)
        for asset_id, path, rank in path_results:
            candidates[asset_id] = candidates.get(asset_id, 0.0) + 0.1

        self._apply_positive_feedback_boost(candidates)

        # Get asset details and build results
        results = []
        asset_ids = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)[:limit]
        assets = self.queries.get_assets_by_ids(asset_ids)
        score_scale = self._score_scale(candidates, asset_ids)

        for asset_id in asset_ids:
            asset = assets.get(asset_id)
            if asset:
                evidence = self.evidence.build(
                    asset_id,
                    query,
                    self._bounded_score(candidates[asset_id] / score_scale),
                )
                results.append(evidence)

        return results

    def _vector_search(self, embedding: np.ndarray, limit: int) -> list[dict]:
        """Search by vector similarity using BLOB fallback."""
        import struct
        from sqlalchemy import text

        # Try sqlite-vec first
        try:
            with self.engine.connect() as conn:
                vec_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                results = conn.execute(
                    text("""
                    SELECT asset_id, distance
                    FROM vec_image_embeddings
                    WHERE embedding MATCH :vec
                      AND k = :limit
                    ORDER BY distance
                    """),
                    {"vec": vec_bytes, "limit": limit},
                ).fetchall()
                if results:
                    return [
                        {
                            "asset_id": row.asset_id,
                            "score": self._bounded_score(1.0 - row.distance),
                        }
                        for row in results
                    ]
        except Exception:
            pass

        return self._blob_vector_search(embedding, limit)

    def _blob_vector_search(self, embedding: np.ndarray, limit: int) -> list[dict]:
        """Search every BLOB embedding when the native vector table is unavailable."""
        from sqlalchemy.orm import Session
        from privatelens.db.schema import ImageEmbedding

        with Session(self.engine) as session:
            embeddings = session.query(ImageEmbedding).yield_per(500)
            results = []
            for emb in embeddings:
                try:
                    vec = np.frombuffer(emb.vector, dtype=np.float32)
                    if vec.shape[0] != embedding.shape[0]:
                        continue
                    denominator = np.linalg.norm(embedding) * np.linalg.norm(vec)
                    if denominator == 0:
                        continue
                    sim = np.dot(embedding, vec) / denominator
                    results.append(
                        {"asset_id": emb.asset_id, "score": self._bounded_score(float(sim))}
                    )
                except Exception:
                    continue

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]

    def _ocr_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search by OCR text content."""
        ocr_results = self.queries.ocr_fts_search(query, limit=limit)
        asset_ids = [r[0] for r in ocr_results]
        assets = self.queries.get_assets_by_ids(asset_ids)

        return [
            {
                "asset_id": r[0],
                "path": assets[r[0]].path if r[0] in assets else "",
                "score": 0.9,
                "explanation": f"OCR match: '{r[1][:100]}'",
                "thumbnail": assets[r[0]].thumbnail_path if r[0] in assets else None,
                "media_type": assets[r[0]].media_type if r[0] in assets else "unknown",
                "is_sensitive": assets[r[0]].is_sensitive if r[0] in assets else False,
            }
            for r in ocr_results
        ]

    def _face_search(self, person_name: str, limit: int) -> list[dict[str, Any]]:
        """Search by person name."""
        person = self.queries.get_person_by_name(person_name)
        if not person:
            return []

        faces = self.queries.get_faces_by_person(person.id, limit=limit)
        asset_ids = [f.asset_id for f in faces]
        assets = self.queries.get_assets_by_ids(asset_ids)

        return [
            {
                "asset_id": f.asset_id,
                "path": assets[f.asset_id].path if f.asset_id in assets else "",
                "score": self._bounded_score(f.confidence or 0.8),
                "explanation": f"Contains person: {person.display_name}",
                "thumbnail": assets[f.asset_id].thumbnail_path if f.asset_id in assets else None,
                "media_type": assets[f.asset_id].media_type if f.asset_id in assets else "unknown",
                "is_sensitive": assets[f.asset_id].is_sensitive if f.asset_id in assets else False,
            }
            for f in faces
        ]

    def _path_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search by filename or path without loading ML models."""
        path_results = self.queries.path_fts_search(query, limit=limit)
        results = []
        for asset_id, _path, _rank in path_results:
            results.append(self.evidence.build(asset_id, query, 0.6))
        return results

    def _metadata_search(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search by metadata (date, location, camera)."""
        from sqlalchemy import and_, or_
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset

        filters = filters or {}
        with Session(self.engine) as session:
            db_query = session.query(Asset)
            applied_structured_filter = False
            if "media_type" in filters:
                db_query = db_query.filter(Asset.media_type == filters["media_type"])
                applied_structured_filter = True
            if "is_sensitive" in filters:
                db_query = db_query.filter(Asset.is_sensitive.is_(bool(filters["is_sensitive"])))
                applied_structured_filter = True
            if filters.get("date_range"):
                date_range = self._date_range_from_query(query)
                if date_range is None:
                    return []
                start, end = date_range
                db_query = db_query.filter(
                    or_(
                        and_(Asset.exif_datetime >= start, Asset.exif_datetime < end),
                        and_(
                            Asset.exif_datetime.is_(None),
                            Asset.created_at >= start,
                            Asset.created_at < end,
                        ),
                    )
                )
                applied_structured_filter = True
            if "aspect_ratio" in filters:
                aspect_ratio = filters["aspect_ratio"]
                min_ratio = aspect_ratio.get("min")
                max_ratio = aspect_ratio.get("max")
                ratio = Asset.width * 1.0 / Asset.height
                db_query = db_query.filter(
                    Asset.width.isnot(None),
                    Asset.height.isnot(None),
                    Asset.height > 0,
                )
                if min_ratio is not None:
                    db_query = db_query.filter(ratio >= float(min_ratio))
                if max_ratio is not None:
                    db_query = db_query.filter(ratio <= float(max_ratio))
                applied_structured_filter = True
            if not applied_structured_filter:
                db_query = db_query.filter(
                    Asset.path.ilike(f"%{query}%")
                    | Asset.exif_make.ilike(f"%{query}%")
                    | Asset.exif_model.ilike(f"%{query}%")
                )
            assets = db_query.limit(limit).all()

            return [
                {
                    "asset_id": a.id,
                    "path": a.path,
                    "score": 0.7,
                    "explanation": self._metadata_explanation(a),
                    "thumbnail": a.thumbnail_path,
                    "media_type": a.media_type,
                    "is_sensitive": a.is_sensitive,
                }
                for a in assets
            ]

    def _metadata_explanation(self, asset: Any) -> str:
        """Build a concise metadata explanation for search output."""
        camera = " ".join(part for part in [asset.exif_make, asset.exif_model] if part)
        if camera:
            return f"Metadata match: {camera}"
        if asset.media_type:
            return f"Metadata match: {asset.media_type}"
        return "Metadata match"

    def _date_range_from_query(self, query: str) -> tuple[datetime, datetime] | None:
        """Parse an explicit YYYY, YYYY-MM, or YYYY-MM-DD date from a query."""
        day_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", query)
        if day_match:
            try:
                start = datetime(
                    int(day_match.group(1)),
                    int(day_match.group(2)),
                    int(day_match.group(3)),
                )
            except ValueError:
                return None
            return start, start + timedelta(days=1)

        month_match = re.search(r"\b(\d{4})-(\d{2})\b", query)
        if month_match:
            year = int(month_match.group(1))
            month = int(month_match.group(2))
            if month < 1 or month > 12:
                return None
            start = datetime(year, month, 1)
            if month == 12:
                return start, datetime(year + 1, 1, 1)
            return start, datetime(year, month + 1, 1)

        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", query)
        if not year_match:
            return None
        year = int(year_match.group(1))
        return datetime(year, 1, 1), datetime(year + 1, 1, 1)

    def search_by_recipe(
        self,
        recipe_name: str,
        query: str,
        limit: int = 50,
        rerank: bool = True,
        use_semantic: bool = True,
        record_event: bool = False,
    ) -> list[dict[str, Any]]:
        """Search using a predefined recipe."""
        from privatelens.search.recipes import get_recipe, init_recipes

        self.last_search_event_id = None
        recipe = get_recipe(recipe_name)
        if not recipe:
            init_recipes()
            recipe = get_recipe(recipe_name)
        if not recipe:
            return self.search(query, limit=limit, record_event=record_event)

        plan = json.loads(recipe.query_plan)
        results = self._execute_plan(plan, query, limit, use_semantic=use_semantic)

        # Apply VLM rerank if configured
        rerank_config = plan.get("rerank")
        if rerank and rerank_config and rerank_config.get("vlm_prompt"):
            results = self.reranker.rerank(
                results,
                rerank_config["vlm_prompt"],
                top_k=rerank_config.get("top_k", 30),
            )

        if record_event:
            self.last_search_event_id = self._record_search_event(
                query,
                recipe_name,
                len(results),
                None,
            )
        return results

    def record_feedback(
        self,
        event_id: int | None,
        feedback: int,
        result_clicked: int | None = None,
    ) -> None:
        """Record user feedback for an existing search event."""
        if event_id is None:
            return

        from sqlalchemy.orm import Session
        from privatelens.db.schema import SearchEvent

        with Session(self.engine) as session:
            event = session.get(SearchEvent, event_id)
            if event is None:
                return
            event.feedback = feedback
            event.result_clicked = result_clicked
            session.commit()

    def _execute_plan(
        self,
        plan: dict,
        query: str,
        limit: int,
        use_semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute a structured search plan."""
        if self.queries.asset_count() == 0:
            return []

        candidates: dict[int, float] = {}

        for signal in plan.get("signals", []):
            signal_type = signal["type"]
            weight = signal.get("weight", 0.1)

            if signal_type == "ocr":
                matched_asset_ids = []
                seen_asset_ids = set()
                for term in self._signal_text_terms(query, signal):
                    ocr_results = self._ocr_search(term, limit=limit * 2)
                    for r in ocr_results:
                        asset_id = r["asset_id"]
                        if asset_id in seen_asset_ids:
                            continue
                        seen_asset_ids.add(asset_id)
                        matched_asset_ids.append(asset_id)
                for asset_id in matched_asset_ids:
                    candidates[asset_id] = candidates.get(asset_id, 0.0) + weight

            elif signal_type == "semantic" and use_semantic:
                emb = self.clip_extractor.encode_text(query)
                if emb is not None:
                    semantic_results = self._vector_search(emb, limit=limit * 2)
                    for r in semantic_results:
                        candidates[r["asset_id"]] = (
                            candidates.get(r["asset_id"], 0.0) + r["score"] * weight
                        )

            elif signal_type == "face":
                if "face_count" in signal:
                    face_count_results = self.queries.face_count_search(
                        int(signal["face_count"]),
                        limit=limit * 2,
                    )
                    for asset_id, _face_count in face_count_results:
                        candidates[asset_id] = candidates.get(asset_id, 0.0) + weight
                else:
                    for name in signal.get("names", [query]):
                        face_results = self._face_search(name, limit=limit * 2)
                        for r in face_results:
                            candidates[r["asset_id"]] = candidates.get(r["asset_id"], 0.0) + weight

            elif signal_type == "metadata":
                metadata_results = self._metadata_search(
                    query,
                    limit=limit * 2,
                    filters=signal.get("filters"),
                )
                for r in metadata_results:
                    candidates[r["asset_id"]] = candidates.get(r["asset_id"], 0.0) + weight

            elif signal_type == "path":
                matched_asset_ids = []
                seen_asset_ids = set()
                for term in self._signal_text_terms(query, signal):
                    path_results = self.queries.path_fts_search(term, limit=limit * 2)
                    for asset_id, _path, _rank in path_results:
                        if asset_id in seen_asset_ids:
                            continue
                        seen_asset_ids.add(asset_id)
                        matched_asset_ids.append(asset_id)
                for asset_id in matched_asset_ids:
                    candidates[asset_id] = candidates.get(asset_id, 0.0) + weight

            elif signal_type == "detection":
                detection_results = self.queries.detection_label_search(
                    signal.get("labels", []),
                    query=query,
                    limit=limit * 2,
                )
                for asset_id, _label, confidence in detection_results:
                    signal_score = weight * (confidence if confidence is not None else 1.0)
                    candidates[asset_id] = candidates.get(asset_id, 0.0) + signal_score

        # Get asset details with evidence
        results = []
        filters = dict(plan.get("filters", {}))
        if "face_count_exact" in filters and not self.queries.has_face_data():
            filters.pop("face_count_exact")
        assets = self.queries.get_assets_by_ids(list(candidates.keys()))
        matching_asset_ids = [
            asset_id
            for asset_id in candidates
            if asset_id in assets and self._asset_matches_filters(assets[asset_id], filters)
        ]
        if filters.get("boost_recent"):
            matching_assets = {asset_id: assets[asset_id] for asset_id in matching_asset_ids}
            self._apply_recent_boost(candidates, matching_assets)
        self._apply_positive_feedback_boost(candidates, matching_asset_ids)

        asset_ids = sorted(
            matching_asset_ids,
            key=lambda x: candidates[x],
            reverse=True,
        )[:limit]
        score_scale = self._score_scale(candidates, asset_ids)

        for asset_id in asset_ids:
            asset = assets.get(asset_id)
            if asset:
                evidence = self.evidence.build(
                    asset_id,
                    query,
                    self._bounded_score(candidates[asset_id] / score_scale),
                )
                results.append(evidence)

        return results

    def _signal_text_terms(self, query: str, signal: dict[str, Any]) -> list[str]:
        """Return unique text terms for a recipe signal."""
        raw_terms = [query, *signal.get("keywords", [])]
        terms = []
        seen = set()
        for term in raw_terms:
            if not isinstance(term, str):
                continue
            cleaned = term.strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(cleaned)
        return terms

    @staticmethod
    def _bounded_score(score: float) -> float:
        """Clamp externally visible scores to the documented zero-to-one range."""
        if not np.isfinite(score):
            return 0.0
        return min(max(float(score), 0.0), 1.0)

    @staticmethod
    def _score_scale(candidates: dict[int, float], asset_ids: list[int]) -> float:
        """Return a scale that preserves relative scores without inflating sub-one values."""
        finite_scores = [
            candidates[asset_id] for asset_id in asset_ids if np.isfinite(candidates[asset_id])
        ]
        return max(1.0, max(finite_scores, default=1.0))

    def _asset_matches_filters(self, asset: Any, filters: dict[str, Any]) -> bool:
        """Return whether an asset record satisfies recipe-level filters."""
        if filters.get("require_sensitive") and not asset.is_sensitive:
            return False

        excluded_types = set(filters.get("exclude", []))
        if asset.media_type in excluded_types:
            return False

        if "face_count_exact" in filters:
            expected_count = int(filters["face_count_exact"])
            if self.queries.asset_face_count(asset.id) != expected_count:
                return False

        return True

    def _apply_recent_boost(
        self,
        candidates: dict[int, float],
        assets: dict[int, Any],
    ) -> None:
        """Apply a small score boost based on capture date recency."""
        dated_assets = [
            (asset_id, asset.exif_datetime or asset.created_at)
            for asset_id, asset in assets.items()
            if asset.exif_datetime or asset.created_at
        ]
        if not dated_assets:
            return

        timestamps = [date.timestamp() for _, date in dated_assets]
        oldest = min(timestamps)
        newest = max(timestamps)
        if oldest == newest:
            for asset_id, _date in dated_assets:
                candidates[asset_id] = candidates.get(asset_id, 0.0) + 0.1
            return

        span = newest - oldest
        for asset_id, date in dated_assets:
            recency = (date.timestamp() - oldest) / span
            candidates[asset_id] = candidates.get(asset_id, 0.0) + 0.1 * recency

    def _apply_positive_feedback_boost(
        self,
        candidates: dict[int, float],
        asset_ids: list[int] | None = None,
    ) -> None:
        """Apply a small capped bonus for assets with explicit positive clicks."""
        eligible_ids = list(candidates) if asset_ids is None else asset_ids
        for asset_id, positive_count in self.queries.positive_feedback_counts(eligible_ids).items():
            boost = min(MAX_POSITIVE_FEEDBACK_BOOST, positive_count * POSITIVE_FEEDBACK_STEP)
            candidates[asset_id] = candidates.get(asset_id, 0.0) + boost

    def _record_search_event(
        self,
        query: str,
        query_type: str,
        results_count: int,
        started_at: float | None,
    ) -> int:
        """Record a search event for future feedback/ranking work."""
        from sqlalchemy.orm import Session
        from privatelens.db.schema import SearchEvent

        elapsed_ms = None
        if started_at is not None:
            elapsed_ms = int((perf_counter() - started_at) * 1000)

        with Session(self.engine) as session:
            event = SearchEvent(
                query=query,
                query_type=query_type,
                results_count=results_count,
                time_to_result_ms=elapsed_ms,
            )
            session.add(event)
            session.flush()
            event_id = event.id
            session.commit()
            return event_id
