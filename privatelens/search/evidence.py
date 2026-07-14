"""Evidence card builder for search results."""

import re
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from privatelens.db.schema import get_engine, Asset, OcrBlock, Caption, Detection, Face, Person


class EvidenceBuilder:
    """Build human-readable explanations of why a result matched."""

    def build(self, asset_id: int, query: str, score: float) -> dict[str, Any]:
        """Build evidence card for a search result.

        Returns:
            Dict with signals, explanation, and metadata
        """
        engine = get_engine()
        with Session(engine) as session:
            asset = session.query(Asset).filter_by(id=asset_id).first()
            if not asset:
                return {"explanation": "", "signals": []}

            signals: list[dict[str, Any]] = []
            path_terms = self._matching_path_terms(asset.path, query)
            if path_terms:
                signals.append(
                    {
                        "source": "path",
                        "text": Path(asset.path).name,
                        "terms": path_terms,
                        "confidence": 0.75,
                        "weight": 0.1,
                    }
                )

            # Check OCR
            ocr_blocks = session.query(OcrBlock).filter_by(asset_id=asset_id).all()
            if ocr_blocks:
                texts = [o.text for o in ocr_blocks if query.lower() in o.text.lower()]
                if texts:
                    signals.append(
                        {
                            "source": "ocr",
                            "text": texts[0][:100],
                            "confidence": 0.9,
                            "weight": 0.4,
                        }
                    )

            # Check captions
            captions = session.query(Caption).filter_by(asset_id=asset_id).all()
            if captions:
                signals.append(
                    {
                        "source": "caption",
                        "text": captions[0].caption[:100],
                        "confidence": captions[0].confidence or 0.8,
                        "weight": 0.2,
                    }
                )

            # Check faces/people
            faces = session.query(Face).filter_by(asset_id=asset_id).all()
            if faces:
                person_names = []
                for f in faces:
                    if f.cluster_id:
                        p = session.query(Person).filter_by(id=f.cluster_id).first()
                        if p and p.display_name:
                            person_names.append(p.display_name)
                if person_names:
                    signals.append(
                        {
                            "source": "face",
                            "people": person_names,
                            "confidence": 0.85,
                            "weight": 0.2,
                        }
                    )
                else:
                    signals.append(
                        {
                            "source": "face",
                            "face_count": len(faces),
                            "confidence": max(
                                (face.confidence or 0.0 for face in faces),
                                default=0.0,
                            ),
                            "weight": 0.2,
                        }
                    )

            # Check detections
            detections = session.query(Detection).filter_by(asset_id=asset_id).all()
            if detections:
                labels = [d.label for d in detections]
                signals.append(
                    {
                        "source": "detection",
                        "labels": labels[:5],
                        "confidence": 0.8,
                        "weight": 0.15,
                    }
                )

            # Semantic similarity fallback
            if not signals:
                signals.append(
                    {
                        "source": "semantic",
                        "score": score,
                        "weight": 1.0,
                    }
                )

            # Build explanation
            parts: list[str] = []
            for sig in signals:
                if sig["source"] == "path":
                    parts.append(f"Path contains '{', '.join(sig['terms'][:3])}'")
                elif sig["source"] == "ocr":
                    parts.append(f"OCR contains '{sig['text'][:50]}'")
                elif sig["source"] == "caption":
                    parts.append(f"Caption: {sig['text'][:50]}")
                elif sig["source"] == "face":
                    if "people" in sig:
                        parts.append(f"People: {', '.join(sig['people'])}")
                    else:
                        parts.append(f"Faces: {sig['face_count']}")
                elif sig["source"] == "detection":
                    parts.append(f"Detected: {', '.join(sig['labels'][:3])}")
                elif sig["source"] == "semantic":
                    parts.append(f"Semantic similarity: {sig['score']:.2f}")

            explanation = "; ".join(parts) if parts else f"Matched with score {score:.2f}"

            return {
                "asset_id": asset_id,
                "path": asset.path,
                "score": score,
                "signals": signals,
                "explanation": explanation,
                "thumbnail": asset.thumbnail_path,
                "media_type": asset.media_type,
                "is_sensitive": asset.is_sensitive,
                "metadata": {
                    "date": str(asset.exif_datetime) if asset.exif_datetime else None,
                    "dimensions": f"{asset.width}x{asset.height}"
                    if asset.width and asset.height
                    else None,
                    "file_size": asset.file_size,
                    "camera": f"{asset.exif_make} {asset.exif_model}"
                    if asset.exif_make or asset.exif_model
                    else None,
                },
            }

    def _matching_path_terms(self, path: str, query: str) -> list[str]:
        """Return query terms that appear in a normalized asset path."""
        normalized_path = self._normalize(path)
        normalized_query = self._normalize(query)
        if not normalized_query:
            return []
        if normalized_query in normalized_path:
            return [normalized_query]

        terms = [term for term in normalized_query.split() if len(term) >= 3]
        return [term for term in terms if term in normalized_path]

    def _normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
