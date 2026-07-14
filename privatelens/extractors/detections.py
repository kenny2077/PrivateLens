"""Derive lightweight search detections from signals already produced during indexing."""

from dataclasses import dataclass
import math
import re
from typing import Any


@dataclass(frozen=True)
class DerivedDetection:
    """A canonical label ready to persist as a ``Detection`` row."""

    label: str
    confidence: float
    source_model: str


_DOCUMENT_TYPES = {"academic", "financial", "medical"}
_SENSITIVE_ALIASES = {
    "driver_license": ("id_card", "document"),
    "passport": ("id_card", "document"),
    "id_card": ("id_card", "document"),
    "receipt": ("receipt", "document"),
    "bank_card": ("document",),
    "ssn": ("document",),
    "medical": ("medical", "document"),
    "academic": ("academic", "document"),
    "immigration": ("document",),
}
_VLM_ALIASES = {
    "receipt": ("receipt", "invoice"),
    "id_card": (
        "id card",
        "identity card",
        "identification card",
        "driver license",
        "drivers license",
        "passport",
    ),
    "screenshot": ("screenshot", "screen capture"),
    "whiteboard": ("whiteboard",),
    "pet": ("pet", "dog", "cat", "puppy", "kitten", "animal"),
    "car": ("car", "vehicle", "automobile", "dashboard", "license plate"),
    "document": ("document", "paper form"),
}
_DOCUMENT_LABELS = {"receipt", "id_card", "whiteboard", "document", *_DOCUMENT_TYPES}


def derive_detections(
    *,
    media_type: str | None,
    ocr_blocks: list[dict[str, Any]] | None = None,
    document_classification: dict[str, Any] | None = None,
    sensitive_detection: dict[str, Any] | None = None,
    vlm_classification: dict[str, Any] | None = None,
    vlm_caption: str | None = None,
    vlm_model: str | None = None,
) -> list[DerivedDetection]:
    """Return canonical, confidence-deduplicated labels from existing index signals."""
    detections: dict[str, DerivedDetection] = {}

    def add(label: str, confidence: Any, source_model: str) -> None:
        normalized = _normalize_label(label)
        if not normalized:
            return
        score = _bounded_confidence(confidence)
        if normalized == "document" and score == 0.0:
            return
        current = detections.get(normalized)
        if current is None or score > current.confidence:
            detections[normalized] = DerivedDetection(normalized, score, source_model)

    normalized_media_type = _normalize_label(media_type or "")
    if normalized_media_type in {"document", "screenshot"}:
        add(normalized_media_type, 0.9, "media-type")

    populated_ocr = [block for block in (ocr_blocks or []) if str(block.get("text", "")).strip()]
    if populated_ocr:
        confidences = [
            _bounded_confidence(block.get("confidence"))
            for block in populated_ocr
            if block.get("confidence") is not None
        ]
        add("text", max(confidences, default=0.9), "rapidocr")

    if document_classification:
        document_type = _canonical_document_type(document_classification.get("type"))
        confidence = document_classification.get("confidence", 0.5)
        if (
            document_type
            and document_type != "unknown"
            and (document_type != "document" or normalized_media_type == "document")
        ):
            add(document_type, confidence, "document-classifier")
            add("document", confidence, "document-classifier")

    if sensitive_detection:
        sensitive_type = _normalize_label(str(sensitive_detection.get("type", "")))
        confidence = sensitive_detection.get("confidence", 0.7)
        source = f"sensitive:{sensitive_detection.get('source', 'heuristic')}"
        for label in _SENSITIVE_ALIASES.get(sensitive_type, ("document",)):
            add(label, confidence, source)

    vlm_source = f"vlm:{vlm_model or 'local'}"
    if vlm_classification:
        confidence = vlm_classification.get("confidence", 0.7)
        evidence = " ".join(str(vlm_classification.get(key, "")) for key in ("type", "description"))
        for label in _vlm_labels(evidence):
            add(label, confidence, vlm_source)
            if label in _DOCUMENT_LABELS:
                add("document", confidence, vlm_source)

    for label in _vlm_labels(vlm_caption or ""):
        add(label, 0.65, vlm_source)
        if label in _DOCUMENT_LABELS:
            add("document", 0.65, vlm_source)

    return list(detections.values())


def _canonical_document_type(value: Any) -> str:
    normalized = _normalize_label(str(value or ""))
    if normalized in {"driver_license", "passport"}:
        return "id_card"
    return normalized


def _vlm_labels(value: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    labels = []
    for label, aliases in _VLM_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
            labels.append(label)
    return labels


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _bounded_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    return min(max(confidence, 0.0), 1.0)
