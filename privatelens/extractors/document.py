"""Document type classifier."""

import re
from pathlib import Path
from typing import Any


class DocumentClassifier:
    """Classify images as documents based on heuristics."""

    DOCUMENT_KEYWORDS = [
        "document",
        "paper",
        "form",
        "contract",
        "letter",
        "certificate",
        "license",
        "permit",
        "id",
        "passport",
        "receipt",
        "invoice",
        "bill",
        "statement",
        "report",
    ]

    def is_document(self, image_path: Path, exif_data: dict[str, Any] | None = None) -> bool:
        """Check if image is likely a document."""
        # Check aspect ratio (documents are usually A4/Letter ratio)
        if exif_data and exif_data.get("width") and exif_data.get("height"):
            w, h = exif_data["width"], exif_data["height"]
            ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
            # A4 is ~1.414, Letter is ~1.294
            has_camera_exif = bool(exif_data.get("make") or exif_data.get("model"))
            if not has_camera_exif and 1.2 <= ratio <= 1.6:
                return True

        # Check filename
        normalized_name = re.sub(r"[^a-z0-9]+", " ", image_path.stem.casefold()).strip()
        for keyword in self.DOCUMENT_KEYWORDS:
            if re.search(
                rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])",
                normalized_name,
            ):
                return True

        return False

    def classify(self, image_path: Path, ocr_text: list[dict] | None = None) -> dict:
        """Classify document type based on OCR and heuristics."""
        if not ocr_text:
            return {"type": "unknown", "confidence": 0.0}

        full_text = " ".join([block["text"] for block in ocr_text]).lower()

        # ID documents
        id_keywords = ["driver license", "identification", "id card", "passport", "national id"]
        for kw in id_keywords:
            if kw in full_text:
                return {"type": "id_card", "confidence": 0.9}

        # Receipts
        receipt_keywords = ["receipt", "total", "tax", "change", "transaction", "payment"]
        if any(kw in full_text for kw in receipt_keywords):
            return {"type": "receipt", "confidence": 0.85}

        # Financial
        financial_keywords = ["invoice", "bill", "statement", "balance", "due", "account"]
        if any(kw in full_text for kw in financial_keywords):
            return {"type": "financial", "confidence": 0.8}

        # Medical
        medical_keywords = ["prescription", "diagnosis", "patient", "medical", "health"]
        if any(kw in full_text for kw in medical_keywords):
            return {"type": "medical", "confidence": 0.8}

        # Academic
        academic_keywords = ["transcript", "diploma", "certificate", "grade", "university"]
        if any(kw in full_text for kw in academic_keywords):
            return {"type": "academic", "confidence": 0.75}

        return {"type": "document", "confidence": 0.5}
