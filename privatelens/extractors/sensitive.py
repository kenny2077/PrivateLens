"""Sensitive content detector."""

import re
from pathlib import Path
from typing import Any


class SensitiveDetector:
    """Detect sensitive documents and photos."""

    SENSITIVE_KEYWORDS = {
        "driver_license": ["driver license", "driver's license", "dl", "driving permit"],
        "passport": ["passport", "travel document", "visa"],
        "id_card": ["identification", "national id", "id card", "resident card"],
        "bank_card": ["credit card", "debit card", "bank card", "card number", "cvv"],
        "ssn": ["social security", "ssn", "national insurance", "tax id"],
        "receipt": ["receipt", "invoice", "transaction"],
        "medical": ["prescription", "diagnosis", "medical record", "patient"],
        "academic": ["transcript", "diploma", "certificate", "grade report"],
        "immigration": ["i-20", "i20", "ds-2019", "sevis", "green card", "permanent resident"],
    }

    SENSITIVE_FILENAMES = {
        "driver_license": ["license", "dl", "driving"],
        "passport": ["passport", "visa"],
        "id_card": ["id", "identification"],
        "bank_card": ["card", "bank", "credit"],
        "ssn": ["ssn", "social"],
        "receipt": ["receipt", "invoice"],
        "medical": ["medical", "health", "prescription"],
        "academic": ["transcript", "diploma", "certificate"],
        "immigration": ["i20", "i-20", "visa", "green_card"],
    }

    def detect(self, image_path: Path, ocr_text: list[dict] | None = None) -> dict | None:
        """Detect if image contains sensitive content."""
        name_lower = image_path.name.lower()
        name_tokens = set(re.split(r"[^a-z0-9]+", image_path.stem.lower()))

        # Check filename patterns
        for doc_type, patterns in self.SENSITIVE_FILENAMES.items():
            for pattern in patterns:
                if self._filename_matches(name_lower, name_tokens, pattern):
                    return {"type": doc_type, "confidence": 0.7, "source": "filename"}

        # Check OCR text
        if ocr_text:
            full_text = " ".join([block["text"] for block in ocr_text]).lower()
            for doc_type, keywords in self.SENSITIVE_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in full_text:
                        return {"type": doc_type, "confidence": 0.85, "source": "ocr"}

        return None

    def _filename_matches(self, name_lower: str, name_tokens: set[str], pattern: str) -> bool:
        """Match short filename patterns only as standalone tokens."""
        pattern = pattern.lower()
        if len(pattern) <= 2:
            return pattern in name_tokens
        return pattern in name_lower

    def is_sensitive(self, image_path: Path, ocr_text: list[dict] | None = None) -> bool:
        """Quick check if image is sensitive."""
        return self.detect(image_path, ocr_text) is not None
