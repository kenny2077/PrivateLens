"""OCR text extraction using RapidOCR."""

import logging
from pathlib import Path
from typing import Any

from privatelens.utils.image_formats import register_image_formats


logger = logging.getLogger(__name__)

register_image_formats()


class OcrExtractor:
    """Extract text from images using RapidOCR."""

    def __init__(self):
        self._engine: Any = None

    def _load_engine(self):
        """Lazy load RapidOCR engine."""
        if self._engine is not None:
            return

        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def extract(self, image_path: Path) -> list[dict[str, Any]] | None:
        """Extract text blocks from image.

        Returns list of dicts with keys: text, bbox, confidence
        """
        try:
            self._load_engine()

            result = self._engine(str(image_path))
            if not result or not result[0]:
                return None

            blocks = []
            for item in result[0]:
                # RapidOCR returns: [bbox, text, confidence]
                bbox, text, confidence = item
                blocks.append(
                    {
                        "text": text,
                        "bbox": bbox,
                        "confidence": confidence,
                    }
                )

            return blocks
        except Exception as e:
            logger.warning("OCR failed for %s: %s", image_path, e)
            return None
