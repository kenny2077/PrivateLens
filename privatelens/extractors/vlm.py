"""VLM caption extraction via Ollama."""

import json
import logging
from pathlib import Path

import httpx

from privatelens.config import settings
from privatelens.privacy.guard import PrivacyError, PrivacyGuard
from privatelens.utils.image_formats import register_image_formats


logger = logging.getLogger(__name__)

register_image_formats()


class VlmExtractor:
    """Generate captions from images using Ollama vision models."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.vlm_model
        self.ollama_url = settings.ollama_url
        self._available: bool | None = None
        self.privacy = PrivacyGuard()

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._available is not None:
            return self._available
        tags_url = f"{self.ollama_url}/api/tags"
        self.privacy.log_outbound(tags_url, "vlm_caption")
        try:
            import urllib.request
            import json

            req = urllib.request.Request(tags_url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                # Check if model or model:tag is available
                self._available = any(
                    self.model == m or self.model.startswith(m.replace(":latest", ""))
                    for m in models
                )
                if not self._available:
                    logger.warning(
                        "VLM unavailable: model %r not in Ollama. Available: %s",
                        self.model,
                        models,
                    )
                return self._available
        except PrivacyError:
            raise
        except Exception as e:
            logger.warning("VLM unavailable (Ollama not running?): %s", e)
            self._available = False
            return False

    def _prepare_image(self, image_path: Path) -> str:
        """Convert image to base64, handling HEIC conversion and resizing for Ollama."""
        import base64
        from io import BytesIO
        from PIL import Image

        source_image = Image.open(image_path)
        # Convert to RGB if needed (handles HEIC, RGBA, P modes)
        img = source_image if source_image.mode == "RGB" else source_image.convert("RGB")

        # Resize to max 1024px on longest side (Ollama vision models expect this)
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Save to JPEG bytes
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def _call_ollama(self, image_path: Path, prompt: str) -> str | None:
        """Call Ollama generate API with urllib (httpx has issues with localhost)."""
        import urllib.request

        generate_url = f"{self.ollama_url}/api/generate"
        self.privacy.log_outbound(generate_url, "vlm_caption")
        image_b64 = self._prepare_image(image_path)

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            generate_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120.0) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()

    def caption(self, image_path: Path) -> str | None:
        """Generate a caption for an image."""
        if not self.is_available():
            return None

        try:
            prompt = (
                "Describe this image in detail. Include: what is shown, "
                "any text visible, people, objects, location clues, "
                "and document type if applicable. Be concise."
            )
            return self._call_ollama(image_path, prompt)
        except PrivacyError:
            raise
        except Exception as e:
            logger.warning("VLM caption failed for %s: %s", image_path, e)
            return None

    def classify_document(self, image_path: Path) -> dict | None:
        """Classify document type using VLM."""
        if not self.is_available():
            return None

        try:
            prompt = (
                "Classify this image using the most specific type: "
                "id_card for driver's licenses or national identity cards; "
                "passport for passports; receipt for receipts or invoices; "
                "screenshot for captured app or device screens; selfie for a self-portrait; "
                "photo for other camera photos; document for a generic document only when no "
                "specialized type applies; otherwise other. Return ONLY a JSON object with keys "
                "type, confidence (0-1), and description (brief). No other text."
            )
            text = self._call_ollama(image_path, prompt)
            if not text:
                return None

            # Try to parse JSON from response
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    return self._normalize_document_classification(json.loads(text[start:end]))
            except json.JSONDecodeError:
                pass

            return self._normalize_document_classification(
                {"type": "other", "confidence": 0.5, "description": text[:200]}
            )
        except PrivacyError:
            raise
        except Exception as e:
            logger.warning("VLM classify failed for %s: %s", image_path, e)
            return None

    @staticmethod
    def _normalize_document_classification(classification: dict) -> dict:
        """Map specific taxonomy evidence to stable machine-readable labels."""
        normalized = dict(classification)
        raw_type = str(normalized.get("type", "other")).strip().lower()
        description = str(normalized.get("description", "")).strip().lower()
        evidence = f"{raw_type} {description}".replace("_", "-")

        aliases = (
            (
                "id_card",
                (
                    "driver license",
                    "driver's license",
                    "driving licence",
                    "identity card",
                    "national id",
                    "id-card",
                ),
            ),
            ("passport", ("passport",)),
            ("receipt", ("receipt", "invoice")),
            ("screenshot", ("screenshot", "screen capture")),
            ("selfie", ("selfie", "self portrait", "self-portrait")),
        )
        for canonical_type, terms in aliases:
            if any(term in evidence for term in terms):
                normalized["type"] = canonical_type
                return normalized

        canonical_type = raw_type.replace("-", "_").replace(" ", "_")
        normalized["type"] = (
            canonical_type
            if canonical_type
            in {
                "id_card",
                "passport",
                "receipt",
                "screenshot",
                "selfie",
                "photo",
                "document",
                "other",
            }
            else "other"
        )
        return normalized
