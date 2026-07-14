"""Face detection and recognition using InsightFace."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from privatelens.config import settings
from privatelens.utils.image_formats import register_image_formats


logger = logging.getLogger(__name__)

register_image_formats()


def select_onnx_providers(available: list[str]) -> list[str]:
    """Select supported local providers without requesting unavailable CUDA."""
    if "CUDAExecutionProvider" in available:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "CoreMLExecutionProvider" in available:
        preferred = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        preferred = ["CPUExecutionProvider"]
    selected = [provider for provider in preferred if provider in available]
    return selected or available


class FaceExtractor:
    """Extract faces from images using InsightFace."""

    def __init__(self, model_name: str = "buffalo_l"):
        self.model_name = model_name
        self._app: Any = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if InsightFace model is downloaded and ready."""
        if self._available is not None:
            return self._available
        try:
            self._app = self._create_app()
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            self._available = True
            return True
        except Exception as e:
            logger.warning("Face detection unavailable (model not downloaded?): %s", e)
            self._available = False
            return False

    def _load_model(self):
        """Lazy load InsightFace model."""
        if self._app is not None:
            return

        self._app = self._create_app()
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def _create_app(self) -> Any:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        providers = select_onnx_providers(ort.get_available_providers())
        root = settings.resolved_model_cache_dir / "insightface"
        root.mkdir(parents=True, exist_ok=True)
        return FaceAnalysis(name=self.model_name, providers=providers, root=str(root))

    def extract(self, image_path: Path) -> list[dict] | None:
        """Extract faces from image.

        Returns list of dicts with keys: bbox, embedding, confidence
        """
        if not self.is_available():
            return None

        try:
            self._load_model()

            with Image.open(image_path) as source_image:
                rgb_image = np.asarray(source_image.convert("RGB"))
            img = np.ascontiguousarray(rgb_image[:, :, ::-1])

            faces = self._app.get(img)
            if not faces:
                return None

            results = []
            for face in faces:
                bbox = face.bbox.astype(int).tolist()
                embedding = (
                    face.embedding.astype(np.float32) if face.embedding is not None else None
                )

                results.append(
                    {
                        "bbox": {
                            "x1": bbox[0],
                            "y1": bbox[1],
                            "x2": bbox[2],
                            "y2": bbox[3],
                        },
                        "embedding": embedding.tobytes() if embedding is not None else None,
                        "confidence": float(face.det_score) if hasattr(face, "det_score") else 0.0,
                    }
                )

            return results
        except Exception as e:
            logger.warning("Face extraction failed for %s: %s", image_path, e)
            return None
