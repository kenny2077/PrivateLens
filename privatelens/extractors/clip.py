"""CLIP image embedding extraction."""

import logging
import numpy as np
from pathlib import Path
from typing import Any
from PIL import Image

from privatelens.config import settings
from privatelens.utils.image_formats import register_image_formats


logger = logging.getLogger(__name__)

register_image_formats()


def canonical_clip_model_name(model_name: str, pretrained: str) -> str:
    """Resolve known checkpoint aliases to architecture-compatible model names."""
    if model_name == "ViT-B-32" and pretrained == "openai":
        return "ViT-B-32-quickgelu"
    return model_name


def clip_model_id(model_name: str, pretrained: str) -> str:
    """Return the persisted identity for a compatible image/text embedding space."""
    canonical_name = canonical_clip_model_name(model_name, pretrained)
    return f"openclip:{canonical_name}:{pretrained}"


class ClipExtractor:
    """Extract CLIP embeddings from images using OpenCLIP."""

    def __init__(self, model_name: str | None = None, pretrained: str | None = None):
        self.pretrained = pretrained or settings.clip_pretrained
        self.model_name = canonical_clip_model_name(
            model_name or settings.clip_model,
            self.pretrained,
        )
        self._model: Any = None
        self._preprocess: Any = None
        self._device: str | None = None

    @property
    def model_id(self) -> str:
        return clip_model_id(self.model_name, self.pretrained)

    def _load_model(self):
        """Lazy load the CLIP model."""
        if self._model is not None:
            return

        import torch
        import open_clip

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_path = settings.resolved_model_cache_dir / "huggingface" / "hub"
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_dir = str(cache_path)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self._device,
            cache_dir=cache_dir,
        )
        self._model.eval()

    def extract(self, image_path: Path) -> np.ndarray | None:
        """Extract 512-dim CLIP embedding from image."""
        try:
            self._load_model()

            import torch

            image = Image.open(image_path).convert("RGB")
            tensor = self._preprocess(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                embedding = self._model.encode_image(tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.cpu().numpy().astype(np.float32).flatten()
        except Exception as e:
            logger.warning("CLIP extraction failed for %s: %s", image_path, e)
            return None

    def encode_text(self, text: str) -> np.ndarray | None:
        """Encode text query to CLIP embedding space."""
        try:
            self._load_model()

            import open_clip
            import torch

            tokenizer = open_clip.get_tokenizer(self.model_name)
            tokens = tokenizer([text]).to(self._device)

            with torch.no_grad():
                text_features = self._model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            return text_features.cpu().numpy().flatten()
        except Exception as e:
            logger.warning("Text encoding failed for %r: %s", text, e)
            return None
