"""VLM reranker for scoring top-N search candidates."""

import logging
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from privatelens.config import settings
from privatelens.privacy.guard import PrivacyError, PrivacyGuard


logger = logging.getLogger(__name__)


class VlmReranker:
    """Rerank search candidates using Ollama vision model."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.vlm_model
        self.ollama_url = settings.ollama_url
        self.privacy = PrivacyGuard()

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        query: str,
        top_k: int = 30,
    ) -> list[dict[str, Any]]:
        """Rerank candidates with VLM scoring.

        Args:
            candidates: List of candidate result dicts with 'asset_id' and 'path'
            query: The search query
            top_k: Number of top candidates to score

        Returns:
            Reranked list of candidates
        """
        if not candidates:
            return candidates

        # Score top candidates
        to_score = candidates[:top_k]
        scored = []

        for candidate in to_score:
            score = self._score_image(candidate["path"], query)
            if score is not None:
                candidate["score"] = candidate.get("score", 0.5) * 0.7 + score * 0.3
                candidate["vlm_score"] = score
            scored.append(candidate)

        # Sort by score descending
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Append remaining unscored candidates
        remaining = candidates[top_k:]
        return scored + remaining

    def _score_image(self, image_path: str, query: str) -> float | None:
        """Score a single image against a query using VLM.

        Returns:
            Score between 0 and 1, or None if failed
        """
        try:
            path = Path(image_path)
            if not path.exists():
                return None

            with open(path, "rb") as f:
                import base64

                image_b64 = base64.b64encode(f.read()).decode()

            prompt = (
                f"Rate how well this image matches the query '{query}' on a scale of 0 to 10. "
                f"Return ONLY a number between 0 and 10. No explanation."
            )

            generate_url = f"{self.ollama_url}/api/generate"
            self.privacy.log_outbound(generate_url, "vlm_caption")
            response = httpx.post(
                generate_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()

            # Extract number from response
            import re

            match = re.search(r"(\d+(?:\.\d+)?)", text)
            if match:
                score = float(match.group(1))
                return min(max(score / 10.0, 0.0), 1.0)

            return None
        except PrivacyError:
            raise
        except Exception as e:
            logger.warning("VLM rerank failed for %s: %s", image_path, e)
            return None
