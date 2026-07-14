#!/usr/bin/env python3
"""Download and cache ML models for PrivateLens."""

import sys
from pathlib import Path

from privatelens.config import settings
from privatelens.extractors.clip import canonical_clip_model_name


def download_clip():
    """Download OpenCLIP model."""
    print("Downloading OpenCLIP model...")
    import torch
    import open_clip

    model_name = canonical_clip_model_name(settings.clip_model, settings.clip_pretrained)
    cache_path = settings.resolved_model_cache_dir / "huggingface" / "hub"
    cache_path.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=settings.clip_pretrained,
        cache_dir=str(cache_path),
    )
    print(f"OpenCLIP {model_name} cached.")


def download_ocr():
    """Download RapidOCR model."""
    print("Downloading RapidOCR model...")
    from rapidocr_onnxruntime import RapidOCR

    RapidOCR()
    print("RapidOCR cached.")


def download_face():
    """Download InsightFace model."""
    print("Downloading InsightFace model...")
    from insightface.app import FaceAnalysis

    root = settings.resolved_model_cache_dir / "insightface"
    root.mkdir(parents=True, exist_ok=True)
    app = FaceAnalysis(name=settings.face_model, root=str(root))
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"InsightFace {settings.face_model} cached.")


def main():
    print("PrivateLens Model Downloader")
    print("=" * 40)

    # Ensure cache directory exists
    settings.resolved_model_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_clip()
    except Exception as e:
        print(f"CLIP download failed: {e}")

    try:
        download_ocr()
    except Exception as e:
        print(f"OCR download failed: {e}")

    try:
        download_face()
    except Exception as e:
        print(f"Face download failed: {e}")

    print("\nDone! Models are cached for future use.")


if __name__ == "__main__":
    main()
