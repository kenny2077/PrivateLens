"""Pydantic settings for PrivateLens."""

import os
from pathlib import Path
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with env var support."""

    model_config = SettingsConfigDict(
        env_prefix="PRIVATELENS_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Paths
    data_dir: Path = Path.home() / ".privatelens"
    db_path: Path | None = None
    thumbnail_dir: Path | None = None
    model_cache_dir: Path | None = None

    # Models
    clip_model: str = "ViT-B-32-quickgelu"
    clip_pretrained: str = "openai"
    ocr_model: str = "ch_PP-OCRv4"
    face_model: str = "buffalo_l"
    # VLM options: llava, gemma3:4b, qwen2.5-vl:3b, minicpm-v:2.6, qwen3:2b
    # qwen3:2b recommended for RTX 4050 6GB (smallest footprint, good quality)
    # On MacBook with AnythingLLM: use qwen3-vl:2b-instruct-q8_0 (copied from AnythingLLM cache)
    vlm_model: str = "qwen3-vl:2b-instruct-q8_0"
    ollama_url: str = "http://localhost:11434"

    # Indexing
    batch_size: int = 32
    thumbnail_size: int = 256
    max_image_size: int = 1024
    skip_duplicates: bool = True

    # Privacy
    encryption_key: str | None = None
    local_only: bool = True
    sensitive_scan: bool = True

    # Search
    default_search_limit: int = 50
    vector_search_limit: int = 200
    rerank_top_k: int = 30

    # AnythingLLM
    anythingllm_url: str | None = None
    anythingllm_api_key: str | None = None
    anythingllm_workspace: str = "privatelens"

    # Immich
    immich_db_url: str | None = None

    @property
    def resolved_db_path(self) -> Path:
        return self.db_path or self.data_dir / "privatelens.db"

    @property
    def resolved_thumbnail_dir(self) -> Path:
        return self.thumbnail_dir or self.data_dir / "thumbnails"

    @property
    def resolved_model_cache_dir(self) -> Path:
        return self.model_cache_dir or Path.home() / ".privatelens" / "models"

    @field_validator("ollama_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("ollama_url must be an http(s) URL with a host")
        return value


settings = Settings()
