# PrivateLens Architecture

## Overview

PrivateLens is a **local-first, read-only photo search sidecar**. It does not
store or manage original photos: it reads existing folders and builds a
searchable, explainable index in a separate local data directory.

## Core Philosophy

1. **Read-only originals** — Scan and index never import, move, rewrite, or delete source photos
2. **Local-first** — Inference and sidecar data stay local unless an integration is explicitly configured
3. **Searchable** — Hybrid search combines CLIP, OCR, faces, captions, detections, and metadata
4. **Explainable** — Results expose the available evidence behind a match
5. **Feedback-aware** — Only explicit CLI `--feedback` selections receive a small, capped local ranking boost

## System Architecture

```
Your Photos (read-only)
    ↓
File Scanner → EXIF → pHash fingerprint
    ↓
Vision Pipeline:
    - Derived thumbnails
    - CLIP embeddings (OpenCLIP)
    - OCR text (RapidOCR)
    - Face detection (InsightFace, opt-in)
    - VLM captions (Ollama, opt-in)
    - Document classification
    - Sensitive content detection
    ↓
Local Index (SQLite + sqlite-vec + FTS5)
    ↓
Hybrid Search Engine
    - Semantic (CLIP vector)
    - Text (OCR + captions FTS5)
    - Face (person clusters)
    - Metadata (path, explicit date, camera, dimensions/type)
    - Search recipes (pre-built query plans)
    ↓
Evidence Cards (why each result matched)
```

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Core: Python 3.11–3.13; full ML: release-gated on 3.11 | Core CI covers 3.11–3.13; the full stack is verified on 3.11, while locked RapidOCR 1.4.4 excludes 3.13 |
| CLI | Click | Modern Python CLI |
| Web API | FastAPI | Async, OpenAPI docs |
| Database | SQLite + sqlite-vec + FTS5 | Zero-config, single-file |
| ORM | SQLAlchemy 2.0 | Familiar, async support |
| Embeddings | OpenCLIP (ViT-B-32) | 512-dim, good balance |
| Face Detection | InsightFace (`settings.face_model`; `buffalo_l` by default) | Optional ONNX runtime; review model terms before use |
| OCR | RapidOCR | Multilingual, fast |
| VLM | Ollama (Qwen3-VL by default) | Optional local captioning and reranking |
| Web/API | FastAPI | Secondary, loopback-only preview surface |

## Data Model

See `privatelens/db/schema.py` for the full SQLAlchemy schema. Key tables:

- `assets` — Core photo metadata
- `image_embeddings` — CLIP vectors (512-dim)
- `ocr_blocks` — OCR text with bounding boxes
- `faces` — Detected faces with embeddings
- `people` — Face clusters (persons)
- `captions` — VLM-generated descriptions
- `sensitive_items` — Sensitive classification plus optional encrypted structured payload
- `search_events` — Plaintext query/result events created only by explicit CLI `--feedback`
- `search_recipes` — Pre-built query templates

## Search Architecture

The search engine combines multiple signals:

1. **CLIP semantic** — Text query → embedding → cosine similarity
2. **OCR text** — FTS5 full-text search on extracted text
3. **Captions** — FTS5 on VLM descriptions
4. **Faces** — Person name → face cluster → matching photos
5. **Metadata** — Path, explicit date, camera, dimensions, and media type
6. **Detections** — Document type, screenshot, receipt labels

Results are fused with weighted scores and can be reranked by VLM.

## Privacy Layer

- `local_only` default-denies non-local app-managed VLM and AnythingLLM calls.
  It is not an operating-system firewall and does not cover package/model-library downloads.
- A Fernet key encrypts only auxiliary sensitive-item provenance (`type`,
  `confidence`, `source`). Sensitivity flags/type/confidence columns, OCR,
  captions, embeddings, paths, thumbnails, and the database remain plaintext.
- `privatelens purge --faces-only` deletes faces, face vectors, and people
  clusters. Full purge deletes sidecar records, search events, and derived
  thumbnails while never deleting source photos.
- `privatelens doctor` checks local configuration without probing the internet.

## Integrations

- **AnythingLLM** — Explicitly export structured photo documents for RAG; the
  payload can contain paths, GPS, OCR, captions, sensitivity state, and named
  people, but not original image bytes or raw face embeddings
- **Immich** — Read Immich PostgreSQL DB, build enhanced index
- **Apple Photos** — Read exported folder metadata without modifying originals
- **Google Takeout** — Read exported metadata without modifying originals

Integrations are secondary to the CLI core and should be treated as preview
features until validated against their upstream versions.

## Development

```bash
# Setup
uv sync --python 3.11 --all-extras
source .venv/bin/activate

# Run tests
python -m pytest tests/ -v

# Start dev server
uvicorn privatelens.api:app --reload
```
