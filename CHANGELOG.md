# Changelog

All notable changes to PrivateLens will be documented in this file.

## [Unreleased]

## [1.0.0] - 2026-07-14

PrivateLens 1.0.0 is the first production-stable CLI release. Hosted checks
pass the core on Python 3.11–3.13, isolated wheel consumption, and a full CPU
image build with HTTP health on Linux amd64 while running non-root with a
read-only root filesystem. Official artifacts are produced only from the
`v1.0.0` tag: wheel/sdist on PyPI and GitHub Releases, plus an explicitly
model-free `1.0.0-core` image on GHCR. The full Compose/Ollama flow remains a
preview.

### Added

- Search quality benchmark covering all ten built-in recipes with a checked-in report.
- Generated-corpus CLIP, OCR, and local VLM quality gate with a checked-in model report.
- Synthetic, non-private terminal demo and evidence-backed JSON search output.
- Multi-stage, non-root CPU Docker and Compose packaging for PrivateLens with local Ollama.
- `doctor`, `prune`, setup diagnostics, and machine-readable CLI modes.
- Incremental folder watcher with debounced scan/index cycles and NDJSON events.
- Python 3.11, 3.12, and 3.13 CI matrix with clean-wheel verification.
- Official CPU-only PyTorch installation path for Linux users.
- Model-free GHCR release image with SBOM and provenance; full ML images remain
  local builds while embedded OCR-model redistribution terms are unclear.

### Changed

- Hardened scan, index, recipe execution, ranking, evidence cards, and ORM session boundaries.
- Added native sqlite-vec loading where supported with a tested BLOB fallback.
- Migrated legacy OpenCLIP configuration to the architecture-compatible QuickGELU model identity.
- Isolated per-asset index writes so one database failure cannot poison the remaining batch.
- Selected CoreML and CPU face providers on Apple Silicon instead of requesting CUDA.

### Verification

- Passed 185 local tests plus lint, typing, bytecode, lock, and diff checks.
- Passed a deterministic 1,000-image scan/index/search/idempotence reliability gate.
- Reached 91.7% hit@1, 100% hit@5, and 95.8% MRR@5 on a 15-image local
  real-photo evaluation; only aggregate metrics are recorded.
- Built core and full CPU images on local arm64; the full image passed CPU-only
  ML imports, HEIC decoding, and a 15/15 scan from a read-only photo mount.
- Passed hosted Python 3.11–3.13 CI with isolated wheel-consumer verification.
- Built and health-checked the full CPU image on hosted Linux amd64 while
  running non-root with a read-only root filesystem.

### Security

- Enforced local-only outbound guards, including the Docker Compose Ollama hostname.
- Kept source photo mounts read-only and benchmark/demo artifacts free of private media.
- Recorded exact RapidOCR wheel/model hashes and kept those models out of the
  published GHCR image pending an unambiguous model-specific license grant.
