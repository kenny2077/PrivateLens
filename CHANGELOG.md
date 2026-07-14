# Changelog

All notable changes to PrivateLens will be documented in this file.

## [Unreleased]

## [1.0.0] - 2026-07-14

This entry describes the 1.0.0 release candidate. Hosted CI, PyPI/GHCR
publication, Linux amd64 validation, and the complete Compose/Ollama gate remain
pending; this changelog does not claim that 1.0.0 has been published.

### Added

- Search quality benchmark covering all ten built-in recipes with a checked-in report.
- Generated-corpus CLIP, OCR, and local VLM quality gate with a checked-in model report.
- Synthetic, non-private terminal demo and evidence-backed JSON search output.
- Multi-stage, non-root CPU Docker and Compose packaging for PrivateLens with local Ollama.
- `doctor`, `prune`, setup diagnostics, and machine-readable CLI modes.
- Incremental folder watcher with debounced scan/index cycles and NDJSON events.
- Python 3.11, 3.12, and 3.13 CI matrix with clean-wheel verification.
- Official CPU-only PyTorch installation path for Linux users.

### Changed

- Hardened scan, index, recipe execution, ranking, evidence cards, and ORM session boundaries.
- Added native sqlite-vec loading where supported with a tested BLOB fallback.
- Migrated legacy OpenCLIP configuration to the architecture-compatible QuickGELU model identity.
- Isolated per-asset index writes so one database failure cannot poison the remaining batch.
- Selected CoreML and CPU face providers on Apple Silicon instead of requesting CUDA.

### Verification

- Passed 181 local tests plus lint, typing, bytecode, lock, and diff checks.
- Passed a deterministic 1,000-image scan/index/search/idempotence reliability gate.
- Reached 91.7% hit@1, 100% hit@5, and 95.8% MRR@5 on a 15-image local
  real-photo evaluation; only aggregate metrics are recorded.
- Built core and full CPU images on local arm64; the full image passed CPU-only
  ML imports, HEIC decoding, and a 15/15 scan from a read-only photo mount.

### Security

- Enforced local-only outbound guards, including the Docker Compose Ollama hostname.
- Kept source photo mounts read-only and benchmark/demo artifacts free of private media.
