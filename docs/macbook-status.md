# MacBook M2 8GB — Verified Local Workflow

This page records the memory-conscious workflow exercised on the primary local
development machine. It does not promise fixed timings, memory use, or
cross-platform compatibility; see the README platform matrix for release status.

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| File scan + EXIF | Verified locally | Scan reads originals and records metadata/fingerprints in the sidecar |
| Derived thumbnails | Verified locally | Generated during the core index pass, not during scan |
| CLIP + RapidOCR | Verified locally | Core index signals; the first run may download model assets |
| Semantic/text/metadata search | Verified locally | Uses vector, FTS, path, explicit date, camera, and dimension/type signals |
| Search recipes + evidence cards | Verified locally | Structured ranking with machine-readable explanations |
| Face extraction | Optional | Opt in only after reviewing the InsightFace model license |
| Ollama VLM | Optional | Run as a separate caption/rerank pass when the configured local model is available |
| Web/API preview | Secondary | Loopback-only default, no access-control guarantee |
| Privacy audit | Verified locally | `privatelens doctor` inspects local configuration without an internet probe |
| CPU Docker | Verified locally on arm64 | Core and full images build; the full image uses CPU-only ML runtimes |

## 1.0.0 release evidence

The 2026-07-14 local gate includes:

- 185 passing tests plus lint, typing, bytecode, lock, and diff checks;
- a deterministic 1,000-image scan/index/search/idempotence reliability run;
- a 15-image local real-photo evaluation at 91.7% hit@1, 100% hit@5, and
  95.8% MRR@5, reported only as aggregate metrics; and
- core and full CPU Docker builds on local arm64, with the full image passing
  CPU-only ML imports, HEIC decoding, and a 15/15 scan from a read-only photo
  mount.

Hosted checks pass Python 3.11–3.13 with isolated wheel-consumer
verification, and the full CPU image builds and passes HTTP health on Linux
amd64 while running non-root with a read-only root filesystem. The release
workflow publishes wheel/sdist artifacts and an explicitly model-free GHCR
core image from `v1.0.0`; the full Compose/Ollama flow remains a preview. CUDA
and the desktop application are unsupported and are not shipped in 1.0.

## Sequential Workflow

Run the heavier phases as separate commands so each Python process releases its
extractor state before the next optional pass:

```bash
# 1. Discover photos; no ML inference
privatelens scan ~/Pictures

# 2. Build core CLIP/OCR index and derived thumbnails
privatelens index --skip-face --skip-vlm --batch-size 1

# 3. Optional biometric pass, disabled by default
privatelens index --only-face --batch-size 1

# 4. Optional local Ollama caption pass, disabled by default
privatelens index --only-vlm --batch-size 1

# Search the resulting sidecar
privatelens search "driver license"
privatelens search "screenshot of code"
privatelens search --recipe find_receipt "Target"
```

`--batch-size` controls how often pending database work is committed. It is not
a promise about model residency or peak RAM. Model downloads, latency, and
memory use vary by runtime version, cache state, media, and enabled passes, so
measure them on the intended workload.

## Recommendation

Use the external gaming PC for a larger, statistically meaningful labeled
real-photo campaign and any future CUDA validation. Do not assume a speedup or
concurrent model fit until that machine passes the
[experimental CUDA checklist](deploy-gaming-pc.md) with recorded measurements.
