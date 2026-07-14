# Experimental NVIDIA / Gaming-PC Validation Guide

This is an **unsupported future-validation checklist**, not a working
installation guide. The 1.0.0 release candidate is CPU-only: it ships no CUDA
image, Compose file, dependency extra, or helper script, and this page
intentionally provides no runnable CUDA commands. The desktop application is
also unsupported and is not shipped in 1.0.

For the currently exercised container path, use the
[CPU Docker quick start](../README.md#docker-cpu-quick-start). Core and full CPU
images build locally on arm64; the full image passes CPU-only ML imports, HEIC
decoding, and a 15/15 scan from a read-only photo mount. The complete CPU
Compose + Ollama flow and Linux amd64 execution remain open gates.

## Intended Target

- Linux or Windows with WSL2 and a supported NVIDIA driver/container runtime
- Docker configured so a test container can see the GPU
- A local Ollama endpoint only when optional VLM captioning/reranking is enabled
- Source photos mounted read-only and PrivateLens sidecar/model data mounted to
  separate writable locations

Follow the current official [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and [Docker Desktop WSL2 documentation](https://docs.docker.com/desktop/features/wsl/)
for the host. PrivateLens does not pin a driver/toolkit combination until the
target machine has been measured and recorded.

## Safety Boundaries

- Keep original media on a read-only mount. The CUDA path must preserve the
  same sidecar boundary as the CPU path.
- Bind the unauthenticated web/API preview to loopback unless a trusted access
  layer is deliberately added. Do not expose it to the internet.
- `local_only=true` default-denies non-local app-managed VLM and AnythingLLM
  requests. It is not a firewall and does not intercept package or model-library
  downloads.
- Face extraction is opt-in. Review the configured InsightFace model license
  before enabling biometric processing.
- The default VLM identifier is the configured Qwen3-VL Ollama tag documented
  in [THIRD_PARTY_MODELS.md](../THIRD_PARTY_MODELS.md); do not substitute model
  names or resource estimates without a recorded run.

## Index Portability

Do not copy the SQLite database between the Mac and container hosts as a normal
sync strategy. Asset and thumbnail records contain absolute paths, so a copied
database is only meaningful when all mount paths are identical. Prefer
scanning/indexing the target read-only photo mount into a target-local sidecar.

## Promotion Gates

An NVIDIA path may become a supported quick start only after a clean target
machine demonstrates all of the following:

1. The CUDA image builds with the full declared Python dependencies and no
   undeclared host files or caches.
2. The container reports the intended GPU execution providers for the actual
   CLIP, OCR, and optional face runtimes; GPU visibility alone is insufficient.
3. Generated-media scan → core index → smart/fast search passes with evidence
   JSON, followed by a consented read-only real-photo smoke.
4. Face and VLM opt-in passes are validated separately, including configured
   model identifiers and licenses.
5. Originals remain read-only, the application runs non-root, and only declared
   sidecar/model-cache/tmp locations are writable.
6. Compose service names, Ollama health/readiness, shutdown behavior, and the
   loopback/LAN exposure choice are verified on both Linux and WSL2 targets.
7. Peak RAM/VRAM, cold/warm startup, per-pass throughput, failures, and model
   download sizes are measured from the target run rather than estimated.
8. The same generated-data regression suite passes after the GPU-specific
   changes, with CPU fallback behavior documented.

Until those gates pass, do not publish speed, VRAM, batch-size, driver-version,
or model-fit claims for this hardware.
