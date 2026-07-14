## 1. Project Architecture — Should not change unless explicitly requested.

The project must always stay aligned with this thesis:

> PrivateLens is a local-first, read-only photo search sidecar: the CLI-first
> "ripgrep of photo search" that indexes existing folders without importing,
> moving, or managing photos, then returns structured, explainable multi-signal
> search results.

Do not narrow the project into a toy demo.

- Dataset: existing user photo folders, generated release fixtures, and optional
  read-only sources such as Immich, Apple Photos exports, and Google Takeout.
- Baselines: manual folder browsing, Apple/Google Photos search,
  Immich/PhotoPrism, and caption/tag-only tools.
- Metrics: scan/index/search success and idempotence, top-k retrieval, evidence
  usefulness, privacy audit status, latency, and install/container reproducibility.
- Related Work: Immich, PhotoPrism, digiKam, Lap, CaptionFoundry, A-Eye,
  ImageIndexer, sqlite-vec, OpenCLIP, RapidOCR, InsightFace, Ollama/VLM, and
  AnythingLLM.

Code Architecture:

- `privatelens/cli.py`: primary Click CLI for scan, index, search, recipes,
  diagnostics, maintenance, clustering, and integrations.
- `privatelens/db/`: SQLite/SQLAlchemy schema, FTS5, and vector persistence.
- `privatelens/extractors/`: EXIF, CLIP, OCR, optional faces/VLM, and classifiers.
- `privatelens/search/`: hybrid retrieval, structured recipes, ranking, and
  evidence cards.
- `privatelens/privacy/`: local-only guard, audit, and limited provenance
  encryption.
- `privatelens/integrations/`: explicit read/export integrations; secondary to
  the core CLI.
- `privatelens/web/` and `privatelens/api.py`: secondary loopback-only preview,
  not an authenticated multi-user service.
- Runtime indexes, thumbnails, model caches, and private media remain outside
  git; only reproducible non-private reports belong in `results/`.

---

## 2. Progress — Update after every meaningful session

Milestones — Three facts only, no raw logs:

- 1. The `1.0.0` release-candidate branch preserves the canonical CLI-first,
  read-only, explainable sidecar thesis and has mature installation, privacy,
  security, support, contribution, licensing, CI, and release documentation.
- 2. Local verification covers 181 tests plus lint/format/typing/bytecode/lock,
  pre-commit, strict wheel/sdist, isolated consumer, dependency-audit, a
  1,001-vector regression, and a deterministic 1,000-image reliability gate.
- 3. Aggregate-only evaluation on 15 local real images reached 91.7% hit@1,
  100% hit@5, and 95.8% MRR@5; core/full CPU images build on local arm64, and
  the full image passes CPU-only ML imports, HEIC decoding, and a 15/15 scan
  from a read-only photo mount.

Critical Bugs / Software or Hardware or Network Issues — Three logs maximum:

- The pre-public development history contains workstation paths and a personal
  commit email; publish only a new clean root plus the final release commit.
- Hosted CI, PyPI/GHCR publication, GitHub release evidence, and Linux amd64
  execution remain pending and must not be claimed as complete.
- Full Compose/Ollama remains unverified; CUDA and desktop are unsupported and
  unshipped, and InsightFace weights require separate license review.

Reflect on current working direction is not worth continuing or have better ideas?

- Worth continuing. The strongest product remains the terminal-first,
  evidence-backed search workflow rather than a new photo manager or UI-first
  rewrite.

---

## 3. Next Stage Implementation Plan — Update after every meaningful session

- Focus 1: Stage the verified release tree, scan it for private material, and
  replace the local development history with a two-commit clean public lineage.
- Focus 2: Create the canonical GitHub repository privately, push main and the
  hardening branch, require hosted Python/container CI, merge, then make it public.
- Focus 3: Configure PyPI trusted publishing and repository security/community
  settings, tag 1.0.0, verify PyPI/GitHub/GHCR publication, and update release truth.

---

## 4. Important Files and Commands

Files:

- `docs/deep-analysis-and-roadmap.md`: canonical thesis and historical roadmap.
- `CONTINUITY.md`: canonical anti-drift project briefing.
- `README.md`: product positioning, installation, evidence, boundaries, and demo.
- `pyproject.toml`: package metadata, Python range, dependencies, and build config.
- `uv.lock`: canonical locked dependency graph and Linux CPU-only Torch source.
- `CHANGELOG.md`: release-candidate change record.
- `LICENSE`: canonical MIT license for PrivateLens source.
- `CONTRIBUTING.md`: contributor workflow and sidecar scope guard.
- `SECURITY.md` and `SUPPORT.md`: private reporting and community support policy.
- `THIRD_PARTY_MODELS.md`: model/runtime provenance and license boundaries.
- `.github/workflows/ci.yml`: hosted Python test and wheel-consumer matrix.
- `.github/workflows/release.yml`: PyPI and GitHub release workflow.
- `.github/workflows/container.yml`: GHCR CPU image workflow.
- `Dockerfile` and `docker-compose.yml`: non-root CPU image and preview stack.
- `privatelens/cli.py`: primary product surface and core orchestration.
- `privatelens/db/schema.py`: persistence, FTS5, and vector backend setup.
- `privatelens/search/`: hybrid search, recipes, ranking, and evidence.
- `privatelens/extractors/`: local model and heuristic extraction boundaries.
- `privatelens/benchmark.py` and `privatelens/model_benchmark.py`: release gates.
- `tests/`: regression, production, HEIC, and scale/reliability coverage.

Commands:

```bash
# inspect continuity and worktree
rtk read CONTINUITY.md
rtk git status --short --branch

# canonical locked setup and validation
rtk proxy uv sync --locked --all-extras
rtk proxy uv lock --check
rtk proxy .venv/bin/python -m pytest
rtk proxy .venv/bin/python -m ruff check .
rtk proxy .venv/bin/python -m ruff format --check .
rtk proxy .venv/bin/python -m mypy privatelens
rtk proxy .venv/bin/python -m compileall privatelens tests
rtk proxy .venv/bin/pre-commit run --all-files

# deterministic product gates
rtk proxy .venv/bin/privatelens benchmark --json
rtk proxy .venv/bin/privatelens benchmark-models --skip-vlm --json

# package and clean-wheel consumer gate
rtk proxy uv build --no-sources
rtk proxy uv venv /tmp/privatelens-wheel-smoke --python 3.11
rtk proxy uv pip install --python /tmp/privatelens-wheel-smoke/bin/python dist/privatelens-1.0.0-py3-none-any.whl
rtk proxy /tmp/privatelens-wheel-smoke/bin/privatelens benchmark --json

# CPU container gates
rtk proxy docker compose config --quiet
rtk proxy docker build --build-arg PRIVATELENS_EXTRAS=core -t privatelens:1.0.0-core .
rtk proxy docker build --build-arg PRIVATELENS_EXTRAS=full -t privatelens:1.0.0 .

# supported first-run CLI path; face and VLM remain opt-in
rtk proxy privatelens scan /path/to/photos --dry-run --json
rtk proxy privatelens index --skip-face --skip-vlm --batch-size 1
rtk proxy privatelens search "receipt" --json --limit 5
rtk proxy privatelens doctor --json
```
