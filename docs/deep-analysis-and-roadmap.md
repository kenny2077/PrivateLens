# PrivateLens — Deep Analysis & 1.0 Roadmap

> **Date:** 2026-07-03  
> **Scope:** Full codebase audit, competitive analysis, strategic value assessment, and improvement pipeline  

> **Historical baseline, not a live completion tracker.** The strategic thesis
> in this document remains the source of truth: PrivateLens is a CLI-first,
> read-only, explainable photo-search sidecar, not a photo manager or toy demo.
> Implementation counts, status tables, risks, and unchecked tasks below record
> the 2026-07-03 starting point. Use the current tests, README release-status
> block, and `CONTINUITY.md` for the live implementation state.

---

## Current Gate Status — 2026-07-14

PrivateLens is now a `1.0.0` release candidate, not a published release. Local
verification on Apple Silicon covers the core product thesis and the original
roadmap's highest-priority reliability gates:

| Gate | Current evidence | Boundary |
|------|------------------|----------|
| Automated and vector correctness | 181 local tests plus lint, typing, bytecode, lock, and diff checks; a 1,001-vector regression covers the BLOB fallback and native-first path; hosted Python 3.11–3.13 and isolated wheel-consumer checks pass | Hosted jobs run on Linux; this is not a multi-OS claim |
| 1,000-image reliability loop | Deterministic generated images complete scan, index, search, and an idempotent rerun | Uses extractor stand-ins; not a relevance or performance claim |
| Real-photo retrieval | A 15-image local run reaches 91.7% hit@1, 100% hit@5, and 95.8% MRR@5 | Aggregate metrics only; the sample is too small for a broad quality claim |
| CPU Docker | Core and full images build locally on arm64; the full image passes local CPU-only ML imports, HEIC decoding, and a 15/15 read-only scan; hosted CI builds it and passes HTTP health on Linux amd64 while running non-root with a read-only root filesystem | GHCR publication and full Compose/Ollama remain pending; hosted health is not a model-quality gate |

PyPI/GHCR publication and the complete Compose/Ollama flow remain open gates.
Hosted Python 3.11–3.13, isolated wheel-consumer, and Linux amd64 full-image
build/HTTP-health checks pass. CUDA and the desktop application are unsupported
and are not shipped in 1.0. The historical body below remains the 2026-07-03
starting-point record rather than a live completion tracker.

## 1. Executive Summary

**Verdict: Worth continuing. High potential as a starrable GitHub product.**

PrivateLens occupies a **genuine, underserved niche** — a non-destructive, local-first AI photo *search sidecar* that indexes your existing folders without importing, moving, or managing files. This positioning is distinct from every major competitor (Immich, PhotoPrism, Lap, digiKam) which are all photo *managers*. The codebase is at a **functional early prototype stage (~40% of Phase 1-2 complete)** with strong architectural foundations but needs significant hardening, testing, and UX polish to reach a credible v1.0 GitHub release.

### Key Numbers

| Metric | Value |
|--------|-------|
| **Total Python LOC** | ~2,600 across 30 source files |
| **Test coverage** | 1 test file, 15 test cases (smoke-level) |
| **Git commits** | 0 (uncommitted — not yet published) |
| **Dependencies** | 18 Python packages (heavy ML stack) |
| **Roadmap completion** | ~35-40% of Phase 1-2 from PLAN.md |
| **Competitive uniqueness** | High — no direct open-source equivalent |

---

## 2. Project Value Assessment

### 2.1 What Makes PrivateLens Valuable

| Strength | Why It Matters |
|----------|---------------|
| **Sidecar architecture** | Reads existing folders read-only. Zero lock-in. No import/export. This is the #1 pain point users cite when rejecting Immich/PhotoPrism |
| **Multi-signal hybrid search** | CLIP + OCR + Faces + Captions + Metadata combined, not just one AI model. This mirrors how Google Photos actually works |
| **Search recipes** | Pre-built query plans for "find my ID", "find receipts", "find selfies" — solves the long-tail of *specific* photo searches that CLIP alone handles poorly |
| **Evidence cards** | "Here's WHY this matched" — no other local tool does this. Critical for trust |
| **Privacy as architecture** | Encrypted sensitive metadata, local-only audit, panic purge — not just a checkbox but a real system |
| **VLM reranker** | Using Ollama vision models to score top-N candidates is genuinely novel for a local tool |
| **AnythingLLM bridge** | Chat-based photo search with RAG citations taps into the local LLM explosion |

### 2.2 Real-World Problem It Solves

> *"I have 30,000 photos across 5 folders and 2 drives. I need to find my driver license backup photo. Google Photos can't see my local files. Apple Photos needs me to import everything. I don't want another photo manager — I just want to SEARCH."*

This is a **real, frequent, unsatisfied need**. The target audience:
- Privacy-conscious power users (r/selfhosted, r/degoogle communities)
- International students / immigrants needing quick document retrieval (I-20s, passports, visas)
- Photographers with existing folder structures they refuse to abandon
- Users of Immich/PhotoPrism who want *better* search without switching platforms

### 2.3 Why It Could Be a GitHub-Starrable Product

1. **No direct competitor** — CaptionFoundry, A-Eye, and ImageIndexer do captioning/tagging, not structured search with recipes and evidence
2. **Trending tech stack** — sqlite-vec + CLIP + local LLM is the hottest intersection in developer tools right now
3. **Demo-able** — A 30-second GIF showing `privatelens search "my driver license"` → result with "WHY: OCR contains DRIVER LICENSE" is compelling
4. **Python-native** — `pip install privatelens` is the gold standard for developer adoption
5. **Immich sidecar story** — "Enhanced search for your Immich library" immediately taps into Immich's 60k+ star community

---

## 3. Competitive Landscape (July 2026)

### 3.1 Direct Competitors

| Tool | Stars | Approach | Key Difference from PrivateLens |
|------|-------|----------|-------------------------------|
| **Immich** | 60k+ | Full photo manager (NestJS + PostgreSQL) | *Manages* photos, requires import. PrivateLens is a sidecar |
| **PhotoPrism** | 35k+ | Full photo manager (Go + MariaDB) | Same — photo manager, not search sidecar |
| **Lap** | ~500 | Desktop photo manager, folder-first | Closer, but still a manager with UI. No search recipes, no evidence |
| **CaptionFoundry** | ~300 | Caption/tag generator | Tagging only, no structured search. No hybrid search |
| **A-Eye** | ~200 | Catalogue + rename tool | Cataloguing, not retrieval. No vector search |
| **ImageIndexer** | ~100 | Sidecar metadata writer | Writes to files (destructive). No search engine |
| **Damselfly** | ~1k | DAM (C#) | Full manager, not Python, different audience |

### 3.2 PrivateLens's Unique Position

```
Photo Managers                    Photo Indexers
(own your files)                  (read your files)
┌─────────────┐                   ┌─────────────────┐
│  Immich      │                   │  CaptionFoundry │ ← Tag only
│  PhotoPrism  │                   │  A-Eye          │ ← Catalogue only
│  digiKam     │                   │  ImageIndexer   │ ← Metadata only
│  Lap         │                   │                 │
└─────────────┘                   │  PrivateLens    │ ← SEARCH + Evidence
                                  │  (← YOU ARE HERE)
                                  └─────────────────┘
```

**The gap:** No tool in the "read-only indexer" category has a **real search engine** with hybrid retrieval, recipes, and explainability. PrivateLens fills this gap.

---

## 4. Current Codebase State (Detailed Audit)

### 4.1 Architecture Overview

```
privatelens/             30 Python files, ~2,600 LOC
├── cli.py              445 LOC  — Click CLI with 9 commands ✅ Working
├── api.py              139 LOC  — FastAPI with 5 endpoints ✅ Working
├── config.py            72 LOC  — Pydantic settings ✅ Solid
├── db/
│   └── schema.py       383 LOC  — 10 SQLAlchemy models + FTS5 + triggers ✅ Solid
├── extractors/
│   ├── exif.py         112 LOC  — EXIF + SHA256 + pHash ✅ Working
│   ├── clip.py          79 LOC  — OpenCLIP embed + text encode ✅ Working
│   ├── ocr.py           57 LOC  — RapidOCR wrapper ✅ Working
│   ├── faces.py         81 LOC  — InsightFace detection ✅ Working
│   ├── vlm.py          141 LOC  — Ollama VLM captions ✅ Working
│   ├── clustering.py   171 LOC  — DBSCAN face clustering ✅ Working
│   ├── document.py      68 LOC  — Heuristic document classifier ✅ Working
│   ├── screenshot.py    51 LOC  — Heuristic screenshot detector ✅ Working
│   └── sensitive.py     57 LOC  — Keyword-based sensitive detector ✅ Working
├── search/
│   ├── engine.py       266 LOC  — Hybrid search (4 signals) ✅ Working
│   ├── recipes.py      247 LOC  — 10 built-in recipes ✅ Working
│   ├── queries.py      132 LOC  — FTS5 + ORM queries ✅ Working
│   ├── evidence.py     117 LOC  — Evidence card builder ✅ Working
│   └── reranker.py     103 LOC  — VLM reranker via Ollama ✅ Working
├── privacy/
│   ├── guard.py         59 LOC  — Local-only enforcement ⚠️ Basic
│   ├── audit.py        126 LOC  — 8-check privacy audit ✅ Working
│   └── encrypt.py       56 LOC  — Fernet encryption ✅ Working
├── integrations/
│   ├── anythingllm.py  183 LOC  — AnythingLLM sync/chat ⚠️ Untested
│   ├── immich.py       ~170 LOC — Immich DB reader ⚠️ Untested
│   ├── apple_photos.py ~120 LOC — Apple Photos import ⚠️ Untested
│   └── google_takeout.py ~100 LOC — Takeout import ⚠️ Untested
├── utils/
│   ├── thumbnails.py    39 LOC  — Thumbnail generation ✅ Working
│   ├── hashing.py       18 LOC  — Hash utilities ✅ Working
│   └── fs.py            30 LOC  — File system helpers ✅ Working
└── web/
    └── index.html      ~200 LOC — Basic search page ⚠️ Minimal
```

### 4.2 What Actually Works vs. Planned

| Feature | PLAN.md Phase | Status | Notes |
|---------|:---:|--------|-------|
| **File scanning + EXIF** | 1 | ✅ Complete | scan command works, SHA256 + pHash + GPS |
| **CLIP embeddings** | 1 | ✅ Complete | OpenCLIP ViT-B-32, lazy loading |
| **OCR extraction** | 1 | ✅ Complete | RapidOCR integration |
| **SQLite schema + FTS5** | 1 | ✅ Complete | 10 tables, 3 FTS5 indexes, triggers |
| **sqlite-vec integration** | 1 | ⚠️ Fallback | Tries to load extension, falls back to brute-force numpy |
| **Hybrid search engine** | 1 | ✅ Complete | 4-signal fusion (CLIP + OCR + Captions + Path) |
| **VLM captions** | 2 | ✅ Complete | Ollama integration with model detection |
| **Search recipes** | 2 | ✅ Complete | 10 built-in recipes with JSON query plans |
| **Evidence cards** | 2 | ✅ Complete | Multi-signal explanation builder |
| **Face detection** | 2 | ✅ Complete | InsightFace buffalo_l |
| **Face clustering** | 2 | ✅ Complete | DBSCAN + greedy fallback |
| **Document classifier** | 2 | ✅ Complete | Heuristic-based (aspect ratio + keywords) |
| **Screenshot detector** | 2 | ✅ Complete | Resolution + filename + EXIF heuristics |
| **Sensitive detector** | 2 | ✅ Complete | Keyword-based OCR + filename matching |
| **VLM reranker** | 2 | ✅ Complete | Ollama-based scoring |
| **Privacy guard** | 3 | ⚠️ Basic | Local-only check, outbound logging skeleton |
| **Privacy audit** | 3 | ✅ Complete | 8-check doctor command |
| **Encryption** | 3 | ✅ Complete | Fernet encrypt/decrypt |
| **AnythingLLM integration** | 3 | ⚠️ Untested | Code exists but unvalidated |
| **Web UI** | 1 | ⚠️ Minimal | Basic HTML page, no real UX |
| **Desktop app (Tauri)** | 4 | ❌ Skeleton | Cargo.toml + config only, no frontend |
| **Immich sidecar** | 5 | ⚠️ Untested | Connector exists, unvalidated |
| **Apple Photos import** | 5 | ⚠️ Untested | Code exists, unvalidated |
| **Google Takeout import** | 5 | ⚠️ Untested | Code exists, unvalidated |
| **Folder watcher** | 4 | ❌ Missing | watchdog is a dependency but not used |
| **Feedback loop** | 3 | ❌ Missing | SearchEvent table exists, no recording |
| **Detection search signal** | 2 | ❌ Missing | `pass` in engine._execute_plan() |

### 4.3 Code Quality Assessment

**Strengths:**
- Clean module separation (extractors / search / privacy / integrations)
- Consistent lazy-loading pattern for ML models (saves RAM)
- Pydantic settings with env var support
- Proper SQLAlchemy 2.0 mapped columns
- FTS5 with trigram tokenizer (good multilingual support)
- HEIC support registration in multiple extractors
- Graceful degradation (sqlite-vec fallback, sklearn fallback)

**Weaknesses:**

| Issue | Severity | Files |
|-------|----------|-------|
| **No git commits** | 🔴 Critical | Project not version-controlled |
| **No CI/CD** | 🔴 Critical | No GitHub Actions, no linting |
| **Thin tests** | 🔴 High | 1 file, 15 tests, mostly import/smoke |
| **No end-to-end test** | 🔴 High | scan→index→search never tested together |
| **init_db returns engine inconsistently** | 🟡 Medium | schema.py L255 returns engine, but type hint says None |
| **Detached ORM objects** | 🟡 Medium | queries.py returns detached objects after session close |
| **No error types** | 🟡 Medium | Bare `except Exception` everywhere |
| **No logging** | 🟡 Medium | `print()` used instead of `logging` module |
| **No progress bars** | 🟡 Medium | Only console.print for progress |
| **Missing .env.example** | 🟡 Low | Referenced in plan but doesn't exist |
| **Duplicate HEIC registration** | 🟢 Low | clip.py, ocr.py, vlm.py all register separately |

---

## 5. Strategic Direction: Where the Value Is

### 5.1 Three Possible Directions (Ranked by Value)

#### 🥇 Direction A: **CLI-First Power Tool** (Recommended)

> *"The `ripgrep` of photo search."*

**Position:** A fast, ergonomic Python CLI that indexes photo folders and provides instant hybrid search from the terminal. Ships as `pip install privatelens`. Zero config, zero server.

**Why this wins:**
- Fastest path to a starrable GitHub product (4-6 weeks to v1.0)
- Developer audience loves CLI tools (see: ripgrep 50k stars, fzf 68k stars)
- Demo-able with a single GIF in README
- Natural pip install story
- Avoids the death trap of competing on UI with Immich/PhotoPrism

**Success metric:** `pip install privatelens && privatelens scan ~/Photos && privatelens search "driver license"` works in <2 minutes.

#### 🥈 Direction B: **Immich Search Plugin**

> *"Enhanced search for Immich."*

**Position:** A sidecar that reads Immich's database/API and provides search recipes, evidence cards, and VLM reranking on top. Immich-compatible, ships as Docker container.

**Why this is valuable:**
- Taps into Immich's 60k+ star community
- Clear distribution channel (Immich docs, community forums)
- Lower bar — "better search for your existing Immich" is easier to sell than "new photo tool"
- Could become the official search enhancement

**Why it's second:** Depends on Immich's API stability, smaller audience than general CLI.

#### 🥉 Direction C: **Local Photo RAG Engine**

> *"Chat with your photo library using Ollama."*

**Position:** An AnythingLLM workspace plugin or standalone RAG engine that lets you chat about your photos: "Show me receipts from March" → results with citations.

**Why it's interesting:** LLM/RAG hype cycle, AnythingLLM integration already started.  
**Why it's third:** Niche audience, slower to demo, RAG quality is fragile.

### 5.2 Recommended Strategy

**Lead with Direction A (CLI tool), build toward B (Immich plugin), keep C (RAG) as a premium feature.**

Reasoning:
1. CLI tool has the fastest time-to-demo and broadest audience
2. Immich sidecar can be built on top of the same CLI core
3. AnythingLLM integration is already partially built and can be a Phase 3 feature
4. All three directions share 90% of the same codebase

---

## 6. Step-by-Step Improvement Pipeline to v1.0

### Overview

```
Current State (v0.1-alpha)
    │
    ▼
Phase 0: Foundation Hardening (1 week)
    │   Git, CI, logging, tests, error handling
    ▼
Phase 1: Core Loop Reliability (1-2 weeks)
    │   scan→index→search must work flawlessly
    ▼
Phase 2: Search Quality (1-2 weeks)
    │   sqlite-vec, better ranking, recipe execution
    ▼
Phase 3: CLI UX Polish (1 week)
    │   Progress bars, rich output, --help docs
    ▼
Phase 4: Packaging & Distribution (1 week)
    │   pip install, Docker, GIF demos
    ▼
Phase 5: README & GitHub Presence (3 days)
    │   README rewrite, badges, demo video
    ▼
v1.0-beta Release 🚀
```

---

### Phase 0: Foundation Hardening (Priority: CRITICAL)

**Goal:** Make the project maintainable and trustworthy.

#### 0.1 Version Control & CI
- [ ] Make initial git commit with clean `.gitignore`
- [ ] Set up GitHub repository (private initially)
- [ ] Add GitHub Actions: lint (ruff), type check (mypy), test (pytest)
- [ ] Add pre-commit hooks (ruff format + ruff check)

#### 0.2 Logging
- [ ] Replace all `print()` calls with Python `logging` module
- [ ] Add `--verbose` / `-v` flag to set log level (already in CLI, not wired)
- [ ] Structured log format: `[module] message` with timestamps

#### 0.3 Error Handling
- [ ] Define exception hierarchy: `PrivateLensError`, `ExtractionError`, `SearchError`, `StorageError`
- [ ] Replace bare `except Exception` with specific catches
- [ ] Add `--debug` flag to show tracebacks

#### 0.4 Config Improvements
- [ ] Create `.env.example` with documented defaults
- [ ] Validate config at startup (e.g., check ollama_url is reachable only if VLM features are used)
- [ ] Fix type annotation: `init_db()` return type should be `Engine`, not `None`

#### 0.5 Test Infrastructure
- [ ] Add `conftest.py` with reusable fixtures (temp DB, sample images of various types)
- [ ] Add integration test: `scan → index → search` end-to-end
- [ ] Add test for each extractor with a real sample image (e.g., a receipt with OCR text)
- [ ] Target: 50+ tests covering all public APIs

**Deliverable:** Green CI, `pytest` passes, `ruff check` clean.

---

### Phase 1: Core Loop Reliability (Priority: HIGH)

**Goal:** `scan → index → search` must work reliably on a real photo folder.

#### 1.1 Fix ORM Session Management
- [ ] Fix detached object issue in `SearchQueries` — return dicts or dataclasses, not ORM objects
- [ ] Consolidate `init_db()` and `get_engine()` into a proper engine singleton
- [ ] Add connection pool configuration for concurrent reads during search

#### 1.2 Fix sqlite-vec Integration
- [ ] Bundle or document sqlite-vec extension installation clearly
- [ ] Add a `privatelens setup` command that checks all dependencies
- [ ] Test vector search with >1000 embeddings for correctness
- [ ] Implement proper cosine similarity index creation

#### 1.3 Indexing Robustness
- [ ] Add proper progress bars (`rich.progress`) for scan and index commands
- [ ] Add `--dry-run` flag to scan/index to preview what would be processed
- [ ] Handle corrupt images gracefully (PIL decompression bombs, truncated files)
- [ ] Add incremental indexing (only new/changed files based on mtime + SHA256)
- [ ] Fix thumbnail generation to work with `asset.id` before commit (currently `id` is None before flush)
- [ ] Add `privatelens status` command (show counts: scanned, indexed, faces, etc.)

#### 1.4 Search Correctness
- [ ] Wire up detection search signal (currently `pass` in `_execute_plan`)
- [ ] Normalize scores across signals to [0, 1] range
- [ ] Add score explanation breakdown in CLI output
- [ ] Handle empty database gracefully (first-run UX)

**Deliverable:** `privatelens scan ~/Photos && privatelens index && privatelens search "receipt"` works on a 1,000-photo folder without crashes.

---

### Phase 2: Search Quality (Priority: HIGH)

**Goal:** Search results should be good enough that users prefer PrivateLens over manual folder browsing.

#### 2.1 Proper Vector Search
- [ ] Ensure sqlite-vec cosine distance works correctly
- [ ] Implement text-to-image search benchmarks (10 known queries → expected results)
- [ ] Add search relevance tests: precision@5, precision@10 for key recipes

#### 2.2 Search Recipe Improvements
- [ ] Implement `face_count` filter in recipe execution (currently recipes reference it but engine doesn't filter by face count)
- [ ] Implement `date_range` filter for `find_memory` recipe
- [ ] Implement `is_sensitive` filter for `find_sensitive` recipe
- [ ] Add recipe auto-detection from natural language query (e.g., "receipt" auto-triggers `find_receipt`)

#### 2.3 Search Feedback Loop
- [ ] Record search events to `search_events` table
- [ ] Add `privatelens search --feedback` for interactive result rating
- [ ] Use click history to boost frequently-accessed assets

#### 2.4 VLM Integration Quality
- [ ] Handle VLM timeouts and retries
- [ ] Add structured caption extraction (JSON output from VLM)
- [ ] Make VLM reranker optional and skippable with `--fast` flag

**Deliverable:** On a curated test set, PrivateLens finds the target photo in top-5 results for 8/10 standard queries.

---

### Phase 3: CLI UX Polish (Priority: MEDIUM)

**Goal:** The CLI should feel professional and delightful.

#### 3.1 Output Quality
- [ ] Rich tables for search results with truncated paths
- [ ] Colored score indicators (green >0.8, yellow >0.5, red <0.5)
- [ ] `--json` output flag for all commands (machine-readable)
- [ ] `--open` flag to open first result in system viewer
- [ ] `privatelens search --preview` to show thumbnails in terminal (via kitty/iterm2 protocol or sixel)

#### 3.2 Help & Discovery
- [ ] Rich `--help` text with examples for every command
- [ ] `privatelens quickstart` command that guides first-time setup
- [ ] `privatelens recipes` already works — add `--detail` flag to show query plans
- [ ] Tab completion for commands and recipe names

#### 3.3 Performance
- [ ] Add timing output: "Searched 10,000 photos in 0.3s"
- [ ] Lazy-load only needed extractors (don't import torch just for `privatelens status`)
- [ ] Background indexing daemon mode: `privatelens watch ~/Photos`

**Deliverable:** First-time user can go from `pip install` to first search in <3 minutes with clear guidance.

---

### Phase 4: Packaging & Distribution (Priority: MEDIUM)

**Goal:** Easy installation on macOS, Linux, and Docker.

#### 4.1 pip Install
- [ ] Verify `pip install privatelens` works from a clean venv
- [ ] Split dependencies into `[core]` and `[full]` extras (core = no torch, just search existing index; full = all ML models)
- [ ] Add `[gpu]` extra for CUDA-enabled torch
- [ ] Test on Python 3.11, 3.12, 3.13

#### 4.2 Docker
- [ ] Simplify Dockerfile (currently CUDA-only, need a CPU version)
- [ ] Multi-stage build: builder (compile deps) → runtime (slim)
- [ ] Docker Compose for `privatelens + ollama` all-in-one
- [ ] Publish to Docker Hub or GitHub Container Registry

#### 4.3 Platform Testing
- [ ] macOS (Apple Silicon) — primary target
- [ ] Linux (x86_64) — Docker and bare metal
- [ ] Windows WSL2 — test in Docker

**Deliverable:** `pip install privatelens` works. `docker run` works. README has installation for all 3 platforms.

---

### Phase 5: README & GitHub Presence (Priority: HIGH for stars)

**Goal:** The GitHub page makes people want to star and try it.

#### 5.1 README Rewrite
- [ ] Hero line: "Find photos by meaning, not by scrolling. Local. Private. Explainable."
- [ ] Demo GIF/video: 30-second clip showing scan → search → result with evidence
- [ ] Feature comparison table vs. Immich, PhotoPrism, etc.
- [ ] Architecture diagram (simplified, not the current complex one)
- [ ] Badges: CI, Python version, license, stars

#### 5.2 Demo Content
- [ ] Record a terminal demo with `asciinema` or `vhs`
- [ ] Create 3 demo scenarios: (1) find ID document, (2) find receipt, (3) find selfie
- [ ] Generate sample output screenshots for README

#### 5.3 GitHub Presence
- [ ] License file (MIT — already declared)
- [ ] CONTRIBUTING.md
- [ ] Issue templates (bug report, feature request)
- [ ] GitHub topics: `photo-search`, `local-first`, `privacy`, `ai`, `clip`, `ocr`
- [ ] Discussions enabled for community

**Deliverable:** GitHub repo page that gets people to try the tool within 30 seconds of landing.

---

## 7. Detailed File-Level Improvement Priorities

### Critical Fixes (Do First)

| File | Issue | Fix |
|------|-------|-----|
| `schema.py` L255 | `init_db` return type mismatch | Change signature to `-> Engine` |
| `schema.py` L369-372 | Silent sqlite-vec failure | Log at warning level, add to `doctor` checks |
| `engine.py` L252 | Detection search is `pass` | Implement or remove from recipe plans |
| `queries.py` L20-26 | Returns detached ORM objects | Return dicts/dataclasses instead |
| `cli.py` L197 | `generate_thumbnail(file_path, asset.id)` may be None before flush | Flush first |
| `reranker.py` L77-86 | Uses raw `httpx` (blocks event loop if async) | Use `urllib` like vlm.py, or make async |

### Quality Improvements (Do Next)

| File | Improvement |
|------|-------------|
| All extractors | Centralize HEIC registration to a single utility |
| `config.py` | Add validation for ollama_url format |
| `cli.py` | Add `rich.progress.Progress` for scan/index loops |
| `search/engine.py` | Normalize candidate scores to [0, 1] properly |
| `privacy/guard.py` | Actually enforce local-only (currently just logs) |
| `sensitive.py` | Too many false positives from filename matching (e.g., any file with "id" in name) |

---

## 8. Effort Estimate to v1.0

| Phase | Effort | Calendar Time | Priority |
|-------|--------|--------------|----------|
| Phase 0: Foundation | 3-4 days | Week 1 | 🔴 Critical |
| Phase 1: Core Loop | 5-7 days | Weeks 1-2 | 🔴 Critical |
| Phase 2: Search Quality | 5-7 days | Weeks 2-3 | 🟡 High |
| Phase 3: CLI UX | 3-4 days | Week 3 | 🟡 High |
| Phase 4: Packaging | 2-3 days | Week 4 | 🟡 Medium |
| Phase 5: README | 2-3 days | Week 4 | 🟡 High |
| **Total** | **~4-5 weeks** | **~1 month** | |

---

## 9. Post-v1.0 Opportunities

### v1.1: Immich Sidecar Mode
- Connect to Immich's PostgreSQL database
- Index Immich library with PrivateLens recipes and evidence
- Ship as Docker sidecar alongside Immich

### v1.2: Web UI
- Replace the basic HTML page with a proper search interface
- Photo grid, evidence panel, recipe selector
- Could use Svelte (like Immich) or htmx (lightweight)

### v1.3: Desktop App
- Tauri shell is already scaffolded
- System tray with background indexing
- File watcher for auto-indexing new photos
- macOS `.dmg` installer

### v1.4: Community & Ecosystem
- Plugin system for custom recipes
- Community recipe marketplace
- Immich/PhotoPrism plugin integrations
- MLX acceleration for Apple Silicon (faster CLIP)

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:---------:|:------:|------------|
| Immich adds search recipes | Medium | High | Position as "enhanced search", become Immich plugin |
| Apple Intelligence does local photo search | Medium | Medium | Apple can't search external drives or non-Apple folders |
| ML dependencies too heavy for pip install | High | Medium | Split into `[core]` (just search) and `[full]` (with models) extras |
| sqlite-vec not mature enough | Low | Medium | Brute-force fallback already works; sqlite-vec is actively maintained |
| VLM quality too low for reranking | Medium | Low | VLM reranker is optional; CLIP + OCR already strong |
| Scope creep into photo management | Medium | High | **Hard rule: never modify, copy, or manage photos** |

---

## 11. Conclusion

PrivateLens is a **well-conceived project** with a genuine market gap, strong architectural foundations, and a functional prototype. The core search loop (CLIP + OCR + Recipes + Evidence) is its primary differentiator and is already mostly implemented.

**To reach a starrable v1.0:**
1. Harden the foundation (git, tests, logging, error handling)
2. Make `scan → index → search` bulletproof
3. Polish the CLI UX with rich output and progress bars
4. Package for easy installation
5. Write a demo-first README

**The single most important thing:** Record a 30-second terminal demo showing `privatelens search "my driver license"` returning the correct photo with an evidence card explaining *why* it matched. That demo alone will get the first 100 stars.
