# Phase 0/1 Core Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the first PrivateLens production slice so git is safe, database initialization is explicit, and recipe detection search contributes real results.

**Architecture:** Keep the product CLI-first and sidecar-only. Make narrow changes at the database/search boundary, and use focused tests that do not load CLIP/OCR/face/VLM models on the MacBook Pro.

**Tech Stack:** Python 3.11+, Click, SQLAlchemy 2.x, SQLite FTS5, pytest, Rich.

---

### Task 1: Protect Git Baseline

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Update git ignores**

Add ignores for OS junk, pytest cache, local env files, private sample media, reference checkout, local databases, thumbnails, models, and generated result artifacts.

```gitignore
.DS_Store
.pytest_cache/
.env
.env.*
!.env.example
photo/
reference/
*.db
*.sqlite
*.sqlite3
data/
results/
models/
thumbnails/
```

- [ ] **Step 2: Verify git status excludes private/reference trees**

Run:

```bash
rtk git status --short --branch
```

Expected: `photo/` and `reference/` are not listed as untracked paths.

### Task 2: Make Database Initialization Contract Explicit

**Files:**
- Modify: `tests/test_smoke.py`
- Modify: `privatelens/db/schema.py`

- [ ] **Step 1: Write failing annotation/logging test**

Add this test near `test_database_schema`:

```python
def test_init_db_declares_engine_return_type(self):
    from sqlalchemy.engine import Engine
    from privatelens.db import schema

    assert schema.init_db.__annotations__["return"] is Engine
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
rtk proxy pytest tests/test_smoke.py::TestScanAndIndex::test_init_db_declares_engine_return_type -v
```

Expected: FAIL because `init_db()` is annotated as returning `None`.

- [ ] **Step 3: Update schema implementation**

In `privatelens/db/schema.py`, import `logging` and `Engine`, create `logger = logging.getLogger(__name__)`, change `def init_db() -> None:` to `def init_db() -> Engine:`, and replace the sqlite-vec `print(...)` fallback with:

```python
logger.warning("sqlite-vec extension not loaded (%s). Using BLOB fallback for vectors.", e)
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
rtk proxy pytest tests/test_smoke.py::TestScanAndIndex::test_init_db_declares_engine_return_type -v
```

Expected: PASS.

### Task 3: Implement Detection Signal Search

**Files:**
- Modify: `tests/test_smoke.py`
- Modify: `privatelens/search/queries.py`
- Modify: `privatelens/search/engine.py`

- [ ] **Step 1: Write failing detection recipe test**

Add this test to `TestSearchEngine`:

```python
def test_detection_signal_contributes_to_recipe_plan(self, temp_db, monkeypatch):
    from sqlalchemy.orm import Session
    from privatelens.db.schema import Asset, Detection

    with Session(temp_db) as session:
        asset = Asset(path="/test/receipt.jpg", sha256="test", media_type="image")
        session.add(asset)
        session.commit()
        session.add(
            Detection(
                asset_id=asset.id,
                label="receipt",
                confidence=0.93,
                source_model="test",
            )
        )
        session.commit()

    engine = SearchEngine()
    monkeypatch.setattr(engine.clip_extractor, "encode_text", lambda _query: None)
    plan = {"signals": [{"type": "detection", "labels": ["receipt"], "weight": 0.7}]}

    results = engine._execute_plan(plan, "receipt", limit=10)

    assert len(results) == 1
    assert results[0]["asset_id"] == asset.id
    assert results[0]["score"] == pytest.approx(0.7)
    assert "Detected: receipt" in results[0]["explanation"]
```

- [ ] **Step 2: Run the detection test to verify it fails**

Run:

```bash
rtk proxy pytest tests/test_smoke.py::TestSearchEngine::test_detection_signal_contributes_to_recipe_plan -v
```

Expected: FAIL because detection signals are currently ignored.

- [ ] **Step 3: Add detection query helper**

Add `detection_label_search(self, labels: list[str], query: str | None = None, limit: int = 200) -> list[tuple[int, str, float | None]]` to `SearchQueries`. It should match labels case-insensitively, optionally fall back to the natural-language query, order by confidence descending with nulls last, and return `(asset_id, label, confidence)` tuples.

- [ ] **Step 4: Wire detection signal execution**

Replace the detection `pass` in `SearchEngine._execute_plan()` with a call to the new helper. Add `weight * confidence` when confidence exists, otherwise add `weight`.

- [ ] **Step 5: Run focused detection test**

Run:

```bash
rtk proxy pytest tests/test_smoke.py::TestSearchEngine::test_detection_signal_contributes_to_recipe_plan -v
```

Expected: PASS.

### Task 4: Verify Focused Phase 0/1 Slice

**Files:**
- No new production files.

- [ ] **Step 1: Run focused search tests**

Run:

```bash
rtk proxy pytest tests/test_smoke.py::TestSearchEngine -v
```

Expected: PASS.

- [ ] **Step 2: Run full existing tests**

Run:

```bash
rtk proxy pytest tests/ -v
```

Expected: PASS or report concrete dependency/runtime failures without hiding them.

- [ ] **Step 3: Update continuity**

Update `CONTINUITY.md` with completed facts, remaining bugs, and next-stage focus.
