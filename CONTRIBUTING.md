# Contributing

Thank you for helping improve PrivateLens. By participating, you agree to follow
the [Code of Conduct](CODE_OF_CONDUCT.md).

PrivateLens is a local-first, read-only photo search sidecar. Contributions
must preserve that product boundary: do not import, move, rewrite, delete, or
manage a user's original photos. The CLI remains the primary product surface;
API and web work should not weaken the scan/index/search workflow.

## Before you start

- Use Python 3.11, 3.12, or 3.13 and `uv` for core work. Use Python 3.11 for
  release-gated ML/full-extra work. The lock resolves on 3.12, but the complete
  ML stack is not a supported 1.0 gate there; RapidOCR excludes Python 3.13.
- Search existing issues before proposing a substantial behavior change.
- Keep patches focused. Discuss large architectural or product-scope changes
  before implementation.
- Report security vulnerabilities privately as described in
  [SECURITY.md](SECURITY.md), not in a public issue.
- Use generated or explicitly redistributable fixtures only. Never commit
  personal photos, OCR text, face embeddings, local databases, model caches,
  tokens, keys, or unredacted paths.

## Development Setup

```bash
uv sync --python 3.11 --extra dev
source .venv/bin/activate
```

Install the full local ML stack when working on indexing, extractors, or
model-dependent search:

```bash
uv sync --python 3.11 --all-extras
```

To print the complete local setup path, including Ollama, model cache warmup, and verification commands:

```bash
privatelens setup
privatelens setup --json
```

Install the repository's local pre-commit hooks:

```bash
pre-commit install
```

The hooks run Ruff checks and formatting using the project virtual environment.

## Development workflow

1. Create a focused branch from the current default branch.
2. Add a failing regression test or a clear verification case before changing
   behavior.
3. Make the smallest change that satisfies that case.
4. Update user-facing documentation when commands, JSON fields, privacy
   behavior, configuration, or model requirements change.
5. Run the checks appropriate to the affected surface.
6. Open a pull request using the repository template and describe any
   unverified platform or model-dependent behavior.

## Verification

Run the lightweight checks before opening a pull request:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy privatelens
python -m compileall privatelens tests
python -m pytest tests/ -v
privatelens benchmark --json
uv build
```

If you changed Docker packaging, also run:

```bash
docker compose config --quiet
```

If you changed CLIP, OCR, face, VLM, or vector retrieval behavior, run the
relevant generated-corpus model benchmark and report the model/runtime details.
Do not describe a generated smoke corpus as broad real-photo quality evidence.

For a Mac-safe smoke run, avoid face/VLM passes unless the local machine has
enough memory:

```bash
privatelens scan /path/to/photos --dry-run
privatelens index --skip-face --skip-vlm --batch-size 1 --dry-run
```

Use the external GPU machine for meaningful CUDA validation. Do not weaken
release goals or replace production paths with toy behavior because a local
8GB Mac cannot run a large experiment.

## Code and CLI expectations

- Match the existing style and keep changes surgical.
- Add or update tests for behavior changes and bug fixes.
- Keep JSON CLI output machine-readable; diagnostics belong on stderr/logging.
- Preserve stable command exit behavior and document new flags.
- Keep source-photo access read-only. Sidecar maintenance may delete only
  PrivateLens-owned index data, thumbnails, and caches after explicit consent.
- Prefer synthetic fixtures small enough for CI. Keep larger, labeled
  evaluation corpora outside git unless their redistribution terms are clear.
- Document optional dependency and graceful-degradation behavior.

## Models and third-party material

Do not add or change a model default without documenting its source, exact
identifier, download path, license, usage restrictions, expected resource
cost, and removal procedure in
[THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md). In particular, do not imply
that PrivateLens's MIT license covers third-party weights.

## Pull Request Guidelines

- Explain the user problem and why the change belongs in a CLI-first,
  read-only sidecar.
- List exact verification commands and results.
- Call out skipped tests, unverified platforms, migrations, network behavior,
  and privacy/security impact.
- Link the relevant issue when one exists.
- Keep unrelated cleanup out of the pull request.

## Contribution licensing

PrivateLens code is licensed under the [MIT License](LICENSE). By submitting a
contribution, you agree that it may be distributed under that license and
represent that you have the right to submit it. Do not contribute material
whose license is incompatible or unclear. Third-party dependencies and model
weights retain their own licenses.
