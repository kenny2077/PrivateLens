## Summary

Describe the user problem and the smallest change that solves it.

## CLI-first sidecar scope

- [ ] Source photos remain read-only: this change does not import, move, rewrite, delete, or manage originals.
- [ ] The primary scan/index/search CLI workflow remains clear and functional.
- [ ] JSON/NDJSON output remains machine-readable, or contract changes are documented.

## Verification

List exact commands and results:

```text
Not run:
```

## Privacy, security, and models

- [ ] Tests and examples use generated or explicitly redistributable data only.
- [ ] No private photos, OCR text, face data, databases, paths, keys, tokens, caches, or build artifacts are included.
- [ ] Network, storage, deletion, encryption, or sensitive-data behavior is described above.
- [ ] New or changed model terms are documented in `THIRD_PARTY_MODELS.md`.
- [ ] Security-sensitive details were reported privately instead of placed in this pull request.

## Documentation and release impact

- [ ] User-facing commands, configuration, limitations, and migration effects are documented.
- [ ] Tests cover behavior changes or the reason they cannot is explained.
- [ ] Unverified operating systems, hardware, models, or full-stack paths are called out explicitly.
- [ ] The change is reflected in `CHANGELOG.md` when release-visible.

