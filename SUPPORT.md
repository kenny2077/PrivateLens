# Support

PrivateLens is maintained on a best-effort basis. There is no guaranteed
response time, compatibility promise for unverified platforms, or individual
deployment SLA.

## Start with self-service diagnostics

```bash
privatelens setup --json
privatelens doctor --json
privatelens status --json
```

Then consult:

- [README troubleshooting](README.md#troubleshooting)
- [Privacy guide](docs/privacy-guide.md)
- [Search recipes](docs/search-recipes.md)
- [Third-party model licenses](THIRD_PARTY_MODELS.md)
- [Changelog](CHANGELOG.md)

## Choose the right channel

### Reproducible bugs

Use the
[bug-report template](https://github.com/kenny2077/PrivateLens/issues/new?template=bug_report.md)
after searching existing issues. Include the PrivateLens version, OS, Python
version, install method, exact command, expected behavior, and a minimal
synthetic reproduction.

### Feature requests

Use the
[feature-request template](https://github.com/kenny2077/PrivateLens/issues/new?template=feature_request.md).
Explain the photo-search problem and how the proposal preserves PrivateLens as a
CLI-first, read-only sidecar. Proposals that turn PrivateLens into a photo
manager are out of scope.

### Setup and usage questions

Use [GitHub Discussions](https://github.com/kenny2077/PrivateLens/discussions)
for non-sensitive usage questions. Use a public issue only when the question
and all diagnostics are safe to disclose.

### Security vulnerabilities

Follow [SECURITY.md](SECURITY.md) and use the repository's
[private vulnerability report](https://github.com/kenny2077/PrivateLens/security/advisories/new).
Never disclose exploit details or sensitive data in a public issue.

### Conduct reports

Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and use a private channel.

## Safe diagnostic sharing

Do not attach:

- private or personal photos;
- OCR text from IDs, receipts, medical records, or financial documents;
- face crops, embeddings, names, or cluster assignments;
- a PrivateLens database, thumbnail directory, or model cache;
- encryption keys, API keys, tokens, cookies, or `.env` files;
- unredacted home-directory names or absolute photo paths.

Prefer the generated demo corpus, replace paths with neutral examples, and
review JSON output before posting it.

## Support boundaries

Maintainers may not be able to provide:

- legal advice about biometric processing or model licensing;
- support for modified forks or old releases;
- debugging for private datasets that cannot be reproduced synthetically;
- guarantees for native Windows, unverified Linux/WSL, CUDA, or third-party
  service behavior;
- custom features that import, move, rewrite, or manage original photos.
