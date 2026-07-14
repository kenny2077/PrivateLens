# Security Policy

PrivateLens indexes potentially sensitive photos, OCR text, document
classifications, locations, and face embeddings. Security reports must avoid
exposing the very data the project is designed to keep private.

## Supported versions

The 1.0.x release line and current default branch receive security fixes on a
best-effort basis.

| Version | Support |
|---------|---------|
| 1.0.x | Supported |
| Current default branch | Supported |
| Versions older than 1.0 and forks | Not supported |

## Report a vulnerability privately

Use the repository's
[private vulnerability report](https://github.com/kenny2077/PrivateLens/security/advisories/new).
Do not file a public issue containing vulnerability details.

If private vulnerability reporting is not enabled, create only a minimal public
request asking the maintainers to provide a private reporting channel. Do not
include the affected command, exploit, private path, photo, OCR text, database,
face data, token, encryption key, or other sensitive details in that request.

Helpful private reports include:

- affected PrivateLens version or commit;
- operating system, Python version, and install method;
- security impact and realistic attack prerequisites;
- minimal reproduction using generated or fully synthetic data;
- redacted logs or `privatelens doctor --json` output;
- suggested remediation, if known.

## In scope

Examples include:

- a scan, index, search, watch, prune, or purge path that can unexpectedly
  modify or delete original photos;
- unintended remote disclosure of photos, OCR text, captions, face data,
  credentials, or index contents;
- bypass of local-only outbound restrictions;
- arbitrary code execution, command injection, path traversal, or unsafe file
  handling;
- exposure or incorrect handling of configured encryption keys;
- access to PrivateLens-owned data outside the configured data directory;
- vulnerable dependency behavior that is reachable through PrivateLens.

## Usually not security issues

Use the public bug template, with redacted synthetic data, for:

- search relevance or model-quality disagreements;
- expected model false positives or false negatives;
- crashes without a security boundary impact;
- slow indexing or excessive memory use;
- unsupported platforms or third-party service outages.

Questions about third-party model licensing belong in
[THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md), not in a vulnerability report.

## Response and disclosure

Maintenance is currently best effort and no response-time SLA is promised.
Maintainers will confirm the report when possible, assess impact, coordinate a
fix and release, and credit reporters who request credit. Please allow time for
a fix before public disclosure and coordinate publication with the maintainers.

## Security boundaries users should know

- Read-only source-photo handling is a core product boundary, but it does not
  sandbox third-party decoders, ML runtimes, or local integrations.
- Local-only mode is not an operating-system firewall. Model downloads are
  network operations, and explicitly configured integrations may create local
  or permitted outbound traffic.
- Optional sensitive-metadata encryption is not full-database, thumbnail, cache,
  or disk encryption.
- Anyone who can read the PrivateLens data directory may be able to access
  derived metadata. Protect it with operating-system permissions and disk
  encryption.
- Face embeddings may be regulated biometric data. Obtain appropriate consent
  and follow applicable law.
