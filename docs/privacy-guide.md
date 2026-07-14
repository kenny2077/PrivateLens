# Privacy Guide

## Local-Only Mode

PrivateLens defaults to `local_only=true`. In this mode:

- OpenCLIP, InsightFace, and RapidOCR run in the local process; VLM calls are
  allowed only to a recognized local endpoint
- App-managed VLM and AnythingLLM requests default-deny non-local endpoints
- The index is stored in a local SQLite file (`~/.privatelens/privatelens.db`)

Recognized local hosts are `localhost`, `127.0.0.1`, `::1`, and the Compose
service name `ollama`. Local-only mode is not an operating-system firewall.
Package installers and model libraries perform their own downloads outside the
application guard. Run `privatelens doctor` and review configuration before
indexing sensitive material; `doctor` checks local state and does not probe the
internet.

## Sensitive Content Detection

PrivateLens heuristically flags likely sensitive documents:

- **Driver licenses** — OCR keywords: "driver license", "driving permit"
- **Passports** — OCR keywords: "passport", "travel document"
- **Bank cards** — OCR keywords: "credit card", "card number", "cvv"
- **SSN** — OCR keywords: "social security", "ssn"
- **Receipts** — OCR keywords: "receipt", "total", "tax"
- **Medical records** — OCR keywords: "prescription", "diagnosis"
- **Immigration docs** — OCR keywords: "I-20", "green card", "visa"

Detected sensitive items are flagged in the `assets` and `sensitive_items`
tables. False positives and false negatives are expected; this is not a data
loss prevention system.

## Encryption

To encrypt the structured classification payload stored for newly detected
sensitive items, configure a Fernet key:

```bash
export PRIVATELENS_ENCRYPTION_KEY="your-fernet-key-here"
```

Or generate a key:

```python
from cryptography.fernet import Fernet
key = Fernet.generate_key().decode()
print(key)
```

Only the auxiliary sensitive-item provenance JSON (`type`, `confidence`, and
`source`) is encrypted. The asset sensitivity flag, asset sensitivity type,
`sensitive_items.type`, `sensitive_items.confidence`, OCR text, captions,
embeddings, paths, thumbnails, SQLite database, model caches, and rows created
without a key remain plaintext. Use operating-system permissions and full-disk
encryption to protect the complete sidecar. Keep the key outside backups that
contain the database.

## Privacy Audit

Run the privacy audit to verify your setup:

```bash
privatelens doctor
```

Checks include local configuration for:

- Database is local
- Thumbnails are local
- Model cache is local
- No cloud API keys in environment
- Ollama endpoint uses an allowed local host
- Encryption key configured
- Sensitive-item classification encryption coverage

## Panic Commands

```bash
# Delete all sidecar index records and derived thumbnails, but keep photos
privatelens purge

# Delete faces, face vectors, and people clusters
privatelens purge --faces-only

# Inspect local privacy/configuration state without an internet probe
privatelens doctor
```

## Face Data

Face extraction is opt-in. When enabled, face embeddings are stored in the
database. To protect privacy:

- Face data is only used for clustering within your own photos
- `privatelens purge --faces-only` deletes face rows, face vectors, and people clusters
- AnythingLLM export omits raw face embeddings but can include display names
  associated with detected face clusters

## Search History

Normal CLI and API searches do not create search-history rows. Only the
interactive CLI `search --feedback` path stores a plaintext query/result event.
If the user selects a positive result, that asset can receive a capped local
ranking boost. Full `privatelens purge` deletes these events.

## AnythingLLM Export

`privatelens sync-anythingllm` is an explicit export. Each structured document
can include the absolute source path, date, dimensions, media type, camera, GPS
coordinates, sensitivity flag, captions, OCR text, identified person
names/associations, and tags. Original image bytes and raw face embeddings are
not sent by this connector. Treat the destination workspace as sensitive.

## Network Activity

The application guard checks app-managed VLM and AnythingLLM endpoints when a
request is attempted. It does not maintain a global or persistent outbound-call
log and it does not intercept dependency/model-library downloads. The web/API
preview disables HTTP access logs, binds to loopback by default, and provides no
authentication or multi-user isolation; keep it local unless you add an
appropriate trusted boundary.

## Best Practices

1. Run `privatelens doctor` after setup
2. Set `PRIVATELENS_ENCRYPTION_KEY` if auxiliary classification-provenance encryption is useful, while remembering that flags/type/confidence and the rest of the index remain plaintext
3. Use `local_only=true` (default)
4. Periodically audit with `privatelens doctor`
5. Purge face data if you no longer need face search

See [SECURITY.md](../SECURITY.md) for vulnerability reporting and
[THIRD_PARTY_MODELS.md](../THIRD_PARTY_MODELS.md) for biometric/model license
considerations.
