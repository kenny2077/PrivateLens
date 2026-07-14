"""Privacy audit and diagnostics."""

import json
import os
from pathlib import Path

from privatelens.config import settings
from privatelens.db.schema import get_engine
from privatelens.privacy.guard import is_local_service_url


class PrivacyAuditor:
    """Run privacy and system diagnostics."""

    def run_audit(self) -> list[dict]:
        """Run comprehensive privacy audit."""
        checks = []

        # Check 1: Database exists and is local
        checks.append(self._check_database())

        # Check 2: Thumbnails are local
        checks.append(self._check_thumbnails())

        # Check 3: Model cache is local
        checks.append(self._check_model_cache())

        # Check 4: No cloud API keys in environment
        checks.append(self._check_cloud_apis())

        # Check 5: Ollama is local
        checks.append(self._check_ollama_local())

        # Check 6: Vector backend
        checks.append(self._check_vector_backend())

        # Check 7: Encryption key configured
        checks.append(self._check_encryption())

        # Check 8: Sensitive items are encrypted
        checks.append(self._check_sensitive_encryption())

        # Check 9: Network status
        checks.append(self._check_network())

        for check in checks:
            check.setdefault("remediation", [])

        return checks

    def _check_database(self) -> dict:
        db_path = settings.resolved_db_path
        return {
            "name": "Database Local",
            "status": "ok" if db_path.exists() else "warning",
            "details": f"DB at {db_path}",
        }

    def _check_thumbnails(self) -> dict:
        thumb_dir = settings.resolved_thumbnail_dir
        return {
            "name": "Thumbnails Local",
            "status": "ok" if thumb_dir.exists() else "warning",
            "details": f"Thumbnails at {thumb_dir}",
        }

    def _check_model_cache(self) -> dict:
        cache_dir = settings.resolved_model_cache_dir
        return {
            "name": "Model Cache Local",
            "status": "ok" if cache_dir.exists() else "warning",
            "details": f"Models at {cache_dir}",
        }

    def _check_cloud_apis(self) -> dict:
        cloud_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AWS_ACCESS_KEY"]
        found = [v for v in cloud_vars if os.environ.get(v)]
        return {
            "name": "No Cloud API Keys",
            "status": "ok" if not found else "warning",
            "details": f"Found: {', '.join(found)}" if found else "No cloud API keys detected",
        }

    def _check_ollama_local(self) -> dict:
        if not is_local_service_url(settings.ollama_url):
            return {
                "name": "Ollama Local",
                "status": "warning",
                "details": f"Ollama at {settings.ollama_url} is not an approved local URL",
                "remediation": ["Set PRIVATELENS_OLLAMA_URL=http://localhost:11434 in .env"],
            }

        import urllib.request

        tags_url = f"{settings.ollama_url.rstrip('/')}/api/tags"
        try:
            req = urllib.request.Request(tags_url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            return {
                "name": "Ollama Local",
                "status": "warning",
                "details": f"Ollama at {settings.ollama_url} is local but not reachable: {e}",
                "remediation": [
                    "open -a Ollama",
                    f"curl -fsS {tags_url}",
                    f"ollama pull {settings.vlm_model}",
                ],
            }

        models = [model.get("name", "") for model in data.get("models", [])]
        model_available = any(
            settings.vlm_model == model
            or settings.vlm_model.startswith(model.replace(":latest", ""))
            for model in models
        )
        if not model_available:
            return {
                "name": "Ollama Local",
                "status": "warning",
                "details": (
                    f"Ollama at {settings.ollama_url} reachable, but model "
                    f"{settings.vlm_model} is not installed. Available: {models}"
                ),
                "remediation": [f"ollama pull {settings.vlm_model}"],
            }

        return {
            "name": "Ollama Local",
            "status": "ok",
            "details": f"Ollama at {settings.ollama_url}; model {settings.vlm_model} available",
            "remediation": [],
        }

    def _check_vector_backend(self) -> dict:
        from sqlalchemy import text

        engine = get_engine()
        try:
            with engine.connect() as conn:
                version_row = conn.execute(text("SELECT vec_version()"))
                version = version_row.fetchone()
                if not version:
                    raise RuntimeError("vec_version() returned no result")
                rows = conn.execute(
                    text("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name IN ('vec_image_embeddings', 'vec_faces')
                """)
                ).fetchall()
            table_names = {row[0] for row in rows}
            has_vec = {"vec_image_embeddings", "vec_faces"}.issubset(table_names)
            fallback_detail = self._diagnose_sqlite_vec_fallback()
            return {
                "name": "Vector Search Backend",
                "status": "ok" if has_vec else "warning",
                "details": (
                    f"sqlite-vec {version[0]} virtual tables available"
                    if has_vec
                    else (
                        "Using BLOB fallback; sqlite-vec virtual tables are not available "
                        f"({fallback_detail})"
                    )
                ),
                "remediation": [] if has_vec else self._sqlite_vec_remediation(fallback_detail),
            }
        except Exception as e:
            fallback_detail = self._diagnose_sqlite_vec_fallback()
            return {
                "name": "Vector Search Backend",
                "status": "warning",
                "details": f"Using BLOB fallback; sqlite-vec check failed: {e} ({fallback_detail})",
                "remediation": self._sqlite_vec_remediation(fallback_detail),
            }

    def _diagnose_sqlite_vec_fallback(self) -> str:
        """Return an actionable sqlite-vec fallback reason for doctor output."""
        try:
            import sqlite_vec
        except ImportError:
            return "sqlite-vec Python package is not installed"

        import sqlite3

        try:
            conn = sqlite3.connect(":memory:")
        except Exception as e:
            return f"sqlite3 in-memory connection failed: {e}"

        try:
            if not hasattr(conn, "load_extension"):
                return "Python sqlite3 does not expose load_extension"
            try:
                sqlite_vec.load(conn)
            except Exception as e:
                return f"sqlite-vec package is installed but extension load failed: {e}"
            return "sqlite-vec extension loads in memory; reinitialize the PrivateLens DB"
        finally:
            conn.close()

    def _sqlite_vec_remediation(self, fallback_detail: str) -> list[str]:
        """Return concise remediation hints for sqlite-vec fallback modes."""
        if "not installed" in fallback_detail:
            return [
                "uv pip install --python .venv/bin/python -e .",
                "privatelens doctor --json",
            ]
        if "load_extension" in fallback_detail:
            return [
                "PrivateLens will use BLOB fallback; no action required for small libraries.",
                "Use the external PC for large libraries or a Python build with SQLite extension loading.",
            ]
        if "extension loads in memory" in fallback_detail:
            return ["Create a fresh PrivateLens data dir or reinitialize the database."]
        return [
            "PrivateLens will use BLOB fallback; rerun privatelens doctor --json after setup changes."
        ]

    def _check_encryption(self) -> dict:
        if not settings.encryption_key:
            return {
                "name": "Encryption Configured",
                "status": "warning",
                "details": "No encryption key configured",
                "remediation": ["Set PRIVATELENS_ENCRYPTION_KEY to a Fernet key before indexing."],
            }

        from privatelens.privacy.encrypt import MetadataEncryptor

        try:
            MetadataEncryptor()
        except (TypeError, ValueError):
            return {
                "name": "Encryption Configured",
                "status": "warning",
                "details": "Encryption key is not a valid Fernet key",
                "remediation": [
                    'Generate a key with: python -c "import sys; '
                    "from cryptography.fernet import Fernet; "
                    'sys.stdout.write(Fernet.generate_key().decode())"'
                ],
            }

        return {
            "name": "Encryption Configured",
            "status": "ok",
            "details": "Valid Fernet encryption key configured",
            "remediation": [],
        }

    def _check_sensitive_encryption(self) -> dict:
        from sqlalchemy import text

        engine = get_engine()
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT encrypted_metadata FROM sensitive_items")
                ).fetchall()
                total_count = len(rows)
                if total_count == 0:
                    return {
                        "name": "Sensitive Items Encrypted",
                        "status": "info",
                        "details": "No sensitive classifications indexed",
                    }

                encrypted_payloads = [row[0] for row in rows if row[0] is not None]
                encrypted_count = len(encrypted_payloads)
                decryptor = None
                if settings.encryption_key:
                    from privatelens.privacy.encrypt import MetadataEncryptor

                    try:
                        decryptor = MetadataEncryptor()
                    except (TypeError, ValueError):
                        pass
                decryptable_count = (
                    sum(decryptor.decrypt(payload) is not None for payload in encrypted_payloads)
                    if decryptor is not None
                    else 0
                )
                fully_verified = (
                    decryptor is not None
                    and encrypted_count == total_count
                    and decryptable_count == total_count
                )
                verification_detail = (
                    f"; {decryptable_count}/{total_count} decryptable with the configured key"
                    if decryptor is not None
                    else "; no valid configured key available to verify ciphertext"
                )
                return {
                    "name": "Sensitive Items Encrypted",
                    "status": "ok" if fully_verified else "warning",
                    "details": (
                        f"{encrypted_count}/{total_count} sensitive classification payloads encrypted"
                        f"{verification_detail}"
                    ),
                    "remediation": (
                        []
                        if fully_verified
                        else [
                            "Configure the original valid PRIVATELENS_ENCRYPTION_KEY, then run "
                            "privatelens index --force to refresh missing or unreadable payloads."
                        ]
                    ),
                }
        except Exception as exc:
            return {
                "name": "Sensitive Items Encrypted",
                "status": "warning",
                "details": (
                    f"Could not inspect sensitive classifications: {type(exc).__name__}: {exc}"
                ),
            }

    def _check_network(self) -> dict:
        return {
            "name": "Network Policy",
            "status": "ok" if settings.local_only else "info",
            "details": (
                "Application local-only guard enabled (this is not a system firewall)"
                if settings.local_only
                else "Application local-only guard disabled"
            ),
            "remediation": (
                []
                if settings.local_only
                else ["Set PRIVATELENS_LOCAL_ONLY=true to block guarded remote integrations."]
            ),
        }
