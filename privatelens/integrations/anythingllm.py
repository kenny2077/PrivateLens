"""AnythingLLM integration for RAG-based photo search."""

import json
import logging
from pathlib import Path

import httpx

from privatelens.config import settings
from privatelens.db.schema import get_engine
from privatelens.privacy.guard import PrivacyError, PrivacyGuard


logger = logging.getLogger(__name__)


class AnythingLLMConnector:
    """Sync photo index to AnythingLLM workspace for RAG search."""

    def __init__(self):
        self.base_url = settings.anythingllm_url or "http://localhost:3001"
        self.api_key = settings.anythingllm_api_key
        self.workspace = settings.anythingllm_workspace
        self.privacy = PrivacyGuard()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def sync(self) -> None:
        """Sync all photo documents to AnythingLLM workspace."""
        from sqlalchemy.orm import Session
        from privatelens.db.schema import Asset, Caption, OcrBlock, Face, Person

        engine = get_engine()
        documents = []

        with Session(engine) as session:
            assets = session.query(Asset).filter(Asset.indexed_at.isnot(None)).all()

            for asset in assets:
                doc = self._build_document(asset, session)
                documents.append(doc)

        # Upload documents in batches
        for i in range(0, len(documents), 100):
            batch = documents[i : i + 100]
            self._upload_batch(batch)

    def _build_document(self, asset, session) -> dict:
        """Build a structured markdown document from photo metadata."""
        from privatelens.db.schema import Caption, OcrBlock, Face, Person

        # Get captions
        captions = session.query(Caption).filter_by(asset_id=asset.id).all()
        caption_text = "\n".join([c.caption for c in captions]) if captions else "No caption"

        # Get OCR
        ocr_blocks = session.query(OcrBlock).filter_by(asset_id=asset.id).all()
        ocr_text = "\n".join([b.text for b in ocr_blocks]) if ocr_blocks else "No OCR text"

        # Get faces/people
        faces = session.query(Face).filter_by(asset_id=asset.id).all()
        people = []
        for face in faces:
            if face.cluster_id:
                person = session.query(Person).filter_by(id=face.cluster_id).first()
                if person and person.display_name:
                    people.append(person.display_name)

        # Build markdown document
        content = f"""# Photo: {Path(asset.path).name}

**Path:** {asset.path}
**Date:** {asset.exif_datetime or asset.created_at}
**Dimensions:** {asset.width}x{asset.height}
**Media Type:** {asset.media_type}
**Camera:** {asset.exif_make} {asset.exif_model}
**Location:** {asset.gps_lat}, {asset.gps_lng}
**Sensitive:** {asset.is_sensitive}

## Caption
{caption_text}

## OCR Text
{ocr_text}

## People
{", ".join(people) if people else "No people identified"}

## Tags
{", ".join([asset.media_type, asset.sensitive_type or ""])}
"""

        return {
            "title": f"Photo: {Path(asset.path).name}",
            "content": content,
            "metadata": {
                "asset_id": asset.id,
                "path": asset.path,
                "media_type": asset.media_type,
                "is_sensitive": asset.is_sensitive,
            },
        }

    def _upload_batch(self, documents: list[dict]) -> None:
        """Upload a batch of documents to AnythingLLM."""
        try:
            # First, ensure workspace exists
            self._ensure_workspace()

            # Upload documents
            for doc in documents:
                document_url = f"{self.base_url}/api/v1/document"
                self.privacy.log_outbound(document_url, "index_upload")
                response = httpx.post(
                    document_url,
                    headers=self._headers(),
                    json={
                        "title": doc["title"],
                        "content": doc["content"],
                    },
                    timeout=30.0,
                )
                response.raise_for_status()

                # Add to workspace
                doc_id = response.json().get("id")
                if doc_id:
                    self._add_to_workspace(doc_id)

        except PrivacyError:
            raise
        except Exception as e:
            logger.warning("AnythingLLM upload failed: %s", e)

    def _ensure_workspace(self) -> None:
        """Create workspace if it doesn't exist."""
        try:
            workspaces_url = f"{self.base_url}/api/v1/workspaces"
            self.privacy.log_outbound(workspaces_url, "index_upload")
            response = httpx.get(
                workspaces_url,
                headers=self._headers(),
                timeout=10.0,
            )
            response.raise_for_status()
            workspaces = response.json().get("workspaces", [])

            exists = any(w["slug"] == self.workspace for w in workspaces)
            if not exists:
                new_workspace_url = f"{self.base_url}/api/v1/workspace/new"
                self.privacy.log_outbound(new_workspace_url, "index_upload")
                httpx.post(
                    new_workspace_url,
                    headers=self._headers(),
                    json={
                        "name": self.workspace,
                        "slug": self.workspace,
                    },
                    timeout=10.0,
                ).raise_for_status()
        except PrivacyError:
            raise
        except Exception as e:
            logger.warning("Workspace check failed: %s", e)

    def _add_to_workspace(self, doc_id: str) -> None:
        """Add document to workspace."""
        try:
            update_url = f"{self.base_url}/api/v1/workspace/{self.workspace}/update-embeddings"
            self.privacy.log_outbound(update_url, "index_upload")
            httpx.post(
                update_url,
                headers=self._headers(),
                json={"adds": [doc_id]},
                timeout=30.0,
            ).raise_for_status()
        except PrivacyError:
            raise
        except Exception as e:
            logger.warning("Add to workspace failed: %s", e)

    def chat(self, message: str) -> dict | None:
        """Chat with AnythingLLM workspace about photos."""
        try:
            chat_url = f"{self.base_url}/api/v1/workspace/{self.workspace}/chat"
            self.privacy.log_outbound(chat_url, "anythingllm_chat")
            response = httpx.post(
                chat_url,
                headers=self._headers(),
                json={
                    "message": message,
                    "mode": "query",  # RAG mode with source citations
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()
        except PrivacyError:
            raise
        except Exception as e:
            logger.warning("Chat failed: %s", e)
            return None
