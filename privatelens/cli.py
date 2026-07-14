"""CLI entry point for PrivateLens."""

import asyncio
import importlib.util
import io
import json
import logging
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import click
from click.shell_completion import CompletionItem
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

import privatelens
from privatelens.config import settings
from privatelens.db.schema import init_db
from privatelens.errors import PrivateLensError
from privatelens.extractors.exif import ExifExtractor
from privatelens.extractors.clip import ClipExtractor, clip_model_id
from privatelens.extractors.ocr import OcrExtractor
from privatelens.extractors.faces import FaceExtractor
from privatelens.extractors.vlm import VlmExtractor
from privatelens.extractors.document import DocumentClassifier
from privatelens.extractors.screenshot import ScreenshotDetector
from privatelens.extractors.sensitive import SensitiveDetector
from privatelens.search.engine import SearchEngine
from privatelens.search.recipes import detect_recipe_for_query, init_recipes
from privatelens.privacy.guard import PrivacyGuard
from privatelens.privacy.audit import PrivacyAuditor
from privatelens.utils.image_formats import SUPPORTED_IMAGE_EXTENSIONS
from privatelens.utils.time import utcnow

console = Console()


class PrivateLensGroup(click.Group):
    """Click group that formats expected product errors by default."""

    def main(self, *args, **kwargs):
        try:
            return super().main(*args, **kwargs)
        except PrivateLensError as exc:
            if self._debug_requested(args, kwargs):
                raise
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from exc
        except Exception as exc:
            if self._debug_requested(args, kwargs):
                raise
            click.echo(f"Unexpected error: {exc}", err=True)
            click.echo("Run with --debug to show a traceback.", err=True)
            raise SystemExit(1) from exc

    @staticmethod
    def _debug_requested(args, kwargs) -> bool:
        raw_args = kwargs.get("args")
        if raw_args is None and args:
            raw_args = args[0]
        if raw_args is None:
            raw_args = sys.argv[1:]
        return "--debug" in raw_args


def complete_recipe_names(ctx, param, incomplete):
    """Complete built-in recipe names for shell completion."""
    from privatelens.search.recipes import BUILTIN_RECIPES

    return [
        CompletionItem(recipe["name"], help=recipe["display_name"])
        for recipe in BUILTIN_RECIPES
        if recipe["name"].startswith(incomplete)
    ]


def configure_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure process-wide logging for CLI commands."""
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )


def format_score(score: float) -> str:
    """Format a search score with Rich color markup."""
    if score >= 0.8:
        color = "green"
    elif score >= 0.5:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{score:.3f}[/{color}]"


def iter_with_progress(items, description: str, enabled: bool):
    """Yield items while advancing a Rich progress task when enabled."""
    if not enabled:
        yield from items
        return

    with Progress() as progress:
        task_id = progress.add_task(description, total=len(items))
        for item in items:
            try:
                yield item
            finally:
                progress.advance(task_id)


def prompt_search_feedback(engine, results: list[dict]) -> None:
    """Prompt for a simple useful/not-useful rating on the current search."""
    event_id = getattr(engine, "last_search_event_id", None)
    if event_id is None:
        console.print("[yellow]No search event available for feedback.[/yellow]")
        return

    answer = (
        click.prompt(
            "Were these results useful? [y/n/s]",
            default="s",
            show_default=False,
        )
        .strip()
        .lower()
    )
    if answer in {"y", "yes"}:
        feedback = 1
        result_clicked = None
        if results:
            selected_rank = click.prompt(
                f"Which result was useful? [1-{len(results)}]",
                type=click.IntRange(1, len(results)),
            )
            result_clicked = results[selected_rank - 1].get("asset_id")
    elif answer in {"n", "no"}:
        feedback = -1
        result_clicked = None
    else:
        console.print("[yellow]Feedback skipped.[/yellow]")
        return

    engine.record_feedback(event_id, feedback, result_clicked=result_clicked)
    console.print("[green]Feedback recorded.[/green]")


def _missing_modules(modules: list[str]) -> list[str]:
    """Return import module names that are not available without importing heavy models."""
    return [module for module in modules if importlib.util.find_spec(module) is None]


def build_setup_plan() -> dict[str, Any]:
    """Build first-run setup commands and lightweight dependency diagnostics."""
    install_full = 'python -m pip install --upgrade "privatelens[full]"'
    install_full_dev = 'python -m pip install --upgrade "privatelens[full,dev]"'
    if sys.platform.startswith("linux"):
        cpu_torch = (
            "python -m pip install --upgrade torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        install_full = f"{cpu_torch} && {install_full}"
        install_full_dev = f"{cpu_torch} && {install_full_dev}"

    commands = {
        "create_venv": "python3 -m venv .venv && source .venv/bin/activate",
        "install_core": 'python -m pip install --upgrade "privatelens"',
        "install_full": install_full,
        "install_full_dev": install_full_dev,
        "copy_env": "export PRIVATELENS_LOCAL_ONLY=true",
        "generate_encryption_key": (
            'python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"'
        ),
        "append_encryption_key": (
            "export PRIVATELENS_ENCRYPTION_KEY='<paste the generated key here>'"
        ),
        "make_runtime_dirs": (
            'mkdir -p "$HOME/.privatelens/thumbnails" "$HOME/.privatelens/models"'
        ),
        "install_ollama": "brew install --cask ollama",
        "start_ollama": "open -a Ollama",
        "pull_vlm_model": f"ollama pull {settings.vlm_model}",
        "verify_ollama": f"curl -fsS {settings.ollama_url.rstrip('/')}/api/tags",
        "warm_model_cache": "privatelens benchmark-models --skip-vlm --json",
        "doctor": "privatelens doctor --json",
        "test": "privatelens benchmark --json",
    }
    package_groups: dict[str, dict[str, Any]] = {
        "core": {
            "command": commands["install_core"],
            "modules": [
                "click",
                "fastapi",
                "uvicorn",
                "sqlalchemy",
                "aiosqlite",
                "PIL",
                "numpy",
                "sqlite_vec",
                "imagehash",
                "httpx",
                "pydantic",
                "pydantic_settings",
                "cryptography",
                "rich",
                "watchdog",
                "multipart",
                "jinja2",
            ],
        },
        "ml": {
            "command": commands["install_full"],
            "modules": [
                "open_clip",
                "torch",
                "torchvision",
                "rapidocr_onnxruntime",
                "insightface",
                "onnxruntime",
            ],
        },
        "dev": {
            "command": commands["install_full_dev"],
            "modules": ["pytest", "ruff", "mypy", "pre_commit"],
        },
    }

    for group in package_groups.values():
        group["missing"] = _missing_modules(group["modules"])

    return {
        "commands": commands,
        "package_groups": package_groups,
        "ollama": {
            "url": settings.ollama_url,
            "model": settings.vlm_model,
        },
    }


def delete_table_rows_if_exists(conn, table_name: str) -> None:
    """Delete rows from an optional SQLite table when it exists."""
    from sqlalchemy import text

    exists = conn.execute(
        text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name"),
        {"name": table_name},
    ).first()
    if exists:
        conn.execute(text(f"DELETE FROM {table_name}"))


def invoke_json_command(ctx, command, **kwargs):
    """Invoke a sibling Click command and return its JSON payload."""
    output = io.StringIO()
    with redirect_stdout(output):
        ctx.invoke(command, json_output=True, **kwargs)
    return json.loads(output.getvalue())


def prune_watched_folder(folder: Path):
    """Remove missing assets scoped to one watched subtree."""
    from sqlalchemy.orm import Session
    from privatelens.db.schema import Asset

    watched_root = folder.expanduser().resolve()
    engine = init_db()
    with Session(engine) as session:
        missing_assets = []
        for asset in session.query(Asset).all():
            asset_path = Path(asset.path).expanduser().resolve()
            if asset_path.is_relative_to(watched_root) and not asset_path.exists():
                missing_assets.append(asset)

        for asset in missing_assets:
            if asset.thumbnail_path:
                Path(asset.thumbnail_path).unlink(missing_ok=True)
            session.delete(asset)
        session.commit()

    removed = len(missing_assets)
    return {"missing_count": removed, "removed": removed}


def run_watch_cycle(ctx, folder: Path, *, recursive, skip_face, skip_vlm, batch_size):
    """Run one scan/index cycle through the canonical CLI implementations."""
    scan_result = invoke_json_command(
        ctx,
        scan,
        folder=str(folder),
        recursive=recursive,
        dry_run=False,
    )
    prune_result = prune_watched_folder(folder)
    index_result = invoke_json_command(
        ctx,
        index,
        folder=str(folder),
        force=False,
        skip_face=skip_face,
        skip_vlm=skip_vlm,
        batch_size=batch_size,
        only_face=False,
        only_vlm=False,
        dry_run=False,
    )
    return {"scan": scan_result, "prune": prune_result, "index": index_result}


@click.group(cls=PrivateLensGroup)
@click.version_option(version=privatelens.__version__)
@click.option("--data-dir", type=click.Path(), help="Data directory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--debug", is_flag=True, help="Show debug logs and tracebacks")
@click.pass_context
def cli(ctx, data_dir, verbose, debug):
    """PrivateLens - Local-first private photo memory indexer."""
    configure_logging(verbose=verbose, debug=debug)
    ctx.ensure_object(dict)
    if data_dir:
        settings.data_dir = Path(data_dir)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    ctx.obj["privacy"] = PrivacyGuard()


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--recursive", "-r", is_flag=True, default=True, help="Scan recursively")
@click.option("--dry-run", is_flag=True, help="Preview images without writing to the index")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def scan(ctx, folder, recursive, dry_run, json_output):
    """Scan a folder for photos and extract metadata."""
    from sqlalchemy.orm import Session
    from privatelens.db.schema import (
        Asset,
        Caption,
        Detection,
        Face,
        ImageEmbedding,
        OcrBlock,
        SensitiveItem,
    )

    if not json_output:
        console.print(f"[bold green]Scanning {folder}...[/bold green]")

    folder_path = Path(folder)

    files = []
    if recursive:
        files = [
            p for p in folder_path.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    else:
        files = [p for p in folder_path.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]

    if not json_output:
        console.print(f"Found {len(files)} images")

    if dry_run:
        if json_output:
            click.echo(
                json.dumps(
                    {
                        "folder": str(folder_path),
                        "recursive": recursive,
                        "dry_run": True,
                        "found": len(files),
                    },
                    indent=2,
                )
            )
            return
        console.print(f"[yellow]Dry run: would scan {len(files)} images.[/yellow]")
        return

    # Ensure data dir exists
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.resolved_thumbnail_dir.mkdir(parents=True, exist_ok=True)

    # Init DB
    engine = init_db()

    exif_extractor = ExifExtractor()
    doc_classifier = DocumentClassifier()
    screenshot_detector = ScreenshotDetector()
    new_count = 0
    updated_count = 0
    unchanged_count = 0
    invalid_count = 0

    with Session(engine) as session:
        for file_path in iter_with_progress(files, "Scanning images", enabled=not json_output):
            # Extract EXIF and content hash.
            exif_data = exif_extractor.extract(file_path)
            if not exif_data.get("valid", True):
                invalid_count += 1
                continue

            # Classify media type
            media_type = "image"
            if screenshot_detector.is_screenshot(file_path, exif_data):
                media_type = "screenshot"
            elif doc_classifier.is_document(file_path, exif_data):
                media_type = "document"

            sha256 = exif_data.get("sha256", "")
            existing = session.query(Asset).filter_by(path=str(file_path)).first()
            if existing:
                content_changed = existing.sha256 != sha256
                classification_changed = existing.media_type != media_type
                needs_reindex = content_changed or classification_changed
                if needs_reindex:
                    updated_count += 1
                else:
                    unchanged_count += 1
                existing.sha256 = sha256
                existing.phash = exif_data.get("phash")
                existing.width = exif_data.get("width")
                existing.height = exif_data.get("height")
                existing.file_size = exif_data.get("file_size")
                existing.media_type = media_type
                existing.modified_at = exif_data.get("modified_at")
                existing.exif_datetime = exif_data.get("datetime")
                existing.exif_make = exif_data.get("make")
                existing.exif_model = exif_data.get("model")
                existing.gps_lat = exif_data.get("gps_lat")
                existing.gps_lng = exif_data.get("gps_lng")
                existing.last_seen_at = utcnow()
                if needs_reindex:
                    if existing.thumbnail_path:
                        Path(existing.thumbnail_path).unlink(missing_ok=True)
                    existing.thumbnail_path = None
                    existing.indexed_at = None
                    existing.is_sensitive = False
                    existing.sensitive_type = None
                    session.query(ImageEmbedding).filter_by(asset_id=existing.id).delete()
                    session.query(OcrBlock).filter_by(asset_id=existing.id).delete()
                    session.query(Face).filter_by(asset_id=existing.id).delete()
                    session.query(Caption).filter_by(asset_id=existing.id).delete()
                    session.query(Detection).filter_by(asset_id=existing.id).delete()
                    session.query(SensitiveItem).filter_by(asset_id=existing.id).delete()
                continue

            # Create asset record
            asset = Asset(
                path=str(file_path),
                sha256=sha256,
                phash=exif_data.get("phash"),
                width=exif_data.get("width"),
                height=exif_data.get("height"),
                file_size=exif_data.get("file_size"),
                media_type=media_type,
                modified_at=exif_data.get("modified_at"),
                exif_datetime=exif_data.get("datetime"),
                exif_make=exif_data.get("make"),
                exif_model=exif_data.get("model"),
                gps_lat=exif_data.get("gps_lat"),
                gps_lng=exif_data.get("gps_lng"),
                last_seen_at=utcnow(),
            )
            session.add(asset)
            new_count += 1

        session.commit()

    if json_output:
        click.echo(
            json.dumps(
                {
                    "folder": str(folder_path),
                    "recursive": recursive,
                    "dry_run": False,
                    "found": len(files),
                    "new": new_count,
                    "updated": updated_count,
                    "unchanged": unchanged_count,
                    "invalid": invalid_count,
                },
                indent=2,
            )
        )
        return

    console.print(
        "[bold green]Scan complete! "
        f"Found {len(files)} images. "
        f"New: {new_count}. Updated: {updated_count}. Unchanged: {unchanged_count}."
        f" Invalid: {invalid_count}."
        "[/bold green]"
    )


@cli.command()
@click.option("--folder", type=click.Path(), help="Specific folder to index")
@click.option("--force", is_flag=True, help="Force re-index")
@click.option(
    "--skip-face/--with-face",
    default=True,
    help="Skip optional face extraction unless explicitly enabled",
)
@click.option(
    "--skip-vlm/--with-vlm",
    default=True,
    help="Skip optional Ollama captions unless explicitly enabled",
)
@click.option(
    "--batch-size",
    type=click.IntRange(1),
    default=5,
    help="Photos per batch (lower = less RAM)",
)
@click.option(
    "--only-face",
    is_flag=True,
    help="Only run optional face extraction (review third-party model terms)",
)
@click.option("--only-vlm", is_flag=True, help="Only run VLM captions (sequential mode)")
@click.option("--dry-run", is_flag=True, help="Preview assets without running extractors")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def index(
    ctx, folder, force, skip_face, skip_vlm, batch_size, only_face, only_vlm, dry_run, json_output
):
    """Generate embeddings, OCR, and thumbnails for scanned photos.

    Sequential mode for low-RAM machines:
      --only-face  : Run only face detection (unload other models)
      --only-vlm   : Run only VLM captions (unload other models)

    Example workflow on 8GB MacBook:
      1. privatelens index --skip-face --skip-vlm       # CLIP + OCR only
      2. privatelens index --only-face --batch-size 3   # Face detection
      3. privatelens index --only-vlm --batch-size 2    # VLM captions
    """
    from sqlalchemy import or_
    from sqlalchemy.orm import Session
    from privatelens.db.schema import (
        Asset,
        Caption,
        Detection,
        Face,
        ImageEmbedding,
        OcrBlock,
        SensitiveItem,
    )
    from privatelens.extractors.detections import derive_detections
    from privatelens.privacy.encrypt import MetadataEncryptor

    if not json_output:
        console.print("[bold green]Starting indexing...[/bold green]")

    engine = init_db()

    # Sequential mode: only load one extractor at a time
    if only_face:
        skip_face = False
        skip_vlm = True
        if not json_output:
            console.print("[yellow]Sequential mode: Face detection only[/yellow]")
    elif only_vlm:
        skip_face = True
        skip_vlm = False
        if not json_output:
            console.print("[yellow]Sequential mode: VLM captions only[/yellow]")

    with Session(engine) as session:
        query = session.query(Asset)
        if folder:
            query = query.filter(Asset.path.like(f"{folder}%"))
        if not force:
            # In sequential mode, skip already-indexed assets for that component
            if only_face:
                query = query.filter(~Asset.faces.any())
            elif only_vlm:
                query = query.filter(~Asset.captions.any())
            else:
                expected_clip_model = clip_model_id(settings.clip_model, settings.clip_pretrained)
                query = query.outerjoin(ImageEmbedding).filter(
                    or_(
                        Asset.indexed_at.is_(None),
                        ImageEmbedding.asset_id.is_(None),
                        ImageEmbedding.model != expected_clip_model,
                    )
                )

        assets = query.all()
        if not json_output:
            console.print(f"Indexing {len(assets)} assets...")

        if dry_run:
            if json_output:
                click.echo(
                    json.dumps(
                        {
                            "dry_run": True,
                            "asset_count": len(assets),
                        },
                        indent=2,
                    )
                )
                return
            console.print(f"[yellow]Dry run: would index {len(assets)} assets.[/yellow]")
            return

        if not assets:
            if json_output:
                click.echo(
                    json.dumps(
                        {
                            "dry_run": False,
                            "asset_count": 0,
                            "indexed": 0,
                            "skipped_missing": 0,
                            "errors": 0,
                            "model_migrations": 0,
                        },
                        indent=2,
                    )
                )
                return
            console.print("[yellow]No assets to index.[/yellow]")
            return

        existing_assets = []
        skipped_missing = 0
        for asset in assets:
            if Path(asset.path).exists():
                existing_assets.append(asset)
            else:
                skipped_missing += 1

        if skipped_missing:
            if not json_output:
                console.print(f"[yellow]Skipped missing files: {skipped_missing}[/yellow]")

        if not existing_assets:
            if json_output:
                click.echo(
                    json.dumps(
                        {
                            "dry_run": False,
                            "asset_count": len(assets),
                            "indexed": 0,
                            "skipped_missing": skipped_missing,
                            "errors": 0,
                            "model_migrations": 0,
                        },
                        indent=2,
                    )
                )
                return
            console.print(
                "[bold green]Indexing complete! "
                f"Indexed 0 assets. Skipped missing files: {skipped_missing}. Errors: 0."
                "[/bold green]"
            )
            return

        # Initialize extractors (lazy-loaded, so only used ones consume RAM)
        clip_extractor = None if (only_face or only_vlm) else ClipExtractor()
        ocr_extractor = None if (only_face or only_vlm) else OcrExtractor()
        face_extractor = None if skip_face else FaceExtractor(settings.face_model)
        vlm_extractor = None if skip_vlm else VlmExtractor()
        document_classifier = DocumentClassifier()
        sensitive_detector = SensitiveDetector() if settings.sensitive_scan else None
        metadata_encryptor = MetadataEncryptor()

        indexed_count = 0
        error_count = 0
        model_migrations = 0
        expected_clip_model = clip_model_id(settings.clip_model, settings.clip_pretrained)
        for i, asset in enumerate(
            iter_with_progress(existing_assets, "Indexing assets", enabled=not json_output)
        ):
            file_path = Path(asset.path)
            savepoint = session.begin_nested()
            asset_model_migrated = False

            try:
                refresh_base_signals = force or asset.indexed_at is None
                refresh_detections = not only_face and (refresh_base_signals or only_vlm)
                existing_detection_values = []
                if refresh_detections:
                    existing_detection_values = [
                        (row.label, row.confidence, row.source_model)
                        for row in session.query(Detection).filter_by(asset_id=asset.id).all()
                    ]
                if force and not only_face and not only_vlm:
                    session.query(ImageEmbedding).filter_by(asset_id=asset.id).delete()
                    session.query(OcrBlock).filter_by(asset_id=asset.id).delete()
                    session.query(SensitiveItem).filter_by(asset_id=asset.id).delete()
                    asset.is_sensitive = False
                    asset.sensitive_type = None

                # Thumbnail (only in full mode or if missing)
                if not only_face and not only_vlm:
                    from privatelens.utils.thumbnails import generate_thumbnail

                    if not asset.thumbnail_path:
                        thumb_path = generate_thumbnail(file_path, asset.id)
                        asset.thumbnail_path = str(thumb_path)

                # CLIP embedding
                if clip_extractor is not None:
                    embedding = clip_extractor.extract(file_path)
                    if embedding is not None:
                        existing = (
                            session.query(ImageEmbedding).filter_by(asset_id=asset.id).first()
                        )
                        if existing is None:
                            emb = ImageEmbedding(
                                asset_id=asset.id,
                                model=expected_clip_model,
                                vector=embedding.tobytes(),
                            )
                            session.add(emb)
                        else:
                            if existing.model != expected_clip_model:
                                asset_model_migrated = True
                            existing.model = expected_clip_model
                            existing.vector = embedding.tobytes()
                            existing.generated_at = utcnow()

                # OCR
                ocr_text = None
                if ocr_extractor is not None and refresh_base_signals:
                    ocr_text = ocr_extractor.extract(file_path)
                    if ocr_text:
                        for block in ocr_text:
                            ocr_block = OcrBlock(
                                asset_id=asset.id,
                                text=block["text"],
                                bbox=json.dumps(block.get("bbox")),
                                confidence=block.get("confidence"),
                            )
                            session.add(ocr_block)

                # Faces
                if face_extractor is not None and (only_face or refresh_base_signals):
                    faces = face_extractor.extract(file_path)
                    if faces is not None:
                        if force:
                            session.query(Face).filter_by(asset_id=asset.id).delete()
                        for face in faces:
                            f = Face(
                                asset_id=asset.id,
                                bbox=json.dumps(face["bbox"]),
                                embedding=face.get("embedding"),
                                confidence=face.get("confidence"),
                            )
                            session.add(f)

                # VLM caption and optional structured classification
                vlm_caption = None
                vlm_classification = None
                detection_vlm_model = None
                vlm_attempted = False
                vlm_classification_supported = False
                if vlm_extractor is not None and (only_vlm or refresh_base_signals):
                    vlm_attempted = True
                    detection_vlm_model = getattr(vlm_extractor, "model", settings.vlm_model)
                    vlm_caption = vlm_extractor.caption(file_path)
                    if vlm_caption:
                        if force:
                            session.query(Caption).filter_by(asset_id=asset.id).delete()
                        cap = Caption(
                            asset_id=asset.id,
                            model=detection_vlm_model,
                            caption=vlm_caption,
                            confidence=0.8,
                        )
                        session.add(cap)
                    classify_document = getattr(vlm_extractor, "classify_document", None)
                    if callable(classify_document):
                        vlm_classification_supported = True
                        vlm_classification = classify_document(file_path)

                # Sensitive detection (only in full mode with OCR)
                sensitive = None
                if (
                    sensitive_detector is not None
                    and ocr_extractor is not None
                    and ocr_text is not None
                ):
                    sensitive = sensitive_detector.detect(file_path, ocr_text)
                    if sensitive:
                        asset.is_sensitive = True
                        asset.sensitive_type = sensitive["type"]
                        si = SensitiveItem(
                            asset_id=asset.id,
                            type=sensitive["type"],
                            confidence=sensitive["confidence"],
                            encrypted_metadata=metadata_encryptor.encrypt(
                                {
                                    "type": sensitive["type"],
                                    "confidence": sensitive["confidence"],
                                    "source": sensitive.get("source"),
                                }
                            ),
                        )
                        session.add(si)

                vlm_refresh_complete = (
                    vlm_attempted
                    and vlm_caption is not None
                    and (not vlm_classification_supported or vlm_classification is not None)
                )
                replace_detections = refresh_detections and not (
                    only_vlm and not vlm_refresh_complete
                )
                if replace_detections:
                    detection_ocr = ocr_text
                    if detection_ocr is None:
                        stored_ocr = session.query(OcrBlock).filter_by(asset_id=asset.id).all()
                        detection_ocr = [
                            {"text": block.text, "confidence": block.confidence}
                            for block in stored_ocr
                        ]

                    if sensitive is None and asset.sensitive_type:
                        stored_sensitive = session.get(SensitiveItem, asset.id)
                        sensitive = {
                            "type": asset.sensitive_type,
                            "confidence": (
                                stored_sensitive.confidence if stored_sensitive is not None else 0.7
                            ),
                            "source": "stored",
                        }

                    if vlm_caption is None:
                        stored_caption = (
                            session.query(Caption)
                            .filter_by(asset_id=asset.id)
                            .order_by(Caption.created_at.desc())
                            .first()
                        )
                        if stored_caption is not None:
                            vlm_caption = stored_caption.caption
                            detection_vlm_model = stored_caption.model

                    document_classification = document_classifier.classify(
                        file_path,
                        detection_ocr,
                    )
                    derived_detections = derive_detections(
                        media_type=asset.media_type,
                        ocr_blocks=detection_ocr,
                        document_classification=document_classification,
                        sensitive_detection=sensitive,
                        vlm_classification=vlm_classification,
                        vlm_caption=vlm_caption,
                        vlm_model=detection_vlm_model,
                    )
                    merged_detections = {
                        detection.label: (
                            detection.confidence,
                            detection.source_model,
                        )
                        for detection in derived_detections
                    }
                    preserve_existing_vlm = not vlm_refresh_complete
                    preserve_existing_base = only_vlm
                    for label, confidence, source_model in existing_detection_values:
                        is_vlm = (source_model or "").startswith("vlm:")
                        if not (
                            (is_vlm and preserve_existing_vlm)
                            or (not is_vlm and preserve_existing_base)
                        ):
                            continue
                        score = float(confidence) if confidence is not None else 0.0
                        if not math.isfinite(score):
                            score = 0.0
                        current = merged_detections.get(label)
                        if current is None or score > current[0]:
                            merged_detections[label] = (score, source_model or "unknown")

                    session.query(Detection).filter_by(asset_id=asset.id).delete()
                    for label, (confidence, source_model) in merged_detections.items():
                        session.add(
                            Detection(
                                asset_id=asset.id,
                                label=label,
                                confidence=confidence,
                                source_model=source_model,
                            )
                        )

                if not only_face and not only_vlm:
                    asset.indexed_at = utcnow()
                session.flush()
                savepoint.commit()
                if asset_model_migrated:
                    model_migrations += 1

            except Exception as e:
                savepoint.rollback()
                error_count += 1
                if not json_output:
                    console.print(f"[red]Error indexing {asset.path}: {e}[/red]")
                continue

            indexed_count += 1
            if indexed_count % batch_size == 0:
                session.commit()

        session.commit()

    if json_output:
        click.echo(
            json.dumps(
                {
                    "dry_run": False,
                    "asset_count": len(assets),
                    "indexed": indexed_count,
                    "skipped_missing": skipped_missing,
                    "errors": error_count,
                    "model_migrations": model_migrations,
                },
                indent=2,
            )
        )
        return

    console.print(
        "[bold green]Indexing complete! "
        f"Indexed {indexed_count} assets. "
        f"Skipped missing files: {skipped_missing}. "
        f"Model migrations: {model_migrations}. "
        f"Errors: {error_count}."
        "[/bold green]"
    )


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--recursive/--no-recursive", default=True, help="Watch subdirectories")
@click.option(
    "--debounce",
    type=click.FloatRange(min=0.1),
    default=2.0,
    show_default=True,
    help="Quiet seconds before processing a batch of changes",
)
@click.option(
    "--initial-scan/--no-initial-scan", default=True, help="Process the folder at startup"
)
@click.option("--skip-face/--with-face", default=True, help="Skip face detection during indexing")
@click.option("--skip-vlm/--with-vlm", default=True, help="Skip VLM captions during indexing")
@click.option(
    "--batch-size",
    type=click.IntRange(1),
    default=1,
    show_default=True,
    help="Photos per indexing batch",
)
@click.option("--json", "json_output", is_flag=True, help="Output newline-delimited JSON events")
@click.pass_context
def watch(
    ctx,
    folder,
    recursive,
    debounce,
    initial_scan,
    skip_face,
    skip_vlm,
    batch_size,
    json_output,
):
    """Watch a photo folder and index debounced changes."""
    from privatelens.watcher import watch_for_changes

    folder = folder.expanduser().resolve()
    if not json_output:
        console.print(f"[bold green]Watching {folder}[/bold green]")
        console.print("[dim]Press Ctrl-C to stop.[/dim]")

    def process_cycle(trigger):
        started_at = perf_counter()
        try:
            payload = run_watch_cycle(
                ctx,
                folder,
                recursive=recursive,
                skip_face=skip_face,
                skip_vlm=skip_vlm,
                batch_size=batch_size,
            )
            record = {
                "event": "cycle",
                "trigger": trigger,
                "folder": str(folder),
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 3),
                **payload,
            }
            if json_output:
                click.echo(json.dumps(record))
                return

            scan_summary = record["scan"]
            index_summary = record["index"]
            console.print(
                f"[green]{trigger.capitalize()} cycle complete:[/green] "
                f"found {scan_summary['found']}, "
                f"new {scan_summary['new']}, updated {scan_summary['updated']}, "
                f"pruned {record['prune']['removed']}, "
                f"indexed {index_summary['indexed']}, errors {index_summary['errors']} "
                f"in {record['elapsed_ms']:.1f} ms"
            )
        except Exception as exc:
            record = {
                "event": "error",
                "trigger": trigger,
                "folder": str(folder),
                "error": str(exc),
            }
            if json_output:
                click.echo(json.dumps(record))
            else:
                console.print(f"[red]Watch cycle failed: {exc}[/red]")

    watch_for_changes(
        folder,
        process_cycle,
        recursive=recursive,
        debounce=debounce,
        initial_scan=initial_scan,
    )

    stopped = {"event": "stopped", "folder": str(folder)}
    if json_output:
        click.echo(json.dumps(stopped))
    else:
        console.print("[yellow]Watcher stopped.[/yellow]")


@cli.command()
@click.argument("query")
@click.option(
    "--type",
    "search_type",
    type=click.Choice(["smart", "ocr", "face", "metadata", "path"]),
    default="smart",
    show_default=True,
    help="Search signal",
)
@click.option(
    "--limit",
    type=click.IntRange(1, 200),
    default=50,
    show_default=True,
    help="Number of results",
)
@click.option("--recipe", shell_complete=complete_recipe_names, help="Use a search recipe")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.option(
    "--open", "open_result", is_flag=True, help="Open the first result in the system viewer"
)
@click.option("--fast", is_flag=True, help="Skip VLM reranking for recipe searches")
@click.option("--feedback", is_flag=True, help="Prompt to rate human search results")
@click.pass_context
def search(ctx, query, search_type, limit, recipe, json_output, open_result, fast, feedback):
    """Search photos with natural language or filters."""
    started_at = perf_counter()
    engine = SearchEngine()
    active_recipe = recipe
    if active_recipe is None and search_type == "smart":
        active_recipe = detect_recipe_for_query(query)

    if active_recipe:
        recipe_options = {"limit": limit, "rerank": not fast}
        if feedback:
            recipe_options["record_event"] = True
        results = engine.search_by_recipe(active_recipe, query, **recipe_options)
    else:
        search_options = {"search_type": search_type, "limit": limit}
        if feedback:
            search_options["record_event"] = True
        results = engine.search(query, **search_options)
    elapsed_ms = round((perf_counter() - started_at) * 1000, 3)

    if json_output:
        click.echo(
            json.dumps(
                {
                    "query": query,
                    "type": search_type,
                    "recipe": active_recipe,
                    "fast": fast,
                    "elapsed_ms": elapsed_ms,
                    "count": len(results),
                    "results": results,
                },
                indent=2,
            )
        )
        return

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        console.print(f"[dim]Searched in {elapsed_ms:.1f} ms[/dim]")
        if feedback:
            prompt_search_feedback(engine, results)
        return

    table = Table(title=f"Search Results: '{query}'")
    table.add_column("Rank", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Score")
    table.add_column("Why Matched", style="blue")

    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            result["path"][:60],
            format_score(result["score"]),
            result.get("explanation", "")[:80],
        )

    console.print(table)
    console.print(f"[dim]Searched in {elapsed_ms:.1f} ms[/dim]")
    if open_result:
        first_path = results[0]["path"]
        click.launch(first_path)
        console.print(f"[green]Opened {first_path}[/green]")
    if feedback:
        prompt_search_feedback(engine, results)


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def status(ctx, json_output):
    """Show local index status without loading ML models."""
    from sqlalchemy.orm import Session
    from privatelens.db.schema import (
        Asset,
        Caption,
        Face,
        ImageEmbedding,
        OcrBlock,
        SensitiveItem,
    )

    engine = init_db()
    with Session(engine) as session:
        counts = {
            "total_assets": session.query(Asset).count(),
            "indexed_assets": session.query(Asset).filter(Asset.indexed_at.isnot(None)).count(),
            "embeddings": session.query(ImageEmbedding).count(),
            "ocr_blocks": session.query(OcrBlock).count(),
            "faces": session.query(Face).count(),
            "captions": session.query(Caption).count(),
            "sensitive_items": session.query(SensitiveItem).count(),
        }

    if json_output:
        click.echo(json.dumps(counts, indent=2))
        return

    table = Table(title="PrivateLens Index Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    for name, count in [
        ("Total Assets", counts["total_assets"]),
        ("Indexed Assets", counts["indexed_assets"]),
        ("Embeddings", counts["embeddings"]),
        ("OCR Blocks", counts["ocr_blocks"]),
        ("Faces", counts["faces"]),
        ("Captions", counts["captions"]),
        ("Sensitive Items", counts["sensitive_items"]),
    ]:
        table.add_row(name, str(count))

    console.print(table)


@cli.command()
@click.pass_context
def quickstart(ctx):
    """Show a safe first-run CLI workflow."""
    console.print("[bold]PrivateLens Quickstart[/bold]")
    console.print("0. Create a safe synthetic demo library:")
    console.print("   privatelens demo --output-dir /tmp/privatelens-demo-photos")
    console.print("1. Scan a folder without moving or importing photos:")
    console.print("   privatelens scan ~/Pictures")
    console.print("2. Index lightly on an 8GB MacBook:")
    console.print("   privatelens index --skip-face --skip-vlm --batch-size 1")
    console.print("3. Search with structured recipes and evidence:")
    console.print("   privatelens search receipt --json --limit 5")
    console.print("4. Run face or VLM passes separately when resources allow:")
    console.print("   privatelens index --only-face --batch-size 1")
    console.print("   privatelens index --only-vlm --batch-size 1")
    console.print("5. Keep the index current with debounced filesystem watching:")
    console.print("   privatelens watch ~/Pictures")


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="privatelens-demo-photos",
    show_default=True,
    help="Directory where synthetic demo photos will be written",
)
@click.option("--force", is_flag=True, help="Overwrite existing generated demo images")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def demo(ctx, output_dir, force, json_output):
    """Create a safe synthetic photo library for terminal demos."""
    from privatelens.demo import build_demo_commands, create_demo_library

    demo_dir = Path(output_dir)
    files = create_demo_library(demo_dir, force=force)
    commands = build_demo_commands(demo_dir)
    payload = {
        "output_dir": str(demo_dir.expanduser()),
        "file_count": len(files),
        "files": files,
        "next_commands": commands,
    }

    if json_output:
        click.echo(json.dumps(payload, indent=2))
        return

    console.print(f"[bold green]Created {len(files)} synthetic demo images.[/bold green]")
    console.print(f"Output: {payload['output_dir']}")
    console.print("Next commands:")
    for command in commands:
        click.echo(f"   {command}")


@cli.command()
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("results/benchmarks/search-quality-v1.json"),
    show_default=True,
    help="Path for the reproducible benchmark report",
)
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def benchmark(ctx, output, json_output):
    """Run the packaged ten-case structured search benchmark."""
    from privatelens.benchmark import run_canonical_benchmark

    output_path = output.expanduser()
    report = run_canonical_benchmark(output_path=output_path)
    if json_output:
        click.echo(json.dumps(report, indent=2))
    else:
        table = Table(title="PrivateLens Search Quality Benchmark")
        table.add_column("Case", style="cyan")
        table.add_column("Recipe", style="green")
        table.add_column("Top-5", justify="center")
        table.add_column("Rank", justify="right")
        for case in report["cases"]:
            table.add_row(
                case["name"],
                case["recipe"],
                "pass" if case["target_in_top_5"] else "fail",
                str(case["first_relevant_rank"] or "-"),
            )
        console.print(table)
        summary = report["summary"]
        console.print(
            f"Hit rate@5: {summary['hit_rate_at_5']:.1%} | "
            f"MRR: {summary['mean_reciprocal_rank']:.3f} | "
            f"Report: {output_path}"
        )

    if not report["summary"]["passed"]:
        raise click.ClickException("Search-quality benchmark did not meet its release gate")


@cli.command(name="benchmark-models")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("results/benchmarks/model-quality-v1.json"),
    show_default=True,
    help="Path for the model-quality benchmark report",
)
@click.option(
    "--vlm/--skip-vlm",
    default=True,
    help="Include local Ollama caption and document-classification gates",
)
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def benchmark_models(ctx, output, vlm, json_output):
    """Run CLIP, OCR, and optional VLM gates on generated images."""
    from privatelens.model_benchmark import run_model_benchmark

    output_path = output.expanduser()
    if json_output:
        incidental_output = io.StringIO()
        with redirect_stdout(incidental_output):
            report = run_model_benchmark(output_path=output_path, include_vlm=vlm)
        click.echo(json.dumps(report, indent=2))
    else:
        report = run_model_benchmark(
            output_path=output_path,
            include_vlm=vlm,
            progress=lambda message: console.print(f"[dim]{message}[/dim]"),
        )
        table = Table(title="PrivateLens Model Quality Benchmark")
        table.add_column("Case", style="cyan")
        table.add_column("CLIP", justify="center")
        table.add_column("OCR", justify="center")
        table.add_column("VLM Type", justify="center")
        table.add_column("Caption", justify="center")
        for case in report["cases"]:
            vlm_case = case["vlm"]
            table.add_row(
                case["name"],
                "pass" if case["clip"]["top_1"] else "fail",
                "pass" if case["ocr"]["top_1"] else "fail",
                ("pass" if vlm_case and vlm_case["classification_correct"] else "fail")
                if vlm_case
                else "skip",
                (f"{vlm_case['caption_term_recall']:.0%}" if vlm_case is not None else "skip"),
            )
        console.print(table)
        summary = report["summary"]
        vlm_summary = (
            f" | VLM type: {summary['vlm_classification_accuracy']:.1%}"
            if summary["vlm_classification_accuracy"] is not None
            else " | VLM: skipped"
        )
        console.print(
            f"CLIP top-1: {summary['clip_top1_rate']:.1%} | "
            f"OCR top-1: {summary['ocr_top1_rate']:.1%}"
            f"{vlm_summary} | Report: {output_path}"
        )

    if not report["summary"]["passed"]:
        raise click.ClickException("Model-quality benchmark did not meet its release gate")


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def setup(ctx, json_output):
    """Show first-run install, model, and verification commands."""
    plan = build_setup_plan()
    if json_output:
        click.echo(json.dumps(plan, indent=2))
        return

    commands = plan["commands"]
    console.print("[bold]PrivateLens Setup[/bold]")
    console.print("Create or activate an environment, then install the published package:")
    click.echo(f"   {commands['create_venv']}")
    click.echo(f"   {commands['install_full']}")

    console.print("Configure local runtime:")
    for key in [
        "copy_env",
        "generate_encryption_key",
        "append_encryption_key",
        "make_runtime_dirs",
    ]:
        click.echo(f"   {commands[key]}")

    console.print("Prepare local VLM and model caches:")
    for key in [
        "install_ollama",
        "start_ollama",
        "pull_vlm_model",
        "verify_ollama",
        "warm_model_cache",
    ]:
        click.echo(f"   {commands[key]}")

    console.print("Verify:")
    click.echo(f"   {commands['doctor']}")
    click.echo(f"   {commands['test']}")

    table = Table(title="Package Diagnostics")
    table.add_column("Group", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Missing modules", style="blue")
    for name, group in plan["package_groups"].items():
        missing = group["missing"]
        table.add_row(name, "ok" if not missing else "missing", ", ".join(missing) or "-")
    console.print(table)


@cli.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
@click.pass_context
def completion(ctx, shell):
    """Print shell completion setup instructions."""
    command = f"_PRIVATELENS_COMPLETE={shell}_source privatelens"
    if shell == "fish":
        console.print("Add this to ~/.config/fish/completions/privatelens.fish:")
        console.print(f"   {command} | source")
        return

    rc_file = "~/.bashrc" if shell == "bash" else "~/.zshrc"
    console.print(f"Add this to {rc_file}:")
    console.print(f'   eval "$({command})"')


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind")
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=8000,
    show_default=True,
    help="TCP port to bind",
)
@click.pass_context
def serve(ctx, host, port):
    """Start the web UI server."""
    import uvicorn
    from privatelens.api import app

    console.print("[bold green]Starting PrivateLens server...[/bold green]")
    browser_host = "localhost" if host in {"127.0.0.1", "0.0.0.0", "::"} else host
    console.print(f"[blue]Open http://{browser_host}:{port} in your browser[/blue]")
    uvicorn.run(app, host=host, port=port, access_log=False)


@cli.command()
@click.option("--detail", is_flag=True, help="Show query plan signal/filter summaries")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def recipes(ctx, detail, json_output):
    """List available search recipes."""
    from privatelens.search.recipes import get_recipes

    init_recipes()  # Ensure recipes are initialized
    recipes = get_recipes()

    if json_output:
        click.echo(
            json.dumps(
                {
                    "count": len(recipes),
                    "recipes": [
                        {
                            "name": r.name,
                            "display_name": r.display_name,
                            "category": r.category,
                            "description": r.description,
                            "query_plan": json.loads(r.query_plan) if detail else None,
                        }
                        for r in recipes
                    ],
                },
                indent=2,
            )
        )
        return

    table = Table(title="Search Recipes")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Display Name", style="green")
    table.add_column("Category", style="magenta")
    table.add_column("Description", style="blue")
    if detail:
        table.add_column("Signals", style="yellow")
        table.add_column("Filters", style="cyan")
        table.add_column("Rerank", style="green")

    for r in recipes:
        row = [r.name, r.display_name, r.category or "", r.description or ""]
        if detail:
            plan = json.loads(r.query_plan)
            signals = ", ".join(signal["type"] for signal in plan.get("signals", [])) or "-"
            filters = ", ".join(plan.get("filters", {}).keys()) or "-"
            rerank = "vlm" if plan.get("rerank") else "-"
            row.extend([signals, filters, rerank])
        table.add_row(*row)

    console.print(table)


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def doctor(ctx, json_output):
    """Run diagnostics and privacy audit."""
    init_db()
    auditor = PrivacyAuditor()
    report = auditor.run_audit()

    if json_output:
        click.echo(
            json.dumps(
                {
                    "count": len(report),
                    "checks": report,
                },
                indent=2,
            )
        )
        return

    table = Table(title="PrivateLens Doctor Report")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="blue")
    table.add_column("Remediation", style="yellow")

    for check in report:
        status_color = "green" if check["status"] == "ok" else "red"
        remediation = "\n".join(check.get("remediation", [])) or "-"
        table.add_row(
            check["name"],
            f"[{status_color}]{check['status']}[/{status_color}]",
            check["details"],
            remediation,
        )

    console.print(table)


@cli.command()
@click.option("--person-id", type=int, help="Assign name to a specific person")
@click.option("--name", help="Name to assign to person")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def cluster(ctx, person_id, name, json_output):
    """Cluster faces into people and optionally assign names."""
    from privatelens.extractors.clustering import FaceClusterer

    clusterer = FaceClusterer()

    if person_id and name:
        success = clusterer.assign_name(person_id, name)
        if json_output:
            click.echo(
                json.dumps(
                    {
                        "action": "assign_name",
                        "person_id": person_id,
                        "name": name,
                        "assigned": success,
                    },
                    indent=2,
                )
            )
            return
        if success:
            console.print(f"[bold green]Named person {person_id} as '{name}'[/bold green]")
        else:
            console.print(f"[red]Person {person_id} not found[/red]")
        return

    if not json_output:
        console.print("[bold green]Clustering faces into people...[/bold green]")
    people_created = clusterer.cluster_all()
    if json_output:
        click.echo(
            json.dumps(
                {
                    "action": "cluster",
                    "people_created": people_created,
                },
                indent=2,
            )
        )
        return
    console.print(f"[bold green]Created {people_created} new people clusters.[/bold green]")
    console.print("[blue]Use --person-id and --name to assign names.[/blue]")


@cli.command()
@click.option("--yes", is_flag=True, help="Remove missing-file records from the index")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def prune(ctx, yes, json_output):
    """Find or remove index records whose photo files no longer exist."""
    from sqlalchemy.orm import Session
    from privatelens.db.schema import Asset

    engine = init_db()
    with Session(engine) as session:
        assets = session.query(Asset).all()
        missing_assets = [asset for asset in assets if not Path(asset.path).exists()]
        paths = sorted(asset.path for asset in missing_assets)
        removed = 0
        if yes:
            for asset in missing_assets:
                session.delete(asset)
            session.commit()
            removed = len(missing_assets)

    payload = {
        "dry_run": not yes,
        "missing_count": len(paths),
        "removed": removed,
        "paths": paths,
    }
    if json_output:
        click.echo(json.dumps(payload, indent=2))
        return

    if not paths:
        console.print("[green]No missing-file records found.[/green]")
    elif yes:
        console.print(f"[green]Removed {removed} missing-file records from the index.[/green]")
    else:
        console.print(f"[yellow]Found {len(paths)} missing-file records:[/yellow]")
        for path in paths:
            click.echo(f"   {path}")
        console.print("[dim]Dry run. Rerun with --yes to remove these index records.[/dim]")


@cli.command()
@click.option("--faces-only", is_flag=True, help="Purge only face embeddings")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.confirmation_option(prompt="Are you sure you want to purge the index?")
@click.pass_context
def purge(ctx, faces_only, json_output):
    """Purge sidecar data without deleting source photos."""
    from sqlalchemy import text

    engine = init_db()
    purged = "faces" if faces_only else "index"
    assets_removed = 0
    people_removed = 0
    thumbnail_paths: list[str] = []

    with engine.begin() as conn:
        if faces_only:
            people_removed = int(conn.execute(text("SELECT COUNT(*) FROM people")).scalar() or 0)
            conn.execute(text("DELETE FROM faces"))
            delete_table_rows_if_exists(conn, "vec_faces")
            conn.execute(text("DELETE FROM people"))
        else:
            assets_removed = int(conn.execute(text("SELECT COUNT(*) FROM assets")).scalar() or 0)
            thumbnail_paths = [
                row[0]
                for row in conn.execute(
                    text("SELECT thumbnail_path FROM assets WHERE thumbnail_path IS NOT NULL")
                ).fetchall()
            ]
            people_removed = int(conn.execute(text("SELECT COUNT(*) FROM people")).scalar() or 0)
            conn.execute(text("DELETE FROM search_events"))
            conn.execute(text("DELETE FROM image_embeddings"))
            delete_table_rows_if_exists(conn, "vec_image_embeddings")
            conn.execute(text("DELETE FROM ocr_blocks"))
            conn.execute(text("DELETE FROM captions"))
            conn.execute(text("DELETE FROM faces"))
            delete_table_rows_if_exists(conn, "vec_faces")
            conn.execute(text("DELETE FROM detections"))
            conn.execute(text("DELETE FROM sensitive_items"))
            conn.execute(text("DELETE FROM assets"))
            conn.execute(text("DELETE FROM people"))

    thumbnails_removed = 0
    if not faces_only:
        thumbnail_root = settings.resolved_thumbnail_dir.expanduser().resolve()
        for stored_path in thumbnail_paths:
            candidate = Path(stored_path).expanduser().resolve()
            if candidate.is_relative_to(thumbnail_root) and candidate.is_file():
                candidate.unlink()
                thumbnails_removed += 1

    if json_output:
        click.echo(
            json.dumps(
                {
                    "purged": purged,
                    "faces_only": faces_only,
                    "assets_removed": assets_removed,
                    "people_removed": people_removed,
                    "thumbnails_removed": thumbnails_removed,
                    "source_photos_removed": 0,
                },
                indent=2,
            )
        )
        return

    if faces_only:
        console.print("[yellow]Face embeddings purged.[/yellow]")
    else:
        console.print("[bold red]Index purged.[/bold red]")


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
@click.pass_context
def sync_anythingllm(ctx, json_output):
    """Sync photo index to AnythingLLM workspace."""
    from privatelens.integrations.anythingllm import AnythingLLMConnector

    connector = AnythingLLMConnector()
    connector.sync()
    if json_output:
        click.echo(
            json.dumps(
                {
                    "target": "anythingllm",
                    "synced": True,
                },
                indent=2,
            )
        )
        return
    console.print("[bold green]Synced to AnythingLLM![/bold green]")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
