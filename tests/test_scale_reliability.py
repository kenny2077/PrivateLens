"""Deterministic release-scale scan/index/search reliability gate."""

import json

import numpy as np
from click.testing import CliRunner
from PIL import Image

import privatelens.cli as cli_module
from privatelens.config import settings
from privatelens.db.schema import reset_engine


def test_one_thousand_photo_scan_index_search_is_crash_free(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    data_dir = tmp_path / "data"
    photos_dir.mkdir()

    seed_path = photos_dir / "photo0000.jpg"
    Image.new("RGB", (8, 8), "blue").save(seed_path, quality=70)
    image_bytes = seed_path.read_bytes()
    for index in range(1, 1_000):
        (photos_dir / f"photo{index:04d}.jpg").write_bytes(image_bytes)

    monkeypatch.setattr(settings, "data_dir", data_dir)
    monkeypatch.setattr(settings, "db_path", None)
    monkeypatch.setattr(settings, "thumbnail_dir", None)
    monkeypatch.setattr(settings, "model_cache_dir", None)
    reset_engine()

    class FakeClipExtractor:
        def extract(self, _path):
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1.0
            return vector

    class FakeOcrExtractor:
        def extract(self, _path):
            return None

    monkeypatch.setattr(cli_module, "ClipExtractor", FakeClipExtractor)
    monkeypatch.setattr(cli_module, "OcrExtractor", FakeOcrExtractor)
    runner = CliRunner()

    try:
        first_scan = runner.invoke(
            cli_module.cli,
            ["--data-dir", str(data_dir), "scan", str(photos_dir), "--json"],
        )
        assert first_scan.exit_code == 0, first_scan.output
        assert json.loads(first_scan.output) == {
            "folder": str(photos_dir.resolve()),
            "recursive": True,
            "dry_run": False,
            "found": 1_000,
            "new": 1_000,
            "updated": 0,
            "unchanged": 0,
            "invalid": 0,
        }

        first_index = runner.invoke(
            cli_module.cli,
            ["--data-dir", str(data_dir), "index", "--batch-size", "100", "--json"],
        )
        assert first_index.exit_code == 0, first_index.output
        index_payload = json.loads(first_index.output)
        assert index_payload["asset_count"] == 1_000
        assert index_payload["indexed"] == 1_000
        assert index_payload["errors"] == 0

        second_scan = runner.invoke(
            cli_module.cli,
            ["--data-dir", str(data_dir), "scan", str(photos_dir), "--json"],
        )
        assert second_scan.exit_code == 0, second_scan.output
        assert json.loads(second_scan.output)["unchanged"] == 1_000

        second_index = runner.invoke(
            cli_module.cli,
            ["--data-dir", str(data_dir), "index", "--batch-size", "100", "--json"],
        )
        assert second_index.exit_code == 0, second_index.output
        assert json.loads(second_index.output)["asset_count"] == 0

        search = runner.invoke(
            cli_module.cli,
            [
                "--data-dir",
                str(data_dir),
                "search",
                "photo0999",
                "--type",
                "path",
                "--limit",
                "1",
                "--json",
            ],
        )
        assert search.exit_code == 0, search.output
        results = json.loads(search.output)["results"]
        assert len(results) == 1
        assert results[0]["path"].endswith("photo0999.jpg")
    finally:
        reset_engine()
