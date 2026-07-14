"""Runtime coverage for advertised HEIC support."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image
from pillow_heif import from_pillow

from privatelens.cli import cli
from privatelens.config import settings
from privatelens.extractors.exif import ExifExtractor
from privatelens.extractors.faces import FaceExtractor
from privatelens.utils.thumbnails import generate_thumbnail


def test_heic_exif_and_thumbnail_decode_real_file(tmp_path, monkeypatch):
    """A real encoded HEIC must work through scan metadata and thumbnails."""
    source_path = tmp_path / "source.heic"
    source_image = Image.new("RGB", (64, 48), color=(20, 100, 180))
    from_pillow(source_image).save(source_path, quality=90)

    metadata = ExifExtractor().extract(source_path)

    assert metadata["valid"] is True
    assert metadata["error"] is None
    assert (metadata["width"], metadata["height"]) == (64, 48)

    monkeypatch.setattr(settings, "thumbnail_dir", tmp_path / "thumbnails")
    thumbnail_path = generate_thumbnail(source_path, asset_id=1, size=32)

    with Image.open(thumbnail_path) as thumbnail:
        assert thumbnail.format == "JPEG"
        assert thumbnail.size == (32, 24)


@pytest.mark.parametrize("suffix", [".heic", ".jpg"])
def test_face_extractor_decodes_pillow_formats_for_insightface(tmp_path, suffix):
    """Face extraction must pass HEIC and ordinary image pixels to InsightFace as BGR."""
    image_path = tmp_path / f"face{suffix}"
    source_image = Image.new("RGB", (64, 48), color=(20, 100, 180))
    if suffix == ".heic":
        from_pillow(source_image).save(image_path, quality=90)
    else:
        source_image.save(image_path, quality=90)
    captured = {}

    class StubFaceApp:
        def get(self, image):
            captured["image"] = image
            return [
                SimpleNamespace(
                    bbox=np.array([1.0, 2.0, 30.0, 40.0]),
                    embedding=np.array([0.25, 0.75], dtype=np.float32),
                    det_score=0.9,
                )
            ]

    extractor = FaceExtractor()
    extractor._app = StubFaceApp()
    extractor._available = True

    faces = extractor.extract(image_path)

    assert faces is not None
    assert faces[0]["bbox"] == {"x1": 1, "y1": 2, "x2": 30, "y2": 40}
    np.testing.assert_allclose(np.frombuffer(faces[0]["embedding"], dtype=np.float32), [0.25, 0.75])
    bgr_image = captured["image"]
    assert bgr_image.shape == (48, 64, 3)
    assert bgr_image.flags.c_contiguous
    np.testing.assert_allclose(bgr_image[0, 0], [180, 100, 20], atol=5)


def test_cli_scan_discovers_real_heif_file(tmp_path):
    """The CLI allowlist must include the HEIF suffix handled by Pillow."""
    image_path = tmp_path / "source.heif"
    source_image = Image.new("RGB", (32, 24), color=(20, 100, 180))
    from_pillow(source_image).save(image_path, quality=90)

    result = CliRunner().invoke(cli, ["scan", str(tmp_path), "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["found"] == 1
