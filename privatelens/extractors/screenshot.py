"""Screenshot detector."""

from pathlib import Path
from typing import Any


class ScreenshotDetector:
    """Detect screenshots based on heuristics."""

    SCREENSHOT_PATTERNS = [
        "screenshot",
        "screen shot",
        "screen-shot",
        "screencap",
        "capture",
        "snip",
    ]

    def is_screenshot(self, image_path: Path, exif_data: dict[str, Any] | None = None) -> bool:
        """Check if image is likely a screenshot."""
        # Check filename
        name_lower = image_path.stem.lower()
        for pattern in self.SCREENSHOT_PATTERNS:
            if pattern in name_lower:
                return True

        # Check dimensions (screenshots often match common screen resolutions)
        if exif_data and exif_data.get("width") and exif_data.get("height"):
            w, h = exif_data["width"], exif_data["height"]
            common_resolutions = [
                (1920, 1080),
                (2560, 1440),
                (3840, 2160),
                (1440, 900),
                (1680, 1050),
                (1280, 720),
                (1170, 2532),
                (1125, 2436),
                (1080, 1920),  # iPhone
                (1080, 2400),
                (1440, 3200),  # Android
            ]
            for rw, rh in common_resolutions:
                if (w == rw and h == rh) or (w == rh and h == rw):
                    return True

        # Screenshots typically have no camera EXIF
        if exif_data and not exif_data.get("make") and not exif_data.get("model"):
            # But they do have a created time
            if exif_data.get("datetime"):
                return True

        return False

    def detect(self, image_path: Path, exif_data: dict[str, Any] | None = None) -> dict:
        """Return screenshot detection result with confidence."""
        is_ss = self.is_screenshot(image_path, exif_data)
        return {
            "is_screenshot": is_ss,
            "confidence": 0.9 if is_ss else 0.0,
        }
