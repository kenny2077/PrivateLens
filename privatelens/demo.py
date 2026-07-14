"""Synthetic demo library generation for reproducible CLI demos."""

import shlex
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEMO_ASSETS: list[dict[str, Any]] = [
    {
        "filename": "target-receipt-lunch.jpg",
        "query": "receipt",
        "title": "TARGET RECEIPT",
        "description": "Receipt with totals and payment text.",
        "size": (720, 1000),
        "background": (250, 248, 238),
        "accent": (168, 35, 35),
        "lines": [
            "Store: Target Downtown",
            "Date: 2026-02-14",
            "Lunch supplies",
            "Subtotal  21.48",
            "Tax        1.72",
            "TOTAL     23.20",
            "Payment: VISA",
        ],
    },
    {
        "filename": "driver-license-backup.jpg",
        "query": "driver license",
        "title": "DRIVER LICENSE",
        "description": "Synthetic government ID style image.",
        "size": (900, 560),
        "background": (229, 239, 250),
        "accent": (36, 77, 135),
        "lines": [
            "STATE OF DEMO",
            "Name: SAMPLE USER",
            "License No: D0000000",
            "DOB: 1990-01-01",
            "Expires: 2030-01-01",
        ],
    },
    {
        "filename": "phone-screenshot-travel.png",
        "query": "screenshot",
        "title": "TRAVEL CHAT SCREENSHOT",
        "description": "Phone screenshot style image.",
        "size": (720, 1280),
        "background": (245, 247, 251),
        "accent": (80, 105, 180),
        "lines": [
            "Messages",
            "Boarding pass is saved",
            "Hotel confirmation: Friday",
            "Screenshot captured at 09:41",
        ],
    },
    {
        "filename": "whiteboard-notes-project.jpg",
        "query": "whiteboard",
        "title": "WHITEBOARD NOTES",
        "description": "Document-like planning notes.",
        "size": (1100, 780),
        "background": (252, 252, 248),
        "accent": (42, 120, 90),
        "lines": [
            "Project Plan",
            "1. Scan existing folders",
            "2. Build private sidecar index",
            "3. Search with evidence cards",
            "4. Ship CLI demo",
        ],
    },
]


def create_demo_library(output_dir: Path, force: bool = False) -> list[dict[str, Any]]:
    """Create a deterministic synthetic photo library for demos."""
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for asset in DEMO_ASSETS:
        path = output_dir / asset["filename"]
        if force or not path.exists():
            _draw_demo_image(path, asset)
        files.append(
            {
                "path": str(path),
                "filename": asset["filename"],
                "query": asset["query"],
                "description": asset["description"],
            }
        )
    return files


def build_demo_commands(output_dir: Path) -> list[str]:
    """Return model-light commands for a reproducible terminal demo."""
    folder = shlex.quote(str(output_dir.expanduser()))
    return [
        f"privatelens scan {folder}",
        "privatelens status",
        "privatelens search receipt --type path --json --limit 5",
        "privatelens recipes --detail",
    ]


def _draw_demo_image(path: Path, asset: dict[str, Any]) -> None:
    image = Image.new("RGB", asset["size"], asset["background"])
    draw = ImageDraw.Draw(image)
    title_font = _font(38)
    body_font = _font(26)
    small_font = _font(18)
    width, height = image.size

    margin = max(36, width // 18)
    accent = asset["accent"]
    draw.rounded_rectangle(
        (margin, margin, width - margin, height - margin),
        radius=18,
        outline=accent,
        width=5,
    )
    draw.rectangle((margin, margin, width - margin, margin + 82), fill=accent)
    draw.text((margin + 28, margin + 22), asset["title"], fill="white", font=title_font)

    y = margin + 125
    for line in asset["lines"]:
        draw.text((margin + 32, y), line, fill=(30, 30, 30), font=body_font)
        y += 58

    draw.text(
        (margin + 32, height - margin - 54),
        "Synthetic PrivateLens demo image. Not a real document.",
        fill=(90, 90, 90),
        font=small_font,
    )
    image.save(path)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()
