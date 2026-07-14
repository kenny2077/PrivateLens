"""Search recipes for common photo retrieval tasks."""

import json
import re
from typing import Any

from privatelens.config import settings
from privatelens.db.schema import init_db, SearchRecipe


# Built-in search recipes
BUILTIN_RECIPES = [
    {
        "name": "find_id_photo",
        "display_name": "Find ID / Passport / License",
        "description": "Find photos of government IDs, passports, driver's licenses",
        "category": "identity",
        "query_plan": json.dumps(
            {
                "signals": [
                    {
                        "type": "ocr",
                        "keywords": ["license", "passport", "identification", "id card"],
                        "weight": 0.4,
                    },
                    {
                        "type": "path",
                        "keywords": ["license", "passport", "id card"],
                        "weight": 0.2,
                    },
                    {"type": "detection", "labels": ["document", "id_card"], "weight": 0.3},
                    {"type": "semantic", "weight": 0.2},
                    {
                        "type": "metadata",
                        "filters": {"aspect_ratio": {"min": 1.2, "max": 1.6}},
                        "weight": 0.1,
                    },
                ],
                "filters": {
                    "exclude": ["group_photo"],
                    "boost_recent": False,
                },
                "rerank": {
                    "vlm_prompt": "Is this a government ID, driver's license, or passport?",
                    "top_k": 20,
                },
            }
        ),
    },
    {
        "name": "find_selfie",
        "display_name": "Find Selfies",
        "description": "Find photos of yourself alone (selfies)",
        "category": "social",
        "query_plan": json.dumps(
            {
                "signals": [
                    {"type": "face", "face_count": 1, "weight": 0.5},
                    {"type": "semantic", "weight": 0.3},
                    {
                        "type": "metadata",
                        "filters": {"aspect_ratio": {"min": 0.5, "max": 0.8}},
                        "weight": 0.1,
                    },
                ],
                "filters": {
                    "exclude": ["group_photo", "document"],
                    "boost_recent": True,
                },
                "rerank": {
                    "vlm_prompt": "Is this a selfie of one person?",
                    "top_k": 30,
                },
            }
        ),
    },
    {
        "name": "find_two_person",
        "display_name": "Find Two-Person Photos",
        "description": "Find photos with exactly two people (e.g., you and a friend)",
        "category": "social",
        "query_plan": json.dumps(
            {
                "signals": [
                    {"type": "face", "face_count": 2, "weight": 0.5},
                    {"type": "semantic", "weight": 0.3},
                    {"type": "ocr", "weight": 0.1},
                ],
                "filters": {
                    "exclude": ["group_photo"],
                    "face_count_exact": 2,
                },
                "rerank": {
                    "vlm_prompt": "Are there exactly two people in this photo?",
                    "top_k": 30,
                },
            }
        ),
    },
    {
        "name": "find_screenshot",
        "display_name": "Find Screenshots",
        "description": "Find screenshots of apps, documents, messages",
        "category": "document",
        "query_plan": json.dumps(
            {
                "signals": [
                    {"type": "metadata", "filters": {"media_type": "screenshot"}, "weight": 0.5},
                    {"type": "ocr", "weight": 0.3},
                    {"type": "semantic", "weight": 0.2},
                ],
                "filters": {
                    "exclude": ["selfie", "document"],
                },
                "rerank": {
                    "vlm_prompt": "Is this a screenshot of a phone or computer screen?",
                    "top_k": 20,
                },
            }
        ),
    },
    {
        "name": "find_receipt",
        "display_name": "Find Receipts",
        "description": "Find receipts and expense documents",
        "category": "finance",
        "query_plan": json.dumps(
            {
                "signals": [
                    {
                        "type": "ocr",
                        "keywords": ["receipt", "total", "tax", "transaction", "payment"],
                        "weight": 0.5,
                    },
                    {
                        "type": "path",
                        "keywords": ["receipt", "invoice", "payment"],
                        "weight": 0.2,
                    },
                    {"type": "detection", "labels": ["receipt"], "weight": 0.3},
                    {"type": "semantic", "weight": 0.2},
                ],
                "filters": {
                    "exclude": ["selfie"],
                },
                "rerank": {
                    "vlm_prompt": "Is this a receipt or invoice?",
                    "top_k": 20,
                },
            }
        ),
    },
    {
        "name": "find_pet",
        "display_name": "Find Pet Photos",
        "description": "Find photos of your pets",
        "category": "memory",
        "query_plan": json.dumps(
            {
                "signals": [
                    {"type": "detection", "labels": ["dog", "cat", "pet", "animal"], "weight": 0.5},
                    {"type": "semantic", "weight": 0.4},
                    {"type": "ocr", "weight": 0.1},
                ],
                "filters": {
                    "exclude": ["document", "screenshot"],
                },
                "rerank": {
                    "vlm_prompt": "Is there a pet or animal in this photo?",
                    "top_k": 30,
                },
            }
        ),
    },
    {
        "name": "find_document",
        "display_name": "Find Documents",
        "description": "Find whiteboards, lecture notes, handwritten documents",
        "category": "document",
        "query_plan": json.dumps(
            {
                "signals": [
                    {
                        "type": "detection",
                        "labels": ["document", "whiteboard", "text"],
                        "weight": 0.4,
                    },
                    {"type": "ocr", "weight": 0.4},
                    {"type": "semantic", "weight": 0.2},
                ],
                "filters": {
                    "exclude": ["selfie", "screenshot"],
                },
                "rerank": {
                    "vlm_prompt": "Is this a document, whiteboard, or page with text?",
                    "top_k": 30,
                },
            }
        ),
    },
    {
        "name": "find_car",
        "display_name": "Find Car Photos",
        "description": "Find photos of cars, dashboards, license plates",
        "category": "memory",
        "query_plan": json.dumps(
            {
                "signals": [
                    {
                        "type": "detection",
                        "labels": ["car", "vehicle", "automobile"],
                        "weight": 0.5,
                    },
                    {"type": "semantic", "weight": 0.4},
                    {"type": "ocr", "keywords": ["license plate", "dashboard"], "weight": 0.1},
                ],
                "filters": {
                    "exclude": ["document", "screenshot"],
                },
                "rerank": {
                    "vlm_prompt": "Is there a car or vehicle in this photo?",
                    "top_k": 30,
                },
            }
        ),
    },
    {
        "name": "find_sensitive",
        "display_name": "Find Sensitive Documents",
        "description": "Find sensitive documents (bank cards, IDs, medical records)",
        "category": "security",
        "query_plan": json.dumps(
            {
                "signals": [
                    {"type": "metadata", "filters": {"is_sensitive": True}, "weight": 0.6},
                    {
                        "type": "ocr",
                        "keywords": ["ssn", "credit card", "passport", "medical"],
                        "weight": 0.3,
                    },
                    {"type": "semantic", "weight": 0.1},
                ],
                "filters": {
                    "require_sensitive": True,
                },
                "rerank": None,
            }
        ),
    },
    {
        "name": "find_memory",
        "display_name": "Find Memory Photos",
        "description": "Find photos from specific times, places, or events",
        "category": "memory",
        "query_plan": json.dumps(
            {
                "signals": [
                    {"type": "semantic", "weight": 0.4},
                    {"type": "metadata", "filters": {"date_range": True}, "weight": 0.3},
                    {"type": "face", "weight": 0.2},
                    {"type": "ocr", "weight": 0.1},
                ],
                "filters": {
                    "boost_recent": True,
                },
                "rerank": {
                    "vlm_prompt": "Does this photo match the described memory or event?",
                    "top_k": 30,
                },
            }
        ),
    },
]


AUTO_RECIPE_RULES = [
    ("find_receipt", ("receipt", "invoice", "expense", "payment", "transaction")),
    (
        "find_id_photo",
        ("passport", "driver license", "driver's license", "id card", "government id"),
    ),
    ("find_two_person", ("two people", "two person", "two-person", "couple")),
    ("find_selfie", ("selfie",)),
    ("find_screenshot", ("screenshot", "screen shot")),
    ("find_sensitive", ("sensitive", "credit card", "bank card", "ssn", "medical")),
    ("find_document", ("document", "whiteboard", "notes", "handwritten")),
    ("find_car", ("car", "vehicle", "dashboard", "license plate")),
    ("find_pet", ("pet", "dog", "cat")),
]


def detect_recipe_for_query(query: str) -> str | None:
    """Return a built-in recipe name for common natural-language searches."""
    normalized = query.casefold()
    for recipe_name, keywords in AUTO_RECIPE_RULES:
        if any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])",
                normalized,
            )
            for keyword in keywords
        ):
            return recipe_name

    if re.search(r"\b\d{4}(-\d{2})?(-\d{2})?\b", normalized):
        return "find_memory"
    return None


def init_recipes() -> None:
    """Initialize built-in recipes in database."""
    from sqlalchemy.orm import Session

    engine = init_db()
    with Session(engine) as session:
        for recipe_data in BUILTIN_RECIPES:
            existing = session.query(SearchRecipe).filter_by(name=recipe_data["name"]).first()
            if existing:
                existing.display_name = recipe_data["display_name"]
                existing.description = recipe_data["description"]
                existing.category = recipe_data["category"]
                existing.query_plan = recipe_data["query_plan"]
                existing.is_builtin = True
            else:
                recipe = SearchRecipe(**recipe_data)
                session.add(recipe)
        session.commit()


def get_recipes() -> list[SearchRecipe]:
    """Get all available search recipes."""
    from sqlalchemy.orm import Session

    engine = init_db()
    with Session(engine) as session:
        return session.query(SearchRecipe).all()


def get_recipe(name: str) -> SearchRecipe | None:
    """Get a specific recipe by name."""
    from sqlalchemy.orm import Session

    engine = init_db()
    with Session(engine) as session:
        return session.query(SearchRecipe).filter_by(name=name).first()
