# Search Recipes

PrivateLens includes 10 built-in search recipes for common photo retrieval tasks.

## Available Recipes

| Recipe | Trigger | Description |
|--------|---------|-------------|
| `find_id_photo` | "driver license", "passport" | Government IDs and licenses |
| `find_selfie` | "selfie" | Likely single-person selfies |
| `find_two_person` | "two people", "couple" | Photos with exactly two detected faces |
| `find_screenshot` | "screenshot" | Screenshots of apps/docs |
| `find_receipt` | "receipt", "invoice" | Receipts and expenses |
| `find_pet` | "my dog", "cat" | Pet photos |
| `find_document` | "whiteboard", "notes" | Documents and notes |
| `find_car` | "car", "dashboard" | Vehicle photos |
| `find_sensitive` | "bank card" | Sensitive documents |
| `find_memory` | "2024", "2024-03" | Photos in an explicit year/month/day range |

## Usage

```bash
# List all recipes
privatelens recipes

# Search with a recipe
privatelens search --recipe find_receipt "Target"
privatelens search --recipe find_selfie "selfie"
privatelens search --recipe find_memory "2024-03"
```

## Recipe Structure

Each recipe is a JSON query plan with:

- `signals` — Weighted search signals (OCR, semantic, face, metadata, detection)
- `filters` — Exclude labels, boost recent, require exact face counts
- `rerank` — Optional VLM rerank prompt

Example:
```json
{
  "signals": [
    {"type": "ocr", "keywords": ["license", "passport"], "weight": 0.4},
    {"type": "detection", "labels": ["document", "id_card"], "weight": 0.3},
    {"type": "semantic", "weight": 0.2},
    {"type": "metadata", "filters": {"aspect_ratio": {"min": 1.2, "max": 1.6}}, "weight": 0.1}
  ],
  "filters": {
    "exclude": ["screenshot", "group_photo"],
    "boost_recent": false
  },
  "rerank": {
    "vlm_prompt": "Is this a government ID, driver's license, or passport?",
    "top_k": 20
  }
}
```

## Custom Recipes

You can add custom recipes to the database:

```python
from privatelens.db.schema import get_engine, SearchRecipe
from sqlalchemy.orm import Session
import json

engine = get_engine()
with Session(engine) as session:
    recipe = SearchRecipe(
        name="find_beach",
        display_name="Find Beach Photos",
        description="Photos at the beach or ocean",
        query_plan=json.dumps({
            "signals": [
                {"type": "semantic", "weight": 0.6},
                {"type": "ocr", "keywords": ["beach", "ocean", "sand"], "weight": 0.4}
            ]
        }),
        category="memory",
        is_builtin=False,
    )
    session.add(recipe)
    session.commit()
```
