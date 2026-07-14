"""FastAPI web UI for PrivateLens."""

from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import privatelens
from privatelens.config import settings
from privatelens.db.schema import init_db, get_engine
from privatelens.search.engine import SearchEngine
from privatelens.search.recipes import get_recipes

app = FastAPI(title="PrivateLens", version=privatelens.__version__)

# Setup templates and static files
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "web"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize search engine
search_engine = SearchEngine()
search_lock = Lock()


@app.get("/api/health")
async def api_health():
    """Return a cheap liveness response for local and container health checks."""
    return {"status": "ok", "version": privatelens.__version__}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main search page."""
    recipes = get_recipes()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"recipes": recipes},
    )


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    type: Literal["smart", "ocr", "face", "metadata", "path"] = Query(
        "smart", description="Search type"
    ),
    limit: int = Query(50, ge=1, le=200, description="Result limit"),
    recipe: str | None = Query(None, description="Search recipe name"),
):
    """API endpoint for search."""
    with search_lock:
        if recipe:
            results = search_engine.search_by_recipe(recipe, q, limit=limit)
        else:
            results = search_engine.search(q, search_type=type, limit=limit)

    return JSONResponse(
        {
            "query": q,
            "type": type,
            "recipe": recipe,
            "count": len(results),
            "results": results,
        }
    )


@app.get("/api/recipes")
async def api_recipes():
    """List available search recipes."""
    recipes = get_recipes()
    return JSONResponse(
        {
            "recipes": [
                {
                    "name": r.name,
                    "display_name": r.display_name,
                    "description": r.description,
                    "category": r.category,
                }
                for r in recipes
            ]
        }
    )


@app.get("/api/stats")
async def api_stats():
    """Get index statistics."""
    from sqlalchemy.orm import Session
    from privatelens.db.schema import Asset, ImageEmbedding, OcrBlock, Face, Caption

    engine = get_engine()
    with Session(engine) as session:
        total_assets = session.query(Asset).count()
        indexed = session.query(Asset).filter(Asset.indexed_at.isnot(None)).count()
        embeddings = session.query(ImageEmbedding).count()
        ocr_blocks = session.query(OcrBlock).count()
        faces = session.query(Face).count()
        captions = session.query(Caption).count()

    return JSONResponse(
        {
            "total_assets": total_assets,
            "indexed": indexed,
            "embeddings": embeddings,
            "ocr_blocks": ocr_blocks,
            "faces": faces,
            "captions": captions,
        }
    )


@app.get("/api/asset/{asset_id}")
async def api_asset(asset_id: int):
    """Get asset details."""
    from sqlalchemy.orm import Session
    from privatelens.db.schema import Asset, Caption, OcrBlock, Face, Person

    engine = get_engine()
    with Session(engine) as session:
        asset = session.query(Asset).filter_by(id=asset_id).first()
        if not asset:
            return JSONResponse({"error": "Asset not found"}, status_code=404)

        captions = session.query(Caption).filter_by(asset_id=asset_id).all()
        ocr = session.query(OcrBlock).filter_by(asset_id=asset_id).all()
        faces = session.query(Face).filter_by(asset_id=asset_id).all()

        people = []
        for f in faces:
            if f.cluster_id:
                p = session.query(Person).filter_by(id=f.cluster_id).first()
                if p:
                    people.append({"name": p.display_name, "confidence": f.confidence})

        return JSONResponse(
            {
                "id": asset.id,
                "path": asset.path,
                "width": asset.width,
                "height": asset.height,
                "media_type": asset.media_type,
                "exif_datetime": str(asset.exif_datetime) if asset.exif_datetime else None,
                "captions": [c.caption for c in captions],
                "ocr_text": [b.text for b in ocr],
                "people": people,
                "is_sensitive": asset.is_sensitive,
            }
        )


@app.get("/thumbnails/{asset_id}", response_class=FileResponse)
async def thumbnail(asset_id: int):
    """Serve an indexed derivative thumbnail without exposing arbitrary paths."""
    from sqlalchemy.orm import Session

    from privatelens.db.schema import Asset

    engine = get_engine()
    with Session(engine) as session:
        asset = session.get(Asset, asset_id)
        thumbnail_path = asset.thumbnail_path if asset is not None else None

    if not thumbnail_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    candidate = Path(thumbnail_path).resolve()
    allowed_root = settings.resolved_thumbnail_dir.resolve()
    if not candidate.is_relative_to(allowed_root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(candidate, media_type="image/jpeg")
