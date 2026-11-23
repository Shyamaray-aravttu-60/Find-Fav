"""FastAPI application: JSON API under /api, static frontend served from /."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .lexicon import SUGGESTED_OCCASIONS
from .recommender import Recommender
from .schemas import RecommendRequest, RecommendResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    recommender = Recommender(settings)
    app.state.recommender = recommender
    # Embedding the catalog can take a few seconds; do it off the startup path.
    threading.Thread(target=recommender.warm_up, name="warm-up", daemon=True).start()
    yield


app = FastAPI(
    title="Occasion-Based Product Recommender",
    description="Hybrid keyword + Ollama-embedding product recommendations for a given occasion.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_recommender(request: Request) -> Recommender:
    return request.app.state.recommender


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status", tags=["meta"])
def status(rec: Recommender = Depends(get_recommender)) -> dict:
    """Catalog size, whether embeddings are ready and whether Ollama / its models are reachable."""
    return rec.status()


@app.get("/api/occasions", tags=["meta"])
def occasions() -> dict:
    """Suggested occasions for the quick-pick chips."""
    return {"occasions": SUGGESTED_OCCASIONS}


@app.post("/api/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(body: RecommendRequest, rec: Recommender = Depends(get_recommender)) -> dict:
    """Rank catalog products for an occasion."""
    return rec.recommend(body.occasion, top_k=body.top_k, mode=body.mode, use_llm=body.use_llm)


@app.get("/api/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend_get(
    occasion: str = Query(..., min_length=1, max_length=120),
    top_k: int = Query(12, ge=1, le=50),
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
    use_llm: bool = True,
    rec: Recommender = Depends(get_recommender),
) -> dict:
    """Same as the POST endpoint, convenient for curl / browser testing."""
    occasion = " ".join(occasion.split())
    if not occasion:
        raise HTTPException(status_code=422, detail="occasion must not be blank")
    return rec.recommend(occasion, top_k=top_k, mode=mode, use_llm=use_llm)


# Mounted last so it never shadows /api/* routes.
if settings.frontend_dir.exists():
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
