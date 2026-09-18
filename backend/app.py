"""
============================================================
CineMind — FastAPI Application
============================================================
Endpoints:
    GET  /                 -> health check + basic info
    GET  /search?q=...     -> search movies by title (local)
    POST /recommend        -> get recommendations for a movie
    GET  /movie/{movie_id} -> full movie details (local)
    POST /chat             -> chatbot (calls local engine + AI reply)
============================================================
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Load env vars from .env if python-dotenv is available (optional).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional; user can export vars manually

# Import the local recommendation engine (loads artifacts ONCE at import).
from . import recommendation_engine as engine
from . import chatbot as chatbot_module
from . import watchlist as watchlist_mod
from . import preferences as prefs_mod
from . import insights as insights_mod
from .chatbot import parse_whatif_phrase

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------
app = FastAPI(
    title="CineMind",
    description=(
        "Semantic movie recommendation system using Sentence Transformer + "
        "FAISS. Movie data and posters come from local metadata only. "
        "Only the chatbot uses an external AI API."
    ),
    version="1.0.0",
)

# Serve the static frontend (HTML/CSS/JS) from /frontend at the root.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
    # Also serve CSS/JS at their plain relative paths so the same index.html
    # works both from FastAPI (http://127.0.0.1:8000/) and from a plain static
    # preview of the frontend/ folder (IDE live preview, Live Server, etc.).
    _CSS_DIR = _FRONTEND_DIR / "css"
    _JS_DIR = _FRONTEND_DIR / "js"
    if _CSS_DIR.exists():
        app.mount("/css", StaticFiles(directory=str(_CSS_DIR)), name="css")
    if _JS_DIR.exists():
        app.mount("/js", StaticFiles(directory=str(_JS_DIR)), name="js")

# Allow cross-origin requests. The frontend is served from this same
# origin normally, but a preview panel / separate host must also work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------
class RecommendRequest(BaseModel):
    movie: str          # movie title
    n: int = 5          # number of recommendations
    personalized: bool = False   # blend preference-based ranking (additive)


class ChatRequest(BaseModel):
    message: str


class WatchlistAddRequest(BaseModel):
    movie_id: str | int   # accept numeric ids from the JSON payload too


class FeedbackRequest(BaseModel):
    movie_id: str | int   # accept numeric ids from the JSON payload too
    action: str         # "like" | "dislike"


class CompareRequest(BaseModel):
    movies: list        # 2-3 movies, each a movie_id (or resolvable title)


class WhatIfRequest(BaseModel):
    movie: str = ""                  # seed movie title
    n: int = 8
    action: int = 0
    romance: int = 0
    emotion: int = 0
    scifi: int = 0
    comedy: int = 0
    seriousness: int = 0
    phrase: Optional[str] = None     # optional NLP slider phrase


class AIDiscoveryRequest(BaseModel):
    query: str
    n: int = 12


# ------------------------------------------------------------------
# Helper: clean NaN / numpy types before sending JSON
# ------------------------------------------------------------------
def _clean(obj):
    """Convert pandas/numpy values to plain JSON-friendly types."""
    import math
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj
    if hasattr(obj, "item"):  # numpy scalar
        return _clean(obj.item())
    return obj


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/")
def root():
    """
    Serve the frontend HTML directly at the root.
    This way the preview panel (which opens http://localhost:3000/)
    shows the website, not JSON.
    """
    index_file = _FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"app": "CineMind", "status": "running", "note": "frontend not found"}


@app.get("/api")
def api_info():
    """Health check + basic catalogue info (moved to /api so / shows the UI)."""
    return {
        "app": "CineMind",
        "status": "running",
        "movies_indexed": len(engine.movie_metadata),
        "embedding_dim": int(engine.loaded_movie_embeddings.shape[1]),
        "endpoints": [
            "GET  /api               (this health check)",
            "GET  /movies?limit&sort_by",
            "GET  /search?q=TITLE",
            "POST /recommend  {movie, n, personalized}",
            "GET  /movie/{movie_id}",
            "POST /chat       {message}",
            "GET  /watchlist  / POST /watchlist/add  / DELETE /watchlist/remove/{id}",
            "POST /preferences/feedback   +  POST /preferences/reset",
            "GET  /insights",
            "POST /compare  {movies:[...]}",
            "POST /what-if   {movie, action, romance, ...}",
            "POST /ai-discovery {query}",
        ],
    }


@app.get("/movies")
def list_movies(
    limit: int = Query(50, description="Number of movies to return"),
    sort_by: str = Query(
        "popularity",
        description=(
            "Sort order: popularity, rating, title, newest (by release date), "
            "favorites (most votes) or gems (high rating + low popularity)."
        ),
    ),
    genre: Optional[str] = Query(None, description="Genre filter (case-insensitive substring)"),
    lang: Optional[str] = Query(None, description="Keep only movies of this original language code"),
    not_lang: Optional[str] = Query(None, description="Exclude movies of this original language code"),
    min_votes: int = Query(0, description="Minimum Vote_Count required"),
):
    """
    Return a batch of movies from local metadata.

    Powers every home-page card row ("Trending Now", "Top Rated", plus the
    curated rows: New Releases, Hidden Gems, Fan Favorites, Around the World
    and genre rows) and is fully local — no external movie API.
    """
    # cap to avoid huge payloads
    limit = max(1, min(limit, 200))
    df = engine.movie_metadata.copy()

    # optional genre filter (case-insensitive substring, NaN-safe)
    if genre:
        g = str(genre).strip().lower()
        if g:
            df = df[df["Genre"].astype(str).str.lower().str.contains(g, na=False, regex=False)]
            if df.empty:
                return {"results": []}

    # optional language filters
    if lang:
        df = df[df["Original_Language"].astype(str).str.lower() == str(lang).strip().lower()]
        if df.empty:
            return {"results": []}
    if not_lang:
        df = df[df["Original_Language"].astype(str).str.lower() != str(not_lang).strip().lower()]
        if df.empty:
            return {"results": []}

    # optional minimum vote count (quality filter for curated rows)
    if min_votes > 0:
        df = df[df["Vote_Count"].astype(float) >= float(min_votes)]
        if df.empty:
            return {"results": []}

    if sort_by == "rating":
        results = df.sort_values("Vote_Average", ascending=False)
    elif sort_by == "title":
        results = df.sort_values("Title")
    elif sort_by == "newest":
        # Release_Date is stored as YYYY-MM-DD strings; lexicographic sort
        # matches chronological order, with any junk rows sinking to the end.
        results = df.sort_values("Release_Date", ascending=False)
    elif sort_by == "favorites":
        results = df.sort_values("Vote_Count", ascending=False)
    elif sort_by == "gems":
        # Hidden gems = highly rated films people may have missed.
        # score = rating − 0.6 × normalised popularity (popularity drags it down).
        pop = df["Popularity"].astype(float)
        pop_range = (pop.max() - pop.min()) + 1e-9
        norm_pop = (pop - pop.min()) / pop_range
        df = df.copy()
        df["_gem_score"] = df["Vote_Average"].astype(float) - 0.6 * norm_pop
        results = df.sort_values("_gem_score", ascending=False)
        results = results.drop(columns=["_gem_score"])
    else:  # default: popularity
        results = df.sort_values("Popularity", ascending=False)

    results = results.head(limit)
    return {"results": _clean([row.to_dict() for _, row in results.iterrows()])}


@app.get("/search")
def search_movies(q: str = Query(..., description="Movie title to search")):
    """
    Search movies by title using local metadata.
    Returns a list of matching movies (substring match, case-insensitive).
    """
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' is required.")
    results = engine.search_movies(q, limit=10)
    insights_mod.log_search(q, kind="search")
    return {"query": q, "results": _clean(results)}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Get recommendations for a movie title using the local
    Sentence Transformer + FAISS engine.

    Calls engine.get_recommended_movie_details(title, n).
    """
    title = req.movie.strip()
    n = req.n
    if not title:
        raise HTTPException(status_code=400, detail="Field 'movie' is required.")
    if not (1 <= n <= 20):
        raise HTTPException(status_code=400, detail="'n' must be between 1 and 20.")

    if req.personalized:
        selected, recommendations = engine.recommend_personalized(
            title, n, prefs_mod.get_preferences()
        )
    else:
        selected, recommendations = engine.get_recommended_movie_details(title, n)

    if selected is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")

    insights_mod.log_recommendation(title, recommendations, kind="recommend")

    return {
        "selected_movie": _clean(selected),
        "recommendations": _clean(recommendations),
        "personalized": req.personalized,
    }


@app.get("/movie/{movie_id}")
def movie_details(movie_id: str):
    """
    Get full details of a movie by its movie_id (from local metadata).

    The engine coerces both string and numeric ids, so "123" and 123 both
    resolve to the same movie.
    """
    details = engine.get_movie_details(movie_id)
    if details is None:
        raise HTTPException(status_code=404, detail=f"movie_id '{movie_id}' not found.")
    return _clean(details)


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Chatbot endpoint. The chatbot:
      1. Extracts movie title + count from the message.
      2. Calls the LOCAL recommendation engine.
      3. Uses the external AI ONLY to write a friendly reply.
    It NEVER invents movies.
    """
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Field 'message' is required.")

    try:
        result = chatbot_module.chat(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {e}")

    # log the interaction if a movie was actually recommended
    src = result.get("source_movie")
    recs = result.get("recommendations", [])
    if src and recs:
        insights_mod.log_recommendation(
            src.get("Title", "unknown"), recs, kind="recommend"
        )
    else:
        insights_mod.log_search(message, kind="search")

    return _clean(result)


# ------------------------------------------------------------------
# Watchlist
# ------------------------------------------------------------------

@app.get("/watchlist")
def get_watchlist():
    """All watchlist movies, enriched with live metadata."""
    items = watchlist_mod.get_watchlist()
    return {"results": _clean(items), "count": len(items)}


@app.post("/watchlist/add")
def watchlist_add(req: WatchlistAddRequest):
    """Add a movie to the watchlist. Duplicates are ignored (returns added=False)."""
    try:
        entry = watchlist_mod.add_to_watchlist(req.movie_id)
    except LookupError:
        raise HTTPException(status_code=404, detail=f"movie_id '{req.movie_id}' not found.")
    return _clean(entry)


@app.delete("/watchlist/remove/{movie_id}")
def watchlist_remove(movie_id: str):
    """Remove a movie from the watchlist."""
    removed = watchlist_mod.remove_from_watchlist(movie_id)
    return {"removed": removed, "movie_id": movie_id, "in_watchlist": False}


@app.get("/watchlist/check/{movie_id}")
def watchlist_check(movie_id: str):
    """Is this movie already in the watchlist?"""
    return {"movie_id": movie_id, "in_watchlist": watchlist_mod.is_in_watchlist(movie_id)}


# ------------------------------------------------------------------
# Preferences & feedback
# ------------------------------------------------------------------

@app.get("/preferences")
def get_preferences():
    """Current user preferences (likes, dislikes, genres, themes)."""
    return _clean(prefs_mod.get_preferences())


@app.post("/preferences/feedback")
def preferences_feedback(req: FeedbackRequest):
    """Record a like / dislike for a movie."""
    action = req.action.lower()
    if action not in prefs_mod.FEEDBACK_ACTIONS:
        raise HTTPException(status_code=400, detail=f"action must be one of {prefs_mod.FEEDBACK_ACTIONS}")
    try:
        prefs = prefs_mod.add_feedback(req.movie_id, action)
    except LookupError:
        raise HTTPException(status_code=404, detail=f"movie_id '{req.movie_id}' not found.")

    title = "unknown"
    row = engine.get_movie_details(req.movie_id)
    if row:
        title = row.get("Title", "unknown")
    insights_mod.log_event(
        "feedback", action=action, movie_id=str(req.movie_id), title=title
    )
    return _clean({"ok": True, "action": action, "preferences": prefs})


@app.post("/preferences/reset")
def preferences_reset():
    """Clear all preferences and feedback."""
    insights_mod.log_event("preferences", reset=True)
    return _clean({"ok": True, "preferences": prefs_mod.clear_preferences()})


# ------------------------------------------------------------------
# Insights
# ------------------------------------------------------------------

@app.get("/insights")
def insights():
    """Aggregate stats built from the local activity log."""
    return _clean(insights_mod.get_insights())


# ------------------------------------------------------------------
# Compare 2-3 movies
# ------------------------------------------------------------------

@app.post("/compare")
def compare(req: CompareRequest):
    """Compare 2-3 movies side by side (by id or title)."""
    movies = [str(m).strip() for m in req.movies if str(m).strip()]
    if len(movies) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 movies to compare.")
    if len(movies) > 3:
        raise HTTPException(status_code=400, detail="Compare supports at most 3 movies.")

    details = []
    for movie in movies:
        # try id first, then a title match
        idx = engine.find_movie_index_by_id(movie)
        if idx is None:
            idx = engine.find_movie(movie)
        if idx is None:
            raise HTTPException(status_code=404, detail=f"Movie '{movie}' not found.")
        chart = {"query": movie}
        chart.update(engine.movie_metadata.iloc[idx].to_dict())
        details.append(chart)

    # similarity between movies (same embedding space) when >2 are given
    if len(details) == 2:
        sim = _pairwise_similarity(details[0], details[1])
        if sim is not None:
            details[0]["Pair_Similarity"] = sim
            details[1]["Pair_Similarity"] = sim

    return {"results": _clean(details)}


def _pairwise_similarity(a: dict, b: dict):
    """Cosine similarity between two movies using the existing embeddings."""
    ia = engine.find_movie_index_by_id(a.get("movie_id"))
    ib = engine.find_movie_index_by_id(b.get("movie_id"))
    if ia is None or ib is None:
        return None
    import numpy as np
    va = engine.loaded_movie_embeddings[ia]
    vb = engine.loaded_movie_embeddings[ib]
    return round(float(np.dot(va, vb)), 4)


# ------------------------------------------------------------------
# What-If recommendations (sliders + optional NLP phrase)
# ------------------------------------------------------------------

@app.post("/what-if")
def what_if(req: WhatIfRequest):
    """Attribute-weighted re-ranking of the existing FAISS neighbourhood."""
    if not (1 <= req.n <= 20):
        raise HTTPException(status_code=400, detail="'n' must be between 1 and 20.")

    movie_title = req.movie.strip()
    weights = {
        "action": req.action,
        "romance": req.romance,
        "emotion": req.emotion,
        "scifi": req.scifi,
        "comedy": req.comedy,
        "seriousness": req.seriousness,
    }

    if req.phrase:
        parsed = parse_whatif_phrase(req.phrase)
        for dim, delta in parsed["adjustments"].items():
            weights[dim] = max(0, min(100, weights.get(dim, 0) + delta))
        if not movie_title and parsed["seed"]:
            movie_title = parsed["seed"]

    if not movie_title:
        raise HTTPException(
            status_code=400,
            detail="Provide a seed movie (or a phrase like 'like Interstellar but more action').",
        )

    selected, recommendations = engine.what_if_recommendations(movie_title, weights, req.n)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found.")

    insights_mod.log_recommendation(movie_title, recommendations, kind="whatif")
    return {
        "selected_movie": _clean(selected),
        "weights": {k: weights[k] for k in ("action", "romance", "emotion", "scifi", "comedy", "seriousness")},
        "recommendations": _clean(recommendations),
    }


# ------------------------------------------------------------------
# AI Discovery (grounded in the local dataset only)
# ------------------------------------------------------------------

@app.post("/ai-discovery")
def ai_discovery(req: AIDiscoveryRequest):
    """Discover movies from a natural-language query (never invents titles)."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Field 'query' is required.")
    n = max(1, min(req.n, 24))
    results = engine.discover_movies(query, n)
    insights_mod.log_search(query, kind="discovery")
    return {"query": query, "results": _clean(results)}


# ------------------------------------------------------------------
# Run directly:  python -m backend.app
# or:            uvicorn backend.app:app --reload --port 8000
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.app:app", host=host, port=port, reload=True)
