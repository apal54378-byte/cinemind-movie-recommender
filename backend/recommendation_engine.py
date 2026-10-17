"""
============================================================
CineMind — Recommendation Engine
============================================================

This module loads the EXISTING saved artifacts ONCE when the
backend starts and exposes the same recommendation logic that
was built in the Jupyter notebook:

    models/movie_embeddings.npy   <- (9828, 384) normalized embeddings
    models/faiss_index.bin          <- faiss.IndexFlatIP index
    models/movie_metadata.csv       <- movie details + Poster_Url

It provides:
    - find_movie(title)
    - get_movie_details(movie_id)
    - get_recommended_movie_details(title, n=10)

Do NOT retrain embeddings or rebuild FAISS per request.
Do NOT replace this with TF-IDF.
This IS the final Sentence Transformer + FAISS model.
============================================================
"""

from pathlib import Path
import re
import math
import threading
import numpy as np
import pandas as pd
import faiss

# ------------------------------------------------------------------
# 1. Robust path handling — locate models/ relative to this file
# ------------------------------------------------------------------
# Project layout:
#   cinemind-python/
#   ├── backend/
#   │   ├── recommendation_engine.py   <- this file
#   │   └── ...
#   └── models/
#       ├── movie_embeddings.npy
#       ├── faiss_index.bin
#       └── movie_metadata.csv
#
# Using __file__ avoids the fragile "../models/..." paths from the
# notebook and works no matter where you run `uvicorn` from.

# backend/ folder -> go up one level -> models/
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

EMBEDDINGS_PATH = _MODELS_DIR / "movie_embeddings.npy"
FAISS_PATH = _MODELS_DIR / "faiss_index.bin"
METADATA_PATH = _MODELS_DIR / "movie_metadata.csv"

# These are the columns the notebook saved into movie_metadata.csv.
METADATA_COLUMNS = [
    "movie_id",
    "Title",
    "Release_Date",
    "Overview",
    "Popularity",
    "Vote_Count",
    "Vote_Average",
    "Original_Language",
    "Genre",
    "Poster_Url",
]


# ------------------------------------------------------------------
# 2. Load artifacts ONCE at import time (module-level singleton)
# ------------------------------------------------------------------
# These globals are created when the module is first imported.
# FastAPI imports this once at startup, so loading happens once.

def _check_artifacts_exist() -> None:
    """Raise a clear error if any artifact file is missing."""
    missing = [p for p in (EMBEDDINGS_PATH, FAISS_PATH, METADATA_PATH) if not p.exists()]
    if missing:
        names = "\n  - ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "CineMind model artifacts not found. Please place these files "
            f"in the models/ folder:\n  - {names}"
        )


_check_artifacts_exist()

print("[CineMind] Loading movie_metadata.csv ...")
movie_metadata: pd.DataFrame = pd.read_csv(METADATA_PATH)

# Some notebooks save movie_id, some don't. If missing, create it from
# the DataFrame index so every row has a stable id (kept aligned with
# embeddings/FAISS row order — see INDEX ALIGNMENT note below).
if "movie_id" not in movie_metadata.columns:
    movie_metadata["movie_id"] = movie_metadata.index

# Keep only the documented metadata columns that actually exist, to
# avoid accidentally returning notebook-internal columns.
_keep = [c for c in METADATA_COLUMNS if c in movie_metadata.columns]
movie_metadata = movie_metadata[_keep].copy()

print(f"[CineMind] Loaded {len(movie_metadata)} movies from metadata.")

print("[CineMind] Loading movie_embeddings.npy ...")
loaded_movie_embeddings: np.ndarray = np.load(EMBEDDINGS_PATH)
# faiss expects float32
if loaded_movie_embeddings.dtype != np.float32:
    loaded_movie_embeddings = loaded_movie_embeddings.astype(np.float32)
print(f"[CineMind] Embeddings shape: {loaded_movie_embeddings.shape}")

print("[CineMind] Loading faiss_index.bin ...")
loaded_faiss_index = faiss.read_index(str(FAISS_PATH))
print(f"[CineMind] FAISS index total vectors: {loaded_faiss_index.ntotal}")

# ------------------------------------------------------------------
# INDEX ALIGNMENT — critical safety check
# ------------------------------------------------------------------
# The FAISS index row order, embeddings row order and metadata row
# order MUST all correspond to the same movies. Verify once at startup.
n_meta = len(movie_metadata)
n_emb = loaded_movie_embeddings.shape[0]
n_faiss = loaded_faiss_index.ntotal
if not (n_meta == n_emb == n_faiss):
    raise RuntimeError(
        "INDEX ALIGNMENT ERROR!\n"
        f"  metadata rows   : {n_meta}\n"
        f"  embeddings rows : {n_emb}\n"
        f"  FAISS vectors   : {n_faiss}\n"
        "These must be equal. Re-export your artifacts from the notebook "
        "without reordering or filtering rows."
    )
print(f"[CineMind] Index alignment OK ({n_meta} movies). Engine ready.\n")


# ------------------------------------------------------------------
# 3. Reusable recommendation functions (notebook logic preserved)
# ------------------------------------------------------------------

def find_movie(title: str):
    """
    Find the DataFrame index of a movie by exact title match
    (case-insensitive, whitespace-trimmed).

    Returns the index (int) or None if not found.
    This preserves the exact logic from the notebook.
    """
    matches = movie_metadata[
        movie_metadata["Title"].str.lower().str.strip()
        == str(title).lower().strip()
    ]
    if matches.empty:
        return None
    return matches.index[0]


def get_movie_details(movie_id):
    """
    Return a movie's details as a dict, looked up by movie_id.
    Returns None if not found.

    Accepts either a numeric or string id (the frontend always sends
    strings, but movie_id may be stored as an int column).

    Source: movie_metadata (local). Never calls any external API.
    """
    # try the value exactly as given first
    movie_row = movie_metadata[movie_metadata["movie_id"] == movie_id]
    if movie_row.empty:
        # coerce numeric-looking ids so "0" and 0 both match int (or float) columns
        for cast in (int, float):
            try:
                movie_row = movie_metadata[
                    movie_metadata["movie_id"] == cast(movie_id)
                ]
            except (ValueError, TypeError):
                movie_row = movie_metadata.iloc[0:0]
            if not movie_row.empty:
                break
    if movie_row.empty:
        return None
    return movie_row.iloc[0].to_dict()


def get_recommended_movie_details(title: str, n: int = 10):
    """
    PRIMARY production recommendation function.

    Steps:
      1. find_movie(title)  -> get the movie's DataFrame index
      2. Take its pre-saved embedding (row = index)
      3. FAISS IndexFlatIP search (inner product == cosine, since
         embeddings are normalized)
      4. Skip the query movie itself + any exact-title duplicates
      5. Return top-n recommendations with Semantic_Similarity score

    Returns:
        (selected_movie_details, recommended_movies)
        selected_movie_details : dict  (the query movie's info)
        recommended_movies     : list[dict]

    If the movie is not found, returns (None, []).
    """
    movie_index = find_movie(title)
    if movie_index is None:
        print(f"Movie '{title}' not found.")
        return None, []

    # The embedding for this movie is at row `movie_index`
    query_vector = np.asarray(
        loaded_movie_embeddings[movie_index], dtype="float32"
    ).reshape(1, -1)

    # Search a few extra to allow skipping the query movie / duplicates
    search_scores, search_indices = loaded_faiss_index.search(
        query_vector, n + 10
    )

    recommended_movies = []
    normalized_title = str(title).lower().strip()

    selected_movie_details = movie_metadata.iloc[movie_index].to_dict()

    for score, movie_idx in zip(search_scores[0], search_indices[0]):
        # safety: faiss may return -1 if fewer results
        if movie_idx < 0 or movie_idx >= len(movie_metadata):
            continue

        current_title = str(movie_metadata.iloc[movie_idx]["Title"]).lower().strip()

        # skip the query movie itself
        if movie_idx == movie_index:
            continue
        # skip exact-title duplicates
        if current_title == normalized_title:
            continue

        movie_details = movie_metadata.iloc[movie_idx].to_dict()
        movie_details["Semantic_Similarity"] = round(float(score), 4)
        movie_details["Reasons"] = build_reasons(selected_movie_details, movie_details)
        recommended_movies.append(movie_details)

        if len(recommended_movies) == n:
            break

    return selected_movie_details, recommended_movies


# ------------------------------------------------------------------
# 4. Small helper used by the search endpoint
# ------------------------------------------------------------------

def search_movies(query: str, limit: int = 10):
    """
    Simple title substring search over local metadata.
    Case-insensitive. Returns a list of dicts.
    """
    q = str(query).lower().strip()
    if not q:
        return []
    mask = movie_metadata["Title"].str.lower().str.contains(q, na=False, regex=False)
    results = movie_metadata[mask].head(limit)
    # convert to plain list[dict] for clean JSON
    return [row.to_dict() for _, row in results.iterrows()]


# =====================================================================
# 5.  New features built ON TOP of the existing engine
#     ─ all of these rely on the same FAISS index + embeddings.
#     ─ No new model is created. No existing function is removed.
# =====================================================================

# ---- Theme / keyword definitions (deterministic — never from LLM) ----

GENRE_SEP_RE = re.compile(r"\s*[,/|]\s*")


def genre_list(value) -> list:
    """Comma/slash/pipe-separated genre string → list[str], NaN-safe."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(g).strip() for g in value if str(g).strip()]
    return [g.strip() for g in GENRE_SEP_RE.split(str(value)) if g.strip()]


THEME_WORDS: dict = {
    "action":    {"action", "fight", "battle", "war", "chase", "hero",
                  "mission", "combat", "shoot", "agent", "crime", "thriller"},
    "romance":   {"romance", "romantic", "love", "lovers", "affair",
                  "wedding", "passion", "date", "relationship", "heart"},
    "emotion":   {"emotional", "feelings", "grief", "loss", "heartbreak",
                  "tear", "family", "bond", "drama", "hope", "healing"},
    "scifi":     {"sci-fi", "scifi", "science fiction", "space", "alien",
                  "future", "time travel", "robot", "technology", "mars",
                  "galaxy", "dimension", "cyberpunk", "dystopia", "ai"},
    "comedy":    {"comedy", "funny", "humor", "laugh", "hilarious",
                  "parody", "joke", "quirky", "slapstick"},
    "seriousness": {"serious", "dark", "grim", "gritty", "political",
                    "biography", "history", "documentary", "mystery", "detective"},
}
ALL_THEME_WORDS = frozenset(
    {w for words in THEME_WORDS.values() for w in words}
)


def themes_in_text(text: str) -> list:
    t = (str(text) or "").lower()
    return [theme for theme, words in THEME_WORDS.items()
            if any(w in t for w in words)]


def find_movie_index_by_id(movie_id):
    """
    Resolve a movie_id to its DataFrame index.
    Accepts str, int, or float. Returns None if not found.
    """
    if movie_id is None:
        return None
    target = str(movie_id).strip()
    mask = movie_metadata["movie_id"].astype(str) == target
    if mask.any():
        return movie_metadata[mask].index[0]
    for cast in (int, float):
        try:
            if movie_metadata["movie_id"].eq(cast(movie_id)).any():
                return movie_metadata[movie_metadata["movie_id"] == cast(movie_id)].index[0]
        except (ValueError, TypeError):
            pass
    return None


# ---- Precomputed lowered text columns (built once at import) ----

_title_lower = movie_metadata["Title"].astype(str).str.lower().str.strip()
_overview_lower = movie_metadata["Overview"].astype(str).str.lower()
_genre_lower = movie_metadata["Genre"].astype(str).str.lower()
_popularity = movie_metadata["Popularity"].astype(float)
_rating = movie_metadata["Vote_Average"].astype(float)


# ------------------------------------------------------------------
# Why Recommended — deterministic explanation builder
# ------------------------------------------------------------------

def build_reasons(query_movie: dict, candidate_movie: dict,
                  user_preferences=None) -> dict:
    """
    Return {factors: {...}, reasons: [...]} for one recommendation
    without calling any LLM.  User preferences are optional.
    """
    q_genres = set(genre_list(query_movie.get("Genre")))
    c_genres = set(genre_list(candidate_movie.get("Genre")))
    genre_overlap = sorted(q_genres & c_genres)

    q_overview = str(query_movie.get("Overview") or "").lower()
    c_overview = str(candidate_movie.get("Overview") or "").lower()
    shared_words = sorted(w for w in ALL_THEME_WORDS if w in q_overview and w in c_overview)

    q_rating = _safe_float(query_movie.get("Vote_Average"))
    c_rating = _safe_float(candidate_movie.get("Vote_Average"))
    rating_close = (q_rating is not None and c_rating is not None
                    and abs(q_rating - c_rating) <= 1.0)

    sim = _safe_float(candidate_movie.get("Semantic_Similarity"))
    strong_semantic = sim is not None and sim >= 0.50

    pref_match = False
    if user_preferences:
        p_genres = set(user_preferences.get("preferred_genres", []))
        if p_genres and bool(c_genres & p_genres):
            pref_match = True

    reasons = []
    factors = {
        "semantic":  round(sim, 4) if sim is not None else None,
        "genre_overlap": genre_overlap,
        "theme_overlap": shared_words,
        "rating_close": rating_close,
        "preference_match": pref_match,
    }

    if strong_semantic:
        reasons.append(
            f"Semantically similar (cosine score {sim:.2f})."
        )
    elif sim is not None and sim > 0:
        reasons.append(
            f"Modest semantic similarity ({sim:.2f}) — a broader find."
        )
    if genre_overlap:
        reasons.append(f"Shares genres: {', '.join(genre_overlap)}.")
    if shared_words:
        reasons.append(f"Overlapping themes: {', '.join(shared_words[:3])}.")
    if rating_close:
        reasons.append(f"Similar audience ratings ({q_rating:.1f} vs {c_rating:.1f}).")
    if pref_match:
        reasons.append("Matches genres you've liked before.")
    if not reasons:
        reasons.append("Emerges from the engine's semantic neighbourhood.")
    return {"factors": factors, "reasons": reasons}


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return f if not math.isnan(f) else None
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------
# More Like This — by movie_id (not just title string)
# ------------------------------------------------------------------

def more_like_this(movie_id, n: int = 8):
    """
    Same FAISS path as get_recommended_movie_details, but keyed on
    movie_id rather than a title string.  Returns (selected, recs).
    """
    movie_index = find_movie_index_by_id(movie_id)
    if movie_index is None:
        return None, []

    query_vector = np.asarray(
        loaded_movie_embeddings[movie_index], dtype="float32"
    ).reshape(1, -1)

    scores, indices = loaded_faiss_index.search(query_vector, n + 12)
    selected_movie = movie_metadata.iloc[movie_index].to_dict()
    normalized_title = str(selected_movie["Title"]).lower().strip()

    recs = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(movie_metadata):
            continue
        if idx == movie_index:
            continue
        current_title = str(movie_metadata.iloc[idx]["Title"]).lower().strip()
        if current_title == normalized_title:
            continue
        movie_details = movie_metadata.iloc[idx].to_dict()
        movie_details["Semantic_Similarity"] = round(float(score), 4)
        movie_details["Reasons"] = build_reasons(selected_movie, movie_details)
        recs.append(movie_details)
        if len(recs) >= n:
            break
    return selected_movie, recs


# ------------------------------------------------------------------
# Personalized ranking
# ------------------------------------------------------------------

def recommend_personalized(title: str, n: int = 10, user_preferences: dict = None):
    """
    Like get_recommended_movie_details but the final ranking blends
    semantic similarity with genre/preference affinity.

    final_score = 0.5 × semantic
               + 0.3 × genre_match
               + 0.2 × preference_match

    Semantic_Similarity is preserved alongside Final_Score.
    """
    movie_index = find_movie(title)
    if movie_index is None:
        return None, []

    query_vector = np.asarray(
        loaded_movie_embeddings[movie_index], dtype="float32"
    ).reshape(1, -1)

    scores, indices = loaded_faiss_index.search(query_vector, n + 14)
    selected_movie = movie_metadata.iloc[movie_index].to_dict()
    query_genres = set(genre_list(selected_movie.get("Genre")))
    pref_genres = set((user_preferences or {}).get("preferred_genres", []))
    normalized_title = str(selected_movie["Title"]).lower().strip()

    recs = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(movie_metadata):
            continue
        if idx == movie_index:
            continue
        current_title = str(movie_metadata.iloc[idx]["Title"]).lower().strip()
        if current_title == normalized_title:
            continue
        movie_details = movie_metadata.iloc[idx].to_dict()
        sem = round(float(score), 4)

        cand_genres = set(genre_list(movie_details.get("Genre")))
        genre_match = _jaccard(query_genres, cand_genres) if query_genres and cand_genres else 0.0
        pref_match  = _jaccard(cand_genres, pref_genres) if pref_genres and cand_genres else 0.0

        final = 0.5 * sem + 0.3 * genre_match + 0.2 * pref_match

        movie_details["Semantic_Similarity"] = sem
        movie_details["Genre_Match"]         = round(genre_match, 4)
        movie_details["Preference_Match"]    = round(pref_match, 4)
        movie_details["Final_Score"]         = round(final, 4)
        movie_details["Reasons"]             = build_reasons(
            selected_movie, movie_details, user_preferences
        )
        recs.append(movie_details)

    recs.sort(key=lambda r: r["Final_Score"], reverse=True)
    return selected_movie, recs[:n]


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ------------------------------------------------------------------
# What-If recommendations — attribute-weighted re-ranking
# ------------------------------------------------------------------

DIMENSION_MAP: dict = {
    "action":    "action",
    "romance":   "romance",
    "emotion":   "emotion",
    "scifi":     "scifi",
    "comedy":    "comedy",
    "seriousness": "seriousness",
}


def what_if_recommendations(title: str, weights: dict, n: int = 8):
    """
    Seed movie → FAISS neighbours → re-ranked by the weighted
    attribute score.  Weights are 0-100 for each dimension.
    """
    movie_index = find_movie(title)
    if movie_index is None:
        return None, []

    query_vector = np.asarray(
        loaded_movie_embeddings[movie_index], dtype="float32"
    ).reshape(1, -1)

    scores, indices = loaded_faiss_index.search(query_vector, max(n * 6, 30))
    selected_movie = movie_metadata.iloc[movie_index].to_dict()
    normalized_title = str(selected_movie["Title"]).lower().strip()

    normalised_weights = {k: max(0.0, min(100.0, weights.get(k, 0))) / 100.0
                          for k in DIMENSION_MAP}
    total_w = sum(normalised_weights.values())

    recs = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(movie_metadata):
            continue
        if idx == movie_index:
            continue
        current_title = str(movie_metadata.iloc[idx]["Title"]).lower().strip()
        if current_title == normalized_title:
            continue

        cand = movie_metadata.iloc[idx].to_dict()
        sem = float(score)
        attr_score = _attribute_score(cand, normalised_weights, total_w)
        final = 0.5 * sem + 0.5 * attr_score

        cand["Semantic_Similarity"] = round(sem, 4)
        cand["Attribute_Score"]     = round(attr_score, 4)
        cand["Final_Score"]         = round(final, 4)
        cand["Reasons"]             = build_reasons(selected_movie, cand)
        recs.append(cand)

    recs.sort(key=lambda r: r["Final_Score"], reverse=True)
    return selected_movie, recs[:n]


def _attribute_score(cand: dict, w: dict, total_w: float) -> float:
    """Simple keyword-hit ratio for each dimension, averaged by weights."""
    if total_w <= 0:
        return 0.0
    text = (
        " "
        .join([
            str(cand.get("Overview") or ""),
            str(cand.get("Genre") or ""),
        ])
    ).lower()
    score = 0.0
    for dim_key, dim_label in DIMENSION_MAP.items():
        dw = w.get(dim_key, 0.0)
        if dw <= 0:
            continue
        words = THEME_WORDS.get(dim_label, set())
        hits = sum(1 for wd in words if wd in text)
        score += dw * (hits / max(len(words), 1))
    return score / total_w


# ------------------------------------------------------------------
# AI Discovery — semantic + keyword grounding
#     When sentence-transformers IS available the query is embedded
#     with all-MiniLM-L6-v2 and searched via FAISS.  When it is
#     NOT available (the current runtime) a deterministic keyword /
#     theme + popularity ranking is used.
#     In every case only real movies from the local dataset appear.
# ------------------------------------------------------------------

_sent_model = None
_sent_available = None  # None = not attempted; False = failed; True = loaded
_sent_model_lock = threading.Lock()

def _load_sentence_model():
    """Lazy-load all-MiniLM-L6-v2 (only used if installed)."""
    global _sent_model, _sent_available
    if _sent_available and _sent_model is not None:
        return True
    if _sent_available is False:
        return False          # already failed once — don't retry
    # Model downloads/initialization are expensive. Serialize the first
    # attempt so concurrent discovery requests share one result.
    with _sent_model_lock:
        if _sent_available is True and _sent_model is not None:
            return True
        if _sent_available is False:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            _sent_model = None
            _sent_available = False
            print(f"[CineMind] WARNING: Sentence Transformer unavailable; AI Discovery will use keyword fallback ({type(exc).__name__}).")
            return False
        _sent_model = model
        _sent_available = True
        return True


_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "for", "and", "with", "like", "movies",
    "movie", "films", "film", "highly", "rated", "popular", "best", "give",
    "me", "suggest", "find", "some", "more", "that", "over", "night",
    "tonight", "i", "want", "need", "look", "show", "tell", "me",
})


def _keywords_from(text: str):
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]


def _normalised_population(idx: int) -> float:
    """Rank the popularity value relative to the dataset range."""
    if len(_popularity) == 0:
        return 0.0
    import numpy as np  # noqa — already imported at top
    min_p, max_p = float(np.nanmin(_popularity)), float(np.nanmax(_popularity))
    if max_p <= min_p:
        return 0.5
    return float((_popularity.iloc[idx] - min_p) / (max_p - min_p))


def _rating_norm(idx: int) -> float:
    if len(_rating) == 0:
        return 0.0
    import numpy as np
    return float(np.nanmax(_rating) - np.nanmin(_rating)) or 1.0


def discover_movies(query: str, limit: int = 12) -> list:
    """
    Grounded movie discovery that never invents titles.

    - Embeds the query with all-MiniLM-L6-v2 if available.
    - Falls back to keyword/theme scoring otherwise.
    - Optionally boosts rating / popularity when the query implies it.
    """
    q = query.strip()
    if not q:
        return []

    keywords = _keywords_from(q)
    wants_rating  = any(k in keywords for k in ("rated", "rating", "best", "great", "top"))
    wants_popular = any(k in keywords for k in ("popular", "trending", "famous", "hit"))

    # --- strategy 1: real semantic search via all-MiniLM-L6-v2 ---
    if _load_sentence_model():
        q_emb = _sent_model.encode([q], normalize_embeddings=True).astype("float32")
        scores, indices = loaded_faiss_index.search(q_emb, limit * 4)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(movie_metadata):
                continue
            sem = float(score)
            popularity_boost = 0.15 * _normalised_population(idx) if wants_popular else 0
            rating_boost = 0.15 * (_rating.iloc[idx] / 10.0 if not math.isnan(_rating.iloc[idx]) else 0) if wants_rating else 0
            final = sem + popularity_boost + rating_boost
            row = movie_metadata.iloc[idx].to_dict()
            row["Relevance"] = round(final, 4)
            row["Semantic_Score"] = round(sem, 4)
            results.append(row)
        results.sort(key=lambda r: r["Relevance"], reverse=True)
        return results[:limit]

    # --- strategy 2 (fallback): keyword / theme + popularity ranking ---
    q_themes = themes_in_text(q)
    q_genre_words = [kw for kw in keywords if len(kw) > 3]  # short names like "sci" skip

    # score = kw_hits + theme_hits*2 + genre_hint*3 + popular_boost*1.5
    scored = []
    for i, row_s in _title_lower.items():
        overview_text = _overview_lower.iloc[i] + " " + _genre_lower.iloc[i] + " " + _title_lower.iloc[i]
        kw_hits = sum(1 for kw in keywords if kw in overview_text)
        theme_hits = sum(1 for th in q_themes if th in overview_text)
        genre_hint_hits = sum(1 for g in genre_list(movie_metadata["Genre"].iloc[i])
                              if g.lower() in q.lower() or q.lower() in g.lower())
        if kw_hits == 0 and theme_hits == 0 and genre_hint_hits == 0:
            continue
        raw = kw_hits + theme_hits * 2.0 + genre_hint_hits * 3.0
        if wants_popular:
            raw += _normalised_population(i) * 1.5
        if wants_rating:
            raw += (_rating.iloc[i] / 10.0) * 1.2
        scored.append((raw, i))
    if not scored:
        return []
    scored.sort(key=lambda t: t[0], reverse=True)
    top_raw = scored[0][0]
    return [
        {
            **movie_metadata.iloc[i].to_dict(),
            "Relevance": round(raw / top_raw, 4) if top_raw > 0 else 0,
        }
        for raw, i in scored[:limit]
    ]
