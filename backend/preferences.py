"""
============================================================
CineMind — User preferences & feedback (local JSON)
============================================================
Likes / dislikes are stored in data/user_preferences.json:

    {
        "liked_movies": [...],
        "disliked_movies": [...],
        "preferred_genres": [...],
        "preferred_themes": [...]
    }

preferred_genres and preferred_themes are *derived* from the
liked movies every time feedback is applied (never from an ML
re-train). They are persisted so the personalized ranking can
read them directly on the next request.
============================================================
"""

from .storage import load_json, save_json, now_iso
from . import recommendation_engine as engine

PREFS_FILE = "user_preferences.json"
PREFS_DEFAULT = {
    "liked_movies": [],
    "disliked_movies": [],
    "preferred_genres": [],
    "preferred_themes": [],
}
FEEDBACK_ACTIONS = ("like", "dislike")


def _load() -> dict:
    data = load_json(PREFS_FILE, PREFS_DEFAULT)
    base = dict(PREFS_DEFAULT)
    base.update(data or {})
    for key in ("liked_movies", "disliked_movies", "preferred_genres", "preferred_themes"):
        if not isinstance(base.get(key), list):
            base[key] = []
    return base


def _save(prefs: dict) -> None:
    save_json(PREFS_FILE, prefs)


def _recompute_tags(prefs: dict) -> dict:
    """
    Re-derive preferred genres & themes from the liked movies list.
    Uses the real dataset genres so we never invent categories.
    """
    genre_counts = {}
    theme_counts = {}
    for liked in prefs.get("liked_movies", []):
        row = engine.get_movie_details(liked.get("movie_id"))
        if not row:
            continue
        for genre in engine.genre_list(row.get("Genre")):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        for theme in _themes_of(row.get("Overview")):
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

    prefs["preferred_genres"] = sorted(genre_counts, key=lambda g: genre_counts[g], reverse=True)
    prefs["preferred_themes"] = sorted(theme_counts, key=lambda t: theme_counts[t], reverse=True)
    prefs["favorite_genres"] = [
        {"genre": g, "count": c}
        for g, c in sorted(genre_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return prefs


def _themes_of(overview: str) -> list:
    """Theme labels present in an overview text (see engine.THEME_WORDS)."""
    return engine.themes_in_text(overview or "")


def get_preferences() -> dict:
    """Current preferences with a few derived stats for the UI."""
    prefs = _load()
    stats = {
        "total_likes": len(prefs.get("liked_movies", [])),
        "total_dislikes": len(prefs.get("disliked_movies", [])),
        "preferred_genre_count": len(prefs.get("preferred_genres", [])),
    }
    return {**prefs, "stats": stats}


def add_feedback(movie_id, action: str) -> dict:
    """
    Register a like or dislike for a movie.

    Raises LookupError if the movie is not in the dataset, and
    ValueError for an unknown action. Liking a movie also removes
    it from dislikes (and vice-versa).
    """
    if action not in FEEDBACK_ACTIONS:
        raise ValueError(f"action must be one of {FEEDBACK_ACTIONS}")

    if engine.find_movie_index_by_id(movie_id) is None:
        raise LookupError(f"movie_id '{movie_id}' not found.")

    movie_id = str(movie_id)
    prefs = _load()

    liked = [m for m in prefs.get("liked_movies", []) if str(m.get("movie_id")) != movie_id]
    disliked = [m for m in prefs.get("disliked_movies", []) if str(m.get("movie_id")) != movie_id]

    if action == "like":
        liked.append({"movie_id": movie_id, "added_at": now_iso()})
    else:
        disliked.append({"movie_id": movie_id, "added_at": now_iso()})

    prefs["liked_movies"] = liked
    prefs["disliked_movies"] = disliked
    _recompute_tags(prefs)
    _save(prefs)
    return prefs


def clear_preferences() -> dict:
    """Reset every preference and re-save the default shape."""
    _save(dict(PREFS_DEFAULT))
    return get_preferences()