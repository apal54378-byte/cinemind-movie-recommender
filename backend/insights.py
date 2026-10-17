"""
============================================================
CineMind — Insights (recommendation history + stats)
============================================================
A small activity log lives in data/recommendation_history.json:

    {
      "entries": [
         {"type": "search",      "timestamp": "...", "query": "interstellar"},
         {"type": "recommend",   "timestamp": "...", "selected_movie": "Interstellar",
          "recommendations": ["Arrival", "..."]},
         {"type": "feedback",    "timestamp": "...", "action": "like", "movie_id": "0"},
         {"type": "discovery",   "timestamp": "...", "query": "emotional sci-fi movies"},
         {"type": "whatif",      "timestamp": "...", "selected_movie": "Interstellar"}
      ]
    }

The /insights endpoint derives totals and favourite genres from
these entries + the watchlist + the preferences.
============================================================
"""

from collections import Counter

from .storage import load_json, save_json, update_json, now_iso
from . import recommendation_engine as engine

HISTORY_FILE = "recommendation_history.json"
HISTORY_DEFAULT = {"entries": []}
MAX_ENTRIES = 500


def _load() -> list:
    data = load_json(HISTORY_FILE, HISTORY_DEFAULT)
    entries = data.get("entries", [])
    return entries if isinstance(entries, list) else []


def _save(entries: list) -> None:
    save_json(HISTORY_FILE, {"entries": entries[-MAX_ENTRIES:]})


def log_event(event_type: str, **fields) -> None:
    """Append one activity entry (timestamped). Never raises."""
    try:
        event = {"type": event_type, "timestamp": now_iso(), **fields}

        def append_event(data):
            entries = data.get("entries", []) if isinstance(data, dict) else []
            if not isinstance(entries, list):
                entries = []
            entries.append(event)
            return {"entries": entries[-MAX_ENTRIES:]}, None

        update_json(HISTORY_FILE, HISTORY_DEFAULT, append_event)
    except Exception:
        pass  # history is best-effort; never break a request over it.


def log_search(query: str, kind: str = "search") -> None:
    log_event(kind, query=str(query)[:200])


def log_recommendation(selected_movie: str, recommendations: list, kind: str = "recommend") -> None:
    titles = [str(r.get("Title") or r) for r in recommendations][:20]
    log_event(kind, selected_movie=str(selected_movie)[:200], recommendations=titles)


def _resolve_genre(title_or_query: str):
    """Best-effort: map a searched/recommended title to a real genre list."""
    if not title_or_query:
        return []
    idx = engine.find_movie(title_or_query)
    row = engine.movie_metadata.iloc[idx].to_dict() if idx is not None else None
    if row is None:
        results = engine.search_movies(title_or_query, limit=1)
        if results:
            row = results[0]
    if row is None:
        return []
    return engine.genre_list(row.get("Genre"))


def get_insights() -> dict:
    """Aggregate the activity log + watchlist + preferences into stats."""
    entries = _load()
    from . import watchlist as watchlist_mod
    from . import preferences as prefs_mod

    watchlist = watchlist_mod.get_watchlist()
    prefs = prefs_mod.get_preferences()

    searches = [e for e in entries if e.get("type") == "search"]
    discoveries = [e for e in entries if e.get("type") == "discovery"]
    recommendations = [e for e in entries if e.get("type") == "recommend"]
    whatifs = [e for e in entries if e.get("type") == "whatif"]
    feedbacks = [e for e in entries if e.get("type") == "feedback"]

    # most searched genres: resolve each search query to its movie's genres
    searched_genre_counter = Counter()
    for e in searches + discoveries:
        for genre in _resolve_genre(e.get("query", "")):
            searched_genre_counter[genre] += 1

    # recently recommended: distinct titles, newest first
    recent_titles = []
    for e in reversed(recommendations):
        for title in e.get("recommendations", []):
            if title and title not in recent_titles:
                recent_titles.append(title)
                if len(recent_titles) >= 10:
                    break
        if len(recent_titles) >= 10:
            break

    # favourite genres come straight from the derived preferences block
    favorite_genres = prefs.get("favorite_genres", [])

    recent_activity = [
        {
            "type": e.get("type"),
            "timestamp": e.get("timestamp"),
            "label": e.get("query") or e.get("selected_movie") or e.get("action"),
        }
        for e in reversed(entries[-20:])
    ]

    return {
        "total_searches": len(searches) + len(discoveries),
        "total_recommendations": len(recommendations) + len(whatifs),
        "total_likes": prefs.get("stats", {}).get("total_likes", 0),
        "total_dislikes": prefs.get("stats", {}).get("total_dislikes", 0),
        "total_feedback": len(feedbacks),
        "watchlist_count": len(watchlist),
        "history_count": len(recommendations) + len(whatifs),
        "favorite_genres": favorite_genres,
        "preferred_themes": prefs.get("preferred_themes", []),
        "most_searched_genres": [
            {"genre": g, "count": c}
            for g, c in searched_genre_counter.most_common(8)
        ],
        "recently_recommended": recent_titles,
        "recent_activity": recent_activity,
    }
