"""
============================================================
CineMind — Watchlist (local JSON storage)
============================================================
Stores {movie_id, title, added_at} entries in
data/watchlist.json. The full movie details (poster, genres…)
are joined from the real dataset when the watchlist is read,
so the file itself stays small and the source of truth for
movie data is always models/movie_metadata.csv.
============================================================
"""

from .storage import load_json, save_json, update_json, now_iso
from . import recommendation_engine as engine

WATCHLIST_FILE = "watchlist.json"
WATCHLIST_DEFAULT = {"watchlist": []}


def _load() -> list:
    data = load_json(WATCHLIST_FILE, WATCHLIST_DEFAULT)
    entries = data.get("watchlist", [])
    return entries if isinstance(entries, list) else []


def _save(entries: list) -> None:
    save_json(WATCHLIST_FILE, {"watchlist": entries})


def is_in_watchlist(movie_id) -> bool:
    movie_id = str(movie_id)
    return any(str(e.get("movie_id")) == movie_id for e in _load())


def add_to_watchlist(movie_id) -> dict:
    """
    Add a movie (by id or title-ish id) to the watchlist.

    Returns the saved entry with `added=True` if it was inserted,
    or the existing one with `added=False` on a duplicate.
    Raises LookupError if the movie is not in the dataset.
    """
    movie_index = engine.find_movie_index_by_id(movie_id)
    if movie_index is None:
        raise LookupError(f"movie_id '{movie_id}' not found.")

    title = str(engine.movie_metadata.iloc[movie_index]["Title"])
    movie_id_key = str(engine.movie_metadata.iloc[movie_index]["movie_id"])

    def add(data):
        entries = data.get("watchlist", []) if isinstance(data, dict) else []
        if not isinstance(entries, list):
            entries = []
        for existing in entries:
            if str(existing.get("movie_id")) == movie_id_key:
                return {"watchlist": entries}, {**existing, "added": False}
        entry = {"movie_id": movie_id_key, "title": title, "added_at": now_iso()}
        entries.append(entry)
        return {"watchlist": entries}, {**entry, "added": True}

    return update_json(WATCHLIST_FILE, WATCHLIST_DEFAULT, add)


def remove_from_watchlist(movie_id) -> bool:
    """Remove a movie. Returns True if something was removed."""
    movie_id = str(movie_id)
    def remove(data):
        entries = data.get("watchlist", []) if isinstance(data, dict) else []
        if not isinstance(entries, list):
            entries = []
        remaining = [e for e in entries if str(e.get("movie_id")) != movie_id]
        return {"watchlist": remaining}, len(remaining) != len(entries)

    return update_json(WATCHLIST_FILE, WATCHLIST_DEFAULT, remove)


def get_watchlist() -> list:
    """Full watchlist, enriched with live movie details from metadata."""
    entries = _load()
    results = []
    for entry in entries:
        row = engine.get_movie_details(entry.get("movie_id"))
        item = dict(entry)
        if row:
            item.update(row)
        results.append(item)
    return results


def watchlist_count() -> int:
    return len(_load())
