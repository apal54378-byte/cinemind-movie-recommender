"""
============================================================
CineMind — Tiny local JSON storage helpers
============================================================
A single place for reading/writing the local JSON "database"
files (watchlist, preferences, history). No real database is
used on purpose: this is a college ML project and the data
footprint is tiny.

All writes are atomic (temp file + os.replace) and guarded by
a lock so the app never corrupts a file if two requests hit
the same file at the same time.
============================================================
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

# data/ lives next to the project root (one level above backend/).
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_locks = {}


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    lock = _locks.get(key)
    if lock is None:
        lock = threading.Lock()
        _locks[key] = lock
    return lock


def ensure_data_dir() -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def load_json(filename: str, default):
    """Read a JSON file, returning `default` if missing or malformed."""
    ensure_data_dir()
    path = _DATA_DIR / filename
    if not path.exists():
        return default
    try:
        with _lock_for(path):
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except (json.JSONDecodeError, OSError, ValueError):
        # Malformed file (e.g. was being written when the app crashed):
        # return the default instead of crashing the request.
        return default


def save_json(filename: str, payload) -> None:
    """Atomically write a JSON file (never leaves a half-written file)."""
    ensure_data_dir()
    path = _DATA_DIR / filename
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _lock_for(path):
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(path))


def now_iso() -> str:
    """ISO-8601 timestamp used for history / watchlist entries."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")