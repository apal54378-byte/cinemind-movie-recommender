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

import copy
import errno
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# data/ lives next to the project root (one level above backend/).
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_locks = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _locks[key] = lock
        return lock


@contextmanager
def _file_lock(path: Path):
    """Coordinate access to one JSON file across threads and processes."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt
            # msvcrt locks a byte range; ensure byte zero exists first.
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if (getattr(exc, "winerror", None) != 33
                            and exc.errno not in (errno.EACCES, errno.EDEADLK)):
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def ensure_data_dir() -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def load_json(filename: str, default):
    """Read a JSON file, returning `default` if missing or malformed."""
    ensure_data_dir()
    path = _DATA_DIR / filename
    try:
        with _lock_for(path):
            with _file_lock(path):
                return _read_json_unlocked(path, default)
    except (json.JSONDecodeError, OSError, ValueError):
        # Malformed file (e.g. was being written when the app crashed):
        # return the default instead of crashing the request.
        return default


def save_json(filename: str, payload) -> None:
    """Atomically write a JSON file (never leaves a half-written file)."""
    ensure_data_dir()
    path = _DATA_DIR / filename
    with _lock_for(path):
        with _file_lock(path):
            _write_json_unlocked(path, payload)


def update_json(filename: str, default, updater):
    """Atomically read, mutate, and write JSON under a cross-process lock.

    `updater` receives the current JSON value and returns `(new_value, result)`.
    The result is returned to the caller while the updated value is persisted.
    """
    ensure_data_dir()
    path = _DATA_DIR / filename
    with _lock_for(path):
        with _file_lock(path):
            current = _read_json_unlocked(path, default)
            updated, result = updater(current)
            _write_json_unlocked(path, updated)
            return result


def _read_json_unlocked(path: Path, default):
    if not path.exists():
        return copy.deepcopy(default)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError, ValueError):
        return copy.deepcopy(default)


def _write_json_unlocked(path: Path, payload) -> None:
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as fh:
            tmp_name = fh.name
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def now_iso() -> str:
    """ISO-8601 timestamp used for history / watchlist entries."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
