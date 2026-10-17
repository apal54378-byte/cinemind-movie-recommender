# CineMind

A **semantic movie recommendation platform** built on Sentence Transformer embeddings and FAISS similarity search, wrapped in a FastAPI backend with a lightweight, dependency-free HTML/CSS/JS frontend.

CineMind recommends films based on the *meaningful content* of a movie's plot description — not just genre keywords. When you search for a movie, the system retrieves conceptually similar titles from a local catalogue of **9,828 movies** using **384-dimensional** vector embeddings.

> **Design principle — local-first.** Every search result, recommendation, movie detail, poster, and home-page row comes from local artifacts in `models/`. The only outbound AI call is the optional chatbot, which merely phrases a natural-language reply around results the local engine already produced. The chatbot is never a source of movie data.

---

## Table of Contents

- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Usage Guide](#usage-guide)
- [Backend Modules](#backend-modules)
- [Frontend](#frontend)
- [Design Principles & Constraints](#design-principles--constraints)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [License](#license)

---

## Key Features

| Feature | Description |
|---------|-------------|
| Semantic search & recommendations | FAISS nearest-neighbour retrieval with cosine similarity scores |
| Curated home page | Hero carousel, Trending Now, Top Rated, New Releases, Hidden Gems, Around the World, Fan Favorites, and daily-rotating genre rows |
| Personalized "Made for You" row | Ranks movies from genres you have liked |
| Movie detail modal | Full local metadata with Like / Dislike, Watchlist, Compare, More Like This, and deterministic "why recommended" explanations |
| Compare tool | Side-by-side comparison of 2–3 movies with pairwise cosine similarity |
| Recommendation Lab | Six tools in one workspace: AI Discovery, What-If, Watchlist, Preferences, Insights, History |
| What-If re-ranking | Adjust attribute sliders (action, romance, emotion, sci-fi, comedy, seriousness) or type a phrase such as *"like Inception but more action"* |
| Chatbot | Extracts intent from plain language, queries the local engine, and optionally phrases the reply via the Groq API |
| Local persistence | Watchlists, preferences, and activity history stored safely as JSON in `data/` |

---

## How It Works

```
Movie overviews (text)
        │
        ▼
Sentence Transformer  (all-MiniLM-L6-v2, 384 dims, normalized)
        │
        ▼
movie_embeddings.npy  ──►  faiss_index.bin (IndexFlatIP)
        │
        ▼
FAISS similarity search  (inner product == cosine similarity on normalized vectors)
        │
        ▼
Recommendations with Semantic_Similarity score + deterministic reasons
```

The Jupyter notebook (`notebook/CineMind.ipynb`) documents the full modelling journey: a TF-IDF baseline, a genre-weighted TF-IDF variant, a hybrid ranking approach, and the final Sentence Transformer + FAISS model. The notebook also exports the three runtime artifacts below.

### The three model artifacts

| Artifact | Description |
|----------|-------------|
| `models/movie_embeddings.npy` | `(9828, 384)` float32 array; row *i* is the embedding for the movie at row *i* of the metadata CSV |
| `models/faiss_index.bin` | Serialized `faiss.IndexFlatIP` index built from the same normalized embeddings |
| `models/movie_metadata.csv` | Movie catalogue with columns `movie_id, Title, Release_Date, Overview, Popularity, Vote_Count, Vote_Average, Original_Language, Genre, Poster_Url` |

All three artifacts share the same row order. On startup the engine verifies alignment (`metadata rows == embeddings rows == FAISS vectors`) and refuses to start on mismatch.

### The core recommendation lookup

1. `find_movie("Interstellar")` — exact, case-insensitive title match → DataFrame index.
2. Use that movie's pre-saved embedding row as the query vector.
3. `faiss_index.search(query, n + 10)` — nearest neighbours by inner product.
4. Skip the query movie itself and any exact-title duplicates.
5. Return the top **n** with a `Semantic_Similarity` score (≈0–1) and a deterministic `Reasons` block.

---

## Architecture

```
 Browser
   │   fetch()  (same-origin; else falls back to http://127.0.0.1:8000)
   ▼
┌──────────────────────────── FastAPI (uvicorn, port 8000) ───────────────────────────┐
│  GET  /                     serves the frontend (index.html)                        │
│  /static  /css  /js         static mounts → frontend/                               │
│  GET  /api                  health check + catalogue stats                          │
│  GET  /movies               catalogue batches (sort, genre, language filters)       │
│  GET  /search?q=            local title search                                      │
│  POST /recommend            FAISS recommendations                                   │
│  GET  /movie/{id}           full local details                                      │
│  POST /chat                 chatbot orchestration                                   │
│  …  /watchlist, /preferences, /insights, /compare, /what-if, /ai-discovery          │
│                                                                                     │
│  backend/recommendation_engine.py   loads .npy / .bin / .csv once, FAISS logic      │
│  backend/chatbot.py                 intent extraction, AI reply orchestration       │
│  backend/storage.py                 atomic JSON read/write helpers (→ data/)        │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 models/  (movie_embeddings.npy, faiss_index.bin, movie_metadata.csv)
```

### Request flow (happy path)

1. The user searches **"Interstellar"** → frontend calls `GET /search?q=Interstellar`.
2. The backend runs a local, case-insensitive title search.
3. If exactly one result matches, the frontend calls `POST /recommend` with that title.
4. The engine executes the FAISS lookup and returns the selected movie plus recommendations, each with a similarity score and a deterministic reasons block.
5. The frontend renders a hero for the selected movie and its similar-movie cards.
6. Clicking a card opens the detail modal via `GET /movie/{id}` with Watchlist, Like/Dislike, Compare, and More Like This actions.

For the full internal walkthrough — sequence diagrams, startup/fallback paths and request tracing — see [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.9+, FastAPI, Uvicorn |
| ML / retrieval | Sentence Transformers (`all-MiniLM-L6-v2`), FAISS, NumPy, pandas |
| Frontend | Plain HTML, CSS, JavaScript (no build step, no bundler) |
| Chatbot | Groq API via the official `groq` Python SDK (`openai/gpt-oss-120b`) |
| Storage | Atomic JSON files under `data/` |

> **Note:** The core recommendation engine loads pre-computed embeddings from disk and does not need `sentence-transformers`. AI Discovery uses `sentence-transformers` to embed query text when available; if the package or model cannot load, it falls back to the deterministic keyword/theme ranker. The dependency is included in `requirements.txt` for the semantic path.

---

## Project Structure

```
CineMind/
├── backend/
│   ├── app.py                    # FastAPI application, endpoints, CORS, static serving
│   ├── recommendation_engine.py  # Loads artifacts once; FAISS search & ranking logic
│   ├── chatbot.py                # Intent extraction, Groq reply orchestration, NLP parser
│   ├── llm.py                    # Centralized Groq client, model config and error mapping
│   ├── watchlist.py              # Watchlist persistence
│   ├── preferences.py            # Like/dislike → derived genres & themes
│   ├── insights.py               # Activity log and aggregated statistics
│   ├── storage.py                # Atomic JSON read/write helpers
│   └── __init__.py
├── frontend/
│   ├── index.html                # Application shell
│   ├── css/
│   │   ├── style.css             # Main theme
│   │   └── responsive.css        # Mobile / tablet breakpoints
│   └── js/
│       ├── api.js                # Central fetch wrapper (single API access point)
│       ├── app.js                # UI logic: search, rows, modal, compare, lab, chat
│       ├── utils.js              # Formatters and safe HTML escaping
│       └── notifications.js      # Toast notifications
├── models/                       # ML artifacts (gitignored; place manually)
│   ├── movie_embeddings.npy
│   ├── faiss_index.bin
│   └── movie_metadata.csv
├── data/                         # Runtime user data (gitignored; auto-created)
│   ├── watchlist.json
│   ├── user_preferences.json
│   └── recommendation_history.json
├── notebook/
│   └── CineMind.ipynb            # Training + artifact export pipeline
├── tests/
│   ├── test_chatbot_groq.py      # Groq provider, client lifecycle, chatbot fallbacks
│   └── test_targeted_fixes.py    # Sentence-model loading, discovery paths, storage locking
├── .env.example                  # Environment variable template
├── .env                          # Local configuration (gitignored)
├── requirements.txt
├── README.md                     # This file
└── PROJECT_STRUCTURE.md          # Internal architecture & workflow documentation
```

---

## Getting Started

### Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- The three model artifacts in `models/` (see [The three model artifacts](#the-three-model-artifacts))

> **Important:** `.gitignore` excludes `models/`, `data/`, and `.env`, so these are never committed. Copy the artifacts into `models/` locally after cloning.

### 1. Place the model artifacts

Copy the three exported files from the notebook into `models/`:

```
models/movie_embeddings.npy
models/faiss_index.bin
models/movie_metadata.csv
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)

```bash
cp .env.example .env
```

The backend auto-loads `.env` on startup via `python-dotenv`; `.env` is ignored by Git and should stay local. In PowerShell:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `GROQ_API_KEY` there (never commit or share the file). Alternatively, set the key only for the current PowerShell session:

```powershell
$env:GROQ_API_KEY = "gsk_your_key_here"
python -m backend.app
```

### 5. Start the server

```bash
uvicorn backend.app:app --reload --port 8000
```

or, to honour `HOST` / `PORT` from `.env`:

```bash
python -m backend.app
```

A healthy startup logs an index-alignment check:

```
[CineMind] Loading movie_metadata.csv ...
[CineMind] Loaded 9828 movies from metadata.
[CineMind] Embeddings shape: (9828, 384)
[CineMind] FAISS index total vectors: 9828
[CineMind] Index alignment OK (9828 movies). Engine ready.

INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 6. Open the application

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. FastAPI serves the entire frontend at the root; the static assets are additionally mounted under `/static`, `/css`, and `/js` for IDE live-preview support.

---

## Configuration

Environment variables (`.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(empty — fallback replies)* | Groq API key for the chatbot |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq-hosted chat model id |
| `GROQ_BASE_URL` | *(Groq SDK default)* | Optional API base URL override (proxy / gateway) |
| `GROQ_TIMEOUT` | `30` | Per-request timeout in seconds |
| `HOST` | `127.0.0.1` | Bind host (used with `python -m backend.app`) |
| `PORT` | `8000` | Bind port |

The chatbot uses the **official Groq Python SDK** (`groq`). No other provider is used: if Groq is not configured or unreachable, the chatbot returns a deterministic local reply built from the local engine's results and never silently switches providers.

The default model, `openai/gpt-oss-120b`, is a current **Groq production model** (see the [Groq models list](https://console.groq.com/docs/models)). Override it with `GROQ_MODEL`, for example `openai/gpt-oss-20b` or `llama-3.1-8b-instant`.

To enable Groq, set `GROQ_API_KEY` in the local, Git-ignored `.env` file and restart the backend. The server logs a warning when the key is missing or empty; it **never** prints the key value. Authentication errors, rate limits (HTTP 429), timeouts, connection failures and invalid model ids are logged with a clear diagnosis, and the chatbot falls back to its local reply.

**Backward compatibility:** the previous `AI_API_KEY` and `AI_MODEL` variables are still read as fallbacks when the corresponding `GROQ_*` variable is unset. They are deprecated — migrate to `GROQ_API_KEY` / `GROQ_MODEL`. The old `AI_BASE_URL` variable is no longer used.

### Port already in use (Windows)

If the port is occupied, identify the listener and inspect its process before deciding whether to stop it:

```powershell
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
$listener | Select-Object LocalAddress, LocalPort, OwningProcess
Get-Process -Id $listener.OwningProcess
```

If you have confirmed it is safe to stop, do so deliberately:

```powershell
Stop-Process -Id $listener.OwningProcess
```

Or leave it running and start CineMind on another available port:

```powershell
uvicorn backend.app:app --reload --port 8021
```

Then open `http://127.0.0.1:8021`; the frontend and API are served by that same backend. When starting with `python -m backend.app`, set `PORT` in the current PowerShell session first (for example `$env:PORT = '8021'`).

---

## API Reference

| Method | Endpoint | Query / Body | Description |
|--------|----------|--------------|-------------|
| GET | `/` | — | Serves the frontend UI |
| GET | `/api` | — | Health check + movie count + embedding dimension |
| GET | `/movies` | `limit`, `sort_by`, `genre`, `lang`, `not_lang`, `min_votes` | Catalogue batches for home rows |
| GET | `/search` | `q=title` | Local, case-insensitive title search |
| POST | `/recommend` | `{"movie","n","personalized"}` | FAISS recommendations (`n ≤ 20`) |
| GET | `/movie/{movie_id}` | — | Full local movie details |
| POST | `/chat` | `{"message"}` | Chatbot (local engine + optional AI reply) |
| GET | `/watchlist` | — | Watchlist entries enriched with live metadata |
| POST | `/watchlist/add` | `{"movie_id"}` | Add to watchlist (duplicates ignored) |
| DELETE | `/watchlist/remove/{movie_id}` | — | Remove from watchlist |
| GET | `/watchlist/check/{movie_id}` | — | Watchlist membership check |
| GET | `/preferences` | — | Likes, dislikes, derived genres and themes |
| POST | `/preferences/feedback` | `{"movie_id","action":"like\|dislike"}` | Record feedback |
| POST | `/preferences/reset` | — | Clear all preferences |
| GET | `/insights` | — | Aggregated activity statistics |
| POST | `/compare` | `{"movies":["id1","id2","id3"]}` | Side-by-side comparison (2–3 movies, pairwise similarity for pairs) |
| POST | `/what-if` | `{"movie","n","action","romance","emotion","scifi","comedy","seriousness","phrase"}` | Attribute-weighted re-ranking |
| POST | `/ai-discovery` | `{"query","n"}` | Grounded natural-language discovery (`n ≤ 24`) |

**`sort_by` values for `/movies`:** `popularity` (default), `rating`, `title`, `newest`, `favorites` (most votes), `gems` (high rating, low popularity).

Every response is passed through a JSON cleaner that converts pandas/NumPy values and `NaN` into plain JSON-safe types.

### Chat response shape

```json
{
  "reply": "...",
  "source_movie": { "Title": "...", "...": "..." },
  "recommendations": [ { "Title": "...", "Semantic_Similarity": 0.76, "...": "..." } ]
}
```

`source_movie` is `null` and `recommendations` is empty when no movie could be identified in the message.

### Examples

```bash
# Pull 5 FAISS recommendations for Interstellar
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"movie":"Interstellar","n":5}'

# Chatbot
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Suggest 5 movies like Interstellar"}'

# What-If re-ranking
curl -X POST http://127.0.0.1:8000/what-if \
  -H "Content-Type: application/json" \
  -d '{"movie":"Interstellar","n":8,"action":70,"romance":10,"phrase":"like Inception but more action"}'
```

---

## Usage Guide

1. **Search a movie.** Type a title in the search bar and press Enter. If there is exactly one match it loads immediately; otherwise click a result card. The hero shows the selected movie with "More Like This" cards below.
2. **Browse home rows.** The carousel auto-scrolls through the most popular films. Curated rows (Trending, Top Rated, New Releases, Hidden Gems, Around the World, Fan Favorites) plus genre rows and the personalized "Made for You" row (once you like a movie) fill the rest of the page.
3. **Open a movie.** Click any card to open the details modal with local metadata and actions: Watchlist, Like / Dislike, Compare (up to 3 movies in the floating dock), and More Like This.
4. **Recommendation Lab.** Open the lab (top-right flask icon) to try AI Discovery ("emotional sci-fi movies about space"), What-If sliders (seed "Interstellar"), Watchlist, Preferences, Insights, and History.
5. **Chat with the bot.** Click Chat and try "Suggest 5 movies like Interstellar", "Give me 3 movies similar to Avengers", or "Recommend movies like Titanic".

---

## Backend Modules

### `backend/app.py` (FastAPI)

- Serves the whole frontend: `index.html` at `/`, and also mounts `/static`, `/css`, `/js` so a plain IDE live-preview of `frontend/` works too.
- **CORS middleware** allows any origin.
- `/movies` supports `sort_by=popularity|rating|title|newest|favorites|gems` plus `genre`, `lang`, `not_lang` and `min_votes` filters — this powers the carousel, all curated rows, the personalized "Made for You" row and the genre rows.
- `/what-if` accepts six integer attribute weights plus an optional `phrase`; NLP phrases from `chatbot.parse_whatif_phrase` are merged into the weights.
- `/compare` accepts 2–3 ids/titles and, for a pair, computes a cosine `Pair_Similarity` from the existing embeddings.
- Every response passes through `_clean()`, which converts pandas/NumPy values (and `NaN`) into JSON-safe types.
- Pydantic models validate every POST body.

### `backend/recommendation_engine.py`

Module-level code runs **once** when the app first imports it: it resolves `models/` from `Path(__file__)`, fails fast with a clear `FileNotFoundError` for missing artifacts, loads the CSV (adding a `movie_id` if absent), loads the embeddings (casting to float32), loads the FAISS index, verifies row alignment, and caches lowercase title/overview/genre columns plus popularity/rating series.

| Function | Purpose |
|----------|---------|
| `find_movie(title)` | exact case-insensitive title → DataFrame index / `None` |
| `find_movie_index_by_id(movie_id)` | resolve `movie_id` (str/int/float) → DataFrame index |
| `get_movie_details(movie_id)` | full-fields dict by id (accepts int or string; NaN-safe) |
| `get_recommended_movie_details(title, n)` | FAISS top-n similar movies with similarity score + reasons |
| `search_movies(query, limit)` | case-insensitive substring title search |
| `more_like_this(movie_id, n)` | same FAISS path keyed on `movie_id` |
| `recommend_personalized(title, n, prefs)` | blended score `0.5·semantic + 0.3·genre + 0.2·pref` |
| `what_if_recommendations(title, weights, n)` | FAISS neighbourhood re-ranked with attribute weights |
| `discover_movies(query, limit)` | grounded discovery (embedding search if available, else keyword fallback) |
| `build_reasons(query, candidate, prefs)` | deterministic "why recommended" factors/reasons |
| `genre_list(value)` | `"Action, Adventure"` → `["Action","Adventure"]`, NaN-safe |
| `themes_in_text(text)` | which of six theme dimensions' keywords appear in a text |

### `backend/chatbot.py`

An orchestration layer, not a model:

1. `_extract_count(message)` — finds a *count* only when the number is clearly tied to movie wording (e.g. `"5 movies"`, `"give me 3"`). Numbers embedded in titles (`"12 Angry Men"`, `"1917"`) are ignored → default **5**.
2. `_extract_title(message)` — handles quotes, `"like X"`, `"similar to X"`, `"such as X"`, and intent-prefixed requests like `"recommend Avengers Endgame please"`.
3. `_best_title_match(candidate)` — exact match first, then **punctuation-folded** substring matching so `"Avengers Endgame"` finds `"Avengers: Endgame"`.
4. `engine.get_recommended_movie_details(...)` — the real FAISS results.
5. `_ai_summarize(...)` — optional: sends the real results to **Groq** with a strict prompt ("only mention movies from this list") to produce a friendly 2–4 sentence reply. If `GROQ_API_KEY` is unset or the call fails, it falls back to a deterministic reply.
6. `parse_whatif_phrase(phrase)` — maps slider-language (`"more emotional"`, `"less serious"`, `"like X but more action"`) onto the six What-If dimensions and optionally extracts a seed title.

### `backend/llm.py`

The single place where the chatbot's external provider is configured and created:

- Wraps the official `groq` SDK; the client is created lazily and cached.
- Reads `GROQ_API_KEY` (legacy `AI_API_KEY` fallback), `GROQ_MODEL` (default `openai/gpt-oss-120b`), `GROQ_BASE_URL` and `GROQ_TIMEOUT` (default 30 s). The key value is never logged.
- `chat_completion(...)` maps Groq errors to safe, actionable messages — authentication (401), permission (403), rate limit (429), timeout, connection error, and invalid/retired model id — then raises `LLMUnavailable` so callers fall back locally.
- `provider_status()` returns a diagnostic snapshot that contains **no** secrets.

### `backend/storage.py`

A tiny, dependency-free JSON store used by the other modules:

- `load_json(filename, default)` — missing or malformed files return the default.
- `save_json(...)` — **atomic** writes (temp file + `os.replace`) guarded by a per-file lock and a cross-process lock, so concurrent requests cannot corrupt a file.
- `update_json(...)` — an atomic read-modify-write under both locks, which prevents lost updates when multiple processes write the same file.
- `now_iso()` — ISO-8601 UTC timestamps.
- Files live in `data/` (auto-created), one level above `backend/`.

### `backend/watchlist.py`

- Stores `{movie_id, title, added_at}` entries in `data/watchlist.json`.
- `add_to_watchlist` ignores duplicates (returns `added:false`) and raises `LookupError` for unknown ids.
- `get_watchlist()` **enriches** each entry with live metadata from `movie_metadata.csv` — the JSON file stays small and the dataset remains the single source of truth.

### `backend/preferences.py`

- Stores likes/dislikes in `data/user_preferences.json`.
- After every `like`/`dislike`, `_recompute_tags` **derives** `preferred_genres`, `preferred_themes` and `favorite_genres` (with counts) from the liked movies using real dataset genres — never invented categories.
- Liking a movie removes it from dislikes, and vice-versa.

### `backend/insights.py`

- Logs activity to `data/recommendation_history.json` (capped at 500 entries): `search`, `recommend`, `discovery`, `whatif`, `feedback`.
- `get_insights()` aggregates totals, favourite genres (from preferences), most-searched genres, recently recommended titles and the last 20 activity events. Best-effort — logging never breaks a request.

---

## Frontend

Static files under `frontend/`, all served by FastAPI (no build step, no bundler):

| File | Purpose |
|------|---------|
| `index.html` | Page skeleton: hero, carousel, home rows, detail modal, Compare dock, Recommendation Lab, chatbot panel |
| `css/style.css` | Main styles (dark premium theme, cards, chat, lab) |
| `css/responsive.css` | Mobile / tablet breakpoints |
| `js/api.js` | Thin `fetch` wrapper — the single place that talks to the API |
| `js/app.js` | UI logic: search, rows, modal, compare dock, RecLab tabs, chat |
| `js/utils.js` | Formatters: year, rating, similarity %, poster, genre, escaping |
| `js/notifications.js` | Toast notifications |

### Key behaviours

- **Search** — pressing Enter or the Search button calls `/search`. A single result immediately loads that movie plus recommendations; otherwise cards render. Clicking a search-result card loads that movie with recommendations, while recommendation / row / chatbot cards open the detail modal.
- **Home rows** — the carousel (top 10 by popularity) auto-scrolls; Trending Now, Top Rated (min 500 votes), New Releases, Hidden Gems (min 150 votes), Around the World (`not_lang=en`), Fan Favorites and the daily-rotating genre rows all come from `/movies` with backend-side sorting. **Made for You** appears once the user has liked at least one movie (built from top preferred genres).
- **Hero stats** — the movie count (`9,828`) and embedding dimension (`384`) are fetched from `/api` at startup so they always match the real catalogue.
- **Detail modal** — full local fields plus action buttons: Watchlist (toggle via `/watchlist/check`), Like/Dislike (updates preferences), Compare (up to 3 chips in the floating dock), More Like This, and the deterministic **Why recommended** reasons.
- **Compare dock** — a floating tray; with 2–3 chips it calls `/compare` and shows side-by-side cards (pairwise similarity for pairs).
- **Recommendation Lab** — `AI Discovery`, `What-If`, `Watchlist`, `Preferences`, `Insights` and `History` as tabs in one workspace, each rendering into `#recLabWorkspace` without reloading the page.
- **Chat box** — quick-reply chips, a typing indicator, and clickable movie cards that open the detail modal.
- **Mobile** — the chat panel slides over the screen; hover-free card layouts.

### HTML escaping

All user-visible strings pass through `Utils.esc()`, and missing posters fall back to an inline SVG data-URI placeholder, so unusual titles never break the layout or inject markup.

---

## Design Principles & Constraints

1. **Local-first** — search, recommendations, details, posters, home rows, and every suggestion come purely from `models/`. No external movie API (TMDB etc.) is called at runtime.
2. **One-time model load** — embeddings / FAISS / CSV load once at import and are reused; the model is never retrained or re-indexed per request.
3. **The chatbot never invents movies** — it can only talk about the engine's real output.
4. **Grounded discovery** — `discover_movies` returns only real rows; the sentence-transformer path (if installed) searches the local index, and the fallback ranker uses keyword/theme hits over local text.
5. **Alignment safety** — metadata, embeddings, and FAISS must have identical row counts; the app refuses to start otherwise.
6. **JSON-safe responses** — pandas/NumPy natives and `NaN` are normalized before returning to the browser.
7. **Deterministic explanations** — `build_reasons` is pure rule-based logic; no LLM is consulted to justify a recommendation.
8. **Safe local persistence** — all user data is written atomically to `data/`; a malformed JSON file degrades to defaults instead of crashing a request.

---

## Testing

The test suite uses `unittest` with mocks only — no network access and no real API key required. It covers the Groq provider and client lifecycle, error mapping and key-safety, the chatbot fallbacks (including "never invents movies"), Sentence-model lazy loading and discovery paths, and storage atomic-write / cross-process locking.

```bash
python -m unittest discover -s tests -v
```

Expected result:

```
Ran 39 tests in 1.344s

OK
```

> The tests exercise the modules directly; they do not spin up the live HTTP layer.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Model artifacts not found` | One of the three files is missing from `models/` |
| `INDEX ALIGNMENT ERROR` | CSV / embeddings / FAISS row counts differ — re-export from the notebook without reordering or filtering rows |
| Cannot reach backend | Uvicorn not running, or the frontend opened from `file://` — use the server URL, or set `API.BASE` in `frontend/js/api.js` |
| Posters missing | `Poster_Url` empty → placeholder rendered; no external poster API is called |
| Chatbot gives plain replies | `GROQ_API_KEY` unset/empty — set it in `.env` and restart the server |
| Chatbot logs "Groq call failed" | Check the message: `401` = bad key, `429` = rate limit (retry), `timed out`/`connection` = network or `GROQ_TIMEOUT`, `invalid/retired model id` = fix `GROQ_MODEL` |
| `.env` values ignored | `python-dotenv` missing from an old environment — `pip install -r requirements.txt` (installs it) or `pip install python-dotenv`, then restart |
| `No module named 'groq'` | Run `python -m pip install -r requirements.txt` in the same interpreter/venv used to start the server |
| Preferences / watchlist not saving | The `data/` directory is auto-created next to `backend/` on first write; verify it exists |
| CORS errors | All origins are allowed; if they persist, ensure you hit `127.0.0.1:8000` (some setups differ between `localhost` and `127.0.0.1`) |

---

## Documentation

- [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) — internal architecture: startup/initialization, request workflows (recommendation, AI Discovery, chatbot), Mermaid diagrams, storage, error/fallback paths, and a request-tracing guide.
- `notebook/CineMind.ipynb` — end-to-end notebook: TF-IDF baseline → Sentence Transformer + FAISS, with artifact export.

---

## License

This project is provided for educational use. No license file is included — clarify ownership and licensing terms before redistribution.
