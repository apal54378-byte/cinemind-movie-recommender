# CineMind · Project Documentation

A deep-dive into the **CineMind** semantic movie recommendation platform.

---

## 1. Project Overview

CineMind is a **movie discovery platform** that recommends films similar to a
movie a user likes. Instead of matching keywords or genres, it uses **semantic
search**: every movie's plot description is converted into a 384-dimensional
vector embedding by a **Sentence Transformer** (`all-MiniLM-L6-v2`), then
similarity search is done with **FAISS** (Facebook AI Similarity Search).

The application is split into three parts that work together:

| Layer    | Technology            | Responsibility                                              |
|----------|-----------------------|-------------------------------------------------------------|
| Frontend | Plain HTML / CSS / JS | Search, hero, home-page rows, detail modal, Compare dock, Recommendation Lab, chatbot UI |
| Backend  | FastAPI + uvicorn     | REST endpoints, startup model loading, local JSON "database" for user data |
| ML model | Sentence Transformer + FAISS | Converts text → vectors and finds similar movies      |

> **Key rule of the project:** movie search, recommendations, movie details,
> posters, home-page rows and every suggestion come from **local files** saved
> in `models/`. The **only** external AI call is the chatbot, which uses an
> OpenAI-compatible API simply to write a friendly natural-language reply about
> the *real* recommendations the local engine already produced. The chatbot is
> never a source of movie data. A second, optional use of an embedding model
> exists in **AI Discovery** — when `sentence-transformers` is installed the
> free-text query is embedded and searched with FAISS; otherwise a deterministic
> keyword/theme ranker runs. Both paths only ever return real movies from the
> local dataset.

### Feature map

| Feature | Backend | Where in the UI |
|---------|---------|-----------------|
| Semantic recommendations | `get_recommended_movie_details` | Hero + "More Like This" cards |
| Home catalogue batches | `GET /movies` (+ `sort_by`, filters) | Carousel + 7 curated rows + genre rows |
| Movie details | `GET /movie/{id}` | Detail modal |
| Deterministic explanations | `build_reasons` | Modal "Why recommended" |
| Watchlist | `watchlist.py` → `data/watchlist.json` | Lab → Watchlist, modal button |
| Likes / dislikes | `preferences.py` → `data/user_preferences.json` | Modal buttons, Lab → Preferences |
| Personalized ranking | `recommend_personalized` | `recommend {personalized:true}`, "Made for You" row |
| What-If re-ranking | `what_if_recommendations` + `parse_whatif_phrase` | Lab → What-If (sliders / NLP phrase) |
| AI Discovery | `discover_movies` | Lab → AI Discovery |
| Compare | `POST /compare` + `_pairwise_similarity` | Compare dock from modal |
| Insights / History | `insights.py` → `data/recommendation_history.json` | Lab → Insights, Lab → History |
| Chatbot | `chatbot.py` | Chat panel / floating launcher |

---

## 2. How the ML Model Works

```
Movie overviews (text)
        │
        ▼
Sentence Transformer  (all-MiniLM-L6-v2)
        │  normalize_embeddings=True
        ▼
movie_embeddings.npy        shape = (9828, 384)
        │
        ▼
FAISS IndexFlatIP           faiss_index.bin
        │
        ▼
similarity search  (inner product == cosine similarity, since vectors are normalized)
```

### The three artifacts

These three files (exported by the Jupyter notebook `notebook/CineMind.ipynb`)
are all the runtime needs — the heavy Sentence Transformer is **not** run at
runtime:

1. **`models/movie_embeddings.npy`** — a `(9828, 384)` float32 NumPy array. Row
   `i` is the embedding for the movie at row `i` of the metadata CSV.
2. **`models/faiss_index.bin`** — a serialized `faiss.IndexFlatIP` index built
   from the same embeddings. Because every vector is **normalized** (unit
   length), inner-product ranking equals cosine similarity ranking.
3. **`models/movie_metadata.csv`** — the movie catalogue itself, with the
   columns: `movie_id, Title, Release_Date, Overview, Popularity, Vote_Count,
   Vote_Average, Original_Language, Genre, Poster_Url`.

### Row alignment (critical)

All three artifacts share the **same row order**. On startup the engine checks
that they are aligned:

```
metadata rows  ==  embeddings rows  ==  FAISS vectors
```

If these counts ever differ the server refuses to start with an
`INDEX ALIGNMENT ERROR`, forcing a clean re-export from the notebook.

### The core recommendation lookup

1. `find_movie("Interstellar")` — exact, case-insensitive title match → DataFrame
   index (e.g. `461`).
2. Take `movie_embeddings[461]` as the query vector.
3. `faiss_index.search(query, n+10)` → nearest neighbours by inner product.
4. Skip the query movie itself and any exact-title duplicates.
5. Return the top **n** with a `Semantic_Similarity` score (0–1) and a
   deterministic `Reasons` block (see §4).

### The notebook pipeline (for reference)

The notebook first builds a **TF-IDF baseline** (with genre-weighted `tags_v2`),
then moves to **Sentence Transformer embeddings** and finally **FAISS**. Multiple
approaches are compared (TF-IDF vs hybrid vs semantic), semantic search wins, and
the three artifacts are exported — the models themselves are **not** needed by
the running app.

---

## 3. Architecture

```
 Browser
   │
   │  fetch()  (same-origin when served by the backend; else falls back to http://127.0.0.1:8000)
   ▼
┌──────────────────────── FastAPI  (uvicorn, port 8000)  ───────────────────────┐
│                                                                                │
│  static mounts  /static /css /js ─► frontend/                                 │
│  GET  /                       serves index.html (the UI)                      │
│  GET  /api                    health check + catalogue stats                   │
│  GET  /movies                 catalogue batches (sort_by, genre, lang, …)      │
│  GET  /search?q=              local title search                               │
│  POST /recommend              FAISS recommendations  → engine                  │
│  GET  /movie/{id}             full local details      → engine                 │
│  POST /chat                   chatbot orchestration    → chatbot.py            │
│  GET/POST/DELETE /watchlist…  watchlist                → watchlist.py          │
│  GET/POST /preferences…       feedback & profile      → preferences.py         │
│  GET  /insights               activity statistics     → insights.py            │
│  POST /compare                side-by-side comparison → engine                 │
│  POST /what-if                attribute re-ranking    → engine                 │
│  POST /ai-discovery           grounded discovery      → engine                 │
│                                                                                │
│  ┌────────────── backend/recommendation_engine.py ────────────────┐            │
│  │ loads .npy / .bin / .csv ONCE at import time                  │            │
│  │ find_movie, find_movie_index_by_id, get_movie_details,        │            │
│  │ get_recommended_movie_details, search_movies, more_like_this, │            │
│  │ recommend_personalized, what_if_recommendations,              │            │
│  │ discover_movies, build_reasons, genre_list, themes_in_text    │            │
│  └────────────────────────────────────────────────────────────────┘            │
│  ┌──────────────────────── backend/chatbot.py ────────────────────┐            │
│  │ intent extraction (count + title)     →  LOCAL engine          │            │
│  │ external AI writes the reply ONLY from real results            │            │
│  │ parse_whatif_phrase — NLP slider phrases ("more action")       │            │
│  └────────────────────────────────────────────────────────────────┘            │
│  ┌──────────── JSON stores (backend/storage.py → data/) ─────────┐             │
│  │ watchlist.json · user_preferences.json · recommendation_history.json │      │
│  └────────────────────────────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 models/  (movie_embeddings.npy, faiss_index.bin, movie_metadata.csv)
```

### Data flow on the happy path

1. User searches **"Interstellar"** → frontend calls `GET /search?q=Interstellar`.
2. The backend does a local, case-insensitive substring search of the title.
3. If exactly one result, the frontend calls `POST /recommend` with that title.
4. The engine runs the FAISS lookup and returns `selected_movie` +
   `recommendations` (each with `Semantic_Similarity` and `Reasons`).
5. The frontend shows a "hero" for the selected movie and its similar-movie cards.
6. Clicking any card opens the detail modal via `GET /movie/{id}` — with
   **Watchlist**, **Like/Dislike**, **Compare** and **More Like This** actions.
7. The chatbot flow reuses the same engine: `POST /chat` extracts the title +
   count, calls the engine, then optionally has the AI polish the reply.

---

## 4. Backend in detail

### `backend/recommendation_engine.py`

Module-level code runs **once** when the app first imports it:

- Resolves `models/` from `Path(__file__)` — safe to run from any working dir.
- Fails fast with a clear `FileNotFoundError` listing any missing artifacts.
- Loads the CSV, adds a `movie_id` column if absent, keeps only the documented
  columns, loads embeddings (casts to float32), loads the FAISS index, and
  verifies row alignment.
- Also caches lowercase title / overview / genre columns and popularity / rating
  series for the keyword-based helpers.

Public functions:

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
| `discover_movies(query, limit)` | grounded discovery (embedding search if available, keyword fallback) |
| `build_reasons(query, candidate, prefs)` | deterministic "why recommended" factors/reasons |
| `genre_list(value)` | `"Action, Adventure"` → `["Action","Adventure"]`, NaN-safe |
| `themes_in_text(text)` | which of 6 theme dimensions key-words appear in a text |

### `backend/app.py` (FastAPI)

- Serves the whole frontend: `index.html` at `/` (and also mounts `/static`,
  `/css`, `/js` so a plain IDE live-preview of `frontend/` works too).
- **CORS middleware** allows any origin.
- `/movies` supports `sort_by=popularity|rating|title|newest|favorites|gems`
  plus `genre`, `lang`, `not_lang` and `min_votes` filters — this powers the
  carousel, Trending / Top Rated / New Releases / Hidden Gems / Around the World
  / Fan Favorites rows, the personalized "Made for You" row and the genre rows.
- `/what-if` accepts six integer attribute weights and an optional `phrase`;
  NLP phrases from `chatbot.parse_whatif_phrase` are merged into the weights.
- `/compare` accepts 2–3 ids/titles and, for exactly a pair, computes a cosine
  `Pair_Similarity` from the existing embeddings.
- Every response is passed through `_clean()`, which converts pandas/numpy
  values (and `NaN`) into plain JSON-safe types.
- Pydantic models validate every POST body.

### `backend/chatbot.py`

An orchestration layer, not a model:

1. `_extract_count(message)` — finds a *count* only when the number is clearly
   tied to movie wording (e.g. `"5 movies"`, `"give me 3"`). Numbers embedded in
   titles (`"12 Angry Men"`, `"1917"`) are ignored → default of **5**.
2. `_extract_title(message)` — handles quotes, `"like X"`, `"similar to X"`,
   `"such as X"`, and plain intent-prefixed requests like
   `"recommend Avengers Endgame please"`.
3. `_best_title_match(candidate)` — exact match first, then **punctuation-folded**
   substring matching so `"Avengers Endgame"` finds `"Avengers: Endgame"`.
4. `engine.get_recommended_movie_details(...)` — the real FAISS results.
5. `_ai_summarize(...)` — optional: sends the real results to an OpenAI-compatible
   API with a strict prompt ("only mention movies from this list") to produce a
   friendly 2–4 sentence reply. If no `AI_API_KEY` is set, it falls back to a
   simple deterministic reply.
6. `parse_whatif_phrase(phrase)` — maps slider-language (`"more emotional"`,
   `"less serious"`, `"like X but more action"`) onto the six What-If dimensions
   and optionally extracts a seed title.

The chat response shape is always:

```json
{
  "reply": "...",
  "source_movie": { "Title": "...", ... } | null,
  "recommendations": [ { "Title": "...", "Semantic_Similarity": 0.76, ... } ]
}
```

### `backend/storage.py`

A tiny, dependency-free JSON store used by the other modules:

- `load_json(filename, default)` — missing / malformed files return the default.
- `save_json(...)` — **atomic** writes (temp file + `os.replace`) guarded by a
  per-file lock so concurrent requests cannot corrupt a file.
- `now_iso()` — ISO-8601 UTC timestamps.
- Files live in `data/` (auto-created), one level above `backend/`.

### `backend/watchlist.py`

- Stores `{movie_id, title, added_at}` entries in `data/watchlist.json`.
- `add_to_watchlist` ignores duplicates (returns `added:false`) and raises
  `LookupError` for unknown ids.
- `get_watchlist()` **enriches** each entry with live metadata from
  `movie_metadata.csv` — the JSON file stays small, the dataset stays the
  single source of truth.

### `backend/preferences.py`

- Stores likes/dislikes in `data/user_preferences.json`.
- After every `like`/`dislike`, `_recompute_tags` **derives**
  `preferred_genres`, `preferred_themes` and `favorite_genres` (with counts)
  from the liked movies using real dataset genres — never invented categories.
- Liking a movie removes it from dislikes and vice-versa.

### `backend/insights.py`

- Logs activity to `data/recommendation_history.json` (capped at 500 entries):
  `search`, `recommend`, `discovery`, `whatif`, `feedback`.
- `get_insights()` aggregates totals, favourite genres
  (from preferences), most-searched genres, recently recommended titles and the
  last 20 activity events. Best-effort — logging never breaks a request.

---

## 5. Frontend in detail

Static files under `frontend/`, all loaded from the FastAPI server (no build
step, no bundler):

| File | Purpose |
|------|---------|
| `index.html` | Page skeleton: hero, carousel, home rows, detail modal, Compare dock, Recommendation Lab, chatbot panel |
| `css/style.css` | Main styles (dark premium theme, cards, chat, lab) |
| `css/responsive.css` | Mobile / tablet breakpoints |
| `js/api.js` | Thin `fetch` wrapper — the single place that talks to the API |
| `js/app.js` | UI logic: search, rows, modal, compare dock, RecLab tabs, chat |
| `js/utils.js` | Formatters: year, rating, similarity %, poster, genre, escaping |
| `js/notifications.js` | Toast notifications |

### Key behaviors

- **Search** — pressing Enter or the Search button calls `/search`. If exactly
  one result it immediately loads that movie + recommendations; otherwise cards
  render. Clicking a search-result card loads that movie with recommendations,
  while recommendation / row / chatbot cards open the detail modal.
- **Home rows** — the carousel (top 10 popularity) auto-scrolls; Trending Now,
  Top Rated (min 500 votes), New Releases, Hidden Gems (min 150 votes),
  Around the World (`not_lang=en`), Fan Favorites and the daily-rotating genre
  rows all come from `/movies` with backend-side sorting. **Made for You** is
  shown only once the user has liked at least one movie (built from top preferred
  genres).
- **Hero stats** — the movie count (`9,828`) and embedding dimension (`384`) are
  fetched from `/api` at startup (animated count-up), so they always match the
  real catalogue.
- **Detail modal** — full local fields + action buttons: Watchlist
  (toggle state via `/watchlist/check`), Like/Dislike (updates preferences,
  re-renders "Made for You"), Compare (adds up to 3 chips to the floating dock),
  More Like This (switches hero to those results) and the deterministic
  **Why recommended** reasons.
- **Compare dock** — a floating tray; with 2–3 chips it calls `/compare` and
  shows side-by-side cards (pairwise similarity for pairs).
- **Recommendation Lab** — `AI Discovery`, `What-If`, `Watchlist`,
  `Preferences`, `Insights` and `History` as tabs in one workspace, each
  rendering into `#recLabWorkspace` without ever reloading the page.
- **Chatbox** — quick-reply chips, a typing indicator, and rendered movie cards
  that are clickable (open the detail modal).
- **Mobile** — the chat panel slides over the screen; hover-free card layouts.

### HTML escaping

All user-visible strings pass through `Utils.esc()`, and missing posters fall
back to an inline SVG data-URI placeholder, so titles with apostrophes, quotes,
or unusual characters never break the layout or inject markup.

---

## 6. Setup & Run

```bash
# 1. deps
pip install -r requirements.txt

# 2. chatbot key (optional) — copy the template
cp .env.example .env
#    then edit .env with your AI_API_KEY / AI_BASE_URL / AI_MODEL
#    (the backend auto-loads .env via python-dotenv)

# 3. start the backend (loads the model artifacts)
uvicorn backend.app:app --reload --port 8000
#    or:  python -m backend.app   (respects HOST / PORT from .env)

# 4. open
http://127.0.0.1:8000
```

Server startup should log:

```
[CineMind] Loading movie_metadata.csv ...
[CineMind] Loaded 9828 movies from metadata.
[CineMind] Embeddings shape: (9828, 384)
[CineMind] FAISS index total vectors: 9828
[CineMind] Index alignment OK (9828 movies). Engine ready.
```

### Environment variables (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_API_KEY` | *(empty — fallback replies)* | API key for the chatbot's AI provider |
| `AI_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint |
| `AI_MODEL` | `llama-3.3-70b-versatile` | Chat model name |
| `HOST` | `127.0.0.1` | Bind host (used with `python -m backend.app`) |
| `PORT` | `8000` | Bind port |

Works with OpenAI, Groq, Together, and local **Ollama**
(`AI_BASE_URL=http://localhost:11434/v1`, `AI_MODEL=llama3`, no key required).
`.env` is loaded automatically on startup via `python-dotenv` (installed as a
regular dependency in `requirements.txt`).

### API reference

| Method | Endpoint | Query / Body | Description |
|--------|----------|--------------|-------------|
| GET | `/` | — | Serves the frontend UI |
| GET | `/api` | — | Health check + movie count + dim |
| GET | `/movies` | `limit`, `sort_by`, `genre`, `lang`, `not_lang`, `min_votes` | Catalogue batches |
| GET | `/search` | `q=title` | Local title search |
| POST | `/recommend` | `{"movie","n","personalized"}` | FAISS recommendations (n ≤ 20) |
| GET | `/movie/{movie_id}` | — | Full local movie details |
| POST | `/chat` | `{"message"}` | Chatbot (engine + optional AI reply) |
| GET | `/watchlist` | — | Saved movies, enriched |
| POST | `/watchlist/add` | `{"movie_id"}` | Add (duplicates ignored) |
| DELETE | `/watchlist/remove/{movie_id}` | — | Remove |
| GET | `/watchlist/check/{movie_id}` | — | Membership check |
| GET | `/preferences` | — | Profile + derived tags |
| POST | `/preferences/feedback` | `{"movie_id","action"}` | Like / dislike |
| POST | `/preferences/reset` | — | Clear profile |
| GET | `/insights` | — | Aggregated stats |
| POST | `/compare` | `{"movies":[...]}` | 2–3 movies, pairwise similarity |
| POST | `/what-if` | `{"movie","n",weights,"phrase"}` | Attribute re-ranking |
| POST | `/ai-discovery` | `{"query","n"}` | Grounded discovery (n ≤ 24) |

---

## 7. Repo Layout

```
cinemind-python/
├── backend/
│   ├── app.py                    # FastAPI app + endpoints + CORS + static serving
│   ├── recommendation_engine.py  # loads .npy/.bin/.csv, FAISS lookups + features
│   ├── chatbot.py                # intent extraction + orchestration + AI reply
│   ├── watchlist.py              # watchlist storage
│   ├── preferences.py            # like/dislike → derived genres & themes
│   ├── insights.py               # activity log + stats
│   ├── storage.py                # atomic JSON read/write helpers
│   └── __init__.py
├── frontend/
│   ├── index.html
│   ├── css/{style.css, responsive.css}
│   └── js/{api.js, app.js, utils.js, notifications.js}
├── models/                       # ML artifacts (gitignored, copied from notebook)
│   ├── movie_embeddings.npy
│   ├── faiss_index.bin
│   └── movie_metadata.csv
├── data/                         # runtime user data (gitignored, auto-created)
│   ├── watchlist.json
│   ├── user_preferences.json
│   └── recommendation_history.json
├── notebook/
│   └── CineMind.ipynb         # TF-IDF baseline → Sentence Transformer + FAISS
├── .env.example                  # configuration template
├── .env                          # your actual config (gitignored)
├── requirements.txt
├── README.md
└── SRS.txt                       # Software Requirements Specification (end-to-end spec)
```

---

## 8. Design Rules & Constraints

1. **Local-first** — search, recommendations, details, posters, home rows, and
   every suggestion are purely from `models/`. No external movie API (TMDB etc.)
   is called at runtime.
2. **One-time model load** — embeddings / FAISS / CSV load once at import and are
   reused; never retrain or re-index per request.
3. **The chatbot never invents movies** — it can only talk about the engine's real
   output.
4. **Grounded discovery** — `discover_movies` returns only real rows; the
   sentence-transformer path (if installed) searches the local index, and the
   fallback ranker uses keyword/theme hits over local text.
5. **Alignment safety** — metadata, embeddings, and FAISS must have identical row
   counts; the app refuses to start otherwise.
6. **JSON-safe responses** — pandas/numpy natives and NaN are normalized before
   returning to the browser.
7. **Deterministic explanations** — `build_reasons` is pure rule-based logic; no
   LLM is consulted to justify a recommendation.
8. **Safe local persistence** — all user data is written atomically to `data/`;
   a malformed JSON file degrades to defaults instead of crashing a request.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Model artifacts not found` | One of the three files is missing from `models/`. |
| `INDEX ALIGNMENT ERROR` | CSV/embeddings/FAISS row counts differ — re-export without reordering. |
| `Cannot reach backend` | `uvicorn` not running, or frontend opened from `file://` — use the server URL (or set `API.BASE` in `frontend/js/api.js`). |
| Posters missing | `Poster_Url` empty → placeholder shown; no external poster API is called. |
| Chatbot gives plain replies | `AI_API_KEY` unset/empty — set it in `.env` (auto-loaded by `python-dotenv`) and restart. |
| `.env` values ignored | `python-dotenv` missing from an old environment — `pip install -r requirements.txt` (installs it) or `pip install python-dotenv`, then restart. |
| Watchlist / prefs / insights seem empty | `data/` is created on first write; check it exists next to `backend/`. |
| CORS errors | All origins are allowed; if they persist, ensure you hit the backend origin, not `localhost` vs `127.0.0.1`. |