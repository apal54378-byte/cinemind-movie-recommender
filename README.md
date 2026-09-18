# CineMind

A **semantic movie recommendation platform** built on Sentence Transformer embeddings and FAISS similarity search, wrapped in a FastAPI backend with a lightweight HTML/CSS/JS frontend.

CineMind recommends films based on the *meaningful content* of a movie's plot description — not just genre keywords. When you search for a movie, the system retrieves visually and conceptually similar titles from a local catalogue of **9,828 movies** using 384-dimensional vector embeddings.

> **Design principle — local-first:** Every search result, recommendation, movie detail, poster, and home-page row comes from local artifacts in `models/`. The only external AI call is the optional chatbot, which merely phrases a natural-language reply around results the local engine already produced. The chatbot is never a source of movie data.

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
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## Key Features

| Feature | Description |
|---------|-------------|
| Semantic search & recommendations | FAISS-based nearest-neighbour retrieval with similarity scores |
| Curated home page | Hero carousel, Trending Now, Top Rated, New Releases, Hidden Gems, Around the World, Fan Favorites, and daily-rotating genre rows |
| Personalized "Made for You" row | Ranks movies from genres you have liked |
| Movie detail modal | Full local metadata with Like / Dislike, Watchlist, Compare, More Like This, and deterministic "why recommended" explanations |
| Compare tool | Side-by-side comparison of 2–3 movies with pairwise cosine similarity |
| Recommendation Lab | Six tools: AI Discovery, What-If, Watchlist, Preferences, Insights, History |
| What-If re-ranking | Adjust attribute sliders (action, romance, emotion, sci-fi, comedy, seriousness) or type a natural-language phrase such as *"like Inception but more action"* |
| Chatbot | Extracts intent from plain language, queries the local engine, and replies via an OpenAI-compatible API (falls back to a deterministic reply without a key) |
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
│  POST /recommend            FAISS recommendations                                  │
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

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.9+, FastAPI, Uvicorn |
| ML / retrieval | Sentence Transformers (`all-MiniLM-L6-v2`), FAISS, NumPy, pandas |
| Frontend | Plain HTML, CSS, JavaScript (no build step, no bundler) |
| Chatbot | OpenAI-compatible API (OpenAI, Groq, Together, or local Ollama) |
| Storage | Atomic JSON files under `data/` |

**Note:** `sentence-transformers` is **not** required at runtime — embeddings are pre-computed and loaded from disk. It is only needed by the notebook to *create* the embeddings and by the optional AI Discovery semantic path (otherwise AI Discovery falls back to a deterministic keyword/theme ranker).

---

## Project Structure

```
CineMind/
├── backend/
│   ├── app.py                    # FastAPI application, endpoints, CORS, static serving
│   ├── recommendation_engine.py  # Loads artifacts once; FAISS search & ranking logic
│   ├── chatbot.py                # Intent extraction, AI reply orchestration, NLP parser
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
│       ├── app.js                # UI logic: search, rows, modal, compare, labs, chat
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
├── .env.example                  # Environment variable template
├── .env                          # Local configuration (gitignored)
├── requirements.txt
├── README.md
└── PROJECT_DOCUMENTATION.md      # Deep-dive technical documentation
```

---

## Getting Started

### Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- The three model artifacts in `models/` (see [Model artifacts](#the-three-model-artifacts))

> **Important:** The `.gitignore` excludes `models/`, `data/`, and `.env` so these are never committed. Copy the artifacts into `models/` locally after cloning.

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

Then edit `.env` (see [Configuration](#configuration)). The backend auto-loads `.env` on startup via `python-dotenv`.

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
| `AI_API_KEY` | *(empty — fallback replies)* | API key for the chatbot's AI provider |
| `AI_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint |
| `AI_MODEL` | `llama-3.3-70b-versatile` | Chat model name |
| `HOST` | `127.0.0.1` | Bind host (used with `python -m backend.app`) |
| `PORT` | `8000` | Bind port |

The chatbot works with any OpenAI-compatible provider:

| Provider | Base URL | Example model |
|----------|----------|---------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq (free) | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Together | `https://api.together.xyz/v1` | `meta-llama/Llama-3-70b` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3` (no key required) |

Without a key the chatbot still works using a deterministic fallback reply.

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

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Model artifacts not found` | One of the three files is missing from `models/` |
| `INDEX ALIGNMENT ERROR` | CSV / embeddings / FAISS row counts differ — re-export from the notebook without reordering or filtering rows |
| Cannot reach backend | Uvicorn not running, or the frontend opened from `file://` — use the server URL, or set `API.BASE` in `frontend/js/api.js` |
| Posters missing | `Poster_Url` empty → placeholder rendered; no external poster API is called |
| Chatbot gives plain replies | `AI_API_KEY` unset — set it in `.env` and restart the server |
| Preferences / watchlist not saving | The `data/` directory is auto-created next to `backend/` on first write; verify it exists |
| CORS errors | All origins are allowed; if they persist, ensure you hit `127.0.0.1:8000` (some setups differ between `localhost` and `127.0.0.1`) |

---

## Documentation

- `PROJECT_DOCUMENTATION.md` — in-depth documentation: ML model internals, backend modules, frontend behaviors, API design, and design constraints.
- `notebook/CineMind.ipynb` — end-to-end notebook: TF-IDF baseline → Sentence Transformer + FAISS, with artifact export.

---

## License

This project is provided for educational use. No license file is included — clarify ownership and licensing terms before redistribution.