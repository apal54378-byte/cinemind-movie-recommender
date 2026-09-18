/* ============================================================
   CineMind — API client (talks to FastAPI backend)
   ============================================================ */

const API = {
  // Leave BASE as null to auto-detect: if the page is served by the
  // FastAPI backend itself, same-origin paths are used (works on any
  // port). When the frontend is opened as a plain static preview
  // (IDE live preview / Live Server, etc.), requests automatically go
  // to the backend at http://127.0.0.1:8000. Set BASE explicitly to
  // override the fallback (e.g. a deployed backend URL).
  BASE: null,
  _basePromise: null,

  async _resolveBase() {
    if (this.BASE != null) return this.BASE;
    if (this._basePromise) return this._basePromise;
    this._basePromise = (async () => {
      // Is this page already served by the backend? Then /api works same-origin.
      try {
        const res = await fetch('/api', { method: 'GET' });
        // Only trust it as the backend if it returns actual JSON — a static
        // server could answer 200 with the index.html for any unknown path.
        const ct = (res.headers.get('content-type') || '').toLowerCase();
        if (res.ok && ct.includes('application/json')) return '';
      } catch (_) {}
      // Not same-origin → static preview → call the FastAPI backend directly.
      return 'http://127.0.0.1:8000';
    })();
    return this._basePromise;
  },

  async _fetch(path, options = {}) {
    try {
      const base = await this._resolveBase();
      const res = await fetch(base + path, options);
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const j = await res.json();
          detail = j.detail || detail;
        } catch (_) {}
        throw new Error(detail);
      }
      return await res.json();
    } catch (err) {
      // Network errors (server not running) also land here
      if (err.message === 'Failed to fetch') {
        throw new Error('Cannot reach backend. Is it running at http://127.0.0.1:8000?');
      }
      throw err;
    }
  },

  // GET /search?q=TITLE
  async search(query) {
    return this._fetch('/search?q=' + encodeURIComponent(query));
  },

  // GET /api — health check + catalogue info (movie count, embedding dim)
  async apiInfo() {
    return this._fetch('/api');
  },

  // GET /movies?limit=N&sort_by=...&genre=...&min_votes=...&lang=...&not_lang=...
  // opts: { genre, lang, notLang, minVotes }  (all optional)
  async movies(limit = 60, sortBy = 'popularity', opts = {}) {
    let path = '/movies?limit=' + limit;
    if (sortBy) path += '&sort_by=' + encodeURIComponent(sortBy);
    if (opts.genre) path += '&genre=' + encodeURIComponent(opts.genre);
    if (opts.lang) path += '&lang=' + encodeURIComponent(opts.lang);
    if (opts.notLang) path += '&not_lang=' + encodeURIComponent(opts.notLang);
    if (opts.minVotes) path += '&min_votes=' + Number(opts.minVotes);
    return this._fetch(path);
  },

  // POST /recommend  {movie, n}
  async recommend(movie, n = 5) {
    return this._fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie, n }),
    });
  },

  // GET /movie/{movie_id}
  async movieDetails(movieId) {
    return this._fetch('/movie/' + encodeURIComponent(movieId));
  },

  // POST /chat  {message}
  async chat(message) {
    return this._fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
  },

  // ---- Watchlist ----
  async watchlist() {
    return this._fetch('/watchlist');
  },

  async watchlistCheck(movieId) {
    return this._fetch('/watchlist/check/' + encodeURIComponent(movieId));
  },

  async watchlistAdd(movieId) {
    return this._fetch('/watchlist/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie_id: String(movieId) }),
    });
  },

  async watchlistRemove(movieId) {
    return this._fetch('/watchlist/remove/' + encodeURIComponent(movieId), {
      method: 'DELETE',
    });
  },

  // ---- Preferences & Feedback ----
  async preferences() {
    return this._fetch('/preferences');
  },

  async feedback(movieId, action) {
    return this._fetch('/preferences/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie_id: String(movieId), action }),
    });
  },

  async prefsReset() {
    return this._fetch('/preferences/reset', { method: 'POST' });
  },

  // ---- Insights ----
  async insights() {
    return this._fetch('/insights');
  },

  // ---- Compare ----
  async compare(movies) {
    return this._fetch('/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movies }),
    });
  },

  // ---- What-If ----
  async whatIf(payload) {
    return this._fetch('/what-if', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  // ---- AI Discovery ----
  async aiDiscovery(query, n) {
    return this._fetch('/ai-discovery', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, n: n || 12 }),
    });
  },
};
