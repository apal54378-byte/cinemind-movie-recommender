/* ============================================================
   CineMind — Main Application Logic (Premium UI)
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  App.init();
});

const App = {
  init() {
    this.searchInput   = document.getElementById('searchInput');
    this.searchBtn     = document.getElementById('searchBtn');
    this.chatToggleBtn = document.getElementById('chatToggleBtn');
    this.loadingBar    = document.getElementById('loadingBar');
    this.loadingText   = document.getElementById('loadingText');
    this.errorBox      = document.getElementById('errorBox');
    this.selectedSec   = document.getElementById('selectedSection');
    this.searchSec     = document.getElementById('searchResultsSection');
    this.recSec        = document.getElementById('recommendationsSection');
    this.welcomeSec    = document.getElementById('welcomeSection');
    this.searchResults = document.getElementById('searchResults');
    this.recsContainer = document.getElementById('recommendations');
    this.recCount      = document.getElementById('recCount');
    // bottom status ticker (hidden while showing search results)
    this.ticker = document.getElementById('statusTicker');
    this.backHomeFab = document.getElementById('backHomeFab');
    // selected / hero
    this.heroBackdrop = document.getElementById('heroBackdrop');
    this.selPoster  = document.getElementById('selectedPoster');
    this.selTitle   = document.getElementById('selectedTitle');
    this.selOver    = document.getElementById('selectedOverview');
    this.selMeta    = document.getElementById('selectedMeta');
    // modal
    this.modal       = document.getElementById('movieModal');
    this.modalClose  = document.getElementById('modalClose');
    this.modalPoster = document.getElementById('modalPoster');
    this.modalTitle  = document.getElementById('modalTitle');
    this.modalOver   = document.getElementById('modalOverview');
    this.modalMeta   = document.getElementById('modalMeta');
    // chat
    this.chatPanel    = document.getElementById('chatPanel');
    this.chatLauncher = document.getElementById('chatLauncher');
    this.chatBackdrop = document.getElementById('chatBackdrop');
    this.chatCloseBtn = document.getElementById('chatCloseBtn');
    this.chatMessages = document.getElementById('chatMessages');
    this.chatForm     = document.getElementById('chatForm');
    this.chatInput    = document.getElementById('chatInput');
    this.heroExploreBtn = document.getElementById('heroExploreBtn');
    this.heroAIBtn      = document.getElementById('heroAIBtn');
    // recommendation lab (RecLab)
    this.recLab          = document.getElementById('recLab');
    this.recLabNavBtn    = document.getElementById('recLabNavBtn');
    this.recLabWsTitle   = document.getElementById('recLabWsTitle');
    this.recLabWsSub     = document.getElementById('recLabWsSub');
    this.recLabWorkspace = document.getElementById('recLabWorkspace');
    this.recLabTab       = null;
    this.recLabActive    = false;
    this._reclabMeta = {
      discovery:   { title: 'AI Discovery', sub: 'Describe a vibe in plain words — every pick is grounded in the real catalogue.' },
      whatif:      { title: 'What-If', sub: 'Seed from a movie you like, then shift attribute sliders to re-rank the FAISS neighbourhood.' },
      watchlist:   { title: 'My Watchlist', sub: 'Movies you saved for later — open any card for details.' },
      preferences: { title: 'My Preferences', sub: 'Your taste profile: likes, dislikes, genres and themes derived from them.' },
      insights:    { title: 'Recommendation Insights', sub: 'Stats and patterns built from your activity across CineMind.' },
      history:     { title: 'Recommendation History', sub: 'A timeline of every search, recommendation and feedback event.' },
    };
    // feature tool modal
    this.featureModal      = document.getElementById('featureModal');
    this.featureModalClose = document.getElementById('featureModalClose');
    this.featureModalTitle = document.getElementById('featureModalTitle');
    this.featureModalBody  = document.getElementById('featureModalBody');
    this.featureBackdrop   = this.featureModal ? this.featureModal.querySelector('.modal-backdrop') : null;
    // movie modal action bar
    this.modalWatchlistBtn = document.getElementById('modalWatchlistBtn');
    this.modalLikeBtn      = document.getElementById('modalLikeBtn');
    this.modalDislikeBtn   = document.getElementById('modalDislikeBtn');
    this.modalCompareBtn   = document.getElementById('modalCompareBtn');
    this.modalMoreBtn      = document.getElementById('modalTargetBtn');
    this.modalReasonsBox   = document.getElementById('modalReasons');
    // compare dock
    this.compareDock     = document.getElementById('compareDock');
    this.compareChips    = document.getElementById('compareChips');
    this.compareActionBtn = document.getElementById('compareActionBtn');
    this.compareClearBtn = document.getElementById('compareClearBtn');

    this.modalId = null;
    this.modalMovieTitle = null;
    this.compareList = [];
    this._reasonsCache = {};

    this._bindEvents();
    this._welcomeChat();
    this._loadStats();       // hero stat numbers come from /api
    this._loadHomeRows();   // fetch trending + top-rated on page load
  },

  _bindEvents() {
    this.searchBtn.addEventListener('click', () => this.doSearch());
    this.searchInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') this.doSearch();
    });
    if (this.heroExploreBtn) {
      this.heroExploreBtn.addEventListener('click', () => {
        const target = document.getElementById('carouselSection');
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
    if (this.heroAIBtn) {
      this.heroAIBtn.addEventListener('click', () => this._toggleChat(true));
    }
    document.querySelectorAll('.chip[data-search]').forEach(chip => {
      chip.addEventListener('click', () => {
        this.searchInput.value = chip.dataset.search;
        this.doSearch();
      });
    });
    this.modalClose.addEventListener('click', () => this._closeModal());
    this.modal.querySelector('.modal-backdrop').addEventListener('click', () => this._closeModal());
    document.querySelectorAll('.btn-back').forEach(btn => {
      btn.addEventListener('click', () => this._backHome());
    });
    if (this.backHomeFab) {
      this.backHomeFab.addEventListener('click', () => this._backHome());
    }
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        if (this.chatPanel.classList.contains('open')) this._toggleChat(false);
        else if (!this.modal.classList.contains('hidden')) this._closeModal();
        else if (this.featureModal && !this.featureModal.classList.contains('hidden')) this._closeFeature();
        else if (this.recLabActive) this._backHome();
      }
    });
    // recommendation lab (RecLab) navigation + sub-tabs
    if (this.recLabNavBtn) this.recLabNavBtn.addEventListener('click', () => this.openRecLab());
    document.querySelectorAll('.reclab-tab').forEach(btn => {
      btn.addEventListener('click', () => this.switchRecLabTab(btn.dataset.tab));
    });
    // feature modal open/close
    if (this.featureModalClose) this.featureModalClose.addEventListener('click', () => this._closeFeature());
    if (this.featureBackdrop) this.featureBackdrop.addEventListener('click', () => this._closeFeature());
    // movie modal actions
    if (this.modalWatchlistBtn) this.modalWatchlistBtn.addEventListener('click', () => this.toggleWatchlist());
    if (this.modalLikeBtn) this.modalLikeBtn.addEventListener('click', () => this.sendFeedback('like'));
    if (this.modalDislikeBtn) this.modalDislikeBtn.addEventListener('click', () => this.sendFeedback('dislike'));
    if (this.modalCompareBtn) this.modalCompareBtn.addEventListener('click', () => {
      if (!this.modalId) return;
      this.addToCompare(this.modalId, this.modalMovieTitle);
      this._setModalCompareAdded();
    });
    if (this.modalMoreBtn) this.modalMoreBtn.addEventListener('click', () => {
      const t = this.modalMovieTitle;
      if (!t) return;
      this._closeModal();
      this._closeFeature();
      this.showMovieAndRecommend(t);
    });
    // compare dock
    if (this.compareActionBtn) this.compareActionBtn.addEventListener('click', () => this.openCompare());
    if (this.compareClearBtn) this.compareClearBtn.addEventListener('click', () => {
      this.compareList = [];
      this._renderCompareDock();
    });
    this.chatToggleBtn.addEventListener('click', () => this._toggleChat(true));
    this.chatLauncher.addEventListener('click', () => {
      // toggle on/off when clicking the floating button
      if (this.chatPanel.classList.contains('open')) this._toggleChat(false);
      else this._toggleChat(true);
    });
    this.chatBackdrop.addEventListener('click', () => this._toggleChat(false));
    this.chatCloseBtn.addEventListener('click', () => this._toggleChat(false));
    // quick reply chips
    document.querySelectorAll('.quick-reply').forEach(chip => {
      chip.addEventListener('click', () => {
        this.chatInput.value = chip.dataset.msg;
        this.sendChat();
      });
    });
    this.chatForm.addEventListener('submit', e => {
      e.preventDefault();
      this.sendChat();
    });
    document.body.addEventListener('click', e => {
      // Why Recommended toggle (inside rec cards)
      const why = e.target.closest('.rec-why-toggle');
      if (why) {
        const body = why.closest('.rec-why').querySelector('.rec-why-body');
        if (body) body.classList.toggle('open');
        why.classList.toggle('open');
        return;
      }
      // remove a chip from the compare tray
      const cx = e.target.closest('[data-compare-remove]');
      if (cx) {
        this._removeCompare(cx.dataset.compareRemove);
        return;
      }
      const card = e.target.closest('[data-movie-id]');
      if (card) {
        if (card.hasAttribute('data-search-title')) {
          this.showMovieAndRecommend(card.dataset.searchTitle);
        } else {
          this.openMovieDetails(card.dataset.movieId, {
            reasons: this._reasonsCache[card.dataset.movieId]
          });
        }
      }
    });
  },

  _showLoading(text = 'Loading…') {
    this.loadingText.textContent = text;
    this.loadingBar.classList.remove('hidden');
  },
  _hideLoading() { this.loadingBar.classList.add('hidden'); },

  _showError(msg) {
    this.errorBox.textContent = msg;
    this.errorBox.classList.remove('hidden');
    setTimeout(() => this.errorBox.classList.add('hidden'), 5000);
  },

  _hideSections() {
    this.selectedSec.classList.add('hidden');
    this.searchSec.classList.add('hidden');
    this.recSec.classList.add('hidden');
    this.welcomeSec.classList.add('hidden');
  },

  // Show/hide the home content (brand hero, carousel, rows) so search
  // results take over the top of the page like a featured stage.
  _toggleHome(show) {
    [
      'brandHeroSection', 'carouselSection', 'trendingSection', 'topRatedSection',
      'newReleasesSection', 'gemsSection', 'worldSection', 'favoritesSection',
      'forYouSection', 'genreRowsContainer',
    ].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.toggle('hidden', !show);
    });
    if (this.ticker) this.ticker.classList.toggle('hidden', !show);
    if (this.backHomeFab) this.backHomeFab.classList.toggle('hidden', show);
  },

  // Back to the home screen (hide results + show home content + ticker)
  _backHome() {
    this._hideSections();
    this._closeRecLab();
    this.searchResults.innerHTML = '';
    this.recsContainer.innerHTML = '';
    this._hideLoading();
    this._toggleHome(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  },

  // ---------- Recommendation Lab (RecLab) ----------
  _toolBody() { return this.recLabWorkspace; },

  openRecLab() {
    if (this.recLabActive) { this._backHome(); return; }
    if (this.featureModal && !this.featureModal.classList.contains('hidden')) this._closeFeature();
    this._hideSections();
    this._toggleHome(false);
    this.recLab.classList.remove('hidden');
    if (this.recLabNavBtn) this.recLabNavBtn.classList.add('active');
    this.recLabActive = true;
    this.switchRecLabTab(this.recLabTab || 'discovery');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  },

  _closeRecLab() {
    if (!this.recLabActive) return;
    this.recLab.classList.add('hidden');
    if (this.recLabNavBtn) this.recLabNavBtn.classList.remove('active');
    this.recLabActive = false;
    if (this.recLabWorkspace) this.recLabWorkspace.innerHTML = '';
  },

  // Seamless sub-tab switching — only the workspace re-renders, no page reload.
  switchRecLabTab(tabKey) {
    const meta = this._reclabMeta[tabKey];
    if (meta) {
      if (this.recLabWsTitle) this.recLabWsTitle.textContent = meta.title;
      if (this.recLabWsSub) this.recLabWsSub.textContent = meta.sub;
    }
    document.querySelectorAll('.reclab-tab').forEach(btn =>
      btn.classList.toggle('active', btn.dataset.tab === tabKey)
    );
    this.recLabTab = tabKey;
    const renderers = {
      discovery: 'openDiscovery',
      whatif: 'openWhatIf',
      watchlist: 'openWatchlist',
      preferences: 'openPreferences',
      insights: 'openInsights',
      history: 'openHistory',
    };
    const fn = renderers[tabKey];
    if (fn && typeof this[fn] === 'function') this[fn]();
  },

  // ---------- Search ----------
  async doSearch() {
    this._closeRecLab();
    const q = this.searchInput.value.trim();
    if (!q) {
      Notifications.show('Please type a movie title.', 'info');
      return;
    }
    this._showLoading('Searching for "' + q + '"…');
    this._hideSections();
    try {
      const data = await API.search(q);
      this._hideLoading();
      if (!data.results || data.results.length === 0) {
        this._toggleHome(true);
        this.welcomeSec.classList.remove('hidden');
        Notifications.show('No matches for "' + q + '".', 'info');
        return;
      }
      if (data.results.length === 1) {
        this.showMovieAndRecommend(data.results[0].Title);
        return;
      }
      this._toggleHome(false);
      this._renderSearchResults(data.results);
    } catch (err) {
      this._hideLoading();
      this._toggleHome(true);
      this.welcomeSec.classList.remove('hidden');
      this._showError(err.message);
    }
  },

  _renderSearchResults(movies) {
    this._hideSections();
    this.searchSec.classList.remove('hidden');
    this.searchResults.innerHTML = movies.map(m => this._movieCardHTML(m, false, true)).join('');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  },

  // ---------- Show a movie + its recommendations ----------
  async showMovieAndRecommend(title) {
    this._closeRecLab();
    this._showLoading('Finding recommendations for "' + title + '"…');
    this._hideSections();
    try {
      const data = await API.recommend(title, 8);
      this._hideLoading();
      this._renderSelected(data.selected_movie);
      this._renderRecommendations(data.recommendations);
      this._toggleHome(false);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      this._hideLoading();
      this._toggleHome(true);
      this.welcomeSec.classList.remove('hidden');
      this._showError(err.message);
    }
  },

  _renderSelected(movie) {
    if (!movie) return;
    this._hideSections();
    this.selectedSec.classList.remove('hidden');
    this.recSec.classList.remove('hidden');

    const poster = Utils.poster(movie.Poster_Url) || Utils.placeholder(movie.Title);

    // Hero backdrop: use poster as blurred background
    if (this.heroBackdrop && poster) {
      this.heroBackdrop.style.backgroundImage = `url("${poster}")`;
    }

    this.selPoster.src = poster;
    this.selPoster.onerror = () => { this.selPoster.src = Utils.placeholder(movie.Title); };
    this.selTitle.textContent = movie.Title;
    this.selOver.textContent = movie.Overview || 'No overview available.';

    const genres = Utils.genres(movie.Genre);
    this.selMeta.innerHTML =
      `<span class="badge rating">★ ${Utils.rating(movie.Vote_Average)}</span>` +
      `<span class="badge">${Utils.year(movie.Release_Date)}</span>` +
      genres.map(g => `<span class="badge">${Utils.esc(g)}</span>`).join('') +
      `<span class="badge">${movie.Vote_Count || 0} votes</span>`;
  },

  _renderRecommendations(movies) {
    if (!movies || movies.length === 0) {
      this.recCount.textContent = '(none)';
      this.recsContainer.innerHTML = '<p class="empty-note">No similar movies found.</p>';
      return;
    }
    this.recCount.textContent = '(' + movies.length + ')';
    this._reasonsCache = {};
    movies.forEach(m => { if (m && m.Reasons) this._reasonsCache[m.movie_id] = m.Reasons; });
    this.recsContainer.innerHTML = movies.map((m, i) => this._recCardHTML(m, i + 1)).join('');
  },

  // "Why Recommended" block — deterministic reasons from the local engine
  _whyBlock(movie) {
    const r = movie && movie.Reasons;
    if (!r || !r.reasons || r.reasons.length === 0) return '';
    return `
      <div class="rec-why">
        <button type="button" class="rec-why-toggle">Why recommended?</button>
        <div class="rec-why-body">
          ${r.reasons.slice(0, 5).map(x => `<span>${Utils.esc(x)}</span>`).join('')}
        </div>
      </div>`;
  },

  _recCardHTML(movie, rank) {
    const poster = Utils.poster(movie.Poster_Url) || Utils.placeholder(movie.Title);
    const ph = Utils.placeholder(movie.Title).replace(/'/g, "\\'");
    const sim = movie.Semantic_Similarity || 0;
    const simPct = Utils.simPercent(sim);
    const width = Math.max(0, Math.min(100, Math.round(sim * 100)));
    const genres = Utils.genres(movie.Genre);
    const top = rank === 1;
    const why = this._whyBlock(movie);
    const topBadge = top
      ? `<span class="top-badge">
           <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="m5 12 5 5L20 7"/></svg>
           TOP MATCH</span>`
      : '';
    return `
      <div class="rec-card${top ? ' top' : ''}" data-movie-id="${movie.movie_id}">
        <div class="rec-poster-wrap">
          <span class="rec-rank">${rank}</span>
          <img src="${poster}" alt="${Utils.esc(movie.Title)}" onerror="this.src='${ph}'" />
          <div class="rec-open">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8Z"/><circle cx="12" cy="12" r="3"/>
            </svg>
          </div>
        </div>
        <div class="rec-info">
          ${topBadge}
          <div class="rec-title">${Utils.esc(movie.Title)}</div>
          <div class="rec-meta">
            <span class="star">★</span> ${Utils.rating(movie.Vote_Average)}
            <span>·</span> ${Utils.year(movie.Release_Date)}
            <span>·</span> <span class="rec-genre">${Utils.esc(genres[0] || 'Film')}</span>
          </div>
          <div class="rec-match">
            <div class="match-top">
              <span class="match-label">Semantic match</span>
              <span class="match-value">${simPct}</span>
            </div>
            <div class="match-bar"><span class="match-fill" style="width:${width}%"></span></div>
          </div>
          ${why}
        </div>
      </div>`;
  },

  _movieCardHTML(movie, showSim, searchResult) {
    const poster = Utils.poster(movie.Poster_Url) || Utils.placeholder(movie.Title);
    const genres = Utils.genres(movie.Genre);
    const sim = showSim ? `<span class="sim-badge">${Utils.simPercent(movie.Semantic_Similarity)}</span>` : '';
    const ph = Utils.placeholder(movie.Title).replace(/'/g, "\\'");
    const searchAttr = searchResult ? ` data-search-title="${Utils.esc(movie.Title)}"` : '';
    return `
      <div class="movie-card small" data-movie-id="${movie.movie_id}"${searchAttr}>
        <div class="poster-wrap">
          <img src="${poster}" alt="${Utils.esc(movie.Title)}"
               onerror="this.src='${ph}'" />
          ${sim}
        </div>
        <div class="card-body">
          <div class="card-title">${Utils.esc(movie.Title)}</div>
          <div class="card-meta">
            <span class="star">★</span> ${Utils.rating(movie.Vote_Average)}
            <span>·</span>
            <span>${Utils.year(movie.Release_Date)}</span>
            <span>·</span>
            <span>${Utils.esc(genres[0] || 'Film')}</span>
          </div>
        </div>
      </div>`;
  },

  // ---------- Movie details modal ----------
  async openMovieDetails(movieId, extra) {
    if (this.featureModal && !this.featureModal.classList.contains('hidden')) this._closeFeature();
    this._showLoading('Loading movie details…');
    try {
      const movie = await API.movieDetails(movieId);
      this._hideLoading();
      this._renderModal(movie, extra);
    } catch (err) {
      this._hideLoading();
      this._showError(err.message);
    }
  },

  _renderModal(movie, extra) {
    const poster = Utils.poster(movie.Poster_Url) || Utils.placeholder(movie.Title);
    this.modalPoster.src = poster;
    this.modalPoster.onerror = () => { this.modalPoster.src = Utils.placeholder(movie.Title); };
    this.modalTitle.textContent = movie.Title;
    this.modalOver.textContent = movie.Overview || 'No overview available.';

    this.modalId = String(movie.movie_id);
    this.modalMovieTitle = movie.Title;

    const genres = Utils.genres(movie.Genre);
    this.modalMeta.innerHTML = `
      <div class="meta-item"><label>Rating</label><span>★ ${Utils.rating(movie.Vote_Average)} (${movie.Vote_Count || 0} votes)</span></div>
      <div class="meta-item"><label>Release Date</label><span>${Utils.esc(movie.Release_Date || 'N/A')}</span></div>
      <div class="meta-item"><label>Genres</label><span>${genres.map(g=>Utils.esc(g)).join(', ') || 'N/A'}</span></div>
      <div class="meta-item"><label>Language</label><span>${Utils.esc(movie.Original_Language || 'N/A')}</span></div>
      <div class="meta-item"><label>Popularity</label><span>${Utils.esc(movie.Popularity || 'N/A')}</span></div>
      <div class="meta-item"><label>Movie ID</label><span>${Utils.esc(movie.movie_id)}</span></div>`;

    // Why Recommended block (shown when this movie came from a recommendation)
    const reasons = (extra && extra.reasons) || movie.Reasons;
    if (this.modalReasonsBox) {
      if (reasons && reasons.reasons && reasons.reasons.length) {
        this.modalReasonsBox.classList.remove('hidden');
        this.modalReasonsBox.innerHTML =
          `<div class="why-kicker">
             <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>
             WHY RECOMMENDED
           </div>
           <ul class="why-list">${reasons.reasons.map(r => `<li>${Utils.esc(r)}</li>`).join('')}</ul>` +
          (reasons.factors && reasons.factors.semantic != null
            ? `<div class="why-sem">Semantic proximity: <b>${reasons.factors.semantic}</b>${reasons.factors.genre_overlap && reasons.factors.genre_overlap.length ? ' · Shared genres: ' + Utils.esc(reasons.factors.genre_overlap.join(', ')) : ''}</div>`
            : '');
      } else {
        this.modalReasonsBox.classList.add('hidden');
        this.modalReasonsBox.innerHTML = '';
      }
    }

    // reset like/dislike + watchlist + compare button states
    this._setModalFeedback(null);
    this._setModalCompareAdded();
    this._setWatchlistBtn(false);
    this._readWatchlistState(movie.movie_id);

    this.modal.classList.remove('hidden');
  },

  _readWatchlistState(movieId) {
    API.watchlistCheck(movieId).then(s => {
      this._setWatchlistBtn(!!(s && s.in_watchlist));
    }).catch(() => {});
  },

  _setWatchlistBtn(state) {
    if (!this.modalWatchlistBtn) return;
    const span = this.modalWatchlistBtn.querySelector('span');
    this.modalWatchlistBtn.classList.toggle('on', !!state);
    if (span) span.textContent = state ? 'In Watchlist' : 'Watchlist';
  },

  _setModalFeedback(action) {
    if (this.modalLikeBtn) this.modalLikeBtn.classList.toggle('on', action === 'like');
    if (this.modalDislikeBtn) this.modalDislikeBtn.classList.toggle('on', action === 'dislike');
  },

  _setModalCompareAdded() {
    if (!this.modalCompareBtn) return;
    const inList = this.modalId && this.compareList.some(c => c.id === this.modalId);
    this.modalCompareBtn.classList.toggle('on', !!inList);
    const span = this.modalCompareBtn.querySelector('span');
    if (span) span.textContent = inList ? 'In Compare' : 'Compare';
  },

  // ---------- Movie modal actions ----------
  async toggleWatchlist() {
    if (!this.modalId) return;
    try {
      const inList = !!(await API.watchlistCheck(this.modalId)).in_watchlist;
      if (inList) {
        await API.watchlistRemove(this.modalId);
        Notifications.show('Removed from watchlist.', 'success');
        this._setWatchlistBtn(false);
        this._updateDockAfterDataChange();
      } else {
        const r = await API.watchlistAdd(this.modalId);
        Notifications.show(r.added ? 'Added to watchlist.' : 'Already in your watchlist.', 'success');
        this._setWatchlistBtn(true);
        this._updateDockAfterDataChange();
      }
    } catch (err) {
      this._showError(err.message);
    }
  },

  async sendFeedback(action) {
    if (!this.modalId) return;
    try {
      await API.feedback(this.modalId, action);
      Notifications.show(
        action === 'like'
          ? 'Liked — saved to your taste profile.'
          : 'Disliked — fewer of these in the future.',
        'success'
      );
      this._setModalFeedback(action);
    } catch (err) {
      this._showError(err.message);
    }
  },

  _updateDockAfterDataChange() {},

  // ---------- Compare ----------
  addToCompare(id, title) {
    if (!id) return;
    if (this.compareList.length >= 3) {
      Notifications.show('Compare supports up to 3 movies.', 'info');
      return;
    }
    if (this.compareList.some(c => c.id === id)) {
      Notifications.show('Already in the compare tray.', 'info');
      return;
    }
    this.compareList.push({ id, title: title || 'Movie' });
    this._renderCompareDock();
    this._setModalCompareAdded();
    Notifications.show('Added to compare. Pick at least one more movie.', 'success');
  },

  _removeCompare(id) {
    this.compareList = this.compareList.filter(c => c.id !== id);
    this._renderCompareDock();
    this._setModalCompareAdded();
  },

  _renderCompareDock() {
    if (!this.compareDock) return;
    const showing = this.compareList.length > 0;
    this.compareDock.classList.toggle('hidden', !showing);
    this.compareChips.innerHTML = this.compareList.map(c =>
      `<span class="compare-chip">${Utils.esc(c.title)}
         <button class="compare-chip-x" data-compare-remove="${Utils.esc(c.id)}" aria-label="Remove">&times;</button>
       </span>`
    ).join('');
    const btn = this.compareActionBtn;
    if (btn) {
      btn.disabled = this.compareList.length < 2;
      btn.textContent = 'Compare (' + this.compareList.length + '/3)';
    }
  },

  async openCompare() {
    if (this.compareList.length < 2) {
      Notifications.show('Add at least 2 movies to compare.', 'info');
      return;
    }
    this._openFeature('Compare Movies');
    const body = this.featureModalBody;
    body.innerHTML = '<div class="feature-loading"><div class="spinner"></div> Comparing movies…</div>';
    try {
      const data = await API.compare(this.compareList.map(c => c.id));
      const results = data.results || [];
      if (!results.length) {
        body.innerHTML = '<div class="empty-note">Nothing to compare.</div>';
        return;
      }
      const pairSim = results[0].Pair_Similarity != null;
      body.innerHTML =
        `<div class="compare-grid cols-${results.length}">` +
        results.map((m, i) => this._compareColHTML(m, i, results.length)).join('') +
        `</div>` +
        (pairSim
          ? `<div class="pair-note">Pairwise cosine similarity (same embedding space): <b>${results[0].Pair_Similarity}</b></div>`
          : '');
    } catch (err) {
      body.innerHTML = `<div class="error-box">${Utils.esc(err.message)}</div>`;
    }
  },

  _compareColHTML(movie, idx, total) {
    const poster = Utils.poster(movie.Poster_Url) || Utils.placeholder(movie.Title);
    const ph = Utils.placeholder(movie.Title).replace(/'/g, "\\'");
    const genres = Utils.genres(movie.Genre);
    const sim = (total === 2 && movie.Pair_Similarity != null)
      ? `<div class="cmp-sim">
           <span class="match-label">Pair similarity</span>
           <span class="match-value">${movie.Pair_Similarity}</span>
         </div>`
      : '';
    const scoreLabel = movie.Final_Score != null ? 'Final_Score' : (movie.Relevance != null ? 'Relevance' : null);
    return `
      <div class="compare-col" data-movie-id="${Utils.esc(movie.movie_id)}">
        <div class="compare-poster-wrap">
          <span class="compare-col-rank">${idx + 1}</span>
          <img src="${poster}" alt="${Utils.esc(movie.Title)}" onerror="this.src='${ph}'"/>
        </div>
        <h4 class="compare-title">${Utils.esc(movie.Title)}</h4>
        <div class="compare-meta">
          <span class="star">★</span> ${Utils.rating(movie.Vote_Average)}
          <span>·</span> ${Utils.year(movie.Release_Date)}
          <span>·</span> ${Utils.esc(genres.join(', ') || 'N/A')}
        </div>
        <p class="compare-overview">${Utils.esc((movie.Overview || 'No overview available.').slice(0, 240))}${(movie.Overview || '').length > 240 ? '…' : ''}</p>
        ${sim}
        <div class="compare-stats">
          <div><span class="match-label">Popularity</span><b>${Utils.esc(movie.Popularity ?? 'N/A')}</b></div>
          <div><span class="match-label">Votes</span><b>${Utils.esc(movie.Vote_Count ?? 'N/A')}</b></div>
          <div><span class="match-label">Language</span><b>${Utils.esc(movie.Original_Language ?? 'N/A')}</b></div>
          ${scoreLabel && movie[scoreLabel] != null ? `<div><span class="match-label">${scoreLabel}</span><b>${movie[scoreLabel]}</b></div>` : ''}
        </div>
      </div>`;
  },

  _closeModal() { this.modal.classList.add('hidden'); },

  // ---------- Feature tool modal ----------
  _openFeature(title) {
    this.featureModalTitle.textContent = title;
    this.featureModalBody.innerHTML = '';
    this.featureModal.classList.remove('hidden');
  },

  _closeFeature() {
    this.featureModal.classList.add('hidden');
    this.featureModalBody.innerHTML = '';
  },

  // generic result card used in feature tool panels
  _featureGridCard(movie) {
    const poster = Utils.poster(movie.Poster_Url) || Utils.placeholder(movie.Title);
    const ph = Utils.placeholder(movie.Title).replace(/'/g, "\\'");
    const genres = Utils.genres(movie.Genre);
    return `
      <div class="feat-item" data-movie-id="${Utils.esc(movie.movie_id)}">
        <div class="feat-poster-wrap">
          <img src="${poster}" alt="${Utils.esc(movie.Title)}" onerror="this.src='${ph}'"/>
        </div>
        <div class="feat-body">
          <div class="feat-title">${Utils.esc(movie.Title)}</div>
          <div class="feat-meta">
            <span class="star">★</span> ${Utils.rating(movie.Vote_Average)}
            <span>·</span> ${Utils.year(movie.Release_Date)}
            ${movie.Relevance != null ? `<span class="relevance">· ${Utils.simPercent(movie.Relevance)}</span>` : ''}
          </div>
        </div>
      </div>`;
  },

  // ---------- Watchlist ----------
  async openWatchlist() {
    const body = this._toolBody();
    body.innerHTML = '<div class="feature-loading"><div class="spinner"></div> Loading watchlist…</div>';
    try {
      const data = await API.watchlist();
      const items = data.results || [];
      if (!items.length) {
        body.innerHTML =
          '<div class="empty-note">Your watchlist is empty. Open a movie and press “Watchlist” to save it here.</div>' +
          '<button class="btn btn-ghost btn-sm" id="wlBackBtn">← Back to movies</button>';
        const bb = document.getElementById('wlBackBtn');
        if (bb) bb.addEventListener('click', () => this._backHome());
        return;
      }
      body.innerHTML =
        `<div class="stat-strip">
           <div class="feature-stat"><b>${items.length}</b><span>Saved movies</span></div>
         </div>
         <div class="feature-grid">` +
        items.map(m => {
          const poster = Utils.poster(m.Poster_Url) || Utils.placeholder(m.Title);
          const ph = Utils.placeholder(m.Title).replace(/'/g, "\\'");
          const genres = Utils.genres(m.Genre);
          return `
            <div class="feat-item wl" data-movie-id="${Utils.esc(m.movie_id)}">
              <div class="feat-poster-wrap">
                <img src="${poster}" alt="${Utils.esc(m.Title)}" onerror="this.src='${ph}'"/>
              </div>
              <div class="feat-body">
                <div class="feat-title">${Utils.esc(m.Title)}</div>
                <div class="feat-meta">
                  <span class="star">★</span> ${Utils.rating(m.Vote_Average)}
                  <span>·</span> ${Utils.year(m.Release_Date)}
                  <span>·</span> ${Utils.esc(genres[0] || 'Film')}
                </div>
                <button class="btn btn-ghost btn-sm" data-remove-watchlist="${Utils.esc(m.movie_id)}">Remove</button>
              </div>
            </div>`;
        }).join('') +
        '</div>';
      body.querySelectorAll('[data-remove-watchlist]').forEach(btn => {
        btn.addEventListener('click', e => {
          e.stopPropagation();
          this.removeFromWatchlist(btn.dataset.removeWatchlist);
        });
      });
    } catch (err) {
      body.innerHTML = `<div class="error-box">${Utils.esc(err.message)}</div>`;
    }
  },

  async removeFromWatchlist(movieId) {
    try {
      await API.watchlistRemove(movieId);
      Notifications.show('Removed from watchlist.', 'success');
      this.openWatchlist();
    } catch (err) {
      this._showError(err.message);
    }
  },

  async _resolveTitles(list) {
    return Promise.all((list || []).map(async item => {
      try {
        const m = await API.movieDetails(item.movie_id);
        return m.Title || String(item.movie_id);
      } catch (_) {
        return String(item.movie_id);
      }
    }));
  },

  // ---------- Preferences ----------
  async openPreferences() {
    const body = this._toolBody();
    body.innerHTML = '<div class="feature-loading"><div class="spinner"></div> Loading preferences…</div>';
    try {
      const p = await API.preferences();
      const stats = p.stats || {};
      const liked = await this._resolveTitles(p.liked_movies || []);
      const disliked = await this._resolveTitles(p.disliked_movies || []);
      const genres = p.preferred_genres || [];
      const themes = p.preferred_themes || [];

      const chipList = (arr, cls) => arr.length
        ? `<div class="feature-chips">${arr.map(x => `<span class="pref-chip ${cls}">${Utils.esc(x)}</span>`).join('')}</div>`
        : '<div class="empty-note">Nothing here yet.</div>';

      body.innerHTML = `
        <div class="stat-strip">
          <div class="feature-stat"><b>${stats.total_likes || 0}</b><span>Likes</span></div>
          <div class="feature-stat"><b>${stats.total_dislikes || 0}</b><span>Dislikes</span></div>
          <div class="feature-stat"><b>${stats.preferred_genre_count || 0}</b><span>Preferred genres</span></div>
        </div>
        <div class="feature-cols">
          <div class="feature-col">
            <div class="feature-block-title">Liked movies</div>
            ${chipList(liked, 'liked')}
          </div>
          <div class="feature-col">
            <div class="feature-block-title">Disliked movies</div>
            ${chipList(disliked, 'disliked')}
          </div>
        </div>
        <div class="feature-block-title">Preferred genres — derived from your likes</div>
        ${chipList(genres, 'genre')}
        <div class="feature-block-title">Preferred themes — derived from your likes</div>
        ${chipList(themes, 'theme')}
        <div class="feature-actions">
          <button class="btn btn-ghost btn-sm" id="prefsResetBtn">Reset preferences</button>
        </div>`;

      const reset = document.getElementById('prefsResetBtn');
      if (reset) reset.addEventListener('click', async () => {
        try {
          await API.prefsReset();
          Notifications.show('Preferences reset.', 'success');
          this.openPreferences();
        } catch (err) { this._showError(err.message); }
      });
    } catch (err) {
      body.innerHTML = `<div class="error-box">${Utils.esc(err.message)}</div>`;
    }
  },

  // ---------- Insights ----------
  async openInsights() {
    const body = this._toolBody();
    body.innerHTML = '<div class="feature-loading"><div class="spinner"></div> Crunching insights…</div>';
    try {
      const ins = await API.insights();
      const genreBars = (list, empty) => {
        if (!list || !list.length) return `<div class="empty-note">${empty}</div>`;
        const max = Math.max(...list.map(g => g.count));
        return `<div class="genre-bars">${list.map(g => `
          <div class="genre-bar-row">
            <span class="genre-bar-label">${Utils.esc(g.genre || g)}</span>
            <span class="genre-bar-track"><i style="width:${max ? Math.round((g.count / max) * 100) : 0}%"></i></span>
            <span class="genre-bar-count">${g.count}</span>
          </div>`).join('')}</div>`;
      };
      body.innerHTML = `
        <div class="stat-strip">
          <div class="feature-stat"><b>${ins.total_searches || 0}</b><span>Searches</span></div>
          <div class="feature-stat"><b>${ins.total_recommendations || 0}</b><span>Recommendations</span></div>
          <div class="feature-stat"><b>${ins.total_likes || 0}</b><span>Likes</span></div>
          <div class="feature-stat"><b>${ins.total_dislikes || 0}</b><span>Dislikes</span></div>
          <div class="feature-stat"><b>${ins.watchlist_count || 0}</b><span>Watchlist</span></div>
        </div>
        <div class="feature-cols">
          <div class="feature-col">
            <div class="feature-block-title">Favourite genres</div>
            ${genreBars(ins.favorite_genres, 'Like movies to grow this list.')}
          </div>
          <div class="feature-col">
            <div class="feature-block-title">Most searched genres</div>
            ${genreBars(ins.most_searched_genres, 'No searches logged yet.')}
          </div>
        </div>
        <div class="feature-block-title">Preferred themes</div>
        ${(ins.preferred_themes || []).length
          ? `<div class="feature-chips">${ins.preferred_themes.map(t => `<span class="pref-chip theme">${Utils.esc(t)}</span>`).join('')}</div>`
          : '<div class="empty-note">Theme signals appear once you rate movies.</div>'}
        <div class="feature-block-title">Recently recommended</div>
        ${(ins.recently_recommended || []).length
          ? `<div class="feature-chips">${ins.recently_recommended.map(t => `<span class="genre-chip">${Utils.esc(t)}</span>`).join('')}</div>`
          : '<div class="empty-note">No recommendations logged yet — run a search to populate history.'}`;
    } catch (err) {
      body.innerHTML = `<div class="error-box">${Utils.esc(err.message)}</div>`;
    }
  },

  // ---------- Recommendation History ----------
  async openHistory() {
    const body = this._toolBody();
    body.innerHTML = '<div class="feature-loading"><div class="spinner"></div> Loading history…</div>';
    try {
      const ins = await API.insights();
      const activity = ins.recent_activity || [];
      const typeLabel = {
        search: 'Search', recommend: 'Recommendation',
        whatif: 'What-If', feedback: 'Feedback', discovery: 'Discovery', preferences: 'Preferences',
      };
      body.innerHTML =
        `<div class="stat-strip">
           <div class="feature-stat"><b>${ins.history_count || 0}</b><span>Recommendations logged</span></div>
           <div class="feature-stat"><b>${activity.length}</b><span>Recent events</span></div>
         </div>` +
        (activity.length
          ? `<div class="timeline">${activity.map(a => {
              const t = a.timestamp ? a.timestamp.replace('T', ' ').slice(0, 16) : '';
              return `
                <div class="timeline-item">
                  <span class="timeline-dot"></span>
                  <div class="timeline-body">
                    <span class="timeline-type">${Utils.esc(typeLabel[a.type] || a.type)}</span>
                    <span class="timeline-label">${Utils.esc(a.label || '—')}</span>
                    <span class="timeline-time">${Utils.esc(t)}</span>
                  </div>
                </div>`;
            }).join('')}</div>`
          : '<div class="empty-note">No activity yet — search a movie to get started.</div>');
    } catch (err) {
      body.innerHTML = `<div class="error-box">${Utils.esc(err.message)}</div>`;
    }
  },

  // ---------- AI Discovery ----------
  openDiscovery() {
    const body = this._toolBody();
    body.innerHTML = `
      <p class="feature-hint">Describe what you feel like watching in plain language. Every result is grounded in the real local catalogue — nothing is invented.</p>
      <form id="discoveryForm" class="feature-form">
        <input id="discoveryInput" type="text" placeholder="e.g. emotional sci-fi movies for a rainy night" autocomplete="off" />
        <button type="submit" class="btn btn-primary">Discover</button>
      </form>
      <div id="discoveryResults" class="movie-grid-feature"></div>`;
    const form = document.getElementById('discoveryForm');
    form.addEventListener('submit', async e => {
      e.preventDefault();
      const q = document.getElementById('discoveryInput').value.trim();
      if (!q) { Notifications.show('Describe what you want to watch.', 'info'); return; }
      const wrap = document.getElementById('discoveryResults');
      wrap.innerHTML = '<div class="feature-loading"><div class="spinner"></div> Discovering…</div>';
      try {
        const data = await API.aiDiscovery(q);
        const results = data.results || [];
        if (!results.length) {
          wrap.innerHTML = '<div class="empty-note">Nothing matched that description. Try different words.</div>';
          return;
        }
        wrap.className = 'movie-grid';
        wrap.innerHTML = results.map(m => this._featureGridCard(m)).join('');
      } catch (err) {
        wrap.className = 'movie-grid-feature';
        wrap.innerHTML = `<div class="error-box">${Utils.esc(err.message)}</div>`;
      }
    });
    setTimeout(() => { const i = document.getElementById('discoveryInput'); if (i) i.focus(); }, 60);
  },

  // ---------- What-If Recommendation ----------
  openWhatIf() {
    const body = this._toolBody();
    const dims = [
      ['action', 'Action'], ['romance', 'Romance'], ['emotion', 'Emotion'],
      ['scifi', 'Sci-Fi'], ['comedy', 'Comedy'], ['seriousness', 'Seriousness'],
    ];
    body.innerHTML = `
      <p class="feature-hint">Seed from a movie you like, then shift the attribute sliders to see how the FAISS neighbourhood re-ranks. You can also type a phrase like “like Interstellar but more emotional”.</p>
      <form id="whatIfForm" class="feature-form">
        <div class="whatif-grid">
          <div class="whatif-left">
            <label class="field-label" for="whatifMovie">Seed movie</label>
            <input id="whatifMovie" type="text" placeholder="e.g. Interstellar" autocomplete="off" />
            <label class="field-label" for="whatifPhrase">Natural-language tweak (optional)</label>
            <input id="whatifPhrase" type="text" placeholder="e.g. like Inception but more action" autocomplete="off" />
            <label class="field-label" for="whatifN">How many results?</label>
            <select id="whatifN" class="field-select">${[4, 6, 8, 10].map(n => `<option value="${n}"${n === 8 ? ' selected' : ''}>${n}</option>`).join('')}</select>
          </div>
          <div class="whatif-right">
            ${dims.map(([k, label]) => `
              <div class="slider-row">
                <label>${label}</label>
                <input type="range" min="0" max="100" value="0" data-dim="${k}" />
                <span class="slider-val" id="whatifVal-${k}">0</span>
              </div>`).join('')}
          </div>
        </div>
        <button type="submit" class="btn btn-primary">Generate What-If</button>
      </form>
      <div id="whatifResults" class="feature-section"></div>`;

    body.querySelectorAll('input[type=range]').forEach(r => {
      r.addEventListener('input', () => {
        const v = document.getElementById('whatifVal-' + r.dataset.dim);
        if (v) v.textContent = r.value;
      });
    });

    const form = document.getElementById('whatIfForm');
    form.addEventListener('submit', async e => {
      e.preventDefault();
      const movie = document.getElementById('whatifMovie').value.trim();
      const phrase = document.getElementById('whatifPhrase').value.trim();
      if (!movie && !phrase) { Notifications.show('Enter a seed movie or a phrase.', 'info'); return; }
      const payload = { movie, n: Number(document.getElementById('whatifN').value), phrase: phrase || null };
      body.querySelectorAll('input[type=range]').forEach(r => { payload[r.dataset.dim] = Number(r.value); });
      const wrap = document.getElementById('whatifResults');
      wrap.innerHTML = '<div class="feature-loading"><div class="spinner"></div> Re-ranking the neighbourhood…</div>';
      try {
        const data = await API.whatIf(payload);
        wrap.innerHTML = this._whatIfResultsHTML(data);
        this._reasonsCache = {};
        (data.recommendations || []).forEach(m => { if (m.Reasons) this._reasonsCache[m.movie_id] = m.Reasons; });
      } catch (err) {
        wrap.innerHTML = `<div class="error-box">${Utils.esc(err.message)}</div>`;
      }
    });
  },

  _whatIfResultsHTML(data) {
    const sel = data.selected_movie;
    const recs = data.recommendations || [];
    const w = data.weights || {};
    const active = Object.keys(w).filter(k => Number(w[k]) > 0);
    const weightsNote = active.length
      ? `<div class="feature-chips">${active.map(k => `<span class="genre-chip">${Utils.esc(k)} ${w[k]}</span>`).join('')}</div>`
      : '';
    return `
      <div class="section-head">
        <h3 class="section-title"><span class="title-glow">What-If Results</span></h3>
        <span class="section-note">re-ranked by attribute weights</span>
      </div>
      ${sel ? `<div class="feature-note">Seeded from <b>${Utils.esc(sel.Title)}</b></div>` : ''}
      ${weightsNote}
      <div class="rec-grid">${recs.map((m, i) => this._recCardHTML(m, i + 1)).join('') || '<p class="empty-note">No results returned.</p>'}</div>`;
  },

  // ---------- Chatbot ----------
  _toggleChat(open) {
    if (open) {
      this.chatPanel.classList.add('open');
      this.chatLauncher.classList.add('open');
      setTimeout(() => this.chatInput.focus(), 300);
    } else {
      this.chatPanel.classList.remove('open');
      this.chatLauncher.classList.remove('open');
    }
  },

  _welcomeChat() {
    this._addChatMessage('bot',
      "Hi! I'm CineMind 🎬 Ask me things like " +
      "'Suggest 5 movies like Interstellar' and I'll use the local " +
      "recommendation engine to find similar films."
    );
  },

  _addChatMessage(role, text) {
    const el = document.createElement('div');
    el.className = 'msg ' + role;
    el.textContent = text;
    this.chatMessages.appendChild(el);
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    return el;
  },

  _addTyping() {
    const el = document.createElement('div');
    el.className = 'msg bot typing';
    el.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    this.chatMessages.appendChild(el);
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    return el;
  },

  _addChatMovies(movies) {
    if (!movies || movies.length === 0) return;
    const wrap = document.createElement('div');
    wrap.className = 'chat-movies';
    movies.forEach(m => {
      const poster = Utils.poster(m.Poster_Url) || Utils.placeholder(m.Title);
      const ph = Utils.placeholder(m.Title).replace(/'/g, "\\'");
      const card = document.createElement('div');
      card.className = 'chat-movie';
      card.dataset.movieId = m.movie_id;
      card.innerHTML =
        `<img src="${poster}" alt="${Utils.esc(m.Title)}" onerror="this.src='${ph}'" />` +
        `<div class="cm-title">${Utils.esc(m.Title)}</div>`;
      wrap.appendChild(card);
    });
    this.chatMessages.appendChild(wrap);
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  },

  async sendChat() {
    const msg = this.chatInput.value.trim();
    if (!msg) return;
    this.chatInput.value = '';
    this._addChatMessage('user', msg);
    const typing = this._addTyping();
    try {
      const data = await API.chat(msg);
      typing.remove();
      this._addChatMessage('bot', data.reply || '(no reply)');
      if (data.recommendations && data.recommendations.length > 0) {
        this._addChatMessage('bot', 'Here are the movies I found:');
        this._addChatMovies(data.recommendations);
      }
    } catch (err) {
      typing.remove();
      this._addChatMessage('bot', 'Sorry, something went wrong: ' + err.message);
    }
  },

  // ---------- Home page rows (carousel + trending + top rated + curated) ----------
  // Backend returns sorted rows (sort_by=popularity / rating / newest / gems /
  // favorites); the frontend only applies presentation filtering (min votes).
  async _loadHomeRows() {
    try {
      const [trendingRes, ratedRes, newestRes, gemsRes, worldRes, favRes] = await Promise.all([
        API.movies(14, 'popularity'),
        API.movies(100, 'rating'),
        API.movies(14, 'newest'),
        API.movies(14, 'gems', { minVotes: 150 }),
        API.movies(14, 'popularity', { notLang: 'en' }),
        API.movies(14, 'favorites'),
      ]);
      const trending = trendingRes.results || [];
      if (trending.length === 0) {
        document.getElementById('welcomeSection').classList.remove('hidden');
        return;
      }
      const ratedAll = ratedRes.results || [];
      // Top Rated = min 500 votes for quality (globally highest rated)
      const topRated = ratedAll
        .filter(m => (Number(m.Vote_Count) || 0) >= 500)
        .slice(0, 14);

      // Carousel = top 10 by popularity (auto-scrolling)
      this._renderCarousel(trending.slice(0, 10));
      // Trending row
      this._renderRow('trendingRow', trending.slice(0, 14), 'trendingCount');
      document.getElementById('trendingSection').classList.remove('hidden');
      // Top Rated row (Netflix-style big rank numbers)
      if (topRated.length) {
        this._renderRow('topRatedRow', topRated.slice(0, 14), 'topRatedCount', true);
        document.getElementById('topRatedSection').classList.remove('hidden');
      }

      // New Releases — most recent films in the catalogue
      this._renderRowHelper('newReleasesRow', 'newReleasesSection', 'newReleasesCount',
        (newestRes.results || []).slice(0, 14));
      // Hidden Gems — highly rated yet low popularity (min 150 votes)
      this._renderRowHelper('gemsRow', 'gemsSection', 'gemsCount',
        (gemsRes.results || []).slice(0, 14));
      // Around the World — popular non-English films
      this._renderRowHelper('worldRow', 'worldSection', 'worldCount',
        (worldRes.results || []).slice(0, 14));
      // Fan Favorites — the most-voted movies
      this._renderRowHelper('favoritesRow', 'favoritesSection', 'favoritesCount',
        (favRes.results || []).slice(0, 14));

      // Personalized row + dynamic genre rows (best-effort, never block)
      this._loadForYouRow();
      this._loadGenreRows();
    } catch (err) {
      console.warn('Home rows failed to load:', err);
      document.getElementById('welcomeSection').classList.remove('hidden');
    }
  },

  // Render one curated home row and reveal its section only if it has cards.
  _renderRowHelper(rowId, sectionId, countId, movies) {
    if (!movies || movies.length === 0) return;
    this._renderRow(rowId, movies, countId);
    const section = document.getElementById(sectionId);
    if (section) section.classList.remove('hidden');
  },

  // "Made for You" — personalized row built from preferred genres (hidden
  // until the user has liked at least one movie).
  async _loadForYouRow() {
    const section = document.getElementById('forYouSection');
    if (!section) return;
    try {
      const prefs = await API.preferences();
      const topGenres = (prefs.preferred_genres || []).slice(0, 2);
      if (!topGenres.length) return;
      const seen = {};
      const results = [];
      for (const g of topGenres) {
        const res = await API.movies(12, 'popularity', { genre: g, minVotes: 50 });
        (res.results || []).forEach(m => {
          const k = String(m.movie_id);
          if (!seen[k]) { seen[k] = true; results.push(m); }
        });
        if (results.length >= 14) break;
      }
      if (!results.length) return;
      this._renderRow('forYouRow', results.slice(0, 14), 'forYouCount');
      section.classList.remove('hidden');
    } catch (err) {
      console.warn('For You row failed to load:', err);
    }
  },

  // Dynamic genre rows — a short rotating set so the homepage always feels
  // fresh, while every pick stays grounded in the local catalogue.
  _genreRowConfig: [
    { term: 'Science Fiction', icon: '🛸', label: 'Sci-Fi & Beyond' },
    { term: 'Action',          icon: '🥋', label: 'Action Packed' },
    { term: 'Romance',         icon: '💞', label: 'Love Stories' },
    { term: 'Animation',       icon: '🎨', label: 'Animation & Family' },
  ],

  async _loadGenreRows() {
    const container = document.getElementById('genreRowsContainer');
    if (!container) return;
    // Rotate a 3-slice window every day so the mix changes without reloading.
    const dayIndex = Math.floor(Date.now() / 86400000) % this._genreRowConfig.length;
    const picks = [0, 1, 2].map(i =>
      this._genreRowConfig[(dayIndex + i) % this._genreRowConfig.length]
    );
    const html = [];
    for (const cfg of picks) {
      try {
        const res = await API.movies(14, 'popularity', { genre: cfg.term, minVotes: 100 });
        const movies = (res.results || []).slice(0, 14);
        if (movies.length < 4) continue;
        html.push(this._genreRowHTML(cfg, movies));
      } catch (err) {
        console.warn('Genre row failed:', cfg.term, err);
      }
    }
    if (!html.length) return;
    container.innerHTML = html.join('');
    container.classList.remove('hidden');
  },

  _genreRowHTML(cfg, movies) {
    const cards = movies.map(m => {
      const poster = Utils.poster(m.Poster_Url) || Utils.placeholder(m.Title);
      const ph = Utils.placeholder(m.Title).replace(/'/g, "\\'");
      return `
        <div class="row-card" data-movie-id="${Utils.esc(m.movie_id)}">
          <div class="poster-wrap">
            <img src="${poster}" alt="${Utils.esc(m.Title)}" onerror="this.src='${ph}'" />
          </div>
          <div class="row-card-body">
            <div class="row-card-title">${Utils.esc(m.Title)}</div>
            <div class="row-card-meta">
              <span class="star">★</span> ${Utils.rating(m.Vote_Average)}
              <span>·</span>
              <span>${Utils.year(m.Release_Date)}</span>
            </div>
          </div>
        </div>`;
    }).join('');
    return `
      <section class="row-section">
        <div class="row-header">
          <h3 class="row-title"><span class="row-icon">${cfg.icon}</span> ${Utils.esc(cfg.label)}</h3>
          <span class="row-sub">curated from local catalogue</span>
        </div>
        <div class="row-scroll">${cards}</div>
      </section>`;
  },

  // Hero stats (movie count) come from the backend.
  async _loadStats() {
    try {
      const info = await API.apiInfo();
      const movies = Number(info.movies_indexed);
      const moviesEl = document.getElementById('statMovies');
      const welcomeCount = document.getElementById('welcomeCount');
      if (moviesEl && !isNaN(movies)) {
        moviesEl.textContent = movies.toLocaleString();
      }
      if (welcomeCount && !isNaN(movies)) {
        welcomeCount.textContent = movies.toLocaleString();
      }
    } catch (err) {
      console.warn('Stats failed to load, keeping defaults:', err);
    }
  },

  // Carousel = auto-scrolling top movies (duplicate the list for seamless loop)
  _renderCarousel(movies) {
    const track = document.getElementById('carouselTrack');
    const doubled = [...movies, ...movies]; // duplicate for infinite scroll
    track.innerHTML = doubled.map((m, i) => {
      const poster = Utils.poster(m.Poster_Url) || Utils.placeholder(m.Title);
      const ph = Utils.placeholder(m.Title).replace(/'/g, "\\'");
      const rank = i % movies.length + 1;
      return `
        <div class="carousel-card" data-movie-id="${m.movie_id}">
          <img src="${poster}" alt="${Utils.esc(m.Title)}" onerror="this.src='${ph}'" />
          <div class="carousel-overlay">
            <div class="carousel-title">${Utils.esc(m.Title)}</div>
            <div class="carousel-meta">★ ${Utils.rating(m.Vote_Average)} · ${Utils.year(m.Release_Date)}</div>
          </div>
          <span class="carousel-rank">#${rank}</span>
        </div>`;
    }).join('');
  },

  // Horizontal scroll row (Trending / Top Rated)
  _renderRow(containerId, movies, countId, bigRank) {
    const container = document.getElementById(containerId);
    const countEl = document.getElementById(countId);
    if (countEl) countEl.textContent = '(' + movies.length + ')';
    container.innerHTML = movies.map((m, i) => {
      const poster = Utils.poster(m.Poster_Url) || Utils.placeholder(m.Title);
      const ph = Utils.placeholder(m.Title).replace(/'/g, "\\'");
      const rankBadge = bigRank
        ? '<span class="row-rank" style="display:none"></span>'
        : `<span class="row-rank">${i + 1}</span>`;
      return `
        <div class="row-card${bigRank ? ' big-rank' : ''}" data-movie-id="${m.movie_id}"${bigRank ? ` data-rank="${i + 1}"` : ''}>
          <div class="poster-wrap">
            <img src="${poster}" alt="${Utils.esc(m.Title)}" onerror="this.src='${ph}'" />
            ${rankBadge}
          </div>
          <div class="row-card-body">
            <div class="row-card-title">${Utils.esc(m.Title)}</div>
            <div class="row-card-meta">
              <span class="star">★</span> ${Utils.rating(m.Vote_Average)}
              <span>·</span>
              <span>${Utils.year(m.Release_Date)}</span>
            </div>
          </div>
        </div>`;
    }).join('');
  },
};
