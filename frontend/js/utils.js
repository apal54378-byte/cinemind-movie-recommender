/* ============================================================
   CineMind — Utility helpers
   ============================================================ */

const Utils = {
  // Get year from a date string like "2021-12-15" -> "2021"
  year(dateStr) {
    if (!dateStr) return 'N/A';
    const s = String(dateStr);
    return s.length >= 4 ? s.slice(0, 4) : s;
  },

  // Format rating with 1 decimal
  rating(val) {
    const n = Number(val);
    return isNaN(n) ? 'N/A' : n.toFixed(1);
  },

  // Format similarity score (0..1) as a percentage
  simPercent(score) {
    const n = Number(score);
    if (isNaN(n)) return '';
    return Math.round(n * 100) + '%';
  },

  // Get full poster URL (backend Poster_Url is already a full URL)
  poster(url) {
    if (!url || url === 'nan' || url === 'None') return '';
    return url;
  },

  // Parse genres (comma-separated string -> array)
  genres(val) {
    if (!val) return [];
    if (Array.isArray(val)) return val;
    return String(val).split(',').map(g => g.trim()).filter(Boolean);
  },

  // Escape HTML to avoid breaking layout with weird titles
  esc(str) {
    if (str == null) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  },

  // Placeholder poster (data URI) when Poster_Url is missing
  placeholder(title) {
    const t = (title || 'Film').slice(0, 18);
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='200' height='300'>
      <rect width='200' height='300' fill='#1a1a2e'/>
      <text x='100' y='150' text-anchor='middle' fill='#6d5efc' font-family='sans-serif' font-size='16' font-weight='bold'>${t}</text>
    </svg>`;
    return 'data:image/svg+xml;utf8,' + encodeURIComponent(svg);
  },
};
