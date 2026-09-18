/* ============================================================
   CineMind — Toast Notifications
   ============================================================ */
const Notifications = (() => {
  let container = null;

  function ensureContainer() {
    if (container) return container;
    container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText =
      'position:fixed;bottom:20px;left:50%;transform:translateX(-50%);' +
      'z-index:9999;display:flex;flex-direction:column;gap:10px;align-items:center;';
    document.body.appendChild(container);
    return container;
  }

  function show(message, type = 'info', duration = 3500) {
    const el = document.createElement('div');
    const colors = {
      info:    'rgba(109,94,252,0.95)',
      success: 'rgba(34,197,94,0.95)',
      error:   'rgba(239,68,68,0.95)',
    };
    el.style.cssText =
      `padding:12px 20px;border-radius:10px;color:#fff;font-size:13px;` +
      `font-family:'Poppins',sans-serif;font-weight:500;` +
      `background:${colors[type] || colors.info};box-shadow:0 8px 24px rgba(0,0,0,0.3);` +
      `opacity:0;transform:translateY(10px);transition:all 0.3s;max-width:90vw;`;
    el.textContent = message;
    ensureContainer().appendChild(el);
    requestAnimationFrame(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    });
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(10px)';
      setTimeout(() => el.remove(), 300);
    }, duration);
  }

  return { show };
})();
