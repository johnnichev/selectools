// Copy as Markdown — for pasting doc pages into AI coding assistants
(function() {
  // Wait for Material to finish rendering
  document.addEventListener('DOMContentLoaded', init);
  // Re-init on instant navigation
  if (typeof document$ !== 'undefined') {
    document$.subscribe(init);
  }

  function init() {
    // Don't add on the landing page
    if (!document.querySelector('.md-content__inner')) return;
    // Don't add duplicates
    if (document.querySelector('.copy-md-inserted')) return;

    // Find the page source path from the canonical URL or page path
    var path = window.location.pathname;
    // Strip /selectools/ prefix and trailing /
    var docPath = path.replace(/^\/selectools\//, '').replace(/\/$/, '');
    // Convert URL path to docs/ source path
    if (!docPath || docPath === 'index.html') docPath = 'index';
    var mdPath = 'docs/' + docPath + '.md';
    // Handle /modules/AGENT/ -> docs/modules/AGENT.md
    if (docPath.endsWith('/index')) {
      mdPath = 'docs/' + docPath.replace(/\/index$/, '') + '.md';
    }

    var rawUrl = 'https://raw.githubusercontent.com/johnnichev/selectools/main/' + mdPath;

    // Insert button next to the page title (h1)
    var h1 = document.querySelector('.md-content__inner h1');
    if (!h1) return;

    var btn = document.createElement('button');
    btn.className = 'copy-md-inserted';
    btn.textContent = 'Copy Markdown';
    btn.title = 'Copy this page as Markdown for your AI assistant (Claude, Cursor, Copilot)';
    btn.style.cssText = 'float:right;font-size:12px;font-weight:500;padding:5px 12px;border-radius:6px;border:1px solid var(--md-default-fg-color--lightest);background:transparent;color:var(--md-default-fg-color--light);cursor:pointer;margin-top:8px;font-family:inherit;transition:all 0.15s;';

    btn.addEventListener('mouseenter', function() {
      btn.style.borderColor = 'var(--md-accent-fg-color)';
      btn.style.color = 'var(--md-accent-fg-color)';
    });
    btn.addEventListener('mouseleave', function() {
      btn.style.borderColor = 'var(--md-default-fg-color--lightest)';
      btn.style.color = 'var(--md-default-fg-color--light)';
    });

    btn.addEventListener('click', async function() {
      try {
        var resp = await fetch(rawUrl);
        if (!resp.ok) throw new Error(resp.status);
        var md = await resp.text();
        await navigator.clipboard.writeText(md);
        btn.textContent = 'Copied!';
        btn.style.color = '#22c55e';
        btn.style.borderColor = '#22c55e';
      } catch (e) {
        // Fallback: copy rendered text
        var content = document.querySelector('.md-content__inner');
        if (content) {
          await navigator.clipboard.writeText(content.innerText);
          btn.textContent = 'Copied (text)';
        }
      }
      setTimeout(function() {
        btn.textContent = 'Copy Markdown';
        btn.style.color = '';
        btn.style.borderColor = '';
      }, 2000);
    });

    h1.insertBefore(btn, h1.firstChild);
  }
})();
