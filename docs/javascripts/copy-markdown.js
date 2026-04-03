// Copy Markdown button — lets users paste doc pages into their IDE agent
document.addEventListener('DOMContentLoaded', function() {
  // Only add on module/doc pages, not the landing page
  if (!document.querySelector('.md-content')) return;

  var btn = document.createElement('button');
  btn.className = 'copy-md-btn';
  btn.textContent = 'Copy as Markdown';
  btn.title = 'Copy this page as Markdown for your AI coding assistant';
  btn.addEventListener('click', async function() {
    // Get the edit URL to find the source file
    var editLink = document.querySelector('a[title="Edit this page"]');
    if (!editLink) {
      btn.textContent = 'No source found';
      setTimeout(function() { btn.textContent = 'Copy as Markdown'; }, 2000);
      return;
    }
    var editUrl = editLink.href;
    // Convert edit URL to raw URL
    var rawUrl = editUrl
      .replace('github.com', 'raw.githubusercontent.com')
      .replace('/edit/', '/');
    try {
      var resp = await fetch(rawUrl);
      if (!resp.ok) throw new Error('Failed to fetch');
      var md = await resp.text();
      await navigator.clipboard.writeText(md);
      btn.textContent = 'Copied!';
      setTimeout(function() { btn.textContent = 'Copy as Markdown'; }, 2000);
    } catch (e) {
      // Fallback: copy the rendered text content
      var content = document.querySelector('.md-content');
      if (content) {
        await navigator.clipboard.writeText(content.textContent);
        btn.textContent = 'Copied (text)';
        setTimeout(function() { btn.textContent = 'Copy as Markdown'; }, 2000);
      }
    }
  });
  document.body.appendChild(btn);
});
