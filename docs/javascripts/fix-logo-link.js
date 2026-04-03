// Fix the header logo link to go to Quickstart instead of root.
// Root URL serves the landing page (not a MkDocs page), which breaks
// Material's instant navigation when loaded inside the docs frame.
(function() {
  function fix() {
    var logos = document.querySelectorAll('a.md-header__button[href="."]');
    logos.forEach(function(a) {
      a.href = 'QUICKSTART/';
    });
    // Also fix the mobile drawer logo
    var navLogos = document.querySelectorAll('.md-nav--primary > .md-nav__title[href="."]');
    navLogos.forEach(function(a) {
      if (a.tagName === 'A' || a.href) a.href = 'QUICKSTART/';
    });
  }
  document.addEventListener('DOMContentLoaded', fix);
  // Re-fix after instant navigation
  if (typeof document$ !== 'undefined') {
    document$.subscribe(fix);
  }
})();
