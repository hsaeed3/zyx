/* title-fx.js — only activate on the Introduction (index.md) page */

(function () {
    // Returns true if we’re on the site “home” (Introduction) page
    function isHome() {
      try {
        // Grab the site root URL from the logo link (robust across bases/subpaths)
        const logo = document.querySelector('a.md-header__button.md-logo, .md-nav__button.md-logo');
        if (!logo) return false;
  
        const rootHref = new URL(logo.getAttribute('href'), location.href);
        const here = new URL(location.href);
  
        // Normalize both to origin + pathname (ignore hash/query)
        const norm = (u) => (u.origin + u.pathname).replace(/\/+$/, '/');
        return norm(here) === norm(rootHref);
      } catch {
        // Fallback: if there’s a canonical link, require it to end with a single '/'
        const can = document.querySelector('link[rel="canonical"]');
        return !!can && /\/$/.test(can.href);
      }
    }
  
    function applyTitleFX() {
      if (!isHome()) return; // ← only on Introduction
  
      const article = document.querySelector(".md-content__inner");
      if (!article) return;
  
      const h1 = article.querySelector("h1");
      if (!h1) return;
  
      if (h1.classList.contains("zyx-hero-ready")) return;
  
      const html =
        `<strong><span class="title-chunk c0" data-key="0">zyx</span></strong>` +
        ` <strong>|</strong> ` +
        `a <span class="title-chunk c1" data-key="1">fun</span> ` +
        `& <span class="title-chunk c2" data-key="2">expressive sdk</span> ` +
        `for <span class="title-chunk c3" data-key="3">context engineering</span>`;
  
      h1.classList.add("zyx-hero", "zyx-hero-ready");
      h1.innerHTML = html;
    }
  
    // Initial load
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", applyTitleFX);
    } else {
      applyTitleFX();
    }
  
    // Re-run after Material’s instant navigation
    if (window.document$ && typeof window.document$.subscribe === "function") {
      window.document$.subscribe(() => {
        applyTitleFX();
      });
    }
  })();