/* markdown-fx.js — Auto-apply glitch effect to .zyx-glitch spans */

(function () {
  function applyGlitchEffect() {
    // Find all elements with the zyx-glitch class
    const glitchElements = document.querySelectorAll('.zyx-glitch');
    
    glitchElements.forEach(el => {
      // Only process if not already processed
      if (!el.hasAttribute('data-text')) {
        const text = el.textContent;
        el.setAttribute('data-text', text);
      }
    });
  }

  // Initial load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyGlitchEffect);
  } else {
    applyGlitchEffect();
  }

  // Re-run after Material's instant navigation
  if (window.document$ && typeof window.document$.subscribe === 'function') {
    window.document$.subscribe(() => {
      applyGlitchEffect();
    });
  }
})();
