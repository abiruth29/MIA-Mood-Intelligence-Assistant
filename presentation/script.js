/* ═══════════════════════════════════════════════════════════
   MIA PRESENTATION — SLIDE ENGINE
   Keyboard / click navigation + progress bar
   ═══════════════════════════════════════════════════════════ */

(function () {
  const slides = document.querySelectorAll('.slide');
  const total = slides.length;
  let current = 1;

  function goTo(n) {
    if (n < 1 || n > total || n === current) return;

    const prev = document.querySelector('.slide.active');
    const next = document.querySelector(`[data-slide="${n}"]`);

    if (prev) {
      prev.classList.remove('active');
      prev.classList.add(n > current ? 'exit-left' : '');
      setTimeout(() => prev.classList.remove('exit-left'), 550);
    }

    if (next) {
      next.classList.add('active');
    }

    current = n;
    updateUI();
  }

  function nextSlide()  { goTo(current + 1); }
  function prevSlide()  { goTo(current - 1); }

  function updateUI() {
    // Counter
    const counter = document.getElementById('slideCounter');
    if (counter) counter.textContent = `${current} / ${total}`;

    // Progress bar
    const bar = document.getElementById('progressBar');
    if (bar) bar.style.width = `${(current / total) * 100}%`;
  }

  // Keyboard navigation
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
      e.preventDefault();
      nextSlide();
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      prevSlide();
    } else if (e.key === 'Home') {
      e.preventDefault();
      goTo(1);
    } else if (e.key === 'End') {
      e.preventDefault();
      goTo(total);
    }
  });

  // Touch swipe support
  let touchStartX = 0;
  let touchStartY = 0;

  document.addEventListener('touchstart', function (e) {
    touchStartX = e.changedTouches[0].screenX;
    touchStartY = e.changedTouches[0].screenY;
  }, { passive: true });

  document.addEventListener('touchend', function (e) {
    const dx = e.changedTouches[0].screenX - touchStartX;
    const dy = e.changedTouches[0].screenY - touchStartY;

    if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 50) {
      if (dx < 0) nextSlide();
      else prevSlide();
    }
  }, { passive: true });

  // Expose for nav buttons
  window.nextSlide = nextSlide;
  window.prevSlide = prevSlide;

  // Initialize
  updateUI();
})();
