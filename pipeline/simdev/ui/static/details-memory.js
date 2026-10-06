// Collapsible sections remember whether they were open, per browser.
//
// Any <details data-remember="key"> takes part: the saved state wins over the
// template's default, and every toggle is saved. Storage can be unavailable
// (private windows), so it only ever falls back to the default.
(function () {
  const KEY = "simdev.open.v1";

  function load() {
    try {
      const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
      return saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {};
    } catch (e) {
      return {};
    }
  }

  function save(saved) {
    try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) { /* not remembered */ }
  }

  function apply(root) {
    const saved = load();
    root.querySelectorAll("details[data-remember]").forEach((el) => {
      const key = el.dataset.remember;
      if (typeof saved[key] === "boolean") el.open = saved[key];
      if (el.dataset.rememberBound) return;
      el.dataset.rememberBound = "1";
      el.addEventListener("toggle", () => {
        const now = load();
        now[key] = el.open;
        save(now);
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => apply(document));
  // Sections that arrive later through htmx swaps take part too.
  document.addEventListener("htmx:afterSwap", (event) => apply(event.target));
})();
