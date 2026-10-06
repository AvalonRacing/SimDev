// pipeline/simdev/ui/static/results.js
// The results table: adjustable column widths and the column tick boxes, both kept per
// browser. Hidden columns are one CSS rule per column class (c-<key>), so rows swapped in
// by htmx after an inline edit are hidden without any extra work.
(function (root) {
  const WIDTHS_KEY = "simdev.results.widths.v1";
  const COLUMNS_KEY = "simdev.results.columns.v1";
  const MIN_WIDTH = 40;

  // --- pure helpers (exercised from node) ------------------------------------

  function clampWidth(w) {
    return Math.max(MIN_WIDTH, Math.round(w));
  }

  // Stored widths win over the defaults, but only sane ones for columns that still exist.
  function widthsFrom(stored, defaults) {
    const out = Object.assign({}, defaults);
    if (stored && typeof stored === "object") {
      for (const key of Object.keys(defaults)) {
        if (Number.isFinite(stored[key])) out[key] = clampWidth(stored[key]);
      }
    }
    return out;
  }

  // The stored value is the list of HIDDEN keys, so a column that appears later (a new
  // group) shows up. Keys that cannot be hidden or no longer exist are dropped.
  function hiddenFrom(stored, defaultHidden, optionalKeys) {
    const list = Array.isArray(stored) ? stored : defaultHidden;
    return new Set(list.filter((k) => optionalKeys.includes(k)));
  }

  function tableWidth(keys, widths, hidden) {
    return keys.reduce((sum, k) => sum + (hidden.has(k) ? 0 : widths[k]), 0);
  }

  function tsvHref(visibleKeys) {
    return `/results.tsv?cols=${visibleKeys.map(encodeURIComponent).join(",")}`;
  }

  function hideRules(hidden, escape) {
    return [...hidden].map((k) => `.c-${escape(k)} { display: none; }`).join("\n");
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { clampWidth, widthsFrom, hiddenFrom, tableWidth, tsvHref, hideRules };
  }
  if (typeof document === "undefined") return;

  // --- the page --------------------------------------------------------------

  const table = document.querySelector("table.results");
  if (!table) return;
  const cols = [...table.querySelectorAll("col[data-key]")];
  const keys = cols.map((c) => c.dataset.key);
  const defaults = {};
  cols.forEach((c) => { defaults[c.dataset.key] = Number(c.dataset.width); });
  const boxes = [...document.querySelectorAll("#column-picker input[data-key]")];
  const optional = boxes.map((b) => b.dataset.key);
  const defaultHidden = boxes.filter((b) => b.dataset.default === "0").map((b) => b.dataset.key);

  function load(key) {
    try { return JSON.parse(localStorage.getItem(key)); } catch (e) { return null; }
  }
  function save(key, value) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, JSON.stringify(value));
    } catch (e) { /* no storage: this visit only */ }
  }

  let widths = widthsFrom(load(WIDTHS_KEY), defaults);
  let hidden = hiddenFrom(load(COLUMNS_KEY), defaultHidden, optional);
  const style = document.createElement("style");
  document.head.append(style);
  const tsvLink = document.getElementById("tsv-link");

  function apply() {
    style.textContent = hideRules(hidden, CSS.escape);
    cols.forEach((c) => { c.style.width = `${widths[c.dataset.key]}px`; });
    table.style.width = `${tableWidth(keys, widths, hidden)}px`;
    boxes.forEach((b) => { b.checked = !hidden.has(b.dataset.key); });
    // Only the ticked columns are exported; the run, design and reference always are.
    if (tsvLink) tsvLink.href = tsvHref(optional.filter((k) => !hidden.has(k)));
  }

  function saveWidths() {
    const changed = {};
    for (const k of keys) if (widths[k] !== defaults[k]) changed[k] = widths[k];
    save(WIDTHS_KEY, Object.keys(changed).length ? changed : null);
  }

  function setHidden(next) {
    hidden = new Set(next);
    save(COLUMNS_KEY, [...hidden]);
    apply();
  }

  boxes.forEach((b) => b.addEventListener("change", () => {
    setHidden(boxes.filter((x) => !x.checked).map((x) => x.dataset.key));
  }));
  document.querySelectorAll("#column-picker [data-pick]").forEach((btn) => btn.addEventListener("click", () => {
    const pick = btn.dataset.pick;
    setHidden(pick === "all" ? [] : pick === "none" ? optional : defaultHidden);
  }));
  const reset = document.getElementById("reset-widths");
  if (reset) reset.addEventListener("click", (e) => {
    e.preventDefault();
    widths = Object.assign({}, defaults);
    saveWidths();
    apply();
  });

  table.querySelectorAll("thead th[data-key]").forEach((th) => {
    const key = th.dataset.key;
    const grip = document.createElement("span");
    grip.className = "rz";
    grip.title = "drag to resize";
    let drag = null;
    grip.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      grip.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, w: widths[key] };
    });
    grip.addEventListener("pointermove", (e) => {
      if (!drag) return;
      widths[key] = clampWidth(drag.w + e.clientX - drag.x);
      apply();
    });
    const end = () => { if (drag) { drag = null; saveWidths(); } };
    grip.addEventListener("pointerup", end);
    grip.addEventListener("pointercancel", end);
    th.append(grip);
  });

  apply();
})(this);
