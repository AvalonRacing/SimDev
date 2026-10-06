// pipeline/simdev/ui/static/compare-restore.js
// /compare opened without runs: go back to the last selection of this browser.
(function () {
  const KEY = "simdev.compare.v1";
  const script = document.currentScript;
  try {
    // Requested runs that are gone: forget them, so the nav link stops
    // redirecting to a "not found" page. Chart choices and limits stay.
    const gone = ((script && script.dataset.missing) || "").split(",").filter(Boolean);
    if (gone.length) {
      const old = JSON.parse(localStorage.getItem(KEY));
      const S = window.SimdevState;
      if (old && S) localStorage.setItem(KEY, JSON.stringify(S.dropRuns(old, gone)));
    }
  } catch (e) { /* no storage */ }
  try {
    const params = new URLSearchParams(location.search);
    if (params.has("runs")) return;
    const saved = JSON.parse(localStorage.getItem(KEY));
    if (!saved || !Array.isArray(saved.runs) || !saved.runs.length) return;
    const q = new URLSearchParams({ runs: saved.runs.join(",") });
    if (typeof saved.ref === "string" && saved.ref) q.set("ref", saved.ref);
    location.replace(`/compare?${q}`);
  } catch (e) { /* no storage or a broken entry: show the empty picker */ }
})();
