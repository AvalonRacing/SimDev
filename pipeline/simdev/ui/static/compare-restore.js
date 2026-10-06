// pipeline/simdev/ui/static/compare-restore.js
// /compare opened without runs: go back to the last selection of this browser.
(function () {
  try {
    const params = new URLSearchParams(location.search);
    if (params.has("runs")) return;
    const saved = JSON.parse(localStorage.getItem("simdev.compare.v1"));
    if (!saved || !Array.isArray(saved.runs) || !saved.runs.length) return;
    const q = new URLSearchParams({ runs: saved.runs.join(",") });
    if (typeof saved.ref === "string" && saved.ref) q.set("ref", saved.ref);
    location.replace(`/compare?${q}`);
  } catch (e) { /* no storage or a broken entry: show the empty picker */ }
})();
