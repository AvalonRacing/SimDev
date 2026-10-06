// pipeline/simdev/ui/static/compare-state.js
// The compare page's pure helpers: plane order, positions by name, and what is
// kept in the browser between visits. No DOM here, so node can run them.
(function (root) {
  const KEY = "simdev.compare.v1";
  const LAYOUTS = ["side", "overlay"];
  const MODES = ["blink", "swipe", "fade"];
  const LIMIT_FIELDS = ["cp", "cpt", "U"];
  const MAX_PANES = 4;

  // Planes run from the most positive offset to the most negative, so the x
  // slices start at the front of the car. The index itself stays ascending.
  const descending = (planes) => planes.slice().sort((a, b) => b.offset - a.offset);

  const posName = (list, pos) => ((list || [])[pos] || {}).name || null;
  function posIndex(list, name) {
    const i = (list || []).findIndex((item) => item.name === name);
    return i < 0 ? 0 : i;
  }

  const sameRuns = (a, b) => Array.isArray(a) && Array.isArray(b)
    && a.length === b.length && a.every((r, i) => r === b[i]);

  function store() {
    try { return root.localStorage || null; } catch (e) { return null; }
  }
  function load() {
    try {
      const s = store();
      const saved = s && JSON.parse(s.getItem(KEY));
      return saved && typeof saved === "object" ? saved : null;
    } catch (e) { return null; }
  }
  function save(obj) {
    try { const s = store(); if (s) s.setItem(KEY, JSON.stringify(obj)); } catch (e) { /* storage unavailable */ }
  }

  // The saved selection without runs the server reported missing. When no run
  // is left, runs and ref go; chart choices and limits stay. Returns a copy.
  function dropRuns(saved, gone) {
    if (!saved || typeof saved !== "object") return saved;
    const out = Object.assign({}, saved);
    if (!Array.isArray(saved.runs) || !Array.isArray(gone) || !gone.length) return out;
    const left = saved.runs.filter((r) => !gone.includes(r));
    if (left.length === saved.runs.length) return out;
    if (!left.length) { delete out.runs; delete out.ref; return out; }
    out.runs = left;
    if (!left.includes(out.ref)) delete out.ref;
    return out;
  }

  // env: { choices(kind, field) -> [{name}], fields(kind) -> [field], kinds: [...] }
  function serialize(state, env) {
    const view = (t) => ({ kind: t.kind, field: t.field, pos: posName(env.choices(t.kind, t.field), t.pos) });
    const o = state.overlay;
    return {
      runs: state.runs, ref: state.ref,
      global: view(state.global), layout: state.layout, limits: state.limits,
      panes: state.panes.map((p) => Object.assign({ run: p.run, sync: p.sync, delta: p.delta }, view(p))),
      overlay: { a: o.a, b: o.b, mode: o.mode, alpha: o.alpha, ms: o.ms },
      force: state.chart.force, cp: state.chart.cp,
    };
  }

  function viewOf(v, env) {
    v = v || {};
    const kind = env.kinds.includes(v.kind) ? v.kind : "x";
    const fields = env.fields(kind);
    const field = fields.includes(v.field) ? v.field : (fields[0] || "cp");
    return { kind, field, pos: posIndex(env.choices(kind, field), v.pos) };
  }

  // Validate a saved selection against what exists now. Never throws; anything
  // unknown falls back to the defaults. The viewer part is kept only when the
  // saved runs equal the current ones; limits and chart choices always are.
  function restore(saved, runs, env) {
    const out = { viewer: null, limits: {}, chart: {} };
    if (!saved || typeof saved !== "object") return out;
    try {
      const lim = saved.limits || {};
      for (const f of LIMIT_FIELDS) {
        // null = the user cleared the field: no limit given, the server decides.
        if (lim[f] === null || (typeof lim[f] === "number" && isFinite(lim[f]) && lim[f] >= 0)) out.limits[f] = lim[f];
      }
      const fo = saved.force || {};
      if (typeof fo.part === "string") out.chart.force = { part: fo.part, relative: fo.relative === true };
      const cp = saved.cp || {};
      if (typeof cp.station === "string" || Array.isArray(cp.patches)) out.chart.cp = {
        station: typeof cp.station === "string" ? cp.station : "",
        patches: Array.isArray(cp.patches) ? cp.patches.filter((p) => typeof p === "string") : null,
      };
      if (!sameRuns(saved.runs, runs)) return out;
      const panes = (Array.isArray(saved.panes) ? saved.panes : [])
        .filter((p) => p && runs.includes(p.run)).slice(0, MAX_PANES)
        .map((p) => Object.assign({ run: p.run, sync: p.sync !== false, delta: p.delta === true,
          zoom: { s: 1, x: 0, y: 0 } }, viewOf(p, env)));
      if (!panes.length) return out;
      const last = panes.length - 1;
      const int = (x) => (Number.isInteger(x) ? Math.max(0, Math.min(last, x)) : 0);
      const o = saved.overlay || {};
      out.viewer = {
        global: viewOf(saved.global, env),
        layout: LAYOUTS.includes(saved.layout) ? saved.layout : "side",
        panes,
        overlay: {
          a: int(o.a), b: Number.isInteger(o.b) ? int(o.b) : Math.min(1, last),
          mode: MODES.includes(o.mode) ? o.mode : "blink",
          alpha: typeof o.alpha === "number" && o.alpha >= 0 && o.alpha <= 1 ? o.alpha : 0.5,
          ms: typeof o.ms === "number" && o.ms >= 100 ? o.ms : 500,
        },
      };
    } catch (e) { out.viewer = null; }
    return out;
  }

  const api = { KEY, descending, posName, posIndex, sameRuns, dropRuns, load, save, serialize, restore };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SimdevState = api;
})(typeof window !== "undefined" ? window : globalThis);
