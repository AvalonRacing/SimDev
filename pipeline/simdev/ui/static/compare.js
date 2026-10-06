// pipeline/simdev/ui/static/compare.js
// The compare page: synced pictures, overlays, on-request deltas (part 1),
// and the charts (part 2, below).
//
// Every picture of one view shares its frame pixel for pixel
// (post_views.yaml), so zoom and pan are kept in fractions of the picture
// and shared between synced panes.
(function () {
  const dataEl = document.getElementById("compare-data");
  if (!dataEl) return;
  const data = JSON.parse(dataEl.textContent);
  const PALETTE = ["#d0542c", "#2c6fd0", "#1f8a4c", "#8e44ad"];
  const DELTA_FIELDS = { plane: ["cp", "cpt", "U"], surface: ["cp"] };
  const SURFACE_VIEWS = ["front", "rear", "left", "right", "top", "bottom", "iso"];

  const state = {
    runs: data.runs.slice(),
    ref: data.ref,
    limits: Object.assign({ U: null }, data.limits),
    global: { kind: "x", field: "cp", pos: 0 },
    zoom: { s: 1, x: 0, y: 0 },
    layout: "side",
    overlay: { a: 0, b: 1, mode: "blink", showB: false, pos: 0.5, alpha: 0.5, timer: null, ms: 500 },
    panes: data.runs.slice(0, 3).map((run) => ({
      run, sync: true, delta: false, kind: "x", field: "cp", pos: 0, zoom: { s: 1, x: 0, y: 0 },
    })),
  };
  const indexes = {};
  // What render() built and update() refreshes in place: rebuilding the DOM
  // on every slider step would end the drag and drop keyboard focus.
  const live = { views: [], positions: [] };
  const colourOf = (run) => PALETTE[state.runs.indexOf(run) % PALETTE.length];
  window.SimdevCompare = { state, indexes, colourOf };

  // --- what a pane shows -----------------------------------------------------

  const settings = (pane) => (pane.sync ? state.global : pane);
  const zoomOf = (pane) => (pane.sync ? state.zoom : pane.zoom);

  function choices(kind, field) {
    // Positions come from REF's index: with the same views digest every run
    // has the same planes, and a run that lacks one shows a placeholder.
    const index = indexes[state.ref] || {};
    if (kind === "surface") return Object.keys((index.surfaces || {})[field] || {})
      .sort((a, b) => SURFACE_VIEWS.indexOf(a) - SURFACE_VIEWS.indexOf(b))
      .map((name) => ({ name, label: name }));
    return (((index.planes || {})[kind] || {})[field] || [])
      .map((p) => ({ name: p.name, label: `${kind} = ${p.offset >= 0 ? "+" : ""}${p.offset.toFixed(3)}` }));
  }

  function fieldsFor(kind) {
    const index = indexes[state.ref] || {};
    if (kind === "surface") return Object.keys(index.surfaces || {});
    return Object.keys((index.planes || {})[kind] || {});
  }

  function pictureOf(run, kind, field, name) {
    const index = indexes[run] || {};
    if (kind === "surface") return ((index.surfaces || {})[field] || {})[name] || null;
    const list = ((index.planes || {})[kind] || {})[field] || [];
    const hit = list.find((p) => p.name === name);
    return hit ? hit.rel : null;
  }

  function deltaAllowed(pane) {
    const s = settings(pane);
    const group = s.kind === "surface" ? "surface" : "plane";
    return pane.run !== state.ref && DELTA_FIELDS[group].includes(s.field);
  }

  function urlOf(pane) {
    const s = settings(pane);
    const list = choices(s.kind, s.field);
    const item = list[Math.min(s.pos, list.length - 1)];
    if (!item) return { url: null, label: "no pictures" };
    if (pane.delta && deltaAllowed(pane)) {
      const view = s.kind === "surface" ? `surface_${item.name}` : item.name;
      const limit = state.limits[s.field];
      const q = new URLSearchParams({ ref: state.ref, view, field: s.field });
      if (limit) q.set("limit", limit);
      // The surface delta matches points within 0.5 mm: on thin parts the
      // other side of the part is inside that radius (handbook, deltas).
      const caption = s.kind === "surface" ? "surface Δ: thin edges (wing) unreliable" : "";
      return { url: `/api/delta/${encodeURIComponent(pane.run)}.png?${q}`, label: `Δ ${item.label}`, delta: true, caption };
    }
    const rel = pictureOf(pane.run, s.kind, s.field, item.name);
    return rel ? { url: `/runs/${encodeURIComponent(pane.run)}/files/${rel}`, label: item.label }
               : { url: null, label: `${item.label}: no picture in this run` };
  }

  // --- loading pictures ------------------------------------------------------

  const deltaCache = new Map();  // url -> object URL, so blink is instant

  const timers = new WeakMap();       // img -> pending delta fetch (debounce or retry)
  const controllers = new WeakMap();  // img -> AbortController of its running fetch

  // A picture the img no longer wants must not keep a browser connection or
  // a server thread busy.
  function cancel(img) {
    clearTimeout(timers.get(img));
    timers.delete(img);
    const controller = controllers.get(img);
    if (controller) controller.abort();
    controllers.delete(img);
  }

  function load(img, note, target) {
    cancel(img);
    img.dataset.want = target.url || "";
    img.onerror = null;
    note.textContent = "";
    if (!target.url) { img.removeAttribute("src"); note.textContent = target.label; return; }
    if (!target.delta) {
      img.onerror = () => { if (img.dataset.want === target.url) note.textContent = "picture failed to load"; };
      img.src = target.url;
      return;
    }
    if (deltaCache.has(target.url)) { img.src = deltaCache.get(target.url); note.textContent = target.caption || ""; return; }
    note.textContent = "computing delta…";
    img.removeAttribute("src");
    const wanted = () => img.isConnected && img.dataset.want === target.url;
    const attempt = async () => {
      timers.delete(img);
      if (!wanted()) return;
      const controller = new AbortController();
      controllers.set(img, controller);
      try {
        const response = await fetch(target.url, { signal: controller.signal });
        if (response.status === 503) {
          // Another delta is being computed: keep "computing delta…" and ask again.
          const wait = (Number(response.headers.get("Retry-After")) || 2) * 1000;
          if (wanted()) timers.set(img, setTimeout(attempt, wait));
          return;
        }
        if (!response.ok) {
          const body = await response.json().catch(() => ({ error: response.statusText }));
          if (wanted()) note.textContent = body.error || "delta failed";
          return;
        }
        const objectUrl = URL.createObjectURL(await response.blob());
        deltaCache.set(target.url, objectUrl);
        if (!wanted()) return;
        img.src = objectUrl;
        note.textContent = target.caption || "";
      } catch (error) {
        if (error.name !== "AbortError" && wanted()) note.textContent = String(error);
      } finally {
        if (controllers.get(img) === controller) controllers.delete(img);
      }
    };
    // Debounced: stepping through planes must not queue a server delta per step.
    timers.set(img, setTimeout(attempt, 300));
  }

  function preload() {
    const s = state.global;
    const list = choices(s.kind, s.field);
    for (const pane of state.panes.filter((p) => p.sync && !p.delta)) {
      for (const step of [-2, -1, 1, 2]) {
        const item = list[s.pos + step];
        const rel = item && pictureOf(pane.run, s.kind, s.field, item.name);
        if (rel) new Image().src = `/runs/${encodeURIComponent(pane.run)}/files/${rel}`;
      }
    }
  }

  // --- zoom and pan ----------------------------------------------------------

  function applyZoom(img, zoom) {
    img.style.transform = `translate(${zoom.x * 100}%, ${zoom.y * 100}%) scale(${zoom.s})`;
  }

  function attachZoom(viewport, getZoom) {
    viewport.addEventListener("wheel", (event) => {
      const zoom = getZoom();
      // Nothing to zoom out of (or a sideways scroll): let the page scroll.
      if (event.deltaY === 0 || (zoom.s === 1 && event.deltaY > 0)) return;
      event.preventDefault();
      const box = viewport.getBoundingClientRect();
      const fx = (event.clientX - box.left) / box.width;
      const fy = (event.clientY - box.top) / box.height;
      const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
      const s = Math.min(20, Math.max(1, zoom.s * factor));
      // keep the point under the cursor where it is
      zoom.x = fx - (fx - zoom.x) * (s / zoom.s);
      zoom.y = fy - (fy - zoom.y) * (s / zoom.s);
      zoom.s = s;
      if (s === 1) { zoom.x = 0; zoom.y = 0; }
      refreshZoom();
    }, { passive: false });
    let drag = null;
    viewport.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".handle")) return;
      drag = { x: event.clientX, y: event.clientY };
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const zoom = getZoom();
      const box = viewport.getBoundingClientRect();
      zoom.x += (event.clientX - drag.x) / box.width;
      zoom.y += (event.clientY - drag.y) / box.height;
      drag = { x: event.clientX, y: event.clientY };
      refreshZoom();
    });
    viewport.addEventListener("pointerup", () => { drag = null; });
    viewport.addEventListener("dblclick", () => {
      Object.assign(getZoom(), { s: 1, x: 0, y: 0 });
      refreshZoom();
    });
  }

  function refreshZoom() {
    document.querySelectorAll("[data-pane]").forEach((img) => {
      const pane = state.panes[Number(img.dataset.pane)];
      if (pane) applyZoom(img, img.dataset.overlay ? state.zoom : zoomOf(pane));
    });
  }

  // --- controls --------------------------------------------------------------

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "on") for (const [e, f] of Object.entries(v)) node.addEventListener(e, f);
      else if (k === "text") node.textContent = v;
      else if (v === true) node.setAttribute(k, "");
      else if (v !== false && v != null) node.setAttribute(k, v);
    }
    for (const child of [].concat(children)) if (child) node.append(child);
    return node;
  }

  function select(options, value, onChange) {
    return el("select", { on: { change: (e) => onChange(e.target.value) } },
      options.map(([v, label]) => el("option", { value: v, selected: String(v) === String(value), text: label })));
  }

  function positionControls(target, onChange) {
    const kinds = [["x", "x-planes"], ["y", "y-planes"], ["z", "z-planes"], ["surface", "surface"]];
    const fields = fieldsFor(target.kind);
    if (!fields.includes(target.field) && fields.length) target.field = fields[0];
    const list = choices(target.kind, target.field);
    target.pos = Math.max(0, Math.min(target.pos, list.length - 1));
    const slider = el("input", { type: "range", min: 0, max: Math.max(0, list.length - 1), value: target.pos,
      on: { input: (e) => { target.pos = Number(e.target.value); onChange(false); } } });
    const label = el("span", { class: "muted", text: (list[target.pos] || {}).label || "–" });
    live.positions.push({ target, slider, label });
    return el("span", { class: "row" }, [
      select(fields.map((f) => [f, f]), target.field, (v) => { target.field = v; onChange(true); }),
      select(kinds, target.kind, (v) => { target.kind = v; target.pos = 0; onChange(true); }),
      slider, label,
    ]);
  }

  function toolbar() {
    const g = state.global;
    const limits = ["cp", "cpt", "U"].map((f) => el("label", { class: "row" }, [
      `Δ ${f} ±`,
      el("input", { type: "number", step: "any", min: 0, size: 5, value: state.limits[f] ?? "",
        placeholder: f === "U" ? "10% u∞" : "", on: { change: (e) => {
          state.limits[f] = e.target.value ? Number(e.target.value) : null; render(); } } }),
    ]));
    return el("div", { class: "toolbar" }, [
      el("div", { class: "row" }, [positionControls(g, (structural) => (structural ? render() : update())),
        el("span", { class: "muted", text: "← → step, Shift ×5, B blink, double-click resets zoom" })]),
      el("div", { class: "row" }, [
        el("label", { class: "row" }, [el("input", { type: "radio", name: "layout", checked: state.layout === "side",
          on: { change: () => { state.layout = "side"; render(); } } }), "side by side"]),
        el("label", { class: "row" }, [el("input", { type: "radio", name: "layout", checked: state.layout === "overlay",
          disabled: state.panes.length < 2, on: { change: () => { state.layout = "overlay"; render(); } } }), "overlay"]),
        el("button", { type: "button", disabled: state.panes.length >= 4,
          text: "+ add pane", title: "another pane of a run on this page", on: { click: addPane } }),
        // A run not yet on the page: reload with it, so the numbers table includes it too.
        state.runs.length >= 4 ? null : select([["", "+ add run…"], ...data.all_runs.filter((r) => !state.runs.includes(r)).map((r) => [r, r])], "",
          (v) => { if (v) location.href = `/compare?${new URLSearchParams({ runs: [...state.runs, v].join(","), ref: state.ref })}`; }),
        ...limits,
      ]),
    ]);
  }

  function addPane() {
    const used = state.panes.map((p) => p.run);
    const run = state.runs.find((r) => !used.includes(r)) || state.runs[0];
    state.panes.push({ run, sync: true, delta: false, kind: state.global.kind, field: state.global.field,
      pos: state.global.pos, zoom: { s: 1, x: 0, y: 0 } });
    render();
  }

  function paneHeader(pane, i) {
    const isRef = pane.run === state.ref;
    return el("div", { class: "pane-head", style: `border-color:${colourOf(pane.run)}` }, [
      el("label", { class: "row" }, [el("input", { type: "checkbox", checked: pane.sync,
        on: { change: (e) => { pane.sync = e.target.checked;
          if (!pane.sync) Object.assign(pane, { kind: state.global.kind, field: state.global.field, pos: state.global.pos });
          render(); } } }), "sync"]),
      select(state.runs.map((r) => [r, r]), pane.run, (v) => { pane.run = v; render(); }),
      isRef ? el("span", { class: "badge", text: "REF" })
            // Reload with the new REF, so the numbers table, bars and pictures share one REF.
            : el("button", { type: "button", text: "make REF", on: { click: () => {
                location.href = `/compare?${new URLSearchParams({ runs: state.runs.join(","), ref: pane.run })}`; } } }),
      el("label", { class: "row" }, [el("input", { type: "checkbox", checked: pane.delta && deltaAllowed(pane),
        disabled: !deltaAllowed(pane), on: { change: (e) => { pane.delta = e.target.checked; update(); } } }), "Δ vs REF"]),
      state.panes.length > 1 ? el("button", { type: "button", text: "×", title: "remove pane",
        on: { click: () => { state.panes.splice(i, 1); render(); } } }) : null,
      pane.sync ? null : positionControls(pane, (structural) => (structural ? render() : update())),
    ]);
  }

  // --- layouts ---------------------------------------------------------------

  function sideBySide(root) {
    const grid = el("div", { class: `panes n${state.panes.length}` });
    state.panes.forEach((pane, i) => {
      const img = el("img", { "data-pane": i, draggable: "false", alt: "" });
      const note = el("div", { class: "pane-note" });
      const viewport = el("div", { class: "viewport" }, [img, note]);
      attachZoom(viewport, () => zoomOf(pane));
      grid.append(el("div", { class: "pane" }, [paneHeader(pane, i), viewport]));
      applyZoom(img, zoomOf(pane));
      live.views.push({ img, note, pane });
    });
    root.append(grid);
  }

  function overlay(root) {
    const o = state.overlay;
    o.a = Math.min(o.a, state.panes.length - 1);
    o.b = Math.min(o.b, state.panes.length - 1);
    const options = state.panes.map((p, i) => [i, `${i + 1}: ${p.run}${p.delta ? " (Δ)" : ""}`]);
    const imgA = el("img", { "data-pane": o.a, "data-overlay": "1", draggable: "false", alt: "" });
    const imgB = el("img", { "data-pane": o.b, "data-overlay": "1", draggable: "false", alt: "", class: "top" });
    const noteA = el("div", { class: "pane-note" });
    const noteB = el("div", { class: "pane-note", style: "top:1.8rem" });
    const handle = el("div", { class: "handle" });
    const viewport = el("div", { class: "viewport overlay" }, [imgA, imgB, handle, noteA, noteB]);
    const label = el("div", { class: "overlay-label" });

    function show() {
      const top = o.mode === "blink" ? (o.showB ? state.panes[o.b] : state.panes[o.a]) : state.panes[o.b];
      label.textContent = o.mode === "blink" ? `showing ${top.run}` : `${state.panes[o.a].run} | ${state.panes[o.b].run}`;
      label.style.color = colourOf(top.run);
      imgB.style.visibility = o.mode === "blink" && !o.showB ? "hidden" : "visible";
      imgB.style.opacity = o.mode === "fade" ? o.alpha : 1;
      imgB.style.clipPath = o.mode === "swipe" ? `inset(0 0 0 ${o.pos * 100}%)` : "none";
      handle.style.display = o.mode === "swipe" ? "block" : "none";
      handle.style.left = `${o.pos * 100}%`;
    }
    state.overlay.flip = () => { o.showB = !o.showB; show(); };

    let dragging = false;
    handle.addEventListener("pointerdown", (e) => { dragging = true; handle.setPointerCapture(e.pointerId); });
    handle.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const box = viewport.getBoundingClientRect();
      o.pos = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
      show();
    });
    handle.addEventListener("pointerup", () => { dragging = false; });
    attachZoom(viewport, () => state.zoom);

    const modes = ["blink", "swipe", "fade"].map((m) => el("label", { class: "row" }, [
      el("input", { type: "radio", name: "omode", checked: o.mode === m,
        on: { change: () => { o.mode = m; show(); } } }), m]));
    const auto = el("label", { class: "row" }, [el("input", { type: "checkbox", checked: !!o.timer,
      on: { change: (e) => {
        clearInterval(o.timer); o.timer = null;
        if (e.target.checked) o.timer = setInterval(state.overlay.flip, o.ms);
      } } }), "auto",
      el("input", { type: "number", min: 100, step: 100, value: o.ms, size: 4,
        on: { change: (e) => { o.ms = Number(e.target.value) || 500;
          if (o.timer) { clearInterval(o.timer); o.timer = setInterval(state.overlay.flip, o.ms); } } } }), "ms"]);

    root.append(el("div", { class: "row" }, [
      "A", select(options, o.a, (v) => { o.a = Number(v); render(); }),
      "B", select(options, o.b, (v) => { o.b = Number(v); render(); }),
      ...modes,
      el("button", { type: "button", text: "blink (B)", on: { click: state.overlay.flip } }),
      auto,
      el("input", { type: "range", min: 0, max: 1, step: 0.05, value: o.alpha, title: "fade",
        on: { input: (e) => { o.alpha = Number(e.target.value); show(); } } }),
    ]));
    root.append(el("div", { class: "pane overlay-pane" }, [label, viewport]));
    applyZoom(imgA, state.zoom);
    applyZoom(imgB, state.zoom);
    live.views.push({ img: imgA, note: noteA, pane: state.panes[o.a] },
                    { img: imgB, note: noteB, pane: state.panes[o.b] });
    show();
  }

  // --- render ----------------------------------------------------------------

  const root = document.getElementById("viewer");

  const failed = [];

  function guards() {
    const box = document.getElementById("guards");
    box.innerHTML = "";
    for (const run of failed) box.append(el("div", { class: "warn", text: `index of ${run} could not be loaded` }));
    const known = state.runs.filter((r) => !failed.includes(r));
    const digests = new Set(known.map((r) => indexes[r].views_digest));
    const states = new Set(known.map((r) => indexes[r].state).filter(Boolean));
    const speeds = new Set(known.map((r) => indexes[r].u_inf).filter((u) => u != null));
    if (digests.size > 1) box.append(el("div", { class: "warn",
      text: "these runs were pictured with different post_views.yaml - their pictures are not comparable and deltas are refused" }));
    if (speeds.size > 1 || states.size > 1) {
      const parts = [];
      if (states.size > 1) parts.push([...states].join(", "));
      if (speeds.size > 1) parts.push(`u∞ ${[...speeds].join(", ")} m/s`);
      box.append(el("div", { class: "warn",
        text: `different driving states (${parts.join("; ")}): compare coefficients with care` }));
    }
  }

  function strip() {
    const box = document.getElementById("strip");
    box.innerHTML = "";
    for (const run of state.runs) {
      const index = indexes[run] || {};
      box.append(el("span", { class: "chip", style: `border-color:${colourOf(run)}` }, [
        el("strong", { text: run }), run === state.ref ? el("span", { class: "badge", text: "REF" }) : null,
        el("span", { class: "muted", text: ` ${index.state || ""} · views ${index.views_digest || "–"}` }),
      ]));
    }
  }

  function render() {
    clearInterval(state.overlay.timer);
    state.overlay.timer = null;
    for (const v of live.views) cancel(v.img);
    root.innerHTML = "";
    live.views = [];
    live.positions = [];
    root.append(toolbar());
    if (state.layout === "overlay" && state.panes.length >= 2) overlay(root); else sideBySide(root);
    strip();
    update();
    document.dispatchEvent(new CustomEvent("simdev:ref", { detail: state.ref }));
  }

  // A new position or a Δ toggle: swap pictures and labels, keep the DOM.
  function update() {
    for (const p of live.positions) {
      p.slider.value = p.target.pos;
      p.label.textContent = (choices(p.target.kind, p.target.field)[p.target.pos] || {}).label || "–";
    }
    for (const v of live.views) load(v.img, v.note, urlOf(v.pane));
    preload();
    document.dispatchEvent(new Event("simdev:moved"));
  }

  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input, select, textarea")) return;
    const g = state.global;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      const n = choices(g.kind, g.field).length;
      const step = (event.shiftKey ? 5 : 1) * (event.key === "ArrowLeft" ? -1 : 1);
      g.pos = Math.max(0, Math.min(n - 1, g.pos + step));
      event.preventDefault();
      update();
    } else if ((event.key === "b" || event.key === "B") && state.overlay.flip && state.layout === "overlay") {
      state.overlay.flip();
    }
  });

  Promise.all(state.runs.map((run) => fetch(`/api/runs/${encodeURIComponent(run)}/index`)
    .then((r) => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
    .then((index) => { indexes[run] = index; })
    .catch(() => { indexes[run] = {}; failed.push(run); })))
    .then(() => { guards(); render(); document.dispatchEvent(new Event("simdev:ready")); });
})();

// --- part 2: charts ----------------------------------------------------------
(function () {
  if (!document.getElementById("compare-data")) return;
  const forces = {}, summaries = {};
  let C = null;

  const width = () => Math.max(320, Math.min(document.querySelector("main").clientWidth - 40, 1100));

  // iteration -> value, last occurrence wins (a restart repeats iterations).
  function lastWins(t, coefficient) {
    const m = new Map();
    const v = (t && t[coefficient]) || [];
    ((t && t.iteration) || []).forEach((it, i) => m.set(it, v[i]));
    return m;
  }

  function partSeries(run, part, coefficient) {
    const f = forces[run];
    if (!f) return null;
    let maps;
    if (!part) {
      maps = [lastWins(f.total, coefficient)];
    } else {
      const members = ((summaries[run] || {}).groups_map || {})[part] || [part];
      const tables = members.map((m) => (f.components || {})[m]).filter(Boolean);
      if (!tables.length || tables.length !== members.length) return null;
      maps = tables.map((t) => lastWins(t, coefficient));
    }
    // Like summary.py: only iterations present in every member, none with a missing value.
    const it = [...maps[0].keys()]
      .filter((x) => maps.every((m) => m.has(x) && m.get(x) != null))
      .sort((x, y) => x - y);
    if (!it.length) return null;
    return { it, v: it.map((x) => maps.reduce((acc, m) => acc + m.get(x), 0)) };
  }

  function windowMean(run, s) {
    const [a, b] = (summaries[run] || {}).window || [0, 0];
    const values = s.v.filter((x, i) => x != null && s.it[i] >= a && s.it[i] <= b);
    return values.length ? values.reduce((x, y) => x + y, 0) / values.length : null;
  }

  const plots = {};

  function drawForces() {
    const part = document.getElementById("force-part").value;
    const relative = document.getElementById("force-relative").checked;
    for (const coefficient of ["Cl", "Cd"]) {
      const box = document.getElementById(`plot-${coefficient}`);
      if (plots[coefficient]) { plots[coefficient].destroy(); plots[coefficient] = null; }
      box.innerHTML = "";
      const entries = [];
      let maxAbs = 0;
      for (const r of C.state.runs) {
        const s = partSeries(r, part, coefficient);
        if (!s) continue;
        let v = s.v;
        if (relative) {
          const mean = windowMean(r, s);
          if (mean == null || mean === 0) continue;
          v = s.v.map((x) => (100 * (x - mean)) / Math.abs(mean));
          const [a, b] = (summaries[r] || {}).window || [0, 0];
          v.forEach((x, i) => { if (s.it[i] >= a && s.it[i] <= b) maxAbs = Math.max(maxAbs, Math.abs(x)); });
        }
        entries.push({ run: r, it: s.it, v });
      }
      if (!entries.length) { box.textContent = "no history"; continue; }
      const runs = entries.map((e) => e.run);
      const joined = uPlot.join(entries.map((e) => [e.it, e.v]));
      const bands = {
        hooks: { drawClear: [(u) => {
          const ctx = u.ctx;
          ctx.save();
          runs.forEach((r) => {
            const [a, b] = (summaries[r] || {}).window || [0, 0];
            ctx.fillStyle = C.colourOf(r) + "18";
            const x0 = u.valToPos(a, "x", true), x1 = u.valToPos(b, "x", true);
            ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height);
          });
          if (relative) for (const [lim, alpha] of [[1, "22"], [0.5, "33"]]) {
            const y0 = u.valToPos(lim, "y", true), y1 = u.valToPos(-lim, "y", true);
            ctx.fillStyle = "#1f8a4c" + alpha;
            ctx.fillRect(u.bbox.left, y0, u.bbox.width, y1 - y0);
          }
          ctx.restore();
        }] },
      };
      // Relative mode zooms on the converged window, not the start-up transient.
      const r = Math.max(3, 1.5 * maxAbs);
      const scales = { x: { time: false } };
      if (relative) scales.y = { range: () => [-r, r] };
      plots[coefficient] = new uPlot({
        width: width(), height: 260, plugins: [bands], scales,
        axes: [{}, { label: relative ? `${coefficient} − mean [%]` : coefficient }],
        series: [{ label: "iteration" }, ...runs.map((x) => ({ label: x, stroke: C.colourOf(x), width: 1.2 }))],
      }, joined, box);
    }
  }

  function drawBars() {
    const box = document.getElementById("bars");
    box.innerHTML = "";
    const ref = summaries[C.state.ref];
    if (!ref) { box.textContent = "no results for REF"; return; }
    const parts = [
      ...Object.keys(ref.groups_map || {}).map((g) => ({ label: g, key: g, group: true })),
      ...Object.keys(ref.patches || {}).sort().map((p) => ({ label: p, key: p, group: false })),
    ];
    const valueOf = (s, part, c) => part.group
      ? (s.values || {})[`${c.toLowerCase()}_${part.key}`]
      : ((s.patches || {})[part.key] || {})[c];
    const noiseOf = (s, part, c) => part.group
      ? (s.noise || {})[`${c.toLowerCase()}_${part.key}`]
      : ((s.patches || {})[part.key] || {})[`${c}_noise`];
    for (const run of C.state.runs.filter((r) => r !== C.state.ref && summaries[r])) {
      const s = summaries[run];
      const rows = [];
      let largest = 1e-9;
      for (const part of parts) for (const c of ["Cl", "Cd"]) {
        const a = valueOf(s, part, c), b = valueOf(ref, part, c);
        if (a == null || b == null) continue;
        const na = noiseOf(s, part, c), nb = noiseOf(ref, part, c);
        const noise = na != null && nb != null ? Math.hypot(na, nb) : null;
        rows.push({ label: `${part.label} ${c}`, d: a - b, noise });
        largest = Math.max(largest, Math.abs(a - b), noise || 0);
      }
      const table = document.createElement("table");
      table.className = "bars";
      table.innerHTML = `<caption style="color:${C.colourOf(run)}">${run} − ${C.state.ref}</caption>`;
      for (const row of rows) {
        const tr = table.insertRow();
        tr.insertCell().textContent = row.label;
        const cell = tr.insertCell();
        const bar = document.createElement("div");
        bar.className = "bar";
        const pct = (50 * Math.abs(row.d)) / largest;
        bar.innerHTML = `<span class="fill" style="${row.d < 0 ? "right:50%" : "left:50%"};width:${pct}%;background:${C.colourOf(run)}"></span>`
          + (row.noise != null ? `<span class="whisker" style="left:${50 - (50 * row.noise) / largest}%;width:${(100 * row.noise) / largest}%"></span>` : "");
        cell.append(bar);
        const num = tr.insertCell();
        num.className = "num";
        num.textContent = `${row.d >= 0 ? "+" : ""}${row.d.toFixed(4)}${row.noise != null ? ` ± ${row.noise.toFixed(4)}` : ""}`;
      }
      box.append(table);
    }
  }

  let cpSeq = 0;
  async function drawCp() {
    const station = document.getElementById("cp-station").value;
    const patches = [...document.querySelectorAll(".cp-patch:checked")].map((b) => b.value);
    const data = {};
    const mine = ++cpSeq;
    await Promise.all(C.state.runs.map(async (run) => {
      try {
        const r = await fetch(`/api/runs/${encodeURIComponent(run)}/cplines/${encodeURIComponent(station)}`);
        if (r.ok) data[run] = await r.json();
      } catch (e) { /* this run just has no points */ }
    }));
    if (mine !== cpSeq) return;  // a newer station was chosen meanwhile
    const points = (key) => Object.entries(data).flatMap(([run, d]) => patches.flatMap((p) => {
      const s = (d.patches || {})[p];
      return s ? s.x.map((x, i) => ({ x, y: s[key][i], run })).filter((q) => q.x != null && q.y != null) : [];
    }));
    scatter(document.getElementById("cp-plot"), points("cp"), true, "cp");
    scatter(document.getElementById("cp-outline"), points("z"), false, "z [m]");
  }

  function scatter(canvas, pts, invert, label) {
    // cp axis inverted (suction up), as viz/cplines.py draws it.
    const ctx = canvas.getContext("2d");
    canvas.width = width();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!pts.length) { ctx.fillText("no cp lines for this station", 20, 20); return; }
    const pad = { l: 50, r: 10, t: 10, b: 24 };
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of pts) {
      if (p.x < x0) x0 = p.x;
      if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.y > y1) y1 = p.y;
    }
    const W = canvas.width - pad.l - pad.r, H = canvas.height - pad.t - pad.b;
    const X = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * W;
    const Y = (y) => pad.t + (invert ? (y - y0) / (y1 - y0 || 1) : 1 - (y - y0) / (y1 - y0 || 1)) * H;
    ctx.strokeStyle = "#999";
    ctx.strokeRect(pad.l, pad.t, W, H);
    ctx.fillStyle = "#333";
    ctx.fillText(label, 4, pad.t + 10);
    ctx.fillText(y0.toFixed(2), 4, invert ? pad.t + 22 : pad.t + H);
    ctx.fillText(y1.toFixed(2), 4, invert ? pad.t + H : pad.t + 22);
    ctx.fillText(`x ${x0.toFixed(3)} … ${x1.toFixed(3)} m`, pad.l, canvas.height - 6);
    for (const p of pts) {
      ctx.fillStyle = C.colourOf(p.run);
      ctx.fillRect(X(p.x) - 1, Y(p.y) - 1, 2, 2);
    }
  }

  function fillSelectors() {
    const parts = new Set();
    for (const run of C.state.runs) {
      Object.keys((summaries[run] || {}).groups_map || {}).forEach((g) => parts.add(g));
      Object.keys((forces[run] || {}).components || {}).forEach((p) => parts.add(p));
    }
    const select = document.getElementById("force-part");
    for (const p of [...parts].sort()) select.append(new Option(p, p));
    const stations = (C.indexes[C.state.ref] || {}).stations || [];
    const station = document.getElementById("cp-station");
    for (const s of stations) station.append(new Option(s.name, s.name));
  }

  document.addEventListener("simdev:ready", async () => {
    C = window.SimdevCompare;
    await Promise.all(C.state.runs.map(async (run) => {
      const get = (path) => fetch(`/api/runs/${encodeURIComponent(run)}/${path}`)
        .then((r) => (r.ok ? r.json() : null)).catch(() => null);
      const [f, s] = await Promise.all([get("forces"), get("summary")]);
      if (f) forces[run] = f;
      if (s) summaries[run] = s;
    }));
    fillSelectors();
    drawForces();
    drawBars();
    drawCp();
    document.getElementById("force-part").addEventListener("change", drawForces);
    document.getElementById("force-relative").addEventListener("change", drawForces);
    document.getElementById("cp-station").addEventListener("change", drawCp);
    document.querySelectorAll(".cp-patch").forEach((b) => b.addEventListener("change", drawCp));
    document.addEventListener("simdev:ref", drawBars);
    // The cp station follows the viewer when it is on a matching y-plane.
    document.addEventListener("simdev:moved", () => {
      const g = C.state.global;
      if (g.kind !== "y") return;
      const planes = (((C.indexes[C.state.ref] || {}).planes || {}).y || {})[g.field] || [];
      const plane = planes[g.pos];
      const select = document.getElementById("cp-station");
      if (plane && [...select.options].some((o) => o.value === plane.name) && select.value !== plane.name) {
        select.value = plane.name;
        drawCp();
      }
    });
  });
})();
