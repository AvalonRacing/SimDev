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
      return { url: `/api/delta/${encodeURIComponent(pane.run)}.png?${q}`, label: `Δ ${item.label}`, delta: true };
    }
    const rel = pictureOf(pane.run, s.kind, s.field, item.name);
    return rel ? { url: `/runs/${encodeURIComponent(pane.run)}/files/${rel}`, label: item.label }
               : { url: null, label: `${item.label}: no picture in this run` };
  }

  // --- loading pictures ------------------------------------------------------

  const deltaCache = new Map();  // url -> object URL, so blink is instant

  async function load(img, note, target) {
    note.textContent = "";
    if (!target.url) { img.removeAttribute("src"); note.textContent = target.label; return; }
    if (!target.delta) { img.src = target.url; return; }
    if (deltaCache.has(target.url)) { img.src = deltaCache.get(target.url); return; }
    note.textContent = "computing delta…";
    img.removeAttribute("src");
    try {
      const response = await fetch(target.url);
      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: response.statusText }));
        note.textContent = body.error || "delta failed";
        return;
      }
      const objectUrl = URL.createObjectURL(await response.blob());
      deltaCache.set(target.url, objectUrl);
      img.src = objectUrl;
      note.textContent = "";
    } catch (error) {
      note.textContent = String(error);
    }
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
      event.preventDefault();
      const zoom = getZoom();
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
        select([["", "+ add run…"], ...data.all_runs.filter((r) => !state.runs.includes(r)).map((r) => [r, r])], "",
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
            : el("button", { type: "button", text: "make REF", on: { click: () => { state.ref = pane.run; render(); } } }),
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
    const handle = el("div", { class: "handle" });
    const viewport = el("div", { class: "viewport overlay" }, [imgA, imgB, handle, noteA]);
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
                    { img: imgB, note: noteA, pane: state.panes[o.b] });
    show();
  }

  // --- render ----------------------------------------------------------------

  const root = document.getElementById("viewer");

  function guards() {
    const box = document.getElementById("guards");
    box.innerHTML = "";
    const digests = new Set(state.runs.map((r) => (indexes[r] || {}).views_digest));
    const states = new Set(state.runs.map((r) => (indexes[r] || {}).state).filter(Boolean));
    if (digests.size > 1) box.append(el("div", { class: "warn",
      text: "these runs were pictured with different post_views.yaml - their pictures are not comparable and deltas are refused" }));
    if (states.size > 1) box.append(el("div", { class: "warn",
      text: `different driving states (${[...states].join(", ")}): different u∞, compare coefficients with care` }));
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
    .then((r) => r.json()).then((index) => { indexes[run] = index; })))
    .then(() => { guards(); render(); document.dispatchEvent(new Event("simdev:ready")); });
})();
