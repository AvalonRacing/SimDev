# SimDev UI: Results Table and Compare Viewer — Design

**Status: approved in conversation section by section, spec awaiting review.**
Written 2026-10-06. Extends the UI of `2026-10-02-simdev-ui-design.md`.

The UI can queue and watch runs, but it shows almost nothing of what a
finished run produced and cannot put two runs next to each other. This adds a
results table that keeps the history of every design iteration, and a compare
page that shows the pictures, deltas and numbers of two or three runs side by
side.

---

## 1. Why this work exists

A finished run already writes everything needed: `results/result.json`
(Cd/Cl with std and amplitude, forces, moments, COP, balance, per-group Cd/Cl,
y+), coefficient histories per patch, about 570 plane and surface pictures
framed identically for every run (`cases/post_views.yaml`), cp-line CSVs and
the sampled planes and surface as `.vtp` under `postProcessing/`.

What happens to it today is a set of throwaway scripts in `~/runs`
(`compare.py`, `ab-report.py`, `prod-compare.py`, `plot_compare.py`, ...), each
re-reading `coefficient.dat` and redoing the windowing, plus the old
pipeline's Excel sheet (`benchmark_old_pipeline/Aeroexcel.xlsx`).

How the user works, stated in the brainstorm:

- the question is almost always **design A against design B**, in one driving
  state, at most two or three;
- only Body and Wing change between iterations;
- the reference is chosen per comparison - usually the previous iteration or
  the best one so far - never a fixed baseline;
- runs do not yet converge within 1 % (target 0.5 %), so **every delta has to
  carry its noise**, or a 1 % "gain" is read off a limit cycle;
- the knowledge comes from **slices and cp surface views**; a free 3D viewer
  is not wanted;
- one user, over Tailscale.

### 1.1 What the Excel sheet does, and keeps

Sheet `TC10`: one row per variant with an ID (`01_002`), a **"Compare with"**
cell holding the row number of the variant it is judged against, the name,
cz/cx, Fx/Fz/Fy, cz/cx of bodyshell and rear wing, COP, efficiency (Fz/Fx),
aero balance, and a **"Geometrie"** text naming the change against that
reference (`01_001_a + korrektur radlauf hinten`). Under every row a
**DeltaRow** (row minus its reference) coloured green/red. Sheet
`TC10_sensitivty` holds the same design across driving states.

Kept: per-row reference, change note, delta row, colouring, the column set.
Not kept: references by row number, which is why so many cells read `#REF!`.
Here a reference is a run name.

## 2. Scope

**In:**

1. A change note in the New Run form, stored with the run.
2. A **Results** page: one table of every run with results, history-keeping,
   editable note and "compare with" per run, Δ row, noise, copy as TSV.
3. A **Compare** page for 1-3 runs (4 allowed): numbers, image viewer with
   per-pane sync, blink/swipe/fade overlay, and **field deltas computed on
   request** for planes and for the cp surface.
4. Charts on the Compare page: overlaid force histories with convergence
   bands, Δ bars per component, overlaid cp lines.

**Out (non-goals):**

- interactive 3D, streamlines, arbitrary new slices;
- deltas of λ2 and y+ (not meaningful to subtract);
- deltas computed from the PNGs (measured useless, §7.1);
- weighting across driving states / lap simulation;
- any change to what the pipeline stages compute. The UI reads files; the
  only new code that touches VTK is the delta helper (§6).

## 3. Change note and run metadata

### 3.1 `<run>/ui/note.json`

```json
{"note": "02_009 + rear wing post 5 mm lower", "compare_with": "c02_combo12",
 "updated_at": "2026-10-06T10:12:00+00:00"}
```

- One file per run, next to the existing `ui/job.json`, so it moves and is
  deleted with the run directory.
- `compare_with` is a run name (a directory under the runs root) or `null`.
  A name that no longer exists is shown as "missing" and kept, not erased.
- Written atomically (temp file in the same directory, then rename).
- **Editable for every run, including runs started from the shell.** Those
  stay read-only for everything else (resume, rerun, strip, delete); the note
  is metadata only and never touches results. The UI creates `ui/` for a shell
  run when the first note is saved.

### 3.2 New Run form

- An optional multi-line field **"What changed in the geometry?"**.
- Stored on the job (new nullable `note` column in `jobs`, added by an
  `ALTER TABLE` migration on start-up) because the run directory may not
  exist yet when the job is queued.
- When the worker creates the run directory and writes `ui/job.json`, it also
  writes `ui/note.json` from the job's note. From then on `note.json` is the
  only truth; editing the job before it starts edits `jobs.note`.

## 4. Results page (`/results`)

New nav entry **Results**. One table, every run under the runs root that has
`results/result.json`, newest first; runs without results are listed greyed
out as "no results yet" so a running iteration is visible in the history.

### 4.1 Columns

| Column | Source |
|---|---|
| ☐ select | - |
| run | directory name, links to the run page |
| state | `driving_state` from `report.tsv` / spec |
| note | `ui/note.json` (editable inline) |
| compare with | `ui/note.json` (dropdown of all runs, inline) |
| Cl, Cd | `cl_mean`, `cd_mean` |
| −Cl/Cd | efficiency, positive for downforce |
| balance % front | `balance_front_pct` |
| Fx, Fy, Fz | `fx`, `fy`, `fz` |
| COP x, y, z | `cop_x`, `cop_y`, `cop_z` |
| Cl/Cd body, wing, other | `groups` |
| Cs | `cs_mean` |
| noise Cl, Cd | §4.3 |
| verdict, iterations, cells | `verdict`, `n_iterations`, `n_cells` |

### 4.2 Δ row

- Under every run whose `compare_with` is set: run minus its reference, per
  numeric column.
- Colour by direction of "better": Cl more negative, Cd lower, −Cl/Cd higher
  are green, the opposite red. Balance, COP, forces and Cs are not coloured
  (better depends on the target).
- A Δ whose magnitude is below its noise (§4.3) is **greyed out** instead of
  coloured, and the cell tooltip gives "Δ −0.004 ± 0.009".

### 4.3 Noise

- Noise of one run, per coefficient: **half the spread (max − min) of the
  rolling mean of length W/2 inside the averaging window** of length W - how
  far the reported mean moves depending on where the run happened to stop.
  This is the `ab-report.py` idea, applied to the run's own window.
- For groups the same is computed from the group's history (sum of its
  patches' `forceCoeffs_<patch>` histories).
- Noise of a Δ: √(noise_A² + noise_B²).
- Computed in the UI from the histories and cached per run against the
  history file's mtime. If it proves itself it can move into `result.json`
  later; the UI then reads it from there.

### 4.4 Interactions

- Filter: state, design, verdict, free text over name and note.
- Tick boxes + **Open in viewer** → `/compare?runs=A,B,C` (order kept; the
  first is the reference unless changed there).
- **Copy as TSV**: the visible rows and Δ rows, tab-separated, empty cell for
  "uncomputable" as in `report.tsv`.
- The run page gets a **Compare with reference** link →
  `/compare?runs=<compare_with>,<this run>` when `compare_with` is set.

## 5. Compare page (`/compare?runs=...`)

Top to bottom.

### 5.1 Run strip

One chip per pane: run, state, verdict, views digest. Remove pane, **+ add
pane** (picks from all runs with results), and a **REF** marker that can be
moved to any pane. REF is the reference for every Δ on the page.

### 5.2 Numbers

The §4 table restricted to the runs on the page, with the Δ row of every
non-REF run taken **against REF** (not against its own `compare_with`).

### 5.3 Image viewer

```
┌ field [cp ▾]  view [x-planes ▾]  ◀━━━━━━━●━━━━━━━━▶ x = −0.100   ← → step ┐
│ layout: (•) side-by-side  ( ) overlay      [+ add pane]                   │
├───────────────────────────┬───────────────────────────┬───────────────────┤
│ ☑ sync  c02_combo12  REF  │ ☑ sync  e05_crest  ☐ Δ    │ ☐ sync  t01b  ☑ Δ │
│        (image)            │        (image)            │ own slider x=+.04 │
└───────────────────────────┴───────────────────────────┴───────────────────┘
overlay:  A [c02 ▾]  B [e05 ▾]   (•) blink  ( ) swipe  ( ) fade   [B] / auto
```

- **Global controls:** field (cp, cpt, U, λ2 on planes; cp, y+ on the
  surface); view (x-, y-, z-planes; surface front, rear, left, right, top,
  bottom, iso); **plane slider** over the offsets that exist, ←/→ one step,
  Shift+←/→ five. Neighbouring planes are preloaded so scrubbing does not
  wait on the network.
- **Panes:** 1-3 by default, 4 allowed; side-by-side as a grid.
- **Sync per pane:** a ticked pane follows the global field, view, plane and
  zoom/pan. An unticked pane shows its own small controls and keeps its own
  position. Zoom (wheel) and pan (drag) are shared in normalised image
  coordinates - every picture of a view has the same frame, pixel for pixel.
- **Overlay mode:** two panes in one viewport - **blink** (button, `B` key,
  or automatic at an adjustable interval, default 0.5 s), **swipe**
  (draggable divider), **fade** (opacity slider). One click back to
  side-by-side.
- **Δ tick box per non-REF pane:** the pane shows *this run − REF* for the
  current view and field, computed on request (§6), as a picture with the
  same frame so it also works in blink and swipe. Colour limits default to
  cp ±0.2, cpt ±0.2, U ±10 % of u_inf, adjustable per field on the page.
- **Guards:** a warning chip when panes differ in `views_digest` (the
  pictures are not comparable) or in driving state (different u_inf); a
  placeholder when one run has no picture at that plane.

### 5.4 Charts (uPlot, run colours = pane colours)

- **Force histories:** Cl and Cd of every run against iteration; selector
  whole car / group / single patch; each run's averaging window shaded.
  Option **relative to window mean (%)** plots each curve as deviation from
  its own window mean with **±0.5 % and ±1 % bands**.
- **Δ by component:** horizontal bars of ΔCl and ΔCd against REF per group
  and per patch, with a noise whisker (§4.3).
- **cp lines:** one y station at a time (the stations in
  `post_views.yaml`), Body/Wing toggles, cp against x as points (as
  `viz/cplines.py` draws them), one colour per run; below it the cut outline
  z against x per run. When the viewer is on a y-plane that matches a
  station, the cp-line station follows it.

## 6. Field deltas

### 6.1 Planes

For run P (pane) and run R (REF), plane `axis_offset`, field f:

1. Take the slice entry from **R's** `results/render_plan.json`: plane
   point, normal, camera focal point, view-up, parallel scale; resolution
   from the plan.
2. Build a uniform grid of points covering exactly that camera frame (focal
   point projected onto the plane, right = view × up, half height = parallel
   scale, aspect from the resolution) at the picture resolution, so the delta
   picture lines up pixel for pixel with the normal pictures.
3. Triangulate both runs' `postProcessing/surfaces/<t>/<plane>.vtp` and
   probe them at the grid points (`vtkProbeFilter`, **tolerance 1e-4 m**:
   cut points sit up to ~50 µm off the plane and a tighter tolerance leaves
   5 mm holes).
4. Build cp, cpt and |U_rel| exactly as `viz/pv_render.py` does, with each
   run's own `frame` (u_inf, omega, origin).
5. Δ = P − R where both are valid. Solid in both: background grey. Fluid in
   exactly one: **black** - geometry that moved.
6. Colour with a diverging banded map (20 bands over ±limit), draw a colour
   bar where the pipeline draws its own, write PNG.

Measured in the spike: probing both runs takes **0.3-1.1 s per plane**.

### 6.2 Surface (cp)

1. Read both runs' `postProcessing/patchSurfaces/<t>/vehicle.vtp`.
2. Interpolate R's `pMean` onto **P's** points (`vtkPointInterpolator`,
   linear kernel, radius 0.5 mm, input arrays not passed through, null
   points masked). A point of P with no point of R within 0.5 mm is masked
   and drawn **black**: surface that is new or moved in P.
3. Δcp = (p_P − p_R) / (0.5 u_inf²), stored as `delta_cp` on P's surface in a
   cached `.vtp` per pair.
4. Render the requested surface view with the camera from P's render plan,
   diverging banded map, ±0.2 default.

Measured in the spike (2.5 M points each): read 8 s, interpolate 0.4 s,
render about 8 s per view. The first surface Δ of a pair therefore takes
about 15-20 s; the page shows "computing" in the pane. Later views of the
same pair only render.

### 6.3 Helper and cache

- `pipeline/simdev/viz/delta.py`, run under `post.paraview_python` (the
  interpreter that has VTK and ParaView), with a clean environment
  (`env -i HOME PATH`) - ParaView hangs under the OpenFOAM environment on
  this machine. Same contract as `viz/pv_render.py`: a JSON request in, the
  PNG written, one JSON summary line on stdout. It imports nothing from
  `simdev`.
- The FastAPI side (`simdev/ui/delta.py`) builds the request, runs the
  helper, and serves the PNG.
- Cache: `<P>/ui/delta/<R>/<view>_<field>_<limit>.png` plus
  `<P>/ui/delta/<R>/surface.vtp`. Key also includes both runs' `spec_hash`
  and `views_digest`; a mismatch recomputes. Deleted with P.
- One delta at a time (a lock); a second request waits. These are seconds of
  one core and must not compete with a solve for more.
- "Strip mesh" keeps `postProcessing/`, so deltas still work on stripped
  runs.

## 7. Spike evidence

`~/runs/ui-delta-spike-2026-10-05` (throwaway; `plane_delta.py`,
`surface_delta.py`, `out/`, `out_surface/`), c02_combo12 against
`av001-tc10-cornering-lowspeed-car-001`, same state and views digest.

### 7.1 Results

- Plane field deltas are clean and readable: the cpt wake difference over the
  rear deck (`y_+0.000`), the tyre wake shift under the car (`z_+0.010`).
- **Deltas decoded from the banded PNGs are useless** for small changes:
  every shifted band edge becomes a ±1-band ring (0.0625 cp). Hence §6 uses
  the sampled fields, never the pictures.
- The whole cp plane of c02 sits about +0.005 above the base (pressure
  level). A diverging map with a dead band at zero would hide it, so the
  band count is even (20) and zero is a band edge: +0.005 shows as the
  faintest positive band.
- The surface Δcp concentrates on the rear wing and rear deck; 0.39 % of the
  points are masked at 0.5 mm (sharp edges where the two surface meshes
  differ). |Δcp| p50 0.006, p95 0.045, p99 0.13.

### 7.2 Not yet tested

- A real geometry change (Body/Wing moved): the black masks of §6.1/§6.2 are
  expected to outline it, but no such pair existed for the spike. The first
  real design iteration checks it.

## 8. Error handling

- Runs without `result.json`: listed as "no results yet", excluded from Δ.
- Missing pictures, cp lines or `.vtp`: a placeholder in the pane with the
  reason, never a broken page.
- Delta helper failure: the pane shows the helper's stderr tail and a hint
  to run `simdev doctor`.
- `note.json` unreadable: treated as empty, shown with a warning; the next
  save rewrites it.
- `compare_with` pointing to a deleted run: shown as "missing", Δ row empty.

## 9. Testing

- Unit: noise (§4.3) on synthetic histories with known spread; Δ and the
  "within noise" decision; better-direction colouring.
- Unit: `note.json` read/write, atomic write, shell-run creation of `ui/`,
  job note copied by the worker.
- Unit: image index (field → view → offsets → file) from a synthetic
  `results/images` tree; views-digest guard.
- Route tests (`tests/test_ui_*` pattern, synthetic run directories):
  `/results`, inline note/compare-with edits, TSV copy, `/compare`, delta
  endpoint with the helper stubbed.
- Helper: two tiny synthetic `.vtp` planes with a known difference → known
  Δ; surface interpolation masks a displaced patch.
- Viewer JavaScript (sync, blink, swipe, fade, preload): a manual check list
  in the plan.
