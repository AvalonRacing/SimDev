# Post-processing: Reports and Images — Design

**Status: approved, not yet implemented.** Written 2026-09-01.

Two-step post-processing for the car pipeline: numbers you can paste into a
spreadsheet, and pictures that can honestly be laid side by side across runs.
Step one extends the existing `post` stage. Step two is a new `images` stage.

---

## 1. Why this work exists

`post` today reports Cd and Cl means, a y+ verdict and three plots. Comparing
two driving states means reading two `result.json` files by eye, and there is
no image output of any kind — `docs/handbook.md` lists "full plane-cut image
suite" and "per-component forces, aero balance" as deferred with the hook in
place. This is that work.

Two requirements drive the whole design and neither is negotiable:

1. **A number must carry its own trustworthiness.** A row pasted into a
   spreadsheet outlives the terminal that printed it. If it does not say which
   iterations it averaged and whether the run converged, someone will compare
   a converged run against a stopped one and believe the difference.
2. **Two pictures must be comparable or they are worse than no pictures.**
   Same plane positions, same camera, same colour limits, same reference
   frame. A picture that silently auto-scales manufactures differences.

---

## 2. Decisions taken

Settled with the user before writing this; recorded so they are not
relitigated.

| Decision | Choice |
|---|---|
| COP convention | Motorsport ratios, plus front aero balance (§4.3) |
| Component groups | `body` = Body, `wing` = Wing, `other` = everything else (§4.4) |
| Slice/surface data path | OpenFOAM samples in parallel, ParaView renders the samples (§6.1) |
| Slice grid anchor | Datum from the `Chassis` patch, not the wheels (§6.3) |
| Report format | Tab-separated, one line per run, old-pipeline shaped (§4.5) |
| Slice suite | cp, cpt, U, normal vorticity, lambda2 on all three axes (§6.5) |
| Colour limits | Fixed per field in the shared views file (§6.4) |

---

## 3. Two findings that shaped it

### 3.1 ParaView's launchers hang on this machine; the module does not

`pvpython` and `pvbatch` (ParaView 6.0.1, Debian package) do not return, not
even for `pvpython --version` — measured, 150 s timeout, no output. But

```
python3 -c "import paraview.simple"   # /usr/bin/python3, python3-paraview
```

imports, creates a render view and writes an offscreen PNG in seconds. System
python3 is 3.14.4, the same version as the venv, though not the same
interpreter and without the venv's packages.

**Consequence:** the renderer runs under `/usr/bin/python3`, never under
`pvpython`, and it cannot import `simdev`. That forces a clean boundary
(§6.2) rather than being merely a workaround.

### 3.2 In a cornering run, `UMean` is the *absolute* velocity

`render/context.py:542` — "In a rotating frame OpenFOAM solves the absolute
velocity". The whole domain is one MRF cell zone. So the written `U`/`UMean`
of a cornering run is velocity in the **ground** frame, in which the far field
is at rest.

Slice that raw and put it beside a straight-line run and the entire freestream
changes colour. None of that is aerodynamics. §5 is the fix.

---

## 4. Part one — the report

### 4.1 What OpenFOAM must write

`forceCoeffs` writes coefficients only. F in newtons and M in newton-metres
need a **`forces` function object**, added to `controlDict.jinja` beside the
existing ones, aggregate only:

```
forces
{
    type            forces;
    libs            ("libforces.so");
    writeControl    timeStep;
    writeInterval   1;
    patches         (<force_patches>);
    rho             rhoInf;
    rhoInf          <spec.flow.rho>;
    CofR            <spec.forces.c_of_r>;
}
```

Cost is two more surface integrals per iteration against the fifteen
`forceCoeffs` objects already there — noise against the solve.

**F and M are not back-derived from Cd/Cs/Cl.** That would require assuming
OpenFOAM's mapping from CmRoll/CmPitch/CmYaw onto Cartesian axes, and this
repo does not assume conventions it can read directly. A `forces` object
writes x, y, z components and there is nothing to guess.

**No per-patch `forces` objects.** The user asked for per-group *coefficients*
only, and `forceCoeffs_<patch>` already supplies those. Eleven more force
objects would buy nothing and add eleven directories to `postProcessing/`.

`force.dat` and `moment.dat` are read **positionally**, not by header name,
with an asserted column count — the same reasoning as
`read_y_plus_area`'s comment: a header whose token count disagrees with the
data does not raise, it silently yields NaN. Capture a real v2412 pair as
`tests/fixtures/force.dat` and `moment.dat` during implementation and pin the
layout in a test rather than trusting this document's memory of it.

### 4.2 The window

Every reported quantity is the mean over **the convergence window `post`
already computes** (`gates/convergence.py`), which is also `fieldAverage`'s
`timeStart`. So the numbers in `report.tsv` and the fields behind the pictures
describe the same iterations. Nothing gets its own window.

### 4.3 COP and aero balance

A net force plus a net moment defines a *line* of action, not a point. The
chosen convention, relative to `spec.forces.c_of_r = (x_ref, y_ref, z_ref)`:

```
COP_x = x_ref − M_y / F_z      where downforce acts longitudinally
COP_y = y_ref + M_x / F_z      where downforce acts laterally
COP_z = z_ref + M_y / F_x      the height drag would act at
```

**These are three diagnostics, not the coordinates of one point.** `COP_x` and
`COP_z` are two different readings of the same pitching moment — `COP_x`
attributes all of `M_y` to downforce, `COP_z` attributes all of it to drag.
Both are standard and useful; together they are not a position. The record
carries `cop_convention: "ratio"` so a future reader cannot mistake them for
one.

Aero balance, with the car nose-forward along +x so `x_front > x_rear`:

```
balance_front_pct = (COP_x − x_rear) / (x_front − x_rear) × 100
```

`x_front` and `x_rear` are the mean x of the front and rear wheel centres,
which `geometry/wheels.py` already measures. COP at the front axle gives
100 %. Steering moves a wheel centre by a few millimetres in x and does not
matter here; the cornering attitude makes "wheelbase along x" approximate, and
the record says so.

**Guards.** Each ratio is undefined as its denominator vanishes. When
`|F_z| < 0.02 · q · A_ref` (i.e. a lift coefficient under 0.02) `COP_x`,
`COP_y` and `balance_front_pct` are written empty with a reason; likewise
`COP_z` on `|F_x|`. Empty, never a large meaningless number.

**Signs are OpenFOAM's, not flipped.** `liftDir` is `(0 0 1)`, so positive
`Cl` and positive `F_z` mean *up*. Downforce is negative. Silently flipping to
"downforce positive" here would disagree with `results/forces.png`,
`result.json` and every existing run.

### 4.4 Groups

| Group | Patches |
|---|---|
| `body` | Body |
| `wing` | Wing |
| `other` | Chassis, SUS_FL, SUS_FR, SUS_RL, SUS_RR, Tire_FL, Tire_FR, Tire_RL, Tire_RR |

Declared in the case config as `post.groups`, validated to cover every
force-bearing patch **exactly once** — a patch in no group or in two is a
config error, not a silent miscount. Because `controlDict` gives every
`forceCoeffs_<patch>` the vehicle's own `Aref`, `lRef` and `CofR`, each
patch's Cd and Cl is its *share* of the total, so groups are plain sums and

```
cd_body + cd_wing + cd_other == cd
```

holds to floating-point. That identity is asserted in a test and makes every
pasted row self-checking.

### 4.5 Output

`results/report.tsv` — one header line, one data line, tab-separated, in the
shape of the old pipeline's `Auswertung_<version>.txt` so it drops into an
existing sheet. **Column order is a contract** and is pinned by a test; new
columns are appended, never inserted.

| Block | Columns |
|---|---|
| identity | `run`, `case_name`, `driving_state`, `spec_hash`, `timestamp` |
| trust | `verdict`, `converged`, `window_start`, `window_end`, `n_iterations`, `cd_amplitude`, `cl_amplitude`, `yplus_passed`, `n_cells` |
| forces (N) | `Fx`, `Fy`, `Fz` |
| moments (N·m) | `Mx`, `My`, `Mz` |
| coefficients | `cd`, `cd_std`, `cl`, `cl_std`, `cs` |
| centre of pressure (m, %) | `COP_x`, `COP_y`, `COP_z`, `balance_front_pct` |
| components | `cd_body`, `cl_body`, `cd_wing`, `cl_wing`, `cd_other`, `cl_other` |

The trust block is not optional decoration. It is the difference between a
comparison and a coincidence.

`simdev report ~/runs/car-* -o summary.tsv` concatenates per-run files on
read. It **never appends to a shared file** — `report/results.py` already
establishes that rule and `aggregate()` already implements the pattern.

`ResultRecord` gains the same fields, all with defaults so existing
`result.json` files still load.

**Where the image-side provenance lives.** `post` runs before `images` and
cannot know the views hash, so `images` writes its own
`results/images.json`: datum, views-file hash, image count, per-image clamped
fraction, and sampling and rendering wall times. `simdev report` reads both
files, joins them per run, and is what raises the mismatched-datum warning of
§6.3. `ResultRecord` is not stretched to carry any of it.

**A run made before the `forces` object existed** has no `force.dat`. Its
force, moment and COP columns are written empty with a reason recorded, and
`post` still succeeds. `post` does not gate on this; missing pictures and
missing newtons are not a failed run.

### 4.6 One new plot

`results/balance.png` — `COP_x` and `balance_front_pct` against iteration,
with the averaging window shaded, alongside the existing `forces.png`.
Balance is the number that actually gets tuned, and whether it swings half a
percent or five across the limit cycle is invisible in a windowed mean.

---

## 5. Part two — the reference frame the pictures are drawn in

Everything here reduces to the straight-line expression at `Ω = 0`, so there
is one code path and the cornering case is not a special branch.

Let `Ω` be the frame's angular velocity vector, `origin` the corner centre
(`domain.centre`, z = 0), `r` the distance from that axis in the xy-plane, and
`U_inf = spec.flow.u_inf`.

```
U_rel   = UMean − Ω × (x − origin)        car-frame velocity  (= UMean when Ω = 0)
U_ff(r) = |Ω| · r                         undisturbed car-frame speed (= U_inf when Ω = 0)

cp      = pMean / (0.5 · U_inf²)
cpt     = (pMean + 0.5·|U_rel|² − 0.5·U_ff(r)²) / (0.5 · U_inf²)
|U|     = |U_rel|
```

`pMean` is kinematic (m²/s²), so no `rho` appears. The outlet fixes `p = 0`
and the far air is at rest in the ground frame — genuinely at rest, therefore
at uniform pressure, with no centrifugal gradient to correct for — so
`p_ref = 0` in both modes.

**Why `cpt` subtracts a *local* head.** At R = 4 m and ω = 3.75 rad/s the
undisturbed car-frame speed runs 12.75–17.25 m/s across the domain half-width
of ±0.6 m. Referenced to a constant `U_inf`, that is a spurious ±0.32 in
`cpt`, graded radially, and it reads exactly like a wake. Subtracting
`0.5·U_ff(r)²` puts the freestream at `cpt = 0` everywhere, so negative `cpt`
means total-pressure loss and nothing else.

**Why `cp`'s denominator stays `0.5·U_inf²`.** That is the same dynamic head
that non-dimensionalises Cd and Cl in `forceCoeffs`. A cp picture and a
coefficient then sit on one scale. Only `cpt`, whose entire purpose is
measuring loss against the local freestream, takes the local reference — and
it takes it in the numerator, where it belongs.

`vorticity` and `Lambda2` are computed from `UMean` (absolute). Absolute and
relative vorticity differ by the constant `2Ω` ≈ 7.5 s⁻¹ against a 2000 s⁻¹
plotting range, and λ₂ by `O(Ω²)` ≈ 14 against 5×10⁴. Both are negligible and
the vortex structures are identical; taking the free simplification is
deliberate and recorded here so it is not later mistaken for an oversight.

**Division of labour, and it follows from what a slice can know:**

| Quantity | Computed by | Because |
|---|---|---|
| `vorticity`, `Lambda2` | OpenFOAM, volume pass | needs 3D velocity gradients, unavailable on a 2D cut |
| `U_rel`, `cp`, `cpt`, `\|U_rel\|` | the renderer, per point | pointwise algebra on fields the slice already carries |

So the frame arithmetic lives in exactly one place — the renderer — with `Ω`,
`origin`, `U_inf` and the colour limits arriving in `render_plan.json`.

---

## 6. Part three — the images

### 6.1 Data flow

```
solve  (decomposed, ~20 M cells, never reconstructed)
   │
   ├─ postProcess -parallel -latestTime  ──►  vorticity, Lambda2  (volume fields)
   │
   ├─ postProcess -parallel -func surfaces ──►  results/samples/**.vtp   (~50 MB)
   │
   └─ /usr/bin/python3 viz/pv_render.py render_plan.json ──►  results/images/**.png
```

`mesh` decomposes once and nothing ever calls `reconstructPar`
(`docs/handbook.md` §"Parallel strategy"). Sampling in parallel respects that:
every rank cuts its own cells and the master writes the surface. The
alternative — ParaView's OpenFOAM reader pulling 20 M cells through one serial
Python process — is both slower and memory-bound, so this choice wins on
performance, which is what the user asked to optimise for. That the ~50 MB
sample archive can be re-rendered later without the mesh is a side effect, not
the justification.

The `vorticity` and `Lambda2` objects are declared **before** `surfaces` in one
dictionary. Function objects execute in dictionary order; the other order finds
nothing and reports nothing, with no error. This is exactly the
`yPlus`/`yPlusArea_*` ordering hazard already documented in
`controlDict.jinja`, and it gets the same treatment: an ordering assertion in
the render test.

Both are `fieldExpression` objects and take `field UMean`. **Verify against
v2412 during implementation** that `field` selects the input and that the
result name is controllable; fall back to a `postProcess -field UMean` pass if
not.

### 6.2 The venv / ParaView boundary

`viz/pv_render.py` imports `paraview.simple` and **nothing from `simdev`**. It
reads one argument: the path to `render_plan.json`.

```
venv side (testable)                    system python3 side
─────────────────────                   ───────────────────
views.py    parse post_views.yaml
datum.py    Chassis datum
plan.py     ──► render_plan.json  ────►  pv_render.py  ──► PNGs
              plane positions
              camera vectors
              colour limits
              Ω, origin, U_inf
              output paths
```

Everything except pixel-pushing is pure data and unit-tested in the venv. The
renderer contains no case logic, no unit conversion and no frame arithmetic
beyond evaluating the expressions the plan hands it.

The interpreter path is configuration (`post.paraview_python`, default
`/usr/bin/python3`) and `simdev doctor` grows a check that it exists and can
import `paraview.simple`. Given §3.1, an environment check here is not
paranoia.

### 6.3 The datum

```
datum = (Chassis bbox centre x, Chassis bbox centre y, 0.0)
```

z is the ground plane, which is physically fixed at 0 in every case.

`Chassis` because the handbook calls it an aero dummy that does not represent
the real car and that nobody iterates — so its bounding box is stable while
Body and Wing are redesigned, which is the whole point. Wheels were rejected
by the user: steering angle moves them.

Measured in `prepare`, which already loads every surface and computes bounds,
and recorded in the run directory beside `caseSpec.json` — not inside the
spec, whose hash must not move for this. Each run records its datum; `simdev
report` warns when runs being compared do not share one, because two
pictures anchored to different datums are not comparable however identical
their file names.

Measured on `CAD/Testcase` (millimetres, scaled by 0.001 into the case):
Chassis spans x −199.9…207.2, y −86.9…112.9, giving a datum near
(3.6, 13.0, 0.0) mm — close enough to the CAD origin that plane offsets read
naturally.

### 6.4 `cases/post_views.yaml`

One file, versioned in git, shared by every run — this is the "somewhere else"
the user asked for. It is copied into each run's `results/` and hashed into the
result record, so any picture can be traced to the definition that produced it.

```yaml
datum:
  patches: [Chassis]
  z: ground

planes:                                  # offsets from the datum, metres
  x: {from: -0.30, to: 0.32, step: 0.02}   # 32 planes; car spans -0.215..0.242
  y: {from: -0.20, to: 0.20, step: 0.02}   # 21 planes; car spans -0.131..0.131
  z: {from:  0.00, to: 0.16, step: 0.01}   # 17 planes; car spans  0.000..0.130

fields:
  cp:      {range: [-3.0, 1.0],      colormap: coolwarm}
  cpt:     {range: [-3.0, 1.0],      colormap: coolwarm}
  U:       {range: [0.0, 22.5],      colormap: viridis}   # 1.5 x u_inf
  vort:    {range: [0.0, 2000.0],    colormap: inferno}
  lambda2: {range: [-5.0e4, 0.0],    colormap: inferno}
  yplus:   {range: [0.0, 5.0],       colormap: viridis}   # the gate band

camera:
  parallel_scale: {x: 0.35, y: 0.35, z: 0.30}
  resolution: [1600, 1200]

streamlines: off        # off | lic | seeded
```

Plane positions are **fixed, never auto-fitted**. Start, end and spacing do not
move between runs. If the geometry pokes outside the requested range the stage
records a reason saying the slices do not cover the car — it does not silently
extend, because an extended range is a different picture wearing the same file
name.

Values outside a colour range clamp to the end colour, and the fraction of the
image that clamped is recorded, so a badly chosen limit is visible rather than
merely invisible.

### 6.5 The suite

| Planes | Fields | Count |
|---|---|---|
| x (32) | cp, cpt, U, vort_x, lambda2 | 160 |
| y (21) | cp, cpt, U, vort_y, lambda2 | 105 |
| z (17) | cp, cpt, U, vort_z, lambda2 | 85 |
| surfaces (7 views) | cp, yplus | 14 |
| | **total** | **364** |

`vort_n` is the vorticity component **normal to its own plane** — the
component that shows streamwise vortices punching through the cut, and what
the old pipeline plotted. The other two components on a given plane are mostly
shear layer and read as noise.

### 6.6 Cameras

Parallel projection with an explicit `CameraParallelScale`. **Never
"reset to fit data"** — a fitted camera silently rezooms when a wing gets
longer, and then two runs are drawn at different magnifications with nothing
saying so.

Slice cameras track their plane along its own normal and are pinned laterally
to the datum, so the car sits in the same pixels every time:

| Planes | Viewed from | Direction | Up | Image reads |
|---|---|---|---|---|
| x | downstream | +x | +z | looking upstream; car-left (+y) on the right |
| y | car's right (−y) | +y | +z | nose to the right |
| z | above | −z | +x | plan view, nose up |

**The surface views are in the car frame too, on the same terms as the
slices** — focal point on the datum, fixed `CameraParallelScale`, direction
vectors in car axes. Nothing about them refers to the domain. That is what
makes a `front.png` from a cornering state and one from a straight-line state
the same picture of two different flows rather than two different pictures.
Because the geometry is never transformed, car axes *are* mesh axes and this
costs nothing to arrange; the datum is what supplies the centring.

Nose is +x and up is +z, so **car-left is +y** (confirmed by the user against
the CAD). The yawed, steered attitude baked into `CAD/Testcase` does not
change that: the attitude is in the geometry, not in the frame.

| View | Camera direction | Up | Sees |
|---|---|---|---|
| `front` | −x | +z | the nose |
| `rear` | +x | +z | the wing |
| `left` | −y (camera at +y) | +z | the car's left flank |
| `right` | +y (camera at −y) | +z | the car's right flank |
| `top` | −z | +x | plan view, nose up |
| `bottom` | +z | +x | floor, nose up |
| `iso` | (−1,−1,−1)/√3 (camera front-left-above) | +z | three-quarter from the car's left |

### 6.7 Every image is stamped

A small corner block on each PNG: run name, `spec_hash[:8]`, views-file hash,
averaging window, field, plane offset from the datum, colour range, and
`MEAN` or `INSTANTANEOUS`.

Two pictures a month apart are worthless if you cannot tell which run, which
window and which colour limits made them. The stamp costs nothing and is the
difference between an archive and a folder of pretty pictures.

`INSTANTANEOUS` appears when `*Mean` fields are absent — a run stopped before
`fieldAverage`'s `timeStart`. Those still get pictures; they are just labelled
as the single arbitrary phase of a limit cycle that they are.

### 6.8 Streamlines

`streamlines: off` by default. The sampled `.vtp` carries `U_rel` as a vector,
so in-plane LIC or seeded traces work on the slice without returning to the
volume. Enabled behind the config key, timed on a real case, and the measured
cost reported — the user's own condition was "depends on rendering time", so
it gets measured rather than guessed.

### 6.9 Layout

```
results/
  report.tsv
  views.yaml                              copy of the definition used
  samples/{x,y,z,surface}/*.vtp
  images/
    slices/x/cp/cp_x_+0.120.png
    slices/y/vort/vort_y_-0.040.png
    surface/cp/{iso,front,left,right,top,rear,bottom}.png
    surface/yplus/...
  index.html                              contact sheet
```

Signed, fixed-width positions so the files sort correctly and two runs'
directories align name-for-name. That alignment is what makes a future
`simdev compare` a small job.

---

## 7. Architecture

```
pipeline/simdev/
  report/
    forces.py        F, M, COP, balance, group sums      (pure pandas)
    tsv.py           per-run report.tsv, aggregate
  viz/
    views.py         post_views.yaml -> Views            (pure data)
    datum.py         Chassis datum
    plan.py          Views + spec + datum -> render_plan.json
    sample.py        render the sampling dict, run postProcess
    pv_render.py     STANDALONE, system python3, imports no simdev
  stages/
    images.py        the new stage
  render/templates/
    controlDict.jinja      + forces function object
    sampleSurfaces.jinja   new: vorticity, Lambda2, surfaces (in that order)
cases/
  post_views.yaml    new, shared, versioned
```

`viz/` rather than extending `render/`: `render/` means "turn a spec into
OpenFOAM dictionary text" throughout this repo and the handbook's dependency
rule depends on that meaning. Sampling dictionaries are dictionary text and do
belong in `render/templates/`; pictures do not.

CLI: `simdev images <run-dir>` as a stage; `simdev report <run-dirs...> -o
summary.tsv` as an aggregate command beside the existing `aggregate`.
`--axes` and `--fields` subset flags on `images`, for iterating on one view
without paying for 364.

No new venv dependency. TSV needs nothing; ParaView is a system package.

**The implementation plan should phase this.** Part one (§4) ships and is
useful on its own — it needs only the `forces` object, a parser and a writer,
and it unblocks spreadsheet work immediately. Part three (§6) is the larger
half and depends on part two (§5) but not on part one. Two phases, part one
first.

---

## 8. Failure handling

Follows the handbook's rule — mesh quality blocks, everything downstream flags.

| Situation | Behaviour |
|---|---|
| `force.dat` absent (pre-`forces` run) | F/M/COP empty, reason recorded, `post` succeeds |
| COP denominator below guard | that COP empty, reason recorded, `post` succeeds |
| A force patch in no group, or in two | config validation error, before anything runs |
| `*Mean` fields absent | pictures rendered from instantaneous fields, stamped `INSTANTANEOUS` |
| Slice range does not cover the geometry | reason recorded, images still produced |
| ParaView interpreter missing or unimportable | `images` raises `StageError` naming the interpreter; `doctor` catches it earlier |
| `images` fails outright | `images` status failed; `post`, `report.tsv` and the run are untouched |

`images` never gates a run. They are pictures.

---

## 9. Testing

| Test | Asserts |
|---|---|
| `test_forces.py` | positional parse of a real v2412 `force.dat`/`moment.dat`; window mean; sign convention (positive `F_z` is up); missing file gives empty + reason |
| `test_cop.py` | golden COP from synthetic F/M; vanishing `F_z` gives empty + reason; balance is 100 % when `COP_x` is at the front axle |
| `test_groups.py` | `cd_body + cd_wing + cd_other == cd`; uncovered patch and doubly-covered patch both rejected |
| `test_report_tsv.py` | column order is stable; aggregate emits one row per run; a run with no record is skipped loudly |
| `test_views.py` | plane list is exactly the configured offsets; two datums shift positions by exactly the datum delta; geometry outside the range warns |
| `test_render_plan.py` | plan is fully determined; `Ω = 0` collapses `cpt` to the straight-line form; camera vectors per §6.6 |
| `test_render_sample_dict.py` | dict renders; plane count matches; `vorticity` and `Lambda2` precede `surfaces` (mirrors the existing `yPlus`/`yPlusArea` ordering test) |
| `test_pv_render.py` | marked `paraview`, like the existing `openfoam` marker: renders one synthetic `.vtp`, asserts a non-uniform PNG of the configured size |

The `paraview` marker joins `openfoam` in `pyproject.toml`, so the suite still
runs on a machine with neither.

---

## 10. Out of scope

- **Cross-run image comparison.** `simdev compare`, side-by-side pairs and
  difference images. Different meshes sample different polygons, so a true
  delta needs resampling onto a common grid — separate work, worth doing once
  this proves out. §6.9's file naming is chosen to make it small.
- **Per-patch forces and moments in newtons.** Per-group coefficients cover
  what was asked; eleven more `forces` objects can be added later without
  redesign.
- **Volume renderings**: Q-criterion isosurfaces, 3D streamlines, anything
  needing the full mesh in ParaView. The sample archive does not support it and
  nothing here asks for it.
- **Rendering during the solve.** Sampling and rendering are post-hoc.

---

## 11. Open risks

- **The `pvpython` hang is unexplained.** Routing around it via
  `/usr/bin/python3` works today; a ParaView upgrade could change either
  behaviour. The `doctor` check is the early-warning.
- **v2412 specifics, confirmed 2026-09-01 against a real run**
  (`car_smoke`, OpenFOAM v2412, `CAD/Testcase`):
  `postProcessing/forces/0/force.dat` is ten columns — `Time` plus
  `total_x/y/z`, `pressure_x/y/z`, `viscous_x/y/z`, total first, exactly as
  `run/parsers.py` assumes. `fieldExpression`'s `field` entry is mandatory (no
  default), and `result` defaults to a bracketed name built from the type and
  its field (e.g. `vorticity(UMean)`) if not set explicitly — both are set
  explicitly in `sampleSurfaces.jinja` for exactly that reason. `postProcess
  -dict` **merges** its dictionary into the run's `controlDict` rather than
  replacing it (`functionObjectList.C:433`); `render/context.py::solver_function_names`
  exists to enumerate every solve-time object so the sampling pass can
  disable all of them, and `viz/sample.py::run_sampling` re-checks
  `forceCoeffs` output timestamps after sampling as a belt-and-braces guard
  against one slipping through. `yPlus` is present in the time directory
  (`processor*/<time>/yPlus`) and sampled without issue. None of these three
  are assumptions any more.
- **Sampling cost, measured only at smoke scale.** On the `car_smoke` profile
  (51,762 cells, 4 ranks decomposed, 70 planes + 1 merged patch surface),
  `postProcess -dict system/sampleSurfaces -latestTime` took **1.5 s**, and
  rendering the resulting 350 PNGs under `/usr/bin/python3 viz/pv_render.py`
  took **191.9 s** (≈0.55 s/image). The on-disk sample archive was **5.1 MB**
  in `postProcessing/surfaces/50/` — note that is where the samples actually
  land; `results/samples/` (§6.1's diagram) is never created, because
  `viz/sample.py` reads the samples from OpenFOAM's own
  `postProcessing/<functionObjectName>/<time>/` output rather than copying
  them into `results/`. **This does not resolve the risk as originally
  framed.** The risk was sampling cost on the ~20 M-cell `car` production
  mesh, and no production mesh has ever been solved (handbook §9, "Not
  verified", item 1) — that number is still unmeasured, because there is
  nothing to measure it against yet. Do not scale the smoke-scale number
  linearly and report it as a production estimate.
- **New: the merged `vehicle` patch surface is not found by the `images`
  stage, found 2026-09-01 on the first real run.** The 7 `surface` views
  (front/rear/left/right/top/bottom/iso) × 2 fields (cp, yplus) — 14 of 364
  planned images — were not drawn. `viz/sample.py::run_sampling` returns
  `sorted((run_dir / "postProcessing" / "surfaces").glob("*"))[-1]` as
  `samples_root`, but the `surfaces`-type function object producing the
  merged `vehicle` patch surface is named `patchSurfaces` in
  `sampleSurfaces.jinja`, and OpenFOAM writes each function object's output
  under `postProcessing/<its own name>/<time>/` — so `vehicle.vtp` lands in
  `postProcessing/patchSurfaces/50/`, a directory `run_sampling` never looks
  in. `find_sample(samples_root, "vehicle")` then finds nothing, `images`
  logs the miss as a note rather than failing (correctly — these are
  pictures, not a gate), and `results/index.html` still links all 364
  filenames from the render plan, so the 14 missing ones 404. The 350 slice
  images are unaffected. Not fixed here — this task is documentation-only —
  but it is a real, measured defect and not a mesh-coverage artifact.
- **`lambda2`'s default colour range is a guess.** Expect to tune it once
  against a real field; the range is in a versioned file precisely so the
  tuning is recorded rather than remembered.
- **§5 assumes the far field is at rest in the ground frame**, which is what
  the MRF absolute-velocity formulation gives. If the cornering frame is ever
  reimplemented in relative velocity, `cpt` inverts silently. `viz/plan.py`
  should assert the formulation it is reading.
- **The `other` group mixes an aero dummy with the tyres and suspension.** Its
  coefficient is a bookkeeping remainder that makes the row sum, not a physical
  quantity to design against. Splitting it later is a config change.
