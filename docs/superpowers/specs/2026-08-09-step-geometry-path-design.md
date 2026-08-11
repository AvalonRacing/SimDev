# STEP/STL Geometry Path — Design (SUPERSEDED, 2026-08-11)

**Status: superseded. Kept for the reasoning, not as instructions.** The
geometry path is built; see `docs/handbook.md` §6 "Bring in CAD" and
`cases/car/config.yaml` for what it actually became.

What changed between this document and the implementation:

- **No STEP.** The CAD now exports STL directly (31 parts, `Testcase.zip`), so
  §3.2's tessellation-inside-`prepare` design and the gmsh dependency are both
  moot. `geometry.scale` handles the millimetres.
- **The ride-height question in §4 answered itself** by moving upstream: parts
  are exported in assembly position with the tyres on the road, and
  `prepare` *checks* placement (`geometry.max_ground_penetration`) rather than
  inferring it. Option 1 in §4, in effect.
- **§3.3's `half_model` fix was implemented as proposed**, with the default
  `geometry.symmetric = False` chosen as the safe direction.
- **§5's open items** are resolved except tessellation quality, which the
  exporter now owns, and the car resolution profile, which exists (`car`,
  `car_dev`) but is sized on paper rather than measured.

One finding here does *not* transfer: the measurements in §2 are of the older
STEP files and disagree with the current STL export. Re-measure rather than
citing them.

---

## 1. Why this work exists

The pipeline validates end-to-end on the procedural Ahmed body. The real goal
is an RC car, whose geometry arrives as STEP files from NX. `geometry.kind:
stl` exists in the schema but **has never been exercised** — no test, no run.
It is the single largest unknown between here and a car result.

This is one of three independent subsystems the user asked for. The other two
are deferred (§7) and are **not blocked by this one**.

---

## 2. The CAD, measured

Files live in `benchmark_old_pipeline/CAD/` (untracked). Measured by
tessellating with gmsh's OCC importer, not by parsing raw `CARTESIAN_POINT`
entries — those include NURBS control points in local frames and gave a
bounding box roughly 6x too large. Do not trust a grep-based bounding box on
a STEP file.

| Part | solids | faces | x (mm) | y (mm) | z (mm) | extent (mm) |
|---|---|---|---|---|---|---|
| body | 1 | 996 | −252.40 … 194.86 | −119.26 … 117.89 | −0.66 … 117.10 | 447 × 237 × 118 |
| chassis | 14 | 848 | −222.07 … 198.73 | −132.33 … 117.17 | −4.50 … 65.38 | 421 × 250 × 70 |
| wing | 1 | 160 | 143.61 … 208.18 | −72.37 … 121.47 | 96.17 … 125.22 | 65 × 194 × 29 |

Facts that follow, each of which shapes the design:

- **Units are millimetres.** `SI_UNIT(.MILLI.,.METRE.)` in all three files.
  A scale of 0.001 is mandatory. The existing `check_geometry` unit-suspicion
  warning would fire on unscaled input, which is the correct behaviour.
- **Schema is AP203, exporter is NX2206.** All solids are
  `MANIFOLD_SOLID_BREP` over `CLOSED_SHELL` — zero open shells, so the
  geometry should tessellate watertight. This has **not** been confirmed on
  the resulting STL yet.
- **All three share one coordinate frame**, and it matches the pipeline's
  convention already: x streamwise, z up, car sitting near z = 0.
- **`chassis.stp` is a 14-solid assembly.** The other two are single solids.
- **Solid names are useless for patch mapping.** Every file reports the same
  product name, `ava_aero_setup_objects`. There is no per-solid name to split
  on, and splitting by solid index would be arbitrary and would break on any
  CAD revision. Patch identity must come from the *filename*.
- **The geometry is not symmetric about y = 0**, and inconsistently so: body
  centred to −0.7 mm, chassis to −7.6 mm, wing to **+24.5 mm** (about 10% of
  vehicle width).
- **Parts dip below z = 0**: body to −0.66 mm, chassis to −4.50 mm.

---

## 3. Decisions made

### 3.1 One STEP file, one patch

Forced by the missing solid names (§2). `chassis.stp`'s 14 solids merge into
a single `chassis` patch. This matches how `geometry.stl_dir` already maps
`{patch.name}.stl`, so the existing convention extends rather than changes.

### 3.2 Conversion happens inside `prepare`, with caching

`geometry.kind: step`, converted automatically during `prepare`, cached on a
hash of (source file + tessellation settings) so it only re-runs when
something actually changed. Converted STLs land in the run directory and go
through the existing watertightness and unit checks like any other geometry.

Rejected: a separate `simdev convert` command. It keeps `prepare` fast and
makes tessellation an inspectable artefact, but it is two commands and the
STLs become an untracked thing that can silently drift from the CAD.

### 3.3 The car runs full, straight, non-symmetric

User's decision. No symmetry plane, whole car meshed, `a_ref` stays the full
frontal area.

**This breaks the current `half_model` derivation and must be fixed as part
of this work.** Today:

```python
half_model = (mode is STRAIGHT) and (yaw_deg == 0.0)
```

A straight, zero-yaw case is therefore *forced* to be a half model, and
`validate()` then demands a `symmetry` patch and halves `a_ref`. That is
correct for the Ahmed body and wrong for this car, because the derivation
accounts only for symmetric **flow**, never for symmetric **geometry**.

Proposed fix, which keeps the property derived and never set:

```python
half_model = geometry.symmetric and (mode is STRAIGHT) and (yaw_deg == 0.0)
```

`geometry.symmetric` is a new declared property of the CAD. Ahmed declares
`true` and is unchanged; the car declares `false` and is correctly full. The
existing validator assertion (half model ⟺ symmetry patch exists) keeps
working in both directions.

This preserves the guarantee in handbook §3.3 — that a half model with a
full-model reference area is unrepresentable — while extending it to cover
asymmetric geometry, which the current rule silently gets wrong.

---

## 4. Open question — where the design stopped

**How the geometry is placed relative to the ground plane.** The chassis dips
4.50 mm below z = 0, about 4% of vehicle height. Ride height drives car aero
more than almost any other single parameter, so this was not guessed at.

Options put to the user, not yet answered:

1. **Explicit transform in config** (recommended) — a per-case scale/translate
   stated outright in the case file, plus a validator reporting where the
   geometry sits relative to the ground and failing beyond a stated
   penetration tolerance. Ride height stays a number chosen, not inferred.
2. **Auto-snap** so the lowest point sits at z = 0 — zero config, but sets
   ride height from whatever stray feature happens to be lowest.
3. **Ground plane at the geometry's lowest point** — preserves the CAD frame,
   but converts 4.5 mm of penetration into 4.5 mm of extra ride height and
   moves the ground off z = 0, which the rest of the domain assumes.

The user asked to clarify rather than answer, and the session ended before
they did. **Resume by asking the clarifying question, not by re-presenting
the options.** Specifically unresolved:

- Is z = 0 the intended ground plane in this CAD frame at all, or is the
  origin some other datum?
- What is actually dipping below zero — a modelled contact patch, a skid
  plate meant to touch, or a chamfer artefact on one of the 14 chassis
  solids? *This has not been measured. Identifying which solid is responsible
  is a few minutes' work and may make the question answer itself.*
- Is there a known ride height this car should be simulated at, in which case
  placement is simply "put it there"?

An earlier question on symmetry was also met with a clarification request
before the user answered it directly with "full, straight, non-symmetric"
(§3.3). Expect the same style: offer measurements before options.

---

## 5. Design areas not yet discussed

Reached in neither questions nor design. All still open:

- **Tessellation quality.** gmsh exposes angular and linear deflection. Too
  coarse gives faceted surfaces that change the aerodynamics; too fine gives
  an unusable STL. Needs config fields and defensible defaults, and the
  defaults should be justified against the 30 mm-scale features on this car.
- **Cache location and invalidation.** Where converted STLs live, and how the
  hash is composed.
- **Per-part transforms**, if parts turn out to need individual placement
  rather than one case-wide transform.
- **Patch roles for the car.** All three parts are `body` role for force
  integration. Whether `wing` should be its own patch for per-component
  forces ties into deferred subsystem §7.1.
- **A resolution profile for the car.** `base_cell_size` is absolute metres
  and `production` is sized for the 1 m Ahmed body; it will not transfer to a
  447 mm car. Needs its own profile.
- **`low_y_plus` has never been run.** The wall-profile machinery is unit
  tested, the physics path is not. The car is intended to use it.

---

## 6. How to resume

1. Re-read §2 (do not re-measure — it is expensive and the numbers are here).
2. Ask the clarifying question in §4 about ride height and the CAD datum.
3. Offer to identify which chassis solid dips below z = 0 before presenting
   options again.
4. Work through §5.
5. Then present the design for approval, write the approved spec, and move to
   `writing-plans`.

The brainstorming skill's hard gate still applies: no implementation until a
design is presented and approved.

---

## 7. Deferred subsystems

Both were requested in the same breath as this one and were deliberately
split out. Neither is blocked by the geometry path, and both would work on
the Ahmed case today.

### 7.1 Force groups and aero balance

Named force groups per component, plus aero balance from lift and pitching
moment. Note for whoever designs this: the old STAR-CCM+ pipeline computed
**no** balance and tracked only global cD/cL monitors averaged over the last
100 iterations (`benchmark_old_pipeline/Scripts/Evalquery.java`), so there is
no prior convention to inherit and the definition must be chosen deliberately.
"Multiple forces for aero balance" is ambiguous between a positional
front/rear split derived from the pitching moment and a per-component
breakdown; they are different things and the user may want both.

### 7.2 Slice post-processing and colour maps

Slices through the fluid domain at tight offsets, and a colour map setting in
the case config.

**A constraint worth raising early:** 10 mm offsets over this car is a lot of
images. The domain around a 447 mm car with a useful wake is order 2 m
streamwise; a 10 mm sweep is ~200 planes on one axis alone, and three axes
puts it in the high hundreds. That affects storage, render time, and whether
anyone looks at them. Worth agreeing a scheme — bounded ranges per axis,
coarser default with a fine sweep on request — before building it.

`pyvista` renders VTK offscreen without a ParaView install and is the
expected route. OpenFOAM's own `surfaces` function object can write the cut
planes during the solve, which avoids reconstructing the case.
