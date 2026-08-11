# SimDev Pipeline Handbook

How the pipeline works, **why** it works that way, and how to change it.

The spec (`docs/superpowers/specs/2026-08-09-openfoam-rc-car-cfd-pipeline-design.md`)
says what we decided. The plan
(`docs/superpowers/plans/2026-08-09-openfoam-pipeline-vertical-slice.md`) says how
to build it. This handbook is for the person who has to live with it afterwards.

> **Status:** §5 (module map), §6 (how to change things) and §9 (what is not
> here yet) describe the code as built, including the cornering and CAD work.
> §8 (debugging) still describes intent and needs a pass once real runs have
> failed in real ways.

---

## 1. What this is, and what it replaces

An automated OpenFOAM pipeline for external aerodynamics of 1/10–1/8 scale RC
cars: geometry in, validated force coefficients out.

It replaces a STAR-CCM+ workflow (kept for reference in
`benchmark_old_pipeline/`) that ran as three chained SLURM jobs driven by Java
macros against a 49 MB binary `.sim` file. That pipeline worked, and several of
its ideas are carried over deliberately. But it had four properties we
specifically designed against, and most of the architecture here only makes
sense once you know what it is defending against:

| Old behaviour | Consequence | What we do instead |
|---|---|---|
| Fixed iteration count, then average the last 100 regardless (`Evalquery.java:171`) | Reported coefficients from runs that never converged | **Convergence gate** (§3.4) |
| `avgIterations=300` in config, `100` hardcoded in the macro | The config file did not describe the run | **One resolved `CaseSpec`**, no downstream defaults (§3.1) |
| `// TODO Bug?? values are not multiplied by 2` on the symmetry factor (`Evalquery.java:703`) | Half-car coefficients possibly 2× wrong, unresolved for years | **`a_ref_effective` derived, asserted** (§3.3) |
| All runs appending to one `Auswertung_*.txt` with no locking | Corrupted results table under parallel sweeps | **Per-run records, aggregated on read** (§3.5) |
| Physics locked inside an undiffable binary | No review, no history, no explaining a result a year later | **`caseSpec.json` provenance** (§3.1) |

The single most valuable thing this pipeline does is make those five failures
structurally impossible rather than merely discouraged.

---

## 2. The mental model

Everything flows one direction. There are no back-edges.

```
  cases/<name>/config.yaml ─┐
  resolution profile ───────┼──► resolve ──► CaseSpec ──► validate ──► render ──► run dir
  wall-treatment profile ───┤                (explicit)    (asserts)    (jinja)
  CLI overrides ────────────┘                    │
                                                 └──► caseSpec.json  (provenance)

  run dir ──► prepare ──► mesh ──► solve ──► post ──► results/result.json
                  │          │        │         │
                 gate       gate     gate      gate
              (validation, (quality) (plateau) (y+ band)
               blockage)
```

Three ideas carry the whole design:

**1. One resolved object.** Config layers merge into a `CaseSpec` in which
*every value is explicit*. Nothing downstream ever applies a default or reads
raw config. If you want to know what a run did, you read one JSON file.

**2. Roles, not names.** Every surface carries a *role* (`body`, `tyre`,
`ground`, `symmetry`, `farfield`, `inlet`, `outlet`, `mrfZone`). The role drives
four things that would otherwise be edited in four different files and silently
drift apart: its boundary condition, its refinement level, whether it counts
toward forces, and whether it participates in MRF.

**3. Gates, not hope.** Each stage ends in a check that must pass before the
next begins. A gate failure stops the chain and is recorded — it is never a
warning you scroll past.

---

## 3. Why the major decisions

### 3.1 Why a resolved `CaseSpec` instead of reading config everywhere

Because "the config says 300 but the code uses 100" is a real bug that shipped,
and it is undetectable by reading either file alone.

Resolution happens in exactly one place (`config/resolve.py`), merging in a
fixed order:

```
built-in defaults ◄ resolution profile ◄ wall-treatment profile ◄ case file ◄ driving state ◄ CLI overrides
```

The driving-state layer lives *inside* the case file and is merged by
`resolve.py` like any other layer, so switching between tight cornering, a
sweeper and braking is one line rather than four edits that have to agree.

The result is validated, then written to the run directory as `caseSpec.json`
with a hash. Every results record carries that hash, so a result can never be
attached to the wrong specification.

**Consequence you must respect when extending:** if you find yourself writing
`config.get("something", default)` anywhere outside `resolve.py`, you are
reintroducing the bug. Put the default in `config/profiles.py`.

### 3.2 Why patch roles

The old pipeline needed an explicit name filter to keep MRF helper surfaces out
of force integration:

```java
new NamePredicate(NameOperator.DoesNotContain, MRFNAME)   // Evalquery.java:105
```

That works until someone names a part badly. Here, `mrfZone` simply has
`in_forces=False` in one table (`geometry/roles.py`), and every consumer reads
that table. The ground cannot accidentally enter the force sum; a wing cannot
accidentally leave it.

Roles are also what make cornering and rotating wheels *additive* rather than a
rewrite. `tyre` and `mrfZone` already exist in the enum and are unused by the
current slice — adding wheels later is a config change plus an STL, not surgery
on the BC writer.

### 3.3 Why `a_ref` is stored as the full-vehicle area, always

This is the canonical CFD error: a symmetry half-model with a full-model
reference area reports every coefficient 2× too high, and the case still runs,
still converges, and still looks plausible.

So the pipeline makes it unrepresentable:

- `ForcesConfig.a_ref_full` is **always** the full-vehicle frontal area.
- `CaseSpec.a_ref_effective` is the only place a halving ever happens.
- `half_model` is a *derived property*, never a config field:

  ```python
  half_model = geometry.symmetric and (mode is STRAIGHT) and (yaw_deg == 0.0)
  ```

- A validator asserts the patch set agrees: half model ⟺ a `symmetry` patch
  exists. Neither can be true alone.

**Note the yaw term.** Yaw breaks symmetry even in straight-line mode. This is
easy to get wrong because "straight line" sounds symmetric. Cornering is not the
only asymmetric case.

**Note the geometry term, which was missing and was a real bug.** The rule
above once read `half_model = straight and yaw == 0`, accounting only for
symmetric *flow* and never for symmetric *geometry*. Any straight zero-yaw
case was therefore forced to be a half model whatever it was a model of, and
`a_ref_effective` was halved for a car with no symmetry plane. The RC car is
asymmetric by up to 24.5 mm, about 10% of its width, so this would have
reported every coefficient 2× high in a run that meshed, converged and looked
entirely ordinary — the exact failure §3.3 exists to prevent, arriving through
the one door that was left open.

`geometry.symmetric` is the third input, declared per case. It **defaults to
`False`**, which is the direction that cannot corrupt a result: a full model of
symmetric geometry merely costs cells, while a half model of asymmetric
geometry is silently wrong. The Ahmed case declares `true` explicitly.

### 3.4 Why the convergence gate reports a windowed mean

Steady aero forces routinely keep drifting long after residuals hit 1e-4.
Residuals alone lie. So:

- The gate tests **Cd and Cl** for a plateau: relative scatter *and* least-squares
  drift over a trailing window, both against a tolerance.
- The **reported coefficient is the mean over that window**, with its standard
  deviation and window bounds recorded — never the last instantaneous iteration.
- A run that hits `max_iterations` without plateauing **terminates and is
  recorded as `converged=False`, with its means still populated.** It never
  hangs, never silently passes, and never vanishes.
- `residualControl` in `fvSolution` is set from `solve.residual_tol`, which
  defaults to **1e-6 — a safety net, not the stopping criterion.** This is the
  subtle half of the argument above: `simpleFoam` *stops* when
  `residualControl` is met, so leaving it at the 1e-4 where a steady aero case
  "looks converged" hands the stopping decision straight back to residuals.
  It also makes the remedy in §8 a lie, because raising `max_iterations`
  cannot help a run that is being stopped by something else.

That last property matters more than it looks: a non-converged run still
produced information, and throwing it away is how people end up re-running the
same failing case three times. The `post` stage therefore accepts a
`gate_failed` solve as valid input. Only a *failed* solve blocks it.

### 3.5 Why per-run result records

Parametric sweeps are the point of a pipeline. Two runs finishing simultaneously
and appending to one file interleave their rows — the old pipeline did exactly
this (`Evalquery.java:685-687`), including a race on the header line.

So each run writes `results/result.json` and a one-row `results/result.csv`, and
`aggregate()` combines them **on read**. There is no shared file to corrupt.

### 3.6 Why wall treatment is a profile, not a constant

Because the right answer depends on Reynolds number, and this project spans two
very different ones.

At 1/10 scale (L ≈ 0.40 m, 15 m/s):

| | body | appendage (chord ≈ 35 mm) |
|---|---|---|
| Re | 4.0×10⁵ | 3.5×10⁴ |
| δ (turbulent estimate) | ≈ 11 mm | ≈ 1.6 mm |
| first cell for y⁺ = 30 | ≈ 1.3 mm | ≈ 1.3 mm |
| first cell / δ | 12 % | **80 %** |

High-y⁺ wall functions are *invalid* on appendages at this scale — the first
cell swallows most of the boundary layer exactly where downforce is generated.
So the RC car runs `low_y_plus` (y⁺ ≈ 1, ~40 µm first cell, 15–20 layers).

But the Ahmed validation case runs at Re ≈ 2.8×10⁶, where wall-resolving means a
~10 µm first cell and a mesh too heavy to iterate against. So it runs
`high_y_plus`, which is also what most published CFD comparisons use.

Same code path, different profile. The **y⁺ gate enforces whichever profile is
active**, so the two can't be confused.

### 3.7 Why the ground is a per-case field

Instinct says "moving ground, always" — static ground under a moving vehicle is
a genuine Critical for ground-effect aero. But the Ahmed body's reference
experiment used a **stationary tunnel floor** with the model on stilts. Running
it with a moving belt would not reproduce the published Cd, and you would not be
able to tell whether your pipeline or your BC was wrong.

So `ground.motion: static | moving` is per-case. The validator keeps the safety
rule as a **loud warning on vehicle cases**, not a hardcoded behaviour.

It renders as `fixedValue uniform (U 0 0)`, not `movingWallVelocity` —
the latter exists for cases where the *mesh* moves. On a static mesh with a
translating road, `fixedValue` expresses the same physics correctly.

### 3.8 Why exit codes are not trusted

OpenFOAM frequently exits 0 on partial failure. The old job chain used
`sbatch -d afterok:`, which only checks exit codes, so a degraded surface mesh
satisfied the dependency and launched a 240-core solve on it.

`Runner` therefore checks the exit code **and** scans the log for `FOAM FATAL`,
floating-point exceptions, and segfaults. A zero exit with a fatal log line is a
failure.

### 3.9 Why staleness is hash-based

The old pipeline skipped post-processing when the output directory "looked
populated" (`if dir has ≥10 files: skip`). Re-run with a new solution, keep last
week's plots, notice nothing.

`should_skip()` compares an **input hash** and requires the previous run of that
stage to have succeeded. It never inspects output directories. `--force`
overrides.

---

## 4. Anatomy of a run

`simdev run cases/ahmed/config.yaml --run-dir ~/runs/ahmed-01 --profile dev`

The case argument is the config **file**, not the case directory.

**prepare** — resolve config → `CaseSpec` → validate (raises on inconsistency,
returns warnings) → generate or import geometry into `constant/triSurface/`,
applying the scale and translate once → check watertightness, units and ground
placement → **measure the wheel axes, centres and rolling radii** → compute the
union bounding box → build the domain (box or sector) → compute the *true
projected* frontal area and assert blockage → render every dictionary → write
`caseSpec.json`.

Runs without OpenFOAM installed. This is the stage that turns a config file
into a complete, hand-runnable case.

What was *measured* rather than configured — the wheels, the frontal area, the
cell count — goes into `status/prepare.json`, not `caseSpec.json`. The spec
records what was asked for; the status records what the geometry turned out to
be.

**mesh** — `blockMesh` → `surfaceFeatureExtract` → `decomposePar` → `snappyHexMesh
-parallel -overwrite` → `checkMesh -parallel`. Parses the layer table and the
quality report, then gates on non-orthogonality, skewness, negative volumes, and
**layer coverage**.

Layer coverage is a gate rather than a log line because snappyHexMesh silently
drops layers on thin trailing edges. Under `low_y_plus` that means the near-wall
resolution you paid for does not exist precisely where the wing loads are.

**solve** — `simpleFoam -parallel` on the *same decomposition the mesh stage
made*. No `reconstructParMesh` in between: reconstructing only to re-decompose
wastes a large fraction of meshing time on a 20M-cell case. Function-object
output goes to `postProcessing/` from the master rank, so forces and y⁺ are
available without reconstruction.

**post** — y⁺ band gate, windowed coefficient means, force-history and residual
plots, and the result record.

---

## 5. Module map

| Module | Owns | Depends on |
|---|---|---|
| `geometry/roles.py` | The role enum and trait table | nothing |
| `config/schema.py` | Typed models, `CaseSpec`, derived properties | roles |
| `config/profiles.py` | Defaults, resolution profiles, wall profiles | schema |
| `config/resolve.py` | Layer merging, YAML loading | profiles, schema |
| `config/validate.py` | Cross-file assertions, `estimate_y_plus` | schema, roles |
| `geometry/ahmed.py` | Procedural Ahmed body | schema |
| `geometry/stl.py` | STL info, watertightness, placement transform, projected area | nothing |
| `geometry/step.py` | STEP tessellation via gmsh, conversion cache | nothing |
| `geometry/wheels.py` | **Wheel axes measured from the surfaces** | roles |
| `domain/base.py` | `Domain` protocol, `DomainBox`, `DomainSector` | schema |
| `domain/box.py` | Rectangular tunnel, blockage | base, roles |
| `domain/annulus.py` | Cornering sector, its block topology | base, roles |
| `render/context.py` | **All derived values and BC logic** | schema, domain, roles |
| `render/render.py` | Template dispatch, `caseSpec.json` | context |
| `render/templates/` | Dictionary text | nothing (logic-free) |
| `run/parsers.py` | Log and `.dat` parsing | nothing |
| `run/runner.py` | Subprocess, log scanning, failure | parsers |
| `run/status.py` | Stage status, hash-based skipping | nothing |
| `gates/*` | The three gates | parsers, schema, roles |
| `stages/*` | Orchestration | everything |
| `report/*` | Result records, plots | nothing |
| `cli.py` | Argument parsing, exit codes | stages |

**The dependency rule:** `render/templates/` depends on nothing and contains no
logic. Everything derived — turbulence quantities, wall-function names, BC
assignments, effective reference area — is computed in `render/context.py`.

This is the rule most likely to erode. A template that starts branching on
`spec.physics.mode` is the beginning of the end; it becomes unreadable and
untestable within a few features. Push the branch into `context.py` and let the
template loop over the result.

---

## 6. How to change things

### Add a config option

1. Add the field to the right model in `config/schema.py`, with a type.
2. If it needs a default, put it in `config/profiles.py` `DEFAULTS` — **not** as
   a pydantic default, unless it is genuinely universal.
3. If it interacts with anything else, add an assertion in `config/validate.py`
   and a test that a bad combination is *rejected*.
4. Use it via `spec.<section>.<field>` — never re-read the YAML.

### Tune the mesh near a surface or in the volume

Four knobs, and the interaction between them is the part that bites.

**Volume refinement** — `domain.refinement_regions`, a list of boxes declared
in *body lengths off the geometry bounding box*, with levels relative to
`base_cell_size`. Regions track the model, so they survive a change of domain
or resolution profile. They are clipped to the domain, because a region that
runs past the boundary still refines every background cell it crosses.

Surface refinement only thickens the mesh against the wall. Wakes and
separations live in the volume: adding a wake box to the Ahmed case moved Cd
by 16% and Cl by 44%.

**Per-patch surface refinement** — `refinement_min` / `refinement_max` on a
patch, falling back to the case-wide levels. One level cannot suit surfaces of
very different size; at the case-wide level the Ahmed stilts landed two cells
across with 39 faces.

**Per-patch layer cap** — `n_layers` on a patch. Applied *after* the
fit calculation, so it can only ever ask for fewer.

**The interaction to understand:** refining a patch **reduces** how many
layers it can carry, because the prism stack has to fit inside the cell it is
carved out of. `CaseSpec.n_layers_for()` reconciles this and nothing else
should. Asking for more layers than fit does not buy near-wall resolution —
snappy truncates and drops them on exactly the curved and thin regions where
they mattered.

Layers follow `is_wall`, not `refinement == "high"`: every wall carries a wall
function, and a wall function assumes its first cell sits in the log layer. A
wall without layers is a turbulence model being evaluated where it is not
valid. See `docs/validation-ahmed.md` §3 for a case where the right answer was
still to decline them.

### Add a patch role

1. Add to `PatchRole` and `ROLE_TRAITS` in `geometry/roles.py`. Decide
   deliberately: does it contribute to forces? Is it a wall? What refinement?
2. Add its BC branch in `build_bcs()` in `render/context.py`.
3. Add a test asserting every field covers every patch (the existing
   `test_every_field_covers_every_patch` will catch omissions automatically).

### Change a numerical scheme

Edit `render/templates/fvSchemes.jinja` or `fvSolution.jinja` directly. These
are the files a CFD engineer should be able to change without touching Python —
that is why they are templates and not generated objects.

If the change should vary by case, promote it to a `CaseSpec` field first
(see above), then interpolate it.

### Bring in CAD

`geometry.kind` is `step` or `stl`, and either way **one file is one patch,
identified by filename**. That is the whole convention, and it is what makes
swapping `Body.step` for a new design a drop-in with no config edit.

**The geometry contract.** Export every part in its **assembly position**, one
complete folder per driving state, using the same part names in every folder.
Units may be millimetres — set `geometry.scale: 0.001`.

**The import transform, in this fixed order:**

```
scale  ->  rotate about z  ->  translate  ->  ground datum
```

Not commutative, which is why it is stated. Rotating after translating turns
the translation into a different offset, and a car 200 mm to the side of where
it was meant to be still meshes and still solves.

- `rotate_z_deg` exists because CAD assemblies are routinely built nose-forward
  along +x while the pipeline frame has the freestream along +x, so the nose
  faces −x. **A rotation, never a mirror** — negating x would also turn the car
  round and would silently swap its left and right, which on geometry
  asymmetric by 10% of its width is a different car.
- `ground_datum: tyre_contact` drops the whole car rigidly until its lowest
  tyre point rests on z = 0. Rigidly, and from the *tyres only*: in the test
  export the four tyres reach −0.70, −1.04, −1.05 and −1.40 mm, and that 0.7 mm
  spread is rake and suspension travel. Snapping each wheel separately would
  flatten the car's attitude; snapping to the lowest point of any surface would
  hand ride height to whatever boss hangs lowest.

Positions are not configured because they cannot be: a driving state changes
ride height, steer and camber together. Everything positional is measured from
the surfaces, so a new driving state is a new export and no case file changes.

**STEP tessellation is a physics setting, not a file-format detail.** Too
coarse and a curved surface becomes a faceted one that separates in the wrong
place; too fine and the surface mesh outweighs the volume mesh built from it.
`geometry.tessellation` is in metres like every other length, and
`curvature_segments` (elements per full circle) is the control that matters on
this car — it decides whether a 6 mm suspension link is a hexagon or a cylinder,
independently of the part's size.

Conversion is cached on the file's content hash plus the tessellation settings
plus the scale. Deliberately **not** on the rotation or translation: those are
rigid transforms applied to the triangles afterwards, so a ride-height change
reuses the cached tessellation instead of re-tessellating a 5 MB body. The
cache sits in `.simdev-cache/` beside the CAD, falling back to the user cache
if the CAD is read-only.

**Never measure a STEP file without tessellating it.** OCC bounding boxes come
from NURBS control hulls and run large — an earlier session recorded one about
6× too big. Every dimension the pipeline reports comes from triangles.

### Switch driving state

A driving state is a named bundle of everything that changes together when the
car is doing something different: which geometry folder to read, how fast it is
going, whether it is cornering and how tightly. Switching is one line:

```yaml
driving_state: testcase

driving_states:
  testcase:
    geometry: {source_dir: CAD/Testcase}
    flow:     {u_inf: 15.0}
    physics:  {mode: cornering, corner_radius: 4.0, corner_direction: left}
    domain:   {kind: annulus}
```

To add one: export a complete folder under `CAD/` with the same part names,
then add an entry. Nothing else changes — the wheel axes, rolling radii and
ride height are measured from whichever state is selected.

It is a **merge layer in `resolve.py`**, not something read later, for the
reason the whole config design exists: after resolution there is one object in
which every value is explicit and nothing downstream consults raw config. The
order is:

```
defaults ◄ resolution profile ◄ wall profile ◄ case file ◄ driving state ◄ CLI --set
```

The table is dropped once the selected state is merged, and only its *name*
survives into the spec. Keeping the whole table would put every unselected
state into the spec hash, so editing the braking state would invalidate cached
cornering runs it cannot possibly have affected.

Two checks are deliberately hard failures rather than warnings:

- **A missing STL for a patch whose role carries one.** Silently skipping it
  used to be the behaviour, and it turns a mistyped part name into a car with
  no rear wing that meshes, solves and converges.
- **Geometry through the road.** snappy meshes the intersection of a wheel and
  the ground into a shape nobody drew, and the run looks entirely normal.
  `geometry.max_ground_penetration` sets the tolerance.

Keep every STL **watertight and whole**, even for half models — half models come
from the *domain* restricting to y ≥ 0 with a `symmetry` patch, never from
cutting geometry. Note that STL stores each facet's vertices separately, so a
sound solid loads as unconnected triangles; `load_surface()` merges them first,
and without that every mesh ever exported reports as leaking.

For a *procedural* source instead, implement a writer with the same contract as
`write_ahmed_stl`: `(params, out_dir) -> dict[str, Path]` mapping patch name to
STL path. Add a branch in `_write_geometry()` in `stages/prepare.py` and a
`kind` value in `GeometryConfig`.

Keep the STL **full-body and watertight** even for half models. Half models are
produced by the *domain* restricting to y ≥ 0 with a `symmetry` patch, not by
cutting geometry. Cutting geometry makes it non-watertight, and snappyHexMesh
leaks into non-watertight surfaces.

### How cornering works

Steady cornering is only steady in a frame that turns with the car, so the
curved domain and the rotating frame are two halves of one model. Neither
means anything alone, and the most common way to get a plausible, wrong
cornering result is to build one and not the other.

**The domain** (`domain/annulus.py`) is an annular sector swept about a
vertical axis through the corner centre. It reads the *same* four numbers as
the box — upstream, downstream, half width, height — and reinterprets them for
a curved path: upstream and downstream become arc length at the vehicle's own
radius, half width becomes radial half-extent. A case therefore keeps its
domain proportions when switched between straight and cornering, so a
difference between the two runs is physics rather than tunnel size.

A sector rather than a box because in the rotating frame the still air outside
is in solid-body rotation and its streamlines are circles. A box cuts those
circles at an angle, making every outer face simultaneously an inlet and an
outlet.

**The frame** is a whole-domain MRF zone at ω = U/R about that same axis. The
`all` cellZone is created by `blockMesh` — the name sits between the vertex
list and the cell counts on each block — and snappy hands it down to every cell
it refines. It is **not** an OpenFOAM keyword: `MRFZone::read` looks the name
up and aborts with `cannot find MRF cellZone all` if nothing made it.

**The boundary conditions follow from the frame, and are not what instinct
says.** OpenFOAM's MRF solves the *absolute* velocity, so:

| Patch | Condition | Why |
|---|---|---|
| inlet | `fixedValue (0 0 0)` | Far from the car the air really is at rest over the track. The onset flow emerges from the frame rotation; prescribing a freestream as well drives the case twice |
| ground | `noSlip`, listed in `nonRotatingPatches` | The road is genuinely stationary. Its sweep under the car comes out of the frame transform — which is why it correctly varies across the track width |
| car body | `noSlip`, *not* excluded | An included patch is forced to Ω×r, i.e. it rotates with the car. Correct, and it overrides whatever is written |
| tyres | `codedFixedValue`, excluded | See below |

`ground.motion` is ignored in cornering, and the validator says so rather than
applying the motion twice.

**Why the tyres need a coded condition.** A cornering tyre is carried around
the corner *and* spins about its own axis — two rotations about different,
non-intersecting axes, which is a screw motion. No single OpenFOAM
rotating-wall condition describes one. Excluded patches carry absolute
velocities, so `render/context.py::_tyre_bc()` sums the two terms explicitly.
Leaving the corner term out parks a spinning wheel in space while the car
drives away from it.

Straight-line needs none of this: one rotation, so `rotatingWallVelocity`
expresses it exactly.

**The signs are derived, never asserted**, because a mirrored cornering case
converges just as happily as a correct one. `corner_side` places the centre on
the side the car turns toward; `omega_signed` follows from that; the inlet end
of the sector follows from both. The test that ties them together asserts the
property that must hold: still air, seen from the rotating frame, arrives at
the car as +x at `u_inf`.

**Blocks are ordered by increasing θ regardless of flow direction.** Hex
handedness depends on the sign of the angular step, so generating them in flow
order inverts every cell on a right-hand corner — and blockMesh reports that as
negative volume, a long way from the cause.

### Add MRF / rotating wheels

Wheels are declared by giving surfaces a shared `wheel` id in the patch list.
Everything else is measured.

- **`geometry/wheels.py` measures the axis, centre, rolling radius and width**
  from the surfaces themselves, per driving state. The axis is the odd one out
  of the area-weighted second-moment tensor of a solid of revolution — one rule
  that works for a wide wheel and a long driveshaft alike, where "largest" or
  "smallest" would get one of them backwards. The moment is integrated exactly
  over each triangle rather than approximated by centroids: tessellators fan a
  flat end cap from one rim vertex, and centroid weighting read that as a 24%
  asymmetry on a machined cylinder.
- **Rolling speed is solved against the road, not configured.** In cornering
  each wheel stands on road moving at its own radius from the corner centre, so
  the outer wheels turn faster — 6.5% at a 3 m corner on a 190 mm track. The
  component the axis cannot roll away is returned as slip, which is how a
  steered wheel announces itself.
- **MRF cell zones** come from a closed STL per zone with role `mrfZone`.
  snappy makes the cellZone directly (`cellZone`/`faceZone`/`cellZoneInside`),
  so no `topoSet` pass is needed, and the faces stay internal so no boundary
  patch is created — which is why `build_bcs()` skips the role entirely.

**Straight-line and cornering cannot both have wheel MRF zones.** An MRF cell
carries exactly one frame rotation, so no cell can be going round the corner
*and* round the wheel. Straight-line gets a zone per wheel and real rim
pumping; cornering gets one frame over everything and drives the tyre surface
through its boundary condition instead. What is lost in cornering is rim
pumping, not wheel rotation, and the validator says so.

One subtlety worth keeping: snappy takes the wheel-zone cells *out* of `all`
when it creates their zones, so cornering also emits a corner-frame entry for
each wheel zone. Without it those few hundred cells would be the only inertial
ones in a car going round a corner.

**Do not let MRF surfaces enter force integration.** They cannot —
`ROLE_TRAITS[MRF_ZONE].in_forces` is `False` — and this is the exact bug the old
pipeline needed a name filter to avoid.

### Add a gate

1. Write a pure function returning `GateResult(passed, reasons, detail)` in
   `gates/`. Take parsed data, not file paths — that is what makes it testable.
2. Add a parser in `run/parsers.py` if needed, with a fixture log in
   `tests/fixtures/logs/`.
3. Wire it into the stage; record it in `StageStatus`.
4. Decide deliberately whether it **blocks** (raise `StageError`) or **flags**
   (record `gate_failed`, continue). Mesh quality blocks — there is no point
   solving on a broken mesh. Convergence and y⁺ flag, because the run still
   produced data worth keeping.

### Add a post-processing output

Add to `stages/post.py` and, if it belongs in the record, to `ResultRecord`.
Keep it in the record if you would ever want to compare it across runs; keep it
as a file if it is only for looking at.

---

## 7. The physics you need to know

**Reynolds number sets everything.** Re = UL/ν. At 1/10 scale you are roughly an
order of magnitude below full-size car Re, and appendages are two orders below
the body. This is why wall treatment is a profile and why transition modelling
is on the roadmap.

**y⁺ is the mesh–turbulence contract.** It measures the first cell centre in
wall units. Wall functions assume the first cell sits in the log layer
(y⁺ 30–300). Wall-resolved meshing assumes it sits in the viscous sublayer
(y⁺ ≈ 1). **The buffer layer (y⁺ ≈ 5–30) is valid for neither** — landing there
by accident is the most common way to get confidently wrong answers, which is
why there is a gate.

`estimate_y_plus()` predicts it before you run, from a flat-plate correlation.
The y⁺ gate measures it after. Trust the second one.

**Blockage** is frontal area over domain cross-section. Under ~1 % for a virtual
tunnel; above that, the domain walls contaminate the pressure field and inflate
drag. For validation against a real experiment you want to *match* the
experiment's blockage instead.

**Convergence is not residuals.** See §3.4.

**Transition.** At Re_chord ≈ 3.5×10⁴ the appendages live in laminar-separation-
bubble territory. Fully-turbulent `kOmegaSST` assumes a turbulent boundary layer
from the leading edge and cannot represent that. The turbulence model is
config-selected so `kOmegaSSTLM` can replace it — but it needs low-y⁺ meshing to
mean anything, which is another reason the RC car runs wall-resolved.

---

## 8. When something goes wrong

**Read `status/*.json` first.** Every stage records `ok` / `failed` /
`gate_failed` with reasons. That tells you which stage and why, before you open
a log.

| Symptom | Look at | Usual cause |
|---|---|---|
| Validation error at `prepare` | The message — they name the two things that disagree | Config edited without its partner (patch set vs mode, layer thickness vs wall treatment) |
| snappyHexMesh leaks into the body | `check_geometry` warnings | Non-watertight STL |
| Mesh gate fails on layer coverage | `logs/log.snappyHexMesh` layer table | Thin trailing edges; reduce `first_layer_thickness` or `expansionRatio`, or relax `minThickness` |
| Mesh gate fails on non-orthogonality | `logs/log.checkMesh` | Too-aggressive refinement jumps; raise `nCellsBetweenLevels` |
| Solve diverges immediately | `logs/log.simpleFoam` | Usually a BC or a bad cell; check `checkMesh` passed and `locationInMesh` is actually in the fluid |
| `converged=False` with tight scatter but a slope | `results/forces.png` | Genuinely still drifting — raise `max_iterations` |
| `converged=False` with large scatter | `results/forces.png` | Unsteadiness the steady solver cannot settle; may need transient |
| y⁺ gate fails | `results/result.json` `yplus` | Wrong wall profile for the condition, or collapsed layers (check the mesh gate detail) |
| Coefficients ~2× expected | `caseSpec.json` `half_model`, `a_ref_full`, `geometry.symmetric` | Should be impossible by construction — if it happens, the derivation is broken and that is a bug worth a test |
| `cannot find MRF cellZone all` | `system/blockMeshDict` | The zone is named on the block, not a keyword. A cornering case whose blockMeshDict came from the box template has no zone |
| `Failed wmake ... corneringWheel*` | `dynamicCode/`, then `0/U` | The coded tyre condition did not compile. The generated file is C++, so vectors are comma-separated — `_cpp_vec`, not `_foam_vec` |
| Cornering forces mirrored | `caseSpec.json` `corner_direction`; `status/prepare.json` wheel speeds | Outer wheels must turn faster than inner ones. If they do not, the corner centre is on the wrong side |
| blockMesh reports negative volumes on a sector | `domain/annulus.py` | Hex handedness follows the sign of the angular step; blocks must be ordered by increasing θ |
| Wheels turn but the road does not | Validation warnings | Straight-line case with `ground.motion: static` and spinning wheels |

**The provenance rule:** every result carries its `spec_hash`. If a number
surprises you, diff the `caseSpec.json` against a run you trust. That comparison
is the whole reason the spec is a text file.

---

## 9. What is deliberately not here yet

| Deferred | Already provided for |
|---|---|
| Yaw / pitch / roll attitude | Mode table drives symmetry derivation; `geometry.symmetric` is a separate input |
| Transition model | `turbulence_model` is config-selected |
| Full plane-cut image suite | `post` stage exists with a minimal set |
| Parametric sweeps | Per-run records aggregate on read; `--set` overrides a single field without copying the case |
| Curved wake refinement for cornering | `refinement_regions` exist but are axis-aligned boxes |
| Per-component forces, aero balance | `forceCoeffs` renders one group; roles already separate force-bearing surfaces |
| Side force and yaw moment in the record | OpenFOAM already writes `Cs`, `CmYaw` and `CmRoll`; `run/parsers.py` reads only Cd and Cl |
| Curved refinement regions | `refinement_regions` are axis-aligned boxes; a cornering wake leaves them, and the validator warns |

Cornering, the annular domain, MRF zones, rotating wheels and the CAD geometry
path are **built and exercised against real OpenFOAM v2412** — a cornering car
case meshes, creates its cell zones, compiles its coded wheel conditions and
solves. What that does *not* mean is that any of it is validated: it means the
plumbing is correct, not the physics.

**Three things that would bite next:**

- **`a_ref_full` for the car is a placeholder** taken from the body bounding
  box. `prepare` reports the true projected frontal area — set it from that and
  record which convention it follows.
- **`low_y_plus` has never been run.** The wall-profile machinery is unit
  tested; the physics path is not. The car is intended to use it, and the
  `car`/`car_dev` profiles are sized on paper rather than measured.
- **Nothing about the car is validated against data.** See the risk note below.

**One open risk, stated plainly:** the Ahmed validation runs at Re ≈ 2.8×10⁶ with
`high_y_plus`. It validates plumbing, numerics, domain construction, force
integration and the gates. It does **not** validate the low-Re wall-resolved
settings the RC car actually uses. Those need separate justification and,
ideally, comparison against track or tunnel data for the real vehicle. Do not
mistake a green Ahmed run for a validated RC car.
