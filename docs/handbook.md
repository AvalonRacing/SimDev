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

- The gate asks **two separate questions** of Cd and Cl, because on a separated
  case they have different answers and only one is about convergence:
  - **Has the mean stopped moving?** The trailing window's mean against the
    mean of the window before it, versus `solve.drift_tol`. *This* is the
    convergence test, and it is the one to keep tight.
  - **How far does it swing about that mean?** `std/|mean|` over the trailing
    window, versus `solve.amplitude_tol`. A stability bound, not a convergence
    test — a limit cycle is physics, and a gate that fails on it is measuring
    the flow rather than the solve.
- **Not a least-squares slope inside one window.** A sine sampled over part of
  a period genuinely has a slope, so that test reads a perfectly stationary
  limit cycle as drift, at a magnitude set by where in the cycle the run
  stopped — it reported "+10.92 % drifting" on a mean that had not moved.
  Two consecutive window means each average the cycle away instead. The
  corollary is that **`plateau_window` wants to be at least one oscillation
  period long**, and judging drift needs two whole windows, so a run shorter
  than `2 × plateau_window` is reported as unjudgeable rather than converged.
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
converting units → check watertightness, units and road placement → **measure
the wheel axes, centres and rolling radii** → compute the union bounding box →
build the domain (box or sector) → assert blockage → render every dictionary →
write `caseSpec.json`.

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

Five knobs, and the interaction between them is the part that bites.

**Volume refinement, as a box** — `domain.refinement_regions`, a list of boxes
declared in *body lengths off the geometry bounding box*, with levels relative
to `base_cell_size`. Regions track the model, so they survive a change of
domain or resolution profile. They are clipped to the domain, because a region
that runs past the boundary still refines every background cell it crosses.

Surface refinement only thickens the mesh against the wall. Wakes and
separations live in the volume: adding a wake box to the Ahmed case moved Cd
by 16% and Cl by 44%.

**Volume refinement, as a distance** — `domain.refinement_shells`, a list of
`{distance, level}` pairs. Distance is in body lengths like everything else;
level is absolute like a region's. snappy refines every cell within `distance`
of the vehicle to at least `level`.

Use a shell rather than a box whenever the refined volume should be the shape
of the *car* instead of the shape of a box drawn round it. On the RC car that
is not a preference, it is the only thing that works: a cornering wake follows
the curve of the path and leaves any axis-aligned box, and the car is posed at
a body-slip angle besides, so a box big enough to contain the near field
spends most of its cells in clean air. A shell has no orientation to get
wrong, which is why the same three lines are correct for the cornering state
and the straight state.

The other thing a shell buys is the flow *through* the car. Distance is to the
nearest surface of any part, so the gaps between body, chassis, wishbones and
rims fall inside the innermost shell automatically. A box refines the air
around the car; a shell refines the air in it.

Two rules, both enforced:

- **Shells must get coarser with distance.** snappy applies the first shell
  whose distance contains a cell, so an outer shell finer than an inner one
  simply never applies. It is not an error snappy reports — it meshes happily
  and ignores you — so `validate()` rejects it.
- **The order in the case file does not matter.** `refinement_shells()` sorts
  them nearest-first before rendering, which is the order snappy needs.

Mechanically, `prepare` writes one extra surface, `constant/triSurface/
vehicle.stl`, that is every wall patch concatenated. It appears in snappy's
`geometry` and **never** in `refinementSurfaces`, so it creates no patch,
carries no boundary condition, is never snapped to and never enters force
integration — it exists only as something to measure distance from. One
combined surface means snappy builds one distance field rather than fifteen.
The MRF sleeves are left out: they are closed volumes *inside* the tyres, and
including them would drag a shell of fine cells into solid rubber.

One consequence worth knowing: a shell thicker than the ride height refines
the road under the car, so `ground_cell_size()` counts shells as well as
boxes. Miss that and the ground's prism stack gets budgeted against a
background cell that does not exist anywhere near the vehicle.

**Per-patch surface refinement** — `refinement_min` / `refinement_max` on a
patch, falling back to the case-wide levels. One level cannot suit surfaces of
very different size; at the case-wide level the Ahmed stilts landed two cells
across with 39 faces.

**Per-patch layer cap** — `n_layers` on a patch. Applied *after* the
fit calculation, so it can only ever ask for fewer.

**The background these levels count from** — `mesh.base_cell_size`. Every
refinement level is a halving of it, so it is not an independent knob: change
it and every level in every profile and every per-patch override has to move
with it, or the surfaces silently change resolution while still being called
level 7.

The car profiles run a **96 mm** background. It was 24 mm, and the reason for
the change is what that cost out in the domain: the first production mesh
spent 1,818,230 cells — a quarter of the whole mesh — on uniform 24 mm
background, most of it in clean air metres from the car, while levels 1 and 2
between the far field and the body held 3,881 and 18,731 cells between them.
At 96 mm the same volume is about 28,000 cells.

**Re-base only by powers of two, and check the metres afterwards.** 96 = 24×4,
so every level moved by +2 and every absolute size is unchanged
(0.096 / 2⁷ = 0.024 / 2⁵ = 0.75 mm). A factor like 3× cannot be absorbed by
any integer level and would move every derived size, the layer budget and
`max_layer_cell_ratio` with it. `tests/test_cell_sizes.py` pins the resulting
cell size **in metres** for every patch and every profile, precisely so the
next re-basing is arithmetic rather than an act of care — if it misses a
patch, the suite fails instead of the physics quietly changing.

What it costs: the ladder is now seven levels deep, and `nCellsBetweenLevels
3` spends buffer cells on every rung, so the far-field saving is not banked in
full.

**A case-wide refinement ceiling** — `mesh.refinement_cap`, applied in
`patch_refinement()` and therefore to per-patch levels as well as case-wide
ones. This is the only way to make a whole case cheap without editing the
patch list: per-patch levels deliberately *override* the resolution profile,
so coarsening `base_cell_size` alone leaves four patches pinned at level 5 and
the mesh expensive anyway. The `car_smoke` profile uses it to answer "does the
pipeline work as a system" in minutes. Do not read a coefficient off a capped
run — the y+ gate will fail it, correctly.

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

**Where layer settings have to live.** `n_layers`, `first_layer_thickness`,
`expansion_ratio` and `max_layer_cell_ratio` cannot be set in a resolution
profile: the wall profile merges *after* it (see the merge order in §6), so
the generic `low_y_plus` numbers would overwrite them without a word. Put them
in the case file, which is also their right owner — the first layer follows
from that vehicle's Reynolds number, and `WALL_PROFILES` is sized for nothing
in particular. `cases/car/config.yaml` carries a worked example.

The stack is graded against the cell it hands off to, not just against y⁺.
Getting the first layer right and then stopping is what leaves a 61 µm prism
against a 414 µm hex. Two numbers are worth computing before a production
mesh: the jump from the last layer to the remaining cell height (aim for
≲ 3×), and the y⁺ at the *top* of the stack (aim to cover the buffer layer,
y⁺ ≈ 20, so the hexes take over in the log layer).

**A sized stack is not an inserted stack, and the gate measures the second.**
Everything above is arithmetic on the spec; snappy then has to build it
against its mesh-quality limits, and that is where a well-sized stack quietly
becomes a 1.4-layer one. The diagnosis lives in the per-iteration trace in
`log.snappyHexMesh`, not in the layer table at the end:

```
Added 4591404 out of 4971498 cells (92.4%)   <- iteration 0, stack fits
Added 2983865 out of 4971498 cells (60.0%)   <- iteration 1, quality took it back
```

The tell is that the extrusion percentage barely moves (89.7% → 86.7%) while
the cell count halves: faces are still being extruded, so the layers are being
*squeezed*, not dropped, and no amount of re-sizing the stack will help.
`meshQualityControls/relaxed` with `nRelaxedIter` is the fix — snappy's design
is to try the strict limits and fall back, and with no `relaxed` block there is
nothing to fall back to.

Note the coupling this exposes, because it is easy to make worse by trying to
be careful: `spec.mesh.max_non_ortho` and `max_skewness` are rendered into
snappy's `meshQualityControls` as well as being the mesh gate's thresholds. So
they are the *mesher's construction constraints* and the *gate's acceptance
criteria* at once, and tightening the gate makes snappy build worse layers,
which then fails the gate. Decoupling them is worth doing.

### Make meshing faster without coarsening the mesh

snappy's cost is not proportional to the cells it produces. It is proportional
to how much geometry querying each refinement iteration does and to how evenly
that work spreads over the ranks — which is why this case meshes slowly at a
cell count that looks modest. Nearly every one of its 6.6 M cells is created
inside a ball a few tenths of a metre across, in a 25.7 m³ domain whose
background is only ~28 k cells.

Two settings exist for this, and neither changes the resolution anywhere.

**`domain.shell_surface_tolerance`** simplifies the combined vehicle surface
that the refinement shells measure distance from. snappy answers a
distance-mode region from an octree over that surface's triangles and rebuilds
it on every rank each time it redistributes the mesh mid-refinement, so the
triangle count is a per-iteration cost. The surface arrives as the CAD
tessellation — sized by `geometry.tessellation` for surfaces that get *snapped
to* — and this one never is: it is not in `refinementSurfaces`, creates no
patch, and never enters force integration. On the real car, 2 mm takes it from
598,796 triangles to 146,450.

The tolerance is not free to raise, and the constraint that bites is not the
one you would expect. Displacement is bounded by the grid, so the refinement
boundary cannot shift further than the tolerance — 2 mm against a 53 mm
innermost shell is nothing. What actually limits it is that a feature *thinner*
than the tolerance can collapse out of the surface entirely, and a suspension
link that is not in the distance field pulls no refinement around itself —
losing exactly the flow between body, chassis and wishbones that the innermost
shell was added for. Measured on the car: at 2 mm the worst part keeps 97.9% of
its area, at 3 mm 93.6%, at 5 mm 81.5%. `prepare` measures retention per part
on every run and warns below 95%, because whether a given feature collapses
depends on where the grid falls across it and cannot be settled by arithmetic.
The before/after counts land in the prepare record under `shell_surface`.

**`mesh.max_load_unbalance`** decides how often snappy redistributes the mesh
between ranks during refinement. `decomposePar` splits the *background* mesh,
which is uniform, so the ranks start with an equal share of a domain in which
the refinement is then poured into that small ball: a handful of ranks create
essentially the whole mesh while the rest hold empty tunnel. It renders
explicitly at the tutorial value of 0.10 rather than being left out, so the
policy is a property of the case and not of whichever build ran it. Lowering it
rebalances more often — worth measuring on this case, where the imbalance is
extreme.

Related trap: `maxLocalCells` is a **silent refinement stop**, not an error. A
rank that hits it stops refining and snappy carries on, so a badly unbalanced
run can return a mesh coarser than the one that was asked for without saying
so.

Before touching either, get the phase breakdown — the three phases respond to
different knobs entirely:

```
grep -nE "Refinement phase|Snapping phase|Layer addition phase|ExecutionTime" logs/log.snappyHexMesh
```

### Make the solve faster

Short answer: the numerics have nothing left to give, but the *memory system*
does. Two things were found after this section was first written, and both are
worth more than anything in the table below: mesh renumbering, which is done
and is in the pipeline, and the operating system, which is not. Everything
here was measured on the production car mesh so that the next person does not
spend the afternoon finding it out again.

**`renumberMesh`, -9.6%, already in the mesh stage.** snappy emits cells in
octree-refinement order, which bears no relation to adjacency, and that
ordering sets the locality of every gather the linear solver makes -
`lduMatrix` reaches `psi` indirectly through `lowerAddr`/`upperAddr`, which is
what this solve is made of. Nothing renumbered before 2026-08-20, and this
table never tested it: everything in it is numerics.

| | band | profile | s/iter |
|---|---|---|---|
| as snappy left it | 174,038 | 5.81e10 | 15.10 |
| after `renumberMesh` | 7,429 | 2.54e10 | **13.65** |

At the old band a gather reached ~1.4 MB away, outside any cache; at the new
one ~59 kB, inside L2. `mesh.renumber` turns it off, and the only reason to is
reproducing a pre-renumbering run iteration for iteration — Gauss-Seidel
sweeps in index order and GAMG agglomerates from the addressing, so the route
to the answer changes even though the discrete system does not.

**The operating system is worth ~1.8x, and that is now measured.** WSL2
delivers 28.5 GB/s of memory bandwidth; the same hardware sustains ~51 GB/s
from native Windows when each socket uses its own memory, and an
OpenFOAM-shaped gather benchmark reproduces the same 1.85x under those two
placements. The four DIMMs are *correctly* installed — one per channel, two
channels per socket, proven by a per-socket rate that exceeds a single
channel's theoretical peak — so there is nothing to fix in the hardware and
nothing to buy yet. See `docs/linux-migration.md` §1b.

With both, 2000 iterations goes from 8.4 h to **4.2-5.1 h**, which is the
five-hour target this section closes by calling unreachable. It is reachable,
and not by cutting cells.

The rest of this section stands: within WSL2, and holding the mesh fixed, the
numerics have nothing left.

**The baseline.** `cases/car` at the `car` profile, 6,929,626 cells, 40 ranks,
100 iterations from the same initial field each time: **14.78 s/iter**. That is
8.2 hours for 2000 iterations, plus about 24 minutes for everything else
(prepare, blockMesh, snappy at ~16 min, checkMesh, and a `post` stage that is
pure Python — the pipeline never calls `reconstructPar`).

| variant | s/iter | vs baseline | cumulative continuity | outcome |
|---|---|---|---|---|
| baseline | 14.78 | — | 2.4e-4 | ok |
| `mpirun --bind-to core --map-by socket` | 14.84 | +0.4% | 2.4e-4 | ok |
| `nNonOrthogonalCorrectors 0` | 35.4 | — | 7.3e+12 | **SIGFPE, iter 29** |
| ” + `nCellsInCoarsestLevel 1000` | 19.9 | — | -4.6e+13 | **SIGFPE, iter 39** |
| ” + unlimited `grad(p)` | 12.28 | -17% | -1.3e-3 | survived, see below |
| `nCellsInCoarsestLevel 1000` | 14.77 | -0.1% | -2.5e-4 | ok |
| ” + `relTol` 0.05 → 0.1 | 14.46 | -2.2% | -1.4e-3 | ok |
| ” + unlimited `grad(p)` | 13.35 | -9.7% | -1.7e-3 | ok |

**`nNonOrthogonalCorrectors 0` diverges. Do not use it here.** It is the
obvious saving and it is the first thing anyone will reach for, because the
corrector costs *more* than the solve it corrects: the log shows the first
pressure solve taking two GAMG cycles and the corrector taking four, since
`relTol` is relative and bites harder on an already-reduced residual. Dropping
it is still fatal. Local continuity error at iteration 20 goes from 9.5e-4 with
the corrector to 1.3e-2 without, and the run dies with a floating point
exception inside forty iterations. 71.7° of non-orthogonality across 68
severely non-orthogonal faces is more than `limited corrected 0.33` can absorb.

The variant that survived — corrector off *and* `grad(p)` unlimited — is a
knife edge, not a setting. Its two siblings died, it carries ten times the
baseline continuity error, and it moved Cd by 4.2% at iteration 100. It is in
the table to be ruled out, not to be used.

**Rank binding does nothing, and that is the expected answer.** Open MPI
defaults to `--bind-to socket` above two ranks, so pinning to cores should help
a NUMA-sensitive code. It measured 0.4% *slower*, i.e. noise. This is
consistent with the STREAM result in `docs/linux-migration.md` §1b: a solver
that saturates memory bandwidth at four threads cannot care where its ranks
sit. The flags are kept because they are correct, not because they are faster;
`SIMDEV_MPI_ARGS` turns them off.

**GAMG's coarsest level does nothing either.** The mechanism is real —
`GAMGAgglomeration.C` stops agglomerating when `nTotalCoarseCells < nProcs *
nCellsInCoarsestLevel`, so the default of 10 at 40 ranks keeps building coarse
levels until the whole coarse grid is under 400 cells, each level costing a
global reduction for almost no work. Raising it to 1000 removes those levels
and changes the wall clock by 0.1%. Whatever this case is waiting on, it is not
coarse-grid reductions.

**Things already ruled out, with the measurement that ruled them out.** Do not
re-chase these:

- *Decomposition quality.* `decomposePar` runs on the 28,420-cell background
  mesh, not the refined one, so it is natural to suspect that snappy's
  mid-refinement redistribution left bad subdomains. It did not: the final mesh
  has 209,940 processor faces against 20,979,929 internal faces (1.0%), and
  cells are balanced across ranks to within 2%.
- *A hidden factor-of-ten.* 14.78 s/iter looks far slower than 6.9 M cells at
  28 GB/s implies. It is not, and the arithmetic that says otherwise is a
  STREAM model applied to a code that does not stream: the matrix is traversed
  through `lowerAddr`/`upperAddr`, and indirect-addressed sparse work runs 3-5x
  below STREAM as a matter of course. Nothing is broken.

**So what is left, if the platform is held fixed.** The cost is cells times
iterations, and both are linear. To reach a five-hour wall clock at 2000
iterations you need 8.28 s/iter, which at this per-cell rate is about 3.9 M
cells — a 44% cut from a mesh whose every level has a documented reason in
`cases/car/config.yaml`. These are the levers that remain *inside WSL2*, and
the migration above reaches the same target without spending any of them, so
price them against that rather than against doing nothing:

1. **Fewer iterations.** 1150 iterations fits five hours with no change to the
   physics at all. The first production solve was stationary in the mean long
   before 3000; whether 1150 is enough is a question about *this* case's limit
   cycle, and `plateau_window` is what answers it.
2. **`nCellsBetweenLevels` 3 → 2.** Levels 3-6 hold 1.77 M cells, much of it
   buffer between rungs of a seven-level ladder. Costs sharper transitions and
   raises non-orthogonality — which the corrector result above says this case
   has no margin for. Measure `checkMesh` before believing it.
3. **The prism stack.** Layer addition takes level 7 from 2,098,855 cells to
   5,126,051, so **3.03 M cells — 44% of the mesh — are prism.** `n_layers`
   6 → 3 recovers half of that. It does not change y+ at the wall (the first
   layer is still 40 µm) but it wrecks the hand-off: three layers span 152 µm
   into a 598 µm remaining cell, a 9.6x jump against the present 2.5x, and the
   stack no longer reaches the log layer. See the long note in
   `cases/car/config.yaml` for why that grading was chosen.
4. **Surface level 7 → 6 somewhere.** Quarters both the surface cells and the
   prism cells on that patch. Every patch currently at 7 has a reason recorded
   next to it, and the tyres were moved 6 → 7 specifically to bring y+ inside
   the gate band.

The real fix is not in this file, and none of those four levers is now the
first thing to reach for. It is the memory system: the machine runs under
WSL2, which delivers 28.5 GB/s of the ~51 GB/s it has. The DIMM population is
*not* the problem it was once written up as — four of twelve channels is a
real ceiling, but the four are correctly placed and the operating system costs
more than they do. Migrate first, measure, and only then price RAM. See
`docs/linux-migration.md` §1b.

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

**The CAD is the truth and the pipeline does not move it.** Yaw, pitch, roll,
steering, camber and ride height are all set in CAD and arrive baked into the
part positions. The only thing applied on import is `geometry.scale`, which is
a unit conversion (millimetres to metres) rather than a placement. There is no
rotation, no translation and no ground snapping, and that is deliberate: every
transform the pipeline is allowed to apply is a place where the simulated car
can differ from the drawn one, invisibly.

Two consequences worth knowing:

- **When the CAD frame and the tunnel disagree about which way the car faces,
  the tunnel is reversed** (`flow.direction`), not the car. See below.
- **Tyres are expected to cross z = 0.** A loaded tyre is modelled deflected
  into the road and the part below the plane is the contact patch. It is
  squared off there before meshing — see below. Bodywork below the road is
  *reported* rather than rejected, and is left alone for snappyHexMesh to clip
  — at a big enough roll or dive a splitter really does touch the road, and
  that is a condition to simulate.

### The tyre contact patch

The one place the pipeline changes the shape of the CAD, and the exception
that proves the rule above.

A tyre drawn deflected into the road meets it *tangentially*, so the wedge
between tread and tarmac closes to zero angle. Left alone, that wedge is
filled with sliver cells: they fail `checkMesh` on skewness, they refuse
layers, and where they do mesh they plant a spurious separation line right
where the wheel wake is born — on an open-wheel car, one of the largest single
contributions to drag.

So `prepare` cuts each `tyre` patch on a horizontal plane just above the road
and extrudes the resulting cross-section straight down through it. The
tangential wedge becomes a vertical wall meeting the ground at ninety degrees,
and the footprint snappy resolves is the real contact patch rather than
whatever the clipping happened to leave.

```yaml
geometry:
  contact_patch:
    enabled: true
    cut_height: 0.0005        # m above the road; the height of the step
    depth_below_road: 0.002   # m the wall runs past z = 0
```

**`cut_height` is the number to think about.** It is the height of the vertical
step, so it has to be something snappy can resolve: a step well under one
surface cell gets smeared back into the ramp you were trying to get rid of.
`prepare` warns, per patch, when it is smaller than that patch's own surface
cell. Raising it costs a little rolling radius and buys a step the mesh can
actually hold.

`depth_below_road` exists so the wall ends *past* `z = 0` rather than on it. A
face coplanar with the `ground` patch is its own class of snapping failure;
running the wall below and letting snappy clip it keeps the intersection a
clean edge. Nothing below the road is simulated — the background mesh starts
at `z = 0` — so the flat cap down there is only holding the STL closed.

Three things about the ordering, all of which have a wrong answer that looks
fine:

- The cut runs **last**, after every check and measurement. Rolling radii,
  ride height and the reported contact depth all describe the CAD as drawn.
- In particular the rolling radius **must** be measured first. The extruded
  corners sit further from the wheel axis than the tread does, so a radius
  taken off the cut surface reads several millimetres high and drives every
  wheel too fast.
- A tyre that does not reach the road is left alone and still trips the
  "the car is floating" warning. Inventing a footprint for it would hide the
  fault.

What the cut produced is recorded per tyre in `status/prepare.json` under
`contact_patches` — footprint area, the depth the CAD drew, and the number of
separate loops (a treaded tyre has more than one). The area is worth a glance
against the load the tyre is carrying; neither it nor the depth is recoverable
from the written STL, because the written STL is the one already squared off.

**Attitude is never failed on.** An RC car spends most of its cornering life
in heavy understeer, so large steer and body-slip angles are the normal
operating point rather than a symptom. Nothing in the pipeline gates on them:
slip is measured and reported as an angle, per wheel, and recorded in
`status/prepare.json` as `slip_deg`. The test suite prepares a car at 10, 25
and 45 degrees of steer to keep it that way.

The one attitude-adjacent failure left is a genuine singularity, not a policy:
a wheel whose axis is within about 6 degrees of vertical has no contact patch
and no rolling speed to solve for.

### Reverse the tunnel instead of the car

`flow.direction` is `+x` or `-x`. A CAD assembly built nose-forward along +x
is a car travelling along +x, so the air comes at it from +x and the
freestream is `-x`.

This one setting drives **every** other direction, and they are all derived
from it rather than written down separately:

| Follows `flow.direction` | Where |
|---|---|
| Which end of the box is the inlet, and which side gets the long wake tail | `domain/box.py`, `DomainBox.x_min_patch` |
| `dragDir` and `pitchAxis` in `forceCoeffs` | `CaseSpec.drag_dir`, `pitch_axis` |
| Which side of the car is its left, hence where the corner centre goes | `CaseSpec.corner_side` |
| The sign of the cornering frame rotation | `CaseSpec.omega_signed` |
| Which end of the sector is the inlet | `domain/annulus.py` |
| The direction the road runs under the tyres | `render/context.py::wheel_speeds` |
| Where snappy's seed point goes | `location_in_mesh` |

Reversing the tunnel and forgetting the drag axis produces a case that
converges to a confidently negative Cd. The test that guards the whole set
asserts one invariant: still air, seen from the rotating frame, arrives at the
car as that case's own freestream.

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

**In cornering the sleeves are not cell zones at all**, and the reason is
sharper than the screw-motion argument above. `all` is every cell that is not
in some other zone — `polyTopoChange` carries one cellZone label per cell
(`polyTopoChange.H:211`), so snappy's zone assignment is a *move*, not a copy,
and a wheel zone therefore shares its entire boundary with the corner frame.
`MRFZone` counts a face as its own when **either** of the face's cells is in
the zone (`MRFZone.C:80-87`), so every one of those faces is an internal face
of *both* zones, and `MRFZoneList::makeRelative` just loops the zones
subtracting each frame's `Ω × r` in turn (`MRFZoneList.C:249-254`). The flux
entering the sleeve is not the flux leaving the fluid around it.

Measured, because the obvious objection is that two zones describing the same
motion should behave like one. A 4×4×1 duct at 1 m/s in a frame turning at
1 rad/s, run twice with everything identical except whether the cells are one
cellZone or two adjacent halves carrying the **same** origin, axis and omega:

| | max \|ΔU\| | mean \|ΔU\| |
|---|---|---|
| one zone vs two adjacent zones, same frame | 0.437 m/s | 0.072 m/s |

On a 1 m/s inlet. So a second zone is wrong at the interface whatever rate it
carries — this is not an approximation that a favourable Coriolis rate ratio
could buy back. Cornering emits one entry, `cornerFrame` on `all`, and the
sleeves stay in the mesh as **refinement regions** (`mode inside`), which
refine the same cells and create no surface, no patch and no zone.

They must not become `refinementSurfaces` either. A refinement surface without
a cellZone is a *wall*: snappy would snap to the sleeve and plug the inside of
each wheel with a patch nothing writes a boundary condition for.

An earlier version of this section said cornering emits a corner-frame entry
per wheel zone so those cells are not left inertial. That fixed the inertial
cells and left the interface error in place; the fix is to not create the
zones.

**Do not let MRF surfaces enter force integration.** They cannot —
`ROLE_TRAITS[MRF_ZONE].in_forces` is `False` — and this is the exact bug the old
pipeline needed a name filter to avoid.

### Decide who owns a mesh-quality threshold

`checkMesh` reaches its own pass/fail verdict using limits compiled into it —
skewness 4, non-orthogonality 70 — and by default that verdict gates the run
alongside `mesh.max_skewness` and `mesh.max_non_ortho`. Two opinions, both
have to be satisfied.

The consequence is easy to trip over: **the spec's numbers can only ever
tighten the gate.** Raising `max_skewness` to 12 leaves checkMesh still
failing the mesh at 4.6, and the config then describes something other than
what the code does — the shape §3.1 exists to prevent.

`mesh.trust_check_mesh_verdict: false` makes the spec the sole authority for
those two quantities. It is **not** a general mute: every other failure
checkMesh reports — negative volumes, non-closed cells, zero-area faces,
multiple regions — is something the pipeline cannot detect for itself and
keeps gating regardless. `gates/mesh_quality.py::SPEC_OWNED_CHECKS` is the
list that gets dropped, and it is deliberately two entries long.

`car_smoke` sets it false, because a mesh coarse enough to run in two minutes
cannot avoid one bad cell where a tyre meets the road. Every other profile
leaves it true.

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

The pipeline once computed a *true projected* area from the triangles and used
that for the blockage check, deliberately keeping it separate from `a_ref_full`
because a reference area is a convention that may exclude parts. **That
computation has been removed.** `forces.a_ref_full` is now declared and is the
only area in play: it non-dimensionalises every coefficient *and* is what
blockage is measured against. Declaring an area smaller than the car's real
silhouette therefore understates blockage in the same proportion, and nothing
checks it against the geometry any more. That is the trade.

The pipeline once computed a *true projected* area from the triangles and used
that for the blockage check, deliberately keeping it separate from `a_ref_full`
because a reference area is a convention that may exclude parts. That
computation has been removed: `forces.a_ref_full` is now declared and is the
only area in play. It non-dimensionalises every coefficient **and** is what
blockage is measured against, so declaring an area smaller than the car's real
silhouette understates blockage in the same proportion. Nothing checks it
against the geometry any more — that is the trade.

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
| Mesh gate fails on layer coverage | `logs/log.snappyHexMesh` "Added N out of M cells" per iteration | **Read the iteration trace before touching the stack sizing.** If iteration 0 places most of the cells and a later one collapses, the geometry accepted the stack and mesh-quality relaxation took it away — that is `meshQualityControls/relaxed` and `nRelaxedIter`, not the layer numbers. If extrusion percentage is low from the start, it really is the geometry: thin trailing edges, `first_layer_thickness`, `expansionRatio`, `minThickness` |
| Mesh gate fails on non-orthogonality | `logs/log.checkMesh` | Too-aggressive refinement jumps; raise `nCellsBetweenLevels` |
| Mesh gate fails although `max_skewness` was raised | `mesh.trust_check_mesh_verdict` | checkMesh judges skewness against its own hardcoded limit of 4 and that verdict gates too. The spec's number can only tighten unless the flag is off — see §6 |
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
| Cd comes out negative | `caseSpec.json` `flow.direction` | The tunnel was reversed and the drag axis was not, or the car faces the wrong way for the direction set |
| Large wheel slip angles | `status/prepare.json` `wheels.*.slip_deg` | The CAD attitude has real steer and body slip in it. Normal for an RC car in understeer; reported, never gated |
| `not a solid of revolution` on a wheel | The export | A `Tire_*` or `MRF_*` file containing more than its own part |
| Part not found, but the file is there | The filename | Patch identity is the filename. A case-only mismatch is accepted with a warning; anything else is an error |

**The provenance rule:** every result carries its `spec_hash`. If a number
surprises you, diff the `caseSpec.json` against a run you trust. That comparison
is the whole reason the spec is a text file.

---

## 9. Where this stands, and what is next

### Verified

The whole chain runs end to end in WSL against the real car CAD, on the
`car_smoke` profile, in one command:

```
prepare  ok    15 STEP parts tessellated, 4 wheels measured, annular sector built
mesh     ok    21,908 cells, non-ortho 69.1, skewness 3.7
solve    gate_failed   50 iterations - has not plateaued, correctly
post     gate_failed   y+ far outside the low_y_plus band, correctly
```

Both gate failures are the right answer for a deliberately absurd mesh. The
claim is that the **plumbing** is correct — geometry import, placement,
wheel measurement, sector meshing, cell zones, coded wheel conditions, the
rotating frame, the gates, the result record. It is not a claim about physics.

Also verified along the way: blockMesh accepts the sector and `checkMesh`
reports its volume as exactly the analytic annulus volume; snappy creates the
MRF cell zones where the wheels are; the coded cornering wheel conditions
compile; `simpleFoam` runs and writes coefficients.

### Not verified, in the order it will matter

1. **No production-resolution run has ever been *solved*.** The `car` profile
   is 1.82 M background cells before refinement — an overnight job. Everything
   in "Verified" above was measured on a mesh nobody should solve on. The mesh
   half is now being exercised at production settings (see the prism-stack note
   below); the solve half still is not.
2. **`low_y_plus` has never been run at a resolution where it means anything.**
   The wall-profile machinery is unit tested; the physics path is not. The
   `car` and `car_dev` profiles are sized on paper, not measured.
3. **The Ahmed validation has never been run**, and its reference number is
   still unpinned — see `docs/validation-ahmed.md`.
4. **Nothing about the car is validated against data.**

### Known loose ends

| Thing | State |
|---|---|
| `a_ref_full = 0.02 m²` | Declared by the user. Nothing checks it against the geometry — the projected-area computation was removed, and it now drives the blockage check too |
| ~~`TIre_RR.step`~~ | Fixed. The export is now `Tire_RR.step` and matches the patch exactly |
| Ground layers far from the car | `ground` is a blockMesh patch at the 96 mm background size, so its 450 µm stack hands off to a 96 mm cell out in the far field. Harmless where it happens (still air over bare track) and well graded under the car, where the innermost refinement shell — thicker than the ride height — takes the floor down with it. Only worth fixing if a floor-refinement region is ever added |
| CAD attitude | A heavy-understeer pose (10° body slip, 14–22° steer). Intentional; the user plans to revisit it for later driving states |
| Production wall clock | 14.78 s/iter measured, so 2000 iterations is ~8.6 h, not the 5 h that was wanted. Solver numerics were measured and are a dead end (~2% safely, and `nNonOrthogonalCorrectors 0` diverges) - see §6, "Make the solve faster". Only cells and iterations are left |
| `car_smoke` mesh limits | `trust_check_mesh_verdict: false`, skewness 20, non-ortho 75. Almost no quality net, by design |

### Deferred, with the hook already in place

| Deferred | Already provided for |
|---|---|
| Side force and yaw moment in the record | OpenFOAM already writes `Cs`, `CmYaw`, `CmRoll`; `run/parsers.py` reads only Cd and Cl |
| Curved **far**-wake refinement for cornering | `domain.refinement_shells` now cover the near field and the flow through the car, and they follow any attitude — but they follow the *car*, not the *path*, so past the outermost shell the cornering far wake is still at background size. `refinement_regions` are axis-aligned boxes and a cornering wake leaves them. The validator says so on every run |
| Per-component forces, aero balance | `forceCoeffs` renders one group; roles already separate force-bearing surfaces |
| Full plane-cut image suite | `post` exists with a minimal set |
| Rim pumping while cornering | The sleeves are meshed and refined but carry no frame of their own — a second MRF zone adjacent to the corner frame double-subtracts the frame flux on every face they share. Needs a sliding mesh to do properly |
| Transition model | `turbulence_model` is config-selected |
| Parametric sweeps | Per-run records aggregate on read; `--set` overrides one field without copying the case |

**One open risk, stated plainly:** the Ahmed validation runs at Re ≈ 2.8×10⁶ with
`high_y_plus`. It validates plumbing, numerics, domain construction, force
integration and the gates. It does **not** validate the low-Re wall-resolved
settings the RC car actually uses. Those need separate justification and,
ideally, comparison against track or tunnel data for the real vehicle. Do not
mistake a green Ahmed run for a validated RC car.
