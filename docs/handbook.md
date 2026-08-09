# SimDev Pipeline Handbook

How the pipeline works, **why** it works that way, and how to change it.

The spec (`docs/superpowers/specs/2026-08-09-openfoam-rc-car-cfd-pipeline-design.md`)
says what we decided. The plan
(`docs/superpowers/plans/2026-08-09-openfoam-pipeline-vertical-slice.md`) says how
to build it. This handbook is for the person who has to live with it afterwards.

> **Status:** written against the design, before implementation. Section 5
> (module map) and Section 8 (debugging) need a pass once the code exists —
> they describe intent, and intent drifts.

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
built-in defaults ◄ resolution profile ◄ wall-treatment profile ◄ case file ◄ CLI overrides
```

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
  half_model = (mode is STRAIGHT) and (yaw_deg == 0.0)
  ```

- A validator asserts the patch set agrees: half model ⟺ a `symmetry` patch
  exists. Neither can be true alone.

**Note the yaw term.** Yaw breaks symmetry even in straight-line mode. This is
easy to get wrong because "straight line" sounds symmetric. Cornering is not the
only asymmetric case.

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
returns warnings) → generate or copy geometry into `constant/triSurface/` →
check watertightness and units → compute the union bounding box → build the
domain → compute the *true projected* frontal area and assert blockage → render
every dictionary → write `caseSpec.json`.

Runs without OpenFOAM installed. This is the stage that turns a config file into
a complete, hand-runnable case.

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
| `geometry/stl.py` | STL info, watertightness, projected area | nothing |
| `domain/base.py` | `DomainBox`, `DomainBuilder` protocol | schema |
| `domain/box.py` | Rectangular tunnel, blockage | base, roles |
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

### Add a new geometry source

Implement a writer with the same contract as `write_ahmed_stl`:
`(params, out_dir) -> dict[str, Path]` mapping patch name to STL path. Add a
branch in `_write_geometry()` in `stages/prepare.py` and a `kind` value in
`GeometryConfig`.

Keep the STL **full-body and watertight** even for half models. Half models are
produced by the *domain* restricting to y ≥ 0 with a `symmetry` patch, not by
cutting geometry. Cutting geometry makes it non-watertight, and snappyHexMesh
leaks into non-watertight surfaces.

### Add the cornering domain

This is the big one, and the architecture was shaped around it.

1. Implement `AnnulusDomainBuilder` in `domain/annulus.py` satisfying the
   `DomainBuilder` protocol. It builds a curved sector; ω = U/R is already
   available as `CaseSpec.omega_rotation`.
2. `DomainBox` will not describe a sector — introduce a common return type or a
   protocol both satisfy. Do this deliberately; do not bolt sector fields onto
   `DomainBox`.
3. Add `blockMeshDict_annulus.jinja` and select the template set by
   `spec.domain.kind`.
4. **The ground BC changes.** In the rotating frame the road surface moves; the
   straight-line `fixedValue (U 0 0)` is wrong there. Add the rotating-frame
   expression in `build_bcs()`.
5. Remove the "cornering requires the annulus domain" error in
   `config/validate.py` — and *only* that one. The half-model/cornering
   assertion must stay.
6. Force axes are in the car frame, and **side force becomes first-class**. Check
   `liftDir`/`dragDir`/`CofR` in `render/context.py`.

The validator already rejects a half-model cornering case, so you cannot
accidentally ship one.

### Add MRF / rotating wheels

Two mechanisms, and you will likely want both:

- **`rotatingWallVelocity` on tyre surfaces** — cheap, correct for a smooth
  tread. Add the BC branch for `PatchRole.TYRE` in `build_bcs()`. Needs the
  wheel axis and ω per wheel in config.
- **A genuine MRF cell zone** — needed where rim and spoke geometry actually
  pump air. Requires a closed STL per zone, a `cellZone` created during
  meshing, and `constant/MRFProperties`. Add an `MRFProperties.jinja` and a
  `mrfZone` branch in `_write_geometry`.

**Do not let MRF surfaces enter force integration.** They already cannot —
`ROLE_TRAITS[MRF_ZONE].in_forces` is `False` — but verify it when you wire this
up, because it is the exact bug the old pipeline needed a name filter to avoid.

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
| Coefficients ~2× expected | `caseSpec.json` `half_model` and `a_ref_full` | Should be impossible by construction — if it happens, the derivation is broken and that is a bug worth a test |

**The provenance rule:** every result carries its `spec_hash`. If a number
surprises you, diff the `caseSpec.json` against a run you trust. That comparison
is the whole reason the spec is a text file.

---

## 9. What is deliberately not here yet

| Deferred | Already provided for |
|---|---|
| Cornering / annular domain | `DomainBuilder` protocol; `Mode.CORNERING` validates and rejects cleanly |
| MRF zones, rotating wheels | `tyre` and `mrfZone` roles defined and unused |
| Yaw / pitch / roll attitude | Mode table drives symmetry derivation |
| Transition model | `turbulence_model` is config-selected |
| Full plane-cut image suite | `post` stage exists with a minimal set |
| Parametric sweeps | Per-run records aggregate on read |

Each is a new module behind an interface the current code already defines. That
was the point of building the slice first.

**One open risk, stated plainly:** the Ahmed validation runs at Re ≈ 2.8×10⁶ with
`high_y_plus`. It validates plumbing, numerics, domain construction, force
integration and the gates. It does **not** validate the low-Re wall-resolved
settings the RC car actually uses. Those need separate justification and,
ideally, comparison against track or tunnel data for the real vehicle. Do not
mistake a green Ahmed run for a validated RC car.
