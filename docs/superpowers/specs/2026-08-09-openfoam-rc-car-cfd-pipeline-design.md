# OpenFOAM CFD Pipeline for RC Car External Aerodynamics — Design

**Date:** 2026-08-09
**Status:** Approved design, pending implementation plan

---

## 1. Purpose

An automated OpenFOAM pipeline for external aerodynamic simulation of 1/10 and
1/8 scale RC cars, covering preparation, meshing, solving and post-processing.

It replaces a STAR-CCM+ pipeline (`benchmark_old_pipeline/`) that ran on the
RWTH SLURM cluster. The new pipeline runs entirely on a local workstation.

The end state supports both straight-line and cornering simulation, with
rotating wheels. This document specifies the full architecture and the scope of
a first vertical slice that must run end to end and be validated against a
known answer before further capability is added.

## 2. Context — what the benchmark got right and wrong

The old pipeline (`benchmark_old_pipeline/`) is the specification-by-example.
Worth carrying over:

- Chained stages with dependency gating (`SLURMshell_Jobchain.txt`)
- Config/preset separation with explicit override semantics (`simConfig.txt:17-23`)
- Domain sizing: 45 m tunnel, 2/3 of length behind the car — roughly 5 body
  lengths upstream, 10 downstream (`simConfig.txt:100,103`)
- Per-component force breakdown and centre of pressure
  (`Templates/MonitorNamesCAR.txt`) — aero balance is the decision-relevant output

Deliberately not carried over, each of which motivates a design decision below:

| Benchmark defect | Location | Design response |
|---|---|---|
| No convergence criterion; fixed iteration count, then average the last 100 regardless | `VMesh_Sim.java:21`, `Evalquery.java:171` | Convergence gate (§7) |
| `avgIterations=300` in config, `100` hardcoded in code | `simConfig.txt:52` vs `Evalquery.java:171` | Single resolved `CaseSpec`; no implicit defaults (§5.1) |
| Unresolved symmetry factor-of-2, left as `TODO Bug??` | `Evalquery.java:703-704` | `Aref` derived from model state, asserted (§5.5) |
| Results appended to one shared file, no locking | `Evalquery.java:685-687` | Per-run records, never shared-append (§8) |
| Stale outputs silently reused when a directory looks populated | `Evalquery.java:179,560` | Input-hash based staleness, `--force` (§7) |
| Physics locked inside an undiffable 49 MB binary | `Cornering.sim` | `caseSpec.json` provenance (§8) |
| Reference velocity inferred from file path substring | `Evalquery.java:1478` (dead code) | All reference values from config, asserted |
| Secrets committed | `simConfig.txt:6-7` | No credentials in repo; not applicable locally |

## 3. Environment and constraints

- **Hardware:** dual Intel Xeon Gold 6148 — 40 physical cores / 80 threads,
  128 GB RAM, 7.45 TB free on `D:`, 330 GB on `C:`
- **Runtime:** WSL2 + Ubuntu + ESI OpenFOAM v2412
- **Language:** Python 3 for orchestration
- Cases live on the ext4 side of WSL, not `/mnt/c` or `/mnt/d` — bind-mounted
  I/O collapses OpenFOAM performance. The WSL virtual disk is relocated to `D:`.
- **Decomposition uses 40 physical cores, not 80 threads.** OpenFOAM is
  memory-bandwidth bound; hyperthreading costs performance. `scotch` with MPI
  pinning, or `hierarchical` split across the two sockets. No claim of
  NUMA-awareness is made — the decomposer does not model it.
- **Runtime budget:** a few hours for a production run is acceptable. The
  development profile must run in minutes.

## 4. Physics and modelling decisions

### 4.1 Regime

Target vehicle: 1/10–1/8 scale RC car. Reference condition L ≈ 0.40 m,
U ≈ 15 m/s, ν = 1.5×10⁻⁵ m²/s.

| | body (L = 0.40 m) | appendage (chord ≈ 35 mm) |
|---|---|---|
| Re | 4.0×10⁵ | 3.5×10⁴ |
| δ (turbulent estimate) | ≈ 11 mm | ≈ 1.6 mm |
| first cell for y⁺ = 30 | ≈ 1.3 mm | ≈ 1.3 mm |
| first cell / δ | 12 % | **80 %** |

Estimates use `Cf ≈ 0.058·Re^-0.2`, `u_τ = √(τ_w/ρ)`, `y = y⁺·ν/u_τ`, first
cell height ≈ 2× the centre distance. They size the mesh; they are not
predictions.

Two consequences drive the modelling choices:

1. **High-y⁺ wall functions are invalid on appendages at this scale.** A first
   cell sized for y⁺ ≈ 30 spans most of the boundary layer on a wing element or
   diffuser vane — where the downforce is generated.
2. **Re_chord ≈ 3.5×10⁴ is low-Re airfoil territory**, governed by laminar
   separation bubbles. Fully-turbulent RANS assumes a turbulent boundary layer
   from the leading edge and cannot represent them.

### 4.2 Decisions

- **Solver:** `simpleFoam`, steady, incompressible. Mach ≪ 0.3 throughout.
- **Turbulence:** `kOmegaSST`, config-selected so `kOmegaSSTLM` (Langtry-Menter
  γ-Reθ) can be substituted without restructuring. Fully turbulent initially;
  the accuracy ceiling on appendages is accepted and documented, not hidden.
- **Wall treatment:** a config profile, not a global constant.
  - `low_y_plus` — y⁺ ≈ 1, first cell ≈ 40 µm at the RC reference condition,
    15–20 layers, `nutLowReWallFunction`. **Default for the RC car.** Also the
    prerequisite for enabling transition modelling later.
  - `high_y_plus` — y⁺ 30–300, 6–8 layers, `nutkWallFunction`. Used for the
    Ahmed validation case, where wall-resolving at Re ≈ 2.8×10⁶ (≈ 10 µm first
    cell) would make the acceptance test too slow to iterate against.
  - `spalding` — `nutUSpaldingWallFunction`, buffer-layer tolerant. Available
    but not a default; it can mask layer collapse rather than expose it.
  - `omegaWallFunction` is correct under both profiles — it blends to the
    low-Re near-wall form.
- **Ground:** a per-case field, `motion: static | moving`. Not baked into the
  patch role. A vehicle case with static ground raises a loud validator warning;
  the Ahmed validation case uses `static` deliberately (§9).

## 5. Architecture

### 5.1 Resolution and provenance

```
config.yaml ─┐
profile ─────┼─► resolve ─► CaseSpec ─► validate ─► render ─► run dir
CLI overrides┘              (explicit)   (asserts)   (jinja)
                                 │
                                 └─► caseSpec.json  (provenance)
```

Config layers merge in order: built-in defaults ◄ profile ◄ case config ◄ CLI
overrides. The result is a **`CaseSpec` in which every value is explicit** —
nothing downstream reads raw config or applies a hidden default. The
`CaseSpec` is written to the run directory as JSON and is the reviewable,
diffable record of what was simulated. Every results record carries its hash.

### 5.2 Repository layout

```
pipeline/simdev/
  config/     schema.py      typed model (pydantic): Domain, Mesh, Physics, Post
              resolve.py     layer merge → CaseSpec
              validate.py    cross-file consistency assertions
  geometry/   ahmed.py       procedural Ahmed body STL
              stl.py         load, unit/scale check, watertightness, bbox
              roles.py       patch role assignment
  domain/     base.py        DomainBuilder interface
              box.py         straight rectangular tunnel
              annulus.py     curved cornering sector (deferred)
  render/     templates/     jinja → case dictionaries
              render.py
  stages/     prepare.py  mesh.py  solve.py  post.py
  gates/      mesh_quality.py    parses log.checkMesh
              convergence.py     force-plateau detection
              yplus.py           post-run y⁺ band check
  run/        runner.py      subprocess/mpirun wrapper, log parsing, exit codes
  report/     results.py     per-run JSON + CSV
              plots.py
cases/ahmed/config.yaml
docs/superpowers/specs/
```

### 5.3 Patch roles

Every surface carries a role. The role drives four things that would otherwise
be edited in four files and drift apart:

| role | `U` BC | in forces | refinement | MRF |
|---|---|---|---|---|
| `body` | `noSlip` | yes | high | — |
| `tyre` | `rotatingWallVelocity` | yes | high | optional zone |
| `ground` | per §4.2 ground field | no | medium | — |
| `symmetry` | `symmetry` | no | — | — |
| `farfield` | `slip` | no | — | — |
| `inlet` | `fixedValue` | no | — | — |
| `outlet` | `inletOutlet` | no | — | — |
| `mrfZone` | cellZone, not a patch | **excluded** | — | yes |

`mrfZone` surfaces are excluded from force integration. The benchmark needed an
explicit name filter for this (`Evalquery.java:105`); here it follows from the
role.

### 5.4 Simulation modes

Symmetry is derived, never set independently. The trigger is
`mode == cornering or yaw != 0` — **yaw alone breaks symmetry even in
straight-line mode.**

| | `straight`, yaw = 0 | `straight`, yaw ≠ 0 | `cornering` |
|---|---|---|---|
| model | half | full | full |
| domain | box + symmetry plane | box | annular sector |
| centre-plane BC | `symmetry` | — | — |
| `Aref` | ½ frontal | full frontal | full frontal |
| frame | inertial | inertial | rotating, ω = U/R |
| ground BC | moving wall | moving wall | rotating-frame consistent |
| force axes | car frame | car frame | car frame; side force first-class |

In the rotating frame the road surface is moving, so the cornering ground BC is
not the same expression as the straight-line one. A half-model cornering case
must be unconstructable, not merely discouraged.

### 5.5 Validation assertions

These run at generation time, in under a second, instead of surfacing as a
wrong number hours later:

- `nut`/`k`/`omega` wall functions match the active wall-treatment profile
- `forceCoeffs.magUInf` equals inlet `U`
- `rhoInf` consistent with the kinematic-pressure convention in use
- **`Aref` is halved if and only if a `symmetry` role is present**
- `liftDir`, `dragDir`, `CofR` consistent with the domain's coordinate frame
- `decomposeParDict.numberOfSubdomains` equals the MPI rank count actually launched
- the turbulence model's required fields are all present, with no orphans
  (no `epsilon` beside a `kOmegaSST`)
- every patch in `0/*` exists in the mesh boundary, and vice versa
- blockage ratio computed and asserted — < 1 % for virtual-tunnel cases; matched
  to the reference tunnel for the Ahmed case
- mode/symmetry/`Aref`/domain row of §5.4 is internally consistent

### 5.6 Profiles

Resolution is a config axis, not a forked case, so the development path and the
production path are the same code:

- `dev` — very coarse, minutes, ~8 ranks
- `production` — full resolution, hours, 40 ranks

Wall treatment is an independent axis (§4.2).

## 6. Stages

Each stage is independently invocable (`simdev mesh <run>`), restartable, and
writes its own log plus a `status.json` recording `ok | failed | gate_failed`
with details.

| stage | actions | exit gate |
|---|---|---|
| `prepare` | resolve → validate → `CaseSpec`; STL generate/load, unit and watertightness check, role assignment; domain build; render dicts | config validation, blockage |
| `mesh` | `blockMesh` → `surfaceFeatures` → `snappyHexMesh` → `checkMesh` | mesh quality |
| `solve` | `decomposePar` → `mpirun simpleFoam`; runtime function objects for forces, residuals, y⁺ | convergence |
| `post` | y⁺ band check, coefficient extraction, plots, plane cuts, results record | y⁺ band |

Note: the utility is `surfaceFeatures` in ESI v2412, not `surfaceFeatureExtract`.

## 7. Gates

**Mesh quality** — parses `log.checkMesh`: max non-orthogonality, skewness,
negative volumes, and **layer coverage fraction**. Layer coverage is a gate
rather than a log line because snappyHexMesh silently drops layers on thin
trailing edges, and under `low_y_plus` that means the near-wall resolution does
not exist precisely where appendage loads are generated.

**Convergence** — `residualControl` plus a force-plateau test on Cd and Cl
(rolling mean and standard deviation over a trailing window, against a
tolerance). It **terminates at `maxIterations` and emits a result explicitly
marked non-converged** — it never hangs, and never silently passes. The
reported coefficient is the **mean over the plateau window**, with window
bounds and standard deviation recorded alongside it.

**y⁺ band** — computes the y⁺ distribution on `body` patches after the solve and
fails or warns on the fraction outside the active profile's band. Under
`low_y_plus` this is the check that catches a collapsed layer stack.

## 8. Failure handling, results, provenance

Every external call's exit code is checked and a failed stage stops the chain.
Because OpenFOAM frequently exits 0 on partial failure, the runner **also**
parses logs for `FOAM FATAL`, floating-point exceptions, and the snappyHexMesh
layer-addition summary.

No stage silently skips work. Re-running a completed stage requires `--force`,
or is triggered automatically when the input hash changes. Staleness is never
inferred from directory contents.

Results are written **per run** as JSON plus a CSV row, each carrying the
`caseSpec.json` hash. Nothing is ever appended to a shared file — the benchmark's
concurrent-append table (`Evalquery.java:685-687`) is not reproduced.

## 9. Validation

**Acceptance case:** Ahmed body, **35° slant**, procedurally generated from
published dimensions, fixed floor, stilts included, `high_y_plus` profile.

35° rather than 25° because the 25° case sits on the separation/reattachment
bifurcation where steady RANS fails for turbulence-model reasons. Validating
there would tell you nothing about the pipeline.

The ground is **static**, matching the fixed tunnel floor of the reference
experiment. This is why ground motion is a per-case field rather than role
behaviour (§4.2).

**First task of the validation work package** is to pin the reference from the
primary source (Ahmed, Ramm & Faltin, SAE 840300, 1984): exact free-stream
velocity and Reynolds number, `Aref`, whether the quoted Cd includes the stilts,
and the target Cd with its citation. Both Re ≈ 2.8×10⁶ (40 m/s) and 4.29×10⁶
appear in the literature; the CFD comparison convention is the former. These
values are written into the case config as the acceptance criterion. A
validation without a named reference number is not a validation.

**Acceptance requires** the pinned Cd matched **within 10 %** on the finest of
**three refinement levels showing monotonic convergence.** One mesh matching one
number can be error cancellation. 10 % is the band published steady-RANS results
for the 35° Ahmed body typically fall in; the figure lives in the case config so
it can be tightened once the reference is pinned and the first results are in.

**Explicit limitation:** Ahmed at Re ≈ 2.8×10⁶ validates plumbing, numerics,
domain construction, force integration and the gates. It does **not** validate
the low-Re, wall-resolved settings the RC car will use. Those require separate
justification and, ideally, comparison against track or tunnel data for the
actual vehicle. This is a known open risk, not an oversight.

## 10. Testing

Unit-testable logic is pure and needs no OpenFOAM installation:

- **unit** — config merge and resolution; every validator, asserting that bad
  configs are *rejected* (half-model cornering, `magUInf`/inlet mismatch,
  wall-function/profile mismatch); Ahmed geometry dimensions; blockage maths;
  log parsers for `checkMesh`, forces, and snappy layer summaries against
  fixture logs
- **smoke** — a deliberately tiny case, a few thousand cells and ~20 iterations,
  running the full chain in 1–2 minutes. This is what makes the pipeline
  developable
- **validation** — the Ahmed acceptance test of §9

## 11. Scope of the vertical slice

**In:**

- Box domain, straight-line, zero yaw, half model with symmetry plane
- Ahmed body 35°, static ground, `high_y_plus`
- snappyHexMesh with layers; `checkMesh` gate
- `simpleFoam` + `kOmegaSST`, steady
- `forceCoeffs`, convergence gate, y⁺ check
- Per-run results record; residual and force-history plots
- `dev` and `production` resolution profiles
- CLI: `simdev run cases/ahmed --profile dev`

**Out, but architecturally provided for:**

| deferred | provision already in the slice |
|---|---|
| cornering / annular domain | `DomainBuilder` interface, one implementation |
| MRF zones, rotating wheels | `tyre` and `mrfZone` roles defined, unused |
| yaw / pitch / roll attitude | mode table (§5.4) drives symmetry |
| transition model | turbulence is config-selected |
| full plane-cut image suite | `post` stage exists with a minimal set |
| parametric sweeps | per-run results records aggregate on read |

Each deferred item is a new module behind an interface the slice already
defines.

## 12. Success criteria for the slice

1. `simdev run cases/ahmed --profile dev` completes all four stages from a clean
   checkout, in minutes.
2. Every gate demonstrably fires: a deliberately broken mesh fails the quality
   gate; a truncated run is marked non-converged; a wall-function/profile
   mismatch is rejected at generation time.
3. The Ahmed 35° acceptance test matches the pinned reference Cd within 10 % on
   the finest of three refinement levels, with monotonic convergence.
4. The full unit and smoke suites pass without OpenFOAM present for the unit
   portion.

## 13. Dependencies

Python: `pydantic`, `jinja2`, `numpy`, `pandas`, `matplotlib`, and an STL
library (`numpy-stl` or `trimesh`). Recorded in `requirements.txt` as they are
introduced.
