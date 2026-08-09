# CFD Review Checklist — OpenFOAM, External Vehicle/Motorsport Aero

This is a working checklist for an experienced reviewer, not a lint rulebook.
Each item says **what tends to be wrong and why it matters** so you can judge
severity against what's actually in the case. Work top to bottom — it mirrors
how a real setup is built and where errors compound.

Defaults assumed for this domain unless the case says otherwise: incompressible
external aero, RANS (`kOmegaSST`) or scale-resolving (`DDES`/`IDDES`/`kOmegaSSTDES`),
`simpleFoam`/`pimpleFoam`, snappyHexMesh on a refined background block, moving
ground + rotating wheels, half-model with a symmetry plane.

---

## 1. Mesh (`blockMeshDict`, `snappyHexMeshDict`, `checkMesh`)

- **y+ vs wall treatment must agree.** Low-Re/wall-resolved wants y+ ≈ 1 and
  enough prism layers to resolve the boundary layer (often 10–20+ with growth
  ~1.2); high-Re/wall-functions wants y+ ≈ 30–300. A first-layer height built
  for one and `nut`/`k`/`omega` BCs written for the other is a top-3 recurring
  bug. Cross-check `firstLayerThickness`/`finalLayerThickness`/`expansionRatio`
  in `addLayersControls` against the wall-function choice in `0/`.
- **Layer coverage.** snappy frequently fails to add layers on curved/complex
  surfaces. If `log.snappyHexMesh` reports layers added on a small fraction of
  faces, the near-wall resolution you designed for doesn't exist. Look for
  `nSurfaceLayers`, relativeSizes, and the layer-addition summary in the log.
- **Refinement strategy.** Wake, underbody, wing wakes, and separation regions
  need refinement boxes/zones. A uniform coarse mesh will not resolve the
  structures that drive the forces you care about. Check `refinementRegions`,
  `refinementSurfaces` levels, and that the background `blockMesh` cell size is
  sane (cubic-ish cells before refinement).
- **`checkMesh` output is non-negotiable.** Look in `log.checkMesh` for: high
  non-orthogonality (>65–70° max is trouble, drives the `fvSchemes` corrected
  schemes and `nNonOrthogonalCorrectors`), skewness (>4 problematic), negative
  volumes (fatal), high aspect ratio in layers (expected, but extreme values
  hurt convergence). If there's no checkMesh log, that absence is a Major
  finding — nobody should run a case they haven't checked.
- **Domain size / blockage.** External aero domain should extend ~3 body-lengths
  upstream, ~5–10 downstream, with frontal blockage typically <1% (virtual
  wind tunnel) or matched to a real tunnel if validating. A cramped domain
  contaminates the pressure field and the forces.

## 2. Numerical schemes (`fvSchemes`)

- **`divSchemes` — the accuracy/stability dial.** For steady RANS aero,
  `div(phi,U)` should be `bounded Gauss linearUpwind grad(U)` — second-order but
  stabilized. Pure `upwind` is robust but adds so much numerical diffusion it
  smears wakes and under-predicts drag/separation; flag it as Major for a
  production aero run. For DES/LES, dissipation must be *low*: `LUST`,
  `linearUpwind` with care, or filtered/blended schemes — pure upwind destroys
  the resolved turbulence and defeats the point of scale-resolving.
- **Turbulence `div` terms** (`div(phi,k)`, `div(phi,omega)`) are almost always
  `bounded Gauss upwind` for stability — that's fine and expected; don't flag.
- **`gradSchemes`.** `Gauss linear`, optionally cell-limited
  (`cellLimited Gauss linear 1`) on poor meshes to bound gradients. Missing
  limiting on a high-skewness mesh → Major.
- **`laplacianSchemes` / `snGradSchemes` must match mesh non-orthogonality.**
  `Gauss linear corrected` for good meshes; `limited 0.33`–`limited 0.5` (or
  `limited corrected`) for non-orthogonal ones. A `corrected` scheme on a
  70°-non-orthogonal mesh with zero non-orthogonal correctors will be unstable
  or inaccurate. This ties directly to §3's `nNonOrthogonalCorrectors`.
- **`ddtSchemes`.** `steadyState` for `simpleFoam`. For transient,
  `backward` or `CrankNicolson 0.9` for accuracy, `Euler` only if robustness
  forces it. A transient case stuck on `Euler` "for stability" is often masking
  a Courant-number or mesh problem — note it.

## 3. Linear solvers & solution control (`fvSolution`)

- **Pressure solver.** `p`/`p_rgh` should use `GAMG` (geometric-algebraic
  multigrid) for external aero — `PCG`/`PBiCG` on a large aero mesh is needlessly
  slow. Check smoother (`GaussSeidel`/`DICGaussSeidel`) and `tolerance`
  (~1e-6–1e-7) with sensible `relTol` (0.01–0.1 transient, 0.05–0.1 steady).
- **`SIMPLE`/`PIMPLE` block consistency.**
  - SIMPLE: needs `nNonOrthogonalCorrectors` matched to the mesh,
    `consistent yes` (SIMPLEC) is common for aero and allows higher relaxation,
    and **`residualControl`** — a steady case with no residual convergence
    criterion just runs to `endTime` regardless of whether it converged.
  - PIMPLE: `nOuterCorrectors` > 1 only earns its cost if you're running large
    Courant numbers; `nCorrectors` ≥ 2; check `nNonOrthogonalCorrectors`.
- **Relaxation factors.** Steady SIMPLE typically `p` 0.3 / `U` 0.7 (or higher
  with SIMPLEC/`consistent`). Over-relaxed → divergence; massively
  under-relaxed (e.g. 0.1 across the board) → "converged" that's really just
  frozen, and wasted iterations. Inconsistent relaxation between fields and the
  `consistent` flag is a real bug.
- **`residualControl` / convergence.** Its absence on a steady case is a Major
  finding. Targets ~1e-4–1e-5 on `U`/`p` are typical, but for aero the forces
  must *also* be plateaued (see §7) — residuals alone lie.

## 4. Time/run control (`controlDict`)

- **Steady:** `endTime` is an iteration count; it must be paired with
  `residualControl` or the run is "however long I guessed," not "converged."
- **Transient Courant number.** `pimpleFoam` with `adjustTimeStep` + `maxCo`
  (often 1–5 for PIMPLE, <1 for accuracy-critical) — or a fixed `deltaT` that
  must be justified against cell size and velocity. A fixed `deltaT` with no Co
  check on a refined aero mesh is a gamble.
- **Write strategy.** `writeInterval`/`purgeWrite` sane? Writing every timestep
  on a transient run fills the disk and kills a cluster job; never writing means
  no restart capability. For DES you need enough write frequency to time-average.
- **`functions` / `libs`.** forceCoeffs, sampling, fieldAverage, residuals
  should be registered here as function objects rather than bolted on later.
  Missing `fieldAverage` on a transient/DES run means there's nothing to
  time-average — the instantaneous forces are not the answer.

## 5. Turbulence (`turbulenceProperties` / `momentumTransport`, `0/` fields)

- **Model fits the physics.** `kOmegaSST` is the workhorse for attached/mildly
  separated aero RANS. Heavy separation, sharp-edge vortices (diffusers, wing
  tips, A-pillars) are where RANS under-delivers and DES/IDDES is warranted —
  if the case has massive separation and is steady RANS, say so honestly (Major,
  with the caveat that RANS is still standard for many production loops).
- **Field consistency with the model.** `kOmegaSST` needs `k`, `omega`, `nut` —
  leftover `epsilon` files or `kEpsilon` BCs are a red flag of a half-converted
  setup. `delta`/DES-specific entries must be present for DES models.
- **Freestream turbulence is physically set, not copy-pasted.** Inlet `k` from a
  sensible turbulence intensity (often 0.1–1% for clean tunnel, higher for
  on-track), `omega`/`epsilon` from a length scale tied to the geometry — not
  textbook defaults left from a tutorial. `nut` consistent with `k`/`omega` at
  inlet. Garbage inlet turbulence → wrong transition/separation → wrong forces.
- **`nut` wall BC matches y+ strategy (again).** `nutkWallFunction`/
  `nutUSpaldingWallFunction` (high-Re) vs a low-Re treatment must match the mesh
  from §1. This is worth re-checking from the turbulence side because the two
  are edited in different files and drift apart.

## 6. Boundary conditions (`0/` fields)

- **Inlet:** `fixedValue` velocity matching the target speed (and the
  `magUInf` you'll use in forceCoeffs — cross-check!); `zeroGradient` p; turbulence
  inlets per §5.
- **Outlet:** `pressureInletOutletVelocity`/`inletOutlet` on `U`, `fixedValue`
  (0) on `p`, with `inletOutlet` on turbulence to handle any backflow. A plain
  `zeroGradient` velocity outlet that allows reversed flow without `inletOutlet`
  is a divergence risk.
- **Moving ground:** the road must be a `movingWallVelocity` (or fixedValue)
  matching freestream — a **static ground under a moving car is physically
  wrong** for ground-effect aero and a classic Critical when the case claims to
  model on-track conditions.
- **Rotating wheels:** `rotatingWallVelocity` on tyre surfaces and/or MRF zones
  in `constant/MRFProperties`. Static wheels badly misrepresent wheel wake and
  underbody flow — Critical/Major depending on what the study is for.
- **Symmetry plane:** `symmetry`/`symmetryPlane` type used correctly on the
  centre-plane of a half-model — and this **must** be reflected in the
  forceCoeffs reference area (§7).
- **Far-field/tunnel walls:** `slip` (inviscid tunnel walls) or symmetry, not
  no-slip, unless modelling a real tunnel with wall boundary layers.
- **BC type sanity:** every patch in `0/*` must have a type, and patch names must
  match `constant/polyMesh/boundary`. A field referencing a patch that no longer
  exists (or `defaultFaces` left as `empty` on a 3D case) is a setup bug.

## 7. Forces & coefficients (`forceCoeffs` function object)

This is where wrong numbers hide in plain sight — the case runs and reports
confident garbage. Scrutinize:

- **`magUInf` == actual inlet speed.** Mismatch scales CD/CL by `(U/Uref)²`.
- **`rhoInf`** set and matching the incompressible reference density used for the
  reported forces (1.0 if working in kinematic pressure, or the real air density
  if you want dimensional forces — must be internally consistent).
- **`Aref` (frontal area) and `lRef`.** For a **symmetry half-model, `Aref`
  must be the half area** (or you must halve the result) — getting this wrong is
  the canonical factor-of-2 error. `lRef` (wheelbase or reference length) drives
  the moment coefficient.
- **`liftDir`, `dragDir`, `pitchAxis`, `CofR`.** Must match the coordinate
  system and the centre of rotation you actually care about (often front-axle or
  CoG). Wrong `CofR` makes the pitching-moment / aero-balance numbers meaningless
  — and balance is often the whole point in motorsport.
- **Patches list** covers exactly the body surfaces (and not the ground/inlet).
- **Binning** (`forceCoeffs` with bins along the body) if they want a CP/load
  distribution — its absence isn't a bug, but flag it if the user's goal needs it.

## 8. Convergence & post-processing

- **Forces must be plateaued, not just residuals low.** Steady aero forces often
  keep drifting after residuals hit 1e-4. If there's a forces log, eyeball
  whether CD/CL have actually flattened. Reporting forces from an unconverged run
  is a Major-to-Critical depending on the drift.
- **Transient/DES averaging window.** Must discard initial transient, then
  time-average over enough flow-through times (several body-lengths/U). Averaging
  from t=0, or over too short a window, gives a non-representative mean. Check
  `fieldAverage` `timeStart` and total averaging duration vs flow-through time.
- **Sampling/probes** placed where they answer the question (wake planes,
  surface CP lines). Misplaced or absent sampling for a study that needs it → Minor/Major.
- **Post-processing scripts** (python/paraview) reading the right fields/times,
  not hardcoded to a stale timestep.

## 9. Automation & reproducibility (`Allrun`, `Allclean`, glue scripts)

- **`Allrun` robustness.** Should source `RunFunctions`, use `runApplication`/
  `runParallel`, and **fail loudly** — `set -e` or explicit exit-code checks. A
  pipeline that plows past a failed `snappyHexMesh` and "succeeds" with a broken
  mesh is the kind of silent failure that wastes a cluster allocation.
- **Mesh checks in the pipeline.** `checkMesh` should run (and ideally gate) after
  meshing. Skipping it in automation is a Major finding for a production loop.
- **Parallel correctness.** `decomposeParDict` present and its `numberOfSubdomains`
  matches the `mpirun -np`/`runParallel` count; `reconstructPar` (or
  `-postProcess`/on-the-fly) handled. Mismatched decomposition counts are a
  common cluster faceplant.
- **Restart / `0.orig` discipline.** `0.orig` copied to `0` at run start so the
  case is re-runnable; `Allclean` actually restores a clean state. A case that
  can't be cleanly re-run isn't reproducible.
- **Hardcoded paths / machine assumptions.** Absolute paths, hardcoded core
  counts, or environment assumptions that break on the cluster → Minor/Major.
- **Logging.** Per-application logs retained (`log.snappyHexMesh`, `log.simpleFoam`)
  so failures are diagnosable. No logs → you're flying blind on the next failure.
