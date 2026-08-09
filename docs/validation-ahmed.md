# Ahmed Body Validation

**Status: NOT VALIDATED.** The machinery below is in place and unit-tested. The
validation itself has not been run, and the reference number it would be judged
against has not been pinned. Do not cite this document as evidence that the
pipeline produces correct force coefficients.

## 1. Reference values — TO BE PINNED

Open Ahmed, Ramm & Faltin, *Some Salient Features of the Time-Averaged Ground
Vehicle Wake*, SAE 840300 (1984), and record all four of the following. The
plan is deliberate about this: **a validation without a named reference number
is not a validation.**

| # | Quantity | Value | Notes |
|---|---|---|---|
| 1 | Free-stream velocity and its Reynolds number | *unpinned* | Both Re ~ 2.8x10^6 (40 m/s) and 4.29x10^6 appear in the literature. The CFD comparison convention is the former. Record which one the target Cd belongs to. |
| 2 | Reference area used to non-dimensionalise the published Cd | *unpinned* | Explicitly record **whether the stilts are included**. `cases/ahmed/config.yaml` currently declares `a_ref_full: 0.112032` (0.389 x 0.288), stilts excluded. |
| 3 | Target Cd for the 35 degree slant | *unpinned* | With full citation to table or figure. |
| 4 | Ground condition of the experiment | *unpinned* | Expected: stationary floor. The case config assumes `ground.motion: static` on this basis. |

Until #1–#4 are filled in, the `acceptance` block is deliberately **absent**
from `cases/ahmed/config.yaml`. A placeholder target would be worse than no
target: it would make `scripts/mesh_independence.py` report a pass or fail
against a number nobody stands behind.

Once pinned, add to `cases/ahmed/config.yaml`:

```yaml
acceptance:
  target_cd: <value from the source>
  tolerance: 0.10
  source: "Ahmed, Ramm & Faltin, SAE 840300 (1984), <table/figure>"
```

## 2. Mesh independence — NOT RUN

Three refinement levels, defined in `scripts/mesh_independence.py`:

| Level | `base_cell_size` | Surface refinement | Cells | Cd | Cl |
|---|---|---|---|---|---|
| coarse | 0.08 | (3, 4) | *not run* | *not run* | *not run* |
| medium | 0.05 | (4, 5) | *not run* | *not run* | *not run* |
| fine | 0.03 | (5, 6) | *not run* | *not run* | *not run* |

**Verdict: not run.** This machine has no WSL distribution installed, so there
is no OpenFOAM to run against. Each level writes its own case YAML under the
output root, so the exact configuration behind every number is recoverable
once the runs happen.

To run it, once §1 is pinned and the environment from `docs/environment-setup.md`
is in place:

```bash
python scripts/mesh_independence.py cases/ahmed/config.yaml \
    --out-root ~/runs/ahmed-independence \
    --target-cd <pinned value> \
    --tolerance 0.10
```

Expected: three runs complete; Cd converges monotonically; the finest level
lands within 10 % of the pinned target. Record the actual numbers in the table
above.

If it fails, diagnose in this order **before** touching the turbulence model:

1. y+ gate output — is the wall treatment actually valid?
2. Layer coverage on `body` — did snappy deliver the layers?
3. Blockage ratio and domain extents.
4. The nose fillet approximation in `pipeline/simdev/geometry/ahmed.py`. The
   nose is a lofted approximation of the 100 mm fillet, not a true spherical
   fillet; the corner where the two curvatures meet is interpolated. This is
   documented at the point of implementation and is the known geometric
   deviation from the reference body.

## 3. What this validation would and would not establish

Once run and passing, this establishes correctness of the pipeline's plumbing,
numerics, domain construction, force integration and gates at the Ahmed
condition. It does not extend beyond that condition. Stated explicitly, because
the RC car work will be tempted to lean on it:

> Ahmed at Re ~ 2.8x10^6 with `high_y_plus` validates the pipeline's plumbing,
> numerics, domain construction, force integration and gates. It does **not**
> validate the low-Re, wall-resolved settings the RC car will use. Those
> require separate justification and, ideally, comparison against track or
> tunnel data for the actual vehicle.

## 4. What *is* established today

Independent of the Ahmed runs, the following are covered by the unit suite
(172 tests, no OpenFOAM required):

- `a_ref` is halved for a half model and only there; the halving lives in one
  property and no other code divides by two.
- Symmetry is derived from mode and yaw, never set, and the patch set is
  asserted to agree with it.
- Force integration includes `body` and `tyre` patches only — never the
  ground, never an MRF zone.
- A command that exits 0 having printed `FOAM FATAL` is treated as a failure.
- A stage is skipped only on an input-hash match, never because its output
  directory looks populated.
- A non-converged solve is recorded and flagged rather than discarded.
- Every result is written per run; nothing appends to a shared table.
