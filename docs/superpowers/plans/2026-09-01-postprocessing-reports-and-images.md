# Post-processing: Reports and Images — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every run a tab-separated force/moment/COP report that can be pasted into a spreadsheet, and a car-frame image suite whose slices, cameras and colour limits are identical across runs.

**Architecture:** Phase one extends the existing `post` stage — one new `forces` function object, a positional parser, COP and group arithmetic, and a TSV writer. Phase two adds an `images` stage: OpenFOAM samples slices and patches to VTK in parallel on the decomposed case, a pure-data render plan is built in the venv, and a standalone script executed by the system Python (which is the only place `paraview` imports) turns that plan into PNGs.

**Tech Stack:** Python 3.11+, pydantic v2, pandas, matplotlib, Jinja2, OpenFOAM ESI v2412 (`postProcess`, `forces`, `vorticity`, `Lambda2`, `surfaces`), ParaView 6.0.1 via `python3-paraview`.

**Spec:** `docs/superpowers/specs/2026-09-01-postprocessing-reports-and-images-design.md`

## Global Constraints

- **Units are metres and SI throughout the case.** The STL cache is millimetres; `geometry.scale` converts. Never mix.
- **Signs are OpenFOAM's.** `liftDir` is `(0 0 1)`, so positive `Cl` and positive `F_z` mean *up*. Downforce is negative. Do not flip.
- **Car axes:** nose +x, up +z, therefore **car-left is +y**. Confirmed against the CAD.
- **The case is never reconstructed.** `decomposePar` runs once in `mesh`; everything after runs `-parallel` on that decomposition. Never call `reconstructPar`.
- **Never append to a shared file.** Per-run records only; aggregate on read. See `report/results.py`.
- **Function objects execute in dictionary order.** Anything reading another's output must be declared after it.
- **Averaging window:** every reported quantity uses the window `check_convergence` already computes. Nothing gets its own.
- **`ResultRecord` fields are added with defaults**, so existing `result.json` files still load.
- **No new venv dependency.** ParaView is a system package reached by subprocess, never imported into the venv.
- **Tests that need OpenFOAM are marked `openfoam`; tests that need ParaView are marked `paraview`.** The suite must pass on a machine with neither.

---

## File Structure

| File | Responsibility |
|---|---|
| `render/templates/controlDict.jinja` | **modify** — add the `forces` function object |
| `render/templates/sampleSurfaces.jinja` | **create** — `vorticity`, `Lambda2`, `surfaces`, and `enabled false` for every inherited object |
| `run/parsers.py` | **modify** — `read_force_vectors()` |
| `report/forces.py` | **create** — window means, COP, balance, group sums, `ForceReport` |
| `report/tsv.py` | **create** — per-run `report.tsv`, aggregate across runs |
| `report/plots.py` | **modify** — `plot_balance()` |
| `report/results.py` | **modify** — `ResultRecord` gains force/moment/COP/group fields |
| `config/schema.py` | **modify** — `PostConfig.groups` |
| `config/validate.py` | **modify** — group coverage check |
| `viz/views.py` | **create** — parse `cases/post_views.yaml` into `Views` |
| `viz/datum.py` | **create** — car datum from the `Chassis` bbox |
| `viz/plan.py` | **create** — `Views` + spec + datum → `render_plan.json` |
| `viz/sample.py` | **create** — render the sampling dict, run `postProcess -parallel` |
| `viz/pv_render.py` | **create** — standalone, system Python, imports no `simdev` |
| `viz/sheet.py` | **create** — contact-sheet `index.html` |
| `stages/post.py` | **modify** — call the report, write `report.tsv` |
| `stages/images.py` | **create** — the new stage |
| `stages/prepare.py` | **modify** — record the datum in `status/prepare.json` |
| `cases/post_views.yaml` | **create** — the shared, versioned view definition |
| `cli.py` | **modify** — `simdev images`, `simdev report`, doctor check |

`viz/` rather than extending `render/`: throughout this repo `render/` means "turn a spec into OpenFOAM dictionary text", and `docs/handbook.md`'s dependency rule depends on that meaning. The sampling dictionary *is* dictionary text and does belong in `render/templates/`; pictures do not.

---

# Phase One — the report

Phase one ships and is useful on its own. It does not depend on anything in phase two.

---

### Task 1: The `forces` function object

**Files:**
- Modify: `pipeline/simdev/render/templates/controlDict.jinja`
- Test: `tests/test_render_solver.py`

**Interfaces:**
- Consumes: the existing `force_patches` and `spec` context variables from `render/context.py::build_context`.
- Produces: `postProcessing/forces/<t>/force.dat` and `moment.dat` at run time. Nothing in Python consumes this task directly; Task 2 reads the files.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_render_solver.py`:

```python
def test_controldict_writes_forces_in_newtons(tmp_path: Path) -> None:
    """forceCoeffs reports coefficients only.

    F in newtons and M in newton-metres need their own object. Recovering
    them from Cd/Cs/Cl would mean assuming how OpenFOAM maps
    CmRoll/CmPitch/CmYaw onto Cartesian axes, and this pipeline does not
    assume conventions it can read directly.
    """
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    block = text[text.index("\n    forces\n") :]
    block = block[: block.index("\n    }")]

    assert "type            forces;" in block
    assert 'libs            ("libforces.so");' in block
    assert "rho             rhoInf;" in block
    assert "CofR            (0.5 0.0 0.0);" in block
    assert "body" in block


def test_the_forces_object_is_aggregate_only(tmp_path: Path) -> None:
    """One forces object, not one per patch.

    The per-patch split the report needs is a split of *coefficients*, which
    forceCoeffs_<patch> already supplies. Eleven more force objects would add
    eleven directories under postProcessing/ and no information.
    """
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    assert "forces_body" not in text
    assert text.count("type            forces;") == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_render_solver.py -k forces -v`
Expected: FAIL — `ValueError: substring not found` on `text.index("\n    forces\n")`.

- [ ] **Step 3: Add the object to the template**

In `pipeline/simdev/render/templates/controlDict.jinja`, immediately after the closing brace of the aggregate `forceCoeffs` block and before the per-patch `{% for p in force_patches %}` loop:

```jinja
    // FORCES IN NEWTONS AND MOMENTS IN NEWTON-METRES.
    //
    // forceCoeffs above reports coefficients only, and a report that gets
    // pasted into a spreadsheet wants dimensional numbers. Recovering them
    // from Cd/Cs/Cl would mean assuming how OpenFOAM maps CmRoll, CmPitch
    // and CmYaw onto Cartesian axes; this object writes x, y and z
    // components directly and there is nothing left to guess.
    //
    // AGGREGATE ONLY. The per-patch split the report needs is a split of
    // *coefficients*, which forceCoeffs_<patch> below already supplies.
    // Eleven more force objects would add eleven directories and no
    // information.
    //
    // v2412 writes force.dat and moment.dat as ten tab-separated columns:
    // Time, then total, pressure and viscous vector triples in that order
    // (forces.C::writeIntegratedDataFileHeader). No porosity here, so there
    // is no fourth triple. run/parsers.py::read_force_vectors depends on it.
    forces
    {
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        log             false;

        patches         ({% for p in force_patches %}{{ p }} {% endfor %});

        rho             rhoInf;
        rhoInf          {{ spec.flow.rho }};
        CofR            ({{ spec.forces.c_of_r[0] }} {{ spec.forces.c_of_r[1] }} {{ spec.forces.c_of_r[2] }});
    }

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_render_solver.py -v`
Expected: PASS, including the pre-existing tests — the new block must not disturb the `yPlus` / `yPlusArea_` ordering assertion.

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/render/templates/controlDict.jinja tests/test_render_solver.py
git commit -m "feat: forces in newtons, because a coefficient does not paste into a load case"
```

---

### Task 2: Read `force.dat` and `moment.dat`

**Files:**
- Modify: `pipeline/simdev/run/parsers.py`
- Create: `tests/fixtures/force.dat`, `tests/fixtures/moment.dat`, `tests/fixtures/force_two_triples.dat`
- Test: `tests/test_parsers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `read_force_vectors(path: Path) -> pd.DataFrame` with columns `Time`, `x`, `y`, `z` — the **total** vector, one row per written iteration.

- [ ] **Step 1: Write the fixtures**

`tests/fixtures/force.dat` — the v2412 layout, ten columns:

```
# Force
# CofR : (0.5 0 0)
#
# Time         	total_x total_y total_z	pressure_x pressure_y pressure_z	viscous_x viscous_y viscous_z
1	1.0 0.2 -10.0	0.9 0.15 -9.5	0.1 0.05 -0.5
2	2.0 0.4 -20.0	1.8 0.30 -19.0	0.2 0.10 -1.0
3	3.0 0.6 -30.0	2.7 0.45 -28.5	0.3 0.15 -1.5
```

`tests/fixtures/moment.dat`:

```
# Moment
# CofR : (0.5 0 0)
#
# Time         	total_x total_y total_z	pressure_x pressure_y pressure_z	viscous_x viscous_y viscous_z
1	0.5 1.0 0.1	0.45 0.9 0.09	0.05 0.1 0.01
2	1.0 2.0 0.2	0.90 1.8 0.18	0.10 0.2 0.02
3	1.5 3.0 0.3	1.35 2.7 0.27	0.15 0.3 0.03
```

`tests/fixtures/force_two_triples.dat` — an older build with no `total` column:

```
# Force
# Time	pressure_x pressure_y pressure_z	viscous_x viscous_y viscous_z
1	0.9 0.15 -9.5	0.1 0.05 -0.5
2	1.8 0.30 -19.0	0.2 0.10 -1.0
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_parsers.py`:

```python
from simdev.run.parsers import read_force_vectors


def test_read_force_vectors_takes_the_total_column() -> None:
    frame = read_force_vectors(FIXTURES / "force.dat")
    assert list(frame["Time"]) == [1.0, 2.0, 3.0]
    assert frame["z"].iloc[-1] == pytest.approx(-30.0)
    assert frame["x"].iloc[0] == pytest.approx(1.0)


def test_read_force_vectors_sums_when_there_is_no_total() -> None:
    """Older builds write pressure and viscous only.

    The total is recoverable by addition, so the layout is accepted rather
    than rejected - but it is recognised by column count, never assumed.
    """
    frame = read_force_vectors(FIXTURES / "force_two_triples.dat")
    assert frame["z"].iloc[0] == pytest.approx(-10.0)
    assert frame["x"].iloc[0] == pytest.approx(1.0)


def test_read_force_vectors_refuses_an_unknown_layout(tmp_path: Path) -> None:
    """A width this parser has not been taught is an error, not a guess.

    read_y_plus_area already learned this: a header whose token count
    disagrees with the data does not raise, it silently yields NaN, and a NaN
    that reaches a report reads as a number nobody checked.
    """
    bad = tmp_path / "force.dat"
    bad.write_text("# Time total_x\n1\t2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has not been taught"):
        read_force_vectors(bad)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_parsers.py -k force_vectors -v`
Expected: FAIL — `ImportError: cannot import name 'read_force_vectors'`.

- [ ] **Step 4: Implement the parser**

Append to `pipeline/simdev/run/parsers.py`:

```python
def read_force_vectors(path: Path) -> pd.DataFrame:
    """Time and the TOTAL force (or moment) vector from a v2412 forces object.

    Read POSITIONALLY, never by header name, for the reason spelled out in
    read_y_plus_area: a header whose token count disagrees with the data
    columns does not raise, it silently produces NaN, and a NaN that reaches
    report.tsv reads as a number somebody measured.

    v2412 writes ten tab-separated columns and no parentheses -
    forces.C::writeIntegratedDataFileHeader:

        Time  total_{x,y,z}  pressure_{x,y,z}  viscous_{x,y,z}

    Older builds omit the total and write seven. Both are accepted because
    the total is recoverable from either - it is the first triple when there
    are three and the sum when there are two - but the layout is recognised
    by width rather than assumed. A `porosity true` object would add a fourth
    triple; the pipeline sets no porous zones, so thirteen columns means
    something changed and is rejected rather than mis-sliced.
    """
    rows: list[dict[str, float]] = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(token) for token in line.split()]
        if len(values) == 10:
            total = values[1:4]
        elif len(values) == 7:
            total = [values[1 + i] + values[4 + i] for i in range(3)]
        else:
            raise ValueError(
                f"{path}: expected 7 or 10 numeric columns per row, got "
                f"{len(values)} - a forces-object layout this parser has not "
                "been taught. Do not widen the slice without checking which "
                "triple is which."
            )
        rows.append(
            {"Time": values[0], "x": total[0], "y": total[1], "z": total[2]}
        )

    if not rows:
        raise ValueError(f"{path}: no data rows")

    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_parsers.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/run/parsers.py tests/test_parsers.py tests/fixtures/force.dat tests/fixtures/moment.dat tests/fixtures/force_two_triples.dat
git commit -m "feat: read force.dat by position, because a header is not a contract"
```

---

### Task 3: Centre of pressure and aero balance

**Files:**
- Create: `pipeline/simdev/report/forces.py`
- Test: `tests/test_cop.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `window_mean(frame: pd.DataFrame, column: str, window: tuple[int, int]) -> float`
  - `CentreOfPressure` dataclass with fields `x`, `y`, `z`, `balance_front_pct` (each `float | None`) and `reasons: list[str]`
  - `centre_of_pressure(force, moment, c_of_r, force_scale, axles=None) -> CentreOfPressure` where `force`, `moment` and `c_of_r` are `tuple[float, float, float]`, `force_scale` is `float` (newtons), `axles` is `tuple[float, float] | None` as `(x_front, x_rear)`
  - `MIN_COEFFICIENT: float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cop.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from simdev.report.forces import (
    MIN_COEFFICIENT,
    centre_of_pressure,
    window_mean,
)

# 0.5 * 1.225 * 15^2 * 0.081 -> a plausible reference force for the car.
FORCE_SCALE = 11.16
ORIGIN = (0.0, 0.0, 0.0)


def test_window_mean_uses_only_the_window() -> None:
    frame = pd.DataFrame({"Time": [1, 2, 3, 4], "Cd": [9.0, 9.0, 1.0, 3.0]})
    assert window_mean(frame, "Cd", (3, 4)) == pytest.approx(2.0)


def test_window_mean_of_an_empty_window_is_nan() -> None:
    frame = pd.DataFrame({"Time": [1, 2], "Cd": [1.0, 2.0]})
    assert pd.isna(window_mean(frame, "Cd", (50, 60)))


def test_cop_x_is_where_downforce_acts() -> None:
    """100 N of downforce 0.1 m ahead of the CofR.

    M = r x F = (0.1, 0, 0) x (0, 0, -100) = (0, 10, 0), so COP_x must come
    back at +0.1 m.
    """
    cop = centre_of_pressure((0.0, 0.0, -100.0), (0.0, 10.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.x == pytest.approx(0.1)


def test_cop_y_is_where_downforce_acts_laterally() -> None:
    # Same force 0.05 m to the car's left: M = (0, 0.05, 0) x (0, 0, -100).
    cop = centre_of_pressure((0.0, 0.0, -100.0), (-5.0, 0.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.y == pytest.approx(0.05)


def test_cop_z_is_the_height_drag_acts_at() -> None:
    # 50 N of drag at 0.08 m: M = (0, 0, 0.08) x (50, 0, 0) = (0, 4, 0).
    cop = centre_of_pressure((50.0, 0.0, 0.0), (0.0, 4.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.z == pytest.approx(0.08)


def test_cop_x_and_cop_z_are_not_one_point() -> None:
    """Both read the same pitching moment and they disagree by construction.

    COP_x attributes all of M_y to downforce; COP_z attributes all of it to
    drag. Pinned as a test so nobody later "fixes" the disagreement and
    silently changes what both columns mean.
    """
    cop = centre_of_pressure((50.0, 0.0, -100.0), (0.0, 10.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.x == pytest.approx(0.1)
    assert cop.z == pytest.approx(0.2)


def test_balance_is_100_percent_at_the_front_axle() -> None:
    cop = centre_of_pressure(
        (0.0, 0.0, -100.0), (0.0, 20.0, 0.0), ORIGIN, FORCE_SCALE, axles=(0.2, -0.2)
    )
    assert cop.x == pytest.approx(0.2)
    assert cop.balance_front_pct == pytest.approx(100.0)


def test_balance_is_50_percent_at_the_wheelbase_centre() -> None:
    cop = centre_of_pressure(
        (0.0, 0.0, -100.0), (0.0, 0.0, 0.0), ORIGIN, FORCE_SCALE, axles=(0.2, -0.2)
    )
    assert cop.balance_front_pct == pytest.approx(50.0)


def test_a_vanishing_denominator_gives_no_number_and_a_reason() -> None:
    """Not a large meaningless number. Empty, with the reason recorded."""
    tiny = MIN_COEFFICIENT * FORCE_SCALE * 0.5
    cop = centre_of_pressure((0.0, 0.0, tiny), (0.0, 10.0, 0.0), ORIGIN, FORCE_SCALE)
    assert cop.x is None and cop.y is None and cop.balance_front_pct is None
    assert any("F_z" in reason for reason in cop.reasons)


def test_cop_is_relative_to_the_centre_of_rotation() -> None:
    cop = centre_of_pressure(
        (0.0, 0.0, -100.0), (0.0, 10.0, 0.0), (0.5, 0.0, 0.0), FORCE_SCALE
    )
    assert cop.x == pytest.approx(0.6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simdev.report.forces'`.

- [ ] **Step 3: Implement**

Create `pipeline/simdev/report/forces.py`:

```python
"""Dimensional forces, moments, centre of pressure and per-group coefficients.

Everything here is a window mean over the iterations `check_convergence`
already chose, which is also `fieldAverage`'s timeStart - so a number in
report.tsv and a picture in results/images describe the same flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

# Below this, in coefficient terms, the denominator of a COP ratio is noise
# and the ratio it produces is a large number with no meaning. Report nothing
# and say why.
MIN_COEFFICIENT = 0.02


def window_mean(frame: pd.DataFrame, column: str, window: tuple[int, int]) -> float:
    """Mean of one column over the averaging window. NaN if the window is empty."""
    rows = frame[(frame["Time"] >= window[0]) & (frame["Time"] <= window[1])]
    if rows.empty or column not in rows:
        return float("nan")
    return float(rows[column].mean())


@dataclass(frozen=True)
class CentreOfPressure:
    """Three diagnostics, NOT the coordinates of one point.

    A net force plus a net moment defines a line of action. `x` and `z` are
    two different readings of the same pitching moment - `x` attributes all
    of M_y to downforce, `z` attributes all of it to drag - so they do not
    describe a single position and were never meant to. The record carries
    cop_convention: "ratio" so a later reader cannot mistake them for one.
    """

    x: float | None
    y: float | None
    z: float | None
    balance_front_pct: float | None
    reasons: list[str] = field(default_factory=list)


def centre_of_pressure(
    force: tuple[float, float, float],
    moment: tuple[float, float, float],
    c_of_r: tuple[float, float, float],
    force_scale: float,
    axles: tuple[float, float] | None = None,
) -> CentreOfPressure:
    """Motorsport-convention COP and front aero balance.

    With r the position of the resultant relative to the centre of rotation,
    M = r x F expands to

        M_x = r_y F_z - r_z F_y
        M_y = r_z F_x - r_x F_z
        M_z = r_x F_y - r_y F_x

    and each reported coordinate drops the second term of its own equation:

        COP_x = x_ref - M_y / F_z
        COP_y = y_ref + M_x / F_z
        COP_z = z_ref + M_y / F_x

    M_z is not used and cannot be: it contains no r_z at all, so it
    constrains nothing this convention reports.

    `force_scale` is 0.5 rho U_inf^2 A_ref in newtons - the force a unit
    coefficient would produce - and only sets the guard threshold.
    """
    fx, fy, fz = force
    mx, my, _mz = moment
    x_ref, y_ref, z_ref = c_of_r
    floor = MIN_COEFFICIENT * abs(force_scale)
    reasons: list[str] = []

    if abs(fz) >= floor:
        cop_x: float | None = x_ref - my / fz
        cop_y: float | None = y_ref + mx / fz
    else:
        cop_x = cop_y = None
        reasons.append(
            f"COP_x and COP_y undefined: |F_z| = {abs(fz):.4g} N is under "
            f"{MIN_COEFFICIENT} of the reference force {abs(force_scale):.4g} N"
        )

    if abs(fx) >= floor:
        cop_z: float | None = z_ref + my / fx
    else:
        cop_z = None
        reasons.append(
            f"COP_z undefined: |F_x| = {abs(fx):.4g} N is under "
            f"{MIN_COEFFICIENT} of the reference force {abs(force_scale):.4g} N"
        )

    balance: float | None = None
    if cop_x is not None and axles is not None:
        x_front, x_rear = axles
        wheelbase = x_front - x_rear
        if abs(wheelbase) > 1e-9:
            balance = (cop_x - x_rear) / wheelbase * 100.0
        else:
            reasons.append(
                "aero balance undefined: the front and rear axles share an x"
            )

    return CentreOfPressure(cop_x, cop_y, cop_z, balance, reasons)
```

Also create `pipeline/simdev/report/__init__.py` content if absent — it already exists as a one-line file; leave it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cop.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/report/forces.py tests/test_cop.py
git commit -m "feat: COP as three diagnostics, and a test that they disagree on purpose"
```

---

### Task 4: Force groups — schema, validation, and the sums

**Files:**
- Modify: `pipeline/simdev/config/schema.py:517-520` (`PostConfig`)
- Modify: `pipeline/simdev/config/validate.py` (inside `validate()`, before the final `raise`)
- Modify: `pipeline/simdev/report/forces.py`
- Modify: `cases/car/config.yaml` (the `post:` block)
- Test: `tests/test_groups.py`

**Interfaces:**
- Consumes: `window_mean` from Task 3; `read_component_coeffs` from `run/parsers.py`.
- Produces:
  - `PostConfig.groups: dict[str, tuple[str, ...]]`, default `{}`
  - `group_coefficients(components, groups, window) -> dict[str, dict[str, float]]` keyed group → `{"Cd": float, "Cl": float}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_groups.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.config.validate import ValidationError, validate
from simdev.report.forces import group_coefficients

WINDOW = (2, 3)


def _component(cd: float, cl: float) -> pd.DataFrame:
    # Constant over the window, and deliberately wrong before it, so a test
    # that ignores the window fails rather than coincidentally passing.
    return pd.DataFrame(
        {"Time": [1, 2, 3], "Cd": [99.0, cd, cd], "Cl": [99.0, cl, cl]}
    )


def test_a_group_is_the_sum_of_its_patches() -> None:
    components = {
        "Body": _component(0.30, -0.80),
        "Wing": _component(0.20, -0.60),
        "Chassis": _component(0.10, -0.10),
    }
    groups = {"body": ("Body",), "wing": ("Wing",), "other": ("Chassis",)}
    result = group_coefficients(components, groups, WINDOW)
    assert result["body"]["Cd"] == pytest.approx(0.30)
    assert result["other"]["Cl"] == pytest.approx(-0.10)


def test_the_groups_sum_to_the_vehicle_total() -> None:
    """The identity that makes a pasted row self-checking.

    controlDict gives every forceCoeffs_<patch> the vehicle's own Aref, lRef
    and CofR, so each patch's Cd is its SHARE of the vehicle coefficient.
    Groups are therefore plain sums.
    """
    components = {
        "Body": _component(0.30, -0.80),
        "Wing": _component(0.20, -0.60),
        "Chassis": _component(0.10, -0.10),
        "Tire_FL": _component(0.05, -0.02),
    }
    groups = {"body": ("Body",), "wing": ("Wing",), "other": ("Chassis", "Tire_FL")}
    result = group_coefficients(components, groups, WINDOW)
    total = sum(g["Cd"] for g in result.values())
    assert total == pytest.approx(0.65)


def test_a_missing_patch_makes_its_group_nan_not_zero() -> None:
    """A patch whose function object never ran is unknown, not zero.

    Silently summing what happens to be present would make the group columns
    disagree with the total by an amount nobody can see.
    """
    groups = {"body": ("Body", "Chassis")}
    result = group_coefficients({"Body": _component(0.3, -0.8)}, groups, WINDOW)
    assert pd.isna(result["body"]["Cd"])
```

Then, in `tests/test_validate.py`, add the coverage checks. Reuse whatever
minimal case dict that file already builds; the three tests are:

```python
def test_a_patch_in_no_group_is_an_error() -> None:
    spec = _spec({"post": {"groups": {"body": ["body"]}},
                  "geometry": {"patches": [
                      {"name": "body", "role": "body"},
                      {"name": "wing", "role": "body"},
                      {"name": "ground", "role": "ground"},
                      {"name": "inlet", "role": "inlet"},
                      {"name": "outlet", "role": "outlet"},
                      {"name": "farfield", "role": "farfield"},
                  ]}})
    with pytest.raises(ValidationError, match="is in no post group"):
        validate(spec)


def test_a_patch_in_two_groups_is_an_error() -> None:
    spec = _spec({"post": {"groups": {"a": ["body"], "b": ["body", "wing"]}},
                  "geometry": {"patches": [
                      {"name": "body", "role": "body"},
                      {"name": "wing", "role": "body"},
                      {"name": "ground", "role": "ground"},
                      {"name": "inlet", "role": "inlet"},
                      {"name": "outlet", "role": "outlet"},
                      {"name": "farfield", "role": "farfield"},
                  ]}})
    with pytest.raises(ValidationError, match="more than one post group"):
        validate(spec)


def test_no_groups_at_all_is_fine() -> None:
    """Ahmed declares none and must not be forced to."""
    validate(_spec())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_groups.py tests/test_validate.py -v`
Expected: FAIL — `ImportError: cannot import name 'group_coefficients'`, and the validate tests fail because `post.groups` is not a field.

- [ ] **Step 3: Add the schema field**

In `pipeline/simdev/config/schema.py`, extend `PostConfig`:

```python
class PostConfig(BaseModel):
    yplus_min: float
    yplus_max: float
    max_fraction_outside: float = 0.1
    # Named bundles of force-bearing patches, for the per-group coefficients
    # in results/report.tsv.
    #
    # Empty means "no grouping", which is what the Ahmed case wants: it has
    # one force patch and nothing to attribute. When it is non-empty,
    # validate() requires it to cover every force patch exactly once, so
    # cd_body + cd_wing + cd_other == cd holds and a pasted row checks
    # itself. A partial grouping would produce columns that quietly do not
    # add up, which is worse than no columns.
    groups: dict[str, tuple[str, ...]] = {}
```

- [ ] **Step 4: Add the validation**

In `pipeline/simdev/config/validate.py`, inside `validate()` just before
`if errors:`, add (the module already imports what it needs from
`simdev.geometry.roles`; add `traits` to that import if absent):

```python
    # --- force groups ----------------------------------------------------
    # Exactly once, or not at all. A patch in no group makes the group
    # columns silently disagree with the vehicle total; a patch in two makes
    # them silently exceed it. Both produce a spreadsheet row that looks
    # arithmetically sound and is not, which is the failure this whole
    # report exists to avoid.
    if spec.post.groups:
        force_patches = [
            p.name for p in spec.geometry.patches if traits(p.role).in_forces
        ]
        membership: dict[str, list[str]] = {}
        for group, patches in spec.post.groups.items():
            for patch in patches:
                membership.setdefault(patch, []).append(group)

        for patch, groups in sorted(membership.items()):
            if patch not in force_patches:
                errors.append(
                    f"post.groups.{groups[0]} names {patch!r}, which is not a "
                    "force-bearing patch"
                )
            elif len(groups) > 1:
                errors.append(
                    f"patch {patch!r} is in more than one post group "
                    f"({', '.join(groups)}): its coefficient would be counted "
                    "twice and the groups would exceed the vehicle total"
                )
        for patch in force_patches:
            if patch not in membership:
                errors.append(
                    f"force patch {patch!r} is in no post group: the group "
                    "coefficients would not sum to the vehicle total"
                )
```

- [ ] **Step 5: Implement the sums**

Append to `pipeline/simdev/report/forces.py`:

```python
COEFFICIENTS = ("Cd", "Cl")


def group_coefficients(
    components: Mapping[str, pd.DataFrame],
    groups: Mapping[str, Sequence[str]],
    window: tuple[int, int],
) -> dict[str, dict[str, float]]:
    """Window-mean Cd and Cl per group, summed from the per-patch objects.

    controlDict gives every forceCoeffs_<patch> the vehicle's own Aref, lRef
    and CofR, so each patch's Cd and Cl is its SHARE of the vehicle
    coefficient and groups are plain sums. That is what makes
    cd_body + cd_wing + cd_other == cd hold to floating point, and it is why
    per-patch reference areas would have been useless here: eleven numbers on
    eleven scales that add up to nothing.

    A patch with no data makes its whole group NaN rather than summing what
    happens to be present. A group that silently omits a member disagrees
    with the total by an amount nobody can see.
    """
    out: dict[str, dict[str, float]] = {}

    for group, patches in groups.items():
        totals = {c: 0.0 for c in COEFFICIENTS}
        for patch in patches:
            frame = components.get(patch)
            if frame is None:
                totals = {c: float("nan") for c in COEFFICIENTS}
                break
            for coefficient in COEFFICIENTS:
                totals[coefficient] += window_mean(frame, coefficient, window)
        out[group] = totals

    return out
```

- [ ] **Step 6: Declare the car's groups**

In `cases/car/config.yaml`, in the `post:` block:

```yaml
  # WHAT THE REPORT ATTRIBUTES FORCES TO, AND WHY IT IS ONLY THREE BUCKETS.
  #
  # Body and Wing are the surfaces under development; everything else on this
  # car is an aero dummy that does not represent the real vehicle, so its
  # coefficient is a bookkeeping remainder rather than something to design
  # against. Splitting `other` further is a config change and costs nothing,
  # but eleven columns nobody reads is how a report stops being read.
  #
  # Every force-bearing patch appears exactly once - validate() requires it -
  # so cd_body + cd_wing + cd_other == cd, and a row pasted into a sheet
  # checks its own arithmetic.
  groups:
    body:  [Body]
    wing:  [Wing]
    other: [Chassis,
            SUS_FL, SUS_FR, SUS_RL, SUS_RR,
            Tire_FL, Tire_FR, Tire_RL, Tire_RR]
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_groups.py tests/test_validate.py tests/test_schema.py -v`
Expected: PASS.

- [ ] **Step 8: Check the real car case still validates**

Run: `.venv/bin/simdev prepare cases/car/config.yaml --run-dir /tmp/car-groups-check --profile car_smoke`
Expected: completes without a group error. If it reports a patch in no group, the group lists and the patch list have drifted — fix the config, not the check.

- [ ] **Step 9: Commit**

```bash
git add pipeline/simdev/config/schema.py pipeline/simdev/config/validate.py pipeline/simdev/report/forces.py cases/car/config.yaml tests/test_groups.py tests/test_validate.py
git commit -m "feat: force groups that must cover the car exactly once"
```

---

### Task 5: Assemble the force report for a run

**Files:**
- Modify: `pipeline/simdev/report/forces.py`
- Test: `tests/test_force_report.py`

**Interfaces:**
- Consumes: `read_force_vectors` (Task 2), `centre_of_pressure` (Task 3), `group_coefficients` (Task 4), `read_component_coeffs` and `find_latest` from the existing modules, `read_status` from `run/status.py`.
- Produces:
  - `ForceReport` dataclass: `force: tuple[float, float, float] | None`, `moment: tuple[float, float, float] | None`, `cop: CentreOfPressure | None`, `groups: dict[str, dict[str, float]]`, `reasons: list[str]`
  - `axles_from_prepare(run_dir: Path) -> tuple[float, float] | None`
  - `build_force_report(run_dir: Path, spec: CaseSpec, window: tuple[int, int]) -> ForceReport`

- [ ] **Step 1: Write the failing test**

Create `tests/test_force_report.py`:

```python
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from simdev.report.forces import axles_from_prepare, build_force_report
from simdev.run.status import StageStatus, write_status
from simdev.stages.common import load_spec
from simdev.stages.prepare import prepare

FIXTURES = Path(__file__).parent / "fixtures"

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
        "symmetric": True,
        "ahmed": {"include_stilts": False},
        "patches": [
            {"name": "body", "role": "body"},
            {"name": "ground", "role": "ground"},
            {"name": "symmetry", "role": "symmetry"},
            {"name": "inlet", "role": "inlet"},
            {"name": "outlet", "role": "outlet"},
            {"name": "farfield", "role": "farfield"},
        ],
    },
}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    case = tmp_path / "config.yaml"
    case.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    target = tmp_path / "run"
    prepare(case, target, profile="dev")
    for name, fixture in (("forces", "force.dat"), ("forces", "moment.dat")):
        out = target / "postProcessing" / name / "0"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIXTURES / fixture, out / fixture)
    return target


def test_the_report_carries_dimensional_forces(run_dir: Path) -> None:
    report = build_force_report(run_dir, load_spec(run_dir), (2, 3))
    # force.dat totals average 2.5, 0.5, -25.0 over Time 2..3.
    assert report.force is not None
    assert report.force[2] == pytest.approx(-25.0)
    assert report.moment is not None
    assert report.moment[1] == pytest.approx(2.5)


def test_a_run_without_a_forces_object_still_reports(run_dir: Path) -> None:
    """A run made before the forces object existed is not a failed run.

    It has coefficients, a y+ verdict and a flow field. It just has no
    newtons, and the report says so rather than refusing.
    """
    shutil.rmtree(run_dir / "postProcessing" / "forces")
    report = build_force_report(run_dir, load_spec(run_dir), (2, 3))
    assert report.force is None
    assert report.cop is None
    assert any("forces" in reason for reason in report.reasons)


def test_axles_come_from_what_prepare_measured(tmp_path: Path) -> None:
    write_status(
        tmp_path,
        StageStatus(
            "prepare", "ok", "h", [],
            {"wheels": {
                "FL": {"origin": [0.30, 0.1, 0.03]},
                "FR": {"origin": [0.30, -0.1, 0.03]},
                "RL": {"origin": [-0.10, 0.1, 0.03]},
                "RR": {"origin": [-0.10, -0.1, 0.03]},
            }},
        ),
    )
    assert axles_from_prepare(tmp_path) == pytest.approx((0.30, -0.10))


def test_no_wheels_means_no_balance(tmp_path: Path) -> None:
    """The Ahmed body has no axles and must not get an invented wheelbase."""
    write_status(tmp_path, StageStatus("prepare", "ok", "h", [], {"wheels": {}}))
    assert axles_from_prepare(tmp_path) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_force_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_force_report'`.

- [ ] **Step 3: Implement**

Append to `pipeline/simdev/report/forces.py`:

```python
@dataclass(frozen=True)
class ForceReport:
    """Everything report.tsv needs beyond what post already computes.

    `force`, `moment` and `cop` are None together: they all come from the
    same two files, and a run made before the forces object existed has
    neither. That is a gap in the record, not a failed run.
    """

    force: tuple[float, float, float] | None
    moment: tuple[float, float, float] | None
    cop: CentreOfPressure | None
    groups: dict[str, dict[str, float]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def axles_from_prepare(run_dir: Path) -> tuple[float, float] | None:
    """Mean front and rear wheel-centre x, as prepare measured them.

    Steering rotates a wheel about its own steering axis and moves the centre
    by millimetres in x, so this is stable enough to divide by - which is why
    the balance denominator comes from here rather than from the geometry
    bounding box, which moves with the bodywork.

    None when the case has no wheels. The Ahmed body must not be given an
    invented wheelbase so that a balance column can be filled in.
    """
    from simdev.run.status import read_status

    status = read_status(run_dir, "prepare")
    if status is None:
        return None
    wheels = status.detail.get("wheels") or {}
    front = [w["origin"][0] for name, w in wheels.items() if name.startswith("F")]
    rear = [w["origin"][0] for name, w in wheels.items() if name.startswith("R")]
    if not front or not rear:
        return None
    return (sum(front) / len(front), sum(rear) / len(rear))


def build_force_report(
    run_dir: Path, spec: "CaseSpec", window: tuple[int, int]
) -> ForceReport:
    """Window means of F and M, the COP triple, and the group coefficients."""
    from simdev.run.parsers import read_component_coeffs, read_force_vectors
    from simdev.stages.common import find_latest

    reasons: list[str] = []

    groups = group_coefficients(
        read_component_coeffs(run_dir), spec.post.groups, window
    )

    try:
        force_frame = read_force_vectors(
            find_latest(run_dir, "forces/*/force.dat")
        )
        moment_frame = read_force_vectors(
            find_latest(run_dir, "forces/*/moment.dat")
        )
    except (FileNotFoundError, ValueError) as error:
        reasons.append(
            "no dimensional forces: could not read the 'forces' function "
            f"object ({error}). Runs made before it was added to controlDict "
            "have coefficients but no newtons"
        )
        return ForceReport(None, None, None, groups, reasons)

    force = tuple(window_mean(force_frame, axis, window) for axis in "xyz")
    moment = tuple(window_mean(moment_frame, axis, window) for axis in "xyz")

    # The force a unit coefficient would produce. Only sets the COP guard.
    force_scale = (
        0.5 * spec.flow.rho * spec.flow.u_inf**2 * spec.a_ref_effective
    )
    cop = centre_of_pressure(
        force, moment, tuple(spec.forces.c_of_r), force_scale,
        axles=axles_from_prepare(run_dir),
    )
    reasons.extend(cop.reasons)

    return ForceReport(force, moment, cop, groups, reasons)
```

Add `from simdev.config.schema import CaseSpec` under a `TYPE_CHECKING` guard at the top of the file, so the annotation resolves without a runtime import cycle.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_force_report.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/report/forces.py tests/test_force_report.py
git commit -m "feat: assemble the force report, and survive a run that predates it"
```

---

### Task 6: `report.tsv` and the aggregate

**Files:**
- Create: `pipeline/simdev/report/tsv.py`
- Test: `tests/test_report_tsv.py`

**Interfaces:**
- Consumes: nothing at import time; `write_report` is handed a finished dict.
- Produces:
  - `FIXED_COLUMNS: tuple[str, ...]` — the stable prefix
  - `report_columns(group_names: Sequence[str]) -> list[str]`
  - `write_report(run_dir: Path, row: Mapping[str, object], group_names: Sequence[str]) -> Path`
  - `read_report(run_dir: Path) -> dict[str, str]`
  - `aggregate_reports(run_dirs: Iterable[Path]) -> tuple[list[str], list[list[str]]]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_tsv.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from simdev.report.tsv import (
    FIXED_COLUMNS,
    aggregate_reports,
    read_report,
    report_columns,
    write_report,
)

ROW: dict[str, object] = {
    "run": "car-01",
    "case_name": "car",
    "driving_state": "testcase",
    "spec_hash": "abc123",
    "timestamp": "2026-09-01T10:00:00+00:00",
    "verdict": "converged",
    "converged": True,
    "window_start": 300,
    "window_end": 400,
    "n_iterations": 400,
    "cd_amplitude": 0.022,
    "cl_amplitude": 0.079,
    "yplus_passed": True,
    "n_cells": 20259975,
    "Fx": 12.34,
    "Fy": -0.56,
    "Fz": -78.9,
    "Mx": 0.12,
    "My": 3.45,
    "Mz": -0.67,
    "cd": 0.99,
    "cd_std": 0.02,
    "cl": -1.62,
    "cl_std": 0.13,
    "cs": -0.04,
    "COP_x": 0.043,
    "COP_y": 0.001,
    "COP_z": 0.279,
    "balance_front_pct": 46.2,
    "cd_body": 0.4,
    "cl_body": -0.9,
    "cd_wing": 0.3,
    "cl_wing": -0.5,
    "cd_other": 0.29,
    "cl_other": -0.22,
}


def test_the_column_order_is_a_contract() -> None:
    """Pinned deliberately. A sheet with pasted rows breaks if columns move.

    New columns are APPENDED, never inserted. If this test fails because you
    inserted one, move it to the end instead of updating the expectation.
    """
    assert FIXED_COLUMNS[:5] == (
        "run", "case_name", "driving_state", "spec_hash", "timestamp",
    )
    assert FIXED_COLUMNS[5:9] == (
        "verdict", "converged", "window_start", "window_end",
    )
    assert FIXED_COLUMNS[-4:] == ("COP_x", "COP_y", "COP_z", "balance_front_pct")


def test_group_columns_follow_the_configured_order() -> None:
    columns = report_columns(["body", "wing", "other"])
    assert columns[len(FIXED_COLUMNS):] == [
        "cd_body", "cl_body", "cd_wing", "cl_wing", "cd_other", "cl_other",
    ]


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    write_report(tmp_path, ROW, ["body", "wing", "other"])
    text = (tmp_path / "results" / "report.tsv").read_text(encoding="utf-8")
    header, data = text.strip().splitlines()
    assert header.split("\t")[0] == "run"
    assert len(header.split("\t")) == len(data.split("\t"))
    assert read_report(tmp_path)["balance_front_pct"] == "46.2"


def test_an_absent_value_is_an_empty_field_not_nan(tmp_path: Path) -> None:
    """"nan" in a spreadsheet cell is a value. An empty cell is a gap.

    A COP that could not be computed must not arrive as text that Excel will
    happily average.
    """
    row = dict(ROW, COP_x=None, COP_z=float("nan"))
    write_report(tmp_path, row, ["body", "wing", "other"])
    fields = read_report(tmp_path)
    assert fields["COP_x"] == ""
    assert fields["COP_z"] == ""


def test_aggregate_puts_one_row_per_run(tmp_path: Path) -> None:
    for name in ("car-01", "car-02"):
        run = tmp_path / name
        write_report(run, dict(ROW, run=name), ["body", "wing", "other"])
    header, rows = aggregate_reports([tmp_path / "car-01", tmp_path / "car-02"])
    assert header[0] == "run"
    assert [r[0] for r in rows] == ["car-01", "car-02"]


def test_aggregate_skips_a_run_with_no_report(tmp_path: Path) -> None:
    write_report(tmp_path / "car-01", ROW, ["body", "wing", "other"])
    header, rows = aggregate_reports([tmp_path / "car-01", tmp_path / "nothing"])
    assert len(rows) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_report_tsv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simdev.report.tsv'`.

- [ ] **Step 3: Implement**

Create `pipeline/simdev/report/tsv.py`:

```python
"""The paste target: one tab-separated line per run, and a way to stack them.

Tab-separated and one line per run because that is the shape the old
StarCCM pipeline's Auswertung_<version>.txt had, and there are sheets built
around it. Written per run and combined ON READ - never appended to a shared
file. See report/results.py for why that rule exists.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# THE COLUMN ORDER IS A CONTRACT. Someone has a sheet with formulas pointing
# at column N. New columns go on the END; nothing is ever inserted or
# reordered. tests/test_report_tsv.py pins it.
#
# The `verdict`..`n_cells` block is not decoration. A row that does not say
# which iterations it averaged and whether the run converged will eventually
# be compared against one that stopped mid-transient, and the difference will
# be believed. It travels with the numbers or it does not exist.
FIXED_COLUMNS: tuple[str, ...] = (
    # identity
    "run", "case_name", "driving_state", "spec_hash", "timestamp",
    # whether to trust the rest of the line
    "verdict", "converged", "window_start", "window_end", "n_iterations",
    "cd_amplitude", "cl_amplitude", "yplus_passed", "n_cells",
    # forces, newtons
    "Fx", "Fy", "Fz",
    # moments about c_of_r, newton-metres
    "Mx", "My", "Mz",
    # coefficients
    "cd", "cd_std", "cl", "cl_std", "cs",
    # centre of pressure, metres, and balance in percent.
    # THREE DIAGNOSTICS, NOT A POINT - see report/forces.py.
    "COP_x", "COP_y", "COP_z", "balance_front_pct",
)


def report_columns(group_names: Sequence[str]) -> list[str]:
    """The fixed prefix, then cd/cl per group in the order the config declares."""
    return [
        *FIXED_COLUMNS,
        *(f"{c}_{g}" for g in group_names for c in ("cd", "cl")),
    ]


def _format(value: object) -> str:
    """Empty for anything absent. NEVER the string "nan".

    A cell containing "nan" is a value a spreadsheet will happily average
    into a summary. An empty cell is a gap, which is what an uncomputable COP
    actually is.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_report(
    run_dir: Path, row: Mapping[str, object], group_names: Sequence[str]
) -> Path:
    """One header line and one data line. Overwrites; never appends."""
    columns = report_columns(group_names)
    target = Path(run_dir) / "results"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "report.tsv"
    path.write_text(
        "\t".join(columns)
        + "\n"
        + "\t".join(_format(row.get(c)) for c in columns)
        + "\n",
        encoding="utf-8",
    )
    return path


def read_report(run_dir: Path) -> dict[str, str]:
    """The one data line of a run's report, as a column -> text mapping."""
    path = Path(run_dir) / "results" / "report.tsv"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: expected a header and one data line")
    return dict(zip(lines[0].split("\t"), lines[1].split("\t")))


def aggregate_reports(
    run_dirs: Iterable[Path],
) -> tuple[list[str], list[list[str]]]:
    """Stack per-run reports on read. Returns (header, rows).

    Columns are unioned in first-seen order, so a run with an extra group
    widens the table rather than being silently truncated to match the first
    run. A run with no report.tsv is skipped - it has not been post-processed
    - and the caller reports which.
    """
    header: list[str] = []
    found: list[dict[str, str]] = []

    for run_dir in run_dirs:
        try:
            fields = read_report(run_dir)
        except (FileNotFoundError, ValueError):
            continue
        for column in fields:
            if column not in header:
                header.append(column)
        found.append(fields)

    return header, [[fields.get(c, "") for c in header] for fields in found]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_report_tsv.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/report/tsv.py tests/test_report_tsv.py
git commit -m "feat: report.tsv, with the trust block that makes a row readable"
```

---

### Task 7: Wire the report into `post`

**Files:**
- Modify: `pipeline/simdev/report/results.py` (`ResultRecord`)
- Modify: `pipeline/simdev/stages/post.py`
- Test: `tests/test_post.py`

**Interfaces:**
- Consumes: `build_force_report` (Task 5), `write_report`/`report_columns` (Task 6).
- Produces: `results/report.tsv` on every successful `post`; `ResultRecord` gains `fx`, `fy`, `fz`, `mx`, `my`, `mz`, `cs_mean`, `cop_x`, `cop_y`, `cop_z`, `balance_front_pct` (all `float | None`, default `None`), `cop_convention: str = "ratio"`, and `groups: dict[str, dict[str, float]]` (default empty).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_post.py`. The existing `run_dir` fixture already stages
`coefficient.dat` and `yPlus.dat`; extend it to also stage the two forces
fixtures under `postProcessing/forces/0/`:

```python
def test_post_writes_the_paste_target(run_dir: Path) -> None:
    _mark_solve(run_dir, "ok")
    post(run_dir)
    text = (run_dir / "results" / "report.tsv").read_text(encoding="utf-8")
    header, data = text.strip().splitlines()
    assert "balance_front_pct" in header.split("\t")
    assert len(header.split("\t")) == len(data.split("\t"))


def test_post_records_the_cop_convention(run_dir: Path) -> None:
    """So a later reader cannot mistake the triple for one point."""
    _mark_solve(run_dir, "ok")
    assert post(run_dir).cop_convention == "ratio"


def test_post_survives_a_run_with_no_forces_object(run_dir: Path) -> None:
    import shutil as _shutil

    _shutil.rmtree(run_dir / "postProcessing" / "forces")
    _mark_solve(run_dir, "ok")
    record = post(run_dir)
    assert record.fz is None
    assert record.cd_mean != 0.0
    assert (run_dir / "results" / "report.tsv").exists()


def test_an_old_result_json_still_loads(run_dir: Path, tmp_path: Path) -> None:
    """New fields are defaulted, so a record written before them still reads."""
    import json

    from simdev.report.results import read_result

    _mark_solve(run_dir, "ok")
    post(run_dir)
    path = run_dir / "results" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("fx", "fy", "fz", "cop_x", "cop_convention", "groups"):
        payload.pop(key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_result(run_dir).fz is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_post.py -v`
Expected: FAIL — no `report.tsv`, and `ResultRecord` has no `cop_convention`.

- [ ] **Step 3: Extend `ResultRecord`**

In `pipeline/simdev/report/results.py`, append to the dataclass — after
`yplus` and before `reasons`, all defaulted so old files still load:

```python
    # Dimensional forces and moments, newtons and newton-metres, window means.
    # None on a run made before the `forces` function object existed.
    fx: float | None = None
    fy: float | None = None
    fz: float | None = None
    mx: float | None = None
    my: float | None = None
    mz: float | None = None
    cs_mean: float | None = None
    # THREE DIAGNOSTICS, NOT ONE POINT. cop_x and cop_z are two readings of
    # the same pitching moment and disagree by construction; the convention
    # is named here so no reader has to guess which one it is holding.
    cop_x: float | None = None
    cop_y: float | None = None
    cop_z: float | None = None
    balance_front_pct: float | None = None
    cop_convention: str = "ratio"
    # group -> {"Cd": ..., "Cl": ...}. Sums to the vehicle total by
    # construction; see config validation.
    groups: dict[str, dict[str, float]] = field(default_factory=dict)
```

- [ ] **Step 4: Wire it into `post`**

In `pipeline/simdev/stages/post.py`, after `components = read_component_coeffs(run_dir)`
and the component plots, before building the record:

```python
    # Dimensional forces, the COP triple and the per-group coefficients.
    # Never fatal: a run made before the `forces` object existed still has
    # coefficients, a y+ verdict and a flow field worth keeping.
    force_report = build_force_report(run_dir, spec, convergence.window)
```

Extend the `ResultRecord(...)` construction with:

```python
        fx=force_report.force[0] if force_report.force else None,
        fy=force_report.force[1] if force_report.force else None,
        fz=force_report.force[2] if force_report.force else None,
        mx=force_report.moment[0] if force_report.moment else None,
        my=force_report.moment[1] if force_report.moment else None,
        mz=force_report.moment[2] if force_report.moment else None,
        cs_mean=window_mean(forces, "Cs", convergence.window),
        cop_x=force_report.cop.x if force_report.cop else None,
        cop_y=force_report.cop.y if force_report.cop else None,
        cop_z=force_report.cop.z if force_report.cop else None,
        balance_front_pct=(
            force_report.cop.balance_front_pct if force_report.cop else None
        ),
        groups=force_report.groups,
        reasons=[*convergence.reasons, *y_plus_gate.reasons, *force_report.reasons],
```

And after `write_result(run_dir, record)`:

```python
    # The paste target. Everything above is for machines; this line is the
    # one a person copies into a sheet, so it carries its own trustworthiness
    # alongside the numbers.
    group_names = list(spec.post.groups)
    write_report(
        run_dir,
        {
            "run": run_dir.name,
            "case_name": record.case_name,
            "driving_state": spec.driving_state,
            "spec_hash": record.spec_hash,
            "timestamp": record.timestamp,
            "verdict": record.verdict,
            "converged": record.converged,
            "window_start": record.window_start,
            "window_end": record.window_end,
            "n_iterations": record.n_iterations,
            "cd_amplitude": convergence.amplitudes.get("Cd"),
            "cl_amplitude": convergence.amplitudes.get("Cl"),
            "yplus_passed": record.yplus_passed,
            "n_cells": record.n_cells,
            "Fx": record.fx, "Fy": record.fy, "Fz": record.fz,
            "Mx": record.mx, "My": record.my, "Mz": record.mz,
            "cd": record.cd_mean, "cd_std": record.cd_std,
            "cl": record.cl_mean, "cl_std": record.cl_std,
            "cs": record.cs_mean,
            "COP_x": record.cop_x, "COP_y": record.cop_y, "COP_z": record.cop_z,
            "balance_front_pct": record.balance_front_pct,
            **{
                f"{c.lower()}_{g}": values[c]
                for g, values in record.groups.items()
                for c in ("Cd", "Cl")
            },
        },
        group_names,
    )
```

Add the imports at the top of `post.py`:

```python
from simdev.report.forces import build_force_report, window_mean
from simdev.report.tsv import write_report
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_post.py tests/test_results.py -v`
Expected: PASS.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: PASS. `post` is called from `test_cli.py` and `test_smoke.py` too; if either now fails on a missing `forces` directory, that is the Step-3-of-Task-5 fallback not firing — fix the fallback, do not stage fixtures into those tests.

- [ ] **Step 7: Commit**

```bash
git add pipeline/simdev/report/results.py pipeline/simdev/stages/post.py tests/test_post.py
git commit -m "feat: post writes the paste target"
```

---

### Task 8: `simdev report`

**Files:**
- Modify: `pipeline/simdev/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `aggregate_reports` (Task 6).
- Produces: the `report` subcommand — `simdev report <run-dirs...> [--out PATH]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_report_stacks_runs_on_read(tmp_path: Path, capsys) -> None:
    from simdev.report.tsv import write_report

    for name in ("car-01", "car-02"):
        write_report(tmp_path / name, {"run": name, "cd": 0.99}, ["body"])
    code = main([
        "report", str(tmp_path / "car-01"), str(tmp_path / "car-02"),
        "--out", str(tmp_path / "summary.tsv"),
    ])
    assert code == 0
    text = (tmp_path / "summary.tsv").read_text(encoding="utf-8")
    assert len(text.strip().splitlines()) == 3  # header + two runs


def test_report_says_which_runs_it_skipped(tmp_path: Path, capsys) -> None:
    """Silently dropping a run from a comparison table is how one goes missing."""
    from simdev.report.tsv import write_report

    write_report(tmp_path / "car-01", {"run": "car-01"}, [])
    main(["report", str(tmp_path / "car-01"), str(tmp_path / "car-99")])
    assert "car-99" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -k report -v`
Expected: FAIL — `SystemExit: 2`, argparse rejects the unknown command `report`.

- [ ] **Step 3: Implement**

In `_parser()`, beside the existing `aggregate` parser:

```python
    rep = subparsers.add_parser(
        "report",
        help=(
            "stack per-run results/report.tsv files into one table. Combines "
            "on read; never appends to a shared file"
        ),
    )
    rep.add_argument("run_dirs", type=Path, nargs="+")
    rep.add_argument("--out", type=Path, default=None)
```

In `main()`, beside the `aggregate` branch:

```python
        if args.command == "report":
            header, rows = aggregate_reports(args.run_dirs)
            missing = [
                str(d) for d in args.run_dirs
                if not (Path(d) / "results" / "report.tsv").exists()
            ]
            if missing:
                # Loudly. A run quietly absent from a comparison table is how
                # a conclusion gets drawn from half the evidence.
                print(
                    "warning: no results/report.tsv, skipped: "
                    + ", ".join(missing),
                    file=sys.stderr,
                )
            text = "\n".join("\t".join(r) for r in [header, *rows]) + "\n"
            if args.out:
                args.out.write_text(text, encoding="utf-8")
            print(text, end="")
            return 0
```

Add the import: `from simdev.report.tsv import aggregate_reports`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/cli.py tests/test_cli.py
git commit -m "feat: simdev report, and it names the runs it could not read"
```

---

### Task 9: The balance history plot

**Files:**
- Modify: `pipeline/simdev/report/forces.py` (`cop_history`)
- Modify: `pipeline/simdev/report/plots.py` (`plot_balance`)
- Modify: `pipeline/simdev/stages/post.py`
- Test: `tests/test_component_plots.py`

**Interfaces:**
- Consumes: `centre_of_pressure` (Task 3).
- Produces:
  - `cop_history(forces, moments, c_of_r, force_scale, axles) -> pd.DataFrame` with columns `Time`, `COP_x`, `balance_front_pct`
  - `plot_balance(history: pd.DataFrame, out_path: Path, window: tuple[int, int]) -> Path`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_component_plots.py`:

```python
def test_cop_history_is_per_iteration() -> None:
    import pandas as pd

    from simdev.report.forces import cop_history

    forces = pd.DataFrame({"Time": [1, 2], "x": [50.0, 50.0], "y": [0.0, 0.0], "z": [-100.0, -200.0]})
    moments = pd.DataFrame({"Time": [1, 2], "x": [0.0, 0.0], "y": [10.0, 10.0], "z": [0.0, 0.0]})
    history = cop_history(forces, moments, (0.0, 0.0, 0.0), 11.16, (0.2, -0.2))
    assert list(history["Time"]) == [1, 2]
    assert history["COP_x"].iloc[0] == pytest.approx(0.1)
    assert history["COP_x"].iloc[1] == pytest.approx(0.05)
    assert history["balance_front_pct"].iloc[0] == pytest.approx(75.0)


def test_plot_balance_writes_a_file(tmp_path: Path) -> None:
    import pandas as pd

    from simdev.report.plots import plot_balance

    history = pd.DataFrame({
        "Time": range(1, 21),
        "COP_x": [0.04] * 20,
        "balance_front_pct": [46.0 + (i % 3) for i in range(20)],
    })
    out = plot_balance(history, tmp_path / "balance.png", (15, 20))
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_component_plots.py -k "cop_history or plot_balance" -v`
Expected: FAIL — `ImportError: cannot import name 'cop_history'`.

- [ ] **Step 3: Implement `cop_history`**

Append to `pipeline/simdev/report/forces.py`:

```python
def cop_history(
    forces: pd.DataFrame,
    moments: pd.DataFrame,
    c_of_r: tuple[float, float, float],
    force_scale: float,
    axles: tuple[float, float] | None,
) -> pd.DataFrame:
    """COP_x and front balance per iteration, for the plot.

    Balance is the number that actually gets tuned, and whether it swings
    half a percent or five across the limit cycle is invisible in a windowed
    mean. Guarded iterations come back NaN so the trace breaks rather than
    spiking - matplotlib draws a gap, which is the honest picture of an
    iteration where the denominator was noise.
    """
    merged = forces.merge(moments, on="Time", suffixes=("_f", "_m"))
    rows = []
    for _, row in merged.iterrows():
        cop = centre_of_pressure(
            (row["x_f"], row["y_f"], row["z_f"]),
            (row["x_m"], row["y_m"], row["z_m"]),
            c_of_r,
            force_scale,
            axles=axles,
        )
        rows.append(
            {
                "Time": row["Time"],
                "COP_x": float("nan") if cop.x is None else cop.x,
                "balance_front_pct": (
                    float("nan")
                    if cop.balance_front_pct is None
                    else cop.balance_front_pct
                ),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Implement `plot_balance`**

Append to `pipeline/simdev/report/plots.py`:

```python
def plot_balance(
    history: pd.DataFrame, out_path: Path, window: tuple[int, int]
) -> Path:
    """COP_x and front aero balance against iteration.

    Two panels sharing an x-axis, with the averaging window shaded, on the
    same reasoning as plot_force_history: the transient is drawn but the axis
    is scaled from the settled region, or the only part anyone reads is a
    flat line one pixel thick.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    start = settled_from(history, window)
    settled = history[history["Time"] >= start]

    for axis, column, label in (
        (axes[0], "COP_x", "COP$_x$ [m]"),
        (axes[1], "balance_front_pct", "front balance [%]"),
    ):
        axis.plot(history["Time"], history[column], linewidth=0.8)
        axis.axvspan(window[0], window[1], alpha=0.12, label="averaging window")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        values = settled[column].dropna()
        if not values.empty:
            pad = max(float(values.std()) * 4.0, 1e-6)
            axis.set_ylim(float(values.mean()) - pad, float(values.mean()) + pad)

    axes[0].legend(loc="upper right", fontsize="small")
    axes[1].set_xlabel("iteration")
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    return out_path
```

- [ ] **Step 5: Call it from `post`**

In `pipeline/simdev/stages/post.py`, after `plot_force_history(...)`, guarded
so a run without the forces object still post-processes:

```python
    if force_report.force is not None:
        plot_balance(
            cop_history(
                read_force_vectors(find_latest(run_dir, "forces/*/force.dat")),
                read_force_vectors(find_latest(run_dir, "forces/*/moment.dat")),
                tuple(spec.forces.c_of_r),
                0.5 * spec.flow.rho * spec.flow.u_inf**2 * spec.a_ref_effective,
                axles_from_prepare(run_dir),
            ),
            results_dir / "balance.png",
            convergence.window,
        )
```

Extend the imports at the top of `post.py`:

```python
from simdev.report.forces import (
    axles_from_prepare, build_force_report, cop_history, window_mean,
)
from simdev.report.plots import plot_balance
from simdev.run.parsers import read_force_vectors
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_component_plots.py tests/test_post.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/simdev/report/forces.py pipeline/simdev/report/plots.py pipeline/simdev/stages/post.py tests/test_component_plots.py
git commit -m "feat: plot the balance, because a windowed mean hides how far it swings"
```

---

**Phase one is complete and shippable here.** `simdev post` writes
`results/report.tsv`, `simdev report ~/runs/car-*` stacks them, and the
spreadsheet work is unblocked. Phase two can follow at any time.

---

# Phase Two — the images

---

### Task 10: The shared view definition

**Files:**
- Create: `pipeline/simdev/viz/__init__.py`, `pipeline/simdev/viz/views.py`
- Create: `cases/post_views.yaml`
- Test: `tests/test_views.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Axis` dataclass with `start: float`, `stop: float`, `step: float` and `offsets() -> list[float]`
  - `FieldStyle` dataclass with `limits: tuple[float, float]`, `colormap: str`
  - `Views` dataclass with `datum_patches: tuple[str, ...]`, `axes: dict[str, Axis]`, `fields: dict[str, FieldStyle]`, `parallel_scale: dict[str, float]`, `focus_height: float`, `resolution: tuple[int, int]`, `streamlines: str`, `digest: str`
  - `load_views(path: Path) -> Views`
  - `DEFAULT_VIEWS_PATH: Path` — `cases/post_views.yaml` relative to the repo root
  - `SLICE_FIELDS: tuple[str, ...] = ("cp", "cpt", "U", "vort", "lambda2")`
  - `SURFACE_FIELDS: tuple[str, ...] = ("cp", "yplus")`

- [ ] **Step 1: Write `cases/post_views.yaml`**

```yaml
# HOW EVERY PICTURE IS FRAMED, FOR EVERY RUN.
#
# This file is the "somewhere else" the slice positions live in. It is shared
# by every run and versioned in git, and each run keeps a copy in
# results/views.yaml with its digest in results/images.json - so any picture
# can be traced to the definition that produced it.
#
# EDITING IT BREAKS COMPARABILITY WITH EVERY PICTURE ALREADY MADE. That is
# the point of it being one file rather than a per-case block: the break is
# visible, deliberate, and recorded in the history.
#
# Offsets are metres from the datum (see below), NOT absolute mesh
# coordinates. The car spans roughly x -0.215..0.242, y -0.131..0.131,
# z 0.000..0.130 about its datum, so every range below starts and ends
# outside it.

datum:
  # Measured from the Chassis bounding box: centre in x and y, ground in z.
  #
  # Chassis and not the wheels, because steering angle moves a wheel. Chassis
  # and not Body or Wing, because those are the surfaces under development
  # and an anchor that moves when the thing being measured is redesigned
  # anchors nothing. docs/handbook.md calls Chassis an aero dummy nobody
  # iterates, which is exactly the property wanted here.
  patches: [Chassis]

planes:
  # Fixed. NOT fitted to the geometry. A range that quietly extends to cover
  # a longer wing produces a different picture wearing the same file name;
  # the images stage warns when the car pokes outside instead.
  x: {from: -0.30, to: 0.32, step: 0.02}   # 32 planes
  y: {from: -0.20, to: 0.20, step: 0.02}   # 21 planes
  z: {from:  0.00, to: 0.16, step: 0.01}   # 17 planes

fields:
  # Fixed limits, on the same reasoning as report/plots.py's FORCE_AXIS_LIMITS:
  # a scale fitted per run makes a 5% change look identical to a 0.5% one.
  # Values outside clamp to the end colour and the clamped fraction is
  # recorded, so a badly chosen limit is visible rather than merely invisible.
  cp:      {limits: [-3.0, 1.0],       colormap: coolwarm}
  cpt:     {limits: [-3.0, 1.0],       colormap: coolwarm}
  U:       {limits: [0.0, 22.5],       colormap: viridis}    # 1.5 x u_inf
  vort:    {limits: [0.0, 2000.0],     colormap: inferno}
  lambda2: {limits: [-50000.0, 0.0],   colormap: inferno}
  yplus:   {limits: [0.0, 5.0],        colormap: viridis}    # the gate band

camera:
  # Parallel projection half-height, per slice axis. Never "fit to data".
  parallel_scale: {x: 0.35, y: 0.35, z: 0.30}
  # Height above the datum the x- and y-slice cameras look at, so the car is
  # centred in frame rather than sitting at the bottom above empty road.
  focus_height: 0.06
  resolution: [1600, 1200]

# off | lic | seeded. Off by default until the cost is measured on a real
# case: the sampled slices carry U_rel as a vector, so this needs no return
# to the volume, but it is still 350 extra filter evaluations.
streamlines: off
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_views.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from simdev.viz.views import DEFAULT_VIEWS_PATH, load_views

REPO = Path(__file__).resolve().parents[1]


def test_the_shipped_definition_loads() -> None:
    views = load_views(REPO / "cases" / "post_views.yaml")
    assert views.datum_patches == ("Chassis",)
    assert views.resolution == (1600, 1200)
    assert views.streamlines == "off"


def test_the_plane_counts_are_what_the_spec_says() -> None:
    views = load_views(REPO / "cases" / "post_views.yaml")
    assert len(views.axes["x"].offsets()) == 32
    assert len(views.axes["y"].offsets()) == 21
    assert len(views.axes["z"].offsets()) == 17


def test_offsets_are_exact_so_two_runs_name_the_same_files() -> None:
    """Float drift in a plane position becomes a differently-named file.

    Two runs whose x_+0.120.png is called x_+0.11999999 in one of them cannot
    be laid side by side by any tool, so the offsets are rounded on the way
    out rather than left to accumulate.
    """
    views = load_views(REPO / "cases" / "post_views.yaml")
    offsets = views.axes["x"].offsets()
    assert offsets[0] == -0.30
    assert offsets[-1] == 0.32
    assert offsets[15] == pytest.approx(0.0, abs=1e-12)


def test_the_digest_changes_with_the_file(tmp_path: Path) -> None:
    """The digest is what ties a picture to the definition that made it."""
    first = tmp_path / "a.yaml"
    first.write_text(
        (REPO / "cases" / "post_views.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    second = tmp_path / "b.yaml"
    second.write_text(
        first.read_text(encoding="utf-8").replace("step: 0.02", "step: 0.04"),
        encoding="utf-8",
    )
    assert load_views(first).digest != load_views(second).digest


def test_an_unknown_streamline_mode_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "views.yaml"
    path.write_text(
        (REPO / "cases" / "post_views.yaml")
        .read_text(encoding="utf-8")
        .replace("streamlines: off", "streamlines: sparkles"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="streamlines"):
        load_views(path)


def test_every_rendered_field_has_limits() -> None:
    """A field with no limits would silently autoscale, which is the one
    thing this file exists to prevent."""
    from simdev.viz.views import SLICE_FIELDS, SURFACE_FIELDS

    views = load_views(REPO / "cases" / "post_views.yaml")
    for field in (*SLICE_FIELDS, *SURFACE_FIELDS):
        assert field in views.fields, field
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_views.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simdev.viz'`.

- [ ] **Step 4: Implement**

Create `pipeline/simdev/viz/__init__.py` containing a single docstring line:

```python
"""Turning a solved run into pictures that can be compared across runs."""
```

Create `pipeline/simdev/viz/views.py`:

```python
"""Parse the shared view definition.

Pure data in, pure data out. Everything here is unit-testable without a run,
a mesh or ParaView, which is the point of keeping the render plan separate
from the renderer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_VIEWS_PATH = (
    Path(__file__).resolve().parents[3] / "cases" / "post_views.yaml"
)

SLICE_FIELDS: tuple[str, ...] = ("cp", "cpt", "U", "vort", "lambda2")
SURFACE_FIELDS: tuple[str, ...] = ("cp", "yplus")
STREAMLINE_MODES: tuple[str, ...] = ("off", "lic", "seeded")


@dataclass(frozen=True)
class Axis:
    start: float
    stop: float
    step: float

    def offsets(self) -> list[float]:
        """Every plane position on this axis, inclusive of both ends.

        Rounded to nanometres on the way out. Accumulating `start + i * step`
        in binary drifts, and a drifted position becomes a differently-named
        file - at which point two runs that were meant to be laid side by
        side no longer have matching file names.
        """
        if self.step <= 0.0:
            raise ValueError(f"plane step must be positive, got {self.step}")
        count = int(round((self.stop - self.start) / self.step))
        return [round(self.start + i * self.step, 9) for i in range(count + 1)]


@dataclass(frozen=True)
class FieldStyle:
    limits: tuple[float, float]
    colormap: str


@dataclass(frozen=True)
class Views:
    datum_patches: tuple[str, ...]
    axes: dict[str, Axis]
    fields: dict[str, FieldStyle]
    parallel_scale: dict[str, float]
    focus_height: float
    resolution: tuple[int, int]
    streamlines: str
    # sha256 of the file text, truncated. Recorded beside every picture so a
    # PNG can always be traced back to the definition that framed it.
    digest: str


def load_views(path: Path) -> Views:
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)

    streamlines = str(raw.get("streamlines", "off"))
    if streamlines not in STREAMLINE_MODES:
        raise ValueError(
            f"streamlines must be one of {STREAMLINE_MODES}, got "
            f"{streamlines!r}"
        )

    axes = {
        name: Axis(float(v["from"]), float(v["to"]), float(v["step"]))
        for name, v in raw["planes"].items()
    }
    fields = {
        name: FieldStyle(
            (float(v["limits"][0]), float(v["limits"][1])), str(v["colormap"])
        )
        for name, v in raw["fields"].items()
    }

    missing = [f for f in (*SLICE_FIELDS, *SURFACE_FIELDS) if f not in fields]
    if missing:
        raise ValueError(
            f"{path}: no limits for {', '.join(missing)}. A field with no "
            "limits would autoscale per run, which is the one thing this "
            "file exists to prevent"
        )

    camera = raw["camera"]
    return Views(
        datum_patches=tuple(raw["datum"]["patches"]),
        axes=axes,
        fields=fields,
        parallel_scale={k: float(v) for k, v in camera["parallel_scale"].items()},
        focus_height=float(camera.get("focus_height", 0.0)),
        resolution=(int(camera["resolution"][0]), int(camera["resolution"][1])),
        streamlines=streamlines,
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_views.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/viz/__init__.py pipeline/simdev/viz/views.py cases/post_views.yaml tests/test_views.py
git commit -m "feat: one versioned file that frames every picture"
```

---

### Task 11: The car datum

**Files:**
- Create: `pipeline/simdev/viz/datum.py`
- Modify: `pipeline/simdev/stages/prepare.py` (the `detail` dict of the status write)
- Test: `tests/test_datum.py`

**Interfaces:**
- Consumes: `trimesh.Trimesh` objects as `prepare` already holds them in `meshes`.
- Produces:
  - `car_datum(meshes: Mapping[str, object], patches: Sequence[str]) -> tuple[tuple[float, float, float], list[str]]` — the datum and any reasons
  - `status/prepare.json` gains `detail["datum"]` as a three-element list

- [ ] **Step 1: Write the failing test**

Create `tests/test_datum.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from simdev.viz.datum import car_datum


def _box(lo: tuple[float, float, float], hi: tuple[float, float, float]):
    mesh = trimesh.creation.box(
        extents=[hi[i] - lo[i] for i in range(3)]
    )
    mesh.apply_translation([(lo[i] + hi[i]) / 2.0 for i in range(3)])
    return mesh


def test_the_datum_is_the_chassis_bbox_centre_on_the_ground() -> None:
    meshes = {
        "Chassis": _box((-0.20, -0.09, 0.003), (0.21, 0.11, 0.06)),
        # Body is bigger and offset; it must not move the datum.
        "Body": _box((-0.50, -0.30, 0.0), (0.50, 0.30, 0.13)),
    }
    datum, reasons = car_datum(meshes, ["Chassis"])
    assert datum[0] == pytest.approx(0.005, abs=1e-6)
    assert datum[1] == pytest.approx(0.010, abs=1e-6)
    assert datum[2] == 0.0
    assert reasons == []


def test_redesigning_the_body_does_not_move_the_datum() -> None:
    """The whole reason Chassis was chosen over Body or the wheels."""
    chassis = _box((-0.20, -0.09, 0.003), (0.21, 0.11, 0.06))
    before, _ = car_datum({"Chassis": chassis, "Wing": _box((-0.3, -0.1, 0.1), (-0.1, 0.1, 0.13))}, ["Chassis"])
    after, _ = car_datum({"Chassis": chassis, "Wing": _box((-0.6, -0.2, 0.1), (-0.1, 0.2, 0.20))}, ["Chassis"])
    assert before == after


def test_a_case_without_the_named_patch_falls_back_and_says_so() -> None:
    """The Ahmed body has no Chassis and still needs a datum."""
    datum, reasons = car_datum({"body": _box((0.0, -0.2, 0.05), (1.0, 0.2, 0.34))}, ["Chassis"])
    assert datum[0] == pytest.approx(0.5)
    assert any("Chassis" in reason for reason in reasons)


def test_no_geometry_at_all_is_an_error() -> None:
    with pytest.raises(ValueError, match="no geometry"):
        car_datum({}, ["Chassis"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_datum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simdev.viz.datum'`.

- [ ] **Step 3: Implement**

Create `pipeline/simdev/viz/datum.py`:

```python
"""The car-frame origin every picture is centred and measured from."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def car_datum(
    meshes: Mapping[str, object], patches: Sequence[str]
) -> tuple[tuple[float, float, float], list[str]]:
    """Bounding-box centre of the datum patches in x and y; ground in z.

    z is the ground plane, which is fixed at 0 in every case this pipeline
    builds - so a picture's vertical framing is anchored to the road rather
    than to whatever the bodywork happens to reach.

    WHY THE CHASSIS. Wheels move with steering angle. Body and Wing are the
    surfaces under development, and an anchor that shifts when the thing
    being measured is redesigned anchors nothing - two runs would be framed
    differently *because* the wing changed, which is precisely the difference
    the pictures are meant to show. docs/handbook.md calls the Chassis an
    aero dummy that nobody iterates. That is the property wanted here.

    Falls back to every available surface, with a reason, when none of the
    named patches exist - the Ahmed validation case has no Chassis and still
    needs to be framed.
    """
    reasons: list[str] = []
    selected = [meshes[p] for p in patches if p in meshes]

    if not selected:
        if not meshes:
            raise ValueError("no geometry to take a datum from")
        reasons.append(
            f"no datum patch among {', '.join(patches)}: the datum is the "
            "bounding box of every surface instead, so it moves when any of "
            "them is redesigned"
        )
        selected = list(meshes.values())

    lo = np.min([m.bounds[0] for m in selected], axis=0)  # type: ignore[attr-defined]
    hi = np.max([m.bounds[1] for m in selected], axis=0)  # type: ignore[attr-defined]

    return (
        (float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0), 0.0),
        reasons,
    )
```

- [ ] **Step 4: Record it in `prepare`**

In `pipeline/simdev/stages/prepare.py`, before the `write_status(...)` call:

```python
    datum, datum_reasons = car_datum(meshes, ["Chassis"])
    warnings.extend(datum_reasons)
```

and inside the status `detail` dict, next to `"wheels"`:

```python
                # The car-frame origin every picture is centred on. Measured,
                # like the wheels above, so it belongs in the run record
                # rather than in caseSpec.json - putting it in the spec would
                # move the spec hash and invalidate every cached run for a
                # value nothing upstream of post consumes.
                "datum": list(datum),
```

Add the import: `from simdev.viz.datum import car_datum`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_datum.py tests/test_prepare.py tests/test_car_prepare.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/viz/datum.py pipeline/simdev/stages/prepare.py tests/test_datum.py
git commit -m "feat: a datum on the one surface nobody redesigns"
```

---

### Task 12: The render plan

**Files:**
- Create: `pipeline/simdev/viz/plan.py`
- Test: `tests/test_render_plan.py`

**Interfaces:**
- Consumes: `Views` (Task 10), `CaseSpec`, `corner_frame` from `render/context.py`, `Domain`.
- Produces:
  - `SURFACE_VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]` — view name → (camera direction, up)
  - `slice_name(axis: str, offset: float) -> str` — e.g. `"x_+0.120"`
  - `build_render_plan(spec, views, datum, frame, samples_dir, images_dir, stamp) -> dict` returning a JSON-serialisable plan

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_plan.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdev.viz.plan import SURFACE_VIEWS, build_render_plan, slice_name
from simdev.viz.views import load_views

REPO = Path(__file__).resolve().parents[1]
VIEWS = load_views(REPO / "cases" / "post_views.yaml")
DATUM = (0.0036, 0.013, 0.0)
STAMP = {"run": "car-01", "spec_hash": "abc12345", "window": "300-400", "mean": True}


def _plan(frame=None):
    return build_render_plan(
        u_inf=15.0,
        views=VIEWS,
        datum=DATUM,
        frame=frame,
        samples_dir=Path("results/samples"),
        images_dir=Path("results/images"),
        stamp=STAMP,
    )


def test_slice_names_sort_and_align_across_runs() -> None:
    """Signed, fixed width. Two runs' directories must line up name for name."""
    assert slice_name("x", 0.12) == "x_+0.120"
    assert slice_name("x", -0.3) == "x_-0.300"
    names = [slice_name("x", o) for o in (-0.30, -0.02, 0.0, 0.30)]
    assert names == sorted(names)


def test_the_plan_is_json_serialisable() -> None:
    """It crosses a process boundary into an interpreter that cannot import
    simdev, so it has to be plain data."""
    json.dumps(_plan())


def test_every_plane_gets_every_slice_field() -> None:
    plan = _plan()
    assert len(plan["slices"]) == 32 + 21 + 17
    assert len(plan["slices"][0]["images"]) == 5
    assert {i["field"] for i in plan["slices"][0]["images"]} == {
        "cp", "cpt", "U", "vort", "lambda2"
    }


def test_the_surface_suite_is_seven_views_of_two_fields() -> None:
    plan = _plan()
    assert len(plan["surfaces"]) == 7
    assert sum(len(s["images"]) for s in plan["surfaces"]) == 14


def test_car_left_is_plus_y() -> None:
    """Confirmed against the CAD. A flipped sign makes every left.png a
    right.png, and nothing in the image would say so."""
    direction, up = SURFACE_VIEWS["left"]
    assert direction == (0.0, -1.0, 0.0)   # looking toward -y, camera at +y
    assert up == (0.0, 0.0, 1.0)


def test_slice_cameras_track_their_own_plane() -> None:
    plan = _plan()
    first = next(s for s in plan["slices"] if s["axis"] == "x")
    assert first["camera"]["focal"][0] == pytest.approx(DATUM[0] - 0.30)
    # Laterally pinned to the datum, so the car does not drift across frame.
    assert first["camera"]["focal"][1] == pytest.approx(DATUM[1])
    assert first["camera"]["focal"][2] == pytest.approx(VIEWS.focus_height)
    assert first["camera"]["parallel_scale"] == pytest.approx(0.35)


def test_a_straight_line_case_carries_no_rotation() -> None:
    plan = _plan(frame=None)
    assert plan["frame"]["mode"] == "straight"
    assert plan["frame"]["omega"] == 0.0


def test_a_cornering_case_carries_the_frame_the_renderer_needs() -> None:
    """U_rel and the local freestream head are computed in the renderer, so
    omega and the corner centre have to travel with the plan."""
    plan = _plan(frame={"omega": 3.75, "origin": (0.0, 4.0, 0.0)})
    assert plan["frame"]["mode"] == "cornering"
    assert plan["frame"]["omega"] == pytest.approx(3.75)
    assert plan["frame"]["origin"][1] == pytest.approx(4.0)


def test_the_plan_carries_no_expression_strings() -> None:
    """The renderer builds its own expressions from these numbers.

    A formula shipped as text in a data file is an eval waiting to be added,
    and it puts the cornering frame arithmetic in two places at once.
    """
    text = json.dumps(_plan(frame={"omega": 3.75, "origin": (0.0, 4.0, 0.0)}))
    assert "coordsX" not in text and "iHat" not in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_render_plan.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simdev.viz.plan'`.

- [ ] **Step 3: Implement**

Create `pipeline/simdev/viz/plan.py`:

```python
"""Everything the renderer needs, as plain data.

The renderer runs under the system Python (see viz/pv_render.py) and cannot
import simdev, so this module is the interface between them. It carries
numbers and file paths and nothing else: no expression strings, because a
formula shipped as text is an eval waiting to be added and it would put the
cornering-frame arithmetic in two places at once.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from simdev.viz.views import SLICE_FIELDS, SURFACE_FIELDS, Views

# Car axes: nose +x, up +z, therefore car-left +y. Confirmed against the CAD.
# Each entry is (direction the camera looks, up vector); the camera is placed
# opposite its direction, so `front` sits at +x and looks back at the nose.
SURFACE_VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front":  ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "rear":   (( 1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "left":   (( 0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "right":  (( 0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "top":    (( 0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    "bottom": (( 0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "iso":    ((-1.0 / math.sqrt(3), -1.0 / math.sqrt(3), -1.0 / math.sqrt(3)),
               (0.0, 0.0, 1.0)),
}

# Parallel projection, so this only has to be outside the geometry.
CAMERA_DISTANCE = 2.0

# The direction each slice family is viewed along, and its up vector.
#   x: from downstream looking upstream, so car-left (+y) is on the right -
#      the conventional way to read streamwise vortices.
#   y: from the car's right, so the nose points right.
#   z: from above, nose up. A plan view.
SLICE_VIEW = {
    "x": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "z": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
}

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def slice_name(axis: str, offset: float) -> str:
    """Signed, fixed-width, three decimals: `x_+0.120`, `x_-0.300`.

    Signed and zero-padded so the names sort in geometric order and two runs'
    image directories line up file for file. That alignment is the whole
    reason a later side-by-side tool is a small job.
    """
    return f"{axis}_{offset:+.3f}"


def _camera(
    focal: tuple[float, float, float],
    direction: Sequence[float],
    up: Sequence[float],
    parallel_scale: float,
) -> dict[str, Any]:
    return {
        "focal": list(focal),
        "position": [focal[i] - CAMERA_DISTANCE * direction[i] for i in range(3)],
        "up": list(up),
        # NEVER fitted to the data. A camera that rescales itself draws two
        # runs at two magnifications with nothing in the image saying so.
        "parallel_scale": parallel_scale,
    }


def build_render_plan(
    u_inf: float,
    views: Views,
    datum: tuple[float, float, float],
    frame: Mapping[str, Any] | None,
    samples_dir: Path,
    images_dir: Path,
    stamp: Mapping[str, Any],
) -> dict[str, Any]:
    """The whole picture suite, as one JSON-serialisable dict."""
    plan: dict[str, Any] = {
        "views_digest": views.digest,
        "resolution": list(views.resolution),
        "streamlines": views.streamlines,
        "datum": list(datum),
        "stamp": dict(stamp),
        # The renderer builds U_rel, cp and cpt from these. Straight-line and
        # cornering differ only in these numbers, so there is one code path.
        "frame": {
            "mode": "cornering" if frame else "straight",
            "omega": float(frame["omega"]) if frame else 0.0,
            "origin": list(frame["origin"]) if frame else [0.0, 0.0, 0.0],
            "u_inf": float(u_inf),
        },
        "fields": {
            name: {"limits": list(style.limits), "colormap": style.colormap}
            for name, style in views.fields.items()
        },
        "slices": [],
        "surfaces": [],
    }

    for axis, spec_axis in views.axes.items():
        direction, up = SLICE_VIEW[axis]
        index = AXIS_INDEX[axis]
        for offset in spec_axis.offsets():
            name = slice_name(axis, offset)

            point = list(datum)
            point[index] = datum[index] + offset

            # Focal point tracks the plane along its own normal and stays
            # pinned to the datum in the other two axes, so the car sits in
            # the same pixels in every run and every state.
            focal = [datum[0], datum[1], datum[2] + views.focus_height]
            if axis == "z":
                focal = [datum[0], datum[1], datum[2] + offset]
            else:
                focal[index] = datum[index] + offset

            plan["slices"].append({
                "name": name,
                "axis": axis,
                "offset": offset,
                "point": point,
                "normal": [1.0 if i == index else 0.0 for i in range(3)],
                "sample": str(samples_dir / f"{name}"),
                "camera": _camera(
                    (focal[0], focal[1], focal[2]),
                    direction,
                    up,
                    views.parallel_scale[axis],
                ),
                "images": [
                    {
                        "field": field,
                        # vort on a plane is the component NORMAL to it - the
                        # one that shows streamwise vortices punching through.
                        "component": index if field == "vort" else None,
                        "out": str(images_dir / "slices" / axis / field / f"{field}_{name}.png"),
                    }
                    for field in SLICE_FIELDS
                ],
            })

    focal = (datum[0], datum[1], datum[2] + views.focus_height)
    for name, (direction, up) in SURFACE_VIEWS.items():
        plan["surfaces"].append({
            "name": name,
            "camera": _camera(focal, direction, up, max(views.parallel_scale.values())),
            "images": [
                {
                    "field": field,
                    "component": None,
                    "out": str(images_dir / "surface" / field / f"{name}.png"),
                }
                for field in SURFACE_FIELDS
            ],
        })

    return plan
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_render_plan.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/viz/plan.py tests/test_render_plan.py
git commit -m "feat: the render plan, as data a foreign interpreter can read"
```

---

### Task 13: The sampling dictionary

**Files:**
- Create: `pipeline/simdev/render/templates/sampleSurfaces.jinja`
- Modify: `pipeline/simdev/render/context.py` (`build_context`: add `solver_function_names`)
- Create: `pipeline/simdev/viz/sample.py`
- Test: `tests/test_render_sample_dict.py`

**Interfaces:**
- Consumes: `CaseSpec`, `Runner` from `run/runner.py`, the slice list from `build_render_plan` (Task 12).
- Produces:
  - `solver_function_names(spec: CaseSpec) -> list[str]` in `render/context.py`, and `build_context(...)["solver_function_names"]` calling it
  - `render_sample_dict(spec: CaseSpec, out_dir: Path, slices: Sequence[Mapping[str, Any]]) -> Path` writing `system/sampleSurfaces`
  - `run_sampling(run_dir: Path, n_ranks: int) -> Path` returning the directory the surfaces landed in
  - `find_sample(samples_root: Path, name: str) -> Path | None`

**Why this takes no `Domain`.** The sampling dictionary needs only the patch
lists and the function-object names, and both follow from the spec alone. Not
taking a domain is what lets the `images` stage render this dictionary from a
run directory without rebuilding the case.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_sample_dict.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from simdev.config.resolve import resolve
from simdev.render.context import solver_function_names
from simdev.viz.sample import render_sample_dict

BASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed", "symmetric": True, "ahmed": {},
        "patches": [
            {"name": "body", "role": "body"},
            {"name": "ground", "role": "ground"},
            {"name": "symmetry", "role": "symmetry"},
            {"name": "inlet", "role": "inlet"},
            {"name": "outlet", "role": "outlet"},
            {"name": "farfield", "role": "farfield"},
        ],
    },
}

SLICES = [
    {"name": "x_+0.000", "point": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]},
    {"name": "z_+0.050", "point": [0.0, 0.0, 0.05], "normal": [0.0, 0.0, 1.0]},
]


def _spec():
    return resolve(BASE, profile="dev")


def _render(tmp_path: Path) -> str:
    render_sample_dict(_spec(), tmp_path, SLICES)
    return (tmp_path / "system" / "sampleSurfaces").read_text(encoding="utf-8")


def test_every_solve_time_function_object_is_disabled(tmp_path: Path) -> None:
    """postProcess -dict MERGES into the run's controlDict.

    functionObjectList.C:433 reads the run's controlDict and merges the -dict
    file over it, so every solve-time object is still live unless it is
    turned off here. fieldAverage is the dangerous one: re-running it would
    overwrite pMean and UMean with a one-sample average, destroying the very
    fields the pictures are made from. forceCoeffs is the quiet one: it would
    write a new one-row time directory that find_latest's mtime sort then
    prefers over the solve's real history.
    """
    names = solver_function_names(_spec())
    text = _render(tmp_path)
    assert names
    for name in names:
        assert f"    {name}\n    {{\n        enabled         false;" in text, name


def test_the_controldict_and_the_suppression_list_agree(tmp_path: Path) -> None:
    """The list is only protective if it names everything controlDict declares."""
    from simdev.domain.box import BoxDomainBuilder
    from simdev.render.render import render_case

    spec = _spec()
    domain = BoxDomainBuilder().build(spec, ((0.0, -0.1945, 0.05), (1.044, 0.1945, 0.338)))
    render_case(spec, domain, {"body": Path("body.stl")}, tmp_path)
    control = (tmp_path / "system" / "controlDict").read_text(encoding="utf-8")
    for name in solver_function_names(spec):
        assert f"\n    {name}\n" in control, name


def test_derived_fields_are_declared_before_the_sampler(tmp_path: Path) -> None:
    """Function objects execute in dictionary order.

    The other order samples fields that do not exist yet and reports nothing,
    with no error - the same hazard controlDict.jinja documents for
    yPlus/yPlusArea.
    """
    text = _render(tmp_path)
    assert text.index("type            vorticity;") < text.index("type            surfaces;")
    assert text.index("type            Lambda2;") < text.index("type            surfaces;")


def test_the_derived_fields_come_from_the_averaged_velocity(tmp_path: Path) -> None:
    """A slice off the instantaneous field is one arbitrary phase of a limit
    cycle. `field` is mandatory on a fieldExpression and `result` defaults to
    vorticity(UMean), so both are set explicitly."""
    text = _render(tmp_path)
    assert "field           UMean;" in text
    assert "result          vorticityMean;" in text
    assert "result          Lambda2Mean;" in text


def test_every_plane_is_cut(tmp_path: Path) -> None:
    text = _render(tmp_path)
    assert "x_+0.000" in text and "z_+0.050" in text
    assert text.count("type        cuttingPlane;") == 2


def test_the_wall_patches_are_sampled_for_the_surface_views(tmp_path: Path) -> None:
    text = _render(tmp_path)
    assert "type        patch;" in text
    assert "yPlus" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_render_sample_dict.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simdev.viz.sample'`.

- [ ] **Step 3: Publish the function-object names**

In `pipeline/simdev/render/context.py`, add a module-level function — it takes
the spec and nothing else, so the sampling dictionary can be rendered from a
run directory without rebuilding the domain:

```python
def solver_function_names(spec: CaseSpec) -> list[str]:
    # EVERY FUNCTION OBJECT controlDict.jinja DECLARES, BY NAME.
    #
    # Published here rather than duplicated in the sampling template because
    # the sampling pass has to switch all of them off: `postProcess -dict`
    # MERGES its file into the run's controlDict rather than replacing it
    # (functionObjectList.C:433), so a name missing from this list is an
    # object that quietly runs again during sampling. For fieldAverage that
    # means overwriting pMean and UMean with a one-sample average.
    #
    # tests/test_render_sample_dict.py asserts this list against the rendered
    # controlDict, so the two cannot drift apart in silence.
    force_patches = [p.name for p in spec.geometry.patches if traits(p.role).in_forces]
    wall_patches = [p.name for p in spec.geometry.patches if traits(p.role).is_wall]
    return [
        "forceCoeffs",
        "forces",
        *[f"forceCoeffs_{p}" for p in force_patches],
        "yPlus",
        *[f"yPlusArea_{p}" for p in wall_patches],
        *(["fieldAverage"] if spec.solve.average_fields else []),
        "residuals",
    ]
```

and in `build_context`'s returned dict:
`"solver_function_names": solver_function_names(spec),`.

- [ ] **Step 4: Write the template**

Create `pipeline/simdev/render/templates/sampleSurfaces.jinja`:

```jinja
{% with foam_class="dictionary", foam_object="sampleSurfaces" %}
{% include "foam_header.jinja" %}
{% endwith %}

// SLICE AND SURFACE SAMPLING, RUN AFTER THE SOLVE.
//
//     mpirun -np N postProcess -dict system/sampleSurfaces -latestTime -parallel
//
// Sampling in parallel is what lets this touch a 20M-cell case at all: the
// run is decomposed once in the mesh stage and never reconstructed, so every
// rank cuts its own cells and only the surfaces are gathered. Pulling the
// whole mesh through one serial process to slice it would be both slower and
// memory-bound.
//
// -dict MERGES INTO THE RUN'S controlDict, IT DOES NOT REPLACE IT.
// See functionObjectList.C:433. Every solve-time function object is
// therefore still live during this pass unless it is switched off, and two
// of them do real damage if they run:
//
//   fieldAverage  would restart and overwrite pMean and UMean with a
//                 one-sample average - destroying the exact fields these
//                 pictures are made from.
//   forceCoeffs   would write a new one-row time directory that
//                 stages/common.py::find_latest, which sorts by mtime,
//                 would then prefer over the solve's real force history.
//
// The list comes from render/context.py so it cannot drift from what
// controlDict declares.
functions
{
{% for name in solver_function_names %}
    {{ name }}
    {
        enabled         false;
    }
{% endfor %}

    // DERIVED FIELDS FIRST. Function objects execute in dictionary order and
    // `surfaces` below samples what these produce; the other order finds
    // nothing and reports nothing, with no error.
    //
    // Both read UMean, not U. A slice taken off the instantaneous field is
    // one arbitrary phase of a limit cycle, and comparing two designs that
    // way compares two arbitrary phases of two different cycles.
    //
    // `field` is mandatory on a fieldExpression and `result` would otherwise
    // default to `vorticity(UMean)` - a name with brackets in it, which then
    // has to be quoted everywhere downstream. Both are set explicitly.
    //
    // These are the ABSOLUTE-frame vorticity and lambda2, because UMean is
    // the absolute velocity. Relative differs by a constant 2*omega, about
    // 7.5 1/s against a 2000 1/s plotting range, and lambda2 by O(omega^2),
    // about 14 against 5e4. Both negligible; the structures are identical.
    vorticityMean
    {
        type            vorticity;
        libs            ("libfieldFunctionObjects.so");
        field           UMean;
        result          vorticityMean;
        executeControl  timeStep;
        writeControl    none;
    }

    Lambda2Mean
    {
        type            Lambda2;
        libs            ("libfieldFunctionObjects.so");
        field           UMean;
        result          Lambda2Mean;
        executeControl  timeStep;
        writeControl    none;
    }

    surfaces
    {
        type            surfaces;
        libs            ("libsampling.so");
        executeControl  timeStep;
        writeControl    timeStep;
        interpolationScheme cellPoint;
        // The vtk writer emits .vtk or .vtp depending on build
        // (vtkSurfaceWriter.H:76). viz/sample.py globs for both.
        surfaceFormat   vtk;

        fields          (pMean UMean vorticityMean Lambda2Mean);

        surfaces
        {
{% for s in slices %}
            {{ s.name }}
            {
                type        cuttingPlane;
                planeType   pointAndNormal;
                pointAndNormalDict
                {
                    point   ({{ s.point[0] }} {{ s.point[1] }} {{ s.point[2] }});
                    normal  ({{ s.normal[0] }} {{ s.normal[1] }} {{ s.normal[2] }});
                }
                interpolate true;
            }
{% endfor %}
        }
    }

    // The car's own surface, for the seven overall views. A separate object
    // because it wants a different field list: yPlus exists only on walls,
    // and asking the slice sampler for it would sample a volume field that
    // is meaningful nowhere off the wall.
    patchSurfaces
    {
        type            surfaces;
        libs            ("libsampling.so");
        executeControl  timeStep;
        writeControl    timeStep;
        interpolationScheme cell;
        surfaceFormat   vtk;

        fields          (pMean yPlus);

        surfaces
        {
{% for p in force_patches %}
            patch_{{ p }}
            {
                type        patch;
                patches     ({{ p }});
                interpolate false;
            }
{% endfor %}
        }
    }
}
```

- [ ] **Step 5: Implement `viz/sample.py`**

Create `pipeline/simdev/viz/sample.py`:

```python
"""Get the slices and surfaces out of a decomposed run, in parallel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from simdev.config.schema import CaseSpec
from simdev.geometry.roles import traits
from simdev.render.context import solver_function_names
from simdev.render.render import env
from simdev.run.runner import Runner, StageError

SAMPLE_DICT = "system/sampleSurfaces"


def render_sample_dict(
    spec: CaseSpec,
    out_dir: Path,
    slices: Sequence[Mapping[str, Any]],
) -> Path:
    """Write system/sampleSurfaces for this run's plane list.

    Takes no Domain and no geometry files: the dictionary needs the patch
    lists and the function-object names, and both follow from the spec alone.
    That is what lets the images stage render this from a run directory
    without rebuilding the case.
    """
    context = {
        "spec": spec,
        "force_patches": [
            p.name for p in spec.geometry.patches if traits(p.role).in_forces
        ],
        "solver_function_names": solver_function_names(spec),
        "slices": list(slices),
    }

    target = Path(out_dir) / SAMPLE_DICT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        env().get_template("sampleSurfaces.jinja").render(**context),
        encoding="utf-8",
    )
    return target


def run_sampling(run_dir: Path, n_ranks: int) -> Path:
    """Cut every plane and patch at the latest written time.

    Returns the directory the surfaces landed in. Raises rather than guessing
    when nothing was written: an empty sample directory after a successful
    postProcess means the fields it wanted were not in the time directory,
    which is a different problem from a failed command and wants a different
    fix.
    """
    run_dir = Path(run_dir)
    before = _force_coeff_times(run_dir)

    Runner(run_dir).run_parallel(
        ["postProcess", "-dict", SAMPLE_DICT, "-latestTime"],
        n_ranks,
        name="postProcess.sample",
    )

    # Belt and braces on the -dict merge. If a solve-time object slipped
    # through the suppression list it would have written here, and a silently
    # corrupted force history is worth one directory listing to rule out.
    if _force_coeff_times(run_dir) != before:
        raise StageError([
            "the sampling pass wrote new forceCoeffs output, which means a "
            "solve-time function object was not disabled. The force history "
            "may now be wrong: check system/sampleSurfaces against "
            "render/context.py::solver_function_names before trusting "
            "results/report.tsv"
        ])

    roots = sorted((run_dir / "postProcessing" / "surfaces").glob("*"))
    if not roots:
        raise StageError([
            "postProcess wrote no surfaces. The usual cause is that the "
            "latest time directory holds no pMean/UMean - a run stopped "
            "before fieldAverage's timeStart has no averaged fields to cut"
        ])
    return roots[-1]


def _force_coeff_times(run_dir: Path) -> set[str]:
    root = Path(run_dir) / "postProcessing" / "forceCoeffs"
    return {p.name for p in root.glob("*")} if root.exists() else set()


def find_sample(samples_root: Path, name: str) -> Path | None:
    """The .vtp or .vtk a named surface produced, whichever the build wrote."""
    for suffix in (".vtp", ".vtk"):
        candidate = Path(samples_root) / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_render_sample_dict.py tests/test_render_solver.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/simdev/render/templates/sampleSurfaces.jinja pipeline/simdev/render/context.py pipeline/simdev/viz/sample.py tests/test_render_sample_dict.py
git commit -m "feat: sample in parallel, and disable every object -dict would have re-run"
```

---

### Task 14: The renderer

**Files:**
- Create: `pipeline/simdev/viz/pv_render.py`
- Test: `tests/test_pv_render.py`
- Modify: `pyproject.toml` (add the `paraview` marker)

**Interfaces:**
- Consumes: `render_plan.json` as written by Task 12. **Imports nothing from `simdev`.**
- Produces: PNGs at the `out` paths in the plan; prints one JSON line to stdout summarising `{"written": int, "clamped": {path: fraction}}`.

- [ ] **Step 1: Register the marker**

In `pyproject.toml`, extend `[tool.pytest.ini_options] markers`:

```toml
    "paraview: requires a working ParaView python (see cli.py::_doctor)",
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_pv_render.py`:

```python
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RENDERER = (
    Path(__file__).resolve().parents[1]
    / "pipeline" / "simdev" / "viz" / "pv_render.py"
)


def test_the_renderer_imports_nothing_from_simdev() -> None:
    """It runs under /usr/bin/python3, which has paraview and no venv.

    A single `from simdev...` here turns every picture into an ImportError on
    the machine that can actually render.
    """
    text = RENDERER.read_text(encoding="utf-8")
    assert "import simdev" not in text
    assert "from simdev" not in text


def _paraview_python() -> str | None:
    for candidate in ("/usr/bin/python3", sys.executable):
        if shutil.which(candidate) is None and not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-c", "import paraview.simple"],
            capture_output=True, timeout=180,
        )
        if probe.returncode == 0:
            return candidate
    return None


@pytest.mark.paraview
def test_it_renders_a_plane_to_a_png(tmp_path: Path) -> None:
    interpreter = _paraview_python()
    if interpreter is None:
        pytest.skip("no python with paraview.simple")

    # A one-cell polydata square carrying pMean and UMean, written by hand so
    # the test needs no OpenFOAM run.
    sample = tmp_path / "x_+0.000.vtp"
    sample.write_text(VTP, encoding="utf-8")

    out = tmp_path / "cp.png"
    plan = {
        "views_digest": "deadbeef1234",
        "resolution": [320, 240],
        "streamlines": "off",
        "datum": [0.0, 0.0, 0.0],
        "stamp": {"run": "test", "spec_hash": "abc12345", "window": "1-1", "mean": True},
        "frame": {"mode": "straight", "omega": 0.0, "origin": [0, 0, 0], "u_inf": 10.0},
        "fields": {"cp": {"limits": [-1.0, 1.0], "colormap": "coolwarm"}},
        "slices": [{
            "name": "x_+0.000",
            "axis": "x",
            "offset": 0.0,
            "sample": str(sample),
            "camera": {
                "focal": [0.0, 0.0, 0.0], "position": [-2.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0], "parallel_scale": 1.0,
            },
            "images": [{"field": "cp", "component": None, "out": str(out)}],
        }],
        "surfaces": [],
    }
    plan_path = tmp_path / "render_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [interpreter, str(RENDERER), str(plan_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 1000
    assert json.loads(result.stdout.strip().splitlines()[-1])["written"] == 1


VTP = """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="4" NumberOfPolys="1">
      <Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">
        0 -1 -1  0 1 -1  0 1 1  0 -1 1
      </DataArray></Points>
      <Polys>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
      </Polys>
      <PointData>
        <DataArray type="Float32" Name="pMean" format="ascii">-40 -10 10 40</DataArray>
        <DataArray type="Float32" Name="UMean" NumberOfComponents="3" format="ascii">
          10 0 0  9 1 0  8 2 0  7 3 0
        </DataArray>
      </PointData>
    </Piece>
  </PolyData>
</VTKFile>
"""
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pv_render.py -v`
Expected: FAIL — the file does not exist, so `RENDERER.read_text` raises `FileNotFoundError`.

- [ ] **Step 4: Implement the renderer**

Create `pipeline/simdev/viz/pv_render.py`:

```python
"""Turn a render plan into PNGs. Runs under the SYSTEM python, not the venv.

    /usr/bin/python3 pv_render.py render_plan.json

THIS FILE MUST NOT IMPORT simdev. ParaView is a system package; the venv has
no access to it and the ParaView interpreter has no access to the venv, so
this module and the rest of the pipeline meet only at render_plan.json.
Everything that needs testing lives on the other side of that file.

pvpython and pvbatch hang on this machine - measured, no output at a 150 s
timeout, not even for --version - while `python3 -c "import paraview.simple"`
imports and renders in seconds. Hence the plain interpreter.
"""

from __future__ import annotations

import json
import os
import sys

from paraview.simple import (  # type: ignore[import-not-found]
    Calculator, ColorBy, CreateRenderView, Delete, GetColorTransferFunction,
    GetScalarBar, Hide, LegacyVTKReader, Render, SaveScreenshot, Show, Text,
    XMLPolyDataReader,
)

# Friendly name -> ParaView preset. Kept here rather than in the yaml so the
# view file stays about the picture rather than about ParaView.
# View-file field name -> the array name actually present on the sampled
# surface. cp, cpt, U and vort are built by the Calculators below and are
# named to match; Lambda2Mean comes from the OpenFOAM function object and
# yPlus is OpenFOAM's own spelling. A mismatch here colours by nothing and
# renders a uniformly grey picture with no error.
ARRAY_NAMES = {"lambda2": "Lambda2Mean", "yplus": "yPlus"}

PRESETS = {
    "coolwarm": "Cool to Warm",
    "viridis": "Viridis (matplotlib)",
    "inferno": "Inferno (matplotlib)",
    "plasma": "Plasma (matplotlib)",
}


def _reader(path):
    if str(path).endswith(".vtp"):
        return XMLPolyDataReader(FileName=[str(path)])
    return LegacyVTKReader(FileNames=[str(path)])


def _derive(source, frame):
    """Add U_rel, cp, cpt, Umag and vorticity magnitude to a sampled surface.

    THE CAR FRAME, AND WHY IT IS NOT COSMETIC. In a cornering run OpenFOAM
    solves the ABSOLUTE velocity, so UMean's far field is at rest in the
    ground frame. Sliced raw beside a straight-line run the whole freestream
    changes colour and none of it is aerodynamics.

        U_rel  = UMean - omega x (x - origin)
        U_ff   = |omega| r          (the undisturbed car-frame speed)
        cp     = pMean / (0.5 u_inf^2)
        cpt    = (pMean + 0.5|U_rel|^2 - 0.5 U_ff^2) / (0.5 u_inf^2)

    cpt subtracts the LOCAL head because at R = 4 m and omega = 3.75 rad/s
    the undisturbed car-frame speed runs 12.75-17.25 m/s across the domain
    half-width. Against a constant u_inf that is a spurious +/-0.32 in cpt,
    graded radially, and it reads exactly like a wake.

    cp keeps 0.5 u_inf^2 in its denominator - the same dynamic head that
    non-dimensionalises Cd and Cl - so a cp picture and a coefficient sit on
    one scale.

    At omega = 0 every expression below collapses to the straight-line form,
    so there is one code path and no cornering branch.
    """
    w = float(frame["omega"])
    ox, oy, _oz = frame["origin"]
    u_inf = float(frame["u_inf"])
    q = 0.5 * u_inf * u_inf

    # omega x d for omega = (0, 0, w) and d = x - origin is (-w*dy, w*dx, 0),
    # so U_rel = (Ux + w*dy, Uy - w*dx, Uz).
    rel = Calculator(Input=source)
    rel.ResultArrayName = "U_rel"
    rel.Function = (
        f"(UMean_X + {w!r}*(coordsY - {oy!r}))*iHat"
        f" + (UMean_Y - {w!r}*(coordsX - {ox!r}))*jHat"
        f" + (UMean_Z)*kHat"
    )

    mag = Calculator(Input=rel)
    mag.ResultArrayName = "U"
    mag.Function = "mag(U_rel)"

    cp = Calculator(Input=mag)
    cp.ResultArrayName = "cp"
    cp.Function = f"pMean/{q!r}"

    # U_ff^2: (w r)^2 in a rotating frame, u_inf^2 when there is no rotation.
    uff2 = (
        f"({w!r}*sqrt((coordsX - {ox!r})^2 + (coordsY - {oy!r})^2))^2"
        if w
        else f"{u_inf * u_inf!r}"
    )
    cpt = Calculator(Input=cp)
    cpt.ResultArrayName = "cpt"
    cpt.Function = f"(pMean + 0.5*mag(U_rel)^2 - 0.5*({uff2}))/{q!r}"

    return cpt


def _vorticity(source, component):
    """|vorticity| normal to this plane - the component that shows vortices
    punching through the cut, which is what the old pipeline plotted."""
    axis = ("X", "Y", "Z")[component]
    out = Calculator(Input=source)
    out.ResultArrayName = "vort"
    out.Function = f"abs(vorticityMean_{axis})"
    return out


def _stamp(view, lines):
    """Provenance in the corner of every image.

    Two pictures a month apart are worthless if you cannot tell which run,
    which window and which colour limits made them.
    """
    text = Text()
    text.Text = "\n".join(lines)
    display = Show(text, view)
    display.WindowLocation = "Lower Left"
    display.FontSize = 9
    return text


def _draw(source, field, style, camera, out_path, resolution, stamp_lines):
    view = CreateRenderView()
    view.CameraParallelProjection = 1
    view.CameraPosition = camera["position"]
    view.CameraFocalPoint = camera["focal"]
    view.CameraViewUp = camera["up"]
    view.CameraParallelScale = camera["parallel_scale"]
    view.OrientationAxesVisibility = 0
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0

    display = Show(source, view)
    ColorBy(display, ("POINTS", field))

    lut = GetColorTransferFunction(field)
    lut.ApplyPreset(PRESETS.get(style["colormap"], style["colormap"]), True)
    low, high = style["limits"]
    lut.RescaleTransferFunction(low, high)

    bar = GetScalarBar(lut, view)
    bar.Title = field
    bar.ComponentTitle = ""

    display.SetScalarBarVisibility(view, True)
    _stamp(view, [*stamp_lines, f"{field}  [{low:g}, {high:g}]"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Render(view)
    SaveScreenshot(out_path, view, ImageResolution=resolution)
    Hide(source, view)
    Delete(view)


def main(argv):
    with open(argv[1], encoding="utf-8") as handle:
        plan = json.load(handle)

    frame = plan["frame"]
    resolution = plan["resolution"]
    stamp = plan["stamp"]
    base_lines = [
        f"{stamp['run']}  spec {stamp['spec_hash']}  views {plan['views_digest']}",
        f"iterations {stamp['window']}  "
        + ("MEAN" if stamp.get("mean") else "INSTANTANEOUS"),
    ]

    written = 0
    missing = []

    for group, label in ((plan["slices"], "slice"), (plan["surfaces"], "surface")):
        for entry in group:
            sample = entry.get("sample")
            if not sample or not os.path.exists(sample):
                missing.append(entry["name"])
                continue
            source = _derive(_reader(sample), frame)
            for image in entry["images"]:
                field = image["field"]
                node = source
                if field == "vort":
                    node = _vorticity(source, image["component"])
                style = plan["fields"][field]
                lines = [*base_lines, f"{label} {entry['name']}"]
                _draw(
                    node,
                    ARRAY_NAMES.get(field, field),
                    style,
                    entry["camera"],
                    image["out"],
                    resolution,
                    lines,
                )
                written += 1

    print(json.dumps({"written": written, "missing": missing}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_pv_render.py -v`
Expected: the import-hygiene test PASSES. The `paraview` test either PASSES or
SKIPS depending on the machine. On this machine it should pass — verify it
does, and if the PNG is blank, check `view.UseColorPaletteForBackground`
first: a white foreground on a white background renders an empty image.

- [ ] **Step 6: Note what is deliberately not wired yet**

`plan["streamlines"]` reaches the renderer and is ignored. That is intentional
and it is the spec's §6.8: the user's condition was "depends on rendering
time", and there is no honest way to decide before Task 18 Step 2 measures
what 364 images already cost. `load_views` still validates the key, so the
setting cannot be misspelled in the meantime.

Add the marker so it is not mistaken for an oversight, immediately after the
plan is loaded in `main()`:

```python
    if plan.get("streamlines", "off") != "off":
        # Not implemented yet - see the plan, Task 14. The sampled surfaces
        # carry U_rel as a vector, so this needs no return to the volume; it
        # is waiting on a measurement, not on a mechanism.
        print(
            f"warning: streamlines={plan['streamlines']} is not implemented; "
            "rendering without them",
            file=sys.stderr,
        )
```

- [ ] **Step 7: Commit**

```bash
git add pipeline/simdev/viz/pv_render.py tests/test_pv_render.py pyproject.toml
git commit -m "feat: a renderer that knows the car frame and nothing about simdev"
```

---

### Task 15: The `images` stage

**Files:**
- Create: `pipeline/simdev/stages/images.py`
- Modify: `pipeline/simdev/stages/prepare.py` (record `corner_frame` and `geometry_bounds`)
- Modify: `pipeline/simdev/config/schema.py` (`PostConfig.paraview_python`, `PostConfig.views`)
- Test: `tests/test_images.py`

**Interfaces:**
- Consumes: `load_views`/`DEFAULT_VIEWS_PATH` (Task 10), `build_render_plan` (Task 12), `render_sample_dict`/`run_sampling`/`find_sample` (Task 13), `pv_render.py` (Task 14).
- Produces:
  - `images(run_dir: Path, force: bool = False, axes: Sequence[str] | None = None, fields: Sequence[str] | None = None) -> dict[str, Any]`
  - `results/images.json`, `results/views.yaml`, `results/samples/`, `results/images/`
  - `status/images.json`

- [ ] **Step 1: Record what the stage needs from `prepare`**

In `pipeline/simdev/stages/prepare.py`, alongside the `"datum"` entry added in
Task 11:

`prepare` already computes the frame for the wheel speeds; name it once above
the `write_status` call —

```python
    frame = corner_frame(spec, domain)
```

— and add to the `detail` dict:

```python
                # The rotating frame, so the images stage can put the pictures
                # in the car's frame without rebuilding the domain. Two
                # numbers are cheaper to record than a domain is to
                # reconstruct, and they are measured rather than configured.
                "corner_frame": (
                    {"omega": frame.omega, "origin": list(frame.origin)}
                    if frame is not None
                    else None
                ),
                # For the coverage warning: slices that stop short of the car
                # are a silent hole in the picture suite.
                "geometry_bounds": [list(lo), list(hi)],
```

- [ ] **Step 2: Add the two config knobs**

In `pipeline/simdev/config/schema.py`, extend `PostConfig`:

```python
    # The interpreter that can import paraview.simple.
    #
    # NOT pvpython. On the development machine pvpython and pvbatch do not
    # return at all - measured, no output at a 150 s timeout, not even for
    # --version - while /usr/bin/python3 with the python3-paraview package
    # imports and renders offscreen in seconds. `simdev doctor` checks it.
    paraview_python: str = "/usr/bin/python3"
    # Path to the shared view definition. None means cases/post_views.yaml,
    # which is the answer for every real run; the override exists so a test
    # can point at a two-plane file instead of a seventy-plane one.
    views: str | None = None
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_images.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from simdev.run.runner import StageError
from simdev.run.status import StageStatus, read_status, write_status
from simdev.stages.images import images
from simdev.stages.prepare import prepare

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed", "symmetric": True, "ahmed": {"include_stilts": False},
        "patches": [
            {"name": "body", "role": "body"},
            {"name": "ground", "role": "ground"},
            {"name": "symmetry", "role": "symmetry"},
            {"name": "inlet", "role": "inlet"},
            {"name": "outlet", "role": "outlet"},
            {"name": "farfield", "role": "farfield"},
        ],
    },
}

TINY_VIEWS = """
datum:  {patches: [Chassis]}
planes:
  x: {from: -0.1, to: 0.1, step: 0.1}
  y: {from: 0.0, to: 0.0, step: 0.1}
  z: {from: 0.0, to: 0.0, step: 0.1}
fields:
  cp:      {limits: [-3.0, 1.0], colormap: coolwarm}
  cpt:     {limits: [-3.0, 1.0], colormap: coolwarm}
  U:       {limits: [0.0, 60.0], colormap: viridis}
  vort:    {limits: [0.0, 2000.0], colormap: inferno}
  lambda2: {limits: [-50000.0, 0.0], colormap: inferno}
  yplus:   {limits: [0.0, 300.0], colormap: viridis}
camera:
  parallel_scale: {x: 0.4, y: 0.6, z: 0.6}
  focus_height: 0.15
  resolution: [320, 240]
streamlines: off
"""


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    views = tmp_path / "views.yaml"
    views.write_text(TINY_VIEWS, encoding="utf-8")
    case = dict(CASE, post={"views": str(views)})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    target = tmp_path / "run"
    prepare(path, target, profile="dev")
    return target


def test_images_refuses_before_a_solve(run_dir: Path) -> None:
    with pytest.raises(StageError, match="solve"):
        images(run_dir)


def test_prepare_records_the_datum_and_the_bounds(run_dir: Path) -> None:
    detail = read_status(run_dir, "prepare").detail
    assert len(detail["datum"]) == 3
    assert detail["datum"][2] == 0.0
    assert detail["corner_frame"] is None       # straight-line case
    assert len(detail["geometry_bounds"]) == 2


def test_the_sampling_dict_is_written_before_anything_runs(
    run_dir: Path, monkeypatch
) -> None:
    """The dict is the reviewable artefact. It must exist even when the
    OpenFOAM call is what fails."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(
        module, "run_sampling",
        lambda *a, **k: (_ for _ in ()).throw(StageError(["postProcess exploded"])),
    )
    with pytest.raises(StageError):
        images(run_dir)
    assert (run_dir / "system" / "sampleSurfaces").exists()


def test_the_plan_covers_the_configured_planes(run_dir: Path, monkeypatch) -> None:
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(module, "run_sampling", lambda *a, **k: run_dir / "nowhere")
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    images(run_dir)
    plan = json.loads((run_dir / "results" / "render_plan.json").read_text())
    # 3 x-planes, 1 y-plane, 1 z-plane.
    assert len(plan["slices"]) == 5
    assert len(plan["surfaces"]) == 7


def test_axes_and_fields_narrow_the_work(run_dir: Path, monkeypatch) -> None:
    """The flag exists so iterating on one view does not cost 364 images."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(module, "run_sampling", lambda *a, **k: run_dir / "nowhere")
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    images(run_dir, axes=["x"], fields=["cp"])
    plan = json.loads((run_dir / "results" / "render_plan.json").read_text())
    assert {s["axis"] for s in plan["slices"]} == {"x"}
    assert all(len(s["images"]) == 1 for s in plan["slices"])


def test_the_views_file_travels_with_the_pictures(run_dir: Path, monkeypatch) -> None:
    """A picture whose framing cannot be reconstructed is an orphan."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(module, "run_sampling", lambda *a, **k: run_dir / "nowhere")
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    record = images(run_dir)
    assert (run_dir / "results" / "views.yaml").exists()
    assert record["views_digest"] == json.loads(
        (run_dir / "results" / "images.json").read_text()
    )["views_digest"]


def test_slices_that_miss_the_car_are_reported(run_dir: Path, monkeypatch) -> None:
    """Silently drawing half the car is worse than saying the range is short."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(module, "run_sampling", lambda *a, **k: run_dir / "nowhere")
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    record = images(run_dir)
    # The Ahmed body is 1.044 m long; the tiny views file spans 0.2 m.
    assert any("does not cover" in r for r in record["reasons"])
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_images.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simdev.stages.images'`.

- [ ] **Step 5: Implement**

Create `pipeline/simdev/stages/images.py`:

```python
"""Pictures of a solved run, framed identically to every other run's."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from simdev.run.runner import StageError
from simdev.run.status import StageStatus, read_status, write_status
from simdev.stages.common import load_spec
from simdev.viz.plan import build_render_plan
from simdev.viz.sample import find_sample, render_sample_dict, run_sampling
from simdev.viz.views import DEFAULT_VIEWS_PATH, load_views

STAGE = "images"

RENDERER = Path(__file__).resolve().parent.parent / "viz" / "pv_render.py"


def _render(interpreter: str, plan_path: Path) -> dict[str, Any]:
    """Hand the plan to the ParaView interpreter and read back what it did."""
    result = subprocess.run(
        [interpreter, str(RENDERER), str(plan_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise StageError([
            f"the renderer failed under {interpreter}",
            result.stderr.strip()[-2000:] or "no stderr",
            "run 'simdev doctor' to check the interpreter can import "
            "paraview.simple. Note that pvpython and pvbatch are NOT the "
            "answer here - they hang on this machine",
        ])
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise StageError([
            "the renderer wrote no summary line",
            result.stdout.strip()[-2000:] or "no stdout",
        ]) from None


def images(
    run_dir: Path,
    force: bool = False,
    axes: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Sample the run, then draw it. Never gates: these are pictures."""
    run_dir = Path(run_dir)
    spec = load_spec(run_dir)

    solve_status = read_status(run_dir, "solve")
    if solve_status is None:
        raise StageError(["stage 'solve' has not been run"])
    if solve_status.state == "failed":
        raise StageError(["stage 'solve' failed outright; there is nothing to draw"])
    # 'gate_failed' is fine, exactly as in post: a run that did not plateau
    # still has a flow field worth looking at, and the stamp says so.

    prepare_status = read_status(run_dir, "prepare")
    if prepare_status is None or "datum" not in prepare_status.detail:
        raise StageError([
            "no datum in status/prepare.json: this run was prepared before "
            "the images stage existed. Re-run 'simdev prepare --force' on it "
            "- it re-renders dictionaries and re-measures geometry, it does "
            "not touch the mesh or the solution"
        ])

    views_path = Path(spec.post.views) if spec.post.views else DEFAULT_VIEWS_PATH
    views = load_views(views_path)
    datum = tuple(prepare_status.detail["datum"])

    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)

    post_status = read_status(run_dir, "post")
    window = (
        int(post_status.detail.get("window_start", 0)) if post_status else 0,
        int(post_status.detail.get("window_end", 0)) if post_status else 0,
    )

    plan = build_render_plan(
        u_inf=spec.flow.u_inf,
        views=views,
        datum=datum,
        frame=prepare_status.detail.get("corner_frame"),
        samples_dir=results / "samples",
        images_dir=results / "images",
        stamp={
            "run": run_dir.name,
            "spec_hash": spec.spec_hash()[:8],
            "window": f"{window[0]}-{window[1]}",
            "mean": bool(spec.solve.average_fields),
        },
    )

    if axes:
        plan["slices"] = [s for s in plan["slices"] if s["axis"] in set(axes)]
    if fields:
        wanted = set(fields)
        for entry in (*plan["slices"], *plan["surfaces"]):
            entry["images"] = [i for i in entry["images"] if i["field"] in wanted]

    reasons: list[str] = []
    lo, hi = prepare_status.detail["geometry_bounds"]
    for index, axis in enumerate("xyz"):
        if axis not in views.axes:
            continue
        offsets = views.axes[axis].offsets()
        low, high = datum[index] + offsets[0], datum[index] + offsets[-1]
        if low > lo[index] or high < hi[index]:
            # NOT silently extended. An auto-fitted range produces a different
            # picture wearing the same file name, and then two runs that look
            # comparable are not.
            reasons.append(
                f"the {axis} slice range {low:+.3f}..{high:+.3f} does not "
                f"cover the geometry {lo[index]:+.3f}..{hi[index]:+.3f}; "
                f"widen planes.{axis} in {views_path.name} if that matters"
            )

    # Written before anything runs, so the dictionary is on disk to read even
    # when the OpenFOAM call is what fails.
    render_sample_dict(spec, run_dir, plan["slices"])

    sampled_at = time.monotonic()
    samples_root = run_sampling(run_dir, spec.solve.n_ranks)
    sample_seconds = time.monotonic() - sampled_at

    for entry in plan["slices"]:
        found = find_sample(samples_root, entry["name"])
        entry["sample"] = str(found) if found else None
    for entry in plan["surfaces"]:
        # One patch surface per force patch; the renderer needs them merged,
        # so the plan names them all and pv_render reads the first that
        # exists. A vehicle whose patches sample separately is drawn from the
        # bodywork patch, which is what the seven overall views are of.
        for patch in spec.geometry.patches:
            found = find_sample(samples_root, f"patch_{patch.name}")
            if found:
                entry["sample"] = str(found)
                break

    plan_path = results / "render_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (results / "views.yaml").write_text(
        views_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    drawn_at = time.monotonic()
    summary = _render(spec.post.paraview_python, plan_path)
    render_seconds = time.monotonic() - drawn_at

    if summary.get("missing"):
        reasons.append(
            f"{len(summary['missing'])} surfaces produced no sample and were "
            f"not drawn (first: {summary['missing'][0]})"
        )

    record = {
        "run": run_dir.name,
        "spec_hash": spec.spec_hash(),
        "views_digest": views.digest,
        "datum": list(datum),
        "images_written": summary.get("written", 0),
        "sample_seconds": round(sample_seconds, 1),
        "render_seconds": round(render_seconds, 1),
        "reasons": reasons,
    }
    (results / "images.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            # Always ok when it completed. These are pictures; a short slice
            # range is a note, not a failed run.
            state="ok",
            input_hash=spec.spec_hash(),
            reasons=reasons,
            detail={
                "images_written": record["images_written"],
                "views_digest": views.digest,
                "sample_seconds": record["sample_seconds"],
                "render_seconds": record["render_seconds"],
            },
        ),
    )
    return record
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_images.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 7: Commit**

```bash
git add pipeline/simdev/stages/images.py pipeline/simdev/stages/prepare.py pipeline/simdev/config/schema.py tests/test_images.py
git commit -m "feat: the images stage, which never gates a run"
```

---

### Task 16: CLI and the doctor check

**Files:**
- Modify: `pipeline/simdev/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `images` (Task 15).
- Produces: `simdev images <run-dir> [--force] [--axes x y] [--fields cp U]`; a ParaView line in `simdev doctor`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_doctor_reports_the_paraview_interpreter(capsys) -> None:
    """The one environment check that would otherwise surface as 364 missing
    pictures at the end of a long run."""
    main(["doctor"])
    assert "paraview" in capsys.readouterr().out.lower()


def test_images_takes_axis_and_field_filters() -> None:
    parser = _parser()
    args = parser.parse_args(["images", "/tmp/run", "--axes", "x", "--fields", "cp", "U"])
    assert args.axes == ["x"]
    assert args.fields == ["cp", "U"]
```

(`_parser` is module-private; import it as `from simdev.cli import _parser, main`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -k "doctor_reports or axis_and_field" -v`
Expected: FAIL — argparse rejects `images`, and `doctor` prints no ParaView line.

- [ ] **Step 3: Add the subcommand**

In `_parser()`:

```python
    img = subparsers.add_parser(
        "images",
        help="render the slice and surface suite for a solved run",
    )
    img.add_argument("run_dir", type=Path)
    img.add_argument("--force", action="store_true")
    img.add_argument(
        "--axes", nargs="+", default=None, choices=["x", "y", "z"],
        help="only these slice axes. Iterating on one view should not cost 364 images",
    )
    img.add_argument(
        "--fields", nargs="+", default=None,
        help="only these fields, e.g. --fields cp U",
    )
```

In `main()`:

```python
        if args.command == "images":
            record = images(
                args.run_dir, force=args.force, axes=args.axes, fields=args.fields
            )
            print(
                f"{record['images_written']} images  "
                f"sample {record['sample_seconds']}s  "
                f"render {record['render_seconds']}s"
            )
            for reason in record["reasons"]:
                print(f"  note: {reason}", file=sys.stderr)
            return 0
```

Add `from simdev.stages.images import images`.

- [ ] **Step 4: Add the doctor check**

In `_doctor()`, after the gmsh check:

```python
    # ParaView, and NOT via pvpython. pvpython and pvbatch do not return on
    # this machine - measured, no output at a 150 s timeout, not even for
    # --version - while the same install imports fine under the plain system
    # interpreter. Checked here because the alternative is discovering it
    # after a solve, when the pictures are what is missing.
    interpreter = "/usr/bin/python3"
    probe = subprocess.run(
        [interpreter, "-c", "import paraview.simple"],
        capture_output=True, timeout=300,
    )
    if probe.returncode == 0:
        print(f"{'paraview (images)':18s} {interpreter}")
    else:
        print(f"{'paraview (images)':18s} BROKEN under {interpreter}")
        print(
            "\nThe images stage needs an interpreter that can 'import "
            "paraview.simple'. On Ubuntu:\n  sudo apt-get install -y "
            "python3-paraview\nSet post.paraview_python if it lives "
            "elsewhere. Do NOT point it at pvpython."
        )
        missing.append("paraview")
```

Add `import subprocess` at the top of `cli.py`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS. Then run `.venv/bin/simdev doctor` by hand and confirm the
ParaView line reports the interpreter rather than BROKEN.

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/cli.py tests/test_cli.py
git commit -m "feat: simdev images, and a doctor check that fails before the solve does"
```

---

### Task 17: The contact sheet

**Files:**
- Create: `pipeline/simdev/viz/sheet.py`
- Modify: `pipeline/simdev/stages/images.py`
- Test: `tests/test_sheet.py`

**Interfaces:**
- Consumes: the plan dict (Task 12) and the record dict (Task 15).
- Produces: `write_contact_sheet(results_dir: Path, plan: dict, record: dict) -> Path` writing `results/index.html`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sheet.py`:

```python
from __future__ import annotations

from pathlib import Path

from simdev.viz.sheet import write_contact_sheet

PLAN = {
    "slices": [
        {"name": "x_+0.000", "axis": "x", "offset": 0.0,
         "images": [{"field": "cp", "out": "results/images/slices/x/cp/cp_x_+0.000.png"}]},
        {"name": "x_+0.020", "axis": "x", "offset": 0.02,
         "images": [{"field": "cp", "out": "results/images/slices/x/cp/cp_x_+0.020.png"}]},
    ],
    "surfaces": [
        {"name": "front",
         "images": [{"field": "yplus", "out": "results/images/surface/yplus/front.png"}]},
    ],
}
RECORD = {
    "run": "car-01", "spec_hash": "abc123", "views_digest": "deadbeef1234",
    "datum": [0.0036, 0.013, 0.0], "images_written": 3,
    "sample_seconds": 12.3, "render_seconds": 45.6, "reasons": [],
}


def test_the_sheet_lists_every_image(tmp_path: Path) -> None:
    out = write_contact_sheet(tmp_path, PLAN, RECORD)
    text = out.read_text(encoding="utf-8")
    assert text.count("<img") == 3
    assert "cp_x_+0.020.png" in text


def test_the_sheet_carries_the_provenance(tmp_path: Path) -> None:
    """It is the index someone opens six months later."""
    text = write_contact_sheet(tmp_path, PLAN, RECORD).read_text(encoding="utf-8")
    assert "deadbeef1234" in text
    assert "car-01" in text


def test_the_sheet_shows_the_warnings(tmp_path: Path) -> None:
    record = dict(RECORD, reasons=["the x slice range does not cover the geometry"])
    text = write_contact_sheet(tmp_path, PLAN, record).read_text(encoding="utf-8")
    assert "does not cover" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sheet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simdev.viz.sheet'`.

- [ ] **Step 3: Implement**

Create `pipeline/simdev/viz/sheet.py`:

```python
"""An index for 364 pictures, so they can be flipped through rather than found."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any, Mapping

_STYLE = """
body { font: 13px/1.4 system-ui, sans-serif; margin: 2rem; background: #fafafa; }
h1 { font-size: 1.2rem; margin-bottom: 0.2rem; }
.meta { color: #555; margin-bottom: 1.5rem; }
.warn { background: #fff4e5; border-left: 3px solid #e08a00; padding: 0.6rem 0.9rem;
        margin-bottom: 1.5rem; }
h2 { font-size: 1rem; margin-top: 2rem; border-bottom: 1px solid #ddd;
     padding-bottom: 0.3rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 0.8rem; }
figure { margin: 0; }
img { width: 100%; border: 1px solid #ddd; background: #fff; }
figcaption { font-size: 11px; color: #666; margin-top: 0.2rem; }
"""


def write_contact_sheet(
    results_dir: Path, plan: Mapping[str, Any], record: Mapping[str, Any]
) -> Path:
    """One page per run, grouped by axis and field, positions in order."""
    results_dir = Path(results_dir)
    out = results_dir / "index.html"

    groups: dict[str, list[tuple[str, str]]] = {}
    for entry in plan["slices"]:
        for image in entry["images"]:
            key = f"slices — {entry['axis']} — {image['field']}"
            groups.setdefault(key, []).append((image["out"], entry["name"]))
    for entry in plan["surfaces"]:
        for image in entry["images"]:
            key = f"surface — {image['field']}"
            groups.setdefault(key, []).append((image["out"], entry["name"]))

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(str(record['run']))}</title>",
        f"<style>{_STYLE}</style>",
        f"<h1>{html.escape(str(record['run']))}</h1>",
        "<div class='meta'>"
        f"spec {html.escape(str(record['spec_hash'])[:8])} &middot; "
        f"views {html.escape(str(record['views_digest']))} &middot; "
        f"datum {record['datum']} &middot; "
        f"{record['images_written']} images &middot; "
        f"sampled in {record['sample_seconds']}s, rendered in "
        f"{record['render_seconds']}s</div>",
    ]

    if record.get("reasons"):
        parts.append("<div class='warn'>")
        for reason in record["reasons"]:
            parts.append(f"<div>{html.escape(str(reason))}</div>")
        parts.append("</div>")

    for key in sorted(groups):
        parts.append(f"<h2>{html.escape(key)}</h2><div class='grid'>")
        for path, label in sorted(groups[key], key=lambda p: p[1]):
            relative = os.path.relpath(path, results_dir)
            parts.append(
                f"<figure><a href='{html.escape(relative)}'>"
                f"<img src='{html.escape(relative)}' loading='lazy'></a>"
                f"<figcaption>{html.escape(label)}</figcaption></figure>"
            )
        parts.append("</div>")

    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out
```

- [ ] **Step 4: Call it from the stage**

In `pipeline/simdev/stages/images.py`, after `images.json` is written:

```python
    write_contact_sheet(results, plan, record)
```

Add `from simdev.viz.sheet import write_contact_sheet`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sheet.py tests/test_images.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/viz/sheet.py pipeline/simdev/stages/images.py tests/test_sheet.py
git commit -m "feat: an index for 364 pictures"
```

---

### Task 18: Documentation, and a measured first run

**Files:**
- Modify: `docs/handbook.md`
- Modify: `docs/superpowers/specs/2026-09-01-postprocessing-reports-and-images-design.md`

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: PASS. Note how many `paraview`- and `openfoam`-marked tests skipped.

- [ ] **Step 2: Run one real case end to end**

Run:

```bash
.venv/bin/simdev run cases/car/config.yaml --run-dir ~/runs/car-images-01 --profile car_smoke
.venv/bin/simdev images ~/runs/car-images-01
.venv/bin/simdev report ~/runs/car-images-01
```

Record the actual numbers: sampling seconds, rendering seconds, total image
count, and the size of `results/samples/`. These are the numbers the spec's
"Sampling cost is unmeasured" risk is waiting for.

**Then decide the streamlines question with them.** §6.8 is the one part of
the spec left unimplemented (Task 14, Step 6), because "depends on rendering
time" cannot be answered before there is a rendering time. If the 364-image
baseline is comfortable, implementing in-plane LIC on the U slices is a small
follow-up — the sampled surfaces already carry `U_rel` as a vector, so it
needs no return to the volume. If it is not comfortable, the honest answer is
to leave the flag at `off` and say so in the handbook.

- [ ] **Step 3: Update the handbook**

In `docs/handbook.md`, under the post-processing material, add a section
covering: what `results/report.tsv` contains and that its column order is a
contract; that COP is three diagnostics and not a point; that
`cases/post_views.yaml` frames every picture and editing it breaks
comparability with everything already rendered; that `images` needs an
interpreter which can `import paraview.simple` and that this is **not**
`pvpython`; and the measured timings from Step 2.

Move these rows out of the "Deferred, with the hook already in place" table,
since they are no longer deferred:

- `Per-component forces, aero balance`
- `Full plane-cut image suite`
- `Side force and yaw moment in the record` — partially: `Cs` is now
  reported, `CmYaw` still is not.

- [ ] **Step 4: Close the risks in the spec**

In the spec's §11, replace the "v2412 specifics are unverified" bullet with
what was found, and fill in the measured sampling cost. Leave the `pvpython`
and `lambda2` range risks open — the first is environmental and the second
needs a real field to tune against.

- [ ] **Step 5: Commit**

```bash
git add docs/handbook.md docs/superpowers/specs/2026-09-01-postprocessing-reports-and-images-design.md
git commit -m "docs: what the report says, what frames the pictures, and what it cost"
```

---

## Notes for the executor

**Phase one is independently shippable.** Stop after Task 9 if the
spreadsheet work is what is urgent; nothing in Tasks 1-9 depends on Tasks
10-18.

**Three places where being clever will cost a day:**

1. **`postProcess -dict` merges, it does not replace** (Task 13). If
   `fieldAverage` is not disabled it will overwrite `pMean` and `UMean` with
   a one-sample average and the pictures will be of a single arbitrary
   iteration, with nothing saying so.
2. **`pv_render.py` must not import `simdev`** (Task 14). It runs under an
   interpreter that has ParaView and no venv.
3. **Never "reset camera to fit data"** (Tasks 12, 14). It silently rezooms
   between runs, which is the exact failure the fixed limits and fixed datum
   exist to prevent.
