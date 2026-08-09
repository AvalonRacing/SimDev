# OpenFOAM RC Car CFD Pipeline — Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenFOAM pipeline that takes a case config to validated force coefficients in four gated stages, proven end to end against the Ahmed body.

**Architecture:** A typed config resolves to a fully-explicit `CaseSpec`, which is validated, then rendered through Jinja templates into a self-contained OpenFOAM case. Four independently-invocable stages (`prepare` → `mesh` → `solve` → `post`) each end in a gate that must pass before the next runs. Almost all logic is pure Python testable without OpenFOAM installed; only the stage runners shell out.

**Tech Stack:** Python 3.11+, pydantic v2, Jinja2, numpy, pandas, matplotlib, trimesh. ESI OpenFOAM v2412 under WSL2 + Ubuntu.

**Spec:** `docs/superpowers/specs/2026-08-09-openfoam-rc-car-cfd-pipeline-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **OpenFOAM:** ESI v2412. The feature-extraction utility is `surfaceFeatures`, **not** `surfaceFeatureExtract`.
- **Decomposition:** 40 physical cores maximum, never 80 threads. `production` profile uses 40 ranks, `dev` uses 8.
- **Case location:** cases run on the WSL ext4 filesystem, never `/mnt/c` or `/mnt/d`.
- **No shared-append results.** Every run writes its own record. Never append to a common file.
- **No implicit defaults downstream.** Only `config/resolve.py` applies defaults. Every other module reads a fully-populated `CaseSpec`.
- **No staleness by directory inspection.** Never skip work because an output directory looks populated. Skip only on input-hash match, or `--force`.
- **`a_ref` is always stored as the full-vehicle frontal area.** The halved value is only ever obtained from `CaseSpec.a_ref_effective`. No other code may divide by two.
- **Reference values are config-derived only.** Never infer a physical quantity from a file path or directory name.
- **Python:** all public functions type-annotated. `from __future__ import annotations` at the top of every module.
- **Commit style:** conventional commits (`feat:`, `test:`, `fix:`, `docs:`, `chore:`).

---

## File Structure

```
pyproject.toml                     package + pytest config, console entry point
requirements.txt                   pinned runtime deps
docs/environment-setup.md          WSL2 + OpenFOAM v2412 install, ext4 placement

pipeline/simdev/
  __init__.py                      version
  geometry/roles.py                PatchRole enum + role trait table
  config/schema.py                 pydantic models, CaseSpec + derived properties
  config/profiles.py               dev/production + wall-treatment profile data
  config/resolve.py                layered merge → CaseSpec
  config/validate.py               cross-file consistency assertions
  geometry/ahmed.py                procedural Ahmed body STL writer
  geometry/stl.py                  STL info, watertightness, projected frontal area
  domain/base.py                   DomainBox dataclass, DomainBuilder protocol
  domain/box.py                    rectangular tunnel builder + blockage ratio
  render/render.py                 CaseSpec + DomainBox → case directory
  render/templates/                Jinja templates for every dictionary
  run/parsers.py                   checkMesh, snappy layers, forces, FOAM FATAL
  run/runner.py                    subprocess wrapper, log capture, exit discipline
  gates/mesh_quality.py            mesh gate
  gates/convergence.py             force-plateau gate
  gates/yplus.py                   y+ band gate
  stages/prepare.py                stage 1
  stages/mesh.py                   stage 2
  stages/solve.py                  stage 3
  stages/post.py                   stage 4
  report/results.py                per-run ResultRecord → JSON + CSV
  report/plots.py                  residual and force-history plots
  cli.py                           `simdev` entry point

cases/ahmed/config.yaml            the validation case

tests/
  conftest.py                      shared fixtures
  fixtures/logs/                   captured OpenFOAM log samples
  test_roles.py  test_schema.py  test_resolve.py  test_validate.py
  test_ahmed.py  test_stl.py  test_domain_box.py  test_render.py
  test_parsers.py  test_gates.py  test_runner.py  test_results.py
  test_smoke.py                    full-chain tiny case (requires OpenFOAM)
```

**Dependency order.** Tasks 1–14 need no OpenFOAM installation and are pure-Python testable. Tasks 15–18 shell out and require the environment from Task 1's documentation to be in place.

---

## Task 1: Project skeleton and environment documentation

**Files:**
- Create: `pyproject.toml`, `pipeline/simdev/__init__.py`, `docs/environment-setup.md`
- Modify: `requirements.txt`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable package `simdev` with `simdev.__version__: str`; `pytest` runnable from repo root; console script `simdev` registered (implemented in Task 17)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_package.py
from __future__ import annotations


def test_package_exposes_version() -> None:
    import simdev

    assert isinstance(simdev.__version__, str)
    assert simdev.__version__.count(".") >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev'`

- [ ] **Step 3: Create the package and build config**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "simdev"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "jinja2>=3.1",
    "numpy>=1.26",
    "pandas>=2.1",
    "matplotlib>=3.8",
    "trimesh>=4.0",
    "pyyaml>=6.0",
]

[project.scripts]
simdev = "simdev.cli:main"

[tool.setuptools]
package-dir = {"" = "pipeline"}

[tool.setuptools.packages.find]
where = ["pipeline"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "openfoam: requires a working OpenFOAM installation",
]
```

```python
# pipeline/simdev/__init__.py
from __future__ import annotations

__version__ = "0.1.0"
```

Create empty `__init__.py` in each subpackage directory: `config`, `geometry`, `domain`, `render`, `run`, `gates`, `stages`, `report`.

```
# requirements.txt
-e .
pytest>=8.0
```

- [ ] **Step 4: Install and run the test**

Run: `pip install -e . && python -m pytest tests/test_package.py -v`
Expected: PASS

Note: the console script will fail to run until Task 17 creates `simdev/cli.py`. That is expected; `pip install -e .` still succeeds because entry points are resolved lazily.

- [ ] **Step 5: Write the environment documentation**

Create `docs/environment-setup.md` with these sections, filled in as literal commands:

````markdown
# Environment Setup

## 1. WSL2 + Ubuntu

From an elevated PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot when prompted, then create your Linux user.

## 2. Move the WSL disk to D:

`C:` has ~330 GB free; CFD runs will exhaust it. Create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=100GB
processors=40
```

Then relocate the distro:

```powershell
wsl --shutdown
wsl --export Ubuntu D:\wsl\ubuntu-backup.tar
wsl --unregister Ubuntu
wsl --import Ubuntu D:\wsl\Ubuntu D:\wsl\ubuntu-backup.tar
```

## 3. OpenFOAM v2412 (ESI)

```bash
curl https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt-get install openfoam2412-default
echo "source /usr/lib/openfoam/openfoam2412/etc/bashrc" >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
simpleFoam -help | head -3
which surfaceFeatures snappyHexMesh checkMesh
```

`surfaceFeatures` must resolve. If only `surfaceFeatureExtract` exists you are on
an older build — the templates target v2412.

## 4. Case location

Run cases under `~/runs` on the ext4 filesystem. **Never** under `/mnt/c` or
`/mnt/d` — bind-mounted I/O collapses OpenFOAM performance.

## 5. Core count

Use 40 ranks maximum (physical cores). OpenFOAM is memory-bandwidth bound;
the 80 logical threads reduce throughput.
````

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt pipeline/simdev docs/environment-setup.md tests/test_package.py
git commit -m "chore: scaffold simdev package and document WSL/OpenFOAM setup"
```

---

## Task 2: Patch roles

**Files:**
- Create: `pipeline/simdev/geometry/roles.py`
- Test: `tests/test_roles.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `PatchRole` — str enum: `BODY`, `TYRE`, `GROUND`, `SYMMETRY`, `FARFIELD`, `INLET`, `OUTLET`, `MRF_ZONE` with values `"body"`, `"tyre"`, `"ground"`, `"symmetry"`, `"farfield"`, `"inlet"`, `"outlet"`, `"mrfZone"`
  - `RoleTraits` — frozen dataclass: `in_forces: bool`, `is_wall: bool`, `refinement: str` (`"high"` / `"medium"` / `"none"`)
  - `ROLE_TRAITS: dict[PatchRole, RoleTraits]`
  - `traits(role: PatchRole) -> RoleTraits`
  - `force_roles() -> frozenset[PatchRole]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_roles.py
from __future__ import annotations

import pytest

from simdev.geometry.roles import ROLE_TRAITS, PatchRole, force_roles, traits


def test_every_role_has_traits() -> None:
    assert set(ROLE_TRAITS) == set(PatchRole)


def test_only_body_and_tyre_contribute_to_forces() -> None:
    assert force_roles() == frozenset({PatchRole.BODY, PatchRole.TYRE})


def test_mrf_zone_is_excluded_from_forces() -> None:
    # Regression guard: the benchmark pipeline needed a name filter for this.
    assert traits(PatchRole.MRF_ZONE).in_forces is False


def test_ground_is_not_in_forces() -> None:
    assert traits(PatchRole.GROUND).in_forces is False


@pytest.mark.parametrize("role", [PatchRole.BODY, PatchRole.TYRE, PatchRole.GROUND])
def test_wall_roles_are_walls(role: PatchRole) -> None:
    assert traits(role).is_wall is True


@pytest.mark.parametrize("role", [PatchRole.SYMMETRY, PatchRole.INLET, PatchRole.OUTLET])
def test_non_wall_roles_are_not_walls(role: PatchRole) -> None:
    assert traits(role).is_wall is False


def test_role_values_match_openfoam_naming() -> None:
    assert PatchRole.MRF_ZONE.value == "mrfZone"
    assert PatchRole.BODY.value == "body"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_roles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.geometry.roles'`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/simdev/geometry/roles.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PatchRole(str, Enum):
    """Role of a surface. Drives BCs, refinement, force inclusion and MRF together."""

    BODY = "body"
    TYRE = "tyre"
    GROUND = "ground"
    SYMMETRY = "symmetry"
    FARFIELD = "farfield"
    INLET = "inlet"
    OUTLET = "outlet"
    MRF_ZONE = "mrfZone"


@dataclass(frozen=True)
class RoleTraits:
    in_forces: bool
    is_wall: bool
    refinement: str


ROLE_TRAITS: dict[PatchRole, RoleTraits] = {
    PatchRole.BODY: RoleTraits(in_forces=True, is_wall=True, refinement="high"),
    PatchRole.TYRE: RoleTraits(in_forces=True, is_wall=True, refinement="high"),
    PatchRole.GROUND: RoleTraits(in_forces=False, is_wall=True, refinement="medium"),
    PatchRole.SYMMETRY: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
    PatchRole.FARFIELD: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
    PatchRole.INLET: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
    PatchRole.OUTLET: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
    # An MRF cell zone is not a patch and must never enter force integration.
    PatchRole.MRF_ZONE: RoleTraits(in_forces=False, is_wall=False, refinement="none"),
}


def traits(role: PatchRole) -> RoleTraits:
    return ROLE_TRAITS[role]


def force_roles() -> frozenset[PatchRole]:
    """Roles whose patches are integrated for forces."""
    return frozenset(r for r, t in ROLE_TRAITS.items() if t.in_forces)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_roles.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/geometry/roles.py tests/test_roles.py
git commit -m "feat: add patch role model with force-inclusion traits"
```

---

## Task 3: Config schema and CaseSpec

**Files:**
- Create: `pipeline/simdev/config/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: `PatchRole` from Task 2
- Produces:
  - Enums: `WallTreatment` (`LOW_Y_PLUS="low_y_plus"`, `HIGH_Y_PLUS="high_y_plus"`, `SPALDING="spalding"`), `Mode` (`STRAIGHT="straight"`, `CORNERING="cornering"`), `GroundMotion` (`STATIC="static"`, `MOVING="moving"`)
  - Models: `FlowConfig`, `GroundConfig`, `PhysicsConfig`, `DomainConfig`, `MeshConfig`, `ForcesConfig`, `SolveConfig`, `PostConfig`, `PatchSpec`, `AhmedParams`, `GeometryConfig`
  - `CaseSpec` with properties `half_model: bool`, `a_ref_effective: float`, `omega_rotation: float | None`, and methods `patches_with_role(role) -> list[str]`, `spec_hash() -> str`

`AhmedParams` is defined here rather than in `geometry/ahmed.py` so the schema has no dependency on the geometry module.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
from __future__ import annotations

import pytest

from simdev.config.schema import CaseSpec, Mode
from simdev.geometry.roles import PatchRole


def _spec(**physics_overrides: object) -> CaseSpec:
    physics = {"wall_treatment": "high_y_plus"}
    physics.update(physics_overrides)
    return CaseSpec.model_validate(
        {
            "name": "t",
            "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
            "ground": {"motion": "static"},
            "physics": physics,
            "domain": {},
            "mesh": {
                "base_cell_size": 0.04,
                "surface_refinement_min": 4,
                "surface_refinement_max": 5,
                "n_layers": 6,
                "first_layer_thickness": 3.0e-4,
            },
            "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
            "solve": {"max_iterations": 2000, "n_ranks": 8},
            "post": {"yplus_min": 30.0, "yplus_max": 300.0},
            "geometry": {
                "kind": "ahmed",
                "ahmed": {},
                "patches": [
                    {"name": "body", "role": "body"},
                    {"name": "ground", "role": "ground"},
                    {"name": "symm", "role": "symmetry"},
                ],
            },
        }
    )


def test_straight_zero_yaw_is_half_model() -> None:
    assert _spec().half_model is True


def test_yaw_alone_breaks_symmetry() -> None:
    # Yaw kills symmetry even in straight-line mode.
    assert _spec(yaw_deg=5.0).half_model is False


def test_cornering_is_never_half_model() -> None:
    spec = _spec(mode="cornering", corner_radius=8.0)
    assert spec.half_model is False


def test_a_ref_is_halved_only_for_half_model() -> None:
    assert _spec().a_ref_effective == pytest.approx(0.056)
    assert _spec(yaw_deg=5.0).a_ref_effective == pytest.approx(0.112)


def test_omega_rotation_from_corner_radius() -> None:
    spec = _spec(mode="cornering", corner_radius=8.0)
    assert spec.omega_rotation == pytest.approx(5.0)


def test_omega_rotation_is_none_for_straight() -> None:
    assert _spec().omega_rotation is None


def test_patches_with_role() -> None:
    assert _spec().patches_with_role(PatchRole.BODY) == ["body"]


def test_spec_hash_is_stable_and_sensitive() -> None:
    assert _spec().spec_hash() == _spec().spec_hash()
    assert _spec().spec_hash() != _spec(yaw_deg=5.0).spec_hash()


def test_mode_enum_values() -> None:
    assert Mode.CORNERING.value == "cornering"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.config.schema'`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/simdev/config/schema.py
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from simdev.geometry.roles import PatchRole


class WallTreatment(str, Enum):
    LOW_Y_PLUS = "low_y_plus"
    HIGH_Y_PLUS = "high_y_plus"
    SPALDING = "spalding"


class Mode(str, Enum):
    STRAIGHT = "straight"
    CORNERING = "cornering"


class GroundMotion(str, Enum):
    STATIC = "static"
    MOVING = "moving"


class FlowConfig(BaseModel):
    u_inf: float = Field(gt=0.0)
    nu: float = 1.5e-5
    rho: float = 1.225
    turbulence_intensity: float = 0.01
    turbulence_length_scale: float = Field(gt=0.0)


class GroundConfig(BaseModel):
    motion: GroundMotion


class PhysicsConfig(BaseModel):
    turbulence_model: Literal["kOmegaSST", "kOmegaSSTLM"] = "kOmegaSST"
    wall_treatment: WallTreatment
    mode: Mode = Mode.STRAIGHT
    yaw_deg: float = 0.0
    corner_radius: float | None = None


class DomainConfig(BaseModel):
    kind: Literal["box", "annulus"] = "box"
    upstream_lengths: float = 5.0
    downstream_lengths: float = 10.0
    half_width_lengths: float = 3.0
    height_lengths: float = 3.0
    max_blockage: float = 0.01


class MeshConfig(BaseModel):
    base_cell_size: float = Field(gt=0.0)
    surface_refinement_min: int
    surface_refinement_max: int
    n_layers: int
    first_layer_thickness: float = Field(gt=0.0)
    expansion_ratio: float = 1.2
    min_layer_coverage: float = 0.7
    max_non_ortho: float = 70.0
    max_skewness: float = 4.0


class ForcesConfig(BaseModel):
    """a_ref_full is ALWAYS the full-vehicle frontal area.

    The halved value is only ever obtained from CaseSpec.a_ref_effective.
    """

    a_ref_full: float = Field(gt=0.0)
    l_ref: float = Field(gt=0.0)
    c_of_r: tuple[float, float, float] = (0.0, 0.0, 0.0)


class SolveConfig(BaseModel):
    max_iterations: int = Field(gt=0)
    n_ranks: int = Field(gt=0)
    plateau_window: int = 200
    plateau_tol: float = 0.002
    residual_tol: float = 1.0e-4


class PostConfig(BaseModel):
    yplus_min: float
    yplus_max: float
    max_fraction_outside: float = 0.1


class PatchSpec(BaseModel):
    name: str
    role: PatchRole


class AhmedParams(BaseModel):
    length: float = 1.044
    width: float = 0.389
    height: float = 0.288
    slant_angle_deg: float = 35.0
    slant_length: float = 0.222
    nose_radius: float = 0.100
    ground_clearance: float = 0.050
    stilt_diameter: float = 0.030
    include_stilts: bool = True


class GeometryConfig(BaseModel):
    kind: Literal["ahmed", "stl"]
    ahmed: AhmedParams | None = None
    stl_dir: str | None = None
    patches: list[PatchSpec]


class CaseSpec(BaseModel):
    """Fully resolved case. Every value explicit; no downstream defaults."""

    name: str
    flow: FlowConfig
    ground: GroundConfig
    physics: PhysicsConfig
    domain: DomainConfig
    mesh: MeshConfig
    forces: ForcesConfig
    solve: SolveConfig
    post: PostConfig
    geometry: GeometryConfig

    @property
    def half_model(self) -> bool:
        """Symmetry is derived, never set. Yaw alone breaks it."""
        return self.physics.mode is Mode.STRAIGHT and self.physics.yaw_deg == 0.0

    @property
    def a_ref_effective(self) -> float:
        if self.half_model:
            return self.forces.a_ref_full / 2.0
        return self.forces.a_ref_full

    @property
    def omega_rotation(self) -> float | None:
        if self.physics.mode is Mode.CORNERING and self.physics.corner_radius:
            return self.flow.u_inf / self.physics.corner_radius
        return None

    def patches_with_role(self, role: PatchRole) -> list[str]:
        return [p.name for p in self.geometry.patches if p.role is role]

    def spec_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schema.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/config/schema.py tests/test_schema.py
git commit -m "feat: add typed CaseSpec with derived symmetry and reference area"
```

---

## Task 4: Profiles and layered config resolution

**Files:**
- Create: `pipeline/simdev/config/profiles.py`, `pipeline/simdev/config/resolve.py`
- Test: `tests/test_resolve.py`

**Interfaces:**
- Consumes: `CaseSpec`, `WallTreatment` from Task 3
- Produces:
  - `DEFAULTS: dict[str, Any]`
  - `RESOLUTION_PROFILES: dict[str, dict]` with keys `"dev"` and `"production"`
  - `WALL_PROFILES: dict[WallTreatment, dict]`
  - `deep_merge(base: dict, over: dict) -> dict`
  - `resolve(case: dict, profile: str, wall_treatment: str | None = None, overrides: dict | None = None) -> CaseSpec`
  - `load_case(path: Path, profile: str, wall_treatment: str | None = None, overrides: dict | None = None) -> CaseSpec`

Merge order is **defaults, then resolution profile, then wall profile, then case file, then CLI overrides**. Later layers win.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolve.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simdev.config.resolve import deep_merge, load_case, resolve
from simdev.config.schema import WallTreatment

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
        "ahmed": {},
        "patches": [{"name": "body", "role": "body"}],
    },
}


def test_deep_merge_overrides_nested_leaves_only() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 9}}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": 9}, "b": 3}


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


def test_dev_profile_uses_eight_ranks() -> None:
    assert resolve(CASE, profile="dev").solve.n_ranks == 8


def test_production_profile_uses_forty_ranks() -> None:
    # 40 physical cores, never the 80 logical threads.
    assert resolve(CASE, profile="production").solve.n_ranks == 40


def test_dev_profile_is_coarser_than_production() -> None:
    dev = resolve(CASE, profile="dev")
    prod = resolve(CASE, profile="production")
    assert dev.mesh.base_cell_size > prod.mesh.base_cell_size
    assert dev.solve.max_iterations < prod.solve.max_iterations


def test_wall_treatment_sets_yplus_band() -> None:
    high = resolve(CASE, profile="dev", wall_treatment="high_y_plus")
    low = resolve(CASE, profile="dev", wall_treatment="low_y_plus")
    assert (high.post.yplus_min, high.post.yplus_max) == (30.0, 300.0)
    assert low.post.yplus_max <= 5.0
    assert low.mesh.first_layer_thickness < high.mesh.first_layer_thickness
    assert low.mesh.n_layers > high.mesh.n_layers


def test_case_file_beats_profile() -> None:
    case = deep_merge(CASE, {"solve": {"n_ranks": 4}})
    assert resolve(case, profile="production").solve.n_ranks == 4


def test_cli_overrides_beat_case_file() -> None:
    case = deep_merge(CASE, {"solve": {"n_ranks": 4}})
    spec = resolve(case, profile="dev", overrides={"solve": {"n_ranks": 2}})
    assert spec.solve.n_ranks == 2


def test_load_case_reads_yaml(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    assert load_case(p, profile="dev").name == "ahmed"


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError):
        resolve(CASE, profile="nonexistent")


def test_resolved_spec_has_no_missing_values() -> None:
    spec = resolve(CASE, profile="dev")
    assert spec.physics.wall_treatment in set(WallTreatment)
    assert spec.mesh.n_layers > 0
    assert spec.post.yplus_max > spec.post.yplus_min
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_resolve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.config.resolve'`

- [ ] **Step 3: Write the profile data**

```python
# pipeline/simdev/config/profiles.py
from __future__ import annotations

from typing import Any

from simdev.config.schema import WallTreatment

# Baseline applied before every other layer.
DEFAULTS: dict[str, Any] = {
    "physics": {
        "turbulence_model": "kOmegaSST",
        "wall_treatment": "high_y_plus",
        "mode": "straight",
        "yaw_deg": 0.0,
    },
    "domain": {
        "kind": "box",
        "upstream_lengths": 5.0,
        "downstream_lengths": 10.0,
        "half_width_lengths": 3.0,
        "height_lengths": 3.0,
        "max_blockage": 0.01,
    },
    "solve": {"plateau_window": 200, "plateau_tol": 0.002, "residual_tol": 1.0e-4},
    "post": {"max_fraction_outside": 0.1},
}

RESOLUTION_PROFILES: dict[str, dict[str, Any]] = {
    # Deliberately very coarse: pipeline development, minutes not hours.
    "dev": {
        "mesh": {
            "base_cell_size": 0.12,
            "surface_refinement_min": 2,
            "surface_refinement_max": 3,
        },
        "solve": {"max_iterations": 300, "n_ranks": 8, "plateau_window": 50},
    },
    "production": {
        "mesh": {
            "base_cell_size": 0.03,
            "surface_refinement_min": 5,
            "surface_refinement_max": 6,
        },
        # 40 physical cores. Never 80 threads: OpenFOAM is bandwidth bound.
        "solve": {"max_iterations": 4000, "n_ranks": 40},
    },
}

WALL_PROFILES: dict[WallTreatment, dict[str, Any]] = {
    WallTreatment.HIGH_Y_PLUS: {
        # 1 mm puts the cell centre at y+ ~= 50 for the Ahmed condition
        # (40 m/s, L = 1.044 m). Cases override this for other conditions.
        "mesh": {"n_layers": 7, "first_layer_thickness": 1.0e-3, "expansion_ratio": 1.2},
        "post": {"yplus_min": 30.0, "yplus_max": 300.0},
    },
    WallTreatment.LOW_Y_PLUS: {
        "mesh": {"n_layers": 18, "first_layer_thickness": 2.0e-5, "expansion_ratio": 1.15},
        "post": {"yplus_min": 0.0, "yplus_max": 5.0},
    },
    WallTreatment.SPALDING: {
        "mesh": {"n_layers": 10, "first_layer_thickness": 1.0e-4, "expansion_ratio": 1.2},
        "post": {"yplus_min": 0.0, "yplus_max": 300.0},
    },
}
```

- [ ] **Step 4: Write the resolver**

```python
# pipeline/simdev/config/resolve.py
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from simdev.config.profiles import DEFAULTS, RESOLUTION_PROFILES, WALL_PROFILES
from simdev.config.schema import CaseSpec, WallTreatment


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `over` onto `base`. Neither input is mutated."""
    result = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve(
    case: dict[str, Any],
    profile: str,
    wall_treatment: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> CaseSpec:
    """Merge defaults, resolution profile, wall profile, case, CLI overrides."""
    if profile not in RESOLUTION_PROFILES:
        raise KeyError(
            f"unknown profile {profile!r}; expected one of {sorted(RESOLUTION_PROFILES)}"
        )

    merged = deep_merge(DEFAULTS, RESOLUTION_PROFILES[profile])

    treatment_name = (
        wall_treatment
        or case.get("physics", {}).get("wall_treatment")
        or merged["physics"]["wall_treatment"]
    )
    treatment = WallTreatment(treatment_name)
    merged = deep_merge(merged, WALL_PROFILES[treatment])
    merged = deep_merge(merged, {"physics": {"wall_treatment": treatment.value}})

    merged = deep_merge(merged, case)
    if overrides:
        merged = deep_merge(merged, overrides)

    return CaseSpec.model_validate(merged)


def load_case(
    path: Path,
    profile: str,
    wall_treatment: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> CaseSpec:
    case = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return resolve(case, profile, wall_treatment, overrides)
```

Ordering subtlety worth understanding: the wall profile is applied *before* the case file, so a case may override individual mesh numbers while inheriting the rest of its treatment. But a `wall_treatment` passed on the CLI wins over the case file when selecting *which* profile applies — that is why `treatment_name` reads the CLI value first.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_resolve.py -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/config/profiles.py pipeline/simdev/config/resolve.py tests/test_resolve.py
git commit -m "feat: add layered config resolution with resolution and wall profiles"
```

---

## Task 5: Cross-file validation assertions

**Files:**
- Create: `pipeline/simdev/config/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `CaseSpec`, `Mode`, `WallTreatment`, `GroundMotion` from Task 3; `PatchRole` from Task 2
- Produces:
  - `ValidationError(Exception)` with attribute `errors: list[str]`
  - `estimate_y_plus(spec: CaseSpec) -> float`
  - `validate(spec: CaseSpec) -> list[str]` — raises `ValidationError` if any error; returns warnings

This is the layer that makes the benchmark pipeline's recurring bugs impossible. It runs in under a second, instead of surfacing as a wrong number three hours later.

`estimate_y_plus` uses the flat-plate correlation `Cf = 0.058 Re^-0.2`, `u_tau = u_inf*sqrt(Cf/2)`, `y_plus = (t/2)*u_tau/nu` where `t` is the first layer thickness. It sizes the mesh; it is not a prediction, which is why the band check tolerates a factor of five before erroring.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate.py
from __future__ import annotations

import copy

import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.config.validate import ValidationError, estimate_y_plus, validate

BASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
        "ahmed": {},
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


def _spec(overrides: dict | None = None, **kw: object):
    case = deep_merge(BASE, overrides or {})
    return resolve(case, profile="dev", **kw)


def test_valid_case_passes() -> None:
    assert validate(_spec()) == []


def test_half_model_without_symmetry_patch_is_rejected() -> None:
    case = copy.deepcopy(BASE)
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("symmetry" in e for e in exc.value.errors)


def test_full_model_with_symmetry_patch_is_rejected() -> None:
    # Yaw breaks symmetry, so a symmetry patch is now an error.
    with pytest.raises(ValidationError) as exc:
        validate(_spec({"physics": {"yaw_deg": 5.0}}))
    assert any("symmetry" in e for e in exc.value.errors)


def test_cornering_half_model_is_unconstructable() -> None:
    case = deep_merge(BASE, {"physics": {"mode": "cornering", "corner_radius": 8.0}})
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    # Symmetry patch present on a cornering case, and box domain unsupported.
    assert exc.value.errors


def test_cornering_without_corner_radius_is_rejected() -> None:
    case = copy.deepcopy(BASE)
    case["physics"] = {"mode": "cornering"}
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("corner_radius" in e for e in exc.value.errors)


def test_cornering_requires_annulus_domain() -> None:
    case = copy.deepcopy(BASE)
    case["physics"] = {"mode": "cornering", "corner_radius": 8.0}
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("annulus" in e for e in exc.value.errors)


def test_missing_inlet_is_rejected() -> None:
    case = copy.deepcopy(BASE)
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "inlet"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("inlet" in e for e in exc.value.errors)


def test_no_body_patch_is_rejected() -> None:
    case = copy.deepcopy(BASE)
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "body"
    ]
    with pytest.raises(ValidationError) as exc:
        validate(resolve(case, profile="dev"))
    assert any("force" in e or "body" in e for e in exc.value.errors)


def test_wall_treatment_mismatch_is_rejected() -> None:
    # low_y_plus band with a first layer sized for wall functions.
    spec = _spec(
        {"mesh": {"first_layer_thickness": 3.0e-3}}, wall_treatment="low_y_plus"
    )
    with pytest.raises(ValidationError) as exc:
        validate(spec)
    assert any("y+" in e for e in exc.value.errors)


def test_rank_count_above_physical_cores_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        validate(_spec({"solve": {"n_ranks": 80}}))
    assert any("rank" in e.lower() or "core" in e.lower() for e in exc.value.errors)


def test_refinement_range_inverted_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        validate(
            _spec({"mesh": {"surface_refinement_min": 6, "surface_refinement_max": 3}})
        )
    assert any("refinement" in e for e in exc.value.errors)


def test_static_ground_on_a_vehicle_case_warns() -> None:
    case = copy.deepcopy(BASE)
    case["geometry"]["kind"] = "stl"
    case["geometry"]["stl_dir"] = "geom"
    case["geometry"]["ahmed"] = None
    warnings = validate(resolve(case, profile="dev"))
    assert any("static ground" in w for w in warnings)


def test_estimate_y_plus_is_in_the_expected_band_for_high_y_plus() -> None:
    # Ahmed at 40 m/s with a 0.3 mm first layer should land in wall-function range.
    y = estimate_y_plus(_spec())
    assert 30.0 <= y <= 300.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.config.validate'`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/simdev/config/validate.py
from __future__ import annotations

from simdev.config.schema import CaseSpec, GroundMotion, Mode
from simdev.geometry.roles import PatchRole, force_roles

MAX_PHYSICAL_CORES = 40
Y_PLUS_ERROR_FACTOR = 5.0


class ValidationError(Exception):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def estimate_y_plus(spec: CaseSpec) -> float:
    """Flat-plate estimate of y+ at the first cell centre.

    Cf = 0.058 Re^-0.2, u_tau = u_inf sqrt(Cf/2), y+ = y u_tau / nu.
    Sizes the mesh; not a prediction.
    """
    re = spec.flow.u_inf * spec.forces.l_ref / spec.flow.nu
    cf = 0.058 * re ** -0.2
    u_tau = spec.flow.u_inf * (cf / 2.0) ** 0.5
    y_centre = spec.mesh.first_layer_thickness / 2.0
    return y_centre * u_tau / spec.flow.nu


def validate(spec: CaseSpec) -> list[str]:
    """Raise ValidationError on inconsistency. Return non-fatal warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    roles = [p.role for p in spec.geometry.patches]
    has_symmetry = PatchRole.SYMMETRY in roles

    # --- symmetry is derived; the patch set must agree with it -------------
    if spec.half_model and not has_symmetry:
        errors.append(
            "half_model is True but no patch has role 'symmetry'; "
            "a_ref would be halved for a model with no symmetry plane"
        )
    if not spec.half_model and has_symmetry:
        errors.append(
            "a 'symmetry' patch is present but the model is not a half model "
            f"(mode={spec.physics.mode.value}, yaw={spec.physics.yaw_deg}); "
            "yaw alone breaks symmetry"
        )

    # --- cornering -------------------------------------------------------
    if spec.physics.mode is Mode.CORNERING:
        if spec.physics.corner_radius is None:
            errors.append("mode is 'cornering' but corner_radius is not set")
        if spec.domain.kind != "annulus":
            errors.append(
                "mode is 'cornering' but domain.kind is "
                f"'{spec.domain.kind}'; cornering requires the 'annulus' domain"
            )

    # --- patch set -------------------------------------------------------
    for required in (PatchRole.INLET, PatchRole.OUTLET, PatchRole.GROUND):
        if required not in roles:
            errors.append(f"no patch has role '{required.value}'")
    if not any(r in force_roles() for r in roles):
        errors.append(
            "no patch contributes to force integration; expected at least one "
            "patch with role 'body' or 'tyre'"
        )

    names = [p.name for p in spec.geometry.patches]
    if len(names) != len(set(names)):
        errors.append("duplicate patch names")

    # --- wall treatment vs mesh ------------------------------------------
    y_plus = estimate_y_plus(spec)
    lo, hi = spec.post.yplus_min, spec.post.yplus_max
    if y_plus > hi * Y_PLUS_ERROR_FACTOR or (lo > 0 and y_plus < lo / Y_PLUS_ERROR_FACTOR):
        errors.append(
            f"first_layer_thickness implies y+ ~= {y_plus:.1f}, far outside the "
            f"[{lo}, {hi}] band for wall treatment "
            f"'{spec.physics.wall_treatment.value}'"
        )
    elif not (lo <= y_plus <= hi):
        warnings.append(
            f"estimated y+ {y_plus:.1f} is outside the [{lo}, {hi}] band; "
            "the post-run y+ gate will confirm"
        )

    # --- mesh ------------------------------------------------------------
    if spec.mesh.surface_refinement_min > spec.mesh.surface_refinement_max:
        errors.append(
            f"surface refinement range inverted: min "
            f"{spec.mesh.surface_refinement_min} > max "
            f"{spec.mesh.surface_refinement_max}"
        )

    # --- parallel --------------------------------------------------------
    if spec.solve.n_ranks > MAX_PHYSICAL_CORES:
        errors.append(
            f"n_ranks {spec.solve.n_ranks} exceeds {MAX_PHYSICAL_CORES} physical "
            "cores; OpenFOAM is memory-bandwidth bound and hyperthreading "
            "reduces throughput"
        )

    # --- ground ----------------------------------------------------------
    if spec.geometry.kind == "stl" and spec.ground.motion is GroundMotion.STATIC:
        warnings.append(
            "static ground under a vehicle case: ground-effect aerodynamics "
            "will be wrong unless you are deliberately matching a fixed-floor "
            "experiment"
        )

    if errors:
        raise ValidationError(errors)
    return warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validate.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/config/validate.py tests/test_validate.py
git commit -m "feat: add generation-time cross-file validation assertions"
```

---

## Task 6: Procedural Ahmed body geometry

**Files:**
- Create: `pipeline/simdev/geometry/ahmed.py`
- Test: `tests/test_ahmed.py`

**Interfaces:**
- Consumes: `AhmedParams` from Task 3
- Produces:
  - `build_body(params: AhmedParams) -> trimesh.Trimesh`
  - `build_stilts(params: AhmedParams) -> trimesh.Trimesh`
  - `write_ahmed_stl(params: AhmedParams, out_dir: Path) -> dict[str, Path]` — returns `{"body": path}` plus `{"stilts": path}` when `params.include_stilts`
  - `frontal_area(params: AhmedParams) -> float` — returns `width * height`

**Coordinate convention** (used by every later task): x downstream from the nose, y spanwise, z up from the ground plane at z = 0. The body's underside sits at `z = ground_clearance`.

**The STL is always the full body.** Half models are produced by the *domain* restricting to y >= 0 with a `symmetry` patch (Task 8), not by cutting the geometry. This keeps the STL watertight.

**Documented approximation:** the nose is built by insetting each cross-section by the circular fillet profile `d(x) = R - sqrt(R^2 - (R-x)^2)` for `x < R`. This gives the correct 100 mm rounding in side view and in plan view, but the corner where the two curvatures meet is a lofted approximation rather than a true spherical fillet. If the acceptance test in Task 18 shows a front-separation discrepancy, this is the first thing to revisit.

**Reference note:** `frontal_area` returns `width * height` = 0.389 x 0.288 = 0.112032 m^2, which is the conventional Ahmed reference area. The stilts are excluded from it. Task 18 confirms this matches the published Cd's reference.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ahmed.py
from __future__ import annotations

import math
from pathlib import Path

import pytest

from simdev.config.schema import AhmedParams
from simdev.geometry.ahmed import build_body, frontal_area, write_ahmed_stl


def test_bounding_box_matches_parameters() -> None:
    p = AhmedParams()
    mesh = build_body(p)
    lo, hi = mesh.bounds
    assert hi[0] - lo[0] == pytest.approx(p.length, abs=1e-6)
    assert hi[1] - lo[1] == pytest.approx(p.width, abs=1e-6)
    assert lo[2] == pytest.approx(p.ground_clearance, abs=1e-6)
    assert hi[2] == pytest.approx(p.ground_clearance + p.height, abs=1e-6)


def test_body_is_watertight() -> None:
    assert build_body(AhmedParams()).is_watertight


def test_body_volume_is_positive_and_below_the_bounding_box() -> None:
    p = AhmedParams()
    mesh = build_body(p)
    box_volume = p.length * p.width * p.height
    assert 0.0 < mesh.volume < box_volume


def test_nose_is_inset_by_the_fillet_radius_at_the_tip() -> None:
    p = AhmedParams()
    mesh = build_body(p)
    tip = mesh.vertices[mesh.vertices[:, 0] < 1e-9]
    assert len(tip) > 0
    # At x = 0 the section is inset by the full nose radius on every side.
    assert tip[:, 1].max() == pytest.approx(p.width / 2 - p.nose_radius, abs=1e-6)


def test_slant_reduces_roof_height_by_the_right_amount() -> None:
    p = AhmedParams()
    mesh = build_body(p)
    theta = math.radians(p.slant_angle_deg)
    tail = mesh.vertices[mesh.vertices[:, 0] > p.length - 1e-9]
    expected = p.ground_clearance + p.height - p.slant_length * math.sin(theta)
    assert tail[:, 2].max() == pytest.approx(expected, abs=1e-6)


def test_frontal_area_is_width_times_height() -> None:
    p = AhmedParams()
    assert frontal_area(p) == pytest.approx(0.112032, abs=1e-6)


def test_write_produces_body_and_stilts(tmp_path: Path) -> None:
    paths = write_ahmed_stl(AhmedParams(include_stilts=True), tmp_path)
    assert set(paths) == {"body", "stilts"}
    assert paths["body"].exists() and paths["body"].stat().st_size > 0
    assert paths["stilts"].exists()


def test_write_omits_stilts_when_disabled(tmp_path: Path) -> None:
    paths = write_ahmed_stl(AhmedParams(include_stilts=False), tmp_path)
    assert set(paths) == {"body"}


def test_slant_angle_is_configurable() -> None:
    steep = build_body(AhmedParams(slant_angle_deg=35.0))
    shallow = build_body(AhmedParams(slant_angle_deg=25.0))
    # A steeper slant removes more material.
    assert steep.volume < shallow.volume
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ahmed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.geometry.ahmed'`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/simdev/geometry/ahmed.py
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

from simdev.config.schema import AhmedParams

N_NOSE_SECTIONS = 24


def _nose_inset(x: float, radius: float) -> float:
    """Circular fillet profile: d(x) = R - sqrt(R^2 - (R-x)^2) for x < R."""
    if x >= radius:
        return 0.0
    return radius - math.sqrt(max(radius * radius - (radius - x) ** 2, 0.0))


def _sections(params: AhmedParams) -> list[tuple[float, float, float, float]]:
    """Return (x, half_width, z_bottom, z_top) for each cross-section."""
    theta = math.radians(params.slant_angle_deg)
    slant_dx = params.slant_length * math.cos(theta)
    slant_dz = params.slant_length * math.sin(theta)
    slant_start = params.length - slant_dx

    xs: list[float] = [
        params.nose_radius * i / N_NOSE_SECTIONS for i in range(N_NOSE_SECTIONS)
    ]
    xs += [params.nose_radius, slant_start, params.length]
    xs = sorted(set(round(x, 12) for x in xs))

    out: list[tuple[float, float, float, float]] = []
    for x in xs:
        inset = _nose_inset(x, params.nose_radius)
        half_width = params.width / 2.0 - inset
        z_bottom = params.ground_clearance + inset
        z_top = params.ground_clearance + params.height - inset
        if x > slant_start:
            z_top -= (x - slant_start) * math.tan(theta)
        out.append((x, half_width, z_bottom, z_top))

    # Guard the arithmetic that produces the tail height.
    assert out[-1][3] == params.ground_clearance + params.height - slant_dz
    return out


def _ring(half_width: float, z_bottom: float, z_top: float, x: float) -> np.ndarray:
    """Four corners of a cross-section, counter-clockwise seen from -x."""
    return np.array(
        [
            [x, -half_width, z_bottom],
            [x, half_width, z_bottom],
            [x, half_width, z_top],
            [x, -half_width, z_top],
        ]
    )


def build_body(params: AhmedParams) -> trimesh.Trimesh:
    sections = _sections(params)
    rings = [_ring(hw, zb, zt, x) for x, hw, zb, zt in sections]

    vertices = np.vstack(rings)
    faces: list[list[int]] = []

    for i in range(len(rings) - 1):
        a = i * 4
        b = (i + 1) * 4
        for j in range(4):
            k = (j + 1) % 4
            faces.append([a + j, b + j, b + k])
            faces.append([a + j, b + k, a + k])

    # Caps. Front ring wound to face -x, rear ring to face +x.
    faces.append([0, 2, 1])
    faces.append([0, 3, 2])
    last = (len(rings) - 1) * 4
    faces.append([last, last + 1, last + 2])
    faces.append([last, last + 2, last + 3])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def build_stilts(params: AhmedParams) -> trimesh.Trimesh:
    """Four cylinders spanning the ground clearance gap."""
    radius = params.stilt_diameter / 2.0
    height = params.ground_clearance
    x_front = 0.25 * params.length
    x_rear = 0.75 * params.length
    y_off = params.width / 4.0

    parts: list[trimesh.Trimesh] = []
    for x in (x_front, x_rear):
        for y in (-y_off, y_off):
            cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
            cyl.apply_translation([x, y, height / 2.0])
            parts.append(cyl)
    return trimesh.util.concatenate(parts)


def frontal_area(params: AhmedParams) -> float:
    """Conventional Ahmed reference area: width x height, stilts excluded."""
    return params.width * params.height


def write_ahmed_stl(params: AhmedParams, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {"body": out_dir / "body.stl"}
    build_body(params).export(paths["body"])

    if params.include_stilts:
        paths["stilts"] = out_dir / "stilts.stl"
        build_stilts(params).export(paths["stilts"])

    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ahmed.py -v`
Expected: PASS (9 tests)

If `test_body_is_watertight` fails, the cap winding is wrong — check that `mesh.fix_normals()` ran and that `process=True` merged the shared ring vertices.

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/geometry/ahmed.py tests/test_ahmed.py
git commit -m "feat: add procedural Ahmed body geometry generator"
```

---

## Task 7: STL inspection and projected frontal area

**Files:**
- Create: `pipeline/simdev/geometry/stl.py`
- Modify: `pyproject.toml` — add `"shapely>=2.0"` to `dependencies`
- Test: `tests/test_stl.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (uses `build_body` from Task 6 in tests only)
- Produces:
  - `StlInfo` — frozen dataclass: `n_triangles: int`, `bounds_min: tuple[float, float, float]`, `bounds_max: tuple[float, float, float]`, `is_watertight: bool`, `surface_area: float`, with property `extent: tuple[float, float, float]`
  - `read_stl_info(path: Path) -> StlInfo`
  - `projected_frontal_area(mesh: trimesh.Trimesh, axis: int = 0) -> float`
  - `check_geometry(info: StlInfo, expected_length: float | None = None) -> list[str]` — returns warnings

`projected_frontal_area` uses `trimesh.Trimesh.projected`, which unions the projected triangles exactly via shapely rather than rasterising. That matters: a rasterised estimate would make the blockage assertion in Task 8 resolution-dependent.

`check_geometry` catches the two classic import faults — a mesh that is not watertight (snappyHexMesh will leak) and a mesh authored in millimetres when the pipeline works in metres.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stl.py
from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from simdev.config.schema import AhmedParams
from simdev.geometry.ahmed import build_body, write_ahmed_stl
from simdev.geometry.stl import (
    check_geometry,
    projected_frontal_area,
    read_stl_info,
)


def test_read_stl_info_round_trips(tmp_path: Path) -> None:
    p = AhmedParams()
    paths = write_ahmed_stl(p, tmp_path)
    info = read_stl_info(paths["body"])
    assert info.n_triangles > 0
    assert info.is_watertight is True
    assert info.extent[0] == pytest.approx(p.length, abs=1e-6)


def test_projected_frontal_area_of_a_unit_box() -> None:
    box = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    # Projected along x, the box presents 3 x 4.
    assert projected_frontal_area(box, axis=0) == pytest.approx(12.0, rel=1e-9)


def test_projected_frontal_area_along_other_axes() -> None:
    box = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    assert projected_frontal_area(box, axis=1) == pytest.approx(8.0, rel=1e-9)
    assert projected_frontal_area(box, axis=2) == pytest.approx(6.0, rel=1e-9)


def test_ahmed_projected_area_is_close_to_width_times_height() -> None:
    p = AhmedParams()
    area = projected_frontal_area(build_body(p), axis=0)
    # Slightly under width*height because the nose fillet insets the section.
    assert 0.9 * p.width * p.height < area <= p.width * p.height + 1e-9


def test_check_geometry_flags_non_watertight(tmp_path: Path) -> None:
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box.faces = box.faces[:-2]  # punch a hole
    path = tmp_path / "leaky.stl"
    box.export(path)
    warnings = check_geometry(read_stl_info(path))
    assert any("watertight" in w for w in warnings)


def test_check_geometry_flags_millimetre_units(tmp_path: Path) -> None:
    box = trimesh.creation.box(extents=[1044.0, 389.0, 288.0])
    path = tmp_path / "mm.stl"
    box.export(path)
    warnings = check_geometry(read_stl_info(path), expected_length=1.044)
    assert any("unit" in w.lower() or "mm" in w for w in warnings)


def test_check_geometry_is_quiet_on_a_good_mesh(tmp_path: Path) -> None:
    paths = write_ahmed_stl(AhmedParams(), tmp_path)
    assert check_geometry(read_stl_info(paths["body"]), expected_length=1.044) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.geometry.stl'`

- [ ] **Step 3: Add the shapely dependency**

In `pyproject.toml`, add to `dependencies`:

```toml
    "shapely>=2.0",
```

Run: `pip install -e .`

- [ ] **Step 4: Write minimal implementation**

```python
# pipeline/simdev/geometry/stl.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import trimesh

# A mesh whose longest edge is this many times the expected length is
# almost certainly authored in millimetres.
UNIT_SUSPICION_FACTOR = 100.0


@dataclass(frozen=True)
class StlInfo:
    n_triangles: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    is_watertight: bool
    surface_area: float

    @property
    def extent(self) -> tuple[float, float, float]:
        return (
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        )


def read_stl_info(path: Path) -> StlInfo:
    mesh = trimesh.load_mesh(Path(path), process=False)
    lo, hi = mesh.bounds
    return StlInfo(
        n_triangles=len(mesh.faces),
        bounds_min=(float(lo[0]), float(lo[1]), float(lo[2])),
        bounds_max=(float(hi[0]), float(hi[1]), float(hi[2])),
        is_watertight=bool(mesh.is_watertight),
        surface_area=float(mesh.area),
    )


def projected_frontal_area(mesh: trimesh.Trimesh, axis: int = 0) -> float:
    """Exact projected area along `axis`, via a shapely union of the triangles."""
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    projection = mesh.projected(normal)
    return float(sum(polygon.area for polygon in projection.polygons_full))


def check_geometry(info: StlInfo, expected_length: float | None = None) -> list[str]:
    warnings: list[str] = []

    if not info.is_watertight:
        warnings.append(
            "geometry is not watertight; snappyHexMesh will leak into the "
            "interior and the mesh will be unusable"
        )

    if expected_length is not None:
        longest = max(info.extent)
        if longest > expected_length * UNIT_SUSPICION_FACTOR:
            warnings.append(
                f"largest extent is {longest:.1f} against an expected "
                f"{expected_length:.3f}; unit mismatch, geometry is probably "
                "in millimetres"
            )

    return warnings
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_stl.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml pipeline/simdev/geometry/stl.py tests/test_stl.py
git commit -m "feat: add STL inspection and exact projected frontal area"
```

---

## Task 8: Box domain builder and blockage

**Files:**
- Create: `pipeline/simdev/domain/base.py`, `pipeline/simdev/domain/box.py`
- Test: `tests/test_domain_box.py`

**Interfaces:**
- Consumes: `CaseSpec` from Task 3; `PatchRole` from Task 2
- Produces:
  - `DomainBox` — frozen dataclass: `x_min, x_max, y_min, y_max, z_min, z_max: float`, `n_cells: tuple[int, int, int]`, `inlet: str`, `outlet: str`, `ground: str`, `farfield: str`, `symmetry: str | None`; properties `size: tuple[float, float, float]`, `cross_section_area: float`, `cell_count: int`
  - `DomainBuilder` — `Protocol` with `build(spec: CaseSpec, geom_bounds: tuple[Sequence[float], Sequence[float]]) -> DomainBox`
  - `BoxDomainBuilder` implementing it
  - `blockage_ratio(frontal_area: float, domain: DomainBox) -> float`
  - `check_blockage(frontal_area: float, domain: DomainBox, max_blockage: float) -> list[str]`

The ground plane is always `z = 0`; geometry sits above it. For a half model the domain spans `y in [0, +w]` with a `symmetry` patch at `y = 0`; for a full model it spans `y in [-w, +w]` and `symmetry` is `None`. The domain, not the STL, is what makes a model a half model.

Patch names come from the spec's declared patches by role, so `blockMeshDict` and `0/*` cannot disagree about what a face is called.

`blockage_ratio` takes the frontal area as an argument rather than reading `a_ref_effective`, because blockage must use the *true projected* area (Task 7) while `a_ref` is a reference convention that may deliberately exclude parts — for the Ahmed body it excludes the stilts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_domain_box.py
from __future__ import annotations

import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.domain.box import BoxDomainBuilder, blockage_ratio, check_blockage

BOUNDS = ((0.0, -0.1945, 0.05), (1.044, 0.1945, 0.338))

BASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
        "ahmed": {},
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


def _spec(overrides: dict | None = None):
    return resolve(deep_merge(BASE, overrides or {}), profile="dev")


def _full_model_spec():
    case = deep_merge(BASE, {"physics": {"yaw_deg": 5.0}})
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    return resolve(case, profile="dev")


def test_streamwise_extent_uses_body_lengths() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    length = 1.044
    assert d.x_min == pytest.approx(0.0 - 5.0 * length)
    assert d.x_max == pytest.approx(1.044 + 10.0 * length)


def test_ground_is_at_z_zero() -> None:
    assert BoxDomainBuilder().build(_spec(), BOUNDS).z_min == 0.0


def test_half_model_starts_at_the_symmetry_plane() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    assert d.y_min == 0.0
    assert d.symmetry == "symmetry"


def test_full_model_is_symmetric_about_y_zero_with_no_symmetry_patch() -> None:
    d = BoxDomainBuilder().build(_full_model_spec(), BOUNDS)
    assert d.y_min == pytest.approx(-d.y_max)
    assert d.symmetry is None


def test_patch_names_come_from_the_spec() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    assert (d.inlet, d.outlet, d.ground, d.farfield) == (
        "inlet",
        "outlet",
        "ground",
        "farfield",
    )


def test_cell_counts_are_positive_and_roughly_cubic() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    nx, ny, nz = d.n_cells
    assert min(nx, ny, nz) >= 1
    sx, sy, sz = d.size
    assert (sx / nx) == pytest.approx(sy / ny, rel=0.35)
    assert (sx / nx) == pytest.approx(sz / nz, rel=0.35)


def test_blockage_ratio_uses_the_domain_cross_section() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    ratio = blockage_ratio(0.056, d)
    assert ratio == pytest.approx(0.056 / d.cross_section_area)


def test_check_blockage_flags_a_cramped_domain() -> None:
    d = BoxDomainBuilder().build(
        _spec({"domain": {"half_width_lengths": 0.3, "height_lengths": 0.4}}), BOUNDS
    )
    assert check_blockage(0.056, d, max_blockage=0.01)


def test_check_blockage_is_quiet_on_a_roomy_domain() -> None:
    d = BoxDomainBuilder().build(_spec(), BOUNDS)
    assert check_blockage(0.056, d, max_blockage=0.01) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_domain_box.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.domain.box'`

- [ ] **Step 3: Write the domain interface**

```python
# pipeline/simdev/domain/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from simdev.config.schema import CaseSpec


@dataclass(frozen=True)
class DomainBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    n_cells: tuple[int, int, int]
    inlet: str
    outlet: str
    ground: str
    farfield: str
    symmetry: str | None

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )

    @property
    def cross_section_area(self) -> float:
        _, dy, dz = self.size
        return dy * dz

    @property
    def cell_count(self) -> int:
        nx, ny, nz = self.n_cells
        return nx * ny * nz


class DomainBuilder(Protocol):
    def build(
        self,
        spec: CaseSpec,
        geom_bounds: tuple[Sequence[float], Sequence[float]],
    ) -> DomainBox: ...
```

- [ ] **Step 4: Write the box builder**

```python
# pipeline/simdev/domain/box.py
from __future__ import annotations

from typing import Sequence

from simdev.config.schema import CaseSpec
from simdev.domain.base import DomainBox
from simdev.geometry.roles import PatchRole


def _only(spec: CaseSpec, role: PatchRole) -> str:
    names = spec.patches_with_role(role)
    if len(names) != 1:
        raise ValueError(
            f"box domain needs exactly one '{role.value}' patch, found {names}"
        )
    return names[0]


class BoxDomainBuilder:
    """Rectangular virtual wind tunnel. Ground plane at z = 0."""

    def build(
        self,
        spec: CaseSpec,
        geom_bounds: tuple[Sequence[float], Sequence[float]],
    ) -> DomainBox:
        lo, hi = geom_bounds
        length = hi[0] - lo[0]
        d = spec.domain

        x_min = lo[0] - d.upstream_lengths * length
        x_max = hi[0] + d.downstream_lengths * length
        z_min = 0.0
        z_max = d.height_lengths * length
        y_max = d.half_width_lengths * length
        y_min = 0.0 if spec.half_model else -y_max

        base = spec.mesh.base_cell_size
        n_cells = (
            max(1, round((x_max - x_min) / base)),
            max(1, round((y_max - y_min) / base)),
            max(1, round((z_max - z_min) / base)),
        )

        symmetry = _only(spec, PatchRole.SYMMETRY) if spec.half_model else None

        return DomainBox(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            n_cells=n_cells,
            inlet=_only(spec, PatchRole.INLET),
            outlet=_only(spec, PatchRole.OUTLET),
            ground=_only(spec, PatchRole.GROUND),
            farfield=_only(spec, PatchRole.FARFIELD),
            symmetry=symmetry,
        )


def blockage_ratio(frontal_area: float, domain: DomainBox) -> float:
    """Frontal area over domain cross-section.

    Pass the *true projected* area, not a_ref: a_ref is a reference
    convention that may deliberately exclude parts.
    """
    return frontal_area / domain.cross_section_area


def check_blockage(
    frontal_area: float, domain: DomainBox, max_blockage: float
) -> list[str]:
    ratio = blockage_ratio(frontal_area, domain)
    if ratio > max_blockage:
        return [
            f"blockage ratio {ratio:.3%} exceeds the {max_blockage:.3%} limit; "
            "the domain is too cramped and will contaminate the pressure field"
        ]
    return []
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_domain_box.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/domain tests/test_domain_box.py
git commit -m "feat: add box domain builder with blockage assertion"
```

---

## Task 9: Render infrastructure and meshing dictionaries

**Files:**
- Create: `pipeline/simdev/render/context.py`, `pipeline/simdev/render/render.py`
- Create: `pipeline/simdev/render/templates/foam_header.jinja`, `blockMeshDict.jinja`, `surfaceFeaturesDict.jinja`, `snappyHexMeshDict.jinja`, `decomposeParDict.jinja`
- Modify: `pyproject.toml` — package the templates
- Test: `tests/test_render_mesh.py`

**Interfaces:**
- Consumes: `CaseSpec` (Task 3), `DomainBox` (Task 8), `PatchRole` (Task 2)
- Produces:
  - `build_context(spec: CaseSpec, domain: DomainBox, geometry_files: dict[str, Path]) -> dict[str, Any]`
  - `WALL_FUNCTIONS: dict[WallTreatment, dict[str, str]]` — keys `"nut"`, `"k"`, `"omega"`
  - `inlet_turbulence(spec: CaseSpec) -> tuple[float, float]` — returns `(k, omega)`
  - `location_in_mesh(domain: DomainBox) -> tuple[float, float, float]`
  - `render_mesh_dicts(spec, domain, geometry_files, out_dir: Path) -> None`
  - `env() -> jinja2.Environment`

**Template discipline (enforced by review):** templates may loop over already-resolved structures and interpolate values. They may **not** branch on raw config, compute physical quantities, or apply defaults. Everything derived lives in `context.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render_mesh.py
from __future__ import annotations

from pathlib import Path

import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.config.schema import WallTreatment
from simdev.domain.box import BoxDomainBuilder
from simdev.render.context import (
    WALL_FUNCTIONS,
    build_context,
    inlet_turbulence,
    location_in_mesh,
)
from simdev.render.render import render_mesh_dicts

BOUNDS = ((0.0, -0.1945, 0.05), (1.044, 0.1945, 0.338))

BASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
        "ahmed": {},
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


def _spec(overrides: dict | None = None, **kw: object):
    return resolve(deep_merge(BASE, overrides or {}), profile="dev", **kw)


def _render(tmp_path: Path, spec=None) -> Path:
    spec = spec or _spec()
    domain = BoxDomainBuilder().build(spec, BOUNDS)
    files = {"body": Path("constant/triSurface/body.stl")}
    render_mesh_dicts(spec, domain, files, tmp_path)
    return tmp_path


def test_inlet_turbulence_matches_the_standard_relations() -> None:
    spec = _spec()
    k, omega = inlet_turbulence(spec)
    expected_k = 1.5 * (0.01 * 40.0) ** 2
    assert k == pytest.approx(expected_k)
    assert omega == pytest.approx(k**0.5 / (0.09**0.25 * 0.01))


def test_wall_functions_differ_by_treatment() -> None:
    assert WALL_FUNCTIONS[WallTreatment.HIGH_Y_PLUS]["nut"] == "nutkWallFunction"
    assert WALL_FUNCTIONS[WallTreatment.LOW_Y_PLUS]["nut"] == "nutLowReWallFunction"
    assert WALL_FUNCTIONS[WallTreatment.SPALDING]["nut"] == "nutUSpaldingWallFunction"
    # omegaWallFunction blends and is correct under every treatment.
    for t in WallTreatment:
        assert WALL_FUNCTIONS[t]["omega"] == "omegaWallFunction"


def test_location_in_mesh_is_upstream_and_inside_a_half_domain() -> None:
    domain = BoxDomainBuilder().build(_spec(), BOUNDS)
    x, y, z = location_in_mesh(domain)
    assert domain.x_min < x < 0.0
    assert domain.y_min < y < domain.y_max
    assert y > 0.0
    assert domain.z_min < z < domain.z_max


def test_context_exposes_effective_reference_area() -> None:
    spec = _spec()
    domain = BoxDomainBuilder().build(spec, BOUNDS)
    ctx = build_context(spec, domain, {"body": Path("body.stl")})
    assert ctx["a_ref"] == pytest.approx(0.056)


def test_blockmesh_has_symmetry_face_for_a_half_model(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "blockMeshDict").read_text()
    assert "symmetry" in text
    assert "type            symmetry;" in text


def test_blockmesh_omits_symmetry_for_a_full_model(tmp_path: Path) -> None:
    case = deep_merge(BASE, {"physics": {"yaw_deg": 5.0}})
    case["geometry"]["patches"] = [
        p for p in case["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    spec = resolve(case, profile="dev")
    text = (_render(tmp_path, spec) / "system" / "blockMeshDict").read_text()
    assert "type            symmetry;" not in text


def test_blockmesh_cell_counts_match_the_domain(tmp_path: Path) -> None:
    spec = _spec()
    domain = BoxDomainBuilder().build(spec, BOUNDS)
    text = (_render(tmp_path, spec) / "system" / "blockMeshDict").read_text()
    nx, ny, nz = domain.n_cells
    assert f"({nx} {ny} {nz})" in text


def test_snappy_uses_absolute_first_layer_thickness(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "snappyHexMeshDict").read_text()
    assert "relativeSizes false;" in text
    assert "firstLayerThickness" in text


def test_snappy_layer_count_matches_the_wall_profile(tmp_path: Path) -> None:
    spec = _spec(wall_treatment="low_y_plus")
    text = (_render(tmp_path, spec) / "system" / "snappyHexMeshDict").read_text()
    assert f"nSurfaceLayers {spec.mesh.n_layers};" in text


def test_decompose_par_matches_rank_count(tmp_path: Path) -> None:
    spec = _spec()
    text = (_render(tmp_path, spec) / "system" / "decomposeParDict").read_text()
    assert f"numberOfSubdomains {spec.solve.n_ranks};" in text


def test_surface_features_lists_every_geometry_file(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "surfaceFeaturesDict").read_text()
    assert "body.stl" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render_mesh.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.render.context'`

- [ ] **Step 3: Package the templates**

In `pyproject.toml` add:

```toml
[tool.setuptools.package-data]
simdev = ["render/templates/*.jinja"]
```

- [ ] **Step 4: Write the context builder**

```python
# pipeline/simdev/render/context.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from simdev.config.schema import CaseSpec, WallTreatment
from simdev.domain.base import DomainBox
from simdev.geometry.roles import PatchRole, traits

C_MU = 0.09

WALL_FUNCTIONS: dict[WallTreatment, dict[str, str]] = {
    WallTreatment.HIGH_Y_PLUS: {
        "nut": "nutkWallFunction",
        "k": "kqRWallFunction",
        "omega": "omegaWallFunction",
    },
    WallTreatment.LOW_Y_PLUS: {
        "nut": "nutLowReWallFunction",
        "k": "kLowReWallFunction",
        "omega": "omegaWallFunction",
    },
    WallTreatment.SPALDING: {
        "nut": "nutUSpaldingWallFunction",
        "k": "kqRWallFunction",
        "omega": "omegaWallFunction",
    },
}


def inlet_turbulence(spec: CaseSpec) -> tuple[float, float]:
    """Freestream k and omega from turbulence intensity and length scale."""
    k = 1.5 * (spec.flow.turbulence_intensity * spec.flow.u_inf) ** 2
    omega = k**0.5 / (C_MU**0.25 * spec.flow.turbulence_length_scale)
    return k, omega


def location_in_mesh(domain: DomainBox) -> tuple[float, float, float]:
    """A point in the fluid, upstream of the body and off the symmetry plane."""
    dx = domain.x_max - domain.x_min
    return (
        domain.x_min + 0.10 * dx,
        domain.y_min + 0.25 * (domain.y_max - domain.y_min),
        domain.z_min + 0.50 * (domain.z_max - domain.z_min),
    )


def build_context(
    spec: CaseSpec, domain: DomainBox, geometry_files: dict[str, Path]
) -> dict[str, Any]:
    k, omega = inlet_turbulence(spec)
    wall_fns = WALL_FUNCTIONS[spec.physics.wall_treatment]

    wall_patches = [
        p.name for p in spec.geometry.patches if traits(p.role).is_wall
    ]
    refined_patches = [
        {
            "name": p.name,
            "file": Path(geometry_files[p.name]).name,
            "level_min": spec.mesh.surface_refinement_min,
            "level_max": spec.mesh.surface_refinement_max,
            "n_layers": spec.mesh.n_layers if traits(p.role).refinement == "high" else 0,
        }
        for p in spec.geometry.patches
        if p.name in geometry_files
    ]

    return {
        "spec": spec,
        "domain": domain,
        "geometry_files": {n: Path(p).name for n, p in geometry_files.items()},
        "refined_patches": refined_patches,
        "wall_patches": wall_patches,
        "force_patches": [
            p.name
            for p in spec.geometry.patches
            if traits(p.role).in_forces
        ],
        "wall_fns": wall_fns,
        "k_inlet": k,
        "omega_inlet": omega,
        "nut_inlet": k / omega,
        "a_ref": spec.a_ref_effective,
        "location_in_mesh": location_in_mesh(domain),
        "ground_is_moving": spec.ground.motion.value == "moving",
        "symmetry_patch": domain.symmetry,
    }
```

- [ ] **Step 5: Write the renderer**

```python
# pipeline/simdev/render/render.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

from simdev.config.schema import CaseSpec
from simdev.domain.base import DomainBox
from simdev.render.context import build_context

TEMPLATE_DIR = Path(__file__).parent / "templates"

MESH_DICTS: dict[str, str] = {
    "blockMeshDict.jinja": "system/blockMeshDict",
    "surfaceFeaturesDict.jinja": "system/surfaceFeaturesDict",
    "snappyHexMeshDict.jinja": "system/snappyHexMeshDict",
    "decomposeParDict.jinja": "system/decomposeParDict",
}


def env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _render_set(
    templates: dict[str, str], context: dict[str, Any], out_dir: Path
) -> None:
    environment = env()
    for template_name, relative_path in templates.items():
        target = Path(out_dir) / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            environment.get_template(template_name).render(**context),
            encoding="utf-8",
        )


def render_mesh_dicts(
    spec: CaseSpec,
    domain: DomainBox,
    geometry_files: dict[str, Path],
    out_dir: Path,
) -> None:
    _render_set(MESH_DICTS, build_context(spec, domain, geometry_files), out_dir)
```

- [ ] **Step 6: Write the templates**

`pipeline/simdev/render/templates/foam_header.jinja`:

```jinja
FoamFile
{
    version     2.0;
    format      ascii;
    class       {{ foam_class }};
    object      {{ foam_object }};
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
```

`blockMeshDict.jinja`:

```jinja
{% with foam_class="dictionary", foam_object="blockMeshDict" %}
{% include "foam_header.jinja" %}
{% endwith %}

scale   1;

vertices
(
    ({{ domain.x_min }} {{ domain.y_min }} {{ domain.z_min }})
    ({{ domain.x_max }} {{ domain.y_min }} {{ domain.z_min }})
    ({{ domain.x_max }} {{ domain.y_max }} {{ domain.z_min }})
    ({{ domain.x_min }} {{ domain.y_max }} {{ domain.z_min }})
    ({{ domain.x_min }} {{ domain.y_min }} {{ domain.z_max }})
    ({{ domain.x_max }} {{ domain.y_min }} {{ domain.z_max }})
    ({{ domain.x_max }} {{ domain.y_max }} {{ domain.z_max }})
    ({{ domain.x_min }} {{ domain.y_max }} {{ domain.z_max }})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({{ domain.n_cells[0] }} {{ domain.n_cells[1] }} {{ domain.n_cells[2] }}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    {{ domain.inlet }}
    {
        type            patch;
        faces           ((0 4 7 3));
    }
    {{ domain.outlet }}
    {
        type            patch;
        faces           ((1 2 6 5));
    }
    {{ domain.ground }}
    {
        type            wall;
        faces           ((0 3 2 1));
    }
{% if symmetry_patch %}
    {{ symmetry_patch }}
    {
        type            symmetry;
        faces           ((0 1 5 4));
    }
    {{ domain.farfield }}
    {
        type            patch;
        faces           ((3 7 6 2) (4 5 6 7));
    }
{% else %}
    {{ domain.farfield }}
    {
        type            patch;
        faces           ((0 1 5 4) (3 7 6 2) (4 5 6 7));
    }
{% endif %}
);

mergePatchPairs ();
```

`surfaceFeaturesDict.jinja`:

```jinja
{% with foam_class="dictionary", foam_object="surfaceFeaturesDict" %}
{% include "foam_header.jinja" %}
{% endwith %}

surfaces
(
{% for name, filename in geometry_files.items() %}
    "{{ filename }}"
{% endfor %}
);

includedAngle   150;
```

`decomposeParDict.jinja`:

```jinja
{% with foam_class="dictionary", foam_object="decomposeParDict" %}
{% include "foam_header.jinja" %}
{% endwith %}

numberOfSubdomains {{ spec.solve.n_ranks }};

method          scotch;
```

`snappyHexMeshDict.jinja`:

```jinja
{% with foam_class="dictionary", foam_object="snappyHexMeshDict" %}
{% include "foam_header.jinja" %}
{% endwith %}

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{
{% for patch in refined_patches %}
    {{ patch.name }}
    {
        type            triSurfaceMesh;
        file            "{{ patch.file }}";
    }
{% endfor %}
}

castellatedMeshControls
{
    maxLocalCells       2000000;
    maxGlobalCells      50000000;
    minRefinementCells  10;
    nCellsBetweenLevels 3;
    resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;

    features
    (
{% for patch in refined_patches %}
        {
            file    "{{ patch.name }}.eMesh";
            level   {{ patch.level_max }};
        }
{% endfor %}
    );

    refinementSurfaces
    {
{% for patch in refined_patches %}
        {{ patch.name }}
        {
            level   ({{ patch.level_min }} {{ patch.level_max }});
        }
{% endfor %}
    }

    refinementRegions {}

    locationInMesh ({{ location_in_mesh[0] }} {{ location_in_mesh[1] }} {{ location_in_mesh[2] }});
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes false;
    expansionRatio {{ spec.mesh.expansion_ratio }};
    firstLayerThickness {{ spec.mesh.first_layer_thickness }};
    minThickness {{ spec.mesh.first_layer_thickness / 4 }};

    layers
    {
{% for patch in refined_patches %}
        {{ patch.name }}
        {
            nSurfaceLayers {{ patch.n_layers }};
        }
{% endfor %}
    }

    nGrow               0;
    featureAngle        130;
    nRelaxIter          5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}

meshQualityControls
{
    maxNonOrtho         {{ spec.mesh.max_non_ortho }};
    maxBoundarySkewness 20;
    maxInternalSkewness {{ spec.mesh.max_skewness }};
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.02;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}

mergeTolerance 1e-6;
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_render_mesh.py -v`
Expected: PASS (11 tests)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml pipeline/simdev/render tests/test_render_mesh.py
git commit -m "feat: render blockMesh, snappy, surfaceFeatures and decomposePar dicts"
```

---

## Task 10: Solver dictionaries and boundary conditions

**Files:**
- Modify: `pipeline/simdev/render/context.py` — add `Bc`, `build_bcs`
- Modify: `pipeline/simdev/render/render.py` — add `SOLVER_DICTS`, `render_solver_dicts`, `render_case`
- Create: templates `controlDict.jinja`, `fvSchemes.jinja`, `fvSolution.jinja`, `transportProperties.jinja`, `turbulenceProperties.jinja`, `field.jinja`
- Test: `tests/test_render_solver.py`

**Interfaces:**
- Consumes: everything from Task 9
- Produces:
  - `Bc` — frozen dataclass: `patch: str`, `entries: dict[str, str]`
  - `build_bcs(spec: CaseSpec, k: float, omega: float, nut: float, wall_fns: dict[str, str]) -> dict[str, list[Bc]]` keyed by field name `"U"`, `"p"`, `"k"`, `"omega"`, `"nut"`
  - `render_solver_dicts(spec, domain, geometry_files, out_dir) -> None`
  - `render_case(spec, domain, geometry_files, out_dir) -> None` — mesh dicts + solver dicts + `caseSpec.json`

BCs are built entirely in Python so the field templates are a single loop with no branching. One `field.jinja` renders all five fields.

**Moving ground uses `fixedValue uniform (U 0 0)`, not `movingWallVelocity`.** The spec's shorthand said `movingWallVelocity`; that BC exists for cases where the *mesh* moves. On a static mesh with a translating road surface, `fixedValue` is the correct expression of the same physics.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render_solver.py
from __future__ import annotations

from pathlib import Path

import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.domain.box import BoxDomainBuilder
from simdev.render.context import WALL_FUNCTIONS, build_bcs, inlet_turbulence
from simdev.render.render import render_case

BOUNDS = ((0.0, -0.1945, 0.05), (1.044, 0.1945, 0.338))

BASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044, "c_of_r": [0.5, 0.0, 0.0]},
    "geometry": {
        "kind": "ahmed",
        "ahmed": {},
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


def _spec(overrides: dict | None = None, **kw: object):
    return resolve(deep_merge(BASE, overrides or {}), profile="dev", **kw)


def _render(tmp_path: Path, spec=None) -> Path:
    spec = spec or _spec()
    domain = BoxDomainBuilder().build(spec, BOUNDS)
    render_case(spec, domain, {"body": Path("body.stl")}, tmp_path)
    return tmp_path


def _bcs(spec):
    k, omega = inlet_turbulence(spec)
    return build_bcs(spec, k, omega, k / omega, WALL_FUNCTIONS[spec.physics.wall_treatment])


def test_static_ground_is_no_slip() -> None:
    bcs = _bcs(_spec())
    ground = next(b for b in bcs["U"] if b.patch == "ground")
    assert ground.entries["type"] == "noSlip"


def test_moving_ground_translates_at_freestream() -> None:
    bcs = _bcs(_spec({"ground": {"motion": "moving"}}))
    ground = next(b for b in bcs["U"] if b.patch == "ground")
    assert ground.entries["type"] == "fixedValue"
    assert "40.0" in ground.entries["value"]


def test_body_is_always_no_slip() -> None:
    body = next(b for b in _bcs(_spec())["U"] if b.patch == "body")
    assert body.entries["type"] == "noSlip"


def test_outlet_uses_inlet_outlet_for_backflow() -> None:
    outlet = next(b for b in _bcs(_spec())["U"] if b.patch == "outlet")
    assert outlet.entries["type"] == "inletOutlet"


def test_farfield_is_slip_not_no_slip() -> None:
    farfield = next(b for b in _bcs(_spec())["U"] if b.patch == "farfield")
    assert farfield.entries["type"] == "slip"


def test_wall_functions_follow_the_treatment() -> None:
    low = _bcs(_spec(wall_treatment="low_y_plus"))
    body_nut = next(b for b in low["nut"] if b.patch == "body")
    assert body_nut.entries["type"] == "nutLowReWallFunction"


def test_every_field_covers_every_patch() -> None:
    spec = _spec()
    names = {p.name for p in spec.geometry.patches}
    for field, bcs in _bcs(spec).items():
        assert {b.patch for b in bcs} == names, field


def test_force_coeffs_uses_the_effective_reference_area(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    # Half model: 0.112 / 2
    assert "Aref            0.056;" in text


def test_force_coeffs_mag_u_inf_matches_the_inlet(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    assert "magUInf         40.0;" in text


def test_force_coeffs_lists_only_force_patches(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "controlDict").read_text()
    block = text.split("forceCoeffs")[1]
    assert "body" in block
    assert "ground" not in block.split("liftDir")[0]


def test_control_dict_end_time_is_the_iteration_cap(tmp_path: Path) -> None:
    spec = _spec()
    text = (_render(tmp_path, spec) / "system" / "controlDict").read_text()
    assert f"endTime         {spec.solve.max_iterations};" in text


def test_div_scheme_is_second_order_stabilised(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "fvSchemes").read_text()
    assert "div(phi,U)      bounded Gauss linearUpwind grad(U);" in text


def test_steady_case_has_residual_control(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "fvSolution").read_text()
    assert "residualControl" in text


def test_pressure_uses_gamg(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "fvSolution").read_text()
    assert "GAMG" in text


def test_transport_properties_carries_nu(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "constant" / "transportProperties").read_text()
    assert "1.5e-05" in text


def test_case_spec_json_is_written(tmp_path: Path) -> None:
    assert (_render(tmp_path) / "caseSpec.json").exists()


def test_field_files_exist_for_every_field(tmp_path: Path) -> None:
    out = _render(tmp_path)
    for field in ("U", "p", "k", "omega", "nut"):
        assert (out / "0" / field).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render_solver.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_bcs'`

- [ ] **Step 3: Add the BC builder to `context.py`**

```python
# append to pipeline/simdev/render/context.py
from dataclasses import dataclass

from simdev.config.schema import GroundMotion


@dataclass(frozen=True)
class Bc:
    patch: str
    entries: dict[str, str]


def _vec(x: float, y: float, z: float) -> str:
    return f"uniform ({x} {y} {z})"


def build_bcs(
    spec: CaseSpec,
    k: float,
    omega: float,
    nut: float,
    wall_fns: dict[str, str],
) -> dict[str, list[Bc]]:
    """Boundary conditions per field per patch. All branching happens here."""
    u = spec.flow.u_inf
    freestream = _vec(u, 0.0, 0.0)
    moving_ground = spec.ground.motion is GroundMotion.MOVING

    fields = ("U", "p", "k", "omega", "nut")
    bcs: dict[str, list[Bc]] = {f: [] for f in fields}

    for patch in spec.geometry.patches:
        role = patch.role
        name = patch.name

        if role is PatchRole.INLET:
            entries = {
                "U": {"type": "fixedValue", "value": freestream},
                "p": {"type": "zeroGradient"},
                "k": {"type": "fixedValue", "value": f"uniform {k}"},
                "omega": {"type": "fixedValue", "value": f"uniform {omega}"},
                "nut": {"type": "calculated", "value": f"uniform {nut}"},
            }
        elif role is PatchRole.OUTLET:
            entries = {
                "U": {
                    "type": "inletOutlet",
                    "inletValue": _vec(0.0, 0.0, 0.0),
                    "value": freestream,
                },
                "p": {"type": "fixedValue", "value": "uniform 0"},
                "k": {
                    "type": "inletOutlet",
                    "inletValue": f"uniform {k}",
                    "value": f"uniform {k}",
                },
                "omega": {
                    "type": "inletOutlet",
                    "inletValue": f"uniform {omega}",
                    "value": f"uniform {omega}",
                },
                "nut": {"type": "calculated", "value": f"uniform {nut}"},
            }
        elif role is PatchRole.SYMMETRY:
            entries = {f: {"type": "symmetry"} for f in fields}
        elif role is PatchRole.FARFIELD:
            entries = {f: {"type": "slip"} for f in fields}
        elif role is PatchRole.GROUND:
            if moving_ground:
                u_entry = {"type": "fixedValue", "value": freestream}
            else:
                u_entry = {"type": "noSlip"}
            entries = {
                "U": u_entry,
                "p": {"type": "zeroGradient"},
                "k": {"type": wall_fns["k"], "value": f"uniform {k}"},
                "omega": {"type": wall_fns["omega"], "value": f"uniform {omega}"},
                "nut": {"type": wall_fns["nut"], "value": "uniform 0"},
            }
        else:  # BODY, TYRE
            entries = {
                "U": {"type": "noSlip"},
                "p": {"type": "zeroGradient"},
                "k": {"type": wall_fns["k"], "value": f"uniform {k}"},
                "omega": {"type": wall_fns["omega"], "value": f"uniform {omega}"},
                "nut": {"type": wall_fns["nut"], "value": "uniform 0"},
            }

        for field in fields:
            bcs[field].append(Bc(patch=name, entries=entries[field]))

    return bcs
```

Then extend `build_context`'s return dict with:

```python
        "bcs": build_bcs(spec, k, omega, k / omega, wall_fns),
        "field_meta": {
            "U": ("volVectorField", "[0 1 -1 0 0 0 0]", f"uniform ({spec.flow.u_inf} 0 0)"),
            "p": ("volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0"),
            "k": ("volScalarField", "[0 2 -2 0 0 0 0]", f"uniform {k}"),
            "omega": ("volScalarField", "[0 0 -1 0 0 0 0]", f"uniform {omega}"),
            "nut": ("volScalarField", "[0 2 -1 0 0 0 0]", f"uniform {k / omega}"),
        },
```

- [ ] **Step 4: Extend the renderer**

```python
# append to pipeline/simdev/render/render.py
import json

SOLVER_DICTS: dict[str, str] = {
    "controlDict.jinja": "system/controlDict",
    "fvSchemes.jinja": "system/fvSchemes",
    "fvSolution.jinja": "system/fvSolution",
    "transportProperties.jinja": "constant/transportProperties",
    "turbulenceProperties.jinja": "constant/turbulenceProperties",
}

FIELDS = ("U", "p", "k", "omega", "nut")


def render_solver_dicts(
    spec: CaseSpec,
    domain: DomainBox,
    geometry_files: dict[str, Path],
    out_dir: Path,
) -> None:
    context = build_context(spec, domain, geometry_files)
    _render_set(SOLVER_DICTS, context, out_dir)

    environment = env()
    template = environment.get_template("field.jinja")
    for field in FIELDS:
        foam_class, dimensions, internal = context["field_meta"][field]
        target = Path(out_dir) / "0" / field
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            template.render(
                field=field,
                foam_class=foam_class,
                dimensions=dimensions,
                internal_field=internal,
                bcs=context["bcs"][field],
            ),
            encoding="utf-8",
        )


def render_case(
    spec: CaseSpec,
    domain: DomainBox,
    geometry_files: dict[str, Path],
    out_dir: Path,
) -> None:
    """Render a complete, self-contained case plus its provenance record."""
    out = Path(out_dir)
    render_mesh_dicts(spec, domain, geometry_files, out)
    render_solver_dicts(spec, domain, geometry_files, out)
    (out / "caseSpec.json").write_text(
        json.dumps(
            {"spec": spec.model_dump(mode="json"), "hash": spec.spec_hash()},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
```

- [ ] **Step 5: Write `field.jinja`**

```jinja
{% with foam_class=foam_class, foam_object=field %}
{% include "foam_header.jinja" %}
{% endwith %}

dimensions      {{ dimensions }};

internalField   {{ internal_field }};

boundaryField
{
{% for bc in bcs %}
    {{ bc.patch }}
    {
{% for key, value in bc.entries.items() %}
        {{ key }}{{ ' ' * (16 - key|length) }}{{ value }};
{% endfor %}
    }
{% endfor %}
}
```

- [ ] **Step 6: Write `controlDict.jinja`**

```jinja
{% with foam_class="dictionary", foam_object="controlDict" %}
{% include "foam_header.jinja" %}
{% endwith %}

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {{ spec.solve.max_iterations }};
deltaT          1;
writeControl    runTime;
writeInterval   {{ spec.solve.max_iterations }};
purgeWrite      2;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;

functions
{
    forceCoeffs
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        log             true;

        patches         ({% for p in force_patches %}{{ p }} {% endfor %});

        rho             rhoInf;
        rhoInf          {{ spec.flow.rho }};
        magUInf         {{ spec.flow.u_inf }};
        lRef            {{ spec.forces.l_ref }};
        Aref            {{ a_ref }};

        liftDir         (0 0 1);
        dragDir         (1 0 0);
        pitchAxis       (0 1 0);
        CofR            ({{ spec.forces.c_of_r[0] }} {{ spec.forces.c_of_r[1] }} {{ spec.forces.c_of_r[2] }});
    }

    yPlus
    {
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        log             true;
    }

    residuals
    {
        type            solverInfo;
        libs            ("libutilityFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        fields          (U p k omega);
    }
}
```

- [ ] **Step 7: Write the remaining templates**

`fvSchemes.jinja`:

```jinja
{% with foam_class="dictionary", foam_object="fvSchemes" %}
{% include "foam_header.jinja" %}
{% endwith %}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         cellLimited Gauss linear 1;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear limited corrected 0.33;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited corrected 0.33;
}

wallDist
{
    method          meshWave;
}
```

`fvSolution.jinja`:

```jinja
{% with foam_class="dictionary", foam_object="fvSolution" %}
{% include "foam_header.jinja" %}
{% endwith %}

solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-07;
        relTol          0.05;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0.1;
    }
}

SIMPLE
{
    consistent          yes;
    nNonOrthogonalCorrectors 1;

    residualControl
    {
        p               {{ spec.solve.residual_tol }};
        U               {{ spec.solve.residual_tol }};
        "(k|omega)"     {{ spec.solve.residual_tol }};
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        ".*"            0.9;
    }
}
```

`transportProperties.jinja`:

```jinja
{% with foam_class="dictionary", foam_object="transportProperties" %}
{% include "foam_header.jinja" %}
{% endwith %}

transportModel  Newtonian;

nu              {{ spec.flow.nu }};
```

`turbulenceProperties.jinja`:

```jinja
{% with foam_class="dictionary", foam_object="turbulenceProperties" %}
{% include "foam_header.jinja" %}
{% endwith %}

simulationType  RAS;

RAS
{
    RASModel        {{ spec.physics.turbulence_model }};
    turbulence      on;
    printCoeffs     on;
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_render_solver.py -v`
Expected: PASS (17 tests)

- [ ] **Step 9: Commit**

```bash
git add pipeline/simdev/render tests/test_render_solver.py
git commit -m "feat: render solver dictionaries, boundary conditions and provenance"
```

---

## Task 11: OpenFOAM log and output parsers

**Files:**
- Create: `pipeline/simdev/run/parsers.py`
- Create: `tests/fixtures/logs/checkMesh_ok.log`, `tests/fixtures/logs/checkMesh_bad.log`, `tests/fixtures/logs/snappy_layers.log`, `tests/fixtures/coefficient.dat`, `tests/fixtures/yPlus.dat`
- Test: `tests/test_parsers.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `CheckMeshResult` — frozen dataclass: `n_cells: int`, `max_non_ortho: float`, `max_skewness: float`, `has_negative_volumes: bool`, `failed_checks: list[str]`
  - `LayerInfo` — frozen dataclass: `patch: str`, `faces: int`, `layers: float`, `thickness: float`
  - `parse_check_mesh(text: str) -> CheckMeshResult`
  - `parse_layer_summary(text: str) -> dict[str, LayerInfo]`
  - `read_force_coeffs(path: Path) -> pandas.DataFrame` — columns from the last `#` header line, including `Time`, `Cd`, `Cl`
  - `read_y_plus(path: Path) -> pandas.DataFrame` — columns `Time`, `patch`, `min`, `max`, `average`
  - `find_fatal_errors(text: str) -> list[str]`

`find_fatal_errors` exists because **OpenFOAM frequently exits 0 on partial failure**. Exit codes alone are not enough, which is the single most expensive lesson from the benchmark pipeline.

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/logs/checkMesh_ok.log`:

```
Mesh stats
    points:           1234567
    faces:            3456789
    internal faces:   3344556
    cells:            1122334
    faces per cell:   6.05
    boundary patches: 5

Checking geometry...
    Overall domain bounding box (-5.22 0 0) (11.484 3.132 3.132)
    Mesh has 3 solution directions (1 1 1)
    Boundary openness (1.2e-16 -3.4e-17 5.6e-17) OK.
    Max cell openness = 2.5e-16 OK.
    Max aspect ratio = 12.34 OK.
    Minimum face area = 1.2e-06. Maximum face area = 3.4e-03.  Face area magnitudes OK.
    Min volume = 1.1e-09. Max volume = 5.6e-05.  Total volume = 123.4.  Cell volumes OK.
    Mesh non-orthogonality Max: 64.23 average: 8.12
    Non-orthogonality check OK.
    Face pyramids OK.
    Max skewness = 3.21 OK.
    Coupled point location match (average 0) OK.

Mesh OK.

End
```

`tests/fixtures/logs/checkMesh_bad.log`:

```
Mesh stats
    points:           1234567
    faces:            3456789
    internal faces:   3344556
    cells:            1122334
    faces per cell:   6.05
    boundary patches: 5

Checking geometry...
    Overall domain bounding box (-5.22 0 0) (11.484 3.132 3.132)
    Mesh has 3 solution directions (1 1 1)
    Max aspect ratio = 980.5 OK.
    Min volume = -1.1e-14. Max volume = 5.6e-05.  Total volume = 123.4.
 ***Zero or negative cell volume detected.  Minimum negative volume: -1.1e-14, Number of negative volume cells: 5
    Mesh non-orthogonality Max: 82.51 average: 15.44
 ***Number of severely non-orthogonal (> 70 degrees) faces: 1234.
    Max skewness = 6.78 FAILED
 ***Error in face pyramids: 12 faces are incorrectly oriented.

Failed 3 mesh checks.

End
```

`tests/fixtures/logs/snappy_layers.log`:

```
Layer mesh : cells:2000000  faces:6000000  points:2100000

Extruding 18345 out of 20000 faces (91.7%). Removed extrusion at 0 faces.

patch      faces    layers   overall thickness
                             [m]       [%]
-----      -----    ------   ---------  ---
body       18345    17.9     0.000358   99.4
stilts     2400     4.2      0.000084   23.3
ground     22000    0        0          0

Finished meshing in = 812.4 s.

End
```

`tests/fixtures/coefficient.dat`:

```
# Force coefficients
# dragDir       : (1 0 0)
# liftDir       : (0 0 1)
# Time          Cd              Cl              CmPitch         Cs
1               0.512000        -0.081000       0.010000        0.000100
2               0.402000        -0.095000       0.011000        0.000090
3               0.351000        -0.101000       0.011500        0.000085
4               0.348000        -0.102000       0.011600        0.000084
5               0.347500        -0.102300       0.011610        0.000084
```

`tests/fixtures/yPlus.dat`:

```
# yPlus
# Time          patch           min             max             average
1               body            28.100000       310.500000      95.400000
1               ground          15.200000       205.300000      78.100000
2               body            30.200000       298.100000      92.700000
2               ground          16.100000       201.700000      77.400000
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_parsers.py
from __future__ import annotations

from pathlib import Path

import pytest

from simdev.run.parsers import (
    find_fatal_errors,
    parse_check_mesh,
    parse_layer_summary,
    read_force_coeffs,
    read_y_plus,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _log(name: str) -> str:
    return (FIXTURES / "logs" / name).read_text(encoding="utf-8")


def test_parse_check_mesh_reads_a_healthy_mesh() -> None:
    result = parse_check_mesh(_log("checkMesh_ok.log"))
    assert result.n_cells == 1122334
    assert result.max_non_ortho == pytest.approx(64.23)
    assert result.max_skewness == pytest.approx(3.21)
    assert result.has_negative_volumes is False
    assert result.failed_checks == []


def test_parse_check_mesh_detects_negative_volumes() -> None:
    result = parse_check_mesh(_log("checkMesh_bad.log"))
    assert result.has_negative_volumes is True


def test_parse_check_mesh_collects_failed_checks() -> None:
    result = parse_check_mesh(_log("checkMesh_bad.log"))
    assert result.failed_checks
    assert any("skewness" in c.lower() for c in result.failed_checks)


def test_parse_check_mesh_reads_bad_metrics() -> None:
    result = parse_check_mesh(_log("checkMesh_bad.log"))
    assert result.max_non_ortho == pytest.approx(82.51)
    assert result.max_skewness == pytest.approx(6.78)


def test_parse_layer_summary_reads_every_patch() -> None:
    layers = parse_layer_summary(_log("snappy_layers.log"))
    assert set(layers) == {"body", "stilts", "ground"}
    assert layers["body"].layers == pytest.approx(17.9)
    assert layers["body"].faces == 18345
    assert layers["body"].thickness == pytest.approx(0.000358)


def test_parse_layer_summary_captures_a_collapsed_stack() -> None:
    layers = parse_layer_summary(_log("snappy_layers.log"))
    assert layers["stilts"].layers == pytest.approx(4.2)
    assert layers["ground"].layers == pytest.approx(0.0)


def test_read_force_coeffs_uses_the_header_row() -> None:
    df = read_force_coeffs(FIXTURES / "coefficient.dat")
    assert list(df.columns[:3]) == ["Time", "Cd", "Cl"]
    assert len(df) == 5
    assert df["Cd"].iloc[-1] == pytest.approx(0.3475)


def test_read_y_plus_returns_per_patch_rows() -> None:
    df = read_y_plus(FIXTURES / "yPlus.dat")
    assert set(df["patch"]) == {"body", "ground"}
    body_last = df[df["patch"] == "body"].iloc[-1]
    assert body_last["max"] == pytest.approx(298.1)


def test_find_fatal_errors_detects_foam_fatal() -> None:
    text = "some output\n--> FOAM FATAL ERROR: keyword nu is undefined\nmore"
    assert find_fatal_errors(text)


def test_find_fatal_errors_detects_floating_point_exception() -> None:
    assert find_fatal_errors("Floating point exception\n")


def test_find_fatal_errors_is_quiet_on_a_clean_log() -> None:
    assert find_fatal_errors(_log("checkMesh_ok.log")) == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_parsers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.run.parsers'`

- [ ] **Step 4: Write minimal implementation**

```python
# pipeline/simdev/run/parsers.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_CELLS = re.compile(r"^\s*cells:\s*(\d+)", re.MULTILINE)
_NON_ORTHO = re.compile(r"Mesh non-orthogonality Max:\s*([-\d.eE+]+)")
_SKEWNESS = re.compile(r"Max skewness\s*=\s*([-\d.eE+]+)")
_NEGATIVE_VOLUME = re.compile(r"Zero or negative cell volume", re.IGNORECASE)
_LAYER_ROW = re.compile(
    r"^\s*(\S+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$"
)

FATAL_PATTERNS = (
    "FOAM FATAL ERROR",
    "FOAM FATAL IO ERROR",
    "Floating point exception",
    "Segmentation fault",
)


@dataclass(frozen=True)
class CheckMeshResult:
    n_cells: int
    max_non_ortho: float
    max_skewness: float
    has_negative_volumes: bool
    failed_checks: list[str]


@dataclass(frozen=True)
class LayerInfo:
    patch: str
    faces: int
    layers: float
    thickness: float


def parse_check_mesh(text: str) -> CheckMeshResult:
    cells = _CELLS.search(text)
    non_ortho = _NON_ORTHO.search(text)
    skewness = _SKEWNESS.search(text)

    failed: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "FAILED" in stripped or stripped.startswith("***"):
            failed.append(stripped.lstrip("* "))

    return CheckMeshResult(
        n_cells=int(cells.group(1)) if cells else 0,
        max_non_ortho=float(non_ortho.group(1)) if non_ortho else float("nan"),
        max_skewness=float(skewness.group(1)) if skewness else float("nan"),
        has_negative_volumes=bool(_NEGATIVE_VOLUME.search(text)),
        failed_checks=failed,
    )


def parse_layer_summary(text: str) -> dict[str, LayerInfo]:
    """Parse snappyHexMesh's final layer table.

    The table is: patch, faces, layers, overall thickness [m], thickness [%].
    """
    result: dict[str, LayerInfo] = {}
    in_table = False

    for line in text.splitlines():
        if line.strip().startswith("-----"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.strip():
            break

        match = _LAYER_ROW.match(line)
        if not match:
            break
        name, faces, layers, thickness, _pct = match.groups()
        result[name] = LayerInfo(
            patch=name,
            faces=int(faces),
            layers=float(layers),
            thickness=float(thickness),
        )

    return result


def _last_header(path: Path) -> list[str]:
    header: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                header = line.lstrip("#").split()
            else:
                break
    return header


def read_force_coeffs(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", comment="#", names=_last_header(path), engine="python"
    )


def read_y_plus(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", comment="#", names=_last_header(path), engine="python"
    )


def find_fatal_errors(text: str) -> list[str]:
    """OpenFOAM often exits 0 on partial failure. Scan the log too."""
    return [
        line.strip()
        for line in text.splitlines()
        if any(pattern in line for pattern in FATAL_PATTERNS)
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_parsers.py -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/run/parsers.py tests/fixtures tests/test_parsers.py
git commit -m "feat: add checkMesh, layer, force and yPlus log parsers"
```

---

## Task 12: Gates

**Files:**
- Create: `pipeline/simdev/gates/base.py`, `pipeline/simdev/gates/mesh_quality.py`, `pipeline/simdev/gates/convergence.py`, `pipeline/simdev/gates/yplus.py`
- Test: `tests/test_gates.py`

**Interfaces:**
- Consumes: `CheckMeshResult`, `LayerInfo` (Task 11); `CaseSpec` (Task 3)
- Produces:
  - `GateResult` — frozen dataclass: `passed: bool`, `reasons: list[str]`, `detail: dict[str, float | str]`
  - `check_mesh_quality(result: CheckMeshResult, layers: dict[str, LayerInfo], spec: CaseSpec) -> GateResult`
  - `ConvergenceResult` — frozen dataclass: `converged: bool`, `reasons: list[str]`, `means: dict[str, float]`, `stds: dict[str, float]`, `window: tuple[int, int]`, `n_iterations: int`
  - `check_convergence(df: pandas.DataFrame, spec: CaseSpec) -> ConvergenceResult`
  - `check_y_plus(df: pandas.DataFrame, spec: CaseSpec) -> GateResult`

**Convergence semantics, fixed here so no later task reinvents them:** the reported coefficient is the **mean over the trailing plateau window**, never the last instantaneous iteration. A run that reaches `max_iterations` without plateauing returns `converged=False` **with** its means still populated — the caller records a non-converged result rather than hanging or silently passing.

Layer coverage is gated only on patches that requested layers. A patch with `nSurfaceLayers 0` is not a failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gates.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simdev.config.resolve import deep_merge, resolve
from simdev.gates.convergence import check_convergence
from simdev.gates.mesh_quality import check_mesh_quality
from simdev.gates.yplus import check_y_plus
from simdev.run.parsers import CheckMeshResult, LayerInfo

BASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
        "ahmed": {},
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


def _spec(overrides: dict | None = None):
    return resolve(deep_merge(BASE, overrides or {}), profile="dev")


def _good_mesh() -> CheckMeshResult:
    return CheckMeshResult(
        n_cells=1000000,
        max_non_ortho=64.0,
        max_skewness=3.2,
        has_negative_volumes=False,
        failed_checks=[],
    )


def _layers(body_layers: float) -> dict[str, LayerInfo]:
    return {
        "body": LayerInfo("body", 18000, body_layers, 3.5e-4),
        "ground": LayerInfo("ground", 22000, 0.0, 0.0),
    }


def _forces(n: int, drift: float = 0.0) -> pd.DataFrame:
    it = np.arange(1, n + 1)
    return pd.DataFrame(
        {
            "Time": it,
            "Cd": 0.35 + drift * it + 1e-5 * np.sin(it),
            "Cl": -0.10 + drift * it,
        }
    )


def test_mesh_gate_passes_a_good_mesh() -> None:
    spec = _spec()
    result = check_mesh_quality(_good_mesh(), _layers(spec.mesh.n_layers), spec)
    assert result.passed is True
    assert result.reasons == []


def test_mesh_gate_fails_on_negative_volumes() -> None:
    spec = _spec()
    bad = CheckMeshResult(1000, 60.0, 3.0, True, [])
    result = check_mesh_quality(bad, _layers(spec.mesh.n_layers), spec)
    assert result.passed is False
    assert any("negative" in r for r in result.reasons)


def test_mesh_gate_fails_on_excessive_non_orthogonality() -> None:
    spec = _spec()
    bad = CheckMeshResult(1000, 85.0, 3.0, False, [])
    result = check_mesh_quality(bad, _layers(spec.mesh.n_layers), spec)
    assert result.passed is False
    assert any("orthogonal" in r for r in result.reasons)


def test_mesh_gate_fails_on_collapsed_layers() -> None:
    spec = _spec()
    # Requested n_layers, achieved a small fraction of them.
    result = check_mesh_quality(_good_mesh(), _layers(1.0), spec)
    assert result.passed is False
    assert any("layer" in r for r in result.reasons)


def test_mesh_gate_ignores_patches_that_requested_no_layers() -> None:
    spec = _spec()
    result = check_mesh_quality(_good_mesh(), _layers(spec.mesh.n_layers), spec)
    assert "ground" not in " ".join(result.reasons)


def test_convergence_detects_a_plateau() -> None:
    spec = _spec()
    result = check_convergence(_forces(300), spec)
    assert result.converged is True
    assert result.means["Cd"] == pytest.approx(0.35, abs=1e-3)


def test_convergence_reports_the_mean_not_the_last_value() -> None:
    spec = _spec()
    df = _forces(300)
    df.loc[df.index[-1], "Cd"] = 99.0  # a single spike must not dominate
    result = check_convergence(df, spec)
    assert result.means["Cd"] < 5.0


def test_convergence_fails_on_a_drifting_signal() -> None:
    spec = _spec()
    result = check_convergence(_forces(300, drift=1e-3), spec)
    assert result.converged is False
    assert any("drift" in r or "plateau" in r for r in result.reasons)


def test_non_converged_run_still_reports_means() -> None:
    spec = _spec()
    result = check_convergence(_forces(300, drift=1e-3), spec)
    assert result.converged is False
    assert "Cd" in result.means and "Cl" in result.means


def test_convergence_fails_when_too_short_to_judge() -> None:
    spec = _spec()
    result = check_convergence(_forces(5), spec)
    assert result.converged is False


def test_convergence_window_is_the_trailing_slice() -> None:
    spec = _spec()
    result = check_convergence(_forces(300), spec)
    start, end = result.window
    assert end - start == spec.solve.plateau_window
    assert end == 300


def test_y_plus_gate_passes_inside_the_band() -> None:
    spec = _spec()
    df = pd.DataFrame(
        {"Time": [1], "patch": ["body"], "min": [35.0], "max": [280.0], "average": [95.0]}
    )
    assert check_y_plus(df, spec).passed is True


def test_y_plus_gate_fails_outside_the_band() -> None:
    spec = _spec()
    df = pd.DataFrame(
        {"Time": [1], "patch": ["body"], "min": [0.4], "max": [3.0], "average": [1.2]}
    )
    result = check_y_plus(df, spec)
    assert result.passed is False
    assert any("y+" in r for r in result.reasons)


def test_y_plus_gate_only_judges_force_patches() -> None:
    spec = _spec()
    df = pd.DataFrame(
        {
            "Time": [1, 1],
            "patch": ["body", "ground"],
            "min": [35.0, 0.1],
            "max": [280.0, 2.0],
            "average": [95.0, 1.0],
        }
    )
    assert check_y_plus(df, spec).passed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.gates.base'`

- [ ] **Step 3: Write the shared result type**

```python
# pipeline/simdev/gates/base.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, float | str] = field(default_factory=dict)
```

- [ ] **Step 4: Write the mesh gate**

```python
# pipeline/simdev/gates/mesh_quality.py
from __future__ import annotations

from simdev.config.schema import CaseSpec
from simdev.gates.base import GateResult
from simdev.geometry.roles import traits
from simdev.run.parsers import CheckMeshResult, LayerInfo


def check_mesh_quality(
    result: CheckMeshResult,
    layers: dict[str, LayerInfo],
    spec: CaseSpec,
) -> GateResult:
    reasons: list[str] = []

    if result.has_negative_volumes:
        reasons.append("mesh contains negative volume cells; it is unusable")

    if result.max_non_ortho > spec.mesh.max_non_ortho:
        reasons.append(
            f"max non-orthogonality {result.max_non_ortho:.1f} exceeds "
            f"{spec.mesh.max_non_ortho:.1f}; the solve will be unstable or inaccurate"
        )

    if result.max_skewness > spec.mesh.max_skewness:
        reasons.append(
            f"max skewness {result.max_skewness:.2f} exceeds "
            f"{spec.mesh.max_skewness:.2f}"
        )

    for check in result.failed_checks:
        reasons.append(f"checkMesh reported: {check}")

    # Layer coverage, only on patches that asked for layers.
    requested = spec.mesh.n_layers
    for patch in spec.geometry.patches:
        if traits(patch.role).refinement != "high":
            continue
        info = layers.get(patch.name)
        if info is None:
            continue
        coverage = info.layers / requested if requested else 1.0
        if coverage < spec.mesh.min_layer_coverage:
            reasons.append(
                f"layer coverage on '{patch.name}' is {coverage:.0%} "
                f"({info.layers:.1f} of {requested} layers); the near-wall "
                "resolution you designed for does not exist there"
            )

    return GateResult(
        passed=not reasons,
        reasons=reasons,
        detail={
            "n_cells": result.n_cells,
            "max_non_ortho": result.max_non_ortho,
            "max_skewness": result.max_skewness,
        },
    )
```

- [ ] **Step 5: Write the convergence gate**

```python
# pipeline/simdev/gates/convergence.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simdev.config.schema import CaseSpec

COEFFICIENTS = ("Cd", "Cl")


@dataclass(frozen=True)
class ConvergenceResult:
    converged: bool
    reasons: list[str] = field(default_factory=list)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    window: tuple[int, int] = (0, 0)
    n_iterations: int = 0


def check_convergence(df: pd.DataFrame, spec: CaseSpec) -> ConvergenceResult:
    """Force-plateau test over the trailing window.

    Always reports the window mean, converged or not, so a non-converged run
    produces a recorded-but-flagged result rather than nothing.
    """
    n = len(df)
    window = spec.solve.plateau_window
    reasons: list[str] = []

    if n < window:
        return ConvergenceResult(
            converged=False,
            reasons=[
                f"only {n} iterations recorded, need at least {window} to judge "
                "a plateau"
            ],
            means={c: float(df[c].mean()) for c in COEFFICIENTS if c in df},
            stds={c: float(df[c].std()) for c in COEFFICIENTS if c in df},
            window=(0, n),
            n_iterations=n,
        )

    tail = df.iloc[-window:]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for coefficient in COEFFICIENTS:
        if coefficient not in df:
            continue
        values = tail[coefficient].to_numpy()
        mean = float(values.mean())
        means[coefficient] = mean
        stds[coefficient] = float(values.std())

        scale = max(abs(mean), 1e-9)

        if stds[coefficient] / scale > spec.solve.plateau_tol:
            reasons.append(
                f"{coefficient} has not reached a plateau: relative scatter "
                f"{stds[coefficient] / scale:.2%} over the last {window} "
                f"iterations exceeds {spec.solve.plateau_tol:.2%}"
            )

        # Least-squares slope across the window, expressed per window.
        iterations = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(iterations, values, 1)[0]) * len(values)
        if abs(slope) / scale > spec.solve.plateau_tol:
            reasons.append(
                f"{coefficient} is still drifting: {slope / scale:+.2%} across "
                f"the last {window} iterations"
            )

    return ConvergenceResult(
        converged=not reasons,
        reasons=reasons,
        means=means,
        stds=stds,
        window=(n - window, n),
        n_iterations=n,
    )
```

- [ ] **Step 6: Write the y+ gate**

```python
# pipeline/simdev/gates/yplus.py
from __future__ import annotations

import pandas as pd

from simdev.config.schema import CaseSpec
from simdev.gates.base import GateResult
from simdev.geometry.roles import traits


def check_y_plus(df: pd.DataFrame, spec: CaseSpec) -> GateResult:
    """Confirm the achieved y+ matches the band the wall treatment assumes."""
    force_patches = {
        p.name for p in spec.geometry.patches if traits(p.role).in_forces
    }
    lo, hi = spec.post.yplus_min, spec.post.yplus_max
    reasons: list[str] = []
    detail: dict[str, float | str] = {}

    latest = df[df["Time"] == df["Time"].max()]

    for _, row in latest.iterrows():
        patch = str(row["patch"])
        if patch not in force_patches:
            continue
        average = float(row["average"])
        detail[f"{patch}_avg_yplus"] = average
        if not (lo <= average <= hi):
            reasons.append(
                f"average y+ on '{patch}' is {average:.1f}, outside the "
                f"[{lo}, {hi}] band assumed by wall treatment "
                f"'{spec.physics.wall_treatment.value}'; the wall model is "
                "not valid there"
            )

    return GateResult(passed=not reasons, reasons=reasons, detail=detail)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_gates.py -v`
Expected: PASS (14 tests)

- [ ] **Step 8: Commit**

```bash
git add pipeline/simdev/gates tests/test_gates.py
git commit -m "feat: add mesh quality, force plateau and y+ band gates"
```

---

## Task 13: Command runner and stage status

**Files:**
- Create: `pipeline/simdev/run/runner.py`, `pipeline/simdev/run/status.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `find_fatal_errors` (Task 11)
- Produces:
  - `StageError(Exception)` with `reasons: list[str]`
  - `CommandResult` — frozen dataclass: `name: str`, `argv: list[str]`, `returncode: int`, `log_path: Path`, `fatal: list[str]`
  - `parallel_argv(argv: list[str], n_ranks: int) -> list[str]`
  - `Runner(case_dir: Path)` with `run(argv, name=None, check=True) -> CommandResult` and `run_parallel(argv, n_ranks, name=None, check=True) -> CommandResult`
  - `StageStatus` — frozen dataclass: `stage: str`, `state: str` (`"ok"` / `"failed"` / `"gate_failed"`), `input_hash: str`, `reasons: list[str]`, `detail: dict`
  - `write_status(run_dir: Path, status: StageStatus) -> Path`
  - `read_status(run_dir: Path, stage: str) -> StageStatus | None`
  - `should_skip(run_dir: Path, stage: str, input_hash: str, force: bool) -> bool`

**Two non-negotiables encoded here.** A command that exits 0 but printed `FOAM FATAL` is a **failure** — exit codes alone are not enough. And `should_skip` compares an input hash; it never inspects whether an output directory looks populated, which is the benchmark pipeline's stale-results bug.

Logs go to `<case_dir>/logs/log.<name>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from simdev.run.runner import (
    CommandResult,
    Runner,
    StageError,
    parallel_argv,
)
from simdev.run.status import (
    StageStatus,
    read_status,
    should_skip,
    write_status,
)


def test_parallel_argv_wraps_with_mpirun_and_parallel_flag() -> None:
    assert parallel_argv(["simpleFoam"], 8) == [
        "mpirun",
        "-np",
        "8",
        "simpleFoam",
        "-parallel",
    ]


def test_successful_command_writes_a_log(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    result = runner.run([sys.executable, "-c", "print('hello')"], name="greet")
    assert result.returncode == 0
    assert result.log_path.exists()
    assert "hello" in result.log_path.read_text(encoding="utf-8")


def test_failing_command_raises(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    with pytest.raises(StageError) as exc:
        runner.run([sys.executable, "-c", "raise SystemExit(3)"], name="boom")
    assert any("exit code 3" in r for r in exc.value.reasons)


def test_failing_command_can_be_tolerated(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    result = runner.run(
        [sys.executable, "-c", "raise SystemExit(3)"], name="boom", check=False
    )
    assert isinstance(result, CommandResult)
    assert result.returncode == 3


def test_foam_fatal_on_a_zero_exit_is_still_a_failure(tmp_path: Path) -> None:
    # The expensive lesson: OpenFOAM exits 0 on partial failure.
    runner = Runner(tmp_path)
    script = "print('--> FOAM FATAL ERROR: keyword nu undefined')"
    with pytest.raises(StageError) as exc:
        runner.run([sys.executable, "-c", script], name="sneaky")
    assert any("FOAM FATAL" in r for r in exc.value.reasons)


def test_status_round_trips(tmp_path: Path) -> None:
    status = StageStatus(
        stage="mesh", state="ok", input_hash="abc123", reasons=[], detail={"n_cells": 10}
    )
    write_status(tmp_path, status)
    loaded = read_status(tmp_path, "mesh")
    assert loaded is not None
    assert loaded.state == "ok"
    assert loaded.input_hash == "abc123"


def test_read_status_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_status(tmp_path, "mesh") is None


def test_should_skip_only_on_matching_hash(tmp_path: Path) -> None:
    write_status(
        tmp_path, StageStatus("mesh", "ok", "abc123", [], {})
    )
    assert should_skip(tmp_path, "mesh", "abc123", force=False) is True
    assert should_skip(tmp_path, "mesh", "different", force=False) is False


def test_force_defeats_skipping(tmp_path: Path) -> None:
    write_status(tmp_path, StageStatus("mesh", "ok", "abc123", [], {}))
    assert should_skip(tmp_path, "mesh", "abc123", force=True) is False


def test_failed_stage_is_never_skipped(tmp_path: Path) -> None:
    write_status(tmp_path, StageStatus("mesh", "gate_failed", "abc123", ["bad"], {}))
    assert should_skip(tmp_path, "mesh", "abc123", force=False) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.run.runner'`

- [ ] **Step 3: Write the runner**

```python
# pipeline/simdev/run/runner.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from simdev.run.parsers import find_fatal_errors


class StageError(Exception):
    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass(frozen=True)
class CommandResult:
    name: str
    argv: list[str]
    returncode: int
    log_path: Path
    fatal: list[str] = field(default_factory=list)


def parallel_argv(argv: list[str], n_ranks: int) -> list[str]:
    return ["mpirun", "-np", str(n_ranks), *argv, "-parallel"]


class Runner:
    """Runs OpenFOAM utilities, captures logs, and fails loudly."""

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = Path(case_dir)
        self.log_dir = self.case_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self, argv: list[str], name: str | None = None, check: bool = True
    ) -> CommandResult:
        name = name or Path(argv[0]).name
        log_path = self.log_dir / f"log.{name}"

        completed = subprocess.run(
            argv,
            cwd=self.case_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        log_path.write_text(output, encoding="utf-8")

        fatal = find_fatal_errors(output)
        result = CommandResult(
            name=name,
            argv=argv,
            returncode=completed.returncode,
            log_path=log_path,
            fatal=fatal,
        )

        if check:
            reasons: list[str] = []
            if completed.returncode != 0:
                reasons.append(f"{name} failed with exit code {completed.returncode}")
            if fatal:
                # Exit code 0 is not proof of success in OpenFOAM.
                reasons.append(f"{name} logged a fatal error: {fatal[0]}")
            if reasons:
                reasons.append(f"see {log_path}")
                raise StageError(reasons)

        return result

    def run_parallel(
        self,
        argv: list[str],
        n_ranks: int,
        name: str | None = None,
        check: bool = True,
    ) -> CommandResult:
        return self.run(
            parallel_argv(argv, n_ranks), name=name or Path(argv[0]).name, check=check
        )
```

- [ ] **Step 4: Write the status store**

```python
# pipeline/simdev/run/status.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATES = ("ok", "failed", "gate_failed")


@dataclass(frozen=True)
class StageStatus:
    stage: str
    state: str
    input_hash: str
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def _path(run_dir: Path, stage: str) -> Path:
    return Path(run_dir) / "status" / f"{stage}.json"


def write_status(run_dir: Path, status: StageStatus) -> Path:
    if status.state not in STATES:
        raise ValueError(f"unknown state {status.state!r}; expected one of {STATES}")
    target = _path(run_dir, status.stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
    return target


def read_status(run_dir: Path, stage: str) -> StageStatus | None:
    target = _path(run_dir, stage)
    if not target.exists():
        return None
    return StageStatus(**json.loads(target.read_text(encoding="utf-8")))


def should_skip(run_dir: Path, stage: str, input_hash: str, force: bool) -> bool:
    """Skip only on an unchanged input hash from a stage that succeeded.

    Never infers staleness from directory contents.
    """
    if force:
        return False
    status = read_status(run_dir, stage)
    if status is None:
        return False
    return status.state == "ok" and status.input_hash == input_hash
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add pipeline/simdev/run/runner.py pipeline/simdev/run/status.py tests/test_runner.py
git commit -m "feat: add command runner with log scanning and stage status store"
```

---

## Task 14: Prepare stage

**Files:**
- Create: `pipeline/simdev/stages/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `load_case` (Task 4), `validate` (Task 5), `write_ahmed_stl` (Task 6), `read_stl_info`/`check_geometry`/`projected_frontal_area` (Task 7), `BoxDomainBuilder`/`check_blockage` (Task 8), `render_case` (Task 10), status helpers (Task 13)
- Produces:
  - `PrepareResult` — frozen dataclass: `spec: CaseSpec`, `domain: DomainBox`, `run_dir: Path`, `warnings: list[str]`, `frontal_area: float`
  - `prepare(case_path: Path, run_dir: Path, profile: str, wall_treatment: str | None = None, overrides: dict | None = None, force: bool = False) -> PrepareResult`

This stage runs **without OpenFOAM installed**, so it is fully unit-testable. It is the stage that turns a config file into a complete case directory plus `caseSpec.json`.

Geometry bounds are the union across all geometry files, so the domain sizes off the whole model rather than one part. The frontal area passed to `check_blockage` is the **true projected area**, halved for a half model — not `a_ref`, which is a reference convention.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prepare.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simdev.config.validate import ValidationError
from simdev.run.status import read_status
from simdev.stages.prepare import prepare

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044, "c_of_r": [0.5, 0.0, 0.0]},
    "geometry": {
        "kind": "ahmed",
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
def case_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    return path


def test_prepare_writes_a_complete_case(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")

    for relative in (
        "system/blockMeshDict",
        "system/snappyHexMeshDict",
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "system/decomposeParDict",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "0/U",
        "0/p",
        "0/k",
        "0/omega",
        "0/nut",
        "caseSpec.json",
    ):
        assert (run_dir / relative).exists(), relative


def test_prepare_writes_the_geometry(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")
    assert (run_dir / "constant" / "triSurface" / "body.stl").exists()


def test_prepare_records_ok_status(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = prepare(case_file, run_dir, profile="dev")
    status = read_status(run_dir, "prepare")
    assert status is not None
    assert status.state == "ok"
    assert status.input_hash == result.spec.spec_hash()


def test_prepare_sizes_the_domain_from_the_geometry(
    case_file: Path, tmp_path: Path
) -> None:
    result = prepare(case_file, tmp_path / "run", profile="dev")
    # 5 body lengths upstream of the nose at x = 0.
    assert result.domain.x_min == pytest.approx(-5.0 * 1.044, rel=1e-6)


def test_prepare_uses_projected_area_for_blockage(
    case_file: Path, tmp_path: Path
) -> None:
    result = prepare(case_file, tmp_path / "run", profile="dev")
    # Half model, so half the projected area, and below width*height.
    assert 0.0 < result.frontal_area < 0.112032 / 2 + 1e-6


def test_prepare_rejects_an_invalid_case(tmp_path: Path) -> None:
    bad = dict(CASE)
    bad["geometry"] = dict(CASE["geometry"])
    bad["geometry"]["patches"] = [
        p for p in CASE["geometry"]["patches"] if p["role"] != "symmetry"
    ]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValidationError):
        prepare(path, tmp_path / "run", profile="dev")


def test_prepare_is_idempotent_and_skips_on_unchanged_input(
    case_file: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")
    marker = run_dir / "system" / "controlDict"
    marker.write_text("TOUCHED", encoding="utf-8")

    prepare(case_file, run_dir, profile="dev")
    assert marker.read_text(encoding="utf-8") == "TOUCHED"


def test_force_rerenders(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")
    marker = run_dir / "system" / "controlDict"
    marker.write_text("TOUCHED", encoding="utf-8")

    prepare(case_file, run_dir, profile="dev", force=True)
    assert marker.read_text(encoding="utf-8") != "TOUCHED"


def test_profile_change_invalidates_the_skip(case_file: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare(case_file, run_dir, profile="dev")
    marker = run_dir / "system" / "controlDict"
    marker.write_text("TOUCHED", encoding="utf-8")

    # A different profile is a different spec hash, so it must re-render.
    prepare(case_file, run_dir, profile="production")
    assert marker.read_text(encoding="utf-8") != "TOUCHED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.stages.prepare'`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/simdev/stages/prepare.py
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh

from simdev.config.resolve import load_case
from simdev.config.schema import CaseSpec
from simdev.config.validate import validate
from simdev.domain.base import DomainBox
from simdev.domain.box import BoxDomainBuilder, check_blockage
from simdev.geometry.ahmed import write_ahmed_stl
from simdev.geometry.stl import check_geometry, projected_frontal_area, read_stl_info
from simdev.render.render import render_case
from simdev.run.status import StageStatus, should_skip, write_status

STAGE = "prepare"


@dataclass(frozen=True)
class PrepareResult:
    spec: CaseSpec
    domain: DomainBox
    run_dir: Path
    frontal_area: float
    warnings: list[str] = field(default_factory=list)


def _write_geometry(spec: CaseSpec, run_dir: Path) -> dict[str, Path]:
    tri_surface = run_dir / "constant" / "triSurface"
    tri_surface.mkdir(parents=True, exist_ok=True)

    if spec.geometry.kind == "ahmed":
        assert spec.geometry.ahmed is not None
        return write_ahmed_stl(spec.geometry.ahmed, tri_surface)

    assert spec.geometry.stl_dir is not None
    files: dict[str, Path] = {}
    for patch in spec.geometry.patches:
        source = Path(spec.geometry.stl_dir) / f"{patch.name}.stl"
        if source.exists():
            target = tri_surface / source.name
            shutil.copy2(source, target)
            files[patch.name] = target
    return files


def _bounds(files: dict[str, Path]) -> tuple[list[float], list[float]]:
    lows: list[list[float]] = []
    highs: list[list[float]] = []
    for path in files.values():
        info = read_stl_info(path)
        lows.append(list(info.bounds_min))
        highs.append(list(info.bounds_max))
    return (
        [min(v[i] for v in lows) for i in range(3)],
        [max(v[i] for v in highs) for i in range(3)],
    )


def prepare(
    case_path: Path,
    run_dir: Path,
    profile: str,
    wall_treatment: str | None = None,
    overrides: dict[str, Any] | None = None,
    force: bool = False,
) -> PrepareResult:
    spec = load_case(Path(case_path), profile, wall_treatment, overrides)
    warnings = validate(spec)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    geometry_files = _write_geometry(spec, run_dir)
    if not geometry_files:
        raise FileNotFoundError("no geometry files were produced or found")

    for path in geometry_files.values():
        warnings.extend(
            check_geometry(read_stl_info(path), expected_length=spec.forces.l_ref)
        )

    lo, hi = _bounds(geometry_files)
    domain = BoxDomainBuilder().build(spec, (lo, hi))

    combined = trimesh.util.concatenate(
        [trimesh.load_mesh(p, process=False) for p in geometry_files.values()]
    )
    frontal_area = projected_frontal_area(combined, axis=0)
    if spec.half_model:
        frontal_area /= 2.0
    warnings.extend(check_blockage(frontal_area, domain, spec.domain.max_blockage))

    result = PrepareResult(
        spec=spec,
        domain=domain,
        run_dir=run_dir,
        frontal_area=frontal_area,
        warnings=warnings,
    )

    if should_skip(run_dir, STAGE, spec.spec_hash(), force):
        return result

    render_case(spec, domain, geometry_files, run_dir)
    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            state="ok",
            input_hash=spec.spec_hash(),
            reasons=warnings,
            detail={
                "frontal_area": frontal_area,
                "background_cells": domain.cell_count,
                "half_model": spec.half_model,
            },
        ),
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prepare.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/simdev/stages/prepare.py tests/test_prepare.py
git commit -m "feat: add prepare stage producing a complete validated case"
```

---

## Task 15: Mesh and solve stages

**Files:**
- Create: `pipeline/simdev/stages/common.py`, `pipeline/simdev/stages/mesh.py`, `pipeline/simdev/stages/solve.py`
- Test: `tests/test_stages_mesh_solve.py`

**Interfaces:**
- Consumes: `Runner`/`StageError` (Task 13), gates (Task 12), parsers (Task 11)
- Produces:
  - `load_spec(run_dir: Path) -> CaseSpec`
  - `require_stage(run_dir: Path, stage: str) -> None` — raises `StageError` if the named stage did not finish `ok`
  - `find_latest(run_dir: Path, pattern: str) -> Path` — newest match under `postProcessing/`, raises `FileNotFoundError`
  - `mesh(run_dir: Path, force: bool = False, runner: Runner | None = None) -> GateResult`
  - `solve(run_dir: Path, force: bool = False, runner: Runner | None = None) -> ConvergenceResult`

**Parallel strategy, fixed here:** `decomposePar` runs once in the mesh stage; `snappyHexMesh` and `simpleFoam` both run `-parallel` on that same decomposition. No `reconstructParMesh` between them — reconstructing only to re-decompose wastes a large fraction of meshing time on a 20M-cell case. Function-object output (`postProcessing/`) is written by the master rank, so forces and y+ are available without reconstruction.

The `runner` parameter exists so tests can inject a recording double. Production callers omit it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stages_mesh_solve.py
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from simdev.run.runner import CommandResult, StageError
from simdev.run.status import read_status
from simdev.stages.mesh import mesh
from simdev.stages.prepare import prepare
from simdev.stages.solve import solve

FIXTURES = Path(__file__).parent / "fixtures"

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
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


class RecordingRunner:
    """Stands in for Runner; records argv and serves canned logs."""

    def __init__(self, case_dir: Path, logs: dict[str, str] | None = None) -> None:
        self.case_dir = Path(case_dir)
        self.log_dir = self.case_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.calls: list[list[str]] = []
        self.logs = logs or {}

    def _emit(self, name: str) -> CommandResult:
        log_path = self.log_dir / f"log.{name}"
        log_path.write_text(self.logs.get(name, "End\n"), encoding="utf-8")
        return CommandResult(name=name, argv=[], returncode=0, log_path=log_path)

    def run(self, argv, name=None, check=True):
        self.calls.append(list(argv))
        return self._emit(name or Path(argv[0]).name)

    def run_parallel(self, argv, n_ranks, name=None, check=True):
        self.calls.append(["mpirun", "-np", str(n_ranks), *argv, "-parallel"])
        return self._emit(name or Path(argv[0]).name)


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    case = tmp_path / "config.yaml"
    case.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    target = tmp_path / "run"
    prepare(case, target, profile="dev")
    return target


def _mesh_logs() -> dict[str, str]:
    return {
        "checkMesh": (FIXTURES / "logs" / "checkMesh_ok.log").read_text(),
        "snappyHexMesh": (FIXTURES / "logs" / "snappy_layers.log").read_text(),
    }


def _good_layers_log(n_layers: int) -> str:
    return (
        "patch      faces    layers   overall thickness\n"
        "                             [m]       [%]\n"
        "-----      -----    ------   ---------  ---\n"
        f"body       18345    {float(n_layers)}     0.000358   99.4\n"
        "ground     22000    0        0          0\n"
        "\n"
    )


def test_mesh_runs_the_expected_command_sequence(run_dir: Path) -> None:
    runner = RecordingRunner(run_dir, _mesh_logs())
    mesh(run_dir, runner=runner)
    executables = [c[0] if c[0] != "mpirun" else c[3] for c in runner.calls]
    assert executables == [
        "blockMesh",
        "surfaceFeatures",
        "decomposePar",
        "snappyHexMesh",
        "checkMesh",
    ]


def test_mesh_uses_surface_features_not_the_legacy_name(run_dir: Path) -> None:
    runner = RecordingRunner(run_dir, _mesh_logs())
    mesh(run_dir, runner=runner)
    flat = [token for call in runner.calls for token in call]
    assert "surfaceFeatures" in flat
    assert "surfaceFeatureExtract" not in flat


def test_mesh_runs_snappy_in_parallel(run_dir: Path) -> None:
    runner = RecordingRunner(run_dir, _mesh_logs())
    mesh(run_dir, runner=runner)
    snappy = next(c for c in runner.calls if "snappyHexMesh" in c)
    assert snappy[0] == "mpirun"
    assert "-parallel" in snappy
    assert "-overwrite" in snappy


def test_mesh_gate_failure_is_recorded_and_raises(run_dir: Path) -> None:
    logs = _mesh_logs()
    logs["checkMesh"] = (FIXTURES / "logs" / "checkMesh_bad.log").read_text()
    runner = RecordingRunner(run_dir, logs)
    with pytest.raises(StageError):
        mesh(run_dir, runner=runner)
    status = read_status(run_dir, "mesh")
    assert status is not None
    assert status.state == "gate_failed"


def test_mesh_records_ok_when_gates_pass(run_dir: Path) -> None:
    from simdev.stages.common import load_spec

    logs = _mesh_logs()
    logs["snappyHexMesh"] = _good_layers_log(load_spec(run_dir).mesh.n_layers)
    runner = RecordingRunner(run_dir, logs)
    result = mesh(run_dir, runner=runner)
    assert result.passed is True
    assert read_status(run_dir, "mesh").state == "ok"


def test_solve_requires_a_successful_mesh(run_dir: Path) -> None:
    with pytest.raises(StageError) as exc:
        solve(run_dir, runner=RecordingRunner(run_dir))
    assert "mesh" in str(exc.value)


def _complete_mesh(run_dir: Path) -> None:
    from simdev.stages.common import load_spec

    logs = _mesh_logs()
    logs["snappyHexMesh"] = _good_layers_log(load_spec(run_dir).mesh.n_layers)
    mesh(run_dir, runner=RecordingRunner(run_dir, logs))


def _install_forces(run_dir: Path) -> None:
    target = run_dir / "postProcessing" / "forceCoeffs" / "0"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURES / "coefficient.dat", target / "coefficient.dat")


def test_solve_runs_simple_foam_in_parallel(run_dir: Path) -> None:
    _complete_mesh(run_dir)
    _install_forces(run_dir)
    runner = RecordingRunner(run_dir)
    solve(run_dir, runner=runner)
    call = next(c for c in runner.calls if "simpleFoam" in c)
    assert call[0] == "mpirun"
    assert "-parallel" in call


def test_solve_does_not_redecompose(run_dir: Path) -> None:
    _complete_mesh(run_dir)
    _install_forces(run_dir)
    runner = RecordingRunner(run_dir)
    solve(run_dir, runner=runner)
    flat = [token for call in runner.calls for token in call]
    assert "decomposePar" not in flat


def test_solve_records_non_converged_without_raising(run_dir: Path) -> None:
    _complete_mesh(run_dir)
    _install_forces(run_dir)  # only 5 iterations, cannot plateau
    result = solve(run_dir, runner=RecordingRunner(run_dir))
    assert result.converged is False
    status = read_status(run_dir, "solve")
    assert status is not None
    assert status.state == "gate_failed"
    # Means are still reported so the run is recorded, not lost.
    assert "Cd" in result.means
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stages_mesh_solve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.stages.common'`

- [ ] **Step 3: Write the shared stage helpers**

```python
# pipeline/simdev/stages/common.py
from __future__ import annotations

import json
from pathlib import Path

from simdev.config.schema import CaseSpec
from simdev.run.runner import StageError
from simdev.run.status import read_status


def load_spec(run_dir: Path) -> CaseSpec:
    payload = json.loads((Path(run_dir) / "caseSpec.json").read_text(encoding="utf-8"))
    return CaseSpec.model_validate(payload["spec"])


def require_stage(run_dir: Path, stage: str) -> None:
    status = read_status(run_dir, stage)
    if status is None:
        raise StageError([f"stage '{stage}' has not been run"])
    if status.state != "ok":
        raise StageError(
            [f"stage '{stage}' did not succeed (state={status.state})", *status.reasons]
        )


def find_latest(run_dir: Path, pattern: str) -> Path:
    matches = sorted(
        (Path(run_dir) / "postProcessing").glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"no file matching 'postProcessing/{pattern}' under {run_dir}"
        )
    return matches[-1]
```

- [ ] **Step 4: Write the mesh stage**

```python
# pipeline/simdev/stages/mesh.py
from __future__ import annotations

from pathlib import Path

from simdev.gates.base import GateResult
from simdev.gates.mesh_quality import check_mesh_quality
from simdev.run.parsers import parse_check_mesh, parse_layer_summary
from simdev.run.runner import Runner, StageError
from simdev.run.status import StageStatus, should_skip, write_status
from simdev.stages.common import load_spec, require_stage

STAGE = "mesh"


def mesh(run_dir: Path, force: bool = False, runner: Runner | None = None) -> GateResult:
    run_dir = Path(run_dir)
    require_stage(run_dir, "prepare")
    spec = load_spec(run_dir)

    if should_skip(run_dir, STAGE, spec.spec_hash(), force):
        return GateResult(passed=True, reasons=[], detail={"skipped": "unchanged input"})

    runner = runner or Runner(run_dir)

    runner.run(["blockMesh"], name="blockMesh")
    runner.run(["surfaceFeatures"], name="surfaceFeatures")
    runner.run(["decomposePar"], name="decomposePar")
    snappy = runner.run_parallel(
        ["snappyHexMesh", "-overwrite"], spec.solve.n_ranks, name="snappyHexMesh"
    )
    check = runner.run_parallel(["checkMesh"], spec.solve.n_ranks, name="checkMesh")

    layers = parse_layer_summary(snappy.log_path.read_text(encoding="utf-8"))
    quality = parse_check_mesh(check.log_path.read_text(encoding="utf-8"))
    gate = check_mesh_quality(quality, layers, spec)

    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            state="ok" if gate.passed else "gate_failed",
            input_hash=spec.spec_hash(),
            reasons=gate.reasons,
            detail=dict(gate.detail),
        ),
    )

    if not gate.passed:
        raise StageError(
            ["mesh quality gate failed", *gate.reasons, "refusing to start the solve"]
        )

    return gate
```

- [ ] **Step 5: Write the solve stage**

```python
# pipeline/simdev/stages/solve.py
from __future__ import annotations

from pathlib import Path

from simdev.gates.convergence import ConvergenceResult, check_convergence
from simdev.run.parsers import read_force_coeffs
from simdev.run.runner import Runner
from simdev.run.status import StageStatus, should_skip, write_status
from simdev.stages.common import find_latest, load_spec, require_stage

STAGE = "solve"


def solve(
    run_dir: Path, force: bool = False, runner: Runner | None = None
) -> ConvergenceResult:
    run_dir = Path(run_dir)
    require_stage(run_dir, "mesh")
    spec = load_spec(run_dir)

    if should_skip(run_dir, STAGE, spec.spec_hash(), force):
        from simdev.run.status import read_status

        previous = read_status(run_dir, STAGE)
        assert previous is not None
        return ConvergenceResult(
            converged=True,
            reasons=["skipped: unchanged input"],
            means={k: float(v) for k, v in previous.detail.items() if k.endswith("_mean")},
        )

    runner = runner or Runner(run_dir)
    # The mesh stage already decomposed; reuse that decomposition.
    runner.run_parallel(["simpleFoam"], spec.solve.n_ranks, name="simpleFoam")

    forces = read_force_coeffs(find_latest(run_dir, "forceCoeffs/*/coefficient.dat"))
    result = check_convergence(forces, spec)

    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            state="ok" if result.converged else "gate_failed",
            input_hash=spec.spec_hash(),
            reasons=result.reasons,
            detail={
                **{f"{k}_mean": v for k, v in result.means.items()},
                **{f"{k}_std": v for k, v in result.stds.items()},
                "n_iterations": result.n_iterations,
                "window_start": result.window[0],
                "window_end": result.window[1],
            },
        ),
    )

    # A non-converged run is recorded and flagged, never silently passed and
    # never fatal: the post stage still reports the means, marked non-converged.
    return result
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_stages_mesh_solve.py -v`
Expected: PASS (10 tests)

- [ ] **Step 7: Commit**

```bash
git add pipeline/simdev/stages tests/test_stages_mesh_solve.py
git commit -m "feat: add mesh and solve stages with quality and convergence gates"
```

---

## Task 16: Post stage, results records and plots

**Files:**
- Create: `pipeline/simdev/report/results.py`, `pipeline/simdev/report/plots.py`, `pipeline/simdev/stages/post.py`
- Test: `tests/test_results.py`, `tests/test_post.py`

**Interfaces:**
- Consumes: parsers (Task 11), gates (Task 12), status (Task 13), `load_spec`/`find_latest` (Task 15)
- Produces:
  - `ResultRecord` — frozen dataclass: `case_name: str`, `spec_hash: str`, `timestamp: str`, `converged: bool`, `cd_mean: float`, `cd_std: float`, `cl_mean: float`, `cl_std: float`, `window_start: int`, `window_end: int`, `n_iterations: int`, `n_cells: int`, `yplus_passed: bool`, `yplus: dict[str, float]`, `reasons: list[str]`
  - `write_result(run_dir: Path, record: ResultRecord) -> tuple[Path, Path]` — returns `(json_path, csv_path)`
  - `read_result(run_dir: Path) -> ResultRecord`
  - `aggregate(run_dirs: Iterable[Path]) -> pandas.DataFrame`
  - `plot_force_history(df, out_path: Path, window: tuple[int, int]) -> Path`
  - `plot_residuals(df, out_path: Path) -> Path`
  - `post(run_dir: Path, force: bool = False) -> ResultRecord`

**Two rules this task exists to enforce.** Results are written **per run** — `write_result` never opens a shared file in append mode, and `aggregate` combines records on read instead. And **`post` runs even when the solve did not converge**: it treats a `gate_failed` solve as valid input and records the result with `converged=False`. Only a `failed` solve blocks it. A run that produced numbers must never vanish because it did not plateau.

Every record carries `spec_hash`, so a result can never be attached to the wrong specification.

- [ ] **Step 1: Write the failing test for results**

```python
# tests/test_results.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdev.report.results import ResultRecord, aggregate, read_result, write_result


def _record(**overrides: object) -> ResultRecord:
    base = dict(
        case_name="ahmed",
        spec_hash="abc123",
        timestamp="2026-08-09T12:00:00Z",
        converged=True,
        cd_mean=0.347,
        cd_std=0.0004,
        cl_mean=-0.102,
        cl_std=0.0006,
        window_start=100,
        window_end=300,
        n_iterations=300,
        n_cells=1122334,
        yplus_passed=True,
        yplus={"body": 95.4},
        reasons=[],
    )
    base.update(overrides)
    return ResultRecord(**base)


def test_write_result_produces_json_and_csv(tmp_path: Path) -> None:
    json_path, csv_path = write_result(tmp_path, _record())
    assert json_path.exists() and csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["spec_hash"] == "abc123"


def test_result_round_trips(tmp_path: Path) -> None:
    write_result(tmp_path, _record())
    assert read_result(tmp_path).cd_mean == pytest.approx(0.347)


def test_csv_holds_exactly_one_row(tmp_path: Path) -> None:
    _, csv_path = write_result(tmp_path, _record())
    lines = [l for l in csv_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2  # header plus one row


def test_rewriting_replaces_rather_than_appends(tmp_path: Path) -> None:
    # The benchmark pipeline appended to a shared table and corrupted it.
    write_result(tmp_path, _record())
    _, csv_path = write_result(tmp_path, _record(cd_mean=0.9))
    lines = [l for l in csv_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert read_result(tmp_path).cd_mean == pytest.approx(0.9)


def test_aggregate_combines_independent_runs(tmp_path: Path) -> None:
    dirs = []
    for i, cd in enumerate([0.34, 0.35, 0.36]):
        d = tmp_path / f"run{i}"
        d.mkdir()
        write_result(d, _record(cd_mean=cd, spec_hash=f"hash{i}"))
        dirs.append(d)
    df = aggregate(dirs)
    assert len(df) == 3
    assert set(df["spec_hash"]) == {"hash0", "hash1", "hash2"}


def test_non_converged_record_is_written_and_flagged(tmp_path: Path) -> None:
    write_result(tmp_path, _record(converged=False, reasons=["Cd still drifting"]))
    loaded = read_result(tmp_path)
    assert loaded.converged is False
    assert loaded.reasons == ["Cd still drifting"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_results.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.report.results'`

- [ ] **Step 3: Write the results module**

```python
# pipeline/simdev/report/results.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ResultRecord:
    case_name: str
    spec_hash: str
    timestamp: str
    converged: bool
    cd_mean: float
    cd_std: float
    cl_mean: float
    cl_std: float
    window_start: int
    window_end: int
    n_iterations: int
    n_cells: int
    yplus_passed: bool
    yplus: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _dir(run_dir: Path) -> Path:
    target = Path(run_dir) / "results"
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_result(run_dir: Path, record: ResultRecord) -> tuple[Path, Path]:
    """Write this run's record. Never appends to a shared file."""
    target = _dir(run_dir)
    json_path = target / "result.json"
    csv_path = target / "result.csv"

    json_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

    flat = {k: v for k, v in asdict(record).items() if not isinstance(v, (dict, list))}
    flat["reasons"] = " | ".join(record.reasons)
    for patch, value in record.yplus.items():
        flat[f"yplus_{patch}"] = value
    pd.DataFrame([flat]).to_csv(csv_path, index=False)

    return json_path, csv_path


def read_result(run_dir: Path) -> ResultRecord:
    payload = json.loads(
        (Path(run_dir) / "results" / "result.json").read_text(encoding="utf-8")
    )
    return ResultRecord(**payload)


def aggregate(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """Combine per-run records on read. This replaces shared-append tables."""
    rows = []
    for run_dir in run_dirs:
        record = read_result(run_dir)
        row = {k: v for k, v in asdict(record).items() if not isinstance(v, (dict, list))}
        row["run_dir"] = str(run_dir)
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Write the plots module**

```python
# pipeline/simdev/report/plots.py
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_force_history(
    df: pd.DataFrame, out_path: Path, window: tuple[int, int]
) -> Path:
    """Cd and Cl against iteration, with the averaging window shaded."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

    for axis, column in zip(axes, ("Cd", "Cl")):
        if column not in df:
            continue
        axis.plot(df["Time"], df[column], linewidth=1.0)
        axis.axvspan(window[0], window[1], alpha=0.15, label="averaging window")
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_residuals(df: pd.DataFrame, out_path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(9, 4))

    for column in df.columns:
        if column == "Time" or not column.endswith("_initial"):
            continue
        axis.semilogy(df["Time"], df[column], linewidth=1.0, label=column)

    axis.set_xlabel("iteration")
    axis.set_ylabel("initial residual")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="best", fontsize="small")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
```

- [ ] **Step 5: Write the failing test for the post stage**

```python
# tests/test_post.py
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from simdev.run.runner import StageError
from simdev.run.status import StageStatus, read_status, write_status
from simdev.stages.post import post
from simdev.stages.prepare import prepare

FIXTURES = Path(__file__).parent / "fixtures"

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
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

    for name, fixture in (
        ("forceCoeffs", "coefficient.dat"),
        ("yPlus", "yPlus.dat"),
    ):
        out = target / "postProcessing" / name / "0"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIXTURES / fixture, out / fixture)

    write_status(
        target, StageStatus("mesh", "ok", "h", [], {"n_cells": 1122334})
    )
    return target


def _mark_solve(run_dir: Path, state: str) -> None:
    from simdev.stages.common import load_spec

    write_status(
        run_dir,
        StageStatus("solve", state, load_spec(run_dir).spec_hash(), [], {}),
    )


def test_post_writes_a_result(run_dir: Path) -> None:
    _mark_solve(run_dir, "ok")
    record = post(run_dir)
    assert (run_dir / "results" / "result.json").exists()
    assert record.n_cells == 1122334


def test_post_runs_after_a_non_converged_solve(run_dir: Path) -> None:
    # A run that produced numbers must not vanish because it did not plateau.
    _mark_solve(run_dir, "gate_failed")
    record = post(run_dir)
    assert record.converged is False
    assert record.cd_mean != 0.0


def test_post_refuses_after_a_failed_solve(run_dir: Path) -> None:
    _mark_solve(run_dir, "failed")
    with pytest.raises(StageError):
        post(run_dir)


def test_post_applies_the_y_plus_gate(run_dir: Path) -> None:
    _mark_solve(run_dir, "ok")
    record = post(run_dir)
    # Fixture body y+ averages 92.7, inside the high_y_plus band.
    assert record.yplus_passed is True
    assert record.yplus["body"] == pytest.approx(92.7)


def test_post_writes_plots(run_dir: Path) -> None:
    _mark_solve(run_dir, "ok")
    post(run_dir)
    assert (run_dir / "results" / "forces.png").exists()


def test_post_records_the_spec_hash(run_dir: Path) -> None:
    from simdev.stages.common import load_spec

    _mark_solve(run_dir, "ok")
    record = post(run_dir)
    assert record.spec_hash == load_spec(run_dir).spec_hash()


def test_post_status_is_recorded(run_dir: Path) -> None:
    _mark_solve(run_dir, "ok")
    post(run_dir)
    assert read_status(run_dir, "post").state == "ok"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_post.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.stages.post'`

- [ ] **Step 7: Write the post stage**

```python
# pipeline/simdev/stages/post.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from simdev.gates.convergence import check_convergence
from simdev.gates.yplus import check_y_plus
from simdev.report.plots import plot_force_history, plot_residuals
from simdev.report.results import ResultRecord, write_result
from simdev.run.parsers import read_force_coeffs, read_y_plus
from simdev.run.runner import StageError
from simdev.run.status import StageStatus, read_status, should_skip, write_status
from simdev.stages.common import find_latest, load_spec

STAGE = "post"


def post(run_dir: Path, force: bool = False) -> ResultRecord:
    run_dir = Path(run_dir)
    spec = load_spec(run_dir)

    solve_status = read_status(run_dir, "solve")
    if solve_status is None:
        raise StageError(["stage 'solve' has not been run"])
    if solve_status.state == "failed":
        raise StageError(
            ["stage 'solve' failed outright; there is nothing to post-process"]
        )
    # 'gate_failed' is acceptable: a non-converged run still produced numbers.

    if should_skip(run_dir, STAGE, spec.spec_hash(), force):
        from simdev.report.results import read_result

        return read_result(run_dir)

    forces = read_force_coeffs(find_latest(run_dir, "forceCoeffs/*/coefficient.dat"))
    convergence = check_convergence(forces, spec)

    y_plus_df = read_y_plus(find_latest(run_dir, "yPlus/*/yPlus.dat"))
    y_plus_gate = check_y_plus(y_plus_df, spec)

    mesh_status = read_status(run_dir, "mesh")
    n_cells = int(mesh_status.detail.get("n_cells", 0)) if mesh_status else 0

    results_dir = run_dir / "results"
    plot_force_history(forces, results_dir / "forces.png", convergence.window)
    try:
        residuals = read_force_coeffs(find_latest(run_dir, "solverInfo/*/solverInfo.dat"))
        plot_residuals(residuals, results_dir / "residuals.png")
    except FileNotFoundError:
        pass

    record = ResultRecord(
        case_name=spec.name,
        spec_hash=spec.spec_hash(),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        converged=convergence.converged,
        cd_mean=convergence.means.get("Cd", float("nan")),
        cd_std=convergence.stds.get("Cd", float("nan")),
        cl_mean=convergence.means.get("Cl", float("nan")),
        cl_std=convergence.stds.get("Cl", float("nan")),
        window_start=convergence.window[0],
        window_end=convergence.window[1],
        n_iterations=convergence.n_iterations,
        n_cells=n_cells,
        yplus_passed=y_plus_gate.passed,
        yplus={
            k.replace("_avg_yplus", ""): float(v)
            for k, v in y_plus_gate.detail.items()
        },
        reasons=[*convergence.reasons, *y_plus_gate.reasons],
    )
    write_result(run_dir, record)

    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            state="ok" if y_plus_gate.passed else "gate_failed",
            input_hash=spec.spec_hash(),
            reasons=record.reasons,
            detail={"cd_mean": record.cd_mean, "cl_mean": record.cl_mean},
        ),
    )
    return record
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_results.py tests/test_post.py -v`
Expected: PASS (13 tests)

- [ ] **Step 9: Commit**

```bash
git add pipeline/simdev/report pipeline/simdev/stages/post.py tests/test_results.py tests/test_post.py
git commit -m "feat: add post stage with per-run result records and plots"
```

---

## Task 17: CLI, Ahmed case, and the smoke test

**Files:**
- Create: `pipeline/simdev/cli.py`, `cases/ahmed/config.yaml`
- Test: `tests/test_cli.py`, `tests/test_smoke.py`

**Interfaces:**
- Consumes: all four stages
- Produces:
  - `main(argv: list[str] | None = None) -> int` — console entry point `simdev`
  - Subcommands: `prepare`, `mesh`, `solve`, `post`, `run`, `doctor`, `aggregate`

`run` chains all four stages and stops at the first that raises. `doctor` reports whether the OpenFOAM utilities the pipeline calls are on `PATH` — the fastest way to diagnose a broken environment.

- [ ] **Step 1: Write the Ahmed case config**

```yaml
# cases/ahmed/config.yaml
# Ahmed body validation case.
#
# Reference: Ahmed, Ramm & Faltin, SAE 840300 (1984).
# 35 degree slant, chosen over 25 degrees because the 25 degree case sits on
# the separation/reattachment bifurcation where steady RANS fails for
# turbulence-model reasons, not pipeline reasons.
#
# Fixed floor, matching the reference experiment's stationary tunnel wall.
# This is why ground motion is a per-case field, not role behaviour.
#
# acceptance.target_cd is confirmed against the primary source in Task 18.
name: ahmed

flow:
  u_inf: 40.0
  nu: 1.5e-05
  rho: 1.225
  turbulence_intensity: 0.005
  turbulence_length_scale: 0.01

ground:
  motion: static

physics:
  turbulence_model: kOmegaSST
  wall_treatment: high_y_plus
  mode: straight
  yaw_deg: 0.0

domain:
  kind: box
  upstream_lengths: 5.0
  downstream_lengths: 10.0
  half_width_lengths: 3.0
  height_lengths: 3.0
  max_blockage: 0.01

forces:
  a_ref_full: 0.112032   # width 0.389 x height 0.288, stilts excluded
  l_ref: 1.044
  c_of_r: [0.522, 0.0, 0.0]

geometry:
  kind: ahmed
  ahmed:
    length: 1.044
    width: 0.389
    height: 0.288
    slant_angle_deg: 35.0
    slant_length: 0.222
    nose_radius: 0.100
    ground_clearance: 0.050
    stilt_diameter: 0.030
    include_stilts: true
  patches:
    - {name: body, role: body}
    - {name: stilts, role: body}
    - {name: ground, role: ground}
    - {name: symmetry, role: symmetry}
    - {name: inlet, role: inlet}
    - {name: outlet, role: outlet}
    - {name: farfield, role: farfield}
```

- [ ] **Step 2: Write the failing CLI test**

```python
# tests/test_cli.py
from __future__ import annotations

from pathlib import Path

import pytest

from simdev.cli import main

CASE = Path("cases/ahmed/config.yaml")


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_prepare_builds_a_case(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert main(["prepare", str(CASE), "--run-dir", str(run_dir), "--profile", "dev"]) == 0
    assert (run_dir / "system" / "controlDict").exists()


def test_prepare_accepts_a_wall_treatment_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    main(
        [
            "prepare",
            str(CASE),
            "--run-dir",
            str(run_dir),
            "--profile",
            "dev",
            "--wall-treatment",
            "low_y_plus",
        ]
    )
    text = (run_dir / "0" / "nut").read_text(encoding="utf-8")
    assert "nutLowReWallFunction" in text


def test_invalid_case_returns_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: broken\n", encoding="utf-8")
    assert main(["prepare", str(bad), "--run-dir", str(tmp_path / "r")]) != 0


def test_doctor_reports_without_crashing(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert "simpleFoam" in out
    assert "surfaceFeatures" in out
    assert code in (0, 1)


def test_unknown_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["nonsense"])
    assert exc.value.code != 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simdev.cli'`

- [ ] **Step 4: Write the CLI**

```python
# pipeline/simdev/cli.py
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from simdev.config.validate import ValidationError
from simdev.report.results import aggregate
from simdev.run.runner import StageError
from simdev.stages.mesh import mesh
from simdev.stages.post import post
from simdev.stages.prepare import prepare
from simdev.stages.solve import solve

REQUIRED_UTILITIES = (
    "blockMesh",
    "surfaceFeatures",
    "snappyHexMesh",
    "checkMesh",
    "decomposePar",
    "simpleFoam",
    "mpirun",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simdev", description="OpenFOAM CFD pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("prepare", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("case", type=Path)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--profile", default="dev")
        sub.add_argument("--wall-treatment", default=None)
        sub.add_argument("--force", action="store_true")

    for name in ("mesh", "solve", "post"):
        sub = subparsers.add_parser(name)
        sub.add_argument("run_dir", type=Path)
        sub.add_argument("--force", action="store_true")

    subparsers.add_parser("doctor")

    agg = subparsers.add_parser("aggregate")
    agg.add_argument("run_dirs", type=Path, nargs="+")
    agg.add_argument("--out", type=Path, default=None)

    return parser


def _doctor() -> int:
    missing = []
    for utility in REQUIRED_UTILITIES:
        location = shutil.which(utility)
        print(f"{utility:18s} {location or 'NOT FOUND'}")
        if location is None:
            missing.append(utility)

    if shutil.which("surfaceFeatureExtract") and not shutil.which("surfaceFeatures"):
        print(
            "\nOnly the legacy 'surfaceFeatureExtract' is present. "
            "This pipeline targets ESI OpenFOAM v2412, which provides "
            "'surfaceFeatures'."
        )

    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print("See docs/environment-setup.md")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        if args.command == "doctor":
            return _doctor()

        if args.command == "aggregate":
            frame = aggregate(args.run_dirs)
            if args.out:
                frame.to_csv(args.out, index=False)
            print(frame.to_string(index=False))
            return 0

        if args.command in ("prepare", "run"):
            prepare(
                args.case,
                args.run_dir,
                profile=args.profile,
                wall_treatment=args.wall_treatment,
                force=args.force,
            )
            if args.command == "prepare":
                return 0

            mesh(args.run_dir, force=args.force)
            result = solve(args.run_dir, force=args.force)
            record = post(args.run_dir, force=args.force)
            print(
                f"Cd = {record.cd_mean:.4f} +/- {record.cd_std:.4f}   "
                f"Cl = {record.cl_mean:.4f} +/- {record.cl_std:.4f}   "
                f"converged = {record.converged}"
            )
            return 0 if result.converged and record.yplus_passed else 1

        if args.command == "mesh":
            mesh(args.run_dir, force=args.force)
            return 0
        if args.command == "solve":
            return 0 if solve(args.run_dir, force=args.force).converged else 1
        if args.command == "post":
            return 0 if post(args.run_dir, force=args.force).yplus_passed else 1

    except (StageError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    return 0
```

- [ ] **Step 5: Write the smoke test**

```python
# tests/test_smoke.py
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from simdev.cli import main

CASE = Path("cases/ahmed/config.yaml")

pytestmark = pytest.mark.skipif(
    shutil.which("simpleFoam") is None, reason="requires OpenFOAM on PATH"
)


@pytest.mark.openfoam
def test_full_chain_runs_end_to_end(tmp_path: Path) -> None:
    """The whole pipeline on a deliberately tiny mesh. Target: 1-2 minutes."""
    run_dir = tmp_path / "run"
    code = main(
        [
            "run",
            str(CASE),
            "--run-dir",
            str(run_dir),
            "--profile",
            "dev",
        ]
    )

    # A dev-profile run is too coarse to converge; exit 1 is expected and fine.
    assert code in (0, 1)

    assert (run_dir / "results" / "result.json").exists()
    assert (run_dir / "results" / "forces.png").exists()
    assert (run_dir / "logs" / "log.snappyHexMesh").exists()

    from simdev.report.results import read_result

    record = read_result(run_dir)
    assert record.n_cells > 0
    assert record.cd_mean == record.cd_mean  # not NaN
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (6 tests)

Run: `python -m pytest tests/test_smoke.py -v -m openfoam`
Expected: PASS if OpenFOAM is installed, SKIPPED otherwise. Fix any failure here before Task 18 — this is the first time real OpenFOAM touches the generated dictionaries, so template syntax errors surface now.

- [ ] **Step 7: Commit**

```bash
git add pipeline/simdev/cli.py cases/ahmed/config.yaml tests/test_cli.py tests/test_smoke.py
git commit -m "feat: add CLI, Ahmed validation case and end-to-end smoke test"
```

---

## Task 18: Ahmed validation and mesh independence

**Files:**
- Create: `docs/validation-ahmed.md`, `scripts/mesh_independence.py`
- Modify: `cases/ahmed/config.yaml` — add the `acceptance` block
- Test: `tests/test_mesh_independence.py`

**Interfaces:**
- Consumes: `aggregate` (Task 16), CLI (Task 17)
- Produces:
  - `monotonic_convergence(values: list[float]) -> bool`
  - `within_tolerance(value: float, target: float, tolerance: float) -> bool`
  - `run_levels(case: Path, out_root: Path, levels: dict[str, dict]) -> pandas.DataFrame`

This is the task that makes the slice's claim testable. **Its first step is not code.**

- [ ] **Step 1: Pin the reference from the primary source**

Open Ahmed, Ramm & Faltin, SAE 840300 (1984), and record in `docs/validation-ahmed.md`:

1. The free-stream velocity and the Reynolds number it corresponds to. **Both Re ~ 2.8x10^6 (40 m/s) and 4.29x10^6 appear in the literature**; the CFD comparison convention is the former. Record which one the target Cd belongs to.
2. The reference area used to non-dimensionalise the published Cd, and **whether the stilts are included in it**.
3. The target Cd for the 35 degree slant, with a full citation.
4. The ground condition of the experiment (expected: stationary floor).

Then add to `cases/ahmed/config.yaml`:

```yaml
acceptance:
  target_cd: <value from the source>
  tolerance: 0.10
  source: "Ahmed, Ramm & Faltin, SAE 840300 (1984), <table/figure>"
```

A validation without a named reference number is not a validation. Do not proceed until these four values are written down.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_mesh_independence.py
from __future__ import annotations

import pytest

from scripts.mesh_independence import monotonic_convergence, within_tolerance


def test_monotonic_increasing_series_is_monotonic() -> None:
    assert monotonic_convergence([0.30, 0.33, 0.345]) is True


def test_monotonic_decreasing_series_is_monotonic() -> None:
    assert monotonic_convergence([0.40, 0.36, 0.348]) is True


def test_oscillating_series_is_not_monotonic() -> None:
    assert monotonic_convergence([0.30, 0.40, 0.33]) is False


def test_two_points_are_insufficient() -> None:
    assert monotonic_convergence([0.30, 0.35]) is False


def test_within_tolerance_accepts_a_ten_percent_error() -> None:
    assert within_tolerance(0.32, target=0.30, tolerance=0.10) is True


def test_within_tolerance_rejects_a_twenty_percent_error() -> None:
    assert within_tolerance(0.36, target=0.30, tolerance=0.10) is False


@pytest.mark.parametrize("value", [0.30, 0.33, 0.27])
def test_tolerance_band_is_symmetric(value: float) -> None:
    assert within_tolerance(value, target=0.30, tolerance=0.10) is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_independence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.mesh_independence'`

- [ ] **Step 4: Write the script**

Create `scripts/__init__.py` (empty) and `scripts/mesh_independence.py`:

```python
# scripts/mesh_independence.py
"""Run the Ahmed case at three refinement levels and check grid convergence."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from simdev.cli import main as cli_main
from simdev.report.results import aggregate

LEVELS: dict[str, dict[str, object]] = {
    "coarse": {"base_cell_size": 0.08, "refinement": (3, 4)},
    "medium": {"base_cell_size": 0.05, "refinement": (4, 5)},
    "fine": {"base_cell_size": 0.03, "refinement": (5, 6)},
}


def monotonic_convergence(values: list[float]) -> bool:
    """True if the series moves consistently in one direction.

    Needs at least three levels: two points can always be joined by a line.
    """
    if len(values) < 3:
        return False
    deltas = [b - a for a, b in zip(values, values[1:])]
    return all(d > 0 for d in deltas) or all(d < 0 for d in deltas)


def within_tolerance(value: float, target: float, tolerance: float) -> bool:
    return abs(value - target) <= abs(target) * tolerance


def run_levels(case: Path, out_root: Path) -> pd.DataFrame:
    run_dirs: list[Path] = []
    for name, settings in LEVELS.items():
        run_dir = Path(out_root) / name
        cli_main(
            [
                "run",
                str(case),
                "--run-dir",
                str(run_dir),
                "--profile",
                "production",
            ]
        )
        run_dirs.append(run_dir)
    return aggregate(run_dirs)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--target-cd", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args()

    frame = run_levels(args.case, args.out_root)
    print(frame.to_string(index=False))

    cds = list(frame["cd_mean"])
    monotonic = monotonic_convergence(cds)
    accurate = within_tolerance(cds[-1], args.target_cd, args.tolerance)

    print(f"\nmonotonic convergence: {monotonic}")
    print(
        f"finest Cd {cds[-1]:.4f} vs target {args.target_cd:.4f} "
        f"({args.tolerance:.0%}): {accurate}"
    )
    return 0 if (monotonic and accurate) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
```

To vary refinement per level, pass the overrides through the config layer rather than editing files: extend `run_levels` to write a temporary case YAML per level with `mesh.base_cell_size` and the refinement pair merged in, using `simdev.config.resolve.deep_merge` on the loaded case dict.

- [ ] **Step 5: Run the unit tests**

Run: `python -m pytest tests/test_mesh_independence.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Run the real validation**

Run:

```bash
python scripts/mesh_independence.py cases/ahmed/config.yaml \
    --out-root ~/runs/ahmed-independence \
    --target-cd <pinned value> \
    --tolerance 0.10
```

Expected: three runs complete; Cd converges monotonically; the finest level lands within 10 % of the pinned target. Record the actual numbers in `docs/validation-ahmed.md`.

If it fails, diagnose in this order before touching the turbulence model:
1. y+ gate output — is the wall treatment actually valid?
2. Layer coverage on `body` — did snappy deliver the layers?
3. Blockage ratio and domain extents.
4. The nose fillet approximation in `geometry/ahmed.py` (documented in Task 6).

- [ ] **Step 7: Write up the result**

Complete `docs/validation-ahmed.md` with: the pinned reference and citation, the three levels and their cell counts, the Cd and Cl at each, the convergence verdict, and any deviation with its explanation.

State the limitation explicitly, in the document:

> Ahmed at Re ~ 2.8x10^6 with `high_y_plus` validates the pipeline's plumbing,
> numerics, domain construction, force integration and gates. It does **not**
> validate the low-Re, wall-resolved settings the RC car will use. Those
> require separate justification and, ideally, comparison against track or
> tunnel data for the actual vehicle.

- [ ] **Step 8: Commit**

```bash
git add scripts docs/validation-ahmed.md cases/ahmed/config.yaml tests/test_mesh_independence.py
git commit -m "feat: add Ahmed validation with mesh independence check"
```

---

## Self-Review

**Spec coverage.** Every section of the design spec maps to a task:

| Spec section | Task |
|---|---|
| §3 environment, WSL, 40 cores | 1 (docs), 4 (rank profiles), 5 (rank assertion) |
| §4.1 regime, y+ estimates | 5 (`estimate_y_plus`) |
| §4.2 solver, turbulence, wall profiles, ground field | 3, 4, 10 |
| §5.1 CaseSpec, provenance | 3 (`spec_hash`), 10 (`caseSpec.json`) |
| §5.2 repo layout | File Structure section |
| §5.3 patch roles | 2 |
| §5.4 mode / symmetry table | 3 (`half_model`), 5 (assertions), 8 (domain) |
| §5.5 validation assertions | 5, plus blockage in 8 |
| §5.6 profiles | 4 |
| §6 stages | 14, 15, 16 |
| §7 gates | 12, wired in 15 and 16 |
| §8 failure handling, per-run results | 13 (`find_fatal_errors`, `should_skip`), 16 |
| §9 validation | 18 |
| §10 testing | every task; smoke in 17 |
| §11 slice scope | 17 CLI + case config |
| §12 success criteria | 17 (criteria 1, 4), 18 (criterion 3), 12+15 (criterion 2) |
| §13 dependencies | 1, plus shapely in 7 |

**Deferred items are correctly absent**: no annulus domain, no MRF, no attitude, no transition model — but `PatchRole.TYRE` / `MRF_ZONE` exist (Task 2), `DomainBuilder` is a protocol with one implementation (Task 8), `Mode.CORNERING` validates and rejects cleanly (Task 5), and `turbulence_model` is already config-selected (Task 3).

**Corrections applied during writing:**
- `HIGH_Y_PLUS` first-layer thickness moved from 3.0e-4 to 1.0e-3. At the Ahmed condition the original value implies y+ ~ 15, which is in the buffer layer and outside the band the profile claims — it would have failed its own validator.
- Moving ground renders as `fixedValue uniform (U 0 0)`, not `movingWallVelocity`. The spec's shorthand named a BC intended for moving *meshes*; on a static mesh with a translating road, `fixedValue` is the correct expression of the same physics (Task 10).
- `require_stage` is deliberately **not** used by the post stage. A `gate_failed` solve is valid input for post-processing, because a non-converged run still produced numbers that must be recorded and flagged (Task 16).

**Type consistency checked** across tasks: `GateResult` (Task 12) is returned by `check_mesh_quality`, `check_y_plus` and `mesh`; `ConvergenceResult` by `check_convergence` and `solve`; `CheckMeshResult` / `LayerInfo` (Task 11) are consumed by Task 12 with matching field names; `StageStatus.state` uses the same three literals in Tasks 13, 15, 16; `spec_hash()` is the input hash everywhere.

**Known limitation carried forward, not hidden:** the Ahmed nose is a lofted approximation of the 100 mm fillet rather than a true spherical fillet (Task 6). It is the fourth diagnostic step in Task 18 if validation misses.
