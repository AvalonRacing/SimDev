"""Two settings that buy meshing time without buying a coarser mesh.

snappyHexMesh's cost is not proportional to the cell count it produces. It is
proportional to how much geometry querying each refinement iteration does, and
to how evenly that work is spread over the ranks. This case is unusually
exposed on both counts: 6.6 M cells packed into a 0.4 m ball inside a 25.7 m
domain, refined against a combined vehicle surface of ~600k triangles.

Neither setting here changes the resolution anywhere in the mesh.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simdev.render.context import VEHICLE_SURFACE
from simdev.stages.prepare import prepare

from tests.test_car_prepare import CORNERS, car_case, write_car
from tests.test_refinement_shells import _block

SHELLS = [{"distance": 0.2, "level": 3}]


@pytest.fixture
def car(tmp_path: Path):
    """The multi-part test car, prepared with shells so a shell surface exists."""
    stl_dir = tmp_path / "cad"
    write_car(stl_dir)

    def run(name: str = "run", shells=SHELLS, **overrides):
        case = car_case(stl_dir, **overrides)
        case["domain"]["refinement_shells"] = shells
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(case), encoding="utf-8")
        return prepare(path, tmp_path / name, profile="car_dev")

    return run


def _snappy(result) -> str:
    return (result.run_dir / "system" / "snappyHexMeshDict").read_text(
        encoding="utf-8"
    )


def _entry(block: str, key: str) -> str:
    """The value of a `key   value;` line, without depending on the alignment.

    The dict is column-aligned for reading, and asserting on the exact run of
    spaces makes a test that fails when someone adds a longer keyword.
    """
    for line in block.splitlines():
        fields = line.strip().rstrip(";").split()
        if len(fields) == 2 and fields[0] == key:
            return fields[1]
    raise AssertionError(f"no entry {key!r} in:\n{block}")


# --- load balance during refinement ---------------------------------------


def test_max_load_unbalance_is_rendered_into_castellated_controls(car) -> None:
    """Left out of the dict, the balancing policy is whatever the build defaults to.

    The refinement here is about as localised as external aero gets, so which
    policy applies is not a detail: the ranks that own the car do essentially
    all the castellation work while the rest idle.
    """
    result = car("balance", **{"mesh.max_load_unbalance": 0.05})
    castellated = _block(_snappy(result), "castellatedMeshControls")

    assert float(_entry(castellated, "maxLoadUnbalance")) == 0.05


def test_max_load_unbalance_defaults_to_the_tutorial_value(car) -> None:
    """0.10 is what every OpenFOAM external-aero tutorial ships.

    Rendered explicitly rather than left out, so the balancing policy is a
    property of the case and not of whichever OpenFOAM build ran it.
    """
    result = car("balance_default")
    castellated = _block(_snappy(result), "castellatedMeshControls")

    assert float(_entry(castellated, "maxLoadUnbalance")) == 0.10


# --- what the distance field is measured against --------------------------
#
# The shell surface is the one surface in the case that exists only to be
# measured from: it is never snapped to, never becomes a patch, and never
# enters force integration. So it is also the one surface that can be
# simplified without changing the mesh anywhere - as long as it keeps the
# features whose *presence* pulls refinement around them.


def _faces(path: Path) -> int:
    import trimesh

    return len(trimesh.load_mesh(path, process=False).faces)


def _wall_faces(result) -> int:
    tri = result.run_dir / "constant" / "triSurface"
    return sum(
        _faces(tri / f"{name}.stl")
        for name in ["Body", *[f"Tire_{w}" for w in CORNERS]]
    )


def _shell_surface(result) -> Path:
    return result.run_dir / "constant" / "triSurface" / f"{VEHICLE_SURFACE}.stl"


def test_no_tolerance_leaves_the_shell_surface_exactly_as_drawn(car) -> None:
    """The default has to be off: it changes what a case meshes to."""
    result = car("asdrawn")

    assert _faces(_shell_surface(result)) == _wall_faces(result)


def test_a_tolerance_shrinks_the_shell_surface(car) -> None:
    result = car("shrunk", **{"domain.shell_surface_tolerance": 0.005})

    assert _faces(_shell_surface(result)) < _wall_faces(result)


def test_decimation_never_touches_the_surfaces_snappy_meshes_against(
    car,
) -> None:
    """Only the distance field is simplified. The walls are meshed and snapped
    to, so simplifying them would coarsen the geometry itself."""
    plain = car("walls_plain")
    thinned = car("walls_thinned", **{"domain.shell_surface_tolerance": 0.005})

    tri_plain = plain.run_dir / "constant" / "triSurface"
    tri_thinned = thinned.run_dir / "constant" / "triSurface"
    for name in ["Body", *[f"Tire_{w}" for w in CORNERS]]:
        assert (tri_plain / f"{name}.stl").read_bytes() == (
            tri_thinned / f"{name}.stl"
        ).read_bytes(), f"{name} was modified"


def test_the_shell_surface_stays_within_the_declared_tolerance(car) -> None:
    """The shell boundary may not move further than the tolerance allows.

    A shell is a distance from this surface, so an error here displaces the
    refinement boundary by the same amount.
    """
    import numpy as np
    import trimesh

    plain = car("bounds_plain")
    thinned = car("bounds_thinned", **{"domain.shell_surface_tolerance": 0.005})

    before = trimesh.load_mesh(_shell_surface(plain), process=False)
    after = trimesh.load_mesh(_shell_surface(thinned), process=False)

    assert np.all(np.abs(after.bounds - before.bounds) <= 0.005)


def test_a_tolerance_that_eats_thin_features_is_warned_about(car) -> None:
    """A feature thinner than the tolerance collapses out of the surface.

    That is not cosmetic. The gaps between body, chassis and wishbones are
    what the innermost shell exists to refine, and a wishbone that has
    vanished from the distance field stops pulling any refinement at all.
    Surface area is the cheap tell: a collapsed feature takes its area with
    it, while honest simplification of a curved panel does not.
    """
    result = car("toocoarse", **{"domain.shell_surface_tolerance": 0.02})

    assert any(
        "Tire_FL" in warning and "shell surface" in warning
        for warning in result.warnings
    ), result.warnings


def test_an_honest_tolerance_is_not_warned_about(car) -> None:
    result = car("honest", **{"domain.shell_surface_tolerance": 0.005})

    assert not [w for w in result.warnings if "shell surface" in w]


def test_what_the_distance_field_was_built_from_is_on_the_record(car) -> None:
    """A run directory should answer "why is it refined there" without
    re-deriving anything - including how much of the car the shells could
    actually see."""
    from simdev.run.status import read_status

    result = car("record", **{"domain.shell_surface_tolerance": 0.005})
    detail = read_status(result.run_dir, "prepare").detail["shell_surface"]

    assert detail["tolerance"] == 0.005
    assert detail["triangles_after"] < detail["triangles_before"]
    assert detail["triangles_before"] == _wall_faces(result)
