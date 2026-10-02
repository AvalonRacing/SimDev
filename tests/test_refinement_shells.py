"""Refinement shells: volume refinement shaped like the car, not like a box.

Surface refinement only thickens the mesh next to the wall. A cell or two off
the bodywork the flow is back at background size, and on this car that
includes the air moving *through* it - between body, chassis and wishbones.
A box cannot fix it here, because a cornering wake leaves any axis-aligned box
and the car is posed at a slip angle besides.

What has to be true is small and checkable: the distances reach where they
say, the order snappy needs is produced whatever order the case declares, the
surface used for the measurement never becomes a patch, and the ground under
the car knows it has been refined.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simdev.config.resolve import load_case as _load_case
from tests.test_split_driving_states import TESTCASE_STATE
from simdev.config.validate import validate
from simdev.render.context import (
    VEHICLE_SURFACE,
    ground_cell_size,
    refinement_shells,
)
from simdev.stages.prepare import prepare

from tests.test_car_prepare import CORNERS, car_case, write_car

CASE = Path(__file__).resolve().parents[1] / "cases" / "car" / "config.yaml"


def load_case(path, profile, wall_treatment, overrides):
    """The car case, in the driving state that used to be selected inline."""
    return _load_case(path, profile, wall_treatment, overrides, TESTCASE_STATE)


def _block(text: str, name: str) -> str:
    """The body of a top-level dictionary entry, by brace matching.

    Splitting on the keyword is not good enough: the word `refinementSurfaces`
    also appears in a comment explaining what is deliberately *not* in it, and
    a naive split lands in the comment and quietly asserts nothing.
    """
    start = next(
        i
        for i, line in enumerate(text.splitlines())
        if line.strip() == name
    )
    lines = text.splitlines()[start:]
    depth = 0
    out: list[str] = []
    for line in lines:
        out.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0 and len(out) > 1:
            break
    return "\n".join(out)


def _rendered_shells(text: str) -> list[tuple[float, int]]:
    """The (distance, level) pairs snappy will read, in the order it reads them."""
    regions = _block(text, "refinementRegions")
    body = _block(regions, VEHICLE_SURFACE)
    pairs = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("(") and not stripped.startswith("(("):
            distance, level = stripped.strip("()").split()
            pairs.append((float(distance), int(level)))
    return pairs


@pytest.fixture
def shelled(tmp_path: Path):
    """The multi-part test car, with shells declared in body lengths."""
    stl_dir = tmp_path / "cad"
    write_car(stl_dir)

    def run(name: str = "run", shells=None, **overrides):
        case = car_case(stl_dir, **overrides)
        if shells is not None:
            case["domain"]["refinement_shells"] = shells
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(case), encoding="utf-8")
        return prepare(path, tmp_path / name, profile="car_dev")

    return run


# --- resolution to metres -------------------------------------------------


def test_distances_are_body_lengths_resolved_against_the_geometry() -> None:
    """Declared in body lengths so a shell means the same thing at any scale."""
    spec = load_case(CASE, "car", None, None)
    assert spec.domain.refinement_shells, "the car case should declare shells"

    from simdev.domain.annulus import AnnulusDomainBuilder

    lo = [-0.22, -0.15, 0.0]
    hi = [0.22, 0.15, 0.12]
    domain = AnnulusDomainBuilder().build(spec, (lo, hi))
    resolved = refinement_shells(spec, domain)

    length = domain.geom_length
    for shell, declared in zip(resolved, sorted(
        spec.domain.refinement_shells, key=lambda s: s.distance
    )):
        assert shell["distance"] == pytest.approx(declared.distance * length)
        assert shell["level"] == declared.level


def test_shells_come_out_nearest_first_whatever_order_they_were_written(
    shelled,
) -> None:
    """snappy applies the first shell containing a cell, so order is load-bearing.

    Declared coarsest-first, an unsorted render would put the outer shell in
    front and the fine ones would never apply - a silently coarser mesh, not
    an error.
    """
    result = shelled(
        "order",
        shells=[
            {"distance": 0.9, "level": 3},
            {"distance": 0.12, "level": 5},
            {"distance": 0.35, "level": 4},
        ],
    )
    snappy = (result.run_dir / "system" / "snappyHexMeshDict").read_text(
        encoding="utf-8"
    )
    pairs = _rendered_shells(snappy)

    assert [level for _, level in pairs] == [5, 4, 3]
    distances = [distance for distance, _ in pairs]
    assert distances == sorted(distances)


# --- the surface used for the measurement ---------------------------------


def test_the_vehicle_surface_is_written_and_covers_every_wall(shelled) -> None:
    result = shelled("surface", shells=[{"distance": 0.2, "level": 3}])
    written = result.run_dir / "constant" / "triSurface" / f"{VEHICLE_SURFACE}.stl"

    assert written.exists()

    import trimesh

    combined = trimesh.load_mesh(written, process=True)
    walls = ["Body", *[f"Tire_{w}" for w in CORNERS]]
    total = 0
    for name in walls:
        part = trimesh.load_mesh(
            result.run_dir / "constant" / "triSurface" / f"{name}.stl", process=True
        )
        total += len(part.faces)
    assert len(combined.faces) == total


def test_the_vehicle_surface_never_becomes_a_patch(shelled) -> None:
    """It is a distance field, not geometry to mesh against.

    Listed in refinementSurfaces it would duplicate every wall in the model,
    create a patch with no boundary condition, and enter force integration.
    """
    result = shelled("nopatch", shells=[{"distance": 0.2, "level": 3}])
    snappy = (result.run_dir / "system" / "snappyHexMeshDict").read_text(
        encoding="utf-8"
    )

    assert VEHICLE_SURFACE not in _block(snappy, "refinementSurfaces")

    # Present as geometry and as a region, though.
    assert f'file            "{VEHICLE_SURFACE}.stl"' in snappy
    assert "mode            distance" in snappy

    # And no boundary condition anywhere.
    field = (result.run_dir / "0" / "U").read_text(encoding="utf-8")
    assert VEHICLE_SURFACE not in field


def test_the_mrf_sleeves_are_left_out_of_the_vehicle_surface(shelled) -> None:
    """They are closed volumes inside the tyres, not wetted surface.

    Including them would put a surface in the middle of each wheel and drag a
    shell of fine cells into rubber, where there is no flow to resolve.
    """
    import trimesh

    result = shelled("nomrf", shells=[{"distance": 0.2, "level": 3}])
    tri = result.run_dir / "constant" / "triSurface"
    combined = trimesh.load_mesh(tri / f"{VEHICLE_SURFACE}.stl", process=True)

    walls = sum(
        len(trimesh.load_mesh(tri / f"{n}.stl", process=True).faces)
        for n in ["Body", *[f"Tire_{w}" for w in CORNERS]]
    )
    sleeves = sum(
        len(trimesh.load_mesh(tri / f"MRF_{w}.stl", process=True).faces)
        for w in CORNERS
    )
    assert len(combined.faces) == walls
    assert len(combined.faces) != walls + sleeves


def test_no_shells_means_no_surface_and_no_region(shelled) -> None:
    result = shelled("noshells", shells=[])
    snappy = (result.run_dir / "system" / "snappyHexMeshDict").read_text(
        encoding="utf-8"
    )

    assert not (
        result.run_dir / "constant" / "triSurface" / f"{VEHICLE_SURFACE}.stl"
    ).exists()
    assert VEHICLE_SURFACE not in snappy


# --- the ground under the car --------------------------------------------


def test_a_shell_thicker_than_the_ride_height_refines_the_ground() -> None:
    """The ground is a blockMesh patch, so only volume refinement reaches it.

    Its prism stack is budgeted against whatever cell it actually sits in. Size
    that against the raw background and it asks for a stack that does not fit
    the far finer cell the shell put there.
    """
    spec = load_case(CASE, "car", None, None)
    from simdev.domain.annulus import AnnulusDomainBuilder

    # Car floating 10 mm off the road: every shell here is thicker than that.
    domain = AnnulusDomainBuilder().build(
        spec, ([-0.22, -0.15, 0.010], [0.22, 0.15, 0.12])
    )
    finest = max(s.level for s in spec.domain.refinement_shells)

    assert ground_cell_size(spec, domain) == pytest.approx(
        spec.mesh.base_cell_size / 2**finest
    )


def test_a_shell_thinner_than_the_ride_height_leaves_the_ground_alone() -> None:
    spec = load_case(
        CASE,
        "car",
        None,
        {"domain": {"refinement_shells": [{"distance": 0.001, "level": 5}]}},
    )
    from simdev.domain.annulus import AnnulusDomainBuilder

    domain = AnnulusDomainBuilder().build(
        spec, ([-0.22, -0.15, 0.10], [0.22, 0.15, 0.30])
    )
    assert ground_cell_size(spec, domain) == pytest.approx(spec.mesh.base_cell_size)


# --- what the validator refuses ------------------------------------------


def test_a_shell_that_gets_finer_with_distance_is_rejected() -> None:
    """snappy would mesh it happily and ignore the outer level in silence."""
    spec = load_case(
        CASE,
        "car",
        None,
        {
            "domain": {
                "refinement_shells": [
                    {"distance": 0.1, "level": 3},
                    {"distance": 0.5, "level": 5},
                ]
            }
        },
    )
    with pytest.raises(Exception) as caught:
        validate(spec)
    assert "coarser with distance" in str(caught.value)


def test_a_case_with_no_shells_is_warned_about() -> None:
    spec = load_case(CASE, "car", None, {"domain": {"refinement_shells": []}})
    assert any("no domain.refinement_shells" in w for w in validate(spec))


def test_the_far_wake_limitation_is_warned_about_without_a_wake_region() -> None:
    """Shells follow the car, not the path - domain.wake is what covers it."""
    spec = load_case(CASE, "car", None, {"domain": {"wake": None}})
    assert any("far wake" in w for w in validate(spec))


def test_a_declared_wake_region_silences_the_far_wake_warning() -> None:
    spec = load_case(CASE, "car", None, None)
    assert spec.domain.wake is not None
    assert not any("far wake" in w for w in validate(spec))
