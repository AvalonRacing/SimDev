"""The MRF sleeve must intersect its tyre, never lie on it.

A sleeve is drawn to the same nominal diameter as the tyre bore, so CAD puts
the two surfaces in the same place. snappyHexMesh cannot resolve that: it has
to snap a wall and insert a faceZone at one location, and what comes out is
baffles, faceZones reported "multiply connected", non-manifold points and
intermittently a reversed face. Measured on the real car at 0.05 mm over three
quarters of the sleeve.

The repair is interference, not clearance - pushed *into* the tyre so the
surfaces plainly cross. Where the sleeve is buried in tyre material there are
no fluid cells, so no zone boundary exists there at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from simdev.geometry.mrf import (
    MrfGeometryError,
    outer_radius,
    push_into_tyre,
    surface_gap,
)

ORIGIN = np.array([0.0, 0.0, 0.033])
AXIS = np.array([0.0, 1.0, 0.0])
BORE = 0.027
TREAD = 0.033
WIDTH = 0.027


def _turned(radius: float, length: float) -> trimesh.Trimesh:
    """A cylinder on the wheel axis, like a machined sleeve."""
    transform = trimesh.transformations.translation_matrix(
        ORIGIN
    ) @ trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0])
    return trimesh.creation.cylinder(
        radius=radius, height=length, sections=64, transform=transform
    )


def tyre() -> trimesh.Trimesh:
    return trimesh.creation.annulus(
        r_min=BORE,
        r_max=TREAD,
        height=WIDTH,
        sections=64,
        transform=trimesh.transformations.translation_matrix(ORIGIN)
        @ trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0]),
    )


# --- measuring the fault --------------------------------------------------


def test_a_sleeve_on_the_bore_measures_as_touching() -> None:
    """The real car's case: sleeve radius equal to the tyre bore."""
    assert surface_gap(_turned(BORE, WIDTH), tyre()) < 1e-4


def test_a_sleeve_with_real_clearance_measures_as_clear() -> None:
    gap = surface_gap(_turned(BORE - 0.004, WIDTH - 0.004), tyre())
    assert gap == pytest.approx(0.004, abs=6e-4)


# --- the repair -----------------------------------------------------------


def test_the_sleeve_grows_radially_into_the_carcass() -> None:
    sleeve = _turned(BORE, WIDTH)
    moved = push_into_tyre(sleeve, ORIGIN, AXIS, 0.002)

    assert outer_radius(sleeve, ORIGIN, AXIS) == pytest.approx(BORE, abs=1e-4)
    assert outer_radius(moved, ORIGIN, AXIS) == pytest.approx(BORE + 0.002, abs=1e-4)


def test_the_end_caps_move_inward_not_outward() -> None:
    """Outward would put the rotating zone in the free stream.

    The cap that matters is the one already flush with the outboard sidewall.
    A normal-direction offset moves it further out, past the tyre, and a zone
    reaching into the free stream spins air that should be still. So the caps
    are pulled back into the wheel's own air instead.
    """
    sleeve = _turned(BORE, WIDTH)
    moved = push_into_tyre(sleeve, ORIGIN, AXIS, 0.002)

    def span(m):
        axial = (m.vertices - ORIGIN) @ AXIS
        return float(axial.min()), float(axial.max())

    lo0, hi0 = span(sleeve)
    lo1, hi1 = span(moved)
    assert lo1 > lo0 and hi1 < hi0
    assert (hi1 - lo1) == pytest.approx((hi0 - lo0) - 2 * 0.002, abs=1e-4)


def test_the_repair_actually_separates_the_surfaces() -> None:
    """The point of the exercise: no longer coincident.

    Intersecting counts as separated here - surface_gap reads ~0 for both
    touching and crossing, so verticality is checked by geometry: every point
    of the sleeve wall is now strictly inside the tyre's radial band.
    """
    moved = push_into_tyre(_turned(BORE, WIDTH), ORIGIN, AXIS, 0.002)
    offsets = moved.vertices - ORIGIN
    axial = offsets @ AXIS
    radial = np.linalg.norm(offsets - axial[:, None] * AXIS, axis=1)

    wall = radial > BORE + 1e-6
    assert wall.any(), "the sleeve wall should now sit inside the carcass"
    assert radial.max() < TREAD, "and must not reach the tread"


def test_it_refuses_to_collapse_a_short_sleeve() -> None:
    with pytest.raises(MrfGeometryError, match="collapse"):
        push_into_tyre(_turned(BORE, 0.003), ORIGIN, AXIS, 0.002)


def test_the_axis_is_unchanged_by_the_repair() -> None:
    """derive_wheels takes the wheel axis from the sleeve; it must still hold."""
    from simdev.geometry.wheels import axis_of_revolution

    moved = push_into_tyre(_turned(BORE, WIDTH), ORIGIN, AXIS, 0.002)
    _, axis, defect = axis_of_revolution(moved)

    assert defect < 0.05
    assert abs(float(np.dot(axis, AXIS))) == pytest.approx(1.0, abs=1e-6)


# --- end to end through prepare ------------------------------------------


def _car(tmp_path, mrf=None, **overrides):
    """The multi-part fixture car, with each sleeve welded to its tyre bore."""
    import yaml
    from tests.test_car_prepare import CORNERS, car_case, write_car
    from simdev.stages.prepare import prepare

    cad = tmp_path / "cad"
    write_car(cad)

    # write_car builds sleeves at r=27 mm inside tyres of bore 5 mm; rebuild
    # them so the sleeve lies exactly on the tyre bore, as the real CAD does.
    from tests.test_car_prepare import TYRE_RADIUS, _place

    for wheel, (x, y) in CORNERS.items():
        transform = _place(x, y)
        bore = 0.012
        tyre_m = trimesh.creation.annulus(
            r_min=bore, r_max=TYRE_RADIUS, height=0.027, transform=transform
        )
        sleeve = trimesh.creation.cylinder(
            radius=bore, height=0.027, sections=64, transform=transform
        )
        for name, m in ((f"Tire_{wheel}", tyre_m), (f"MRF_{wheel}", sleeve)):
            m = m.copy()
            m.apply_scale(1000.0)
            (cad / f"{name}.stl").write_bytes(trimesh.exchange.stl.export_stl(m))

    case = car_case(cad, **overrides)
    if mrf is not None:
        case["geometry"]["mrf_interference"] = mrf
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    return prepare(path, tmp_path / "run", profile="car_dev")


def test_a_welded_sleeve_is_detected_and_pushed_in(tmp_path) -> None:
    result = _car(tmp_path)

    assert result.sleeve_fits, "every wheel should be assessed"
    for wheel, fit in result.sleeve_fits.items():
        assert fit.clearance < 1e-3, f"{wheel} should read as coincident"
        assert fit.moved, f"{wheel} should have been pushed in"
        assert fit.radius_after > fit.radius_before
    assert any("pushed into it" in w for w in result.warnings)


def test_the_pushed_sleeve_is_what_gets_written(tmp_path) -> None:
    """The file OpenFOAM meshes is the repaired one, not the CAD one."""
    result = _car(tmp_path)
    fit = result.sleeve_fits["FL"]

    written = trimesh.load_mesh(
        result.run_dir / "constant" / "triSurface" / "MRF_FL.stl", process=True
    )
    wheel = result.wheels["FL"]
    assert outer_radius(written, wheel.centre, wheel.direction) == pytest.approx(
        fit.radius_after, rel=1e-3
    )


def test_a_sleeve_with_clearance_is_left_exactly_alone(tmp_path) -> None:
    """Only a coincident sleeve is touched. The CAD is otherwise the truth."""
    result = _car(tmp_path, mrf={"min_clearance": 1e-9})

    for wheel, fit in result.sleeve_fits.items():
        assert not fit.moved
        assert fit.radius_after == fit.radius_before


def test_the_rolling_radius_is_measured_before_the_sleeve_moves(tmp_path) -> None:
    """derive_wheels reads the axis off the sleeve as drawn, not as repaired."""
    from tests.test_car_prepare import TYRE_RADIUS

    result = _car(tmp_path)
    for wheel in result.wheels.values():
        assert wheel.radius == pytest.approx(TYRE_RADIUS, rel=1e-2)
        assert abs(float(wheel.direction[1])) == pytest.approx(1.0, abs=1e-6)


def test_the_measurement_is_recorded_in_the_run_status(tmp_path) -> None:
    import json

    result = _car(tmp_path)
    status = json.loads(
        (result.run_dir / "status" / "prepare.json").read_text(encoding="utf-8")
    )
    recorded = status["detail"]["mrf_sleeves"]

    assert set(recorded) == set(result.sleeve_fits)
    for entry in recorded.values():
        assert entry["interference"] > 0.0
        assert entry["radius_after"] > entry["radius_before"]
