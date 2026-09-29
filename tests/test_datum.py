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
