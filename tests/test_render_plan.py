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
        images_dir=Path("results/images"),
        stamp=STAMP,
    )


def test_slice_names_are_stable_across_runs() -> None:
    """Same offset always yields same name, enabling file-by-file alignment across runs.

    The name format is signed, fixed-width, three decimals. Lexicographic order
    is NOT geometric order ('+' is ASCII 43, '-' is 45, so positives sort first),
    but that's okay: the property that matters is identity, not sorting.
    Two runs comparing a slice at x=-0.30 both call slice_name("x", -0.30) and
    both get "x_-0.300", so their image directories align file-by-file.
    """
    # Exact format for positive, negative, and zero
    assert slice_name("x", 0.12) == "x_+0.120"
    assert slice_name("x", -0.3) == "x_-0.300"
    assert slice_name("z", 0.0) == "z_+0.000"

    # Cross-run identity: same offset always yields same name
    offsets = [-0.30, -0.02, 0.0, 0.30]
    names1 = [slice_name("x", o) for o in offsets]
    names2 = [slice_name("x", o) for o in offsets]
    assert names1 == names2, "Run 1 and run 2 generate identical names"


def test_the_plan_is_json_serialisable() -> None:
    """It crosses a process boundary into an interpreter that cannot import
    simdev, so it has to be plain data."""
    json.dumps(_plan())


def test_no_entry_is_seeded_with_a_sample_path() -> None:
    """"sample" is filled in later, by images.py's on-disk discovery
    (find_sample) - never by the plan builder. A path seeded here and never
    overwritten (a caller that skips that discovery step) would be a
    results/samples/... path nothing produced. Slices and surfaces must
    agree on this: neither carries the key until discovery runs.
    """
    plan = _plan()
    assert all("sample" not in s for s in plan["slices"])
    assert all("sample" not in s for s in plan["surfaces"])


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
    # From the views file, not a literal: the camera scale is a tuning knob
    # and hardcoding it here makes every adjustment look like a regression.
    # What is being tested is that the plan USES the configured value.
    assert first["camera"]["parallel_scale"] == pytest.approx(
        VIEWS.parallel_scale["x"]
    )


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


YAWED = (
    (0.9834, 0.1815, 0.0),   # x_car, ~10.5 deg of body slip
    (-0.1815, 0.9834, 0.0),  # y_car
    (0.0, 0.0, 1.0),         # z_car
)


def _yawed_plan():
    return build_render_plan(
        u_inf=15.0, views=VIEWS, datum=DATUM, frame=None,
        images_dir=Path("results/images"), stamp=STAMP, axes=YAWED,
    )


def test_car_axes_rotate_the_slice_normals() -> None:
    """A slice must cut across the CAR, not across the domain.

    The CAD carries its attitude baked in - this pose is ~10.5 degrees of
    body slip - so a plane normal to mesh +x cuts the car at that angle, and
    a state exported at a different slip angle cuts it at a different one.
    Plane k of two states would then be two different cuts sharing a name.
    """
    first = next(s for s in _yawed_plan()["slices"] if s["axis"] == "x")
    assert first["normal"][0] == pytest.approx(0.9834, abs=1e-3)
    assert first["normal"][1] == pytest.approx(0.1815, abs=1e-3)


def test_car_axes_rotate_the_surface_cameras() -> None:
    """`front` means the car's nose, not the domain's -x face."""
    plan = _yawed_plan()
    front = next(s for s in plan["surfaces"] if s["name"] == "front")
    direction = [
        front["camera"]["focal"][i] - front["camera"]["position"][i]
        for i in range(3)
    ]
    mag = sum(d * d for d in direction) ** 0.5
    # Looking along -x_car: the camera sits on the nose side and looks back.
    assert direction[0] / mag == pytest.approx(-0.9834, abs=1e-3)
    assert direction[1] / mag == pytest.approx(-0.1815, abs=1e-3)


def test_slice_offsets_walk_along_the_car_axis() -> None:
    """The offset must step along the rotated normal, not along mesh x.

    Otherwise the planes stay parallel to the car but march across it at an
    angle, and the spacing a run reports is not the spacing on the car.
    """
    plan = _yawed_plan()
    at_zero = next(s for s in plan["slices"] if s["name"] == "x_+0.000")
    at_200 = next(s for s in plan["slices"] if s["name"] == "x_+0.200")
    step = [at_200["point"][i] - at_zero["point"][i] for i in range(3)]
    assert step[0] == pytest.approx(0.200 * 0.9834, abs=1e-3)
    assert step[1] == pytest.approx(0.200 * 0.1815, abs=1e-3)


def test_identity_axes_leave_everything_in_mesh_coordinates() -> None:
    """A case with no measured yaw must be framed exactly as before."""
    plan = _plan()
    first = next(s for s in plan["slices"] if s["axis"] == "x")
    assert first["normal"] == pytest.approx([1.0, 0.0, 0.0])


def test_images_go_in_one_flat_directory_per_field_and_axis() -> None:
    plan = _plan()
    first = next(s for s in plan["slices"] if s["axis"] == "x")
    out = next(i["out"] for i in first["images"] if i["field"] == "cp")
    assert "/cp_x/" in out and "/slices/" not in out
    surface = next(s for s in plan["surfaces"] if s["name"] == "iso")
    sout = next(i["out"] for i in surface["images"] if i["field"] == "yplus")
    assert sout.endswith("surface_yplus/iso.png")
