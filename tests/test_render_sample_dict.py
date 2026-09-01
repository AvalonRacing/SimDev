from __future__ import annotations

import copy
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

# A second fixture with TWO force-bearing patches (RULING C3's own body plus
# a "wing"), used only by tests that need to tell "one combined surface"
# apart from "one surface per patch that happens to look the same because
# there is only one patch". BASE alone cannot distinguish the two: with a
# single force patch, "one surface per patch" and "one combined surface"
# render identically except for the surface's name.
MULTI_FORCE_PATCH: dict = copy.deepcopy(BASE)
MULTI_FORCE_PATCH["geometry"]["patches"].insert(1, {"name": "wing", "role": "body"})

SLICES = [
    {"name": "x_+0.000", "point": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]},
    {"name": "z_+0.050", "point": [0.0, 0.0, 0.05], "normal": [0.0, 0.0, 1.0]},
]


def _spec(base: dict = BASE):
    return resolve(base, profile="dev")


def _render(tmp_path: Path, base: dict = BASE) -> str:
    render_sample_dict(_spec(base), tmp_path, SLICES)
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


def test_the_vehicle_surface_samples_UMean_too(tmp_path: Path) -> None:
    """RULING C4: the field list on patchSurfaces is (pMean UMean yPlus), not
    (pMean yPlus). render/plan.py's Calculators build velocity-derived fields
    off UMean unconditionally, so a surface sample missing it fails at render
    time - and adding it here is one vector field on one surface, negligible
    against the cost of a second sampling code path for surfaces alone."""
    text = _render(tmp_path)
    assert "fields          (pMean UMean yPlus);" in text


def test_one_combined_vehicle_surface_not_one_per_patch(tmp_path: Path) -> None:
    """RULING C3: the brief's per-patch `patch_<name>` surfaces are wrong.

    Sampled per patch, a consumer feeding the seven "overall" views would
    pick whichever surface it found first and draw one part of the car
    labelled as the whole vehicle. There must be exactly one patch-type
    surface, named `vehicle`, whose own `patches` entry lists every
    force-bearing patch - which BASE alone (one force patch) cannot
    distinguish from "one surface per patch", so this uses
    MULTI_FORCE_PATCH's two.
    """
    text = _render(tmp_path, MULTI_FORCE_PATCH)
    assert text.count("type        patch;") == 1
    assert "patch_body" not in text
    assert "patch_wing" not in text
    # Both force patches on the one surface's own patches list.
    idx = text.index("type        patch;")
    block = text[max(0, idx - 200) : idx + 200]
    assert "vehicle" in block
    idx = text.index("patches     (")
    line_end = text.index(")", idx)
    patches_line = text[idx : line_end + 1]
    assert "body" in patches_line and "wing" in patches_line
