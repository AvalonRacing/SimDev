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
