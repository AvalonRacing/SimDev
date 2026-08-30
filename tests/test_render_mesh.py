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
    layer_patches,
    location_in_mesh,
    refinement_boxes,
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
        "symmetric": True,
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
    text = (_render(tmp_path) / "system" / "surfaceFeatureExtractDict").read_text()
    assert "body.stl" in text


def test_surface_feature_dict_uses_the_esi_per_surface_format(tmp_path: Path) -> None:
    """ESI v2412 keys each surface to its own sub-dictionary.

    The flat `surfaces ( "body.stl" );` list is the OpenFOAM Foundation
    format. Rendering it against an ESI build produces a dictionary that
    surfaceFeatureExtract reads without error and acts on incorrectly, so
    assert the structure rather than merely the filename.
    """
    text = (_render(tmp_path) / "system" / "surfaceFeatureExtractDict").read_text()
    assert "extractionMethod    extractFromSurface;" in text
    assert "includedAngle       150;" in text
    assert "surfaces\n(" not in text


def _spec_with_wake(**region):
    base = {
        "name": "wake",
        "level": 2,
        "x_start": -0.2,
        "x_end": 3.0,
        "half_width": 0.5,
        "height": 0.6,
    }
    base.update(region)
    return _spec({"domain": {"refinement_regions": [base]}})


def test_refinement_region_is_declared_as_a_searchable_box(tmp_path: Path) -> None:
    """snappy needs the box in geometry{} as well as refinementRegions{}.

    Referencing a region that was never declared is a FOAM FATAL IO ERROR,
    not a warning.
    """
    text = (_render(tmp_path, _spec_with_wake()) / "system" / "snappyHexMeshDict").read_text()
    assert "type            searchableBox;" in text
    assert "mode            inside;" in text
    assert "levels          ((1.0 2));" in text


def test_refinement_region_is_anchored_to_the_geometry(tmp_path: Path) -> None:
    """Offsets are body lengths off the model's bounding box, not the domain."""
    spec = _spec_with_wake()
    domain = BoxDomainBuilder().build(spec, BOUNDS)
    boxes = refinement_boxes(spec, domain)
    length = domain.geom_length

    assert len(boxes) == 1
    assert boxes[0]["min"][0] == pytest.approx(domain.geom_min[0] - 0.2 * length)
    assert boxes[0]["max"][0] == pytest.approx(domain.geom_max[0] + 3.0 * length)


def test_refinement_region_is_clipped_to_the_domain(tmp_path: Path) -> None:
    """A region running past the domain still refines every cell it crosses.

    Left unclipped that silently multiplies the cell count, so an oversized
    request is trimmed rather than honoured.
    """
    spec = _spec_with_wake(x_end=500.0, half_width=500.0, height=500.0)
    domain = BoxDomainBuilder().build(spec, BOUNDS)
    box = refinement_boxes(spec, domain)[0]

    assert box["max"][0] <= domain.x_max
    assert box["max"][1] <= domain.y_max
    assert box["max"][2] <= domain.z_max
    assert box["min"][1] >= domain.y_min


def test_no_regions_renders_an_empty_block(tmp_path: Path) -> None:
    text = (_render(tmp_path) / "system" / "snappyHexMeshDict").read_text()
    assert "searchableBox" not in text
    assert "refinementRegions" in text


def test_patch_layer_override_only_ever_reduces(tmp_path: Path) -> None:
    """A per-patch n_layers caps the stack; it cannot exceed what fits."""
    case = deep_merge(BASE, {})
    case["geometry"]["patches"] = [
        {"name": "body", "role": "body", "n_layers": 2},
        {"name": "ground", "role": "ground", "n_layers": 999},
        {"name": "symmetry", "role": "symmetry"},
        {"name": "inlet", "role": "inlet"},
        {"name": "outlet", "role": "outlet"},
        {"name": "farfield", "role": "farfield"},
    ]
    spec = resolve(case, profile="dev")
    domain = BoxDomainBuilder().build(spec, BOUNDS)
    by_name = {p["name"]: p["n_layers"] for p in layer_patches(spec, domain)}

    body = next(p for p in spec.geometry.patches if p.name == "body")
    assert by_name["body"] == 2
    assert by_name["ground"] == spec.n_layers_in_cell(spec.mesh.base_cell_size)
    assert by_name["ground"] < 999
    assert by_name["body"] < spec.n_layers_for(body)


def test_every_wall_patch_gets_layers(tmp_path: Path) -> None:
    """Layers follow is_wall, not refinement level.

    The ground is a blockMesh patch with a wall function on it; skipping it
    left its first cell 60 mm off the floor and y+ around 2000.
    """
    text = (_render(tmp_path) / "system" / "snappyHexMeshDict").read_text()
    layers_block = text.split("layers\n    {")[1].split("}\n\n")[0]
    assert "ground" in layers_block
    assert "body" in layers_block


# --- layer normal smoothing ------------------------------------------------


def test_layer_normal_smoothing_defaults_to_snappys_own_values(
    tmp_path: Path,
) -> None:
    """1 and 3 are snappy's defaults and what every run before 2026-08-30
    used, so an unset case renders exactly what it used to."""
    text = (_render(tmp_path) / "system" / "snappyHexMeshDict").read_text()
    assert "nSmoothSurfaceNormals 1;" in text
    assert "nSmoothNormals      3;" in text


def test_layer_normal_smoothing_is_settable(tmp_path: Path) -> None:
    """These two decide how hard snappy smooths the extrusion direction before
    building the prism stack, and on a car that is 24% suspension and tyre by
    wetted area they are the knob that decides whether the stack survives on
    thin, curved parts. They were compared by hand-editing a rendered
    dictionary, which made the better setting unreproducible - the caseSpec of
    both runs was byte-identical.
    """
    spec = _spec(
        {"mesh": {"n_smooth_surface_normals": 3, "n_smooth_normals": 10}}
    )
    text = (_render(tmp_path, spec) / "system" / "snappyHexMeshDict").read_text()
    assert "nSmoothSurfaceNormals 3;" in text
    assert "nSmoothNormals      10;" in text


def test_layer_normal_smoothing_changes_the_spec_hash() -> None:
    """It changes the mesh, so it has to change the hash - otherwise the stage
    cache would skip re-meshing and hand back the previous mesh."""
    assert _spec().spec_hash() != _spec(
        {"mesh": {"n_smooth_normals": 10}}
    ).spec_hash()


# --- deterministic decomposition -------------------------------------------


def test_decompose_par_dict_is_deterministic_by_default(tmp_path: Path) -> None:
    """scotch is a randomised graph partitioner with no fixed seed, and it was
    measured producing different partitions from an identical blockMesh:
    processor 0 got 5788 cells in ~/runs/car-nut10-smooth and 5731 in
    ~/runs/car-long2000. Everything downstream inherits that - snappy
    redistributes with the same method - so two runs of one spec produced two
    different meshes, and the pipeline could not reproduce its own result.
    """
    text = (_render(tmp_path) / "system" / "decomposeParDict").read_text()
    assert "method          hierarchical;" in text
    # the directive, not the word - the comment above it names scotch on purpose
    assert "method          scotch;" not in text
    assert "order           xyz;" in text


def test_decompose_par_dict_split_multiplies_to_the_rank_count(tmp_path: Path) -> None:
    import re
    spec = _spec({"solve": {"n_ranks": 8}})
    text = (_render(tmp_path, spec) / "system" / "decomposeParDict").read_text()
    n = re.search(r"n\s+\((\d+) (\d+) (\d+)\)", text)
    assert n is not None
    assert int(n[1]) * int(n[2]) * int(n[3]) == 8


def test_scotch_is_still_reachable_for_a_deliberate_choice(tmp_path: Path) -> None:
    """Determinism costs partition quality, so the faster partitioner stays
    available - as a recorded choice in the spec, not a silent default."""
    spec = _spec({"solve": {"decomposition": "scotch"}})
    text = (_render(tmp_path, spec) / "system" / "decomposeParDict").read_text()
    assert "method          scotch;" in text
