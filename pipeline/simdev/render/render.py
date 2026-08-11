from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jinja2

from simdev.config.schema import CaseSpec
from simdev.domain.base import Domain
from simdev.geometry.wheels import Wheel
from simdev.render.context import build_context

TEMPLATE_DIR = Path(__file__).parent / "templates"

# blockMeshDict is chosen by the domain rather than listed here: a straight
# tunnel and a cornering sector are different block topologies, not one
# template with a branch in it.
MESH_DICTS: dict[str, str] = {
    "surfaceFeatureExtractDict.jinja": "system/surfaceFeatureExtractDict",
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
    domain: Domain,
    geometry_files: dict[str, Path],
    out_dir: Path,
    wheels: dict[str, Wheel] | None = None,
) -> None:
    context = build_context(spec, domain, geometry_files, wheels)
    templates = dict(MESH_DICTS)
    templates[domain.block_mesh_template] = "system/blockMeshDict"
    _render_set(templates, context, out_dir)


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
    domain: Domain,
    geometry_files: dict[str, Path],
    out_dir: Path,
    wheels: dict[str, Wheel] | None = None,
) -> None:
    context = build_context(spec, domain, geometry_files, wheels)
    templates = dict(SOLVER_DICTS)
    # Only written when something actually rotates. An empty MRFProperties is
    # valid OpenFOAM and does nothing, but it leaves a rotating-frame file in
    # a straight-line case for someone to read and mistrust later.
    if context["mrf_zones"]:
        templates["MRFProperties.jinja"] = "constant/MRFProperties"
    _render_set(templates, context, out_dir)

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
    domain: Domain,
    geometry_files: dict[str, Path],
    out_dir: Path,
    wheels: dict[str, Wheel] | None = None,
) -> None:
    """Render a complete, self-contained case plus its provenance record."""
    out = Path(out_dir)
    render_mesh_dicts(spec, domain, geometry_files, out, wheels)
    render_solver_dicts(spec, domain, geometry_files, out, wheels)
    (out / "caseSpec.json").write_text(
        json.dumps(
            {"spec": spec.model_dump(mode="json"), "hash": spec.spec_hash()},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
