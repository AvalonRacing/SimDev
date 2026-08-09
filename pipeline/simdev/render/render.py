from __future__ import annotations

import json
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
