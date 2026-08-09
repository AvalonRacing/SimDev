from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh

from simdev.config.resolve import load_case
from simdev.config.schema import CaseSpec
from simdev.config.validate import validate
from simdev.domain.base import DomainBox
from simdev.domain.box import BoxDomainBuilder, check_blockage
from simdev.geometry.ahmed import write_ahmed_stl
from simdev.geometry.stl import check_geometry, projected_frontal_area, read_stl_info
from simdev.render.render import render_case
from simdev.run.status import StageStatus, should_skip, write_status

STAGE = "prepare"


@dataclass(frozen=True)
class PrepareResult:
    spec: CaseSpec
    domain: DomainBox
    run_dir: Path
    frontal_area: float
    warnings: list[str] = field(default_factory=list)


def _write_geometry(spec: CaseSpec, run_dir: Path) -> dict[str, Path]:
    tri_surface = run_dir / "constant" / "triSurface"
    tri_surface.mkdir(parents=True, exist_ok=True)

    if spec.geometry.kind == "ahmed":
        assert spec.geometry.ahmed is not None
        return write_ahmed_stl(spec.geometry.ahmed, tri_surface)

    assert spec.geometry.stl_dir is not None
    files: dict[str, Path] = {}
    for patch in spec.geometry.patches:
        source = Path(spec.geometry.stl_dir) / f"{patch.name}.stl"
        if source.exists():
            target = tri_surface / source.name
            shutil.copy2(source, target)
            files[patch.name] = target
    return files


def _bounds(files: dict[str, Path]) -> tuple[list[float], list[float]]:
    lows: list[list[float]] = []
    highs: list[list[float]] = []
    for path in files.values():
        info = read_stl_info(path)
        lows.append(list(info.bounds_min))
        highs.append(list(info.bounds_max))
    return (
        [min(v[i] for v in lows) for i in range(3)],
        [max(v[i] for v in highs) for i in range(3)],
    )


def prepare(
    case_path: Path,
    run_dir: Path,
    profile: str,
    wall_treatment: str | None = None,
    overrides: dict[str, Any] | None = None,
    force: bool = False,
) -> PrepareResult:
    spec = load_case(Path(case_path), profile, wall_treatment, overrides)
    warnings = validate(spec)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    geometry_files = _write_geometry(spec, run_dir)
    if not geometry_files:
        raise FileNotFoundError("no geometry files were produced or found")

    for path in geometry_files.values():
        warnings.extend(
            check_geometry(read_stl_info(path), expected_length=spec.forces.l_ref)
        )

    lo, hi = _bounds(geometry_files)
    domain = BoxDomainBuilder().build(spec, (lo, hi))

    combined = trimesh.util.concatenate(
        [trimesh.load_mesh(p, process=False) for p in geometry_files.values()]
    )
    frontal_area = projected_frontal_area(combined, axis=0)
    if spec.half_model:
        frontal_area /= 2.0
    warnings.extend(check_blockage(frontal_area, domain, spec.domain.max_blockage))

    result = PrepareResult(
        spec=spec,
        domain=domain,
        run_dir=run_dir,
        frontal_area=frontal_area,
        warnings=warnings,
    )

    if should_skip(run_dir, STAGE, spec.spec_hash(), force):
        return result

    render_case(spec, domain, geometry_files, run_dir)
    write_status(
        run_dir,
        StageStatus(
            stage=STAGE,
            state="ok",
            input_hash=spec.spec_hash(),
            reasons=warnings,
            detail={
                "frontal_area": frontal_area,
                "background_cells": domain.cell_count,
                "half_model": spec.half_model,
                # Derived, so it is not in caseSpec.json: record it here or a
                # clamped layer count is invisible after the fact.
                "surface_cell_size": spec.surface_cell_size,
                "n_layers_requested": spec.mesh.n_layers,
                "n_layers_effective": spec.n_layers_effective,
                "layer_stack_thickness": spec.layer_stack_thickness(
                    spec.n_layers_effective
                ),
            },
        ),
    )
    return result
