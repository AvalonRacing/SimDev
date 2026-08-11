from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh

from simdev.config.resolve import load_case
from simdev.config.schema import CaseSpec, GroundDatum
from simdev.config.validate import validate
from simdev.domain.annulus import AnnulusDomainBuilder, check_sweep
from simdev.domain.base import Domain, DomainSector
from simdev.domain.box import BoxDomainBuilder, check_blockage
from simdev.geometry.ahmed import write_ahmed_stl
from simdev.geometry.roles import PatchRole, traits
from simdev.geometry.step import Tessellation, convert, default_cache_dir
from simdev.geometry.stl import (
    check_geometry,
    load_surface,
    place_surface,
    projected_frontal_area,
    stl_info,
    write_surface,
)
from simdev.geometry.wheels import Wheel, derive_wheels
from simdev.render.context import corner_frame, layer_patches, wheel_speeds
from simdev.render.render import render_case
from simdev.run.status import StageStatus, should_skip, write_status

STAGE = "prepare"

DOMAIN_BUILDERS = {
    "box": BoxDomainBuilder,
    "annulus": AnnulusDomainBuilder,
}


@dataclass(frozen=True)
class PrepareResult:
    spec: CaseSpec
    domain: Domain
    run_dir: Path
    frontal_area: float
    wheels: dict[str, Wheel] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _write_geometry(
    spec: CaseSpec, run_dir: Path
) -> tuple[dict[str, Path], dict[str, trimesh.Trimesh]]:
    """Place every surface into the run directory, and keep what was loaded.

    Returns the meshes alongside the paths because the car is 1.36 M
    triangles across 31 parts, and the bounding box, the frontal area and the
    wheel axes all want the same surfaces. Re-reading them from disk three
    times was the difference between prepare taking seconds and taking a
    minute.
    """
    tri_surface = run_dir / "constant" / "triSurface"
    tri_surface.mkdir(parents=True, exist_ok=True)

    if spec.geometry.kind == "ahmed":
        assert spec.geometry.ahmed is not None
        files = write_ahmed_stl(spec.geometry.ahmed, tri_surface)
        return files, {name: load_surface(p) for name, p in files.items()}

    assert spec.geometry.source_dir is not None
    source_dir = Path(spec.geometry.source_dir)
    suffix = ".step" if spec.geometry.kind == "step" else ".stl"

    tessellation = Tessellation(
        max_edge=spec.geometry.tessellation.max_edge,
        min_edge=spec.geometry.tessellation.min_edge,
        curvature_segments=spec.geometry.tessellation.curvature_segments,
    )
    cache_dir = (
        default_cache_dir(source_dir) if spec.geometry.kind == "step" else None
    )

    meshes: dict[str, trimesh.Trimesh] = {}
    missing: list[str] = []

    for patch in spec.geometry.patches:
        if not traits(patch.role).from_stl:
            continue
        source = source_dir / f"{patch.name}{suffix}"
        if not source.exists():
            missing.append(f"{patch.name} (role {patch.role.value})")
            continue

        if spec.geometry.kind == "step":
            source, _ = convert(source, cache_dir, tessellation, spec.geometry.scale)

        meshes[patch.name] = place_surface(
            load_surface(source),
            scale=spec.geometry.scale,
            rotate_z_deg=spec.geometry.rotate_z_deg,
            translate=spec.geometry.translate,
        )

    # Silently skipping a missing file used to be the behaviour, and it turns
    # a mistyped part name into a car with no rear wing that meshes, solves
    # and converges.
    if missing:
        raise FileNotFoundError(
            f"no {suffix} found in {source_dir} for: {', '.join(missing)}. "
            f"Every patch whose role carries a surface needs "
            f"<patch name>{suffix}"
        )

    # The ground datum needs every surface placed first, because it is a
    # single rigid move of the whole car chosen from where its tyres ended up.
    _apply_ground_datum(spec, meshes)

    files = {
        name: write_surface(mesh, tri_surface / f"{name}.stl")
        for name, mesh in meshes.items()
    }
    return files, meshes


def _apply_ground_datum(
    spec: CaseSpec, meshes: dict[str, trimesh.Trimesh]
) -> float:
    """Drop the car onto the road, as one rigid body. Returns the shift."""
    if spec.geometry.ground_datum is not GroundDatum.TYRE_CONTACT:
        return 0.0

    tyres = [
        p.name
        for p in spec.geometry.patches
        if p.role is PatchRole.TYRE and p.name in meshes
    ]
    if not tyres:
        raise ValueError(
            "geometry.ground_datum is 'tyre_contact' but no patch has role "
            "'tyre'; there is nothing for the car to stand on"
        )

    lowest = min(float(meshes[name].bounds[0][2]) for name in tyres)
    shift = -lowest
    if shift:
        for mesh in meshes.values():
            mesh.apply_translation([0.0, 0.0, shift])
    return shift


def _bounds(
    meshes: dict[str, trimesh.Trimesh],
) -> tuple[list[float], list[float]]:
    lows = [list(mesh.bounds[0]) for mesh in meshes.values()]
    highs = [list(mesh.bounds[1]) for mesh in meshes.values()]
    return (
        [min(v[i] for v in lows) for i in range(3)],
        [max(v[i] for v in highs) for i in range(3)],
    )


def check_ground_placement(
    spec: CaseSpec, meshes: dict[str, trimesh.Trimesh]
) -> list[str]:
    """Where the geometry sits relative to the road, as a hard check.

    Ride height drives car aerodynamics more strongly than almost any other
    single dimension, so the pipeline states where the car ended up rather
    than assuming the CAD arrived correctly placed. Anything meaningfully
    below z = 0 is an error: snappyHexMesh will happily mesh the intersection
    of a wheel and the road into a shape nobody designed, and the run that
    follows looks entirely normal.
    """
    limit = -spec.geometry.max_ground_penetration
    warnings: list[str] = []
    offenders: list[str] = []

    for name, mesh in meshes.items():
        z_min = float(mesh.bounds[0][2])
        if z_min < limit:
            offenders.append(f"{name} at {z_min * 1e3:.2f} mm")

    if offenders:
        raise ValueError(
            "geometry penetrates the ground plane at z = 0: "
            + "; ".join(sorted(offenders))
            + f". The limit is {spec.geometry.max_ground_penetration * 1e3:.2f} mm. "
            "Raise the car with geometry.translate, or re-export it sitting on "
            "the road"
        )

    lowest = min(float(m.bounds[0][2]) for m in meshes.values())
    if lowest > 0.05 * (max(float(m.bounds[1][2]) for m in meshes.values())):
        warnings.append(
            f"the lowest point of the geometry is {lowest * 1e3:.1f} mm above "
            "the road; nothing is touching the ground, so either the car is "
            "floating or the ride height is deliberate"
        )
    return warnings


def _wheel_warnings(
    spec: CaseSpec, domain: Domain, wheels: dict[str, Wheel]
) -> list[str]:
    """Report what the wheels turned out to be, and when that looks wrong."""
    if not wheels:
        return []

    speeds = wheel_speeds(spec, wheels, corner_frame(spec, domain))
    warnings: list[str] = []

    for name in sorted(speeds):
        entry = speeds[name]
        wheel = entry["wheel"]
        # Slip beyond a few per cent of the road speed means the wheel's axis
        # and its direction of travel disagree - a steer angle that was not
        # expected, or a mislabelled corner.
        if entry["road_speed"] > 0.0 and entry["slip"] > 0.05 * entry["road_speed"]:
            warnings.append(
                f"wheel '{name}' cannot roll off "
                f"{entry['slip']:.2f} m/s of its {entry['road_speed']:.2f} m/s "
                "road speed; its axis is not square to the direction of "
                "travel. Expected for a steered wheel, a mistake otherwise"
            )
        if wheel.axis_surface_radius > wheel.radius:
            warnings.append(
                f"wheel '{name}': the MRF sleeve '{wheel.axis_source}' reaches "
                f"{wheel.axis_surface_radius * 1e3:.1f} mm, outside the "
                f"{wheel.radius * 1e3:.1f} mm tyre; the rotating cell zone "
                "would extend into the free stream"
            )

    spread = [speeds[n]["surface_speed"] for n in speeds]
    if spread and max(spread) > 0.0:
        warnings.append(
            "wheel surface speeds "
            + ", ".join(
                f"{n} {speeds[n]['surface_speed']:.2f} m/s "
                f"({speeds[n]['omega']:+.1f} rad/s)"
                for n in sorted(speeds)
            )
        )
    return warnings


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

    geometry_files, meshes = _write_geometry(spec, run_dir)
    if not geometry_files:
        raise FileNotFoundError("no geometry files were produced or found")

    for name, mesh in meshes.items():
        warnings.extend(
            [
                f"{name}: {message}"
                for message in check_geometry(
                    stl_info(mesh), expected_length=spec.forces.l_ref
                )
            ]
        )

    warnings.extend(check_ground_placement(spec, meshes))

    # Measured after placement, so the axes are those of the driving state
    # actually being run rather than of the CAD's authoring position.
    wheels = derive_wheels(spec.geometry.patches, meshes)

    lo, hi = _bounds(meshes)
    domain = DOMAIN_BUILDERS[spec.domain.kind]().build(spec, (lo, hi))
    if isinstance(domain, DomainSector):
        warnings.extend(check_sweep(domain))

    combined = trimesh.util.concatenate(list(meshes.values()))
    frontal_area = projected_frontal_area(combined, axis=0)
    if spec.half_model:
        frontal_area /= 2.0
    warnings.extend(check_blockage(frontal_area, domain, spec.domain.max_blockage))
    warnings.extend(_wheel_warnings(spec, domain, wheels))

    result = PrepareResult(
        spec=spec,
        domain=domain,
        run_dir=run_dir,
        frontal_area=frontal_area,
        wheels=wheels,
        warnings=warnings,
    )

    if should_skip(run_dir, STAGE, spec.spec_hash(), force):
        return result

    speeds = wheel_speeds(spec, wheels, corner_frame(spec, domain))
    render_case(spec, domain, geometry_files, run_dir, wheels)
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
                # Measured, not configured, so it belongs in the run record
                # rather than in caseSpec.json: this is what the geometry
                # turned out to be, not what was asked for.
                "wheels": {
                    name: {
                        "origin": w.origin,
                        "axis": w.axis,
                        "radius": w.radius,
                        "width": w.width,
                        "omega": speeds[name]["omega"],
                        "surface_speed": speeds[name]["surface_speed"],
                        "slip": speeds[name]["slip"],
                    }
                    for name, w in wheels.items()
                },
                # Derived, so it is not in caseSpec.json: record it here or a
                # clamped layer count is invisible after the fact.
                "surface_cell_size": spec.surface_cell_size,
                "n_layers_requested": spec.mesh.n_layers,
                "n_layers_effective": spec.n_layers_effective,
                "layer_stack_thickness": spec.layer_stack_thickness(
                    spec.n_layers_effective
                ),
                # What snappy was actually asked for, per wall patch. The mesh
                # gate needs this to judge coverage, and it cannot recompute
                # the ground's share without the domain.
                "requested_layers": {
                    p["name"]: p["n_layers"] for p in layer_patches(spec, domain)
                },
            },
        ),
    )
    return result
