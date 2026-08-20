from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh

from simdev.config.resolve import load_case
from simdev.config.schema import CaseSpec
from simdev.config.validate import validate
from simdev.domain.annulus import AnnulusDomainBuilder, check_sweep
from simdev.domain.base import Domain, DomainSector
from simdev.domain.box import BoxDomainBuilder, check_blockage
from simdev.geometry.ahmed import build_ahmed_surfaces
from simdev.geometry.contact import ContactPatch, extrude_contact_patch
from simdev.geometry.decimate import cluster_vertices
from simdev.geometry.mrf import (
    SleeveFit,
    outer_radius,
    push_into_tyre,
    surface_gap,
)
from simdev.geometry.roles import PatchRole, traits
from simdev.geometry.step import Tessellation, convert, default_cache_dir
from simdev.geometry.stl import (
    check_geometry,
    load_surface,
    place_surface,
    stl_info,
    write_surface,
)
from simdev.geometry.wheels import Wheel, derive_wheels
from simdev.render.context import (
    VEHICLE_SURFACE,
    corner_frame,
    layer_patches,
    wheel_speeds,
)
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
    wheels: dict[str, Wheel] = field(default_factory=dict)
    contact_patches: dict[str, ContactPatch] = field(default_factory=dict)
    sleeve_fits: dict[str, SleeveFit] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _load_geometry(spec: CaseSpec) -> tuple[dict[str, trimesh.Trimesh], list[str]]:
    """Every surface as the CAD draws it, in metres.

    Loading and writing are separate steps because the contact patch sits
    between them: the bounding box, the ground check and the wheel axes must
    all see the geometry as drawn, and only then is it modified for meshing.

    The meshes are kept in memory rather than re-read per consumer because
    the car is several hundred thousand triangles across fifteen parts, and
    re-reading them was the difference between prepare taking seconds and
    taking a minute.
    """
    if spec.geometry.kind == "ahmed":
        assert spec.geometry.ahmed is not None
        # Generated rather than imported, and already in metres: no scale to
        # apply, and no tyres to cut.
        return build_ahmed_surfaces(spec.geometry.ahmed), []

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
    renamed: list[str] = []

    for patch in spec.geometry.patches:
        if not traits(patch.role).from_stl:
            continue
        source = source_dir / f"{patch.name}{suffix}"
        if not source.exists():
            near = _case_insensitive_match(source_dir, patch.name, suffix)
            if near is None:
                missing.append(f"{patch.name} (role {patch.role.value})")
                continue
            renamed.append(f"{near.name} for patch '{patch.name}'")
            source = near

        if spec.geometry.kind == "step":
            source, _ = convert(source, cache_dir, tessellation, spec.geometry.scale)

        meshes[patch.name] = place_surface(
            load_surface(source), scale=spec.geometry.scale
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

    return meshes, renamed


def _write_geometry(
    meshes: dict[str, trimesh.Trimesh], run_dir: Path
) -> dict[str, Path]:
    """Write the surfaces OpenFOAM will actually mesh against.

    The file written is the file the pipeline modified, so any question about
    what was simulated is answered by opening the run directory rather than by
    re-deriving the contact patch.
    """
    tri_surface = run_dir / "constant" / "triSurface"
    tri_surface.mkdir(parents=True, exist_ok=True)
    return {
        name: write_surface(mesh, tri_surface / f"{name}.stl")
        for name, mesh in meshes.items()
    }


# How much of a part's surface area may vanish into the shell surface's
# simplification before it is worth saying so.
#
# Calibrated on the real car rather than picked. At the 2 mm tolerance the
# case runs, the worst part is the suspension at 97.9% retained - honest
# simplification of curved panels, nothing lost. At 3 mm it is 93.6% and at
# 5 mm 81.5%, which is links collapsing out of the surface altogether. 95%
# therefore passes a sound tolerance and fails the first one that is not.
MIN_SHELL_AREA_RETAINED = 0.95


def _write_vehicle_surface(
    spec: CaseSpec, meshes: dict[str, trimesh.Trimesh], run_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    """One surface covering the whole vehicle, for measuring distance to.

    Refinement shells refine every cell within a distance of the car, and
    snappy takes that distance against a named surface. Handing it the fifteen
    parts separately would make it build fifteen distance fields and take the
    minimum; one combined surface is the same answer for a fifteenth of the
    work.

    Walls only. The MRF sleeves are closed volumes *inside* the tyres, so
    including them would put a surface in the middle of each wheel and pull a
    shell of fine cells into rubber, where there is no flow to resolve.

    Simplified first, if the case asked for it. This is the only surface in
    the case that can be: it is never snapped to, never becomes a patch and
    never enters force integration, so its triangle count buys nothing but the
    cost of snappy's own distance queries - which it pays on every rank, every
    time the mesh is redistributed mid-refinement. See
    domain.shell_surface_tolerance and geometry/decimate.py.

    Each part is simplified on its own rather than the concatenation, for two
    reasons: a shared grid would weld parts that pass within a tolerance of
    each other, and per-part area retention is the only cheap way to notice a
    thin feature that has collapsed out of the distance field.

    Written even though it is never meshed against, because it is what the
    mesh was actually built from: a run directory should answer "why is it
    refined there" without re-deriving anything.
    """
    if not spec.domain.refinement_shells:
        return {}, []

    named = [
        (patch.name, meshes[patch.name])
        for patch in spec.geometry.patches
        if patch.name in meshes and traits(patch.role).is_wall
    ]
    if not named:
        return {}, []

    tolerance = spec.domain.shell_surface_tolerance
    warnings: list[str] = []
    parts: list[trimesh.Trimesh] = []
    before = 0

    for name, mesh in named:
        before += len(mesh.faces)
        simplified = cluster_vertices(mesh, tolerance)
        parts.append(simplified)

        if tolerance <= 0.0 or mesh.area <= 0.0:
            continue
        retained = simplified.area / mesh.area
        if retained < MIN_SHELL_AREA_RETAINED:
            warnings.append(
                f"{name}: simplifying the shell surface at "
                f"{tolerance * 1000:.1f} mm cost {(1 - retained) * 100:.1f}% "
                "of the part's area, which means features thinner than the "
                "tolerance have collapsed out of it and are pulling no "
                "refinement around themselves; lower "
                "domain.shell_surface_tolerance"
            )

    combined = trimesh.util.concatenate(parts)
    write_surface(
        combined, run_dir / "constant" / "triSurface" / f"{VEHICLE_SURFACE}.stl"
    )
    return (
        {
            "tolerance": tolerance,
            "triangles_before": before,
            "triangles_after": len(combined.faces),
        },
        warnings,
    )


def fit_mrf_sleeves(
    spec: CaseSpec,
    meshes: dict[str, trimesh.Trimesh],
    wheels: dict[str, Wheel],
) -> tuple[dict[str, SleeveFit], list[str]]:
    """Push any sleeve that is lying on its tyre a little way into it.

    Modifies `meshes` in place. A sleeve with real clearance is left exactly
    as drawn; only a coincident one is moved, and the measured gap that
    triggered it is recorded either way.

    Must run after derive_wheels(), which takes each wheel's axis from the
    sleeve as the CAD drew it. Measuring the axis off a sleeve this has
    already grown would be measuring the repair.
    """
    settings = spec.geometry.mrf_interference
    if not settings.enabled:
        return {}, []

    sleeves = {
        p.wheel: p
        for p in spec.geometry.patches
        if p.role is PatchRole.MRF_ZONE and p.wheel is not None
    }
    tyres = {
        p.wheel: p
        for p in spec.geometry.patches
        if p.role is PatchRole.TYRE and p.wheel is not None
    }

    fits: dict[str, SleeveFit] = {}
    warnings: list[str] = []

    for wheel, sleeve_patch in sorted(sleeves.items()):
        tyre_patch = tyres.get(wheel)
        if tyre_patch is None or sleeve_patch.name not in meshes:
            continue
        if tyre_patch.name not in meshes or wheel not in wheels:
            continue

        sleeve = meshes[sleeve_patch.name]
        tyre = meshes[tyre_patch.name]
        measured = wheels[wheel]

        gap = surface_gap(sleeve, tyre)
        before = outer_radius(sleeve, measured.centre, measured.direction)

        if gap >= settings.min_clearance:
            fits[wheel] = SleeveFit(
                wheel=wheel,
                sleeve=sleeve_patch.name,
                tyre=tyre_patch.name,
                clearance=gap,
                interference=0.0,
                radius_before=before,
                radius_after=before,
                tyre_radius=measured.radius,
            )
            continue

        moved = push_into_tyre(
            sleeve, measured.centre, measured.direction, settings.interference
        )
        meshes[sleeve_patch.name] = moved
        after = outer_radius(moved, measured.centre, measured.direction)

        fits[wheel] = SleeveFit(
            wheel=wheel,
            sleeve=sleeve_patch.name,
            tyre=tyre_patch.name,
            clearance=gap,
            interference=settings.interference,
            radius_before=before,
            radius_after=after,
            tyre_radius=measured.radius,
        )

        # Pushed in radially, so this is the check that it did not come out
        # the other side. A zone reaching past the tread spins the free
        # stream, which is worse than the coincidence being repaired.
        if after >= measured.radius:
            warnings.append(
                f"wheel '{wheel}': sleeve '{sleeve_patch.name}' was pushed to "
                f"{after * 1e3:.1f} mm to clear the tyre surface, which reaches "
                f"or passes the {measured.radius * 1e3:.1f} mm tread. The "
                "rotating cell zone would extend into the free stream; reduce "
                "geometry.mrf_interference.interference"
            )

    moved_fits = {w: f for w, f in fits.items() if f.moved}
    if moved_fits:
        warnings.append(
            "MRF sleeve sat on the tyre surface and was pushed into it: "
            + ", ".join(
                f"{f.sleeve} gap {f.clearance * 1e6:.0f} um, radius "
                f"{f.radius_before * 1e3:.1f} -> {f.radius_after * 1e3:.1f} mm"
                for _, f in sorted(moved_fits.items())
            )
            + ". A faceZone lying on a wall makes baffles, multiply connected "
            "zones and non-manifold points; an intersecting one does not"
        )
    return fits, warnings


def apply_contact_patches(
    spec: CaseSpec, meshes: dict[str, trimesh.Trimesh]
) -> tuple[dict[str, ContactPatch], list[str]]:
    """Cut every tyre near the road and extrude the cut down through it.

    Modifies `meshes` in place, and only for patches with role 'tyre'.
    Bodywork that crosses the road is left for snappyHexMesh to clip: a
    splitter touching the tarmac is a condition being simulated, and squaring
    it off would change the shape rather than repair it.

    Must run after check_ground_placement() and derive_wheels(). Both measure
    the tyre as drawn - the depth it sinks into the road, and its rolling
    radius - and both are wrong if they see the extrusion instead. The
    extruded corners sit further from the wheel axis than the tread does, so
    a rolling radius taken from the cut surface reads several millimetres
    high and drives every wheel too fast.
    """
    settings = spec.geometry.contact_patch
    if not settings.enabled:
        return {}, []

    patches: dict[str, ContactPatch] = {}
    warnings: list[str] = []
    floor_z = -settings.depth_below_road

    for patch in spec.geometry.patches:
        if patch.role is not PatchRole.TYRE or patch.name not in meshes:
            continue

        cut, record = extrude_contact_patch(
            meshes[patch.name], settings.cut_height, floor_z
        )
        if record is None:
            continue
        meshes[patch.name] = cut
        patches[patch.name] = record

        cell = spec.surface_cell_size_for(patch)
        if settings.cut_height < cell:
            warnings.append(
                f"{patch.name}: the contact patch step is "
                f"{settings.cut_height * 1e3:.2f} mm against a "
                f"{cell * 1e3:.2f} mm surface cell on this patch. "
                "snappyHexMesh will smear a step it cannot resolve back into "
                "a ramp; raise geometry.contact_patch.cut_height above the "
                "cell size, or refine the tyre further"
            )

    if patches:
        warnings.append(
            "tyre contact patch: "
            + ", ".join(
                f"{n} {p.area * 1e6:.0f} mm2 over {p.depth * 1e3:.2f} mm"
                for n, p in sorted(patches.items())
            )
            + f", extruded to z = {floor_z * 1e3:.1f} mm"
        )
    return patches, warnings


def _case_insensitive_match(
    source_dir: Path, name: str, suffix: str
) -> Path | None:
    """A file whose name differs from the patch only in capitalisation.

    A concession to real exports, which arrive with things like TIre_RR.step.
    Accepted with a warning rather than silently, because the filename is the
    patch identity and a folder where that only nearly holds will eventually
    hold two files that differ by case alone.
    """
    wanted = f"{name}{suffix}".lower()
    matches = [p for p in source_dir.glob(f"*{suffix}") if p.name.lower() == wanted]
    return matches[0] if len(matches) == 1 else None


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
    """Where the geometry sits relative to the road.

    **Tyres are expected to cross z = 0.** A loaded tyre is modelled deflected
    into the road, and the part below the plane is the contact patch.
    Reporting that as an error would be rejecting correct CAD - and the cut in
    apply_contact_patches() needs something below the plane to work with, so a
    tyre that floats is now a mesh that has no footprint at all.

    Measured before the contact patch is applied, so the depth reported here
    is what the CAD drew rather than what the extrusion left.

    Bodywork below the road is reported loudly but does not stop the run.
    Attitude and ride height are set in CAD and are the modeller's to own, and
    at a big enough roll or dive a splitter really can touch the road - which
    is a condition to simulate, not a defect to reject. snappy clips it
    against the ground the same way it clips a tyre. Since it is equally a
    symptom of a mis-positioned export, it is never silent.
    """
    warnings: list[str] = []
    tyres = {p.name for p in spec.geometry.patches if p.role is PatchRole.TYRE}

    offenders = [
        f"{name} at {float(mesh.bounds[0][2]) * 1e3:.2f} mm"
        for name, mesh in meshes.items()
        if name not in tyres and float(mesh.bounds[0][2]) < 0.0
    ]
    if offenders:
        warnings.append(
            "geometry other than a tyre crosses the road plane at z = 0: "
            + "; ".join(sorted(offenders))
            + ". snappyHexMesh will clip it against the ground. Intended if "
            "the part is meant to touch the road at this attitude; otherwise "
            "the ride height in CAD is wrong"
        )

    contact = {
        name: float(meshes[name].bounds[0][2])
        for name in sorted(tyres & set(meshes))
    }
    if contact:
        if all(depth >= 0.0 for depth in contact.values()):
            warnings.append(
                "no tyre reaches the road plane at z = 0 (lowest is "
                f"{min(contact.values()) * 1e3:.2f} mm above it); the car is "
                "floating and has no contact patch"
            )
        else:
            warnings.append(
                "tyre contact patch depth: "
                + ", ".join(f"{n} {-d * 1e3:.2f} mm" for n, d in contact.items())
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

    # Slip is reported, never judged. An RC car spends most of its cornering
    # life in heavy understeer, so a wheel whose axis is well off square to
    # its direction of travel is the normal operating point rather than a
    # symptom. The pose is a modelling decision made in CAD; the pipeline's
    # job is to say what it measured, not to have an opinion about it.
    slips = [
        f"{n} {math.degrees(math.asin(min(1.0, speeds[n]['slip'] / speeds[n]['road_speed']))):.1f} deg"
        for n in sorted(speeds)
        if speeds[n]["road_speed"] > 0.0
    ]
    if slips:
        warnings.append("wheel slip angle: " + ", ".join(slips))

    for name in sorted(speeds):
        entry = speeds[name]
        wheel = entry["wheel"]
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

    meshes, renamed = _load_geometry(spec)
    if not meshes:
        raise FileNotFoundError("no geometry files were produced or found")

    if renamed:
        warnings.append(
            "filenames matched only by capitalisation: "
            + "; ".join(renamed)
            + ". The filename is the patch identity - rename the files so the "
            "match is exact"
        )

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

    # Blockage is checked against the *declared* reference area. The
    # pipeline used to compute a true projected area from the triangles and
    # use that instead; it no longer does, so this number is only as good as
    # forces.a_ref_full. Declaring an area smaller than the car's real
    # silhouette understates blockage in exactly the same proportion.
    warnings.extend(
        check_blockage(spec.a_ref_effective, domain, spec.domain.max_blockage)
    )
    warnings.extend(_wheel_warnings(spec, domain, wheels))

    # Last, and deliberately so: everything above measured the CAD as drawn,
    # and these are the only steps that change it.
    sleeve_fits, sleeve_warnings = fit_mrf_sleeves(spec, meshes, wheels)
    warnings.extend(sleeve_warnings)

    contact_patches, contact_warnings = apply_contact_patches(spec, meshes)
    warnings.extend(contact_warnings)

    geometry_files = _write_geometry(meshes, run_dir)
    shell_surface, shell_warnings = _write_vehicle_surface(spec, meshes, run_dir)
    warnings.extend(shell_warnings)

    result = PrepareResult(
        spec=spec,
        domain=domain,
        run_dir=run_dir,
        wheels=wheels,
        contact_patches=contact_patches,
        sleeve_fits=sleeve_fits,
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
                        "road_speed": speeds[name]["road_speed"],
                        "slip": speeds[name]["slip"],
                        # The attitude the CAD encodes, as a number rather
                        # than a warning. Large values are the normal
                        # operating point for an RC car in understeer.
                        "slip_deg": math.degrees(
                            math.asin(
                                min(
                                    1.0,
                                    speeds[name]["slip"]
                                    / speeds[name]["road_speed"],
                                )
                            )
                        )
                        if speeds[name]["road_speed"] > 0.0
                        else 0.0,
                    }
                    for name, w in wheels.items()
                },
                # What the cut actually produced. The footprint area is the
                # number to sanity-check against the load the tyre carries,
                # and `depth` says how far into the road the CAD drew it -
                # neither is recoverable from the written STL, because the
                # written STL is the one that has already been squared off.
                # Measured, and the repair if any. A sleeve welded to its
                # tyre is invisible in the written STL once it has been
                # pushed in, so the gap that triggered it is only on record
                # here.
                "mrf_sleeves": {
                    w: {
                        "clearance": f.clearance,
                        "interference": f.interference,
                        "radius_before": f.radius_before,
                        "radius_after": f.radius_after,
                        "tyre_radius": f.tyre_radius,
                    }
                    for w, f in sleeve_fits.items()
                },
                "contact_patches": {
                    name: {
                        "area": patch.area,
                        "depth": patch.depth,
                        "n_loops": patch.n_loops,
                        "cut_height": patch.cut_height,
                        "floor_z": patch.floor_z,
                    }
                    for name, patch in contact_patches.items()
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
                # What the shells could actually see. The written STL says
                # what the distance field was, but not what it was simplified
                # *from*, and that ratio is the whole reason the setting
                # exists. Empty when the case declares no shells.
                "shell_surface": shell_surface,
            },
        ),
    )
    return result
