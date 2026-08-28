from __future__ import annotations

from simdev.config.schema import CaseSpec, GroundMotion, Mode, WheelRotation
from simdev.geometry.roles import PatchRole, force_roles, traits

MAX_PHYSICAL_CORES = 40
Y_PLUS_ERROR_FACTOR = 5.0

# Below this the prism stack cannot represent a boundary layer at all: the
# wall function is being applied across what is effectively one cell.
MIN_USEFUL_LAYERS = 3


class ValidationError(Exception):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def estimate_y_plus(spec: CaseSpec) -> float:
    """Flat-plate estimate of y+ at the first cell centre.

    Cf = 0.058 Re^-0.2, u_tau = u_inf sqrt(Cf/2), y+ = y u_tau / nu.
    Sizes the mesh; not a prediction.
    """
    re = spec.flow.u_inf * spec.forces.l_ref / spec.flow.nu
    cf = 0.058 * re ** -0.2
    u_tau = spec.flow.u_inf * (cf / 2.0) ** 0.5
    y_centre = spec.mesh.first_layer_thickness / 2.0
    return y_centre * u_tau / spec.flow.nu


def validate(spec: CaseSpec) -> list[str]:
    """Raise ValidationError on inconsistency. Return non-fatal warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    roles = [p.role for p in spec.geometry.patches]
    has_symmetry = PatchRole.SYMMETRY in roles

    # --- symmetry is derived; the patch set must agree with it -------------
    if spec.half_model and not has_symmetry:
        errors.append(
            "half_model is True but no patch has role 'symmetry'; "
            "a_ref would be halved for a model with no symmetry plane"
        )
    if not spec.half_model and has_symmetry:
        errors.append(
            "a 'symmetry' patch is present but the model is not a half model "
            f"(mode={spec.physics.mode.value}, yaw={spec.physics.yaw_deg}); "
            "yaw alone breaks symmetry"
        )

    # --- cornering -------------------------------------------------------
    if spec.physics.mode is Mode.CORNERING:
        if spec.physics.corner_radius is None:
            errors.append("mode is 'cornering' but corner_radius is not set")
        if spec.domain.kind != "annulus":
            errors.append(
                "mode is 'cornering' but domain.kind is "
                f"'{spec.domain.kind}'; cornering requires the 'annulus' domain"
            )
        if spec.geometry.patches and any(
            p.role is PatchRole.MRF_ZONE for p in spec.geometry.patches
        ):
            warnings.append(
                "cornering makes no MRF zone out of the wheel sleeves: the "
                "corner frame already covers every cell, and two adjacent "
                "MRF zones subtract each other's frame flux on the faces "
                "they share, which is every face of a sleeve boundary. "
                "Measured on a duct at 1 m/s, splitting one zone into two "
                "adjacent halves carrying the SAME frame moved the solution "
                "by 0.44 m/s. The sleeves stay in the mesh as refinement "
                "regions, and the wheels still turn - the tyre walls carry "
                "the full spin-plus-carry velocity through their coded "
                "condition. What is lost is rim pumping, which needs a "
                "sliding mesh rather than a second frame"
            )
        if spec.ground.motion is GroundMotion.MOVING:
            warnings.append(
                "ground.motion is 'moving', which does nothing in a cornering "
                "case: the road is stationary in the ground frame and its "
                "apparent motion comes from the rotating frame instead. The "
                "setting is ignored rather than applied twice"
            )
        if spec.domain.refinement_regions:
            warnings.append(
                f"{len(spec.domain.refinement_regions)} refinement region(s) "
                "are axis-aligned boxes, but a cornering wake follows the "
                "curve of the path and will leave them. Expect the far wake "
                "to be resolved at the background cell size"
            )
        if not spec.domain.refinement_shells:
            warnings.append(
                "no domain.refinement_shells: nothing refines the volume "
                "around the car, so the near field and the flow through the "
                "bodywork sit at the background cell size a cell or two off "
                "the wall. Shells are measured from the vehicle surface, so "
                "unlike a box they follow a cornering attitude"
            )
        elif spec.domain.wake is None:
            reach = max(s.distance for s in spec.domain.refinement_shells)
            warnings.append(
                f"refinement shells reach {reach:.2f} body lengths from the "
                "car. Beyond that the cornering far wake is still at the "
                "background cell size: shells follow the car, not the path, "
                "and no domain.wake region is declared to pick it up"
            )
        elif spec.domain.wake.end * 1.0 < max(
            s.distance for s in spec.domain.refinement_shells
        ):
            warnings.append(
                f"domain.wake.end ({spec.domain.wake.end:.2f} body lengths) is "
                "shorter than the vehicle shells' own reach "
                f"({max(s.distance for s in spec.domain.refinement_shells):.2f}"
                "); the wake region ends before the vehicle shells would have "
                "handed off to it"
            )
    elif spec.domain.kind == "annulus":
        errors.append(
            f"domain.kind is 'annulus' but mode is '{spec.physics.mode.value}'; "
            "the curved domain only describes a curved path"
        )

    # --- rotating wheels --------------------------------------------------
    wheel_ids = spec.geometry.wheel_ids()
    if wheel_ids and spec.physics.wheel_rotation is WheelRotation.LOCKED:
        warnings.append(
            f"{len(wheel_ids)} wheel(s) are declared but wheel_rotation is "
            "'locked', so they are modelled as stationary walls. Stationary "
            "wheels change the wake and the drag they generate; this is a "
            "known simplification, not a neutral one"
        )
    if (
        wheel_ids
        and spec.physics.mode is Mode.STRAIGHT
        and spec.ground.motion is GroundMotion.STATIC
        and spec.physics.wheel_rotation is WheelRotation.SPINNING
    ):
        warnings.append(
            "the ground is static but the wheels are set to spin, so the tyre "
            "contact patch moves and the road under it does not. Either both "
            "move or neither does"
        )
    for patch in spec.geometry.patches:
        if patch.wheel is not None and not (
            traits(patch.role).is_wall or patch.role is PatchRole.MRF_ZONE
        ):
            errors.append(
                f"patch '{patch.name}' declares wheel '{patch.wheel}' but its "
                f"role '{patch.role.value}' is neither a wall nor an MRF zone; "
                "nothing about it can rotate"
            )

    # --- patch set -------------------------------------------------------
    for required in (PatchRole.INLET, PatchRole.OUTLET, PatchRole.GROUND):
        if required not in roles:
            errors.append(f"no patch has role '{required.value}'")
    if not any(r in force_roles() for r in roles):
        errors.append(
            "no patch contributes to force integration; expected at least one "
            "patch with role 'body' or 'tyre'"
        )

    names = [p.name for p in spec.geometry.patches]
    if len(names) != len(set(names)):
        errors.append("duplicate patch names")

    # --- wall treatment vs mesh ------------------------------------------
    y_plus = estimate_y_plus(spec)
    lo, hi = spec.post.yplus_min, spec.post.yplus_max
    if y_plus > hi * Y_PLUS_ERROR_FACTOR or (lo > 0 and y_plus < lo / Y_PLUS_ERROR_FACTOR):
        errors.append(
            f"first_layer_thickness implies y+ ~= {y_plus:.1f}, far outside the "
            f"[{lo}, {hi}] band for wall treatment "
            f"'{spec.physics.wall_treatment.value}'"
        )
    elif not (lo <= y_plus <= hi):
        warnings.append(
            f"estimated y+ {y_plus:.1f} is outside the [{lo}, {hi}] band; "
            "the post-run y+ gate will confirm"
        )

    # --- prism stack vs the cell it is carved out of ----------------------
    # estimate_y_plus above checks first_layer_thickness against the *physics*.
    # This checks the whole stack against the *mesh*. Both must hold, and they
    # come from different config layers: wall treatment sets the layer sizes,
    # the resolution profile sets the cell size, and nothing else reconciles
    # them. A case that fails here meshes for 40 minutes and then loses its
    # layers, which the mesh gate reports as a coverage failure long after the
    # cause has scrolled past.
    requested = spec.mesh.n_layers
    stack = spec.layer_stack_thickness(requested)

    for patch in spec.geometry.patches:
        if traits(patch.role).refinement != "high":
            continue
        cell = spec.surface_cell_size_for(patch)
        budget = spec.layer_budget_for(patch)
        fits = spec.n_layers_for(patch)
        _min, level = spec.patch_refinement(patch)

        if fits == 0:
            errors.append(
                f"on patch '{patch.name}': first_layer_thickness "
                f"{spec.mesh.first_layer_thickness * 1e3:.3f} mm exceeds the "
                f"{budget * 1e3:.3f} mm layer budget of a {cell * 1e3:.3f} mm "
                f"surface cell (base_cell_size {spec.mesh.base_cell_size} / "
                f"2^{level}): not one layer fits. This refinement implies a "
                f"wall-resolved mesh, but wall treatment is "
                f"'{spec.physics.wall_treatment.value}'. Coarsen the "
                "refinement on this patch, or switch wall treatment"
            )
        elif fits < MIN_USEFUL_LAYERS:
            # A warning, not an error: a small appendage legitimately supports
            # fewer layers than the main body, and whether the wall treatment
            # actually holds there is measured by the y+ gate after the run,
            # not guessed from a flat-plate correlation before it.
            warnings.append(
                f"only {fits} prism layer(s) fit on patch '{patch.name}' "
                f"({cell * 1e3:.3f} mm cell, {budget * 1e3:.3f} mm budget); "
                f"fewer than {MIN_USEFUL_LAYERS} barely represents a boundary "
                "layer, so check the y+ gate on this patch"
            )
        elif fits < requested:
            warnings.append(
                f"requested {requested} prism layers but only {fits} fit on "
                f"patch '{patch.name}' ({cell * 1e3:.3f} mm cell); using "
                f"{fits}. The full stack would be {stack * 1e3:.3f} mm against "
                f"a {budget * 1e3:.3f} mm budget"
            )

    # --- mesh ------------------------------------------------------------
    if spec.mesh.surface_refinement_min > spec.mesh.surface_refinement_max:
        errors.append(
            f"surface refinement range inverted: min "
            f"{spec.mesh.surface_refinement_min} > max "
            f"{spec.mesh.surface_refinement_max}"
        )

    # --- refinement shells -----------------------------------------------
    # snappy applies the first shell whose distance contains the cell, so a
    # shell further out than a finer one can never take effect. Written the
    # wrong way round it is not an error snappy reports: it meshes happily and
    # quietly ignores the level you asked for.
    shells = sorted(spec.domain.refinement_shells, key=lambda s: s.distance)
    for nearer, further in zip(shells, shells[1:]):
        if further.level >= nearer.level:
            errors.append(
                f"refinement shell at {further.distance} body lengths asks for "
                f"level {further.level}, which is not coarser than the "
                f"level {nearer.level} shell at {nearer.distance}. Shells must "
                "get coarser with distance, or the outer one never applies"
            )

    for shell in shells:
        if shell.level > spec.mesh.surface_refinement_max:
            warnings.append(
                f"refinement shell at {shell.distance} body lengths asks for "
                f"level {shell.level}, finer than the case-wide surface "
                f"refinement of {spec.mesh.surface_refinement_max}. The volume "
                "away from the wall would be finer than the wall itself"
            )

    # --- parallel --------------------------------------------------------
    if spec.solve.n_ranks > MAX_PHYSICAL_CORES:
        errors.append(
            f"n_ranks {spec.solve.n_ranks} exceeds {MAX_PHYSICAL_CORES} physical "
            "cores; OpenFOAM is memory-bandwidth bound and hyperthreading "
            "reduces throughput"
        )

    # --- ground ----------------------------------------------------------
    # Not in cornering: there the road is *correctly* stationary in the
    # ground frame and its motion under the car comes from the rotating
    # frame. Warning about it there would contradict the cornering advice
    # above, and a pair of warnings that disagree teaches people to ignore
    # both.
    if (
        spec.geometry.kind != "ahmed"
        and spec.ground.motion is GroundMotion.STATIC
        and spec.physics.mode is not Mode.CORNERING
    ):
        warnings.append(
            "static ground under a vehicle case: ground-effect aerodynamics "
            "will be wrong unless you are deliberately matching a fixed-floor "
            "experiment"
        )

    if errors:
        raise ValidationError(errors)
    return warnings
