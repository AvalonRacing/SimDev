from __future__ import annotations

from simdev.config.schema import CaseSpec, GroundMotion, Mode
from simdev.geometry.roles import PatchRole, force_roles

MAX_PHYSICAL_CORES = 40
Y_PLUS_ERROR_FACTOR = 5.0


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

    # --- mesh ------------------------------------------------------------
    if spec.mesh.surface_refinement_min > spec.mesh.surface_refinement_max:
        errors.append(
            f"surface refinement range inverted: min "
            f"{spec.mesh.surface_refinement_min} > max "
            f"{spec.mesh.surface_refinement_max}"
        )

    # --- parallel --------------------------------------------------------
    if spec.solve.n_ranks > MAX_PHYSICAL_CORES:
        errors.append(
            f"n_ranks {spec.solve.n_ranks} exceeds {MAX_PHYSICAL_CORES} physical "
            "cores; OpenFOAM is memory-bandwidth bound and hyperthreading "
            "reduces throughput"
        )

    # --- ground ----------------------------------------------------------
    if spec.geometry.kind == "stl" and spec.ground.motion is GroundMotion.STATIC:
        warnings.append(
            "static ground under a vehicle case: ground-effect aerodynamics "
            "will be wrong unless you are deliberately matching a fixed-floor "
            "experiment"
        )

    if errors:
        raise ValidationError(errors)
    return warnings
