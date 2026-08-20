from __future__ import annotations

from simdev.config.schema import CaseSpec
from simdev.gates.base import GateResult
from simdev.geometry.roles import traits
from simdev.run.parsers import CheckMeshResult, LayerInfo


# The two quantities the spec carries its own thresholds for, and therefore
# the only checkMesh verdicts that `trust_check_mesh_verdict: false` drops.
#
# Deliberately not "ignore checkMesh". Everything else it fails a mesh on -
# negative volumes, non-closed cells, zero-area faces, multiple regions - is
# something the pipeline cannot detect for itself, so those keep gating
# whatever the case says. Turning the flag into a blanket mute would let a
# smoke profile sail past a mesh in two disconnected pieces.
SPEC_OWNED_CHECKS = ("skew", "non-orthogonal")


def _is_spec_owned(check: str) -> bool:
    lowered = check.lower()
    return any(token in lowered for token in SPEC_OWNED_CHECKS)


def check_mesh_quality(
    result: CheckMeshResult,
    layers: dict[str, LayerInfo],
    spec: CaseSpec,
    requested_layers: dict[str, int] | None = None,
) -> GateResult:
    """requested_layers is what snappy was actually asked for, per patch.

    Passed in rather than recomputed because it depends on the domain (the
    ground's cell size falls wherever a refinement region reaches the floor),
    and the gate has no domain.
    """
    reasons: list[str] = []
    # Recorded on the result but not fatal. Same pattern as the y+ gate:
    # worth knowing, not worth refusing a mesh over.
    notes: list[str] = []

    if result.has_negative_volumes:
        reasons.append("mesh contains negative volume cells; it is unusable")

    # Non-orthogonality and skewness are judged by how much of the mesh they
    # affect, not by the single worst face.
    #
    # The worst face used to gate directly, and on anything of production size
    # that is not a quality criterion, it is a lottery. The real mesh this was
    # changed for had 58 severely non-orthogonal faces and 27 highly skew ones
    # out of 21.1 M, an average non-orthogonality of 9.6, and checkMesh's own
    # non-orthogonality verdict was OK - and the gate failed it. Meanwhile the
    # same spec numbers are snappy's construction limits, so tightening the
    # gate made snappy build a worse mesh, which then failed the tighter gate.
    #
    # The max is still recorded in `detail` - it is worth knowing, it is just
    # not a verdict.
    if (
        result.mean_non_ortho == result.mean_non_ortho  # not NaN
        and result.mean_non_ortho > spec.mesh.max_mean_non_ortho
    ):
        reasons.append(
            f"average non-orthogonality {result.mean_non_ortho:.1f} exceeds "
            f"{spec.mesh.max_mean_non_ortho:.1f}; the non-orthogonal "
            "correction will not recover this and the solve will be inaccurate"
        )

    # Judging extent needs a denominator. Without one - an older log, a
    # truncated one, a caller that built the result by hand - fall back to the
    # single worst face rather than skipping the criterion, because a gate
    # that quietly stops checking is worse than one that is too strict.
    if result.n_faces > 0:
        limit = spec.mesh.max_bad_face_fraction
        non_ortho_fraction = result.n_severely_non_ortho / result.n_faces
        if non_ortho_fraction > limit:
            reasons.append(
                f"{result.n_severely_non_ortho} faces "
                f"({non_ortho_fraction:.2%}) are severely non-orthogonal, "
                f"above the {limit:.2%} allowed; that is enough of the mesh "
                "to make the solve unstable or inaccurate"
            )
        skew_fraction = result.n_highly_skew / result.n_faces
        if skew_fraction > limit:
            reasons.append(
                f"{result.n_highly_skew} faces ({skew_fraction:.2%}) are "
                f"highly skew, above the {limit:.2%} allowed"
            )
    else:
        if result.max_non_ortho > spec.mesh.max_non_ortho:
            reasons.append(
                f"max non-orthogonality {result.max_non_ortho:.1f} exceeds "
                f"{spec.mesh.max_non_ortho:.1f}, and checkMesh reported no "
                "face count to judge how much of the mesh is affected"
            )
        if result.max_skewness > spec.mesh.max_skewness:
            reasons.append(
                f"max skewness {result.max_skewness:.2f} exceeds "
                f"{spec.mesh.max_skewness:.2f}, and checkMesh reported no "
                "face count to judge how much of the mesh is affected"
            )

    # checkMesh's own verdict on skewness and non-orthogonality is the worst
    # face by another route: it judges against limits compiled into it (4 for
    # internal skewness), so one bad face out of twenty million fails the
    # mesh. Where this gate has a denominator it has already judged both
    # quantities by mean and by extent, which is strictly more informative, so
    # the verdict is recorded rather than fatal - otherwise it re-admits
    # through the back door exactly what the extent criteria were written to
    # stop gating on, and no production mesh can pass however good it is.
    #
    # Only the two quantities the spec measures for itself are demoted.
    # Everything else checkMesh fails a mesh on - negative volumes, non-closed
    # cells, zero-area faces, multiple regions - the pipeline cannot detect,
    # so those still gate whatever the case says.
    for check in result.failed_checks:
        if _is_spec_owned(check):
            if not spec.mesh.trust_check_mesh_verdict:
                continue
            if result.n_faces > 0:
                notes.append(f"checkMesh reported: {check}")
                continue
        reasons.append(f"checkMesh reported: {check}")

    # Layer coverage on every wall patch, since every wall carries a wall
    # function that assumes its first cell sits in the log layer. The
    # denominator is per patch and is what was actually asked of snappy, not
    # the raw request - otherwise a patch whose stack was clamped to fit its
    # own cell reports a coverage shortfall it never had.
    for patch in spec.geometry.patches:
        if not traits(patch.role).is_wall:
            continue
        requested = (
            requested_layers.get(patch.name, 0)
            if requested_layers is not None
            else spec.n_layers_for(patch)
        )
        if requested == 0:
            continue
        info = layers.get(patch.name)
        if info is None:
            # Never 'continue'. A patch that asked for layers and has no row
            # in the table means either snappy extruded nothing on it or the
            # table did not parse; both are failures, and skipping turns the
            # gate into a no-op that reports success.
            reasons.append(
                f"no layer data for '{patch.name}', which requested "
                f"{requested} layers; snappy extruded nothing there or the "
                "layer table could not be parsed (see logs/log.snappyHexMesh)"
            )
            continue
        coverage = info.layers / requested if requested else 1.0
        if coverage < spec.mesh.min_layer_coverage:
            reasons.append(
                f"layer coverage on '{patch.name}' is {coverage:.0%} "
                f"({info.layers:.1f} of {requested} layers); the near-wall "
                "resolution you designed for does not exist there"
            )

    return GateResult(
        passed=not reasons,
        reasons=reasons + notes,
        detail={
            "n_cells": result.n_cells,
            "n_faces": result.n_faces,
            "max_non_ortho": result.max_non_ortho,
            "mean_non_ortho": result.mean_non_ortho,
            "max_skewness": result.max_skewness,
            "n_severely_non_ortho": result.n_severely_non_ortho,
            "n_highly_skew": result.n_highly_skew,
        },
    )
