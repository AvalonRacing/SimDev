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

    if result.has_negative_volumes:
        reasons.append("mesh contains negative volume cells; it is unusable")

    if result.max_non_ortho > spec.mesh.max_non_ortho:
        reasons.append(
            f"max non-orthogonality {result.max_non_ortho:.1f} exceeds "
            f"{spec.mesh.max_non_ortho:.1f}; the solve will be unstable or inaccurate"
        )

    if result.max_skewness > spec.mesh.max_skewness:
        reasons.append(
            f"max skewness {result.max_skewness:.2f} exceeds "
            f"{spec.mesh.max_skewness:.2f}"
        )

    for check in result.failed_checks:
        if _is_spec_owned(check) and not spec.mesh.trust_check_mesh_verdict:
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
        reasons=reasons,
        detail={
            "n_cells": result.n_cells,
            "max_non_ortho": result.max_non_ortho,
            "max_skewness": result.max_skewness,
        },
    )
