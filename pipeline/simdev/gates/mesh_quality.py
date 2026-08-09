from __future__ import annotations

from simdev.config.schema import CaseSpec
from simdev.gates.base import GateResult
from simdev.geometry.roles import traits
from simdev.run.parsers import CheckMeshResult, LayerInfo


def check_mesh_quality(
    result: CheckMeshResult,
    layers: dict[str, LayerInfo],
    spec: CaseSpec,
) -> GateResult:
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
        reasons.append(f"checkMesh reported: {check}")

    # Layer coverage, only on patches that asked for layers. The denominator
    # is per patch and is what was actually asked of snappy (n_layers_for),
    # not the raw request - otherwise a patch whose stack was clamped to fit
    # its own cell reports a coverage shortfall it never had.
    for patch in spec.geometry.patches:
        if traits(patch.role).refinement != "high":
            continue
        requested = spec.n_layers_for(patch)
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
