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

    # Layer coverage, only on patches that asked for layers.
    requested = spec.mesh.n_layers
    for patch in spec.geometry.patches:
        if traits(patch.role).refinement != "high":
            continue
        info = layers.get(patch.name)
        if info is None:
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
