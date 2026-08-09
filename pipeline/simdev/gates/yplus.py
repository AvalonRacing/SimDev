from __future__ import annotations

import pandas as pd

from simdev.config.schema import CaseSpec
from simdev.gates.base import GateResult
from simdev.geometry.roles import traits


def check_y_plus(df: pd.DataFrame, spec: CaseSpec) -> GateResult:
    """Confirm the achieved y+ matches the band the wall treatment assumes."""
    force_patches = {
        p.name for p in spec.geometry.patches if traits(p.role).in_forces
    }
    lo, hi = spec.post.yplus_min, spec.post.yplus_max
    reasons: list[str] = []
    detail: dict[str, float | str] = {}

    latest = df[df["Time"] == df["Time"].max()]

    for _, row in latest.iterrows():
        patch = str(row["patch"])
        if patch not in force_patches:
            continue
        average = float(row["average"])
        detail[f"{patch}_avg_yplus"] = average
        if not (lo <= average <= hi):
            reasons.append(
                f"average y+ on '{patch}' is {average:.1f}, outside the "
                f"[{lo}, {hi}] band assumed by wall treatment "
                f"'{spec.physics.wall_treatment.value}'; the wall model is "
                "not valid there"
            )

    return GateResult(passed=not reasons, reasons=reasons, detail=detail)
