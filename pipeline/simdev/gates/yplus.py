from __future__ import annotations

import pandas as pd

from simdev.config.schema import CaseSpec
from simdev.gates.base import GateResult
from simdev.geometry.roles import traits
# Imported from the renderer deliberately: it is the authority on which wall
# function each patch actually gets, and the gate asking that authority is the
# alternative to the two of them holding the same fact separately - which is
# exactly how a gate ends up judging against a treatment the case stopped
# using.
from simdev.render.context import y_plus_band_for


def check_y_plus(
    df: pd.DataFrame,
    spec: CaseSpec,
    area_weighted: dict[str, float] | None = None,
) -> GateResult:
    """Confirm the achieved y+ matches the band the wall treatment assumes.

    JUDGED ON THE AREA-WEIGHTED MEAN, NOT THE FACE MEAN, and the difference is
    not academic. `df` carries what OpenFOAM's yPlus function object reports,
    which is an unweighted average over the faces of a patch. It therefore
    counts a tiny face and a huge one equally, and the error that introduces
    is set by how y+ correlates with face size on that particular patch.

    THE DIRECTION IS NOT UNIVERSAL, which is worth stating because it is easy
    to assume otherwise. Measured on car-nut10-smooth (21.6 M cells):

        ground     face-mean 10.37   area-weighted 20.65    +99%
        SUS_FL     face-mean  1.130  area-weighted  1.031    -9%
        Tire_RL    face-mean  2.236  area-weighted  2.235     -0%

    The ground is bimodal - many tiny refined faces under the car at low y+,
    a few huge bare faces far away at high y+ - so the unweighted mean is
    dragged down and halves the real number. On the vehicle patches the
    correlation runs the other way and the unweighted mean reads slightly
    high instead. Either way it is the wrong average; only the sign of being
    wrong changes.

    `area_weighted` comes from the per-patch surfaceFieldValue function
    objects (see parsers.read_y_plus_area). It is allowed to be empty or
    partial: runs meshed before those objects existed still post-process, and
    a patch without a weighted number falls back to its face mean on its own.
    Every fallback is said out loud rather than assumed, because a gate that
    silently changed which number it judged would be worse than one that never
    had the weighted number at all.
    """
    area_weighted = area_weighted or {}
    force_patches = {
        p.name for p in spec.geometry.patches if traits(p.role).in_forces
    }
    wall_patches = {p.name for p in spec.geometry.patches if traits(p.role).is_wall}
    case_band = (spec.post.yplus_min, spec.post.yplus_max)
    roles = {p.name: p.role for p in spec.geometry.patches}
    reasons: list[str] = []
    warnings: list[str] = []
    detail: dict[str, float | str] = {}
    fell_back: list[str] = []

    latest = df[df["Time"] == df["Time"].max()]

    for _, row in latest.iterrows():
        patch = str(row["patch"])
        if patch not in wall_patches:
            continue

        face_mean = float(row["average"])
        weighted = area_weighted.get(patch)

        if weighted is None:
            fell_back.append(patch)
            judged, basis = face_mean, "face-mean"
        else:
            judged, basis = weighted, "area-weighted"
            # Recorded even though it was not judged: the gap between the two
            # is a direct read on how uneven the patch's cell size is, and it
            # is the number that says whether a patch near the bound is near
            # it honestly.
            detail[f"{patch}_facemean_yplus"] = face_mean

        detail[f"{patch}_avg_yplus"] = judged

        # The band this patch's own wall function is valid over, which is not
        # always the case-wide one - the ground carries Spalding whatever the
        # vehicle runs.
        lo, hi = y_plus_band_for(roles[patch], case_band)
        if lo <= judged <= hi:
            continue

        treatment = (
            spec.physics.wall_treatment.value
            if (lo, hi) == case_band
            else "this patch's own (see ROLE_WALL_FUNCTIONS)"
        )
        message = (
            f"{basis} y+ on '{patch}' is {judged:.1f}, outside the "
            f"[{lo}, {hi}] band assumed by wall treatment "
            f"'{treatment}'; the wall model is "
            "not valid there"
        )
        # Only force-bearing surfaces fail the gate - they are where the
        # reported numbers come from. Other walls still carry a wall function
        # and are still worth knowing about, so they are recorded rather than
        # dropped: an unresolved floor shapes the flow the body sits in.
        if patch in force_patches:
            reasons.append(message)
        else:
            warnings.append(message)

    if fell_back:
        warnings.append(
            "y+ on "
            + ", ".join(f"'{p}'" for p in sorted(fell_back))
            + " is NOT area-weighted: no surfaceFieldValue output was found "
            "for those patches, so the unweighted face mean was judged "
            "instead. That mean weights every face equally regardless of "
            "area, so it misreads any patch whose cell size varies - re-mesh "
            "with the current controlDict to get the weighted number"
        )
        detail["weighting"] = "face-mean" if not area_weighted else "mixed"
    else:
        detail["weighting"] = "area-weighted"

    return GateResult(
        passed=not reasons, reasons=reasons + warnings, detail=detail
    )
