from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from simdev.config.profiles import DEFAULTS, RESOLUTION_PROFILES, WALL_PROFILES
from simdev.config.schema import CaseSpec, WallTreatment


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `over` onto `base`. Neither input is mutated."""
    result = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class LegacyDrivingStatesError(ValueError):
    pass


def _refuse_inline_states(case: dict[str, Any]) -> None:
    """Driving states moved to the CAD library (CAD/states/<name>/state.yaml).

    Refused loudly rather than ignored: the schema ignores unknown keys, so a
    leftover driving_states block would otherwise be silently dropped and the
    run would use the case file's own speed and mode.
    """
    if "driving_states" in case or "driving_state" in case:
        raise LegacyDrivingStatesError(
            "the case file still carries driving_state/driving_states. States now "
            "live in CAD/states/<name>/state.yaml and are selected with --state; "
            "run `python -m scripts.migration.split_driving_states` and remove "
            "them from the case file"
        )


def resolve(
    case: dict[str, Any],
    profile: str,
    wall_treatment: str | None = None,
    overrides: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> CaseSpec:
    """Merge defaults, resolution profile, wall profile, case, state, overrides."""
    if profile not in RESOLUTION_PROFILES:
        raise KeyError(
            f"unknown profile {profile!r}; expected one of {sorted(RESOLUTION_PROFILES)}"
        )

    _refuse_inline_states(case)

    # A state from the CAD library, before the wall profile is chosen,
    # because a state may set the wall treatment along with everything else
    # it changes.
    if state:
        case = deep_merge(case, state)

    merged = deep_merge(DEFAULTS, RESOLUTION_PROFILES[profile])

    treatment_name = (
        wall_treatment
        or case.get("physics", {}).get("wall_treatment")
        or merged["physics"]["wall_treatment"]
    )
    treatment = WallTreatment(treatment_name)
    merged = deep_merge(merged, WALL_PROFILES[treatment])
    merged = deep_merge(merged, {"physics": {"wall_treatment": treatment.value}})

    merged = deep_merge(merged, case)

    # An explicit CLI treatment selected the wall profile above, so it must also
    # win the field itself. Otherwise the case file's value would survive the
    # merge and the spec would name one treatment while carrying another's mesh
    # and y+ numbers.
    if wall_treatment is not None:
        merged = deep_merge(merged, {"physics": {"wall_treatment": treatment.value}})

    if overrides:
        merged = deep_merge(merged, overrides)

    return CaseSpec.model_validate(merged)


def load_case(
    path: Path,
    profile: str,
    wall_treatment: str | None = None,
    overrides: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> CaseSpec:
    case = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return resolve(case, profile, wall_treatment, overrides, state)
