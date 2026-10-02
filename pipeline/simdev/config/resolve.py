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


class UnknownDrivingStateError(KeyError):
    pass


def apply_driving_state(case: dict[str, Any]) -> dict[str, Any]:
    """Fold the selected driving state into the case, and drop the table.

    A driving state is a named bundle of everything that changes together
    when the car is doing something different: which geometry folder to read,
    how fast it is going, whether it is cornering and how tightly. Switching
    between tight cornering, a wide sweeper and braking is then one line at
    the top of the case file rather than an edit in four places that must
    agree.

    It is a merge layer rather than something read later, for the reason the
    whole config design exists: after resolve() there is one object in which
    every value is explicit, and nothing downstream may consult raw config to
    find out what it should have been.

    The table itself is removed once the selected state is merged. Keeping it
    would put every *unselected* state into the spec hash, so editing the
    braking state would invalidate cached cornering runs that cannot possibly
    have been affected by it.
    """
    states = case.get("driving_states")
    selected = case.get("driving_state")

    if not states:
        if selected:
            raise UnknownDrivingStateError(
                f"driving_state is {selected!r} but the case defines no "
                "driving_states block"
            )
        return case

    case = {k: v for k, v in case.items() if k != "driving_states"}

    if not selected:
        raise UnknownDrivingStateError(
            "the case defines driving_states "
            f"({', '.join(sorted(states))}) but driving_state selects none"
        )
    if selected not in states:
        raise UnknownDrivingStateError(
            f"unknown driving_state {selected!r}; the case defines "
            f"{', '.join(sorted(states))}"
        )

    return deep_merge(case, states[selected] or {})


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

    # Before the wall profile is chosen, because a driving state may set the
    # wall treatment along with everything else it changes.
    case = apply_driving_state(case)

    # A state from the CAD library (CAD/states/<name>/state.yaml) sits at the
    # same point in the merge as an inline driving state, for the same reason.
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
