"""The New Run form: what it offers, what it accepts, and what it previews."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from simdev.cad.library import Library, LibraryError
from simdev.config.profiles import RESOLUTION_PROFILES
from simdev.config.resolve import deep_merge, resolve
from simdev.config.schema import CaseSpec, Mode
from simdev.ui.queue import Queue

RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DEFAULT_PROFILE = "car_dev"
GRAVITY = 9.81


class FormError(ValueError):
    pass


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: type
    choices: tuple[str, ...] = ()
    cornering_only: bool = False


# The short list the spec settled on. Anything else is a new driving state or
# a profile, not a per-run knob.
FIELDS = (
    Field("flow.u_inf", "Speed u_inf [m/s]", float),
    Field("physics.corner_radius", "Corner radius [m]", float, cornering_only=True),
    Field("physics.corner_direction", "Corner direction", str, ("left", "right"), True),
    Field("solve.n_ranks", "MPI ranks", int),
    Field("solve.max_iterations", "Max iterations", int),
)


def profiles() -> list[str]:
    return sorted(RESOLUTION_PROFILES)


def split_pair(text: str | None) -> tuple[str, str] | None:
    if not text or "/" not in text:
        return None
    design, state = text.split("/", 1)
    return design, state


def nest(flat: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in flat.items():
        target = out
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return out


def _get(data: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        data = data.get(part) if isinstance(data, dict) else None
    return data


def resolve_run(
    library: Library,
    case_path: Path,
    design: str,
    state: str,
    profile: str,
    overrides: Mapping[str, Any],
) -> CaseSpec:
    """The same resolve() prepare will make, minus the CAD copy."""
    if profile not in RESOLUTION_PROFILES:
        raise FormError(f"unknown profile {profile!r}")
    if (design, state) not in library.pairs():
        raise FormError(f"design {design!r} has no Body and Wing for state {state!r}")
    case = yaml.safe_load(Path(case_path).read_text(encoding="utf-8"))
    layers = deep_merge({"driving_state": state}, nest(overrides))
    try:
        return resolve(case, profile, None, layers, state=library.state(state).params)
    except LibraryError as error:
        raise FormError(str(error)) from error
    except (ValueError, KeyError) as error:
        # pydantic's ValidationError is a ValueError.
        raise FormError(f"the run does not resolve: {error}") from error


def _visible(spec: CaseSpec) -> list[Field]:
    cornering = spec.physics.mode is Mode.CORNERING
    return [f for f in FIELDS if cornering or not f.cornering_only]


def defaults(spec: CaseSpec) -> dict[str, Any]:
    dumped = spec.model_dump(mode="json")
    return {f.key: _get(dumped, f.key) for f in FIELDS}


def field_rows(spec: CaseSpec, values: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    base = defaults(spec)
    values = values or {}
    return [
        {"key": f.key, "label": f.label, "choices": f.choices,
         "value": values.get(f.key, base[f.key])}
        for f in _visible(spec)
    ]


def changed_overrides(form: Mapping[str, Any], base: CaseSpec) -> dict[str, Any]:
    """Parse the override inputs and keep only values that differ from base.

    Unchanged values are dropped so the stored job, and the run's
    caseSpec.json, show exactly what was overridden and nothing else.
    """
    reference = defaults(base)
    out: dict[str, Any] = {}
    for f in _visible(base):
        raw = str(form.get(f.key, "") or "").strip()
        if raw == "":
            continue
        try:
            value = f.kind(raw)
        except ValueError:
            raise FormError(f"{f.label}: {raw!r} is not a number") from None
        if f.choices and value not in f.choices:
            raise FormError(f"{f.label}: must be one of {', '.join(f.choices)}")
        if value != reference[f.key]:
            out[f.key] = value
    return out


def preview_numbers(spec: CaseSpec) -> dict[str, str]:
    rows = {
        "Mode": spec.physics.mode.value,
        "Speed": f"{spec.flow.u_inf:g} m/s",
    }
    if spec.physics.mode is Mode.CORNERING and spec.physics.corner_radius:
        radius = spec.physics.corner_radius
        rows["Corner"] = f"{radius:g} m, {spec.physics.corner_direction.value}"
        rows["Yaw rate ω"] = f"{spec.omega_rotation:.3f} rad/s"
        rows["Lateral acceleration"] = f"{spec.flow.u_inf ** 2 / radius / GRAVITY:.2f} g"
    rows["ν_t/ν at inlet"] = f"{spec.nut_ratio:.1f}"
    rows["Domain"] = str(spec.domain.kind)
    rows["Ground"] = spec.ground.motion.value
    rows["MPI ranks"] = str(spec.solve.n_ranks)
    rows["Max iterations"] = str(spec.solve.max_iterations)
    return rows


def default_run_name(runs_root: Path, queue: Queue, design: str, state: str, profile: str) -> str:
    for n in range(1, 1000):
        name = f"{design}-{state}-{profile}-{n:03d}"
        if not (Path(runs_root) / name).exists() and not queue.name_taken(name):
            return name
    raise FormError("no free run name; pick one by hand")


def check_run_name(
    name: str, runs_root: Path, queue: Queue, exclude_id: int | None = None
) -> Path:
    # The name becomes a directory under runs_root; a slash or '..' in it
    # would put the run (and a later delete) somewhere else entirely.
    if not RUN_NAME.fullmatch(name or "") or ".." in name:
        raise FormError(
            f"run name {name!r}: use letters, digits, '.', '_' and '-', "
            "starting with a letter or digit"
        )
    path = Path(runs_root) / name
    if path.exists():
        raise FormError(f"run directory {path} already exists; pick another name")
    if queue.name_taken(name, exclude_id):
        raise FormError(f"a queued or running job already uses the name {name!r}")
    return path
