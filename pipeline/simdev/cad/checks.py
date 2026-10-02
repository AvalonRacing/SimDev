"""The library's two injected checks, wired to the real pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import ValidationError as SchemaValidationError

from simdev.cad.library import REPO_ROOT, Library, LibraryError, default_cad_root
from simdev.cad.stepcheck import check_step_files
from simdev.config.resolve import resolve
from simdev.config.validate import ValidationError, validate

DEFAULT_CASE = REPO_ROOT / "cases" / "car" / "config.yaml"


def state_params_check(
    case_path: Path, profile: str = "car_dev"
) -> Callable[[dict[str, Any]], None]:
    """A state that uploads is a state that resolves.

    Runs the same resolve() and validate() a real run will, so a mode typo or
    a cornering state without a radius is refused at upload, not at 2 a.m.
    """

    def check(params: dict[str, Any]) -> None:
        case = yaml.safe_load(Path(case_path).read_text(encoding="utf-8"))
        try:
            validate(resolve(case, profile, state=params))
        except (SchemaValidationError, ValidationError, ValueError, KeyError) as error:
            raise LibraryError(f"state parameters do not resolve: {error}") from error

    return check


def default_library(cad_root: Path | None = None, case_path: Path | None = None) -> Library:
    return Library(
        cad_root or default_cad_root(),
        step_check=check_step_files,
        params_check=state_params_check(case_path or DEFAULT_CASE),
    )
