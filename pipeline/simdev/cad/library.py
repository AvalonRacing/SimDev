"""The CAD library: driving states and design iterations, as plain files.

    CAD/states/<state>/            the 13 attitude parts + state.yaml
    CAD/designs/<design>/<state>/  Body.step + Wing.step for that state

A body is exported in the attitude of the state it will run in - ride height,
roll and pitch are baked into Body.step - so a design iteration is not one
Body.step. It is one Body/Wing pair per state, called a slot, and a run can
only be built from a (design, state) pair that has one.

Nothing here parses STEP or knows the case schema. Both checks are injected,
so the library can be tested with fake files and the web server can run the
STEP check out of process (see stepcheck.py and checks.py).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from simdev.geometry.step import CACHE_DIR_NAME

STATE_PARTS = (
    "Chassis",
    "SUS_FL", "SUS_FR", "SUS_RL", "SUS_RR",
    "Tire_FL", "Tire_FR", "Tire_RL", "Tire_RR",
    "MRF_FL", "MRF_FR", "MRF_RL", "MRF_RR",
)
DESIGN_PARTS = ("Body", "Wing")
STEP_SUFFIXES = (".step", ".stp")
STATE_FILE = "state.yaml"
NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")

# pipeline/simdev/cad/library.py -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

StepCheck = Callable[[list[Path]], dict[Path, str]]
ParamsCheck = Callable[[dict[str, Any]], None]


def default_cad_root() -> Path:
    return REPO_ROOT / "CAD"


class LibraryError(ValueError):
    """An operation the library refused, worded for the person who asked.

    A ValueError so the CLI's existing handler prints it as `error: ...`.
    """


def check_name(name: str, kind: str) -> str:
    # The name becomes a directory, part of a URL and part of a run name, so
    # it is restricted rather than escaped in three different ways.
    if not NAME_PATTERN.fullmatch(name or ""):
        raise LibraryError(
            f"{kind} name {name!r} must be lowercase letters, digits and dashes, "
            "starting with a letter or digit"
        )
    return name


def match_parts(
    files: Mapping[str, Path], expected: tuple[str, ...], require_all: bool
) -> dict[str, Path]:
    """Map uploaded filenames onto part names, ignoring case and .stp/.step.

    Every problem is collected before raising, so one failed upload reports
    everything wrong with it rather than one problem per attempt.
    """
    lookup = {part.lower(): part for part in expected}
    matched: dict[str, Path] = {}
    matched_files: dict[str, str] = {}  # part -> filename
    problems: list[str] = []

    for filename, path in files.items():
        stem, suffix = os.path.splitext(Path(filename).name)
        part = lookup.get(stem.lower())
        if suffix.lower() not in STEP_SUFFIXES:
            problems.append(f"{filename}: not a .step or .stp file")
        elif part is None:
            problems.append(
                f"{filename}: not one of the expected parts ({', '.join(expected)})"
            )
        elif part in matched:
            first_file = matched_files[part]
            problems.append(f"{filename}: a second file for part {part} (also {first_file})")
        else:
            matched[part] = Path(path)
            matched_files[part] = filename

    if require_all:
        missing = [part for part in expected if part not in matched]
        if missing:
            problems.append("missing parts: " + ", ".join(missing))
    if problems:
        raise LibraryError("; ".join(problems))
    if not matched:
        raise LibraryError("no files were uploaded")
    return matched


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_state_file(path: Path) -> tuple[str, dict[str, Any]]:
    """The description and the parameters of a state.yaml."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    description = str(raw.pop("description", "") or "")
    return description, raw


@dataclass(frozen=True)
class State:
    name: str
    description: str
    params: dict[str, Any]


class Library:
    def __init__(
        self,
        root: Path,
        step_check: StepCheck | None = None,
        params_check: ParamsCheck | None = None,
    ) -> None:
        self.root = Path(root)
        self._step_check = step_check
        self._params_check = params_check

    @property
    def states_dir(self) -> Path:
        return self.root / "states"

    @property
    def designs_dir(self) -> Path:
        return self.root / "designs"

    # --- reading -------------------------------------------------------------

    def states(self) -> list[State]:
        if not self.states_dir.is_dir():
            return []
        return [
            self.state(folder.name)
            for folder in sorted(self.states_dir.iterdir())
            if (folder / STATE_FILE).is_file()
        ]

    def state(self, name: str) -> State:
        if not NAME_PATTERN.fullmatch(name or ""):
            raise LibraryError(f"no driving state {name!r}")
        path = self.states_dir / name / STATE_FILE
        if not path.is_file():
            raise LibraryError(f"no driving state {name!r}")
        description, params = read_state_file(path)
        return State(name=name, description=description, params=params)

    def designs(self) -> dict[str, list[str]]:
        if not self.designs_dir.is_dir():
            return {}
        return {
            design.name: sorted(s.name for s in design.iterdir() if s.is_dir())
            for design in sorted(self.designs_dir.iterdir())
            if design.is_dir() and not design.name.startswith(".")
        }

    def pairs(self) -> list[tuple[str, str]]:
        return [
            (design, state)
            for design, states in self.designs().items()
            for state in states
            if (self.states_dir / state / STATE_FILE).is_file()
        ]

    def slots_for_state(self, state: str) -> list[str]:
        return [d for d, states in self.designs().items() if state in states]

    # --- checks --------------------------------------------------------------

    def _check_files(self, parts: dict[str, Path]) -> None:
        if self._step_check is None:
            return
        errors = self._step_check(list(parts.values()))
        if errors:
            part_of = {path: part for part, path in parts.items()}
            raise LibraryError(
                "; ".join(
                    f"{part_of.get(path, path.name)}: {message}"
                    for path, message in errors.items()
                )
            )

    def _check_params(self, params: dict[str, Any]) -> None:
        if self._params_check is not None:
            self._params_check(params)

    # --- writing helpers -----------------------------------------------------

    def _staging(self) -> Path:
        # Inside the library root so the final rename stays on one filesystem
        # and is atomic: a failed upload never leaves half a state behind.
        path = self.root / f".tmp-{uuid.uuid4().hex}"
        path.mkdir(parents=True)
        return path

    def _install(self, staging: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)

    @staticmethod
    def _copy_parts(parts: dict[str, Path], folder: Path) -> None:
        for part, source in parts.items():
            shutil.copyfile(source, folder / f"{part}.step")

    @staticmethod
    def _replace_files(folder: Path, parts: dict[str, Path]) -> None:
        for part, source in parts.items():
            temporary = folder / f".{part}.step.tmp"
            shutil.copyfile(source, temporary)
            os.replace(temporary, folder / f"{part}.step")

    @staticmethod
    def _write_state_file(folder: Path, description: str, params: dict[str, Any]) -> None:
        temporary = folder / f".{STATE_FILE}.tmp"
        temporary.write_text(
            yaml.safe_dump({"description": description, **params}, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, folder / STATE_FILE)

    def _build(self, target: Path, fill: Callable[[Path], None]) -> None:
        staging = self._staging()
        try:
            fill(staging)
            self._install(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    # --- states --------------------------------------------------------------

    def create_state(
        self, name: str, description: str, params: dict[str, Any], files: Mapping[str, Path]
    ) -> State:
        check_name(name, "state")
        target = self.states_dir / name
        if target.exists():
            raise LibraryError(f"driving state {name!r} already exists")
        parts = match_parts(files, STATE_PARTS, require_all=True)
        self._check_params(params)
        self._check_files(parts)

        def fill(folder: Path) -> None:
            self._copy_parts(parts, folder)
            self._write_state_file(folder, description, params)

        self._build(target, fill)
        return self.state(name)

    def update_state(self, name: str, description: str, params: dict[str, Any]) -> State:
        self.state(name)
        self._check_params(params)
        self._write_state_file(self.states_dir / name, description, params)
        return self.state(name)

    def replace_state_parts(self, name: str, files: Mapping[str, Path]) -> None:
        self.state(name)
        parts = match_parts(files, STATE_PARTS, require_all=False)
        self._check_files(parts)
        self._replace_files(self.states_dir / name, parts)

    def rename_state(self, old: str, new: str) -> None:
        check_name(new, "state")
        self.state(old)
        if (self.states_dir / new).exists():
            raise LibraryError(f"driving state {new!r} already exists")
        clashes = [d for d in self.slots_for_state(new)]
        if clashes:
            raise LibraryError(
                f"designs {', '.join(clashes)} already have a folder named {new!r}"
            )
        (self.states_dir / old).rename(self.states_dir / new)
        for design in self.slots_for_state(old):
            (self.designs_dir / design / old).rename(self.designs_dir / design / new)

    def delete_state(self, name: str) -> None:
        self.state(name)
        used = self.slots_for_state(name)
        if used:
            raise LibraryError(
                f"driving state {name!r} still has design slots: "
                + ", ".join(f"{d}/{name}" for d in used)
                + "; move or delete them first"
            )
        shutil.rmtree(self.states_dir / name)

    # --- designs and slots ---------------------------------------------------

    def _require_design(self, design: str) -> Path:
        if not NAME_PATTERN.fullmatch(design or ""):
            raise LibraryError(f"no design {design!r}")
        folder = self.designs_dir / design
        if not folder.is_dir():
            raise LibraryError(f"no design {design!r}")
        return folder

    def _require_slot(self, design: str, state: str) -> Path:
        if not NAME_PATTERN.fullmatch(state or ""):
            raise LibraryError(f"design {design!r} has no Body and Wing for state {state!r}")
        slot = self._require_design(design) / state
        if not slot.is_dir():
            raise LibraryError(f"design {design!r} has no Body and Wing for state {state!r}")
        return slot

    def create_design(self, name: str) -> None:
        check_name(name, "design")
        folder = self.designs_dir / name
        if folder.exists():
            raise LibraryError(f"design {name!r} already exists")
        folder.mkdir(parents=True)

    def add_slot(self, design: str, state: str, files: Mapping[str, Path]) -> None:
        folder = self._require_design(design)
        self.state(state)
        target = folder / state
        if target.exists():
            raise LibraryError(
                f"design {design!r} already has Body and Wing for {state!r}; "
                "replace them instead"
            )
        parts = match_parts(files, DESIGN_PARTS, require_all=True)
        self._check_files(parts)
        self._build(target, lambda staging: self._copy_parts(parts, staging))

    def replace_slot_parts(self, design: str, state: str, files: Mapping[str, Path]) -> None:
        slot = self._require_slot(design, state)
        parts = match_parts(files, DESIGN_PARTS, require_all=False)
        self._check_files(parts)
        self._replace_files(slot, parts)

    def move_slot(self, design: str, src: str, dst: str, overwrite: bool = False) -> None:
        """Re-file a Body/Wing pair under another state - the fix for an
        upload made against the wrong one."""
        slot = self._require_slot(design, src)
        self.state(dst)
        if src == dst:
            return
        target = self.designs_dir / design / dst
        if target.exists():
            if not overwrite:
                raise LibraryError(
                    f"design {design!r} already has Body and Wing for {dst!r}; "
                    "tick overwrite to replace them"
                )
            shutil.rmtree(target)
        slot.rename(target)

    def delete_slot(self, design: str, state: str) -> None:
        shutil.rmtree(self._require_slot(design, state))

    def rename_design(self, old: str, new: str) -> None:
        check_name(new, "design")
        folder = self._require_design(old)
        if (self.designs_dir / new).exists():
            raise LibraryError(f"design {new!r} already exists")
        folder.rename(self.designs_dir / new)

    def delete_design(self, name: str) -> None:
        shutil.rmtree(self._require_design(name))

    # --- assembly ------------------------------------------------------------

    def assemble(self, design: str, state: str, target: Path) -> dict[str, str]:
        """Copy the 15 parts of (design, state) into target; return their digests.

        Copied, not linked: deleting or replacing library CAD later must not
        change or break a run that was built from it. The tessellation cache
        is shared through a symlink instead - it is keyed by file content, so
        sharing it is safe, and without it every run would re-tessellate.
        """
        self.state(state)
        slot = self._require_slot(design, state)
        sources = {
            **{part: self.states_dir / state / f"{part}.step" for part in STATE_PARTS},
            **{part: slot / f"{part}.step" for part in DESIGN_PARTS},
        }
        missing = [part for part, path in sources.items() if not path.is_file()]
        if missing:
            raise LibraryError(
                f"{design}/{state} is incomplete, missing: {', '.join(missing)}"
            )

        target = Path(target)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

        digests: dict[str, str] = {}
        for part, source in sources.items():
            destination = target / f"{part}.step"
            shutil.copy2(source, destination)
            digests[part] = sha256(destination)
        # The parameters travel with the parts, so a run resumed after the
        # library state was edited still runs with the values it was built with.
        shutil.copy2(self.states_dir / state / STATE_FILE, target / STATE_FILE)

        cache = self.root / CACHE_DIR_NAME
        cache.mkdir(parents=True, exist_ok=True)
        (target / CACHE_DIR_NAME).symlink_to(cache, target_is_directory=True)
        return digests
