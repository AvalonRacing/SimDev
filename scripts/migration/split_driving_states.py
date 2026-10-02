"""One-off: move cases/car/config.yaml's driving states into the CAD library.

    .venv/bin/python -m scripts.migration.split_driving_states

For each entry under driving_states: the 13 attitude parts and the state's
parameters become CAD/states/<name>/, and that folder's Body and Wing become
the slot CAD/designs/baseline/<name>/. Files are copied, so the old
CAD/Testcase folder is untouched; delete it once a run from the library has
been checked.

Safe to run twice: a state that already exists in the library is skipped.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml

from simdev.cad.library import DESIGN_PARTS, Library

REPO = Path(__file__).resolve().parents[2]


def split(
    config: dict[str, Any], repo: Path, cad_root: Path, design: str = "baseline"
) -> list[str]:
    library = Library(cad_root)
    design_names = {part.lower() for part in DESIGN_PARTS}
    done: list[str] = []

    for name, entry in (config.get("driving_states") or {}).items():
        if (library.states_dir / name).exists():
            print(f"{name}: already in the library, skipped")
            continue

        params = copy.deepcopy(entry or {})
        geometry = params.get("geometry", {})
        source = repo / geometry.pop("source_dir", "")
        if not geometry:
            params.pop("geometry", None)
        if not source.is_dir():
            raise SystemExit(f"{name}: geometry folder {source} does not exist")

        files = {
            path.name: path
            for path in source.iterdir()
            if path.suffix.lower() in (".step", ".stp")
        }
        state_files = {k: v for k, v in files.items() if Path(k).stem.lower() not in design_names}
        design_files = {k: v for k, v in files.items() if Path(k).stem.lower() in design_names}

        library.create_state(
            name, f"migrated from cases/car/config.yaml ({source.name})", params, state_files
        )
        if not (library.designs_dir / design).exists():
            library.create_design(design)
        library.add_slot(design, name, design_files)
        done.append(name)
        print(f"{name}: states/{name} and designs/{design}/{name} written")

    return done


def main() -> int:
    config = yaml.safe_load((REPO / "cases/car/config.yaml").read_text(encoding="utf-8"))
    split(config, REPO, REPO / "CAD")
    print(
        "\nNext: remove driving_state/driving_states from cases/car/config.yaml, then\n"
        "  simdev run cases/car/config.yaml --run-dir ~/runs/<name> "
        "--profile car_dev --design baseline --state testcase"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
