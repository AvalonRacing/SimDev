"""Run the Ahmed case at three refinement levels and check grid convergence."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from simdev.cli import main as cli_main
from simdev.config.resolve import deep_merge
from simdev.report.results import aggregate

LEVELS: dict[str, dict[str, Any]] = {
    "coarse": {"base_cell_size": 0.08, "refinement": (3, 4)},
    "medium": {"base_cell_size": 0.05, "refinement": (4, 5)},
    "fine": {"base_cell_size": 0.03, "refinement": (5, 6)},
}


def monotonic_convergence(values: list[float]) -> bool:
    """True if the series moves consistently in one direction.

    Needs at least three levels: two points can always be joined by a line.
    """
    if len(values) < 3:
        return False
    deltas = [b - a for a, b in zip(values, values[1:])]
    return all(d > 0 for d in deltas) or all(d < 0 for d in deltas)


def within_tolerance(value: float, target: float, tolerance: float) -> bool:
    """True if `value` is within `tolerance` (a fraction) of `target`.

    A value sitting exactly on the band edge must be accepted. Comparing with
    a bare `<=` decides that case on the last bit of the subtraction, so a
    deviation of exactly 10% can read as 10.0000000000000009%.
    """
    error = abs(value - target)
    limit = abs(target) * tolerance
    return error <= limit or math.isclose(error, limit, rel_tol=1e-9)


def _level_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    level_min, level_max = settings["refinement"]
    return {
        "mesh": {
            "base_cell_size": settings["base_cell_size"],
            "surface_refinement_min": level_min,
            "surface_refinement_max": level_max,
        }
    }


def run_levels(
    case: Path,
    out_root: Path,
    levels: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run the case once per refinement level and aggregate the records.

    The level's mesh settings go through the config layer as a written-out
    case file per level, rather than being applied by hand. Running the same
    profile three times would produce three identical meshes and a grid
    independence study that cannot fail.
    """
    levels = levels if levels is not None else LEVELS
    base = yaml.safe_load(Path(case).read_text(encoding="utf-8"))

    run_dirs: list[Path] = []
    for name, settings in levels.items():
        level_dir = Path(out_root) / name
        level_dir.mkdir(parents=True, exist_ok=True)

        # Written out so each level's exact configuration is recoverable.
        level_case = level_dir / "config.yaml"
        level_case.write_text(
            yaml.safe_dump(deep_merge(base, _level_overrides(settings))),
            encoding="utf-8",
        )

        run_dir = level_dir / "run"
        cli_main(
            [
                "run",
                str(level_case),
                "--run-dir",
                str(run_dir),
                "--profile",
                "production",
            ]
        )
        run_dirs.append(run_dir)

    return aggregate(run_dirs)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--target-cd", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args()

    frame = run_levels(args.case, args.out_root)
    print(frame.to_string(index=False))

    cds = list(frame["cd_mean"])
    monotonic = monotonic_convergence(cds)
    accurate = within_tolerance(cds[-1], args.target_cd, args.tolerance)

    print(f"\nmonotonic convergence: {monotonic}")
    print(
        f"finest Cd {cds[-1]:.4f} vs target {args.target_cd:.4f} "
        f"({args.tolerance:.0%}): {accurate}"
    )
    return 0 if (monotonic and accurate) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
