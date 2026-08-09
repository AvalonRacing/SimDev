from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pydantic import ValidationError as SchemaValidationError

from simdev.config.validate import ValidationError
from simdev.report.results import aggregate
from simdev.run.runner import StageError
from simdev.stages.mesh import mesh
from simdev.stages.post import post
from simdev.stages.prepare import prepare
from simdev.stages.solve import solve

REQUIRED_UTILITIES = (
    "blockMesh",
    "surfaceFeatures",
    "snappyHexMesh",
    "checkMesh",
    "decomposePar",
    "simpleFoam",
    "mpirun",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simdev", description="OpenFOAM CFD pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("prepare", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("case", type=Path)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--profile", default="dev")
        sub.add_argument("--wall-treatment", default=None)
        sub.add_argument("--force", action="store_true")

    for name in ("mesh", "solve", "post"):
        sub = subparsers.add_parser(name)
        sub.add_argument("run_dir", type=Path)
        sub.add_argument("--force", action="store_true")

    subparsers.add_parser("doctor")

    agg = subparsers.add_parser("aggregate")
    agg.add_argument("run_dirs", type=Path, nargs="+")
    agg.add_argument("--out", type=Path, default=None)

    return parser


def _doctor() -> int:
    missing = []
    for utility in REQUIRED_UTILITIES:
        location = shutil.which(utility)
        print(f"{utility:18s} {location or 'NOT FOUND'}")
        if location is None:
            missing.append(utility)

    if shutil.which("surfaceFeatureExtract") and not shutil.which("surfaceFeatures"):
        print(
            "\nOnly the legacy 'surfaceFeatureExtract' is present. "
            "This pipeline targets ESI OpenFOAM v2412, which provides "
            "'surfaceFeatures'."
        )

    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print("See docs/environment-setup.md")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        if args.command == "doctor":
            return _doctor()

        if args.command == "aggregate":
            frame = aggregate(args.run_dirs)
            if args.out:
                frame.to_csv(args.out, index=False)
            print(frame.to_string(index=False))
            return 0

        if args.command in ("prepare", "run"):
            prepare(
                args.case,
                args.run_dir,
                profile=args.profile,
                wall_treatment=args.wall_treatment,
                force=args.force,
            )
            if args.command == "prepare":
                return 0

            mesh(args.run_dir, force=args.force)
            result = solve(args.run_dir, force=args.force)
            record = post(args.run_dir, force=args.force)
            print(
                f"Cd = {record.cd_mean:.4f} +/- {record.cd_std:.4f}   "
                f"Cl = {record.cl_mean:.4f} +/- {record.cl_std:.4f}   "
                f"converged = {record.converged}"
            )
            return 0 if result.converged and record.yplus_passed else 1

        if args.command == "mesh":
            mesh(args.run_dir, force=args.force)
            return 0
        if args.command == "solve":
            return 0 if solve(args.run_dir, force=args.force).converged else 1
        if args.command == "post":
            return 0 if post(args.run_dir, force=args.force).yplus_passed else 1

    # SchemaValidationError is pydantic's: a case file that is malformed rather
    # than merely inconsistent fails in model_validate, before our own
    # cross-file assertions ever run.
    except (StageError, ValidationError, SchemaValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    return 0
