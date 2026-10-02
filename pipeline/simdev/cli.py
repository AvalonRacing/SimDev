from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError as SchemaValidationError

from simdev.config.validate import ValidationError
from simdev.report.results import aggregate
from simdev.report.tsv import aggregate_reports
from simdev.run.runner import StageError
from simdev.stages.images import images
from simdev.stages.mesh import mesh
from simdev.stages.post import post
from simdev.stages.prepare import prepare
from simdev.stages.solve import solve

REQUIRED_UTILITIES = (
    "blockMesh",
    "surfaceFeatureExtract",
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
        sub.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="path.to.field=value",
            help=(
                "override one resolved field, e.g. "
                "--set physics.mode=cornering. Repeatable. This is the last "
                "layer of the merge, so it wins over the case file, and it "
                "lands in caseSpec.json like any other value - a sweep does "
                "not need a copy of the case per point."
            ),
        )
        sub.add_argument(
            "--design", default=None,
            help="design iteration from the CAD library (with --state)",
        )
        sub.add_argument(
            "--state", default=None,
            help="driving state from the CAD library (with --design)",
        )
        sub.add_argument(
            "--cad-root", type=Path, default=None,
            help="CAD library folder, default <repo>/CAD",
        )

    for name in ("mesh", "solve", "post"):
        sub = subparsers.add_parser(name)
        sub.add_argument("run_dir", type=Path)
        sub.add_argument("--force", action="store_true")

    img = subparsers.add_parser(
        "images",
        help="render the slice and surface suite for a solved run",
    )
    img.add_argument("run_dir", type=Path)
    img.add_argument("--force", action="store_true")
    img.add_argument(
        "--axes", nargs="+", default=None, choices=["x", "y", "z"],
        help="only these slice axes. Iterating on one view should not cost 562 images",
    )
    img.add_argument(
        "--fields", nargs="+", default=None,
        help="only these fields, e.g. --fields cp U",
    )

    subparsers.add_parser("doctor")

    agg = subparsers.add_parser("aggregate")
    agg.add_argument("run_dirs", type=Path, nargs="+")
    agg.add_argument("--out", type=Path, default=None)

    rep = subparsers.add_parser(
        "report",
        help=(
            "stack per-run results/report.tsv files into one table. Combines "
            "on read; never appends to a shared file"
        ),
    )
    rep.add_argument("run_dirs", type=Path, nargs="+")
    rep.add_argument("--out", type=Path, default=None)

    ui = subparsers.add_parser(
        "ui", help="serve the web UI on the Tailscale address (needs simdev[ui])"
    )
    ui.add_argument(
        "--host", default=None,
        help="address to listen on; default: this machine's Tailscale IPv4",
    )
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--runs-root", type=Path, default=Path.home() / "runs")
    ui.add_argument("--cad-root", type=Path, default=None)
    ui.add_argument("--case", type=Path, default=None, help="default: cases/car/config.yaml")
    ui.add_argument(
        "--db", type=Path, default=Path.home() / ".local/share/simdev-ui/ui.db"
    )

    return parser


def _doctor() -> int:
    missing = []
    for utility in REQUIRED_UTILITIES:
        location = shutil.which(utility)
        print(f"{utility:18s} {location or 'NOT FOUND'}")
        if location is None:
            missing.append(utility)

    # ESI builds (openfoam.com, vXXXX) ship 'surfaceFeatureExtract' and read
    # 'surfaceFeatureExtractDict'. The Foundation builds (openfoam.org) ship
    # 'surfaceFeatures' with a different, flat dictionary format. The templates
    # target ESI, so finding only 'surfaceFeatures' means the wrong distribution.
    if shutil.which("surfaceFeatures") and not shutil.which("surfaceFeatureExtract"):
        print(
            "\nOnly 'surfaceFeatures' is present, which is the OpenFOAM "
            "Foundation utility. This pipeline targets ESI OpenFOAM v2412 "
            "(openfoam.com), which provides 'surfaceFeatureExtract' and reads "
            "a differently-structured surfaceFeatureExtractDict."
        )

    # gmsh is a Python import rather than a utility on PATH, and it fails at
    # import time on a missing system library rather than at install time -
    # so a venv that pip reports as complete still cannot read a STEP file.
    # Checked here because this is the command someone runs when a fresh
    # machine misbehaves.
    try:
        import gmsh  # noqa: F401

        print(f"{'gmsh (STEP import)':18s} {gmsh.GMSH_API_VERSION}")
    except Exception as error:
        print(f"{'gmsh (STEP import)':18s} BROKEN: {error}")
        missing.append("gmsh")

    # ParaView, and NOT via pvpython. pvpython and pvbatch do not return on
    # this machine - measured, no output at a 150 s timeout, not even for
    # --version - while the same install imports fine under the plain system
    # interpreter. Checked here because the alternative is discovering it
    # after a solve, when the pictures are what is missing.
    interpreter = "/usr/bin/python3"
    probe = subprocess.run(
        [interpreter, "-c", "import paraview.simple"],
        capture_output=True, timeout=300,
    )
    if probe.returncode == 0:
        print(f"{'paraview (images)':18s} {interpreter}")
    else:
        print(f"{'paraview (images)':18s} BROKEN under {interpreter}")
        print(
            "\nThe images stage needs an interpreter that can 'import "
            "paraview.simple'. On Ubuntu:\n  sudo apt-get install -y "
            "python3-paraview\nSet post.paraview_python if it lives "
            "elsewhere. Do NOT point it at pvpython."
        )
        missing.append("paraview")

    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        if "gmsh" in missing:
            print(
                "gmsh imports a system library the wheel does not carry. On "
                "Ubuntu:\n  sudo apt-get install -y libglu1-mesa libopengl0 "
                "libxft2\nNote libGLU pulls in libOpenGL, so installing only "
                "libglu1-mesa moves the error rather than fixing it."
            )
        print("See docs/environment-setup.md")
        return 1
    return 0


def parse_overrides(assignments: list[str]) -> dict:
    """Turn `a.b=value` strings into the nested dict resolve() expects.

    Values go through the YAML scalar parser, so 3.0 is a float, cornering is
    a string and true is a boolean - the same reading they would get from the
    case file, rather than everything arriving as text and failing schema
    validation for the wrong reason.
    """
    import yaml

    overrides: dict = {}
    for assignment in assignments:
        key, separator, raw = assignment.partition("=")
        if not separator:
            raise ValueError(
                f"--set expects path.to.field=value, got {assignment!r}"
            )
        target = overrides
        parts = key.strip().split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(raw)
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        if args.command == "ui":
            from simdev.ui.serve import serve

            return serve(args.host, args.port, args.runs_root, args.cad_root,
                         args.case, args.db)
        if args.command == "doctor":
            return _doctor()

        if args.command == "aggregate":
            frame = aggregate(args.run_dirs)
            if args.out:
                frame.to_csv(args.out, index=False)
            print(frame.to_string(index=False))
            return 0

        if args.command == "report":
            header, rows = aggregate_reports(args.run_dirs)
            missing = [
                str(d) for d in args.run_dirs
                if not (Path(d) / "results" / "report.tsv").exists()
            ]
            if missing:
                # Loudly. A run quietly absent from a comparison table is how
                # a conclusion gets drawn from half the evidence.
                print(
                    "warning: no results/report.tsv, skipped: "
                    + ", ".join(missing),
                    file=sys.stderr,
                )
            text = "\n".join("\t".join(r) for r in [header, *rows]) + "\n"
            if args.out:
                args.out.write_text(text, encoding="utf-8")
            print(text, end="")
            return 0

        if args.command in ("prepare", "run"):
            prepare(
                args.case,
                args.run_dir,
                profile=args.profile,
                wall_treatment=args.wall_treatment,
                overrides=parse_overrides(args.overrides),
                force=args.force,
                design=args.design,
                state=args.state,
                cad_root=args.cad_root,
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
        if args.command == "images":
            record = images(
                args.run_dir, force=args.force, axes=args.axes, fields=args.fields
            )
            print(
                f"{record['images_written']} images  "
                f"sample {record['sample_seconds']}s  "
                f"render {record['render_seconds']}s"
            )
            for reason in record["reasons"]:
                print(f"  note: {reason}", file=sys.stderr)
            return 0

    # SchemaValidationError is pydantic's: a case file that is malformed rather
    # than merely inconsistent fails in model_validate, before our own
    # cross-file assertions ever run.
    except (StageError, ValidationError, SchemaValidationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    return 0
