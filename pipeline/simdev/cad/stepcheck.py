"""Does a file open as STEP? Asked of every upload before it enters the library.

gmsh keeps global state and installs signal handlers in initialize(), which
must not happen on one of the web server's worker threads. The check therefore
runs in a child interpreter, one per batch, with the files checked in turn.
The header test runs first and in-process: it catches the common mistake (an
STL or a zip uploaded as .step) without starting gmsh at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HEADER = b"ISO-10303-21"
TIMEOUT_S = 300


def check_step_files(paths: list[Path]) -> dict[Path, str]:
    errors: dict[Path, str] = {}
    to_open: list[Path] = []
    for path in paths:
        with Path(path).open("rb") as handle:
            head = handle.read(256)
        if HEADER not in head:
            errors[path] = "not a STEP file (no ISO-10303-21 header)"
        else:
            to_open.append(path)

    if not to_open:
        return errors

    try:
        process = subprocess.run(
            [sys.executable, "-m", "simdev.cad.stepcheck", *map(str, to_open)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        for path in to_open:
            errors[path] = f"the STEP check did not finish within {TIMEOUT_S} s"
        return errors

    try:
        reported = json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        lines = process.stderr.strip().splitlines() or ["the STEP check crashed"]
        for path in to_open:
            errors[path] = lines[-1]
        return errors

    for path in to_open:
        message = reported.get(str(path))
        if message:
            errors[path] = message
    return errors


def _open_each(paths: list[str]) -> dict[str, str | None]:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        out: dict[str, str | None] = {}
        for path in paths:
            gmsh.clear()
            try:
                gmsh.model.occ.importShapes(path)
                gmsh.model.occ.synchronize()
                if not gmsh.model.getEntities(2):
                    out[path] = "opens, but contains no surfaces"
                else:
                    out[path] = None
            except Exception as error:
                out[path] = f"gmsh could not read it: {error}"
        return out
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    # One JSON line, last, so anything gmsh prints before it is ignored.
    print(json.dumps(_open_each(sys.argv[1:])))
