"""Freeing disk space. Both operations are refused for a running job by the
caller; nothing here knows about the queue."""

from __future__ import annotations

import shutil
from pathlib import Path


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def _is_time_dir(path: Path) -> bool:
    try:
        return path.is_dir() and float(path.name) != 0.0
    except ValueError:
        return False


def strip_mesh(run_dir: Path) -> int:
    """Delete the mesh and field data; keep everything a result is read from.

    Kept: results/, logs/, status/, postProcessing/, caseSpec.json, cad/ and
    the 0/ initial conditions (small, and they document the boundary setup).
    """
    run_dir = Path(run_dir)
    doomed = [
        *run_dir.glob("processor*"),
        run_dir / "constant" / "polyMesh",
        *(p for p in run_dir.iterdir() if _is_time_dir(p)),
    ]
    freed = 0
    for path in doomed:
        if path.exists():
            freed += _size(path)
            shutil.rmtree(path)
    return freed


def delete_run(run_dir: Path) -> None:
    shutil.rmtree(Path(run_dir))
