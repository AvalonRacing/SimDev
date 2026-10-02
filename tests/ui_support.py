"""Shared fixtures for the web UI tests: a stocked fake library and a client."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from simdev.cad.library import DESIGN_PARTS, REPO_ROOT, STATE_PARTS, Library

CORNER = {
    "flow": {"u_inf": 15.0},
    "physics": {"mode": "cornering", "corner_radius": 4.0, "corner_direction": "left"},
    "domain": {"kind": "annulus"},
    "ground": {"motion": "static"},
}
STRAIGHT = {
    "flow": {"u_inf": 15.0},
    "physics": {"mode": "straight"},
    "domain": {"kind": "box"},
    "ground": {"motion": "moving"},
}
CAR_CASE = REPO_ROOT / "cases" / "car" / "config.yaml"


def upload(tmp_path: Path, parts, tag: str = "a") -> dict[str, Path]:
    folder = tmp_path / f"upload-{tag}"
    folder.mkdir(exist_ok=True)
    out = {}
    for part in parts:
        path = folder / f"{part}.step"
        path.write_bytes(b"ISO-10303-21;\n" + f"{part} {tag}".encode())
        out[path.name] = path
    return out


def stock_library(root: Path, tmp_path: Path) -> Library:
    """corner + straight states, design v01 with a slot for corner only."""
    lib = Library(root)
    lib.create_state("corner", "4 m left", CORNER, upload(tmp_path, STATE_PARTS, "c"))
    lib.create_state("straight", "", STRAIGHT, upload(tmp_path, STATE_PARTS, "s"))
    lib.create_design("v01")
    lib.add_slot("v01", "corner", upload(tmp_path, DESIGN_PARTS, "d"))
    return lib


@contextmanager
def make_client(tmp_path: Path, start_worker: bool = False, worker_command=None):
    from fastapi.testclient import TestClient

    from simdev.ui.app import create_app
    from simdev.ui.context import UIConfig

    library = stock_library(tmp_path / "CAD", tmp_path)
    config = UIConfig(
        runs_root=tmp_path / "runs",
        cad_root=tmp_path / "CAD",
        case_path=CAR_CASE,
        db_path=tmp_path / "ui.db",
        start_worker=start_worker,
    )
    app = create_app(config, library=library, worker_command=worker_command)
    with TestClient(app) as client:
        yield client, app
