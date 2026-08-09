from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_CELLS = re.compile(r"^\s*cells:\s*(\d+)", re.MULTILINE)
_NON_ORTHO = re.compile(r"Mesh non-orthogonality Max:\s*([-\d.eE+]+)")
_SKEWNESS = re.compile(r"Max skewness\s*=\s*([-\d.eE+]+)")
_NEGATIVE_VOLUME = re.compile(r"Zero or negative cell volume", re.IGNORECASE)
_LAYER_ROW = re.compile(
    r"^\s*(\S+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$"
)

FATAL_PATTERNS = (
    "FOAM FATAL ERROR",
    "FOAM FATAL IO ERROR",
    "Floating point exception",
    "Segmentation fault",
)


@dataclass(frozen=True)
class CheckMeshResult:
    n_cells: int
    max_non_ortho: float
    max_skewness: float
    has_negative_volumes: bool
    failed_checks: list[str]


@dataclass(frozen=True)
class LayerInfo:
    patch: str
    faces: int
    layers: float
    thickness: float


def parse_check_mesh(text: str) -> CheckMeshResult:
    cells = _CELLS.search(text)
    non_ortho = _NON_ORTHO.search(text)
    skewness = _SKEWNESS.search(text)

    failed: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "FAILED" in stripped or stripped.startswith("***"):
            failed.append(stripped.lstrip("* "))

    return CheckMeshResult(
        n_cells=int(cells.group(1)) if cells else 0,
        max_non_ortho=float(non_ortho.group(1)) if non_ortho else float("nan"),
        max_skewness=float(skewness.group(1)) if skewness else float("nan"),
        has_negative_volumes=bool(_NEGATIVE_VOLUME.search(text)),
        failed_checks=failed,
    )


def parse_layer_summary(text: str) -> dict[str, LayerInfo]:
    """Parse snappyHexMesh's final layer table.

    The table is: patch, faces, layers, overall thickness [m], thickness [%].
    """
    result: dict[str, LayerInfo] = {}
    in_table = False

    for line in text.splitlines():
        if line.strip().startswith("-----"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.strip():
            break

        match = _LAYER_ROW.match(line)
        if not match:
            break
        name, faces, layers, thickness, _pct = match.groups()
        result[name] = LayerInfo(
            patch=name,
            faces=int(faces),
            layers=float(layers),
            thickness=float(thickness),
        )

    return result


def _last_header(path: Path) -> list[str]:
    header: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                header = line.lstrip("#").split()
            else:
                break
    return header


def read_force_coeffs(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", comment="#", names=_last_header(path), engine="python"
    )


def read_y_plus(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", comment="#", names=_last_header(path), engine="python"
    )


def find_fatal_errors(text: str) -> list[str]:
    """OpenFOAM often exits 0 on partial failure. Scan the log too."""
    return [
        line.strip()
        for line in text.splitlines()
        if any(pattern in line for pattern in FATAL_PATTERNS)
    ]
