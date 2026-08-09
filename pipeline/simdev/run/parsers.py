from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_CELLS = re.compile(r"^\s*cells:\s*(\d+)", re.MULTILINE)
_NON_ORTHO = re.compile(r"Mesh non-orthogonality Max:\s*([-\d.eE+]+)")
_SKEWNESS = re.compile(r"Max skewness\s*=\s*([-\d.eE+]+)")
_NEGATIVE_VOLUME = re.compile(r"Zero or negative cell volume", re.IGNORECASE)
_LAYER_HEADER = re.compile(r"^\s*patch\s+faces\s+layers\b", re.IGNORECASE)

FATAL_PATTERNS = (
    "FOAM FATAL ERROR",
    "FOAM FATAL IO ERROR",
    # The stack frames OpenFOAM's own signal handlers print. More specific
    # than the OS messages below, and present even when the shell's
    # "(core dumped)" line is not captured.
    "Foam::sigFpe::sigHandler",
    "Foam::sigSegv::sigHandler",
    "Floating point exception",
    "Segmentation fault",
)

# Lines that contain a fatal-looking substring but report normal startup.
#
# Every OpenFOAM utility prints
#     trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).
# when FOAM_SIGFPE is set, which is the default in a stock v2412 install. It
# announces that FPE *trapping* is armed - it is not an FPE. Matching it as
# fatal fails every command in the pipeline on its own environment banner,
# including a blockMesh that ran perfectly.
BENIGN_PATTERNS = (
    "trapFpe:",
    "trapping enabled",
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

    v2412 prints six columns, splitting 'layers' into what was asked for and
    what the mesh actually got::

        patch  faces        layers        overall thickness
                        target   mesh     [m]       [%]
        -----  -----    -----    ----     ---       ---
        body   2485     5        4.28     0.00687   92.3

    Older builds print five, with a single 'layers' column. Both are accepted;
    the reported value is always the layers *achieved*, since that is what the
    coverage gate has to judge.

    Anchored on the header rather than on a run of dashes: snappy underlines
    its section headings the same way, and arming on the first dashed line
    finds 'Outer iteration : 0' long before the table, parses nothing, and
    hands the gate an empty dict it silently treats as 'no layers requested'.
    """
    lines = text.splitlines()

    # The last table is the final one; earlier iterations print their own.
    header = None
    for index, line in enumerate(lines):
        if _LAYER_HEADER.match(line):
            header = index
    if header is None:
        return {}

    separator = None
    for index in range(header, min(header + 4, len(lines))):
        if lines[index].lstrip().startswith("-----"):
            separator = index
            break
    if separator is None:
        return {}

    result: dict[str, LayerInfo] = {}
    for line in lines[separator + 1 :]:
        if not line.strip():
            break
        fields = line.split()
        # name faces target achieved thickness pct | name faces layers thickness pct
        if len(fields) == 6:
            name, faces, _target, layers, thickness = fields[:5]
        elif len(fields) == 5:
            name, faces, layers, thickness = fields[:4]
        else:
            break
        try:
            result[name] = LayerInfo(
                patch=name,
                faces=int(faces),
                layers=float(layers),
                thickness=float(thickness),
            )
        except ValueError:
            break

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
    """OpenFOAM often exits 0 on partial failure. Scan the log too.

    Benign startup banners are exempted first: a scanner that cries wolf on
    every successful run gets switched off, which costs more than the checks
    it was protecting.
    """
    return [
        line.strip()
        for line in text.splitlines()
        if any(pattern in line for pattern in FATAL_PATTERNS)
        and not any(benign in line for benign in BENIGN_PATTERNS)
    ]
