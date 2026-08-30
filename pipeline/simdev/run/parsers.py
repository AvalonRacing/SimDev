from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_CELLS = re.compile(r"^\s*cells:\s*(\d+)", re.MULTILINE)
_NON_ORTHO = re.compile(r"Mesh non-orthogonality Max:\s*([-\d.eE+]+)")
_MEAN_NON_ORTHO = re.compile(
    r"Mesh non-orthogonality Max:\s*[-\d.eE+]+\s*average:\s*([-\d.eE+]+)"
)
# 'faces:' but not 'internal faces:', so the leading anchor matters.
_FACES = re.compile(r"^\s*faces:\s*(\d+)", re.MULTILINE)
# checkMesh names these counts only when there are any, so an absent match
# means none - never unknown. Both are the quantity the gate actually judges:
# a max is one face, a count is how much of the mesh is affected.
_N_NON_ORTHO = re.compile(
    r"Number of severely non-orthogonal[^:]*:\s*(\d+)"
)
_N_SKEW = re.compile(r"(\d+)\s+highly skew faces")
_SKEWNESS = re.compile(r"Max skewness\s*=\s*([-\d.eE+]+)")
_NEGATIVE_VOLUME = re.compile(r"Zero or negative cell volume", re.IGNORECASE)
_LAYER_HEADER = re.compile(r"^\s*patch\s+faces\s+layers\b", re.IGNORECASE)
# renumberMesh reports the matrix band once for the mesh it read and once for
# the mesh it wrote, in that order, from the master rank only.
_BAND = re.compile(r"^\s*band\s*:\s*(\d+)", re.MULTILINE)

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
    # Everything below describes how *much* of the mesh is affected, rather
    # than how bad its single worst face is. Defaulted so the many callers
    # that build a result positionally keep working.
    n_faces: int = 0
    mean_non_ortho: float = float("nan")
    n_severely_non_ortho: int = 0
    n_highly_skew: int = 0


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

    faces = _FACES.search(text)
    mean_non_ortho = _MEAN_NON_ORTHO.search(text)
    n_non_ortho = _N_NON_ORTHO.search(text)
    n_skew = _N_SKEW.search(text)

    return CheckMeshResult(
        n_cells=int(cells.group(1)) if cells else 0,
        max_non_ortho=float(non_ortho.group(1)) if non_ortho else float("nan"),
        max_skewness=float(skewness.group(1)) if skewness else float("nan"),
        has_negative_volumes=bool(_NEGATIVE_VOLUME.search(text)),
        failed_checks=failed,
        n_faces=int(faces.group(1)) if faces else 0,
        mean_non_ortho=(
            float(mean_non_ortho.group(1)) if mean_non_ortho else float("nan")
        ),
        n_severely_non_ortho=int(n_non_ortho.group(1)) if n_non_ortho else 0,
        n_highly_skew=int(n_skew.group(1)) if n_skew else 0,
    )


def parse_renumber_band(text: str) -> tuple[int, int] | None:
    """Matrix band before and after renumbering, from a renumberMesh log.

    The band is how far apart the two cells sharing a face can be in the cell
    ordering, so it bounds how far the linear solver's indirect gathers reach
    through memory. Returns None when the log does not carry both numbers -
    a renumberMesh that failed, or one whose output was not captured - rather
    than guessing, because the pair is only meaningful as a ratio.
    """
    bands = _BAND.findall(text)
    if len(bands) < 2:
        return None
    return int(bands[0]), int(bands[1])


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


# Directory prefix the controlDict gives each per-patch area-average function
# object. One spelling, shared with the template's loop by convention and
# asserted by tests/test_render_solver.py.
Y_PLUS_AREA_PREFIX = "yPlusArea_"


def read_y_plus_area(run_dir: Path) -> dict[str, float]:
    """Area-weighted y+ per wall patch, at the latest time available.

    One surfaceFieldValue function object writes one directory per patch, so
    this collects them rather than reading a single file. See the controlDict
    template for why the weighted number is the one that gets judged: the
    plain yPlus object's mean is unweighted and reads low on any patch whose
    cell size varies.

    Returns {} rather than raising when the directories are absent, because
    runs meshed before these function objects existed are still worth
    post-processing - the gate falls back to the unweighted mean and says so.
    A file with a header and no rows is 'no measurement', not zero: a solve
    killed before its first write time leaves exactly that.
    """
    root = Path(run_dir) / "postProcessing"
    values: dict[str, float] = {}

    for directory in sorted(root.glob(f"{Y_PLUS_AREA_PREFIX}*")):
        patch = directory.name[len(Y_PLUS_AREA_PREFIX) :]
        latest: tuple[float, float] | None = None

        for dat in directory.glob("*/surfaceFieldValue.dat"):
            # Read positionally, NOT by header name. surfaceFieldValue's
            # header varies between OpenFOAM versions, and a header whose
            # token count disagrees with the data columns does not raise - it
            # silently produces NaN, which would then be compared against the
            # y+ band and quietly pass. Time is the first column and the value
            # is the last; nothing else about the header is load-bearing.
            try:
                frame = pd.read_csv(
                    dat, sep=r"\s+", comment="#", header=None, engine="python"
                )
            except pd.errors.EmptyDataError:
                # Comments and no rows: a solve killed before its first write
                # time leaves exactly this. No measurement, not a zero.
                continue
            if frame.empty:
                continue
            row = frame.iloc[-1]
            time = float(row.iloc[0])
            if latest is None or time > latest[0]:
                latest = (time, float(row.iloc[-1]))

        if latest is not None:
            values[patch] = latest[1]

    return values


def is_fatal_line(line: str) -> bool:
    """Whether one log line reports a fatal error.

    Split out of find_fatal_errors so the runner can apply the same judgement
    to a line as it streams past, without holding the whole log to scan it
    afterwards - a six-hour solve writes more log than is worth keeping in
    memory, and a run killed by a timeout never reaches the afterwards.
    """
    return any(pattern in line for pattern in FATAL_PATTERNS) and not any(
        benign in line for benign in BENIGN_PATTERNS
    )


def find_fatal_errors(text: str) -> list[str]:
    """OpenFOAM often exits 0 on partial failure. Scan the log too.

    Benign startup banners are exempted first: a scanner that cries wolf on
    every successful run gets switched off, which costs more than the checks
    it was protecting.
    """
    return [line.strip() for line in text.splitlines() if is_fatal_line(line)]
