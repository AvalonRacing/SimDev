from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from simdev.cli import main

CASE = Path(__file__).parent / "fixtures" / "ahmed.yaml"

pytestmark = pytest.mark.skipif(
    shutil.which("simpleFoam") is None, reason="requires OpenFOAM on PATH"
)


@pytest.mark.openfoam
def test_full_chain_runs_end_to_end(tmp_path: Path) -> None:
    """The whole pipeline on a deliberately tiny mesh. Target: 1-2 minutes."""
    run_dir = tmp_path / "run"
    code = main(
        [
            "run",
            str(CASE),
            "--run-dir",
            str(run_dir),
            "--profile",
            "dev",
        ]
    )

    # A dev-profile run is too coarse to converge; exit 1 is expected and fine.
    assert code in (0, 1)

    assert (run_dir / "results" / "result.json").exists()
    assert (run_dir / "results" / "forces.png").exists()
    assert (run_dir / "logs" / "log.snappyHexMesh").exists()

    from simdev.report.results import read_result

    record = read_result(run_dir)
    assert record.n_cells > 0
    assert record.cd_mean == record.cd_mean  # not NaN
