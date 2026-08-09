from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from simdev.run.runner import CommandResult, StageError
from simdev.run.status import read_status
from simdev.stages.mesh import mesh
from simdev.stages.prepare import prepare
from simdev.stages.solve import solve

FIXTURES = Path(__file__).parent / "fixtures"

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed",
        "ahmed": {"include_stilts": False},
        "patches": [
            {"name": "body", "role": "body"},
            {"name": "ground", "role": "ground"},
            {"name": "symmetry", "role": "symmetry"},
            {"name": "inlet", "role": "inlet"},
            {"name": "outlet", "role": "outlet"},
            {"name": "farfield", "role": "farfield"},
        ],
    },
}


class RecordingRunner:
    """Stands in for Runner; records argv and serves canned logs."""

    def __init__(self, case_dir: Path, logs: dict[str, str] | None = None) -> None:
        self.case_dir = Path(case_dir)
        self.log_dir = self.case_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.calls: list[list[str]] = []
        self.logs = logs or {}

    def _emit(self, name: str) -> CommandResult:
        log_path = self.log_dir / f"log.{name}"
        log_path.write_text(self.logs.get(name, "End\n"), encoding="utf-8")
        return CommandResult(name=name, argv=[], returncode=0, log_path=log_path)

    def run(self, argv, name=None, check=True):
        self.calls.append(list(argv))
        return self._emit(name or Path(argv[0]).name)

    def run_parallel(self, argv, n_ranks, name=None, check=True):
        self.calls.append(["mpirun", "-np", str(n_ranks), *argv, "-parallel"])
        return self._emit(name or Path(argv[0]).name)


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    case = tmp_path / "config.yaml"
    case.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    target = tmp_path / "run"
    prepare(case, target, profile="dev")
    return target


def _mesh_logs() -> dict[str, str]:
    return {
        "checkMesh": (FIXTURES / "logs" / "checkMesh_ok.log").read_text(),
        "snappyHexMesh": (FIXTURES / "logs" / "snappy_layers.log").read_text(),
    }


def _good_layers_log(n_layers: int) -> str:
    return (
        "patch      faces    layers   overall thickness\n"
        "                             [m]       [%]\n"
        "-----      -----    ------   ---------  ---\n"
        f"body       18345    {float(n_layers)}     0.000358   99.4\n"
        "ground     22000    0        0          0\n"
        "\n"
    )


def test_mesh_runs_the_expected_command_sequence(run_dir: Path) -> None:
    runner = RecordingRunner(run_dir, _mesh_logs())
    mesh(run_dir, runner=runner)
    executables = [c[0] if c[0] != "mpirun" else c[3] for c in runner.calls]
    assert executables == [
        "blockMesh",
        "surfaceFeatureExtract",
        "decomposePar",
        "snappyHexMesh",
        "checkMesh",
    ]


def test_mesh_uses_the_esi_feature_extraction_utility(run_dir: Path) -> None:
    """ESI OpenFOAM (openfoam.com, which the templates target) ships
    'surfaceFeatureExtract'. 'surfaceFeatures' is the OpenFOAM Foundation
    utility and does not exist in v2412 — calling it aborts the mesh stage
    on its second command."""
    runner = RecordingRunner(run_dir, _mesh_logs())
    mesh(run_dir, runner=runner)
    flat = [token for call in runner.calls for token in call]
    assert "surfaceFeatureExtract" in flat
    assert "surfaceFeatures" not in flat


def test_mesh_decomposes_with_force_so_it_can_be_re_run(run_dir: Path) -> None:
    """decomposePar aborts on an already-decomposed case.

    Without -force the mesh stage only ever works on a virgin run directory,
    which makes the pipeline's own --force flag unable to deliver a re-run.
    """
    runner = RecordingRunner(run_dir, _mesh_logs())
    mesh(run_dir, runner=runner)
    decompose = next(c for c in runner.calls if c[0] == "decomposePar")
    assert "-force" in decompose


def test_mesh_runs_snappy_in_parallel(run_dir: Path) -> None:
    runner = RecordingRunner(run_dir, _mesh_logs())
    mesh(run_dir, runner=runner)
    snappy = next(c for c in runner.calls if "snappyHexMesh" in c)
    assert snappy[0] == "mpirun"
    assert "-parallel" in snappy
    assert "-overwrite" in snappy


def test_mesh_gate_failure_is_recorded_and_raises(run_dir: Path) -> None:
    logs = _mesh_logs()
    logs["checkMesh"] = (FIXTURES / "logs" / "checkMesh_bad.log").read_text()
    runner = RecordingRunner(run_dir, logs)
    with pytest.raises(StageError):
        mesh(run_dir, runner=runner)
    status = read_status(run_dir, "mesh")
    assert status is not None
    assert status.state == "gate_failed"


def test_mesh_records_ok_when_gates_pass(run_dir: Path) -> None:
    from simdev.stages.common import load_spec

    logs = _mesh_logs()
    logs["snappyHexMesh"] = _good_layers_log(load_spec(run_dir).mesh.n_layers)
    runner = RecordingRunner(run_dir, logs)
    result = mesh(run_dir, runner=runner)
    assert result.passed is True
    assert read_status(run_dir, "mesh").state == "ok"


def test_solve_requires_a_successful_mesh(run_dir: Path) -> None:
    with pytest.raises(StageError) as exc:
        solve(run_dir, runner=RecordingRunner(run_dir))
    assert "mesh" in str(exc.value)


def _complete_mesh(run_dir: Path) -> None:
    from simdev.stages.common import load_spec

    logs = _mesh_logs()
    logs["snappyHexMesh"] = _good_layers_log(load_spec(run_dir).mesh.n_layers)
    mesh(run_dir, runner=RecordingRunner(run_dir, logs))


def _install_forces(run_dir: Path) -> None:
    target = run_dir / "postProcessing" / "forceCoeffs" / "0"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURES / "coefficient.dat", target / "coefficient.dat")


def test_solve_runs_simple_foam_in_parallel(run_dir: Path) -> None:
    _complete_mesh(run_dir)
    _install_forces(run_dir)
    runner = RecordingRunner(run_dir)
    solve(run_dir, runner=runner)
    call = next(c for c in runner.calls if "simpleFoam" in c)
    assert call[0] == "mpirun"
    assert "-parallel" in call


def test_solve_does_not_redecompose(run_dir: Path) -> None:
    _complete_mesh(run_dir)
    _install_forces(run_dir)
    runner = RecordingRunner(run_dir)
    solve(run_dir, runner=runner)
    flat = [token for call in runner.calls for token in call]
    assert "decomposePar" not in flat


def test_solve_records_non_converged_without_raising(run_dir: Path) -> None:
    _complete_mesh(run_dir)
    _install_forces(run_dir)  # only 5 iterations, cannot plateau
    result = solve(run_dir, runner=RecordingRunner(run_dir))
    assert result.converged is False
    status = read_status(run_dir, "solve")
    assert status is not None
    assert status.state == "gate_failed"
    # Means are still reported so the run is recorded, not lost.
    assert "Cd" in result.means


def test_mesh_reseeds_processor_zero_dirs_after_snappy(run_dir: Path) -> None:
    """decomposePar runs before snappy, so processor*/0 predates the surfaces.

    Left alone, those fields have no patchField for the patches snappy
    creates and simpleFoam aborts reading them. The stage must overwrite
    them from the rendered 0/ once the mesh exists.
    """
    (run_dir / "0").mkdir(parents=True, exist_ok=True)
    (run_dir / "0" / "p").write_text("rendered p with body entry", encoding="utf-8")

    stale = run_dir / "processor0" / "0"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "p").write_text("decomposed before snappy, no body", encoding="utf-8")

    mesh(run_dir, runner=RecordingRunner(run_dir, _mesh_logs()))

    assert (stale / "p").read_text(encoding="utf-8") == "rendered p with body entry"
