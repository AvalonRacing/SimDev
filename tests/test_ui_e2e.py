"""Queue a real dev run through the HTTP API and wait for the pipeline.

Long (tens of minutes: a full mesh and 60 iterations), so it runs only when
asked: SIMDEV_UI_E2E=1 with OpenFOAM on PATH and the library migrated.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.openfoam,
    pytest.mark.skipif(shutil.which("simpleFoam") is None, reason="requires OpenFOAM on PATH"),
    pytest.mark.skipif(os.environ.get("SIMDEV_UI_E2E") != "1", reason="set SIMDEV_UI_E2E=1"),
]


def test_a_dev_run_queued_through_the_api_finishes(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from simdev.cad.library import REPO_ROOT, Library, default_cad_root
    from simdev.ui.app import create_app
    from simdev.ui.context import UIConfig

    library = Library(default_cad_root())
    if ("baseline", "testcase") not in library.pairs():
        pytest.skip("run scripts/migration/split_driving_states.py first")

    config = UIConfig(
        runs_root=tmp_path / "runs", cad_root=default_cad_root(),
        case_path=REPO_ROOT / "cases" / "car" / "config.yaml",
        db_path=tmp_path / "ui.db", start_worker=True,
    )
    app = create_app(config, library=library)
    with TestClient(app) as client:
        response = client.post(
            "/runs",
            data={"pair": "baseline/testcase", "profile": "car_dev", "run_name": "e2e",
                  "solve.max_iterations": "60"},
            follow_redirects=False,
        )
        assert response.status_code == 303, response.text
        queue = app.state.ctx.queue
        job = queue.latest_for("e2e")
        deadline = time.time() + 3 * 3600
        while time.time() < deadline:
            job = queue.get(job.id)
            if job.status not in ("queued", "running"):
                break
            time.sleep(10)

    # 60 iterations cannot fill two plateau windows, so the convergence gate
    # flags it: gate_failed is the expected verdict, a crash is not.
    assert job.status in ("done", "gate_failed"), job.error
    run_dir = tmp_path / "runs" / "e2e"
    assert (run_dir / "status" / "solve.json").is_file()
    assert (run_dir / "cad" / "Body.step").is_file()
