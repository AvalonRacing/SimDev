from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from simdev.run.runner import StageError
from simdev.run.status import StageStatus, read_status, write_status
from simdev.stages.images import images
from simdev.stages.prepare import prepare
from simdev.viz.sample import SampleRoots

CASE: dict = {
    "name": "ahmed",
    "flow": {"u_inf": 40.0, "turbulence_intensity": 0.01, "turbulence_length_scale": 0.01},
    "ground": {"motion": "static"},
    "forces": {"a_ref_full": 0.112032, "l_ref": 1.044},
    "geometry": {
        "kind": "ahmed", "symmetric": True, "ahmed": {"include_stilts": False},
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

TINY_VIEWS = """
datum:  {patches: [Chassis]}
planes:
  x: {from: -0.1, to: 0.1, step: 0.1}
  y: {from: 0.0, to: 0.0, step: 0.1}
  z: {from: 0.0, to: 0.0, step: 0.1}
fields:
  cp:      {limits: [-3.0, 1.0], colormap: spectrum}
  cpt:     {limits: [-3.0, 1.0], colormap: spectrum}
  U:       {limits: [0.0, 60.0], colormap: spectrum}
  lambda2: {limits: [-50000.0, 0.0], colormap: spectrum}
  yplus:   {limits: [0.0, 300.0], colormap: thermal}
camera:
  parallel_scale: {x: 0.4, y: 0.6, z: 0.6}
  focus_height: 0.15
  resolution: [320, 240]
streamlines: off
"""


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    views = tmp_path / "views.yaml"
    views.write_text(TINY_VIEWS, encoding="utf-8")
    case = dict(CASE, post={"views": str(views)})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    target = tmp_path / "run"
    prepare(path, target, profile="dev")
    return target


def test_images_refuses_before_a_solve(run_dir: Path) -> None:
    with pytest.raises(StageError, match="solve"):
        images(run_dir)


def test_prepare_records_the_datum_and_the_bounds(run_dir: Path) -> None:
    detail = read_status(run_dir, "prepare").detail
    assert len(detail["datum"]) == 3
    assert detail["datum"][2] == 0.0
    assert detail["corner_frame"] is None       # straight-line case
    assert len(detail["geometry_bounds"]) == 2


def test_the_sampling_dict_is_written_before_anything_runs(
    run_dir: Path, monkeypatch
) -> None:
    """The dict is the reviewable artefact. It must exist even when the
    OpenFOAM call is what fails."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(
        module, "run_sampling",
        lambda *a, **k: (_ for _ in ()).throw(StageError(["postProcess exploded"])),
    )
    with pytest.raises(StageError):
        images(run_dir)
    assert (run_dir / "system" / "sampleSurfaces").exists()


def test_the_plan_covers_the_configured_planes(run_dir: Path, monkeypatch) -> None:
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(
        module, "run_sampling",
        lambda *a, **k: SampleRoots(run_dir / "nowhere", run_dir / "nowhere"),
    )
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    images(run_dir)
    plan = json.loads((run_dir / "results" / "render_plan.json").read_text())
    # 3 x-planes, 1 y-plane, 1 z-plane.
    assert len(plan["slices"]) == 5
    assert len(plan["surfaces"]) == 7


def test_axes_and_fields_narrow_the_work(run_dir: Path, monkeypatch) -> None:
    """The flag exists so iterating on one view does not cost 562 images."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(
        module, "run_sampling",
        lambda *a, **k: SampleRoots(run_dir / "nowhere", run_dir / "nowhere"),
    )
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    images(run_dir, axes=["x"], fields=["cp"])
    plan = json.loads((run_dir / "results" / "render_plan.json").read_text())
    assert {s["axis"] for s in plan["slices"]} == {"x"}
    assert all(len(s["images"]) == 1 for s in plan["slices"])


def test_the_views_file_travels_with_the_pictures(run_dir: Path, monkeypatch) -> None:
    """A picture whose framing cannot be reconstructed is an orphan."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(
        module, "run_sampling",
        lambda *a, **k: SampleRoots(run_dir / "nowhere", run_dir / "nowhere"),
    )
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    record = images(run_dir)
    assert (run_dir / "results" / "views.yaml").exists()
    assert record["views_digest"] == json.loads(
        (run_dir / "results" / "images.json").read_text()
    )["views_digest"]


def test_slices_that_miss_the_car_are_reported(run_dir: Path, monkeypatch) -> None:
    """Silently drawing half the car is worse than saying the range is short."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    monkeypatch.setattr(
        module, "run_sampling",
        lambda *a, **k: SampleRoots(run_dir / "nowhere", run_dir / "nowhere"),
    )
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    record = images(run_dir)
    # The Ahmed body is 1.044 m long; the tiny views file spans 0.2 m.
    assert any("does not cover" in r for r in record["reasons"])


def test_the_vehicle_sample_is_used_for_every_surface_view(
    run_dir: Path, monkeypatch
) -> None:
    """Task 13 samples ONE combined 'vehicle' surface listing every force
    patch - not one sample per patch. Every one of the seven surface views is
    an overall view of the whole car, so each surface entry's sample must be
    that one file, not whichever per-patch sample happened to exist first.

    REGRESSION (found on the first real end-to-end run, 2026-09-01): 'vehicle'
    lives under postProcessing/patchSurfaces/<time>/, a SEPARATE directory
    from postProcessing/surfaces/<time>/ where the slice samples land -
    they are two different OpenFOAM function objects (see
    sampleSurfaces.jinja). The two roots are built here with that exact
    layout, matching what the real run produced; putting vehicle.vtp under
    the surfaces root instead - as an earlier version of this test did - made
    it indistinguishable from a bug that silently drew 350 of 364 images."""
    from simdev.stages import images as module
    from simdev.viz.sample import SampleRoots

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    surfaces_root = run_dir / "postProcessing" / "surfaces" / "400"
    surfaces_root.mkdir(parents=True)
    patches_root = run_dir / "postProcessing" / "patchSurfaces" / "400"
    patches_root.mkdir(parents=True)
    (patches_root / "vehicle.vtp").write_text("not a real vtp", encoding="utf-8")
    monkeypatch.setattr(
        module, "run_sampling",
        lambda *a, **k: SampleRoots(surfaces=surfaces_root, patches=patches_root),
    )
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    images(run_dir)
    plan = json.loads((run_dir / "results" / "render_plan.json").read_text())
    for entry in plan["surfaces"]:
        assert entry["sample"] == str(patches_root / "vehicle.vtp")


def test_a_solve_that_never_ran_is_refused(run_dir: Path) -> None:
    with pytest.raises(StageError):
        images(run_dir)


def test_a_solve_that_failed_outright_is_refused(run_dir: Path) -> None:
    write_status(run_dir, StageStatus("solve", "failed", "h", ["blew up"], {}))
    with pytest.raises(StageError):
        images(run_dir)


def test_a_gate_failed_solve_is_accepted(run_dir: Path, monkeypatch) -> None:
    """A run that did not plateau still has a flow field worth looking at."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "gate_failed", "h", ["did not plateau"], {}))
    monkeypatch.setattr(
        module, "run_sampling",
        lambda *a, **k: SampleRoots(run_dir / "nowhere", run_dir / "nowhere"),
    )
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    images(run_dir)  # must not raise


def test_a_run_prepared_before_this_feature_existed_is_refused(
    run_dir: Path,
) -> None:
    """No datum in status/prepare.json means prepare pre-dates the images
    stage; there is nothing measured to frame the pictures from."""
    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    prepare_status = read_status(run_dir, "prepare")
    stale_detail = {k: v for k, v in prepare_status.detail.items() if k != "datum"}
    write_status(
        run_dir,
        StageStatus("prepare", "ok", prepare_status.input_hash, [], stale_detail),
    )
    with pytest.raises(StageError):
        images(run_dir)


def test_the_window_comes_from_result_json_not_post_status(
    run_dir: Path, monkeypatch
) -> None:
    """RULING C2: post's status detail carries only cd_mean/cl_mean/verdict -
    window_start/window_end live on results/result.json's ResultRecord. When
    post has not run, images falls back to (0, 0) rather than raising, since
    images does not require post."""
    from simdev.stages import images as module

    write_status(run_dir, StageStatus("solve", "ok", "h", [], {}))
    # post's status detail deliberately has no window_start/window_end -
    # mirrors what stages/post.py actually writes.
    write_status(
        run_dir,
        StageStatus(
            "post", "ok", "h", [],
            {"cd_mean": 0.3, "cl_mean": -0.1, "verdict": "converged"},
        ),
    )
    monkeypatch.setattr(
        module, "run_sampling",
        lambda *a, **k: SampleRoots(run_dir / "nowhere", run_dir / "nowhere"),
    )
    monkeypatch.setattr(module, "_render", lambda *a, **k: {"written": 0, "missing": []})
    record = images(run_dir)
    plan = json.loads((run_dir / "results" / "render_plan.json").read_text())
    assert plan["stamp"]["window"] == "0-0"
    assert record  # completed without raising despite the missing keys
