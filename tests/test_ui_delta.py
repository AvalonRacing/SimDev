from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from simdev.ui import delta as service

FRAME = {"mode": "straight", "u_inf": 10.0, "omega": 0.0, "origin": [0, 0, 0]}


def make_run(root: Path, name: str, digest: str = "v1", spec_hash: str = "s") -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "postProcessing/surfaces/400").mkdir(parents=True)
    (run / "postProcessing/surfaces/400/x_+0.000.vtp").write_text("vtp")
    plan = {"views_digest": digest, "resolution": [16, 12], "frame": FRAME,
            "slices": [{"name": "x_+0.000", "axis": "x", "offset": 0.0,
                        "point": [0, 0, 0], "normal": [1, 0, 0], "camera": {"parallel_scale": 0.1},
                        "sample": f"/old/{name}/postProcessing/surfaces/400/x_+0.000.vtp",
                        "images": []}],
            "surfaces": [{"name": "top", "camera": {}, "sample": f"/old/{name}/postProcessing/patchSurfaces/400/vehicle.vtp", "images": []}]}
    (run / "results/render_plan.json").write_text(json.dumps(plan))
    (run / "results/result.json").write_text(json.dumps({"spec_hash": spec_hash}))
    return run


class FakeHelper:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, argv, **kwargs):
        with self.lock:
            self.calls += 1
        request = json.loads(Path(argv[-1]).read_text())
        assert kwargs["env"]["PATH"] == "/usr/bin:/bin"
        Path(request["out"]).write_bytes(b"\x89PNG")
        class Done:
            returncode = 0
            stdout = json.dumps({"ok": True, "out": request["out"]}) + "\n"
            stderr = ""
        return Done()


def test_request_uses_the_reference_frame_and_rerooted_samples(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    request = service.build_request(pane, ref, "x_+0.000", "cp", 0.2)
    assert request["kind"] == "plane"
    assert request["pane"]["sample"] == str(pane / "postProcessing/surfaces/400/x_+0.000.vtp")
    assert request["ref"]["sample"] == str(ref / "postProcessing/surfaces/400/x_+0.000.vtp")
    assert request["resolution"] == [16, 12]


def test_refusals(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a", digest="v2")
    with pytest.raises(service.DeltaError) as e:
        service.build_request(pane, ref, "x_+0.000", "cp", 0.2)
    assert e.value.status == 409
    ref = make_run(tmp_path, "c")
    for view, field, status in (("x_+9.000", "cp", 404), ("x_+0.000", "lambda2", 422),
                                ("surface_top", "cpt", 422)):
        with pytest.raises(service.DeltaError) as e:
            service.build_request(pane, ref, view, field, 0.2)
        assert e.value.status == status


def test_default_limits() -> None:
    assert service.default_limit("cp", 15.0) == 0.2
    assert service.default_limit("U", 15.0) == 1.5


def test_result_is_cached_and_concurrent_requests_run_the_helper_once(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    paths, busy = [], []

    def ask() -> None:
        try:
            paths.append(service.delta_png(pane, ref, "x_+0.000", "cp", None, runner=helper))
        except service.DeltaError as error:
            busy.append(error.status)

    threads = [threading.Thread(target=ask) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Requests that met a running compute were told to come back (503), not queued.
    assert set(busy) <= {503}
    assert helper.calls == 1
    paths.append(service.delta_png(pane, ref, "x_+0.000", "cp", None, runner=helper))
    assert helper.calls == 1
    assert len(set(paths)) == 1 and paths[0].read_bytes() == b"\x89PNG"
    assert paths[0].is_relative_to(pane / "ui" / "delta" / "a")


def test_a_busy_helper_answers_503_without_waiting(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    with service._LOCK:
        with pytest.raises(service.DeltaError) as e:
            service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    assert e.value.status == 503 and "another delta" in str(e.value)
    assert helper.calls == 0


def test_a_cached_picture_is_served_while_another_delta_computes(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    first = service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    with service._LOCK:
        again = service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    assert again == first and helper.calls == 1


def test_a_changed_plan_stamp_invalidates_the_cache(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    plan_file = ref / "results/render_plan.json"
    plan = json.loads(plan_file.read_text())
    plan["stamp"] = {"window": "100-400"}
    plan_file.write_text(json.dumps(plan))
    service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    assert helper.calls == 2


def test_the_helper_version_is_part_of_the_cache_key(tmp_path: Path, monkeypatch) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    monkeypatch.setattr(service, "HELPER_VERSION", service.HELPER_VERSION + 1)
    service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    assert helper.calls == 2


def _edit_plan(run: Path, change) -> None:
    plan_file = run / "results/render_plan.json"
    plan = json.loads(plan_file.read_text())
    change(plan)
    plan_file.write_text(json.dumps(plan))


def test_a_different_attitude_is_refused(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    _edit_plan(ref, lambda p: p["slices"][0].update(point=[0, 0, 0.01]))
    with pytest.raises(service.DeltaError) as e:
        service.build_request(pane, ref, "x_+0.000", "cp", 0.2)
    assert e.value.status == 409 and "not the same planes" in str(e.value)
    ref2 = make_run(tmp_path, "c")
    _edit_plan(ref2, lambda p: p["slices"][0].update(normal=[1, 0, 0.02]))
    with pytest.raises(service.DeltaError) as e:
        service.build_request(pane, ref2, "x_+0.000", "cp", 0.2)
    assert e.value.status == 409
    ref3 = make_run(tmp_path, "d")
    _edit_plan(ref3, lambda p: p["surfaces"][0].update(camera={"position": [1, 2, 3]}))
    with pytest.raises(service.DeltaError) as e:
        service.build_request(pane, ref3, "surface_top", "cp", 0.2)
    assert e.value.status == 409
    # identical framing up to round-off is fine
    ref4 = make_run(tmp_path, "e")
    _edit_plan(ref4, lambda p: p["slices"][0].update(point=[0, 0, 1e-9]))
    assert service.build_request(pane, ref4, "x_+0.000", "cp", 0.2)["kind"] == "plane"


def test_a_datum_a_few_micrometres_off_is_the_same_framing(tmp_path: Path) -> None:
    # c02_combo12 and its base differ by 2.0e-6 m in every slice point: the
    # datum is measured from each run's own Chassis tessellation.
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    _edit_plan(ref, lambda p: p["slices"][0].update(point=[2e-6, -2e-6, 0]))
    _edit_plan(ref, lambda p: p["surfaces"][0].update(camera={"focal": [2e-6, 0, 0]}))
    _edit_plan(pane, lambda p: p["surfaces"][0].update(camera={"focal": [0, 0, 0]}))
    assert service.build_request(pane, ref, "x_+0.000", "cp", 0.2)["kind"] == "plane"
    assert service.build_request(pane, ref, "surface_top", "cp", 0.2)["kind"] == "surface"


REAL_PANE = Path.home() / "runs/meshstudy-2026-10-04/sweep_tess1mm/c02_combo12"
REAL_REF = Path.home() / "runs/av001-tc10-cornering-lowspeed-car-001"


@pytest.mark.skipif(not (REAL_PANE / "results/render_plan.json").is_file()
                    or not (REAL_REF / "results/render_plan.json").is_file(),
                    reason="the real study runs are not on this machine")
def test_the_real_mesh_variant_pair_is_comparable_on_every_plane() -> None:
    plan = json.loads((REAL_REF / "results/render_plan.json").read_text())
    for entry in plan["slices"]:
        service.build_request(REAL_PANE, REAL_REF, entry["name"], "cp", 0.2)


def test_a_changed_reference_invalidates_the_cache(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    (ref / "results/result.json").write_text(json.dumps({"spec_hash": "changed"}))
    service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    assert helper.calls == 2


def test_helper_failure_is_reported(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")

    def failing(argv, **kwargs):
        class Done:
            returncode = 1
            stdout = json.dumps({"ok": False, "error": "KeyError: 'pMean'"}) + "\n"
            stderr = "trace"
        return Done()

    with pytest.raises(service.DeltaError) as e:
        service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=failing)
    assert e.value.status == 500 and "pMean" in str(e.value)


def test_invalid_limits(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    import math
    for bad_limit, desc in ((float('nan'), "nan"), (float('inf'), "inf"),
                            (0, "zero"), (-0.5, "negative")):
        with pytest.raises(service.DeltaError) as e:
            service.build_request(pane, ref, "x_+0.000", "cp", bad_limit)
        assert e.value.status == 422, f"failed for {desc}: {e.value}"


def test_limit_rounding_matches_filenames(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    # Two limits differing only beyond 6 decimals should use same file
    path1 = service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    path2 = service.delta_png(pane, ref, "x_+0.000", "cp", 0.2000001, runner=helper)
    assert path1 == path2
    assert helper.calls == 1


def test_different_limits_use_different_files(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    # Two clearly different limits should produce different files
    path1 = service.delta_png(pane, ref, "x_+0.000", "cp", 0.2, runner=helper)
    path2 = service.delta_png(pane, ref, "x_+0.000", "cp", 0.3, runner=helper)
    assert path1 != path2
    assert helper.calls == 2


def test_invalid_ref_name_is_refused(tmp_path: Path) -> None:
    pane, ref = make_run(tmp_path, "b"), make_run(tmp_path, "a")
    helper = FakeHelper()
    # Test with a ref_dir that has name ".."
    escaped_ref = tmp_path / ".."
    escaped_ref.mkdir(exist_ok=True)
    (escaped_ref / "results").mkdir(parents=True, exist_ok=True)
    (escaped_ref / "postProcessing/surfaces/400").mkdir(parents=True, exist_ok=True)
    (escaped_ref / "postProcessing/surfaces/400/x_+0.000.vtp").write_text("vtp")
    plan = {"views_digest": "v1", "resolution": [16, 12], "frame": FRAME,
            "slices": [{"name": "x_+0.000", "axis": "x", "offset": 0.0,
                        "point": [0, 0, 0], "normal": [1, 0, 0], "camera": {"parallel_scale": 0.1},
                        "sample": "/old/x/postProcessing/surfaces/400/x_+0.000.vtp",
                        "images": []}],
            "surfaces": []}
    (escaped_ref / "results/render_plan.json").write_text(json.dumps(plan))
    (escaped_ref / "results/result.json").write_text(json.dumps({"spec_hash": "s"}))

    with pytest.raises(service.DeltaError) as e:
        service.delta_png(pane, escaped_ref, "x_+0.000", "cp", 0.2, runner=helper)
    assert e.value.status == 404
    # Verify nothing outside ui/delta was touched
    assert not (pane / "ui" / "delta" / ".." / "results").exists()
