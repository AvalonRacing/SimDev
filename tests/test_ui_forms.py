from __future__ import annotations

from pathlib import Path

import pytest

from simdev.ui import forms
from simdev.ui.queue import JobSpec, Queue
from tests.ui_support import CAR_CASE, stock_library


@pytest.fixture
def lib(tmp_path: Path):
    return stock_library(tmp_path / "CAD", tmp_path)


def test_resolve_run_applies_the_state(lib) -> None:
    spec = forms.resolve_run(lib, CAR_CASE, "v01", "corner", "car_dev", {})
    assert spec.physics.corner_radius == 4.0
    assert spec.driving_state == "corner"


def test_resolve_run_refuses_a_pair_without_a_slot(lib) -> None:
    with pytest.raises(forms.FormError, match="no Body and Wing"):
        forms.resolve_run(lib, CAR_CASE, "v01", "straight", "car_dev", {})


def test_only_changed_values_become_overrides(lib) -> None:
    base = forms.resolve_run(lib, CAR_CASE, "v01", "corner", "car_dev", {})
    form = {"flow.u_inf": "15", "physics.corner_radius": "6", "physics.corner_direction": "left",
            "solve.n_ranks": str(base.solve.n_ranks), "solve.max_iterations": ""}
    assert forms.changed_overrides(form, base) == {"physics.corner_radius": 6.0}


def test_a_bad_number_is_a_form_error(lib) -> None:
    base = forms.resolve_run(lib, CAR_CASE, "v01", "corner", "car_dev", {})
    with pytest.raises(forms.FormError, match="Speed"):
        forms.changed_overrides({"flow.u_inf": "fast"}, base)


def test_cornering_fields_are_hidden_for_a_straight_state(tmp_path: Path, lib) -> None:
    lib.add_slot("v01", "straight", {
        p.name: p for p in (tmp_path / "upload-d").iterdir()
    })
    base = forms.resolve_run(lib, CAR_CASE, "v01", "straight", "car_dev", {})
    keys = [row["key"] for row in forms.field_rows(base)]
    assert "physics.corner_radius" not in keys
    assert forms.changed_overrides({"physics.corner_radius": "9"}, base) == {}


def test_preview_numbers_for_a_corner(lib) -> None:
    spec = forms.resolve_run(lib, CAR_CASE, "v01", "corner", "car_dev", {})
    numbers = forms.preview_numbers(spec)
    assert numbers["Yaw rate ω"] == "3.750 rad/s"
    assert numbers["Lateral acceleration"] == "5.73 g"


@pytest.mark.parametrize("name", ["../evil", "a/b", "", ".hidden", "a b"])
def test_run_names_with_paths_are_refused(tmp_path: Path, name: str) -> None:
    queue = Queue(tmp_path / "ui.db")
    with pytest.raises(forms.FormError, match="run name"):
        forms.check_run_name(name, tmp_path / "runs", queue)


def test_a_run_name_is_refused_when_its_directory_exists(tmp_path: Path) -> None:
    (tmp_path / "runs" / "taken").mkdir(parents=True)
    with pytest.raises(forms.FormError, match="already exists"):
        forms.check_run_name("taken", tmp_path / "runs", Queue(tmp_path / "ui.db"))


def test_default_run_names_count_up(tmp_path: Path) -> None:
    queue = Queue(tmp_path / "ui.db")
    first = forms.default_run_name(tmp_path / "runs", queue, "v01", "corner", "car_dev")
    assert first == "v01-corner-car_dev-001"
    queue.enqueue(JobSpec(first, "x", "c", "v01", "corner", "car_dev", 8))
    assert forms.default_run_name(tmp_path / "runs", queue, "v01", "corner", "car_dev").endswith("-002")
