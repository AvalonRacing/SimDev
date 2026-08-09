from __future__ import annotations

from pathlib import Path

import pytest

from simdev.run.parsers import (
    find_fatal_errors,
    parse_check_mesh,
    parse_layer_summary,
    read_force_coeffs,
    read_y_plus,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _log(name: str) -> str:
    return (FIXTURES / "logs" / name).read_text(encoding="utf-8")


def test_parse_check_mesh_reads_a_healthy_mesh() -> None:
    result = parse_check_mesh(_log("checkMesh_ok.log"))
    assert result.n_cells == 1122334
    assert result.max_non_ortho == pytest.approx(64.23)
    assert result.max_skewness == pytest.approx(3.21)
    assert result.has_negative_volumes is False
    assert result.failed_checks == []


def test_parse_check_mesh_detects_negative_volumes() -> None:
    result = parse_check_mesh(_log("checkMesh_bad.log"))
    assert result.has_negative_volumes is True


def test_parse_check_mesh_collects_failed_checks() -> None:
    result = parse_check_mesh(_log("checkMesh_bad.log"))
    assert result.failed_checks
    assert any("skewness" in c.lower() for c in result.failed_checks)


def test_parse_check_mesh_reads_bad_metrics() -> None:
    result = parse_check_mesh(_log("checkMesh_bad.log"))
    assert result.max_non_ortho == pytest.approx(82.51)
    assert result.max_skewness == pytest.approx(6.78)


def test_parse_layer_summary_reads_every_patch() -> None:
    layers = parse_layer_summary(_log("snappy_layers.log"))
    assert set(layers) == {"body", "stilts", "ground"}
    assert layers["body"].layers == pytest.approx(17.9)
    assert layers["body"].faces == 18345
    assert layers["body"].thickness == pytest.approx(0.000358)


def test_parse_layer_summary_captures_a_collapsed_stack() -> None:
    layers = parse_layer_summary(_log("snappy_layers.log"))
    assert layers["stilts"].layers == pytest.approx(4.2)
    assert layers["ground"].layers == pytest.approx(0.0)


def test_read_force_coeffs_uses_the_header_row() -> None:
    df = read_force_coeffs(FIXTURES / "coefficient.dat")
    assert list(df.columns[:3]) == ["Time", "Cd", "Cl"]
    assert len(df) == 5
    assert df["Cd"].iloc[-1] == pytest.approx(0.3475)


def test_read_y_plus_returns_per_patch_rows() -> None:
    df = read_y_plus(FIXTURES / "yPlus.dat")
    assert set(df["patch"]) == {"body", "ground"}
    body_last = df[df["patch"] == "body"].iloc[-1]
    assert body_last["max"] == pytest.approx(298.1)


def test_find_fatal_errors_detects_foam_fatal() -> None:
    text = "some output\n--> FOAM FATAL ERROR: keyword nu is undefined\nmore"
    assert find_fatal_errors(text)


def test_find_fatal_errors_detects_floating_point_exception() -> None:
    assert find_fatal_errors("Floating point exception\n")


def test_find_fatal_errors_is_quiet_on_a_clean_log() -> None:
    assert find_fatal_errors(_log("checkMesh_ok.log")) == []
