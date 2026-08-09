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


def test_parse_layer_summary_reads_the_v2412_six_column_table() -> None:
    """Captured verbatim from a real v2412 run.

    v2412 splits 'layers' into target and achieved, giving six columns. The
    achieved value is the one the coverage gate must see: here snappy was
    asked for 5 layers on both patches and delivered 4.28 on the body but
    only 1.23 on the stilts.
    """
    layers = parse_layer_summary(_log("snappy_layers_v2412.log"))
    assert set(layers) == {"body", "stilts"}
    assert layers["body"].faces == 2485
    assert layers["body"].layers == pytest.approx(4.28)
    assert layers["body"].thickness == pytest.approx(0.00687)
    assert layers["stilts"].layers == pytest.approx(1.23)


def test_parse_layer_summary_ignores_section_underlines() -> None:
    """'Outer iteration : 0' is underlined with dashes before the table.

    Arming the parse on the first dashed line finds that instead, parses
    nothing, and returns {} - which the gate used to read as 'nothing to
    check' and pass.
    """
    text = _log("snappy_layers_v2412.log")
    assert "Outer iteration" in text and "-------" in text
    assert parse_layer_summary(text) != {}


def test_parse_layer_summary_returns_empty_when_there_is_no_table() -> None:
    assert parse_layer_summary("snappyHexMesh died early\nEnd\n") == {}


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


def test_sigfpe_startup_banner_is_not_a_fatal_error() -> None:
    """Verbatim from a successful blockMesh under a stock v2412 install.

    FOAM_SIGFPE is on by default, so every utility prints this line. Treating
    it as fatal fails the entire pipeline at its first command.
    """
    banner = "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE)."
    assert find_fatal_errors(banner) == []
    assert find_fatal_errors(_log("blockMesh_ok.log")) == []


def test_a_real_sigfpe_is_still_caught_alongside_the_banner() -> None:
    """The banner must not mask an actual FPE later in the same log."""
    text = (
        "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n"
        "Time = 1\n"
        "#1  Foam::sigFpe::sigHandler(int) at ??:?\n"
    )
    found = find_fatal_errors(text)
    assert len(found) == 1
    assert "sigHandler" in found[0]
