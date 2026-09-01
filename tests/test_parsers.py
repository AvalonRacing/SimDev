from __future__ import annotations

from pathlib import Path

import pytest

from simdev.run.parsers import (
    find_fatal_errors,
    parse_check_mesh,
    parse_layer_summary,
    read_force_coeffs,
    read_force_vectors,
    read_y_plus,
    read_y_plus_area,
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


def test_parse_check_mesh_reads_the_face_count_not_the_internal_one() -> None:
    result = parse_check_mesh(_log("checkMesh_production.log"))
    assert result.n_faces == 21080791


def test_parse_check_mesh_reads_average_non_orthogonality() -> None:
    """The average is what governs solve accuracy; the max is one face."""
    result = parse_check_mesh(_log("checkMesh_production.log"))
    assert result.mean_non_ortho == pytest.approx(9.5751389)


def test_parse_check_mesh_counts_severely_non_orthogonal_faces() -> None:
    result = parse_check_mesh(_log("checkMesh_production.log"))
    assert result.n_severely_non_ortho == 58


def test_parse_check_mesh_counts_highly_skew_faces() -> None:
    result = parse_check_mesh(_log("checkMesh_production.log"))
    assert result.n_highly_skew == 27


def test_parse_check_mesh_counts_are_zero_when_checkmesh_is_silent() -> None:
    """A clean log names no counts, and absent must not read as unknown."""
    result = parse_check_mesh(_log("checkMesh_ok.log"))
    assert result.n_severely_non_ortho == 0
    assert result.n_highly_skew == 0
    assert result.mean_non_ortho == pytest.approx(8.12)


def test_parse_check_mesh_counts_a_bad_mesh() -> None:
    result = parse_check_mesh(_log("checkMesh_bad.log"))
    assert result.n_severely_non_ortho == 1234
    assert result.mean_non_ortho == pytest.approx(15.44)


def test_parse_layer_summary_reads_every_patch() -> None:
    layers = parse_layer_summary(_log("snappy_layers.log"))
    assert set(layers) == {"body", "stilts", "ground"}
    assert layers["body"].layers == pytest.approx(17.9)
    assert layers["body"].faces == 18345
    assert layers["body"].thickness == pytest.approx(0.000358)


def test_parse_layer_summary_captures_a_collapsed_stack() -> None:
    layers = parse_layer_summary(_log("snappy_layers.log"))
    assert layers["stilts"].layers == pytest.approx(4.2)
    assert layers["ground"].layers == pytest.approx(6.9)


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


# --- area-weighted y+ ------------------------------------------------------

_SFV = """\
# Region type : patch {patch}
# Faces  : 220096
# Area   : 1.8802373e+01
# Time          \tareaAverage(yPlus)
50\t{first}
125\t{last}
"""


def _write_area(run_dir: Path, patch: str, first: float, last: float) -> None:
    d = run_dir / "postProcessing" / f"yPlusArea_{patch}" / "0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "surfaceFieldValue.dat").write_text(
        _SFV.format(patch=patch, first=first, last=last), encoding="utf-8"
    )


def test_read_y_plus_area_returns_the_latest_value_per_patch(tmp_path: Path) -> None:
    _write_area(tmp_path, "ground", first=9.1, last=12.40)
    _write_area(tmp_path, "Body", first=1.0, last=1.34)

    assert read_y_plus_area(tmp_path) == pytest.approx({"ground": 12.40, "Body": 1.34})


def test_read_y_plus_area_is_empty_when_the_run_predates_the_function_objects(
    tmp_path: Path,
) -> None:
    """Runs meshed before these function objects existed have no such
    directories. The gate has to fall back rather than crash on them."""
    (tmp_path / "postProcessing").mkdir()
    assert read_y_plus_area(tmp_path) == {}


def test_read_y_plus_area_ignores_a_header_only_file(tmp_path: Path) -> None:
    """A solve killed before its first write time leaves the header and no
    rows. That is 'no measurement', not a measurement of zero."""
    d = tmp_path / "postProcessing" / "yPlusArea_ground" / "0"
    d.mkdir(parents=True)
    (d / "surfaceFieldValue.dat").write_text(
        "# Region type : patch ground\n# Time\tareaAverage(yPlus)\n", encoding="utf-8"
    )
    assert read_y_plus_area(tmp_path) == {}


def test_read_y_plus_area_does_not_depend_on_the_header_shape(
    tmp_path: Path,
) -> None:
    """surfaceFieldValue's header varies between OpenFOAM versions, and a
    header whose token count disagrees with the data columns is enough to make
    a name-based read raise. Time is the first column and the value is the
    last; nothing else about the header is load-bearing.
    """
    d = tmp_path / "postProcessing" / "yPlusArea_ground" / "0"
    d.mkdir(parents=True)
    (d / "surfaceFieldValue.dat").write_text(
        "# Region type : patch ground\n"
        "# Faces  : 220096\n"
        "# Area   : 1.8802373e+01\n"
        "# Time            areaAverage(yPlus) of field yPlus\n"
        "125\t12.40\n",
        encoding="utf-8",
    )
    assert read_y_plus_area(tmp_path) == pytest.approx({"ground": 12.40})


# --- force/moment vectors --------------------------------------------------


def test_read_force_vectors_takes_the_total_column() -> None:
    frame = read_force_vectors(FIXTURES / "force.dat")
    assert list(frame["Time"]) == [1.0, 2.0, 3.0]
    assert frame["z"].iloc[-1] == pytest.approx(-30.0)
    assert frame["x"].iloc[0] == pytest.approx(1.0)


def test_read_force_vectors_sums_when_there_is_no_total() -> None:
    """Older builds write pressure and viscous only.

    The total is recoverable by addition, so the layout is accepted rather
    than rejected - but it is recognised by column count, never assumed.
    """
    frame = read_force_vectors(FIXTURES / "force_two_triples.dat")
    assert frame["z"].iloc[0] == pytest.approx(-10.0)
    assert frame["x"].iloc[0] == pytest.approx(1.0)


def test_read_force_vectors_refuses_an_unknown_layout(tmp_path: Path) -> None:
    """A width this parser has not been taught is an error, not a guess.

    read_y_plus_area already learned this: a header whose token count
    disagrees with the data does not raise, it silently yields NaN, and a NaN
    that reaches a report reads as a number nobody checked.
    """
    bad = tmp_path / "force.dat"
    bad.write_text("# Time total_x\n1\t2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has not been taught"):
        read_force_vectors(bad)
