from __future__ import annotations

from pathlib import Path

import pytest

from simdev.viz.views import DEFAULT_VIEWS_PATH, load_views

REPO = Path(__file__).resolve().parents[1]


def test_the_shipped_definition_loads() -> None:
    views = load_views(REPO / "cases" / "post_views.yaml")
    assert views.datum_patches == ("Chassis",)
    assert views.resolution == (1600, 1200)
    assert views.streamlines == "off"


def test_the_plane_counts_are_what_the_spec_says() -> None:
    views = load_views(REPO / "cases" / "post_views.yaml")
    assert len(views.axes["x"].offsets()) == 32
    assert len(views.axes["y"].offsets()) == 21
    assert len(views.axes["z"].offsets()) == 17


def test_offsets_are_exact_so_two_runs_name_the_same_files() -> None:
    """Float drift in a plane position becomes a differently-named file.

    Two runs whose x_+0.120.png is called x_+0.11999999 in one of them cannot
    be laid side by side by any tool, so the offsets are rounded on the way
    out rather than left to accumulate.
    """
    views = load_views(REPO / "cases" / "post_views.yaml")
    offsets = views.axes["x"].offsets()
    assert offsets[0] == -0.30
    assert offsets[-1] == 0.32
    assert offsets[15] == pytest.approx(0.0, abs=1e-12)


def test_the_digest_changes_with_the_file(tmp_path: Path) -> None:
    """The digest is what ties a picture to the definition that made it."""
    first = tmp_path / "a.yaml"
    first.write_text(
        (REPO / "cases" / "post_views.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    second = tmp_path / "b.yaml"
    second.write_text(
        first.read_text(encoding="utf-8").replace("step: 0.02", "step: 0.04"),
        encoding="utf-8",
    )
    assert load_views(first).digest != load_views(second).digest


def test_the_shipped_definition_uses_the_starccm_colourmaps() -> None:
    views = load_views(REPO / "cases" / "post_views.yaml")
    assert {s.colormap for s in views.fields.values()} <= {"spectrum", "thermal"}


def test_an_unknown_colormap_is_rejected(tmp_path: Path) -> None:
    """ParaView ignores an unknown preset silently and keeps its default.

    Caught here or not at all: the symptom downstream is 364 pictures in the
    wrong colours and no error anywhere.
    """
    path = tmp_path / "views.yaml"
    path.write_text(
        (REPO / "cases" / "post_views.yaml")
        .read_text(encoding="utf-8")
        .replace("colormap: spectrum}", "colormap: chartreuse}", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown colormap"):
        load_views(path)


def test_an_unknown_streamline_mode_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "views.yaml"
    path.write_text(
        (REPO / "cases" / "post_views.yaml")
        .read_text(encoding="utf-8")
        .replace('streamlines: "off"', "streamlines: sparkles"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="streamlines"):
        load_views(path)


def test_every_rendered_field_has_limits() -> None:
    """A field with no limits would silently autoscale, which is the one
    thing this file exists to prevent."""
    from simdev.viz.views import SLICE_FIELDS, SURFACE_FIELDS

    views = load_views(REPO / "cases" / "post_views.yaml")
    for field in (*SLICE_FIELDS, *SURFACE_FIELDS):
        assert field in views.fields, field


def test_every_colour_limit_is_a_round_number() -> None:
    """Limits are read against other people's plots, so they must be numbers
    someone else would also have picked.

    A limit fitted to one run's percentiles - -0.96693, say - cannot be
    matched by anyone working from different data, and a scale nobody else
    can reproduce defeats the point of fixing it. Every limit here should be
    expressible in at most two significant figures.
    """
    views = load_views(REPO / "cases" / "post_views.yaml")
    for name, style in views.fields.items():
        for value in style.limits:
            if value == 0.0:
                continue
            magnitude = abs(value)
            scale = 10 ** (len(f"{int(magnitude)}") - 1) if magnitude >= 1 else 0.1
            quantum = scale / 10.0
            assert abs(value / quantum - round(value / quantum)) < 1e-9, (
                f"{name} limit {value} is not round to {quantum}"
            )
