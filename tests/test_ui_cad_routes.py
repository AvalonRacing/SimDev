from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from simdev.cad.library import DESIGN_PARTS, STATE_PARTS, LibraryError  # noqa: E402
from simdev.ui.routes_cad import params_from_form  # noqa: E402
from tests.ui_support import make_client  # noqa: E402


def files(parts, tag: str = "x"):
    return [("files", (f"{p}.step", b"ISO-10303-21;\n" + f"{p}{tag}".encode(),
                       "application/octet-stream")) for p in parts]


def test_params_for_a_corner() -> None:
    params = params_from_form("cornering", "12", "6", "right")
    assert params == {
        "flow": {"u_inf": 12.0},
        "physics": {"mode": "cornering", "corner_radius": 6.0, "corner_direction": "right"},
        "domain": {"kind": "annulus"},
        "ground": {"motion": "static"},
    }


def test_switching_to_straight_drops_the_corner() -> None:
    existing = params_from_form("cornering", "12", "6", "right")
    params = params_from_form("straight", "12", "", "left", existing)
    assert params["physics"] == {"mode": "straight"}
    assert params["domain"] == {"kind": "box"}


def test_a_corner_needs_a_radius() -> None:
    with pytest.raises(LibraryError, match="radius"):
        params_from_form("cornering", "12", "", "left")


def test_the_library_page_lists_states_and_slots(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        text = client.get("/cad").text
        assert "corner" in text and "straight" in text and "v01" in text


def test_upload_a_new_state(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        response = client.post(
            "/cad/states",
            data={"name": "braking", "description": "", "mode": "straight", "u_inf": "20",
                  "corner_radius": "", "corner_direction": "left"},
            files=files(STATE_PARTS), follow_redirects=False,
        )
        assert response.status_code == 303
        assert app.state.ctx.library.state("braking").params["flow"]["u_inf"] == 20.0


def test_a_state_upload_missing_a_part_is_explained(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        response = client.post(
            "/cad/states",
            data={"name": "braking", "mode": "straight", "u_inf": "20"},
            files=files(STATE_PARTS[1:]),
        )
        assert response.status_code == 400
        assert "missing parts: Chassis" in response.text


def test_add_a_slot_then_move_it(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        client.post("/cad/designs", data={"name": "v02"}, follow_redirects=False)
        client.post("/cad/designs/v02/slots", data={"state": "corner"},
                    files=files(DESIGN_PARTS), follow_redirects=False)
        client.post("/cad/designs/v02/slots/corner/move", data={"to": "straight"},
                    follow_redirects=False)
        assert ("v02", "straight") in app.state.ctx.library.pairs()


def test_moving_onto_an_existing_slot_asks_for_overwrite(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        client.post("/cad/designs/v01/slots", data={"state": "straight"},
                    files=files(DESIGN_PARTS, "s"), follow_redirects=False)
        response = client.post("/cad/designs/v01/slots/corner/move", data={"to": "straight"})
        assert response.status_code == 400
        assert "overwrite" in response.text
        response = client.post("/cad/designs/v01/slots/corner/move",
                               data={"to": "straight", "overwrite": "true"}, follow_redirects=False)
        assert response.status_code == 303
        assert app.state.ctx.library.pairs() == [("v01", "straight")]


def test_deleting_a_state_with_slots_is_refused(tmp_path: Path) -> None:
    with make_client(tmp_path) as (client, app):
        response = client.post("/cad/states/corner/delete")
        assert response.status_code == 400
        assert "v01/corner" in response.text


def test_a_hostile_name_cannot_break_out_of_the_confirm_dialogs(tmp_path: Path) -> None:
    # The library restricts names to [a-z0-9-], so no route can be fed a
    # hostile one; render the template directly with stub rows instead.
    from types import SimpleNamespace

    name = "x');alert(1);('"
    with make_client(tmp_path) as (client, app):
        row = {"state": SimpleNamespace(name=name, description=""), "mode": "straight",
               "u_inf": 1, "corner_radius": "", "corner_direction": "left", "slots": [name]}
        page = app.state.ctx.templates.get_template("cad.html").render(
            states=[row], designs={name: [name]}, state_names=[name, "other"],
            error=None, message=None,
        )
        assert "confirm('" not in page
        assert "confirm(&#39;" not in page
        assert page.count("\\u0027);alert(1);(\\u0027") == 4  # state, slot (design + state), design
