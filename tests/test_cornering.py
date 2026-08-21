from __future__ import annotations

import math

import numpy as np
import pytest

from simdev.config.schema import CaseSpec
from simdev.config.validate import ValidationError, validate
from simdev.domain.annulus import (
    MAX_USEFUL_SWEEP,
    AnnulusDomainBuilder,
    SectorMesh,
    check_sweep,
)
from simdev.domain.base import DomainSector
from simdev.geometry.wheels import measure_wheel
from simdev.render.context import build_bcs, corner_frame, mrf_zones, wheel_speeds

GEOM_BOUNDS = ([-0.22, -0.10, 0.0], [0.22, 0.10, 0.12])


def _case(**physics) -> dict:
    base = {
        "mode": "cornering",
        "corner_radius": 3.0,
        "corner_direction": "left",
        # The car's own wall treatment: at 0.44 m and 12 m/s the appendages
        # are far too small for wall functions to be valid on them.
        "wall_treatment": "low_y_plus",
    }
    base.update(physics)
    return {
        "name": "corner",
        "flow": {"u_inf": 12.0, "turbulence_length_scale": 0.01},
        "ground": {"motion": "static"},
        "physics": base,
        "domain": {"kind": "annulus"},
        "mesh": {
            "base_cell_size": 0.024,
            "surface_refinement_min": 3,
            "surface_refinement_max": 4,
            "n_layers": 12,
            "first_layer_thickness": 2.0e-5,
            "expansion_ratio": 1.15,
        },
        "forces": {"a_ref_full": 0.02, "l_ref": 0.44},
        "solve": {"max_iterations": 500, "n_ranks": 8},
        "post": {"yplus_min": 0.0, "yplus_max": 5.0},
        "geometry": {
            "kind": "stl",
            "source_dir": "cad",
            "patches": [
                {"name": "Body", "role": "body"},
                {"name": "Tire_FL", "role": "tyre", "wheel": "FL"},
                {"name": "MRF_FL", "role": "mrfZone", "wheel": "FL"},
                {"name": "ground", "role": "ground"},
                {"name": "inlet", "role": "inlet"},
                {"name": "outlet", "role": "outlet"},
                {"name": "farfield", "role": "farfield"},
            ],
        },
    }


def spec(**physics) -> CaseSpec:
    return CaseSpec.model_validate(_case(**physics))


def sector(**physics) -> DomainSector:
    return AnnulusDomainBuilder().build(spec(**physics), GEOM_BOUNDS)


# --- the derived half-model rule ------------------------------------------


def test_asymmetric_geometry_is_never_a_half_model() -> None:
    """The bug this fixes: straight and zero-yaw used to be enough."""
    straight = spec(mode="straight", corner_radius=None)
    assert straight.physics.yaw_deg == 0.0
    assert straight.half_model is False
    assert straight.a_ref_effective == straight.forces.a_ref_full


def test_symmetric_geometry_running_straight_is_a_half_model() -> None:
    case = _case(mode="straight", corner_radius=None)
    case["domain"]["kind"] = "box"
    case["geometry"]["symmetric"] = True
    case["geometry"]["patches"].append({"name": "symm", "role": "symmetry"})

    resolved = CaseSpec.model_validate(case)

    assert resolved.half_model is True
    assert resolved.a_ref_effective == pytest.approx(0.01)


def test_symmetric_geometry_cornering_is_still_not_a_half_model() -> None:
    case = _case()
    case["geometry"]["symmetric"] = True
    assert CaseSpec.model_validate(case).half_model is False


# --- the sign of the corner -----------------------------------------------


def test_left_corner_puts_the_centre_to_negative_y() -> None:
    """Car travels along -x, so with z up its left-hand side faces -y."""
    assert sector(corner_direction="left").centre[1] < 0.0


def test_right_corner_mirrors_the_left_one() -> None:
    left = sector(corner_direction="left")
    right = sector(corner_direction="right")

    assert left.centre[1] == pytest.approx(-right.centre[1])
    assert spec(corner_direction="left").omega_signed == pytest.approx(
        -spec(corner_direction="right").omega_signed
    )


def test_omega_is_u_over_r_with_a_sign() -> None:
    assert abs(spec().omega_signed) == pytest.approx(12.0 / 3.0)


def test_inlet_lies_upstream_of_the_car_for_both_directions() -> None:
    """The property that keeps a mirrored case from converging quietly.

    In the rotating frame the air at the car flows along +x. The inlet must
    therefore be on the -x side of the car, whichever way the corner goes.
    """
    for direction in ("left", "right"):
        domain = sector(corner_direction=direction)
        car = np.array(domain.point(domain.theta_car, domain.radius, 0.0))
        inlet = np.array(domain.point(domain.theta_inlet, domain.radius, 0.0))
        assert inlet[0] < car[0], direction


def test_the_frame_reproduces_the_freestream_at_the_car() -> None:
    """The check that ties omega, the sector and the flow direction together.

    Still air in the ground frame, seen from the rotating frame, must arrive
    at the vehicle as the case's own freestream: +x at u_inf. If any one of
    the three signs is wrong this comes out reversed or sideways.
    """
    for direction in ("left", "right"):
        resolved = spec(corner_direction=direction)
        domain = sector(corner_direction=direction)
        frame = corner_frame(resolved, domain)

        car = domain.point(domain.theta_car, domain.radius, 0.0)
        onset = frame.road_velocity_at(car)

        assert onset[0] == pytest.approx(resolved.flow.u_inf, rel=1e-9), direction
        assert onset[1] == pytest.approx(0.0, abs=1e-9), direction


# --- sector geometry ------------------------------------------------------


def test_arc_length_matches_the_requested_domain_length() -> None:
    """Sector length is the box's length, wrapped round the corner."""
    resolved = spec()
    domain = sector()
    body_length = GEOM_BOUNDS[1][0] - GEOM_BOUNDS[0][0]
    expected = (
        resolved.domain.upstream_lengths + resolved.domain.downstream_lengths
    ) * body_length

    assert domain.arc_length == pytest.approx(expected, rel=1e-9)


def test_radial_extent_straddles_the_path_radius() -> None:
    domain = sector()
    assert domain.r_min < domain.radius < domain.r_max
    assert domain.radius - domain.r_min == pytest.approx(domain.r_max - domain.radius)


def test_blocks_are_ordered_by_increasing_theta_whichever_way_it_turns() -> None:
    """Block handedness depends on the sign of the angular step."""
    for direction in ("left", "right"):
        domain = sector(corner_direction=direction)
        starts = [b.theta_start for b in domain.blocks]
        assert starts == sorted(starts), direction
        assert all(b.theta_end > b.theta_start for b in domain.blocks), direction


def test_the_sector_is_split_finely_enough_to_stay_round() -> None:
    domain = sector()
    limit = (math.pi / 2.0) / domain.blocks[0].n_circumferential  # generous bound
    for block in domain.blocks:
        assert block.theta_end - block.theta_start <= max(limit, math.pi / 24.0) + 1e-9


def test_a_tight_corner_warns_that_the_tunnel_meets_itself() -> None:
    domain = sector(corner_radius=0.6)
    assert domain.sweep > MAX_USEFUL_SWEEP
    assert any("re-enters" in w for w in check_sweep(domain))


# --- the generated block mesh ---------------------------------------------


def test_every_block_is_right_handed() -> None:
    """A left-handed hex is negative volume, reported far from its cause."""
    for direction in ("left", "right"):
        domain = sector(corner_direction=direction)
        mesh = SectorMesh(domain)
        points = np.asarray(mesh.vertices)

        for corners, _ in mesh.hexes:
            origin = points[corners[0]]
            local_x = points[corners[1]] - origin
            local_y = points[corners[3]] - origin
            local_z = points[corners[4]] - origin
            triple = float(np.dot(np.cross(local_x, local_y), local_z))
            assert triple > 0.0, (direction, triple)


def test_vertices_lie_on_the_two_radii() -> None:
    domain = sector()
    mesh = SectorMesh(domain)
    centre = np.array([domain.centre[0], domain.centre[1]])

    for point in mesh.vertices:
        radius = float(np.linalg.norm(np.asarray(point[:2]) - centre))
        assert radius == pytest.approx(domain.r_min, rel=1e-9) or radius == pytest.approx(
            domain.r_max, rel=1e-9
        )


def test_arc_midpoints_sit_on_the_circle_not_the_chord() -> None:
    """Without arc edges blockMesh would draw a polygon, not a sector."""
    domain = sector()
    mesh = SectorMesh(domain)
    centre = np.array([domain.centre[0], domain.centre[1]])

    for _, _, mid in mesh.arcs:
        radius = float(np.linalg.norm(np.asarray(mid[:2]) - centre))
        assert radius == pytest.approx(domain.r_min, rel=1e-9) or radius == pytest.approx(
            domain.r_max, rel=1e-9
        )


def test_boundary_covers_every_outer_face_exactly_once() -> None:
    domain = sector()
    mesh = SectorMesh(domain)
    faces = [f for _, _, group in mesh.boundary() for f in group]

    # Two end caps, plus four outer faces per block: both radial walls, the
    # roof and the road.
    assert len(faces) == 2 + 4 * len(mesh.hexes)
    assert len({frozenset(f) for f in faces}) == len(faces)


def test_cell_count_matches_the_blocks() -> None:
    domain = sector()
    mesh = SectorMesh(domain)
    from_blocks = sum(nx * ny * nz for _, (nx, ny, nz) in mesh.hexes)
    assert from_blocks == domain.cell_count


# --- boundary conditions and the rotating frame ---------------------------


def _wheel(offset):
    import trimesh

    transform = trimesh.transformations.translation_matrix(
        offset
    ) @ trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0])
    return measure_wheel(
        wheel="FL",
        axis_mesh=trimesh.creation.cylinder(radius=0.027, height=0.022, transform=transform),
        axis_source="MRF_FL",
        radius_mesh=trimesh.creation.annulus(
            r_min=0.005, r_max=0.033, height=0.027, transform=transform
        ),
        radius_source="Tire_FL",
    )


def _bcs(resolved, domain, wheels):
    frame = corner_frame(resolved, domain)
    speeds = wheel_speeds(resolved, wheels, frame)
    return build_bcs(resolved, 0.1, 10.0, 0.01, {"k": "kqRWallFunction",
                     "omega": "omegaWallFunction", "nut": "nutkWallFunction"},
                     speeds, frame), speeds, frame


def test_mrf_zone_gets_no_boundary_condition() -> None:
    """It is a closed volume; naming it as a patch aborts the solver."""
    resolved = spec()
    bcs, _, _ = _bcs(resolved, sector(), {"FL": _wheel([0.15, -0.09, 0.033])})
    assert "MRF_FL" not in [bc.patch for bc in bcs["U"]]
    assert "Tire_FL" in [bc.patch for bc in bcs["U"]]


def test_cornering_ground_is_stationary_in_the_ground_frame() -> None:
    resolved = spec()
    bcs, _, _ = _bcs(resolved, sector(), {})
    ground = next(bc for bc in bcs["U"] if bc.patch == "ground")
    assert ground.entries["type"] == "noSlip"


def test_cornering_inlet_prescribes_still_air_not_a_freestream() -> None:
    """The onset flow comes out of the frame rotation, not out of the inlet."""
    resolved = spec()
    bcs, _, _ = _bcs(resolved, sector(), {})
    inlet = next(bc for bc in bcs["U"] if bc.patch == "inlet")
    assert inlet.entries["value"] == "uniform (0.0 0.0 0.0)"


def test_cornering_tyre_composes_both_rotations() -> None:
    """Spin plus carry-round. Omitting the second leaves the wheel behind."""
    resolved = spec()
    domain = sector()
    bcs, _, frame = _bcs(resolved, domain, {"FL": _wheel([0.15, -0.09, 0.033])})

    tyre = next(bc for bc in bcs["U"] if bc.patch == "Tire_FL")
    assert tyre.entries["type"] == "codedFixedValue"
    assert "wheelOmega" in tyre.entries["code"]
    assert "frameOmega" in tyre.entries["code"]
    assert f"{frame.omega:.10g}" in tyre.entries["code"].replace("-", "")


def test_straight_tyre_uses_a_single_rotating_wall() -> None:
    case = _case(mode="straight", corner_radius=None)
    case["domain"]["kind"] = "box"
    case["ground"]["motion"] = "moving"
    resolved = CaseSpec.model_validate(case)

    from simdev.domain.box import BoxDomainBuilder

    domain = BoxDomainBuilder().build(resolved, GEOM_BOUNDS)
    bcs, speeds, _ = _bcs(resolved, domain, {"FL": _wheel([0.15, -0.09, 0.033])})

    tyre = next(bc for bc in bcs["U"] if bc.patch == "Tire_FL")
    assert tyre.entries["type"] == "rotatingWallVelocity"
    assert speeds["FL"]["surface_speed"] == pytest.approx(12.0, rel=1e-6)


def test_locked_wheels_fall_back_to_a_plain_wall() -> None:
    resolved = spec(wheel_rotation="locked")
    bcs, _, _ = _bcs(resolved, sector(), {"FL": _wheel([0.15, -0.09, 0.033])})
    tyre = next(bc for bc in bcs["U"] if bc.patch == "Tire_FL")
    assert tyre.entries["type"] == "noSlip"


# --- MRF zones ------------------------------------------------------------


def test_cornering_turns_the_background_at_the_corner_rate_and_wheels_at_their_own() -> None:
    """No cell may be left inertial inside a car going round a corner.

    snappyHexMesh moves the wheel-sleeve cells out of the background zone
    when it creates their own, so covering only 'all' would leave those cells
    with no frame rotation at all - closed by giving the wheel zone its own
    rotation instead, rather than the background's (2026-08-21): the wheel's
    own spin is a far better approximation of what that cell is actually
    doing than the corner rate is.
    """
    resolved = spec()
    domain = sector()
    frame = corner_frame(resolved, domain)
    wheel = _wheel([0.15, -0.09, 0.033])
    speeds = wheel_speeds(resolved, {"FL": wheel}, frame)

    zones = mrf_zones(resolved, domain, speeds, frame)

    assert [z["cell_zone"] for z in zones] == ["all", "MRF_FL"]
    background, wheel_zone = zones

    assert background["axis"] == (0.0, 0.0, 1.0)
    assert background["origin"] == frame.origin
    assert background["omega"] == pytest.approx(frame.omega)

    assert wheel_zone["origin"] == wheel.origin
    assert wheel_zone["axis"] == wheel.axis
    assert wheel_zone["omega"] == pytest.approx(speeds["FL"]["omega"])
    assert wheel_zone["omega"] != pytest.approx(frame.omega)


def test_the_road_and_the_tyre_are_excluded_from_the_rotating_frame() -> None:
    """Both carry absolute velocities; the car body does not.

    Asserted on EVERY cornering zone, not just the corner frame. A tyre face
    is an included face of whichever zone owns the cell behind it, and
    MRFZone::correctBoundaryVelocity overwrites included faces with the
    zone's own Omega ^ (Cf - origin) every SIMPLE iteration. Leave the wheel
    sleeves off this list and the tyre faces that border a sleeve cell lose
    _tyre_bc's corner-carry term while the rest of the same patch keeps it -
    a step of |omega_corner x r| ~ 15 m/s across a cellZone boundary.
    """
    resolved = spec()
    domain = sector()
    frame = corner_frame(resolved, domain)
    speeds = wheel_speeds(resolved, {"FL": _wheel([0.15, -0.09, 0.033])}, frame)

    zones = mrf_zones(resolved, domain, speeds, frame)
    assert len(zones) > 1, "expected the wheel sleeves alongside the corner frame"

    for zone in zones:
        excluded = zone["non_rotating_patches"]
        assert "ground" in excluded, zone["cell_zone"]
        assert "Tire_FL" in excluded, zone["cell_zone"]
        assert "Body" not in excluded, zone["cell_zone"]


def test_straight_line_gets_one_zone_per_wheel() -> None:
    case = _case(mode="straight", corner_radius=None)
    case["domain"]["kind"] = "box"
    case["ground"]["motion"] = "moving"
    resolved = CaseSpec.model_validate(case)

    from simdev.domain.box import BoxDomainBuilder

    domain = BoxDomainBuilder().build(resolved, GEOM_BOUNDS)
    wheels = {"FL": _wheel([0.15, -0.09, 0.033])}
    speeds = wheel_speeds(resolved, wheels, None)

    zones = mrf_zones(resolved, domain, speeds, None)

    assert [z["cell_zone"] for z in zones] == ["MRF_FL"]
    assert zones[0]["non_rotating_patches"] == []


# --- cornering wheel speeds differ per corner -----------------------------


def test_outer_wheels_turn_faster_than_inner_ones() -> None:
    """The reason a single wheel speed cannot be configured."""
    resolved = spec(corner_direction="left")
    domain = sector(corner_direction="left")
    frame = corner_frame(resolved, domain)

    # Left corner: the centre is at -y, so +y is the outside of the turn.
    inner = _wheel([0.15, -0.09, 0.033])
    outer = _wheel([0.15, 0.09, 0.033])
    speeds = wheel_speeds(resolved, {"FL": inner, "FR": outer}, frame)

    assert speeds["FR"]["surface_speed"] > speeds["FL"]["surface_speed"]
    ratio = speeds["FR"]["surface_speed"] / speeds["FL"]["surface_speed"]
    assert ratio == pytest.approx((3.0 + 0.09) / (3.0 - 0.09), rel=1e-3)


# --- validation -----------------------------------------------------------


def test_annulus_without_cornering_is_rejected() -> None:
    case = _case(mode="straight", corner_radius=None)
    with pytest.raises(ValidationError, match="curved domain"):
        validate(CaseSpec.model_validate(case))


def test_cornering_in_a_box_is_rejected() -> None:
    case = _case()
    case["domain"]["kind"] = "box"
    with pytest.raises(ValidationError, match="requires the 'annulus' domain"):
        validate(CaseSpec.model_validate(case))


def test_cornering_warns_that_the_corner_rate_is_dropped_from_wheel_zones() -> None:
    warnings = validate(spec())
    assert any("own spin rate" in w for w in warnings)


def test_a_wheel_id_on_a_non_rotating_role_is_rejected() -> None:
    case = _case()
    case["geometry"]["patches"].append(
        {"name": "inlet2", "role": "farfield", "wheel": "FL"}
    )
    with pytest.raises(ValidationError, match="nothing about it can rotate"):
        validate(CaseSpec.model_validate(case))


# --- a reversed tunnel ----------------------------------------------------
#
# The CAD is never rotated to suit the pipeline, so a car built nose-forward
# along +x is met by air coming from +x. Every direction in the case has to
# follow that one setting together; any one left behind is a sign error that
# still converges.


def _reversed(**physics) -> CaseSpec:
    case = _case(**physics)
    case["flow"]["direction"] = "-x"
    return CaseSpec.model_validate(case)


def test_reversing_the_tunnel_reverses_the_freestream() -> None:
    assert spec().freestream[0] > 0.0
    assert _reversed().freestream[0] < 0.0
    assert abs(_reversed().freestream[0]) == pytest.approx(spec().freestream[0])


def test_drag_and_side_axes_follow_the_tunnel() -> None:
    """Leaving dragDir at +x with the tunnel reversed reports a negative Cd."""
    assert spec().drag_dir == (1.0, 0.0, 0.0)
    assert _reversed().drag_dir == (-1.0, 0.0, 0.0)
    # side = lift x drag, so it flips too, or every pitching moment changes sign.
    assert _reversed().pitch_axis == (0.0, -1.0, 0.0)


def test_the_corner_centre_moves_to_the_other_side() -> None:
    """The car's left is the other way round when it travels the other way."""
    assert spec(corner_direction="left").corner_side == pytest.approx(-1.0)
    assert _reversed(corner_direction="left").corner_side == pytest.approx(1.0)


def test_the_frame_still_reproduces_the_freestream_at_the_car() -> None:
    """The invariant that ties omega, the sector and the flow together.

    Whichever way the tunnel points, still air seen from the rotating frame
    must arrive at the vehicle as that case's own freestream.
    """
    for direction in ("left", "right"):
        resolved = _reversed(corner_direction=direction)
        domain = AnnulusDomainBuilder().build(resolved, GEOM_BOUNDS)
        frame = corner_frame(resolved, domain)

        car = domain.point(domain.theta_car, domain.radius, 0.0)
        onset = frame.road_velocity_at(car)

        assert onset[0] == pytest.approx(resolved.freestream[0], rel=1e-9), direction
        assert onset[1] == pytest.approx(0.0, abs=1e-9), direction


def test_the_inlet_is_still_upstream_of_the_car() -> None:
    """With the flow along -x, upstream means the +x side."""
    for direction in ("left", "right"):
        resolved = _reversed(corner_direction=direction)
        domain = AnnulusDomainBuilder().build(resolved, GEOM_BOUNDS)
        car = np.array(domain.point(domain.theta_car, domain.radius, 0.0))
        inlet = np.array(domain.point(domain.theta_inlet, domain.radius, 0.0))
        assert inlet[0] > car[0], direction


def test_a_reversed_box_puts_the_long_tail_downstream() -> None:
    """Otherwise the wake runs straight out of the inlet."""
    from simdev.domain.box import BoxDomainBuilder

    case = _case(mode="straight", corner_radius=None)
    case["domain"]["kind"] = "box"
    case["flow"]["direction"] = "-x"
    resolved = CaseSpec.model_validate(case)

    domain = BoxDomainBuilder().build(resolved, GEOM_BOUNDS)
    # Flow runs -x, so the car's nose is its +x end and the wake trails to -x.
    nose, tail = GEOM_BOUNDS[1][0], GEOM_BOUNDS[0][0]
    upstream = domain.x_max - nose
    downstream = tail - domain.x_min

    assert downstream > upstream
    assert downstream / upstream == pytest.approx(
        resolved.domain.downstream_lengths / resolved.domain.upstream_lengths
    )
    assert domain.x_max_patch == "inlet"
    assert domain.x_min_patch == "outlet"
    assert domain.inlet_x == domain.x_max


def test_a_reversed_box_seeds_snappy_near_its_inlet() -> None:
    from simdev.domain.box import BoxDomainBuilder
    from simdev.render.context import location_in_mesh

    case = _case(mode="straight", corner_radius=None)
    case["domain"]["kind"] = "box"
    case["flow"]["direction"] = "-x"
    resolved = CaseSpec.model_validate(case)

    domain = BoxDomainBuilder().build(resolved, GEOM_BOUNDS)
    seed = location_in_mesh(domain)

    assert domain.x_min < seed[0] < domain.x_max
    assert seed[0] > GEOM_BOUNDS[1][0]


def test_a_reversed_straight_case_runs_the_road_the_other_way() -> None:
    """The contact patch must still match the road it stands on."""
    from simdev.domain.box import BoxDomainBuilder

    case = _case(mode="straight", corner_radius=None)
    case["domain"]["kind"] = "box"
    case["ground"]["motion"] = "moving"
    case["flow"]["direction"] = "-x"
    resolved = CaseSpec.model_validate(case)

    domain = BoxDomainBuilder().build(resolved, GEOM_BOUNDS)
    wheel = _wheel([0.15, -0.09, 0.033])
    speeds = wheel_speeds(resolved, {"FL": wheel}, None)

    velocity = wheel.surface_velocity(
        wheel.contact_point()[None, :], speeds["FL"]["omega"]
    )[0]

    assert velocity[0] == pytest.approx(-12.0, rel=1e-6)
    assert speeds["FL"]["slip"] == pytest.approx(0.0, abs=1e-9)
