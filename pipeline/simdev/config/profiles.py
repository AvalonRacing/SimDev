from __future__ import annotations

from typing import Any

from simdev.config.schema import WallTreatment

# Baseline applied before every other layer.
DEFAULTS: dict[str, Any] = {
    "physics": {
        "turbulence_model": "kOmegaSST",
        "wall_treatment": "high_y_plus",
        "mode": "straight",
        "yaw_deg": 0.0,
    },
    "domain": {
        "kind": "box",
        "upstream_lengths": 5.0,
        "downstream_lengths": 10.0,
        "half_width_lengths": 3.0,
        "height_lengths": 3.0,
        "max_blockage": 0.01,
    },
    # residual_tol is a safety net, deliberately far below the 1e-4 at which
    # a steady aero case "looks converged". simpleFoam *stops* when
    # residualControl is met, so setting it where residuals plateau hands the
    # stopping decision to the very criterion this pipeline exists to
    # distrust: forces routinely keep drifting long after residuals flatten.
    # At 1e-6 it only fires when the solution really has stopped moving, and
    # the force-plateau gate owns termination the rest of the time - which is
    # also what makes "raise max_iterations" a remedy that works.
    "solve": {"plateau_window": 200, "plateau_tol": 0.002, "residual_tol": 1.0e-6},
    "post": {"max_fraction_outside": 0.1},
}

RESOLUTION_PROFILES: dict[str, dict[str, Any]] = {
    # Deliberately very coarse: pipeline development, minutes not hours.
    "dev": {
        "mesh": {
            "base_cell_size": 0.12,
            "surface_refinement_min": 2,
            "surface_refinement_max": 3,
        },
        "solve": {"max_iterations": 300, "n_ranks": 8, "plateau_window": 50},
    },
    # Sized for the Ahmed validation, which runs high_y_plus.
    #
    # 0.09 / 2^3 gives 11.25 mm surface cells, and that is as fine as wall
    # functions allow at this condition. y+ 30-300 at 40 m/s pins the first
    # layer at 1 mm; four layers at ER 1.2 stack up to 5.37 mm; and the stack
    # has to fit inside half the surface cell. Refining past this does not
    # improve the mesh, it stops the layers fitting - which is exactly what
    # the old (0.03, levels 5-6) setting did, asking for a 1 mm first layer
    # inside a 0.47 mm cell.
    #
    # base_cell_size is absolute metres, so this profile does not transfer to
    # the 1/10-scale RC car: that work runs low_y_plus, where a 2e-5 first
    # layer permits far finer surface cells, and it will need its own profile.
    "production": {
        "mesh": {
            "base_cell_size": 0.09,
            "surface_refinement_min": 2,
            "surface_refinement_max": 3,
        },
        # 40 physical cores. Never 80 threads: OpenFOAM is bandwidth bound.
        "solve": {"max_iterations": 4000, "n_ranks": 40},
    },
}

WALL_PROFILES: dict[WallTreatment, dict[str, Any]] = {
    WallTreatment.HIGH_Y_PLUS: {
        # 1 mm puts the cell centre at y+ ~= 50 for the Ahmed condition
        # (40 m/s, L = 1.044 m). Cases override this for other conditions.
        "mesh": {"n_layers": 7, "first_layer_thickness": 1.0e-3, "expansion_ratio": 1.2},
        "post": {"yplus_min": 30.0, "yplus_max": 300.0},
    },
    WallTreatment.LOW_Y_PLUS: {
        "mesh": {"n_layers": 18, "first_layer_thickness": 2.0e-5, "expansion_ratio": 1.15},
        "post": {"yplus_min": 0.0, "yplus_max": 5.0},
    },
    WallTreatment.SPALDING: {
        "mesh": {"n_layers": 10, "first_layer_thickness": 1.0e-4, "expansion_ratio": 1.2},
        "post": {"yplus_min": 0.0, "yplus_max": 300.0},
    },
}
