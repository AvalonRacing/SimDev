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
    # The RC car. A different vehicle scale needs its own absolute cell size,
    # because base_cell_size is metres and every refinement level is relative
    # to it.
    #
    # The car is ~0.44 m long against the Ahmed body's 1.044 m, and its
    # features are far smaller: a 42 mm wing chord, 4 mm suspension links,
    # 66 mm tyres. 24 mm background cells with five levels of surface
    # refinement give 0.75 mm on refined surfaces - about 56 cells across the
    # wing chord and 5 across a control arm, which is the coarsest that still
    # resolves the appendages generating the downforce.
    #
    # It runs low_y_plus, so the first layer is 20 um and 18 layers stack to
    # 0.79 mm. That does *not* fit inside half a 0.75 mm cell, and validate()
    # will say so patch by patch rather than letting snappy discover it forty
    # minutes into meshing - see docs/handbook.md section 6.
    # Deliberately far too coarse to believe. This exists to answer "does the
    # pipeline work as a system", not "what is the drag": it meshes, solves and
    # posts in minutes so the plumbing can be exercised end to end.
    #
    # refinement_cap is the load-bearing setting. Per-patch levels override the
    # profile - correct for production, since a 42 mm wing chord needs its own
    # level whatever the profile says - so coarsening base_cell_size alone
    # leaves the wing and suspension pinned at level 5 and the mesh expensive
    # anyway. The cap overrides them in turn.
    #
    # Do not read a coefficient off this profile. The y+ gate will fail and it
    # is right to.
    "car_smoke": {
        "mesh": {
            "base_cell_size": 0.05,
            "surface_refinement_min": 1,
            "surface_refinement_max": 2,
            "refinement_cap": 2,
        },
        "solve": {"max_iterations": 50, "n_ranks": 4, "plateau_window": 20},
    },
    "car_dev": {
        "mesh": {
            "base_cell_size": 0.024,
            "surface_refinement_min": 3,
            "surface_refinement_max": 4,
        },
        "solve": {"max_iterations": 400, "n_ranks": 8, "plateau_window": 50},
    },
    "car": {
        "mesh": {
            "base_cell_size": 0.024,
            "surface_refinement_min": 4,
            "surface_refinement_max": 5,
        },
        "solve": {"max_iterations": 5000, "n_ranks": 40},
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
