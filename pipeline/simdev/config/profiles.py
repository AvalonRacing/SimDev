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
    "solve": {
        "plateau_window": 200,
        "drift_tol": 0.002,
        "amplitude_tol": 0.10,
        "residual_tol": 1.0e-6,
    },
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
    # 66 mm tyres. Refined surfaces sit at 0.75 mm - about 56 cells across the
    # wing chord and 5 across a control arm, which is the coarsest that still
    # resolves the appendages generating the downforce.
    #
    # THE BACKGROUND IS 96 mm AND THE LEVELS ARE COUNTED FROM THERE. It used
    # to be 24 mm at level 5, which is the same 0.75 mm by a shorter ladder,
    # and the reason for the change is what that cost out in the domain: the
    # first production mesh spent 1,818,230 cells - a quarter of the whole
    # mesh - on uniform 24 mm background, most of it in clean air metres from
    # the car. 96 mm is the same volume in about 28k cells.
    #
    # Re-basing is exact rather than approximate, and it has to stay that way:
    # 96 = 24 x 4, so every level moved by +2 and every absolute cell size is
    # unchanged (0.096 / 2^7 == 0.024 / 2^5 == 0.75 mm). A factor that is not
    # a power of two would move every derived size, the layer budget and
    # max_layer_cell_ratio with it. tests/test_cell_sizes.py pins the absolute
    # sizes in metres so the next re-basing cannot quietly miss a patch.
    #
    # What it costs: the ladder is now seven levels deep, and
    # nCellsBetweenLevels 3 spends buffer cells on each rung, so the far-field
    # saving is not banked in full.
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
            # 0.2 / 2^4 == 0.05 / 2^2 == 12.5 mm: the same smoke-test
            # resolution on the same x4 re-basing as the other car profiles.
            "base_cell_size": 0.2,
            "surface_refinement_min": 3,
            "surface_refinement_max": 4,
            "refinement_cap": 4,
            # A mesh this coarse cannot avoid one bad cell where a tyre meets
            # the road, and checkMesh fails it at its own hardcoded skewness
            # limit of 4 no matter what max_skewness says. Since the point of
            # this profile is to exercise the plumbing rather than to produce
            # a mesh anyone would solve on, the spec's own limits are the
            # authority here - loosened to match.
            #
            # Structural failures still gate: a mesh in two pieces stops the
            # run on this profile exactly as on any other.
            "trust_check_mesh_verdict": False,
            "max_skewness": 20.0,
            "max_non_ortho": 75.0,
        },
        "solve": {"max_iterations": 50, "n_ranks": 4, "plateau_window": 20},
    },
    "car_dev": {
        "mesh": {
            # 0.096 / 2^6 == 0.024 / 2^4 == 1.5 mm.
            "base_cell_size": 0.096,
            "surface_refinement_min": 5,
            "surface_refinement_max": 6,
        },
        "solve": {"max_iterations": 400, "n_ranks": 8, "plateau_window": 50},
    },
    "car": {
        "mesh": {
            # Re-based from 0.096 m / level 6-7 (2026-08-21), targeting 15M
            # cells. NOT a plain shrink of base_cell_size: an earlier attempt
            # (0.0687 m, same levels) hit 14.4M but silently halved the
            # finest wall cell to 0.537 mm, invalidating the layer stack
            # below (tuned for exactly 0.75 mm) without retuning it - see
            # tests/test_cell_sizes.py's docstring on why that is the wrong
            # move. A correct re-basing halves base_cell_size and drops
            # every level by 1 in lockstep (here and in every per-patch
            # override in cases/car/config.yaml, plus every shell/wake
            # level), which leaves every absolute cell size - 0.75 mm walls,
            # 1.5 mm MRF sleeves - exactly where it was. The added cells
            # come entirely from a finer background/mid-field, which is
            # also where the case's own history says the mesh was thin
            # (see the "first production mesh" note below).
            "base_cell_size": 0.048,
            "surface_refinement_min": 5,
            "surface_refinement_max": 6,
        },
        # 2000 iterations. At the measured 14.78 s/iter this is ~8.6 h
        # wall clock, not the 5 h that was hoped for - see the handbook.
        #
        # WHY 2000 AND NOT 5000. The first production solve reached a
        # stationary mean well before iteration 3000 and then oscillated about
        # it - a limit cycle, which is the honest answer for steady RANS on a
        # massively separated cornering open-wheel car. Iterations past that
        # buy a different point on the same cycle, not a better answer. Read
        # the coefficients as time-averages over plateau_window.
        #
        # CUT TO 750 (2026-08-21): the native-Linux 6.93M-cell run showed the
        # windowed mean fully settled by iteration ~200-400 and unchanged
        # through iteration 2000 - only the oscillation kept sampling, not a
        # different answer. 750 gives ~2x margin over that while convergence
        # work is ongoing and mesh size just went up; revisit once the
        # numerics/scheme changes below are validated.
        #
        # THE PRESSURE SETTINGS STAY AT THEIR DEFAULTS, AND THAT IS A MEASURED
        # RESULT, NOT AN OVERSIGHT. Do not "optimise" them again without
        # reading docs/handbook.md, "Make the solve faster" - all of this was
        # tried on this exact mesh and the numbers are recorded there.
        #
        # n_non_orth_correctors = 0 DIVERGES on this mesh. It is the obvious
        # saving - the corrector solve costs more than the solve it corrects -
        # and it kills the run: SIGFPE at iteration 29 with GAMG's default
        # coarsest level and at iteration 39 with 1000. Local continuity error
        # at iteration 20 goes from 9.5e-4 with the corrector to 1.3e-2
        # without. 71.7 degrees of non-orthogonality over 68 severely
        # non-orthogonal faces is more than `limited corrected 0.33` can
        # absorb.
        #
        # gamg_coarsest_cells = 1000 measured 14.77 s/iter against 14.78 at
        # the default of 10 - no effect. The mechanism is real (agglomeration
        # stops at nProcs x this value, so the default builds coarse levels
        # down to 400 cells total) but the reductions it saves are not what
        # this case is waiting on.
        # THE TOLERANCES ARE SPLIT BECAUSE THIS CASE HAS A LIMIT CYCLE, and
        # both numbers below are measured on it rather than chosen. The
        # 6.93M-cell run held its rolling 200-iteration mean to +/-0.35 % in
        # Cd and +/-0.9 % in Cl while swinging +/-2.2 % and +/-7.9 % about it
        # (docs/linux-migration.md). Those are answers to two different
        # questions and they differ by an order of magnitude, which is why a
        # single plateau_tol could not be set to anything useful: at 0.002 it
        # failed every run on the swing, and loose enough to pass the swing it
        # would no longer notice a mean that was still moving.
        #
        # drift_tol 0.035 is MEASURED, and it is loose because plateau_window
        # is shorter than the oscillation. Sweeping every stopping point of
        # the 2000-iteration 6.93M run, over the stationary part only (both
        # windows past iteration 500), the drift between consecutive windows
        # comes out:
        #
        #   window 200   Cd median 0.80 % worst 1.91 %   Cl 1.34 % / 3.00 %
        #   window 400   Cd median 0.64 % worst 1.33 %   Cl 1.07 % / 2.15 %
        #
        # A periodogram of that history says why: Cd's limit cycle runs at a
        # period of 500-750 iterations (autocorrelation zero-crossing implies
        # ~628), while Cl's is ~40. A 200-iteration window averages Cl's
        # cycle away and does not touch Cd's, so what is left in the Cd drift
        # is the cycle itself rather than any movement of the mean.
        #
        # 0.035 clears the worst case above with a little margin. THE BETTER
        # FIX IS A LONGER WINDOW, NOT A LOOSER BOUND: plateau_window 400 with
        # max_iterations ~1500 (500 transient + two 400-windows + margin)
        # would let drift_tol come back to ~0.025, and 750/2500 would make it
        # tighter still. Both cost wall clock that has not been budgeted, so
        # this stays measured-and-documented rather than quietly optimistic.
        #
        # amplitude_tol is left at the 0.10 default, which clears the measured
        # 2.4-3.6 % Cl standard deviation with margin while still catching a
        # solve that comes apart. Tighten drift_tol, never amplitude_tol, if
        # you want a stricter run - only the first one is about convergence.
        # 250 ITERATIONS, AND CONVERGENCE IS NOT JUDGED. Both halves of that
        # are deliberate. Read this before restoring either.
        #
        # THIS PROFILE NO LONGER ASKS "HAS IT CONVERGED". It takes a
        # fixed-cost sample: run 250 iterations, average Cd and Cl over the
        # last 50, report them. drift_tol is None, which switches the drift
        # test from a verdict to a recorded number (see gates/convergence.py -
        # it still measures and prints the drift, it just stops failing on
        # it). amplitude_tol stays on, because it catches a solve coming apart
        # and that is worth knowing however the stopping point was picked.
        #
        # WHAT THIS BUYS: ~45 min a run against ~2.9 h at 500, which is what
        # makes a mesh sweep affordable.
        #
        # WHAT IT COSTS, MEASURED, so nobody has to rediscover it. Cl is still
        # moving at iteration 250 on every mesh tried, always toward more
        # downforce, and the finer the mesh the more it moves:
        #
        #   mesh      Cl at 250    Cl at 500    move
        #    6.93M      -1.5959      -1.6787    -5.2 %
        #   13.28M      -0.9782      -1.0600    -8.4 %
        #   17.80M      -1.0150      -1.1453   -12.8 %
        #
        # So a Cl from this profile UNDERSTATES DOWNFORCE, by roughly a tenth
        # on a fine mesh, and understates it more as the mesh grows.
        #
        # THE "EVERY RUN CARRIES THE SAME BIAS SO THE DELTA SURVIVES" DEFENCE
        # THAT USED TO SIT HERE IS FALSE, and the 17.80M/22.93M pair killed
        # it. Those two meshes differ only in where the fine cells sit, and
        # the Cl difference between them CHANGES SIGN with the stopping point:
        #
        #   read at iteration 250:   -7.1 %
        #   read at iteration 500:  +13.7 %
        #
        # The bias is not common-mode, because each mesh has its own
        # oscillation phase and its own transient length.
        #
        # WHAT ACTUALLY LIMITS THIS, measured by sweeping every stopping point
        # past iteration 200 and taking the windowed mean at each: a Cl read
        # this way moves 12.7-27.9 % depending only on where the run stopped,
        # and widening the window to 150 still leaves 8-17 %. The oscillation
        # is slow relative to the run, so averaging inside one run cannot
        # remove it. THE ERROR BAR ON ANY Cl FROM THIS PROFILE IS ROUGHLY
        # +/-15 %, and any mesh or design effect smaller than that is not
        # resolvable here - including the ~10 % allocation effect the pair
        # above was built to measure.
        #
        # Cd is far better behaved: the same sweep gives 3.6-5.1 %.
        #
        # So: use this profile to screen for LARGE effects and to keep the
        # pipeline exercised. Do not set a number from it against the StarCCM+
        # benchmark, do not compare it with the 500- and 750-iteration results
        # already in the run history, and do not read a small delta off two
        # runs at this length and believe it.
        #
        # FOR A TRUSTWORTHY ABSOLUTE Cl, raise max_iterations on that one run
        # and put drift_tol back. Do not widen a bound to make a gate green.
        #
        # THE GENERAL LESSON, which cost a night to learn: the transient gets
        # longer as the mesh gets finer, so an iteration count calibrated on
        # one mesh is not evidence about a finer one. Cd is far less affected
        # than Cl at every stop measured.
        #
        # WRITE_INTERVAL MUST DIVIDE MAX_ITERATIONS. It did not: 500 into 750
        # wrote fields at iteration 500 and then never again, because
        # simpleFoam does not force a write at endTime unless residualControl
        # stops it first. The 13.28M run therefore finished cleanly with its
        # newest field data 250 iterations stale, which is invisible until
        # someone opens the case in ParaView and reads a coefficient off the
        # wrong time. Coefficients were unaffected - the function objects
        # write every timestep to postProcessing/ - so nothing in the reported
        # numbers was wrong, only the fields you can look at.
        #
        # 125 divides 250, so this writes at 125 and at 250. purgeWrite 2 keeps
        # both. If either number changes, keep the division exact.
        "solve": {
            "max_iterations": 250,
            "n_ranks": 40,
            # The averaging window, and at 250 iterations it is also the whole
            # of what this profile reports: Cd and Cl are the mean over the
            # last 50. Two windows still fit (100 of 250), so the drift is
            # measurable and gets reported - it just no longer gates.
            "plateau_window": 50,
            # None, not a large number. See the block above and the field
            # comment in schema.SolveConfig: a huge tolerance would record
            # 'converged' for a run nobody judged; None records the truth.
            "drift_tol": None,
            # Never write only at the end, and never on an interval that does
            # not divide max_iterations - see above.
            "write_interval": 125,
        },
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
