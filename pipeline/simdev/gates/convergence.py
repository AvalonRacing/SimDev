from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from simdev.config.schema import CaseSpec

COEFFICIENTS = ("Cd", "Cl")


@dataclass(frozen=True)
class ConvergenceResult:
    converged: bool
    reasons: list[str] = field(default_factory=list)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    # Relative oscillation, std/|mean| over the trailing window. Reported
    # whether or not it passes: on a limit-cycle case this is a property of
    # the flow that the engineer needs, not a defect of the solve.
    amplitudes: dict[str, float] = field(default_factory=dict)
    window: tuple[int, int] = (0, 0)
    n_iterations: int = 0


def check_convergence(df: pd.DataFrame, spec: CaseSpec) -> ConvergenceResult:
    """Force-plateau test over the trailing window.

    Two independent questions, because on this class of case they have
    different answers and only one of them is about convergence.

    **Has the mean stopped moving?** The convergence question. Answered by
    comparing the mean of the trailing window against the mean of the window
    before it, against `solve.drift_tol`.

    **How far does it swing about that mean?** A property of the flow.
    Answered by std/|mean| over the trailing window, against
    `solve.amplitude_tol`, which exists to catch a solve going unstable
    rather than to grade the physics.

    Splitting them is not a loosening. Steady RANS on a massively separated
    cornering open-wheel car has no fixed point to find: it reaches a
    stationary mean and then oscillates about it forever. The measured car
    case (docs/linux-migration.md) holds its rolling 200-iteration mean to
    +/-0.35 % in Cd and +/-0.9 % in Cl while swinging +/-2.2 % and +/-7.9 %
    about it. One tolerance covering both had to be set loose enough for the
    swing, which then made it far too loose to notice a drifting mean - the
    single failure this gate exists to catch.

    Nor is a least-squares slope inside one window a substitute. A sine
    sampled over part of a period genuinely has a slope, so that test reads a
    perfectly stationary limit cycle as drift, at a magnitude set by where in
    the cycle the run happened to stop - 10.9 % for the Cl case above. Two
    consecutive window means each average the cycle away instead, which is
    why the window wants to be at least one oscillation period long.

    Always reports the window mean, converged or not, so a non-converged run
    produces a recorded-but-flagged result rather than nothing.
    """
    n = len(df)
    window = spec.solve.plateau_window
    present = [c for c in COEFFICIENTS if c in df]

    if n < window:
        return ConvergenceResult(
            converged=False,
            reasons=[
                f"only {n} iterations recorded, need at least {window} to judge "
                "a plateau"
            ],
            means={c: float(df[c].mean()) for c in present},
            stds={c: float(df[c].std()) for c in present},
            amplitudes={},
            window=(0, n),
            n_iterations=n,
        )

    tail = df.iloc[-window:]
    # None when the run is too short to hold two whole windows; the mean is
    # still reported, it just cannot be compared against anything yet.
    previous = df.iloc[-2 * window : -window] if n >= 2 * window else None

    # Split deliberately. `failures` decides the verdict; `notes` is measured
    # detail that is reported either way. A disabled tolerance has to produce
    # a note rather than a failure, or turning the gate off would flip every
    # run to 'converged' - which is a claim nobody made.
    failures: list[str] = []
    notes: list[str] = []
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    amplitudes: dict[str, float] = {}

    drift_tol = spec.solve.drift_tol
    amplitude_tol = spec.solve.amplitude_tol

    for coefficient in present:
        values = tail[coefficient].to_numpy()
        mean = float(values.mean())
        means[coefficient] = mean
        stds[coefficient] = float(values.std())

        scale = max(abs(mean), 1e-9)
        amplitudes[coefficient] = stds[coefficient] / scale

        if amplitude_tol is not None and amplitudes[coefficient] > amplitude_tol:
            failures.append(
                f"{coefficient} oscillates by {amplitudes[coefficient]:.2%} of "
                f"its mean over the last {window} iterations, beyond the "
                f"{amplitude_tol:.2%} bound; that is wider than a "
                "limit cycle and suggests the solve is not stable"
            )

        if previous is None:
            continue

        drift = (mean - float(previous[coefficient].to_numpy().mean())) / scale
        if drift_tol is None:
            notes.append(
                f"{coefficient} mean moved {drift:+.2%} between the last two "
                f"{window}-iteration windows (not judged: drift_tol is unset)"
            )
        elif abs(drift) > drift_tol:
            failures.append(
                f"{coefficient} is still drifting: its mean moved {drift:+.2%} "
                f"between the last two {window}-iteration windows, beyond the "
                f"{drift_tol:.2%} bound"
            )

    if previous is None:
        failures.append(
            f"only {n} iterations recorded, need at least {2 * window} to "
            f"compare two consecutive {window}-iteration window means and so "
            "tell a settled mean from a drifting one"
        )

    if drift_tol is None:
        notes.append(
            f"convergence was NOT judged: drift_tol is unset, so Cd and Cl are "
            f"means over the last {window} iterations at a stopping point "
            "chosen deliberately rather than reached by a plateau test. Valid "
            "for comparing runs that all stop at the same iteration; not an "
            "absolute, and not a converged value"
        )

    return ConvergenceResult(
        converged=not failures,
        reasons=failures + notes,
        means=means,
        stds=stds,
        amplitudes=amplitudes,
        window=(n - window, n),
        n_iterations=n,
    )
