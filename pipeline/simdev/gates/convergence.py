from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simdev.config.schema import CaseSpec

COEFFICIENTS = ("Cd", "Cl")


@dataclass(frozen=True)
class ConvergenceResult:
    converged: bool
    reasons: list[str] = field(default_factory=list)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    window: tuple[int, int] = (0, 0)
    n_iterations: int = 0


def check_convergence(df: pd.DataFrame, spec: CaseSpec) -> ConvergenceResult:
    """Force-plateau test over the trailing window.

    Always reports the window mean, converged or not, so a non-converged run
    produces a recorded-but-flagged result rather than nothing.
    """
    n = len(df)
    window = spec.solve.plateau_window
    reasons: list[str] = []

    if n < window:
        return ConvergenceResult(
            converged=False,
            reasons=[
                f"only {n} iterations recorded, need at least {window} to judge "
                "a plateau"
            ],
            means={c: float(df[c].mean()) for c in COEFFICIENTS if c in df},
            stds={c: float(df[c].std()) for c in COEFFICIENTS if c in df},
            window=(0, n),
            n_iterations=n,
        )

    tail = df.iloc[-window:]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for coefficient in COEFFICIENTS:
        if coefficient not in df:
            continue
        values = tail[coefficient].to_numpy()
        mean = float(values.mean())
        means[coefficient] = mean
        stds[coefficient] = float(values.std())

        scale = max(abs(mean), 1e-9)

        if stds[coefficient] / scale > spec.solve.plateau_tol:
            reasons.append(
                f"{coefficient} has not reached a plateau: relative scatter "
                f"{stds[coefficient] / scale:.2%} over the last {window} "
                f"iterations exceeds {spec.solve.plateau_tol:.2%}"
            )

        # Least-squares slope across the window, expressed per window.
        iterations = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(iterations, values, 1)[0]) * len(values)
        if abs(slope) / scale > spec.solve.plateau_tol:
            reasons.append(
                f"{coefficient} is still drifting: {slope / scale:+.2%} across "
                f"the last {window} iterations"
            )

    return ConvergenceResult(
        converged=not reasons,
        reasons=reasons,
        means=means,
        stds=stds,
        window=(n - window, n),
        n_iterations=n,
    )
