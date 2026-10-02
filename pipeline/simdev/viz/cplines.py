"""cp over x on Body and Wing, one plot per car-y station.

The renderer (viz/pv_render.py) cuts the parts and writes each station's
points to CSV; this draws them. Split that way because the cut needs VTK,
which only the ParaView interpreter has, while the plot is ordinary
matplotlib and can be tested here without either.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One colour per part, fixed, so Body is the same colour on every plot.
# Beyond these two the matplotlib cycle takes over.
PATCH_COLOURS = {"Body": "#1f5fbf", "Wing": "#d0452b"}


def read_station(
    path: Path,
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """patch -> (x, z, pMean) for every point the cut produced."""
    points: dict[str, tuple[list[float], list[float], list[float]]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            xs, zs, ps = points.setdefault(row["patch"], ([], [], []))
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
            ps.append(float(row["pMean"]))
    return points


def draw_cp_lines(spec: Mapping[str, Any] | None, u_inf: float) -> int:
    """Draw every station whose CSV exists. Returns how many were drawn.

    cp = pMean / (0.5 u_inf^2): the dynamic head the force coefficients and
    the cp pictures already use, so a curve and a colour read on one scale.

    POINTS, NOT LINES. A cut through a closed part returns upper and lower
    surface in no particular order, and joining them in file order draws
    chords straight across the section. Small markers show both surfaces
    honestly without having to sort out which is which.
    """
    if not spec:
        return 0
    q = 0.5 * u_inf * u_inf
    drawn = 0
    for station in spec["stations"]:
        source = Path(station["csv"])
        if not source.exists():
            continue
        points = read_station(source)

        # cp on top; underneath, the section the curve was taken from, on
        # the same x axis, so a peak can be read straight down onto the
        # part of the car that makes it.
        z_low, z_high = spec.get("z_limits", (0.0, 0.14))
        x_low, x_high = spec["x_limits"]
        fig, (axis, section) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 7.2),
            gridspec_kw={"height_ratios": [3, 2]},
        )
        for patch in spec["patches"]:
            if patch not in points:
                continue
            xs, zs, ps = points[patch]
            colour = PATCH_COLOURS.get(patch)
            axis.scatter(
                xs, [p / q for p in ps], s=3, linewidths=0,
                color=colour, label=patch,
            )
            section.scatter(xs, zs, s=1.5, linewidths=0, color=colour)
        # Nose on the left, matching the y slices; suction (negative cp)
        # upward, the convention for pressure distributions.
        axis.set_xlim(x_high, x_low)
        cp_low, cp_high = spec["cp_limits"]
        axis.set_ylim(cp_high, cp_low)
        axis.axhline(0.0, color="#999", linewidth=0.6)
        axis.set_ylabel("cp")
        axis.set_title(f"cp over x at y = {station['offset']:+.3f} m")
        axis.grid(True, linewidth=0.3, alpha=0.6)
        # TRUE TO SCALE. A section squashed to fill the panel puts the
        # wing's camber and the roof's curvature at shapes they do not have.
        #
        # Fixed by the BOX's proportions, not by set_aspect: an equal-aspect
        # panel too short for its slot gives up width instead, and then its
        # x positions no longer sit under the cp curve above. The slot is
        # sized tall enough that only height is given up.
        section.set_ylim(z_low, z_high)
        section.set_box_aspect((z_high - z_low) / (x_high - x_low))
        section.axhline(0.0, color="#555", linewidth=0.8)
        section.set_xlabel("x from datum [m]  (nose left)")
        section.set_ylabel("z [m]")
        section.grid(True, linewidth=0.3, alpha=0.6)
        if points:
            # Upper left: ahead of the nose at strong suction, the one
            # corner no curve reaches.
            axis.legend(loc="upper left", markerscale=4, frameon=False)
        fig.tight_layout()

        out = Path(station["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        drawn += 1
    return drawn
