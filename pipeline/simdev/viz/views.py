"""Parse the shared view definition.

Pure data in, pure data out. Everything here is unit-testable without a run,
a mesh or ParaView, which is the point of keeping the render plan separate
from the renderer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_VIEWS_PATH = (
    Path(__file__).resolve().parents[3] / "cases" / "post_views.yaml"
)

SLICE_FIELDS: tuple[str, ...] = ("cp", "cpt", "U", "lambda2")
SURFACE_FIELDS: tuple[str, ...] = ("cp", "yplus")
STREAMLINE_MODES: tuple[str, ...] = ("off", "lic", "seeded")

# `banded` is the old pipeline's 32-band pressure bar and the default: a
# field that names no colormap gets it. `spectrum` and `thermal` are the
# StarCCM+ ramps, named as the old pipeline's simConfig.txt named them.
# viz/pv_render.py builds `banded` itself and maps the other two onto
# ParaView presets.
#
# VALIDATED HERE, IN THE VENV, because the failure downstream is silent:
# ParaView's ApplyPreset returns without complaint on a preset name it does
# not know and leaves the default map in place. A typo would produce 562
# pictures in the wrong colours with nothing anywhere saying so.
COLORMAPS: tuple[str, ...] = ("banded", "spectrum", "thermal")
DEFAULT_COLORMAP = "banded"


@dataclass(frozen=True)
class Axis:
    start: float
    stop: float
    step: float

    def offsets(self) -> list[float]:
        """Every plane position on this axis, inclusive of both ends.

        Rounded to nanometres on the way out. Accumulating `start + i * step`
        in binary drifts, and a drifted position becomes a differently-named
        file - at which point two runs that were meant to be laid side by
        side no longer have matching file names.
        """
        if self.step <= 0.0:
            raise ValueError(f"plane step must be positive, got {self.step}")
        count = int(round((self.stop - self.start) / self.step))
        return [round(self.start + i * self.step, 9) for i in range(count + 1)]


@dataclass(frozen=True)
class FieldStyle:
    limits: tuple[float, float]
    colormap: str


@dataclass(frozen=True)
class CpLines:
    """cp-over-x section plots: which parts, at which car-y stations.

    The axis limits are fixed for the same reason every colour scale is:
    a plot that autoscales cannot be laid beside another run's.
    """
    patches: tuple[str, ...]
    stations: Axis
    cp_limits: tuple[float, float]
    x_limits: tuple[float, float]
    # Height range of the section panel under each plot, from the datum.
    z_limits: tuple[float, float]


@dataclass(frozen=True)
class Views:
    datum_patches: tuple[str, ...]
    axes: dict[str, Axis]
    fields: dict[str, FieldStyle]
    parallel_scale: dict[str, float]
    surface_scale: dict[str, float]
    # Per slice axis, x and y only - z slices look straight down and centre
    # on the datum.
    focus_height: dict[str, float]
    surface_focus_height: float
    resolution: tuple[int, int]
    streamlines: str
    # None when the views file has no cp_lines block: an older definition
    # still loads, it just draws no section plots.
    cp_lines: CpLines | None
    # sha256 of the file text, truncated. Recorded beside every picture so a
    # PNG can always be traced back to the definition that framed it.
    digest: str


# The seven overall views, and how wide each needs to be when the views file
# does not say. Expressed as a multiple of the largest slice scale: the iso
# view looks along the car's diagonal, which is longer than its length, so
# it needs the most; front and rear see only the car's width and need least.
SURFACE_SCALE_FALLBACK = {
    "iso": 1.5,
    "front": 1.0,
    "rear": 1.0,
    "left": 1.3,
    "right": 1.3,
    "top": 1.3,
    "bottom": 1.3,
}


def _surface_scale(camera: dict) -> dict[str, float]:
    """Per-view half-heights for the surface renders.

    OPTIONAL in the views file. A definition written before these existed is
    still a valid definition, and failing to load it would mean an old run
    cannot be re-rendered at all - so the fallback scales off the slice
    settings rather than raising.
    """
    declared = camera.get("surface_scale") or {}
    widest = max(float(v) for v in camera["parallel_scale"].values())
    return {
        name: float(declared.get(name, widest * factor))
        for name, factor in SURFACE_SCALE_FALLBACK.items()
    }


def _focus_height(camera: dict) -> dict[str, float]:
    """Slice focal heights for x and y.

    One number for both, or a per-axis mapping. A single number is what
    every views file written before the x zoom says, and it must still load
    and frame the way it used to.
    """
    declared = camera.get("focus_height", 0.0)
    if isinstance(declared, dict):
        return {axis: float(declared.get(axis, 0.0)) for axis in ("x", "y")}
    return {"x": float(declared), "y": float(declared)}


def _cp_lines(raw: dict | None) -> CpLines | None:
    if not raw:
        return None
    y = raw["y"]
    return CpLines(
        patches=tuple(str(p) for p in raw["patches"]),
        stations=Axis(float(y["from"]), float(y["to"]), float(y["step"])),
        cp_limits=(float(raw["cp_limits"][0]), float(raw["cp_limits"][1])),
        x_limits=(float(raw["x_limits"][0]), float(raw["x_limits"][1])),
        z_limits=(
            float(raw.get("z_limits", (0.0, 0.14))[0]),
            float(raw.get("z_limits", (0.0, 0.14))[1]),
        ),
    )


def load_views(path: Path) -> Views:
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)

    value = raw.get("streamlines", "off")
    # YAML 1.1 reads a bare `off` as the boolean False, so the natural
    # spelling `streamlines: off` never arrives as the string "off".
    # Accept it rather than rejecting the reader's own spelling. `true` is
    # deliberately NOT accepted: it names no mode.
    if value is False:
        value = "off"
    streamlines = str(value)
    if streamlines not in STREAMLINE_MODES:
        raise ValueError(
            f"streamlines must be one of {STREAMLINE_MODES}, got "
            f"{streamlines!r}"
        )

    axes = {
        name: Axis(float(v["from"]), float(v["to"]), float(v["step"]))
        for name, v in raw["planes"].items()
    }
    fields = {
        name: FieldStyle(
            (float(v["limits"][0]), float(v["limits"][1])),
            str(v.get("colormap", DEFAULT_COLORMAP)),
        )
        for name, v in raw["fields"].items()
    }

    unknown = sorted({s.colormap for s in fields.values()} - set(COLORMAPS))
    if unknown:
        raise ValueError(
            f"{path}: unknown colormap {', '.join(unknown)}. Known: "
            f"{', '.join(COLORMAPS)}. ParaView ignores a preset name it does "
            "not recognise without raising, so this is checked here rather "
            "than discovered in the pictures"
        )

    missing = [f for f in (*SLICE_FIELDS, *SURFACE_FIELDS) if f not in fields]
    if missing:
        raise ValueError(
            f"{path}: no limits for {', '.join(missing)}. A field with no "
            "limits would autoscale per run, which is the one thing this "
            "file exists to prevent"
        )

    camera = raw["camera"]
    return Views(
        datum_patches=tuple(raw["datum"]["patches"]),
        axes=axes,
        fields=fields,
        parallel_scale={k: float(v) for k, v in camera["parallel_scale"].items()},
        surface_scale=_surface_scale(camera),
        focus_height=_focus_height(camera),
        # Defaults to the slice height when unset, so an older views
        # file still loads and still frames the way it used to.
        surface_focus_height=float(
            camera.get("surface_focus_height", _focus_height(camera)["y"])
        ),
        resolution=(int(camera["resolution"][0]), int(camera["resolution"][1])),
        streamlines=streamlines,
        cp_lines=_cp_lines(raw.get("cp_lines")),
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    )
