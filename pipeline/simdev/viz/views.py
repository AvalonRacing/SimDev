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

SLICE_FIELDS: tuple[str, ...] = ("cp", "cpt", "U", "vort", "lambda2")
SURFACE_FIELDS: tuple[str, ...] = ("cp", "yplus")
STREAMLINE_MODES: tuple[str, ...] = ("off", "lic", "seeded")

# The StarCCM+ colourmaps, named as the old pipeline's simConfig.txt named
# them. viz/pv_render.py maps them onto ParaView presets.
#
# VALIDATED HERE, IN THE VENV, because the failure downstream is silent:
# ParaView's ApplyPreset returns without complaint on a preset name it does
# not know and leaves the default map in place. A typo would produce 364
# pictures in the wrong colours with nothing anywhere saying so.
COLORMAPS: tuple[str, ...] = ("spectrum", "thermal")


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
class Views:
    datum_patches: tuple[str, ...]
    axes: dict[str, Axis]
    fields: dict[str, FieldStyle]
    parallel_scale: dict[str, float]
    focus_height: float
    resolution: tuple[int, int]
    streamlines: str
    # sha256 of the file text, truncated. Recorded beside every picture so a
    # PNG can always be traced back to the definition that framed it.
    digest: str


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
            (float(v["limits"][0]), float(v["limits"][1])), str(v["colormap"])
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
        focus_height=float(camera.get("focus_height", 0.0)),
        resolution=(int(camera["resolution"][0]), int(camera["resolution"][1])),
        streamlines=streamlines,
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    )
