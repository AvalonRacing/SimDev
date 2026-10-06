"""Field deltas between two runs, as pictures framed like the run's own.

Runs under the ParaView interpreter (post.paraview_python), like
viz/pv_render.py, and imports nothing from simdev. VTK and ParaView are
imported inside the functions that need them, so the pure parts - grid,
fields, colours - can be imported and tested from the venv.

Never decodes the existing PNGs: on banded pictures a small change turns
every shifted band edge into a one-band ring, which reads as noise.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

BANDS = 20  # even, so zero is a band edge and a small offset still shows
SOLID = (212, 212, 212)
MOVED = (0, 0, 0)
PROBE_TOLERANCE = 1e-4  # cut points sit up to ~50 um off the plane
SURFACE_RADIUS = 5e-4

_NEG = [(5, 48, 97), (33, 102, 172), (67, 147, 195), (146, 197, 222), (209, 229, 240),
        (247, 247, 247)]
_POS = [(247, 247, 247), (253, 219, 199), (244, 165, 130), (214, 96, 77), (178, 24, 43),
        (103, 0, 31)]


def _ramp(anchors, t: float) -> tuple[int, int, int]:
    x = t * (len(anchors) - 1)
    i = min(int(x), len(anchors) - 2)
    f = x - i
    a, b = anchors[i], anchors[i + 1]
    return tuple(int(round(a[k] + (b[k] - a[k]) * f)) for k in range(3))


def diverging_table(n: int = BANDS) -> list[tuple[int, int, int]]:
    half = n // 2
    # Band centres; the band touching zero is the palest but never white.
    neg = [_ramp(_NEG, (i + 0.5) / half * 0.9) for i in range(half)]
    pos = [_ramp(_POS, 0.1 + (i + 0.5) / half * 0.9) for i in range(half)]
    return neg + pos


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def frame_grid(slice_entry: dict, resolution: Sequence[int]) -> tuple[np.ndarray, tuple[int, int]]:
    cam = slice_entry["camera"]
    n = _unit(slice_entry["normal"])
    point = np.asarray(slice_entry["point"], dtype=float)
    focal = np.asarray(cam["focal"], dtype=float)
    focal = focal - np.dot(focal - point, n) * n
    view = _unit(focal - np.asarray(cam["position"], dtype=float))
    up = np.asarray(cam["up"], dtype=float)
    up = _unit(up - np.dot(up, n) * n)
    right = _unit(np.cross(view, up))
    nx, ny = int(resolution[0]), int(resolution[1])
    hh = float(cam["parallel_scale"])
    hw = hh * nx / ny
    uu, vv = np.meshgrid(np.linspace(-hw, hw, nx), np.linspace(-hh, hh, ny))
    pts = focal + uu[..., None] * right + vv[..., None] * up
    return pts.reshape(-1, 3), (ny, nx)


def derived_fields(p, U, valid, pts, frame) -> dict[str, np.ndarray]:
    """cp, cpt and |U_rel| exactly as viz/pv_render.py builds them."""
    w = float(frame.get("omega") or 0.0)
    ox, oy, _ = frame["origin"]
    u_inf = float(frame["u_inf"])
    q = 0.5 * u_inf * u_inf
    U = np.asarray(U, dtype=float)
    dx, dy = pts[:, 0] - ox, pts[:, 1] - oy
    rel = np.stack([U[:, 0] + w * dy, U[:, 1] - w * dx, U[:, 2]], axis=1)
    umag = np.linalg.norm(rel, axis=1)
    p = np.asarray(p, dtype=float)
    uff2 = (w * np.hypot(dx, dy)) ** 2 if w else u_inf * u_inf
    out = {"cp": p / q, "cpt": (p + 0.5 * umag**2 - 0.5 * uff2) / q + 1.0, "U": umag}
    valid = np.asarray(valid, dtype=bool)
    return {k: np.where(valid, v, np.nan) for k, v in out.items()}


def delta_rgb(delta: np.ndarray, solid: np.ndarray, moved: np.ndarray, limit: float) -> np.ndarray:
    table = np.array(diverging_table(), dtype=np.uint8)
    scaled = np.nan_to_num((delta + limit) / (2 * limit) * BANDS, nan=0.0)
    index = np.clip(np.floor(scaled).astype(int), 0, BANDS - 1)
    rgb = table[index]
    rgb[solid] = SOLID
    rgb[moved] = MOVED
    return rgb[::-1].copy()  # grid row 0 is the bottom, image row 0 the top


def draw_colour_bar(rgb: np.ndarray, limit: float, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    h, w = rgb.shape[:2]
    # Same place as the pipeline's own bar: bottom centre.
    x0, x1 = int(w * 0.37), int(w * 0.63)
    y0, y1 = int(h * 0.90), int(h * 0.93)
    table = diverging_table()
    step = (x1 - x0) / BANDS
    for i, colour in enumerate(table):
        draw.rectangle([x0 + i * step, y0, x0 + (i + 1) * step, y1], fill=colour)
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0))
    # No text anchors: they need a FreeType font, which is not guaranteed.
    for text, x in ((f"{-limit:g}", x0), ("0", (x0 + x1) // 2), (f"+{limit:g}", x1)):
        draw.text((x - 4 * len(text), y0 - 14), text, fill=(0, 0, 0))
    draw.text(((x0 + x1) // 2 - 4 * len(label), y1 + 4), label, fill=(0, 0, 0))
    return np.asarray(image)


def _write_png(rgb: np.ndarray, out: Path) -> None:
    from PIL import Image

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(out, ".png")
    try:
        Image.fromarray(rgb).save(tmp)
        os.replace(tmp, out)  # a reader never gets half a picture
    finally:
        tmp.unlink(missing_ok=True)


def _tmp_name(target: Path, suffix: str) -> Path:
    """A scratch name next to target, unique per process."""
    return target.with_name(f"{target.stem}.{os.getpid()}.tmp{suffix}")


def _read(path: str):
    import vtk

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def _probe(path: str, pts: np.ndarray) -> dict[str, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(_read(path))
    tri.Update()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
    target = vtk.vtkPolyData()
    target.SetPoints(points)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(target)
    probe.SetSourceData(tri.GetOutput())
    probe.SetComputeTolerance(False)
    probe.SetTolerance(PROBE_TOLERANCE)
    probe.Update()
    data = probe.GetOutput().GetPointData()
    return {
        "p": vtk_to_numpy(data.GetArray("pMean")),
        "U": vtk_to_numpy(data.GetArray("UMean")),
        "valid": vtk_to_numpy(data.GetArray("vtkValidPointMask")).astype(bool),
    }


def plane_delta(request: dict[str, Any]) -> dict[str, Any]:
    field, limit = request["field"], float(request["limit"])
    pts, shape = frame_grid(request["slice"], request["resolution"])
    pane = _probe(request["pane"]["sample"], pts)
    ref = _probe(request["ref"]["sample"], pts)
    a = derived_fields(pane["p"], pane["U"], pane["valid"], pts, request["pane"]["frame"])[field]
    b = derived_fields(ref["p"], ref["U"], ref["valid"], pts, request["ref"]["frame"])[field]
    a, b = a.reshape(shape), b.reshape(shape)
    delta = a - b
    solid = ~np.isfinite(a) & ~np.isfinite(b)
    moved = np.isfinite(a) ^ np.isfinite(b)
    rgb = draw_colour_bar(delta_rgb(delta, solid, moved, limit), limit, f"delta {field}")
    _write_png(rgb, Path(request["out"]))
    both = np.isfinite(delta)
    return {"max_abs": float(np.abs(delta[both]).max()) if both.any() else None,
            "moved_pct": float(100 * moved.mean())}


def surface_delta_field(pane_path: str, ref_path: str, q_pane: float, q_ref: float):
    """Ref's pMean interpolated onto the pane's surface, as delta_cp.

    Each run's pressure is normalised by its own dynamic head.

    A pane point with no ref point within SURFACE_RADIUS gets NaN: surface
    that is new or moved in the pane, drawn black.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    pane, ref = _read(pane_path), _read(ref_path)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(ref)
    locator.BuildLocator()
    kernel = vtk.vtkLinearKernel()
    kernel.SetRadius(SURFACE_RADIUS)
    kernel.SetKernelFootprintToRadius()
    interp = vtk.vtkPointInterpolator()
    interp.SetInputData(pane)
    interp.SetSourceData(ref)
    interp.SetKernel(kernel)
    interp.SetLocator(locator)
    interp.PassPointArraysOff()  # else the pane's own pMean comes through under the same name
    interp.SetNullPointsStrategyToMaskPoints()
    interp.SetValidPointsMaskArrayName("has_ref")
    interp.Update()
    out = interp.GetOutput().GetPointData()
    p_pane = vtk_to_numpy(pane.GetPointData().GetArray("pMean"))
    p_ref = vtk_to_numpy(out.GetArray("pMean"))
    has = vtk_to_numpy(out.GetArray("has_ref")).astype(bool)
    d = np.where(has, p_pane / q_pane - p_ref / q_ref, np.nan).astype(np.float32)
    surface = vtk.vtkPolyData()
    surface.ShallowCopy(pane)
    array = numpy_to_vtk(d, deep=True)
    array.SetName("delta_cp")
    surface.GetPointData().AddArray(array)
    stats = {"moved_pct": float(100 * (~has).mean()),
             "max_abs": float(np.nanmax(np.abs(d))) if has.any() else None}
    return surface, stats


def surface_delta(request: dict[str, Any]) -> dict[str, Any]:
    import vtk

    limit = float(request["limit"])
    cache = Path(request["cache_vtp"])
    pane_s, ref_s = request["pane"]["sample"], request["ref"]["sample"]
    stats_path = cache.with_suffix(".json")
    fresh = stats_path.is_file() and cache.is_file() and cache.stat().st_mtime > max(
        Path(pane_s).stat().st_mtime, Path(ref_s).stat().st_mtime)
    if not fresh:
        q_pane = 0.5 * float(request["pane"]["frame"]["u_inf"]) ** 2
        q_ref = 0.5 * float(request["ref"]["frame"]["u_inf"]) ** 2
        surface, stats = surface_delta_field(pane_s, ref_s, q_pane, q_ref)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_name(cache, ".vtp")
        try:
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(str(tmp))
            writer.SetInputData(surface)
            writer.Write()
            os.replace(tmp, cache)
        finally:
            tmp.unlink(missing_ok=True)
        # Sidecar last: its presence marks a complete cache.
        side = _tmp_name(stats_path, ".json")
        try:
            side.write_text(json.dumps(stats), encoding="utf-8")
            os.replace(side, stats_path)
        finally:
            side.unlink(missing_ok=True)
    else:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    from paraview.simple import (
        CreateRenderView, GetColorTransferFunction, Render, SaveScreenshot, Show,
        XMLPolyDataReader, _DisableFirstRenderCameraReset,
    )

    # The first Render of a session otherwise refits the camera to the data
    # (as in pv_render.py), so the picture would not match the pipeline's.
    _DisableFirstRenderCameraReset()

    view = CreateRenderView()
    view.ViewSize = list(request["resolution"])
    view.OrientationAxesVisibility = 0
    try:
        view.UseColorPaletteForBackground = 0
    except AttributeError:
        pass
    view.Background = [c / 255 for c in SOLID]
    display = Show(XMLPolyDataReader(FileName=[str(cache)]), view)
    display.ColorArrayName = ["POINTS", "delta_cp"]
    lut = GetColorTransferFunction("delta_cp")
    points = []
    for i, colour in enumerate(diverging_table()):
        rgb = [c / 255 for c in colour]
        points += [-limit + 2 * limit * i / BANDS, *rgb, -limit + 2 * limit * (i + 1) / BANDS, *rgb]
    lut.ColorSpace = "RGB"
    lut.RGBPoints = points
    lut.Discretize = 1
    lut.NumberOfTableValues = BANDS
    lut.AutomaticRescaleRangeMode = "Never"
    lut.NanColor = [c / 255 for c in MOVED]
    display.LookupTable = lut
    cam = request["camera"]
    view.CameraFocalPoint = cam["focal"]
    view.CameraPosition = cam["position"]
    view.CameraViewUp = cam["up"]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = cam["parallel_scale"]
    Render(view)
    out = Path(request["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = _tmp_name(out, ".raw.png")
    from PIL import Image

    try:
        SaveScreenshot(str(raw), view, ImageResolution=list(request["resolution"]))
        shot = np.asarray(Image.open(raw).convert("RGB"))
    finally:
        raw.unlink(missing_ok=True)
    _write_png(draw_colour_bar(shot, limit, "delta cp"), out)
    return stats


def main(argv: Sequence[str]) -> int:
    started = time.time()
    try:
        request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        handler = {"plane": plane_delta, "surface": surface_delta}[request["kind"]]
        stats = handler(request)
    except Exception as error:  # the caller shows this text in the pane
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}))
        return 1
    print(json.dumps({"ok": True, "out": request["out"],
                      "seconds": round(time.time() - started, 2), **stats}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
