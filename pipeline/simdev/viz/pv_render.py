"""Turn a render plan into PNGs. Runs under the SYSTEM python, not the venv.

    /usr/bin/python3 pv_render.py render_plan.json

THIS FILE MUST NOT IMPORT simdev. ParaView is a system package; the venv has
no access to it and the ParaView interpreter has no access to the venv, so
this module and the rest of the pipeline meet only at render_plan.json.
Everything that needs testing lives on the other side of that file.

pvpython and pvbatch hang on this machine - measured, no output at a 150 s
timeout, not even for --version - while `python3 -c "import paraview.simple"`
imports and renders in seconds. Hence the plain interpreter.
"""

from __future__ import annotations

import json
import os
import sys

from paraview.simple import (  # type: ignore[import-not-found]
    Calculator, ColorBy, CreateRenderView, Delete, GetColorTransferFunction,
    GetScalarBar, Hide, LegacyVTKReader, Render, SaveScreenshot, Show,
    XMLPolyDataReader,
)

# View-file field name -> the array name actually present on the sampled
# surface. cp, cpt, U and vort are built by the Calculators below and are
# named to match; Lambda2Mean comes from the OpenFOAM function object and
# yPlus is OpenFOAM's own spelling. A mismatch here colours by nothing and
# renders a uniformly grey picture with no error.
ARRAY_NAMES = {"lambda2": "Lambda2Mean", "yplus": "yPlus"}

# StarCCM+ colourmap -> ParaView preset. Kept here rather than in the yaml so
# the view file stays about the picture rather than about ParaView.
#
#   spectrum  "Blue to Red Rainbow" is two stops, blue to red, interpolated
#             in HSV - which is exactly the blue-cyan-green-yellow-red ramp
#             StarCCM+ draws. Verified against the preset's own ColorSpace,
#             not assumed from its name.
#
#   thermal   "Black-Body Radiation": black, red, orange, white.
#
# DO NOT USE THE PRESET LITERALLY NAMED "Spectrum". It is an IndexedColors
# preset with no RGBPoints at all - a categorical palette meant for
# annotations - so colouring a continuous scalar field with it produces
# banded nonsense. The name is the only thing about it that fits.
PRESETS = {
    "spectrum": "Blue to Red Rainbow",
    "thermal": "Black-Body Radiation",
}


def _reader(path):
    if str(path).endswith(".vtp"):
        return XMLPolyDataReader(FileName=[str(path)])
    return LegacyVTKReader(FileNames=[str(path)])


def _derive(source, frame):
    """Add U_rel, cp, cpt, Umag and vorticity magnitude to a sampled surface.

    THE CAR FRAME, AND WHY IT IS NOT COSMETIC. In a cornering run OpenFOAM
    solves the ABSOLUTE velocity, so UMean's far field is at rest in the
    ground frame. Sliced raw beside a straight-line run the whole freestream
    changes colour and none of it is aerodynamics.

        U_rel  = UMean - omega x (x - origin)
        U_ff   = |omega| r          (the undisturbed car-frame speed)
        cp     = pMean / (0.5 u_inf^2)
        cpt    = (pMean + 0.5|U_rel|^2 - 0.5 U_ff^2) / (0.5 u_inf^2)

    cpt subtracts the LOCAL head because at R = 4 m and omega = 3.75 rad/s
    the undisturbed car-frame speed runs 12.75-17.25 m/s across the domain
    half-width. Against a constant u_inf that is a spurious +/-0.32 in cpt,
    graded radially, and it reads exactly like a wake.

    cp keeps 0.5 u_inf^2 in its denominator - the same dynamic head that
    non-dimensionalises Cd and Cl - so a cp picture and a coefficient sit on
    one scale.

    At omega = 0 every expression below collapses to the straight-line form,
    so there is one code path and no cornering branch.
    """
    w = float(frame["omega"])
    ox, oy, _oz = frame["origin"]
    u_inf = float(frame["u_inf"])
    q = 0.5 * u_inf * u_inf

    # omega x d for omega = (0, 0, w) and d = x - origin is (-w*dy, w*dx, 0),
    # so U_rel = (Ux + w*dy, Uy - w*dx, Uz).
    rel = Calculator(Input=source)
    rel.ResultArrayName = "U_rel"
    rel.Function = (
        f"(UMean_X + {w!r}*(coordsY - {oy!r}))*iHat"
        f" + (UMean_Y - {w!r}*(coordsX - {ox!r}))*jHat"
        f" + (UMean_Z)*kHat"
    )

    mag = Calculator(Input=rel)
    mag.ResultArrayName = "U"
    mag.Function = "mag(U_rel)"

    cp = Calculator(Input=mag)
    cp.ResultArrayName = "cp"
    cp.Function = f"pMean/{q!r}"

    # U_ff^2: (w r)^2 in a rotating frame, u_inf^2 when there is no rotation.
    uff2 = (
        f"({w!r}*sqrt((coordsX - {ox!r})^2 + (coordsY - {oy!r})^2))^2"
        if w
        else f"{u_inf * u_inf!r}"
    )
    cpt = Calculator(Input=cp)
    cpt.ResultArrayName = "cpt"
    cpt.Function = f"(pMean + 0.5*mag(U_rel)^2 - 0.5*({uff2}))/{q!r}"

    return cpt


def _vorticity(source, component):
    """|vorticity| normal to this plane - the component that shows vortices
    punching through the cut, which is what the old pipeline plotted."""
    axis = ("X", "Y", "Z")[component]
    out = Calculator(Input=source)
    out.ResultArrayName = "vort"
    out.Function = f"abs(vorticityMean_{axis})"
    return out


# NO TEXT IS DRAWN INTO THE IMAGE. Only the colour bar.
#
# These pictures get read side by side and cropped into reports, and an
# overlay block in the corner is in the way for both. The provenance that
# block used to carry has not been dropped - it is in results/images.json
# (run, spec hash, views digest, datum, counts, timings), in the
# results/views.yaml copy of the definition that framed every picture, and
# across the top of results/index.html. What is gone is only the copy that
# was burnt into the pixels.


def _draw(source, field, style, camera, out_path, resolution):
    view = CreateRenderView()
    view.CameraParallelProjection = 1
    view.CameraPosition = camera["position"]
    view.CameraFocalPoint = camera["focal"]
    view.CameraViewUp = camera["up"]
    view.CameraParallelScale = camera["parallel_scale"]
    view.OrientationAxesVisibility = 0
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0

    display = Show(source, view)
    ColorBy(display, ("POINTS", field))

    lut = GetColorTransferFunction(field)
    # Strict. ApplyPreset does not raise on a name it does not know, it just
    # leaves the default map in place - so an unmapped name has to fail here
    # or it will not fail anywhere.
    if style["colormap"] not in PRESETS:
        raise KeyError(
            f"no ParaView preset for colormap {style['colormap']!r}; "
            f"known: {', '.join(sorted(PRESETS))}"
        )
    lut.ApplyPreset(PRESETS[style["colormap"]], True)
    low, high = style["limits"]
    lut.RescaleTransferFunction(low, high)

    bar = GetScalarBar(lut, view)
    bar.Title = field
    bar.ComponentTitle = ""
    # Smaller than ParaView's default (0.33 long, 16 thick). The bar is a
    # key, not a feature of the picture; at the default size it takes a
    # noticeable bite out of a 1600x1200 frame that the flow should have.
    bar.ScalarBarLength = 0.22
    bar.ScalarBarThickness = 10
    bar.TitleFontSize = 11
    bar.LabelFontSize = 10

    display.SetScalarBarVisibility(view, True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Render(view)
    SaveScreenshot(out_path, view, ImageResolution=resolution)
    Hide(source, view)
    Delete(view)


def main(argv):
    with open(argv[1], encoding="utf-8") as handle:
        plan = json.load(handle)

    if plan.get("streamlines", "off") != "off":
        # Not implemented yet - see the plan, Task 14. The sampled surfaces
        # carry U_rel as a vector, so this needs no return to the volume; it
        # is waiting on a measurement, not on a mechanism.
        print(
            f"warning: streamlines={plan['streamlines']} is not implemented; "
            "rendering without them",
            file=sys.stderr,
        )

    frame = plan["frame"]
    resolution = plan["resolution"]

    written = 0
    missing = []

    for group in (plan["slices"], plan["surfaces"]):
        for entry in group:
            sample = entry.get("sample")
            if not sample or not os.path.exists(sample):
                missing.append(entry["name"])
                continue
            source = _derive(_reader(sample), frame)
            for image in entry["images"]:
                field = image["field"]
                node = source
                if field == "vort":
                    node = _vorticity(source, image["component"])
                _draw(
                    node,
                    ARRAY_NAMES.get(field, field),
                    plan["fields"][field],
                    entry["camera"],
                    image["out"],
                    resolution,
                )
                written += 1

    print(json.dumps({"written": written, "missing": missing}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
