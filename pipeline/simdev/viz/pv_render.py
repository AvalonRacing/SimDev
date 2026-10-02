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

from paraview import servermanager  # type: ignore[import-not-found]
from paraview.simple import (  # type: ignore[import-not-found]
    Calculator, ColorBy, CreateRenderView, Delete, GetColorTransferFunction,
    GetScalarBar, Hide, LegacyVTKReader, MergeBlocks, Render, SaveScreenshot,
    Show, Slice, XMLPolyDataReader,
)

# View-file field name -> the array name actually present on the sampled
# surface. cp, cpt and U are built by the Calculators below and are
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

# The old pipeline's pressure-coefficient bar, read off a screenshot of it:
# 32 flat bands, black through blue, cyan, green, yellow, red and magenta to
# white. Not a ParaView preset, so it is built here rather than looked up -
# 0-255 RGB, lowest band first, stretched across whatever limits the field
# declares.
BANDED = (
    (0, 0, 0), (0, 0, 121), (0, 0, 171), (0, 0, 210),
    (0, 0, 242), (0, 92, 255), (0, 152, 255), (0, 194, 255),
    (0, 229, 255), (0, 255, 251), (0, 255, 220), (0, 255, 183),
    (0, 255, 137), (0, 255, 65), (102, 255, 0), (159, 255, 0),
    (200, 255, 0), (234, 255, 0), (255, 247, 0), (255, 215, 0),
    (255, 177, 0), (255, 130, 0), (255, 46, 0), (255, 0, 112),
    (255, 0, 165), (255, 0, 205), (255, 0, 238), (255, 79, 255),
    (255, 145, 255), (255, 189, 255), (255, 224, 255), (255, 255, 255),
)


def _apply_colormap(lut, name, low, high):
    """Point `lut` at the named colourmap across [low, high].

    Strict. ApplyPreset does not raise on a name it does not know, it just
    leaves the default map in place - so an unmapped name has to fail here
    or it will not fail anywhere.
    """
    if name == "banded":
        # HARD STEPS, NOT A RAMP. Each band is written as two points of the
        # same colour at its two edges, so nothing is interpolated between
        # neighbours. Discretizing to exactly one table entry per band keeps
        # the colour bar drawing the same 32 blocks the old pipeline did.
        n = len(BANDED)
        points = []
        for i, rgb in enumerate(BANDED):
            colour = [c / 255.0 for c in rgb]
            points += [low + (high - low) * i / n, *colour]
            points += [low + (high - low) * (i + 1) / n, *colour]
        lut.ColorSpace = "RGB"
        lut.RGBPoints = points
        lut.Discretize = 1
        lut.NumberOfTableValues = n
        lut.RescaleOnVisibilityChange = 0
        lut.AutomaticRescaleRangeMode = "Never"
        return
    if name not in PRESETS:
        raise KeyError(
            f"no ParaView colormap for {name!r}; "
            f"known: banded, {', '.join(sorted(PRESETS))}"
        )
    lut.ApplyPreset(PRESETS[name], True)
    lut.RescaleTransferFunction(low, high)


BACKGROUND = [0.85, 0.85, 0.85]


def _reader(path):
    if str(path).endswith(".vtp"):
        return XMLPolyDataReader(FileName=[str(path)])
    return LegacyVTKReader(FileNames=[str(path)])


def _derive(source, frame):
    """Add U_rel, cp, cpt and Umag to a sampled surface.

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
    # THE +1 IS THE CONVENTION, NOT A FUDGE.
    #
    # cpt is referenced to freestream TOTAL pressure, so undisturbed flow
    # reads 1 and loss reads below it - the same scale StarCCM+ reports and
    # the same one published plots use, which is the point: these pictures
    # get read against other people's. Written out,
    #
    #   cpt = (p + 0.5|U_rel|^2) / q      with the local head corrected
    #
    # and in undisturbed flow p = 0 and |U_rel| = U_ff, so the first term is
    # 0.5*U_ff^2/q - which is 1 in straight-line flow and NOT 1 in a
    # rotating frame, where the undisturbed speed varies with radius. The
    # local-head subtraction removes that variation and the +1 puts the
    # datum back where the convention wants it. Both are needed: without the
    # subtraction a cornering freestream is graded by radius; without the +1
    # it sits at 0 and nobody else's scale matches.
    cpt = Calculator(Input=cp)
    cpt.ResultArrayName = "cpt"
    cpt.Function = f"(pMean + 0.5*mag(U_rel)^2 - 0.5*({uff2}))/{q!r} + 1"

    return cpt



# NO TEXT IS DRAWN INTO THE IMAGE. Only the colour bar.
#
# These pictures get read side by side and cropped into reports, and an
# overlay block in the corner is in the way for both. The provenance that
# block used to carry has not been dropped - it is in results/images.json
# (run, spec hash, views digest, datum, counts, timings), in the
# results/views.yaml copy of the definition that framed every picture, and
# across the top of results/index.html. What is gone is only the copy that
# was burnt into the pixels.


def _draw(source, field, style, camera, out_path, resolution, unlit=False):
    view = CreateRenderView()
    view.OrientationAxesVisibility = 0
    # LIGHT GREY, NOT WHITE. The banded map's top band is white, and
    # freestream sits there in cpt (1) and lambda2 (0) - on a white page the
    # plane's edge and the ground line vanished into the background.
    view.Background = BACKGROUND
    view.UseColorPaletteForBackground = 0

    display = Show(source, view)
    # CAMERA AFTER Show, NOT BEFORE. The first Show in a session resets the
    # camera to fit the data, so a camera set beforehand is thrown away for
    # that one picture: the first slice of every run came out zoomed to the
    # whole domain plane, with nothing saying why only that one was wrong.
    view.CameraParallelProjection = 1
    view.CameraPosition = camera["position"]
    view.CameraFocalPoint = camera["focal"]
    view.CameraViewUp = camera["up"]
    view.CameraParallelScale = camera["parallel_scale"]

    ColorBy(display, ("POINTS", field))
    if unlit:
        # A slice is a flat cut, so shading tells nothing about shape - it
        # only darkens every colour by ~20%, which turns the top band's
        # white into grey and makes the picture disagree with its own bar.
        # Pure ambient draws the map's exact colours. The car surface views
        # keep their lighting: there it is what shows the shape.
        display.Ambient = 1.0
        display.Diffuse = 0.0
        display.Specular = 0.0

    lut = GetColorTransferFunction(field)
    low, high = style["limits"]
    _apply_colormap(lut, style["colormap"], low, high)

    bar = GetScalarBar(lut, view)
    # NO TITLE. "only colourscale" - and the field is already named by the
    # directory the picture sits in (cp_x/, surface_yplus/) and by its own
    # filename, so a label in the pixels would only repeat it. The numeric
    # labels stay: without them the bar is a decoration rather than a scale.
    bar.Title = ""
    bar.ComponentTitle = ""
    # HORIZONTAL, ALONG THE BOTTOM. The default is a vertical bar in the
    # lower-right corner, which draws its tick labels to the RIGHT of the
    # bar - off the edge of the frame, so the numbers come out clipped and
    # the scale is unreadable. Laid flat along the bottom the labels sit
    # under the bar with room for them, and it fills the dead band beneath
    # the ground plane rather than covering flow.
    bar.AutoOrient = 0
    bar.Orientation = "Horizontal"
    # Placed by hand rather than by WindowLocation. A horizontal bar draws
    # its tick labels BELOW itself, and "Lower Center" sits flush with the
    # bottom edge - so the bar appeared but its numbers fell off the frame.
    # Lifting it to 7% of frame height leaves room for them underneath.
    bar.WindowLocation = "Any Location"
    bar.Position = [0.37, 0.07]
    # Smaller than ParaView's default (0.33 long, 16 thick). The bar is a
    # key, not a feature of the picture.
    bar.ScalarBarLength = 0.26
    bar.ScalarBarThickness = 9
    bar.TitleFontSize = 11
    bar.LabelFontSize = 10
    # Two significant figures. The limits are round numbers by design, so
    # the default "1.0e+00" spelling of 1 is noise - and a long label is
    # what overflowed the frame in the first place.
    bar.LabelFormat = "%-#.3g"
    bar.RangeLabelFormat = "%-#.3g"
    # BLACK. ParaView defaults the bar's text to white, which is sized for
    # its own dark viewport - on the light background these images use it is
    # drawn, correctly, in white on white. The numbers were there the whole
    # time and invisible, which reads exactly like a bar with no labels.
    bar.LabelColor = [0.0, 0.0, 0.0]
    bar.TitleColor = [0.0, 0.0, 0.0]

    display.SetScalarBarVisibility(view, True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Render(view)
    SaveScreenshot(out_path, view, ImageResolution=resolution)
    Hide(source, view)
    Delete(view)


def _cut_cp_lines(spec, missing):
    """Cut each part at each car-y station and write the points to CSV.

    Only the cut happens here - it needs VTK, which the venv does not have.
    The plots are drawn from these CSVs by viz/cplines.py on the venv side,
    where they can be tested without ParaView. Raw pMean goes out, not cp:
    the dynamic head is the venv's to apply, from the same u_inf as the
    force coefficients.

    Returns how many station CSVs were written.
    """
    if not spec:
        return 0
    readers = {}
    for patch, sample in spec.get("samples", {}).items():
        if sample and os.path.exists(sample):
            readers[patch] = _reader(sample)
        else:
            missing.append(f"cpline_{patch}")
    if not readers:
        return 0

    datum = spec["datum"]
    x_axis, z_axis = spec["x_axis"], spec["z_axis"]
    written = 0
    for station in spec["stations"]:
        rows = []
        for patch, reader in readers.items():
            cut = Slice(Input=reader)
            cut.SliceType = "Plane"
            cut.SliceType.Origin = station["point"]
            cut.SliceType.Normal = station["normal"]
            merged = MergeBlocks(Input=cut)
            data = servermanager.Fetch(merged)
            pressure = data.GetPointData().GetArray("pMean")
            if pressure is not None:
                for i in range(data.GetNumberOfPoints()):
                    point = data.GetPoint(i)
                    d = [point[k] - datum[k] for k in range(3)]
                    rows.append((
                        patch,
                        sum(d[k] * x_axis[k] for k in range(3)),
                        sum(d[k] * z_axis[k] for k in range(3)),
                        pressure.GetValue(i),
                    ))
            Delete(merged)
            Delete(cut)
        os.makedirs(os.path.dirname(station["csv"]), exist_ok=True)
        with open(station["csv"], "w", encoding="utf-8") as handle:
            handle.write("patch,x,z,pMean\n")
            for row in rows:
                handle.write(f"{row[0]},{row[1]!r},{row[2]!r},{row[3]!r}\n")
        written += 1
    return written


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
                _draw(
                    source,
                    ARRAY_NAMES.get(field, field),
                    plan["fields"][field],
                    entry["camera"],
                    image["out"],
                    resolution,
                    unlit=group is plan["slices"],
                )
                written += 1

    cut = _cut_cp_lines(plan.get("cp_lines"), missing)

    print(json.dumps({"written": written, "missing": missing, "cut": cut}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
