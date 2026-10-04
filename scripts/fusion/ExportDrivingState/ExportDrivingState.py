"""Export the driving state of the Fusion vehicle model as 13 STEP files.

Chassis, SUS_*, Tire_*, MRF_* - the STATE_PARTS of pipeline/simdev/cad/library.py,
one file each, named so the CAD library and the web UI upload accept them as
they are, in the attitude the model is in right now.

Fusion's STEP export writes what is visible, so every file is exported with
everything else hidden. The light bulbs are put back afterwards, whatever
happens, and each file is checked against what should have been visible
before it is written.

WHICH PART. The kind of an occurrence - Setup, tire, MRF zone, suspension -
comes from its name (the words below). An occurrence belongs to the kind of
the nearest occurrence at or above it that matches, so a Tire nested inside
the Setup assembly is a tire, not chassis, and a bearing inside a Wheelhub is
suspension.

WHICH CORNER. Tires and suspension parts are the same component in every
corner ("Tire:1" ... "Tire:4"), so the corner cannot come from the name. It
comes from where the part is: the four tires give the car's own frame (front
is the axle with the larger x, left is +y with z up, as the SimDev CAD is
built), and every tire, MRF and suspension occurrence goes to the quadrant of
that frame its bounding box centre sits in. Measuring in the car's frame
rather than against the origin keeps a yawed or offset car right. A part too
close to the centre line or the mid-wheelbase to call is refused, not
guessed.

Everything is shown for confirmation before anything is written.

Bodies whose own light bulb is off stay hidden and are not exported: that is
how helper or construction bodies are kept out of a part.

Install: Fusion > Utilities > Scripts and Add-Ins > "+" > pick this folder.
"""

import math
import os
import re
import shutil
import traceback

import adsk.core
import adsk.fusion

# --- Naming ---------------------------------------------------------------
# Edit these if the model names things differently. Matching ignores case and
# is tried against both the occurrence name (without ":1") and the component
# name. A word only counts where no letter comes before it, so "Hub" does not
# match "Github" but "Tierodfront" and "DS_combined" do match.

TIRE_WORDS = ["Tire", "Tyre"]
MRF_WORDS = ["MRF"]
# LCA, UCA, wheel hub, tie rod front/rear, drive shaft (combined).
SUS_WORDS = ["LCA", "UCA", "Wheel_?hub", "Hub", "Tie_?rod", "DS", "Drive_?shaft"]
# The Setup assembly is exported whole (minus the parts above), as Chassis.
CHASSIS_PATTERN = r"^Setup(?![a-z])"

# A part is refused rather than guessed when its centre is closer than this to
# the car's centre line (fraction of half the track) or to the middle of the
# wheelbase (fraction of half the wheelbase).
LATERAL_MARGIN = 0.10
LONGITUDINAL_MARGIN = 0.25


def _starts_word(words):
    return re.compile(r"(?<![a-z])(?:%s)" % "|".join(words), re.IGNORECASE)


KINDS = {
    "Chassis": re.compile(CHASSIS_PATTERN, re.IGNORECASE),
    "SUS": _starts_word(SUS_WORDS),
    "Tire": _starts_word(TIRE_WORDS),
    "MRF": _starts_word(MRF_WORDS),
}
CORNER_KINDS = ("SUS", "Tire", "MRF")
CORNERS = ("FL", "FR", "RL", "RR")

# Export order, and the file names: <part>.step.
PART_ORDER = ("Chassis",) + tuple(
    "%s_%s" % (kind, corner) for kind in CORNER_KINDS for corner in CORNERS
)
TMP_DIR_NAME = ".driving_state_export"


class ExportError(Exception):
    pass


# --- The model ------------------------------------------------------------


class Model:
    """Which part every occurrence and body of the design belongs to."""

    def __init__(self, root):
        self.root = root
        self.problems = []
        self.occs = {}        # fullPathName -> occurrence
        self.parent = {}      # fullPathName -> parent fullPathName or None
        self.hits = {}        # fullPathName -> kinds its own name matched
        for occ in root.allOccurrences:
            key = occ.fullPathName
            self.occs[key] = occ
            context = occ.assemblyContext
            self.parent[key] = context.fullPathName if context else None
            self.hits[key] = _matching_kinds(occ)
        for key, hits in self.hits.items():
            if len(hits) > 1:
                self.problems.append(
                    "%s matches more than one kind: %s" % (_label(key), ", ".join(hits))
                )

        self.kind = {}
        for key in self.occs:
            self._kind_of(key)

        # A unit is a run of one corner kind: the topmost occurrence of it and
        # everything below it of the same kind. A unit sits in one corner.
        self.unit = {}
        for key in self.occs:
            self._unit_of(key)
        members = {}
        for key, unit in self.unit.items():
            if unit:
                members.setdefault(unit, []).append(key)

        centres = {u: self._centre(keys) for u, keys in members.items()}
        corners = self._corners(centres)

        self.owner = {}
        self.units_by_part = {part: [] for part in PART_ORDER}
        for key in self.occs:
            kind = self.kind[key]
            if kind == "Chassis":
                self.owner[key] = "Chassis"
                if "Chassis" in self.hits[key] and self.kind.get(self.parent[key]) != "Chassis":
                    self.units_by_part["Chassis"].append(key)
            elif kind in CORNER_KINDS and corners.get(self.unit[key]):
                part = "%s_%s" % (kind, corners[self.unit[key]])
                self.owner[key] = part
                if self.unit[key] == key:
                    self.units_by_part[part].append(key)
            else:
                self.owner[key] = None

        self.owned = {part: set() for part in PART_ORDER}
        for key, part in self.owner.items():
            if part:
                self.owned[part].add(key)

    def _kind_of(self, key):
        if key in self.kind:
            return self.kind[key]
        hits = self.hits[key]
        if len(hits) == 1:
            kind = hits[0]
        elif len(hits) > 1:
            kind = None  # reported as a problem
        elif self.parent[key] is not None:
            kind = self._kind_of(self.parent[key])
        else:
            kind = None
        self.kind[key] = kind
        return kind

    def _unit_of(self, key):
        if key in self.unit:
            return self.unit[key]
        kind = self.kind[key]
        parent = self.parent[key]
        if kind not in CORNER_KINDS:
            unit = None
        elif parent is not None and self.kind[parent] == kind:
            unit = self._unit_of(parent)
        else:
            unit = key
        self.unit[key] = unit
        return unit

    def _centre(self, keys):
        """Bounding box centre of the exported bodies, in root coordinates."""
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for _, body in self.bodies(keys):
            if not body.isLightBulbOn:
                continue
            box = body.boundingBox
            for i, (a, b) in enumerate(zip(box.minPoint.asArray(), box.maxPoint.asArray())):
                lo[i] = min(lo[i], a)
                hi[i] = max(hi[i], b)
        if lo[0] == math.inf:
            return None
        return [(a + b) / 2 for a, b in zip(lo, hi)]

    def _corners(self, centres):
        """unit -> corner, from the car frame the four tires span."""
        tires = [u for u in centres if self.kind[u] == "Tire" and centres[u]]
        if len(tires) != 4:
            self.problems.append(
                "found %d tires, need 4 to tell the corners apart: %s"
                % (len(tires), ", ".join(_label(u) for u in tires) or "none")
            )
            return {}

        points = [centres[u] for u in tires]
        centre = _mean(points)
        by_x = sorted(points, key=lambda p: p[0], reverse=True)
        forward = _sub(_mean(by_x[:2]), _mean(by_x[2:]))
        forward[2] = 0.0
        length = math.hypot(forward[0], forward[1])
        if length == 0.0:
            self.problems.append("the four tires do not span a front and a rear axle")
            return {}
        forward = [forward[0] / length, forward[1] / length, 0.0]
        left = [-forward[1], forward[0], 0.0]  # z x forward

        def frame(p):
            d = _sub(p, centre)
            return _dot(d, forward), _dot(d, left)

        half_wheelbase = _mean([[abs(frame(p)[0])] for p in points])[0]
        half_track = _mean([[abs(frame(p)[1])] for p in points])[0]

        corners = {}
        for unit, p in centres.items():
            if p is None:
                continue  # no visible bodies: nothing to export either way
            along, across = frame(p)
            if (abs(along) < LONGITUDINAL_MARGIN * half_wheelbase
                    or abs(across) < LATERAL_MARGIN * half_track):
                self.problems.append(
                    "%s sits too close to the car's centre to tell its corner "
                    "(%.0f%% of half the wheelbase forward, %.0f%% of half the "
                    "track left)" % (_label(unit), 100 * along / half_wheelbase,
                                     100 * across / half_track)
                )
                continue
            corners[unit] = ("F" if along > 0 else "R") + ("L" if across > 0 else "R")

        tire_corners = sorted(corners.get(u, "?") for u in tires)
        if tire_corners != sorted(CORNERS):
            self.problems.append(
                "the four tires land in corners %s, not one in each" % ", ".join(tire_corners)
            )
            return {}
        return corners

    def ancestors(self, part):
        """Occurrences above the part's own that are not the part's."""
        found = set()
        for key in self.owned[part]:
            up = self.parent[key]
            while up is not None and up not in found:
                if self.owner[up] != part:
                    found.add(up)
                up = self.parent[up]
        return found

    def matched_names(self, part):
        return [_label(k) for k in sorted(self.units_by_part[part])]

    def unexported(self):
        """Occurrences with visible-bulb bodies of their own that no part takes."""
        return [
            _label(k) for k in sorted(self.occs)
            if self.owner[k] is None
            and any(b.isLightBulbOn for b in self.occs[k].bRepBodies)
        ]

    def bodies(self, keys):
        for key in keys:
            for body in self.occs[key].bRepBodies:
                yield key, body

    def all_bodies(self):
        for body in self.root.bRepBodies:
            yield "(root)", body
        for key in self.occs:
            for body in self.occs[key].bRepBodies:
                yield key, body

    def expected_bodies(self, part):
        return [(k, b) for k, b in self.bodies(self.owned[part]) if b.isLightBulbOn]


def _matching_kinds(occ):
    names = {re.sub(r":\d+$", "", occ.name), occ.component.name}
    return [kind for kind, rx in KINDS.items() if any(rx.search(n) for n in names)]


def _label(key):
    return key.replace("+", " / ")


def _mean(points):
    return [sum(c) / len(points) for c in zip(*points)]


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# --- Visibility -----------------------------------------------------------


class Visibility:
    """Remembers every light bulb this script may touch, and puts them back."""

    def __init__(self, model):
        self.model = model
        self.occ_bulbs = {k: o.isLightBulbOn for k, o in model.occs.items()}
        self.comps = {}
        for comp in [model.root] + [o.component for o in model.occs.values()]:
            token = comp.entityToken
            if token not in self.comps:
                self.comps[token] = (comp, comp.isBodiesFolderLightBulbOn)

    def show_only(self, part):
        model = self.model
        owned = model.owned[part]
        ancestors = model.ancestors(part)

        for key, occ in model.occs.items():
            want = key in owned or key in ancestors
            if occ.isLightBulbOn != want:
                occ.isLightBulbOn = want

        # An ancestor's own bodies belong to another part (or to none), so its
        # bodies folder goes off. The root is everyone's ancestor.
        on = {model.occs[k].component.entityToken for k in owned}
        off = {model.root.entityToken}
        off.update(model.occs[k].component.entityToken for k in ancestors)
        for token in on & off:
            comp = self.comps[token][0]
            if comp.bRepBodies.count:
                raise ExportError(
                    "%s: component %r is used both inside this part and above "
                    "it, so its bodies cannot be shown in one place and hidden "
                    "in the other. Give the two uses separate components."
                    % (part, comp.name)
                )
        for token in on:
            self._folder(token, True)
        for token in off - on:
            self._folder(token, False)
        adsk.doEvents()

    def _folder(self, token, want):
        comp = self.comps[token][0]
        if comp.isBodiesFolderLightBulbOn != want:
            comp.isBodiesFolderLightBulbOn = want

    def restore(self):
        failed = []
        for key, bulb in self.occ_bulbs.items():
            try:
                occ = self.model.occs[key]
                if occ.isLightBulbOn != bulb:
                    occ.isLightBulbOn = bulb
            except Exception:
                failed.append(key)
        for comp, bulb in self.comps.values():
            try:
                if comp.isBodiesFolderLightBulbOn != bulb:
                    comp.isBodiesFolderLightBulbOn = bulb
            except Exception:
                failed.append(comp.name + " (bodies folder)")
        adsk.doEvents()
        return failed


# --- Export ---------------------------------------------------------------


def _check_visible(model, part):
    """Raise unless exactly the part's bodies are visible."""
    expected = {b.entityToken: (k, b.name) for k, b in model.expected_bodies(part)}
    visible = {
        b.entityToken: (k, b.name) for k, b in model.all_bodies() if b.isVisible
    }
    extra = [visible[t] for t in visible if t not in expected]
    missing = [expected[t] for t in expected if t not in visible]
    if not visible:
        raise ExportError("%s: no body is visible, nothing to export" % part)
    if extra or missing:
        lines = ["%s: the visible bodies are not the part's own." % part]
        lines += ["  visible but not in %s: %s / %s" % (part, k, n) for k, n in extra[:10]]
        lines += ["  in %s but hidden: %s / %s" % (part, k, n) for k, n in missing[:10]]
        raise ExportError("\n".join(lines))
    return sum(1 for k, b in model.expected_bodies(part) if b.isSolid)


def _count_step_solids(path):
    with open(path, "r", errors="replace") as handle:
        text = handle.read()
    return len(re.findall(r"=\s*MANIFOLD_SOLID_BREP\s*\(", text))


def export_all(design, model, folder):
    tmp = os.path.join(folder, TMP_DIR_NAME)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    manager = design.exportManager
    visibility = Visibility(model)
    report = []
    try:
        for part in PART_ORDER:
            visibility.show_only(part)
            solids = _check_visible(model, part)
            path = os.path.join(tmp, part + ".step")
            options = manager.createSTEPExportOptions(path, model.root)
            if not manager.execute(options) or not os.path.isfile(path):
                raise ExportError("%s: Fusion did not write %s" % (part, path))
            in_file = _count_step_solids(path)
            note = "" if in_file == solids else "   <-- %d solids in the file" % in_file
            report.append("%-8s %3d solids%s" % (part, solids, note))
    except Exception:
        failed = visibility.restore()
        shutil.rmtree(tmp, ignore_errors=True)
        if failed:
            raise ExportError(
                "Export failed, and these light bulbs could not be put back:\n"
                + "\n".join(failed) + "\n\n" + traceback.format_exc()
            )
        raise

    failed = visibility.restore()
    # Only a complete state replaces the old files: a folder mixing a new
    # Chassis with an old SUS_FL would still upload and run.
    for part in PART_ORDER:
        os.replace(os.path.join(tmp, part + ".step"), os.path.join(folder, part + ".step"))
    shutil.rmtree(tmp, ignore_errors=True)
    return report, failed


# --- Dialogs --------------------------------------------------------------


def _problems(model):
    problems = list(model.problems)
    for part in PART_ORDER:
        if not model.owned[part]:
            problems.append("%s: nothing found" % part)
        elif not model.expected_bodies(part):
            problems.append("%s: found, but has no visible-bulb bodies" % part)
    return problems


def _ask_folder(ui):
    dialog = ui.createFolderDialog()
    dialog.title = "Folder for the driving state STEP files"
    if dialog.showDialog() != adsk.core.DialogResults.DialogOK:
        return None
    return dialog.folder


def _confirm(ui, design, model, folder):
    lines = ["Export to:", folder, ""]
    for part in PART_ORDER:
        names = model.matched_names(part)
        shown = ", ".join(names[:4]) + (" + %d more" % (len(names) - 4) if len(names) > 4 else "")
        lines.append("%s.step  (%d bodies)  <-  %s"
                     % (part, len(model.expected_bodies(part)), shown))
    # The corners are copies of each other, so a corner with a different
    # number of suspension parts has one misplaced or missing.
    counts = {len(model.units_by_part["SUS_" + c]) for c in CORNERS}
    if len(counts) > 1:
        lines += ["", "Warning: the corners have different numbers of suspension parts."]
    unexported = model.unexported()
    if unexported:
        shown = ", ".join(unexported[:8]) + (" + %d more" % (len(unexported) - 8)
                                            if len(unexported) > 8 else "")
        lines += ["", "In no file: " + shown]
    units = design.unitsManager.defaultLengthUnits
    if units != "mm":
        lines += ["", "Warning: the design's units are %r; SimDev expects millimetres." % units]
    existing = [p for p in PART_ORDER if os.path.exists(os.path.join(folder, p + ".step"))]
    if existing:
        lines += ["", "These files will be overwritten: " + ", ".join(existing)]
    result = ui.messageBox(
        "\n".join(lines),
        "Export driving state",
        adsk.core.MessageBoxButtonTypes.OKCancelButtonType,
        adsk.core.MessageBoxIconTypes.QuestionIconType,
    )
    return result == adsk.core.DialogResults.DialogOK


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox("Open the vehicle design (Design workspace) first.")
            return

        model = Model(design.rootComponent)
        problems = _problems(model)
        if problems:
            top = sorted(o.name for o in design.rootComponent.occurrences)
            ui.messageBox(
                "Nothing was exported:\n\n" + "\n".join(problems)
                + "\n\nTop-level occurrences: " + ", ".join(top[:40])
                + "\n\nAdjust the names at the top of ExportDrivingState.py.",
                "Export driving state",
            )
            return

        folder = _ask_folder(ui)
        if not folder or not _confirm(ui, design, model, folder):
            return

        report, failed = export_all(design, model, folder)
        message = "Exported 13 STEP files to\n%s\n\n%s" % (folder, "\n".join(report))
        if any("<--" in line for line in report):
            message += ("\n\nA solid count differs from the visible bodies. A body "
                        "with several lumps writes several solids; anything else "
                        "means hidden geometry was exported - check that file.")
        if failed:
            message += "\n\nThese light bulbs could not be put back:\n" + "\n".join(failed)
        ui.messageBox(message, "Export driving state")
    except ExportError as error:
        if ui:
            ui.messageBox("Nothing was exported.\n\n%s" % error, "Export driving state")
    except Exception:
        if ui:
            ui.messageBox("Nothing was exported.\n\n" + traceback.format_exc(),
                          "Export driving state")
