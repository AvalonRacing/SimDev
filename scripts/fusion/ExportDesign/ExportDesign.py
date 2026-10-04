"""Export the design parts of the Fusion vehicle model: Body.step and Wing.step.

The DESIGN_PARTS of pipeline/simdev/cad/library.py, named so the CAD library
and the web UI upload accept them as they are. They belong in
CAD/designs/<design>/<state>/: the attitude is baked into the export, so pose
the model in the driving state first, and export once per state.

The parts are found by part number: AVxxx_01_... is the bodyshell (Body),
AVxxx_02_... the wing (Wing). An occurrence belongs to the part of the nearest
occurrence at or above it that matches, so everything inside the AVxxx_01
assembly goes into Body unless it is itself an AVxxx_02 part.

Fusion's STEP export writes what is visible, so each file is exported with
everything else hidden. The light bulbs are put back afterwards, whatever
happens, and each file is checked against what should have been visible
before it is written. Bodies whose own light bulb is off stay hidden and are
not exported.

Install: Fusion > Utilities > Scripts and Add-Ins > "+" > pick this folder.
"""

import os
import re
import shutil
import traceback

import adsk.core
import adsk.fusion

# --- Naming ---------------------------------------------------------------
# Matched, ignoring case, against the occurrence name (without ":1") and the
# component name. The part number takes no underscore, so "AV123_02_x_01"
# is a wing, not a bodyshell.
PART_PATTERNS = {
    "Body": r"^AV[0-9a-z]*_01(?![0-9])",
    "Wing": r"^AV[0-9a-z]*_02(?![0-9])",
}

# Export order, and the file names: <part>.step.
PART_ORDER = ("Body", "Wing")
TMP_DIR_NAME = ".design_export"

_COMPILED = {p: re.compile(rx, re.IGNORECASE) for p, rx in PART_PATTERNS.items()}


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
        self.hits = {}        # fullPathName -> parts its own name matched
        for occ in root.allOccurrences:
            key = occ.fullPathName
            self.occs[key] = occ
            context = occ.assemblyContext
            self.parent[key] = context.fullPathName if context else None
            self.hits[key] = _matching_parts(occ)
            if len(self.hits[key]) > 1:
                self.problems.append("%s matches both Body and Wing" % _label(key))

        self.owner = {}
        for key in self.occs:
            self._owner_of(key)

        self.owned = {part: set() for part in PART_ORDER}
        self.units_by_part = {part: [] for part in PART_ORDER}
        for key, part in self.owner.items():
            if part:
                self.owned[part].add(key)
                if part in self.hits[key] and self.owner.get(self.parent[key]) != part:
                    self.units_by_part[part].append(key)

    def _owner_of(self, key):
        if key in self.owner:
            return self.owner[key]
        hits = self.hits[key]
        if len(hits) == 1:
            owner = hits[0]
        elif len(hits) > 1:
            owner = None  # reported as a problem
        elif self.parent[key] is not None:
            owner = self._owner_of(self.parent[key])
        else:
            owner = None
        self.owner[key] = owner
        return owner

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


def _matching_parts(occ):
    names = {re.sub(r":\d+$", "", occ.name), occ.component.name}
    return [p for p in PART_ORDER if any(_COMPILED[p].search(n) for n in names)]


def _label(key):
    return key.replace("+", " / ")


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
            report.append("%-5s %3d solids%s" % (part, solids, note))
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
    # Only a complete pair replaces the old files: a new Body next to an old
    # Wing would still upload and run.
    for part in PART_ORDER:
        os.replace(os.path.join(tmp, part + ".step"), os.path.join(folder, part + ".step"))
    shutil.rmtree(tmp, ignore_errors=True)
    return report, failed


# --- Dialogs --------------------------------------------------------------


def _problems(model):
    problems = list(model.problems)
    for part in PART_ORDER:
        if not model.owned[part]:
            problems.append("%s: nothing matches %s" % (part, PART_PATTERNS[part]))
        elif not model.expected_bodies(part):
            problems.append("%s: found, but has no visible-bulb bodies" % part)
    return problems


def _ask_folder(ui):
    dialog = ui.createFolderDialog()
    dialog.title = "Folder for Body.step and Wing.step (CAD/designs/<design>/<state>)"
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
    lines += ["", "The model's current attitude is baked in: is it posed in the "
                  "driving state this folder is for?"]
    units = design.unitsManager.defaultLengthUnits
    if units != "mm":
        lines += ["", "Warning: the design's units are %r; SimDev expects millimetres." % units]
    existing = [p for p in PART_ORDER if os.path.exists(os.path.join(folder, p + ".step"))]
    if existing:
        lines += ["", "These files will be overwritten: " + ", ".join(existing)]
    result = ui.messageBox(
        "\n".join(lines),
        "Export design",
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
                + "\n\nAdjust PART_PATTERNS at the top of ExportDesign.py.",
                "Export design",
            )
            return

        folder = _ask_folder(ui)
        if not folder or not _confirm(ui, design, model, folder):
            return

        report, failed = export_all(design, model, folder)
        message = "Exported Body.step and Wing.step to\n%s\n\n%s" % (folder, "\n".join(report))
        if any("<--" in line for line in report):
            message += ("\n\nA solid count differs from the visible bodies. A body "
                        "with several lumps writes several solids; anything else "
                        "means hidden geometry was exported - check that file.")
        if failed:
            message += "\n\nThese light bulbs could not be put back:\n" + "\n".join(failed)
        ui.messageBox(message, "Export design")
    except ExportError as error:
        if ui:
            ui.messageBox("Nothing was exported.\n\n%s" % error, "Export design")
    except Exception:
        if ui:
            ui.messageBox("Nothing was exported.\n\n" + traceback.format_exc(),
                          "Export design")
