from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from simdev.geometry.roles import PatchRole


class WallTreatment(str, Enum):
    LOW_Y_PLUS = "low_y_plus"
    HIGH_Y_PLUS = "high_y_plus"
    SPALDING = "spalding"


class Mode(str, Enum):
    STRAIGHT = "straight"
    CORNERING = "cornering"


class GroundMotion(str, Enum):
    STATIC = "static"
    MOVING = "moving"


class FlowDirection(str, Enum):
    """Which way along x the freestream blows.

    The car is never rotated to suit the pipeline - attitude belongs to the
    CAD - so the tunnel is turned around instead. A CAD assembly built
    nose-forward along +x is a car travelling along +x, which means the air
    comes at it from +x and the freestream is '-x'.

    Everything directional is derived from this: which end of the domain is
    the inlet, the drag axis, which side of the car is its left, the sign of
    the cornering frame, and the direction the road moves under the tyres.
    None of those may be written down separately, because a case with the
    tunnel reversed and the drag axis not is one that converges to a
    confidently negative Cd.
    """

    PLUS_X = "+x"
    MINUS_X = "-x"


class FlowConfig(BaseModel):
    u_inf: float = Field(gt=0.0)
    nu: float = 1.5e-5
    rho: float = 1.225
    turbulence_intensity: float = 0.01
    turbulence_length_scale: float = Field(gt=0.0)
    direction: FlowDirection = FlowDirection.PLUS_X


class GroundConfig(BaseModel):
    motion: GroundMotion


class CornerDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class WheelRotation(str, Enum):
    """How tyre and rim surfaces are driven.

    SPINNING is the physical answer and the default. LOCKED exists because a
    stationary wheel is a recognisable, well-documented modelling error rather
    than a silent one: if a coded boundary condition cannot be compiled in
    some environment, the fallback should be a choice on the record, not a
    surprise.
    """

    SPINNING = "spinning"
    LOCKED = "locked"


class PhysicsConfig(BaseModel):
    turbulence_model: Literal["kOmegaSST", "kOmegaSSTLM"] = "kOmegaSST"
    wall_treatment: WallTreatment
    mode: Mode = Mode.STRAIGHT
    yaw_deg: float = 0.0
    corner_radius: float | None = None
    # Which way the car turns. The corner centre is placed this far to the
    # named side of the vehicle, so the sign of the frame rotation - and with
    # it every wheel speed - follows from the flow direction rather than from
    # a hardcoded sign. See render/context.py::corner_frame().
    corner_direction: CornerDirection = CornerDirection.LEFT
    wheel_rotation: WheelRotation = WheelRotation.SPINNING


class RefinementRegion(BaseModel):
    """A box of extra volume refinement, sized in body lengths.

    Offsets are multiples of the geometry's own length and are measured from
    its bounding box, so a region tracks the model rather than the domain:
    x_start = -0.2 begins a fifth of a body length ahead of the nose, and
    x_end = 3.0 reaches three body lengths past the tail.

    Surface refinement only thickens the mesh next to the wall. Wakes,
    separations and vortices live in the volume, and without a region there
    they are resolved at the background cell size no matter how fine the
    surface is. That is the difference between a mesh that converges and one
    that converges to the wrong number.
    """

    name: str
    level: int = Field(ge=1)
    x_start: float = 0.0
    x_end: float
    half_width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)


class RefinementShell(BaseModel):
    """A shell of refinement at a distance from the vehicle's own surfaces.

    Where a RefinementRegion is a box you place, a shell is a distance you
    declare: snappy refines every cell within `distance` of the vehicle to at
    least `level`, so the refined volume is the shape of the car rather than
    the shape of a box someone drew round it.

    That difference is what makes it work on this case at all. A cornering
    wake follows the curve of the path and leaves any axis-aligned box, and
    the car is posed at a body-slip angle besides, so a box sized to contain
    the near field spends most of its cells in clean air. A shell has no
    orientation to get wrong, and it costs nothing to switch a case between
    the straight and cornering states.

    The other thing it buys is the flow *through* the car. Distance is
    measured to the nearest surface of any part, so the gaps between body,
    chassis, wishbones and rims are inside the innermost shell automatically -
    the internal airflow that surface refinement alone leaves at background
    size once you are a cell or two off the wall.

    `distance` is in body lengths, like every other length in DomainConfig.
    `level` is absolute, like RefinementRegion.level.
    """

    distance: float = Field(gt=0.0)
    level: int = Field(ge=1)


class DomainConfig(BaseModel):
    """Extents are in body lengths for both domain kinds.

    The annulus reads the same four numbers as the box and converts them to a
    sector: upstream/downstream become arc length along the path at the
    vehicle's own corner radius, half_width becomes radial half-extent, and
    height is unchanged. Declaring a cornering domain therefore needs no new
    numbers, and a case can be flipped between straight and cornering without
    its domain silently changing size.
    """

    kind: Literal["box", "annulus"] = "box"
    upstream_lengths: float = 5.0
    downstream_lengths: float = 10.0
    half_width_lengths: float = 3.0
    height_lengths: float = 3.0
    max_blockage: float = 0.01
    refinement_regions: list[RefinementRegion] = Field(default_factory=list)
    # Ordered coarse-to-fine or fine-to-coarse as you like; validate() checks
    # that they are consistent rather than trusting the order.
    refinement_shells: list[RefinementShell] = Field(default_factory=list)
    # Grid size, in metres, for simplifying the combined surface the shells
    # measure their distance from. 0 leaves it exactly as the CAD tessellated
    # it, which is the default because it changes what a case meshes to.
    #
    # This is the one surface in the case that can be simplified without
    # coarsening anything: it is never snapped to, never becomes a patch and
    # never enters force integration - it exists only to be measured from. The
    # walls snappy actually meshes against are untouched.
    #
    # WHAT IT BUYS. snappy answers a distance-mode region from an octree over
    # this surface's triangles, and rebuilds it on every rank each time the
    # mesh is redistributed mid-refinement. On the real car the combined
    # surface is 598,654 triangles; at 2 mm it is 146,598, and every part
    # keeps at least 97.9% of its area - the suspension, which is the thin-
    # feature risk, keeps exactly that.
    #
    # HOW TO SIZE IT. Two constraints, and the second is the one that bites.
    # The refinement boundary can shift by up to this distance, so it wants to
    # be small against the innermost shell (2 mm against 53 mm on the car).
    # And a feature thinner than the tolerance MAY collapse out of the surface
    # entirely, depending on where the grid falls across it - at 5 mm the
    # car's suspension loses 18.5% of its area to links disappearing, and a
    # link that is not in the distance field pulls no refinement around
    # itself. prepare measures the retained area per part and warns, because
    # that second constraint cannot be checked by arithmetic.
    shell_surface_tolerance: float = Field(default=0.0, ge=0.0)
    # Arc segments per 90 degrees of sector. blockMesh draws block edges as
    # circular arcs, but snappy's background cells are still hexes, so a
    # sector spanned by too few segments has visibly faceted radial walls.
    arc_segments_per_quadrant: int = Field(default=12, ge=2)


class MeshConfig(BaseModel):
    """n_layers is a *request*, not a promise.

    The prism stack has to fit inside the surface cell it is carved out of,
    and that cell is set by base_cell_size and surface_refinement_max - two
    values that arrive from a different config layer than the layer settings
    do. CaseSpec.n_layers_effective reconciles them; nothing else should.
    """

    base_cell_size: float = Field(gt=0.0)
    surface_refinement_min: int
    surface_refinement_max: int
    n_layers: int
    first_layer_thickness: float = Field(gt=0.0)
    expansion_ratio: float = Field(default=1.2, ge=1.0)
    min_layer_coverage: float = 0.7
    # Rendered straight into castellatedMeshControls. Costs no resolution
    # anywhere: it buys meshing *time* by deciding how often snappy
    # redistributes the mesh between ranks during refinement.
    #
    # 0.10 is the OpenFOAM tutorial value and a safe default. Lower it on a
    # case whose refinement is concentrated in a small part of the domain -
    # which is every external-aero case, and this one especially, where the
    # whole mesh is created inside a ball a few tenths of a metre across in a
    # 25 m domain. See the note in the template.
    max_load_unbalance: float = Field(default=0.10, ge=0.0)
    # These two are snappy's *construction* limits, rendered straight into
    # meshQualityControls. They are no longer the gate's acceptance criteria:
    # gating a 21 M-face mesh on its single worst face fails every production
    # mesh ever built, and tightening the number here only made snappy build
    # worse layers trying to satisfy it. What the gate judges is below.
    max_non_ortho: float = 70.0
    max_skewness: float = 4.0
    # Acceptance criteria, which ask how *much* of the mesh is affected rather
    # than how bad one face is.
    #
    # The average non-orthogonality is what governs the accuracy of the
    # non-orthogonal correction, and it is the number to watch: a real
    # production mesh runs under 10 while its worst face sits near 70.
    max_mean_non_ortho: float = 25.0
    # Fraction of all faces allowed to exceed checkMesh's own severe limits
    # (70 degrees non-orthogonality, skewness 4 internal / 20 boundary).
    # 0.1% of 21 M faces is 21,000 - far above the 58 and 27 a good mesh has,
    # and far below the tens of thousands a genuinely broken one has.
    max_bad_face_fraction: float = Field(default=1.0e-3, ge=0.0, le=1.0)
    # Layer iterations that run under the strict limits above before snappy
    # falls back to the `relaxed` sub-dictionary.
    #
    # 1, NOT 0, AND THE DIFFERENCE WAS MEASURED. snappyLayerDriver selects the
    # dictionary as `iteration < nRelaxedIter ? strict : relaxed`, so 0 puts
    # the relaxed limits in force for iteration 0 - the iteration that decides
    # which faces get extruded at all. That sounds like what you want, and on
    # this case it is not: the relaxed block sets minVol, minTetQuality and
    # minDeterminant to ~1e-30, and with those in force from the start snappy
    # successfully extrudes the full 40 um stack across the whole 18.8 m2 of
    # far-field track, where the background cell is 96 mm wide. That is an
    # aspect ratio of ~4800 (measured: 5547, and checkMesh counts 6,430 cells
    # over its limit) on layers that resolve nothing - the ground more than a
    # car length away carries still air over bare tarmac.
    #
    # At 1 the strict limits govern iteration 0, those slivers are refused,
    # and 96% of the ground keeps its unlayered 48 mm cell at aspect ratio 2,
    # which is the right answer for it. The 0.8% of ground under the car is
    # refined by the shells and is layered either way - 21 um against 24 um,
    # i.e. unchanged where it matters.
    n_relaxed_iter: int = Field(default=1, ge=0)
    # Fraction of the surface cell the whole prism stack may occupy. Also
    # rendered as snappy's own maxFaceThicknessRatio, so the pipeline's limit
    # and snappy's truncation threshold can never disagree.
    max_layer_cell_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    # A ceiling on every refinement level in the case, per-patch ones
    # included. None means no ceiling.
    #
    # This exists because per-patch levels deliberately *override* the
    # resolution profile, which is right for production - a 42 mm wing chord
    # needs its own level whatever the profile says - but leaves no way to
    # ask for a genuinely cheap mesh. Coarsening the profile alone does
    # nothing when four patches are pinned to level 6. A cap is the one knob
    # that makes a whole case cheap without editing the patch list, which is
    # what a smoke test of the plumbing needs.
    refinement_cap: int | None = Field(default=None, ge=0)
    # Whether checkMesh's own pass/fail verdict gates the run alongside the
    # thresholds above.
    #
    # checkMesh judges skewness and non-orthogonality against limits compiled
    # into it (4 and 70), which is a genuine second opinion and the default is
    # to respect it. But it also means max_skewness and max_non_ortho can only
    # ever *tighten* the gate: raising max_skewness to 12 leaves checkMesh
    # still failing the mesh at 4.6, and the config then describes something
    # other than what the code does - the exact "config says 300, code uses
    # 100" shape section 3.1 exists to prevent.
    #
    # Setting this False makes the spec's own numbers the sole authority for
    # those two quantities. It does *not* silence checkMesh generally: every
    # other failure it reports - negative volumes, non-closed cells,
    # zero-area faces, multiple regions - has no equivalent in the spec and
    # still gates.
    trust_check_mesh_verdict: bool = True


class ForcesConfig(BaseModel):
    """a_ref_full is ALWAYS the full-vehicle frontal area.

    The halved value is only ever obtained from CaseSpec.a_ref_effective.
    """

    a_ref_full: float = Field(gt=0.0)
    l_ref: float = Field(gt=0.0)
    c_of_r: tuple[float, float, float] = (0.0, 0.0, 0.0)


class SolveConfig(BaseModel):
    max_iterations: int = Field(gt=0)
    n_ranks: int = Field(gt=0)
    # How often the solver writes a full field set, in iterations.
    #
    # None means 'only at max_iterations', which is what this used to do
    # unconditionally - and it cost a 12-hour production solve its entire
    # flow field. simpleFoam writes on reaching endTime or on satisfying
    # residualControl; a run stopped any other way (a timeout, a full disk, a
    # walked-over ssh session) writes nothing at all, and 3,012 iterations of
    # a 5,000-iteration case left behind force histories, residuals, and not
    # one cell of U or p to look at.
    #
    # Set it on any run long enough that losing it would hurt. purgeWrite in
    # the template keeps only the most recent writes, so the cost is bounded
    # disk rather than growing disk, and the newest write doubles as a
    # restart point.
    write_interval: int | None = Field(default=None, gt=0)
    plateau_window: int = 200
    plateau_tol: float = 0.002
    residual_tol: float = 1.0e-4

    # --- pressure equation cost -------------------------------------------
    #
    # These three exist because the pressure solve is where a bandwidth-bound
    # case spends its wall clock, and the defaults are sized for correctness
    # on an arbitrary mesh rather than for throughput on this one. All three
    # default to the previous behaviour, so raising them is opt-in and the old
    # numbers stay reachable for a comparison run.

    # Extra pressure solves per SIMPLE iteration to recover the
    # non-orthogonal part of the Laplacian.
    #
    # It is not the cheap half. On the production car mesh the corrector solve
    # took FOUR GAMG cycles against the first solve's two, because relTol is
    # relative and bites harder on an already-reduced residual - so one
    # corrector is not a 2x on the pressure equation, it is nearer 3x.
    #
    # 0 is safe here only because snGradSchemes and laplacianSchemes both run
    # `limited corrected 0.33`, which damps the non-orthogonal correction to
    # at most a third of the orthogonal part. It is still a mesh-dependent
    # choice: watch `time step continuity errors` in the log, not just the
    # clock. On a mesh with severe non-orthogonality the error grows and the
    # iterations you saved get spent again on slower convergence.
    n_non_orth_correctors: int = Field(default=1, ge=0)

    # Relative tolerance on the pressure solve within one SIMPLE iteration.
    #
    # Loosening this trades inner cycles for outer iterations, which is a good
    # trade only when the run stops on convergence. With a fixed iteration
    # budget it is a bad one - the outer iterations are not there to be spent.
    p_rel_tol: float = Field(default=0.05, gt=0.0, lt=1.0)

    # GAMG's coarsest level, in cells.
    #
    # OpenFOAM's default is 10. On a 6.9 M cell mesh that agglomerates all the
    # way down to a handful of cells, and every extra coarse level costs a
    # global reduction across the ranks for a level that carries almost no
    # work. A coarsest level of ~1000 stops the ladder while the levels still
    # earn their keep and hands the remainder to the direct solve.
    #
    # Rendered explicitly at the OpenFOAM default rather than left out, for
    # the same reason maxLoadUnbalance is: the policy should be a property of
    # the case, not of whichever build happened to run it.
    gamg_coarsest_cells: int = Field(default=10, gt=0)


class PostConfig(BaseModel):
    yplus_min: float
    yplus_max: float
    max_fraction_outside: float = 0.1


class PatchSpec(BaseModel):
    """refinement_min/max override the case-wide levels for this patch.

    Refinement levels are relative to the background cell, so one level
    cannot suit surfaces of very different size: on the Ahmed body a level
    that resolves the 1044 mm body leaves a 30 mm stilt two cells across.
    Left as None, the patch follows MeshConfig.surface_refinement_min/max.
    """

    name: str
    role: PatchRole
    refinement_min: int | None = None
    refinement_max: int | None = None
    # Caps the prism stack on this patch. Still clamped by what fits the
    # cell, so this only ever asks for fewer layers, never more.
    n_layers: int | None = None
    # Which wheel this surface belongs to, e.g. "FL". Declared, never parsed
    # out of the patch name: a rotating wall that silently stopped rotating
    # because a part was renamed is not a failure anyone would notice.
    #
    # Every surface carrying a wheel id rotates with that wheel - tyre, rim,
    # hub, upright face. The axis, centre and radius are measured from the
    # geometry itself (geometry/wheels.py), so they follow the driving state
    # instead of being restated per case.
    wheel: str | None = None


class AhmedParams(BaseModel):
    length: float = 1.044
    width: float = 0.389
    height: float = 0.288
    slant_angle_deg: float = 35.0
    slant_length: float = 0.222
    nose_radius: float = 0.100
    ground_clearance: float = 0.050
    stilt_diameter: float = 0.030
    include_stilts: bool = True


class TessellationConfig(BaseModel):
    """Surface mesh sizing for STEP import, in metres.

    Not a file-format detail: too coarse and a curved surface becomes a
    faceted one that separates in the wrong place, too fine and the surface
    mesh outweighs the volume mesh built from it.

    curvature_segments is elements per full circle and is the control that
    matters most on this vehicle - it decides whether a 6 mm suspension link
    is a hexagon or a cylinder, independently of the part's size.
    """

    max_edge: float = Field(default=0.004, gt=0.0)
    min_edge: float = Field(default=0.0004, gt=0.0)
    curvature_segments: int = Field(default=24, ge=6)


class ContactPatchConfig(BaseModel):
    """How the tyres are joined to the road, in metres.

    A loaded tyre is drawn deflected into the road, so tread and ground meet
    tangentially and the wedge between them closes to zero angle. Left alone
    that wedge is filled with sliver cells that fail on skewness, refuse
    layers, and separate the flow in the wrong place.

    So the tyre is cut on a horizontal plane just above the road and the
    cross-section is extruded straight down through it: a vertical wall
    meeting the ground at ninety degrees, and a footprint of the right size.

    `cut_height` is the one number to think about. It is the height of the
    vertical step, so it wants to be something snappyHexMesh can resolve -
    a step well under one surface cell gets smeared back into a ramp and buys
    nothing. prepare() warns when it is smaller than the tyre's own surface
    cell.

    `depth_below_road` exists so the extrusion ends *past* z = 0 rather than
    on it. A face coplanar with the ground patch is its own class of snapping
    failure; running the wall below and letting snappy clip it keeps the
    intersection a clean edge. Nothing simulates the part below the road.
    """

    enabled: bool = True
    cut_height: float = Field(default=0.0005, gt=0.0)
    depth_below_road: float = Field(default=0.002, gt=0.0)


class MrfInterferenceConfig(BaseModel):
    """Keeping the MRF sleeve off the tyre's surface, in metres.

    A sleeve is drawn to the same nominal diameter as the tyre bore, so CAD
    puts the two surfaces in exactly the same place. For snappyHexMesh that is
    degenerate: a faceZone lying on a wall produces baffles, faceZones that
    come back "multiply connected", non-manifold points and intermittently a
    reversed face. Measured on this car at 0.05 mm over three quarters of the
    sleeve.

    So when the gap is under `min_clearance` the sleeve is pushed
    `interference` into the tyre - deliberately intersecting rather than
    touching, because where it is buried in tyre material there are no fluid
    cells and therefore no zone boundary at all. The cellZone ends up bounded
    by the tyre's own wall, which is what should have bounded it.

    Never the other way round. Shrinking the sleeve clear of the tyre would
    leave the zone short of the air it exists to rotate, and a near miss is as
    fragile as a hit.
    """

    enabled: bool = True
    # Gap below which sleeve and tyre count as the same surface. Generous
    # against the ~0 of a real coincidence and the millimetres of a sleeve
    # with genuine clearance, so it does not have to be precise.
    min_clearance: float = Field(default=0.001, gt=0.0)
    # How far into the tyre it is pushed. Wants to be a few surface cells so
    # snappy cannot resolve the two surfaces as one; validate() checks it
    # against the sleeve's own cell size.
    interference: float = Field(default=0.002, gt=0.0)


class GeometryConfig(BaseModel):
    """Where the surfaces come from.

    **The CAD is the truth, and the pipeline does not move it.** Yaw, pitch,
    roll, steering, camber and ride height are all set in CAD and arrive
    baked into the part positions, so there is no rotation, no translation
    and no ground snapping here. The only thing applied on import is `scale`,
    which is a unit conversion rather than a placement: OpenFOAM works in
    metres and the CAD is authored in millimetres.

    This is deliberate and worth keeping. Every transform the pipeline is
    allowed to apply is a place where the simulated car can differ from the
    drawn one, and the difference is invisible in the result.

    The one exception is `contact_patch`, and it is an exception rather than a
    hole in the rule: it changes the tyres near the road, where the CAD's own
    shape is a modelling artefact (a rigid tyre pushed through a rigid road)
    rather than something to reproduce. It is bounded to a fraction of a
    millimetre above z = 0, it happens *after* every measurement and check, so
    rolling radii and ride height still come from the CAD as drawn, and what
    it did is recorded per tyre in status/prepare.json.
    """

    kind: Literal["ahmed", "stl", "step"]
    ahmed: AhmedParams | None = None
    source_dir: str | None = None
    scale: float = Field(default=1.0, gt=0.0)
    tessellation: TessellationConfig = Field(default_factory=TessellationConfig)
    contact_patch: ContactPatchConfig = Field(default_factory=ContactPatchConfig)
    mrf_interference: MrfInterferenceConfig = Field(
        default_factory=MrfInterferenceConfig
    )
    # A declared property of the CAD, not of the flow.
    #
    # half_model used to be derived from flow symmetry alone, which forced a
    # symmetry plane onto any straight zero-yaw case - correct for the Ahmed
    # body, and a silent halving of a_ref for an asymmetric vehicle. Defaults
    # to False because that is the direction that cannot corrupt a result: a
    # full model of symmetric geometry merely costs cells, whereas a half
    # model of asymmetric geometry reports coefficients that are wrong and
    # plausible at the same time.
    symmetric: bool = False
    patches: list[PatchSpec]

    def wheel_ids(self) -> list[str]:
        """Distinct wheel ids, in first-appearance order."""
        seen: list[str] = []
        for patch in self.patches:
            if patch.wheel is not None and patch.wheel not in seen:
                seen.append(patch.wheel)
        return seen


class CaseSpec(BaseModel):
    """Fully resolved case. Every value explicit; no downstream defaults."""

    name: str
    # Which driving state was selected. Provenance only: the state's contents
    # have already been merged into the fields below, so nothing downstream
    # reads this. It is here so a result can be traced to a state by name
    # without diffing the whole spec.
    driving_state: str | None = None
    flow: FlowConfig
    ground: GroundConfig
    physics: PhysicsConfig
    domain: DomainConfig
    mesh: MeshConfig
    forces: ForcesConfig
    solve: SolveConfig
    post: PostConfig
    geometry: GeometryConfig

    @property
    def half_model(self) -> bool:
        """Symmetry is derived, never set.

        Three independent things must all hold, and every one of them has
        been got wrong somewhere in the literature:

        - the *geometry* is symmetric about y = 0 (declared per case);
        - the *mode* is straight, so the path does not curve;
        - the *yaw* is zero, because sideslip breaks symmetry on its own.

        The geometry term is the one this pipeline originally missed. Without
        it a straight zero-yaw case is forced to be a half model whatever it
        is a model *of*, and a_ref_effective is halved for a car that has no
        symmetry plane.
        """
        return (
            self.geometry.symmetric
            and self.physics.mode is Mode.STRAIGHT
            and self.physics.yaw_deg == 0.0
        )

    @property
    def a_ref_effective(self) -> float:
        if self.half_model:
            return self.forces.a_ref_full / 2.0
        return self.forces.a_ref_full

    def _capped(self, level: int) -> int:
        cap = self.mesh.refinement_cap
        return level if cap is None else min(level, cap)

    def patch_refinement(self, patch: PatchSpec) -> tuple[int, int]:
        """This patch's (min, max) refinement levels, case-wide unless set.

        The one place refinement is resolved, so the one place the cap is
        applied. Anything reading raw config levels instead would let a
        capped case still mesh a patch at level 6.
        """
        return (
            self._capped(
                self.mesh.surface_refinement_min
                if patch.refinement_min is None
                else patch.refinement_min
            ),
            self._capped(
                self.mesh.surface_refinement_max
                if patch.refinement_max is None
                else patch.refinement_max
            ),
        )

    def surface_cell_size_for(self, patch: PatchSpec) -> float:
        return self.mesh.base_cell_size / 2 ** self.patch_refinement(patch)[1]

    def layer_budget_for(self, patch: PatchSpec) -> float:
        return self.mesh.max_layer_cell_ratio * self.surface_cell_size_for(patch)

    def n_layers_in_cell(self, cell_size: float) -> int:
        """Layers that fit in a cell of the given size, never more than asked."""
        return self._layers_within(self.mesh.max_layer_cell_ratio * cell_size)

    def n_layers_for(self, patch: PatchSpec) -> int:
        """Layers that fit against *this patch's* cell size.

        Note the direction of the trade: refining a patch buys geometric
        resolution but shrinks the layer budget, because the stack has to fit
        inside a smaller cell. A finely-refined small part therefore carries
        fewer, thinner-relative layers - and asking for the case-wide count
        anyway is what makes snappy truncate.
        """
        return self._layers_within(self.layer_budget_for(patch))

    @property
    def surface_cell_size(self) -> float:
        """Edge length of a cell at the case-wide finest refinement level.

        snappy refinement levels are relative to the background cell, so this
        is the only meaningful measure of near-wall cell size - and the thing
        the prism stack has to fit inside. Patches carrying their own levels
        use surface_cell_size_for().
        """
        return self.mesh.base_cell_size / 2 ** self._capped(
            self.mesh.surface_refinement_max
        )

    def layer_stack_thickness(self, n_layers: int) -> float:
        """Total height of a geometric prism stack of n layers."""
        t_1 = self.mesh.first_layer_thickness
        ratio = self.mesh.expansion_ratio
        if n_layers <= 0:
            return 0.0
        if ratio == 1.0:
            return t_1 * n_layers
        return t_1 * (ratio**n_layers - 1.0) / (ratio - 1.0)

    @property
    def layer_budget(self) -> float:
        """How much of the surface cell the prism stack may occupy."""
        return self.mesh.max_layer_cell_ratio * self.surface_cell_size

    @property
    def n_layers_effective(self) -> int:
        """Layers that actually fit, never more than requested.

        snappyHexMesh carves the prism stack out of the surface cell and caps
        its height at maxFaceThicknessRatio of the local face. Asking for more
        layers than fit does not buy near-wall resolution: snappy squeezes the
        stack, then drops layers on exactly the curved and thin regions where
        the near-wall resolution mattered. Requesting a number that fits is
        strictly better than requesting one that does not.

        Zero means the first layer alone exceeds the budget - the refinement
        and the wall treatment disagree, and validate() rejects the case.
        """
        return self._layers_within(self.layer_budget)

    def _layers_within(self, budget: float) -> int:
        t_1 = self.mesh.first_layer_thickness
        ratio = self.mesh.expansion_ratio

        if t_1 > budget:
            return 0
        if ratio == 1.0:
            fits = int(budget // t_1)
        else:
            fits = int(
                math.floor(math.log1p(budget * (ratio - 1.0) / t_1) / math.log(ratio))
            )
        return max(0, min(self.mesh.n_layers, fits))

    @property
    def omega_rotation(self) -> float | None:
        """Magnitude of the frame rotation, |omega| = U / R."""
        if self.physics.mode is Mode.CORNERING and self.physics.corner_radius:
            return self.flow.u_inf / self.physics.corner_radius
        return None

    @property
    def flow_sign(self) -> float:
        """+1 if the freestream blows along +x, -1 if along -x.

        The single number every other direction is derived from. Nothing else
        may hard-code a streamwise sign.
        """
        return 1.0 if self.flow.direction is FlowDirection.PLUS_X else -1.0

    @property
    def freestream(self) -> tuple[float, float, float]:
        """Freestream velocity vector."""
        return (self.flow_sign * self.flow.u_inf, 0.0, 0.0)

    @property
    def drag_dir(self) -> tuple[float, float, float]:
        """Drag acts along the flow, so it follows the tunnel."""
        return (self.flow_sign, 0.0, 0.0)

    @property
    def pitch_axis(self) -> tuple[float, float, float]:
        """Completes a right-handed (drag, side, lift) set.

        side = lift x drag, so reversing the tunnel reverses it too. Leaving
        it at +y while the drag axis flips silently changes the sign of every
        pitching moment - and aero balance is computed from that moment.
        """
        return (0.0, self.flow_sign, 0.0)

    @property
    def corner_side(self) -> float:
        """Sign of the y offset from the car to the corner centre.

        The car travels *against* the freestream, so its forward direction is
        -flow_sign in x, and with z up its left-hand side faces -flow_sign in
        y. A left-hand corner puts the centre of the turn on that side.
        """
        turn = -1.0 if self.physics.corner_direction is CornerDirection.LEFT else 1.0
        return self.flow_sign * turn

    @property
    def omega_signed(self) -> float | None:
        """Frame rotation about +z, sign included.

        Derived rather than asserted: with the centre at y = corner_side * R
        and the car moving at -flow_sign in x, the angular velocity works out
        as -corner_side * flow_sign * U / R. Getting this backwards mirrors
        the entire cornering case, which looks perfectly converged.
        """
        magnitude = self.omega_rotation
        if magnitude is None:
            return None
        return -self.corner_side * self.flow_sign * magnitude

    def patches_with_role(self, role: PatchRole) -> list[str]:
        return [p.name for p in self.geometry.patches if p.role is role]

    def spec_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
