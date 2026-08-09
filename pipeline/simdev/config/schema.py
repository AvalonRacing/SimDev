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


class FlowConfig(BaseModel):
    u_inf: float = Field(gt=0.0)
    nu: float = 1.5e-5
    rho: float = 1.225
    turbulence_intensity: float = 0.01
    turbulence_length_scale: float = Field(gt=0.0)


class GroundConfig(BaseModel):
    motion: GroundMotion


class PhysicsConfig(BaseModel):
    turbulence_model: Literal["kOmegaSST", "kOmegaSSTLM"] = "kOmegaSST"
    wall_treatment: WallTreatment
    mode: Mode = Mode.STRAIGHT
    yaw_deg: float = 0.0
    corner_radius: float | None = None


class DomainConfig(BaseModel):
    kind: Literal["box", "annulus"] = "box"
    upstream_lengths: float = 5.0
    downstream_lengths: float = 10.0
    half_width_lengths: float = 3.0
    height_lengths: float = 3.0
    max_blockage: float = 0.01


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
    max_non_ortho: float = 70.0
    max_skewness: float = 4.0
    # Fraction of the surface cell the whole prism stack may occupy. Also
    # rendered as snappy's own maxFaceThicknessRatio, so the pipeline's limit
    # and snappy's truncation threshold can never disagree.
    max_layer_cell_ratio: float = Field(default=0.5, gt=0.0, le=1.0)


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
    plateau_window: int = 200
    plateau_tol: float = 0.002
    residual_tol: float = 1.0e-4


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


class GeometryConfig(BaseModel):
    kind: Literal["ahmed", "stl"]
    ahmed: AhmedParams | None = None
    stl_dir: str | None = None
    patches: list[PatchSpec]


class CaseSpec(BaseModel):
    """Fully resolved case. Every value explicit; no downstream defaults."""

    name: str
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
        """Symmetry is derived, never set. Yaw alone breaks it."""
        return self.physics.mode is Mode.STRAIGHT and self.physics.yaw_deg == 0.0

    @property
    def a_ref_effective(self) -> float:
        if self.half_model:
            return self.forces.a_ref_full / 2.0
        return self.forces.a_ref_full

    def patch_refinement(self, patch: PatchSpec) -> tuple[int, int]:
        """This patch's (min, max) refinement levels, case-wide unless set."""
        return (
            self.mesh.surface_refinement_min
            if patch.refinement_min is None
            else patch.refinement_min,
            self.mesh.surface_refinement_max
            if patch.refinement_max is None
            else patch.refinement_max,
        )

    def surface_cell_size_for(self, patch: PatchSpec) -> float:
        return self.mesh.base_cell_size / 2 ** self.patch_refinement(patch)[1]

    def layer_budget_for(self, patch: PatchSpec) -> float:
        return self.mesh.max_layer_cell_ratio * self.surface_cell_size_for(patch)

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
        return self.mesh.base_cell_size / 2**self.mesh.surface_refinement_max

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
        if self.physics.mode is Mode.CORNERING and self.physics.corner_radius:
            return self.flow.u_inf / self.physics.corner_radius
        return None

    def patches_with_role(self, role: PatchRole) -> list[str]:
        return [p.name for p in self.geometry.patches if p.role is role]

    def spec_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
