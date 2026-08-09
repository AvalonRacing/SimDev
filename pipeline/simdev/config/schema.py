from __future__ import annotations

import hashlib
import json
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
    base_cell_size: float = Field(gt=0.0)
    surface_refinement_min: int
    surface_refinement_max: int
    n_layers: int
    first_layer_thickness: float = Field(gt=0.0)
    expansion_ratio: float = 1.2
    min_layer_coverage: float = 0.7
    max_non_ortho: float = 70.0
    max_skewness: float = 4.0


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
    name: str
    role: PatchRole


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
