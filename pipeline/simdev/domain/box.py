from __future__ import annotations

from typing import Sequence

from simdev.config.schema import CaseSpec
from simdev.domain.base import DomainBox
from simdev.geometry.roles import PatchRole


def _only(spec: CaseSpec, role: PatchRole) -> str:
    names = spec.patches_with_role(role)
    if len(names) != 1:
        raise ValueError(
            f"box domain needs exactly one '{role.value}' patch, found {names}"
        )
    return names[0]


class BoxDomainBuilder:
    """Rectangular virtual wind tunnel. Ground plane at z = 0."""

    def build(
        self,
        spec: CaseSpec,
        geom_bounds: tuple[Sequence[float], Sequence[float]],
    ) -> DomainBox:
        lo, hi = geom_bounds
        length = hi[0] - lo[0]
        d = spec.domain

        # Upstream and downstream are relative to the flow, not to +x. With
        # the tunnel reversed the long tail of the domain has to be on the
        # other side of the car, or the wake runs straight out of the inlet.
        if spec.flow_sign > 0:
            x_min = lo[0] - d.upstream_lengths * length
            x_max = hi[0] + d.downstream_lengths * length
        else:
            x_min = lo[0] - d.downstream_lengths * length
            x_max = hi[0] + d.upstream_lengths * length
        z_min = 0.0
        z_max = d.height_lengths * length
        y_max = d.half_width_lengths * length
        y_min = 0.0 if spec.half_model else -y_max

        base = spec.mesh.base_cell_size
        n_cells = (
            max(1, round((x_max - x_min) / base)),
            max(1, round((y_max - y_min) / base)),
            max(1, round((z_max - z_min) / base)),
        )

        symmetry = _only(spec, PatchRole.SYMMETRY) if spec.half_model else None

        return DomainBox(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            n_cells=n_cells,
            inlet=_only(spec, PatchRole.INLET),
            outlet=_only(spec, PatchRole.OUTLET),
            ground=_only(spec, PatchRole.GROUND),
            farfield=_only(spec, PatchRole.FARFIELD),
            symmetry=symmetry,
            flow_sign=spec.flow_sign,
            geom_min=(lo[0], lo[1], lo[2]),
            geom_max=(hi[0], hi[1], hi[2]),
        )


def blockage_ratio(frontal_area: float, domain: DomainBox) -> float:
    """Frontal area over domain cross-section.

    Pass the *true projected* area, not a_ref: a_ref is a reference
    convention that may deliberately exclude parts.
    """
    return frontal_area / domain.cross_section_area


def check_blockage(
    frontal_area: float, domain: DomainBox, max_blockage: float
) -> list[str]:
    ratio = blockage_ratio(frontal_area, domain)
    if ratio > max_blockage:
        return [
            f"blockage ratio {ratio:.3%} exceeds the {max_blockage:.3%} limit; "
            "the domain is too cramped and will contaminate the pressure field"
        ]
    return []
