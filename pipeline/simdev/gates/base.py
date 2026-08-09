from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, float | str] = field(default_factory=dict)
