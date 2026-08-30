from __future__ import annotations

import pytest

from simdev.render.context import hierarchical_split


def test_split_always_uses_every_rank() -> None:
    """A product short of numberOfSubdomains makes decomposePar abort, and a
    product over it silently leaves ranks idle."""
    for n in (1, 2, 4, 8, 12, 16, 40, 48, 64, 80):
        nx, ny, nz = hierarchical_split(n, (10.0, 4.0, 2.0))
        assert nx * ny * nz == n, n


def test_split_favours_the_long_axis() -> None:
    """Splitting a long thin domain across its short axis makes tall narrow
    blocks and a large interface area, which is processor-boundary traffic on
    a bandwidth-bound solver."""
    nx, ny, nz = hierarchical_split(8, (100.0, 10.0, 1.0))
    assert nx > ny >= nz


def test_split_follows_whichever_axis_is_long() -> None:
    nx, ny, nz = hierarchical_split(8, (1.0, 10.0, 100.0))
    assert nz > ny >= nx


def test_split_of_a_cube_is_balanced() -> None:
    assert sorted(hierarchical_split(8, (1.0, 1.0, 1.0))) == [2, 2, 2]


def test_a_prime_rank_count_still_resolves() -> None:
    """7 ranks cannot be balanced; it must still produce a usable split
    rather than raising or silently dropping a factor."""
    assert hierarchical_split(7, (10.0, 4.0, 2.0)) == (7, 1, 1)


def test_split_is_deterministic() -> None:
    """The whole point. Two identical calls, and by extension two identical
    runs, must partition the mesh the same way."""
    a = [hierarchical_split(40, (12.0, 9.0, 1.3)) for _ in range(5)]
    assert len(set(a)) == 1
