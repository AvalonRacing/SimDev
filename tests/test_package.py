from __future__ import annotations


def test_package_exposes_version() -> None:
    import simdev

    assert isinstance(simdev.__version__, str)
    assert simdev.__version__.count(".") >= 1
