"""Starting the server, on the tailnet and nowhere else."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

EVERYWHERE = {"0.0.0.0", "::", ""}
SNAP_TAILSCALE = Path("/snap/bin/tailscale")


def resolve_host(
    host: str | None,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    if host is not None:
        if host.strip() in EVERYWHERE:
            raise ValueError(
                "refusing to listen on every interface: the UI has no logins, so "
                "anyone who can reach the port can start and delete runs. Use the "
                "Tailscale address (the default) or --host 127.0.0.1"
            )
        return host.strip()

    # The snap's bin directory is not on a systemd user service's PATH.
    exe = which("tailscale") or (str(SNAP_TAILSCALE) if SNAP_TAILSCALE.exists() else None)
    if exe is None:
        raise ValueError(
            "tailscale not found; install it (docs/ui-setup.md) or pass "
            "--host 127.0.0.1 for local use"
        )
    result = run([exe, "ip", "-4"], capture_output=True, text=True, timeout=10)
    lines = result.stdout.strip().splitlines() if result.returncode == 0 else []
    if not lines:
        raise ValueError(
            "could not read the Tailscale address: "
            + (result.stderr.strip() or "is tailscale up?")
        )
    return lines[0].strip()


def serve(
    host: str | None,
    port: int,
    runs_root: Path,
    cad_root: Path | None,
    case_path: Path | None,
    db_path: Path,
) -> int:
    import uvicorn

    from simdev.cad.checks import DEFAULT_CASE
    from simdev.cad.library import default_cad_root
    from simdev.ui.app import create_app
    from simdev.ui.context import UIConfig

    address = resolve_host(host)
    runs_root = Path(runs_root).expanduser()
    runs_root.mkdir(parents=True, exist_ok=True)
    config = UIConfig(
        runs_root=runs_root,
        cad_root=Path(cad_root or default_cad_root()).expanduser().resolve(),
        case_path=Path(case_path or DEFAULT_CASE).expanduser().resolve(),
        db_path=Path(db_path).expanduser(),
    )
    print(f"SimDev UI on http://{address}:{port}", flush=True)
    uvicorn.run(create_app(config), host=address, port=port, log_level="info")
    return 0
