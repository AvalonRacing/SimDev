"""Machine health and the two queue settings."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from fastapi import APIRouter, Form, Request

from simdev.ui.context import back, ctx, render

router = APIRouter()
MAX_PARALLEL = 8


@router.get("/system")
def system(request: Request, error: str | None = None, message: str | None = None):
    c = ctx(request)
    c.config.runs_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(c.config.runs_root)
    return render(
        request, "system.html",
        disk_free=disk.free / 1e9, disk_total=disk.total / 1e9,
        load=os.getloadavg(), cpus=os.cpu_count(),
        core_budget=c.queue.setting("core_budget"),
        max_parallel=c.queue.setting("max_parallel"),
        config=c.config, error=error, message=message,
    )


@router.post("/system/settings")
def settings(request: Request, core_budget: int = Form(...), max_parallel: int = Form(...)):
    cpus = os.cpu_count() or 1
    if not 1 <= core_budget <= cpus:
        return back("/system", error=f"core budget must be between 1 and {cpus}")
    if not 1 <= max_parallel <= MAX_PARALLEL:
        return back("/system", error=f"parallel runs must be between 1 and {MAX_PARALLEL}")
    queue = ctx(request).queue
    # A queued job wider than the budget would never start; refuse rather
    # than strand it.
    widest = max(queue.queued(), key=lambda job: job.n_ranks, default=None)
    if widest is not None and widest.n_ranks > core_budget:
        return back(
            "/system",
            error=f"core budget {core_budget} is below the {widest.n_ranks} cores the "
            f"queued run {widest.run_name} needs",
        )
    queue.set_setting("core_budget", core_budget)
    queue.set_setting("max_parallel", max_parallel)
    return back("/system", message="settings saved")


@router.post("/system/doctor")
def doctor(request: Request):
    # Out of process: doctor probes ParaView with a 300 s timeout of its own.
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from simdev.cli import main; raise SystemExit(main(['doctor']))"],
            capture_output=True, text=True, timeout=600,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = "simdev doctor did not finish within 10 minutes"
    return render(request, "_doctor.html", output=output)
