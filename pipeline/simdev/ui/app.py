"""The SimDev web UI: one process, a worker thread, server-rendered pages."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from simdev.cad.checks import default_library
from simdev.cad.library import Library
from simdev.ui import routes_queue, routes_runs
from simdev.ui.context import Context, UIConfig
from simdev.ui.queue import Job, Queue
from simdev.ui.worker import Worker, default_command

HERE = Path(__file__).parent


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    seconds = int(seconds)
    hours, minutes = divmod(seconds // 60, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if seconds >= 60:
        return f"{seconds // 60} min"
    return f"{seconds} s"


def overrides_text(values: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {v}" for k, v in sorted(values.items())) or "–"


def clock(timestamp: float | None) -> str:
    if not timestamp:
        return "–"
    return datetime.fromtimestamp(timestamp).strftime("%d.%m. %H:%M")


def create_app(
    config: UIConfig,
    library: Library | None = None,
    worker_command: Callable[[Job], list[str]] | None = None,
) -> FastAPI:
    queue = Queue(config.db_path)
    library = library or default_library(config.cad_root, config.case_path)
    worker = Worker(queue, command=worker_command or default_command, cad_root=config.cad_root)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters.update(duration=duration, overrides=overrides_text, clock=clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.start_worker:
            worker.start()
        yield
        worker.stop()

    app = FastAPI(title="SimDev", lifespan=lifespan)
    app.state.ctx = Context(config, queue, library, worker, templates)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    for router in (routes_queue.router, routes_runs.router):
        app.include_router(router)
    return app
