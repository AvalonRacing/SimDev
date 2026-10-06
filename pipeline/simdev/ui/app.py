"""The SimDev web UI: one process, a worker thread, server-rendered pages."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.templating import Jinja2Templates

from simdev.cad.checks import default_library
from simdev.cad.library import Library
from simdev.ui import routes_cad, routes_compare, routes_queue, routes_results, routes_runs, routes_system
from simdev.ui.context import Context, UIConfig
from simdev.ui.queue import Job, Queue
from simdev.ui.worker import Worker, default_command

HERE = Path(__file__).parent
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
DEFAULT_PORTS = {"http": 80, "https": 443}


def _endpoint(netloc: str, scheme: str) -> tuple[str, int | None] | None:
    """(host, port) of a netloc, with the scheme's default port filled in."""
    try:
        parts = urlsplit(f"//{netloc}")
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname:
        return None
    return parts.hostname.lower(), port or DEFAULT_PORTS.get(scheme)


def cross_site(request: Request) -> bool:
    """Whether a request was sent by a page from another site.

    There are no logins, so any page open in a browser on the tailnet could
    otherwise post a form here and delete runs. Browsers send Origin (or at
    least Referer) with such requests; a request carrying neither did not come
    from a web page (curl, the test client) and is allowed.
    """
    source = request.headers.get("origin") or request.headers.get("referer")
    if source is None:
        return False
    sent_from = urlsplit(source)
    origin = _endpoint(sent_from.netloc, sent_from.scheme)
    target = _endpoint(request.headers.get("host", ""), request.url.scheme)
    return origin is None or origin != target


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


def number(value: float | None, fmt: str = "{:.4f}") -> str:
    return "" if value is None else fmt.format(value)


def create_app(
    config: UIConfig,
    library: Library | None = None,
    worker_command: Callable[[Job], list[str]] | None = None,
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    """allowed_hosts: the Host names the server answers to (None: any). The
    server sets it so a DNS-rebinding page cannot reach the UI under a name
    of its own; the tests leave it unset."""
    queue = Queue(config.db_path)
    library = library or default_library(config.cad_root, config.case_path)
    worker = Worker(queue, command=worker_command or default_command, cad_root=config.cad_root)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters.update(duration=duration, overrides=overrides_text, clock=clock, num=number)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.start_worker:
            worker.start()
        yield
        worker.stop()

    app = FastAPI(title="SimDev", lifespan=lifespan)
    app.state.ctx = Context(config, queue, library, worker, templates)

    @app.middleware("http")
    async def same_origin_only(request: Request, call_next):
        if request.method not in SAFE_METHODS and cross_site(request):
            return PlainTextResponse("cross-site request refused", status_code=403)
        return await call_next(request)

    if allowed_hosts is not None:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    for router in (routes_queue.router, routes_runs.router, routes_results.router, routes_compare.router, routes_cad.router, routes_system.router):
        app.include_router(router)
    return app
