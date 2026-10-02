"""What every route needs, in one place, so the route modules stay thin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from simdev.cad.library import Library
from simdev.ui.queue import Queue
from simdev.ui.worker import Worker


@dataclass(frozen=True)
class UIConfig:
    runs_root: Path
    cad_root: Path
    case_path: Path
    db_path: Path
    start_worker: bool = True


@dataclass
class Context:
    config: UIConfig
    queue: Queue
    library: Library
    worker: Worker
    templates: Jinja2Templates


def ctx(request: Request) -> Context:
    return request.app.state.ctx


def render(request: Request, name: str, /, status_code: int = 200, **values: Any):
    return ctx(request).templates.TemplateResponse(
        request, name, values, status_code=status_code
    )


def back(url: str, **params: Any) -> RedirectResponse:
    """Redirect after a POST, carrying a one-line message or error along."""
    query = urlencode({k: v for k, v in params.items() if v is not None})
    return RedirectResponse(f"{url}?{query}" if query else url, status_code=303)
