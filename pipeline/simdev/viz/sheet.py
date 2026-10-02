"""An index for 562 pictures, so they can be flipped through rather than found."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any, Mapping

_STYLE = """
body { font: 13px/1.4 system-ui, sans-serif; margin: 2rem; background: #fafafa; }
h1 { font-size: 1.2rem; margin-bottom: 0.2rem; }
.meta { color: #555; margin-bottom: 1.5rem; }
.warn { background: #fff4e5; border-left: 3px solid #e08a00; padding: 0.6rem 0.9rem;
        margin-bottom: 1.5rem; }
h2 { font-size: 1rem; margin-top: 2rem; border-bottom: 1px solid #ddd;
     padding-bottom: 0.3rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 0.8rem; }
figure { margin: 0; }
img { width: 100%; border: 1px solid #ddd; background: #fff; }
figcaption { font-size: 11px; color: #666; margin-top: 0.2rem; }
"""

# Fixed order for surface views: the canonical sense makes reading consistent.
SURFACE_ORDER = ["front", "rear", "left", "right", "top", "bottom", "iso"]


def write_contact_sheet(
    results_dir: Path, plan: Mapping[str, Any], record: Mapping[str, Any]
) -> Path:
    """One page per run, grouped by axis and field, positions in order.

    Slices are sorted by numeric offset (smallest to largest). Surfaces are
    sorted by the canonical view order (front, rear, left, right, top, bottom, iso).
    """
    results_dir = Path(results_dir)
    out = results_dir / "index.html"

    # Build groups: key -> [(image_path, label, sort_key), ...]
    # For slices, sort_key is the numeric offset; for surfaces, it's the position
    # in SURFACE_ORDER.
    groups: dict[str, list[tuple[str, str, float | int]]] = {}

    for entry in plan["slices"]:
        offset = entry.get("offset", 0.0)  # numeric offset
        for image in entry["images"]:
            # Named for the directory the pictures are actually in, so a
            # heading on the sheet and a folder on disk read the same.
            key = f"{image['field']}_{entry['axis']}"
            groups.setdefault(key, []).append(
                (image["out"], entry["name"], offset)
            )

    for entry in plan["surfaces"]:
        # Sort surfaces by position in SURFACE_ORDER
        surface_name = entry.get("name", "unknown")
        sort_key = SURFACE_ORDER.index(surface_name) if surface_name in SURFACE_ORDER else 999
        for image in entry["images"]:
            key = f"surface_{image['field']}"
            groups.setdefault(key, []).append(
                (image["out"], surface_name, sort_key)
            )

    # cp-over-x plots, one per car-y station, in y order like the y slices.
    for station in (plan.get("cp_lines") or {}).get("stations", []):
        groups.setdefault("cp_line_y", []).append(
            (station["out"], station["name"], station["offset"])
        )

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(str(record['run']))}</title>",
        f"<style>{_STYLE}</style>",
        f"<h1>{html.escape(str(record['run']))}</h1>",
        "<div class='meta'>"
        f"spec {html.escape(str(record['spec_hash'])[:8])} &middot; "
        f"views {html.escape(str(record['views_digest']))} &middot; "
        f"datum {record['datum']} &middot; "
        f"{record['images_written']} images &middot; "
        f"sampled in {record['sample_seconds']}s, rendered in "
        f"{record['render_seconds']}s</div>",
    ]

    if record.get("reasons"):
        parts.append("<div class='warn'>")
        for reason in record["reasons"]:
            parts.append(f"<div>{html.escape(str(reason))}</div>")
        parts.append("</div>")

    for key in sorted(groups):
        parts.append(f"<h2>{html.escape(key)}</h2><div class='grid'>")
        # Sort by the sort_key (offset for slices, position for surfaces)
        for path, label, _ in sorted(groups[key], key=lambda p: p[2]):
            relative = os.path.relpath(path, results_dir)
            parts.append(
                f"<figure><a href='{html.escape(relative)}'>"
                f"<img src='{html.escape(relative)}' loading='lazy'></a>"
                f"<figcaption>{html.escape(label)}</figcaption></figure>"
            )
        parts.append("</div>")

    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out
