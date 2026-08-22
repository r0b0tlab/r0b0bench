from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LaneResult(BaseModel):
    lane_id: str
    status: str  # PASS | FAIL | SKIP | ERROR | NOT_IMPLEMENTED
    summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    infra_errors: int = 0
    imported: bool = False
    elapsed_s: float | None = None


def wilson_ci(k: int, n: int, z: float = 1.96) -> dict[str, float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    margin = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"low": max(0.0, centre - margin), "high": min(1.0, centre + margin)}


def load_profile(name: str) -> dict[str, Any]:
    key = name.replace("_", "-")
    allowed = ("core", "core-subset", "systems", "hard-subset")
    if key not in allowed:
        raise ValueError(f"unknown profile {name!r}; allowed: {', '.join(allowed)}")
    fname = {
        "core": "core.yaml",
        "core-subset": "core_subset.yaml",
        "systems": "systems.yaml",
        "hard-subset": "hard_subset.yaml",
    }[key]
    path = Path(__file__).resolve().parent / "profiles" / fname
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    systems = yaml.safe_load(
        (Path(__file__).resolve().parent / "profiles" / "_systems_block.yaml").read_text(encoding="utf-8")
    )
    data["systems"] = systems
    data["profile"] = key
    return data


def ensure_outside_checkout(output: Path, checkout_root: Path | None = None) -> Path:
    out = output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = (checkout_root or Path(__file__).resolve().parents[2]).resolve()
    try:
        out.relative_to(root)
    except ValueError:
        return out
    # allow if explicitly under out/ which is gitignored — still warn via marker
    if out == root or root in out.parents:
        # Permit only dedicated out dirs that are gitignored
        if out.name in {"out", "r0b0bench-out"} or "r0b0bench-out" in out.parts or out.name.startswith("out"):
            return out
        raise RuntimeError(
            f"output root {out} is inside source checkout {root}. "
            "Set --output to a path outside the repo (e.g. /tmp/r0b0bench-out)."
        )
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
