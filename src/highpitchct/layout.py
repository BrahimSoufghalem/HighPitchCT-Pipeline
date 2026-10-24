"""Deterministic local paths for downloaded and indexed data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .manifest import SeriesSpec

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_component(value: str) -> str:
    cleaned = _SAFE.sub("_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError("Path component becomes empty after sanitization")
    return cleaned


@dataclass(frozen=True)
class SeriesPaths:
    archive: Path
    partial: Path
    extracted: Path
    state: Path


def series_paths(
    data_root: str | Path,
    spec: SeriesSpec,
    *,
    incoming_root: str | Path | None = None,
) -> SeriesPaths:
    root = Path(data_root)
    base_incoming = Path(incoming_root) if incoming_root is not None else (root / "incoming")
    case_id = safe_component(spec.case_id)
    role = safe_component(spec.role)
    uid = safe_component(spec.series_instance_uid)
    incoming = base_incoming / case_id / role
    return SeriesPaths(
        archive=incoming / f"{uid}.zip",
        partial=incoming / f"{uid}.zip.part",
        extracted=root / "raw" / "tcia" / case_id / role / uid,
        state=root / "manifests" / "downloads" / f"{uid}.json",
    )
