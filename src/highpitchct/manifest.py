"""Strict input manifest parsing for TCIA series downloads."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

_UID_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_ALLOWED_ROLES = {
    "projection_full",
    "projection_low",
    "reconstruction_published",
    "unknown",
}


@dataclass(frozen=True)
class SeriesSpec:
    case_id: str
    series_instance_uid: str
    role: str
    description: str = ""


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def read_series_manifest(path: str | Path) -> list[SeriesSpec]:
    """Read, validate, and return enabled series entries.

    Duplicate SeriesInstanceUID values are rejected to prevent two labels from
    silently referring to the same download.
    """
    path = Path(path)
    required = {"case_id", "series_instance_uid", "role", "description", "enabled"}
    items: list[SeriesSpec] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            if not _as_bool(row["enabled"]):
                continue
            case_id = row["case_id"].strip()
            uid = row["series_instance_uid"].strip()
            role = row["role"].strip()
            if not case_id:
                raise ValueError(f"Line {line_number}: case_id is empty")
            if not _UID_RE.fullmatch(uid) or len(uid) > 64:
                raise ValueError(f"Line {line_number}: invalid DICOM SeriesInstanceUID")
            if role not in _ALLOWED_ROLES:
                raise ValueError(
                    f"Line {line_number}: role {role!r} is not one of {sorted(_ALLOWED_ROLES)}"
                )
            if uid in seen:
                raise ValueError(f"Line {line_number}: duplicate SeriesInstanceUID {uid}")
            seen.add(uid)
            items.append(SeriesSpec(case_id, uid, role, row["description"].strip()))
    return items
