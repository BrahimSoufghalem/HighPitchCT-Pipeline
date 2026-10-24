"""Strict input manifest parsing and serialization for TCIA series downloads."""

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
    "reconstruction_low",
    "unknown",
}


@dataclass(frozen=True)
class SeriesSpec:
    case_id: str
    series_instance_uid: str
    role: str
    description: str = ""
    expected_files: int | None = None
    expected_bytes: int | None = None
    body_part: str = ""


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _optional_positive_int(raw: str | None, field_name: str, line_number: int) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        val = int(raw.strip())
    except ValueError as err:
        raise ValueError(f"Line {line_number}: invalid integer for {field_name}: {raw!r}") from err
    if val < 0:
        raise ValueError(f"Line {line_number}: {field_name} cannot be negative")
    return val


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
            expected_files = _optional_positive_int(
                row.get("expected_files"), "expected_files", line_number
            )
            expected_bytes = _optional_positive_int(
                row.get("expected_bytes"), "expected_bytes", line_number
            )
            body_part = (row.get("body_part") or "").strip()
            items.append(
                SeriesSpec(
                    case_id=case_id,
                    series_instance_uid=uid,
                    role=role,
                    description=row["description"].strip(),
                    expected_files=expected_files,
                    expected_bytes=expected_bytes,
                    body_part=body_part,
                )
            )
    return items


def write_series_manifest(
    path: str | Path,
    specs: list[SeriesSpec],
    *,
    enabled: bool = True,
) -> Path:
    """Write a validated, reproducible CSV manifest."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "series_instance_uid",
        "role",
        "body_part",
        "expected_files",
        "expected_bytes",
        "description",
        "enabled",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "case_id": spec.case_id,
                    "series_instance_uid": spec.series_instance_uid,
                    "role": spec.role,
                    "body_part": spec.body_part,
                    "expected_files": "" if spec.expected_files is None else spec.expected_files,
                    "expected_bytes": "" if spec.expected_bytes is None else spec.expected_bytes,
                    "description": spec.description,
                    "enabled": "true" if enabled else "false",
                }
            )
    return out
