"""ZIP validation, uncompressed byte accounting, and safe extraction."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from .runtime import DiskBudgetExceededError, DiskBudgetGuard


def verify_zip(path: str | Path) -> tuple[int, int]:
    """Verify ZIP CRC integrity and return (file_member_count, uncompressed_bytes)."""
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        members = [m for m in archive.infolist() if not m.is_dir()]
        uncompressed_bytes = sum(int(m.file_size) for m in members)
        return len(members), uncompressed_bytes


def safe_extract_zip(
    path: str | Path,
    destination: str | Path,
    *,
    budget_guard: DiskBudgetGuard | None = None,
) -> int:
    """Safely extract ZIP preventing path traversal and enforcing disk quota beforehand."""
    destination = Path(destination).resolve()
    extracted = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        uncompressed_total = sum(int(m.file_size) for m in members if not m.is_dir())
        if budget_guard is not None and not budget_guard.check_within_budget(uncompressed_total):
            snap = budget_guard.snapshot()
            raise DiskBudgetExceededError(
                f"Extracting {uncompressed_total} bytes would exceed disk quota "
                f"(on_disk={snap['on_disk_bytes']}, max={snap['max_disk_bytes']})"
            )

        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
            mode = member.external_attr >> 16
            if mode:
                os.chmod(target, mode & 0o777)
            extracted += 1
    return extracted
