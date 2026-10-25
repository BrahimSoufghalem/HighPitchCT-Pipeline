"""Fast single-pass CRC-verified atomic ZIP extraction with Zip-Slip and quota guards."""

from __future__ import annotations

import os
import shutil
import threading
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from .runtime import DiskBudgetExceededError, DiskBudgetGuard


def _validate_rel_member(raw_name: str) -> PurePosixPath:
    norm = raw_name.replace("\\", "/")
    if not norm or norm.startswith("/"):
        raise ValueError(f"Unsafe ZIP member path: {raw_name}")
    rel = PurePosixPath(norm)
    parts = rel.parts
    if not parts:
        raise ValueError(f"Unsafe ZIP member path: {raw_name}")
    if ":" in parts[0] or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe ZIP member path: {raw_name}")
    return rel


def inspect_zip_headers(path: str | Path) -> tuple[int, int]:
    """Read ZIP central directory without decompressing payloads; return (file_count, uncompressed_bytes)."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            uncompressed_bytes = sum(int(m.file_size) for m in members)
            return len(members), uncompressed_bytes
    except zipfile.BadZipFile as error:
        raise ValueError(f"Corrupt ZIP archive: {error}") from error


def verify_zip(path: str | Path) -> tuple[int, int]:
    """Verify full ZIP CRC integrity and return (file_member_count, uncompressed_bytes)."""
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt ZIP member: {bad}")
            members = [m for m in archive.infolist() if not m.is_dir()]
            uncompressed_bytes = sum(int(m.file_size) for m in members)
            return len(members), uncompressed_bytes
    except zipfile.BadZipFile as error:
        raise ValueError(f"Corrupt ZIP archive: {error}") from error


def safe_extract_zip_with_stats(
    path: str | Path,
    destination: str | Path,
    *,
    budget_guard: DiskBudgetGuard | None = None,
    prior_reserved_bytes: int = 0,
) -> tuple[int, int, int]:
    """Single-pass CRC-verified atomic extraction.

    Returns (extracted_files, zip_member_count, uncompressed_bytes).
    Validates Zip-Slip and disk quota upfront from central directory headers, extracts into
    a transient sibling directory while Python's ZipExtFile verifies every member's CRC-32,
    and atomically swaps into `destination` on success.
    """
    destination = Path(destination).resolve()
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)

    try:
        archive_ctx = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as error:
        raise ValueError(f"Corrupt ZIP archive: {error}") from error

    with archive_ctx as archive:
        members = archive.infolist()
        validated_files: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        unique_rel_dirs: set[PurePosixPath] = set()
        uncompressed_total = 0

        for member in members:
            rel = _validate_rel_member(member.filename)
            if member.is_dir():
                unique_rel_dirs.add(rel)
            else:
                validated_files.append((member, rel))
                uncompressed_total += int(member.file_size)
                if str(rel.parent) != ".":
                    unique_rel_dirs.add(rel.parent)

        # Secondary resolved-path check on unique directories only (O(dirs) instead of O(files))
        for rel_dir in unique_rel_dirs:
            resolved_dir = (destination / Path(*rel_dir.parts)).resolve()
            if resolved_dir != destination and destination not in resolved_dir.parents:
                raise ValueError(f"Unsafe ZIP member path: {rel_dir}")

        active_reservation = 0
        if budget_guard is not None:
            fits, snap = budget_guard.try_update_reservation(
                prior_reserved_bytes, uncompressed_total
            )
            if not fits:
                raise DiskBudgetExceededError(
                    f"Extracting {uncompressed_total} bytes would exceed disk quota "
                    f"(on_disk={snap['on_disk_bytes']}, reserved={snap['reserved_bytes']}, "
                    f"max={snap['max_disk_bytes']})"
                )
            active_reservation = uncompressed_total

        staging_dir = (
            destination_parent
            / f".{destination.name}.extracting_{os.getpid()}_{threading.get_ident()}"
        )
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
            for rel_dir in unique_rel_dirs:
                (staging_dir / Path(*rel_dir.parts)).mkdir(parents=True, exist_ok=True)

            extracted = 0
            for member, rel in validated_files:
                target = staging_dir / Path(*rel.parts)
                with archive.open(member, "r") as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
                mode = (member.external_attr >> 16) & 0o777
                # Only issue chmod syscall when executable bits are explicitly present
                if mode & 0o111:
                    os.chmod(target, mode)
                extracted += 1

            if destination.exists():
                shutil.rmtree(destination)
            os.replace(staging_dir, destination)
            return extracted, len(validated_files), uncompressed_total
        except (zipfile.BadZipFile, zlib.error, EOFError) as error:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ValueError(f"Corrupt ZIP member during CRC-verified extraction: {error}") from error
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        finally:
            if budget_guard is not None and active_reservation > 0:
                budget_guard.release(active_reservation)


def safe_extract_zip(
    path: str | Path,
    destination: str | Path,
    *,
    budget_guard: DiskBudgetGuard | None = None,
) -> int:
    """Safely extract ZIP preventing path traversal and enforcing disk quota beforehand."""
    extracted, _, _ = safe_extract_zip_with_stats(
        path, destination, budget_guard=budget_guard, prior_reserved_bytes=0
    )
    return extracted
