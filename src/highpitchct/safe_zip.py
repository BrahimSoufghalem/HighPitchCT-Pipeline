"""ZIP validation and extraction with zip-slip protection."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path


def verify_zip(path: str | Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt ZIP member: {bad}")


def safe_extract_zip(path: str | Path, destination: str | Path) -> int:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
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
