"""Runtime and storage detection for local machines and Kaggle notebooks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def is_kaggle() -> bool:
    """Return True only when the standard Kaggle filesystem is present."""
    return Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir()


def default_data_root() -> Path:
    """Choose a writable data root without ever writing into /kaggle/input."""
    configured = os.environ.get("HPCT_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    if is_kaggle():
        return Path("/kaggle/working/highpitchct-data")
    return Path("data")


def ensure_writable_output(path: str | Path) -> Path:
    """Reject Kaggle's read-only input mount and verify that output is writable."""
    path = Path(path).expanduser().resolve()
    kaggle_input = Path("/kaggle/input")
    if path == kaggle_input or kaggle_input in path.parents:
        raise ValueError(
            "/kaggle/input is read-only. Use /kaggle/working or set HPCT_DATA_ROOT."
        )
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".hpct-write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)
    return path


def _ram_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def _gpus() -> list[dict[str, str | int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    result = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if len(parts) != 3:
            continue
        index, name, memory_mib = parts
        try:
            result.append({"index": int(index), "name": name, "memory_mib": int(memory_mib)})
        except ValueError:
            continue
    return result


def runtime_report(data_root: str | Path | None = None) -> dict:
    root = ensure_writable_output(data_root or default_data_root())
    disk = shutil.disk_usage(root)
    return {
        "kaggle": is_kaggle(),
        "data_root": str(root),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpus": _gpus(),
        "warnings": [
            "GPU memories are separate; two T4 GPUs do not form one 30 GB memory pool."
        ],
    }


def print_runtime_report(data_root: str | Path | None = None) -> None:
    print(json.dumps(runtime_report(data_root), indent=2))
