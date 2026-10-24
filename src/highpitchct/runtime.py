"""Runtime environment detection, smart Kaggle profile, and disk quota guard."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
KAGGLE_DEFAULT_MAX_DISK_GB = 18.0
ALLOWED_PROFILES = {"auto", "kaggle", "full"}


class DiskBudgetExceededError(RuntimeError):
    """Raised when writing additional data would exceed the configured disk budget."""


@dataclass(frozen=True)
class EnvironmentProfile:
    """Resolved execution profile controlling storage paths and disk limits."""

    name: str
    is_kaggle_env: bool
    data_root: Path
    incoming_root: Path
    max_disk_bytes: int | None
    cleanup_archive_after_extract: bool

    @property
    def max_disk_gb(self) -> float | None:
        if self.max_disk_bytes is None:
            return None
        return round(self.max_disk_bytes / GIB, 3)


def is_kaggle() -> bool:
    """Return True when running inside a Kaggle kernel/notebook environment."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
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


def directory_used_bytes(path: str | Path) -> int:
    """Compute total regular-file bytes inside a directory tree."""
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for entry in root.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def resolve_environment_profile(
    *,
    profile: str | None = None,
    data_root: str | Path | None = None,
    incoming_root: str | Path | None = None,
    max_disk_gb: float | None = None,
    cleanup_zip: bool | None = None,
) -> EnvironmentProfile:
    """Resolve smart Kaggle vs Full execution rules explicitly (never silently)."""
    raw_profile = (profile or os.environ.get("HPCT_ENV_PROFILE") or "auto").strip().lower()
    if raw_profile not in ALLOWED_PROFILES:
        raise ValueError(
            f"Invalid profile {raw_profile!r}; must be one of {sorted(ALLOWED_PROFILES)}"
        )

    kaggle_detected = is_kaggle()
    effective_name = ("kaggle" if kaggle_detected else "full") if raw_profile == "auto" else raw_profile

    if data_root is not None:
        resolved_data = ensure_writable_output(data_root)
    elif os.environ.get("HPCT_DATA_ROOT"):
        resolved_data = ensure_writable_output(os.environ["HPCT_DATA_ROOT"])
    elif effective_name == "kaggle" and Path("/kaggle/working").is_dir():
        resolved_data = ensure_writable_output("/kaggle/working/highpitchct-data")
    else:
        resolved_data = ensure_writable_output("data")

    env_incoming = os.environ.get("HPCT_INCOMING_ROOT")
    if incoming_root is not None:
        resolved_incoming = ensure_writable_output(incoming_root)
    elif env_incoming:
        resolved_incoming = ensure_writable_output(env_incoming)
    elif effective_name == "kaggle":
        # Stage ZIP archives in /tmp so they do not consume /kaggle/working's 20 GB limit
        resolved_incoming = ensure_writable_output("/tmp/highpitchct-incoming")
    else:
        resolved_incoming = ensure_writable_output(resolved_data / "incoming")

    env_max_gb = os.environ.get("HPCT_MAX_DISK_GB")
    if max_disk_gb is not None:
        effective_gb: float | None = float(max_disk_gb)
    elif env_max_gb is not None and env_max_gb.strip() != "":
        effective_gb = float(env_max_gb)
    elif effective_name == "kaggle":
        effective_gb = KAGGLE_DEFAULT_MAX_DISK_GB
    else:
        effective_gb = None

    if effective_gb is not None and effective_gb <= 0:
        raise ValueError(f"max_disk_gb must be positive, got {effective_gb}")

    max_bytes = int(effective_gb * GIB) if effective_gb is not None else None
    effective_cleanup = (
        cleanup_zip if cleanup_zip is not None else (effective_name == "kaggle")
    )

    return EnvironmentProfile(
        name=effective_name,
        is_kaggle_env=kaggle_detected,
        data_root=resolved_data,
        incoming_root=resolved_incoming,
        max_disk_bytes=max_bytes,
        cleanup_archive_after_extract=effective_cleanup,
    )


class DiskBudgetGuard:
    """Thread-safe disk usage enforcer for data_root (e.g. <= 18 GiB on Kaggle)."""

    def __init__(self, data_root: str | Path, max_disk_bytes: int | None):
        self.data_root = Path(data_root)
        self.max_disk_bytes = max_disk_bytes
        self._reserved_bytes = 0
        self._lock = threading.Lock()

    def snapshot(self) -> dict[str, int | float | None]:
        with self._lock:
            on_disk = directory_used_bytes(self.data_root)
            committed = on_disk + self._reserved_bytes
            remaining = (
                max(0, self.max_disk_bytes - committed)
                if self.max_disk_bytes is not None
                else None
            )
            return {
                "on_disk_bytes": on_disk,
                "reserved_bytes": self._reserved_bytes,
                "committed_bytes": committed,
                "max_disk_bytes": self.max_disk_bytes,
                "remaining_bytes": remaining,
            }

    def try_reserve(self, estimated_bytes: int) -> tuple[bool, dict[str, int | float | None]]:
        """Atomically reserve bytes if within max_disk_bytes and filesystem free space."""
        needed = max(0, int(estimated_bytes))
        with self._lock:
            on_disk = directory_used_bytes(self.data_root)
            committed = on_disk + self._reserved_bytes
            fs_free = shutil.disk_usage(self.data_root).free
            info = {
                "on_disk_bytes": on_disk,
                "reserved_bytes": self._reserved_bytes,
                "needed_bytes": needed,
                "max_disk_bytes": self.max_disk_bytes,
                "filesystem_free_bytes": fs_free,
            }
            if needed > 0 and needed > fs_free:
                return False, info
            if self.max_disk_bytes is not None and (committed + needed) > self.max_disk_bytes:
                return False, info
            self._reserved_bytes += needed
            return True, info

    def release(self, reserved_bytes: int) -> None:
        with self._lock:
            self._reserved_bytes = max(0, self._reserved_bytes - max(0, int(reserved_bytes)))

    def check_within_budget(self, extra_bytes: int = 0) -> bool:
        if self.max_disk_bytes is None:
            return True
        with self._lock:
            on_disk = directory_used_bytes(self.data_root)
            return (on_disk + max(0, int(extra_bytes))) <= self.max_disk_bytes


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


def runtime_report(
    data_root: str | Path | None = None,
    *,
    profile: str | None = None,
    max_disk_gb: float | None = None,
) -> dict:
    resolved = resolve_environment_profile(
        profile=profile,
        data_root=data_root,
        max_disk_gb=max_disk_gb,
    )
    disk = shutil.disk_usage(resolved.data_root)
    used_bytes = directory_used_bytes(resolved.data_root)
    return {
        "kaggle": resolved.is_kaggle_env,
        "profile": resolved.name,
        "data_root": str(resolved.data_root),
        "incoming_root": str(resolved.incoming_root),
        "data_root_used_bytes": used_bytes,
        "data_root_used_gb": round(used_bytes / GIB, 3),
        "max_disk_bytes": resolved.max_disk_bytes,
        "max_disk_gb": resolved.max_disk_gb,
        "cleanup_archive_after_extract": resolved.cleanup_archive_after_extract,
        "cpu_count": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpus": _gpus(),
        "warnings": [
            "GPU memories are separate; two T4 GPUs do not form one 32 GB single-GPU memory pool.",
            (
                f"Kaggle smart quota active: downloads into data_root stop cleanly at <= {resolved.max_disk_gb} GiB."
                if resolved.max_disk_bytes is not None
                else "Full profile active: no artificial disk quota cap (downloads full manifest)."
            ),
        ],
    }


def print_runtime_report(
    data_root: str | Path | None = None,
    *,
    profile: str | None = None,
    max_disk_gb: float | None = None,
) -> None:
    print(json.dumps(runtime_report(data_root, profile=profile, max_disk_gb=max_disk_gb), indent=2))
