"""Resumable, series-level downloader for the public NBIA/TCIA API."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .layout import series_paths
from .manifest import SeriesSpec
from .safe_zip import safe_extract_zip, verify_zip

DEFAULT_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def image_url(series_instance_uid: str, base_url: str = DEFAULT_BASE_URL) -> str:
    query = urllib.parse.urlencode({"SeriesInstanceUID": series_instance_uid})
    return f"{base_url.rstrip('/')}/getImage?{query}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _download_once(url: str, partial: Path, timeout: int) -> tuple[int, bool]:
    """Download once. Returns (bytes on disk, resumed).

    A partial file is appended only when the server answers HTTP 206. If a Range
    request is ignored and HTTP 200 is returned, the file is safely restarted.
    """
    partial.parent.mkdir(parents=True, exist_ok=True)
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "HighPitchCT-Pipeline/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        resumed = bool(existing and status == 206)
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    return partial.stat().st_size, resumed


def download_series(
    spec: SeriesSpec,
    data_root: str | Path,
    *,
    retries: int = 5,
    timeout: int = 120,
    base_url: str = DEFAULT_BASE_URL,
    extract: bool = True,
) -> dict:
    """Download, validate, atomically finalize, and optionally extract one series."""
    paths = series_paths(data_root, spec)
    url = image_url(spec.series_instance_uid, base_url)

    if paths.archive.exists():
        verify_zip(paths.archive)
        status = "already_present"
        attempts = 0
        resumed = False
    else:
        last_error: Exception | None = None
        resumed = False
        for attempts in range(1, retries + 1):
            try:
                _, resumed_now = _download_once(url, paths.partial, timeout)
                resumed = resumed or resumed_now
                verify_zip(paths.partial)
                paths.archive.parent.mkdir(parents=True, exist_ok=True)
                os.replace(paths.partial, paths.archive)
                last_error = None
                break
            except (OSError, ValueError, urllib.error.URLError) as error:
                last_error = error
                if attempts < retries:
                    time.sleep(min(2 ** (attempts - 1), 30))
        if last_error is not None:
            payload = {
                "status": "failed",
                "series": asdict(spec),
                "url": url,
                "attempts": attempts,
                "error": f"{type(last_error).__name__}: {last_error}",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(paths.state, payload)
            raise RuntimeError(payload["error"]) from last_error
        status = "downloaded"

    extracted_files = None
    if extract:
        if paths.extracted.exists() and any(paths.extracted.rglob("*")):
            extracted_files = sum(1 for p in paths.extracted.rglob("*") if p.is_file())
        else:
            extracted_files = safe_extract_zip(paths.archive, paths.extracted)

    payload = {
        "status": status,
        "series": asdict(spec),
        "url": url,
        "attempts": attempts,
        "resumed": resumed,
        "archive": str(paths.archive),
        "archive_bytes": paths.archive.stat().st_size,
        "archive_sha256": sha256_file(paths.archive),
        "extracted_to": str(paths.extracted) if extract else None,
        "extracted_files": extracted_files,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(paths.state, payload)
    return payload
