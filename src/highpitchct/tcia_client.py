"""High-throughput resumable, quota-aware TCIA NBIA downloader with live tqdm progress."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .layout import safe_component, series_paths
from .manifest import SeriesSpec
from .progress import ProgressReporter
from .runtime import DiskBudgetExceededError, DiskBudgetGuard, EnvironmentProfile
from .safe_zip import inspect_zip_headers, safe_extract_zip_with_stats, verify_zip

DEFAULT_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DEFAULT_COLLECTION = "LDCT-and-Projection-data"
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB streaming chunks for high throughput
DEFAULT_WORKERS = 4

_REPORT_FIELDS = [
    "case_id",
    "role",
    "body_part",
    "series_instance_uid",
    "status",
    "expected_mb",
    "zip_downloaded_mb",
    "uncompressed_mb",
    "extracted_files",
    "elapsed_s",
    "avg_speed_mbps",
    "resumed",
    "attempts",
]

_thread_local = threading.local()


def make_http_session(*, pool_maxsize: int = 8, retries: int = 4) -> requests.Session:
    """Create a connection-pooled requests.Session with automatic retry on transient 5xx/429."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=2,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    size = max(4, int(pool_maxsize))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=size, pool_maxsize=size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "HighPitchCT-Pipeline/0.3.1"})
    return session


def get_http_session() -> requests.Session:
    """Return a thread-local connection-pooled requests.Session."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = make_http_session()
        _thread_local.session = session
    return session


@contextmanager
def _open_http_stream(
    url: str,
    headers: dict[str, str],
    timeout: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
) -> Iterator[tuple[int, dict[str, str], Iterator[bytes]]]:
    """Open a streaming HTTP GET via pooled requests.Session and yield (status, headers, chunks)."""
    session = get_http_session()
    connect_timeout = min(20, max(5, int(timeout)))
    with session.get(
        url,
        headers=headers,
        stream=True,
        timeout=(connect_timeout, timeout),
    ) as resp:
        if resp.status_code not in {200, 206}:
            resp.raise_for_status()
        norm_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        yield resp.status_code, norm_headers, resp.iter_content(chunk_size=chunk_size)


def image_url(series_instance_uid: str, base_url: str = DEFAULT_BASE_URL) -> str:
    query = urllib.parse.urlencode({"SeriesInstanceUID": series_instance_uid})
    return f"{base_url.rstrip('/')}/getImage?{query}"


def series_metadata_url(series_instance_uid: str, base_url: str = DEFAULT_BASE_URL) -> str:
    query = urllib.parse.urlencode({"SeriesInstanceUID": series_instance_uid})
    return f"{base_url.rstrip('/')}/getSeriesMetaData?{query}"


def collection_series_url(
    collection: str = DEFAULT_COLLECTION,
    body_part: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    params: dict[str, str] = {"Collection": collection}
    if body_part:
        params["BodyPartExamined"] = body_part.strip().upper()
    query = urllib.parse.urlencode(params)
    return f"{base_url.rstrip('/')}/getSeries?{query}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(DEFAULT_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _get_json(url: str, timeout: int = 60) -> list[dict]:
    session = get_http_session()
    resp = session.get(
        url,
        headers={"Accept": "application/json"},
        timeout=(min(15, max(5, timeout)), timeout),
    )
    resp.raise_for_status()
    raw = resp.text
    if not raw.strip():
        return []
    parsed = resp.json()
    return parsed if isinstance(parsed, list) else [parsed]


def fetch_series_metadata(
    series_instance_uid: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 60,
) -> dict | None:
    """Query NBIA /getSeriesMetaData to inspect expected bytes and file count before download."""
    url = series_metadata_url(series_instance_uid, base_url)
    try:
        items = _get_json(url, timeout=timeout)
    except (OSError, ValueError, requests.RequestException, urllib.error.URLError):
        return None
    return items[0] if items else None


def classify_tcia_series_role(series_description: str, modality: str = "CT") -> str:
    """Map TCIA LDCT-and-Projection-data SeriesDescription to explicit pipeline role."""
    desc = (series_description or "").strip().lower()
    if "projection" in desc:
        if "full" in desc:
            return "projection_full"
        if "low" in desc:
            return "projection_low"
        return "unknown"
    if "image" in desc:
        if "full" in desc:
            return "reconstruction_published"
        if "low" in desc:
            return "reconstruction_low"
    return "unknown"


def discover_tcia_series(
    *,
    collection: str = DEFAULT_COLLECTION,
    body_part: str | None = None,
    roles: set[str] | None = None,
    max_cases: int | None = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 120,
) -> list[SeriesSpec]:
    """Query NBIA public API and build deterministic SeriesSpec entries grouped by patient."""
    wanted_roles = roles or {"projection_full", "reconstruction_published"}
    url = collection_series_url(collection=collection, body_part=body_part, base_url=base_url)
    records = _get_json(url, timeout=timeout)

    by_patient: dict[str, list[SeriesSpec]] = {}
    for rec in records:
        uid = str(rec.get("SeriesInstanceUID") or "").strip()
        patient_id = str(rec.get("PatientID") or "").strip()
        if not uid or not patient_id:
            continue
        desc = str(rec.get("SeriesDescription") or "").strip()
        role = classify_tcia_series_role(desc, str(rec.get("Modality") or "CT"))
        if role not in wanted_roles:
            continue
        bp = str(rec.get("BodyPartExamined") or "").strip().upper()
        if body_part and bp and bp != body_part.strip().upper():
            continue

        raw_count = rec.get("ImageCount")
        raw_bytes = rec.get("FileSize") or rec.get("TotalSizeInBytes")
        exp_files = int(raw_count) if raw_count not in (None, "") else None
        exp_bytes = int(raw_bytes) if raw_bytes not in (None, "") else None

        case_id = f"case_{safe_component(patient_id)}"
        spec = SeriesSpec(
            case_id=case_id,
            series_instance_uid=uid,
            role=role,
            description=desc or f"{collection} {role}",
            expected_files=exp_files,
            expected_bytes=exp_bytes,
            body_part=bp,
        )
        by_patient.setdefault(case_id, []).append(spec)

    sorted_cases = sorted(by_patient.keys())
    if max_cases is not None:
        sorted_cases = sorted_cases[: max(0, int(max_cases))]

    role_order = {
        "projection_full": 0,
        "reconstruction_published": 1,
        "projection_low": 2,
        "reconstruction_low": 3,
        "unknown": 4,
    }
    output: list[SeriesSpec] = []
    for cid in sorted_cases:
        series_list = sorted(
            by_patient[cid], key=lambda s: (role_order.get(s.role, 99), s.series_instance_uid)
        )
        output.extend(series_list)
    return output


def _download_stream_to_part(
    url: str,
    partial: Path,
    timeout: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
    on_chunk: Callable[[int], None] | None = None,
    on_restart: Callable[[], None] | None = None,
    on_content_length: Callable[[int], None] | None = None,
) -> tuple[int, bool, str]:
    """Stream HTTP payload to `partial` while computing SHA-256 incrementally on-the-fly.

    Returns (bytes_on_disk, resumed, sha256_hex).
    Appends to an existing `.part` file ONLY when the server answers HTTP 206.
    If the server ignores `Range` and responds HTTP 200, safely restarts from byte 0.
    """
    partial.parent.mkdir(parents=True, exist_ok=True)
    existing = partial.stat().st_size if partial.exists() else 0
    headers: dict[str, str] = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    with _open_http_stream(url, headers, timeout, chunk_size=chunk_size) as (
        status,
        resp_headers,
        chunks,
    ):
        resumed = bool(existing and status == 206)
        digest = hashlib.sha256()

        if resumed:
            # Hash existing prefix once so final digest covers the entire resumed file
            with partial.open("rb") as prev:
                while prefix_chunk := prev.read(chunk_size):
                    digest.update(prefix_chunk)
            raw_cl = resp_headers.get("content-length")
            if on_content_length is not None and raw_cl and raw_cl.isdigit():
                on_content_length(existing + int(raw_cl))
            mode = "ab"
        else:
            if on_restart is not None:
                on_restart()
            raw_cl = resp_headers.get("content-length")
            if on_content_length is not None and raw_cl and raw_cl.isdigit():
                on_content_length(int(raw_cl))
            mode = "wb"

        written_now = 0
        with partial.open(mode) as output:
            for chunk in chunks:
                if not chunk:
                    continue
                output.write(chunk)
                digest.update(chunk)
                n = len(chunk)
                written_now += n
                if on_chunk is not None:
                    on_chunk(n)

    total_size = (existing + written_now) if resumed else written_now
    return total_size, resumed, digest.hexdigest()


def _download_once(url: str, partial: Path, timeout: int) -> tuple[int, bool]:
    """Backwards-compatible single-attempt downloader returning (bytes_on_disk, resumed)."""
    total_size, resumed, _ = _download_stream_to_part(url, partial, timeout)
    return total_size, resumed


def _count_extracted_files(directory: Path) -> tuple[int, int]:
    """Fast os.scandir count returning (total_regular_files, dcm_files)."""
    if not directory.exists():
        return 0, 0
    total_files = 0
    dcm_files = 0
    stack = [str(directory)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total_files += 1
                        if entry.name.lower().endswith(".dcm"):
                            dcm_files += 1
        except OSError:
            continue
    return total_files, dcm_files


def _existing_extracted_state(paths, spec: SeriesSpec) -> dict | None:
    """Return prior completed state if extracted files are already verified on disk."""
    if not paths.state.exists() or not paths.extracted.exists():
        return None
    try:
        prior = json.loads(paths.state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if prior.get("status") not in {"downloaded", "already_present", "already_extracted"}:
        return None
    on_disk_files, dcm_files = _count_extracted_files(paths.extracted)
    expected = spec.expected_files or prior.get("extracted_files")
    if on_disk_files > 0 and (
        expected is None or on_disk_files == int(expected) or dcm_files == int(expected)
    ):
        prior["status"] = "already_extracted"
        prior["extracted_files"] = on_disk_files
        prior["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        return prior
    return None


def estimate_series_bytes(
    spec: SeriesSpec,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 60,
) -> tuple[int | None, int | None]:
    """Return (expected_bytes, expected_files) from manifest or NBIA metadata endpoint."""
    if spec.expected_bytes is not None and spec.expected_files is not None:
        return spec.expected_bytes, spec.expected_files
    meta = fetch_series_metadata(spec.series_instance_uid, base_url=base_url, timeout=timeout)
    if not meta:
        return spec.expected_bytes, spec.expected_files
    raw_size = meta.get("File Size") or meta.get("FileSize") or meta.get("TotalSizeInBytes")
    raw_count = meta.get("Image Count") or meta.get("ImageCount")
    exp_bytes = (
        spec.expected_bytes
        if spec.expected_bytes is not None
        else (int(raw_size) if raw_size not in (None, "") else None)
    )
    exp_files = (
        spec.expected_files
        if spec.expected_files is not None
        else (int(raw_count) if raw_count not in (None, "") else None)
    )
    return exp_bytes, exp_files


def download_series(
    spec: SeriesSpec,
    data_root: str | Path,
    *,
    incoming_root: str | Path | None = None,
    retries: int = 5,
    timeout: int = 120,
    base_url: str = DEFAULT_BASE_URL,
    extract: bool = True,
    cleanup_archive: bool = False,
    budget_guard: DiskBudgetGuard | None = None,
    pre_reserved_bytes: int = 0,
    progress: ProgressReporter | None = None,
    progress_target_bytes: int = 0,
) -> dict:
    """Download, stream-hash, enforce quota, single-pass CRC-extract, and optionally clean ZIP."""
    paths = series_paths(data_root, spec, incoming_root=incoming_root)
    url = image_url(spec.series_instance_uid, base_url)
    label = f"{spec.case_id} | {spec.role}"
    t_start = time.monotonic()

    reserved_bytes = max(0, int(pre_reserved_bytes))
    credited_bar_bytes = 0
    bar_expected_bytes = max(0, int(progress_target_bytes))

    def _on_chunk(n_bytes: int) -> None:
        nonlocal credited_bar_bytes
        credited_bar_bytes += n_bytes
        if progress is not None:
            progress.update(n_bytes)

    def _on_restart() -> None:
        nonlocal credited_bar_bytes
        if credited_bar_bytes > 0:
            if progress is not None:
                progress.update(-credited_bar_bytes)
            credited_bar_bytes = 0

    def _on_content_length(actual_total: int) -> None:
        nonlocal bar_expected_bytes
        if progress is not None and bar_expected_bytes > 0 and actual_total > 0:
            delta = actual_total - bar_expected_bytes
            if delta != 0:
                progress.adjust_total(delta)
                bar_expected_bytes = actual_total

    try:
        if extract:
            existing_payload = _existing_extracted_state(paths, spec)
            if existing_payload is not None:
                _atomic_json(paths.state, existing_payload)
                if progress is not None:
                    progress.write(
                        f"⏭  [{label}] already extracted ({existing_payload.get('extracted_files')} files)"
                    )
                return existing_payload

        expected_bytes = spec.expected_bytes
        expected_files = spec.expected_files
        if (
            reserved_bytes == 0
            and budget_guard is not None
            and budget_guard.max_disk_bytes is not None
            and expected_bytes is None
        ):
            expected_bytes, expected_files = estimate_series_bytes(
                spec, base_url=base_url, timeout=min(timeout, 60)
            )

        if reserved_bytes == 0 and budget_guard is not None and expected_bytes is not None:
            fits, quota_info = budget_guard.try_reserve(expected_bytes)
            if not fits:
                payload = {
                    "status": "skipped_quota_reached",
                    "series": asdict(spec),
                    "url": url,
                    "reason": "Pre-download series size exceeds configured disk budget",
                    "quota": quota_info,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _atomic_json(paths.state, payload)
                if progress is not None:
                    progress.write(f"⏭  [{label}] skipped (disk quota reached)")
                return payload
            reserved_bytes = expected_bytes

        est_mb_str = (
            f"~{expected_bytes / (1024 * 1024):.1f} MiB est."
            if expected_bytes
            else "streaming ZIP"
        )

        archive_sha256 = ""
        extracted_files: int | None = None
        zip_members = 0
        uncompressed_bytes = 0

        if paths.archive.exists():
            status = "already_present"
            attempts = 0
            resumed = False
            archive_bytes = paths.archive.stat().st_size
            archive_sha256 = sha256_file(paths.archive)
            if progress is not None and bar_expected_bytes > 0:
                progress.adjust_total(-bar_expected_bytes)
                bar_expected_bytes = 0
            if extract:
                zip_members, uncompressed_bytes = inspect_zip_headers(paths.archive)
                if budget_guard is not None:
                    fits, snap = budget_guard.try_update_reservation(
                        reserved_bytes, uncompressed_bytes
                    )
                    if not fits:
                        if cleanup_archive:
                            paths.archive.unlink(missing_ok=True)
                        payload = {
                            "status": "skipped_quota_reached",
                            "series": asdict(spec),
                            "url": url,
                            "archive_bytes": archive_bytes,
                            "uncompressed_bytes": uncompressed_bytes,
                            "reason": (
                                f"Uncompressed series ({uncompressed_bytes} B) exceeds disk quota"
                            ),
                            "quota": snap,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        _atomic_json(paths.state, payload)
                        return payload
                    reserved_bytes = uncompressed_bytes
                if progress is not None:
                    progress.write(
                        f"📦  [{label}] extracting existing archive ({zip_members} files) ..."
                    )
                extracted_files, zip_members, uncompressed_bytes = safe_extract_zip_with_stats(
                    paths.archive, paths.extracted, budget_guard=None
                )
                if budget_guard is not None and reserved_bytes > 0:
                    budget_guard.release(reserved_bytes)
                    reserved_bytes = 0
            else:
                zip_members, uncompressed_bytes = verify_zip(paths.archive)
        else:
            if progress is not None:
                progress.write(f"⬇  [{label}] downloading ({est_mb_str}) ...")

            last_error: Exception | None = None
            resumed = False
            attempts = 0
            archive_bytes = 0

            for attempts in range(1, retries + 1):
                try:
                    existing_before = paths.partial.stat().st_size if paths.partial.exists() else 0
                    if existing_before > 0 and credited_bar_bytes == 0 and progress is not None:
                        # Credit existing .part bytes if HTTP 206 resumes
                        pass

                    archive_bytes, resumed_now, archive_sha256 = _download_stream_to_part(
                        url,
                        paths.partial,
                        timeout,
                        on_chunk=_on_chunk,
                        on_restart=_on_restart,
                        on_content_length=_on_content_length,
                    )
                    resumed = resumed or resumed_now
                    if archive_bytes < 22:
                        raise ValueError(f"Downloaded ZIP too small ({archive_bytes} bytes)")

                    # Reconcile chunked transfer stream length with progress bar total
                    if progress is not None and bar_expected_bytes > 0:
                        delta = credited_bar_bytes - bar_expected_bytes
                        if delta != 0:
                            progress.adjust_total(delta)
                            bar_expected_bytes = credited_bar_bytes

                    paths.archive.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(paths.partial, paths.archive)

                    if extract:
                        zip_members, uncompressed_bytes = inspect_zip_headers(paths.archive)
                        if budget_guard is not None:
                            fits, snap = budget_guard.try_update_reservation(
                                reserved_bytes, uncompressed_bytes
                            )
                            if not fits:
                                if cleanup_archive:
                                    paths.archive.unlink(missing_ok=True)
                                payload = {
                                    "status": "skipped_quota_reached",
                                    "series": asdict(spec),
                                    "url": url,
                                    "archive_bytes": archive_bytes,
                                    "uncompressed_bytes": uncompressed_bytes,
                                    "reason": (
                                        f"Extracting {uncompressed_bytes} bytes would exceed disk quota "
                                        f"(on_disk={snap['on_disk_bytes']}, max={snap['max_disk_bytes']})"
                                    ),
                                    "quota": snap,
                                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                }
                                _atomic_json(paths.state, payload)
                                if progress is not None:
                                    progress.write(f"⏭  [{label}] skipped before extract (quota reached)")
                                return payload
                            reserved_bytes = uncompressed_bytes

                        if progress is not None:
                            progress.write(
                                f"📦  [{label}] extracting & verifying CRC-32 ({zip_members} files) ..."
                            )
                        extracted_files, zip_members, uncompressed_bytes = (
                            safe_extract_zip_with_stats(
                                paths.archive, paths.extracted, budget_guard=None
                            )
                        )
                        if budget_guard is not None and reserved_bytes > 0:
                            budget_guard.release(reserved_bytes)
                            reserved_bytes = 0
                    else:
                        zip_members, uncompressed_bytes = verify_zip(paths.archive)

                    last_error = None
                    break
                except DiskBudgetExceededError as quota_err:
                    if paths.extracted.exists():
                        shutil.rmtree(paths.extracted, ignore_errors=True)
                    if cleanup_archive:
                        paths.archive.unlink(missing_ok=True)
                    payload = {
                        "status": "skipped_quota_reached",
                        "series": asdict(spec),
                        "url": url,
                        "archive_bytes": archive_bytes,
                        "uncompressed_bytes": uncompressed_bytes,
                        "reason": str(quota_err),
                        "quota": budget_guard.snapshot() if budget_guard else None,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    _atomic_json(paths.state, payload)
                    return payload
                except (
                    OSError,
                    ValueError,
                    requests.RequestException,
                    urllib.error.URLError,
                ) as error:
                    last_error = error
                    # If archive was corrupt after download, remove it so next retry starts cleanly
                    if isinstance(error, ValueError) and "Corrupt ZIP" in str(error):
                        paths.archive.unlink(missing_ok=True)
                        paths.partial.unlink(missing_ok=True)
                        _on_restart()
                    if progress is not None:
                        progress.write(
                            f"⚠  [{label}] attempt {attempts}/{retries} failed: {str(error)[:90]}"
                        )
                    if attempts < retries:
                        time.sleep(min(2 ** (attempts - 1), 20))

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
                if progress is not None:
                    progress.write(f"❌  [{label}] failed after {attempts} attempts")
                raise RuntimeError(payload["error"]) from last_error
            status = "downloaded"

        if extract and expected_files is not None and extracted_files != expected_files:
            _, dcm_count = _count_extracted_files(paths.extracted)
            if dcm_count != expected_files and extracted_files != expected_files:
                raise RuntimeError(
                    f"Extracted DICOM count ({dcm_count}) does not match expected_files ({expected_files})"
                )

        archive_cleaned_up = False
        if extract and cleanup_archive and paths.archive.exists():
            paths.archive.unlink(missing_ok=True)
            archive_cleaned_up = True

        elapsed_s = max(time.monotonic() - t_start, 1e-3)
        zip_mb = archive_bytes / (1024 * 1024)
        uncomp_mb = uncompressed_bytes / (1024 * 1024)
        speed_mbps = (zip_mb / elapsed_s) if status == "downloaded" else 0.0

        if progress is not None:
            if status == "downloaded":
                progress.write(
                    f"✅  [{label}] {zip_mb:.1f} MiB ZIP -> {uncomp_mb:.1f} MiB "
                    f"({extracted_files if extract else zip_members} files) "
                    f"in {elapsed_s:.1f}s @ {speed_mbps:.2f} MiB/s"
                )
            else:
                progress.write(
                    f"✅  [{label}] {status} ({uncomp_mb:.1f} MiB, "
                    f"{extracted_files if extract else zip_members} files)"
                )

        payload = {
            "status": status,
            "series": asdict(spec),
            "url": url,
            "attempts": attempts,
            "resumed": resumed,
            "archive": str(paths.archive),
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha256,
            "archive_cleaned_up": archive_cleaned_up,
            "zip_member_count": zip_members,
            "uncompressed_bytes": uncompressed_bytes,
            "extracted_to": str(paths.extracted) if extract else None,
            "extracted_files": extracted_files,
            "elapsed_seconds": round(elapsed_s, 3),
            "avg_speed_mbps": round(speed_mbps, 2),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(paths.state, payload)
        return payload
    finally:
        if budget_guard is not None and reserved_bytes > 0:
            budget_guard.release(reserved_bytes)


def _write_download_report_csv(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_REPORT_FIELDS)
        writer.writeheader()
        for r in results:
            s = r.get("series") or {}
            exp_b = s.get("expected_bytes")
            zip_b = r.get("archive_bytes")
            unc_b = r.get("uncompressed_bytes")
            writer.writerow(
                {
                    "case_id": s.get("case_id", ""),
                    "role": s.get("role", ""),
                    "body_part": s.get("body_part", ""),
                    "series_instance_uid": s.get("series_instance_uid", ""),
                    "status": r.get("status", ""),
                    "expected_mb": round(int(exp_b) / (1024 * 1024), 2) if exp_b else "",
                    "zip_downloaded_mb": round(int(zip_b) / (1024 * 1024), 2) if zip_b else "",
                    "uncompressed_mb": round(int(unc_b) / (1024 * 1024), 2) if unc_b else "",
                    "extracted_files": r.get("extracted_files", ""),
                    "elapsed_s": r.get("elapsed_seconds", ""),
                    "avg_speed_mbps": r.get("avg_speed_mbps", ""),
                    "resumed": r.get("resumed", ""),
                    "attempts": r.get("attempts", ""),
                }
            )
    os.replace(tmp, path)


def download_manifest_with_profile(
    specs: list[SeriesSpec],
    env_profile: EnvironmentProfile,
    *,
    workers: int = DEFAULT_WORKERS,
    retries: int = 5,
    timeout: int = 120,
    extract: bool = True,
    base_url: str = DEFAULT_BASE_URL,
    show_progress: bool = False,
) -> dict:
    """Download series concurrently across admitted patients with live tqdm progress & quota guard."""
    t_run_start = time.monotonic()
    budget_guard = DiskBudgetGuard(env_profile.data_root, env_profile.max_disk_bytes)

    # 1. Fast resume inspection so already-extracted series skip metadata & download queues
    already_done_map: dict[str, dict] = {}
    pending_specs: list[SeriesSpec] = []
    for spec in specs:
        paths = series_paths(
            env_profile.data_root, spec, incoming_root=env_profile.incoming_root
        )
        if extract:
            prior = _existing_extracted_state(paths, spec)
            if prior is not None:
                _atomic_json(paths.state, prior)
                already_done_map[spec.series_instance_uid] = prior
                continue
        pending_specs.append(spec)

    # 2. Parallel NBIA metadata pre-fetch for pending series missing expected_bytes
    enriched_by_uid: dict[str, SeriesSpec] = {s.series_instance_uid: s for s in specs}
    need_meta = [
        s
        for s in pending_specs
        if s.expected_bytes is None
        and (budget_guard.max_disk_bytes is not None or show_progress)
    ]
    if need_meta:
        meta_workers = max(1, min(8, len(need_meta)))
        with ThreadPoolExecutor(max_workers=meta_workers) as meta_pool:
            fut_map = {
                meta_pool.submit(
                    estimate_series_bytes, s, base_url=base_url, timeout=min(timeout, 45)
                ): s
                for s in need_meta
            }
            for fut in as_completed(fut_map):
                orig = fut_map[fut]
                try:
                    exp_b, exp_f = fut.result()
                except Exception:
                    exp_b, exp_f = orig.expected_bytes, orig.expected_files
                enriched_by_uid[orig.series_instance_uid] = SeriesSpec(
                    case_id=orig.case_id,
                    series_instance_uid=orig.series_instance_uid,
                    role=orig.role,
                    description=orig.description,
                    expected_files=exp_f,
                    expected_bytes=exp_b,
                    body_part=orig.body_part,
                )

    # 3. Group by case_id in manifest order and perform deterministic case-level quota admission
    cases: OrderedDict[str, list[SeriesSpec]] = OrderedDict()
    for orig in specs:
        cases.setdefault(orig.case_id, []).append(enriched_by_uid[orig.series_instance_uid])

    results_by_uid: dict[str, dict] = dict(already_done_map)
    admitted_jobs: list[tuple[SeriesSpec, int]] = []  # (spec, pre_reserved_bytes)
    quota_stopped = False

    for case_id, case_specs in cases.items():
        case_pending = [
            s for s in case_specs if s.series_instance_uid not in already_done_map
        ]
        if not case_pending:
            continue

        if quota_stopped:
            snap = budget_guard.snapshot()
            for spec in case_pending:
                paths = series_paths(
                    env_profile.data_root, spec, incoming_root=env_profile.incoming_root
                )
                skipped = {
                    "status": "skipped_quota_reached",
                    "series": asdict(spec),
                    "reason": f"Disk quota already reached prior to case {case_id}",
                    "quota": snap,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _atomic_json(paths.state, skipped)
                results_by_uid[spec.series_instance_uid] = skipped
            continue

        if budget_guard.max_disk_bytes is not None:
            known_case_bytes = sum(s.expected_bytes or 0 for s in case_pending)
            if known_case_bytes > 0:
                fits, snap = budget_guard.try_reserve(known_case_bytes)
                if not fits:
                    quota_stopped = True
                    for spec in case_pending:
                        paths = series_paths(
                            env_profile.data_root, spec, incoming_root=env_profile.incoming_root
                        )
                        skipped = {
                            "status": "skipped_quota_reached",
                            "series": asdict(spec),
                            "case_pending_bytes": known_case_bytes,
                            "reason": (
                                f"Case {case_id} requires ~{known_case_bytes} bytes which exceeds "
                                f"remaining budget in {env_profile.name} profile"
                            ),
                            "quota": snap,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        _atomic_json(paths.state, skipped)
                        results_by_uid[spec.series_instance_uid] = skipped
                    continue
                for spec in case_pending:
                    admitted_jobs.append((spec, int(spec.expected_bytes or 0)))
                continue

        for spec in case_pending:
            admitted_jobs.append((spec, 0))

    # 4. Execute all admitted series concurrently across one shared worker pool with live tqdm bar
    est_total_bytes = sum(int(spec.expected_bytes or 0) for spec, _ in admitted_jobs)
    quota_label = (
        f"<= {env_profile.max_disk_gb:.1f} GiB"
        if env_profile.max_disk_gb is not None
        else "Unlimited (Full)"
    )

    if show_progress:
        sys.stderr.write(
            f"📊 Pre-flight [{env_profile.name.upper()} | quota: {quota_label} | workers: {workers}]\n"
            f"   ✅ Already complete : {len(already_done_map)} series\n"
            f"   ⏭  Skipped (quota)  : {len(results_by_uid) - len(already_done_map)} series\n"
            f"   ⬇  Queued download  : {len(admitted_jobs)} series "
            f"(~{est_total_bytes / (1024 * 1024):.1f} MiB est.)\n"
        )
        sys.stderr.flush()

    failed_count = 0
    if admitted_jobs:
        effective_workers = max(1, min(int(workers), len(admitted_jobs)))
        with ProgressReporter(
            total=est_total_bytes if est_total_bytes > 0 else None,
            desc="📥 Downloading TCIA Series",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            colour="cyan",
            enabled=show_progress,
        ) as bar:
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(
                        download_series,
                        spec,
                        env_profile.data_root,
                        incoming_root=env_profile.incoming_root,
                        retries=retries,
                        timeout=timeout,
                        base_url=base_url,
                        extract=extract,
                        cleanup_archive=env_profile.cleanup_archive_after_extract,
                        budget_guard=budget_guard,
                        pre_reserved_bytes=pre_res,
                        progress=bar,
                        progress_target_bytes=int(spec.expected_bytes or 0),
                    ): (spec, pre_res)
                    for spec, pre_res in admitted_jobs
                }
                for fut in as_completed(futures):
                    spec, _ = futures[fut]
                    try:
                        res = fut.result()
                        results_by_uid[spec.series_instance_uid] = res
                        if res.get("status") == "skipped_quota_reached":
                            quota_stopped = True
                    except Exception as error:
                        failed_count += 1
                        results_by_uid[spec.series_instance_uid] = {
                            "status": "failed",
                            "series": asdict(spec),
                            "error": str(error),
                        }

    # Preserve exact deterministic manifest order in results and CSV report
    ordered_results = [
        results_by_uid[s.series_instance_uid]
        for s in specs
        if s.series_instance_uid in results_by_uid
    ]

    report_csv = env_profile.data_root / "manifests" / "download_report.csv"
    _write_download_report_csv(report_csv, ordered_results)

    elapsed_total = max(time.monotonic() - t_run_start, 1e-3)
    downloaded_zip_bytes = sum(
        int(r.get("archive_bytes") or 0)
        for r in ordered_results
        if r.get("status") == "downloaded"
    )
    downloaded_zip_mb = round(downloaded_zip_bytes / (1024 * 1024), 2)
    overall_speed_mbps = (
        round(downloaded_zip_mb / elapsed_total, 2) if downloaded_zip_bytes > 0 else 0.0
    )

    final_snap = budget_guard.snapshot()
    summary = {
        "profile": env_profile.name,
        "kaggle_detected": env_profile.is_kaggle_env,
        "data_root": str(env_profile.data_root),
        "incoming_root": str(env_profile.incoming_root),
        "max_disk_bytes": env_profile.max_disk_bytes,
        "max_disk_gb": env_profile.max_disk_gb,
        "final_data_root_bytes": final_snap["on_disk_bytes"],
        "final_data_root_gb": round(int(final_snap["on_disk_bytes"] or 0) / (1024**3), 3),
        "total_series_in_manifest": len(specs),
        "completed_series": sum(
            1
            for r in ordered_results
            if r.get("status") in {"downloaded", "already_present", "already_extracted"}
        ),
        "skipped_quota_series": sum(
            1 for r in ordered_results if r.get("status") == "skipped_quota_reached"
        ),
        "failed_series": failed_count,
        "quota_stopped": quota_stopped,
        "elapsed_seconds": round(elapsed_total, 2),
        "downloaded_zip_mb": downloaded_zip_mb,
        "average_speed_mbps": overall_speed_mbps,
        "download_report_csv": str(report_csv),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": ordered_results,
    }
    _atomic_json(env_profile.data_root / "manifests" / "download_run_summary.json", summary)

    if show_progress:
        sys.stderr.write(
            f"\n{'=' * 58}\n"
            f"🎉 DONE in {elapsed_total / 60.0:.2f} min ({elapsed_total:.1f}s)\n"
            f"⚡ Average network speed : {overall_speed_mbps:.2f} MiB/s\n"
            f"📦 Downloaded ZIP stream : {downloaded_zip_mb:.1f} MiB\n"
            f"💾 Final data_root usage : {summary['final_data_root_gb']:.3f} GiB ({quota_label})\n"
            f"📊 CSV Report            : {report_csv}\n"
            f"{'=' * 58}\n"
        )
        sys.stderr.flush()

    return summary
