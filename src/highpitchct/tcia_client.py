"""Resumable, quota-aware downloader and TCIA NBIA series discovery client."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .layout import safe_component, series_paths
from .manifest import SeriesSpec
from .runtime import DiskBudgetExceededError, DiskBudgetGuard, EnvironmentProfile
from .safe_zip import safe_extract_zip, verify_zip

DEFAULT_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DEFAULT_COLLECTION = "LDCT-and-Projection-data"


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
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _get_json(url: str, timeout: int = 60) -> list[dict]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "HighPitchCT-Pipeline/0.3", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    if not raw.strip():
        return []
    parsed = json.loads(raw)
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
    except (OSError, ValueError, urllib.error.URLError):
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
        raw_bytes = rec.get("FileSize")
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


def _download_once(url: str, partial: Path, timeout: int) -> tuple[int, bool]:
    """Download once. Returns (bytes on disk, resumed).

    A partial file is appended only when the server answers HTTP 206. If a Range
    request is ignored and HTTP 200 is returned, the file is safely restarted.
    """
    partial.parent.mkdir(parents=True, exist_ok=True)
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "HighPitchCT-Pipeline/0.3"}
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
    on_disk_files = sum(1 for p in paths.extracted.rglob("*") if p.is_file())
    expected = spec.expected_files or prior.get("extracted_files")
    if on_disk_files > 0 and (expected is None or on_disk_files == int(expected)):
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
    if spec.expected_bytes is not None:
        return spec.expected_bytes, spec.expected_files
    meta = fetch_series_metadata(spec.series_instance_uid, base_url=base_url, timeout=timeout)
    if not meta:
        return None, spec.expected_files
    raw_size = meta.get("File Size") or meta.get("FileSize")
    raw_count = meta.get("Image Count") or meta.get("ImageCount")
    exp_bytes = int(raw_size) if raw_size not in (None, "") else None
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
) -> dict:
    """Download, validate, enforce disk quota, extract, and optionally clean up ZIP."""
    paths = series_paths(data_root, spec, incoming_root=incoming_root)
    url = image_url(spec.series_instance_uid, base_url)

    if extract:
        existing_payload = _existing_extracted_state(paths, spec)
        if existing_payload is not None:
            _atomic_json(paths.state, existing_payload)
            return existing_payload

    expected_bytes = spec.expected_bytes
    expected_files = spec.expected_files
    if budget_guard is not None and budget_guard.max_disk_bytes is not None and expected_bytes is None:
        expected_bytes, expected_files = estimate_series_bytes(
            spec, base_url=base_url, timeout=min(timeout, 60)
        )

    reserved_bytes = 0
    if budget_guard is not None and expected_bytes is not None:
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
            return payload
        reserved_bytes = expected_bytes

    try:
        if paths.archive.exists():
            zip_members, uncompressed_bytes = verify_zip(paths.archive)
            status = "already_present"
            attempts = 0
            resumed = False
        else:
            last_error: Exception | None = None
            resumed = False
            attempts = 0
            zip_members = 0
            uncompressed_bytes = 0
            for attempts in range(1, retries + 1):
                try:
                    _, resumed_now = _download_once(url, paths.partial, timeout)
                    resumed = resumed or resumed_now
                    zip_members, uncompressed_bytes = verify_zip(paths.partial)
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

        archive_bytes = paths.archive.stat().st_size
        archive_sha256 = sha256_file(paths.archive)

        # Release pre-extraction estimate right before exact uncompressed quota check
        if budget_guard is not None and reserved_bytes > 0:
            budget_guard.release(reserved_bytes)
            reserved_bytes = 0

        extracted_files = None
        if extract:
            try:
                extracted_files = safe_extract_zip(
                    paths.archive, paths.extracted, budget_guard=budget_guard
                )
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

            if expected_files is not None and extracted_files != expected_files:
                # NBIA ZIP sometimes includes 1 license/manifest file; count .dcm specifically too
                dcm_count = sum(
                    1
                    for p in paths.extracted.rglob("*")
                    if p.is_file() and p.suffix.lower() == ".dcm"
                )
                if dcm_count != expected_files and extracted_files != expected_files:
                    raise RuntimeError(
                        f"Extracted DICOM count ({dcm_count}) does not match expected_files ({expected_files})"
                    )

        archive_cleaned_up = False
        if extract and cleanup_archive and paths.archive.exists():
            paths.archive.unlink(missing_ok=True)
            archive_cleaned_up = True

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
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(paths.state, payload)
        return payload
    finally:
        if budget_guard is not None and reserved_bytes > 0:
            budget_guard.release(reserved_bytes)


def download_manifest_with_profile(
    specs: list[SeriesSpec],
    env_profile: EnvironmentProfile,
    *,
    workers: int = 2,
    retries: int = 5,
    timeout: int = 120,
    extract: bool = True,
    base_url: str = DEFAULT_BASE_URL,
) -> dict:
    """Download series grouped by case_id with smart Kaggle <=18 GiB vs Full rules."""
    budget_guard = DiskBudgetGuard(env_profile.data_root, env_profile.max_disk_bytes)

    cases: OrderedDict[str, list[SeriesSpec]] = OrderedDict()
    for spec in specs:
        cases.setdefault(spec.case_id, []).append(spec)

    results: list[dict] = []
    failed_count = 0
    quota_stopped = False

    for case_id, case_specs in cases.items():
        if quota_stopped:
            for spec in case_specs:
                paths = series_paths(
                    env_profile.data_root, spec, incoming_root=env_profile.incoming_root
                )
                skipped = {
                    "status": "skipped_quota_reached",
                    "series": asdict(spec),
                    "reason": f"Disk quota already reached prior to case {case_id}",
                    "quota": budget_guard.snapshot(),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _atomic_json(paths.state, skipped)
                results.append(skipped)
            continue

        # Case-level pre-check when quota is active so we avoid partial patients
        if budget_guard.max_disk_bytes is not None:
            pending_bytes = 0
            enriched_specs: list[SeriesSpec] = []
            for spec in case_specs:
                paths = series_paths(
                    env_profile.data_root, spec, incoming_root=env_profile.incoming_root
                )
                if extract and _existing_extracted_state(paths, spec) is not None:
                    enriched_specs.append(spec)
                    continue
                exp_b, exp_f = estimate_series_bytes(
                    spec, base_url=base_url, timeout=min(timeout, 60)
                )
                updated_spec = SeriesSpec(
                    case_id=spec.case_id,
                    series_instance_uid=spec.series_instance_uid,
                    role=spec.role,
                    description=spec.description,
                    expected_files=exp_f,
                    expected_bytes=exp_b,
                    body_part=spec.body_part,
                )
                enriched_specs.append(updated_spec)
                if exp_b is not None:
                    pending_bytes += exp_b
            case_specs = enriched_specs

            if pending_bytes > 0 and not budget_guard.check_within_budget(pending_bytes):
                quota_stopped = True
                snap = budget_guard.snapshot()
                for spec in case_specs:
                    paths = series_paths(
                        env_profile.data_root, spec, incoming_root=env_profile.incoming_root
                    )
                    skipped = {
                        "status": "skipped_quota_reached",
                        "series": asdict(spec),
                        "case_pending_bytes": pending_bytes,
                        "reason": (
                            f"Case {case_id} requires ~{pending_bytes} bytes which exceeds "
                            f"remaining budget in {env_profile.name} profile"
                        ),
                        "quota": snap,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    _atomic_json(paths.state, skipped)
                    results.append(skipped)
                continue

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(case_specs)))) as pool:
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
                ): spec
                for spec in case_specs
            }
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    if res.get("status") == "skipped_quota_reached":
                        quota_stopped = True
                except Exception as error:
                    failed_count += 1
                    results.append(
                        {
                            "status": "failed",
                            "series": asdict(spec),
                            "error": str(error),
                        }
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
            for r in results
            if r.get("status") in {"downloaded", "already_present", "already_extracted"}
        ),
        "skipped_quota_series": sum(
            1 for r in results if r.get("status") == "skipped_quota_reached"
        ),
        "failed_series": failed_count,
        "quota_stopped": quota_stopped,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    _atomic_json(env_profile.data_root / "manifests" / "download_run_summary.json", summary)
    return summary
