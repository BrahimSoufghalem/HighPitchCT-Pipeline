"""Header-only DICOM indexer. Pixel arrays are never loaded."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

_FIELDS = [
    "relative_path",
    "file_bytes",
    "sop_class_uid",
    "sop_instance_uid",
    "study_instance_uid",
    "series_instance_uid",
    "modality",
    "manufacturer",
    "series_description",
    "instance_number",
    "rows",
    "columns",
    "number_of_frames",
    "patient_id_sha256_12",
    "read_error",
]


def _text(dataset, keyword: str) -> str:
    value = getattr(dataset, keyword, "")
    return str(value) if value is not None else ""


def _patient_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12] if value else ""


def index_dicom_tree(data_root: str | Path) -> dict:
    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError("pydicom is required for the index command") from error

    root = Path(data_root)
    scan_root = root / "raw" / "tcia"
    output_csv = root / "manifests" / "dicom_index.csv"
    output_json = root / "manifests" / "dicom_index_summary.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    candidates = [p for p in scan_root.rglob("*") if p.is_file()] if scan_root.exists() else []
    for path in sorted(candidates):
        record: dict[str, str | int] = {field: "" for field in _FIELDS}
        record["relative_path"] = str(path.relative_to(root))
        record["file_bytes"] = path.stat().st_size
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
            for keyword, field in [
                ("SOPClassUID", "sop_class_uid"),
                ("SOPInstanceUID", "sop_instance_uid"),
                ("StudyInstanceUID", "study_instance_uid"),
                ("SeriesInstanceUID", "series_instance_uid"),
                ("Modality", "modality"),
                ("Manufacturer", "manufacturer"),
                ("SeriesDescription", "series_description"),
                ("InstanceNumber", "instance_number"),
                ("Rows", "rows"),
                ("Columns", "columns"),
                ("NumberOfFrames", "number_of_frames"),
            ]:
                record[field] = _text(ds, keyword)
            record["patient_id_sha256_12"] = _patient_hash(_text(ds, "PatientID"))
        except Exception as error:  # index must report unreadable files rather than hide them
            record["read_error"] = f"{type(error).__name__}: {error}"
        rows.append(record)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    errors = sum(bool(row["read_error"]) for row in rows)
    series = sorted({str(row["series_instance_uid"]) for row in rows if row["series_instance_uid"]})
    summary = {
        "scanned_root": str(scan_root),
        "candidate_files": len(candidates),
        "indexed_files": len(rows) - errors,
        "read_errors": errors,
        "series_count": len(series),
        "series_instance_uids": series,
        "csv": str(output_csv),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
