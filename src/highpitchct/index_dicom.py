"""Header-only DICOM & DICOM-CT-PD v4 indexer with trajectory and truncation QA."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .ctpd_tags import (
    classify_dicom_sop,
    decode_named_ctpd_tag,
    decode_tag,
    extract_referenced_raw_series_uid,
)

_FILE_FIELDS = [
    "source_path",
    "file_bytes",
    "dicom_kind",
    "sop_class_uid",
    "sop_instance_uid",
    "study_instance_uid",
    "series_instance_uid",
    "referenced_raw_series_uid",
    "modality",
    "manufacturer",
    "body_part",
    "series_description",
    "instance_number",
    "rows",
    "columns",
    "number_of_frames",
    "phi0_rad",
    "z0_mm",
    "slice_z_mm",
    "patient_id_sha256_12",
    "read_error",
]

_SERIES_FIELDS = [
    "series_instance_uid",
    "study_instance_uid",
    "patient_id_sha256_12",
    "dicom_kind",
    "body_part",
    "manufacturer",
    "series_description",
    "file_count",
    "total_mb",
    "instance_min",
    "instance_max",
    "missing_instances",
    "duplicate_instances",
    "detector_rows",
    "detector_cols",
    "detector_shape",
    "ffs_mode",
    "angular_steps_per_rotation",
    "nominal_spiral_pitch",
    "z_min_mm",
    "z_max_mm",
    "axial_span_mm",
    "table_direction",
    "estimated_rotations",
    "coverage_isocenter_mm",
    "estimated_table_feed_mm_per_rot",
    "estimated_normalized_pitch",
    "pitch_fit_rmse_mm",
    "referenced_raw_series_uid",
    "linked_image_series_uids",
    "validation_status",
    "validation_notes",
]


def _text(dataset, keyword: str) -> str:
    value = getattr(dataset, keyword, "")
    return str(value).strip() if value is not None else ""


def _patient_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12] if value else ""


def _unwrap_angles_rad(angles: list[float]) -> list[float]:
    if not angles:
        return []
    unwrapped = [angles[0]]
    two_pi = 2.0 * math.pi
    for idx in range(1, len(angles)):
        diff = angles[idx] - angles[idx - 1]
        diff_wrapped = (diff + math.pi) % two_pi - math.pi
        if diff_wrapped == -math.pi and diff > 0:
            diff_wrapped = math.pi
        unwrapped.append(unwrapped[-1] + diff_wrapped)
    return unwrapped


def _linear_fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, rmse) for y = slope * x + intercept."""
    n = len(x)
    if n < 2:
        return 0.0, (y[0] if y else 0.0), 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx <= 1e-18:
        return 0.0, mean_y, 0.0
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    mse = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y)) / n
    return slope, intercept, math.sqrt(mse)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def index_dicom_tree(
    data_root: str | Path,
    *,
    scan_dirs: list[str | Path] | None = None,
) -> dict:
    """Scan DICOM headers without loading PixelData and audit series geometry & completeness."""
    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError("pydicom is required for the index command") from error

    root = Path(data_root).expanduser().resolve()
    default_scan = root / "raw" / "tcia"
    roots_to_scan = (
        [Path(p).expanduser().resolve() for p in scan_dirs]
        if scan_dirs
        else [default_scan]
    )

    output_csv = root / "manifests" / "dicom_index.csv"
    series_csv = root / "manifests" / "series_inventory.csv"
    output_json = root / "manifests" / "dicom_index_summary.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[Path, Path]] = []
    for sroot in roots_to_scan:
        if not sroot.exists():
            continue
        if sroot.is_file():
            candidates.append((sroot.parent, sroot))
            continue
        for p in sorted(sroot.rglob("*")):
            if not p.is_file():
                continue
            # Skip non-DICOM metadata files such as NBIA metadata.csv or .json/.txt
            if p.suffix.lower() in {".csv", ".json", ".txt", ".md", ".zip", ".part", ".xlsx"}:
                continue
            candidates.append((sroot, p))

    file_rows: list[dict[str, str | int | float]] = []
    series_accum: dict[str, dict] = {}

    for base_dir, path in candidates:
        record: dict[str, str | int | float] = {field: "" for field in _FILE_FIELDS}
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path.relative_to(base_dir))
        record["source_path"] = rel
        fbytes = path.stat().st_size
        record["file_bytes"] = fbytes

        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
            kind = classify_dicom_sop(ds)
            record["dicom_kind"] = kind
            for keyword, field in [
                ("SOPClassUID", "sop_class_uid"),
                ("SOPInstanceUID", "sop_instance_uid"),
                ("StudyInstanceUID", "study_instance_uid"),
                ("SeriesInstanceUID", "series_instance_uid"),
                ("Modality", "modality"),
                ("Manufacturer", "manufacturer"),
                ("BodyPartExamined", "body_part"),
                ("SeriesDescription", "series_description"),
                ("InstanceNumber", "instance_number"),
                ("Rows", "rows"),
                ("Columns", "columns"),
                ("NumberOfFrames", "number_of_frames"),
            ]:
                record[field] = _text(ds, keyword)

            pid_hash = _patient_hash(_text(ds, "PatientID"))
            record["patient_id_sha256_12"] = pid_hash

            inst_str = str(record["instance_number"])
            inst_int = int(inst_str) if inst_str.lstrip("-").isdigit() else None

            phi0_val: float | None = None
            z0_val: float | None = None
            slice_z_val: float | None = None
            ref_raw_uid = ""

            if kind == "dicom_ct_pd_projection":
                phi0_val = decode_named_ctpd_tag(ds, "phi0_rad")
                z0_val = decode_named_ctpd_tag(ds, "z0_mm")
                if phi0_val is not None:
                    record["phi0_rad"] = round(float(phi0_val), 7)
                if z0_val is not None:
                    record["z0_mm"] = round(float(z0_val), 5)
            elif kind == "ct_image":
                ref_raw_uid = extract_referenced_raw_series_uid(ds)
                record["referenced_raw_series_uid"] = ref_raw_uid
                if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                    slice_z_val = float(ds.ImagePositionPatient[2])
                    record["slice_z_mm"] = round(slice_z_val, 5)

            suid = str(record["series_instance_uid"])
            if suid:
                acc = series_accum.get(suid)
                if acc is None:
                    acc = {
                        "series_instance_uid": suid,
                        "study_instance_uid": str(record["study_instance_uid"]),
                        "patient_id_sha256_12": pid_hash,
                        "dicom_kind": kind,
                        "body_part": str(record["body_part"]),
                        "manufacturer": str(record["manufacturer"]),
                        "series_description": str(record["series_description"]),
                        "file_count": 0,
                        "total_bytes": 0,
                        "instances": [],
                        "traj": [],  # (inst, phi0, z0)
                        "slice_zs": [],
                        "referenced_raw_uids": set(),
                        "meta": {},
                    }
                    if kind == "dicom_ct_pd_projection":
                        acc["meta"] = {
                            "detector_rows": decode_named_ctpd_tag(ds, "detector_rows"),
                            "detector_cols": decode_named_ctpd_tag(ds, "detector_cols"),
                            "du_mm": decode_named_ctpd_tag(ds, "du_mm"),
                            "dv_mm": decode_named_ctpd_tag(ds, "dv_mm"),
                            "rho0_mm": decode_named_ctpd_tag(ds, "rho0_mm"),
                            "d0_mm": decode_named_ctpd_tag(ds, "d0_mm"),
                            "detector_shape": decode_named_ctpd_tag(ds, "detector_shape"),
                            "ffs_mode": decode_named_ctpd_tag(ds, "ffs_mode"),
                            "angular_steps_per_rotation": decode_named_ctpd_tag(
                                ds, "angular_steps_per_rotation"
                            ),
                            "nominal_spiral_pitch": decode_tag(ds, 0x0018, 0x9311, "FD"),
                        }
                    series_accum[suid] = acc

                acc["file_count"] += 1
                acc["total_bytes"] += fbytes
                if inst_int is not None:
                    acc["instances"].append(inst_int)
                if phi0_val is not None and z0_val is not None and inst_int is not None:
                    acc["traj"].append((inst_int, float(phi0_val), float(z0_val)))
                if slice_z_val is not None:
                    acc["slice_zs"].append(slice_z_val)
                if ref_raw_uid:
                    acc["referenced_raw_uids"].add(ref_raw_uid)
        except Exception as error:
            record["read_error"] = f"{type(error).__name__}: {error}"

        file_rows.append(record)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FILE_FIELDS)
        writer.writeheader()
        writer.writerows(file_rows)

    # Map reverse links: raw_series_uid -> list of image series UIDs
    raw_to_images: dict[str, list[str]] = {}
    for suid, acc in series_accum.items():
        for raw_uid in acc["referenced_raw_uids"]:
            raw_to_images.setdefault(raw_uid, []).append(suid)

    # First pass to compute axial spans for all series
    series_spans: dict[str, float] = {}
    for suid, acc in series_accum.items():
        if acc["dicom_kind"] == "dicom_ct_pd_projection" and acc["traj"]:
            zs = [t[2] for t in acc["traj"]]
            series_spans[suid] = max(zs) - min(zs)
        elif acc["dicom_kind"] == "ct_image" and acc["slice_zs"]:
            zs = acc["slice_zs"]
            series_spans[suid] = max(zs) - min(zs)
        else:
            series_spans[suid] = 0.0

    # Group projection counts per patient to catch mismatched partial downloads
    patient_max_proj_count: dict[str, int] = {}
    patient_max_image_span: dict[str, float] = {}
    for suid, acc in series_accum.items():
        ph = acc["patient_id_sha256_12"]
        if not ph:
            continue
        if acc["dicom_kind"] == "dicom_ct_pd_projection":
            patient_max_proj_count[ph] = max(
                patient_max_proj_count.get(ph, 0), acc["file_count"]
            )
        elif acc["dicom_kind"] == "ct_image":
            patient_max_image_span[ph] = max(
                patient_max_image_span.get(ph, 0.0), series_spans[suid]
            )

    series_rows: list[dict] = []
    for suid in sorted(series_accum.keys()):
        acc = series_accum[suid]
        notes: list[str] = []
        status = "VALID"

        insts = acc["instances"]
        inst_min = min(insts) if insts else ""
        inst_max = max(insts) if insts else ""
        unique_insts = len(set(insts))
        dup_count = len(insts) - unique_insts
        missing_count = (
            (int(inst_max) - int(inst_min) + 1 - unique_insts) if insts else 0
        )

        if dup_count > 0:
            status = "INVALID_DUPLICATE_VIEWS"
            notes.append(f"{dup_count} duplicate InstanceNumbers")
        if missing_count > 0:
            status = "INVALID_MISSING_VIEWS"
            notes.append(f"{missing_count} missing InstanceNumbers between {inst_min}..{inst_max}")

        srow: dict[str, str | int | float] = {f: "" for f in _SERIES_FIELDS}
        srow.update(
            {
                "series_instance_uid": suid,
                "study_instance_uid": acc["study_instance_uid"],
                "patient_id_sha256_12": acc["patient_id_sha256_12"],
                "dicom_kind": acc["dicom_kind"],
                "body_part": acc["body_part"],
                "manufacturer": acc["manufacturer"],
                "series_description": acc["series_description"],
                "file_count": acc["file_count"],
                "total_mb": round(acc["total_bytes"] / (1024**2), 2),
                "instance_min": inst_min,
                "instance_max": inst_max,
                "missing_instances": missing_count,
                "duplicate_instances": dup_count,
                "referenced_raw_series_uid": ";".join(sorted(acc["referenced_raw_uids"])),
                "linked_image_series_uids": ";".join(sorted(raw_to_images.get(suid, []))),
            }
        )

        if acc["dicom_kind"] == "dicom_ct_pd_projection":
            meta = acc["meta"]
            for k in (
                "detector_rows",
                "detector_cols",
                "detector_shape",
                "ffs_mode",
                "angular_steps_per_rotation",
                "nominal_spiral_pitch",
            ):
                if meta.get(k) is not None:
                    srow[k] = meta[k]

            traj = sorted(acc["traj"], key=lambda t: t[0])
            if traj:
                phis = [t[1] for t in traj]
                zs = [t[2] for t in traj]
                z_min, z_max = min(zs), max(zs)
                span_mm = z_max - z_min
                srow["z_min_mm"] = round(z_min, 4)
                srow["z_max_mm"] = round(z_max, 4)
                srow["axial_span_mm"] = round(span_mm, 4)
                srow["table_direction"] = (
                    1 if zs[-1] > zs[0] else (-1 if zs[-1] < zs[0] else 0)
                )

                phi_unwrap = _unwrap_angles_rad(phis)
                total_rad = abs(phi_unwrap[-1] - phi_unwrap[0])
                rotations = total_rad / (2.0 * math.pi)
                srow["estimated_rotations"] = round(rotations, 4)

                if len(phi_unwrap) >= 3:
                    dphis = [abs(phi_unwrap[i + 1] - phi_unwrap[i]) for i in range(len(phi_unwrap) - 1)]
                    med_dphi = _median(dphis)
                    if med_dphi > 0 and max(dphis) > 3.5 * med_dphi:
                        status = "INVALID_ANGLE_DISCONTINUITY"
                        notes.append(
                            f"Max angular step {max(dphis):.6f} rad exceeds 3.5x median {med_dphi:.6f} rad"
                        )

                slope_mm_per_rad, _, rmse_mm = _linear_fit(phi_unwrap, zs)
                feed_mm_rot = abs(slope_mm_per_rad) * 2.0 * math.pi
                srow["estimated_table_feed_mm_per_rot"] = round(feed_mm_rot, 4)
                srow["pitch_fit_rmse_mm"] = round(rmse_mm, 5)

                nrow = meta.get("detector_rows")
                dv = meta.get("dv_mm")
                rho0 = meta.get("rho0_mm")
                d0 = meta.get("d0_mm")
                if nrow and dv and rho0 and d0:
                    w_iso = float(nrow) * float(dv) * (float(rho0) / float(d0))
                    srow["coverage_isocenter_mm"] = round(w_iso, 4)
                    if w_iso > 0:
                        srow["estimated_normalized_pitch"] = round(feed_mm_rot / w_iso, 5)

                # Cross-series axial truncation check against linked images or patient image span
                linked_imgs = raw_to_images.get(suid, [])
                ref_img_span = max(
                    [series_spans.get(iuid, 0.0) for iuid in linked_imgs],
                    default=patient_max_image_span.get(acc["patient_id_sha256_12"], 0.0),
                )
                if ref_img_span > 0 and span_mm < 0.90 * ref_img_span:
                    if status == "VALID":
                        status = "WARNING_TRUNCATED_PROJECTION_SERIES"
                    notes.append(
                        f"Projection z-span ({span_mm:.2f} mm) < linked image z-span ({ref_img_span:.2f} mm); partial download likely"
                    )

                max_p_count = patient_max_proj_count.get(acc["patient_id_sha256_12"], 0)
                if max_p_count > 0 and acc["file_count"] < 0.95 * max_p_count:
                    if status == "VALID":
                        status = "WARNING_TRUNCATED_PROJECTION_SERIES"
                    notes.append(
                        f"View count ({acc['file_count']}) < sibling projection series ({max_p_count})"
                    )
            else:
                status = "INVALID_MISSING_GEOMETRY"
                notes.append("Missing private trajectory tags (7031,1001)/(7031,1002)")

        elif acc["dicom_kind"] == "ct_image" and acc["slice_zs"]:
            z_min, z_max = min(acc["slice_zs"]), max(acc["slice_zs"])
            srow["z_min_mm"] = round(z_min, 4)
            srow["z_max_mm"] = round(z_max, 4)
            srow["axial_span_mm"] = round(z_max - z_min, 4)

        srow["validation_status"] = status
        srow["validation_notes"] = " | ".join(notes)
        series_rows.append(srow)

    with series_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SERIES_FIELDS)
        writer.writeheader()
        writer.writerows(series_rows)

    errors = sum(bool(row["read_error"]) for row in file_rows)
    summary = {
        "scanned_roots": [str(p) for p in roots_to_scan],
        "candidate_files": len(candidates),
        "indexed_files": len(file_rows) - errors,
        "read_errors": errors,
        "series_count": len(series_rows),
        "series_instance_uids": [str(r["series_instance_uid"]) for r in series_rows],
        "csv": str(output_csv),
        "series_inventory_csv": str(series_csv),
        "series_summaries": series_rows,
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
