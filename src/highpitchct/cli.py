"""Command-line interface for Stage 0 & Stage 1 (download, indexing, geometry & reader audit)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from .ctpd_reader import (
    load_projections_memmap,
    read_ct_image_volume,
    scan_projection_series,
    verify_projection_orientation,
)
from .index_dicom import index_dicom_tree
from .manifest import read_series_manifest, write_series_manifest
from .phantom import (
    build_thorax_phantom,
    make_synthetic_helical_geometry,
    project_phantom_helical,
    voxelize_phantom,
    write_synthetic_ctpd_series,
)
from .runtime import print_runtime_report, resolve_environment_profile
from .tcia_client import DEFAULT_COLLECTION, discover_tcia_series, download_manifest_with_profile


def _discover_local_case_series(
    tcia_root: Path, case_id: str | None = None
) -> list[tuple[str, Path, Path | None]]:
    """Discover downloaded (case_id, projection_dir, optional_image_dir) under raw/tcia."""
    if not tcia_root.exists():
        return []
    case_dirs = (
        [tcia_root / case_id]
        if case_id
        else sorted(p for p in tcia_root.iterdir() if p.is_dir())
    )
    found: list[tuple[str, Path, Path | None]] = []
    for cdir in case_dirs:
        if not cdir.is_dir():
            continue
        proj_parent = cdir / "projection_full"
        if not proj_parent.is_dir():
            continue
        proj_subdirs = sorted(p for p in proj_parent.iterdir() if p.is_dir())
        if not proj_subdirs:
            continue
        img_parent = cdir / "reconstruction_published"
        img_dir: Path | None = None
        if img_parent.is_dir():
            img_subdirs = sorted(p for p in img_parent.iterdir() if p.is_dir())
            if img_subdirs:
                img_dir = img_subdirs[0]
        for pdir in proj_subdirs:
            found.append((cdir.name, pdir, img_dir))
    return found


def _run_synthetic_phantom_roundtrip(workers: int = 4) -> dict[str, Any]:
    """Execute an end-to-end synthetic 3D Thorax Phantom DICOM-CT-PD v4 round-trip check."""
    t0 = time.monotonic()
    geom_gt = make_synthetic_helical_geometry(
        n_views=384,
        n_rows=32,
        n_cols=128,
        angular_steps_per_rotation=192,
        normalized_pitch=0.85,
        detector_shape="CYLINDRICAL",
        ffs_mode="FFSZ",
        rotation_direction=-1,
        quarter_detector_offset_cols=1.125,
    )
    phantom = build_thorax_phantom(axial_scale=0.5)
    vox = voxelize_phantom(
        phantom,
        shape_zyx=(48, 96, 96),
        spacing_zyx_mm=(2.0, 3.2, 3.2),
        water_mu_mm_inv=geom_gt.water_mu_mm_inv,
    )
    proj_gt = project_phantom_helical(phantom, geom_gt, batch_views=64, show_progress=False)

    with tempfile.TemporaryDirectory(prefix="hpct_stage1_phantom_") as tmp:
        tmp_path = Path(tmp)
        dcm_dir = tmp_path / "dcm"
        mm_path = tmp_path / "proj.f32.mmap"
        write_synthetic_ctpd_series(dcm_dir, geom_gt, proj_gt)
        geom_loaded, mm = load_projections_memmap(
            dcm_dir, mm_path, chunk_views=128, workers=workers, show_progress=False
        )
        diff = np.asarray(mm, dtype=np.float32) - proj_gt
        rmse = float(np.sqrt(np.mean(diff * diff)))
        max_abs_err = float(np.max(np.abs(diff)))
        orient = verify_projection_orientation(dcm_dir, workers=workers)
        del mm

    roundtrip_pass = bool(
        orient["overall_pass"]
        and rmse < 2.0e-4
        and max_abs_err < 5.0e-4
        and abs(geom_loaded.center_col_0based - geom_gt.center_col_0based) < 1e-5
    )
    return {
        "pass": roundtrip_pass,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "voxel_hu_min_max": [
            round(float(np.min(vox["hu_volume"])), 2),
            round(float(np.max(vox["hu_volume"])), 2),
        ],
        "projection_shape": list(proj_gt.shape),
        "quantization_rmse": round(rmse, 7),
        "quantization_max_abs_err": round(max_abs_err, 7),
        "orientation_verification": orient,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpct",
        description="HighPitchCT CLI (Stage 0 download/index & Stage 1 geometry/reader/phantom audit)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser(
        "discover-tcia",
        help="Query NBIA public API and generate a reproducible series CSV manifest",
    )
    discover.add_argument("--collection", default=DEFAULT_COLLECTION)
    discover.add_argument(
        "--body-part",
        help="Optional BodyPartExamined filter, e.g., CHEST or ABDOMEN",
    )
    discover.add_argument(
        "--roles",
        default="projection_full,reconstruction_published",
        help="Comma-separated roles to include (default: projection_full,reconstruction_published)",
    )
    discover.add_argument("--max-cases", type=int, help="Optional maximum number of patient cases")
    discover.add_argument("--output", required=True, help="Output CSV path")
    discover.add_argument("--timeout", type=int, default=120)

    validate = sub.add_parser(
        "validate-manifest", help="Validate a series CSV without downloading"
    )
    validate.add_argument("--series-file", required=True)

    download = sub.add_parser(
        "download",
        help="Download TCIA series with smart Kaggle (<=18 GiB) vs Full disk quota management",
    )
    download.add_argument("--series-file", required=True)
    download.add_argument("--data-root", help="Writable root for raw/ and manifests/")
    download.add_argument(
        "--incoming-root",
        help="Staging directory for .zip/.part files (defaults to /tmp/highpitchct-incoming on Kaggle)",
    )
    download.add_argument(
        "--profile",
        choices=["auto", "kaggle", "full"],
        default=None,
        help="Execution profile: auto (default), kaggle (<=18 GiB cap + /tmp staging), or full (no cap)",
    )
    download.add_argument(
        "--max-disk-gb",
        type=float,
        default=None,
        help="Explicit disk quota cap in GiB for data-root (overrides profile default)",
    )
    zip_group = download.add_mutually_exclusive_group()
    zip_group.add_argument(
        "--cleanup-zip",
        action="store_true",
        help="Delete verified .zip archive after extraction (default on Kaggle)",
    )
    zip_group.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep .zip archive after extraction (default on Full profile)",
    )
    download.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel series download workers across admitted patients (default: 4)",
    )
    download.add_argument("--retries", type=int, default=5)
    download.add_argument("--timeout", type=int, default=120)
    download.add_argument("--no-extract", action="store_true")
    download.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live tqdm progress bars and stderr status logs",
    )

    index = sub.add_parser(
        "index",
        help="Create header-only DICOM & DICOM-CT-PD v4 inventory and series audit",
    )
    index.add_argument("--data-root", help="Writable root where manifests/ will be saved")
    index.add_argument(
        "--scan-dir",
        action="append",
        dest="scan_dirs",
        help="Optional directory to scan (e.g., /kaggle/input/... or external drive); repeatable",
    )
    index.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel threads for header-only DICOM scanning (default: 8)",
    )
    index.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live tqdm progress bar during indexing",
    )

    stage1 = sub.add_parser(
        "stage1-audit",
        help="Stage 1 physical audit: 3D helical geometry, FFS, orientation proofs, and 3D phantom check",
    )
    stage1.add_argument("--data-root", help="Writable root containing raw/tcia and manifests/")
    stage1.add_argument("--case-id", help="Optional patient case filter under raw/tcia (e.g., C012)")
    stage1.add_argument("--series-dir", help="Explicit DICOM-CT-PD projection directory to audit")
    stage1.add_argument(
        "--image-dir",
        help="Optional published CT image directory to verify UID linkage and HU/mu ranges",
    )
    stage1.add_argument(
        "--memmap-out",
        help="Optional output path to stream projections into a float32 disk memmap + .geometry.npz",
    )
    stage1.add_argument(
        "--run-phantom-check",
        action="store_true",
        help="Run end-to-end synthetic 3D Thorax Phantom DICOM-CT-PD round-trip verification",
    )
    stage1.add_argument("--workers", type=int, default=8)
    stage1.add_argument("--no-progress", action="store_true")

    environment = sub.add_parser(
        "env", help="Report Kaggle/Full profile, disk quota, RAM, disk space, and GPUs"
    )
    environment.add_argument("--data-root")
    environment.add_argument("--profile", choices=["auto", "kaggle", "full"], default=None)
    environment.add_argument("--max-disk-gb", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "discover-tcia":
        roles = {r.strip() for r in args.roles.split(",") if r.strip()}
        specs = discover_tcia_series(
            collection=args.collection,
            body_part=args.body_part,
            roles=roles,
            max_cases=args.max_cases,
            timeout=args.timeout,
        )
        out_path = write_series_manifest(args.output, specs, enabled=True)
        total_expected = sum(s.expected_bytes or 0 for s in specs)
        cases = sorted({s.case_id for s in specs})
        print(
            json.dumps(
                {
                    "output_csv": str(out_path),
                    "cases_discovered": len(cases),
                    "series_discovered": len(specs),
                    "total_expected_gb": round(total_expected / (1024**3), 3),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate-manifest":
        specs = read_series_manifest(args.series_file)
        cases = sorted({s.case_id for s in specs})
        total_expected = sum(s.expected_bytes or 0 for s in specs)
        print(
            json.dumps(
                {
                    "enabled_cases": len(cases),
                    "enabled_series": len(specs),
                    "known_expected_gb": round(total_expected / (1024**3), 3)
                    if total_expected > 0
                    else None,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "env":
        print_runtime_report(
            args.data_root, profile=args.profile, max_disk_gb=args.max_disk_gb
        )
        return 0

    if args.command == "index":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        env_prof = resolve_environment_profile(data_root=args.data_root)
        summary = index_dicom_tree(
            env_prof.data_root,
            scan_dirs=args.scan_dirs,
            workers=args.workers,
            show_progress=not args.no_progress,
        )
        compact = {k: v for k, v in summary.items() if k != "series_summaries"}
        compact["series_validation_statuses"] = {
            s["series_instance_uid"]: {
                "kind": s["dicom_kind"],
                "files": s["file_count"],
                "pitch": s["estimated_normalized_pitch"],
                "status": s["validation_status"],
                "notes": s["validation_notes"],
            }
            for s in summary["series_summaries"]
        }
        print(json.dumps(compact, indent=2))
        return 0

    if args.command == "stage1-audit":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        t_start = time.monotonic()
        env_prof = resolve_environment_profile(data_root=args.data_root)
        show_prog = not args.no_progress

        targets: list[tuple[str, Path, Path | None]] = []
        if args.series_dir:
            s_path = Path(args.series_dir).expanduser().resolve()
            i_path = Path(args.image_dir).expanduser().resolve() if args.image_dir else None
            targets.append((args.case_id or s_path.name, s_path, i_path))
        else:
            targets = _discover_local_case_series(
                env_prof.data_root / "raw" / "tcia", case_id=args.case_id
            )
            if args.image_dir and len(targets) == 1:
                cid, pdir, _ = targets[0]
                targets = [(cid, pdir, Path(args.image_dir).expanduser().resolve())]

        if not targets and not args.run_phantom_check:
            raise SystemExit(
                f"No projection series found under {env_prof.data_root / 'raw' / 'tcia'}. "
                "Provide --series-dir or pass --run-phantom-check."
            )

        audited_series: list[dict[str, Any]] = []
        all_pass = True

        for idx_t, (cid, pdir, idir) in enumerate(targets):
            geom, files = scan_projection_series(
                pdir, workers=args.workers, show_progress=show_prog
            )
            orient = verify_projection_orientation(
                files, geometry=geom, workers=min(args.workers, 4)
            )
            if not orient["overall_pass"]:
                all_pass = False

            entry: dict[str, Any] = {
                "case_id": cid,
                "series_dir": str(pdir),
                "geometry": geom.summary_dict(),
                "orientation_verification": orient,
            }

            if args.memmap_out and idx_t == 0:
                _, mm = load_projections_memmap(
                    pdir,
                    args.memmap_out,
                    workers=min(args.workers, 4),
                    show_progress=show_prog,
                )
                entry["memmap"] = {
                    "path": str(Path(args.memmap_out).expanduser().resolve()),
                    "shape": list(mm.shape),
                    "dtype": str(mm.dtype),
                }
                del mm

            if idir is not None and idir.exists():
                vol = read_ct_image_volume(idir, water_mu_mm_inv=geom.water_mu_mm_inv)
                uid_linked = vol["referenced_raw_series_uid"] == geom.series_instance_uid
                entry["published_image_sanity"] = {
                    "image_dir": str(idir),
                    "series_instance_uid": vol["series_instance_uid"],
                    "referenced_raw_series_uid": vol["referenced_raw_series_uid"],
                    "deterministic_uid_link_verified": uid_linked,
                    "convolution_kernel": vol["convolution_kernel"],
                    "shape_zyx": list(vol["shape_zyx"]),
                    "z_range_mm": [
                        round(float(vol["z_mm"][0]), 4),
                        round(float(vol["z_mm"][-1]), 4),
                    ],
                    "axial_span_mm": round(float(vol["z_mm"][-1] - vol["z_mm"][0]), 4),
                    "hu_min_max": [
                        round(float(np.min(vol["hu_volume"])), 2),
                        round(float(np.max(vol["hu_volume"])), 2),
                    ],
                    "mu_min_max_mm_inv": [
                        round(float(np.min(vol["mu_volume"])), 6),
                        round(float(np.max(vol["mu_volume"])), 6),
                    ],
                    "role_warning": vol["role_warning"],
                }
            audited_series.append(entry)

        phantom_report = None
        if args.run_phantom_check:
            phantom_report = _run_synthetic_phantom_roundtrip(workers=min(args.workers, 4))
            if not phantom_report["pass"]:
                all_pass = False

        report = {
            "stage": "Stage 1 (v0.4.0)",
            "overall_pass": all_pass,
            "elapsed_seconds": round(time.monotonic() - t_start, 3),
            "audited_series_count": len(audited_series),
            "audited_series": audited_series,
            "synthetic_phantom_roundtrip": phantom_report,
        }
        out_json = env_prof.data_root / "manifests" / "stage1_audit_report.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_json"] = str(out_json)
        print(json.dumps(report, indent=2))
        return 0 if all_pass else 1

    # download command
    specs = read_series_manifest(args.series_file)
    if not specs:
        raise SystemExit("No enabled series in the manifest")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    cleanup_override: bool | None = None
    if args.cleanup_zip:
        cleanup_override = True
    elif args.keep_zip:
        cleanup_override = False

    env_prof = resolve_environment_profile(
        profile=args.profile,
        data_root=args.data_root,
        incoming_root=args.incoming_root,
        max_disk_gb=args.max_disk_gb,
        cleanup_zip=cleanup_override,
    )
    summary = download_manifest_with_profile(
        specs,
        env_prof,
        workers=args.workers,
        retries=args.retries,
        timeout=args.timeout,
        extract=not args.no_extract,
        show_progress=not args.no_progress,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 1 if summary["failed_series"] > 0 else 0
