"""Command-line interface for Stage 0, Stage 1, and Stage 2 of HighPitchCT-Pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from .ctpd_reader import (
    _extract_raw_matrix_fortran,
    load_projections_memmap,
    read_ct_image_volume,
    read_single_projection,
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
from .rebinning import (
    project_phantom_planar_fanbeam_2d,
    rebin_curved_to_flat,
    rebin_curved_to_flat_scalar_ref,
    rebin_helical_ssrb,
    rebin_helical_ssrb_scalar_ref,
    rebin_series_ssrb_streaming,
)
from .runtime import print_runtime_report, resolve_environment_profile
from .tcia_client import DEFAULT_COLLECTION, discover_tcia_series, download_manifest_with_profile
from .visual_qa import (
    save_stage1_phantom_visual_qa,
    save_stage1_real_visual_qa,
    save_stage2_phantom_ssrb_visual_qa,
    save_stage2_real_ssrb_visual_qa,
)


def _discover_local_case_series(
    tcia_root: Path, case_id: str | None = None
) -> list[tuple[str, Path, Path | None]]:
    """Discover downloaded (case_id, projection_dir, optional_image_dir) under raw/tcia.

    Matches both `C012` and `case_C012` case-insensitively so CLI `--case-id C012` works
    seamlessly with manifests whose `case_id` column is `case_C012`.
    """
    if not tcia_root.exists():
        return []

    all_dirs = sorted(p for p in tcia_root.iterdir() if p.is_dir())
    if case_id:
        raw_tok = case_id.strip()
        bare_tok = raw_tok[5:] if raw_tok.lower().startswith("case_") else raw_tok
        accepted_names = {
            raw_tok.lower(),
            bare_tok.lower(),
            f"case_{bare_tok.lower()}",
        }
        case_dirs = [p for p in all_dirs if p.name.lower() in accepted_names]
    else:
        case_dirs = all_dirs

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


def _generate_stage1_real_qa_png(
    out_png: Path,
    *,
    case_id: str,
    files: list[Path],
    n_rows: int,
    n_cols: int,
    vendor_vol: dict[str, Any] | None,
) -> Path:
    import pydicom

    mid_idx = len(files) // 2
    ds_mid = pydicom.dcmread(files[mid_idx], force=False)
    slope = float(getattr(ds_mid, "RescaleSlope", 1.0))
    intercept = float(getattr(ds_mid, "RescaleIntercept", 0.0))
    proj_f = read_single_projection(files[mid_idx], expected_rows=n_rows, expected_cols=n_cols)
    proj_c = (
        _extract_raw_matrix_fortran(ds_mid, n_rows=n_rows, n_cols=n_cols, order="C").astype(
            np.float32
        )
        * np.float32(slope)
        + np.float32(intercept)
    )

    sample_idx = np.linspace(0, len(files) - 1, num=min(96, len(files)), dtype=int)
    scout = np.empty((sample_idx.size, n_cols), dtype=np.float32)
    mid_r = n_rows // 2
    for k, vi in enumerate(sample_idx):
        scout[k] = read_single_projection(
            files[int(vi)], expected_rows=n_rows, expected_cols=n_cols
        )[mid_r]

    v_slice = None
    v_z = None
    if vendor_vol is not None:
        hu_v = vendor_vol["hu_volume"]
        mid_z_idx = hu_v.shape[0] // 2
        v_slice = hu_v[mid_z_idx]
        v_z = float(vendor_vol["z_mm"][mid_z_idx])

    return save_stage1_real_visual_qa(
        out_png,
        case_id=case_id,
        proj_fortran=proj_f,
        proj_c_order_corrupted=proj_c,
        scout_sinogram=scout,
        vendor_mid_slice_hu=v_slice,
        vendor_mid_z_mm=v_z,
    )


def _run_synthetic_phantom_roundtrip(
    workers: int = 4,
    *,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
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

    qa_png_str: str | None = None
    if figures_dir is not None:
        qa_png = save_stage1_phantom_visual_qa(
            figures_dir / "stage1_phantom_visual_qa.png",
            hu_volume=vox["hu_volume"],
            projections=proj_gt,
        )
        qa_png_str = str(qa_png)

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
        "visual_qa_png": qa_png_str,
    }


def _run_stage2_synthetic_phantom_check(
    *,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Stage 2 synthetic verification: scalar parity + exact 2D GT comparison + high-pitch mask."""
    t0 = time.monotonic()
    phantom = build_thorax_phantom(axial_scale=0.6)

    # 1. Scalar vs Vectorized parity on small FFSZ geometry
    geom_small = make_synthetic_helical_geometry(
        n_views=48,
        n_rows=16,
        n_cols=48,
        angular_steps_per_rotation=24,
        normalized_pitch=0.90,
        detector_shape="CYLINDRICAL",
        ffs_mode="FFSZ",
        rotation_direction=-1,
        quarter_detector_offset_cols=1.125,
    )
    proj_small = project_phantom_helical(phantom, geom_small, batch_views=48, show_progress=False)
    flat_vec, mask_vec = rebin_curved_to_flat(proj_small, geom_small)
    flat_ref, mask_ref = rebin_curved_to_flat_scalar_ref(proj_small, geom_small)
    c2f_mask_exact = bool(np.array_equal(mask_vec, mask_ref))
    c2f_max_diff = float(np.max(np.abs(flat_vec - flat_ref)))

    ssrb_vec = rebin_helical_ssrb(flat_vec, mask_vec, geom_small, [0.0], angular_mode="short_scan")
    ssrb_ref = rebin_helical_ssrb_scalar_ref(
        flat_vec, mask_vec, geom_small, [0.0], angular_mode="short_scan"
    )
    ssrb_mask_exact = bool(np.array_equal(ssrb_vec.measured_mask, ssrb_ref.measured_mask))
    ssrb_max_diff = float(np.max(np.abs(ssrb_vec.sinograms - ssrb_ref.sinograms)))

    # 2. 3D Helical FFSZ (pitch=0.85) vs Exact 2D Planar Fan-Beam Ground Truth at z = 0 mm
    geom_low = make_synthetic_helical_geometry(
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
    proj_low = project_phantom_helical(phantom, geom_low, batch_views=64, show_progress=False)
    flat_low, valid_low = rebin_curved_to_flat(proj_low, geom_low)
    ssrb_low = rebin_helical_ssrb(
        flat_low, valid_low, geom_low, [0.0], angular_mode="short_scan"
    )
    exact_2d = project_phantom_planar_fanbeam_2d(
        phantom,
        z_slices_mm=[0.0],
        beta_rad=ssrb_low.beta_rad,
        n_u=ssrb_low.n_u,
        du_mm=ssrb_low.du_mm,
        rho0_mm=ssrb_low.rho0_mm,
        d0_mm=ssrb_low.d0_mm,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
    )
    abs_err_2d = np.abs(ssrb_low.sinograms[0] - exact_2d[0])
    mae_vs_2d_gt = float(np.mean(abs_err_2d[ssrb_low.measured_mask[0]]))
    rmse_vs_2d_gt = float(np.sqrt(np.mean((abs_err_2d[ssrb_low.measured_mask[0]]) ** 2)))
    low_meas_frac = float(ssrb_low.overall_measured_fraction)

    # 3. High-Pitch (pitch=1.85) missing-data wedge & strict zero-fill check
    geom_high = make_synthetic_helical_geometry(
        n_views=384,
        n_rows=32,
        n_cols=128,
        angular_steps_per_rotation=192,
        normalized_pitch=1.85,
        detector_shape="CYLINDRICAL",
        ffs_mode="FFSZ",
        rotation_direction=-1,
        quarter_detector_offset_cols=1.125,
    )
    proj_high = project_phantom_helical(phantom, geom_high, batch_views=64, show_progress=False)
    flat_high, valid_high = rebin_curved_to_flat(proj_high, geom_high)
    ssrb_high = rebin_helical_ssrb(
        flat_high, valid_high, geom_high, [0.0], angular_mode="short_scan"
    )
    high_meas_frac = float(ssrb_high.overall_measured_fraction)
    unmeasured_strictly_zero = bool(
        np.all(ssrb_high.sinograms[~ssrb_high.measured_mask] == 0.0)
    )

    qa_png_str: str | None = None
    if figures_dir is not None:
        qa_png = save_stage2_phantom_ssrb_visual_qa(
            figures_dir / "stage2_phantom_ssrb_visual_qa.png",
            exact_planar_sino=exact_2d[0],
            ssrb_low_pitch_sino=ssrb_low.sinograms[0],
            ssrb_low_pitch_mask=ssrb_low.measured_mask[0],
            ssrb_high_pitch_sino=ssrb_high.sinograms[0],
            ssrb_high_pitch_mask=ssrb_high.measured_mask[0],
            z_slice_mm=0.0,
            low_pitch=0.85,
            high_pitch=1.85,
        )
        qa_png_str = str(qa_png)

    overall_pass = bool(
        c2f_mask_exact
        and c2f_max_diff < 1.0e-5
        and ssrb_mask_exact
        and ssrb_max_diff < 1.0e-5
        and low_meas_frac >= 0.98
        and mae_vs_2d_gt < 0.025
        and 0.45 <= high_meas_frac <= 0.92
        and unmeasured_strictly_zero
    )
    return {
        "pass": overall_pass,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "scalar_parity": {
            "curved_to_flat_mask_exact": c2f_mask_exact,
            "curved_to_flat_max_abs_diff": round(c2f_max_diff, 8),
            "ssrb_mask_exact": ssrb_mask_exact,
            "ssrb_max_abs_diff": round(ssrb_max_diff, 8),
        },
        "planar_2d_gt_comparison_pitch_0p85": {
            "measured_fraction": round(low_meas_frac, 6),
            "mae_line_integral": round(mae_vs_2d_gt, 6),
            "rmse_line_integral": round(rmse_vs_2d_gt, 6),
        },
        "high_pitch_1p85_mask_verification": {
            "measured_fraction": round(high_meas_frac, 6),
            "missing_fraction": round(1.0 - high_meas_frac, 6),
            "unmeasured_strictly_zero": unmeasured_strictly_zero,
        },
        "visual_qa_png": qa_png_str,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpct",
        description="HighPitchCT CLI (Stage 0 download/index, Stage 1 geometry/reader audit, Stage 2 SSRB)",
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
        help="Stage 1 physical audit: 3D helical geometry, FFS, orientation proofs, phantom & visual QA",
    )
    stage1.add_argument("--data-root", help="Writable root containing raw/tcia and manifests/")
    stage1.add_argument(
        "--case-id",
        help="Optional patient case filter under raw/tcia (matches C012 or case_C012)",
    )
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

    stage2 = sub.add_parser(
        "stage2-rebin",
        help="Stage 2 Curved-to-Flat & Helical SSRB with 3D FFS, explicit measured_mask, and Visual QA",
    )
    stage2.add_argument("--data-root", help="Writable root containing raw/tcia and manifests/")
    stage2.add_argument(
        "--case-id",
        help="Optional patient case filter under raw/tcia (matches C012 or case_C012)",
    )
    stage2.add_argument("--series-dir", help="Explicit DICOM-CT-PD projection directory to rebin")
    stage2.add_argument("--image-dir", help="Optional published CT image directory for Visual QA")
    stage2.add_argument(
        "--num-slices",
        type=int,
        default=5,
        help="Number of interior axial z slices to rebin (default: 5)",
    )
    stage2.add_argument(
        "--angular-mode",
        choices=["short_scan", "full_scan"],
        default="short_scan",
        help="Target planar sinogram angular span: short_scan (pi + 2*gamma_max) or full_scan (2*pi)",
    )
    stage2.add_argument(
        "--sinograms-out",
        help="Optional compressed .npz path to save rebinned sinograms, measured_mask, and geometry grids",
    )
    stage2.add_argument(
        "--run-phantom-check",
        action="store_true",
        help="Run Stage 2 synthetic verification (scalar parity + exact 2D GT comparison + high-pitch mask)",
    )
    stage2.add_argument("--workers", type=int, default=8)
    stage2.add_argument("--no-progress", action="store_true")

    environment = sub.add_parser(
        "env", help="Report Kaggle/Full profile, disk quota, RAM, disk space, and GPUs"
    )
    environment.add_argument("--data-root")
    environment.add_argument("--profile", choices=["auto", "kaggle", "full"], default=None)
    environment.add_argument("--max-disk-gb", type=float, default=None)
    return parser


def _resolve_cli_series_targets(
    tcia_root: Path,
    *,
    case_id: str | None,
    series_dir: str | None,
    image_dir: str | None,
    allow_empty_for_phantom: bool,
) -> list[tuple[str, Path, Path | None]]:
    """Resolve local projection series targets with strict fail-fast on missing explicit --case-id."""
    if series_dir:
        s_path = Path(series_dir).expanduser().resolve()
        i_path = Path(image_dir).expanduser().resolve() if image_dir else None
        return [(case_id or s_path.name, s_path, i_path)]

    targets = _discover_local_case_series(tcia_root, case_id=case_id)
    if case_id and not targets:
        existing = (
            sorted(p.name for p in tcia_root.iterdir() if p.is_dir())
            if tcia_root.exists()
            else []
        )
        raise SystemExit(
            f"Explicit --case-id {case_id!r} matched zero local projection series under {tcia_root} "
            f"(available case folders: {existing}). Refusing silent fallback."
        )
    if image_dir and len(targets) == 1:
        cid, pdir, _ = targets[0]
        targets = [(cid, pdir, Path(image_dir).expanduser().resolve())]

    if not targets and not allow_empty_for_phantom:
        raise SystemExit(
            f"No projection series found under {tcia_root}. "
            "Provide --series-dir, download data first, or pass --run-phantom-check."
        )
    return targets


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
        figures_dir = env_prof.data_root / "manifests" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        targets = _resolve_cli_series_targets(
            env_prof.data_root / "raw" / "tcia",
            case_id=args.case_id,
            series_dir=args.series_dir,
            image_dir=args.image_dir,
            allow_empty_for_phantom=bool(args.run_phantom_check),
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

            vol_data: dict[str, Any] | None = None
            if idir is not None and idir.exists():
                vol_data = read_ct_image_volume(idir, water_mu_mm_inv=geom.water_mu_mm_inv)
                uid_linked = vol_data["referenced_raw_series_uid"] == geom.series_instance_uid
                entry["published_image_sanity"] = {
                    "image_dir": str(idir),
                    "series_instance_uid": vol_data["series_instance_uid"],
                    "referenced_raw_series_uid": vol_data["referenced_raw_series_uid"],
                    "deterministic_uid_link_verified": uid_linked,
                    "convolution_kernel": vol_data["convolution_kernel"],
                    "shape_zyx": list(vol_data["shape_zyx"]),
                    "z_range_mm": [
                        round(float(vol_data["z_mm"][0]), 4),
                        round(float(vol_data["z_mm"][-1]), 4),
                    ],
                    "axial_span_mm": round(float(vol_data["z_mm"][-1] - vol_data["z_mm"][0]), 4),
                    "hu_min_max": [
                        round(float(np.min(vol_data["hu_volume"])), 2),
                        round(float(np.max(vol_data["hu_volume"])), 2),
                    ],
                    "mu_min_max_mm_inv": [
                        round(float(np.min(vol_data["mu_volume"])), 6),
                        round(float(np.max(vol_data["mu_volume"])), 6),
                    ],
                    "role_warning": vol_data["role_warning"],
                }

            qa_png = _generate_stage1_real_qa_png(
                figures_dir / f"stage1_{cid}_visual_qa.png",
                case_id=cid,
                files=files,
                n_rows=geom.n_rows,
                n_cols=geom.n_cols,
                vendor_vol=vol_data,
            )
            entry["visual_qa_png"] = str(qa_png)
            audited_series.append(entry)

        phantom_report = None
        if args.run_phantom_check:
            phantom_report = _run_synthetic_phantom_roundtrip(
                workers=min(args.workers, 4), figures_dir=figures_dir
            )
            if not phantom_report["pass"]:
                all_pass = False

        report = {
            "stage": "Stage 1 (v0.5.0)",
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

    if args.command == "stage2-rebin":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        if args.num_slices < 1:
            raise SystemExit("--num-slices must be at least 1")

        t_start = time.monotonic()
        env_prof = resolve_environment_profile(data_root=args.data_root)
        show_prog = not args.no_progress
        figures_dir = env_prof.data_root / "manifests" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        targets = _resolve_cli_series_targets(
            env_prof.data_root / "raw" / "tcia",
            case_id=args.case_id,
            series_dir=args.series_dir,
            image_dir=args.image_dir,
            allow_empty_for_phantom=bool(args.run_phantom_check),
        )

        rebinned_series: list[dict[str, Any]] = []
        all_pass = True

        for idx_t, (cid, pdir, idir) in enumerate(targets):
            geom, files = scan_projection_series(
                pdir, workers=args.workers, show_progress=show_prog
            )
            traj = geom.trajectory_metrics()
            native_pitch = float(traj["normalized_pitch"])
            z_min_raw = float(np.min(geom.z0_mm))
            z_max_raw = float(np.max(geom.z0_mm))
            margin_mm = min(
                1.5 * geom.coverage_isocenter_mm,
                0.25 * max(0.0, z_max_raw - z_min_raw),
            )
            if args.num_slices == 1:
                z_slices = np.array([0.5 * (z_min_raw + z_max_raw)], dtype=np.float64)
            else:
                z_slices = np.linspace(
                    z_min_raw + margin_mm,
                    z_max_raw - margin_mm,
                    args.num_slices,
                    dtype=np.float64,
                )

            ssrb_main = rebin_series_ssrb_streaming(
                files,
                z_slices,
                geometry=geom,
                angular_mode=args.angular_mode,
                workers=min(args.workers, 6),
                show_progress=show_prog,
            )

            # Diagnostic mid-slice comparisons: Short-Scan vs Full-Scan vs High-Pitch RDC (50% rows)
            mid_slice_idx = len(z_slices) // 2
            z_mid = float(z_slices[mid_slice_idx])

            ssrb_short_mid = (
                ssrb_main
                if args.angular_mode == "short_scan"
                else rebin_series_ssrb_streaming(
                    files,
                    [z_mid],
                    geometry=geom,
                    angular_mode="short_scan",
                    workers=min(args.workers, 6),
                    show_progress=False,
                )
            )
            short_sino_mid = (
                ssrb_main.sinograms[mid_slice_idx]
                if args.angular_mode == "short_scan"
                else ssrb_short_mid.sinograms[0]
            )
            short_mask_mid = (
                ssrb_main.measured_mask[mid_slice_idx]
                if args.angular_mode == "short_scan"
                else ssrb_short_mid.measured_mask[0]
            )

            ssrb_full_mid = (
                ssrb_main
                if args.angular_mode == "full_scan"
                else rebin_series_ssrb_streaming(
                    files,
                    [z_mid],
                    geometry=geom,
                    angular_mode="full_scan",
                    workers=min(args.workers, 6),
                    show_progress=False,
                )
            )
            full_sino_mid = (
                ssrb_main.sinograms[mid_slice_idx]
                if args.angular_mode == "full_scan"
                else ssrb_full_mid.sinograms[0]
            )
            full_mask_mid = (
                ssrb_main.measured_mask[mid_slice_idx]
                if args.angular_mode == "full_scan"
                else ssrb_full_mid.measured_mask[0]
            )

            r_lo_rdc = float(geom.n_rows // 4)
            r_hi_rdc = float(geom.n_rows - (geom.n_rows // 4) - 1)
            active_rows_rdc = max(1.0, r_hi_rdc - r_lo_rdc + 1.0)
            rdc_eff_pitch = native_pitch * (float(geom.n_rows) / active_rows_rdc)
            ssrb_rdc_mid = rebin_series_ssrb_streaming(
                files,
                [z_mid],
                geometry=geom,
                angular_mode="short_scan",
                active_row_range=(r_lo_rdc, r_hi_rdc),
                workers=min(args.workers, 6),
                show_progress=False,
            )

            # Single mid-view Curved->Flat diagnostic for Visual QA
            mid_v_idx = int(np.argmin(np.abs(geom.z0_mm - z_mid)))
            raw_mid_view = read_single_projection(
                files[mid_v_idx], expected_rows=geom.n_rows, expected_cols=geom.n_cols
            )
            flat_mid_arr, valid_mid_arr = rebin_curved_to_flat(
                raw_mid_view[None, :, :], geom.slice_views(mid_v_idx, mid_v_idx + 1)
            )

            vendor_slice_hu: np.ndarray | None = None
            if idir is not None and idir.exists():
                vol_data = read_ct_image_volume(idir, water_mu_mm_inv=geom.water_mu_mm_inv)
                z_vol = vol_data["z_mm"]
                if float(z_vol[0]) <= z_mid <= float(z_vol[-1]):
                    v_idx_z = int(np.argmin(np.abs(z_vol - z_mid)))
                else:
                    frac = (z_mid - z_min_raw) / max(z_max_raw - z_min_raw, 1e-6)
                    v_idx_z = int(np.clip(round(frac * (z_vol.size - 1)), 0, z_vol.size - 1))
                vendor_slice_hu = vol_data["hu_volume"][v_idx_z]

            qa_png = save_stage2_real_ssrb_visual_qa(
                figures_dir / f"stage2_{cid}_ssrb_visual_qa.png",
                case_id=cid,
                raw_curved_view=raw_mid_view,
                rebinned_flat_view=flat_mid_arr[0],
                rebinned_flat_valid=valid_mid_arr[0],
                z_slice_mm=z_mid,
                native_pitch=native_pitch,
                short_scan_sino=short_sino_mid,
                short_scan_mask=short_mask_mid,
                full_scan_sino=full_sino_mid,
                full_scan_mask=full_mask_mid,
                rdc_high_pitch_sino=ssrb_rdc_mid.sinograms[0],
                rdc_high_pitch_mask=ssrb_rdc_mid.measured_mask[0],
                rdc_effective_pitch=rdc_eff_pitch,
                vendor_slice_hu=vendor_slice_hu,
            )

            unmeasured_zero = bool(
                np.all(ssrb_main.sinograms[~ssrb_main.measured_mask] == 0.0)
            )
            short_mid_frac = float(np.mean(short_mask_mid))
            rdc_mid_frac = float(ssrb_rdc_mid.overall_measured_fraction)
            series_pass = bool(
                unmeasured_zero
                and (short_mid_frac >= 0.98 if native_pitch <= 1.05 else short_mid_frac > 0.30)
                and (rdc_mid_frac < short_mid_frac if rdc_eff_pitch > 1.35 else True)
            )
            if not series_pass:
                all_pass = False

            entry = {
                "case_id": cid,
                "series_dir": str(pdir),
                "native_normalized_pitch": round(native_pitch, 5),
                "ffs_mode": geom.ffs_mode,
                "series_pass": series_pass,
                "unmeasured_strictly_zero": unmeasured_zero,
                "ssrb_summary": ssrb_main.summary_dict(),
                "mid_slice_diagnostics": {
                    "z_mid_mm": round(z_mid, 4),
                    "short_scan_measured_fraction": round(short_mid_frac, 6),
                    "full_scan_measured_fraction": round(float(np.mean(full_mask_mid)), 6),
                    "rdc_50pct_rows_range": [r_lo_rdc, r_hi_rdc],
                    "rdc_effective_pitch": round(rdc_eff_pitch, 5),
                    "rdc_short_scan_measured_fraction": round(rdc_mid_frac, 6),
                    "rdc_short_scan_missing_fraction": round(1.0 - rdc_mid_frac, 6),
                },
                "visual_qa_png": str(qa_png),
            }

            if args.sinograms_out and idx_t == 0:
                s_out = Path(args.sinograms_out).expanduser().resolve()
                s_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    s_out,
                    sinograms=ssrb_main.sinograms,
                    measured_mask=ssrb_main.measured_mask,
                    z_slices_mm=ssrb_main.z_slices_mm,
                    beta_rad=ssrb_main.beta_rad,
                    phi_center_rad=ssrb_main.phi_center_rad,
                    u_coords_mm=ssrb_main.u_coords_mm,
                    du_mm=np.array(ssrb_main.du_mm, dtype=np.float64),
                    rho0_mm=np.array(ssrb_main.rho0_mm, dtype=np.float64),
                    d0_mm=np.array(ssrb_main.d0_mm, dtype=np.float64),
                    gamma_max_rad=np.array(ssrb_main.gamma_max_rad, dtype=np.float64),
                    angular_mode=np.array(ssrb_main.angular_mode, dtype=np.str_),
                )
                entry["sinograms_npz"] = str(s_out)

            rebinned_series.append(entry)

        phantom_stage2 = None
        if args.run_phantom_check:
            phantom_stage2 = _run_stage2_synthetic_phantom_check(figures_dir=figures_dir)
            if not phantom_stage2["pass"]:
                all_pass = False

        report = {
            "stage": "Stage 2 (v0.5.0)",
            "overall_pass": all_pass,
            "elapsed_seconds": round(time.monotonic() - t_start, 3),
            "rebinned_series_count": len(rebinned_series),
            "rebinned_series": rebinned_series,
            "synthetic_phantom_ssrb_check": phantom_stage2,
        }
        out_json = env_prof.data_root / "manifests" / "stage2_rebin_report.json"
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


if __name__ == "__main__":
    raise SystemExit(main())
