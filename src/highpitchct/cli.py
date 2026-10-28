"""Command-line interface for Stage 0, Stage 1, Stage 2, and Stage 3 of HighPitchCT-Pipeline."""

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
from .dual_domain_dl import (
    DEFAULT_CLINICAL_HIGH_PITCHES,
    DEFAULT_ITF_TRAINING_RATIO,
    build_hybrid_phantom_dataset,
    build_hybrid_series_dataset_streaming,
    create_dual_domain_model,
    save_dual_domain_checkpoint,
    save_dual_domain_dataset_npz,
    train_dual_domain_model,
)
from .forward_project import (
    build_dual_domain_pair_from_ssrb,
    compute_rdc_active_row_range,
    forward_project_fanbeam_2d,
    forward_project_fanbeam_2d_scalar_ref,
    forward_project_helical_3d,
    simulate_series_high_pitch_streaming,
    synthesize_itf_helical_geometry,
)
from .generalization import (
    build_multi_phantom_generalization_split,
    build_real_series_generalization_split,
    run_generalization_experiment,
)
from .index_dicom import index_dicom_tree
from .manifest import read_series_manifest, write_series_manifest
from .phantom import (
    Ellipsoid3D,
    build_thorax_phantom,
    make_synthetic_helical_geometry,
    project_phantom_helical,
    voxelize_phantom,
    voxelize_phantom_slices_radiological,
    write_synthetic_ctpd_series,
)
from .rebinning import (
    SSRBResult,
    project_phantom_planar_fanbeam_2d,
    rebin_curved_to_flat,
    rebin_curved_to_flat_scalar_ref,
    rebin_helical_ssrb,
    rebin_helical_ssrb_scalar_ref,
    rebin_series_ssrb_streaming,
)
from .recon_fbp import (
    compute_reconstruction_metrics,
    normalize_filter_name,
    reconstruct_fanbeam_fbp,
    reconstruct_fanbeam_fbp_scalar_ref,
    reconstruct_series_reference_streaming,
)
from .runtime import print_runtime_report, resolve_environment_profile
from .tcia_client import DEFAULT_COLLECTION, discover_tcia_series, download_manifest_with_profile
from .visual_qa import (
    save_stage1_phantom_visual_qa,
    save_stage1_real_visual_qa,
    save_stage2_phantom_ssrb_visual_qa,
    save_stage2_real_ssrb_visual_qa,
    save_stage3_phantom_fbp_visual_qa,
    save_stage3_real_fbp_visual_qa,
    save_stage4_phantom_forward_visual_qa,
    save_stage4_real_simulation_visual_qa,
    save_stage5_dual_domain_visual_qa,
    save_stage6_generalization_visual_qa,
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


def _run_stage3_synthetic_phantom_check(
    *,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Stage 3 synthetic verification: 0-HU water calibration, scalar parity, and Thorax FBP QA."""
    t0 = time.monotonic()

    # 1. Exact uniform 0 HU water cylinder calibration (zero-bias spatial Ram-Lak proof)
    water_cyl = [Ellipsoid3D("water_cylinder", (0.0, 0.0, 0.0), (110.0, 110.0, 220.0), 1000.0, 0.0)]
    geom_w = make_synthetic_helical_geometry(
        n_views=576,
        n_rows=24,
        n_cols=256,
        angular_steps_per_rotation=384,
        normalized_pitch=0.85,
        detector_shape="CYLINDRICAL",
        ffs_mode="FFSZ",
        rotation_direction=-1,
        quarter_detector_offset_cols=1.125,
        du_mm=2.0,
        dv_mm=2.0,
    )
    proj_w = project_phantom_helical(water_cyl, geom_w, batch_views=128, show_progress=False)
    flat_w, valid_w = rebin_curved_to_flat(proj_w, geom_w)
    ssrb_w = rebin_helical_ssrb(flat_w, valid_w, geom_w, [0.0], angular_mode="short_scan")
    fbp_w = reconstruct_fanbeam_fbp(
        ssrb_w,
        image_size=192,
        fov_mm=340.0,
        water_mu_mm_inv=geom_w.water_mu_mm_inv,
        filter_name="ram_lak",
    )
    yy_w, xx_w = np.meshgrid(fbp_w.y_mm, fbp_w.x_mm, indexing="ij")
    roi_water = (xx_w * xx_w + yy_w * yy_w) <= (85.0 ** 2)
    water_mean_hu = float(np.mean(fbp_w.hu_slices[0][roi_water]))
    water_std_hu = float(np.std(fbp_w.hu_slices[0][roi_water]))

    # 2. Vectorized vs Scalar FBP parity on 32 x 32 grid
    fbp_vec_small = reconstruct_fanbeam_fbp(
        ssrb_w,
        image_size=32,
        fov_mm=340.0,
        water_mu_mm_inv=geom_w.water_mu_mm_inv,
        filter_name="ram_lak",
        lateral_pad_cols=32,
    )
    fbp_ref_small = reconstruct_fanbeam_fbp_scalar_ref(
        ssrb_w,
        image_size=32,
        fov_mm=340.0,
        water_mu_mm_inv=geom_w.water_mu_mm_inv,
        filter_name="ram_lak",
        lateral_pad_cols=32,
    )
    scalar_max_hu_diff = float(np.max(np.abs(fbp_vec_small.hu_slices - fbp_ref_small.hu_slices)))
    scalar_max_mu_diff = float(np.max(np.abs(fbp_vec_small.mu_slices - fbp_ref_small.mu_slices)))

    # 3. 3D Asymmetric Thorax Phantom: Reference (pitch=0.85) vs High-Pitch (pitch=1.85) FBP
    phantom = build_thorax_phantom(axial_scale=0.8)
    z_test = -12.0  # Intersects heart (+16, +32, -14.4), lungs, aorta, and posterior spine!
    geom_low = make_synthetic_helical_geometry(
        n_views=768,
        n_rows=32,
        n_cols=320,  # R_FOV=189.6 mm > thorax trunk 155 mm (no truncation)
        angular_steps_per_rotation=384,
        normalized_pitch=0.85,
        detector_shape="CYLINDRICAL",
        ffs_mode="FFSZ",
        rotation_direction=-1,
        quarter_detector_offset_cols=1.125,
        du_mm=2.2,
        dv_mm=1.8,
        z_center_mm=z_test,
    )
    proj_low = project_phantom_helical(phantom, geom_low, batch_views=128, show_progress=False)
    flat_low, valid_low = rebin_curved_to_flat(proj_low, geom_low)
    ssrb_low = rebin_helical_ssrb(flat_low, valid_low, geom_low, [z_test], angular_mode="short_scan")
    fbp_ref = reconstruct_fanbeam_fbp(
        ssrb_low,
        image_size=256,
        fov_mm=360.0,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
        filter_name="ram_lak",
    )

    geom_high = make_synthetic_helical_geometry(
        n_views=768,
        n_rows=32,
        n_cols=320,  # R_FOV=189.6 mm > thorax trunk 155 mm (no truncation)
        angular_steps_per_rotation=384,
        normalized_pitch=1.85,
        detector_shape="CYLINDRICAL",
        ffs_mode="FFSZ",
        rotation_direction=-1,
        quarter_detector_offset_cols=1.125,
        du_mm=2.2,
        dv_mm=1.8,
        z_center_mm=z_test,
    )
    proj_high = project_phantom_helical(phantom, geom_high, batch_views=128, show_progress=False)
    flat_high, valid_high = rebin_curved_to_flat(proj_high, geom_high)
    ssrb_high = rebin_helical_ssrb(
        flat_high,
        valid_high,
        geom_high,
        [z_test],
        angular_mode="short_scan",
        active_row_range=(4.0, 27.0),
    )
    fbp_hp_raw = reconstruct_fanbeam_fbp(
        ssrb_high,
        image_size=256,
        fov_mm=360.0,
        water_mu_mm_inv=geom_high.water_mu_mm_inv,
        filter_name="ram_lak",
        inpaint_mode="none",
    )
    fbp_hp_inpaint = reconstruct_fanbeam_fbp(
        ssrb_high,
        image_size=256,
        fov_mm=360.0,
        water_mu_mm_inv=geom_high.water_mu_mm_inv,
        filter_name="ram_lak",
        inpaint_mode="conjugate_feather",
    )

    gt_vox = voxelize_phantom_slices_radiological(
        phantom,
        z_slices_mm=[z_test],
        image_size=256,
        fov_mm=360.0,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
    )
    gt_hu = gt_vox["hu_slices"][0]
    ref_metrics = compute_reconstruction_metrics(
        fbp_ref.hu_slices[0], gt_hu, fov_mask=fbp_ref.fov_mask
    )
    hp_raw_metrics = compute_reconstruction_metrics(
        fbp_hp_raw.hu_slices[0], gt_hu, fov_mask=fbp_ref.fov_mask
    )
    hp_inpaint_metrics = compute_reconstruction_metrics(
        fbp_hp_inpaint.hu_slices[0], gt_hu, fov_mask=fbp_ref.fov_mask
    )

    # Radiological orientation ROI sanity checks (+x Right, -x Left, +y Anterior, -y Posterior)
    yy_p, xx_p = np.meshgrid(fbp_ref.y_mm, fbp_ref.x_mm, indexing="ij")
    roi_heart = ((xx_p + 8.0) ** 2 + (yy_p - 36.0) ** 2) <= (10.0 ** 2)
    roi_right_lung = ((xx_p - 64.0) ** 2 + (yy_p - 8.0) ** 2) <= (22.0 ** 2)
    roi_spine = ((xx_p - 0.0) ** 2 + (yy_p + 68.0) ** 2) <= (8.0 ** 2)

    heart_hu = float(np.mean(fbp_ref.hu_slices[0][roi_heart]))
    right_lung_hu = float(np.mean(fbp_ref.hu_slices[0][roi_right_lung]))
    spine_hu = float(np.mean(fbp_ref.hu_slices[0][roi_spine]))
    orientation_rois_pass = bool(
        (25.0 <= heart_hu <= 85.0)
        and (-800.0 <= right_lung_hu <= -700.0)
        and (spine_hu >= 120.0)
    )

    qa_png_str: str | None = None
    if figures_dir is not None:
        qa_png = save_stage3_phantom_fbp_visual_qa(
            figures_dir / "stage3_phantom_fbp_visual_qa.png",
            gt_hu=gt_hu,
            ref_fbp_hu=fbp_ref.hu_slices[0],
            high_pitch_raw_fbp_hu=fbp_hp_raw.hu_slices[0],
            high_pitch_inpaint_fbp_hu=fbp_hp_inpaint.hu_slices[0],
            fov_mask=fbp_ref.fov_mask,
            fov_mm=fbp_ref.fov_mm,
            z_slice_mm=z_test,
            low_pitch=0.85,
            high_pitch=1.85,
            ref_metrics=ref_metrics,
            hp_raw_metrics=hp_raw_metrics,
            hp_inpaint_metrics=hp_inpaint_metrics,
        )
        qa_png_str = str(qa_png)

    overall_pass = bool(
        abs(water_mean_hu) < 2.0
        and water_std_hu < 4.0
        and scalar_max_hu_diff < 0.05
        and orientation_rois_pass
        and ref_metrics["ssim"] >= 0.88
        and hp_inpaint_metrics["rmse_hu"] < hp_raw_metrics["rmse_hu"]
    )
    return {
        "pass": overall_pass,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "filter_name": "ram_lak",
        "water_cylinder_calibration": {
            "mean_hu": round(water_mean_hu, 4),
            "std_hu": round(water_std_hu, 4),
            "pass": bool(abs(water_mean_hu) < 2.0 and water_std_hu < 4.0),
        },
        "scalar_parity": {
            "max_abs_hu_diff": round(scalar_max_hu_diff, 6),
            "max_abs_mu_diff_mm_inv": round(scalar_max_mu_diff, 9),
            "pass": bool(scalar_max_hu_diff < 0.05),
        },
        "radiological_orientation_rois_hu": {
            "heart_anterior_left_hu": round(heart_hu, 2),
            "right_lung_minus_x_hu": round(right_lung_hu, 2),
            "spine_posterior_minus_y_hu": round(spine_hu, 2),
            "pass": orientation_rois_pass,
        },
        "thorax_phantom_pitch_0p85_vs_gt": ref_metrics,
        "thorax_phantom_pitch_1p85_raw_no_fill_vs_gt": hp_raw_metrics,
        "thorax_phantom_pitch_1p85_conjugate_fill_vs_gt": hp_inpaint_metrics,
        "visual_qa_png": qa_png_str,
    }


def _run_stage4_synthetic_phantom_check(
    *,
    figures_dir: Path | None = None,
    filter_name: str = "ram_lak",
    backend: str = "numpy",
) -> dict[str, Any]:
    """Run Stage 4 synthetic verification: 2D scalar parity, Forward->FBP roundtrip, and RDC/ITF simulation."""
    t0 = time.monotonic()
    fname = normalize_filter_name(filter_name)

    phantom = build_thorax_phantom(axial_scale=0.8)
    z_test = -12.0
    geom_low = make_synthetic_helical_geometry(
        n_views=768,
        n_rows=32,
        n_cols=320,
        angular_steps_per_rotation=384,
        normalized_pitch=0.85,
        detector_shape="CYLINDRICAL",
        ffs_mode="FFSZ",
        rotation_direction=-1,
        quarter_detector_offset_cols=1.125,
        du_mm=2.2,
        dv_mm=1.8,
        z_center_mm=z_test,
    )

    # 1. Level-2 Scalar Parity for 2D Fan-Beam Forward Projector on 32 x 32 slice
    gt_small = voxelize_phantom_slices_radiological(
        phantom,
        z_slices_mm=[z_test],
        image_size=32,
        fov_mm=360.0,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
    )
    beta_small = np.linspace(0.0, np.pi + 0.5, 24, dtype=np.float64)
    u_small = np.linspace(-180.0, 180.0, 48, dtype=np.float64)
    sino_vec_small = forward_project_fanbeam_2d(
        gt_small["mu_slices"][0],
        beta_small,
        u_small,
        rho0_mm=geom_low.rho0_mm,
        d0_mm=geom_low.d0_mm,
        fov_mm=360.0,
        backend="numpy",
    )
    sino_ref_small = forward_project_fanbeam_2d_scalar_ref(
        gt_small["mu_slices"][0],
        beta_small,
        u_small,
        rho0_mm=geom_low.rho0_mm,
        d0_mm=geom_low.d0_mm,
        fov_mm=360.0,
    )
    scalar_max_diff = float(np.max(np.abs(sino_vec_small - sino_ref_small)))

    # 2. Level-3 2D Forward -> Ram-Lak FBP Self-Consistency Round-Trip on 192 x 192 slice
    proj_low = project_phantom_helical(phantom, geom_low, batch_views=128, show_progress=False)
    flat_low, valid_low = rebin_curved_to_flat(proj_low, geom_low)
    ssrb_low = rebin_helical_ssrb(
        flat_low, valid_low, geom_low, [z_test], angular_mode="short_scan"
    )
    gt_vox = voxelize_phantom_slices_radiological(
        phantom,
        z_slices_mm=[z_test],
        image_size=192,
        fov_mm=360.0,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
    )
    gt_mu = gt_vox["mu_slices"]
    gt_hu = gt_vox["hu_slices"][0]

    sino_rt = forward_project_fanbeam_2d(
        gt_mu,
        ssrb_low.beta_rad,
        ssrb_low.u_coords_mm,
        rho0_mm=ssrb_low.rho0_mm,
        d0_mm=ssrb_low.d0_mm,
        fov_mm=360.0,
        backend=backend,
    )
    ssrb_rt = SSRBResult(
        sinograms=sino_rt,
        measured_mask=np.ones_like(sino_rt, dtype=bool),
        z_slices_mm=ssrb_low.z_slices_mm.copy(),
        beta_rad=ssrb_low.beta_rad.copy(),
        phi_center_rad=ssrb_low.phi_center_rad.copy(),
        u_coords_mm=ssrb_low.u_coords_mm.copy(),
        du_mm=ssrb_low.du_mm,
        rho0_mm=ssrb_low.rho0_mm,
        d0_mm=ssrb_low.d0_mm,
        gamma_max_rad=ssrb_low.gamma_max_rad,
        angular_mode=ssrb_low.angular_mode,
        angular_span_rad=ssrb_low.angular_span_rad,
        active_row_range=ssrb_low.active_row_range,
    )
    fbp_rt = reconstruct_fanbeam_fbp(
        ssrb_rt,
        image_size=192,
        fov_mm=360.0,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
        filter_name=fname,
        backend=backend,
    )
    rt_metrics = compute_reconstruction_metrics(
        fbp_rt.hu_slices[0], gt_hu, fov_mask=fbp_rt.fov_mask
    )

    # 3. High-Pitch RDC (target_pitch = 2.4) & ITF (target_pitch = 2.4 via 3D Helical Forward Projection)
    r_lo_rdc, r_hi_rdc, eff_rdc_pitch = compute_rdc_active_row_range(
        geom_low.n_rows, 0.85, 2.4
    )
    ssrb_rdc = rebin_helical_ssrb(
        flat_low,
        valid_low,
        geom_low,
        [z_test],
        angular_mode="short_scan",
        active_row_range=(r_lo_rdc, r_hi_rdc),
    )
    pair_rdc = build_dual_domain_pair_from_ssrb(
        ssrb_low,
        ssrb_rdc,
        scenario="rdc",
        target_pitch=2.4,
        effective_pitch=eff_rdc_pitch,
        native_pitch=0.85,
        active_row_range=(r_lo_rdc, r_hi_rdc),
        image_size=192,
        fov_mm=360.0,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
        filter_name=fname,
        backend=backend,
    )

    # 3D Helical Forward Projection for ITF (target_pitch = 2.4)
    z_slab = np.linspace(z_test - 35.0, z_test + 35.0, 9, dtype=np.float64)
    slab_vox = voxelize_phantom_slices_radiological(
        phantom,
        z_slices_mm=z_slab,
        image_size=128,
        fov_mm=360.0,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
    )
    geom_base_itf = make_synthetic_helical_geometry(
        n_views=384,
        n_rows=32,
        n_cols=160,
        angular_steps_per_rotation=192,
        normalized_pitch=0.85,
        detector_shape="CYLINDRICAL",
        ffs_mode="FFSZ",
        rotation_direction=-1,
        quarter_detector_offset_cols=1.125,
        du_mm=4.4,
        dv_mm=1.8,
        z_center_mm=z_test,
    )
    proj_base_itf = project_phantom_helical(
        phantom, geom_base_itf, batch_views=128, show_progress=False
    )
    flat_base_itf, valid_base_itf = rebin_curved_to_flat(proj_base_itf, geom_base_itf)
    ssrb_base_itf = rebin_helical_ssrb(
        flat_base_itf, valid_base_itf, geom_base_itf, [z_test], angular_mode="short_scan"
    )
    geom_itf = synthesize_itf_helical_geometry(
        geom_base_itf, 2.4, z_center_mm=z_test, z_span_mm=10.0, min_rotations=1.0
    )
    proj_itf = forward_project_helical_3d(
        slab_vox["mu_slices"],
        z_slab,
        slab_vox["y_mm"],
        slab_vox["x_mm"],
        geom_itf,
        step_mm=3.0,
        z_boundary_mode="clamp",
        backend=backend,
    )
    flat_itf, valid_itf = rebin_curved_to_flat(proj_itf, geom_itf)
    ssrb_itf = rebin_helical_ssrb(
        flat_itf, valid_itf, geom_itf, [z_test], angular_mode="short_scan"
    )
    eff_itf_pitch = float(geom_itf.trajectory_metrics()["normalized_pitch"])
    pair_itf = build_dual_domain_pair_from_ssrb(
        ssrb_base_itf,
        ssrb_itf,
        scenario="itf",
        target_pitch=2.4,
        effective_pitch=eff_itf_pitch,
        native_pitch=0.85,
        active_row_range=(0.0, float(geom_itf.n_rows - 1)),
        image_size=192,
        fov_mm=360.0,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
        filter_name=fname,
        backend=backend,
    )

    qa_png_str: str | None = None
    if figures_dir is not None:
        qa_png = save_stage4_phantom_forward_visual_qa(
            figures_dir / "stage4_phantom_forward_visual_qa.png",
            gt_hu=gt_hu,
            roundtrip_fbp_hu=fbp_rt.hu_slices[0],
            roundtrip_metrics=rt_metrics,
            s_hp_raw=pair_rdc.s_hp_raw[0],
            mask_hp=pair_rdc.mask_hp[0],
            s_reproj=pair_rdc.s_reproj[0],
            rdc_prelim_hu=pair_rdc.x_prelim_hu[0],
            rdc_metrics=pair_rdc.prelim_vs_target_metrics,
            rdc_pitch=pair_rdc.effective_pitch,
            itf_prelim_hu=pair_itf.x_prelim_hu[0],
            itf_metrics=pair_itf.prelim_vs_target_metrics,
            itf_pitch=pair_itf.effective_pitch,
            fov_mm=360.0,
            z_slice_mm=z_test,
            filter_name=fname,
        )
        qa_png_str = str(qa_png)

    overall_pass = bool(
        scalar_max_diff < 1e-5
        and rt_metrics["ssim"] >= 0.88
        and pair_rdc.prelim_vs_target_metrics["rmse_hu"] < pair_rdc.raw_nofill_vs_target_metrics["rmse_hu"]
        and pair_rdc.sinogram_metrics["reproj_missing_wedge_mae"]
        < pair_rdc.sinogram_metrics["raw_missing_wedge_mae"]
        and pair_itf.prelim_vs_target_metrics["rmse_hu"] < pair_itf.raw_nofill_vs_target_metrics["rmse_hu"]
    )
    return {
        "pass": overall_pass,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "filter_name": fname,
        "scalar_parity_2d_forward": {
            "max_abs_line_integral_diff": round(scalar_max_diff, 8),
            "pass": bool(scalar_max_diff < 1e-5),
        },
        "forward_to_fbp_roundtrip_vs_gt": rt_metrics,
        "rdc_simulation_summary": pair_rdc.summary_dict(),
        "itf_3d_helical_simulation_summary": pair_itf.summary_dict(),
        "visual_qa_png": qa_png_str,
    }


def _run_stage5_synthetic_phantom_check(
    *,
    figures_dir: Path | None = None,
    filter_name: str = "ram_lak",
    backend: str = "torch",
    epochs: int = 8,
    itf_ratio: float = DEFAULT_ITF_TRAINING_RATIO,
) -> dict[str, Any]:
    """Run Stage 5 synthetic verification: 60% ITF + 40% RDC across p in {1.5, 2.0, 2.5} + Dual-Domain DL."""
    t0 = time.monotonic()
    fname = normalize_filter_name(filter_name)
    samples = build_hybrid_phantom_dataset(
        pitches=DEFAULT_CLINICAL_HIGH_PITCHES,
        scenarios=("itf", "rdc"),
        z_slices_mm=(-12.0,),
        native_pitch=0.85,
        image_size=96,
        fov_mm=360.0,
        filter_name=fname,
        backend=backend,
    )
    model, device = create_dual_domain_model(
        sino_base_channels=16,
        img_base_channels=16,
        filter_name=fname,
        backend=backend,
        seed=42,
    )
    train_res = train_dual_domain_model(
        model,
        samples,
        device=device,
        epochs=max(2, int(epochs)),
        lr=3.0e-3,
        itf_ratio=itf_ratio,
        seed=42,
    )
    preds = train_res["predictions"]
    summary = train_res["summary"]

    # Select primary display prediction: ITF at p=2.0 (or highest available ITF sample)
    disp_pred = preds[0]
    for p in preds:
        if p.sample.scenario == "itf" and abs(p.sample.target_pitch - 2.0) < 0.1:
            disp_pred = p
            break

    qa_png_str: str | None = None
    if figures_dir is not None:
        qa_png = save_stage5_dual_domain_visual_qa(
            figures_dir / "stage5_phantom_dl_visual_qa.png",
            title_prefix="Stage 5 Phantom Visual QA",
            scenario=disp_pred.sample.scenario,
            target_pitch=disp_pred.sample.target_pitch,
            effective_pitch=disp_pred.sample.effective_pitch,
            native_pitch=disp_pred.sample.native_pitch,
            z_slice_mm=disp_pred.sample.z_slice_mm,
            fov_mm=disp_pred.sample.fov_mm,
            fov_mask=disp_pred.sample.fov_mask,
            filter_name=fname,
            s_hp_raw=disp_pred.sample.s_hp_raw,
            mask_hp=disp_pred.sample.mask_hp,
            s_dc=disp_pred.s_dc,
            x_target_hu=disp_pred.sample.x_target_hu,
            x_prelim_hu=disp_pred.sample.x_prelim_hu,
            x_dc_hu=disp_pred.x_dc_hu,
            x_final_hu=disp_pred.x_final_hu,
            prelim_metrics=disp_pred.prelim_metrics,
            domain1_dc_metrics=disp_pred.domain1_dc_metrics,
            final_dual_metrics=disp_pred.final_dual_metrics,
            sino_metrics=disp_pred.sinogram_progression_metrics,
            display_vmax_hu=600.0,
        )
        qa_png_str = str(qa_png)

    prog = summary["overall_progression"]
    overall_pass = bool(
        prog["stage3_final_dual_domain"]["rmse_hu"] <= prog["stage1_prelim_fbp"]["rmse_hu"]
        and prog["stage1_prelim_fbp"]["rmse_hu"] < prog["stage0_raw_nofill"]["rmse_hu"]
        and prog["sinogram_progression"]["measured_rays_max_abs_dev"] == 0.0
    )
    return {
        "pass": overall_pass,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "filter_name": fname,
        "backend": backend,
        "summary": summary,
        "visual_qa_png": qa_png_str,
    }


def _run_stage6_synthetic_generalization_check(
    *,
    figures_dir: Path | None = None,
    filter_name: str = "ram_lak",
    backend: str = "torch",
    epochs: int = 8,
    itf_ratio: float = DEFAULT_ITF_TRAINING_RATIO,
) -> dict[str, Any]:
    """Run Stage 6 synthetic multi-phantom zero-leakage generalization check."""
    t0 = time.monotonic()
    fname = normalize_filter_name(filter_name)
    split = build_multi_phantom_generalization_split(
        pitches=DEFAULT_CLINICAL_HIGH_PITCHES,
        scenarios=("itf", "rdc"),
        image_size=96,
        fov_mm=360.0,
        filter_name=fname,
        backend=backend,
    )
    res = run_generalization_experiment(
        split["train_samples"],
        split["test_samples"],
        split_mode="patient_split",
        sino_base_channels=16,
        img_base_channels=16,
        filter_name=fname,
        epochs=max(3, int(epochs)),
        lr=3.0e-3,
        itf_ratio=itf_ratio,
        backend=backend,
        seed=42,
    )
    test_preds = res["unseen_test_predictions"]
    disp_pred = test_preds[0]
    for p in test_preds:
        if p.sample.scenario == "itf" and abs(p.sample.target_pitch - 2.0) < 0.1:
            disp_pred = p
            break

    qa_png_str: str | None = None
    if figures_dir is not None:
        qa_png = save_stage6_generalization_visual_qa(
            figures_dir / "stage6_phantom_generalization_visual_qa.png",
            title_prefix="Stage 6 Multi-Phantom Visual QA",
            split_mode="patient_split",
            train_cases_str="PHANTOM_TRAIN_A,PHANTOM_TRAIN_B",
            test_case_id=disp_pred.sample.case_id,
            scenario=disp_pred.sample.scenario,
            target_pitch=disp_pred.sample.target_pitch,
            effective_pitch=disp_pred.sample.effective_pitch,
            native_pitch=disp_pred.sample.native_pitch,
            z_slice_mm=disp_pred.sample.z_slice_mm,
            fov_mm=disp_pred.sample.fov_mm,
            fov_mask=disp_pred.sample.fov_mask,
            filter_name=fname,
            s_hp_raw=disp_pred.sample.s_hp_raw,
            mask_hp=disp_pred.sample.mask_hp,
            s_dc=disp_pred.s_dc,
            x_target_hu=disp_pred.sample.x_target_hu,
            x_prelim_hu=disp_pred.sample.x_prelim_hu,
            x_dc_hu=disp_pred.x_dc_hu,
            x_final_hu=disp_pred.x_final_hu,
            prelim_metrics=disp_pred.prelim_metrics,
            domain1_dc_metrics=disp_pred.domain1_dc_metrics,
            final_dual_metrics=disp_pred.final_dual_metrics,
            sino_metrics=disp_pred.sinogram_progression_metrics,
            display_vmax_hu=600.0,
        )
        qa_png_str = str(qa_png)

    gen_m = res["generalization_metrics"]
    overall_pass = bool(
        gen_m["zero_leakage_verified"]
        and gen_m["unseen_test_monotonic_pass"]
        and gen_m["unseen_test_measured_rays_max_abs_dev"] == 0.0
    )
    return {
        "pass": overall_pass,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "filter_name": fname,
        "backend": backend,
        "zero_leakage_audit": res["zero_leakage_audit"],
        "generalization_metrics": gen_m,
        "train_summary": res["train_summary"],
        "unseen_test_summary": res["unseen_test_summary"],
        "visual_qa_png": qa_png_str,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpct",
        description="HighPitchCT CLI (Stage 0-6: Download, Audit, SSRB, Parker Ram-Lak FBP, Forward Sim, Dual-Domain DL, Generalization)",
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
        aliases=["index-dicom"],
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

    stage3 = sub.add_parser(
        "stage3-recon",
        help="Stage 3 Sinogram Completion + Exact Parker Weighting + Fan-Beam FBP (`x_reference`) & Visual QA",
    )
    stage3.add_argument("--data-root", help="Writable root containing raw/tcia and manifests/")
    stage3.add_argument(
        "--case-id",
        help="Optional patient case filter under raw/tcia (matches C012 or case_C012)",
    )
    stage3.add_argument("--series-dir", help="Explicit DICOM-CT-PD projection directory to reconstruct")
    stage3.add_argument("--image-dir", help="Optional published CT image directory for side-by-side Visual QA")
    stage3.add_argument(
        "--num-slices",
        type=int,
        default=5,
        help="Number of interior axial z slices to reconstruct (default: 5)",
    )
    stage3.add_argument(
        "--image-size",
        "--n-xy",
        dest="image_size",
        type=int,
        default=512,
        help="Reconstructed axial matrix size H = W (default: 512)",
    )
    stage3.add_argument(
        "--fov-mm",
        type=float,
        default=None,
        help="Reconstruction FOV diameter [mm] (default: matched to paired Vendor CT FOV or 2*rho0*sin(gamma_max))",
    )
    stage3.add_argument(
        "--filter",
        "--filter-name",
        dest="filter_name",
        choices=["ram_lak", "ram-lak", "ramp", "shepp_logan", "cosine", "hamming", "hann"],
        default="ram_lak",
        help="FBP frequency filter (default: ram_lak)",
    )
    stage3.add_argument(
        "--cutoff-ratio",
        type=float,
        default=1.0,
        help="Normalized Nyquist frequency cutoff in (0, 1] (default: 1.0)",
    )
    stage3.add_argument(
        "--angular-mode",
        choices=["short_scan", "full_scan"],
        default="short_scan",
        help="Target planar sinogram angular span: short_scan (Parker) or full_scan (0.5 weight)",
    )
    stage3.add_argument(
        "--inpaint-mode",
        choices=["conjugate_feather", "angular_feather", "none"],
        default="conjugate_feather",
        help="High-pitch missing-wedge completion strategy (default: conjugate_feather)",
    )
    stage3.add_argument(
        "--backend",
        choices=["numpy", "torch", "torch_cuda"],
        default="numpy",
        help="Backprojection execution backend (default: numpy; torch_cuda requires GPU without fallback)",
    )
    stage3.add_argument(
        "--recon-out",
        help="Optional compressed .npz path to save reconstructed mu_slices, hu_slices, and coordinate grids",
    )
    stage3.add_argument(
        "--save-reference-npz",
        action="store_true",
        help="Automatically save reconstructed x_reference slices to <data-root>/manifests/stage3_<case_id>_x_reference.npz",
    )
    stage3.add_argument(
        "--run-phantom-check",
        action="store_true",
        help="Run Stage 3 synthetic verification (0-HU water calibration + scalar parity + Thorax FBP QA)",
    )
    stage3.add_argument("--workers", type=int, default=8)
    stage3.add_argument("--no-progress", action="store_true")

    stage4 = sub.add_parser(
        "stage4-sim",
        help="Stage 4 2D/3D Forward Projection + High-Pitch (RDC & ITF) Simulation + Fig. 2(d) Reprojection & Visual QA",
    )
    stage4.add_argument("--data-root", help="Writable root containing raw/tcia and manifests/")
    stage4.add_argument(
        "--case-id",
        help="Optional patient case filter under raw/tcia (matches C012 or case_C012)",
    )
    stage4.add_argument("--series-dir", help="Explicit DICOM-CT-PD projection directory to simulate")
    stage4.add_argument("--image-dir", help="Optional published CT image directory for FOV matching")
    stage4.add_argument(
        "--scenario",
        choices=["rdc", "itf", "both"],
        default="both",
        help="High-pitch simulation scenario: rdc (Fig. 1b), itf (Fig. 1c), or both (default: both)",
    )
    stage4.add_argument(
        "--target-pitch",
        type=float,
        default=2.0,
        help="Primary high-pitch factor to simulate for Visual QA and .npz export (default: 2.0)",
    )
    stage4.add_argument(
        "--pitches",
        default="1.5,2.0,2.5",
        help="Comma-separated list of target high pitches to benchmark (default: 1.5,2.0,2.5)",
    )
    stage4.add_argument(
        "--num-slices",
        type=int,
        default=1,
        help="Number of interior axial z slices to simulate (default: 1)",
    )
    stage4.add_argument(
        "--image-size",
        "--n-xy",
        dest="image_size",
        type=int,
        default=256,
        help="Reconstructed axial matrix size H = W (default: 256)",
    )
    stage4.add_argument(
        "--fov-mm",
        type=float,
        default=None,
        help="Reconstruction FOV diameter [mm] (default: matched to paired Vendor CT FOV or scanner FOV)",
    )
    stage4.add_argument(
        "--filter",
        "--filter-name",
        dest="filter_name",
        choices=["ram_lak", "ram-lak", "ramp", "shepp_logan", "cosine", "hamming", "hann"],
        default="ram_lak",
        help="FBP frequency filter (default: ram_lak)",
    )
    stage4.add_argument(
        "--cutoff-ratio",
        type=float,
        default=1.0,
        help="Normalized Nyquist frequency cutoff in (0, 1] (default: 1.0)",
    )
    stage4.add_argument(
        "--inpaint-mode",
        choices=["conjugate_feather", "angular_feather", "none"],
        default="conjugate_feather",
        help="High-pitch missing-wedge completion strategy (default: conjugate_feather)",
    )
    stage4.add_argument(
        "--itf-slab-slices",
        type=int,
        default=7,
        help="Number of axial reference slices in the 3D slab for ITF helical forward projection (default: 7)",
    )
    stage4.add_argument(
        "--backend",
        choices=["numpy", "torch", "torch_cuda"],
        default="numpy",
        help="Forward/backprojection execution backend (default: numpy; torch_cuda requires GPU without fallback)",
    )
    stage4.add_argument(
        "--pairs-out",
        help="Optional compressed .npz path to save Dual-Domain training/evaluation arrays",
    )
    stage4.add_argument(
        "--save-pairs-npz",
        action="store_true",
        help="Automatically save simulated pairs to <data-root>/manifests/stage4_<case_id>_pairs.npz",
    )
    stage4.add_argument(
        "--run-phantom-check",
        action="store_true",
        help="Run Stage 4 synthetic verification (2D scalar parity + Forward->FBP roundtrip + RDC/ITF QA)",
    )
    stage4.add_argument("--workers", type=int, default=8)
    stage4.add_argument("--no-progress", action="store_true")

    stage5 = sub.add_parser(
        "stage5-train",
        aliases=["stage5-dl"],
        help="Stage 5 Physics-Conditioned Dual-Domain DL (60% ITF + 40% RDC, p in {1.5, 2.0, 2.5}, Ram-Lak FBP Bridge) & Visual QA",
    )
    stage5.add_argument(
        "--data-root",
        "--workspace",
        dest="data_root",
        help="Writable root containing raw/tcia and manifests/",
    )
    stage5.add_argument(
        "--case-id",
        help="Optional patient case filter under raw/tcia (matches C012 or case_C012)",
    )
    stage5.add_argument("--series-dir", help="Explicit DICOM-CT-PD projection directory")
    stage5.add_argument("--image-dir", help="Optional published CT image directory for FOV matching")
    stage5.add_argument(
        "--scenario",
        choices=["rdc", "itf", "both"],
        default="both",
        help="High-pitch scenarios to include in dataset (default: both -> 60% ITF + 40% RDC)",
    )
    stage5.add_argument(
        "--itf-ratio",
        "--itf-weight",
        dest="itf_ratio",
        type=float,
        default=DEFAULT_ITF_TRAINING_RATIO,
        help="Training curriculum weight assigned to ITF vs RDC when both are enabled (default: 0.60)",
    )
    stage5.add_argument(
        "--pitches",
        default="1.5,2.0,2.5",
        help="Comma-separated clinical high-pitch targets (default: 1.5,2.0,2.5)",
    )
    stage5.add_argument(
        "--num-slices",
        type=int,
        default=1,
        help="Number of interior axial z slices to extract per series (default: 1)",
    )
    stage5.add_argument(
        "--image-size",
        "--n-xy",
        dest="image_size",
        type=int,
        default=256,
        help="Reconstructed axial matrix size H = W (default: 256)",
    )
    stage5.add_argument(
        "--fov-mm",
        type=float,
        default=None,
        help="Reconstruction FOV diameter [mm] (default: matched to paired Vendor CT FOV or scanner FOV)",
    )
    stage5.add_argument(
        "--filter",
        "--filter-name",
        dest="filter_name",
        choices=["ram_lak", "ram-lak", "ramp", "shepp_logan", "cosine", "hamming", "hann"],
        default="ram_lak",
        help="FBP frequency filter (default: ram_lak)",
    )
    stage5.add_argument(
        "--cutoff-ratio",
        type=float,
        default=1.0,
        help="Normalized Nyquist frequency cutoff in (0, 1] (default: 1.0)",
    )
    stage5.add_argument(
        "--inpaint-mode",
        choices=["conjugate_feather", "angular_feather", "none"],
        default="conjugate_feather",
        help="Preliminary missing-wedge completion strategy (default: conjugate_feather)",
    )
    stage5.add_argument(
        "--itf-slab-slices",
        type=int,
        default=7,
        help="Number of axial reference slices in the 3D slab for ITF helical forward projection (default: 7)",
    )
    stage5.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of Dual-Domain optimization epochs (default: 15)",
    )
    stage5.add_argument(
        "--lr",
        type=float,
        default=2.5e-3,
        help="AdamW initial learning rate (default: 2.5e-3)",
    )
    stage5.add_argument(
        "--sino-channels",
        type=int,
        default=64,
        help="Base channel width for 4-level Domain-1 SinoInpaintUNet: 64->128->256->512 (default: 64)",
    )
    stage5.add_argument(
        "--img-channels",
        type=int,
        default=64,
        help="Base channel width for 4-level Domain-2 ImageRefineUNet: 64->128->256->512 (default: 64)",
    )
    stage5.add_argument(
        "--backend",
        choices=["torch", "torch_cuda"],
        default="torch",
        help="PyTorch execution backend (default: torch; torch_cuda requires GPU without silent fallback)",
    )
    stage5.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained DualDomainHighPitchNet weights to <data-root>/manifests/stage5_<case_id>_dual_domain.pt",
    )
    stage5.add_argument(
        "--save-dataset-npz",
        action="store_true",
        help="Save multi-pitch ITF+RDC dataset to <data-root>/manifests/stage5_<case_id>_dataset.npz",
    )
    stage5.add_argument(
        "--run-phantom-check",
        action="store_true",
        help="Run Stage 5 synthetic phantom verification alongside patient training",
    )
    stage5.add_argument("--workers", type=int, default=4)
    stage5.add_argument("--no-progress", action="store_true")

    stage6 = sub.add_parser(
        "stage6-generalize",
        aliases=["stage6-eval"],
        help="Stage 6 Zero-Leakage Multi-Patient / Multi-Zone Generalization & Out-of-Sample Evaluation",
    )
    stage6.add_argument(
        "--data-root",
        "--workspace",
        dest="data_root",
        help="Writable root containing raw/tcia and manifests/",
    )
    stage6.add_argument(
        "--case-id",
        help="Optional patient case filter under raw/tcia (e.g. C012)",
    )
    stage6.add_argument("--series-dir", help="Explicit DICOM-CT-PD projection directory")
    stage6.add_argument("--image-dir", help="Optional published CT image directory for FOV matching")
    stage6.add_argument(
        "--split-mode",
        choices=["auto", "patient_split", "axial_guardband"],
        default="auto",
        help="Zero-leakage split protocol: 'patient_split' (>=2 patients), 'axial_guardband' (disjoint thorax slabs), or 'auto'",
    )
    stage6.add_argument(
        "--train-cases",
        help="Comma-separated training patient case IDs for patient_split (e.g. C012)",
    )
    stage6.add_argument(
        "--test-cases",
        help="Comma-separated unseen test patient case IDs for patient_split (e.g. C027)",
    )
    stage6.add_argument(
        "--train-slices-per-case",
        type=int,
        default=2,
        help="Number of training axial slices extracted per training case (default: 2)",
    )
    stage6.add_argument(
        "--test-slices-per-case",
        type=int,
        default=1,
        help="Number of held-out unseen test axial slices extracted per test case (default: 1)",
    )
    stage6.add_argument(
        "--min-guardband-mm",
        type=float,
        default=45.0,
        help="Minimum helical cone-beam dead-zone guard band [mm] between train and test zones in axial_guardband mode (default: 45.0)",
    )
    stage6.add_argument(
        "--scenario",
        choices=["rdc", "itf", "both"],
        default="both",
        help="High-pitch scenarios to include (default: both -> 60% ITF + 40% RDC)",
    )
    stage6.add_argument(
        "--itf-ratio",
        "--itf-weight",
        dest="itf_ratio",
        type=float,
        default=DEFAULT_ITF_TRAINING_RATIO,
        help="Training curriculum weight assigned to ITF vs RDC (default: 0.60)",
    )
    stage6.add_argument(
        "--pitches",
        default="1.5,2.0,2.5",
        help="Comma-separated clinical high-pitch targets (default: 1.5,2.0,2.5)",
    )
    stage6.add_argument(
        "--image-size",
        "--n-xy",
        dest="image_size",
        type=int,
        default=256,
        help="Reconstructed axial matrix size H = W (default: 256)",
    )
    stage6.add_argument(
        "--fov-mm",
        type=float,
        default=None,
        help="Reconstruction FOV diameter [mm] (default: matched to paired Vendor CT FOV or scanner FOV)",
    )
    stage6.add_argument(
        "--filter",
        "--filter-name",
        dest="filter_name",
        choices=["ram_lak", "ram-lak", "ramp", "shepp_logan", "cosine", "hamming", "hann"],
        default="ram_lak",
        help="FBP frequency filter (default: ram_lak)",
    )
    stage6.add_argument(
        "--cutoff-ratio",
        type=float,
        default=1.0,
        help="Normalized Nyquist frequency cutoff in (0, 1] (default: 1.0)",
    )
    stage6.add_argument(
        "--inpaint-mode",
        choices=["conjugate_feather", "angular_feather", "none"],
        default="conjugate_feather",
        help="Preliminary missing-wedge completion strategy (default: conjugate_feather)",
    )
    stage6.add_argument(
        "--itf-slab-slices",
        type=int,
        default=7,
        help="Number of axial reference slices in each 3D ITF slab (default: 7)",
    )
    stage6.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Total 3-Stage Dual-Domain training epochs on train_samples (default: 30)",
    )
    stage6.add_argument(
        "--lr",
        type=float,
        default=1.5e-3,
        help="AdamW initial learning rate (default: 1.5e-3)",
    )
    stage6.add_argument(
        "--sino-channels",
        type=int,
        default=64,
        help="Base channel width for 4-level Domain-1 SinoInpaintUNet: 64->128->256->512 (default: 64)",
    )
    stage6.add_argument(
        "--img-channels",
        type=int,
        default=64,
        help="Base channel width for 4-level Domain-2 ImageRefineUNet: 64->128->256->512 (default: 64)",
    )
    stage6.add_argument(
        "--backend",
        choices=["torch", "torch_cuda"],
        default="torch",
        help="PyTorch execution backend (default: torch; torch_cuda requires GPU without silent fallback)",
    )
    stage6.add_argument(
        "--save-model",
        action="store_true",
        help="Save generalized DualDomainHighPitchNet weights to <data-root>/manifests/stage6_generalized_dual_domain.pt",
    )
    stage6.add_argument(
        "--load-model",
        default=None,
        help="Optional path to a saved .pt DualDomainHighPitchNet checkpoint for frozen zero-shot evaluation without retraining",
    )
    stage6.add_argument(
        "--run-phantom-check",
        action="store_true",
        help="Run Stage 6 synthetic multi-phantom zero-leakage generalization check",
    )
    stage6.add_argument("--workers", type=int, default=4)
    stage6.add_argument("--no-progress", action="store_true")

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

    if args.command in ("index", "index-dicom"):
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
            "stage": "Stage 1 (v0.6.0)",
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
            "stage": "Stage 2 (v0.6.0)",
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

    if args.command == "stage3-recon":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        if args.num_slices < 1:
            raise SystemExit("--num-slices must be at least 1")
        if args.image_size < 16:
            raise SystemExit("--image-size must be at least 16")

        t_start = time.monotonic()
        env_prof = resolve_environment_profile(data_root=args.data_root)
        show_prog = not args.no_progress
        figures_dir = env_prof.data_root / "manifests" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fname = normalize_filter_name(args.filter_name)

        targets = _resolve_cli_series_targets(
            env_prof.data_root / "raw" / "tcia",
            case_id=args.case_id,
            series_dir=args.series_dir,
            image_dir=args.image_dir,
            allow_empty_for_phantom=bool(args.run_phantom_check),
        )

        reconstructed_series: list[dict[str, Any]] = []
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

            # Pre-read paired Vendor CT volume (if available) to match its clinical FOV (`400.0 mm`)
            vol_data: dict[str, Any] | None = None
            vendor_fov_mm: float | None = None
            vendor_kernel: str | None = None
            if idir is not None and idir.exists():
                vol_data = read_ct_image_volume(idir, water_mu_mm_inv=geom.water_mu_mm_inv)
                ps_yx = vol_data["pixel_spacing_yx_mm"]
                vendor_fov_mm = float(vol_data["shape_zyx"][2]) * float(ps_yx[1])
                vendor_kernel = vol_data["convolution_kernel"] or None

            effective_fov_mm = (
                args.fov_mm
                if args.fov_mm is not None
                else (vendor_fov_mm if vendor_fov_mm is not None and vendor_fov_mm > 0.0 else None)
            )

            ssrb_ref, fbp_ref = reconstruct_series_reference_streaming(
                files,
                z_slices,
                geometry=geom,
                angular_mode=args.angular_mode,
                image_size=args.image_size,
                fov_mm=effective_fov_mm,
                filter_name=fname,
                cutoff_ratio=args.cutoff_ratio,
                inpaint_mode=args.inpaint_mode,
                backend=args.backend,
                workers=min(args.workers, 6),
                show_progress=show_prog,
            )

            mid_slice_idx = len(z_slices) // 2
            z_mid = float(z_slices[mid_slice_idx])
            ref_mid_hu = fbp_ref.hu_slices[mid_slice_idx]

            # Mid-slice High-Pitch RDC (50% central rows) FBP ablation: No-Fill vs Conjugate-Feather Fill
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
            fbp_rdc_raw = reconstruct_fanbeam_fbp(
                ssrb_rdc_mid,
                image_size=args.image_size,
                fov_mm=fbp_ref.fov_mm,
                water_mu_mm_inv=geom.water_mu_mm_inv,
                filter_name=fname,
                cutoff_ratio=args.cutoff_ratio,
                inpaint_mode="none",
                backend=args.backend,
            )
            fbp_rdc_inpaint = reconstruct_fanbeam_fbp(
                ssrb_rdc_mid,
                image_size=args.image_size,
                fov_mm=fbp_ref.fov_mm,
                water_mu_mm_inv=geom.water_mu_mm_inv,
                filter_name=fname,
                cutoff_ratio=args.cutoff_ratio,
                inpaint_mode="conjugate_feather",
                backend=args.backend,
            )

            rdc_raw_vs_ref = compute_reconstruction_metrics(
                fbp_rdc_raw.hu_slices[0], ref_mid_hu, fov_mask=fbp_ref.fov_mask
            )
            rdc_inpaint_vs_ref = compute_reconstruction_metrics(
                fbp_rdc_inpaint.hu_slices[0], ref_mid_hu, fov_mask=fbp_ref.fov_mask
            )

            vendor_slice_hu: np.ndarray | None = None
            if vol_data is not None:
                z_vol = vol_data["z_mm"]
                if float(z_vol[0]) <= z_mid <= float(z_vol[-1]):
                    v_idx_z = int(np.argmin(np.abs(z_vol - z_mid)))
                else:
                    frac = (z_mid - z_min_raw) / max(z_max_raw - z_min_raw, 1e-6)
                    v_idx_z = int(np.clip(round(frac * (z_vol.size - 1)), 0, z_vol.size - 1))
                vendor_slice_hu = vol_data["hu_volume"][v_idx_z]

            qa_png = save_stage3_real_fbp_visual_qa(
                figures_dir / f"stage3_{cid}_fbp_visual_qa.png",
                case_id=cid,
                z_slice_mm=z_mid,
                native_pitch=native_pitch,
                ref_fbp_hu=ref_mid_hu,
                fov_mask=fbp_ref.fov_mask,
                fov_mm=fbp_ref.fov_mm,
                filter_name=fbp_ref.filter_name,
                rdc_effective_pitch=rdc_eff_pitch,
                rdc_raw_fbp_hu=fbp_rdc_raw.hu_slices[0],
                rdc_inpaint_fbp_hu=fbp_rdc_inpaint.hu_slices[0],
                rdc_vs_ref_metrics=rdc_inpaint_vs_ref,
                rdc_raw_vs_ref_metrics=rdc_raw_vs_ref,
                vendor_slice_hu=vendor_slice_hu,
                vendor_fov_mm=vendor_fov_mm,
                vendor_kernel=vendor_kernel,
            )

            finite_ok = bool(
                np.all(np.isfinite(fbp_ref.mu_slices)) and np.all(np.isfinite(fbp_ref.hu_slices))
            )
            hu_fov = fbp_ref.hu_slices[:, fbp_ref.fov_mask]
            hu_range_ok = bool(
                hu_fov.size > 0
                and float(np.min(hu_fov)) <= -850.0
                and float(np.max(hu_fov)) >= 100.0
            )
            inpaint_improves_rdc = bool(
                rdc_inpaint_vs_ref["rmse_hu"] <= rdc_raw_vs_ref["rmse_hu"]
                if rdc_eff_pitch > 1.35
                else True
            )
            series_pass = bool(finite_ok and hu_range_ok and inpaint_improves_rdc)
            if not series_pass:
                all_pass = False

            entry = {
                "case_id": cid,
                "series_dir": str(pdir),
                "native_normalized_pitch": round(native_pitch, 5),
                "ffs_mode": geom.ffs_mode,
                "series_pass": series_pass,
                "finite_values_verified": finite_ok,
                "physical_hu_range_verified": hu_range_ok,
                "ssrb_measured_fraction": round(ssrb_ref.overall_measured_fraction, 6),
                "reference_fbp_summary": fbp_ref.summary_dict(),
                "mid_slice_high_pitch_rdc_ablation": {
                    "z_mid_mm": round(z_mid, 4),
                    "rdc_50pct_rows_range": [r_lo_rdc, r_hi_rdc],
                    "rdc_effective_pitch": round(rdc_eff_pitch, 5),
                    "rdc_measured_fraction": round(ssrb_rdc_mid.overall_measured_fraction, 6),
                    "rdc_no_fill_vs_x_reference": rdc_raw_vs_ref,
                    "rdc_conjugate_feather_vs_x_reference": rdc_inpaint_vs_ref,
                    "conjugate_fill_rmse_reduction_hu": round(
                        rdc_raw_vs_ref["rmse_hu"] - rdc_inpaint_vs_ref["rmse_hu"], 4
                    ),
                },
                "visual_qa_png": str(qa_png),
            }

            target_npz_out: Path | None = None
            if args.recon_out and idx_t == 0:
                target_npz_out = Path(args.recon_out).expanduser().resolve()
            elif getattr(args, "save_reference_npz", False):
                target_npz_out = env_prof.data_root / "manifests" / f"stage3_{cid}_x_reference.npz"

            if target_npz_out is not None:
                target_npz_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target_npz_out,
                    mu_slices=fbp_ref.mu_slices,
                    hu_slices=fbp_ref.hu_slices,
                    fov_mask=fbp_ref.fov_mask,
                    z_slices_mm=fbp_ref.z_slices_mm,
                    y_mm=fbp_ref.y_mm,
                    x_mm=fbp_ref.x_mm,
                    fov_mm=np.array(fbp_ref.fov_mm, dtype=np.float64),
                    pixel_spacing_mm=np.array(fbp_ref.pixel_spacing_mm, dtype=np.float64),
                    water_mu_mm_inv=np.array(fbp_ref.water_mu_mm_inv, dtype=np.float64),
                    filter_name=np.array(fbp_ref.filter_name, dtype=np.str_),
                )
                entry["recon_npz"] = str(target_npz_out)

            reconstructed_series.append(entry)

        phantom_stage3 = None
        if args.run_phantom_check:
            phantom_stage3 = _run_stage3_synthetic_phantom_check(figures_dir=figures_dir)
            if not phantom_stage3["pass"]:
                all_pass = False

        report = {
            "stage": "Stage 3 (v0.7.0)",
            "overall_pass": all_pass,
            "elapsed_seconds": round(time.monotonic() - t_start, 3),
            "reconstructed_series_count": len(reconstructed_series),
            "reconstructed_series": reconstructed_series,
            "synthetic_phantom_fbp_check": phantom_stage3,
        }
        out_json = env_prof.data_root / "manifests" / "stage3_recon_report.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_json"] = str(out_json)
        print(json.dumps(report, indent=2))
        return 0 if all_pass else 1

    if args.command == "stage4-sim":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        if args.num_slices < 1:
            raise SystemExit("--num-slices must be at least 1")
        if args.image_size < 16:
            raise SystemExit("--image-size must be at least 16")
        if args.target_pitch <= 0.0:
            raise SystemExit("--target-pitch must be positive")

        t_start = time.monotonic()
        env_prof = resolve_environment_profile(data_root=args.data_root)
        show_prog = not args.no_progress
        figures_dir = env_prof.data_root / "manifests" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fname = normalize_filter_name(args.filter_name)

        pitch_list = sorted(
            {
                float(x.strip())
                for x in str(args.pitches).split(",")
                if x.strip()
            }
            | {float(args.target_pitch)}
        )

        targets = _resolve_cli_series_targets(
            env_prof.data_root / "raw" / "tcia",
            case_id=args.case_id,
            series_dir=args.series_dir,
            image_dir=args.image_dir,
            allow_empty_for_phantom=bool(args.run_phantom_check),
        )

        simulated_series: list[dict[str, Any]] = []
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

            vendor_fov_mm: float | None = None
            if idir is not None and idir.exists():
                vol_data = read_ct_image_volume(idir, water_mu_mm_inv=geom.water_mu_mm_inv)
                ps_yx = vol_data["pixel_spacing_yx_mm"]
                vendor_fov_mm = float(vol_data["shape_zyx"][2]) * float(ps_yx[1])

            effective_fov_mm = (
                args.fov_mm
                if args.fov_mm is not None
                else (vendor_fov_mm if vendor_fov_mm is not None and vendor_fov_mm > 0.0 else None)
            )

            # Reconstruct shared low-pitch reference target (s*, x*) once
            ssrb_ref, fbp_ref = reconstruct_series_reference_streaming(
                files,
                z_slices,
                geometry=geom,
                angular_mode="short_scan",
                image_size=args.image_size,
                fov_mm=effective_fov_mm,
                filter_name=fname,
                cutoff_ratio=args.cutoff_ratio,
                inpaint_mode="conjugate_feather",
                backend=args.backend,
                workers=min(args.workers, 6),
                show_progress=show_prog,
            )

            rdc_multi_pitch_summary: dict[str, Any] = {}
            primary_rdc_pair = None
            primary_itf_pair = None

            if args.scenario in {"rdc", "both"}:
                for p_val in pitch_list:
                    pair_p = simulate_series_high_pitch_streaming(
                        files,
                        z_slices,
                        geometry=geom,
                        scenario="rdc",
                        target_pitch=p_val,
                        image_size=args.image_size,
                        fov_mm=fbp_ref.fov_mm,
                        filter_name=fname,
                        cutoff_ratio=args.cutoff_ratio,
                        inpaint_mode=args.inpaint_mode,
                        backend=args.backend,
                        workers=min(args.workers, 6),
                        ssrb_ref_precomputed=ssrb_ref,
                        fbp_ref_precomputed=fbp_ref,
                        show_progress=False,
                    )
                    rdc_multi_pitch_summary[f"p_{p_val:.1f}"] = pair_p.summary_dict()
                    if abs(p_val - float(args.target_pitch)) < 1e-4 or primary_rdc_pair is None:
                        primary_rdc_pair = pair_p

            if args.scenario in {"itf", "both"}:
                primary_itf_pair = simulate_series_high_pitch_streaming(
                    files,
                    z_slices,
                    geometry=geom,
                    scenario="itf",
                    target_pitch=float(args.target_pitch),
                    image_size=args.image_size,
                    fov_mm=fbp_ref.fov_mm,
                    filter_name=fname,
                    cutoff_ratio=args.cutoff_ratio,
                    inpaint_mode=args.inpaint_mode,
                    backend=args.backend,
                    itf_slab_slices=args.itf_slab_slices,
                    workers=min(args.workers, 6),
                    ssrb_ref_precomputed=ssrb_ref,
                    fbp_ref_precomputed=fbp_ref,
                    show_progress=show_prog,
                )

            pair_for_rdc_panel = primary_rdc_pair if primary_rdc_pair is not None else primary_itf_pair
            pair_for_itf_panel = primary_itf_pair if primary_itf_pair is not None else primary_rdc_pair
            assert pair_for_rdc_panel is not None and pair_for_itf_panel is not None

            mid_idx = len(z_slices) // 2
            qa_png = save_stage4_real_simulation_visual_qa(
                figures_dir / f"stage4_{cid}_rdc_itf_visual_qa.png",
                case_id=cid,
                z_slice_mm=float(z_slices[mid_idx]),
                native_pitch=native_pitch,
                fov_mm=fbp_ref.fov_mm,
                fov_mask=fbp_ref.fov_mask,
                filter_name=fname,
                x_target_hu=fbp_ref.hu_slices[mid_idx],
                s_target=pair_for_rdc_panel.s_target[mid_idx],
                s_hp_raw=pair_for_rdc_panel.s_hp_raw[mid_idx],
                mask_hp=pair_for_rdc_panel.mask_hp[mid_idx],
                s_reproj=pair_for_rdc_panel.s_reproj[mid_idx],
                rdc_pitch=pair_for_rdc_panel.effective_pitch,
                rdc_prelim_hu=pair_for_rdc_panel.x_prelim_hu[mid_idx],
                rdc_metrics=pair_for_rdc_panel.prelim_vs_target_metrics,
                itf_pitch=pair_for_itf_panel.effective_pitch,
                itf_prelim_hu=pair_for_itf_panel.x_prelim_hu[mid_idx],
                itf_metrics=pair_for_itf_panel.prelim_vs_target_metrics,
            )

            series_pass = True
            for pair_chk in (primary_rdc_pair, primary_itf_pair):
                if pair_chk is not None:
                    if not (
                        np.all(np.isfinite(pair_chk.s_reproj))
                        and np.all(np.isfinite(pair_chk.x_prelim_hu))
                        and pair_chk.prelim_vs_target_metrics["rmse_hu"]
                        <= pair_chk.raw_nofill_vs_target_metrics["rmse_hu"]
                    ):
                        series_pass = False
            if not series_pass:
                all_pass = False

            entry = {
                "case_id": cid,
                "series_dir": str(pdir),
                "native_normalized_pitch": round(native_pitch, 5),
                "fov_mm": round(float(fbp_ref.fov_mm), 4),
                "filter_name": fname,
                "series_pass": series_pass,
                "rdc_multi_pitch_summary": rdc_multi_pitch_summary,
                "primary_rdc_summary": (
                    primary_rdc_pair.summary_dict() if primary_rdc_pair is not None else None
                ),
                "primary_itf_summary": (
                    primary_itf_pair.summary_dict() if primary_itf_pair is not None else None
                ),
                "visual_qa_png": str(qa_png),
            }

            target_npz_out: Path | None = None
            if args.pairs_out and idx_t == 0:
                target_npz_out = Path(args.pairs_out).expanduser().resolve()
            elif getattr(args, "save_pairs_npz", False):
                target_npz_out = env_prof.data_root / "manifests" / f"stage4_{cid}_pairs.npz"

            if target_npz_out is not None:
                target_npz_out.parent.mkdir(parents=True, exist_ok=True)
                active_pair = primary_rdc_pair if primary_rdc_pair is not None else primary_itf_pair
                assert active_pair is not None
                np.savez_compressed(
                    target_npz_out,
                    s_target=active_pair.s_target,
                    s_hp_raw=active_pair.s_hp_raw,
                    mask_hp=active_pair.mask_hp,
                    s_reproj=active_pair.s_reproj,
                    s_reproj_dc=active_pair.s_reproj_dc,
                    x_target_hu=active_pair.x_target_hu,
                    mu_target=active_pair.mu_target,
                    x_prelim_hu=active_pair.x_prelim_hu,
                    mu_prelim=active_pair.mu_prelim,
                    fov_mask=active_pair.fov_mask,
                    z_slices_mm=active_pair.z_slices_mm,
                    y_mm=active_pair.y_mm,
                    x_mm=active_pair.x_mm,
                    beta_rad=active_pair.beta_rad,
                    u_coords_mm=active_pair.u_coords_mm,
                    fov_mm=np.array(active_pair.fov_mm, dtype=np.float64),
                    filter_name=np.array(active_pair.filter_name, dtype=np.str_),
                )
                entry["pairs_npz"] = str(target_npz_out)

            simulated_series.append(entry)

        phantom_stage4 = None
        if args.run_phantom_check:
            phantom_stage4 = _run_stage4_synthetic_phantom_check(
                figures_dir=figures_dir,
                filter_name=fname,
                backend=args.backend,
            )
            if not phantom_stage4["pass"]:
                all_pass = False

        report = {
            "stage": "Stage 4 (v0.8.0)",
            "overall_pass": all_pass,
            "elapsed_seconds": round(time.monotonic() - t_start, 3),
            "simulated_series_count": len(simulated_series),
            "simulated_series": simulated_series,
            "synthetic_phantom_forward_check": phantom_stage4,
        }
        out_json = env_prof.data_root / "manifests" / "stage4_sim_report.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_json"] = str(out_json)
        print(json.dumps(report, indent=2))
        return 0 if all_pass else 1

    if args.command in {"stage5-train", "stage5-dl"}:
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        if args.num_slices < 1:
            raise SystemExit("--num-slices must be at least 1")
        if args.image_size < 16:
            raise SystemExit("--image-size must be at least 16")
        if args.epochs < 1:
            raise SystemExit("--epochs must be at least 1")
        if not (0.0 <= float(args.itf_ratio) <= 1.0):
            raise SystemExit("--itf-ratio must be in [0.0, 1.0]")

        t_start = time.monotonic()
        env_prof = resolve_environment_profile(data_root=args.data_root)
        show_prog = not args.no_progress
        figures_dir = env_prof.data_root / "manifests" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fname = normalize_filter_name(args.filter_name)

        pitch_list = sorted(
            {
                float(x.strip())
                for x in str(args.pitches).split(",")
                if x.strip()
            }
        )
        if not pitch_list or any(p <= 0.0 for p in pitch_list):
            raise SystemExit("--pitches must contain positive floats, e.g. 1.5,2.0,2.5")

        if args.scenario == "both":
            scenarios_list = ("itf", "rdc")
        else:
            scenarios_list = (args.scenario,)

        targets = _resolve_cli_series_targets(
            env_prof.data_root / "raw" / "tcia",
            case_id=args.case_id,
            series_dir=args.series_dir,
            image_dir=args.image_dir,
            allow_empty_for_phantom=bool(args.run_phantom_check),
        )

        trained_series: list[dict[str, Any]] = []
        all_pass = True

        for cid, pdir, idir in targets:
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

            vendor_fov_mm: float | None = None
            if idir is not None and idir.exists():
                vol_data = read_ct_image_volume(idir, water_mu_mm_inv=geom.water_mu_mm_inv)
                ps_yx = vol_data["pixel_spacing_yx_mm"]
                vendor_fov_mm = float(vol_data["shape_zyx"][2]) * float(ps_yx[1])

            effective_fov_mm = (
                args.fov_mm
                if args.fov_mm is not None
                else (vendor_fov_mm if vendor_fov_mm is not None and vendor_fov_mm > 0.0 else None)
            )

            samples = build_hybrid_series_dataset_streaming(
                files,
                z_slices,
                case_id=cid,
                geometry=geom,
                pitches=pitch_list,
                scenarios=scenarios_list,
                image_size=args.image_size,
                fov_mm=effective_fov_mm,
                filter_name=fname,
                cutoff_ratio=args.cutoff_ratio,
                inpaint_mode=args.inpaint_mode,
                itf_slab_slices=args.itf_slab_slices,
                backend=args.backend,
                workers=min(args.workers, 6),
                show_progress=show_prog,
            )

            model, device = create_dual_domain_model(
                sino_base_channels=args.sino_channels,
                img_base_channels=args.img_channels,
                filter_name=fname,
                cutoff_ratio=args.cutoff_ratio,
                backend=args.backend,
                seed=42,
            )
            train_res = train_dual_domain_model(
                model,
                samples,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                itf_ratio=float(args.itf_ratio),
                seed=42,
            )
            preds = train_res["predictions"]
            summary = train_res["summary"]

            # Choose primary display sample: prefer ITF at middle pitch (e.g. 2.0)
            disp_pred = preds[len(preds) // 2]
            mid_pitch = pitch_list[len(pitch_list) // 2]
            for p in preds:
                if p.sample.scenario == "itf" and abs(p.sample.target_pitch - mid_pitch) < 0.15:
                    disp_pred = p
                    break

            qa_png = save_stage5_dual_domain_visual_qa(
                figures_dir / f"stage5_{cid}_dl_visual_qa.png",
                title_prefix=f"Stage 5 Real-Data Visual QA ({cid})",
                scenario=disp_pred.sample.scenario,
                target_pitch=disp_pred.sample.target_pitch,
                effective_pitch=disp_pred.sample.effective_pitch,
                native_pitch=disp_pred.sample.native_pitch,
                z_slice_mm=disp_pred.sample.z_slice_mm,
                fov_mm=disp_pred.sample.fov_mm,
                fov_mask=disp_pred.sample.fov_mask,
                filter_name=fname,
                s_hp_raw=disp_pred.sample.s_hp_raw,
                mask_hp=disp_pred.sample.mask_hp,
                s_dc=disp_pred.s_dc,
                x_target_hu=disp_pred.sample.x_target_hu,
                x_prelim_hu=disp_pred.sample.x_prelim_hu,
                x_dc_hu=disp_pred.x_dc_hu,
                x_final_hu=disp_pred.x_final_hu,
                prelim_metrics=disp_pred.prelim_metrics,
                domain1_dc_metrics=disp_pred.domain1_dc_metrics,
                final_dual_metrics=disp_pred.final_dual_metrics,
                sino_metrics=disp_pred.sinogram_progression_metrics,
                display_vmax_hu=400.0,
            )

            prog = summary["overall_progression"]
            series_pass = bool(
                prog["stage3_final_dual_domain"]["rmse_hu"]
                <= prog["stage1_prelim_fbp"]["rmse_hu"]
                and prog["stage1_prelim_fbp"]["rmse_hu"]
                <= prog["stage0_raw_nofill"]["rmse_hu"]
                and prog["sinogram_progression"]["measured_rays_max_abs_dev"] == 0.0
            )
            if not series_pass:
                all_pass = False

            entry: dict[str, Any] = {
                "case_id": cid,
                "series_dir": str(pdir),
                "native_normalized_pitch": round(native_pitch, 5),
                "fov_mm": round(float(disp_pred.sample.fov_mm), 4),
                "filter_name": fname,
                "backend": args.backend,
                "epochs": int(args.epochs),
                "itf_ratio": round(float(args.itf_ratio), 4),
                "pitches": pitch_list,
                "series_pass": series_pass,
                "summary": summary,
                "visual_qa_png": str(qa_png),
            }

            if getattr(args, "save_model", False):
                ckpt_path = save_dual_domain_checkpoint(
                    env_prof.data_root / "manifests" / f"stage5_{cid}_dual_domain.pt",
                    model,
                    summary=summary,
                    sino_base_channels=args.sino_channels,
                    img_base_channels=args.img_channels,
                    filter_name=fname,
                    cutoff_ratio=args.cutoff_ratio,
                )
                entry["model_checkpoint_pt"] = str(ckpt_path)

            if getattr(args, "save_dataset_npz", False):
                ds_path = save_dual_domain_dataset_npz(
                    env_prof.data_root / "manifests" / f"stage5_{cid}_dataset.npz",
                    samples,
                    itf_ratio=float(args.itf_ratio),
                )
                entry["dataset_npz"] = str(ds_path)

            trained_series.append(entry)

        phantom_stage5 = None
        if args.run_phantom_check:
            phantom_stage5 = _run_stage5_synthetic_phantom_check(
                figures_dir=figures_dir,
                filter_name=fname,
                backend=args.backend,
                epochs=min(max(4, args.epochs), 10),
                itf_ratio=float(args.itf_ratio),
            )
            if not phantom_stage5["pass"]:
                all_pass = False

        report = {
            "stage": "Stage 5 (v0.8.2)",
            "overall_pass": all_pass,
            "elapsed_seconds": round(time.monotonic() - t_start, 3),
            "trained_series_count": len(trained_series),
            "trained_series": trained_series,
            "synthetic_phantom_dl_check": phantom_stage5,
        }
        out_json = env_prof.data_root / "manifests" / "stage5_dl_report.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_json"] = str(out_json)
        print(json.dumps(report, indent=2))
        return 0 if all_pass else 1

    if args.command in ("stage6-generalize", "stage6-eval"):
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        t_start = time.monotonic()
        env_prof = resolve_environment_profile(data_root=args.data_root)
        show_prog = not args.no_progress
        figures_dir = env_prof.data_root / "manifests" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fname = normalize_filter_name(args.filter_name)

        pitch_list = [float(p.strip()) for p in str(args.pitches).split(",") if p.strip()]
        if not pitch_list:
            raise SystemExit("--pitches must specify at least one high-pitch target")
        scenarios_list = (
            ["itf", "rdc"] if args.scenario == "both" else [args.scenario.strip().lower()]
        )

        targets = _resolve_cli_series_targets(
            env_prof.data_root / "raw" / "tcia",
            case_id=args.case_id,
            series_dir=args.series_dir,
            image_dir=args.image_dir,
            allow_empty_for_phantom=bool(args.run_phantom_check),
        )

        all_pass = True
        real_generalization: dict[str, Any] | None = None

        if targets:
            tr_cases = (
                [c.strip() for c in str(args.train_cases).split(",") if c.strip()]
                if getattr(args, "train_cases", None)
                else None
            )
            te_cases = (
                [c.strip() for c in str(args.test_cases).split(",") if c.strip()]
                if getattr(args, "test_cases", None)
                else None
            )
            sim_backend = "torch_cuda" if args.backend == "torch_cuda" else "numpy"
            split = build_real_series_generalization_split(
                targets,
                split_mode=args.split_mode,
                train_case_ids=tr_cases,
                test_case_ids=te_cases,
                n_train_slices_per_case=int(args.train_slices_per_case),
                n_test_slices_per_case=int(args.test_slices_per_case),
                min_guardband_mm=float(args.min_guardband_mm),
                pitches=pitch_list,
                scenarios=scenarios_list,
                image_size=int(args.image_size),
                fov_mm=args.fov_mm,
                filter_name=fname,
                cutoff_ratio=float(args.cutoff_ratio),
                inpaint_mode=args.inpaint_mode,
                itf_slab_slices=int(args.itf_slab_slices),
                backend=sim_backend,
                workers=int(args.workers),
                show_progress=show_prog,
            )
            audit_obj = split["zero_leakage_audit"]
            exp_res = run_generalization_experiment(
                split["train_samples"],
                split["test_samples"],
                split_mode=split["split_mode"],
                min_guardband_mm=float(args.min_guardband_mm),
                itf_slab_half_span_mm=float(audit_obj.itf_slab_half_span_mm),
                sino_base_channels=int(args.sino_channels),
                img_base_channels=int(args.img_channels),
                filter_name=fname,
                cutoff_ratio=float(args.cutoff_ratio),
                epochs=int(args.epochs),
                lr=float(args.lr),
                itf_ratio=float(args.itf_ratio),
                backend=args.backend,
                seed=42,
                load_checkpoint_path=getattr(args, "load_model", None),
            )

            test_preds = exp_res["unseen_test_predictions"]
            disp_pred = test_preds[len(test_preds) // 2]
            mid_pitch = pitch_list[len(pitch_list) // 2]
            for p in test_preds:
                if p.sample.scenario == "itf" and abs(p.sample.target_pitch - mid_pitch) < 0.15:
                    disp_pred = p
                    break

            train_cids_str = ",".join(exp_res["zero_leakage_audit"]["train_case_ids"])
            test_cid_str = disp_pred.sample.case_id
            qa_png = save_stage6_generalization_visual_qa(
                figures_dir / f"stage6_{test_cid_str}_generalization_visual_qa.png",
                title_prefix=f"Stage 6 Real-Data Generalization QA ({test_cid_str})",
                split_mode=split["split_mode"],
                train_cases_str=train_cids_str,
                test_case_id=test_cid_str,
                scenario=disp_pred.sample.scenario,
                target_pitch=disp_pred.sample.target_pitch,
                effective_pitch=disp_pred.sample.effective_pitch,
                native_pitch=disp_pred.sample.native_pitch,
                z_slice_mm=disp_pred.sample.z_slice_mm,
                fov_mm=disp_pred.sample.fov_mm,
                fov_mask=disp_pred.sample.fov_mask,
                filter_name=fname,
                s_hp_raw=disp_pred.sample.s_hp_raw,
                mask_hp=disp_pred.sample.mask_hp,
                s_dc=disp_pred.s_dc,
                x_target_hu=disp_pred.sample.x_target_hu,
                x_prelim_hu=disp_pred.sample.x_prelim_hu,
                x_dc_hu=disp_pred.x_dc_hu,
                x_final_hu=disp_pred.x_final_hu,
                prelim_metrics=disp_pred.prelim_metrics,
                domain1_dc_metrics=disp_pred.domain1_dc_metrics,
                final_dual_metrics=disp_pred.final_dual_metrics,
                sino_metrics=disp_pred.sinogram_progression_metrics,
                display_vmax_hu=400.0,
            )

            gen_m = exp_res["generalization_metrics"]
            real_pass = bool(
                gen_m["zero_leakage_verified"]
                and gen_m["unseen_test_monotonic_pass"]
                and gen_m["unseen_test_measured_rays_max_abs_dev"] == 0.0
            )
            if not real_pass:
                all_pass = False

            real_generalization = {
                "pass": real_pass,
                "split_mode": split["split_mode"],
                "case_metadata": split["case_metadata"],
                "zero_leakage_audit": exp_res["zero_leakage_audit"],
                "generalization_metrics": gen_m,
                "train_summary": exp_res["train_summary"],
                "unseen_test_summary": exp_res["unseen_test_summary"],
                "visual_qa_png": str(qa_png),
            }

            if getattr(args, "save_model", False):
                ckpt_path = save_dual_domain_checkpoint(
                    env_prof.data_root / "manifests" / "stage6_generalized_dual_domain.pt",
                    exp_res["model"],
                    summary=exp_res["unseen_test_summary"],
                    sino_base_channels=int(args.sino_channels),
                    img_base_channels=int(args.img_channels),
                    filter_name=fname,
                    cutoff_ratio=float(args.cutoff_ratio),
                )
                real_generalization["model_checkpoint_pt"] = str(ckpt_path)

        phantom_stage6 = None
        if args.run_phantom_check:
            phantom_stage6 = _run_stage6_synthetic_generalization_check(
                figures_dir=figures_dir,
                filter_name=fname,
                backend=args.backend,
                epochs=min(max(4, args.epochs), 10),
                itf_ratio=float(args.itf_ratio),
            )
            if not phantom_stage6["pass"]:
                all_pass = False

        report = {
            "stage": "Stage 6 (v0.9.5)",
            "overall_pass": all_pass,
            "elapsed_seconds": round(time.monotonic() - t_start, 3),
            "real_patient_generalization": real_generalization,
            "synthetic_multi_phantom_generalization": phantom_stage6,
        }
        out_json = env_prof.data_root / "manifests" / "stage6_generalization_report.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        report["report_json"] = str(out_json)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"[Stage 6] Saved full report -> {out_json} | overall_pass={all_pass}",
            file=sys.stderr,
            flush=True,
        )
        # Keep stdout compact (< 32 KiB) so sequential stderr->stdout Popen callers never hit OS 64 KiB pipe deadlock
        stdout_report = json.loads(json.dumps(report))
        if stdout_report.get("real_patient_generalization"):
            ts = stdout_report["real_patient_generalization"].get("train_summary")
            if isinstance(ts, dict):
                ts.pop("per_sample", None)
                ts.pop("training_history", None)
        print(json.dumps(stdout_report, separators=(",", ":")), flush=True)
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
