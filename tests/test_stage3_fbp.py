from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from highpitchct.cli import main as cli_main
from highpitchct.phantom import (
    Ellipsoid3D,
    build_thorax_phantom,
    make_synthetic_helical_geometry,
    project_phantom_helical,
    voxelize_phantom_slices_radiological,
    write_synthetic_ctpd_series,
)
from highpitchct.rebinning import (
    rebin_curved_to_flat,
    rebin_helical_ssrb,
)
from highpitchct.recon_fbp import (
    SUPPORTED_FILTER_NAMES,
    build_fbp_filter_kernel_rfft,
    compute_parker_weights,
    compute_reconstruction_metrics,
    extrapolate_sinogram,
    reconstruct_fanbeam_fbp,
    reconstruct_fanbeam_fbp_scalar_ref,
    reconstruct_series_reference_streaming,
)
from highpitchct.visual_qa import save_stage3_phantom_fbp_visual_qa

try:
    import pydicom  # noqa: F401

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"File {path} does not start with valid PNG signature")
    if raw[12:16] != b"IHDR":
        raise AssertionError(f"File {path} missing IHDR chunk")
    width, height = struct.unpack(">II", raw[16:24])
    return int(width), int(height)


class TestStage3ParkerAndFBP(unittest.TestCase):
    """Comprehensive mathematical, physical calibration, parity, and Visual QA tests for Stage 3."""

    def test_parker_conjugate_partition_of_unity_and_pi_quadrature(self) -> None:
        """Prove w(b_tilde, alpha) + w(b_tilde + pi + 2*alpha, -alpha) == 1.0 (<1e-13) & sum == pi."""
        d0_mm = 1085.6
        n_u = 128
        du_mm = 3.5
        u_coords = (np.arange(n_u, dtype=np.float64) - 0.5 * (n_u - 1.0)) * du_mm
        alpha = np.arctan2(u_coords, d0_mm)
        gamma_max = float(np.max(np.abs(alpha)))
        span = math.pi + 2.0 * gamma_max

        n_beta = 721
        beta_1d = -1.2 + np.linspace(0.0, span, n_beta, endpoint=True, dtype=np.float64)
        w = compute_parker_weights(beta_1d, u_coords, d0_mm, angular_mode="short_scan")

        self.assertEqual(w.shape, (n_beta, n_u))
        self.assertTrue(np.all(w >= -1e-15))
        self.assertTrue(np.all(w <= 1.0 + 1e-15))

        # Check discrete integral over beta equals pi across ALL columns including +/-gamma_max
        d_beta = span / float(n_beta - 1)
        col_integrals = np.sum(w, axis=0) * d_beta
        np.testing.assert_allclose(col_integrals, math.pi, atol=2.5e-3, rtol=1e-3)

        # Check exact algebraic conjugate sum w(b1, alpha_m) + w(b1 + pi + 2*alpha_m, -alpha_m) == 1.0
        b_tilde = beta_1d - beta_1d[0]
        for m in range(n_u):
            m_conj = (n_u - 1) - m  # Since u_coords is symmetric, alpha[m_conj] == -alpha[m]
            alpha_m = float(alpha[m])
            upper_low_zone = 2.0 * (gamma_max - alpha_m)
            # Test 25 exact points across the lower redundant zone [0, 2*(gamma_max - alpha_m)]
            b1_samples = np.linspace(0.0, upper_low_zone, 25, endpoint=True, dtype=np.float64)
            b2_samples = b1_samples + math.pi + 2.0 * alpha_m
            # Evaluate Parker weights directly on exact paired grids
            b_grid = np.unique(
                np.concatenate([np.array([0.0, span]), b1_samples, b2_samples])
            )
            w_exact = compute_parker_weights(b_grid, u_coords, d0_mm, angular_mode="short_scan")
            idx1 = np.searchsorted(b_grid, b1_samples)
            idx2 = np.searchsorted(b_grid, b2_samples)
            conj_sum = w_exact[idx1, m] + w_exact[idx2, m_conj]
            np.testing.assert_allclose(
                conj_sum,
                1.0,
                atol=1e-13,
                rtol=1e-13,
                err_msg=f"Failed Parker conjugate partition of unity at column m={m}",
            )

        # Also verify singular corners (0, +gamma_max) and (span, -gamma_max) equal 0.5
        self.assertAlmostEqual(float(w[0, -1]), 0.5, places=13)
        self.assertAlmostEqual(float(w[-1, 0]), 0.5, places=13)
        _ = b_tilde

    def test_spatial_ramp_filters_real_non_negative_and_calibrated(self) -> None:
        """Verify all 5 filter windows preserve finite-aperture DC calibration and symmetry."""
        n_fft = 1024
        du_iso = 0.705
        h_ramp = build_fbp_filter_kernel_rfft(n_fft, du_iso, filter_name="ramp")
        self.assertEqual(h_ramp.shape, (n_fft // 2 + 1,))
        self.assertTrue(np.all(h_ramp >= 0.0))
        # Finite-aperture spatial Ram-Lak DC term is small positive (~ 1 / (pi^2 * N_fft * du_iso)), NEVER negative
        self.assertGreater(float(h_ramp[0]), 0.0)
        self.assertLess(float(h_ramp[0]), 1e-2)

        for fname in SUPPORTED_FILTER_NAMES:
            h_win = build_fbp_filter_kernel_rfft(n_fft, du_iso, filter_name=fname, cutoff_ratio=0.85)
            self.assertEqual(h_win.shape, h_ramp.shape)
            self.assertAlmostEqual(float(h_win[0]), float(h_ramp[0]), places=14)
            # Beyond cutoff_ratio=0.85, filter response must be strictly zero
            f_norm = np.fft.rfftfreq(n_fft, d=1.0) / 0.5
            self.assertTrue(np.all(h_win[f_norm > 0.850001] == 0.0))

    def test_vectorized_vs_scalar_fbp_parity_and_explicit_backend_policy(self) -> None:
        """Verify vectorized NumPy FBP matches scalar reference (<0.05 HU) and torch policy holds."""
        phantom = build_thorax_phantom(axial_scale=0.6)
        geom = make_synthetic_helical_geometry(
            n_views=144,
            n_rows=16,
            n_cols=64,
            angular_steps_per_rotation=72,
            normalized_pitch=0.90,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=6.0,
            dv_mm=3.0,
        )
        proj = project_phantom_helical(phantom, geom, batch_views=72)
        flat_p, valid_p = rebin_curved_to_flat(proj, geom)

        for mode in ("short_scan", "full_scan"):
            ssrb = rebin_helical_ssrb(flat_p, valid_p, geom, [-5.0, 5.0], angular_mode=mode)
            vec_res = reconstruct_fanbeam_fbp(
                ssrb,
                image_size=28,
                fov_mm=340.0,
                filter_name="shepp_logan",
                lateral_pad_cols=24,
                angle_chunk=11,
            )
            ref_res = reconstruct_fanbeam_fbp_scalar_ref(
                ssrb,
                image_size=28,
                fov_mm=340.0,
                filter_name="shepp_logan",
                lateral_pad_cols=24,
            )
            self.assertTrue(np.array_equal(vec_res.fov_mask, ref_res.fov_mask))
            np.testing.assert_allclose(vec_res.mu_slices, ref_res.mu_slices, atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(vec_res.hu_slices, ref_res.hu_slices, atol=0.05, rtol=1e-4)

        if HAS_TORCH:
            ssrb_t = rebin_helical_ssrb(flat_p, valid_p, geom, [0.0], angular_mode="short_scan")
            np_res = reconstruct_fanbeam_fbp(ssrb_t, image_size=48, fov_mm=340.0, backend="numpy")
            th_res = reconstruct_fanbeam_fbp(ssrb_t, image_size=48, fov_mm=340.0, backend="torch")
            np.testing.assert_allclose(np_res.hu_slices, th_res.hu_slices, atol=0.05, rtol=1e-4)
            with mock.patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "Refusing silent fallback"):
                    reconstruct_fanbeam_fbp(
                        ssrb_t, image_size=32, fov_mm=340.0, backend="torch_cuda"
                    )

    def test_water_cylinder_exact_zero_hu_calibration_short_and_full_scan(self) -> None:
        """Prove uniform water cylinder reconstructs to 0.0 HU (+/- 1.5 HU) in short & full scan."""
        water_cyl = [Ellipsoid3D("water", (0.0, 0.0, 0.0), (110.0, 110.0, 220.0), 1000.0, 0.0)]
        geom = make_synthetic_helical_geometry(
            n_views=576,
            n_rows=24,
            n_cols=256,
            angular_steps_per_rotation=384,
            normalized_pitch=0.80,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=2.0,
            dv_mm=2.0,
        )
        proj = project_phantom_helical(water_cyl, geom, batch_views=128)
        flat_p, valid_p = rebin_curved_to_flat(proj, geom)

        ssrb_short = rebin_helical_ssrb(flat_p, valid_p, geom, [0.0], angular_mode="short_scan")
        ssrb_full = rebin_helical_ssrb(flat_p, valid_p, geom, [0.0], angular_mode="full_scan")

        fbp_short = reconstruct_fanbeam_fbp(ssrb_short, image_size=192, fov_mm=340.0, filter_name="hann")
        fbp_full = reconstruct_fanbeam_fbp(ssrb_full, image_size=192, fov_mm=340.0, filter_name="hann")

        yy, xx = np.meshgrid(fbp_short.y_mm, fbp_short.x_mm, indexing="ij")
        roi = (xx * xx + yy * yy) <= (85.0 ** 2)

        mean_short = float(np.mean(fbp_short.hu_slices[0][roi]))
        std_short = float(np.std(fbp_short.hu_slices[0][roi]))
        mean_full = float(np.mean(fbp_full.hu_slices[0][roi]))
        std_full = float(np.std(fbp_full.hu_slices[0][roi]))
        diff_sf = float(np.mean(np.abs(fbp_short.hu_slices[0][roi] - fbp_full.hu_slices[0][roi])))

        self.assertLess(abs(mean_short), 1.5, msg=f"Short-scan water HU bias: {mean_short}")
        self.assertLess(std_short, 3.5)
        self.assertLess(abs(mean_full), 1.5, msg=f"Full-scan water HU bias: {mean_full}")
        self.assertLess(std_full, 3.5)
        self.assertLess(diff_sf, 1.0)

    def test_thorax_phantom_radiological_orientation_and_high_pitch_completion(self) -> None:
        """Verify anatomical ROI orientation, SSIM >= 0.88, mask immutability, & streak reduction."""
        phantom = build_thorax_phantom(axial_scale=0.8)
        z_sl = -12.0

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
            z_center_mm=z_sl,
        )
        proj_low = project_phantom_helical(phantom, geom_low, batch_views=128)
        flat_low, valid_low = rebin_curved_to_flat(proj_low, geom_low)
        ssrb_low = rebin_helical_ssrb(flat_low, valid_low, geom_low, [z_sl], angular_mode="short_scan")
        fbp_ref = reconstruct_fanbeam_fbp(ssrb_low, image_size=256, fov_mm=360.0, filter_name="hann")

        gt = voxelize_phantom_slices_radiological(
            phantom, z_slices_mm=[z_sl], image_size=256, fov_mm=360.0
        )["hu_slices"][0]
        ref_metrics = compute_reconstruction_metrics(
            fbp_ref.hu_slices[0], gt, fov_mask=fbp_ref.fov_mask
        )
        self.assertGreaterEqual(ref_metrics["ssim"], 0.88)

        # Verify radiological supine coordinates: Row 0 = +y (Sternum), Row H-1 = -y (Spine)
        self.assertGreater(float(fbp_ref.y_mm[0]), 0.0)
        self.assertLess(float(fbp_ref.y_mm[-1]), 0.0)
        self.assertLess(float(fbp_ref.x_mm[0]), 0.0)
        self.assertGreater(float(fbp_ref.x_mm[-1]), 0.0)

        yy, xx = np.meshgrid(fbp_ref.y_mm, fbp_ref.x_mm, indexing="ij")
        roi_heart = ((xx - 8.0) ** 2 + (yy - 36.0) ** 2) <= 10.0 ** 2
        heart_hu = float(np.mean(fbp_ref.hu_slices[0][roi_heart]))
        gt_heart_hu = float(np.mean(gt[roi_heart]))
        rlung_hu = float(np.mean(fbp_ref.hu_slices[0][((xx + 64.0) ** 2 + (yy - 8.0) ** 2) <= 22.0 ** 2]))
        spine_hu = float(np.mean(fbp_ref.hu_slices[0][((xx - 0.0) ** 2 + (yy + 68.0) ** 2) <= 8.0 ** 2]))
        self.assertTrue(40.0 <= heart_hu <= 75.0, msg=f"Unexpected heart HU={heart_hu} (GT={gt_heart_hu})")
        self.assertLess(abs(heart_hu - gt_heart_hu), 10.0)
        self.assertTrue(-800.0 <= rlung_hu <= -700.0, msg=f"Unexpected right lung HU={rlung_hu}")
        self.assertGreaterEqual(spine_hu, 120.0, msg=f"Unexpected spine HU={spine_hu}")

        # High-pitch = 1.85 missing-wedge extrapolation & mask immutability
        geom_hp = make_synthetic_helical_geometry(
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
            z_center_mm=z_sl,
        )
        proj_hp = project_phantom_helical(phantom, geom_hp, batch_views=128)
        flat_hp, valid_hp = rebin_curved_to_flat(proj_hp, geom_hp)
        ssrb_hp = rebin_helical_ssrb(
            flat_hp,
            valid_hp,
            geom_hp,
            [z_sl],
            angular_mode="short_scan",
            active_row_range=(4.0, 27.0),
        )
        mask_snapshot = ssrb_hp.measured_mask.copy()

        prep_cf = extrapolate_sinogram(ssrb_hp, inpaint_mode="conjugate_feather")
        # Original SSRBResult.measured_mask MUST NEVER BE MUTATED!
        self.assertTrue(np.array_equal(ssrb_hp.measured_mask, mask_snapshot))
        self.assertTrue(np.array_equal(prep_cf.original_measured_mask, mask_snapshot))
        self.assertGreater(float(np.mean(prep_cf.conjugate_filled_mask)), 0.02)

        fbp_hp_raw = reconstruct_fanbeam_fbp(
            ssrb_hp, image_size=256, fov_mm=360.0, filter_name="hann", inpaint_mode="none"
        )
        fbp_hp_cf = reconstruct_fanbeam_fbp(
            ssrb_hp,
            image_size=256,
            fov_mm=360.0,
            filter_name="hann",
            inpaint_mode="conjugate_feather",
        )
        m_raw = compute_reconstruction_metrics(fbp_hp_raw.hu_slices[0], gt, fov_mask=fbp_ref.fov_mask)
        m_cf = compute_reconstruction_metrics(fbp_hp_cf.hu_slices[0], gt, fov_mask=fbp_ref.fov_mask)
        m_raw_ref = compute_reconstruction_metrics(
            fbp_hp_raw.hu_slices[0], fbp_ref.hu_slices[0], fov_mask=fbp_ref.fov_mask
        )
        m_cf_ref = compute_reconstruction_metrics(
            fbp_hp_cf.hu_slices[0], fbp_ref.hu_slices[0], fov_mask=fbp_ref.fov_mask
        )
        self.assertLess(m_cf["rmse_hu"], m_raw["rmse_hu"] - 40.0)
        self.assertGreater(m_cf["ssim"], m_raw["ssim"] + 0.15)
        self.assertLess(m_cf_ref["rmse_hu"], m_raw_ref["rmse_hu"] - 40.0)

        # Verify mandatory Visual QA PNG generation (both matplotlib & pure-stdlib fallback)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "stage3_mpl.png"
            save_stage3_phantom_fbp_visual_qa(
                p1,
                gt_hu=gt,
                ref_fbp_hu=fbp_ref.hu_slices[0],
                high_pitch_raw_fbp_hu=fbp_hp_raw.hu_slices[0],
                high_pitch_inpaint_fbp_hu=fbp_hp_cf.hu_slices[0],
                fov_mask=fbp_ref.fov_mask,
                fov_mm=fbp_ref.fov_mm,
                z_slice_mm=z_sl,
                low_pitch=0.85,
                high_pitch=1.85,
                ref_metrics=ref_metrics,
                hp_raw_metrics=m_raw,
                hp_inpaint_metrics=m_cf,
            )
            w1, h1 = _read_png_dimensions(p1)
            self.assertGreater(w1, 600)
            self.assertGreater(h1, 400)

            p2 = Path(tmp) / "stage3_stdlib.png"
            with mock.patch("highpitchct.visual_qa._has_matplotlib", return_value=False):
                save_stage3_phantom_fbp_visual_qa(
                    p2,
                    gt_hu=gt,
                    ref_fbp_hu=fbp_ref.hu_slices[0],
                    high_pitch_raw_fbp_hu=fbp_hp_raw.hu_slices[0],
                    high_pitch_inpaint_fbp_hu=fbp_hp_cf.hu_slices[0],
                    fov_mask=fbp_ref.fov_mask,
                    fov_mm=fbp_ref.fov_mm,
                    z_slice_mm=z_sl,
                    low_pitch=0.85,
                    high_pitch=1.85,
                    ref_metrics=ref_metrics,
                    hp_raw_metrics=m_raw,
                    hp_inpaint_metrics=m_cf,
                )
            w2, h2 = _read_png_dimensions(p2)
            self.assertGreater(w2, 600)
            self.assertGreater(h2, 400)


@unittest.skipUnless(HAS_PYDICOM, "pydicom required for DICOM-CT-PD streaming Stage 3 CLI test")
class TestStage3StreamingAndCLI(unittest.TestCase):
    """End-to-end streaming DICOM-CT-PD -> SSRB -> Parker -> FBP and `hpct stage3-recon` CLI test."""

    def test_streaming_reference_recon_and_cli_stage3_recon(self) -> None:
        phantom = build_thorax_phantom(axial_scale=0.6)
        geom = make_synthetic_helical_geometry(
            n_views=384,
            n_rows=32,
            n_cols=128,
            angular_steps_per_rotation=192,
            normalized_pitch=0.90,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=3.8,
            dv_mm=2.0,
        )
        proj = project_phantom_helical(phantom, geom, batch_views=96)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj_dir = (
                root / "raw" / "tcia" / "case_C012" / "projection_full" / geom.series_instance_uid
            )
            write_synthetic_ctpd_series(proj_dir, geom, proj)

            # Verify streaming matches in-memory FBP bit-for-bit
            flat_m, valid_m = rebin_curved_to_flat(proj, geom)
            ssrb_mem = rebin_helical_ssrb(flat_m, valid_m, geom, [-6.0, 0.0, 6.0])
            fbp_mem = reconstruct_fanbeam_fbp(
                ssrb_mem,
                image_size=128,
                water_mu_mm_inv=geom.water_mu_mm_inv,
                filter_name="hann",
            )
            ssrb_str, fbp_str = reconstruct_series_reference_streaming(
                proj_dir,
                [-6.0, 0.0, 6.0],
                image_size=128,
                filter_name="hann",
                workers=2,
            )
            self.assertTrue(np.array_equal(ssrb_mem.measured_mask, ssrb_str.measured_mask))
            np.testing.assert_allclose(fbp_mem.hu_slices, fbp_str.hu_slices, atol=1e-4, rtol=1e-5)

            # Run CLI `hpct stage3-recon --case-id C012 --run-phantom-check`
            npz_out = root / "manifests" / "c012_stage3_ref.npz"
            rc = cli_main(
                [
                    "stage3-recon",
                    "--data-root",
                    str(root),
                    "--case-id",
                    "C012",
                    "--num-slices",
                    "3",
                    "--image-size",
                    "128",
                    "--recon-out",
                    str(npz_out),
                    "--run-phantom-check",
                    "--workers",
                    "2",
                    "--no-progress",
                ]
            )
            self.assertEqual(rc, 0)
            report_path = root / "manifests" / "stage3_recon_report.json"
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["overall_pass"])
            self.assertEqual(report["reconstructed_series_count"], 1)
            self.assertTrue(npz_out.is_file())

            p_real = Path(report["reconstructed_series"][0]["visual_qa_png"])
            p_phan = Path(report["synthetic_phantom_fbp_check"]["visual_qa_png"])
            self.assertTrue(p_real.is_file())
            self.assertTrue(p_phan.is_file())
            self.assertGreater(_read_png_dimensions(p_real)[0], 600)
            self.assertGreater(_read_png_dimensions(p_phan)[0], 600)


if __name__ == "__main__":
    unittest.main()
