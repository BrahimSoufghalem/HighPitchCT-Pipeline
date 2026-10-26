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
from highpitchct.forward_project import (
    build_dual_domain_pair_from_ssrb,
    compute_rdc_active_row_range,
    forward_project_fanbeam_2d,
    forward_project_fanbeam_2d_scalar_ref,
    forward_project_helical_3d,
    synthesize_itf_helical_geometry,
)
from highpitchct.phantom import (
    Ellipsoid3D,
    build_thorax_phantom,
    make_synthetic_helical_geometry,
    project_phantom_helical,
    voxelize_phantom_slices_radiological,
    write_synthetic_ctpd_series,
)
from highpitchct.rebinning import (
    SSRBResult,
    rebin_curved_to_flat,
    rebin_helical_ssrb,
)
from highpitchct.recon_fbp import (
    compute_reconstruction_metrics,
    reconstruct_fanbeam_fbp,
)
from highpitchct.visual_qa import (
    save_stage4_phantom_forward_visual_qa,
    save_stage4_real_simulation_visual_qa,
)

try:
    import pydicom  # noqa: F401
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

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


def _write_synthetic_vendor_ct_slices(
    img_dir: Path,
    hu_slices: np.ndarray,
    z_slices_mm: np.ndarray,
    fov_mm: float = 400.0,
    patient_id: str = "C012",
    series_uid: str | None = None,
) -> str:
    """Write a minimal valid DICOM CT image series with explicit FOV for vendor zoom matching."""
    img_dir.mkdir(parents=True, exist_ok=True)
    suid = series_uid or generate_uid()
    study_uid = generate_uid()
    n_slices, n_y, n_x = hu_slices.shape
    ps_mm = float(fov_mm) / float(n_x)
    half_fov = 0.5 * float(fov_mm) - 0.5 * ps_mm

    for idx in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(
            str(img_dir / f"slice_{idx:04d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128
        )
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = suid
        ds.SeriesNumber = 2
        ds.InstanceNumber = idx + 1
        ds.Rows = int(n_y)
        ds.Columns = int(n_x)
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleIntercept = -1024.0
        ds.RescaleSlope = 1.0
        ds.PixelSpacing = [f"{ps_mm:.6f}", f"{ps_mm:.6f}"]
        ds.SliceThickness = 1.0
        ds.ConvolutionKernel = "B40f"
        z_val = float(z_slices_mm[idx])
        ds.SliceLocation = z_val
        ds.ImagePositionPatient = [f"{-half_fov:.4f}", f"{-half_fov:.4f}", f"{z_val:.4f}"]
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]

        raw_u16 = np.clip(np.round(hu_slices[idx] + 1024.0), 0.0, 65535.0).astype("<u2")
        ds.PixelData = raw_u16.tobytes()
        ds.save_as(str(img_dir / f"slice_{idx:04d}.dcm"), enforce_file_format=True)
    return suid


class TestStage4ForwardProjectAndHighPitchSim(unittest.TestCase):
    """Comprehensive mathematical, physical, parity, and Visual QA tests for Stage 4."""

    def test_rdc_active_row_range_and_itf_geometry_synthesis(self) -> None:
        """Verify poster Fig. 1(b) RDC row ranges & Fig. 1(c) ITF helical trajectory synthesis."""
        # Poster nominal p_native = 0.6 with 64 rows -> 16, 12, 8 central rows for p = 2.4, 3.2, 4.8
        r_lo_24, r_hi_24, eff_24 = compute_rdc_active_row_range(64, 0.6, 2.4)
        self.assertEqual((r_lo_24, r_hi_24), (24.0, 39.0))
        self.assertEqual(int(r_hi_24 - r_lo_24 + 1.0), 16)
        self.assertAlmostEqual(eff_24, 2.4, places=5)

        r_lo_32, r_hi_32, eff_32 = compute_rdc_active_row_range(64, 0.6, 3.2)
        self.assertEqual((r_lo_32, r_hi_32), (26.0, 37.0))
        self.assertEqual(int(r_hi_32 - r_lo_32 + 1.0), 12)
        self.assertAlmostEqual(eff_32, 3.2, places=5)

        r_lo_48, r_hi_48, eff_48 = compute_rdc_active_row_range(64, 0.6, 4.8)
        self.assertEqual((r_lo_48, r_hi_48), (28.0, 35.0))
        self.assertEqual(int(r_hi_48 - r_lo_48 + 1.0), 8)
        self.assertAlmostEqual(eff_48, 4.8, places=5)

        # Case C012 actual p_native = 0.90 with 64 rows -> 24, 18, 12 central rows
        self.assertEqual(compute_rdc_active_row_range(64, 0.90, 2.4)[:2], (20.0, 43.0))
        self.assertEqual(compute_rdc_active_row_range(64, 0.90, 3.2)[:2], (23.0, 40.0))
        self.assertEqual(compute_rdc_active_row_range(64, 0.90, 4.8)[:2], (26.0, 37.0))

        with self.assertRaises(ValueError):
            compute_rdc_active_row_range(1, 0.90, 2.4)
        with self.assertRaises(ValueError):
            compute_rdc_active_row_range(64, -0.5, 2.4)

        # Verify ITF helical trajectory synthesis preserves exact 3D FFS perturbations
        geom_low = make_synthetic_helical_geometry(
            n_views=384,
            n_rows=32,
            n_cols=128,
            angular_steps_per_rotation=192,
            normalized_pitch=0.90,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=3.5,
            dv_mm=1.2,
        )
        for p_target in (2.4, 3.2, 4.8):
            geom_itf = synthesize_itf_helical_geometry(geom_low, target_pitch=p_target)
            traj_itf = geom_itf.trajectory_metrics()
            self.assertAlmostEqual(float(traj_itf["normalized_pitch"]), p_target, places=4)
            n_chk = min(geom_itf.n_views, geom_low.n_views)
            np.testing.assert_allclose(
                geom_itf.drho_mm[:n_chk], geom_low.drho_mm[:n_chk], atol=1e-12
            )
            np.testing.assert_allclose(
                geom_itf.dphi_rad[:n_chk], geom_low.dphi_rad[:n_chk], atol=1e-12
            )
            np.testing.assert_allclose(
                geom_itf.dz_mm[:n_chk], geom_low.dz_mm[:n_chk], atol=1e-12
            )

    def test_fanbeam_2d_forward_vectorized_vs_scalar_parity_and_backends(self) -> None:
        """Verify vectorized 2D fan-beam projector matches scalar reference (<1e-5) & torch policy."""
        phantom = build_thorax_phantom(axial_scale=0.6)
        z_slices = np.array([-4.0, 4.0], dtype=np.float64)
        vox = voxelize_phantom_slices_radiological(
            phantom, z_slices_mm=z_slices, image_size=24, fov_mm=320.0
        )
        mu_slices = vox["mu_slices"]

        d0_mm = 1085.6
        rho0_mm = 595.0
        n_u = 48
        du_mm = 6.5
        u_coords = (np.arange(n_u, dtype=np.float64) - 0.5 * (n_u - 1.0)) * du_mm
        alpha = np.arctan2(u_coords, d0_mm)
        span = math.pi + 2.0 * float(np.max(np.abs(alpha)))
        beta_rad = -1.1 + np.linspace(0.0, span, 36, endpoint=True, dtype=np.float64)

        sino_vec = forward_project_fanbeam_2d(
            mu_slices,
            beta_rad,
            u_coords,
            rho0_mm=rho0_mm,
            d0_mm=d0_mm,
            fov_mm=320.0,
            step_factor=0.5,
            backend="numpy",
            angle_chunk=9,
        )
        sino_ref_0 = forward_project_fanbeam_2d_scalar_ref(
            mu_slices[0],
            beta_rad,
            u_coords,
            rho0_mm=rho0_mm,
            d0_mm=d0_mm,
            fov_mm=320.0,
            step_factor=0.5,
        )
        sino_ref_1 = forward_project_fanbeam_2d_scalar_ref(
            mu_slices[1],
            beta_rad,
            u_coords,
            rho0_mm=rho0_mm,
            d0_mm=d0_mm,
            fov_mm=320.0,
            step_factor=0.5,
        )
        self.assertEqual(sino_vec.shape, (2, 36, 48))
        np.testing.assert_allclose(sino_vec[0], sino_ref_0, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(sino_vec[1], sino_ref_1, atol=1e-5, rtol=1e-5)

        if HAS_TORCH:
            sino_th = forward_project_fanbeam_2d(
                mu_slices,
                beta_rad,
                u_coords,
                rho0_mm=rho0_mm,
                d0_mm=d0_mm,
                fov_mm=320.0,
                step_factor=0.5,
                backend="torch",
                angle_chunk=12,
            )
            np.testing.assert_allclose(sino_vec, sino_th, atol=1e-4, rtol=1e-4)

            with mock.patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no silent fallback"):
                    forward_project_fanbeam_2d(
                        mu_slices,
                        beta_rad,
                        u_coords,
                        rho0_mm=rho0_mm,
                        d0_mm=d0_mm,
                        fov_mm=320.0,
                        step_factor=0.5,
                        backend="torch_cuda",
                    )

    def test_forward_to_ram_lak_fbp_self_consistency_roundtrip(self) -> None:
        """Prove 2D Forward Projection -> Ram-Lak FBP reconstructs water to 0 HU & thorax SSIM >= 0.90."""
        d0_mm = 1085.6
        rho0_mm = 595.0
        n_u = 320
        du_mm = 2.2
        u_coords = (np.arange(n_u, dtype=np.float64) - 0.5 * (n_u - 1.0)) * du_mm
        alpha = np.arctan2(u_coords, d0_mm)
        gamma_max = float(np.max(np.abs(alpha)))
        span = math.pi + 2.0 * gamma_max
        beta_1d = -0.5 * math.pi + np.linspace(0.0, span, 280, endpoint=True, dtype=np.float64)

        # 1. Uniform water cylinder roundtrip
        water_cyl = [Ellipsoid3D("water", (0.0, 0.0, 0.0), (110.0, 110.0, 200.0), 1000.0, 0.0)]
        vox_w = voxelize_phantom_slices_radiological(
            water_cyl, z_slices_mm=[0.0], image_size=192, fov_mm=340.0
        )
        sino_w = forward_project_fanbeam_2d(
            vox_w["mu_slices"],
            beta_1d,
            u_coords,
            rho0_mm=rho0_mm,
            d0_mm=d0_mm,
            fov_mm=340.0,
        )
        ssrb_w = SSRBResult(
            sinograms=sino_w,
            measured_mask=np.ones_like(sino_w, dtype=bool),
            z_slices_mm=np.array([0.0], dtype=np.float64),
            beta_rad=beta_1d[None, :],
            phi_center_rad=np.array([0.0], dtype=np.float64),
            u_coords_mm=u_coords,
            du_mm=du_mm,
            rho0_mm=rho0_mm,
            d0_mm=d0_mm,
            gamma_max_rad=gamma_max,
            angular_mode="short_scan",
            angular_span_rad=span,
            active_row_range=(0.0, 31.0),
        )
        fbp_w = reconstruct_fanbeam_fbp(ssrb_w, image_size=192, fov_mm=340.0, filter_name="ram_lak")
        yy, xx = np.meshgrid(fbp_w.y_mm, fbp_w.x_mm, indexing="ij")
        roi_w = (xx * xx + yy * yy) <= (85.0 ** 2)
        mean_water_hu = float(np.mean(fbp_w.hu_slices[0][roi_w]))
        self.assertLess(
            abs(mean_water_hu), 1.5, msg=f"Ram-Lak water roundtrip HU bias: {mean_water_hu}"
        )

        # 2. 3D Thorax Phantom axial slice roundtrip with Ram-Lak filter
        phantom = build_thorax_phantom(axial_scale=0.8)
        z_sl = -12.0
        vox_t = voxelize_phantom_slices_radiological(
            phantom, z_slices_mm=[z_sl], image_size=192, fov_mm=360.0
        )
        sino_t = forward_project_fanbeam_2d(
            vox_t["mu_slices"],
            beta_1d,
            u_coords,
            rho0_mm=rho0_mm,
            d0_mm=d0_mm,
            fov_mm=360.0,
        )
        ssrb_t = SSRBResult(
            sinograms=sino_t,
            measured_mask=np.ones_like(sino_t, dtype=bool),
            z_slices_mm=np.array([z_sl], dtype=np.float64),
            beta_rad=beta_1d[None, :],
            phi_center_rad=np.array([0.0], dtype=np.float64),
            u_coords_mm=u_coords,
            du_mm=du_mm,
            rho0_mm=rho0_mm,
            d0_mm=d0_mm,
            gamma_max_rad=gamma_max,
            angular_mode="short_scan",
            angular_span_rad=span,
            active_row_range=(0.0, 31.0),
        )
        fbp_t = reconstruct_fanbeam_fbp(ssrb_t, image_size=192, fov_mm=360.0, filter_name="ram_lak")
        m_rt = compute_reconstruction_metrics(
            fbp_t.hu_slices[0], vox_t["hu_slices"][0], fov_mask=fbp_t.fov_mask
        )
        self.assertGreaterEqual(m_rt["ssim"], 0.90)
        self.assertLessEqual(m_rt["rmse_hu"], 35.0)

    def test_helical_3d_forward_projector_and_dual_domain_pair_generation(self) -> None:
        """Verify 3D helical forward projector & AAPM 2026 Fig. 2(a)-(d) dual-domain pair builder."""
        phantom = build_thorax_phantom(axial_scale=0.8)
        z_sl = -12.0
        geom_low = make_synthetic_helical_geometry(
            n_views=576,
            n_rows=32,
            n_cols=256,
            angular_steps_per_rotation=288,
            normalized_pitch=0.85,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=2.6,
            dv_mm=1.8,
            z_center_mm=z_sl,
        )
        proj_low = project_phantom_helical(phantom, geom_low, batch_views=144)
        flat_low, valid_low = rebin_curved_to_flat(proj_low, geom_low)
        ssrb_low = rebin_helical_ssrb(flat_low, valid_low, geom_low, [z_sl], angular_mode="short_scan")

        # Simulate RDC high-pitch p=2.4
        r_min, r_max, eff_rdc_p = compute_rdc_active_row_range(geom_low.n_rows, 0.85, 2.4)
        ssrb_rdc = rebin_helical_ssrb(
            flat_low,
            valid_low,
            geom_low,
            [z_sl],
            angular_mode="short_scan",
            active_row_range=(r_min, r_max),
        )
        mask_copy = ssrb_rdc.measured_mask.copy()

        pair = build_dual_domain_pair_from_ssrb(
            ssrb_low,
            ssrb_rdc,
            scenario="rdc",
            target_pitch=2.4,
            effective_pitch=eff_rdc_p,
            native_pitch=0.85,
            active_row_range=(r_min, r_max),
            image_size=160,
            fov_mm=360.0,
            water_mu_mm_inv=geom_low.water_mu_mm_inv,
            filter_name="ram_lak",
            inpaint_mode="conjugate_feather",
        )
        # Ensure original SSRBResult.measured_mask was NEVER mutated
        self.assertTrue(np.array_equal(ssrb_rdc.measured_mask, mask_copy))
        self.assertTrue(np.array_equal(pair.mask_hp, mask_copy))

        # Verify exact data consistency on measured pixels: s_reproj_dc[mask_hp] == s_hp_raw[mask_hp]
        np.testing.assert_allclose(
            pair.s_reproj_dc[pair.mask_hp],
            pair.s_hp_raw[pair.mask_hp],
            atol=1e-7,
            rtol=1e-7,
        )
        # Verify s_reproj dramatically lowers missing-wedge error against s_target
        self.assertLess(
            pair.sinogram_metrics["reproj_missing_wedge_mae"],
            0.35 * pair.sinogram_metrics["raw_missing_wedge_mae"],
        )

        # Verify 3D helical forward projector on a multi-slice slab (numpy vs torch & CUDA fail-fast)
        z_slab = np.linspace(z_sl - 20.0, z_sl + 20.0, 9, dtype=np.float64)
        vox_slab = voxelize_phantom_slices_radiological(
            phantom, z_slices_mm=z_slab, image_size=64, fov_mm=340.0
        )
        geom_sub = make_synthetic_helical_geometry(
            n_views=32,
            n_rows=16,
            n_cols=64,
            angular_steps_per_rotation=32,
            normalized_pitch=2.4,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=5.0,
            dv_mm=2.5,
            z_center_mm=z_sl,
        )
        proj_3d_np = forward_project_helical_3d(
            vox_slab["mu_slices"],
            z_slab,
            vox_slab["y_mm"],
            vox_slab["x_mm"],
            geom_sub,
            step_mm=3.0,
            backend="numpy",
        )
        self.assertEqual(proj_3d_np.shape, (32, 16, 64))
        self.assertGreater(float(np.max(proj_3d_np)), 0.1)

        if HAS_TORCH:
            proj_3d_th = forward_project_helical_3d(
                vox_slab["mu_slices"],
                z_slab,
                vox_slab["y_mm"],
                vox_slab["x_mm"],
                geom_sub,
                step_mm=3.0,
                backend="torch",
            )
            np.testing.assert_allclose(proj_3d_np, proj_3d_th, atol=2e-4, rtol=1e-4)

            with mock.patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no silent fallback"):
                    forward_project_helical_3d(
                        vox_slab["mu_slices"],
                        z_slab,
                        vox_slab["y_mm"],
                        vox_slab["x_mm"],
                        geom_sub,
                        step_mm=3.0,
                        backend="torch_cuda",
                    )

        # Verify Stage 4 Visual QA PNG generators (both matplotlib and pure-stdlib fallback)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "s4_phantom_mpl.png"
            p2 = Path(tmp) / "s4_phantom_std.png"
            p3 = Path(tmp) / "s4_real_std.png"

            save_stage4_phantom_forward_visual_qa(
                p1,
                gt_hu=pair.x_target_hu[0],
                roundtrip_fbp_hu=pair.x_target_hu[0],
                roundtrip_metrics={"mae_hu": 8.0, "rmse_hu": 12.0, "psnr_db": 38.0, "ssim": 0.95},
                s_hp_raw=pair.s_hp_raw[0],
                mask_hp=pair.mask_hp[0],
                s_reproj=pair.s_reproj[0],
                rdc_prelim_hu=pair.x_prelim_hu[0],
                rdc_metrics=pair.prelim_vs_target_metrics,
                rdc_pitch=pair.effective_pitch,
                itf_prelim_hu=pair.x_prelim_hu[0],
                itf_metrics=pair.prelim_vs_target_metrics,
                itf_pitch=2.4,
                fov_mm=pair.fov_mm,
                z_slice_mm=z_sl,
                filter_name="ram_lak",
            )
            self.assertGreater(_read_png_dimensions(p1)[0], 600)

            with mock.patch("highpitchct.visual_qa._has_matplotlib", return_value=False):
                save_stage4_phantom_forward_visual_qa(
                    p2,
                    gt_hu=pair.x_target_hu[0],
                    roundtrip_fbp_hu=pair.x_target_hu[0],
                    roundtrip_metrics={"mae_hu": 8.0, "rmse_hu": 12.0, "psnr_db": 38.0, "ssim": 0.95},
                    s_hp_raw=pair.s_hp_raw[0],
                    mask_hp=pair.mask_hp[0],
                    s_reproj=pair.s_reproj[0],
                    rdc_prelim_hu=pair.x_prelim_hu[0],
                    rdc_metrics=pair.prelim_vs_target_metrics,
                    rdc_pitch=pair.effective_pitch,
                    itf_prelim_hu=pair.x_prelim_hu[0],
                    itf_metrics=pair.prelim_vs_target_metrics,
                    itf_pitch=2.4,
                    fov_mm=pair.fov_mm,
                    z_slice_mm=z_sl,
                    filter_name="ram_lak",
                )
                save_stage4_real_simulation_visual_qa(
                    p3,
                    case_id="case_C012",
                    z_slice_mm=z_sl,
                    native_pitch=0.85,
                    fov_mm=pair.fov_mm,
                    fov_mask=pair.fov_mask,
                    filter_name="ram_lak",
                    x_target_hu=pair.x_target_hu[0],
                    s_target=pair.s_target[0],
                    s_hp_raw=pair.s_hp_raw[0],
                    mask_hp=pair.mask_hp[0],
                    s_reproj=pair.s_reproj[0],
                    rdc_pitch=pair.effective_pitch,
                    rdc_prelim_hu=pair.x_prelim_hu[0],
                    rdc_metrics=pair.prelim_vs_target_metrics,
                    itf_pitch=2.4,
                    itf_prelim_hu=pair.x_prelim_hu[0],
                    itf_metrics=pair.prelim_vs_target_metrics,
                )
            self.assertGreater(_read_png_dimensions(p2)[0], 600)
            self.assertGreater(_read_png_dimensions(p3)[0], 600)


@unittest.skipUnless(HAS_PYDICOM, "pydicom required for Stage 4 DICOM-CT-PD & CLI test")
class TestStage4StreamingAndCLI(unittest.TestCase):
    """End-to-end test for `hpct stage4-sim` CLI (RDC + ITF) and automatic Vendor CT FOV matching."""

    def test_cli_stage4_sim_streaming_rdc_itf_and_vendor_fov_matching(self) -> None:
        phantom = build_thorax_phantom(axial_scale=0.6)
        geom = make_synthetic_helical_geometry(
            n_views=384,
            n_rows=32,
            n_cols=144,  # max_fov_mm = 456.8 mm > 400.0 mm vendor FOV
            angular_steps_per_rotation=192,
            normalized_pitch=0.90,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=3.6,
            dv_mm=2.0,
        )
        proj = project_phantom_helical(phantom, geom, batch_views=96)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_uid = generate_uid()
            proj_dir = (
                root / "raw" / "tcia" / "case_C012" / "projection_full" / geom.series_instance_uid
            )
            img_dir = (
                root / "raw" / "tcia" / "case_C012" / "reconstruction_published" / img_uid
            )

            write_synthetic_ctpd_series(proj_dir, geom, proj, patient_id="C012")
            z_vendor = np.linspace(-18.0, 18.0, 9, dtype=np.float64)
            vox_vendor = voxelize_phantom_slices_radiological(
                phantom, z_slices_mm=z_vendor, image_size=64, fov_mm=400.0
            )
            _write_synthetic_vendor_ct_slices(
                img_dir,
                vox_vendor["hu_slices"],
                z_vendor,
                fov_mm=400.0,
                patient_id="C012",
                series_uid=img_uid,
            )

            # 1. Verify stage3-recon defaults to ram_lak AND automatically matches vendor 400.0 mm FOV
            rc_s3 = cli_main(
                [
                    "stage3-recon",
                    "--data-root",
                    str(root),
                    "--case-id",
                    "C012",
                    "--num-slices",
                    "1",
                    "--n-xy",
                    "64",
                    "--workers",
                    "2",
                    "--no-progress",
                ]
            )
            self.assertEqual(rc_s3, 0)
            s3_rep = json.loads(
                (root / "manifests" / "stage3_recon_report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(s3_rep["overall_pass"])
            s3_summary = s3_rep["reconstructed_series"][0]["reference_fbp_summary"]
            self.assertEqual(s3_summary["filter_name"], "ram_lak")
            self.assertAlmostEqual(float(s3_summary["fov_mm"]), 400.0, places=3)

            # 2. Verify stage4-sim runs both RDC and ITF, matches 400.0 mm FOV, writes NPZ and PNGs
            rc_s4 = cli_main(
                [
                    "stage4-sim",
                    "--data-root",
                    str(root),
                    "--case-id",
                    "C012",
                    "--scenario",
                    "both",
                    "--target-pitch",
                    "2.4",
                    "--pitches",
                    "2.4,3.2",
                    "--num-slices",
                    "1",
                    "--n-xy",
                    "64",
                    "--itf-slab-slices",
                    "5",
                    "--save-pairs-npz",
                    "--run-phantom-check",
                    "--workers",
                    "2",
                    "--no-progress",
                ]
            )
            self.assertEqual(rc_s4, 0)
            s4_path = root / "manifests" / "stage4_sim_report.json"
            self.assertTrue(s4_path.is_file())
            s4_rep = json.loads(s4_path.read_text(encoding="utf-8"))
            self.assertTrue(s4_rep["overall_pass"])
            self.assertEqual(s4_rep["simulated_series_count"], 1)

            sim_entry = s4_rep["simulated_series"][0]
            self.assertEqual(sim_entry["filter_name"], "ram_lak")
            self.assertAlmostEqual(float(sim_entry["fov_mm"]), 400.0, places=3)
            self.assertIn("p_2.4", sim_entry["rdc_multi_pitch_summary"])
            self.assertIn("p_3.2", sim_entry["rdc_multi_pitch_summary"])
            self.assertIsNotNone(sim_entry["primary_rdc_summary"])
            self.assertIsNotNone(sim_entry["primary_itf_summary"])

            npz_path = Path(sim_entry["pairs_npz"])
            self.assertTrue(npz_path.is_file())
            with np.load(npz_path) as data:
                self.assertIn("s_target", data)
                self.assertIn("x_target_hu", data)
                self.assertIn("s_reproj", data)
                self.assertIn("s_reproj_dc", data)

            png_p = Path(sim_entry["visual_qa_png"])
            self.assertTrue(png_p.is_file())
            self.assertGreater(_read_png_dimensions(png_p)[0], 600)

            phan_png = Path(s4_rep["synthetic_phantom_forward_check"]["visual_qa_png"])
            self.assertTrue(phan_png.is_file())
            self.assertGreater(_read_png_dimensions(phan_png)[0], 600)


if __name__ == "__main__":
    unittest.main()
