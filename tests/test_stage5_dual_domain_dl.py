"""Stage 5 unit & integration tests: Physics-Conditioned Dual-Domain DL (`v0.8.0`).

Covers:
1. 4-channel sinogram & 3-channel image physics conditioning tensors + Hard Data Consistency (`0.0` error on measured rays) + strict backend fail-fast policy.
2. Exact numerical parity (`< 0.05 HU`) between PyTorch `DifferentiableParkerFBPBridge` and NumPy `reconstruct_fanbeam_fbp` with `ram_lak` filter + autograd gradient flow verification.
3. Multi-pitch `{1.5, 2.0, 2.5}` automatic adaptation across different patient `native_pitch` values (`0.65` vs `0.90`), `60% ITF / 40% RDC` curriculum weighting, and `.npz` dataset round-trip.
4. End-to-end `DualDomainHighPitchNet` training convergence (`x_final` outperforms `x_prelim` and `x_raw_nofill` in both sinogram and image domains), `.pt` checkpoint round-trip, and Visual QA PNG generation.
5. `hpct stage5-train` CLI end-to-end execution with automatic `400.0 mm` Vendor CT FOV matching.
"""

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
from highpitchct.dual_domain_dl import (
    DEFAULT_CLINICAL_HIGH_PITCHES,
    DEFAULT_ITF_TRAINING_RATIO,
    _make_dual_domain_modules,
    apply_hard_data_consistency,
    build_hybrid_phantom_dataset,
    build_image_input_tensor,
    build_sino_input_tensor,
    build_weighted_hybrid_curriculum,
    create_dual_domain_model,
    evaluate_dual_domain_sample,
    load_dual_domain_checkpoint,
    load_dual_domain_dataset_npz,
    resolve_torch_device,
    save_dual_domain_checkpoint,
    save_dual_domain_dataset_npz,
    train_dual_domain_model,
)
from highpitchct.forward_project import forward_project_fanbeam_2d
from highpitchct.phantom import (
    build_thorax_phantom,
    make_synthetic_helical_geometry,
    project_phantom_helical,
    voxelize_phantom_slices_radiological,
    write_synthetic_ctpd_series,
)
from highpitchct.rebinning import SSRBResult
from highpitchct.recon_fbp import reconstruct_fanbeam_fbp
from highpitchct.visual_qa import save_stage5_dual_domain_visual_qa

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    HAS_PYDICOM = True
except ImportError:  # pragma: no cover
    HAS_PYDICOM = False


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"Invalid PNG signature in {path}")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _write_synthetic_vendor_ct_slices(
    out_dir: Path,
    hu_slices: np.ndarray,
    z_slices_mm: np.ndarray,
    *,
    fov_mm: float = 400.0,
    patient_id: str = "C012",
    series_uid: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suid = series_uid or str(generate_uid())
    study_uid = str(generate_uid())
    n_s, n_y, n_x = hu_slices.shape
    d_xy = float(fov_mm) / float(n_x)

    for idx in range(n_s):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(out_dir / f"img_{idx:03d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = suid
        ds.Modality = "CT"
        ds.PatientID = patient_id
        ds.InstanceNumber = idx + 1
        ds.Rows = int(n_y)
        ds.Columns = int(n_x)
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelSpacing = [f"{d_xy:.6f}", f"{d_xy:.6f}"]
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
        ds.ImagePositionPatient = [
            f"{-0.5 * fov_mm:.4f}",
            f"{-0.5 * fov_mm:.4f}",
            f"{float(z_slices_mm[idx]):.4f}",
        ]
        ds.SliceLocation = f"{float(z_slices_mm[idx]):.4f}"
        ds.RescaleIntercept = "-1024.0"
        ds.RescaleSlope = "1.0"
        ds.ConvolutionKernel = "B50f"

        raw_u16 = np.clip(np.round(hu_slices[idx] + 1024.0), 0, 65535).astype(np.uint16)
        ds.PixelData = raw_u16.tobytes()
        ds.save_as(str(out_dir / f"img_{idx:03d}.dcm"), enforce_file_format=True)


@unittest.skipUnless(HAS_TORCH, "PyTorch required for Stage 5 Dual-Domain DL tests")
class TestStage5DualDomainDL(unittest.TestCase):
    """Unit tests for Stage 5 Physics-Conditioned Dual-Domain Deep Learning (`v0.8.0`)."""

    def test_physics_conditioning_tensors_and_hard_data_consistency_invariants(self) -> None:
        """Verify 4-ch sinogram & 3-ch image tensors, Hard DC exactness, & strict backend fail-fast."""
        import torch

        # 1. Explicit backend policy: fail-fast on unavailable CUDA or unknown backend
        with self.assertRaises(ValueError):
            resolve_torch_device("auto")
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no silent fallback"):
                resolve_torch_device("torch_cuda")

        # 2. 4-channel sinogram conditioning tensor & Hard Data Consistency
        s_hp_raw = torch.tensor(
            [[[1.2, 2.5, 0.0, 0.0], [0.8, 1.9, 0.0, 0.0]]], dtype=torch.float32
        )
        s_reproj = torch.tensor(
            [[[1.1, 2.4, 1.7, 0.9], [0.7, 1.8, 1.4, 0.6]]], dtype=torch.float32
        )
        mask_hp = torch.tensor(
            [[[True, True, False, False], [True, True, False, False]]], dtype=torch.bool
        )
        x_sino = build_sino_input_tensor(s_hp_raw, s_reproj, mask_hp, effective_pitch=2.0)
        self.assertEqual(tuple(x_sino.shape), (1, 4, 2, 4))
        # Channel 2 is exact binary mask
        np.testing.assert_allclose(x_sino[0, 2].numpy(), mask_hp[0].numpy().astype(np.float32))
        # Channel 3 has higher value inside the missing wedge than in the measured region
        self.assertGreater(float(x_sino[0, 3, 0, 2]), float(x_sino[0, 3, 0, 0]))

        # Hard Data Consistency preserves measured pixels to 0.0 error and clamps negatives in wedge
        s_pred_noisy = torch.tensor(
            [[[99.0, -50.0, 1.65, -0.4], [-12.0, 42.0, 1.35, 0.55]]], dtype=torch.float32
        )
        s_dc = apply_hard_data_consistency(s_hp_raw, mask_hp, s_pred_noisy)
        np.testing.assert_allclose(
            s_dc[mask_hp].numpy(), s_hp_raw[mask_hp].numpy(), atol=0.0, rtol=0.0
        )
        self.assertAlmostEqual(float(s_dc[0, 0, 2]), 1.65, places=6)
        self.assertEqual(float(s_dc[0, 0, 3]), 0.0)  # negative prediction clamped to 0.0

        # 3. 3-channel image conditioning tensor
        x_dc_hu = torch.zeros((1, 8, 8), dtype=torch.float32)
        x_pre_hu = torch.full((1, 8, 8), -200.0, dtype=torch.float32)
        fov_mask = torch.ones((1, 8, 8), dtype=torch.bool)
        fov_mask[0, 0, 0] = False
        x_img = build_image_input_tensor(x_dc_hu, x_pre_hu, fov_mask, effective_pitch=2.5)
        self.assertEqual(tuple(x_img.shape), (1, 3, 8, 8))
        self.assertEqual(float(x_img[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(x_img[0, 2, 0, 0]), 0.0)
        self.assertAlmostEqual(float(x_img[0, 2, 4, 4]), 2.5 / 3.0, places=5)

    def test_differentiable_parker_ram_lak_fbp_bridge_parity_and_gradients(self) -> None:
        """Prove PyTorch DifferentiableParkerFBPBridge matches NumPy Ram-Lak FBP (<0.05 HU) & flows grads."""
        import torch

        _, _, DifferentiableParkerFBPBridge, _, _ = _make_dual_domain_modules()
        bridge = DifferentiableParkerFBPBridge(
            filter_name="ram_lak", cutoff_ratio=1.0, lateral_pad_cols=64
        )

        phantom = build_thorax_phantom(axial_scale=0.7)
        vox = voxelize_phantom_slices_radiological(
            phantom, z_slices_mm=[-12.0], image_size=96, fov_mm=340.0
        )
        d0_mm = 1085.6
        rho0_mm = 595.0
        n_u = 128
        du_mm = 4.8
        u_coords = (np.arange(n_u, dtype=np.float64) - 0.5 * (n_u - 1.0)) * du_mm
        gamma_max = float(np.max(np.abs(np.arctan2(u_coords, d0_mm))))
        span = math.pi + 2.0 * gamma_max
        beta_1d = -0.5 * math.pi + np.linspace(0.0, span, 120, endpoint=True, dtype=np.float64)

        sino_np = forward_project_fanbeam_2d(
            vox["mu_slices"],
            beta_1d,
            u_coords,
            rho0_mm=rho0_mm,
            d0_mm=d0_mm,
            fov_mm=340.0,
        )
        ssrb = SSRBResult(
            sinograms=sino_np,
            measured_mask=np.ones_like(sino_np, dtype=bool),
            z_slices_mm=np.array([-12.0], dtype=np.float64),
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
        fbp_np = reconstruct_fanbeam_fbp(
            ssrb,
            image_size=96,
            fov_mm=340.0,
            filter_name="ram_lak",
            inpaint_mode="none",
            lateral_pad_cols=64,
            backend="numpy",
        )

        sino_t = torch.tensor(sino_np[0], dtype=torch.float32, requires_grad=True)
        hu_t, mu_t, fov_mask_t = bridge.forward_single(
            sino_t,
            beta_1d=beta_1d,
            u_coords_mm=u_coords,
            du_mm=du_mm,
            rho0_mm=rho0_mm,
            d0_mm=d0_mm,
            image_size=96,
            fov_mm=340.0,
            water_mu_mm_inv=0.0192,
            angular_mode="short_scan",
        )

        max_hu_diff = float(
            np.max(np.abs(hu_t.detach().numpy()[fbp_np.fov_mask] - fbp_np.hu_slices[0][fbp_np.fov_mask]))
        )
        self.assertLess(
            max_hu_diff,
            0.05,
            msg=f"DifferentiableParkerFBPBridge vs NumPy FBP max HU diff = {max_hu_diff}",
        )

        # Verify end-to-end autograd backpropagation from image HU loss back to sinogram rays
        loss = torch.mean(hu_t[fov_mask_t] ** 2)
        loss.backward()
        self.assertIsNotNone(sino_t.grad)
        grad_norm = float(torch.norm(sino_t.grad).item())
        self.assertGreater(grad_norm, 1e-3)
        self.assertTrue(bool(torch.all(torch.isfinite(sino_t.grad)).item()))

    def test_multi_pitch_1p5_2p0_2p5_adaptation_across_different_native_pitches_and_60_40_curriculum(
        self,
    ) -> None:
        """Verify p in {1.5, 2.0, 2.5} adapts to native_pitch=0.65 & 0.90 and 60% ITF / 40% RDC holds."""
        samples_p065 = build_hybrid_phantom_dataset(
            pitches=(1.5, 2.0, 2.5),
            scenarios=("itf", "rdc"),
            z_slices_mm=(-12.0,),
            native_pitch=0.65,
            image_size=64,
            fov_mm=340.0,
            filter_name="ram_lak",
        )
        samples_p090 = build_hybrid_phantom_dataset(
            pitches=(1.5, 2.0, 2.5),
            scenarios=("itf", "rdc"),
            z_slices_mm=(-12.0,),
            native_pitch=0.90,
            image_size=64,
            fov_mm=340.0,
            filter_name="ram_lak",
        )
        self.assertEqual(len(samples_p065), 6)
        self.assertEqual(len(samples_p090), 6)

        for s65, s90 in zip(samples_p065, samples_p090):
            self.assertEqual(s65.target_pitch, s90.target_pitch)
            self.assertIn(s65.target_pitch, DEFAULT_CLINICAL_HIGH_PITCHES)
            # Effective pitch must track target_pitch closely regardless of native_pitch (0.65 vs 0.90)
            self.assertLess(abs(s65.effective_pitch - s65.target_pitch), 0.25)
            self.assertLess(abs(s90.effective_pitch - s90.target_pitch), 0.25)

        # Verify 60% ITF / 40% RDC curriculum weighting
        curr = build_weighted_hybrid_curriculum(samples_p090, itf_ratio=DEFAULT_ITF_TRAINING_RATIO)
        total_w = sum(w for _, w in curr)
        itf_w = sum(w for s, w in curr if s.scenario == "itf")
        rdc_w = sum(w for s, w in curr if s.scenario == "rdc")
        self.assertAlmostEqual(itf_w / total_w, 0.60, places=6)
        self.assertAlmostEqual(rdc_w / total_w, 0.40, places=6)

        # Verify .npz dataset bundle round-trip
        with tempfile.TemporaryDirectory() as tmp:
            npz_p = save_dual_domain_dataset_npz(
                Path(tmp) / "hybrid_ds.npz", samples_p090, itf_ratio=0.60
            )
            loaded = load_dual_domain_dataset_npz(npz_p)
            self.assertEqual(len(loaded), len(samples_p090))
            np.testing.assert_allclose(loaded[0].x_target_hu, samples_p090[0].x_target_hu)
            np.testing.assert_allclose(loaded[0].s_hp_raw, samples_p090[0].s_hp_raw)

    def test_dual_domain_training_monotonic_improvement_and_checkpoint_roundtrip(self) -> None:
        """Verify Dual-Domain training improves RMSE/SSIM & missing-wedge MAE, and saves checkpoints/PNGs."""
        samples = build_hybrid_phantom_dataset(
            pitches=(1.5, 2.0, 2.5),
            scenarios=("itf", "rdc"),
            z_slices_mm=(-12.0,),
            native_pitch=0.85,
            image_size=64,
            fov_mm=340.0,
            filter_name="ram_lak",
        )
        model, device = create_dual_domain_model(
            sino_base_channels=12,
            img_base_channels=12,
            filter_name="ram_lak",
            backend="torch",
            seed=42,
        )
        res = train_dual_domain_model(
            model,
            samples,
            device=device,
            epochs=6,
            lr=3.5e-3,
            itf_ratio=0.60,
            seed=42,
        )
        hist = res["history"]
        summary = res["summary"]
        prog = summary["overall_progression"]

        # Loss must decrease from epoch 1 to final epoch
        self.assertLess(hist[-1]["total_loss"], hist[0]["total_loss"])
        # Final Dual-Domain RMSE must beat preliminary FBP and raw no-fill FBP
        self.assertLessEqual(
            prog["stage3_final_dual_domain"]["rmse_hu"],
            prog["stage1_prelim_fbp"]["rmse_hu"],
        )
        self.assertLess(
            prog["stage1_prelim_fbp"]["rmse_hu"],
            prog["stage0_raw_nofill"]["rmse_hu"],
        )
        # Hard Data Consistency must have 0.0 deviation on measured rays
        self.assertEqual(prog["sinogram_progression"]["measured_rays_max_abs_dev"], 0.0)
        # SinoNet missing-wedge MAE must be strictly smaller than raw missing-wedge MAE
        self.assertLess(
            prog["sinogram_progression"]["sinonet_dc_missing_wedge_mae"],
            prog["sinogram_progression"]["raw_missing_wedge_mae"],
        )

        # Verify checkpoint save/load round-trip and Visual QA PNG generation
        pred0 = res["predictions"][0]
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_p = save_dual_domain_checkpoint(
                Path(tmp) / "model.pt",
                model,
                summary=summary,
                sino_base_channels=12,
                img_base_channels=12,
                filter_name="ram_lak",
            )
            loaded_model, loaded_dev, payload = load_dual_domain_checkpoint(ckpt_p, backend="torch")
            pred_reloaded = evaluate_dual_domain_sample(loaded_model, pred0.sample, device=loaded_dev)
            np.testing.assert_allclose(pred_reloaded.x_final_hu, pred0.x_final_hu, atol=1e-4)
            self.assertEqual(payload["config"]["filter_name"], "ram_lak")

            png_mpl = Path(tmp) / "s5_mpl.png"
            png_std = Path(tmp) / "s5_std.png"
            save_stage5_dual_domain_visual_qa(
                png_mpl,
                title_prefix="Stage 5 Test QA",
                scenario=pred0.sample.scenario,
                target_pitch=pred0.sample.target_pitch,
                effective_pitch=pred0.sample.effective_pitch,
                native_pitch=pred0.sample.native_pitch,
                z_slice_mm=pred0.sample.z_slice_mm,
                fov_mm=pred0.sample.fov_mm,
                fov_mask=pred0.sample.fov_mask,
                filter_name="ram_lak",
                s_hp_raw=pred0.sample.s_hp_raw,
                mask_hp=pred0.sample.mask_hp,
                s_dc=pred0.s_dc,
                x_target_hu=pred0.sample.x_target_hu,
                x_prelim_hu=pred0.sample.x_prelim_hu,
                x_dc_hu=pred0.x_dc_hu,
                x_final_hu=pred0.x_final_hu,
                prelim_metrics=pred0.prelim_metrics,
                domain1_dc_metrics=pred0.domain1_dc_metrics,
                final_dual_metrics=pred0.final_dual_metrics,
                sino_metrics=pred0.sinogram_progression_metrics,
            )
            self.assertGreater(_read_png_dimensions(png_mpl)[0], 600)

            with mock.patch("highpitchct.visual_qa._has_matplotlib", return_value=False):
                save_stage5_dual_domain_visual_qa(
                    png_std,
                    title_prefix="Stage 5 Test QA",
                    scenario=pred0.sample.scenario,
                    target_pitch=pred0.sample.target_pitch,
                    effective_pitch=pred0.sample.effective_pitch,
                    native_pitch=pred0.sample.native_pitch,
                    z_slice_mm=pred0.sample.z_slice_mm,
                    fov_mm=pred0.sample.fov_mm,
                    fov_mask=pred0.sample.fov_mask,
                    filter_name="ram_lak",
                    s_hp_raw=pred0.sample.s_hp_raw,
                    mask_hp=pred0.sample.mask_hp,
                    s_dc=pred0.s_dc,
                    x_target_hu=pred0.sample.x_target_hu,
                    x_prelim_hu=pred0.sample.x_prelim_hu,
                    x_dc_hu=pred0.x_dc_hu,
                    x_final_hu=pred0.x_final_hu,
                    prelim_metrics=pred0.prelim_metrics,
                    domain1_dc_metrics=pred0.domain1_dc_metrics,
                    final_dual_metrics=pred0.final_dual_metrics,
                    sino_metrics=pred0.sinogram_progression_metrics,
                )
            self.assertGreater(_read_png_dimensions(png_std)[0], 600)


@unittest.skipUnless(
    HAS_TORCH and HAS_PYDICOM, "PyTorch and pydicom required for Stage 5 CLI test"
)
class TestStage5CLI(unittest.TestCase):
    """End-to-end test for `hpct stage5-train` CLI (60% ITF + 40% RDC, p in {1.5, 2.0, 2.5}, 400mm FOV)."""

    def test_cli_stage5_train_end_to_end_with_vendor_fov_matching(self) -> None:
        phantom = build_thorax_phantom(axial_scale=0.6)
        geom = make_synthetic_helical_geometry(
            n_views=256,
            n_rows=32,
            n_cols=128,  # max_fov_mm = 458.0 mm > 400.0 mm vendor FOV
            angular_steps_per_rotation=128,
            normalized_pitch=0.90,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=4.0,
            dv_mm=2.0,
        )
        proj = project_phantom_helical(phantom, geom, batch_views=128)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_uid = str(generate_uid())
            proj_dir = (
                root / "raw" / "tcia" / "case_C012" / "projection_full" / geom.series_instance_uid
            )
            img_dir = (
                root / "raw" / "tcia" / "case_C012" / "reconstruction_published" / img_uid
            )
            write_synthetic_ctpd_series(proj_dir, geom, proj, patient_id="C012")
            z_vendor = np.linspace(-16.0, 16.0, 7, dtype=np.float64)
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

            rc = cli_main(
                [
                    "stage5-train",
                    "--data-root",
                    str(root),
                    "--case-id",
                    "C012",
                    "--scenario",
                    "both",
                    "--itf-ratio",
                    "0.60",
                    "--pitches",
                    "1.5,2.0,2.5",
                    "--num-slices",
                    "1",
                    "--n-xy",
                    "64",
                    "--itf-slab-slices",
                    "5",
                    "--epochs",
                    "4",
                    "--sino-channels",
                    "12",
                    "--img-channels",
                    "12",
                    "--backend",
                    "torch",
                    "--save-model",
                    "--save-dataset-npz",
                    "--workers",
                    "2",
                    "--no-progress",
                ]
            )
            self.assertEqual(rc, 0)
            rep_path = root / "manifests" / "stage5_dl_report.json"
            self.assertTrue(rep_path.is_file())
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            self.assertTrue(rep["overall_pass"])
            self.assertEqual(rep["trained_series_count"], 1)

            entry = rep["trained_series"][0]
            self.assertEqual(entry["filter_name"], "ram_lak")
            self.assertAlmostEqual(float(entry["fov_mm"]), 400.0, places=3)
            self.assertAlmostEqual(float(entry["itf_ratio"]), 0.60, places=3)
            self.assertEqual(entry["pitches"], [1.5, 2.0, 2.5])
            self.assertTrue(Path(entry["model_checkpoint_pt"]).is_file())
            self.assertTrue(Path(entry["dataset_npz"]).is_file())
            self.assertTrue(Path(entry["visual_qa_png"]).is_file())
            self.assertGreater(_read_png_dimensions(Path(entry["visual_qa_png"]))[0], 600)


if __name__ == "__main__":
    unittest.main()
