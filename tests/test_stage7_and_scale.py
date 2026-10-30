"""Unit and integration tests for v1.0 Workstation-Scale Pipeline & Stage 7 Ablation Study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from highpitchct.ablation import ABLATION_METHOD_ORDER, run_stage7_ablation_study
from highpitchct.dual_domain_dl import (
    _make_dual_domain_modules,
    build_hybrid_phantom_dataset,
    build_image_input_tensor,
    create_dual_domain_model,
    load_dual_domain_checkpoint,
    save_dual_domain_checkpoint,
)
from highpitchct.forward_project import forward_project_fanbeam_2d
from highpitchct.phantom import (
    build_thorax_phantom,
    make_synthetic_helical_geometry,
    project_phantom_helical,
    write_synthetic_ctpd_series,
)
from highpitchct.rebinning import rebin_curved_to_flat, rebin_helical_ssrb
from highpitchct.recon_fbp import (
    compute_batch_reconstruction_metrics_torch,
    compute_reconstruction_metrics,
    reconstruct_fanbeam_fbp,
)
from highpitchct.scale_pipeline import (
    ShardedDualDomainDataset,
    build_cohort_shards_resumable,
    build_patient_full_volume_shard,
    build_synthetic_cohort_shards,
    compute_patient_full_slice_grid,
    plan_three_way_patient_split,
    train_sharded_dual_domain,
    verify_three_way_zero_leakage,
)


class TestStage7AndWorkstationScale(unittest.TestCase):
    def test_three_way_patient_split_35_patients_zero_leakage(self) -> None:
        cases = (
            [f"case_C{i:03d}" for i in range(1, 21)]
            + [f"case_L{i:03d}" for i in range(1, 11)]
            + [f"case_N{i:03d}" for i in range(1, 11)]
        )
        split = plan_three_way_patient_split(cases, n_train=25, n_val=5, n_test=5, seed=42)
        self.assertEqual(len(split["train_case_ids"]), 25)
        self.assertEqual(len(split["val_case_ids"]), 5)
        self.assertEqual(len(split["test_case_ids"]), 5)

        audit = verify_three_way_zero_leakage(
            split["train_case_ids"],
            split["val_case_ids"],
            split["test_case_ids"],
            train_sample_count=35700,
            val_sample_count=7140,
            test_sample_count=7140,
        )
        self.assertTrue(audit["overall_zero_leakage_pass"])
        self.assertEqual(audit["total_patient_overlap_count"], 0)
        self.assertEqual(audit["total_sample_count"], 49980)

        bad_audit = verify_three_way_zero_leakage(
            ["case_C001", "case_C002"], ["case_C002"], ["case_C003"]
        )
        self.assertFalse(bad_audit["overall_zero_leakage_pass"])
        self.assertEqual(bad_audit["train_val_overlap"], ["case_C002"])

    def test_full_patient_slice_grid_extracts_entire_scan_span(self) -> None:
        geom = make_synthetic_helical_geometry(
            n_views=4608,
            n_rows=64,
            n_cols=256,
            angular_steps_per_rotation=576,
            normalized_pitch=0.95,
            dv_mm=1.1,
        )
        z_req, dz = compute_patient_full_slice_grid(geom, target_slices_per_patient=238)
        self.assertEqual(z_req.size, 238)
        self.assertGreater(dz, 0.2)
        # Verify strictly uniform spacing across all 238 slices
        diffs = np.diff(z_req)
        self.assertLess(float(np.max(np.abs(diffs - dz))), 1e-8)

    def test_activated_4channel_imgnet_and_checkpoint_roundtrip(self) -> None:
        h, w = 48, 48
        x_dc = torch.linspace(-800.0, 300.0, h * w, dtype=torch.float32).reshape(2, h // 2, w)
        x_dc = torch.cat([x_dc, x_dc], dim=1)
        x_pre = x_dc - 45.0
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij"
        )
        fov = (xx * xx + yy * yy) <= 0.85

        in3 = build_image_input_tensor(x_dc, x_pre, fov, 2.0, in_channels=3)
        self.assertEqual(tuple(in3.shape), (2, 3, h, w))

        in4 = build_image_input_tensor(
            x_dc, x_pre, fov, torch.tensor([1.5, 2.5]), in_channels=4
        )
        self.assertEqual(tuple(in4.shape), (2, 4, h, w))
        # Channel 3 inside FOV must equal clamp((x_dc - x_pre) / 200, -3, 3) = 45 / 200 = 0.225
        self.assertAlmostEqual(float(in4[0, 3, h // 2, w // 2].item()), 0.225, places=5)
        self.assertEqual(float(in4[0, 3, 0, 0].item()), 0.0)

        model, dev = create_dual_domain_model(
            sino_base_channels=16, img_base_channels=16, img_in_channels=4, backend="torch"
        )
        out = model.img_net(
            x_dc,
            x_pre,
            fov,
            torch.tensor([1.5, 2.5]),
            missing_fraction=torch.tensor([0.05, 0.42]),
        )
        self.assertEqual(tuple(out.shape), (2, h, w))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "m4ch.pt"
            save_dual_domain_checkpoint(
                ckpt,
                model,
                summary={"test": True},
                sino_base_channels=16,
                img_base_channels=16,
            )
            loaded, _, payload = load_dual_domain_checkpoint(ckpt, backend="torch")
            self.assertEqual(payload["config"]["img_in_channels"], 4)
            self.assertEqual(loaded.img_net.in_channels, 4)

    def test_batched_fbp_bridge_and_batched_metrics_parity(self) -> None:
        ph = build_thorax_phantom(axial_scale=0.65)
        samples = build_hybrid_phantom_dataset(
            phantom=ph,
            z_slices_mm=[-12.0, -6.0],
            native_pitch=0.85,
            pitches=(1.5, 2.0),
            scenarios=("rdc",),
            image_size=64,
            fov_mm=300.0,
            backend="numpy",
        )
        self.assertGreaterEqual(len(samples), 3)
        _, _, DifferentiableParkerFBPBridge, _, _ = _make_dual_domain_modules()
        bridge = DifferentiableParkerFBPBridge(filter_name="ram_lak", lateral_pad_cols=32)

        s_stack = torch.tensor(
            np.stack([s.s_reproj_dc for s in samples], axis=0),
            dtype=torch.float32,
            requires_grad=True,
        )
        b_stack = torch.tensor(
            np.stack([s.beta_rad for s in samples], axis=0), dtype=torch.float64
        )
        s0 = samples[0]
        hu_batch, _, fov_b = bridge.forward_batch(
            s_stack,
            beta_batch=b_stack,
            u_coords_mm=s0.u_coords_mm,
            du_mm=s0.du_mm,
            rho0_mm=s0.rho0_mm,
            d0_mm=s0.d0_mm,
            image_size=64,
            fov_mm=s0.fov_mm,
            water_mu_mm_inv=s0.water_mu_mm_inv,
            angular_mode=s0.angular_mode,
        )
        loss = torch.mean(hu_batch[:, fov_b])
        loss.backward()
        self.assertIsNotNone(s_stack.grad)
        self.assertGreater(float(torch.max(torch.abs(s_stack.grad)).item()), 0.0)

        for i, smp in enumerate(samples):
            hu_single, _, _ = bridge.forward_single(
                torch.tensor(smp.s_reproj_dc, dtype=torch.float32),
                beta_1d=smp.beta_rad,
                u_coords_mm=smp.u_coords_mm,
                du_mm=smp.du_mm,
                rho0_mm=smp.rho0_mm,
                d0_mm=smp.d0_mm,
                image_size=64,
                fov_mm=smp.fov_mm,
                water_mu_mm_inv=smp.water_mu_mm_inv,
                angular_mode=smp.angular_mode,
            )
            max_diff = float(torch.max(torch.abs(hu_batch[i].detach() - hu_single)).item())
            self.assertLess(max_diff, 0.05)

        # Verify compute_batch_reconstruction_metrics_torch exact parity with NumPy reference
        pred_np = np.stack([s.x_prelim_hu for s in samples], axis=0)
        tgt_np = np.stack([s.x_target_hu for s in samples], axis=0)
        m_ref = compute_reconstruction_metrics(pred_np, tgt_np, fov_mask=s0.fov_mask)
        m_torch = compute_batch_reconstruction_metrics_torch(
            torch.from_numpy(pred_np),
            torch.from_numpy(tgt_np),
            fov_mask=torch.from_numpy(s0.fov_mask),
        )
        self.assertAlmostEqual(m_torch["rmse_hu"], m_ref["rmse_hu"], places=3)
        self.assertAlmostEqual(m_torch["mae_hu"], m_ref["mae_hu"], places=3)
        self.assertAlmostEqual(m_torch["psnr_db"], m_ref["psnr_db"], places=3)
        self.assertAlmostEqual(m_torch["ssim"], m_ref["ssim"], places=4)

        # Also check multi-slice batched FBP & 2D fan-beam projection torch backend parity
        geom = make_synthetic_helical_geometry(
            n_views=192, n_rows=16, n_cols=64, angular_steps_per_rotation=96, normalized_pitch=0.85
        )
        proj = np.ones((192, 16, 64), dtype=np.float32) * 0.5
        f_np, m_np = rebin_curved_to_flat(proj, geom, backend="numpy")
        f_pt, m_pt = rebin_curved_to_flat(proj, geom, backend="torch")
        self.assertTrue(np.array_equal(m_np, m_pt))
        self.assertLess(float(np.max(np.abs(f_np - f_pt))), 1e-5)

        ssrb_2 = rebin_helical_ssrb(f_np, m_np, geom, [-2.0, 2.0], workers=2)
        fbp_np = reconstruct_fanbeam_fbp(ssrb_2, image_size=48, fov_mm=240.0, backend="numpy")
        fbp_pt = reconstruct_fanbeam_fbp(ssrb_2, image_size=48, fov_mm=240.0, backend="torch")
        self.assertLess(float(np.max(np.abs(fbp_np.hu_slices - fbp_pt.hu_slices))), 0.05)

        fp_np = forward_project_fanbeam_2d(
            np.maximum(fbp_np.mu_slices, 0.0),
            ssrb_2.beta_rad,
            ssrb_2.u_coords_mm,
            rho0_mm=ssrb_2.rho0_mm,
            d0_mm=ssrb_2.d0_mm,
            fov_mm=240.0,
            backend="numpy",
        )
        fp_pt = forward_project_fanbeam_2d(
            np.maximum(fbp_np.mu_slices, 0.0),
            ssrb_2.beta_rad,
            ssrb_2.u_coords_mm,
            rho0_mm=ssrb_2.rho0_mm,
            d0_mm=ssrb_2.d0_mm,
            fov_mm=240.0,
            backend="torch",
        )
        self.assertLess(float(np.max(np.abs(fp_np - fp_pt))), 2e-4)

    def test_sharded_val_training_exact_resume_and_stage7_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards_dir = root / "shards"
            ckpt_dir = root / "ckpts"
            ab_dir = root / "ablation"

            cohort = build_synthetic_cohort_shards(
                shards_dir,
                n_train_cases=2,
                n_val_cases=1,
                n_test_cases=1,
                slices_per_case=1,
                image_size=32,
                pitches=(1.5, 2.0),
                scenarios=("itf", "rdc"),
                backend="numpy",
            )
            self.assertTrue(cohort["zero_leakage_audit"]["overall_zero_leakage_pass"])

            train_ds = ShardedDualDomainDataset(cohort["train_shard_dirs"], seed=42)
            val_ds = ShardedDualDomainDataset(cohort["val_shard_dirs"], seed=42)
            test_ds = ShardedDualDomainDataset(cohort["test_shard_dirs"], seed=42)
            self.assertEqual(len(train_ds), 8)
            self.assertEqual(len(val_ds), 4)
            self.assertEqual(len(test_ds), 4)

            # Part 1: Run 2 out of 3 epochs (`stop_after_epoch=2`) to simulate interruption
            res_part1 = train_sharded_dual_domain(
                train_ds,
                val_ds,
                checkpoint_dir=ckpt_dir,
                test_dataset=test_ds,
                epochs=3,
                stage_epochs=(1, 1, 1),
                batch_size=4,
                lr=1.5e-3,
                sino_base_channels=12,
                img_base_channels=12,
                img_in_channels=4,
                backend="torch",
                num_workers=0,
                resume=True,
                stop_after_epoch=2,
                seed=42,
                show_progress=False,
            )
            self.assertEqual(res_part1["epochs_completed"], 2)
            self.assertEqual(len(res_part1["history"]), 2)
            self.assertTrue(Path(res_part1["latest_resume_state_pt"]).is_file())

            # Part 2: Resume without `stop_after_epoch` — must resume at Epoch 3 and finish 3/3!
            res_full = train_sharded_dual_domain(
                train_ds,
                val_ds,
                checkpoint_dir=ckpt_dir,
                test_dataset=test_ds,
                epochs=3,
                stage_epochs=(1, 1, 1),
                batch_size=4,
                lr=1.5e-3,
                sino_base_channels=12,
                img_base_channels=12,
                img_in_channels=4,
                backend="torch",
                num_workers=0,
                resume=True,
                seed=42,
                show_progress=False,
            )
            self.assertEqual(res_full["epochs_completed"], 3)
            self.assertEqual([h["epoch"] for h in res_full["history"]], [1, 2, 3])
            for h in res_full["history"]:
                self.assertIn("rmse_hu", h["val"]["final_dual_domain"])
                self.assertIn("psnr_db", h["val"]["final_dual_domain"])
                self.assertIn("ssim", h["val"]["final_dual_domain"])
            self.assertEqual(res_full["unseen_test_metrics"]["measured_rays_max_abs_dev"], 0.0)

            # Part 3: Stage 7 6-Method Quantitative Ablation Study (`M0..M5`)
            ab_report = run_stage7_ablation_study(
                test_ds,
                dual_checkpoint_path=res_full["best_checkpoint_pt"],
                train_dataset=train_ds,
                val_dataset=val_ds,
                output_dir=ab_dir,
                m3_epochs=1,
                batch_size=4,
                backend="torch",
                num_workers=0,
                resume=True,
                seed=42,
                show_progress=False,
            )
            self.assertEqual(ab_report["measured_rays_max_abs_dev"], 0.0)
            self.assertTrue(ab_report["m3_standalone_trained"])
            for m_key, _ in ABLATION_METHOD_ORDER:
                self.assertIn(m_key, ab_report["overall_ablation_matrix"])

            csv_p = Path(ab_report["ablation_csv"])
            json_p = Path(ab_report["report_json"])
            grid_p = Path(ab_report["visual_qa_grid_png"])
            curves_p = Path(ab_report["pitch_curves_png"])
            self.assertTrue(csv_p.is_file())
            self.assertTrue(json_p.is_file())
            self.assertTrue(grid_p.is_file() and grid_p.stat().st_size > 500)
            self.assertTrue(curves_p.is_file() and curves_p.stat().st_size > 500)

            with csv_p.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertGreaterEqual(len(rows), 6)
            saved_json = json.loads(json_p.read_text(encoding="utf-8"))
            self.assertEqual(saved_json["test_sample_count"], 4)

    def test_build_patient_full_volume_shard_from_dicom_with_oversampling_guard(self) -> None:
        """Verify real DICOM shard construction: zero crash at line 737 and physical slice clamping."""
        geom = make_synthetic_helical_geometry(
            n_views=256,
            n_rows=16,
            n_cols=96,
            angular_steps_per_rotation=128,
            normalized_pitch=0.85,
            dv_mm=1.5,
        )
        phantom = build_thorax_phantom(axial_scale=0.4)
        proj = project_phantom_helical(phantom, geom, batch_views=64)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj_dir = (
                root / "raw" / "tcia" / "case_TEST01" / "projection_full" / geom.series_instance_uid
            )
            shard_dir = root / "shards" / "train" / "case_TEST01"

            write_synthetic_ctpd_series(proj_dir, geom, proj, patient_id="TEST01")

            info = build_patient_full_volume_shard(
                "case_TEST01",
                proj_dir,
                shard_dir,
                split_name="train",
                target_slices_per_patient=2500,
                pitches=[1.5],
                scenarios=["rdc"],
                image_size=32,
                backend="numpy",
                workers=2,
            )

            self.assertFalse(info["resumed_from_disk"])
            # Verify slice count was clamped to scanner physical thin-slice capacity (dz >= 0.45 mm)
            self.assertLess(info["num_slices"], 100)
            self.assertGreaterEqual(info["slice_spacing_mm"], 0.45)
            self.assertTrue(Path(info["shard_dir"]).is_dir())
            self.assertTrue((shard_dir / "shard_complete.json").is_file())
            self.assertTrue((shard_dir / "s_target.npy").is_file())
            self.assertTrue((shard_dir / "x_target_hu.npy").is_file())

            # Verify ShardedDualDomainDataset loads this shard without issues
            ds = ShardedDualDomainDataset([shard_dir], seed=42)
            self.assertEqual(len(ds), info["num_samples"])
            sample = ds[0]
            self.assertEqual(sample["x_target_hu"].shape, (32, 32))
            self.assertEqual(sample["s_hp_raw"].shape, sample["s_target"].shape)


if __name__ == "__main__":
    unittest.main()
