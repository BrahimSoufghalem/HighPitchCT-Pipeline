"""Stage 6 unit and integration tests for Zero-Leakage Generalization & Out-of-Sample Evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from highpitchct.cli import main
from highpitchct.generalization import (
    build_multi_phantom_generalization_split,
    build_variant_thorax_phantom,
    plan_axial_guardband_slices,
    run_generalization_experiment,
    verify_zero_leakage_split,
)
from highpitchct.phantom import (
    make_synthetic_helical_geometry,
    project_phantom_helical,
    write_synthetic_ctpd_series,
)


class TestStage6Generalization(unittest.TestCase):
    """Verify Stage 6 Zero-Leakage Audit, Axial Guard-Band Planner, and Out-of-Sample Generalization."""

    def test_zero_leakage_audit_rejects_patient_and_axial_leakage(self) -> None:
        """Prove verify_zero_leakage_split fails fast on patient overlap, sample overlap, or guardband violation."""
        split = build_multi_phantom_generalization_split(
            pitches=(1.5, 2.0),
            scenarios=("rdc",),
            image_size=64,
            fov_mm=360.0,
            filter_name="ram_lak",
            backend="torch",
        )
        tr = split["train_samples"]
        te = split["test_samples"]

        # Valid patient_split passes
        audit_ok = verify_zero_leakage_split(tr, te, split_mode="patient_split")
        self.assertTrue(audit_ok.overall_zero_leakage_pass)
        self.assertEqual(audit_ok.patient_id_overlap_count, 0)

        # 1. Sample overlap must fail fast
        with self.assertRaises(ValueError):
            verify_zero_leakage_split(tr, [tr[0]], split_mode="patient_split")

        # 2. Same patient ID in train and test with different z must fail patient_split
        s_same_case = tr[0]
        from dataclasses import replace

        s_leak_case = replace(
            s_same_case,
            sample_id="PHANTOM_TRAIN_A_rdc_p2.00_z+15.0",
            z_slice_mm=15.0,
        )
        with self.assertRaises(ValueError):
            verify_zero_leakage_split(tr, [s_leak_case], split_mode="patient_split")

        # 3. Axial guardband violation (|z_tr - z_te| = 27.0 mm < 45.0 mm) must fail fast
        with self.assertRaises(ValueError):
            verify_zero_leakage_split(
                [s_same_case],
                [s_leak_case],
                split_mode="axial_guardband",
                min_guardband_mm=45.0,
                itf_slab_half_span_mm=12.0,
            )

        # 4. 3D ITF slab overlap violation (min_sep = 30 mm, 2 * slab_half_span = 36 mm -> gap = -6 mm)
        s_far_z = replace(
            s_same_case,
            sample_id="PHANTOM_TRAIN_A_rdc_p2.00_z+18.0",
            z_slice_mm=18.0,
        )
        with self.assertRaises(ValueError):
            verify_zero_leakage_split(
                [s_same_case],
                [s_far_z],
                split_mode="axial_guardband",
                min_guardband_mm=25.0,
                itf_slab_half_span_mm=18.0,
            )

    def test_axial_guardband_slice_planner_and_3d_itf_slab_disjointness(self) -> None:
        """Verify plan_axial_guardband_slices guarantees >= 45 mm guardband and positive 3D ITF slab gap."""
        plan = plan_axial_guardband_slices(
            z_min_raw_mm=81.0,
            z_max_raw_mm=336.0,
            coverage_isocenter_mm=38.4,
            n_train_slices=2,
            n_test_slices=1,
            min_guardband_mm=45.0,
        )
        self.assertEqual(len(plan["train_z_slices_mm"]), 2)
        self.assertEqual(len(plan["test_z_slices_mm"]), 1)
        self.assertGreaterEqual(plan["actual_guardband_mm"], 45.0)
        self.assertGreater(plan["itf_slab_gap_mm"], 0.0)
        max_tr = max(plan["train_z_slices_mm"])
        min_te = min(plan["test_z_slices_mm"])
        self.assertAlmostEqual(min_te - max_tr, plan["actual_guardband_mm"], places=2)

    def test_multi_phantom_out_of_sample_generalization_across_unseen_anatomy_and_native_pitch(
        self,
    ) -> None:
        """Prove DualDomainHighPitchNet trained on 2 phantoms generalizes zero-shot to an unseen 3rd phantom."""
        split = build_multi_phantom_generalization_split(
            pitches=(1.5, 2.0, 2.5),
            scenarios=("itf", "rdc"),
            image_size=64,
            fov_mm=360.0,
            filter_name="ram_lak",
            backend="torch",
        )
        res = run_generalization_experiment(
            split["train_samples"],
            split["test_samples"],
            split_mode="patient_split",
            sino_base_channels=16,
            img_base_channels=16,
            filter_name="ram_lak",
            epochs=6,
            lr=3.0e-3,
            itf_ratio=0.60,
            backend="torch",
            seed=42,
        )
        gen_m = res["generalization_metrics"]
        self.assertTrue(gen_m["zero_leakage_verified"])
        self.assertTrue(gen_m["unseen_test_monotonic_pass"])
        self.assertEqual(gen_m["unseen_test_measured_rays_max_abs_dev"], 0.0)
        self.assertLess(
            gen_m["unseen_test_final_dual_rmse_hu"],
            gen_m["unseen_test_prelim_rmse_hu"],
        )
        self.assertGreaterEqual(
            gen_m["unseen_test_final_dual_ssim"],
            gen_m["unseen_test_prelim_ssim"],
        )

    def test_cli_stage6_generalize_on_multi_patient_synthetic_dicom_series(self) -> None:
        """Verify CLI stage6-generalize runs end-to-end on two distinct patient series (patient_split)."""
        import numpy as np
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian

        with tempfile.TemporaryDirectory(prefix="hpct_stage6_cli_") as tmp:
            root = Path(tmp)
            pdir_train = root / "raw" / "tcia" / "case_C012" / "projection_full" / "1"
            idir_train = root / "raw" / "tcia" / "case_C012" / "reconstruction_published" / "1"
            pdir_test = root / "raw" / "tcia" / "case_C027" / "projection_full" / "1"

            idir_train.mkdir(parents=True, exist_ok=True)
            fm = FileMetaDataset()
            fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            fm.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.10.999.2001"
            fm.TransferSyntaxUID = ExplicitVRLittleEndian
            ct_ds = FileDataset(
                str(idir_train / "slice_0001.dcm"),
                {},
                file_meta=fm,
                preamble=b"\0" * 128,
            )
            ct_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ct_ds.SOPInstanceUID = "1.2.826.0.1.3680043.10.999.2001"
            ct_ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.999.2000"
            ct_ds.Modality = "CT"
            ct_ds.InstanceNumber = 1
            ct_ds.ImagePositionPatient = [-180.0, -180.0, 0.0]
            ct_ds.Rows = 64
            ct_ds.Columns = 64
            ct_ds.PixelSpacing = [5.625, 5.625]
            ct_ds.SliceThickness = 1.5
            ct_ds.BitsAllocated = 16
            ct_ds.BitsStored = 16
            ct_ds.HighBit = 15
            ct_ds.PixelRepresentation = 0
            ct_ds.SamplesPerPixel = 1
            ct_ds.PhotometricInterpretation = "MONOCHROME2"
            ct_ds.RescaleSlope = 1.0
            ct_ds.RescaleIntercept = -1024.0
            ct_ds.PixelData = np.full((64, 64), 1024, dtype=np.uint16).tobytes()
            ct_ds.save_as(str(idir_train / "slice_0001.dcm"), enforce_file_format=True)

            geom_train = make_synthetic_helical_geometry(
                n_views=192,
                n_rows=32,
                n_cols=96,
                angular_steps_per_rotation=96,
                normalized_pitch=0.85,
                detector_shape="CYLINDRICAL",
                ffs_mode="FFSZ",
                rotation_direction=-1,
                quarter_detector_offset_cols=1.125,
                du_mm=6.8,
                dv_mm=1.8,
                z_center_mm=0.0,
            )
            proj_train = project_phantom_helical(
                build_variant_thorax_phantom("train_a"),
                geom_train,
                batch_views=96,
                show_progress=False,
            )
            write_synthetic_ctpd_series(pdir_train, geom_train, proj_train)

            geom_test = make_synthetic_helical_geometry(
                n_views=192,
                n_rows=32,
                n_cols=96,
                angular_steps_per_rotation=96,
                normalized_pitch=0.90,
                detector_shape="CYLINDRICAL",
                ffs_mode="FFSZ",
                rotation_direction=-1,
                quarter_detector_offset_cols=1.125,
                du_mm=6.8,
                dv_mm=1.8,
                z_center_mm=0.0,
            )
            proj_test = project_phantom_helical(
                build_variant_thorax_phantom("test_unseen"),
                geom_test,
                batch_views=96,
                show_progress=False,
            )
            write_synthetic_ctpd_series(pdir_test, geom_test, proj_test)

            rc = main(
                [
                    "stage6-generalize",
                    "--data-root",
                    str(root),
                    "--split-mode",
                    "patient_split",
                    "--train-cases",
                    "C012",
                    "--test-cases",
                    "C027",
                    "--train-slices-per-case",
                    "1",
                    "--test-slices-per-case",
                    "1",
                    "--pitches",
                    "1.5,2.0,2.5",
                    "--itf-ratio",
                    "0.60",
                    "--image-size",
                    "64",
                    "--fov-mm",
                    "360.0",
                    "--sino-channels",
                    "16",
                    "--img-channels",
                    "16",
                    "--epochs",
                    "4",
                    "--backend",
                    "torch",
                    "--save-model",
                    "--no-progress",
                ]
            )
            self.assertEqual(rc, 0)

            rep_path = root / "manifests" / "stage6_generalization_report.json"
            self.assertTrue(rep_path.exists())
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            self.assertTrue(rep["overall_pass"])
            real_gen = rep["real_patient_generalization"]
            self.assertEqual(real_gen["split_mode"], "patient_split")
            self.assertEqual(
                real_gen["zero_leakage_audit"]["train_case_ids"],
                ["case_C012"],
            )
            self.assertEqual(
                real_gen["zero_leakage_audit"]["test_case_ids"],
                ["case_C027"],
            )
            self.assertTrue(Path(real_gen["visual_qa_png"]).exists())
            ckpt_pt = Path(real_gen["model_checkpoint_pt"])
            self.assertTrue(ckpt_pt.exists())
            self.assertIn(
                "unseen_test_domain2_vs_domain1_rmse_delta_hu",
                real_gen["generalization_metrics"],
            )

            # Verify frozen zero-shot evaluation via --load-model without retraining
            rc_eval = main(
                [
                    "stage6-generalize",
                    "--data-root",
                    str(root),
                    "--split-mode",
                    "patient_split",
                    "--train-cases",
                    "C012",
                    "--test-cases",
                    "C027",
                    "--train-slices-per-case",
                    "1",
                    "--test-slices-per-case",
                    "1",
                    "--pitches",
                    "1.5,2.0,2.5",
                    "--itf-ratio",
                    "0.60",
                    "--image-size",
                    "64",
                    "--fov-mm",
                    "360.0",
                    "--load-model",
                    str(ckpt_pt),
                    "--backend",
                    "torch",
                    "--no-progress",
                ]
            )
            self.assertEqual(rc_eval, 0)
            rep_eval = json.loads(rep_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                rep_eval["real_patient_generalization"]["generalization_metrics"][
                    "unseen_test_final_dual_rmse_hu"
                ],
                real_gen["generalization_metrics"]["unseen_test_final_dual_rmse_hu"],
                places=4,
            )


if __name__ == "__main__":
    unittest.main()
