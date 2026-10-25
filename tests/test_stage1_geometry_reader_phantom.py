from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from highpitchct.cli import main as cli_main
from highpitchct.ctpd_reader import (
    hu_to_mu,
    iter_projection_chunks,
    load_projections_memmap,
    mu_to_hu,
    read_ct_image_volume,
    read_single_projection,
    scan_projection_series,
    verify_projection_orientation,
)
from highpitchct.ctpd_tags import CT_IMAGE_SOP_CLASS_UID
from highpitchct.geometry import HelicalGeometry
from highpitchct.index_dicom import index_dicom_tree
from highpitchct.phantom import (
    build_thorax_phantom,
    make_synthetic_helical_geometry,
    project_phantom_helical,
    voxelize_phantom,
    write_synthetic_ctpd_series,
)

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ImplicitVRLittleEndian

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class TestStage1PureNumPyPhysics(unittest.TestCase):
    """Pure NumPy mathematical & geometric proofs (no pydicom required)."""

    def test_hu_mu_exact_conversion_and_roundtrip(self) -> None:
        w = 0.0192
        self.assertAlmostEqual(float(hu_to_mu(-1000.0, w)), 0.0, places=10)
        self.assertAlmostEqual(float(hu_to_mu(0.0, w)), w, places=10)
        self.assertAlmostEqual(float(hu_to_mu(1000.0, w)), 2.0 * w, places=10)
        self.assertAlmostEqual(float(hu_to_mu(-1024.0, w, clip_non_negative=True)), 0.0, places=10)

        hu_arr = np.array([-1000.0, -750.0, 0.0, 20.0, 55.0, 820.0], dtype=np.float32)
        mu_arr = hu_to_mu(hu_arr, w)
        self.assertEqual(mu_arr.dtype, np.float32)
        hu_back = mu_to_hu(mu_arr, w)
        np.testing.assert_allclose(hu_back, hu_arr, rtol=1e-6, atol=1e-4)

    def test_helical_geometry_basis_centers_and_npz_roundtrip(self) -> None:
        geom = make_synthetic_helical_geometry(
            n_views=192,
            n_rows=64,
            n_cols=736,
            angular_steps_per_rotation=192,
            normalized_pitch=0.90,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=1.2858,
            dv_mm=1.0947,
        )
        # Verify exact Siemens C012 1-based <-> 0-based central element convention
        c1, r1 = geom.detector_central_element_1based
        self.assertAlmostEqual(c1, 369.625, places=6)
        self.assertAlmostEqual(r1, 32.5, places=6)
        self.assertAlmostEqual(geom.center_col_0based, 368.625, places=6)
        self.assertAlmostEqual(geom.center_row_0based, 31.5, places=6)

        # Orthonormal detector basis check across all views
        e_ray, e_u, e_z = geom.detector_basis_vectors(geom.phi0_rad)
        np.testing.assert_allclose(np.linalg.norm(e_ray, axis=-1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(e_u, axis=-1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(e_ray * e_u, axis=-1), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(e_ray * e_z, axis=-1), 0.0, atol=1e-12)

        # Row 0 must be at +z and Row N-1 at -z
        v_rows = geom.row_axial_offsets_mm()
        self.assertGreater(v_rows[0], 0.0)
        self.assertLess(v_rows[-1], 0.0)
        self.assertAlmostEqual(float(v_rows[0] + v_rows[-1]), 0.0, places=9)

        # Dynamic pitch fit check
        metrics = geom.trajectory_metrics()
        self.assertAlmostEqual(float(metrics["normalized_pitch"]), 0.90, places=6)
        self.assertAlmostEqual(float(metrics["pitch_fit_rmse_mm"]), 0.0, places=9)
        self.assertEqual(int(metrics["rotation_direction"]), -1)
        self.assertEqual(int(metrics["table_direction"]), 1)

        # NPZ serialization roundtrip
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "geom.npz"
            geom.to_npz(npz_path)
            loaded = HelicalGeometry.from_npz(npz_path)
            self.assertEqual(loaded.series_instance_uid, geom.series_instance_uid)
            self.assertEqual(loaded.detector_shape, geom.detector_shape)
            np.testing.assert_allclose(loaded.phi0_unwrapped_rad, geom.phi0_unwrapped_rad)
            np.testing.assert_allclose(loaded.drho_mm, geom.drho_mm)

    def test_exact_3d_helical_conjugate_ray_line_identity(self) -> None:
        """Prove algebraically to machine precision (<1e-11 mm) that 3D helical conjugate rays coincide."""
        n_cols = 128
        n_rows = 32
        c0 = 64.25
        r0 = 15.5
        du = 2.0
        dv = 1.5
        rho0 = 595.0
        d0 = 1085.6
        c1 = 44  # non-central column
        gamma1 = (c1 - c0) * (du / d0)
        c2 = 2.0 * c0 - c1  # exact conjugate column

        phi1 = 0.35
        phi2 = phi1 + math.pi + 2.0 * gamma1
        z1 = 10.0
        z2 = 26.0  # 16 mm half-turn advance

        # Solve exact 3D helical conjugate row offsets:
        s0 = rho0 * math.cos(gamma1)
        m_slope = (z2 - z1) / (2.0 * s0)
        v1 = m_slope * d0
        v2 = -m_slope * d0
        r1 = r0 - v1 / dv
        r2 = r0 - v2 / dv

        # Build 2-view geometry where r0_view0 puts row 0 at v1 and view1 puts row 0 at v2
        geom = HelicalGeometry(
            series_instance_uid="1.2.3",
            detector_shape="CYLINDRICAL",
            n_rows=n_rows,
            n_cols=n_cols,
            du_mm=du,
            dv_mm=dv,
            rho0_mm=rho0,
            d0_mm=d0,
            center_col_0based=c0,
            center_row_0based=r0,
            ffs_mode="FFSNONE",
            angular_steps_per_rotation=360,
            water_mu_mm_inv=0.0192,
            nominal_spiral_pitch=0.8,
            log_flag="YES",
            rescale_slope=1.0,
            rescale_intercept=0.0,
            instance_numbers=np.array([1, 2], dtype=np.int32),
            phi0_rad=np.array([phi1, phi2], dtype=np.float64),
            phi0_unwrapped_rad=np.array([phi1, phi2], dtype=np.float64),
            z0_mm=np.array([z1, z2], dtype=np.float64),
            dphi_rad=np.zeros(2),
            dz_mm=np.zeros(2),
            drho_mm=np.zeros(2),
        )
        s_xyz = geom.source_cartesian_mm()  # (2, 3)
        e_ray, e_u, e_z = geom.detector_basis_vectors(geom.phi0_rad)
        f0 = geom.focal_center_cartesian_mm()

        d1_xyz = f0[0] + d0 * math.cos(gamma1) * e_ray[0] + d0 * math.sin(gamma1) * e_u[0] + v1 * e_z[0]
        gamma2 = (c2 - c0) * (du / d0)
        d2_xyz = f0[1] + d0 * math.cos(gamma2) * e_ray[1] + d0 * math.sin(gamma2) * e_u[1] + v2 * e_z[1]

        u1 = (d1_xyz - s_xyz[0]) / np.linalg.norm(d1_xyz - s_xyz[0])
        u2 = (d2_xyz - s_xyz[1]) / np.linalg.norm(d2_xyz - s_xyz[1])
        # Unit ray vectors must be exact opposites in 3D!
        np.testing.assert_allclose(u1, -u2, atol=1e-12)
        # Distance from S2 to the 3D line (S1 + t*u1) must be zero!
        cross_dist = float(np.linalg.norm(np.cross(s_xyz[1] - s_xyz[0], u1)))
        self.assertLess(cross_dist, 1e-10)
        # Both rows must lie well within [0, n_rows - 1]
        self.assertTrue(0.0 <= r1 <= n_rows - 1.0)
        self.assertTrue(0.0 <= r2 <= n_rows - 1.0)

    def test_phantom_voxelizer_and_analytical_vs_numerical_ray_marching(self) -> None:
        phantom = build_thorax_phantom(axial_scale=1.0)
        vox = voxelize_phantom(
            phantom,
            shape_zyx=(65, 129, 129),
            spacing_zyx_mm=(2.5, 2.8, 2.8),
            water_mu_mm_inv=0.0192,
        )
        hu = vox["hu_volume"]
        # Air outside body ~ -1000 HU, cortical spine ~ +820 HU
        self.assertAlmostEqual(float(hu[0, 0, 0]), -1000.0, places=3)
        self.assertAlmostEqual(float(np.max(hu)), 820.0, places=2)

        # Compare analytical helical cone-beam integral with FFSZ against fine 3D numerical ray marching
        geom = make_synthetic_helical_geometry(
            n_views=4,
            n_rows=8,
            n_cols=16,
            angular_steps_per_rotation=64,
            normalized_pitch=0.9,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            du_mm=24.0,
            dv_mm=6.0,
        )
        proj_analytic = project_phantom_helical(phantom, geom)
        # Pick oblique view 1 (odd view with drho=+5.45, dz=-0.66), row 2, col 9
        src, u_dirs, lengths = geom.ray_origins_and_directions(1)
        r_idx, c_idx = 2, 9
        L = float(lengths[r_idx, c_idx])
        u = u_dirs[r_idx, c_idx]

        # 40,000-step midpoint rule along the exact 3D segment [0, L]
        n_steps = 40000
        dt = L / n_steps
        t_vals = (np.arange(n_steps, dtype=np.float64) + 0.5) * dt
        pts = src[None, :] + t_vals[:, None] * u[None, :]  # (N, 3)

        mu_samples = np.zeros(n_steps, dtype=np.float64)
        for ell in phantom:
            cx, cy, cz = ell.center_xyz_mm
            ax, ay, az = ell.semi_axes_xyz_mm
            px = pts[:, 0] - cx
            py = pts[:, 1] - cy
            pz = pts[:, 2] - cz
            if ell.yaw_rad != 0.0:
                c_y = math.cos(ell.yaw_rad)
                s_y = math.sin(ell.yaw_rad)
                px, py = px * c_y + py * s_y, -px * s_y + py * c_y
            inside = (px / ax) ** 2 + (py / ay) ** 2 + (pz / az) ** 2 <= 1.0
            mu_samples[inside] += ell.delta_mu_mm_inv(geom.water_mu_mm_inv)

        num_integral = float(np.sum(np.maximum(mu_samples, 0.0)) * dt)
        ana_integral = float(proj_analytic[1, r_idx, c_idx])
        self.assertGreater(ana_integral, 0.5)
        self.assertAlmostEqual(ana_integral, num_integral, delta=5e-4)


@unittest.skipUnless(HAS_PYDICOM, "pydicom is required for DICOM-CT-PD .dcm round-trip tests")
class TestStage1DicomCtpdRoundTripAndOrientation(unittest.TestCase):
    """End-to-end synthetic DICOM-CT-PD v4 file generation, reader, orientation proof, and CLI audit."""

    def _write_synthetic_vendor_ct_slices(
        self,
        img_dir: Path,
        *,
        hu_vol: np.ndarray,
        z_mm: np.ndarray,
        referenced_raw_series_uid: str,
    ) -> None:
        img_dir.mkdir(parents=True, exist_ok=True)
        series_uid = "1.2.826.0.1.3680043.10.999.2001"
        study_uid = "1.2.826.0.1.3680043.10.999.2000"
        for k in range(hu_vol.shape[0]):
            sop_uid = f"{series_uid}.{k + 1}"
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = CT_IMAGE_SOP_CLASS_UID
            meta.MediaStorageSOPInstanceUID = sop_uid
            meta.TransferSyntaxUID = ImplicitVRLittleEndian

            ds = FileDataset(None, {}, file_meta=meta, preamble=b"\x00" * 128)
            ds.SOPClassUID = CT_IMAGE_SOP_CLASS_UID
            ds.SOPInstanceUID = sop_uid
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.999.9999"
            ds.Modality = "CT"
            ds.Manufacturer = "Siemens Healthcare"
            ds.BodyPartExamined = "CHEST"
            ds.SeriesDescription = "Synthetic Full Dose Images"
            ds.PatientID = "SYNTH_THORAX_01"
            ds.ConvolutionKernel = "B50f"
            ds.InstanceNumber = k + 1
            ds.Rows = int(hu_vol.shape[1])
            ds.Columns = int(hu_vol.shape[2])
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelSpacing = [4.5, 4.5]
            ds.SliceThickness = 1.5
            ds.ImagePositionPatient = [-150.0, -320.0, float(z_mm[k]) - 200.0]
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = -1024.0

            ref_series_item = Dataset()
            ref_series_item.add_new((0x0020, 0x000E), "UI", referenced_raw_series_uid)
            ref_raw_item = Dataset()
            ref_raw_item.add_new((0x0008, 0x1115), "SQ", Sequence([ref_series_item]))
            ds.add_new((0x0008, 0x9121), "SQ", Sequence([ref_raw_item]))

            raw_u16 = np.clip(np.rint(hu_vol[k] + 1024.0), 0.0, 65535.0).astype("<u2")
            ds.PixelData = raw_u16.tobytes(order="C")
            out_slice = img_dir / f"1-{k + 1:03d}.dcm"
            try:
                pydicom.dcmwrite(out_slice, ds, enforce_file_format=True)
            except TypeError:
                ds.is_little_endian = True
                ds.is_implicit_VR = True
                pydicom.dcmwrite(out_slice, ds, write_like_original=False)

    def test_full_dicom_ctpd_roundtrip_orientation_proofs_and_cli_audit(self) -> None:
        geom = make_synthetic_helical_geometry(
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
        proj_gt = project_phantom_helical(phantom, geom)
        # Reconstructed slice z-span (31 * 1.5 = 46.5 mm) fits inside helical z0-span (59.4 mm)
        vox = voxelize_phantom(
            phantom, shape_zyx=(32, 64, 64), spacing_zyx_mm=(1.5, 4.5, 4.5)
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj_dir = root / "raw" / "tcia" / "SYNTH01" / "projection_full" / geom.series_instance_uid
            img_dir = root / "raw" / "tcia" / "SYNTH01" / "reconstruction_published" / "img_series_01"

            written = write_synthetic_ctpd_series(proj_dir, geom, proj_gt)
            self.assertEqual(len(written), 384)
            self._write_synthetic_vendor_ct_slices(
                img_dir,
                hu_vol=vox["hu_volume"],
                z_mm=vox["z_mm"],
                referenced_raw_series_uid=geom.series_instance_uid,
            )

            # 1. Stage 0 indexer compatibility check
            idx_summary = index_dicom_tree(root, workers=4, show_progress=False)
            self.assertEqual(idx_summary["read_errors"], 0)
            statuses = {s["dicom_kind"]: s["validation_status"] for s in idx_summary["series_summaries"]}
            self.assertEqual(statuses["dicom_ct_pd_projection"], "VALID")
            self.assertEqual(statuses["ct_image"], "VALID")

            # 2. Chunked & memmap reader fidelity check
            geom_read, files = scan_projection_series(proj_dir, workers=4)
            self.assertAlmostEqual(geom_read.center_col_0based, geom.center_col_0based, places=5)
            self.assertAlmostEqual(geom_read.center_row_0based, geom.center_row_0based, places=5)

            p0 = read_single_projection(files[0])
            self.assertEqual(p0.shape, (32, 128))
            self.assertEqual(p0.dtype, np.float32)

            chunks = list(iter_projection_chunks(files, geom_read, chunk_views=100, workers=2))
            self.assertEqual(len(chunks), 4)

            mm_file = root / "cache" / "proj.mmap"
            _, mm = load_projections_memmap(proj_dir, mm_file, chunk_views=128, workers=2)
            rmse = float(np.sqrt(np.mean((np.asarray(mm) - proj_gt) ** 2)))
            self.assertLess(rmse, 1.0e-4)
            del mm

            # 3. Quantitative 3-part physical orientation verification
            orient = verify_projection_orientation(proj_dir, workers=2)
            self.assertTrue(orient["overall_pass"])
            self.assertTrue(orient["layout_proof"]["pass"])
            self.assertLess(orient["layout_proof"]["fortran_over_c_ratio"], 0.25)

            conj = orient["helical_conjugate_ray_proof"]
            self.assertTrue(conj["available"])
            self.assertTrue(conj["axial_row_sign_pass"])
            self.assertTrue(conj["transverse_col_and_c0_pass"])
            # True geometry MAE must be dramatically smaller than flipped rows or flipped cols
            self.assertLess(conj["mae_true_geometry"] * 5.0, conj["mae_flipped_axial_rows"])
            self.assertLess(conj["mae_true_geometry"] * 5.0, conj["mae_flipped_detector_cols"])
            self.assertLess(conj["mae_true_geometry"], conj["mae_unconverted_1based_c0"])
            self.assertLess(conj["mae_true_geometry"], conj["mae_mirrored_c0"])

            ft = orient["full_turn_axial_overlap_proof"]
            self.assertTrue(ft["available"])
            self.assertTrue(ft["pass"])
            self.assertLess(ft["mae_true_row0_plus_z"] * 3.0, ft["mae_flipped_row0_minus_z"])

            # 4. Published CT image volume reader check
            vol_info = read_ct_image_volume(img_dir)
            self.assertEqual(vol_info["referenced_raw_series_uid"], geom.series_instance_uid)
            self.assertEqual(vol_info["shape_zyx"], (32, 64, 64))
            np.testing.assert_allclose(vol_info["hu_volume"], vox["hu_volume"], atol=1.0)

            # 5. CLI hpct stage1-audit end-to-end check
            rc = cli_main(
                [
                    "stage1-audit",
                    "--data-root",
                    str(root),
                    "--run-phantom-check",
                    "--workers",
                    "2",
                    "--no-progress",
                ]
            )
            self.assertEqual(rc, 0)
            audit_json = root / "manifests" / "stage1_audit_report.json"
            self.assertTrue(audit_json.exists())
            audit_data = json.loads(audit_json.read_text(encoding="utf-8"))
            self.assertTrue(audit_data["overall_pass"])
            self.assertEqual(audit_data["audited_series_count"], 1)
            self.assertTrue(audit_data["audited_series"][0]["published_image_sanity"]["deterministic_uid_link_verified"])
            self.assertTrue(audit_data["synthetic_phantom_roundtrip"]["pass"])

    def test_negative_gates_catch_corrupted_order_and_missing_views(self) -> None:
        geom = make_synthetic_helical_geometry(
            n_views=48,
            n_rows=16,
            n_cols=64,
            angular_steps_per_rotation=48,
            normalized_pitch=0.8,
            ffs_mode="FFSZ",
        )
        phantom = build_thorax_phantom(axial_scale=0.3)
        proj = project_phantom_helical(phantom, geom)

        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "series"
            files = write_synthetic_ctpd_series(pdir, geom, proj)

            # Corrupt PixelData to wrong C-order reshape -> layout check MUST fail!
            for fpath in files:
                ds = pydicom.dcmread(fpath)
                raw_f = np.frombuffer(ds.PixelData, dtype="<u2").reshape((16, 64), order="F")
                # Intentionally write C-order bytes while header says Rows=64, Columns=16
                ds.PixelData = raw_f.tobytes(order="C")
                try:
                    pydicom.dcmwrite(fpath, ds, enforce_file_format=True)
                except TypeError:
                    pydicom.dcmwrite(fpath, ds, write_like_original=False)

            orient_bad = verify_projection_orientation(pdir, workers=1)
            self.assertFalse(orient_bad["layout_proof"]["pass"])
            self.assertFalse(orient_bad["overall_pass"])

            # Delete middle view -> scan_projection_series MUST raise ValueError (no silent fallback)
            files[10].unlink()
            with self.assertRaises(ValueError):
                scan_projection_series(pdir, workers=1)


if __name__ == "__main__":
    unittest.main()
