from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from highpitchct.cli import _discover_local_case_series, main as cli_main
from highpitchct.ctpd_reader import read_single_projection, scan_projection_series
from highpitchct.phantom import (
    Ellipsoid3D,
    build_thorax_phantom,
    make_synthetic_helical_geometry,
    project_phantom_helical,
    write_synthetic_ctpd_series,
)
from highpitchct.rebinning import (
    _solve_ray_detector_hit_local,
    project_phantom_planar_fanbeam_2d,
    rebin_curved_to_flat,
    rebin_curved_to_flat_scalar_ref,
    rebin_helical_ssrb,
    rebin_helical_ssrb_scalar_ref,
    rebin_series_ssrb_streaming,
)
from highpitchct.visual_qa import (
    save_stage1_phantom_visual_qa,
    save_stage2_phantom_ssrb_visual_qa,
)

try:
    import pydicom  # noqa: F401

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"File {path} does not start with valid PNG signature")
    if raw[12:16] != b"IHDR":
        raise AssertionError(f"File {path} missing IHDR chunk")
    width, height = struct.unpack(">II", raw[16:24])
    return int(width), int(height)


class TestStage2PureNumPyRebinningAndSSRB(unittest.TestCase):
    """Pure NumPy mathematical, geometric, scalar-parity, and ground-truth tests for Stage 2."""

    def test_exact_3d_ray_collinearity_across_all_detector_shapes_with_ffs(self) -> None:
        """Prove (<1e-11 rad) that virtual flat pixel rays hit exact physical (r_hit, c_hit)."""
        for shape in ("CYLINDRICAL", "FLAT", "SPHERICAL"):
            geom = make_synthetic_helical_geometry(
                n_views=4,
                n_rows=32,
                n_cols=96,
                angular_steps_per_rotation=96,
                normalized_pitch=0.9,
                detector_shape=shape,
                ffs_mode="FFSZ",
                rotation_direction=-1,
                quarter_detector_offset_cols=1.125,
            )
            # View 1 has non-zero 3D FFS: drho=+5.45 mm, dz=-0.66 mm, dphi=0.000333 rad
            i_view = 1
            src_xyz = geom.source_cartesian_mm(i_view)
            phi_s = float(geom.phi0_rad[i_view] + geom.dphi_rad[i_view])
            e_ray_s, e_u_s, e_z_s = geom.detector_basis_vectors(phi_s)

            phi_0 = float(geom.phi0_rad[i_view])
            f0_xyz = geom.focal_center_cartesian_mm(i_view)
            e_ray_0, e_u_0, e_z_0 = geom.detector_basis_vectors(phi_0)

            for u_mm in (-140.0, 0.0, 95.0):
                for v_mm in (-22.0, 0.0, 18.5):
                    r_hit_arr, c_hit_arr = _solve_ray_detector_hit_local(
                        shape,
                        d0_mm=geom.d0_mm,
                        du_mm=geom.du_mm,
                        dv_mm=geom.dv_mm,
                        c0_0based=geom.center_col_0based,
                        r0_0based=geom.center_row_0based,
                        rho0_mm=geom.rho0_mm,
                        drho_mm=float(geom.drho_mm[i_view]),
                        dphi_rad=float(geom.dphi_rad[i_view]),
                        dz_mm=float(geom.dz_mm[i_view]),
                        u_flat_mm=u_mm,
                        v_flat_mm=v_mm,
                    )
                    rf = float(r_hit_arr)
                    cf = float(c_hit_arr)

                    # 3D point on virtual centered flat detector
                    p_flat = src_xyz + geom.d0_mm * e_ray_s + u_mm * e_u_s + v_mm * e_z_s
                    dir_flat = p_flat - src_xyz
                    dir_flat /= np.linalg.norm(dir_flat)

                    # 3D point on physical detector surface at continuous (rf, cf)
                    u_phys = (cf - geom.center_col_0based) * geom.du_mm
                    v_phys = (geom.center_row_0based - rf) * geom.dv_mm
                    if shape == "CYLINDRICAL":
                        gam = u_phys / geom.d0_mm
                        a_loc = geom.d0_mm * math.cos(gam)
                        b_loc = geom.d0_mm * math.sin(gam)
                        c_loc = v_phys
                    elif shape == "FLAT":
                        a_loc = geom.d0_mm
                        b_loc = u_phys
                        c_loc = v_phys
                    else:
                        gam = u_phys / geom.d0_mm
                        alp = v_phys / geom.d0_mm
                        a_loc = geom.d0_mm * math.cos(alp) * math.cos(gam)
                        b_loc = geom.d0_mm * math.cos(alp) * math.sin(gam)
                        c_loc = geom.d0_mm * math.sin(alp)

                    p_phys = f0_xyz + a_loc * e_ray_0 + b_loc * e_u_0 + c_loc * e_z_0
                    dir_phys = p_phys - src_xyz
                    dir_phys /= np.linalg.norm(dir_phys)

                    sin_angle_err = float(np.linalg.norm(np.cross(dir_flat, dir_phys)))
                    dot = float(np.dot(dir_flat, dir_phys))
                    self.assertGreater(dot, 0.999999999999, msg=f"Opposite ray direction for {shape}")
                    self.assertLess(sin_angle_err, 1e-13, msg=f"Failed collinearity for {shape}")

    def test_vectorized_vs_scalar_curved_to_flat_and_ssrb_parity(self) -> None:
        """Verify vectorized Curved-to-Flat and SSRB match explicit scalar references (<1e-6)."""
        phantom = build_thorax_phantom(axial_scale=0.5)
        for shape in ("CYLINDRICAL", "FLAT", "SPHERICAL"):
            geom = make_synthetic_helical_geometry(
                n_views=36,
                n_rows=12,
                n_cols=32,
                angular_steps_per_rotation=18,
                normalized_pitch=0.90,
                detector_shape=shape,
                ffs_mode="FFSZ",
                rotation_direction=-1,
                quarter_detector_offset_cols=1.125,
                du_mm=10.0,
                dv_mm=4.0,
            )
            proj = project_phantom_helical(phantom, geom, batch_views=36)
            flat_vec, mask_vec = rebin_curved_to_flat(proj, geom, batch_views=7)
            flat_ref, mask_ref = rebin_curved_to_flat_scalar_ref(proj, geom)
            self.assertTrue(np.array_equal(mask_vec, mask_ref))
            np.testing.assert_allclose(flat_vec, flat_ref, atol=1e-6, rtol=1e-6)
            self.assertTrue(np.all(flat_vec[~mask_vec] == 0.0))

        for rot_dir in (-1, 1):
            geom_ssrb = make_synthetic_helical_geometry(
                n_views=48,
                n_rows=14,
                n_cols=36,
                angular_steps_per_rotation=24,
                normalized_pitch=1.35,
                detector_shape="CYLINDRICAL",
                ffs_mode="FFSZ",
                rotation_direction=rot_dir,
                quarter_detector_offset_cols=1.125,
                du_mm=9.0,
                dv_mm=3.5,
            )
            proj_s = project_phantom_helical(phantom, geom_ssrb, batch_views=48)
            flat_s, mask_s = rebin_curved_to_flat(proj_s, geom_ssrb)
            for mode in ("short_scan", "full_scan"):
                res_vec = rebin_helical_ssrb(
                    flat_s, mask_s, geom_ssrb, [-4.0, 0.0, 5.0], angular_mode=mode
                )
                res_ref = rebin_helical_ssrb_scalar_ref(
                    flat_s, mask_s, geom_ssrb, [-4.0, 0.0, 5.0], angular_mode=mode
                )
                self.assertTrue(np.array_equal(res_vec.measured_mask, res_ref.measured_mask))
                np.testing.assert_allclose(
                    res_vec.sinograms, res_ref.sinograms, atol=1e-6, rtol=1e-6
                )
                # Verify monotonically increasing beta_rad
                self.assertTrue(np.all(np.diff(res_vec.beta_rad, axis=1) > 0.0))

    def test_z_extruded_phantom_exact_recovery_and_3d_ffs_drho_necessity(self) -> None:
        """Prove exact 2D sinogram recovery on z-invariant phantom and necessity of 3D FFS drho."""
        extruded = [
            Ellipsoid3D("trunk", (0.0, 0.0, 0.0), (140.0, 100.0, 2500.0), 1000.0, 0.0),
            Ellipsoid3D("offcenter_bone", (110.0, -35.0, 0.0), (20.0, 16.0, 2500.0), 900.0, 0.25),
            Ellipsoid3D("right_lung", (-60.0, 15.0, 0.0), (42.0, 55.0, 2500.0), -750.0, -0.15),
        ]
        geom = make_synthetic_helical_geometry(
            n_views=960,
            n_rows=32,
            n_cols=256,
            angular_steps_per_rotation=576,
            normalized_pitch=0.90,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=2.0,
            dv_mm=2.0,
        )
        proj = project_phantom_helical(extruded, geom, batch_views=128)
        flat_true, valid_true = rebin_curved_to_flat(proj, geom)
        ssrb_true = rebin_helical_ssrb(
            flat_true, valid_true, geom, [0.0], angular_mode="short_scan"
        )
        exact_2d = project_phantom_planar_fanbeam_2d(
            extruded,
            z_slices_mm=[0.0],
            beta_rad=ssrb_true.beta_rad,
            n_u=ssrb_true.n_u,
            du_mm=ssrb_true.du_mm,
            rho0_mm=ssrb_true.rho0_mm,
            d0_mm=ssrb_true.d0_mm,
            water_mu_mm_inv=geom.water_mu_mm_inv,
        )
        mae_true = float(np.mean(np.abs(ssrb_true.sinograms[0] - exact_2d[0])))
        self.assertGreaterEqual(ssrb_true.overall_measured_fraction, 0.99)
        self.assertLess(mae_true, 0.004)

        # Now deliberately ignore FFS shifts (Helix2Fan bug: drho=0, dz=0, dphi=0)
        from dataclasses import replace

        geom_no_ffs = replace(
            geom,
            drho_mm=np.zeros_like(geom.drho_mm),
            dz_mm=np.zeros_like(geom.dz_mm),
            dphi_rad=np.zeros_like(geom.dphi_rad),
        )
        flat_bug, valid_bug = rebin_curved_to_flat(proj, geom_no_ffs)
        ssrb_bug = rebin_helical_ssrb(
            flat_bug, valid_bug, geom_no_ffs, [0.0], angular_mode="short_scan"
        )
        mae_bug = float(np.mean(np.abs(ssrb_bug.sinograms[0] - exact_2d[0])))
        self.assertGreater(mae_bug, 2.0 * mae_true)

    def test_high_pitch_and_rdc_missing_mask_strict_zero_no_smearing(self) -> None:
        """Verify high-pitch (p=1.9) and RDC collimation produce true missing-data masks with 0.0."""
        phantom = build_thorax_phantom(axial_scale=0.6)
        geom_hp = make_synthetic_helical_geometry(
            n_views=384,
            n_rows=32,
            n_cols=128,
            angular_steps_per_rotation=192,
            normalized_pitch=1.90,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
        )
        proj_hp = project_phantom_helical(phantom, geom_hp, batch_views=96)
        flat_hp, valid_hp = rebin_curved_to_flat(proj_hp, geom_hp)
        ssrb_hp = rebin_helical_ssrb(
            flat_hp, valid_hp, geom_hp, [0.0], angular_mode="short_scan"
        )
        frac_hp = ssrb_hp.overall_measured_fraction
        self.assertGreater(frac_hp, 0.45)
        self.assertLess(frac_hp, 0.92)
        self.assertTrue(np.all(ssrb_hp.sinograms[~ssrb_hp.measured_mask] == 0.0))

        # Verify RDC active_row_range on native pitch=0.90 reduces measured_fraction as expected
        geom_nat = make_synthetic_helical_geometry(
            n_views=384,
            n_rows=32,
            n_cols=128,
            angular_steps_per_rotation=192,
            normalized_pitch=0.90,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
        )
        proj_nat = project_phantom_helical(phantom, geom_nat, batch_views=96)
        flat_nat, valid_nat = rebin_curved_to_flat(proj_nat, geom_nat)
        ssrb_full_rows = rebin_helical_ssrb(flat_nat, valid_nat, geom_nat, [0.0])
        ssrb_rdc_rows = rebin_helical_ssrb(
            flat_nat, valid_nat, geom_nat, [0.0], active_row_range=(8.0, 23.0)
        )
        self.assertGreaterEqual(ssrb_full_rows.overall_measured_fraction, 0.99)
        self.assertLess(ssrb_rdc_rows.overall_measured_fraction, 0.92)
        self.assertTrue(np.all(ssrb_rdc_rows.sinograms[~ssrb_rdc_rows.measured_mask] == 0.0))

    def test_case_id_discovery_and_visual_qa_png_outputs(self) -> None:
        """Verify C012 <-> case_C012 discovery, strict fail-fast, and valid PNG generation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tcia = root / "raw" / "tcia"
            pdir = tcia / "case_C012" / "projection_full" / "1.2.3.4"
            idir = tcia / "case_C012" / "reconstruction_published" / "1.2.3.5"
            pdir.mkdir(parents=True)
            idir.mkdir(parents=True)

            # Both 'C012' and 'case_C012' must find `case_C012`
            found_short = _discover_local_case_series(tcia, case_id="C012")
            found_prefixed = _discover_local_case_series(tcia, case_id="case_C012")
            self.assertEqual(len(found_short), 1)
            self.assertEqual(found_short[0][0], "case_C012")
            self.assertEqual(found_short, found_prefixed)

            # Missing explicit --case-id MUST fail fast even when --run-phantom-check is set
            with self.assertRaises(SystemExit):
                cli_main(
                    [
                        "stage2-rebin",
                        "--data-root",
                        str(root),
                        "--case-id",
                        "C999",
                        "--run-phantom-check",
                        "--no-progress",
                    ]
                )

            # Test both default renderer AND forced stdlib PNG renderer
            sino = np.linspace(0.0, 3.0, 64 * 80, dtype=np.float32).reshape(64, 80)
            mask = np.ones((64, 80), dtype=bool)
            mask[:10, :20] = False
            png1 = save_stage2_phantom_ssrb_visual_qa(
                root / "fig_default.png",
                exact_planar_sino=sino,
                ssrb_low_pitch_sino=sino * 0.99,
                ssrb_low_pitch_mask=np.ones_like(mask),
                ssrb_high_pitch_sino=np.where(mask, sino, 0.0),
                ssrb_high_pitch_mask=mask,
                z_slice_mm=0.0,
                low_pitch=0.85,
                high_pitch=1.85,
            )
            w1, h1 = _read_png_dimensions(png1)
            self.assertGreater(w1, 200)
            self.assertGreater(h1, 200)

            with mock.patch("highpitchct.visual_qa._has_matplotlib", return_value=False):
                png2 = save_stage1_phantom_visual_qa(
                    root / "fig_stdlib.png",
                    hu_volume=np.zeros((16, 32, 32), dtype=np.float32),
                    projections=np.ones((24, 16, 32), dtype=np.float32),
                )
                w2, h2 = _read_png_dimensions(png2)
                self.assertGreater(w2, 200)
                self.assertGreater(h2, 200)


@unittest.skipUnless(HAS_PYDICOM, "pydicom required for streaming DICOM-CT-PD Stage 2 tests")
class TestStage2StreamingAndCLIDicomIntegration(unittest.TestCase):
    """End-to-end DICOM-CT-PD streaming SSRB equivalence and CLI integration tests."""

    def test_streaming_ssrb_exact_equivalence_to_in_memory_and_cli_stage2(self) -> None:
        phantom = build_thorax_phantom(axial_scale=0.5)
        geom = make_synthetic_helical_geometry(
            n_views=288,
            n_rows=24,
            n_cols=96,
            angular_steps_per_rotation=96,
            normalized_pitch=0.92,
            detector_shape="CYLINDRICAL",
            ffs_mode="FFSZ",
            rotation_direction=-1,
            quarter_detector_offset_cols=1.125,
            du_mm=5.0,
            dv_mm=2.2,
        )
        proj = project_phantom_helical(phantom, geom, batch_views=96)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdir = root / "raw" / "tcia" / "case_C012" / "projection_full" / geom.series_instance_uid
            write_synthetic_ctpd_series(pdir, geom, proj, patient_id="C012")

            # Load quantized projections in memory to compare streaming vs full in-memory
            geom_disk, files = scan_projection_series(pdir, workers=2)
            full_disk = np.stack(
                [
                    read_single_projection(f, expected_rows=geom.n_rows, expected_cols=geom.n_cols)
                    for f in files
                ],
                axis=0,
            )
            flat_mem, valid_mem = rebin_curved_to_flat(full_disk, geom_disk)

            # Intentionally pass unsorted z slices with small batch span to stress sliding cache
            z_test = [12.0, -14.0, 0.0, 5.0, -7.0, 15.0]
            res_mem = rebin_helical_ssrb(
                flat_mem, valid_mem, geom_disk, z_test, angular_mode="short_scan"
            )
            res_stream = rebin_series_ssrb_streaming(
                files,
                z_test,
                geometry=geom_disk,
                angular_mode="short_scan",
                max_batch_z_span_mm=6.0,
                workers=2,
                show_progress=False,
            )
            self.assertTrue(np.array_equal(res_stream.measured_mask, res_mem.measured_mask))
            np.testing.assert_allclose(res_stream.sinograms, res_mem.sinograms, atol=1e-6)

            # Now test `hpct stage2-rebin --case-id C012 --run-phantom-check` (matching case_C012!)
            sino_npz = root / "manifests" / "c012_sino.npz"
            rc = cli_main(
                [
                    "stage2-rebin",
                    "--data-root",
                    str(root),
                    "--case-id",
                    "C012",
                    "--num-slices",
                    "3",
                    "--sinograms-out",
                    str(sino_npz),
                    "--run-phantom-check",
                    "--workers",
                    "2",
                    "--no-progress",
                ]
            )
            self.assertEqual(rc, 0)

            rep_path = root / "manifests" / "stage2_rebin_report.json"
            self.assertTrue(rep_path.exists())
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            self.assertTrue(rep["overall_pass"])
            self.assertEqual(rep["rebinned_series_count"], 1)
            self.assertEqual(rep["rebinned_series"][0]["case_id"], "case_C012")
            self.assertTrue(sino_npz.exists())

            real_png = Path(rep["rebinned_series"][0]["visual_qa_png"])
            phan_png = Path(rep["synthetic_phantom_ssrb_check"]["visual_qa_png"])
            self.assertTrue(real_png.exists())
            self.assertTrue(phan_png.exists())
            self.assertGreater(_read_png_dimensions(real_png)[0], 200)
            self.assertGreater(_read_png_dimensions(phan_png)[0], 200)


if __name__ == "__main__":
    unittest.main()
