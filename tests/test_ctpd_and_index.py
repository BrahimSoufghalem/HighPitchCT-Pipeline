import math
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ImplicitVRLittleEndian

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from highpitchct.ctpd_tags import (
    CT_IMAGE_SOP_CLASS_UID,
    RAW_DATA_SOP_CLASS_UID,
    classify_dicom_sop,
    decode_named_ctpd_tag,
    extract_referenced_raw_series_uid,
)
from highpitchct.index_dicom import index_dicom_tree


def _write_synthetic_ctpd_projection(
    path: Path,
    *,
    series_uid: str,
    study_uid: str,
    instance_number: int,
    phi0_rad: float,
    z0_mm: float,
    patient_id: str = "SYNTH_01",
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = RAW_DATA_SOP_CLASS_UID
    meta.MediaStorageSOPInstanceUID = f"{series_uid}.{instance_number}"
    meta.TransferSyntaxUID = ImplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = RAW_DATA_SOP_CLASS_UID
    ds.SOPInstanceUID = f"{series_uid}.{instance_number}"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.Manufacturer = "SIEMENS"
    ds.BodyPartExamined = "CHEST"
    ds.SeriesDescription = "Full-dose-projections"
    ds.PatientID = patient_id
    ds.InstanceNumber = instance_number
    ds.Rows = 736
    ds.Columns = 64

    # Store private tags as raw VR='UN' bytes exactly like TCIA Implicit VR files
    ds.add_new((0x7029, 0x1010), "UN", struct.pack("<H", 64))
    ds.add_new((0x7029, 0x1011), "UN", struct.pack("<H", 736))
    ds.add_new((0x7029, 0x1002), "UN", struct.pack("<f", 1.2858393))
    ds.add_new((0x7029, 0x1006), "UN", struct.pack("<f", 1.0947227))
    ds.add_new((0x7029, 0x100B), "UN", b"CYLINDRICAL ")
    ds.add_new((0x7031, 0x1001), "UN", struct.pack("<f", phi0_rad))
    ds.add_new((0x7031, 0x1002), "UN", struct.pack("<f", z0_mm))
    ds.add_new((0x7031, 0x1003), "UN", struct.pack("<f", 595.0))
    ds.add_new((0x7031, 0x1031), "UN", struct.pack("<f", 1085.6))
    ds.add_new((0x7031, 0x1033), "UN", struct.pack("<2f", 369.625, 32.5))
    ds.add_new((0x7033, 0x100E), "UN", b"FFSZ")
    ds.add_new((0x7033, 0x1013), "UN", struct.pack("<H", 1152))
    ds.add_new((0x7041, 0x1001), "UN", b"0.0192")
    ds.save_as(str(path), write_like_original=False)


def _write_synthetic_ct_image(
    path: Path,
    *,
    series_uid: str,
    study_uid: str,
    referenced_raw_series_uid: str,
    instance_number: int,
    slice_z_mm: float,
    patient_id: str = "SYNTH_01",
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CT_IMAGE_SOP_CLASS_UID
    meta.MediaStorageSOPInstanceUID = f"{series_uid}.{instance_number}"
    meta.TransferSyntaxUID = ImplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = CT_IMAGE_SOP_CLASS_UID
    ds.SOPInstanceUID = f"{series_uid}.{instance_number}"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.Manufacturer = "Siemens Healthcare"
    ds.BodyPartExamined = "CHEST"
    ds.SeriesDescription = "Full Dose Images"
    ds.PatientID = patient_id
    ds.InstanceNumber = instance_number
    ds.Rows = 512
    ds.Columns = 512
    ds.ImagePositionPatient = [-199.6, -368.1, slice_z_mm]

    ref_series_item = Dataset()
    ref_series_item.SeriesInstanceUID = referenced_raw_series_uid
    ref_raw_item = Dataset()
    ref_raw_item.add_new((0x0008, 0x1115), "SQ", Sequence([ref_series_item]))
    ds.add_new((0x0008, 0x9121), "SQ", Sequence([ref_raw_item]))
    ds.save_as(str(path), write_like_original=False)


@unittest.skipUnless(HAS_PYDICOM, "pydicom is required for DICOM-CT-PD indexing tests")
class CtpdAndIndexTests(unittest.TestCase):
    def test_private_un_tags_and_reference_link_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pfile = root / "1-00001.dcm"
            ifile = root / "img-001.dcm"
            _write_synthetic_ctpd_projection(
                pfile,
                series_uid="1.2.840.999.1",
                study_uid="1.2.840.999.10",
                instance_number=1,
                phi0_rad=3.0,
                z0_mm=10.0,
            )
            _write_synthetic_ct_image(
                ifile,
                series_uid="1.2.840.999.2",
                study_uid="1.2.840.999.20",
                referenced_raw_series_uid="1.2.840.999.1",
                instance_number=1,
                slice_z_mm=-50.0,
            )

            p_ds = pydicom.dcmread(pfile, stop_before_pixels=True)
            self.assertEqual(classify_dicom_sop(p_ds), "dicom_ct_pd_projection")
            self.assertEqual(decode_named_ctpd_tag(p_ds, "detector_rows"), 64)
            self.assertEqual(decode_named_ctpd_tag(p_ds, "detector_cols"), 736)
            self.assertEqual(decode_named_ctpd_tag(p_ds, "detector_shape"), "CYLINDRICAL")
            self.assertAlmostEqual(decode_named_ctpd_tag(p_ds, "water_attenuation_mm_inv"), 0.0192)
            central = decode_named_ctpd_tag(p_ds, "detector_central_element_1based")
            self.assertAlmostEqual(central[0], 369.625, places=3)
            self.assertAlmostEqual(central[1], 32.5, places=3)

            i_ds = pydicom.dcmread(ifile, stop_before_pixels=True)
            self.assertEqual(classify_dicom_sop(i_ds), "ct_image")
            self.assertEqual(extract_referenced_raw_series_uid(i_ds), "1.2.840.999.1")

    def test_indexer_estimates_dynamic_pitch_and_flags_truncated_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan_dir = root / "external_case"
            proj_dir = scan_dir / "projections"
            img_dir = scan_dir / "images"
            proj_dir.mkdir(parents=True)
            img_dir.mkdir(parents=True)

            # Simulate pitch = 0.6 (feed = 0.6 * 38.4 = 23.04 mm/rot), 20 views spanning only ~4 mm
            feed_06 = 0.6 * 38.4
            dphi = -2.0 * math.pi / 100.0
            for k in range(1, 21):
                phi = 3.0 + (k - 1) * dphi
                z = 10.0 + (k - 1) * (feed_06 / 100.0)
                _write_synthetic_ctpd_projection(
                    proj_dir / f"1-{k:05d}.dcm",
                    series_uid="1.2.840.999.101",
                    study_uid="1.2.840.999.100",
                    instance_number=k,
                    phi0_rad=phi,
                    z0_mm=z,
                )

            # Linked image series spans 100 mm -> projections (~4.4 mm) should be flagged truncated!
            _write_synthetic_ct_image(
                img_dir / "1-001.dcm",
                series_uid="1.2.840.999.201",
                study_uid="1.2.840.999.200",
                referenced_raw_series_uid="1.2.840.999.101",
                instance_number=1,
                slice_z_mm=-10.0,
            )
            _write_synthetic_ct_image(
                img_dir / "1-002.dcm",
                series_uid="1.2.840.999.201",
                study_uid="1.2.840.999.200",
                referenced_raw_series_uid="1.2.840.999.101",
                instance_number=2,
                slice_z_mm=-110.0,
            )

            out_data_root = root / "data_out"
            summary = index_dicom_tree(out_data_root, scan_dirs=[scan_dir])
            self.assertEqual(summary["read_errors"], 0)
            self.assertEqual(summary["series_count"], 2)

            by_uid = {s["series_instance_uid"]: s for s in summary["series_summaries"]}
            proj_s = by_uid["1.2.840.999.101"]
            self.assertAlmostEqual(float(proj_s["estimated_normalized_pitch"]), 0.6, places=3)
            self.assertEqual(proj_s["linked_image_series_uids"], "1.2.840.999.201")
            self.assertEqual(proj_s["validation_status"], "WARNING_TRUNCATED_PROJECTION_SERIES")


if __name__ == "__main__":
    unittest.main()
