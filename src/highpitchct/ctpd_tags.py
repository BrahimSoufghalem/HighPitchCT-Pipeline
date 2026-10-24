"""Header-only DICOM-CT-PD v4 private tag decoding and SOP classification.

In TCIA LDCT-and-Projection-data (Implicit VR Little Endian, 1.2.840.10008.1.2),
pydicom loads private tags (7029,xxxx)-(7041,xxxx) with VR='UN' as raw bytes.
This module decodes them strictly according to Tables 1 and 2 of the
DICOM-CT-PD User Manual Version 4 without loading PixelData.
"""

from __future__ import annotations

import struct
from typing import Any

RAW_DATA_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66"
CT_IMAGE_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.2"

# Private tag registry: name -> (group, element, VR)
CTPD_PRIVATE_TAGS: dict[str, tuple[int, int, str]] = {
    "detector_rows": (0x7029, 0x1010, "US"),
    "detector_cols": (0x7029, 0x1011, "US"),
    "du_mm": (0x7029, 0x1002, "FL"),
    "dv_mm": (0x7029, 0x1006, "FL"),
    "detector_shape": (0x7029, 0x100B, "CS"),
    "phi0_rad": (0x7031, 0x1001, "FL"),
    "z0_mm": (0x7031, 0x1002, "FL"),
    "rho0_mm": (0x7031, 0x1003, "FL"),
    "d0_mm": (0x7031, 0x1031, "FL"),
    "detector_central_element_1based": (0x7031, 0x1033, "FL"),
    "dphi_rad": (0x7033, 0x100B, "FL"),
    "dz_mm": (0x7033, 0x100C, "FL"),
    "drho_mm": (0x7033, 0x100D, "FL"),
    "ffs_mode": (0x7033, 0x100E, "CS"),
    "angular_steps_per_rotation": (0x7033, 0x1013, "US"),
    "number_of_spectra": (0x7033, 0x1061, "US"),
    "spectrum_index": (0x7033, 0x1063, "US"),
    "photon_statistics": (0x7033, 0x1065, "FL"),
    "timestamp_ms": (0x7033, 0x1067, "FL"),
    "projection_data_type": (0x7037, 0x1009, "CS"),
    "projection_geometry_type": (0x7037, 0x100A, "CS"),
    "beam_hardening_flag": (0x7039, 0x1003, "CS"),
    "gain_correction_flag": (0x7039, 0x1004, "CS"),
    "dark_field_flag": (0x7039, 0x1005, "CS"),
    "flat_field_flag": (0x7039, 0x1006, "CS"),
    "bad_pixel_flag": (0x7039, 0x1007, "CS"),
    "scatter_correction_flag": (0x7039, 0x1008, "CS"),
    "log_flag": (0x7039, 0x1009, "CS"),
    "water_attenuation_mm_inv": (0x7041, 0x1001, "DS"),
}


def decode_tag(dataset: Any, group: int, element: int, vr_expected: str) -> Any:
    """Decode a standard or private DICOM tag even when VR='UN'."""
    tag = (group, element)
    if tag not in dataset:
        return None
    raw_value = dataset[tag].value
    if raw_value is None:
        return None

    if isinstance(raw_value, (bytes, bytearray)):
        if len(raw_value) == 0:
            return None
        if vr_expected == "FL":
            if len(raw_value) % 4 != 0:
                raise ValueError(f"Tag ({group:04X},{element:04X}) byte length {len(raw_value)} not divisible by 4")
            count = len(raw_value) // 4
            values = struct.unpack(f"<{count}f", raw_value)
            return float(values[0]) if count == 1 else [float(v) for v in values]
        if vr_expected == "FD":
            if len(raw_value) % 8 != 0:
                raise ValueError(f"Tag ({group:04X},{element:04X}) byte length {len(raw_value)} not divisible by 8")
            count = len(raw_value) // 8
            values = struct.unpack(f"<{count}d", raw_value)
            return float(values[0]) if count == 1 else [float(v) for v in values]
        if vr_expected == "US":
            if len(raw_value) % 2 != 0:
                raise ValueError(f"Tag ({group:04X},{element:04X}) byte length {len(raw_value)} not divisible by 2")
            count = len(raw_value) // 2
            values = struct.unpack(f"<{count}H", raw_value)
            return int(values[0]) if count == 1 else [int(v) for v in values]
        text = raw_value.decode("ascii", errors="replace").strip("\x00 ")
        if not text:
            return None
        if vr_expected == "DS":
            return float(text)
        if vr_expected == "IS":
            return int(text)
        return text

    if hasattr(raw_value, "__iter__") and not isinstance(raw_value, str):
        items = list(raw_value)
        if vr_expected in {"FL", "FD", "DS"}:
            return [float(v) for v in items]
        if vr_expected in {"US", "IS"}:
            return [int(v) for v in items]
        return [str(v).strip() for v in items]

    if vr_expected in {"FL", "FD", "DS"}:
        return float(raw_value)
    if vr_expected in {"US", "IS"}:
        return int(raw_value)
    return str(raw_value).strip()


def decode_named_ctpd_tag(dataset: Any, name: str) -> Any:
    group, element, vr = CTPD_PRIVATE_TAGS[name]
    return decode_tag(dataset, group, element, vr)


def classify_dicom_sop(dataset: Any) -> str:
    """Deterministically classify a DICOM dataset using SOPClassUID and private tags."""
    sop_uid = str(getattr(dataset, "SOPClassUID", "") or "").strip()
    if sop_uid == RAW_DATA_SOP_CLASS_UID:
        if (0x7029, 0x1010) in dataset and (0x7031, 0x1001) in dataset:
            return "dicom_ct_pd_projection"
        return "raw_data_other"
    if sop_uid == CT_IMAGE_SOP_CLASS_UID:
        return "ct_image"
    return "other_dicom"


def extract_referenced_raw_series_uid(dataset: Any) -> str:
    """Extract the linked projection SeriesInstanceUID from ReferenceRawDataSequence (0008,9121).

    In TCIA LDCT-and-Projection-data, reconstructed CT images have different
    StudyInstanceUID and FrameOfReferenceUID from raw projections. The exact
    deterministic link is stored inside (0008,9121) -> (0008,1115) -> (0020,000E).
    """
    if (0x0008, 0x9121) not in dataset:
        return ""
    raw_seq = dataset[(0x0008, 0x9121)].value
    if not raw_seq:
        return ""
    for item in raw_seq:
        if (0x0008, 0x1115) in item:
            series_seq = item[(0x0008, 0x1115)].value
            for series_item in series_seq or []:
                if (0x0020, 0x000E) in series_item:
                    uid = str(series_item[(0x0020, 0x000E)].value or "").strip()
                    if uid:
                        return uid
    return ""
