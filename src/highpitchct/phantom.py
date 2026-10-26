"""3D Analytical & Voxelized Thorax Phantom and Synthetic DICOM-CT-PD v4 Writer.

Provides:
1. Asymmetric 3D chest ellipsoid phantom (body, left/right lungs, heart, spine, marrow,
   aorta, sternum, cranial & caudal nodules) so any flip in x, y, or z is immediately caught.
2. Exact closed-form 3D ray-ellipsoid helical cone-beam forward projection supporting
   CYLINDRICAL/FLAT/SPHERICAL detectors and full 3D Flying Focal Spot (FFS) shifts.
3. Voxelizer producing exact Ground Truth mu [mm^-1] and HU volumes for Stage 2-9 tests.
4. Synthetic DICOM-CT-PD v4 (.dcm) writer reproducing Mayo/TCIA Implicit VR Little Endian
   private binary tags (VR='UN') and MATLAB Fortran-order PixelData layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct

import numpy as np

from .ctpd_reader import mu_to_hu
from .ctpd_tags import CTPD_PRIVATE_TAGS, RAW_DATA_SOP_CLASS_UID
from .geometry import HelicalGeometry, unwrap_angles_rad
from .progress import ProgressReporter


@dataclass(frozen=True)
class Ellipsoid3D:
    """3D ellipsoid with additive Hounsfield Unit step relative to background."""

    name: str
    center_xyz_mm: tuple[float, float, float]
    semi_axes_xyz_mm: tuple[float, float, float]
    delta_hu: float
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if any(ax <= 0.0 for ax in self.semi_axes_xyz_mm):
            raise ValueError(
                f"Ellipsoid {self.name!r} semi-axes must be positive, got {self.semi_axes_xyz_mm}"
            )

    def delta_mu_mm_inv(self, water_mu_mm_inv: float = 0.0192) -> float:
        return float(water_mu_mm_inv) * (float(self.delta_hu) / 1000.0)


def build_thorax_phantom(axial_scale: float = 1.0) -> list[Ellipsoid3D]:
    """Build an asymmetric 3D thorax phantom centered around (0, 0, 0) [mm].

    Patient coordinate convention (supine, DICOM-CT-PD Manual Fig. 5):
      +x = patient's left, -x = patient's right
      +y = anterior (sternum), -y = posterior (spine)
      +z = cranial (apical), -z = caudal (basal)

    Internal structures inside the soft-tissue trunk (+20 HU, so delta_hu=+1020 over -1000 HU air)
    are mutually disjoint except nested sub-structures (marrow inside spine, nodules inside lungs),
    so additive superposition produces exact clinical HU values everywhere.
    """
    s = float(axial_scale)
    if s <= 0.0:
        raise ValueError(f"axial_scale must be positive, got {axial_scale}")

    return [
        # Outer thorax trunk: raises air (-1000 HU) -> soft tissue (+20 HU)
        Ellipsoid3D(
            name="thorax_trunk",
            center_xyz_mm=(0.0, 0.0, 0.0),
            semi_axes_xyz_mm=(155.0, 108.0, 140.0 * s),
            delta_hu=1020.0,
            yaw_rad=0.0,
        ),
        # Right lung (-x): lowers soft tissue (+20 HU) -> lung parenchyma (-750 HU)
        Ellipsoid3D(
            name="right_lung",
            center_xyz_mm=(-64.0, 8.0, 4.0 * s),
            semi_axes_xyz_mm=(52.0, 68.0, 95.0 * s),
            delta_hu=-770.0,
            yaw_rad=-0.12,
        ),
        # Left lung (+x): smaller to accommodate anterior-left heart -> (-750 HU)
        Ellipsoid3D(
            name="left_lung",
            center_xyz_mm=(68.0, 2.0, 6.0 * s),
            semi_axes_xyz_mm=(45.0, 64.0, 90.0 * s),
            delta_hu=-770.0,
            yaw_rad=0.14,
        ),
        # Heart (anterior-left, caudal bias): raises (+20 HU) -> blood/myocardium (+55 HU)
        Ellipsoid3D(
            name="heart",
            center_xyz_mm=(16.0, 32.0, -18.0 * s),
            semi_axes_xyz_mm=(34.0, 38.0, 46.0 * s),
            delta_hu=35.0,
            yaw_rad=0.20,
        ),
        # Posterior spine cortical bone (-y): raises (+20 HU) -> cortical bone (+820 HU)
        Ellipsoid3D(
            name="spine_cortex",
            center_xyz_mm=(0.0, -68.0, 0.0),
            semi_axes_xyz_mm=(22.0, 20.0, 118.0 * s),
            delta_hu=800.0,
            yaw_rad=0.0,
        ),
        # Trabecular marrow nested inside spine: lowers (+820 HU) -> (+180 HU)
        Ellipsoid3D(
            name="spine_marrow",
            center_xyz_mm=(0.0, -68.0, 0.0),
            semi_axes_xyz_mm=(13.0, 12.0, 112.0 * s),
            delta_hu=-640.0,
            yaw_rad=0.0,
        ),
        # Descending aorta (posterior-left): raises (+20 HU) -> contrast blood (+140 HU)
        Ellipsoid3D(
            name="descending_aorta",
            center_xyz_mm=(18.0, -40.0, -8.0 * s),
            semi_axes_xyz_mm=(12.0, 12.0, 105.0 * s),
            delta_hu=120.0,
            yaw_rad=0.0,
        ),
        # Anterior sternum (+y): raises (+20 HU) -> (+520 HU)
        Ellipsoid3D(
            name="sternum",
            center_xyz_mm=(0.0, 88.0, 12.0 * s),
            semi_axes_xyz_mm=(18.0, 7.0, 72.0 * s),
            delta_hu=500.0,
            yaw_rad=0.0,
        ),
        # Right apical pulmonary nodule (cranial +z, right -x) inside right_lung: (-750 -> +40 HU)
        Ellipsoid3D(
            name="right_apical_nodule",
            center_xyz_mm=(-58.0, 18.0, 42.0 * s),
            semi_axes_xyz_mm=(9.0, 9.0, 9.0 * max(0.5, s)),
            delta_hu=790.0,
            yaw_rad=0.0,
        ),
        # Left basal high-density lesion (caudal -z, left +x) inside left_lung: (-750 -> +90 HU)
        Ellipsoid3D(
            name="left_basal_lesion",
            center_xyz_mm=(66.0, -14.0, -36.0 * s),
            semi_axes_xyz_mm=(12.0, 10.0, 11.0 * max(0.5, s)),
            delta_hu=840.0,
            yaw_rad=0.0,
        ),
    ]


def voxelize_phantom(
    ellipsoids: list[Ellipsoid3D],
    *,
    shape_zyx: tuple[int, int, int] = (128, 256, 256),
    spacing_zyx_mm: tuple[float, float, float] = (1.5, 1.4, 1.4),
    origin_zyx_mm: tuple[float, float, float] | None = None,
    water_mu_mm_inv: float = 0.0192,
) -> dict[str, np.ndarray]:
    """Voxelize 3D ellipsoid phantom onto a regular (Z, Y, X) Cartesian grid."""
    nz, ny, nx = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    dz, dy, dx = (float(spacing_zyx_mm[0]), float(spacing_zyx_mm[1]), float(spacing_zyx_mm[2]))

    if origin_zyx_mm is None:
        z0 = -0.5 * (nz - 1) * dz
        y0 = -0.5 * (ny - 1) * dy
        x0 = -0.5 * (nx - 1) * dx
    else:
        z0, y0, x0 = (float(origin_zyx_mm[0]), float(origin_zyx_mm[1]), float(origin_zyx_mm[2]))

    z_mm = z0 + np.arange(nz, dtype=np.float64) * dz
    y_mm = y0 + np.arange(ny, dtype=np.float64) * dy
    x_mm = x0 + np.arange(nx, dtype=np.float64) * dx

    mu_vol = np.zeros((nz, ny, nx), dtype=np.float32)
    yy_2d, xx_2d = np.meshgrid(y_mm, x_mm, indexing="ij")  # (ny, nx)

    for ell in ellipsoids:
        cx, cy, cz = ell.center_xyz_mm
        ax, ay, az = ell.semi_axes_xyz_mm
        dmu = np.float32(ell.delta_mu_mm_inv(water_mu_mm_inv))

        # Quick z bounding-box culling
        z_norm_sq = ((z_mm - cz) / az) ** 2
        active_z = np.where(z_norm_sq <= 1.0)[0]
        if active_z.size == 0:
            continue

        px = xx_2d - cx
        py = yy_2d - cy
        if ell.yaw_rad != 0.0:
            c_yaw = math.cos(ell.yaw_rad)
            s_yaw = math.sin(ell.yaw_rad)
            rx = px * c_yaw + py * s_yaw
            ry = -px * s_yaw + py * c_yaw
        else:
            rx, ry = px, py

        xy_norm_sq = (rx / ax) ** 2 + (ry / ay) ** 2  # (ny, nx)
        for kz in active_z:
            mask_2d = (xy_norm_sq + z_norm_sq[kz]) <= 1.0
            mu_vol[kz, mask_2d] += dmu

    mu_vol = np.maximum(mu_vol, np.float32(0.0))
    hu_vol = mu_to_hu(mu_vol, water_mu_mm_inv=water_mu_mm_inv)
    return {
        "mu_volume": mu_vol,
        "hu_volume": hu_vol,
        "z_mm": z_mm,
        "y_mm": y_mm,
        "x_mm": x_mm,
    }


def voxelize_phantom_slices_radiological(
    ellipsoids: list[Ellipsoid3D],
    *,
    z_slices_mm: np.ndarray | list[float],
    image_size: int = 256,
    fov_mm: float = 360.0,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    water_mu_mm_inv: float = 0.0192,
) -> dict[str, np.ndarray]:
    """Evaluate exact 3D phantom slices on radiological (Row 0 = +y Anterior, Col 0 = -x Right) grid."""
    n_xy = int(image_size)
    if n_xy < 2:
        raise ValueError(f"image_size must be >= 2, got {image_size}")
    if float(fov_mm) <= 0.0:
        raise ValueError(f"fov_mm must be positive, got {fov_mm}")

    z_arr = np.asarray(z_slices_mm, dtype=np.float64).reshape(-1)
    d_xy = float(fov_mm) / float(n_xy)
    half_idx = 0.5 * (n_xy - 1.0)
    x_mm = float(center_xy_mm[0]) + (np.arange(n_xy, dtype=np.float64) - half_idx) * d_xy
    y_mm = float(center_xy_mm[1]) + (half_idx - np.arange(n_xy, dtype=np.float64)) * d_xy

    mu_slices = np.zeros((z_arr.size, n_xy, n_xy), dtype=np.float32)
    yy_2d, xx_2d = np.meshgrid(y_mm, x_mm, indexing="ij")

    for ell in ellipsoids:
        cx, cy, cz = ell.center_xyz_mm
        ax, ay, az = ell.semi_axes_xyz_mm
        dmu = np.float32(ell.delta_mu_mm_inv(water_mu_mm_inv))
        z_norm_sq = ((z_arr - cz) / az) ** 2
        active_z = np.where(z_norm_sq <= 1.0)[0]
        if active_z.size == 0:
            continue

        px = xx_2d - cx
        py = yy_2d - cy
        if ell.yaw_rad != 0.0:
            c_yaw = math.cos(ell.yaw_rad)
            s_yaw = math.sin(ell.yaw_rad)
            rx = px * c_yaw + py * s_yaw
            ry = -px * s_yaw + py * c_yaw
        else:
            rx, ry = px, py

        xy_norm_sq = (rx / ax) ** 2 + (ry / ay) ** 2
        for kz in active_z:
            mask_2d = (xy_norm_sq + z_norm_sq[kz]) <= 1.0
            mu_slices[kz, mask_2d] += dmu

    mu_slices = np.maximum(mu_slices, np.float32(0.0))
    hu_slices = mu_to_hu(mu_slices, water_mu_mm_inv=water_mu_mm_inv)
    return {
        "mu_slices": mu_slices,
        "hu_slices": hu_slices,
        "z_mm": z_arr,
        "y_mm": y_mm,
        "x_mm": x_mm,
        "pixel_spacing_mm": np.array(d_xy, dtype=np.float64),
    }


def _ray_ellipsoid_chord_lengths(
    origins_xyz: np.ndarray,
    unit_dirs_xyz: np.ndarray,
    max_lengths_mm: np.ndarray,
    ell: Ellipsoid3D,
) -> np.ndarray:
    """Exact closed-form chord lengths [mm] of 3D ray segments intersecting an ellipsoid.

    origins_xyz:    (B, 3)
    unit_dirs_xyz:  (B, R, C, 3)
    max_lengths_mm: (B, R, C)
    Returns float64 chord lengths of shape (B, R, C).
    """
    cx, cy, cz = ell.center_xyz_mm
    ax, ay, az = ell.semi_axes_xyz_mm

    p0x = origins_xyz[:, 0] - cx  # (B,)
    p0y = origins_xyz[:, 1] - cy
    p0z = origins_xyz[:, 2] - cz

    rx = unit_dirs_xyz[..., 0]  # (B, R, C)
    ry = unit_dirs_xyz[..., 1]
    rz = unit_dirs_xyz[..., 2]

    if ell.yaw_rad != 0.0:
        c_yaw = math.cos(ell.yaw_rad)
        s_yaw = math.sin(ell.yaw_rad)
        p0x, p0y = (p0x * c_yaw + p0y * s_yaw, -p0x * s_yaw + p0y * c_yaw)
        rx, ry = (rx * c_yaw + ry * s_yaw, -rx * s_yaw + ry * c_yaw)

    px_s = (p0x / ax)[:, None, None]
    py_s = (p0y / ay)[:, None, None]
    pz_s = (p0z / az)[:, None, None]

    rx_s = rx / ax
    ry_s = ry / ay
    rz_s = rz / az

    a_quad = rx_s * rx_s + ry_s * ry_s + rz_s * rz_s  # (B, R, C)
    b_half = px_s * rx_s + py_s * ry_s + pz_s * rz_s  # (B, R, C)
    c_quad = (px_s * px_s + py_s * py_s + pz_s * pz_s) - 1.0  # (B, 1, 1)

    disc = b_half * b_half - a_quad * c_quad
    hit = disc > 0.0
    if not np.any(hit):
        return np.zeros_like(max_lengths_mm, dtype=np.float64)

    chords = np.zeros_like(max_lengths_mm, dtype=np.float64)
    sqrt_disc = np.sqrt(disc[hit])
    a_hit = a_quad[hit]
    b_hit = b_half[hit]
    t_enter = np.maximum(0.0, (-b_hit - sqrt_disc) / a_hit)
    t_exit = np.minimum(max_lengths_mm[hit], (-b_hit + sqrt_disc) / a_hit)
    chords[hit] = np.maximum(0.0, t_exit - t_enter)
    return chords


def project_phantom_helical(
    ellipsoids: list[Ellipsoid3D],
    geometry: HelicalGeometry,
    *,
    view_indices: np.ndarray | list[int] | slice | None = None,
    batch_views: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute exact analytical 3D helical cone-beam line integrals [V, n_rows, n_cols].

    Evaluates int_{S(i)}^{D(r,c;i)} mu(x) dl in float64 per batch and stores float32,
    keeping peak RAM under ~50 MiB even for 13,000+ views.
    """
    if view_indices is None:
        all_idx = np.arange(geometry.n_views, dtype=np.int64)
    elif isinstance(view_indices, slice):
        all_idx = np.arange(geometry.n_views, dtype=np.int64)[view_indices]
    else:
        all_idx = np.asarray(view_indices, dtype=np.int64)

    n_out = int(all_idx.size)
    out = np.zeros((n_out, geometry.n_rows, geometry.n_cols), dtype=np.float32)
    if n_out == 0 or not ellipsoids:
        return out

    dmus = [ell.delta_mu_mm_inv(geometry.water_mu_mm_inv) for ell in ellipsoids]
    step = max(1, int(batch_views))

    with ProgressReporter(
        total=n_out,
        desc="☢️ Analytical helical projection",
        unit="view",
        unit_scale=False,
        colour="magenta",
        enabled=show_progress,
    ) as bar:
        for start in range(0, n_out, step):
            stop = min(n_out, start + step)
            batch_idx = all_idx[start:stop]
            origins, unit_dirs, lengths = geometry.ray_origins_and_directions(batch_idx)
            acc = np.zeros((stop - start, geometry.n_rows, geometry.n_cols), dtype=np.float64)
            for ell, dmu in zip(ellipsoids, dmus):
                if dmu == 0.0:
                    continue
                acc += dmu * _ray_ellipsoid_chord_lengths(origins, unit_dirs, lengths, ell)
            out[start:stop] = np.maximum(acc, 0.0).astype(np.float32)
            bar.update(stop - start)

    return out


def make_synthetic_helical_geometry(
    *,
    n_views: int = 384,
    n_rows: int = 32,
    n_cols: int = 128,
    angular_steps_per_rotation: int = 192,
    normalized_pitch: float = 0.85,
    detector_shape: str = "CYLINDRICAL",
    ffs_mode: str = "FFSZ",
    rotation_direction: int = -1,
    quarter_detector_offset_cols: float = 1.125,
    du_mm: float = 4.0,
    dv_mm: float = 2.0,
    rho0_mm: float = 595.0,
    d0_mm: float = 1085.6,
    z_center_mm: float = 0.0,
    phi_start_rad: float = 0.4,
    series_instance_uid: str = "1.2.826.0.1.3680043.10.999.1001",
) -> HelicalGeometry:
    """Create a realistic synthetic HelicalGeometry (including Siemens-like FFSZ & rot_dir=-1)."""
    if rotation_direction not in (-1, 1):
        raise ValueError("rotation_direction must be +1 or -1")

    w_iso = float(n_rows) * float(dv_mm) * (float(rho0_mm) / float(d0_mm))
    feed_mm_per_rot = float(normalized_pitch) * w_iso
    dphi_step = float(rotation_direction) * (2.0 * math.pi / float(angular_steps_per_rotation))
    dz_step = feed_mm_per_rot / float(angular_steps_per_rotation)

    idx = np.arange(n_views, dtype=np.float64)
    phi0_unwrap = float(phi_start_rad) + idx * dphi_step
    phi0_wrapped = np.mod(phi0_unwrap, 2.0 * math.pi)
    z0_start = float(z_center_mm) - 0.5 * (n_views - 1) * dz_step
    z0 = z0_start + idx * dz_step

    ffs_norm = ffs_mode.strip().upper()
    if ffs_norm == "FFSZ":
        parity = (np.arange(n_views) % 2).astype(np.float64)
        drho = parity * 5.45
        dz = parity * (-0.66)
        dphi = np.where(parity == 0.0, 0.000336, 0.000333)
    elif ffs_norm == "FFSNONE":
        drho = np.zeros(n_views, dtype=np.float64)
        dz = np.zeros(n_views, dtype=np.float64)
        dphi = np.zeros(n_views, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported synthetic ffs_mode={ffs_mode!r}; use FFSZ or FFSNONE")

    center_col_0 = 0.5 * (n_cols - 1.0) + float(quarter_detector_offset_cols)
    center_row_0 = 0.5 * (n_rows - 1.0)

    return HelicalGeometry(
        series_instance_uid=series_instance_uid,
        detector_shape=detector_shape,
        n_rows=n_rows,
        n_cols=n_cols,
        du_mm=du_mm,
        dv_mm=dv_mm,
        rho0_mm=rho0_mm,
        d0_mm=d0_mm,
        center_col_0based=center_col_0,
        center_row_0based=center_row_0,
        ffs_mode=ffs_norm,
        angular_steps_per_rotation=angular_steps_per_rotation,
        water_mu_mm_inv=0.0192,
        nominal_spiral_pitch=normalized_pitch,
        log_flag="YES",
        rescale_slope=0.0001,
        rescale_intercept=-0.05,
        instance_numbers=np.arange(1, n_views + 1, dtype=np.int32),
        phi0_rad=phi0_wrapped,
        phi0_unwrapped_rad=unwrap_angles_rad(phi0_wrapped),
        z0_mm=z0,
        dphi_rad=dphi,
        dz_mm=dz,
        drho_mm=drho,
    )


def _pad_dicom_ascii(text: str) -> bytes:
    raw = text.encode("ascii")
    if len(raw) % 2 == 1:
        raw += b" "
    return raw


def write_synthetic_ctpd_series(
    out_dir: str | Path,
    geometry: HelicalGeometry,
    projections: np.ndarray,
    *,
    patient_id: str = "SYNTH_THORAX_01",
    study_instance_uid: str = "1.2.826.0.1.3680043.10.999.1000",
    series_description: str = "Synthetic Full dose projections",
) -> list[Path]:
    """Write a complete synthetic DICOM-CT-PD v4 series matching Mayo/TCIA byte conventions.

    Uses Implicit VR Little Endian (1.2.840.10008.1.2), VR='UN' private binary tags,
    transposed (Rows=n_cols, Columns=n_rows), and Fortran-order uint16 PixelData.
    """
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
        from pydicom.uid import ImplicitVRLittleEndian
    except ImportError as error:
        raise RuntimeError("pydicom is required to write synthetic DICOM-CT-PD files") from error

    proj_arr = np.asarray(projections, dtype=np.float32)
    expected_shape = (geometry.n_views, geometry.n_rows, geometry.n_cols)
    if proj_arr.shape != expected_shape:
        raise ValueError(f"projections shape {proj_arr.shape} != expected {expected_shape}")

    target_dir = Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    g_min = float(np.min(proj_arr))
    g_max = float(np.max(proj_arr))
    intercept = min(0.0, g_min) - 0.01
    span = max(g_max - intercept, 0.5)
    slope = span / 65000.0

    c_x_1based, r_y_1based = geometry.detector_central_element_1based
    static_un_tags: dict[tuple[int, int], bytes] = {
        CTPD_PRIVATE_TAGS["detector_rows"][:2]: struct.pack("<H", geometry.n_rows),
        CTPD_PRIVATE_TAGS["detector_cols"][:2]: struct.pack("<H", geometry.n_cols),
        CTPD_PRIVATE_TAGS["du_mm"][:2]: struct.pack("<f", geometry.du_mm),
        CTPD_PRIVATE_TAGS["dv_mm"][:2]: struct.pack("<f", geometry.dv_mm),
        CTPD_PRIVATE_TAGS["detector_shape"][:2]: _pad_dicom_ascii(geometry.detector_shape),
        CTPD_PRIVATE_TAGS["rho0_mm"][:2]: struct.pack("<f", geometry.rho0_mm),
        CTPD_PRIVATE_TAGS["d0_mm"][:2]: struct.pack("<f", geometry.d0_mm),
        CTPD_PRIVATE_TAGS["detector_central_element_1based"][:2]: struct.pack(
            "<2f", c_x_1based, r_y_1based
        ),
        CTPD_PRIVATE_TAGS["ffs_mode"][:2]: _pad_dicom_ascii(geometry.ffs_mode),
        CTPD_PRIVATE_TAGS["angular_steps_per_rotation"][:2]: struct.pack(
            "<H", geometry.angular_steps_per_rotation
        ),
        CTPD_PRIVATE_TAGS["projection_data_type"][:2]: _pad_dicom_ascii("HELICAL"),
        CTPD_PRIVATE_TAGS["projection_geometry_type"][:2]: _pad_dicom_ascii("FANBEAM"),
        CTPD_PRIVATE_TAGS["log_flag"][:2]: _pad_dicom_ascii(geometry.log_flag),
        CTPD_PRIVATE_TAGS["water_attenuation_mm_inv"][:2]: _pad_dicom_ascii(
            f"{geometry.water_mu_mm_inv:.6f}"
        ),
    }

    written: list[Path] = []
    for k in range(geometry.n_views):
        inst_num = int(geometry.instance_numbers[k])
        sop_uid = f"{geometry.series_instance_uid}.{inst_num}"

        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationVersion = b"\x00\x01"
        file_meta.MediaStorageSOPClassUID = RAW_DATA_SOP_CLASS_UID
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.999.1"

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)

        ds.SOPClassUID = RAW_DATA_SOP_CLASS_UID
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = geometry.series_instance_uid
        ds.Modality = "CT"
        ds.Manufacturer = "SIEMENS"
        ds.BodyPartExamined = "CHEST"
        ds.SeriesDescription = series_description
        ds.PatientID = patient_id
        ds.InstanceNumber = inst_num

        # Mayo MATLAB dicomwrite transposed convention: Rows = n_cols, Columns = n_rows
        ds.Rows = int(geometry.n_cols)
        ds.Columns = int(geometry.n_rows)
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = f"{slope:.8e}"
        ds.RescaleIntercept = f"{intercept:.8e}"
        if geometry.nominal_spiral_pitch is not None:
            ds.add_new((0x0018, 0x9311), "FD", float(geometry.nominal_spiral_pitch))

        # Private creator tags matching Mayo Clinic v1
        for grp in (0x7029, 0x7031, 0x7033, 0x7037, 0x7039, 0x7041):
            ds.add_new((grp, 0x0010), "LO", "CtProjectionData-MayoClinc-v1")

        for tag_pair, raw_val in static_un_tags.items():
            ds.add_new(tag_pair, "UN", raw_val)

        ds.add_new(CTPD_PRIVATE_TAGS["phi0_rad"][:2], "UN", struct.pack("<f", geometry.phi0_rad[k]))
        ds.add_new(CTPD_PRIVATE_TAGS["z0_mm"][:2], "UN", struct.pack("<f", geometry.z0_mm[k]))
        ds.add_new(CTPD_PRIVATE_TAGS["dphi_rad"][:2], "UN", struct.pack("<f", geometry.dphi_rad[k]))
        ds.add_new(CTPD_PRIVATE_TAGS["dz_mm"][:2], "UN", struct.pack("<f", geometry.dz_mm[k]))
        ds.add_new(CTPD_PRIVATE_TAGS["drho_mm"][:2], "UN", struct.pack("<f", geometry.drho_mm[k]))

        quant = np.clip(np.rint((proj_arr[k] - intercept) / slope), 0.0, 65535.0).astype("<u2")
        # Write column-major (Fortran order) bytes == quant.T in C-order
        ds.PixelData = quant.tobytes(order="F")

        out_path = target_dir / f"1-{inst_num:05d}.dcm"
        try:
            pydicom.dcmwrite(out_path, ds, enforce_file_format=True)
        except TypeError:
            ds.is_little_endian = True
            ds.is_implicit_VR = True
            pydicom.dcmwrite(out_path, ds, write_like_original=False)
        written.append(out_path)

    return written
