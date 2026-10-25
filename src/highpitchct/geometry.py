"""Explicit 3D Helical & Detector Geometry Model for DICOM-CT-PD v4.

Coordinate System (DICOM-CT-PD User Manual Version 4, Pages 3-5, Figures 2, 3, 5):
- Patient Cartesian coordinates (x, y, z) form a left-handed system:
    x = -rho * sin(phi)
    y = +rho * cos(phi)
    z =  z
  where phi = 0 is at 12 o'clock (0, +rho), phi increases counter-clockwise
  towards -x (9 o'clock is phi = +pi/2 at (-rho, 0)), and +z points horizontally
  from the table base into the gantry.
- Detector Focal Center at view i:
    F0(i) = (-rho0 * sin(phi0_i), +rho0 * cos(phi0_i), z0_i)
- True X-ray Focal Spot Position with 3D Flying Focal Spot (FFS) shifts:
    rho_s(i) = rho0 + drho_i
    phi_s(i) = phi0_i + dphi_i
    z_s(i)   = z0_i + dz_i
    S(i)     = (-rho_s(i) * sin(phi_s(i)), +rho_s(i) * cos(phi_s(i)), z_s(i))
- Detector 0-based central element converted from 1-based tag (7031,1033) [Col X, Row Y]:
    center_col_0based (c0) = X_1based - 1.0
    center_row_0based (r0) = Y_1based - 1.0
- Detector element indexing on reshaped projection array [row_idx, col_idx] of shape (N_rows, N_cols):
    * Row 0 (1-based Row 1) lies in the +z direction; Row N_rows-1 lies in the -z direction:
          v(r) = (r0 - r) * dv_mm
    * Column 0 (1-based Col 1) lies at -x when phi0 = 0; Column N_cols-1 lies at +x when phi0 = 0:
          u(c) = (c - c0) * du_mm
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_DETECTOR_SHAPES = frozenset({"CYLINDRICAL", "FLAT", "SPHERICAL"})


def unwrap_angles_rad(angles_rad: np.ndarray | list[float]) -> np.ndarray:
    """Unwrap azimuthal angles [rad] continuously in float64."""
    arr = np.asarray(angles_rad, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"angles_rad must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        return arr.copy()
    return np.unwrap(arr)


def fit_helical_trajectory(
    phi0_unwrapped_rad: np.ndarray,
    z0_mm: np.ndarray,
    *,
    n_rows: int,
    dv_mm: float,
    rho0_mm: float,
    d0_mm: float,
) -> dict[str, float | int]:
    """Fit z0(phi0_unwrapped) linearly to estimate dynamic table feed and normalized pitch."""
    phi = np.asarray(phi0_unwrapped_rad, dtype=np.float64)
    z = np.asarray(z0_mm, dtype=np.float64)
    if phi.shape != z.shape or phi.ndim != 1:
        raise ValueError("phi0_unwrapped_rad and z0_mm must be 1D arrays of equal length")
    if phi.size == 0:
        raise ValueError("Cannot fit helical trajectory on empty view arrays")

    w_iso = float(n_rows) * float(dv_mm) * (float(rho0_mm) / float(d0_mm)) if d0_mm > 0 else 0.0
    if phi.size < 2:
        return {
            "slope_mm_per_rad": 0.0,
            "intercept_mm": float(z[0]),
            "table_feed_mm_per_rot": 0.0,
            "coverage_isocenter_mm": w_iso,
            "normalized_pitch": 0.0,
            "pitch_fit_rmse_mm": 0.0,
            "axial_span_mm": 0.0,
            "total_rotations": 0.0,
            "table_direction": 0,
            "rotation_direction": 0,
        }

    mean_phi = float(np.mean(phi))
    mean_z = float(np.mean(z))
    dphi = phi - mean_phi
    sxx = float(np.dot(dphi, dphi))
    if sxx <= 1e-18:
        slope = 0.0
        intercept = mean_z
        rmse = 0.0
    else:
        slope = float(np.dot(dphi, z - mean_z) / sxx)
        intercept = mean_z - slope * mean_phi
        residuals = z - (slope * phi + intercept)
        rmse = float(math.sqrt(float(np.mean(residuals * residuals))))

    feed_mm_per_rot = abs(slope) * (2.0 * math.pi)
    norm_pitch = (feed_mm_per_rot / w_iso) if w_iso > 0.0 else 0.0
    total_rot = abs(float(phi[-1] - phi[0])) / (2.0 * math.pi)
    table_dir = 1 if z[-1] > z[0] else (-1 if z[-1] < z[0] else 0)
    rot_dir = 1 if phi[-1] > phi[0] else (-1 if phi[-1] < phi[0] else 0)

    return {
        "slope_mm_per_rad": slope,
        "intercept_mm": intercept,
        "table_feed_mm_per_rot": feed_mm_per_rot,
        "coverage_isocenter_mm": w_iso,
        "normalized_pitch": norm_pitch,
        "pitch_fit_rmse_mm": rmse,
        "axial_span_mm": float(np.max(z) - np.min(z)),
        "total_rotations": total_rot,
        "table_direction": table_dir,
        "rotation_direction": rot_dir,
    }


@dataclass(frozen=True)
class HelicalGeometry:
    """Complete, explicit 3D helical acquisition & detector geometry for one CT-PD series."""

    series_instance_uid: str
    detector_shape: str
    n_rows: int
    n_cols: int
    du_mm: float
    dv_mm: float
    rho0_mm: float
    d0_mm: float
    center_col_0based: float
    center_row_0based: float
    ffs_mode: str
    angular_steps_per_rotation: int
    water_mu_mm_inv: float
    nominal_spiral_pitch: float | None
    log_flag: str
    rescale_slope: float
    rescale_intercept: float
    instance_numbers: np.ndarray
    phi0_rad: np.ndarray
    phi0_unwrapped_rad: np.ndarray
    z0_mm: np.ndarray
    dphi_rad: np.ndarray
    dz_mm: np.ndarray
    drho_mm: np.ndarray

    def __post_init__(self) -> None:
        shape_norm = str(self.detector_shape).strip().upper()
        if shape_norm not in SUPPORTED_DETECTOR_SHAPES:
            raise ValueError(
                f"Unsupported detector_shape={self.detector_shape!r}; "
                f"expected one of {sorted(SUPPORTED_DETECTOR_SHAPES)}"
            )
        object.__setattr__(self, "detector_shape", shape_norm)
        object.__setattr__(self, "ffs_mode", str(self.ffs_mode).strip().upper())
        object.__setattr__(self, "log_flag", str(self.log_flag).strip().upper())

        if self.n_rows < 1 or self.n_cols < 1:
            raise ValueError(f"Invalid detector dimensions: ({self.n_rows}, {self.n_cols})")
        if self.du_mm <= 0.0 or self.dv_mm <= 0.0:
            raise ValueError(f"Detector spacings must be positive: du={self.du_mm}, dv={self.dv_mm}")
        if self.rho0_mm <= 0.0 or self.d0_mm <= self.rho0_mm:
            raise ValueError(
                f"Require 0 < rho0_mm < d0_mm, got rho0_mm={self.rho0_mm}, d0_mm={self.d0_mm}"
            )
        if self.water_mu_mm_inv <= 0.0:
            raise ValueError(f"water_mu_mm_inv must be positive, got {self.water_mu_mm_inv}")

        inst = np.asarray(self.instance_numbers, dtype=np.int32)
        object.__setattr__(self, "instance_numbers", inst)
        n = inst.shape[0]
        if inst.ndim != 1 or n < 1:
            raise ValueError("HelicalGeometry requires at least 1 projection view")

        for field_name in (
            "phi0_rad",
            "phi0_unwrapped_rad",
            "z0_mm",
            "dphi_rad",
            "dz_mm",
            "drho_mm",
        ):
            arr = np.asarray(getattr(self, field_name), dtype=np.float64)
            if arr.shape != (n,):
                raise ValueError(
                    f"Field {field_name} has shape {arr.shape}, expected ({n},)"
                )
            object.__setattr__(self, field_name, arr)

    @property
    def n_views(self) -> int:
        return int(self.phi0_rad.shape[0])

    @property
    def detector_central_element_1based(self) -> tuple[float, float]:
        """Return 1-based (Column X, Row Y) matching DICOM-CT-PD tag (7031,1033)."""
        return (float(self.center_col_0based) + 1.0, float(self.center_row_0based) + 1.0)

    @property
    def coverage_isocenter_mm(self) -> float:
        return float(self.n_rows) * float(self.dv_mm) * (float(self.rho0_mm) / float(self.d0_mm))

    def trajectory_metrics(self) -> dict[str, float | int]:
        return fit_helical_trajectory(
            self.phi0_unwrapped_rad,
            self.z0_mm,
            n_rows=self.n_rows,
            dv_mm=self.dv_mm,
            rho0_mm=self.rho0_mm,
            d0_mm=self.d0_mm,
        )

    def with_modified_centers(
        self,
        *,
        center_col_0based: float | None = None,
        center_row_0based: float | None = None,
    ) -> HelicalGeometry:
        return replace(
            self,
            center_col_0based=(
                float(center_col_0based)
                if center_col_0based is not None
                else self.center_col_0based
            ),
            center_row_0based=(
                float(center_row_0based)
                if center_row_0based is not None
                else self.center_row_0based
            ),
        )

    def slice_views(self, start: int = 0, stop: int | None = None, step: int = 1) -> HelicalGeometry:
        """Return a new HelicalGeometry restricted to views [start:stop:step]."""
        sl = slice(start, stop, step)
        return replace(
            self,
            instance_numbers=self.instance_numbers[sl].copy(),
            phi0_rad=self.phi0_rad[sl].copy(),
            phi0_unwrapped_rad=self.phi0_unwrapped_rad[sl].copy(),
            z0_mm=self.z0_mm[sl].copy(),
            dphi_rad=self.dphi_rad[sl].copy(),
            dz_mm=self.dz_mm[sl].copy(),
            drho_mm=self.drho_mm[sl].copy(),
        )

    def _normalize_view_indices(
        self, view_indices: int | slice | np.ndarray | list[int] | None
    ) -> tuple[np.ndarray, bool]:
        if view_indices is None:
            return np.arange(self.n_views, dtype=np.int64), False
        if isinstance(view_indices, (int, np.integer)):
            idx = int(view_indices)
            if idx < 0:
                idx += self.n_views
            if idx < 0 or idx >= self.n_views:
                raise IndexError(f"view_index {view_indices} out of bounds for {self.n_views} views")
            return np.array([idx], dtype=np.int64), True
        if isinstance(view_indices, slice):
            return np.arange(self.n_views, dtype=np.int64)[view_indices], False
        arr = np.asarray(view_indices, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"view_indices must be scalar, slice, or 1D, got shape {arr.shape}")
        return arr, False

    def column_transverse_offsets_mm(self, *, flip_cols: bool = False) -> np.ndarray:
        """Signed transverse detector coordinate u(c) [mm] for c = 0 .. n_cols-1."""
        cols = np.arange(self.n_cols, dtype=np.float64)
        if flip_cols:
            cols = (self.n_cols - 1.0) - cols
        return (cols - self.center_col_0based) * self.du_mm

    def column_fan_angles_rad(self, *, flip_cols: bool = False) -> np.ndarray:
        """Signed fan angle gamma(c) [rad] relative to detector focal center F0."""
        u_mm = self.column_transverse_offsets_mm(flip_cols=flip_cols)
        if self.detector_shape in {"CYLINDRICAL", "SPHERICAL"}:
            return u_mm / self.d0_mm
        if self.detector_shape == "FLAT":
            return np.arctan2(u_mm, self.d0_mm)
        raise ValueError(f"Unsupported detector_shape={self.detector_shape!r}")

    def row_axial_offsets_mm(self, *, flip_rows: bool = False) -> np.ndarray:
        """Signed axial detector offset v(r) [mm] relative to z0.

        Per DICOM-CT-PD Manual Figure 2, Row 1 (0-based r=0) is at +z and
        Row N (0-based r=n_rows-1) is at -z, so v(r) = (r0 - r) * dv_mm.
        """
        rows = np.arange(self.n_rows, dtype=np.float64)
        if flip_rows:
            rows = (self.n_rows - 1.0) - rows
        return (self.center_row_0based - rows) * self.dv_mm

    def focal_center_cartesian_mm(
        self, view_indices: int | slice | np.ndarray | list[int] | None = None
    ) -> np.ndarray:
        """Return detector focal center F0 = (-rho0*sin(phi0), +rho0*cos(phi0), z0) [mm]."""
        idx, scalar = self._normalize_view_indices(view_indices)
        phi0 = self.phi0_rad[idx]
        z0 = self.z0_mm[idx]
        pts = np.stack(
            [-self.rho0_mm * np.sin(phi0), self.rho0_mm * np.cos(phi0), z0],
            axis=-1,
        )
        return pts[0] if scalar else pts

    def source_cylindrical(
        self, view_indices: int | slice | np.ndarray | list[int] | None = None
    ) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        """Return actual FFS-shifted source cylindrical coordinates (rho_s, phi_s_unwrapped, z_s)."""
        idx, scalar = self._normalize_view_indices(view_indices)
        rho_s = self.rho0_mm + self.drho_mm[idx]
        phi_s = self.phi0_unwrapped_rad[idx] + self.dphi_rad[idx]
        z_s = self.z0_mm[idx] + self.dz_mm[idx]
        if scalar:
            return float(rho_s[0]), float(phi_s[0]), float(z_s[0])
        return rho_s, phi_s, z_s

    def source_cartesian_mm(
        self, view_indices: int | slice | np.ndarray | list[int] | None = None
    ) -> np.ndarray:
        """Return actual X-ray focal spot S(i) in patient Cartesian (x, y, z) [mm]."""
        idx, scalar = self._normalize_view_indices(view_indices)
        rho_s = self.rho0_mm + self.drho_mm[idx]
        phi_s = self.phi0_rad[idx] + self.dphi_rad[idx]
        z_s = self.z0_mm[idx] + self.dz_mm[idx]
        pts = np.stack(
            [-rho_s * np.sin(phi_s), rho_s * np.cos(phi_s), z_s],
            axis=-1,
        )
        return pts[0] if scalar else pts

    @staticmethod
    def detector_basis_vectors(
        phi0_rad: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return orthonormal detector basis (e_ray, e_u, e_z) at angle(s) phi0_rad."""
        phi = np.asarray(phi0_rad, dtype=np.float64)
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)
        zeros = np.zeros_like(phi)
        ones = np.ones_like(phi)
        e_ray = np.stack([sin_p, -cos_p, zeros], axis=-1)
        e_u = np.stack([cos_p, sin_p, zeros], axis=-1)
        e_z = np.stack([zeros, zeros, ones], axis=-1)
        return e_ray, e_u, e_z

    def detector_element_cartesian_mm(
        self,
        view_indices: int | slice | np.ndarray | list[int] | None = None,
        *,
        flip_rows: bool = False,
        flip_cols: bool = False,
    ) -> np.ndarray:
        """Compute exact 3D Cartesian coordinates D(view, row, col, xyz) [mm] of detector centers."""
        idx, scalar = self._normalize_view_indices(view_indices)
        phi0 = self.phi0_rad[idx]  # (K,)
        f0 = self.focal_center_cartesian_mm(idx)  # (K, 3)
        e_ray, e_u, e_z = self.detector_basis_vectors(phi0)  # each (K, 3)

        v_r = self.row_axial_offsets_mm(flip_rows=flip_rows)  # (R,)
        u_c = self.column_transverse_offsets_mm(flip_cols=flip_cols)  # (C,)

        if self.detector_shape == "CYLINDRICAL":
            gamma_c = u_c / self.d0_mm  # (C,)
            a_rc = np.broadcast_to((self.d0_mm * np.cos(gamma_c))[None, :], (self.n_rows, self.n_cols))
            b_rc = np.broadcast_to((self.d0_mm * np.sin(gamma_c))[None, :], (self.n_rows, self.n_cols))
            c_rc = np.broadcast_to(v_r[:, None], (self.n_rows, self.n_cols))
        elif self.detector_shape == "FLAT":
            a_rc = np.full((self.n_rows, self.n_cols), self.d0_mm, dtype=np.float64)
            b_rc = np.broadcast_to(u_c[None, :], (self.n_rows, self.n_cols))
            c_rc = np.broadcast_to(v_r[:, None], (self.n_rows, self.n_cols))
        elif self.detector_shape == "SPHERICAL":
            gamma_c = u_c / self.d0_mm  # (C,)
            alpha_r = v_r / self.d0_mm  # (R,)
            cos_a = np.cos(alpha_r)[:, None]
            sin_a = np.sin(alpha_r)[:, None]
            a_rc = self.d0_mm * cos_a * np.cos(gamma_c)[None, :]
            b_rc = self.d0_mm * cos_a * np.sin(gamma_c)[None, :]
            c_rc = np.broadcast_to((self.d0_mm * sin_a), (self.n_rows, self.n_cols))
        else:
            raise ValueError(f"Unsupported detector_shape={self.detector_shape!r}")

        # Combine (K, 1, 1, 3) with (1, R, C, 1)
        det = (
            f0[:, None, None, :]
            + a_rc[None, :, :, None] * e_ray[:, None, None, :]
            + b_rc[None, :, :, None] * e_u[:, None, None, :]
            + c_rc[None, :, :, None] * e_z[:, None, None, :]
        )
        return det[0] if scalar else det

    def ray_origins_and_directions(
        self,
        view_indices: int | slice | np.ndarray | list[int] | None = None,
        *,
        flip_rows: bool = False,
        flip_cols: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (source_xyz, unit_ray_dirs, ray_lengths_mm) for requested view(s).

        Shapes:
          - If view_indices is scalar int: ((3,), (R, C, 3), (R, C))
          - Otherwise: ((K, 3), (K, R, C, 3), (K, R, C))
        """
        idx, scalar = self._normalize_view_indices(view_indices)
        src = self.source_cartesian_mm(idx)  # (K, 3)
        det = self.detector_element_cartesian_mm(
            idx, flip_rows=flip_rows, flip_cols=flip_cols
        )  # (K, R, C, 3)
        vec = det - src[:, None, None, :]
        lengths = np.linalg.norm(vec, axis=-1)
        unit_dirs = vec / np.maximum(lengths[..., None], 1e-12)
        if scalar:
            return src[0], unit_dirs[0], lengths[0]
        return src, unit_dirs, lengths

    def summary_dict(self) -> dict[str, Any]:
        traj = self.trajectory_metrics()
        ffs_unique_states = int(
            np.unique(
                np.round(
                    np.stack([self.drho_mm, self.dphi_rad, self.dz_mm], axis=-1),
                    decimals=4,
                ),
                axis=0,
            ).shape[0]
        )
        return {
            "series_instance_uid": self.series_instance_uid,
            "detector_shape": self.detector_shape,
            "n_views": self.n_views,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "du_mm": round(self.du_mm, 6),
            "dv_mm": round(self.dv_mm, 6),
            "rho0_mm": round(self.rho0_mm, 4),
            "d0_mm": round(self.d0_mm, 4),
            "center_col_0based": round(self.center_col_0based, 6),
            "center_row_0based": round(self.center_row_0based, 6),
            "detector_central_element_1based": [
                round(x, 6) for x in self.detector_central_element_1based
            ],
            "ffs_mode": self.ffs_mode,
            "ffs_unique_states": ffs_unique_states,
            "drho_range_mm": [
                round(float(np.min(self.drho_mm)), 5),
                round(float(np.max(self.drho_mm)), 5),
            ],
            "dphi_range_rad": [
                round(float(np.min(self.dphi_rad)), 7),
                round(float(np.max(self.dphi_rad)), 7),
            ],
            "dz_range_mm": [
                round(float(np.min(self.dz_mm)), 5),
                round(float(np.max(self.dz_mm)), 5),
            ],
            "angular_steps_per_rotation": self.angular_steps_per_rotation,
            "water_mu_mm_inv": self.water_mu_mm_inv,
            "nominal_spiral_pitch": self.nominal_spiral_pitch,
            "log_flag": self.log_flag,
            "z0_range_mm": [
                round(float(np.min(self.z0_mm)), 4),
                round(float(np.max(self.z0_mm)), 4),
            ],
            "axial_span_mm": round(float(traj["axial_span_mm"]), 4),
            "total_rotations": round(float(traj["total_rotations"]), 4),
            "table_feed_mm_per_rot": round(float(traj["table_feed_mm_per_rot"]), 4),
            "coverage_isocenter_mm": round(float(traj["coverage_isocenter_mm"]), 4),
            "estimated_normalized_pitch": round(float(traj["normalized_pitch"]), 5),
            "pitch_fit_rmse_mm": round(float(traj["pitch_fit_rmse_mm"]), 5),
            "table_direction": int(traj["table_direction"]),
            "rotation_direction": int(traj["rotation_direction"]),
        }

    def to_npz(self, path: str | Path) -> Path:
        """Serialize HelicalGeometry deterministically to a compressed .npz archive."""
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "series_instance_uid": self.series_instance_uid,
            "detector_shape": self.detector_shape,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "du_mm": self.du_mm,
            "dv_mm": self.dv_mm,
            "rho0_mm": self.rho0_mm,
            "d0_mm": self.d0_mm,
            "center_col_0based": self.center_col_0based,
            "center_row_0based": self.center_row_0based,
            "ffs_mode": self.ffs_mode,
            "angular_steps_per_rotation": self.angular_steps_per_rotation,
            "water_mu_mm_inv": self.water_mu_mm_inv,
            "nominal_spiral_pitch": self.nominal_spiral_pitch,
            "log_flag": self.log_flag,
            "rescale_slope": self.rescale_slope,
            "rescale_intercept": self.rescale_intercept,
        }
        np.savez_compressed(
            out,
            meta_json=np.array(json.dumps(meta), dtype=np.str_),
            instance_numbers=self.instance_numbers,
            phi0_rad=self.phi0_rad,
            phi0_unwrapped_rad=self.phi0_unwrapped_rad,
            z0_mm=self.z0_mm,
            dphi_rad=self.dphi_rad,
            dz_mm=self.dz_mm,
            drho_mm=self.drho_mm,
        )
        return out

    @classmethod
    def from_npz(cls, path: str | Path) -> HelicalGeometry:
        """Load HelicalGeometry from a .npz archive written by to_npz."""
        src = Path(path).expanduser().resolve()
        with np.load(src, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"].item()))
            return cls(
                series_instance_uid=str(meta["series_instance_uid"]),
                detector_shape=str(meta["detector_shape"]),
                n_rows=int(meta["n_rows"]),
                n_cols=int(meta["n_cols"]),
                du_mm=float(meta["du_mm"]),
                dv_mm=float(meta["dv_mm"]),
                rho0_mm=float(meta["rho0_mm"]),
                d0_mm=float(meta["d0_mm"]),
                center_col_0based=float(meta["center_col_0based"]),
                center_row_0based=float(meta["center_row_0based"]),
                ffs_mode=str(meta["ffs_mode"]),
                angular_steps_per_rotation=int(meta["angular_steps_per_rotation"]),
                water_mu_mm_inv=float(meta["water_mu_mm_inv"]),
                nominal_spiral_pitch=(
                    float(meta["nominal_spiral_pitch"])
                    if meta.get("nominal_spiral_pitch") is not None
                    else None
                ),
                log_flag=str(meta["log_flag"]),
                rescale_slope=float(meta["rescale_slope"]),
                rescale_intercept=float(meta["rescale_intercept"]),
                instance_numbers=data["instance_numbers"],
                phi0_rad=data["phi0_rad"],
                phi0_unwrapped_rad=data["phi0_unwrapped_rad"],
                z0_mm=data["z0_mm"],
                dphi_rad=data["dphi_rad"],
                dz_mm=data["dz_mm"],
                drho_mm=data["drho_mm"],
            )
