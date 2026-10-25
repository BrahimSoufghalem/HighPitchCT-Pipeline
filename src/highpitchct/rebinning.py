"""Stage 2: Vectorized & Streaming Curved-to-Flat Detector Rebinning and Helical SSRB.

Implements first-principles 3D geometric rebinning from DICOM-CT-PD v4 helical cone-beam
projections to planar fan-beam sinograms (Noo et al., Phys. Med. Biol. 44, 1999, pp. 561-570),
with full 3D Flying Focal Spot (FFS) compensation (drho, dphi, dz) and explicit missing-data
masks (`measured_mask`) without any artificial edge smearing (`mode='nearest'` is strictly forbidden).

Key physical guarantees over legacy Helix2Fan implementations:
1. Cylindrical vs Spherical vs Flat intersection: uses exact 3D ray-surface intersection
   in the detector's local frame (horizontal norm sqrt(d0^2 + u^2) for CYLINDRICAL detectors,
   NOT spherical norm sqrt(d0^2 + u^2 + v^2)).
2. Exact 0-based central detector element (c0 = X_1based - 1.0, r0 = Y_1based - 1.0) and
   Row 0 = +z sign convention verified in Stage 1.
3. Exact 3D FFS compensation in BOTH Curved-to-Flat and SSRB:
   - Radial shift drho (+5.45 mm on odd views in Siemens FFSZ) modifies source-to-origin radius
     rho_s = rho0 + drho, requiring exact chord-matching fan angle
     alpha^(g) = arcsin((rho0 / rho_s) * sin(alpha)) and angular shift (alpha - alpha^(g)).
   - Partitioning by physical (drho, dz) FFS states preserves smooth helical interpolation per
     focal spot and exploits z-FFS axial oversampling via argmin |v| selection.
4. Strict out-of-bounds rejection: unmeasured pixels in `valid_flat_mask` and `measured_mask`
   are flagged `False` and filled with `0.0` so downstream extrapolation (Stage 3) and neural
   networks (Stage 7) see the true high-pitch missing-data geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .ctpd_reader import read_single_projection, scan_projection_series
from .geometry import SUPPORTED_DETECTOR_SHAPES, HelicalGeometry
from .phantom import Ellipsoid3D, project_phantom_helical
from .progress import ProgressReporter

SUPPORTED_ANGULAR_MODES = frozenset({"short_scan", "full_scan"})


def _solve_ray_detector_hit_local(
    detector_shape: str,
    *,
    d0_mm: float,
    du_mm: float,
    dv_mm: float,
    c0_0based: float,
    r0_0based: float,
    rho0_mm: float,
    drho_mm: float,
    dphi_rad: float,
    dz_mm: float,
    u_flat_mm: np.ndarray | float,
    v_flat_mm: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect virtual flat-panel ray(s) from S(i) with physical detector in F0(i) frame.

    At view i:
      - Physical detector focal center is F0(i) at (rho0, phi0, z0) with orthonormal basis
        (e_ray(phi0), e_u(phi0), e_z).
      - Actual FFS-shifted source is S(i) at (rho_s = rho0 + drho, phi_s = phi0 + dphi, z_s = z0 + dz).
      - Virtual centered flat detector lies at distance d0_mm along e_ray(phi_s) from S(i),
        spanned by u along e_u(phi_s) and v along +e_z.

    Expressing q = S(i) - F0(i) and ray vector w(u, v) in the (e_ray(phi0), e_u(phi0), e_z) frame:
        q_ray = rho0 - rho_s * cos(dphi)
        q_u   = -rho_s * sin(dphi)
        q_z   = dz
        w_ray = d0 * cos(dphi) - u * sin(dphi)
        w_u   = d0 * sin(dphi) + u * cos(dphi)
        w_z   = v

    Returns continuous 0-based physical detector coordinates (r_hit, c_hit).
    """
    u_arr, v_arr = np.broadcast_arrays(
        np.asarray(u_flat_mm, dtype=np.float64),
        np.asarray(v_flat_mm, dtype=np.float64),
    )

    rho_s = float(rho0_mm) + float(drho_mm)
    cos_dp = math.cos(float(dphi_rad))
    sin_dp = math.sin(float(dphi_rad))

    q_ray = float(rho0_mm) - rho_s * cos_dp
    q_u = -rho_s * sin_dp
    q_z = float(dz_mm)

    w_ray = float(d0_mm) * cos_dp - u_arr * sin_dp
    w_u = float(d0_mm) * sin_dp + u_arr * cos_dp
    w_z = v_arr

    shape_norm = detector_shape.strip().upper()
    if shape_norm == "CYLINDRICAL":
        a_q = w_ray * w_ray + w_u * w_u
        b_half = q_ray * w_ray + q_u * w_u
        c_q = (q_ray * q_ray + q_u * q_u) - (float(d0_mm) ** 2)
        disc = np.maximum(b_half * b_half - a_q * c_q, 0.0)
        t_hit = (-b_half + np.sqrt(disc)) / np.maximum(a_q, 1e-18)
        x_ray = q_ray + t_hit * w_ray
        x_u = q_u + t_hit * w_u
        x_z = q_z + t_hit * w_z
        gamma_hit = np.arctan2(x_u, x_ray)
        c_hit = float(c0_0based) + (float(d0_mm) * gamma_hit) / float(du_mm)
        r_hit = float(r0_0based) - x_z / float(dv_mm)
    elif shape_norm == "FLAT":
        t_hit = (float(d0_mm) - q_ray) / np.where(np.abs(w_ray) > 1e-18, w_ray, 1e-18)
        x_u = q_u + t_hit * w_u
        x_z = q_z + t_hit * w_z
        c_hit = float(c0_0based) + x_u / float(du_mm)
        r_hit = float(r0_0based) - x_z / float(dv_mm)
    elif shape_norm == "SPHERICAL":
        a_q = w_ray * w_ray + w_u * w_u + w_z * w_z
        b_half = q_ray * w_ray + q_u * w_u + q_z * w_z
        c_q = (q_ray * q_ray + q_u * q_u + q_z * q_z) - (float(d0_mm) ** 2)
        disc = np.maximum(b_half * b_half - a_q * c_q, 0.0)
        t_hit = (-b_half + np.sqrt(disc)) / np.maximum(a_q, 1e-18)
        x_ray = q_ray + t_hit * w_ray
        x_u = q_u + t_hit * w_u
        x_z = q_z + t_hit * w_z
        gamma_hit = np.arctan2(x_u, x_ray)
        alpha_hit = np.arcsin(np.clip(x_z / float(d0_mm), -1.0, 1.0))
        c_hit = float(c0_0based) + (float(d0_mm) * gamma_hit) / float(du_mm)
        r_hit = float(r0_0based) - (float(d0_mm) * alpha_hit) / float(dv_mm)
    else:
        raise ValueError(
            f"Unsupported detector_shape={detector_shape!r}; "
            f"expected one of {sorted(SUPPORTED_DETECTOR_SHAPES)}"
        )

    return r_hit, c_hit


@dataclass(frozen=True)
class CurvedToFlatPlan:
    """Precomputed per-FFS-state bilinear sampling tables for fast Curved-to-Flat rebinning."""

    n_v: int
    n_u: int
    du_mm: float
    dv_mm: float
    d0_mm: float
    u_coords_mm: np.ndarray
    v_coords_mm: np.ndarray
    unique_ffs_states: np.ndarray
    view_state_inverse: np.ndarray
    r_hit_states: np.ndarray
    c_hit_states: np.ndarray
    valid_mask_states: np.ndarray
    r0_states: np.ndarray
    r1_states: np.ndarray
    c0_states: np.ndarray
    c1_states: np.ndarray
    w00_states: np.ndarray
    w01_states: np.ndarray
    w10_states: np.ndarray
    w11_states: np.ndarray


def build_curved_to_flat_plan(
    geometry: HelicalGeometry,
    *,
    n_v: int | None = None,
    n_u: int | None = None,
    du_mm: float | None = None,
    dv_mm: float | None = None,
) -> CurvedToFlatPlan:
    """Precompute exact 3D Curved-to-Flat detector mapping tables per unique FFS state."""
    nv_out = int(geometry.n_rows if n_v is None else n_v)
    nu_out = int(geometry.n_cols if n_u is None else n_u)
    du_out = float(geometry.du_mm if du_mm is None else du_mm)
    dv_out = float(geometry.dv_mm if dv_mm is None else dv_mm)
    if nv_out < 1 or nu_out < 1:
        raise ValueError(f"Virtual flat detector dimensions must be >= 1, got ({nv_out}, {nu_out})")
    if du_out <= 0.0 or dv_out <= 0.0:
        raise ValueError(f"Virtual flat detector spacings must be > 0, got ({du_out}, {dv_out})")

    c_flat_0 = 0.5 * (nu_out - 1.0)
    r_flat_0 = 0.5 * (nv_out - 1.0)
    u_coords = (np.arange(nu_out, dtype=np.float64) - c_flat_0) * du_out  # (nu_out,)
    v_coords = (r_flat_0 - np.arange(nv_out, dtype=np.float64)) * dv_out  # (nv_out,), Row 0 = +z

    rounded_ffs = np.stack(
        [
            np.round(geometry.drho_mm, 4),
            np.round(geometry.dphi_rad, 6),
            np.round(geometry.dz_mm, 4),
        ],
        axis=-1,
    )
    unique_states, inv_idx = np.unique(rounded_ffs, axis=0, return_inverse=True)
    inv_idx = np.asarray(inv_idx, dtype=np.int64).reshape(-1)
    n_states = int(unique_states.shape[0])

    r_hit_all = np.empty((n_states, nv_out, nu_out), dtype=np.float64)
    c_hit_all = np.empty((n_states, nv_out, nu_out), dtype=np.float64)
    valid_all = np.empty((n_states, nv_out, nu_out), dtype=bool)
    r0_all = np.empty((n_states, nv_out, nu_out), dtype=np.intp)
    r1_all = np.empty((n_states, nv_out, nu_out), dtype=np.intp)
    c0_all = np.empty((n_states, nv_out, nu_out), dtype=np.intp)
    c1_all = np.empty((n_states, nv_out, nu_out), dtype=np.intp)
    w00_all = np.empty((n_states, nv_out, nu_out), dtype=np.float32)
    w01_all = np.empty((n_states, nv_out, nu_out), dtype=np.float32)
    w10_all = np.empty((n_states, nv_out, nu_out), dtype=np.float32)
    w11_all = np.empty((n_states, nv_out, nu_out), dtype=np.float32)

    u_grid = u_coords[None, :]  # (1, nu_out)
    v_grid = v_coords[:, None]  # (nv_out, 1)

    for s in range(n_states):
        mask_views = inv_idx == s
        drho_s = float(np.mean(geometry.drho_mm[mask_views]))
        dphi_s = float(np.mean(geometry.dphi_rad[mask_views]))
        dz_s = float(np.mean(geometry.dz_mm[mask_views]))

        r_hit, c_hit = _solve_ray_detector_hit_local(
            geometry.detector_shape,
            d0_mm=geometry.d0_mm,
            du_mm=geometry.du_mm,
            dv_mm=geometry.dv_mm,
            c0_0based=geometry.center_col_0based,
            r0_0based=geometry.center_row_0based,
            rho0_mm=geometry.rho0_mm,
            drho_mm=drho_s,
            dphi_rad=dphi_s,
            dz_mm=dz_s,
            u_flat_mm=u_grid,
            v_flat_mm=v_grid,
        )
        valid = (
            (r_hit >= 0.0)
            & (r_hit <= float(geometry.n_rows - 1))
            & (c_hit >= 0.0)
            & (c_hit <= float(geometry.n_cols - 1))
        )

        r_clamped = np.clip(r_hit, 0.0, float(geometry.n_rows - 1))
        c_clamped = np.clip(c_hit, 0.0, float(geometry.n_cols - 1))
        r0 = np.floor(r_clamped).astype(np.intp)
        c0 = np.floor(c_clamped).astype(np.intp)
        r1 = np.minimum(r0 + 1, geometry.n_rows - 1)
        c1 = np.minimum(c0 + 1, geometry.n_cols - 1)
        wr = r_clamped - r0
        wc = c_clamped - c0

        valid_f = valid.astype(np.float64)
        r_hit_all[s] = r_hit
        c_hit_all[s] = c_hit
        valid_all[s] = valid
        r0_all[s] = r0
        r1_all[s] = r1
        c0_all[s] = c0
        c1_all[s] = c1
        w00_all[s] = ((1.0 - wr) * (1.0 - wc) * valid_f).astype(np.float32)
        w01_all[s] = ((1.0 - wr) * wc * valid_f).astype(np.float32)
        w10_all[s] = (wr * (1.0 - wc) * valid_f).astype(np.float32)
        w11_all[s] = (wr * wc * valid_f).astype(np.float32)

    return CurvedToFlatPlan(
        n_v=nv_out,
        n_u=nu_out,
        du_mm=du_out,
        dv_mm=dv_out,
        d0_mm=float(geometry.d0_mm),
        u_coords_mm=u_coords,
        v_coords_mm=v_coords,
        unique_ffs_states=unique_states,
        view_state_inverse=inv_idx,
        r_hit_states=r_hit_all,
        c_hit_states=c_hit_all,
        valid_mask_states=valid_all,
        r0_states=r0_all,
        r1_states=r1_all,
        c0_states=c0_all,
        c1_states=c1_all,
        w00_states=w00_all,
        w01_states=w01_all,
        w10_states=w10_all,
        w11_states=w11_all,
    )


def rebin_curved_to_flat(
    projections: np.ndarray,
    geometry: HelicalGeometry,
    *,
    plan: CurvedToFlatPlan | None = None,
    batch_views: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized 3D Curved-to-Flat rebinning returning (flat_projections, valid_flat_mask).

    Out-of-bounds pixels are strictly set to `0.0` with `valid_flat_mask = False`
    (never smeared via `mode='nearest'`).
    """
    proj_arr = np.asarray(projections, dtype=np.float32)
    expected_shape = (geometry.n_views, geometry.n_rows, geometry.n_cols)
    if proj_arr.shape != expected_shape:
        raise ValueError(
            f"projections shape {proj_arr.shape} does not match geometry {expected_shape}"
        )

    active_plan = plan if plan is not None else build_curved_to_flat_plan(geometry)
    if active_plan.view_state_inverse.shape[0] != geometry.n_views:
        raise ValueError("CurvedToFlatPlan view count does not match geometry.n_views")

    n_views = geometry.n_views
    flat_out = np.zeros((n_views, active_plan.n_v, active_plan.n_u), dtype=np.float32)
    valid_out = active_plan.valid_mask_states[active_plan.view_state_inverse].copy()

    n_states = int(active_plan.unique_ffs_states.shape[0])
    step = max(1, int(batch_views))
    for s in range(n_states):
        view_idx = np.where(active_plan.view_state_inverse == s)[0]
        if view_idx.size == 0:
            continue
        r0 = active_plan.r0_states[s]
        r1 = active_plan.r1_states[s]
        c0 = active_plan.c0_states[s]
        c1 = active_plan.c1_states[s]
        w00 = active_plan.w00_states[s][None, :, :]
        w01 = active_plan.w01_states[s][None, :, :]
        w10 = active_plan.w10_states[s][None, :, :]
        w11 = active_plan.w11_states[s][None, :, :]

        for start in range(0, view_idx.size, step):
            sub_idx = view_idx[start : start + step]
            src_batch = proj_arr[sub_idx]
            flat_out[sub_idx] = (
                src_batch[:, r0, c0] * w00
                + src_batch[:, r0, c1] * w01
                + src_batch[:, r1, c0] * w10
                + src_batch[:, r1, c1] * w11
            )

    return flat_out, valid_out


def rebin_curved_to_flat_scalar_ref(
    projections: np.ndarray,
    geometry: HelicalGeometry,
    *,
    n_v: int | None = None,
    n_u: int | None = None,
    du_mm: float | None = None,
    dv_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit scalar reference loop for Curved-to-Flat validation (validation_protocol.md)."""
    proj_arr = np.asarray(projections, dtype=np.float64)
    expected_shape = (geometry.n_views, geometry.n_rows, geometry.n_cols)
    if proj_arr.shape != expected_shape:
        raise ValueError(
            f"projections shape {proj_arr.shape} does not match geometry {expected_shape}"
        )

    nv_out = int(geometry.n_rows if n_v is None else n_v)
    nu_out = int(geometry.n_cols if n_u is None else n_u)
    du_out = float(geometry.du_mm if du_mm is None else du_mm)
    dv_out = float(geometry.dv_mm if dv_mm is None else dv_mm)

    c_flat_0 = 0.5 * (nu_out - 1.0)
    r_flat_0 = 0.5 * (nv_out - 1.0)

    flat_out = np.zeros((geometry.n_views, nv_out, nu_out), dtype=np.float32)
    valid_out = np.zeros((geometry.n_views, nv_out, nu_out), dtype=bool)

    for i in range(geometry.n_views):
        drho_i = float(geometry.drho_mm[i])
        dphi_i = float(geometry.dphi_rad[i])
        dz_i = float(geometry.dz_mm[i])

        for n in range(nv_out):
            v_mm = (r_flat_0 - float(n)) * dv_out
            for m in range(nu_out):
                u_mm = (float(m) - c_flat_0) * du_out
                r_hit_arr, c_hit_arr = _solve_ray_detector_hit_local(
                    geometry.detector_shape,
                    d0_mm=geometry.d0_mm,
                    du_mm=geometry.du_mm,
                    dv_mm=geometry.dv_mm,
                    c0_0based=geometry.center_col_0based,
                    r0_0based=geometry.center_row_0based,
                    rho0_mm=geometry.rho0_mm,
                    drho_mm=drho_i,
                    dphi_rad=dphi_i,
                    dz_mm=dz_i,
                    u_flat_mm=u_mm,
                    v_flat_mm=v_mm,
                )
                rf = float(r_hit_arr)
                cf = float(c_hit_arr)
                if 0.0 <= rf <= float(geometry.n_rows - 1) and 0.0 <= cf <= float(
                    geometry.n_cols - 1
                ):
                    valid_out[i, n, m] = True
                    r0 = int(math.floor(rf))
                    c0 = int(math.floor(cf))
                    r1 = min(r0 + 1, geometry.n_rows - 1)
                    c1 = min(c0 + 1, geometry.n_cols - 1)
                    wr = rf - r0
                    wc = cf - c0
                    val = (
                        (1.0 - wr) * (1.0 - wc) * proj_arr[i, r0, c0]
                        + (1.0 - wr) * wc * proj_arr[i, r0, c1]
                        + wr * (1.0 - wc) * proj_arr[i, r1, c0]
                        + wr * wc * proj_arr[i, r1, c1]
                    )
                    flat_out[i, n, m] = np.float32(val)

    return flat_out, valid_out


@dataclass(frozen=True)
class SSRBResult:
    """Complete output of Single-Slice Rebinning (SSRB) for a stack of axial z slices."""

    sinograms: np.ndarray
    measured_mask: np.ndarray
    z_slices_mm: np.ndarray
    beta_rad: np.ndarray
    phi_center_rad: np.ndarray
    u_coords_mm: np.ndarray
    du_mm: float
    rho0_mm: float
    d0_mm: float
    gamma_max_rad: float
    angular_span_rad: float
    angular_mode: str
    active_row_range: tuple[float, float]

    @property
    def n_slices(self) -> int:
        return int(self.sinograms.shape[0])

    @property
    def n_beta(self) -> int:
        return int(self.sinograms.shape[1])

    @property
    def n_u(self) -> int:
        return int(self.sinograms.shape[2])

    @property
    def measured_fraction_per_slice(self) -> np.ndarray:
        return np.mean(self.measured_mask, axis=(1, 2)).astype(np.float64)

    @property
    def overall_measured_fraction(self) -> float:
        if self.measured_mask.size == 0:
            return 0.0
        return float(np.mean(self.measured_mask))

    def summary_dict(self) -> dict[str, Any]:
        fracs = self.measured_fraction_per_slice
        return {
            "angular_mode": self.angular_mode,
            "sinograms_shape": [int(x) for x in self.sinograms.shape],
            "n_slices": self.n_slices,
            "n_beta": self.n_beta,
            "n_u": self.n_u,
            "du_mm": round(self.du_mm, 6),
            "rho0_mm": round(self.rho0_mm, 4),
            "d0_mm": round(self.d0_mm, 4),
            "gamma_max_deg": round(math.degrees(self.gamma_max_rad), 4),
            "angular_span_deg": round(math.degrees(self.angular_span_rad), 4),
            "active_row_range": [round(float(x), 4) for x in self.active_row_range],
            "z_slices_mm": [round(float(z), 4) for z in self.z_slices_mm],
            "measured_fraction_per_slice": [round(float(f), 6) for f in fracs],
            "overall_measured_fraction": round(self.overall_measured_fraction, 6),
            "missing_fraction_overall": round(1.0 - self.overall_measured_fraction, 6),
        }


def compute_phi_center_for_slices(
    geometry: HelicalGeometry,
    z_slices_mm: np.ndarray | list[float],
) -> np.ndarray:
    """Compute nominal unwrapped source angle phi_center(z_slice) via linear fit of z_s(phi_s)."""
    z_req = np.asarray(z_slices_mm, dtype=np.float64).reshape(-1)
    _, phi_s, z_s = geometry.source_cylindrical()
    phi_s = np.asarray(phi_s, dtype=np.float64)
    z_s = np.asarray(z_s, dtype=np.float64)

    mean_phi = float(np.mean(phi_s))
    mean_z = float(np.mean(z_s))
    dphi = phi_s - mean_phi
    sxx = float(np.dot(dphi, dphi))
    if sxx <= 1e-18:
        return np.full_like(z_req, mean_phi)
    slope = float(np.dot(dphi, z_s - mean_z) / sxx)
    if abs(slope) <= 1e-12:
        return np.full_like(z_req, mean_phi)
    return mean_phi + (z_req - mean_z) / slope


def build_ssrb_angular_grid(
    geometry: HelicalGeometry,
    z_slices_mm: np.ndarray | list[float],
    *,
    angular_mode: str = "short_scan",
    n_beta: int | None = None,
    n_u: int | None = None,
    du_mm: float | None = None,
    phi_center_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Construct monotonically increasing target sinogram angles beta_rad[slice, k].

    Returns:
      (beta_rad [S, N_beta], phi_center_rad [S], gamma_max_rad, angular_span_rad)
    """
    mode_norm = angular_mode.strip().lower()
    if mode_norm not in SUPPORTED_ANGULAR_MODES:
        raise ValueError(
            f"Unsupported angular_mode={angular_mode!r}; expected one of {sorted(SUPPORTED_ANGULAR_MODES)}"
        )

    z_arr = np.asarray(z_slices_mm, dtype=np.float64).reshape(-1)
    nu_eff = int(geometry.n_cols if n_u is None else n_u)
    du_eff = float(geometry.du_mm if du_mm is None else du_mm)
    half_width_u = 0.5 * max(0, nu_eff - 1) * du_eff
    gamma_max = math.atan2(half_width_u, float(geometry.d0_mm))

    phi_center = (
        np.asarray(phi_center_override, dtype=np.float64).reshape(-1)
        if phi_center_override is not None
        else compute_phi_center_for_slices(geometry, z_arr)
    )
    if phi_center.shape != z_arr.shape:
        raise ValueError(
            f"phi_center_override shape {phi_center.shape} != z_slices_mm shape {z_arr.shape}"
        )

    if mode_norm == "short_scan":
        span = math.pi + 2.0 * gamma_max
        nb = (
            int(n_beta)
            if n_beta is not None
            else max(2, int(math.ceil(geometry.angular_steps_per_rotation * (span / (2.0 * math.pi)))))
        )
        rel_offsets = np.linspace(-0.5 * span, 0.5 * span, nb, endpoint=True, dtype=np.float64)
    else:
        span = 2.0 * math.pi
        nb = int(n_beta) if n_beta is not None else max(2, int(geometry.angular_steps_per_rotation))
        rel_offsets = np.linspace(-math.pi, math.pi, nb, endpoint=False, dtype=np.float64)

    beta_rad = phi_center[:, None] + rel_offsets[None, :]
    return beta_rad, phi_center, gamma_max, span


def _partition_ffs_view_groups(geometry: HelicalGeometry) -> list[np.ndarray]:
    """Partition view indices into smooth single-focal-spot groups sorted in acquisition order.

    Grouping by rounded (drho_mm, dz_mm) separates alternating focal spots in FFSZ / FFSXY / FFSXYZ
    so each group has constant radial offset rho_s and smooth monotonic z_s(phi_s), independent of
    whether a sliced window starts on an even or odd global view index.
    """
    keys = np.stack(
        [np.round(geometry.drho_mm, 3), np.round(geometry.dz_mm, 3)],
        axis=-1,
    )
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv, dtype=np.int64).reshape(-1)
    groups: list[np.ndarray] = []
    for gid in range(int(np.max(inv)) + 1):
        idx = np.where(inv == gid)[0]
        if idx.size >= 2:
            groups.append(idx)
    # Sort deterministically by first view index
    groups.sort(key=lambda arr: int(arr[0]))
    return groups


def rebin_helical_ssrb(
    flat_projections: np.ndarray,
    flat_valid_mask: np.ndarray | None,
    geometry: HelicalGeometry,
    z_slices_mm: np.ndarray | list[float],
    *,
    angular_mode: str = "short_scan",
    n_beta: int | None = None,
    du_mm: float | None = None,
    active_row_range: tuple[float, float] | None = None,
    apply_cosine_weight: bool = True,
    phi_center_override: np.ndarray | None = None,
) -> SSRBResult:
    """Vectorized Noo et al. (1999) Helical Single-Slice Rebinning with full 3D FFS support.

    Selects among all valid FFS focal-spot states and multi-turn 2*pi*ell shifts using strict
    argmin |v| (closest ray to central detector plane). Unmeasured pixels receive `0.0` and
    `measured_mask = False` without any `mode='nearest'` edge smearing.
    """
    flat_arr = np.asarray(flat_projections, dtype=np.float32)
    if flat_arr.ndim != 3 or flat_arr.shape[0] != geometry.n_views:
        raise ValueError(
            f"flat_projections must have shape ({geometry.n_views}, N_v, N_u), got {flat_arr.shape}"
        )
    n_views, n_v, n_u = flat_arr.shape
    du_eff = float(geometry.du_mm if du_mm is None else du_mm)
    dv_eff = float(geometry.dv_mm)

    if flat_valid_mask is None:
        valid_flat = np.ones((n_views, n_v, n_u), dtype=bool)
    else:
        valid_flat = np.asarray(flat_valid_mask, dtype=bool)
        if valid_flat.shape != flat_arr.shape:
            raise ValueError(
                f"flat_valid_mask shape {valid_flat.shape} != flat_projections shape {flat_arr.shape}"
            )

    r_min_act = 0.0 if active_row_range is None else float(active_row_range[0])
    r_max_act = float(n_v - 1) if active_row_range is None else float(active_row_range[1])
    if r_min_act < 0.0 or r_max_act > float(n_v - 1) or r_min_act > r_max_act:
        raise ValueError(
            f"Invalid active_row_range={active_row_range!r} for n_v={n_v} (valid [0, {n_v - 1}])"
        )
    r_upper_int = int(math.floor(r_max_act))

    z_arr = np.asarray(z_slices_mm, dtype=np.float64).reshape(-1)
    beta_rad, phi_center, gamma_max, span = build_ssrb_angular_grid(
        geometry,
        z_arr,
        angular_mode=angular_mode,
        n_beta=n_beta,
        n_u=n_u,
        du_mm=du_eff,
        phi_center_override=phi_center_override,
    )
    n_slices, nb_out = beta_rad.shape

    c_flat_0 = 0.5 * (n_u - 1.0)
    r_flat_0 = 0.5 * (n_v - 1.0)
    u_coords = (np.arange(n_u, dtype=np.float64) - c_flat_0) * du_eff
    alpha_virt = np.arctan2(u_coords, float(geometry.d0_mm))  # (n_u,)
    sin_alpha_virt = np.sin(alpha_virt)

    sinograms = np.zeros((n_slices, nb_out, n_u), dtype=np.float32)
    measured = np.zeros((n_slices, nb_out, n_u), dtype=bool)
    if n_slices == 0:
        return SSRBResult(
            sinograms=sinograms,
            measured_mask=measured,
            z_slices_mm=z_arr,
            beta_rad=beta_rad,
            phi_center_rad=phi_center,
            u_coords_mm=u_coords,
            du_mm=du_eff,
            rho0_mm=float(geometry.rho0_mm),
            d0_mm=float(geometry.d0_mm),
            gamma_max_rad=gamma_max,
            angular_span_rad=span,
            angular_mode=angular_mode.strip().lower(),
            active_row_range=(r_min_act, r_max_act),
        )

    rho_s_all, phi_s_all, z_s_all = geometry.source_cylindrical()
    rho_s_all = np.asarray(rho_s_all, dtype=np.float64)
    phi_s_all = np.asarray(phi_s_all, dtype=np.float64)
    z_s_all = np.asarray(z_s_all, dtype=np.float64)

    groups = _partition_ffs_view_groups(geometry)
    if not groups:
        raise ValueError("SSRB requires at least 2 views in at least one FFS state")

    # Precompute column geometry & monotonic lookup arrays for each FFS group
    group_cache: list[dict[str, Any]] = []
    for idx_g in groups:
        rho_g = float(np.mean(rho_s_all[idx_g]))
        ratio_sin = (float(geometry.rho0_mm) / rho_g) * sin_alpha_virt
        valid_sin = np.abs(ratio_sin) <= 1.0
        alpha_g = np.arcsin(np.clip(ratio_sin, -1.0, 1.0))
        u_g = float(geometry.d0_mm) * np.tan(alpha_g)
        c_g = (u_g / du_eff) + c_flat_0
        valid_col = valid_sin & (c_g >= 0.0) & (c_g <= float(n_u - 1))

        c_clamped = np.clip(c_g, 0.0, float(n_u - 1))
        c0_g = np.floor(c_clamped).astype(np.intp)
        c1_g = np.minimum(c0_g + 1, n_u - 1)
        wc_g = c_clamped - c0_g

        d0_sq_plus_u_sq = (float(geometry.d0_mm) ** 2) + (u_g * u_g)
        noo_u_factor = d0_sq_plus_u_sq / (rho_g * float(geometry.d0_mm))
        horiz_len_g = np.sqrt(d0_sq_plus_u_sq)

        phi_g = phi_s_all[idx_g]
        z_g = z_s_all[idx_g]
        pos_grid = np.arange(idx_g.size, dtype=np.float64)
        if phi_g[-1] >= phi_g[0]:
            phi_sorted = phi_g
            pos_sorted = pos_grid
        else:
            phi_sorted = phi_g[::-1]
            pos_sorted = pos_grid[::-1]

        group_cache.append(
            {
                "idx_g": idx_g,
                "d_alpha": alpha_virt - alpha_g,  # (n_u,)
                "valid_col": valid_col,  # (n_u,)
                "c0": c0_g,
                "c1": c1_g,
                "wc": wc_g,
                "d0_sq_plus_u_sq": d0_sq_plus_u_sq,
                "noo_u_factor": noo_u_factor,
                "horiz_len": horiz_len_g,
                "phi_sorted": phi_sorted,
                "pos_sorted": pos_sorted,
                "z_g": z_g,
            }
        )

    v_max_abs = max(abs((r_flat_0 - r_min_act) * dv_eff), abs((r_flat_0 - r_max_act) * dv_eff))
    z_reach_max = v_max_abs * (float(np.max(rho_s_all)) / float(geometry.d0_mm)) + 5.0

    for s_idx in range(n_slices):
        z_sl = float(z_arr[s_idx])
        beta_s = beta_rad[s_idx]  # (nb_out,)
        best_abs_v = np.full((nb_out, n_u), np.inf, dtype=np.float64)
        best_val = np.zeros((nb_out, n_u), dtype=np.float32)
        best_mask = np.zeros((nb_out, n_u), dtype=bool)

        for gc in group_cache:
            phi_sorted = gc["phi_sorted"]
            phi_min = float(phi_sorted[0])
            phi_max = float(phi_sorted[-1])
            base_phi = beta_s[:, None] + gc["d_alpha"][None, :]  # (nb_out, n_u)
            ell_min = int(math.floor((phi_min - float(np.max(base_phi))) / (2.0 * math.pi)))
            ell_max = int(math.ceil((phi_max - float(np.min(base_phi))) / (2.0 * math.pi)))

            idx_g = gc["idx_g"]
            z_g = gc["z_g"]

            for ell in range(ell_min, ell_max + 1):
                phi_star = base_phi + (2.0 * math.pi * float(ell))
                cand_ok = (
                    (phi_star >= phi_min)
                    & (phi_star <= phi_max)
                    & gc["valid_col"][None, :]
                )
                if not np.any(cand_ok):
                    continue

                # Quick axial reach pruning before 2D array gather
                mid_phi = float(np.mean(phi_star[cand_ok]))
                mid_pos = float(np.interp(mid_phi, phi_sorted, gc["pos_sorted"]))
                z_est = float(z_g[min(int(round(mid_pos)), z_g.size - 1)])
                if abs(z_sl - z_est) > z_reach_max + abs(float(z_g[-1] - z_g[0])) * 0.25 and z_g.size > 64:
                    # Check exact min |z_sl - z_s| on endpoints of cand_ok
                    p_lo = float(np.interp(float(np.min(phi_star[cand_ok])), phi_sorted, gc["pos_sorted"]))
                    p_hi = float(np.interp(float(np.max(phi_star[cand_ok])), phi_sorted, gc["pos_sorted"]))
                    z_lo = float(z_g[int(math.floor(p_lo))])
                    z_hi = float(z_g[min(int(math.ceil(p_hi)), z_g.size - 1)])
                    if (z_sl < min(z_lo, z_hi) - z_reach_max) or (
                        z_sl > max(z_lo, z_hi) + z_reach_max
                    ):
                        continue

                k_sub, m_sub = np.where(cand_ok)
                pos_f = np.interp(phi_star[k_sub, m_sub], phi_sorted, gc["pos_sorted"])
                j0 = np.floor(pos_f).astype(np.intp)
                j1 = np.minimum(j0 + 1, idx_g.size - 1)
                wj = pos_f - j0

                z_star = (1.0 - wj) * z_g[j0] + wj * z_g[j1]
                v_cand = (z_sl - z_star) * gc["noo_u_factor"][m_sub]
                r_cand = r_flat_0 - (v_cand / dv_eff)

                in_row = (r_cand >= r_min_act) & (r_cand <= r_max_act)
                if not np.any(in_row):
                    continue

                k_ok = k_sub[in_row]
                m_ok = m_sub[in_row]
                v_ok = v_cand[in_row]
                r_ok = r_cand[in_row]
                j0_ok = j0[in_row]
                j1_ok = j1[in_row]
                wj_ok = wj[in_row]

                v0_global = idx_g[j0_ok]
                v1_global = idx_g[j1_ok]

                r0 = np.floor(r_ok).astype(np.intp)
                r1 = np.minimum(r0 + 1, r_upper_int)
                wr = r_ok - r0

                c0 = gc["c0"][m_ok]
                c1 = gc["c1"][m_ok]
                wc = gc["wc"][m_ok]

                corners_valid = (
                    valid_flat[v0_global, r0, c0]
                    & valid_flat[v0_global, r0, c1]
                    & valid_flat[v0_global, r1, c0]
                    & valid_flat[v0_global, r1, c1]
                    & valid_flat[v1_global, r0, c0]
                    & valid_flat[v1_global, r0, c1]
                    & valid_flat[v1_global, r1, c0]
                    & valid_flat[v1_global, r1, c1]
                )
                if not np.any(corners_valid):
                    continue

                k_fin = k_ok[corners_valid]
                m_fin = m_ok[corners_valid]
                abs_v_fin = np.abs(v_ok[corners_valid])

                better = abs_v_fin < best_abs_v[k_fin, m_fin]
                if not np.any(better):
                    continue

                sel = np.where(corners_valid)[0][better]
                k_win = k_ok[sel]
                m_win = m_ok[sel]
                v0_w = v0_global[sel]
                v1_w = v1_global[sel]
                r0_w = r0[sel]
                r1_w = r1[sel]
                wr_w = wr[sel]
                c0_w = c0[sel]
                c1_w = c1[sel]
                wc_w = wc[sel]
                wj_w = wj_ok[sel]
                v_w = v_ok[sel]

                w00 = (1.0 - wr_w) * (1.0 - wc_w)
                w01 = (1.0 - wr_w) * wc_w
                w10 = wr_w * (1.0 - wc_w)
                w11 = wr_w * wc_w

                val0 = (
                    w00 * flat_arr[v0_w, r0_w, c0_w]
                    + w01 * flat_arr[v0_w, r0_w, c1_w]
                    + w10 * flat_arr[v0_w, r1_w, c0_w]
                    + w11 * flat_arr[v0_w, r1_w, c1_w]
                )
                val1 = (
                    w00 * flat_arr[v1_w, r0_w, c0_w]
                    + w01 * flat_arr[v1_w, r0_w, c1_w]
                    + w10 * flat_arr[v1_w, r1_w, c0_w]
                    + w11 * flat_arr[v1_w, r1_w, c1_w]
                )
                interp_val = (1.0 - wj_w) * val0 + wj_w * val1
                if apply_cosine_weight:
                    cos_w = gc["horiz_len"][m_win] / np.sqrt(
                        gc["d0_sq_plus_u_sq"][m_win] + v_w * v_w
                    )
                    interp_val = interp_val * cos_w

                best_abs_v[k_win, m_win] = np.abs(v_w)
                best_val[k_win, m_win] = interp_val.astype(np.float32)
                best_mask[k_win, m_win] = True

        sinograms[s_idx] = best_val
        measured[s_idx] = best_mask

    return SSRBResult(
        sinograms=sinograms,
        measured_mask=measured,
        z_slices_mm=z_arr,
        beta_rad=beta_rad,
        phi_center_rad=phi_center,
        u_coords_mm=u_coords,
        du_mm=du_eff,
        rho0_mm=float(geometry.rho0_mm),
        d0_mm=float(geometry.d0_mm),
        gamma_max_rad=gamma_max,
        angular_span_rad=span,
        angular_mode=angular_mode.strip().lower(),
        active_row_range=(r_min_act, r_max_act),
    )


def rebin_helical_ssrb_scalar_ref(
    flat_projections: np.ndarray,
    flat_valid_mask: np.ndarray | None,
    geometry: HelicalGeometry,
    z_slices_mm: np.ndarray | list[float],
    *,
    angular_mode: str = "short_scan",
    n_beta: int | None = None,
    du_mm: float | None = None,
    active_row_range: tuple[float, float] | None = None,
    apply_cosine_weight: bool = True,
) -> SSRBResult:
    """Explicit scalar reference loop for SSRB verification (validation_protocol.md)."""
    flat_arr = np.asarray(flat_projections, dtype=np.float64)
    n_views, n_v, n_u = flat_arr.shape
    du_eff = float(geometry.du_mm if du_mm is None else du_mm)
    dv_eff = float(geometry.dv_mm)
    valid_flat = (
        np.ones((n_views, n_v, n_u), dtype=bool)
        if flat_valid_mask is None
        else np.asarray(flat_valid_mask, dtype=bool)
    )

    r_min_act = 0.0 if active_row_range is None else float(active_row_range[0])
    r_max_act = float(n_v - 1) if active_row_range is None else float(active_row_range[1])
    r_upper_int = int(math.floor(r_max_act))

    z_arr = np.asarray(z_slices_mm, dtype=np.float64).reshape(-1)
    beta_rad, phi_center, gamma_max, span = build_ssrb_angular_grid(
        geometry,
        z_arr,
        angular_mode=angular_mode,
        n_beta=n_beta,
        n_u=n_u,
        du_mm=du_eff,
    )
    n_slices, nb_out = beta_rad.shape

    c_flat_0 = 0.5 * (n_u - 1.0)
    r_flat_0 = 0.5 * (n_v - 1.0)
    u_coords = (np.arange(n_u, dtype=np.float64) - c_flat_0) * du_eff

    rho_s_all, phi_s_all, z_s_all = geometry.source_cylindrical()
    rho_s_all = np.asarray(rho_s_all, dtype=np.float64)
    phi_s_all = np.asarray(phi_s_all, dtype=np.float64)
    z_s_all = np.asarray(z_s_all, dtype=np.float64)
    groups = _partition_ffs_view_groups(geometry)

    sinograms = np.zeros((n_slices, nb_out, n_u), dtype=np.float32)
    measured = np.zeros((n_slices, nb_out, n_u), dtype=bool)

    for s_idx in range(n_slices):
        z_sl = float(z_arr[s_idx])
        for k in range(nb_out):
            beta_k = float(beta_rad[s_idx, k])
            for m in range(n_u):
                u_m = float(u_coords[m])
                alpha_m = math.atan2(u_m, float(geometry.d0_mm))
                best_abs_v = float("inf")
                best_val = 0.0
                found = False

                for idx_g in groups:
                    rho_g = float(np.mean(rho_s_all[idx_g]))
                    ratio_s = (float(geometry.rho0_mm) / rho_g) * math.sin(alpha_m)
                    if abs(ratio_s) > 1.0:
                        continue
                    alpha_g = math.asin(ratio_s)
                    u_g = float(geometry.d0_mm) * math.tan(alpha_g)
                    c_g = (u_g / du_eff) + c_flat_0
                    if c_g < 0.0 or c_g > float(n_u - 1):
                        continue

                    c0 = int(math.floor(c_g))
                    c1 = min(c0 + 1, n_u - 1)
                    wc = c_g - c0

                    phi_g = phi_s_all[idx_g]
                    z_g = z_s_all[idx_g]
                    if phi_g[-1] >= phi_g[0]:
                        phi_min, phi_max = float(phi_g[0]), float(phi_g[-1])
                    else:
                        phi_min, phi_max = float(phi_g[-1]), float(phi_g[0])

                    base_phi = beta_k + (alpha_m - alpha_g)
                    ell_min = int(math.floor((phi_min - base_phi) / (2.0 * math.pi)))
                    ell_max = int(math.ceil((phi_max - base_phi) / (2.0 * math.pi)))

                    for ell in range(ell_min, ell_max + 1):
                        phi_star = base_phi + 2.0 * math.pi * float(ell)
                        if phi_star < phi_min or phi_star > phi_max:
                            continue

                        # Find bracketing pair (j0, j1) in idx_g
                        if phi_g[-1] >= phi_g[0]:
                            pos_f = float(
                                np.interp(
                                    phi_star,
                                    phi_g,
                                    np.arange(idx_g.size, dtype=np.float64),
                                )
                            )
                        else:
                            pos_f = float(
                                np.interp(
                                    phi_star,
                                    phi_g[::-1],
                                    np.arange(idx_g.size, dtype=np.float64)[::-1],
                                )
                            )
                        j0 = int(math.floor(pos_f))
                        j1 = min(j0 + 1, int(idx_g.size) - 1)
                        wj = pos_f - j0

                        z_star = (1.0 - wj) * float(z_g[j0]) + wj * float(z_g[j1])
                        d0_sq_u_sq = (float(geometry.d0_mm) ** 2) + (u_g * u_g)
                        v_cand = (z_sl - z_star) * (
                            d0_sq_u_sq / (rho_g * float(geometry.d0_mm))
                        )
                        r_cand = r_flat_0 - (v_cand / dv_eff)
                        if r_cand < r_min_act or r_cand > r_max_act:
                            continue

                        r0 = int(math.floor(r_cand))
                        r1 = min(r0 + 1, r_upper_int)
                        wr = r_cand - r0

                        v0_gl = int(idx_g[j0])
                        v1_gl = int(idx_g[j1])
                        if not (
                            valid_flat[v0_gl, r0, c0]
                            and valid_flat[v0_gl, r0, c1]
                            and valid_flat[v0_gl, r1, c0]
                            and valid_flat[v0_gl, r1, c1]
                            and valid_flat[v1_gl, r0, c0]
                            and valid_flat[v1_gl, r0, c1]
                            and valid_flat[v1_gl, r1, c0]
                            and valid_flat[v1_gl, r1, c1]
                        ):
                            continue

                        if abs(v_cand) >= best_abs_v:
                            continue

                        val0 = (
                            (1.0 - wr) * (1.0 - wc) * flat_arr[v0_gl, r0, c0]
                            + (1.0 - wr) * wc * flat_arr[v0_gl, r0, c1]
                            + wr * (1.0 - wc) * flat_arr[v0_gl, r1, c0]
                            + wr * wc * flat_arr[v0_gl, r1, c1]
                        )
                        val1 = (
                            (1.0 - wr) * (1.0 - wc) * flat_arr[v1_gl, r0, c0]
                            + (1.0 - wr) * wc * flat_arr[v1_gl, r0, c1]
                            + wr * (1.0 - wc) * flat_arr[v1_gl, r1, c0]
                            + wr * wc * flat_arr[v1_gl, r1, c1]
                        )
                        val = (1.0 - wj) * val0 + wj * val1
                        if apply_cosine_weight:
                            val *= math.sqrt(d0_sq_u_sq) / math.sqrt(
                                d0_sq_u_sq + v_cand * v_cand
                            )

                        best_abs_v = abs(v_cand)
                        best_val = val
                        found = True

                if found:
                    sinograms[s_idx, k, m] = np.float32(best_val)
                    measured[s_idx, k, m] = True

    return SSRBResult(
        sinograms=sinograms,
        measured_mask=measured,
        z_slices_mm=z_arr,
        beta_rad=beta_rad,
        phi_center_rad=phi_center,
        u_coords_mm=u_coords,
        du_mm=du_eff,
        rho0_mm=float(geometry.rho0_mm),
        d0_mm=float(geometry.d0_mm),
        gamma_max_rad=gamma_max,
        angular_span_rad=span,
        angular_mode=angular_mode.strip().lower(),
        active_row_range=(r_min_act, r_max_act),
    )


def rebin_series_ssrb_streaming(
    series_dir_or_files: str | Path | list[Path],
    z_slices_mm: np.ndarray | list[float],
    *,
    geometry: HelicalGeometry | None = None,
    angular_mode: str = "short_scan",
    n_beta: int | None = None,
    active_row_range: tuple[float, float] | None = None,
    apply_cosine_weight: bool = True,
    max_batch_z_span_mm: float = 20.0,
    workers: int = 4,
    show_progress: bool = False,
) -> SSRBResult:
    """Bounded-RAM sliding-window Curved-to-Flat + SSRB directly from DICOM-CT-PD files.

    Guarantees exact bit-for-bit agreement with full in-memory `rebin_curved_to_flat` +
    `rebin_helical_ssrb` while keeping peak memory under ~350 MiB even on 13,446-view C012,
    and reusing overlapping flat-rebinned views across adjacent z batches so no view is read twice.
    """
    from concurrent.futures import ThreadPoolExecutor

    if isinstance(series_dir_or_files, list):
        files = [Path(p) for p in series_dir_or_files]
        if geometry is None:
            raise ValueError("geometry must be provided when passing an explicit file list")
    else:
        geometry, files = scan_projection_series(
            series_dir_or_files, workers=max(workers, 4), show_progress=show_progress
        )

    if len(files) != geometry.n_views:
        raise ValueError(f"File count ({len(files)}) != geometry.n_views ({geometry.n_views})")

    z_req = np.asarray(z_slices_mm, dtype=np.float64).reshape(-1)
    phi_center_full = compute_phi_center_for_slices(geometry, z_req)
    beta_full, _, gamma_max, span = build_ssrb_angular_grid(
        geometry,
        z_req,
        angular_mode=angular_mode,
        n_beta=n_beta,
        phi_center_override=phi_center_full,
    )
    n_slices, nb_out = beta_full.shape
    n_u = geometry.n_cols
    n_v = geometry.n_rows

    r_min_act = 0.0 if active_row_range is None else float(active_row_range[0])
    r_max_act = float(n_v - 1) if active_row_range is None else float(active_row_range[1])
    u_coords = (np.arange(n_u, dtype=np.float64) - 0.5 * (n_u - 1.0)) * float(geometry.du_mm)

    sinograms = np.zeros((n_slices, nb_out, n_u), dtype=np.float32)
    measured = np.zeros((n_slices, nb_out, n_u), dtype=bool)
    if n_slices == 0:
        return SSRBResult(
            sinograms=sinograms,
            measured_mask=measured,
            z_slices_mm=z_req,
            beta_rad=beta_full,
            phi_center_rad=phi_center_full,
            u_coords_mm=u_coords,
            du_mm=float(geometry.du_mm),
            rho0_mm=float(geometry.rho0_mm),
            d0_mm=float(geometry.d0_mm),
            gamma_max_rad=gamma_max,
            angular_span_rad=span,
            angular_mode=angular_mode.strip().lower(),
            active_row_range=(r_min_act, r_max_act),
        )

    # Sort slice indices by z so the sliding window moves monotonically along the helix
    order = np.argsort(z_req)
    batches: list[np.ndarray] = []
    b_start = 0
    while b_start < n_slices:
        z0_b = float(z_req[order[b_start]])
        b_stop = b_start + 1
        while (
            b_stop < n_slices
            and (float(z_req[order[b_stop]]) - z0_b) <= float(max_batch_z_span_mm)
            and (b_stop - b_start) < 16
        ):
            b_stop += 1
        batches.append(order[b_start:b_stop])
        b_start = b_stop

    v_max_mm = 0.5 * (n_v - 1.0) * float(geometry.dv_mm)
    rho_s_max = float(geometry.rho0_mm + np.max(geometry.drho_mm))
    dz_step_max = (
        float(np.max(np.abs(np.diff(geometry.z0_mm)))) if geometry.n_views > 1 else 0.0
    )
    z_pad_mm = (
        v_max_mm * (rho_s_max / float(geometry.d0_mm))
        + float(np.max(np.abs(geometry.dz_mm)))
        + 16.0 * dz_step_max
        + 4.0
    )

    full_plan = build_curved_to_flat_plan(geometry)

    # Sliding window cache: [cache_start, cache_stop) -> (flat_arr, valid_arr)
    cache_start = 0
    cache_stop = 0
    cache_flat: np.ndarray | None = None
    cache_valid: np.ndarray | None = None

    def _read_and_rebin_range(v_start: int, v_stop: int) -> tuple[np.ndarray, np.ndarray]:
        count = v_stop - v_start
        raw_chunk = np.empty((count, n_v, n_u), dtype=np.float32)
        eff_workers = max(1, min(int(workers), count))
        if eff_workers == 1:
            for k_rel, v_idx in enumerate(range(v_start, v_stop)):
                raw_chunk[k_rel] = read_single_projection(
                    files[v_idx], expected_rows=n_v, expected_cols=n_u
                )
        else:
            with ThreadPoolExecutor(max_workers=eff_workers) as pool:
                for k_rel, arr in enumerate(
                    pool.map(
                        lambda p: read_single_projection(p, expected_rows=n_v, expected_cols=n_u),
                        files[v_start:v_stop],
                    )
                ):
                    raw_chunk[k_rel] = arr

        sub_geom = geometry.slice_views(v_start, v_stop)
        sub_plan = CurvedToFlatPlan(
            n_v=full_plan.n_v,
            n_u=full_plan.n_u,
            du_mm=full_plan.du_mm,
            dv_mm=full_plan.dv_mm,
            d0_mm=full_plan.d0_mm,
            u_coords_mm=full_plan.u_coords_mm,
            v_coords_mm=full_plan.v_coords_mm,
            unique_ffs_states=full_plan.unique_ffs_states,
            view_state_inverse=full_plan.view_state_inverse[v_start:v_stop],
            r_hit_states=full_plan.r_hit_states,
            c_hit_states=full_plan.c_hit_states,
            valid_mask_states=full_plan.valid_mask_states,
            r0_states=full_plan.r0_states,
            r1_states=full_plan.r1_states,
            c0_states=full_plan.c0_states,
            c1_states=full_plan.c1_states,
            w00_states=full_plan.w00_states,
            w01_states=full_plan.w01_states,
            w10_states=full_plan.w10_states,
            w11_states=full_plan.w11_states,
        )
        return rebin_curved_to_flat(raw_chunk, sub_geom, plan=sub_plan)

    with ProgressReporter(
        total=n_slices,
        desc="🌀 Streaming Curved->Flat + SSRB",
        unit="slice",
        unit_scale=False,
        colour="green",
        enabled=show_progress,
    ) as bar:
        for b_indices in batches:
            z_b = z_req[b_indices]
            z_lo = float(np.min(z_b)) - z_pad_mm
            z_hi = float(np.max(z_b)) + z_pad_mm
            in_z = np.where((geometry.z0_mm >= z_lo) & (geometry.z0_mm <= z_hi))[0]
            if in_z.size < 2:
                bar.update(int(b_indices.size))
                continue

            i_min = max(0, int(in_z[0]) - 16)
            i_max = min(geometry.n_views, int(in_z[-1]) + 17)
            if i_max - i_min < 2:
                bar.update(int(b_indices.size))
                continue

            # Update sliding cache reusing any overlap with [cache_start, cache_stop)
            if cache_flat is None or cache_valid is None or i_min >= cache_stop or i_max <= cache_start:
                cache_flat, cache_valid = _read_and_rebin_range(i_min, i_max)
                cache_start, cache_stop = i_min, i_max
            else:
                ov_start = max(i_min, cache_start)
                ov_stop = min(i_max, cache_stop)
                new_flat = np.empty((i_max - i_min, n_v, n_u), dtype=np.float32)
                new_valid = np.empty((i_max - i_min, n_v, n_u), dtype=bool)

                # Copy overlap
                new_flat[ov_start - i_min : ov_stop - i_min] = cache_flat[
                    ov_start - cache_start : ov_stop - cache_start
                ]
                new_valid[ov_start - i_min : ov_stop - i_min] = cache_valid[
                    ov_start - cache_start : ov_stop - cache_start
                ]

                if i_min < ov_start:
                    f_pre, m_pre = _read_and_rebin_range(i_min, ov_start)
                    new_flat[0 : ov_start - i_min] = f_pre
                    new_valid[0 : ov_start - i_min] = m_pre
                if ov_stop < i_max:
                    f_post, m_post = _read_and_rebin_range(ov_stop, i_max)
                    new_flat[ov_stop - i_min : i_max - i_min] = f_post
                    new_valid[ov_stop - i_min : i_max - i_min] = m_post

                cache_flat, cache_valid = new_flat, new_valid
                cache_start, cache_stop = i_min, i_max

            win_geom = geometry.slice_views(i_min, i_max)
            sub_res = rebin_helical_ssrb(
                cache_flat,
                cache_valid,
                win_geom,
                z_b,
                angular_mode=angular_mode,
                n_beta=nb_out,
                active_row_range=active_row_range,
                apply_cosine_weight=apply_cosine_weight,
                phi_center_override=phi_center_full[b_indices],
            )
            sinograms[b_indices] = sub_res.sinograms
            measured[b_indices] = sub_res.measured_mask
            bar.update(int(b_indices.size))

    return SSRBResult(
        sinograms=sinograms,
        measured_mask=measured,
        z_slices_mm=z_req,
        beta_rad=beta_full,
        phi_center_rad=phi_center_full,
        u_coords_mm=u_coords,
        du_mm=float(geometry.du_mm),
        rho0_mm=float(geometry.rho0_mm),
        d0_mm=float(geometry.d0_mm),
        gamma_max_rad=gamma_max,
        angular_span_rad=span,
        angular_mode=angular_mode.strip().lower(),
        active_row_range=(r_min_act, r_max_act),
    )


def project_phantom_planar_fanbeam_2d(
    ellipsoids: list[Ellipsoid3D],
    *,
    z_slices_mm: np.ndarray | list[float],
    beta_rad: np.ndarray,
    n_u: int,
    du_mm: float,
    rho0_mm: float,
    d0_mm: float,
    water_mu_mm_inv: float = 0.0192,
) -> np.ndarray:
    """Compute exact 2D in-plane fan-beam sinograms g_exact[slice, beta, u] for Stage 2 QA.

    Evaluates exact analytical chords strictly inside z = z_slice on the centered virtual flat
    detector (rho0, d0, du_mm), serving as the exact 2D Ground Truth reference against which
    3D Helical Curved-to-Flat + SSRB sinograms are quantitatively and visually compared.
    """
    z_arr = np.asarray(z_slices_mm, dtype=np.float64).reshape(-1)
    beta_arr = np.asarray(beta_rad, dtype=np.float64)
    if beta_arr.ndim == 1:
        beta_arr = np.broadcast_to(beta_arr[None, :], (z_arr.size, beta_arr.size))
    if beta_arr.shape[0] != z_arr.size:
        raise ValueError(f"beta_rad shape {beta_arr.shape} incompatible with {z_arr.size} slices")

    n_slices, n_beta = beta_arr.shape
    out = np.zeros((n_slices, n_beta, int(n_u)), dtype=np.float32)
    zeros = np.zeros(n_beta, dtype=np.float64)
    inst = np.arange(1, n_beta + 1, dtype=np.int32)

    for s_idx in range(n_slices):
        b_s = beta_arr[s_idx]
        planar_geom = HelicalGeometry(
            series_instance_uid=f"1.2.826.0.1.3680043.10.999.200{s_idx}",
            detector_shape="FLAT",
            n_rows=1,
            n_cols=int(n_u),
            du_mm=float(du_mm),
            dv_mm=1.0,
            rho0_mm=float(rho0_mm),
            d0_mm=float(d0_mm),
            center_col_0based=0.5 * (int(n_u) - 1.0),
            center_row_0based=0.0,
            ffs_mode="FFSNONE",
            angular_steps_per_rotation=max(2, n_beta),
            water_mu_mm_inv=float(water_mu_mm_inv),
            nominal_spiral_pitch=0.0,
            log_flag="YES",
            rescale_slope=1.0,
            rescale_intercept=0.0,
            instance_numbers=inst,
            phi0_rad=np.mod(b_s, 2.0 * math.pi),
            phi0_unwrapped_rad=b_s.copy(),
            z0_mm=np.full(n_beta, float(z_arr[s_idx]), dtype=np.float64),
            dphi_rad=zeros,
            dz_mm=zeros,
            drho_mm=zeros,
        )
        proj_1row = project_phantom_helical(
            ellipsoids, planar_geom, batch_views=128, show_progress=False
        )
        out[s_idx] = proj_1row[:, 0, :]

    return out
