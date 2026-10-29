"""Stage 4: 2D Fan-Beam & 3D Helical Forward Projectors and High-Pitch (RDC / ITF) Simulation Engine.

Implements the physical data-generation and reprojection pipeline of the AAPM 2026 poster
(Zeng & Lee, "Efficient Dual-Domain Deep Learning Model for High-Pitch Helical CT Reconstruction"):

1. **2D Equispaced Fan-Beam Forward Projector (`forward_project_fanbeam_2d`)**:
   - Exact geometric forward operator `P_fan: mu(s, y, x) -> g(s, k, m)` matched to the
     radiological coordinate grid (`Col 0 = +x_CTPD` Patient Right, `Col W-1 = -x_CTPD` Patient Left,
     `Row 0 = +y_CTPD` Anterior, `Row H-1 = -y_CTPD` Posterior) and virtual flat detector geometry
     of `reconstruct_fanbeam_fbp` (`Stage 3`).
   - Used in **Fig. 2(d)** to forward-project the preliminary high-pitch FBP reconstruction
     `x_prelim` (`Fig. 2(c)`) into the reprojected input sinogram `s_reproj` for the Sinogram U-Net.
   - Includes an explicit scalar-loop reference `forward_project_fanbeam_2d_scalar_ref` for
     Level-2 validation parity and supports explicit backends (`numpy`, `torch`, `torch_cuda`)
     with strict fail-fast (no silent fallback).

2. **3D Helical Cone-Beam Forward Projector (`forward_project_helical_3d`)**:
   - Ray-driven 3D helical cone-beam projector `P_helical: mu(z, y, x) -> g(i, r, c)` supporting
     cylindrical, flat, and spherical detectors and 3D Flying Focal Spot (`FFS`) shifts
     `(drho, dphi, dz)`.

3. **High-Pitch Acquisition Scenarios (`Fig. 1(b)` RDC & `Fig. 1(c)` ITF)**:
   - **RDC (`Reduced Detector Coverage`)**: Restricts active detector rows to the central
     `N_active = round(N_rows * p_native / p_target)` rows while keeping table feed constant
     (e.g., `p = 2.4, 3.2, 4.8`).
   - **ITF (`Increased Table Feed`)**: Synthesizes a high-pitch `HelicalGeometry` with all `N_rows`
     active and table travel per rotation scaled to `p_target * W_coll_iso` (`synthesize_itf_helical_geometry`),
     forward-projects a 3D reference attenuation volume `mu_reference(z, y, x)`, and runs
     `Curved-to-Flat + SSRB + Extrapolation + Ram-Lak FBP + 2D Fan-Beam Reprojection`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .ctpd_reader import hu_to_mu, scan_projection_series
from .geometry import HelicalGeometry
from .progress import ProgressReporter
from .rebinning import (
    SSRBResult,
    rebin_curved_to_flat,
    rebin_helical_ssrb,
    rebin_series_ssrb_streaming,
)
from .recon_fbp import (
    FBPReconstructionResult,
    SUPPORTED_FBP_BACKENDS,
    _interp_columns_linear_2d,
    compute_parker_weights,
    compute_reconstruction_metrics,
    default_scanner_fov_mm,
    extrapolate_sinogram,
    normalize_filter_name,
    reconstruct_fanbeam_fbp,
    reconstruct_series_reference_streaming,
)

SUPPORTED_HIGH_PITCH_SCENARIOS = frozenset({"rdc", "itf"})


@dataclass(frozen=True)
class HighPitchSimulationPair:
    """Complete AAPM 2026 Fig. 2 Dual-Domain training/evaluation data tuple for a high-pitch scenario."""

    scenario: str
    target_pitch: float
    effective_pitch: float
    native_pitch: float
    active_row_range: tuple[float, float]
    z_slices_mm: np.ndarray
    y_mm: np.ndarray
    x_mm: np.ndarray
    fov_mm: float
    fov_mask: np.ndarray
    filter_name: str
    # Sinogram domain: shape (S, N_beta, N_u)
    s_target: np.ndarray
    s_hp_raw: np.ndarray
    mask_hp: np.ndarray
    s_hp_extrap: np.ndarray
    s_hp_parker_weighted: np.ndarray
    s_reproj: np.ndarray
    s_reproj_dc: np.ndarray
    beta_rad: np.ndarray
    u_coords_mm: np.ndarray
    # Image domain: shape (S, H, W)
    mu_target: np.ndarray
    x_target_hu: np.ndarray
    mu_prelim: np.ndarray
    x_prelim_hu: np.ndarray
    mu_raw_nofill: np.ndarray
    x_raw_nofill_hu: np.ndarray
    # Diagnostics & metrics
    prelim_vs_target_metrics: dict[str, float]
    raw_nofill_vs_target_metrics: dict[str, float]
    sinogram_metrics: dict[str, float]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "target_pitch": round(float(self.target_pitch), 4),
            "effective_pitch": round(float(self.effective_pitch), 4),
            "native_pitch": round(float(self.native_pitch), 4),
            "active_row_range": [round(float(r), 2) for r in self.active_row_range],
            "n_slices": int(self.z_slices_mm.size),
            "z_slices_mm": [round(float(z), 4) for z in self.z_slices_mm],
            "image_shape_syx": [int(x) for x in self.x_prelim_hu.shape],
            "sinogram_shape_sku": [int(x) for x in self.s_reproj.shape],
            "fov_mm": round(float(self.fov_mm), 4),
            "filter_name": self.filter_name,
            "measured_fraction": round(float(np.mean(self.mask_hp)), 6),
            "missing_fraction": round(float(1.0 - np.mean(self.mask_hp)), 6),
            "unmeasured_raw_strictly_zero": bool(np.all(self.s_hp_raw[~self.mask_hp] == 0.0)),
            "prelim_vs_target_metrics": self.prelim_vs_target_metrics,
            "raw_nofill_vs_target_metrics": self.raw_nofill_vs_target_metrics,
            "sinogram_metrics": self.sinogram_metrics,
        }


def compute_rdc_active_row_range(
    n_rows: int,
    native_pitch: float,
    target_pitch: float,
) -> tuple[float, float, float]:
    """Compute centered detector row subset `(r_lo, r_hi, effective_pitch)` for RDC (Fig. 1(b)).

    In Reduced Detector Coverage (RDC), keeping table feed constant and restricting active
    detector rows from `N_rows` to the central `N_active` rows increases the effective
    normalized pitch to `p_eff = p_native * (N_rows / N_active)`.
    """
    if int(n_rows) < 2:
        raise ValueError(f"n_rows must be >= 2, got {n_rows}")
    if float(native_pitch) <= 0.0:
        raise ValueError(f"native_pitch must be positive, got {native_pitch}")
    if float(target_pitch) <= 0.0:
        raise ValueError(f"target_pitch must be positive, got {target_pitch}")

    nr = int(n_rows)
    if float(target_pitch) <= float(native_pitch):
        return 0.0, float(nr - 1), float(native_pitch)

    raw_active = float(nr) * (float(native_pitch) / float(target_pitch))
    if nr % 2 == 0:
        # Round to nearest even number >= 2 so the window is strictly symmetric around (nr - 1)/2
        n_active = max(2, min(nr, int(2 * round(raw_active / 2.0))))
    else:
        n_active = max(2, min(nr, int(round(raw_active))))

    r_lo = 0.5 * float(nr - n_active)
    r_hi = r_lo + float(n_active - 1)
    eff_pitch = float(native_pitch) * (float(nr) / float(n_active))
    return float(r_lo), float(r_hi), float(eff_pitch)


def _project_fanbeam_slice_numpy(
    mu_2d: np.ndarray,
    beta_1d: np.ndarray,
    u_coords_mm: np.ndarray,
    *,
    rho0_mm: float,
    d0_mm: float,
    fov_mm: float,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    step_factor: float = 0.5,
    angle_chunk: int = 64,
) -> np.ndarray:
    """Vectorized 2D equispaced fan-beam ray-driven line-integral projector in pure NumPy."""
    img = np.asarray(mu_2d, dtype=np.float32)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"Expected square 2D slice (N_xy, N_xy), got {img.shape}")

    n_xy = int(img.shape[0])
    d_xy = float(fov_mm) / float(n_xy)
    half_idx = 0.5 * (n_xy - 1.0)
    cx = float(center_xy_mm[0])
    cy = float(center_xy_mm[1])

    # Ray-marching interval around closest approach t_iso(m) = rho0 * cos(alpha_m)
    alpha_u = np.arctan2(np.asarray(u_coords_mm, dtype=np.float64), float(d0_mm))  # (N_u,)
    t_iso = float(rho0_mm) * np.cos(alpha_u)  # (N_u,)

    r_bound = 0.5 * math.sqrt(2.0) * float(fov_mm) + math.hypot(cx, cy) + d_xy
    ds = max(0.1 * d_xy, float(step_factor) * d_xy)
    n_steps = max(8, int(math.ceil((2.0 * r_bound) / ds)) + 1)
    tau = np.linspace(-r_bound, r_bound, n_steps, dtype=np.float64)
    dt = float(tau[1] - tau[0]) if n_steps > 1 else float(2.0 * r_bound)

    # t_grid: shape (N_u, N_steps)
    t_grid = (t_iso[:, None] + tau[None, :]).astype(np.float32)

    n_beta = int(beta_1d.size)
    n_u = int(u_coords_mm.size)
    sino_out = np.zeros((n_beta, n_u), dtype=np.float32)

    inv_dxy = np.float32(1.0 / d_xy)
    half_idx_f = np.float32(half_idx)
    cx_f = np.float32(cx)
    cy_f = np.float32(cy)
    rho0_f = np.float32(rho0_mm)
    max_idx_f = np.float32(n_xy - 1.0)

    chunk = max(1, int(angle_chunk))
    for k0 in range(0, n_beta, chunk):
        k1 = min(n_beta, k0 + chunk)
        b_sub = np.asarray(beta_1d[k0:k1], dtype=np.float64)  # (K,)
        sx = (-rho0_f * np.sin(b_sub)).astype(np.float32)[:, None, None]  # (K, 1, 1)
        sy = (rho0_f * np.cos(b_sub)).astype(np.float32)[:, None, None]  # (K, 1, 1)

        theta = b_sub[:, None] + alpha_u[None, :]  # (K, N_u)
        ex = np.sin(theta).astype(np.float32)[:, :, None]  # (K, N_u, 1)
        ey = (-np.cos(theta)).astype(np.float32)[:, :, None]  # (K, N_u, 1)

        # Ray points in DICOM-CT-PD Cartesian (x, y) [mm]: shape (K, N_u, N_steps)
        x_pts = sx + t_grid[None, :, :] * ex
        y_pts = sy + t_grid[None, :, :] * ey

        # Radiological pixel indices: x_c = cx + (half_idx - c)*d_xy => c = half_idx - (x - cx)/d_xy
        col_f = half_idx_f - (x_pts - cx_f) * inv_dxy
        row_f = half_idx_f - (y_pts - cy_f) * inv_dxy

        valid = (
            (col_f >= 0.0)
            & (col_f <= max_idx_f)
            & (row_f >= 0.0)
            & (row_f <= max_idx_f)
        )
        col_c = np.clip(col_f, 0.0, max_idx_f)
        row_c = np.clip(row_f, 0.0, max_idx_f)

        c0 = np.floor(col_c).astype(np.int32)
        r0 = np.floor(row_c).astype(np.int32)
        c1 = np.minimum(c0 + 1, n_xy - 1)
        r1 = np.minimum(r0 + 1, n_xy - 1)

        wc = col_c - c0
        wr = row_c - r0

        v00 = img[r0, c0]
        v01 = img[r0, c1]
        v10 = img[r1, c0]
        v11 = img[r1, c1]

        vals = (
            (1.0 - wr) * ((1.0 - wc) * v00 + wc * v01)
            + wr * ((1.0 - wc) * v10 + wc * v11)
        )
        vals = np.where(valid, vals, np.float32(0.0))
        sino_out[k0:k1] = np.sum(vals, axis=2, dtype=np.float32) * np.float32(dt)

    return sino_out


def _project_fanbeam_slice_torch(
    mu_2d: np.ndarray,
    beta_1d: np.ndarray,
    u_coords_mm: np.ndarray,
    *,
    rho0_mm: float,
    d0_mm: float,
    fov_mm: float,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    step_factor: float = 0.5,
    require_cuda: bool = False,
    angle_chunk: int = 128,
) -> np.ndarray:
    """Vectorized 2D equispaced fan-beam projector in PyTorch (`torch` or `torch_cuda`, no silent fallback)."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required when backend is 'torch' or 'torch_cuda' (no silent fallback)."
        ) from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "backend='torch_cuda' was explicitly requested for forward projection, "
            "but torch.cuda.is_available() is False (no silent fallback)."
        )

    device = torch.device("cuda" if require_cuda else "cpu")
    img = np.asarray(mu_2d, dtype=np.float32)
    n_xy = int(img.shape[0])
    d_xy = float(fov_mm) / float(n_xy)
    half_idx = 0.5 * (n_xy - 1.0)
    cx = float(center_xy_mm[0])
    cy = float(center_xy_mm[1])

    alpha_u = np.arctan2(np.asarray(u_coords_mm, dtype=np.float64), float(d0_mm))
    t_iso = float(rho0_mm) * np.cos(alpha_u)

    r_bound = 0.5 * math.sqrt(2.0) * float(fov_mm) + math.hypot(cx, cy) + d_xy
    ds = max(0.1 * d_xy, float(step_factor) * d_xy)
    n_steps = max(8, int(math.ceil((2.0 * r_bound) / ds)) + 1)
    tau = np.linspace(-r_bound, r_bound, n_steps, dtype=np.float64)
    dt = float(tau[1] - tau[0]) if n_steps > 1 else float(2.0 * r_bound)

    n_beta = int(beta_1d.size)
    n_u = int(u_coords_mm.size)
    sino_out = np.zeros((n_beta, n_u), dtype=np.float32)

    with torch.no_grad():
        # Shape (1, 1, H, W) for grid_sample with align_corners=True
        img_t = torch.as_tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        alpha_t = torch.as_tensor(alpha_u, dtype=torch.float32, device=device)  # (N_u,)
        t_grid_t = torch.as_tensor(
            (t_iso[:, None] + tau[None, :]).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )  # (N_u, N_steps)

        scale_norm = float(2.0 / max(float(n_xy - 1), 1.0))
        inv_dxy = float(1.0 / d_xy)
        half_idx_f = float(half_idx)
        cx_f = float(cx)
        cy_f = float(cy)
        rho0_f = float(rho0_mm)

        chunk = max(1, int(angle_chunk))
        for k0 in range(0, n_beta, chunk):
            k1 = min(n_beta, k0 + chunk)
            b_sub = torch.as_tensor(beta_1d[k0:k1], dtype=torch.float32, device=device)  # (K,)
            sx = (-rho0_f * torch.sin(b_sub)).view(-1, 1, 1)
            sy = (rho0_f * torch.cos(b_sub)).view(-1, 1, 1)

            theta = b_sub.unsqueeze(1) + alpha_t.unsqueeze(0)  # (K, N_u)
            ex = torch.sin(theta).unsqueeze(2)
            ey = (-torch.cos(theta)).unsqueeze(2)

            x_pts = sx + t_grid_t.unsqueeze(0) * ex  # (K, N_u, N_steps)
            y_pts = sy + t_grid_t.unsqueeze(0) * ey

            col_f = half_idx_f - (x_pts - cx_f) * inv_dxy
            row_f = half_idx_f - (y_pts - cy_f) * inv_dxy
            valid_m = (
                (col_f >= 0.0)
                & (col_f <= float(n_xy - 1))
                & (row_f >= 0.0)
                & (row_f <= float(n_xy - 1))
            )

            # Normalize [0, N_xy - 1] -> [-1, +1] for align_corners=True
            gx = col_f * scale_norm - 1.0
            gy = row_f * scale_norm - 1.0
            grid = torch.stack([gx, gy], dim=-1).view(1, (k1 - k0) * n_u, n_steps, 2)

            sampled = F.grid_sample(
                img_t,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).view(k1 - k0, n_u, n_steps)
            sampled = torch.where(valid_m, sampled, torch.zeros_like(sampled))
            sino_chunk = torch.sum(sampled, dim=2) * float(dt)
            sino_out[k0:k1] = sino_chunk.detach().cpu().numpy().astype(np.float32)

    return sino_out


def _project_fanbeam_batched_torch(
    mu_slices: np.ndarray,
    beta_rad: np.ndarray,
    u_coords_mm: np.ndarray,
    *,
    rho0_mm: float,
    d0_mm: float,
    fov_mm: float,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    step_factor: float = 0.5,
    require_cuda: bool = False,
    angle_chunk: int = 64,
    progress_bar: Any = None,
) -> np.ndarray:
    """Batched multi-slice 2D equispaced fan-beam projector in PyTorch (`torch` or `torch_cuda`)."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required when backend is 'torch' or 'torch_cuda' (no silent fallback)."
        ) from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "backend='torch_cuda' was explicitly requested for forward projection, "
            "but torch.cuda.is_available() is False (no silent fallback)."
        )

    n_slices, n_xy, _ = mu_slices.shape
    n_beta = int(beta_rad.shape[1])
    n_u = int(u_coords_mm.size)
    out = np.zeros((n_slices, n_beta, n_u), dtype=np.float32)
    if n_slices == 0 or n_beta == 0 or n_u == 0:
        return out

    device = torch.device("cuda" if require_cuda else "cpu")
    d_xy = float(fov_mm) / float(n_xy)
    half_idx = 0.5 * (n_xy - 1.0)
    cx = float(center_xy_mm[0])
    cy = float(center_xy_mm[1])

    alpha_u = np.arctan2(np.asarray(u_coords_mm, dtype=np.float64), float(d0_mm))
    t_iso = float(rho0_mm) * np.cos(alpha_u)

    r_bound = 0.5 * math.sqrt(2.0) * float(fov_mm) + math.hypot(cx, cy) + d_xy
    ds = max(0.1 * d_xy, float(step_factor) * d_xy)
    n_steps = max(8, int(math.ceil((2.0 * r_bound) / ds)) + 1)
    tau = np.linspace(-r_bound, r_bound, n_steps, dtype=np.float64)
    dt = float(tau[1] - tau[0]) if n_steps > 1 else float(2.0 * r_bound)

    if require_cuda:
        vram_bytes = torch.cuda.get_device_properties(device).total_memory
        slice_batch = 16 if (vram_bytes >= 22 * (1024**3) and n_xy <= 256) else 8
    else:
        slice_batch = 4
    k_step = max(16, min(int(angle_chunk), 64 if n_xy > 256 else 96))

    with torch.no_grad():
        alpha_t = torch.as_tensor(alpha_u, dtype=torch.float32, device=device).view(1, 1, n_u)
        t_grid_t = torch.as_tensor(
            (t_iso[:, None] + tau[None, :]).astype(np.float32),
            dtype=torch.float32,
            device=device,
        ).view(1, 1, n_u, n_steps)

        scale_norm = float(2.0 / max(float(n_xy - 1), 1.0))
        inv_dxy = float(1.0 / d_xy)
        half_idx_f = float(half_idx)
        cx_f = float(cx)
        cy_f = float(cy)
        rho0_f = float(rho0_mm)
        max_xy_f = float(n_xy - 1)

        for s0 in range(0, n_slices, slice_batch):
            s1 = min(n_slices, s0 + slice_batch)
            bsz = s1 - s0
            img_t = torch.as_tensor(
                np.ascontiguousarray(mu_slices[s0:s1], dtype=np.float32), device=device
            ).unsqueeze(1)
            beta_b = torch.as_tensor(
                np.ascontiguousarray(beta_rad[s0:s1], dtype=np.float32), device=device
            )
            sino_gpu = torch.empty((bsz, n_beta, n_u), dtype=torch.float32, device=device)

            for k0 in range(0, n_beta, k_step):
                k1 = min(n_beta, k0 + k_step)
                kc = k1 - k0
                b_sub = beta_b[:, k0:k1]
                sx = (-rho0_f * torch.sin(b_sub)).view(bsz, kc, 1, 1)
                sy = (rho0_f * torch.cos(b_sub)).view(bsz, kc, 1, 1)

                theta = b_sub.unsqueeze(2) + alpha_t
                ex = torch.sin(theta).unsqueeze(3)
                ey = (-torch.cos(theta)).unsqueeze(3)

                x_pts = sx + t_grid_t * ex
                y_pts = sy + t_grid_t * ey

                col_f = half_idx_f - (x_pts - cx_f) * inv_dxy
                row_f = half_idx_f - (y_pts - cy_f) * inv_dxy
                valid_m = (
                    (col_f >= 0.0)
                    & (col_f <= max_xy_f)
                    & (row_f >= 0.0)
                    & (row_f <= max_xy_f)
                )

                gx = col_f * scale_norm - 1.0
                gy = row_f * scale_norm - 1.0
                grid = torch.stack([gx, gy], dim=-1).view(bsz, kc * n_u, n_steps, 2)

                sampled = F.grid_sample(
                    img_t,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).view(bsz, kc, n_u, n_steps)
                sampled = torch.where(valid_m, sampled, torch.zeros_like(sampled))
                sino_gpu[:, k0:k1, :] = torch.sum(sampled, dim=3) * float(dt)

            out[s0:s1] = sino_gpu.detach().cpu().numpy().astype(np.float32)
            if progress_bar is not None:
                progress_bar.update(bsz)

    return out


def forward_project_fanbeam_2d(
    mu_slices: np.ndarray,
    beta_rad: np.ndarray,
    u_coords_mm: np.ndarray,
    *,
    rho0_mm: float,
    d0_mm: float,
    fov_mm: float,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    step_factor: float = 0.5,
    backend: str = "numpy",
    angle_chunk: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Forward-project 2D or 3D axial attenuation slice(s) `mu [mm^-1]` into fan-beam sinogram(s).

    Parameters
    ----------
    mu_slices : np.ndarray
        Linear attenuation array `[mm^-1]` of shape `(N_xy, N_xy)` or `(S, N_xy, N_xy)` in the
        radiological coordinate grid (`Col 0 = +x_CTPD` Right, `Row 0 = +y_CTPD` Anterior).
    beta_rad : np.ndarray
        Fan-beam source angles `[rad]` of shape `(N_beta,)` or `(S, N_beta)`.
    u_coords_mm : np.ndarray
        Virtual flat-detector transverse coordinates `[mm]` of shape `(N_u,)`.
    """
    backend_norm = backend.strip().lower()
    if backend_norm not in SUPPORTED_FBP_BACKENDS:
        raise ValueError(
            f"Unsupported backend={backend!r}; expected one of {sorted(SUPPORTED_FBP_BACKENDS)}"
        )
    if float(rho0_mm) <= 0.0 or float(d0_mm) <= 0.0 or float(fov_mm) <= 0.0:
        raise ValueError("rho0_mm, d0_mm, and fov_mm must all be strictly positive")

    arr = np.asarray(mu_slices, dtype=np.float32)
    single_2d = arr.ndim == 2
    if single_2d:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Expected (N_xy, N_xy) or (S, N_xy, N_xy) mu_slices, got {mu_slices.shape}")

    n_slices = int(arr.shape[0])
    u_arr = np.asarray(u_coords_mm, dtype=np.float64)
    b_arr = np.asarray(beta_rad, dtype=np.float64)
    if b_arr.ndim == 1:
        b_arr = np.broadcast_to(b_arr[None, :], (n_slices, b_arr.size)).copy()
    elif b_arr.shape[0] != n_slices:
        raise ValueError(f"beta_rad shape {b_arr.shape} incompatible with {n_slices} slices")

    n_beta = int(b_arr.shape[1])
    n_u = int(u_arr.size)
    out = np.zeros((n_slices, n_beta, n_u), dtype=np.float32)

    with ProgressReporter(
        total=n_slices,
        desc="🔦 Forward-projecting 2D Fan-Beam",
        unit="slice",
        unit_scale=False,
        colour="magenta",
        enabled=show_progress and n_slices > 0,
    ) as bar:
        if backend_norm in {"torch", "torch_cuda"} and n_slices > 1:
            out = _project_fanbeam_batched_torch(
                arr,
                b_arr,
                u_arr,
                rho0_mm=rho0_mm,
                d0_mm=d0_mm,
                fov_mm=fov_mm,
                center_xy_mm=center_xy_mm,
                step_factor=step_factor,
                require_cuda=(backend_norm == "torch_cuda"),
                angle_chunk=max(angle_chunk, 64),
                progress_bar=bar,
            )
        else:
            for s_idx in range(n_slices):
                if backend_norm == "numpy":
                    out[s_idx] = _project_fanbeam_slice_numpy(
                        arr[s_idx],
                        b_arr[s_idx],
                        u_arr,
                        rho0_mm=rho0_mm,
                        d0_mm=d0_mm,
                        fov_mm=fov_mm,
                        center_xy_mm=center_xy_mm,
                        step_factor=step_factor,
                        angle_chunk=angle_chunk,
                    )
                else:
                    out[s_idx] = _project_fanbeam_slice_torch(
                        arr[s_idx],
                        b_arr[s_idx],
                        u_arr,
                        rho0_mm=rho0_mm,
                        d0_mm=d0_mm,
                        fov_mm=fov_mm,
                        center_xy_mm=center_xy_mm,
                        step_factor=step_factor,
                        require_cuda=(backend_norm == "torch_cuda"),
                        angle_chunk=max(angle_chunk, 128),
                    )
                bar.update(1)

    return out[0] if single_2d else out


def forward_project_fanbeam_2d_scalar_ref(
    mu_2d: np.ndarray,
    beta_1d: np.ndarray,
    u_coords_mm: np.ndarray,
    *,
    rho0_mm: float,
    d0_mm: float,
    fov_mm: float,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    step_factor: float = 0.5,
) -> np.ndarray:
    """Explicit scalar loop reference for 2D fan-beam forward projection (`validation_protocol.md`)."""
    img = np.asarray(mu_2d, dtype=np.float32)
    n_xy = int(img.shape[0])
    d_xy = float(fov_mm) / float(n_xy)
    half_idx = 0.5 * (n_xy - 1.0)
    cx = float(center_xy_mm[0])
    cy = float(center_xy_mm[1])

    r_bound = 0.5 * math.sqrt(2.0) * float(fov_mm) + math.hypot(cx, cy) + d_xy
    ds = max(0.1 * d_xy, float(step_factor) * d_xy)
    n_steps = max(8, int(math.ceil((2.0 * r_bound) / ds)) + 1)
    tau = np.linspace(-r_bound, r_bound, n_steps, dtype=np.float64)
    dt = float(tau[1] - tau[0]) if n_steps > 1 else float(2.0 * r_bound)

    n_beta = int(beta_1d.size)
    n_u = int(u_coords_mm.size)
    out = np.zeros((n_beta, n_u), dtype=np.float32)

    for k in range(n_beta):
        b = float(beta_1d[k])
        sx = -float(rho0_mm) * math.sin(b)
        sy = float(rho0_mm) * math.cos(b)
        for m in range(n_u):
            u = float(u_coords_mm[m])
            alpha = math.atan2(u, float(d0_mm))
            t_iso = float(rho0_mm) * math.cos(alpha)
            theta = b + alpha
            ex = math.sin(theta)
            ey = -math.cos(theta)
            acc = 0.0
            for n in range(n_steps):
                t = t_iso + float(tau[n])
                xv = sx + t * ex
                yv = sy + t * ey
                cf = half_idx - (xv - cx) / d_xy
                rf = half_idx - (yv - cy) / d_xy
                if 0.0 <= cf <= float(n_xy - 1) and 0.0 <= rf <= float(n_xy - 1):
                    c0 = int(math.floor(cf))
                    r0 = int(math.floor(rf))
                    c1 = min(c0 + 1, n_xy - 1)
                    r1 = min(r0 + 1, n_xy - 1)
                    wc = cf - c0
                    wr = rf - r0
                    v = (1.0 - wr) * (
                        (1.0 - wc) * float(img[r0, c0]) + wc * float(img[r0, c1])
                    ) + wr * ((1.0 - wc) * float(img[r1, c0]) + wc * float(img[r1, c1]))
                    acc += v
            out[k, m] = np.float32(acc * dt)

    return out


def synthesize_itf_helical_geometry(
    base_geometry: HelicalGeometry,
    target_pitch: float,
    *,
    z_center_mm: float | None = None,
    z_span_mm: float | None = None,
    min_rotations: float = 1.25,
) -> HelicalGeometry:
    """Synthesize a high-pitch `ITF (Increased Table Feed)` `HelicalGeometry` (Fig. 1(c)).

    Preserves the exact detector geometry (`n_rows`, `n_cols`, `du_mm`, `dv_mm`, `rho0_mm`, `d0_mm`,
    `center_col_0based`, `center_row_0based`, `detector_shape`), angular sampling rate
    `angular_steps_per_rotation`, and periodic 3D Flying Focal Spot pattern `(drho, dphi, dz)`,
    while scaling the table travel per rotation to `target_pitch * coverage_isocenter_mm`.
    """
    if float(target_pitch) <= 0.0:
        raise ValueError(f"target_pitch must be positive, got {target_pitch}")

    traj = base_geometry.trajectory_metrics()
    rot_dir = 1.0 if float(traj["rotation_direction"]) >= 0.0 else -1.0
    table_dir = 1.0 if float(traj["table_direction"]) >= 0.0 else -1.0

    n_rot = int(base_geometry.angular_steps_per_rotation)
    dphi_step = rot_dir * (2.0 * math.pi / float(n_rot))

    table_feed_per_rot = float(target_pitch) * float(base_geometry.coverage_isocenter_mm)
    dz_step = table_dir * (table_feed_per_rot / float(n_rot))

    z_mid = (
        float(0.5 * (np.min(base_geometry.z0_mm) + np.max(base_geometry.z0_mm)))
        if z_center_mm is None
        else float(z_center_mm)
    )
    req_span = (
        float(max(abs(base_geometry.z0_mm[-1] - base_geometry.z0_mm[0]), 1.25 * table_feed_per_rot))
        if z_span_mm is None
        else float(z_span_mm) + 1.25 * table_feed_per_rot
    )
    n_views_span = int(math.ceil(req_span / max(abs(dz_step), 1e-6))) + 1
    n_views_min = int(math.ceil(float(min_rotations) * float(n_rot)))
    # Align n_views to a multiple of 8 so periodic FFS states remain balanced
    n_views = int(math.ceil(max(n_views_span, n_views_min) / 8.0) * 8)

    idx = np.arange(n_views, dtype=np.float64)
    i_ref = int(np.argmin(np.abs(base_geometry.z0_mm - z_mid)))
    phi0_mid = float(base_geometry.phi0_unwrapped_rad[i_ref])
    mid_i = float(n_views // 2)
    phi0_unwrap = phi0_mid + (idx - mid_i) * dphi_step
    phi0_wrapped = ((phi0_unwrap + math.pi) % (2.0 * math.pi)) - math.pi
    z0_arr = z_mid + (idx - mid_i) * dz_step

    # Preserve the periodic FFS shift pattern from base_geometry
    base_len = int(base_geometry.n_views)
    ffs_idx = np.arange(n_views, dtype=np.int64) % base_len
    dphi_arr = base_geometry.dphi_rad[ffs_idx].copy()
    dz_arr = base_geometry.dz_mm[ffs_idx].copy()
    drho_arr = base_geometry.drho_mm[ffs_idx].copy()

    return HelicalGeometry(
        series_instance_uid=f"{base_geometry.series_instance_uid}.ITF_p{target_pitch:.2f}",
        detector_shape=base_geometry.detector_shape,
        n_rows=base_geometry.n_rows,
        n_cols=base_geometry.n_cols,
        du_mm=base_geometry.du_mm,
        dv_mm=base_geometry.dv_mm,
        rho0_mm=base_geometry.rho0_mm,
        d0_mm=base_geometry.d0_mm,
        center_col_0based=base_geometry.center_col_0based,
        center_row_0based=base_geometry.center_row_0based,
        ffs_mode=base_geometry.ffs_mode,
        angular_steps_per_rotation=n_rot,
        water_mu_mm_inv=base_geometry.water_mu_mm_inv,
        nominal_spiral_pitch=float(target_pitch),
        log_flag="YES",
        rescale_slope=base_geometry.rescale_slope,
        rescale_intercept=base_geometry.rescale_intercept,
        instance_numbers=np.arange(1, n_views + 1, dtype=np.int32),
        phi0_rad=phi0_wrapped,
        phi0_unwrapped_rad=phi0_unwrap,
        z0_mm=z0_arr,
        dphi_rad=dphi_arr,
        dz_mm=dz_arr,
        drho_mm=drho_arr,
    )


def forward_project_helical_3d(
    mu_volume_zyx: np.ndarray,
    z_grid_mm: np.ndarray,
    y_mm: np.ndarray,
    x_mm: np.ndarray,
    geometry: HelicalGeometry,
    *,
    step_mm: float | None = None,
    z_boundary_mode: str = "clamp",
    backend: str = "numpy",
    view_batch: int | None = None,
    progress_callback: Any = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Ray-driven 3D Helical Cone-Beam Forward Projector `P_helical: mu(z, y, x) -> g(i, r, c)`.

    Uses exact 3D source positions `S(i)` (including 3D Flying Focal Spot shifts `drho, dphi, dz`)
    and 3D detector element coordinates `D(i, r, c)` from `geometry.ray_origins_and_directions(...)`.
    """
    backend_norm = backend.strip().lower()
    if backend_norm not in SUPPORTED_FBP_BACKENDS:
        raise ValueError(
            f"Unsupported backend={backend!r}; expected one of {sorted(SUPPORTED_FBP_BACKENDS)}"
        )
    if z_boundary_mode not in {"clamp", "zero"}:
        raise ValueError(f"z_boundary_mode must be 'clamp' or 'zero', got {z_boundary_mode!r}")

    vol = np.asarray(mu_volume_zyx, dtype=np.float32)
    z_arr = np.asarray(z_grid_mm, dtype=np.float64)
    y_arr = np.asarray(y_mm, dtype=np.float64)
    x_arr = np.asarray(x_mm, dtype=np.float64)
    if vol.ndim != 3 or vol.shape != (z_arr.size, y_arr.size, x_arr.size):
        raise ValueError(
            f"mu_volume_zyx shape {vol.shape} does not match (Nz={z_arr.size}, Ny={y_arr.size}, Nx={x_arr.size})"
        )

    nz, ny, nx = vol.shape
    dx = abs(float(x_arr[0] - x_arr[-1])) / max(float(nx - 1), 1.0)
    dy = abs(float(y_arr[0] - y_arr[-1])) / max(float(ny - 1), 1.0)
    dz = abs(float(z_arr[-1] - z_arr[0])) / max(float(nz - 1), 1.0) if nz > 1 else dx

    ds = float(0.85 * min(dx, dy)) if step_mm is None else float(step_mm)
    r_xy = 0.5 * max(
        abs(float(x_arr[0] - x_arr[-1])) + dx,
        abs(float(y_arr[0] - y_arr[-1])) + dy,
    )
    t_min = max(10.0, float(geometry.rho0_mm) - r_xy - 8.0)
    t_max = float(geometry.rho0_mm) + r_xy + 8.0
    n_steps = max(8, int(math.ceil((t_max - t_min) / ds)) + 1)
    t_samples = np.linspace(t_min, t_max, n_steps, dtype=np.float32)
    dt = float(t_samples[1] - t_samples[0]) if n_steps > 1 else float(t_max - t_min)

    # Coordinate mapping: x_arr[0] -> c=0, x_arr[-1] -> c=nx-1
    x0 = float(x_arr[0])
    inv_span_x = float((nx - 1) / (x_arr[-1] - x_arr[0])) if nx > 1 and x_arr[-1] != x_arr[0] else 0.0
    y0 = float(y_arr[0])
    inv_span_y = float((ny - 1) / (y_arr[-1] - y_arr[0])) if ny > 1 and y_arr[-1] != y_arr[0] else 0.0
    z0 = float(z_arr[0])
    inv_span_z = float((nz - 1) / (z_arr[-1] - z_arr[0])) if nz > 1 and z_arr[-1] != z_arr[0] else 0.0

    projections = np.zeros((geometry.n_views, geometry.n_rows, geometry.n_cols), dtype=np.float32)

    if backend_norm in {"torch", "torch_cuda"}:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required when backend is 'torch' or 'torch_cuda' (no silent fallback)."
            ) from error
        if backend_norm == "torch_cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "backend='torch_cuda' was requested for 3D helical forward projection, "
                "but torch.cuda.is_available() is False (no silent fallback)."
            )
        device = torch.device("cuda" if backend_norm == "torch_cuda" else "cpu")
        if view_batch is not None:
            vb = max(1, int(view_batch))
        elif device.type == "cuda":
            vram_bytes = torch.cuda.get_device_properties(device).total_memory
            samples_per_view = geometry.n_rows * geometry.n_cols * n_steps
            if vram_bytes >= 60 * (1024**3) and samples_per_view <= 35_000_000:
                vb = 24
            elif vram_bytes >= 22 * (1024**3) and samples_per_view <= 35_000_000:
                vb = 12
            else:
                vb = 4
        else:
            vb = 2

        with torch.no_grad():
            vol_t = torch.as_tensor(vol, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            t_t = torch.as_tensor(t_samples, dtype=torch.float32, device=device).view(1, 1, 1, -1)
            sx_norm = float(2.0 / max(float(nx - 1), 1.0))
            sy_norm = float(2.0 / max(float(ny - 1), 1.0))
            sz_norm = float(2.0 / max(float(nz - 1), 1.0)) if nz > 1 else 0.0
            max_nx = float(nx - 1)
            max_ny = float(ny - 1)
            max_nz = float(max(nz - 1, 0))

            with ProgressReporter(
                total=geometry.n_views,
                desc="🌀 3D Helical Forward Projection (Torch)",
                unit="view",
                unit_scale=False,
                colour="magenta",
                enabled=show_progress and geometry.n_views > 0,
            ) as bar:
                for i0 in range(0, geometry.n_views, vb):
                    i1 = min(geometry.n_views, i0 + vb)
                    kv = i1 - i0
                    src, dirs, _ = geometry.ray_origins_and_directions(slice(i0, i1))
                    src_t = torch.as_tensor(src, dtype=torch.float32, device=device)  # (kv, 3)
                    dirs_t = torch.as_tensor(dirs, dtype=torch.float32, device=device)  # (kv, R, C, 3)
                    x_pts = src_t[:, 0].view(kv, 1, 1, 1) + t_t * dirs_t[:, :, :, 0:1]  # (kv, R, C, S)
                    y_pts = src_t[:, 1].view(kv, 1, 1, 1) + t_t * dirs_t[:, :, :, 1:2]
                    z_pts = src_t[:, 2].view(kv, 1, 1, 1) + t_t * dirs_t[:, :, :, 2:3]

                    cf = (x_pts - x0) * inv_span_x
                    rf = (y_pts - y0) * inv_span_y
                    zf = (z_pts - z0) * inv_span_z if nz > 1 else torch.zeros_like(x_pts)

                    in_xy = (cf >= 0.0) & (cf <= max_nx) & (rf >= 0.0) & (rf <= max_ny)
                    if z_boundary_mode == "clamp":
                        zf_use = torch.clamp(zf, 0.0, max_nz)
                        valid_m = in_xy
                    else:
                        zf_use = zf
                        valid_m = in_xy & (zf >= 0.0) & (zf <= max_nz)

                    gx = cf * sx_norm - 1.0
                    gy = rf * sy_norm - 1.0
                    gz = (zf_use * sz_norm - 1.0) if nz > 1 else torch.zeros_like(gx)
                    grid = torch.stack([gx, gy, gz], dim=-1).view(
                        1, kv * geometry.n_rows, geometry.n_cols, n_steps, 3
                    )
                    sampled = F.grid_sample(
                        vol_t,
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    ).view(kv, geometry.n_rows, geometry.n_cols, n_steps)
                    sampled = torch.where(valid_m, sampled, torch.zeros_like(sampled))
                    proj_kv = torch.sum(sampled, dim=3) * float(dt)
                    projections[i0:i1] = proj_kv.detach().cpu().numpy().astype(np.float32)
                    bar.update(kv)
                    if progress_callback is not None:
                        progress_callback(i1, geometry.n_views)
        return projections

    # Pure NumPy vectorized per-view implementation
    with ProgressReporter(
        total=geometry.n_views,
        desc="🌀 3D Helical Forward Projection (NumPy)",
        unit="view",
        unit_scale=False,
        colour="magenta",
        enabled=show_progress and geometry.n_views > 0,
    ) as bar:
        for i in range(geometry.n_views):
            src, dirs, _ = geometry.ray_origins_and_directions(i)
            dirs_f = dirs.astype(np.float32)  # (R, C, 3)
            x_pts = np.float32(src[0]) + dirs_f[:, :, 0:1] * t_samples[None, None, :]
            y_pts = np.float32(src[1]) + dirs_f[:, :, 1:2] * t_samples[None, None, :]
            z_pts = np.float32(src[2]) + dirs_f[:, :, 2:3] * t_samples[None, None, :]

            cf = (x_pts - np.float32(x0)) * np.float32(inv_span_x)
            rf = (y_pts - np.float32(y0)) * np.float32(inv_span_y)
            zf = (
                (z_pts - np.float32(z0)) * np.float32(inv_span_z)
                if nz > 1
                else np.zeros_like(cf)
            )

            valid = (
                (cf >= 0.0)
                & (cf <= np.float32(nx - 1))
                & (rf >= 0.0)
                & (rf <= np.float32(ny - 1))
            )
            if z_boundary_mode == "zero" and nz > 1:
                valid = valid & (zf >= 0.0) & (zf <= np.float32(nz - 1))

            cc = np.clip(cf, 0.0, np.float32(nx - 1))
            rc = np.clip(rf, 0.0, np.float32(ny - 1))
            zc = np.clip(zf, 0.0, np.float32(max(nz - 1, 0)))

            c0_i = np.floor(cc).astype(np.int32)
            r0_i = np.floor(rc).astype(np.int32)
            z0_i = np.floor(zc).astype(np.int32)
            c1_i = np.minimum(c0_i + 1, nx - 1)
            r1_i = np.minimum(r0_i + 1, ny - 1)
            z1_i = np.minimum(z0_i + 1, max(nz - 1, 0))

            wc = cc - c0_i
            wr = rc - r0_i
            wz = zc - z0_i

            v000 = vol[z0_i, r0_i, c0_i]
            v001 = vol[z0_i, r0_i, c1_i]
            v010 = vol[z0_i, r1_i, c0_i]
            v011 = vol[z0_i, r1_i, c1_i]
            v0 = (1.0 - wr) * ((1.0 - wc) * v000 + wc * v001) + wr * (
                (1.0 - wc) * v010 + wc * v011
            )
            if nz > 1:
                v100 = vol[z1_i, r0_i, c0_i]
                v101 = vol[z1_i, r0_i, c1_i]
                v110 = vol[z1_i, r1_i, c0_i]
                v111 = vol[z1_i, r1_i, c1_i]
                v1 = (1.0 - wr) * ((1.0 - wc) * v100 + wc * v101) + wr * (
                    (1.0 - wc) * v110 + wc * v111
                )
                vals = (1.0 - wz) * v0 + wz * v1
            else:
                vals = v0

            vals = np.where(valid, vals, np.float32(0.0))
            projections[i] = np.sum(vals, axis=2, dtype=np.float32) * np.float32(dt)
            bar.update(1)
            if progress_callback is not None:
                progress_callback(i + 1, geometry.n_views)

    return projections


def _blend_data_consistent_reprojected_sinogram(
    s_hp_raw: np.ndarray,
    mask_hp: np.ndarray,
    s_reproj: np.ndarray,
    *,
    s_hp_extrap: np.ndarray | None = None,
    feather_passes: int = 4,
) -> np.ndarray:
    """Combine measured rays `s_hp_raw[mask_hp]` with boundary-anchored reprojected rays `s_reproj[~mask_hp]`."""
    raw = np.asarray(s_hp_raw, dtype=np.float32)
    mask = np.asarray(mask_hp, dtype=bool)
    rep = np.asarray(s_reproj, dtype=np.float32)
    ext = np.asarray(s_hp_extrap, dtype=np.float32) if s_hp_extrap is not None else None

    out = np.where(mask, raw, rep).astype(np.float64)
    unmeasured = ~mask
    if not np.any(unmeasured):
        return out.astype(np.float32)

    n_slices, _, _ = out.shape
    for s_idx in range(n_slices):
        um = unmeasured[s_idx]
        if not np.any(um):
            continue
        cur_mask = mask[s_idx].copy()
        cur = out[s_idx].copy()
        rep_s = rep[s_idx].astype(np.float64)

        # Step 0: Pre-fill any completely unmeasured lateral border column (e.g. quarter-offset edge)
        # from its nearest measured column at angles where that neighbor is measured.
        col_has_meas = np.any(cur_mask, axis=0)
        if not np.all(col_has_meas) and np.any(col_has_meas):
            meas_cols = np.where(col_has_meas)[0]
            for m_un in np.where(~col_has_meas)[0]:
                nearest_m = int(meas_cols[np.argmin(np.abs(meas_cols - m_un))])
                nb_mask = cur_mask[:, nearest_m]
                cur[nb_mask, m_un] = cur[nb_mask, nearest_m]
                cur_mask[nb_mask, m_un] = True

        # Step 1: Vectorized boundary-anchored reprojection along angle beta:
        # s_corr(beta, u) = s_reproj(beta, u) + interp_beta[s_raw - s_reproj](beta, u)
        diff_interp_2d, col_filled = _interp_columns_linear_2d(cur - rep_s, cur_mask)
        if np.any(col_filled):
            reproj_anchored = np.maximum(0.0, rep_s + diff_interp_2d)
            if ext is not None:
                blended = 0.5 * reproj_anchored + 0.5 * ext[s_idx].astype(np.float64)
            else:
                blended = reproj_anchored
            cur = np.where(col_filled, blended, cur)

        for _ in range(max(0, int(feather_passes))):
            pad_u = np.pad(cur, ((0, 0), (1, 1)), mode="edge")
            sm_u = 0.25 * pad_u[:, :-2] + 0.50 * pad_u[:, 1:-1] + 0.25 * pad_u[:, 2:]
            cur = np.where(um, sm_u, cur)

            pad_k = np.pad(cur, ((1, 1), (0, 0)), mode="edge")
            sm_k = 0.25 * pad_k[:-2, :] + 0.50 * pad_k[1:-1, :] + 0.25 * pad_k[2:, :]
            cur = np.where(um, sm_k, cur)
        out[s_idx] = np.where(mask[s_idx], raw[s_idx], np.maximum(0.0, cur))

    return out.astype(np.float32)


def compute_sinogram_metrics(
    pred_sino: np.ndarray,
    target_sino: np.ndarray,
    *,
    measured_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute quantitative line-integral accuracy metrics across full sinogram and missing wedge."""
    p = np.asarray(pred_sino, dtype=np.float64)
    t = np.asarray(target_sino, dtype=np.float64)
    diff = p - t
    mae_all = float(np.mean(np.abs(diff)))
    rmse_all = math.sqrt(float(np.mean(diff * diff)))

    if measured_mask is not None:
        m = np.asarray(measured_mask, dtype=bool)
        um = ~m
        if np.any(um):
            diff_um = diff[um]
            mae_um = float(np.mean(np.abs(diff_um)))
            rmse_um = math.sqrt(float(np.mean(diff_um * diff_um)))
        else:
            mae_um = 0.0
            rmse_um = 0.0
    else:
        mae_um = mae_all
        rmse_um = rmse_all

    return {
        "mae_all": round(mae_all, 6),
        "rmse_all": round(rmse_all, 6),
        "mae_missing_wedge": round(mae_um, 6),
        "rmse_missing_wedge": round(rmse_um, 6),
    }


def build_dual_domain_pair_from_ssrb(
    ssrb_ref: SSRBResult,
    ssrb_hp: SSRBResult,
    *,
    scenario: str,
    target_pitch: float,
    effective_pitch: float,
    native_pitch: float,
    active_row_range: tuple[float, float],
    image_size: int = 512,
    fov_mm: float | None = None,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    water_mu_mm_inv: float = 0.0192,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    inpaint_mode: str = "conjugate_feather",
    lateral_pad_cols: int = 96,
    backend: str = "numpy",
    fbp_ref_precomputed: FBPReconstructionResult | None = None,
    show_progress: bool = False,
) -> HighPitchSimulationPair:
    """Execute AAPM 2026 Fig. 2(a)->(b)->(c)->(d) pipeline from reference and high-pitch SSRB results."""
    scen = scenario.strip().lower()
    if scen not in SUPPORTED_HIGH_PITCH_SCENARIOS:
        raise ValueError(
            f"Unsupported scenario={scenario!r}; expected one of {sorted(SUPPORTED_HIGH_PITCH_SCENARIOS)}"
        )
    fname = normalize_filter_name(filter_name)

    # 1. Ground-Truth / Reference target (s* and x*)
    prep_ref = extrapolate_sinogram(
        ssrb_ref, inpaint_mode="conjugate_feather", lateral_pad_cols=lateral_pad_cols
    )
    s_target = prep_ref.completed_sinograms.astype(np.float32)

    if fbp_ref_precomputed is not None:
        fbp_ref = fbp_ref_precomputed
    else:
        fbp_ref = reconstruct_fanbeam_fbp(
            ssrb_ref,
            image_size=image_size,
            fov_mm=fov_mm,
            center_xy_mm=center_xy_mm,
            water_mu_mm_inv=water_mu_mm_inv,
            filter_name=fname,
            cutoff_ratio=cutoff_ratio,
            inpaint_mode="conjugate_feather",
            lateral_pad_cols=lateral_pad_cols,
            backend=backend,
            show_progress=show_progress,
        )

    # 2. High-Pitch Fig. 2(a) & Fig. 2(b): Raw SSRB sinogram + Extrapolation + Parker weighting
    s_hp_raw = ssrb_hp.sinograms.copy()
    mask_hp = ssrb_hp.measured_mask.copy()
    prep_hp = extrapolate_sinogram(
        ssrb_hp, inpaint_mode=inpaint_mode, lateral_pad_cols=lateral_pad_cols
    )
    s_hp_extrap = prep_hp.completed_sinograms.astype(np.float32)

    s_hp_parker = np.empty_like(s_hp_extrap, dtype=np.float32)
    for s_idx in range(ssrb_hp.n_slices):
        w_p = compute_parker_weights(
            ssrb_hp.beta_rad[s_idx],
            ssrb_hp.u_coords_mm,
            ssrb_hp.d0_mm,
            angular_mode=ssrb_hp.angular_mode,
        )
        s_hp_parker[s_idx] = (s_hp_extrap[s_idx].astype(np.float64) * w_p).astype(np.float32)

    # 3. High-Pitch Fig. 2(c): Preliminary FBP reconstruction (and raw no-fill ablation)
    fbp_prelim = reconstruct_fanbeam_fbp(
        ssrb_hp,
        image_size=fbp_ref.image_size,
        fov_mm=fbp_ref.fov_mm,
        center_xy_mm=fbp_ref.center_xy_mm,
        water_mu_mm_inv=water_mu_mm_inv,
        filter_name=fname,
        cutoff_ratio=cutoff_ratio,
        inpaint_mode=inpaint_mode,
        lateral_pad_cols=lateral_pad_cols,
        backend=backend,
        show_progress=show_progress,
    )
    fbp_raw_nofill = reconstruct_fanbeam_fbp(
        ssrb_hp,
        image_size=fbp_ref.image_size,
        fov_mm=fbp_ref.fov_mm,
        center_xy_mm=fbp_ref.center_xy_mm,
        water_mu_mm_inv=water_mu_mm_inv,
        filter_name=fname,
        cutoff_ratio=cutoff_ratio,
        inpaint_mode="none",
        lateral_pad_cols=lateral_pad_cols,
        backend=backend,
        show_progress=False,
    )

    # 4. High-Pitch Fig. 2(d): 2D Fan-Beam Reprojection of non-negative mu_prelim
    # Clip negative streak undershoots below air (mu < 0) before forward projection!
    mu_prelim_nonneg = np.maximum(fbp_prelim.mu_slices, np.float32(0.0))
    same_grid = (
        s_hp_raw.shape == s_target.shape
        and np.allclose(ssrb_hp.beta_rad, ssrb_ref.beta_rad, atol=1e-4)
        and np.allclose(ssrb_hp.u_coords_mm, ssrb_ref.u_coords_mm, atol=1e-4)
    )
    active_beta_rad = ssrb_ref.beta_rad if same_grid else ssrb_hp.beta_rad
    active_u_coords = ssrb_ref.u_coords_mm if same_grid else ssrb_hp.u_coords_mm
    if not same_grid:
        mu_target_nonneg = np.maximum(fbp_ref.mu_slices, np.float32(0.0))
        s_target = forward_project_fanbeam_2d(
            mu_target_nonneg,
            active_beta_rad,
            active_u_coords,
            rho0_mm=ssrb_hp.rho0_mm,
            d0_mm=ssrb_hp.d0_mm,
            fov_mm=fbp_ref.fov_mm,
            center_xy_mm=fbp_ref.center_xy_mm,
            backend=backend,
            show_progress=False,
        )
    s_reproj = forward_project_fanbeam_2d(
        mu_prelim_nonneg,
        active_beta_rad,
        active_u_coords,
        rho0_mm=ssrb_hp.rho0_mm,
        d0_mm=ssrb_hp.d0_mm,
        fov_mm=fbp_ref.fov_mm,
        center_xy_mm=fbp_ref.center_xy_mm,
        backend=backend,
        show_progress=show_progress,
    )
    s_reproj_dc = _blend_data_consistent_reprojected_sinogram(
        s_hp_raw, mask_hp, s_reproj, s_hp_extrap=s_hp_extrap
    )
    eval_mask = mask_hp

    prelim_metrics = compute_reconstruction_metrics(
        fbp_prelim.hu_slices, fbp_ref.hu_slices, fov_mask=fbp_ref.fov_mask
    )
    raw_metrics = compute_reconstruction_metrics(
        fbp_raw_nofill.hu_slices, fbp_ref.hu_slices, fov_mask=fbp_ref.fov_mask
    )

    sino_raw_m = compute_sinogram_metrics(s_hp_raw, s_target, measured_mask=eval_mask)
    sino_rep_m = compute_sinogram_metrics(s_reproj, s_target, measured_mask=eval_mask)
    sino_dc_m = compute_sinogram_metrics(s_reproj_dc, s_target, measured_mask=eval_mask)

    sino_metrics = {
        "raw_missing_wedge_mae": sino_raw_m["mae_missing_wedge"],
        "raw_all_mae": sino_raw_m["mae_all"],
        "reproj_all_mae": sino_rep_m["mae_all"],
        "reproj_all_rmse": sino_rep_m["rmse_all"],
        "reproj_missing_wedge_mae": sino_rep_m["mae_missing_wedge"],
        "reproj_dc_all_mae": sino_dc_m["mae_all"],
        "reproj_dc_all_rmse": sino_dc_m["rmse_all"],
        "reproj_dc_missing_wedge_mae": sino_dc_m["mae_missing_wedge"],
    }

    return HighPitchSimulationPair(
        scenario=scen,
        target_pitch=float(target_pitch),
        effective_pitch=float(effective_pitch),
        native_pitch=float(native_pitch),
        active_row_range=(float(active_row_range[0]), float(active_row_range[1])),
        z_slices_mm=fbp_ref.z_slices_mm.copy(),
        y_mm=fbp_ref.y_mm.copy(),
        x_mm=fbp_ref.x_mm.copy(),
        fov_mm=float(fbp_ref.fov_mm),
        fov_mask=fbp_ref.fov_mask.copy(),
        filter_name=fname,
        s_target=s_target,
        s_hp_raw=s_hp_raw,
        mask_hp=mask_hp,
        s_hp_extrap=s_hp_extrap,
        s_hp_parker_weighted=s_hp_parker,
        s_reproj=s_reproj,
        s_reproj_dc=s_reproj_dc,
        beta_rad=active_beta_rad.copy(),
        u_coords_mm=active_u_coords.copy(),
        mu_target=fbp_ref.mu_slices.copy(),
        x_target_hu=fbp_ref.hu_slices.copy(),
        mu_prelim=fbp_prelim.mu_slices.copy(),
        x_prelim_hu=fbp_prelim.hu_slices.copy(),
        mu_raw_nofill=fbp_raw_nofill.mu_slices.copy(),
        x_raw_nofill_hu=fbp_raw_nofill.hu_slices.copy(),
        prelim_vs_target_metrics=prelim_metrics,
        raw_nofill_vs_target_metrics=raw_metrics,
        sinogram_metrics=sino_metrics,
    )


def simulate_series_high_pitch_streaming(
    series_dir_or_files: str | Path | list[Path],
    z_slices_mm: np.ndarray | list[float],
    *,
    geometry: HelicalGeometry | None = None,
    scenario: str = "rdc",
    target_pitch: float = 2.0,
    image_size: int = 512,
    fov_mm: float | None = None,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    inpaint_mode: str = "conjugate_feather",
    backend: str = "numpy",
    itf_slab_slices: int = 9,
    workers: int = 4,
    ssrb_ref_precomputed: SSRBResult | None = None,
    fbp_ref_precomputed: FBPReconstructionResult | None = None,
    show_progress: bool = False,
) -> HighPitchSimulationPair:
    """Simulate a complete High-Pitch (`RDC` or `ITF`) Dual-Domain pair on a real DICOM-CT-PD series."""
    scen = scenario.strip().lower()
    if scen not in SUPPORTED_HIGH_PITCH_SCENARIOS:
        raise ValueError(
            f"Unsupported scenario={scenario!r}; expected one of {sorted(SUPPORTED_HIGH_PITCH_SCENARIOS)}"
        )
    fname = normalize_filter_name(filter_name)

    if isinstance(series_dir_or_files, list):
        files = [Path(p) for p in series_dir_or_files]
        if geometry is None:
            raise ValueError("geometry must be provided when passing an explicit file list")
    else:
        geometry, files = scan_projection_series(
            series_dir_or_files, workers=max(workers, 4), show_progress=show_progress
        )

    z_req = np.asarray(z_slices_mm, dtype=np.float64).reshape(-1)
    traj = geometry.trajectory_metrics()
    native_pitch = float(traj["normalized_pitch"])

    if ssrb_ref_precomputed is not None and fbp_ref_precomputed is not None:
        ssrb_ref = ssrb_ref_precomputed
        fbp_ref = fbp_ref_precomputed
    else:
        ssrb_ref, fbp_ref = reconstruct_series_reference_streaming(
            files,
            z_req,
            geometry=geometry,
            angular_mode="short_scan",
            image_size=image_size,
            fov_mm=fov_mm,
            center_xy_mm=center_xy_mm,
            filter_name=fname,
            cutoff_ratio=cutoff_ratio,
            inpaint_mode="conjugate_feather",
            backend=backend,
            workers=workers,
            show_progress=show_progress,
        )

    if scen == "rdc":
        r_lo, r_hi, eff_pitch = compute_rdc_active_row_range(
            geometry.n_rows, native_pitch, target_pitch
        )
        ssrb_hp = rebin_series_ssrb_streaming(
            files,
            z_req,
            geometry=geometry,
            angular_mode="short_scan",
            active_row_range=(r_lo, r_hi),
            workers=workers,
            show_progress=show_progress,
        )
        return build_dual_domain_pair_from_ssrb(
            ssrb_ref,
            ssrb_hp,
            scenario="rdc",
            target_pitch=target_pitch,
            effective_pitch=eff_pitch,
            native_pitch=native_pitch,
            active_row_range=(r_lo, r_hi),
            image_size=fbp_ref.image_size,
            fov_mm=fbp_ref.fov_mm,
            center_xy_mm=fbp_ref.center_xy_mm,
            water_mu_mm_inv=geometry.water_mu_mm_inv,
            filter_name=fname,
            cutoff_ratio=cutoff_ratio,
            inpaint_mode=inpaint_mode,
            backend=backend,
            fbp_ref_precomputed=fbp_ref,
            show_progress=show_progress,
        )

    # Scenario == "itf" (Increased Table Feed via 3D Helical Cone-Beam Forward Projection)
    # Build a local 3D reference attenuation slab mu_slab(z, y, x) spanning the target z_req
    z_min_raw = float(np.min(geometry.z0_mm))
    z_max_raw = float(np.max(geometry.z0_mm))
    half_cov = 0.85 * float(geometry.coverage_isocenter_mm)
    slab_z_lo = max(z_min_raw + 0.5 * half_cov, float(np.min(z_req)) - half_cov)
    slab_z_hi = min(z_max_raw - 0.5 * half_cov, float(np.max(z_req)) + half_cov)
    if slab_z_hi <= slab_z_lo:
        slab_z_lo = float(np.min(z_req)) - 10.0
        slab_z_hi = float(np.max(z_req)) + 10.0

    n_slab = max(5, int(itf_slab_slices))
    z_slab = np.unique(np.concatenate([np.linspace(slab_z_lo, slab_z_hi, n_slab), z_req]))
    z_slab.sort()

    _, fbp_slab = reconstruct_series_reference_streaming(
        files,
        z_slab,
        geometry=geometry,
        angular_mode="short_scan",
        image_size=fbp_ref.image_size,
        fov_mm=fbp_ref.fov_mm,
        center_xy_mm=fbp_ref.center_xy_mm,
        filter_name=fname,
        cutoff_ratio=cutoff_ratio,
        inpaint_mode="conjugate_feather",
        backend=backend,
        workers=workers,
        show_progress=show_progress,
    )
    mu_slab = np.maximum(fbp_slab.mu_slices, np.float32(0.0))

    geom_itf = synthesize_itf_helical_geometry(
        geometry,
        target_pitch,
        z_center_mm=float(0.5 * (np.min(z_req) + np.max(z_req))),
        z_span_mm=float(np.max(z_req) - np.min(z_req)),
    )
    proj_itf = forward_project_helical_3d(
        mu_slab,
        z_slab,
        fbp_slab.y_mm,
        fbp_slab.x_mm,
        geom_itf,
        z_boundary_mode="clamp",
        backend=backend,
        show_progress=show_progress,
    )
    flat_itf, valid_itf = rebin_curved_to_flat(proj_itf, geom_itf)
    ssrb_itf = rebin_helical_ssrb(
        flat_itf,
        valid_itf,
        geom_itf,
        z_req,
        angular_mode="short_scan",
        active_row_range=None,
    )
    eff_itf_pitch = float(geom_itf.trajectory_metrics()["normalized_pitch"])
    return build_dual_domain_pair_from_ssrb(
        ssrb_ref,
        ssrb_itf,
        scenario="itf",
        target_pitch=target_pitch,
        effective_pitch=eff_itf_pitch,
        native_pitch=native_pitch,
        active_row_range=(0.0, float(geom_itf.n_rows - 1)),
        image_size=fbp_ref.image_size,
        fov_mm=fbp_ref.fov_mm,
        center_xy_mm=fbp_ref.center_xy_mm,
        water_mu_mm_inv=geometry.water_mu_mm_inv,
        filter_name=fname,
        cutoff_ratio=cutoff_ratio,
        inpaint_mode=inpaint_mode,
        backend=backend,
        fbp_ref_precomputed=fbp_ref,
        show_progress=show_progress,
    )
