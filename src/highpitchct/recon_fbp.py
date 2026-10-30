"""Stage 3: Sinogram Completion/Extrapolation, Parker Short-Scan Weighting, and Fan-Beam FBP.

Implements the complete deterministic analytical reconstruction baseline (`x_reference` and
high-pitch baseline `FBP`) from Stage 2 virtual-flat-detector `SSRBResult` sinograms:

1. Physical Missing-Wedge Completion & Lateral Truncation Extrapolation (`extrapolate_sinogram`):
   - Preserves the original `SSRBResult.measured_mask` untouched for downstream neural networks.
   - Fills high-pitch unmeasured wedges (`~measured_mask`) first from exact 2D fan-beam conjugate
     rays `(beta +/- (pi + 2*alpha), -u)` when measured, then via smooth directional angular and
     transverse boundary interpolation (preventing catastrophic zero-step Gibbs streaks).
   - Applies C^1 raised-cosine lateral detector roll-off before 1D ramp filtering to suppress
     FOV edge truncation rings and cupping when the table or outer chest touches column 0 / N_u-1.
2. Exact Parker Short-Scan Redundancy Weighting (`compute_parker_weights`, Parker 1982, Med. Phys. 9):
   - Smooth sin^2 weights on `[0, pi + 2*delta]` satisfying the exact conjugate-ray partition of
     unity `w(beta_tilde, alpha) + w(beta_tilde + pi + 2*alpha, -alpha) == 1` everywhere,
     including the `0/0` singular corners `(0, +delta)` and `(pi + 2*delta, -delta)` (`w = 0.5`).
3. Spatial-Domain Bandlimited Kak & Slaney Ramp Kernel (`build_fbp_filter_kernel_rfft`):
   - Constructed from exact real-space Ram-Lak impulse response (`h[0] = 1/(4*tau)`,
     `h[odd n] = -1/(pi^2 * n^2 * tau)`, `tau = (rho0/d0)*du`) transformed via zero-padded `rfft`
     so DC calibration is exact (`0.0 HU` on water cylinders without artificial negative HU bias),
     with selectable apodization windows (`ramp`, `shepp_logan`, `cosine`, `hamming`, `hann`).
4. Equispaced Collinear-Detector Fan-Beam Backprojection (`reconstruct_fanbeam_fbp`):
   - Reconstructs calibrated `mu [mm^-1]` and `HU` slices in standard supine radiological
     orientation (`Row 0 = +y Anterior`, `Row H-1 = -y Posterior`, `Col 0 = -x Patient Right`,
     `Col W-1 = +x Patient Left`).
   - Supports deterministic vectorized NumPy CPU execution (`backend='numpy'`, default) and
     explicit PyTorch CPU/CUDA execution (`backend='torch'` / `'torch_cuda'`, never silent fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .ctpd_reader import mu_to_hu, scan_projection_series
from .geometry import HelicalGeometry
from .progress import ProgressReporter
from .rebinning import SUPPORTED_ANGULAR_MODES, SSRBResult, rebin_series_ssrb_streaming

SUPPORTED_FILTER_NAMES = frozenset(
    {"ram_lak", "ram-lak", "ramp", "shepp_logan", "cosine", "hamming", "hann"}
)
SUPPORTED_INPAINT_MODES = frozenset({"conjugate_feather", "angular_feather", "none"})
SUPPORTED_FBP_BACKENDS = frozenset({"numpy", "torch", "torch_cuda"})


def normalize_filter_name(filter_name: str) -> str:
    """Validate and normalize FBP ramp filter name (`ram_lak` / `ram-lak` / `ramp` -> `ram_lak`)."""
    fname = filter_name.strip().lower()
    if fname not in SUPPORTED_FILTER_NAMES:
        raise ValueError(
            f"Unsupported filter_name={filter_name!r}; expected one of {sorted(SUPPORTED_FILTER_NAMES)}"
        )
    if fname in {"ram_lak", "ram-lak", "ramp"}:
        return "ram_lak"
    return fname


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


@dataclass(frozen=True)
class PreparedSinogram:
    """Completed sinogram ready for Parker weighting + Ramp filtering, keeping original mask intact."""

    completed_sinograms: np.ndarray
    original_measured_mask: np.ndarray
    conjugate_filled_mask: np.ndarray
    interpolated_mask: np.ndarray
    inpaint_mode: str
    lateral_pad_cols: int

    def summary_dict(self) -> dict[str, Any]:
        orig_frac = float(np.mean(self.original_measured_mask)) if self.original_measured_mask.size else 0.0
        conj_frac = float(np.mean(self.conjugate_filled_mask)) if self.conjugate_filled_mask.size else 0.0
        interp_frac = float(np.mean(self.interpolated_mask)) if self.interpolated_mask.size else 0.0
        return {
            "inpaint_mode": self.inpaint_mode,
            "lateral_pad_cols": int(self.lateral_pad_cols),
            "original_measured_fraction": round(orig_frac, 6),
            "conjugate_filled_fraction": round(conj_frac, 6),
            "directional_interpolated_fraction": round(interp_frac, 6),
        }


@dataclass(frozen=True)
class FBPReconstructionResult:
    """Calibrated axial CT reconstruction output (`mu [mm^-1]` and `HU`) from Stage 3 FBP."""

    mu_slices: np.ndarray
    hu_slices: np.ndarray
    fov_mask: np.ndarray
    z_slices_mm: np.ndarray
    y_mm: np.ndarray
    x_mm: np.ndarray
    fov_mm: float
    pixel_spacing_mm: float
    center_xy_mm: tuple[float, float]
    water_mu_mm_inv: float
    filter_name: str
    cutoff_ratio: float
    inpaint_mode: str
    lateral_pad_cols: int
    angular_mode: str
    backend: str
    sinogram_prep_summary: dict[str, Any]
    extrapolated_sinograms: np.ndarray | None = None

    @property
    def n_slices(self) -> int:
        return int(self.mu_slices.shape[0])

    @property
    def image_size(self) -> int:
        return int(self.mu_slices.shape[1])

    def summary_dict(self) -> dict[str, Any]:
        in_fov = np.broadcast_to(self.fov_mask[None, :, :], self.hu_slices.shape)
        hu_vals = self.hu_slices[in_fov] if np.any(in_fov) else self.hu_slices.reshape(-1)
        mu_vals = self.mu_slices[in_fov] if np.any(in_fov) else self.mu_slices.reshape(-1)
        return {
            "shape_syx": [int(x) for x in self.mu_slices.shape],
            "fov_mm": round(float(self.fov_mm), 4),
            "pixel_spacing_mm": round(float(self.pixel_spacing_mm), 6),
            "center_xy_mm": [round(float(c), 4) for c in self.center_xy_mm],
            "z_slices_mm": [round(float(z), 4) for z in self.z_slices_mm],
            "water_mu_mm_inv": round(float(self.water_mu_mm_inv), 6),
            "filter_name": self.filter_name,
            "cutoff_ratio": round(float(self.cutoff_ratio), 4),
            "inpaint_mode": self.inpaint_mode,
            "lateral_pad_cols": int(self.lateral_pad_cols),
            "angular_mode": self.angular_mode,
            "backend": self.backend,
            "hu_in_fov_min_max": [
                round(float(np.min(hu_vals)), 2),
                round(float(np.max(hu_vals)), 2),
            ]
            if hu_vals.size
            else [0.0, 0.0],
            "hu_in_fov_mean": round(float(np.mean(hu_vals)), 3) if hu_vals.size else 0.0,
            "mu_in_fov_min_max_mm_inv": [
                round(float(np.min(mu_vals)), 6),
                round(float(np.max(mu_vals)), 6),
            ]
            if mu_vals.size
            else [0.0, 0.0],
            "sinogram_preparation": self.sinogram_prep_summary,
        }


def default_scanner_fov_mm(rho0_mm: float, gamma_max_rad: float) -> float:
    """Compute exact non-truncated circular fan-beam FOV diameter 2 * rho0 * sin(gamma_max) [mm]."""
    fov = 2.0 * float(rho0_mm) * math.sin(float(gamma_max_rad))
    if fov <= 0.0:
        raise ValueError(
            f"Scanner FOV must be positive, got {fov} from rho0={rho0_mm}, gamma_max={gamma_max_rad}"
        )
    return fov


def _interp_columns_linear_2d(
    arr_2d: np.ndarray,
    valid_mask_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized column-wise 1D linear interpolation matching `np.interp` clamping along axis 0."""
    n_beta, _ = arr_2d.shape
    partial_cols = np.any(valid_mask_2d, axis=0) & (~np.all(valid_mask_2d, axis=0))
    if not np.any(partial_cols):
        return arr_2d, np.zeros_like(valid_mask_2d, dtype=bool)

    out = arr_2d.copy()
    filled = np.zeros_like(valid_mask_2d, dtype=bool)
    col_indices = np.where(partial_cols)[0]
    sub_mask = valid_mask_2d[:, col_indices]
    sub_val = out[:, col_indices]
    n_p = col_indices.size

    rows = np.arange(n_beta, dtype=np.int32)[:, None]
    prev_idx = np.maximum.accumulate(np.where(sub_mask, rows, -1), axis=0)
    next_idx = np.minimum.accumulate(np.where(sub_mask, rows, n_beta)[::-1], axis=0)[::-1]

    first_valid = next_idx[0, :]
    last_valid = prev_idx[-1, :]
    i0 = np.where(prev_idx < 0, first_valid[None, :], prev_idx)
    i1 = np.where(next_idx >= n_beta, last_valid[None, :], next_idx)

    col_arange = np.arange(n_p, dtype=np.int32)[None, :]
    v0 = sub_val[i0, col_arange]
    v1 = sub_val[i1, col_arange]
    denom = (i1 - i0).astype(np.float64)
    w = np.where(i1 > i0, (rows.astype(np.float64) - i0.astype(np.float64)) / np.maximum(denom, 1.0), 0.0)
    interp_vals = v0 + w * (v1 - v0)

    miss_sub = ~sub_mask
    out[:, col_indices] = np.where(miss_sub, interp_vals, sub_val)
    filled[:, col_indices] = miss_sub
    return out, filled


def extrapolate_sinogram(
    ssrb_result: SSRBResult,
    *,
    inpaint_mode: str = "conjugate_feather",
    lateral_pad_cols: int = 96,
    workers: int = 1,
) -> PreparedSinogram:
    """Fill unmeasured high-pitch sinogram wedges without mutating `SSRBResult.measured_mask`.

    Modes:
      - 'conjugate_feather' (default): first recovers unmeasured rays `(beta, u)` from exact
        2D fan-beam conjugate rays `(beta +/- (pi + 2*alpha), -u)` when measured inside the scan
        window, then smoothly interpolates any remaining high-pitch gaps along angle `beta` and
        transverse column `u`.
      - 'angular_feather': directional angular + transverse boundary interpolation without
        conjugate-ray lookup.
      - 'none': leaves unmeasured pixels at `0.0` (for ablation / artifact demonstration).
    """
    from concurrent.futures import ThreadPoolExecutor

    mode = inpaint_mode.strip().lower()
    if mode not in SUPPORTED_INPAINT_MODES:
        raise ValueError(
            f"Unsupported inpaint_mode={inpaint_mode!r}; expected one of {sorted(SUPPORTED_INPAINT_MODES)}"
        )
    if int(lateral_pad_cols) < 0:
        raise ValueError(f"lateral_pad_cols must be >= 0, got {lateral_pad_cols}")

    sino_orig = np.asarray(ssrb_result.sinograms, dtype=np.float32)
    mask_orig = np.asarray(ssrb_result.measured_mask, dtype=bool)
    if sino_orig.shape != mask_orig.shape or sino_orig.ndim != 3:
        raise ValueError(
            f"sinograms shape {sino_orig.shape} and measured_mask shape {mask_orig.shape} must match (S, N_beta, N_u)"
        )

    n_slices, n_beta, n_u = sino_orig.shape
    completed = np.where(mask_orig, sino_orig, np.float32(0.0)).copy()
    conj_filled = np.zeros_like(mask_orig, dtype=bool)
    interp_filled = np.zeros_like(mask_orig, dtype=bool)

    if mode == "none" or n_slices == 0 or np.all(mask_orig):
        return PreparedSinogram(
            completed_sinograms=completed,
            original_measured_mask=mask_orig.copy(),
            conjugate_filled_mask=conj_filled,
            interpolated_mask=interp_filled,
            inpaint_mode=mode,
            lateral_pad_cols=int(lateral_pad_cols),
        )

    u_coords = np.asarray(ssrb_result.u_coords_mm, dtype=np.float64)
    du = float(ssrb_result.du_mm)
    c_flat_0 = 0.5 * (n_u - 1.0)
    alpha_u = np.arctan2(u_coords, float(ssrb_result.d0_mm))  # (n_u,)
    k_grid = np.arange(n_beta, dtype=np.float64)
    m_grid = np.arange(n_u, dtype=np.float64)
    is_full = ssrb_result.angular_mode == "full_scan"

    def _process_slice(s_idx: int) -> None:
        cur_mask = mask_orig[s_idx].copy()
        if np.all(cur_mask) or not np.any(cur_mask):
            return

        cur_sino = completed[s_idx].astype(np.float64, copy=True)
        beta_s = np.asarray(ssrb_result.beta_rad[s_idx], dtype=np.float64)
        beta_min = float(beta_s[0])
        beta_max = float(beta_s[-1])

        # Step 0: Pre-fill any completely unmeasured lateral border column (e.g. quarter-offset edge)
        # from its nearest measured column at angles where that neighbor is measured, so border columns
        # never get interpolated across 180 deg in Step 2.
        col_has_meas = np.any(cur_mask, axis=0)
        if not np.all(col_has_meas) and np.any(col_has_meas):
            meas_cols = np.where(col_has_meas)[0]
            for m_un in np.where(~col_has_meas)[0]:
                nearest_m = int(meas_cols[np.argmin(np.abs(meas_cols - m_un))])
                nb_mask = cur_mask[:, nearest_m]
                cur_sino[nb_mask, m_un] = cur_sino[nb_mask, nearest_m]
                cur_mask[nb_mask, m_un] = True
                interp_filled[s_idx, nb_mask, m_un] = True

        if np.all(cur_mask):
            completed[s_idx] = np.maximum(cur_sino, 0.0).astype(np.float32)
            return

        # Step 1: Exact 2D fan-beam conjugate-ray fill (beta' = beta + 2*alpha +/- pi, u' = -u)
        if mode == "conjugate_feather" and n_beta >= 2:
            miss_k, miss_m = np.where(~cur_mask)
            if miss_k.size > 0:
                u_conj = -u_coords[miss_m]
                m_conj_f = (u_conj / du) + c_flat_0
                valid_m_conj = (m_conj_f >= -1e-11) & (m_conj_f <= float(n_u - 1) + 1e-11)
                m_conj_cl = np.clip(m_conj_f, 0.0, float(n_u - 1))
                m0 = np.floor(m_conj_cl).astype(np.intp)
                m1 = np.minimum(m0 + 1, n_u - 1)
                wm = m_conj_cl - m0

                still_missing = np.ones(miss_k.size, dtype=bool)
                for sign in (+1.0, -1.0):
                    if not np.any(still_missing):
                        break
                    cand_idx = np.where(still_missing & valid_m_conj)[0]
                    if cand_idx.size == 0:
                        continue
                    ks = miss_k[cand_idx]
                    ms = miss_m[cand_idx]
                    b_conj = beta_s[ks] + 2.0 * alpha_u[ms] + sign * math.pi
                    if is_full and n_beta >= 2:
                        b_conj = beta_min + np.mod(b_conj - beta_min, 2.0 * math.pi)
                        in_b = np.ones(cand_idx.size, dtype=bool)
                    else:
                        in_b = (b_conj >= beta_min - 1e-11) & (b_conj <= beta_max + 1e-11)

                    if not np.any(in_b):
                        continue

                    sub = cand_idx[in_b]
                    b_cl = np.clip(b_conj[in_b], beta_min, beta_max)
                    if is_full:
                        d_beta = (2.0 * math.pi) / float(n_beta)
                        k_f = (b_cl - beta_min) / d_beta
                        k0 = (np.floor(k_f).astype(np.intp)) % n_beta
                        k1 = (k0 + 1) % n_beta
                        wk = k_f - np.floor(k_f)
                    else:
                        k_f = np.interp(b_cl, beta_s, k_grid)
                        k0 = np.floor(k_f).astype(np.intp)
                        k1 = np.minimum(k0 + 1, n_beta - 1)
                        wk = k_f - k0

                    mc0 = m0[sub]
                    mc1 = m1[sub]
                    wmc = wm[sub]
                    ok4 = (
                        cur_mask[k0, mc0]
                        & cur_mask[k0, mc1]
                        & cur_mask[k1, mc0]
                        & cur_mask[k1, mc1]
                    )
                    if not np.any(ok4):
                        continue

                    win = sub[ok4]
                    k0w, k1w, wkw = k0[ok4], k1[ok4], wk[ok4]
                    m0w, m1w, wmw = mc0[ok4], mc1[ok4], wmc[ok4]
                    vals = (
                        (1.0 - wkw) * (1.0 - wmw) * cur_sino[k0w, m0w]
                        + (1.0 - wkw) * wmw * cur_sino[k0w, m1w]
                        + wkw * (1.0 - wmw) * cur_sino[k1w, m0w]
                        + wkw * wmw * cur_sino[k1w, m1w]
                    )
                    tgt_k = miss_k[win]
                    tgt_m = miss_m[win]
                    cur_sino[tgt_k, tgt_m] = vals
                    cur_mask[tgt_k, tgt_m] = True
                    conj_filled[s_idx, tgt_k, tgt_m] = True
                    still_missing[win] = False

        # Step 2: Smooth angular continuity interpolation per column m
        if is_full:
            for m_idx in range(n_u):
                col_mask = cur_mask[:, m_idx]
                if np.all(col_mask) or not np.any(col_mask):
                    continue
                valid_k = np.where(col_mask)[0]
                valid_v = cur_sino[valid_k, m_idx]
                miss_k_col = np.where(~col_mask)[0]
                if valid_k.size >= 2:
                    k_ext = np.concatenate([valid_k - n_beta, valid_k, valid_k + n_beta]).astype(np.float64)
                    v_ext = np.concatenate([valid_v, valid_v, valid_v])
                    cur_sino[miss_k_col, m_idx] = np.interp(miss_k_col.astype(np.float64), k_ext, v_ext)
                else:
                    cur_sino[miss_k_col, m_idx] = np.interp(
                        miss_k_col.astype(np.float64),
                        valid_k.astype(np.float64),
                        valid_v,
                    )
                cur_mask[miss_k_col, m_idx] = True
                interp_filled[s_idx, miss_k_col, m_idx] = True
        else:
            cur_sino, col_filled = _interp_columns_linear_2d(cur_sino, cur_mask)
            if np.any(col_filled):
                cur_mask |= col_filled
                interp_filled[s_idx] |= col_filled

        # Step 3: Fill any remaining unmeasured pixels along u
        if not np.all(cur_mask):
            for k_idx in range(n_beta):
                row_mask = cur_mask[k_idx, :]
                if np.all(row_mask) or not np.any(row_mask):
                    continue
                valid_m = np.where(row_mask)[0]
                miss_m_row = np.where(~row_mask)[0]
                cur_sino[k_idx, miss_m_row] = np.interp(
                    m_grid[miss_m_row],
                    m_grid[valid_m],
                    cur_sino[k_idx, valid_m],
                )
                cur_mask[k_idx, miss_m_row] = True
                interp_filled[s_idx, k_idx, miss_m_row] = True

        # Step 4: Feather transitions across all unmeasured pixels (~mask_orig[s_idx]) along (u, k)
        # while keeping every originally measured pixel (mask_orig[s_idx]) 100% untouched and exact.
        unmeas_s = ~mask_orig[s_idx]
        cur_f32 = cur_sino.astype(np.float32)
        if np.any(unmeas_s) and n_u >= 3 and n_beta >= 3:
            for _ in range(6):
                pad_u = np.pad(cur_f32, ((0, 0), (1, 1)), mode="edge")
                smooth_u = np.float32(0.25) * pad_u[:, :-2] + np.float32(0.50) * pad_u[:, 1:-1] + np.float32(0.25) * pad_u[:, 2:]
                pad_k = np.pad(smooth_u, ((1, 1), (0, 0)), mode="edge")
                smooth_ku = np.float32(0.25) * pad_k[:-2, :] + np.float32(0.50) * pad_k[1:-1, :] + np.float32(0.25) * pad_k[2:, :]
                cur_f32 = np.where(unmeas_s, smooth_ku, cur_f32)

        completed[s_idx] = np.maximum(cur_f32, np.float32(0.0))

    n_workers = max(1, int(workers))
    if n_workers > 1 and n_slices >= 4:
        with ThreadPoolExecutor(max_workers=min(n_workers, n_slices)) as pool:
            list(pool.map(_process_slice, range(n_slices)))
    else:
        for s_idx in range(n_slices):
            _process_slice(s_idx)

    return PreparedSinogram(
        completed_sinograms=completed,
        original_measured_mask=mask_orig.copy(),
        conjugate_filled_mask=conj_filled,
        interpolated_mask=interp_filled,
        inpaint_mode=mode,
        lateral_pad_cols=int(lateral_pad_cols),
    )


def compute_parker_weights(
    beta_rad_1d: np.ndarray,
    u_coords_mm: np.ndarray,
    d0_mm: float,
    *,
    angular_mode: str = "short_scan",
) -> np.ndarray:
    """Compute exact redundancy weights `w(beta, u)` (Parker 1982 for short-scan; 0.5 for full-scan).

    For `short_scan` over `[beta_0, beta_0 + pi + 2*delta]` (`delta >= gamma_max`):
      - `0 <= beta_tilde <= 2*(delta - alpha)`:  `sin^2((pi / 4) * beta_tilde / (delta - alpha))`
      - `2*(delta - alpha) < beta_tilde < pi - 2*alpha`:  `1.0`
      - `pi - 2*alpha <= beta_tilde <= pi + 2*delta`:  `sin^2((pi / 4) * (pi + 2*delta - beta_tilde) / (delta + alpha))`
      - Singular corners `(0, +delta)` and `(pi + 2*delta, -delta)` evaluate to `0.5` so that BOTH
        `w(beta_tilde, alpha) + w(beta_tilde + pi + 2*alpha, -alpha) == 1` AND trapezoidal column
        quadrature `sum_k w_k * d_beta == pi` hold to machine precision across all columns!
    """
    mode = angular_mode.strip().lower()
    if mode not in SUPPORTED_ANGULAR_MODES:
        raise ValueError(
            f"Unsupported angular_mode={angular_mode!r}; expected one of {sorted(SUPPORTED_ANGULAR_MODES)}"
        )

    beta = np.asarray(beta_rad_1d, dtype=np.float64).reshape(-1)
    u = np.asarray(u_coords_mm, dtype=np.float64).reshape(-1)
    n_beta = beta.size
    n_u = u.size
    if n_beta < 2 or n_u < 1:
        raise ValueError(f"Require n_beta >= 2 and n_u >= 1, got ({n_beta}, {n_u})")
    if float(d0_mm) <= 0.0:
        raise ValueError(f"d0_mm must be positive, got {d0_mm}")

    if mode == "full_scan":
        return np.full((n_beta, n_u), 0.5, dtype=np.float64)

    span = float(beta[-1] - beta[0])
    if span <= 0.0 or not np.all(np.diff(beta) > 0.0):
        raise ValueError("beta_rad_1d must be strictly monotonically increasing")

    alpha = np.arctan2(u, float(d0_mm))  # (n_u,)
    gamma_max = float(np.max(np.abs(alpha)))
    delta = 0.5 * (span - math.pi)
    if delta < gamma_max - 1e-6:
        raise ValueError(
            f"Short-scan angular span ({math.degrees(span):.4f} deg) is narrower than minimum "
            f"pi + 2*gamma_max ({math.degrees(math.pi + 2.0 * gamma_max):.4f} deg)"
        )
    delta = max(delta, gamma_max)

    b_tilde = np.broadcast_to((beta - float(beta[0]))[:, None], (n_beta, n_u))
    alpha_row = alpha[None, :]  # (1, n_u)
    lower_bound = 2.0 * (delta - alpha_row)  # >= 0
    upper_start = math.pi - 2.0 * alpha_row  # <= pi + 2*delta
    total_span = math.pi + 2.0 * delta

    weights = np.ones((n_beta, n_u), dtype=np.float64)

    # Lower overscan zone: 0 <= b_tilde <= 2*(delta - alpha)
    in_low = (b_tilde >= -1e-12) & (b_tilde <= lower_bound + 1e-12)
    denom_low = np.broadcast_to(delta - alpha_row, (n_beta, n_u))
    regular_low = in_low & (denom_low > 1e-12)
    singular_low = in_low & (denom_low <= 1e-12)
    if np.any(regular_low):
        eta_low = (0.25 * math.pi) * np.clip(b_tilde[regular_low] / denom_low[regular_low], 0.0, 2.0)
        weights[regular_low] = np.sin(eta_low) ** 2
    if np.any(singular_low):
        weights[singular_low] = 0.5

    # Upper overscan zone: pi - 2*alpha <= b_tilde <= pi + 2*delta
    in_high = (b_tilde >= upper_start - 1e-12) & (b_tilde <= total_span + 1e-12)
    denom_high = np.broadcast_to(delta + alpha_row, (n_beta, n_u))
    regular_high = in_high & (denom_high > 1e-12)
    singular_high = in_high & (denom_high <= 1e-12)
    if np.any(regular_high):
        eta_high = (0.25 * math.pi) * np.clip(
            (total_span - b_tilde[regular_high]) / denom_high[regular_high], 0.0, 2.0
        )
        weights[regular_high] = np.sin(eta_high) ** 2
    if np.any(singular_high):
        weights[singular_high] = 0.5

    outside = (b_tilde < -1e-12) | (b_tilde > total_span + 1e-12)
    if np.any(outside):
        weights[outside] = 0.0

    return weights


def build_fbp_filter_kernel_rfft(
    n_fft: int,
    du_iso_mm: float,
    *,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
) -> np.ndarray:
    """Build exact spatial-domain Kak & Slaney discrete ramp filter in `rfft` domain `[mm^-1]`.

    Uses the analytical real-space impulse response `k[0] = 1/(4*du_iso)`,
    `k[odd n] = -1/(pi^2 * n^2 * du_iso)` (Kak & Slaney 1988, Eq. 61) so finite-aperture
    DC bias is strictly zero (avoiding the ~-10 HU cupping bias of naive `|fftfreq|`),
    multiplied by a smooth frequency apodization window with `W(0) = 1`.
    """
    fname = normalize_filter_name(filter_name)
    if int(n_fft) < 4 or (int(n_fft) % 2) != 0:
        raise ValueError(f"n_fft must be an even integer >= 4, got {n_fft}")
    if float(du_iso_mm) <= 0.0:
        raise ValueError(f"du_iso_mm must be positive, got {du_iso_mm}")
    if not (0.0 < float(cutoff_ratio) <= 1.0):
        raise ValueError(f"cutoff_ratio must lie in (0, 1], got {cutoff_ratio}")

    n_f = int(n_fft)
    tau = float(du_iso_mm)
    spatial = np.zeros(n_f, dtype=np.float64)
    spatial[0] = 1.0 / (4.0 * tau)

    half = n_f // 2
    odd_n = np.arange(1, half + 1, 2, dtype=np.int64)
    odd_vals = -1.0 / ((math.pi * math.pi) * (odd_n.astype(np.float64) ** 2) * tau)
    spatial[odd_n] = odd_vals
    mirror_mask = odd_n < half
    spatial[n_f - odd_n[mirror_mask]] = odd_vals[mirror_mask]

    h_rfft = np.real(np.fft.rfft(spatial))  # Purely real by even symmetry

    # Normalized frequency f_norm in [0, 1] relative to cutoff * Nyquist
    f_nyq_ratio = np.fft.rfftfreq(n_f, d=1.0) / 0.5  # [0.0 .. 1.0]
    f_scaled = f_nyq_ratio / float(cutoff_ratio)
    within_cut = f_scaled <= 1.0
    fc = np.clip(f_scaled, 0.0, 1.0)

    if fname == "ram_lak":
        win = np.ones_like(fc)
    elif fname == "shepp_logan":
        # np.sinc(x) computes sin(pi * x) / (pi * x)
        win = np.sinc(0.5 * fc)
    elif fname == "cosine":
        win = np.cos(0.5 * math.pi * fc)
    elif fname == "hamming":
        win = 0.54 + 0.46 * np.cos(math.pi * fc)
    elif fname == "hann":
        win = 0.50 + 0.50 * np.cos(math.pi * fc)
    else:
        raise AssertionError("Unreachable")

    win = np.where(within_cut, win, 0.0)
    win[0] = 1.0  # Preserve exact DC water calibration across all windows/cutoffs
    return (h_rfft * win).astype(np.float64)


def filter_weighted_sinogram_2d(
    weighted_sino_2d: np.ndarray,
    *,
    du_iso_mm: float,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    lateral_pad_cols: int = 96,
) -> np.ndarray:
    """Apply smooth raised-cosine lateral truncation roll-off + zero-padded 1D FFT ramp filter."""
    g_w = np.asarray(weighted_sino_2d, dtype=np.float64)
    if g_w.ndim != 2:
        raise ValueError(f"Expected 2D weighted sinogram (N_beta, N_u), got {g_w.shape}")

    n_beta, n_u = g_w.shape
    pad = max(0, int(lateral_pad_cols))
    n_ext = n_u + 2 * pad
    n_fft = max(64, _next_power_of_two(2 * n_ext))

    buf = np.zeros((n_beta, n_fft), dtype=np.float64)
    buf[:, pad : pad + n_u] = g_w
    if pad > 0:
        idx_p = np.arange(1, pad + 1, dtype=np.float64)
        left_taper = 0.5 * (1.0 - np.cos(math.pi * idx_p / float(pad + 1)))  # 0 -> 1
        right_taper = left_taper[::-1]  # 1 -> 0
        buf[:, :pad] = g_w[:, 0:1] * left_taper[None, :]
        buf[:, pad + n_u : pad + n_u + pad] = g_w[:, -1:] * right_taper[None, :]

    h_rfft = build_fbp_filter_kernel_rfft(
        n_fft,
        du_iso_mm,
        filter_name=filter_name,
        cutoff_ratio=cutoff_ratio,
    )
    filtered_full = np.fft.irfft(np.fft.rfft(buf, axis=1) * h_rfft[None, :], n=n_fft, axis=1)
    return filtered_full[:, pad : pad + n_u]


def _quadrature_d_beta(beta_rad_1d: np.ndarray, angular_mode: str) -> float:
    beta = np.asarray(beta_rad_1d, dtype=np.float64).reshape(-1)
    if beta.size < 2:
        raise ValueError("Require at least 2 angles in beta_rad_1d")
    if angular_mode.strip().lower() == "full_scan":
        return (2.0 * math.pi) / float(beta.size)
    return float(beta[-1] - beta[0]) / float(beta.size - 1)


def _backproject_slice_numpy(
    q_filtered: np.ndarray,
    beta_rad_1d: np.ndarray,
    *,
    d_beta: float,
    x_fov: np.ndarray,
    y_fov: np.ndarray,
    rho0_mm: float,
    d0_mm: float,
    du_mm: float,
    angle_chunk: int = 64,
) -> np.ndarray:
    """Vectorized chunked-angle fan-beam backprojection onto active circular-FOV pixels `[mm^-1]`."""
    n_beta, n_u = q_filtered.shape
    n_pix = x_fov.size
    if n_pix == 0:
        return np.zeros(0, dtype=np.float32)

    q_f32 = np.ascontiguousarray(q_filtered, dtype=np.float32)
    x_f32 = np.ascontiguousarray(x_fov, dtype=np.float32)
    y_f32 = np.ascontiguousarray(y_fov, dtype=np.float32)
    rho0_f = np.float32(rho0_mm)
    inv_rho0_f = np.float32(1.0 / rho0_mm)
    d0_over_du_f = np.float32(d0_mm / du_mm)
    c_flat_0_f = np.float32(0.5 * (n_u - 1.0))
    max_col_f = np.float32(n_u - 1.0)
    n_u_minus_1 = n_u - 1

    acc = np.zeros(n_pix, dtype=np.float32)
    step = max(1, int(angle_chunk))

    for k0 in range(0, n_beta, step):
        k1 = min(n_beta, k0 + step)
        b_chunk = beta_rad_1d[k0:k1]
        sin_b = np.sin(b_chunk).astype(np.float32)[:, None]  # (B, 1)
        cos_b = np.cos(b_chunk).astype(np.float32)[:, None]  # (B, 1)

        # Ray distance along e_ray(beta) = (sin(beta), -cos(beta)) from source (-rho0*sin, +rho0*cos)
        d_ray = rho0_f + sin_b * x_f32[None, :] - cos_b * y_f32[None, :]  # (B, P)
        d_ray = np.maximum(d_ray, np.float32(1e-3))
        inv_d_ray = np.float32(1.0) / d_ray

        # Transverse coordinate along e_u(beta) = (cos(beta), sin(beta))
        u_perp = cos_b * x_f32[None, :] + sin_b * y_f32[None, :]  # (B, P)
        m_star = u_perp * inv_d_ray * d0_over_du_f + c_flat_0_f  # (B, P)

        in_det = (m_star >= np.float32(0.0)) & (m_star <= max_col_f)
        m_cl = np.clip(m_star, np.float32(0.0), max_col_f)
        m0 = np.floor(m_cl).astype(np.int32)
        m1 = np.minimum(m0 + 1, n_u_minus_1)
        wm = m_cl - m0

        rows = np.arange(k1 - k0, dtype=np.int32)[:, None]
        q_sub = q_f32[k0:k1]
        q_interp = ((np.float32(1.0) - wm) * q_sub[rows, m0] + wm * q_sub[rows, m1]) * in_det

        # U = d_ray / rho0 => 1 / U^2 = (rho0 / d_ray)^2
        inv_u = rho0_f * inv_d_ray
        # Note: inv_rho0_f kept unused if rho0_f * inv_d_ray is direct; verify: rho0 / d_ray == 1 / U
        _ = inv_rho0_f
        weighted = q_interp * (inv_u * inv_u)
        acc += np.sum(weighted, axis=0, dtype=np.float32)

    acc *= np.float32(d_beta)
    return acc


def _backproject_slice_torch(
    q_filtered: np.ndarray,
    beta_rad_1d: np.ndarray,
    *,
    d_beta: float,
    x_fov: np.ndarray,
    y_fov: np.ndarray,
    rho0_mm: float,
    d0_mm: float,
    du_mm: float,
    require_cuda: bool,
    angle_chunk: int = 128,
) -> np.ndarray:
    """Explicit PyTorch CPU/CUDA backprojection (fails fast if `require_cuda=True` without GPU)."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch (`torch`) is required when backend='torch' or 'torch_cuda'; "
            "install torch or pass backend='numpy'."
        ) from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "Explicit backend='torch_cuda' requested, but torch.cuda.is_available() is False. "
            "Refusing silent fallback."
        )

    device = torch.device("cuda" if require_cuda else "cpu")

    n_beta, n_u = q_filtered.shape
    n_pix = x_fov.size
    if n_pix == 0:
        return np.zeros(0, dtype=np.float32)

    with torch.no_grad():
        q_t = torch.as_tensor(np.ascontiguousarray(q_filtered, dtype=np.float32), device=device)
        x_t = torch.as_tensor(np.ascontiguousarray(x_fov, dtype=np.float32), device=device)
        y_t = torch.as_tensor(np.ascontiguousarray(y_fov, dtype=np.float32), device=device)
        beta_t = torch.as_tensor(np.ascontiguousarray(beta_rad_1d, dtype=np.float32), device=device)

        rho0_f = float(rho0_mm)
        d0_over_du_f = float(d0_mm / du_mm)
        c_flat_0_f = float(0.5 * (n_u - 1.0))
        max_col_f = float(n_u - 1.0)
        acc = torch.zeros(n_pix, dtype=torch.float32, device=device)
        step = max(1, int(angle_chunk))

        for k0 in range(0, n_beta, step):
            k1 = min(n_beta, k0 + step)
            b_sub = beta_t[k0:k1]
            sin_b = torch.sin(b_sub).unsqueeze(1)
            cos_b = torch.cos(b_sub).unsqueeze(1)

            d_ray = torch.clamp(rho0_f + sin_b * x_t.unsqueeze(0) - cos_b * y_t.unsqueeze(0), min=1e-3)
            inv_d_ray = 1.0 / d_ray
            u_perp = cos_b * x_t.unsqueeze(0) + sin_b * y_t.unsqueeze(0)
            m_star = u_perp * inv_d_ray * d0_over_du_f + c_flat_0_f

            in_det = (m_star >= 0.0) & (m_star <= max_col_f)
            m_cl = torch.clamp(m_star, 0.0, max_col_f)
            m0 = torch.floor(m_cl).to(torch.long)
            m1 = torch.clamp(m0 + 1, max=n_u - 1)
            wm = m_cl - m0.to(torch.float32)

            q_sub = q_t[k0:k1]
            v0 = torch.gather(q_sub, 1, m0)
            v1 = torch.gather(q_sub, 1, m1)
            q_interp = ((1.0 - wm) * v0 + wm * v1) * in_det.to(torch.float32)

            inv_u = rho0_f * inv_d_ray
            acc += torch.sum(q_interp * (inv_u * inv_u), dim=0)

        acc *= float(d_beta)
        return acc.detach().cpu().numpy().astype(np.float32)


def _reconstruct_fanbeam_fbp_batched_torch(
    completed_sinograms: np.ndarray,
    beta_rad: np.ndarray,
    *,
    u_coords: np.ndarray,
    d0_mm: float,
    rho0_mm: float,
    du_mm: float,
    du_iso_mm: float,
    cos_alpha: np.ndarray,
    filter_name: str,
    cutoff_ratio: float,
    lateral_pad_cols: int,
    angular_mode: str,
    x_fov: np.ndarray,
    y_fov: np.ndarray,
    fov_iy: np.ndarray,
    fov_ix: np.ndarray,
    n_xy: int,
    require_cuda: bool,
    angle_chunk: int = 128,
    progress_bar: Any = None,
) -> np.ndarray:
    """Batched multi-slice PyTorch CPU/CUDA Parker + Ram-Lak FBP reconstruction."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError(
            "PyTorch (`torch`) is required when backend='torch' or 'torch_cuda'; "
            "install torch or pass backend='numpy'."
        ) from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "Explicit backend='torch_cuda' requested, but torch.cuda.is_available() is False. "
            "Refusing silent fallback."
        )

    n_slices, n_beta, n_u = completed_sinograms.shape
    mu_slices = np.zeros((n_slices, n_xy, n_xy), dtype=np.float32)
    n_pix = int(x_fov.size)
    if n_slices == 0 or n_pix == 0:
        return mu_slices

    device = torch.device("cuda" if require_cuda else "cpu")
    pad = max(0, int(lateral_pad_cols))
    n_ext = n_u + 2 * pad
    n_fft = max(64, _next_power_of_two(2 * n_ext))
    h_rfft = build_fbp_filter_kernel_rfft(
        n_fft,
        du_iso_mm,
        filter_name=filter_name,
        cutoff_ratio=cutoff_ratio,
    ).astype(np.float32)

    if pad > 0:
        idx_p = np.arange(1, pad + 1, dtype=np.float64)
        left_taper_np = (0.5 * (1.0 - np.cos(math.pi * idx_p / float(pad + 1)))).astype(np.float32)
        right_taper_np = left_taper_np[::-1].copy()
    else:
        left_taper_np = np.zeros(0, dtype=np.float32)
        right_taper_np = np.zeros(0, dtype=np.float32)

    if require_cuda:
        try:
            vram_bytes = torch.cuda.get_device_properties(device).total_memory
        except Exception:
            vram_bytes = 14 * (1024**3)
        if vram_bytes >= 70 * (1024**3):
            # A100 (80GB) or RTX PRO 6000 (96GB)
            slice_batch = 24 if n_pix <= 70000 else 12
            step = 128 if n_pix <= 70000 else 64
        elif vram_bytes >= 35 * (1024**3):
            # A6000 (48GB) or A40 (48GB)
            slice_batch = 16 if n_pix <= 70000 else 8
            step = 96 if n_pix <= 70000 else 64
        elif vram_bytes >= 20 * (1024**3):
            # RTX 3090 / 4090 (24GB)
            slice_batch = 12 if n_pix <= 70000 else 6
            step = 64
        else:
            # Tesla T4 (15GB on Kaggle), RTX 3080/4070 (10-12GB)
            slice_batch = 8 if n_pix <= 70000 else 4
            step = 48 if n_pix <= 70000 else 32
        torch.cuda.empty_cache()
    else:
        slice_batch = 4
        step = max(16, min(max(int(angle_chunk), 32), 64))

    # Parker weights depend only on relative angle b_tilde = beta - beta[0]; reuse when uniform across slices!
    beta_f64 = np.asarray(beta_rad, dtype=np.float64)
    b_tilde_all = beta_f64 - beta_f64[:, 0:1]
    shared_parker = bool(n_slices == 1 or np.allclose(b_tilde_all, b_tilde_all[0:1], atol=1e-7))
    w_red_shared_np = (
        compute_parker_weights(
            beta_f64[0],
            u_coords,
            d0_mm,
            angular_mode=angular_mode,
        ).astype(np.float32)
        if shared_parker
        else None
    )

    with torch.no_grad():
        h_t = torch.as_tensor(h_rfft, dtype=torch.float32, device=device).view(1, 1, -1)
        x_t = torch.as_tensor(np.ascontiguousarray(x_fov, dtype=np.float32), device=device).view(1, 1, -1)
        y_t = torch.as_tensor(np.ascontiguousarray(y_fov, dtype=np.float32), device=device).view(1, 1, -1)
        cos_t = torch.as_tensor(np.ascontiguousarray(cos_alpha, dtype=np.float32), device=device).view(1, 1, -1)
        left_t = torch.as_tensor(left_taper_np, dtype=torch.float32, device=device).view(1, 1, -1)
        right_t = torch.as_tensor(right_taper_np, dtype=torch.float32, device=device).view(1, 1, -1)
        w_red_shared_t = (
            torch.as_tensor(w_red_shared_np, dtype=torch.float32, device=device).unsqueeze(0)
            if w_red_shared_np is not None
            else None
        )

        rho0_f = float(rho0_mm)
        d0_over_du_f = float(d0_mm / du_mm)
        c_flat_0_f = float(0.5 * (n_u - 1.0))
        max_col_f = float(n_u - 1.0)

        for s0 in range(0, n_slices, slice_batch):
            s1 = min(n_slices, s0 + slice_batch)
            bsz = s1 - s0
            if w_red_shared_t is not None:
                w_red_b = w_red_shared_t
            else:
                w_red_np = np.stack(
                    [
                        compute_parker_weights(
                            beta_f64[s],
                            u_coords,
                            d0_mm,
                            angular_mode=angular_mode,
                        ).astype(np.float32)
                        for s in range(s0, s1)
                    ],
                    axis=0,
                )
                w_red_b = torch.as_tensor(w_red_np, dtype=torch.float32, device=device)

            d_beta_np = np.asarray(
                [
                    _quadrature_d_beta(beta_f64[s], angular_mode)
                    for s in range(s0, s1)
                ],
                dtype=np.float32,
            )

            sino_b = torch.as_tensor(
                np.ascontiguousarray(completed_sinograms[s0:s1], dtype=np.float32), device=device
            )
            beta_b = torch.as_tensor(
                np.ascontiguousarray(beta_rad[s0:s1], dtype=np.float32), device=device
            )
            d_beta_b = torch.as_tensor(d_beta_np, dtype=torch.float32, device=device).view(bsz, 1)

            g_w = sino_b * w_red_b * cos_t
            if pad > 0:
                left_pad = g_w[:, :, 0:1] * left_t
                right_pad = g_w[:, :, -1:] * right_t
                g_ext = torch.cat([left_pad, g_w, right_pad], dim=2)
            else:
                g_ext = g_w
            g_fft_in = F.pad(g_ext, (0, n_fft - g_ext.shape[2]), mode="constant", value=0.0)
            q_full = torch.fft.irfft(
                torch.fft.rfft(g_fft_in, n=n_fft, dim=-1) * h_t, n=n_fft, dim=-1
            )
            q_b = q_full[:, :, pad : pad + n_u]

            acc = torch.zeros((bsz, n_pix), dtype=torch.float32, device=device)
            for k0 in range(0, n_beta, step):
                k1 = min(n_beta, k0 + step)
                b_sub = beta_b[:, k0:k1]
                sin_b = torch.sin(b_sub).unsqueeze(2)
                cos_b = torch.cos(b_sub).unsqueeze(2)

                d_ray = torch.clamp(rho0_f + sin_b * x_t - cos_b * y_t, min=1e-3)
                inv_d_ray = 1.0 / d_ray
                u_perp = cos_b * x_t + sin_b * y_t
                m_star = u_perp * inv_d_ray * d0_over_du_f + c_flat_0_f

                in_det = (m_star >= 0.0) & (m_star <= max_col_f)
                m_cl = torch.clamp(m_star, 0.0, max_col_f)
                m0 = torch.floor(m_cl).to(torch.long)
                m1 = torch.clamp(m0 + 1, max=n_u - 1)
                wm = m_cl - m0.to(torch.float32)

                q_sub = q_b[:, k0:k1, :]
                v0 = torch.gather(q_sub, 2, m0)
                v1 = torch.gather(q_sub, 2, m1)
                q_interp = ((1.0 - wm) * v0 + wm * v1) * in_det.to(torch.float32)

                inv_u = rho0_f * inv_d_ray
                acc += torch.sum(q_interp * (inv_u * inv_u), dim=1)
                del sin_b, cos_b, d_ray, inv_d_ray, u_perp, m_star, in_det, m_cl, m0, m1, wm, v0, v1, q_interp, inv_u

            acc *= d_beta_b
            mu_slices[s0:s1, fov_iy, fov_ix] = acc.detach().cpu().numpy().astype(np.float32)
            del q_b, q_full, g_ext, g_w, sino_b, beta_b, d_beta_b, acc
            if require_cuda:
                torch.cuda.empty_cache()
            if progress_bar is not None:
                progress_bar.update(bsz)

    return mu_slices


def reconstruct_fanbeam_fbp(
    ssrb_result: SSRBResult,
    *,
    image_size: int = 512,
    fov_mm: float | None = None,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    water_mu_mm_inv: float = 0.0192,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    inpaint_mode: str = "conjugate_feather",
    lateral_pad_cols: int = 96,
    backend: str = "numpy",
    angle_chunk: int = 64,
    min_hu_clip: float | None = -1024.0,
    show_progress: bool = False,
) -> FBPReconstructionResult:
    """Reconstruct calibrated axial CT slices (`mu [mm^-1]` and `HU`) from `SSRBResult`."""
    n_xy = int(image_size)
    if n_xy < 2:
        raise ValueError(f"image_size must be >= 2, got {image_size}")
    if float(water_mu_mm_inv) <= 0.0:
        raise ValueError(f"water_mu_mm_inv must be positive, got {water_mu_mm_inv}")

    fname = normalize_filter_name(filter_name)
    backend_norm = backend.strip().lower()
    if backend_norm not in SUPPORTED_FBP_BACKENDS:
        raise ValueError(
            f"Unsupported backend={backend!r}; expected one of {sorted(SUPPORTED_FBP_BACKENDS)}"
        )

    scanner_fov = default_scanner_fov_mm(ssrb_result.rho0_mm, ssrb_result.gamma_max_rad)
    active_fov = scanner_fov if fov_mm is None else float(fov_mm)
    if active_fov <= 0.0:
        raise ValueError(f"fov_mm must be positive, got {active_fov}")

    d_xy = active_fov / float(n_xy)
    half_idx = 0.5 * (n_xy - 1.0)
    # Radiological supine axial display convention (matching DICOM LPS / Vendor CT slices):
    #   In DICOM-CT-PD Figure 5 (left-handed table-view system), +x is Patient Right (Liver)
    #   and -x is Patient Left (Heart). Standard radiological viewing (feet-to-head) places:
    #   Col 0 -> +x (Patient Right), Col W-1 -> -x (Patient Left)
    #   Row 0 -> +y (Anterior / Sternum), Row H-1 -> -y (Posterior / Spine & Table)
    cx = float(center_xy_mm[0])
    cy = float(center_xy_mm[1])
    x_mm = cx + (half_idx - np.arange(n_xy, dtype=np.float64)) * d_xy
    y_mm = cy + (half_idx - np.arange(n_xy, dtype=np.float64)) * d_xy

    yy_grid, xx_grid = np.meshgrid(y_mm, x_mm, indexing="ij")
    r_recon = 0.5 * active_fov
    r_scan = 0.5 * scanner_fov
    fov_mask = (
        (((xx_grid - cx) ** 2 + (yy_grid - cy) ** 2) <= (r_recon * r_recon))
        & (((xx_grid * xx_grid) + (yy_grid * yy_grid)) <= (r_scan * r_scan))
    )
    fov_iy, fov_ix = np.where(fov_mask)
    x_fov = xx_grid[fov_iy, fov_ix]
    y_fov = yy_grid[fov_iy, fov_ix]

    n_slices = ssrb_result.n_slices
    prep = extrapolate_sinogram(
        ssrb_result,
        inpaint_mode=inpaint_mode,
        lateral_pad_cols=lateral_pad_cols,
        workers=min(12, os.cpu_count() or 4) if n_slices >= 4 else 1,
    )

    mu_slices = np.zeros((n_slices, n_xy, n_xy), dtype=np.float32)

    u_coords = np.asarray(ssrb_result.u_coords_mm, dtype=np.float64)
    d0 = float(ssrb_result.d0_mm)
    rho0 = float(ssrb_result.rho0_mm)
    du = float(ssrb_result.du_mm)
    du_iso = (rho0 / d0) * du
    cos_alpha = d0 / np.sqrt((d0 * d0) + (u_coords * u_coords))  # (N_u,)

    with ProgressReporter(
        total=n_slices,
        desc="🩻 Reconstructing Fan-Beam FBP",
        unit="slice",
        unit_scale=False,
        colour="cyan",
        enabled=show_progress and n_slices > 0,
    ) as bar:
        if backend_norm in {"torch", "torch_cuda"} and n_slices > 1:
            mu_slices = _reconstruct_fanbeam_fbp_batched_torch(
                prep.completed_sinograms,
                ssrb_result.beta_rad,
                u_coords=u_coords,
                d0_mm=d0,
                rho0_mm=rho0,
                du_mm=du,
                du_iso_mm=du_iso,
                cos_alpha=cos_alpha,
                filter_name=fname,
                cutoff_ratio=cutoff_ratio,
                lateral_pad_cols=lateral_pad_cols,
                angular_mode=ssrb_result.angular_mode,
                x_fov=x_fov,
                y_fov=y_fov,
                fov_iy=fov_iy,
                fov_ix=fov_ix,
                n_xy=n_xy,
                require_cuda=(backend_norm == "torch_cuda"),
                angle_chunk=max(angle_chunk, 128),
                progress_bar=bar,
            )
        else:
            for s_idx in range(n_slices):
                beta_1d = np.asarray(ssrb_result.beta_rad[s_idx], dtype=np.float64)
                w_red = compute_parker_weights(
                    beta_1d,
                    u_coords,
                    d0,
                    angular_mode=ssrb_result.angular_mode,
                )
                g_weighted = (
                    prep.completed_sinograms[s_idx].astype(np.float64) * w_red * cos_alpha[None, :]
                )
                q_filtered = filter_weighted_sinogram_2d(
                    g_weighted,
                    du_iso_mm=du_iso,
                    filter_name=fname,
                    cutoff_ratio=cutoff_ratio,
                    lateral_pad_cols=lateral_pad_cols,
                )
                d_beta = _quadrature_d_beta(beta_1d, ssrb_result.angular_mode)

                if backend_norm == "numpy":
                    mu_fov = _backproject_slice_numpy(
                        q_filtered,
                        beta_1d,
                        d_beta=d_beta,
                        x_fov=x_fov,
                        y_fov=y_fov,
                        rho0_mm=rho0,
                        d0_mm=d0,
                        du_mm=du,
                        angle_chunk=angle_chunk,
                    )
                else:
                    mu_fov = _backproject_slice_torch(
                        q_filtered,
                        beta_1d,
                        d_beta=d_beta,
                        x_fov=x_fov,
                        y_fov=y_fov,
                        rho0_mm=rho0,
                        d0_mm=d0,
                        du_mm=du,
                        require_cuda=(backend_norm == "torch_cuda"),
                        angle_chunk=max(angle_chunk, 128),
                    )

                mu_slices[s_idx, fov_iy, fov_ix] = mu_fov
                bar.update(1)

    hu_slices = mu_to_hu(mu_slices, water_mu_mm_inv=water_mu_mm_inv)
    if min_hu_clip is not None:
        hu_slices = np.maximum(hu_slices, np.float32(min_hu_clip))
    # Set outside circular FOV cleanly to air (-1000 HU, mu = 0.0)
    out_fov = ~fov_mask
    if np.any(out_fov):
        mu_slices[:, out_fov] = np.float32(0.0)
        hu_slices[:, out_fov] = np.float32(-1000.0)

    return FBPReconstructionResult(
        mu_slices=mu_slices,
        hu_slices=hu_slices,
        fov_mask=fov_mask,
        z_slices_mm=ssrb_result.z_slices_mm.copy(),
        y_mm=y_mm,
        x_mm=x_mm,
        fov_mm=active_fov,
        pixel_spacing_mm=d_xy,
        center_xy_mm=(cx, cy),
        water_mu_mm_inv=float(water_mu_mm_inv),
        filter_name=fname,
        cutoff_ratio=float(cutoff_ratio),
        inpaint_mode=prep.inpaint_mode,
        lateral_pad_cols=int(lateral_pad_cols),
        angular_mode=ssrb_result.angular_mode,
        backend=backend_norm,
        sinogram_prep_summary=prep.summary_dict(),
        extrapolated_sinograms=prep.completed_sinograms,
    )


def reconstruct_fanbeam_fbp_scalar_ref(
    ssrb_result: SSRBResult,
    *,
    image_size: int = 32,
    fov_mm: float | None = None,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    water_mu_mm_inv: float = 0.0192,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    inpaint_mode: str = "conjugate_feather",
    lateral_pad_cols: int = 32,
    min_hu_clip: float | None = -1024.0,
) -> FBPReconstructionResult:
    """Explicit scalar loop reference for Fan-Beam FBP parity verification (`validation_protocol.md`)."""
    fname = normalize_filter_name(filter_name)
    n_xy = int(image_size)
    scanner_fov = default_scanner_fov_mm(ssrb_result.rho0_mm, ssrb_result.gamma_max_rad)
    active_fov = scanner_fov if fov_mm is None else float(fov_mm)
    d_xy = active_fov / float(n_xy)
    half_idx = 0.5 * (n_xy - 1.0)
    cx = float(center_xy_mm[0])
    cy = float(center_xy_mm[1])
    x_mm = cx + (half_idx - np.arange(n_xy, dtype=np.float64)) * d_xy
    y_mm = cy + (half_idx - np.arange(n_xy, dtype=np.float64)) * d_xy

    r_recon = 0.5 * active_fov
    r_scan = 0.5 * scanner_fov
    yy_grid, xx_grid = np.meshgrid(y_mm, x_mm, indexing="ij")
    fov_mask = (
        (((xx_grid - cx) ** 2 + (yy_grid - cy) ** 2) <= (r_recon * r_recon))
        & (((xx_grid * xx_grid) + (yy_grid * yy_grid)) <= (r_scan * r_scan))
    )

    prep = extrapolate_sinogram(
        ssrb_result,
        inpaint_mode=inpaint_mode,
        lateral_pad_cols=lateral_pad_cols,
    )

    n_slices = ssrb_result.n_slices
    n_u = ssrb_result.n_u
    c_flat_0 = 0.5 * (n_u - 1.0)
    u_coords = np.asarray(ssrb_result.u_coords_mm, dtype=np.float64)
    d0 = float(ssrb_result.d0_mm)
    rho0 = float(ssrb_result.rho0_mm)
    du = float(ssrb_result.du_mm)
    du_iso = (rho0 / d0) * du
    cos_alpha = d0 / np.sqrt((d0 * d0) + (u_coords * u_coords))

    mu_slices = np.zeros((n_slices, n_xy, n_xy), dtype=np.float32)

    for s_idx in range(n_slices):
        beta_1d = np.asarray(ssrb_result.beta_rad[s_idx], dtype=np.float64)
        w_red = compute_parker_weights(
            beta_1d, u_coords, d0, angular_mode=ssrb_result.angular_mode
        )
        g_weighted = (
            prep.completed_sinograms[s_idx].astype(np.float64) * w_red * cos_alpha[None, :]
        )
        q_filtered = filter_weighted_sinogram_2d(
            g_weighted,
            du_iso_mm=du_iso,
            filter_name=fname,
            cutoff_ratio=cutoff_ratio,
            lateral_pad_cols=lateral_pad_cols,
        )
        d_beta = _quadrature_d_beta(beta_1d, ssrb_result.angular_mode)
        sin_all = [math.sin(float(b)) for b in beta_1d]
        cos_all = [math.cos(float(b)) for b in beta_1d]

        for iy in range(n_xy):
            yv = float(y_mm[iy])
            for ix in range(n_xy):
                xv = float(x_mm[ix])
                if not fov_mask[iy, ix]:
                    continue
                acc = 0.0
                for k in range(beta_1d.size):
                    sb = sin_all[k]
                    cb = cos_all[k]
                    d_ray = max(rho0 + xv * sb - yv * cb, 1e-3)
                    u_star = d0 * (xv * cb + yv * sb) / d_ray
                    m_star = (u_star / du) + c_flat_0
                    if 0.0 <= m_star <= float(n_u - 1):
                        m0 = int(math.floor(m_star))
                        m1 = min(m0 + 1, n_u - 1)
                        wm = m_star - m0
                        qv = (1.0 - wm) * float(q_filtered[k, m0]) + wm * float(q_filtered[k, m1])
                        inv_u = rho0 / d_ray
                        acc += qv * (inv_u * inv_u)
                mu_slices[s_idx, iy, ix] = np.float32(acc * d_beta)

    hu_slices = mu_to_hu(mu_slices, water_mu_mm_inv=water_mu_mm_inv)
    if min_hu_clip is not None:
        hu_slices = np.maximum(hu_slices, np.float32(min_hu_clip))
    out_fov = ~fov_mask
    if np.any(out_fov):
        mu_slices[:, out_fov] = np.float32(0.0)
        hu_slices[:, out_fov] = np.float32(-1000.0)

    return FBPReconstructionResult(
        mu_slices=mu_slices,
        hu_slices=hu_slices,
        fov_mask=fov_mask,
        z_slices_mm=ssrb_result.z_slices_mm.copy(),
        y_mm=y_mm,
        x_mm=x_mm,
        fov_mm=active_fov,
        pixel_spacing_mm=d_xy,
        center_xy_mm=(cx, cy),
        water_mu_mm_inv=float(water_mu_mm_inv),
        filter_name=fname,
        cutoff_ratio=float(cutoff_ratio),
        inpaint_mode=prep.inpaint_mode,
        lateral_pad_cols=int(lateral_pad_cols),
        angular_mode=ssrb_result.angular_mode,
        backend="scalar_ref",
        sinogram_prep_summary=prep.summary_dict(),
        extrapolated_sinograms=prep.completed_sinograms,
    )


def _separable_gaussian_blur_2d(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """Apply separable 1D symmetric convolution with edge replication in pure NumPy."""
    pad = kernel_1d.size // 2
    padded_x = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    blur_x = np.empty_like(img, dtype=np.float64)
    for j in range(img.shape[1]):
        blur_x[:, j] = np.tensordot(padded_x[:, j : j + kernel_1d.size], kernel_1d, axes=([1], [0]))

    padded_y = np.pad(blur_x, ((pad, pad), (0, 0)), mode="edge")
    blur_xy = np.empty_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        blur_xy[i, :] = np.tensordot(padded_y[i : i + kernel_1d.size, :], kernel_1d, axes=([0], [0]))
    return blur_xy


def compute_ssim_2d(
    pred_2d: np.ndarray,
    target_2d: np.ndarray,
    *,
    mask_2d: np.ndarray | None = None,
    data_range: float = 2000.0,
) -> float:
    """Compute exact 11x11 Gaussian Structural Similarity Index (SSIM, Wang et al. 2004)."""
    x = np.asarray(pred_2d, dtype=np.float64)
    y = np.asarray(target_2d, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"SSIM requires matching 2D arrays, got {x.shape} and {y.shape}")

    coords = np.arange(-5, 6, dtype=np.float64)
    g1d = np.exp(-(coords * coords) / (2.0 * (1.5 ** 2)))
    g1d /= np.sum(g1d)

    mu_x = _separable_gaussian_blur_2d(x, g1d)
    mu_y = _separable_gaussian_blur_2d(y, g1d)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = np.maximum(_separable_gaussian_blur_2d(x * x, g1d) - mu_x2, 0.0)
    sigma_y2 = np.maximum(_separable_gaussian_blur_2d(y * y, g1d) - mu_y2, 0.0)
    sigma_xy = _separable_gaussian_blur_2d(x * y, g1d) - mu_xy

    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    if mask_2d is not None:
        m = np.asarray(mask_2d, dtype=bool)
        if np.any(m):
            return float(np.mean(ssim_map[m]))
    return float(np.mean(ssim_map))


def compute_reconstruction_metrics(
    pred_hu: np.ndarray,
    target_hu: np.ndarray,
    *,
    fov_mask: np.ndarray | None = None,
    data_range_hu: float = 2000.0,
    hu_zero_offset: float = 1000.0,
) -> dict[str, float]:
    """Compute quantitative image-domain metrics (`RMSE [HU]`, `MAE [HU]`, `PSNR [dB]`, `SSIM`).

    Note on CT SSIM: Wang et al. (2004) luminance similarity `(2*mu_x*mu_y + C1)/(mu_x^2 + mu_y^2 + C1)`
    requires non-negative intensities (`>= 0`). Signed Hounsfield units place water (`0 HU`) at zero,
    where tiny +/- ripples make `mu_x * mu_y < 0`. Shifting by `hu_zero_offset=1000.0` maps air
    (`-1000 HU`, `mu=0`) to `0.0` (`HU_mod = max(HU + 1000, 0) = 1000 * mu / mu_water >= 0`), matching
    the AAPM Mayo LDCT Challenge evaluation convention while keeping `RMSE`, `MAE`, and `PSNR` invariant.
    """
    p = np.asarray(pred_hu, dtype=np.float64)
    t = np.asarray(target_hu, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"pred_hu shape {p.shape} != target_hu shape {t.shape}")
    if p.ndim == 2:
        p = p[None, :, :]
        t = t[None, :, :]
    if p.ndim != 3:
        raise ValueError(f"Expected 2D or 3D HU arrays, got {p.shape}")

    if fov_mask is not None:
        m2d = np.asarray(fov_mask, dtype=bool)
        m3d = np.broadcast_to(m2d[None, :, :], p.shape)
    else:
        m2d = None
        m3d = np.ones_like(p, dtype=bool)

    diff = (p - t)[m3d]
    if diff.size == 0:
        return {"rmse_hu": 0.0, "mae_hu": 0.0, "max_abs_hu": 0.0, "psnr_db": 99.0, "ssim": 1.0}

    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    psnr = 99.0 if mse <= 1e-12 else float(10.0 * math.log10((float(data_range_hu) ** 2) / mse))
    p_pos = np.maximum(p + float(hu_zero_offset), 0.0)
    t_pos = np.maximum(t + float(hu_zero_offset), 0.0)
    ssim_vals = [
        compute_ssim_2d(p_pos[s], t_pos[s], mask_2d=m2d, data_range=data_range_hu)
        for s in range(p.shape[0])
    ]
    return {
        "rmse_hu": round(rmse, 4),
        "mae_hu": round(mae, 4),
        "max_abs_hu": round(max_abs, 4),
        "psnr_db": round(psnr, 4),
        "ssim": round(float(np.mean(ssim_vals)), 6),
    }


def compute_batch_reconstruction_metrics_torch(
    pred_hu: Any,
    target_hu: Any,
    *,
    fov_mask: Any = None,
    data_range_hu: float = 2000.0,
    hu_zero_offset: float = 1000.0,
    return_per_slice: bool = False,
) -> dict[str, Any]:
    """Exact GPU/CPU batched evaluator for `RMSE [HU]`, `MAE [HU]`, `PSNR [dB]`, and 11x11 Gaussian `SSIM`.

    Matches `compute_reconstruction_metrics` to machine precision (`float64` separable 11x11 Gaussian
    with replicate boundary padding and `hu_zero_offset=1000.0`) while evaluating multi-slice batches
    in a single vectorized PyTorch pass.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        p = pred_hu if isinstance(pred_hu, torch.Tensor) else torch.as_tensor(pred_hu)
        t = target_hu if isinstance(target_hu, torch.Tensor) else torch.as_tensor(target_hu, device=p.device)
        t = t.to(device=p.device)
        if p.ndim == 2:
            p = p.unsqueeze(0)
            t = t.unsqueeze(0)
        elif p.ndim == 4 and p.shape[1] == 1:
            p = p.squeeze(1)
            t = t.squeeze(1)
        if p.shape != t.shape or p.ndim != 3:
            raise ValueError(f"Expected matching (B, H, W) tensors, got {tuple(p.shape)} and {tuple(t.shape)}")

        bsz, h, w = p.shape
        if bsz == 0:
            base = {"rmse_hu": 0.0, "mae_hu": 0.0, "max_abs_hu": 0.0, "psnr_db": 99.0, "ssim": 1.0}
            if return_per_slice:
                empty = np.zeros(0, dtype=np.float64)
                base.update(
                    {
                        "per_slice_rmse_hu": empty,
                        "per_slice_mae_hu": empty,
                        "per_slice_max_abs_hu": empty,
                        "per_slice_psnr_db": empty,
                        "per_slice_ssim": empty,
                    }
                )
            return base

        p64 = p.to(dtype=torch.float64)
        t64 = t.to(dtype=torch.float64)

        if fov_mask is not None:
            m = fov_mask if isinstance(fov_mask, torch.Tensor) else torch.as_tensor(fov_mask, device=p.device)
            m = m.to(device=p.device, dtype=torch.bool)
            if m.ndim == 2:
                m = m.unsqueeze(0).expand(bsz, h, w)
            elif m.ndim == 3 and m.shape[0] == 1 and bsz > 1:
                m = m.expand(bsz, h, w)
        else:
            m = torch.ones((bsz, h, w), dtype=torch.bool, device=p.device)

        m_f = m.to(dtype=torch.float64)
        pix_per_slice = torch.clamp(torch.sum(m_f, dim=(1, 2)), min=1.0)
        diff = (p64 - t64) * m_f

        mse_per_slice = torch.sum(diff * diff, dim=(1, 2)) / pix_per_slice
        rmse_per_slice = torch.sqrt(mse_per_slice)
        mae_per_slice = torch.sum(torch.abs(diff), dim=(1, 2)) / pix_per_slice
        max_per_slice = torch.amax(torch.abs(diff), dim=(1, 2))
        dr2 = float(data_range_hu) ** 2
        psnr_per_slice = torch.where(
            mse_per_slice <= 1e-12,
            torch.full_like(mse_per_slice, 99.0),
            10.0 * torch.log10(dr2 / torch.clamp(mse_per_slice, min=1e-12)),
        )

        coords = torch.arange(-5, 6, dtype=torch.float64, device=p.device)
        g1d = torch.exp(-(coords * coords) / (2.0 * (1.5 ** 2)))
        g1d = g1d / torch.sum(g1d)
        kx = g1d.view(1, 1, 1, 11)
        ky = g1d.view(1, 1, 11, 1)

        def _blur2d(inp: torch.Tensor) -> torch.Tensor:
            u = inp.unsqueeze(1)
            ux = F.conv2d(F.pad(u, (5, 5, 0, 0), mode="replicate"), kx)
            uxy = F.conv2d(F.pad(ux, (0, 0, 5, 5), mode="replicate"), ky)
            return uxy.squeeze(1)

        x_pos = torch.clamp(p64 + float(hu_zero_offset), min=0.0)
        y_pos = torch.clamp(t64 + float(hu_zero_offset), min=0.0)

        mu_x = _blur2d(x_pos)
        mu_y = _blur2d(y_pos)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = torch.clamp(_blur2d(x_pos * x_pos) - mu_x2, min=0.0)
        sigma_y2 = torch.clamp(_blur2d(y_pos * y_pos) - mu_y2, min=0.0)
        sigma_xy = _blur2d(x_pos * y_pos) - mu_xy

        c1 = (0.01 * float(data_range_hu)) ** 2
        c2 = (0.03 * float(data_range_hu)) ** 2
        ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        ssim_per_slice = torch.sum(ssim_map * m_f, dim=(1, 2)) / pix_per_slice

        total_pix = float(torch.sum(m_f).item())
        if total_pix <= 0.0:
            global_mse = 0.0
            global_mae = 0.0
            global_max = 0.0
        else:
            global_mse = float((torch.sum(diff * diff) / total_pix).item())
            global_mae = float((torch.sum(torch.abs(diff)) / total_pix).item())
            global_max = float(torch.max(max_per_slice).item())
        global_rmse = math.sqrt(global_mse)
        global_psnr = 99.0 if global_mse <= 1e-12 else float(10.0 * math.log10(dr2 / global_mse))
        global_ssim = float(torch.mean(ssim_per_slice).item())

        result: dict[str, Any] = {
            "rmse_hu": round(global_rmse, 4),
            "mae_hu": round(global_mae, 4),
            "max_abs_hu": round(global_max, 4),
            "psnr_db": round(global_psnr, 4),
            "ssim": round(global_ssim, 6),
        }
        if return_per_slice:
            result["per_slice_rmse_hu"] = rmse_per_slice.detach().cpu().numpy()
            result["per_slice_mae_hu"] = mae_per_slice.detach().cpu().numpy()
            result["per_slice_max_abs_hu"] = max_per_slice.detach().cpu().numpy()
            result["per_slice_psnr_db"] = psnr_per_slice.detach().cpu().numpy()
            result["per_slice_ssim"] = ssim_per_slice.detach().cpu().numpy()
        return result


def reconstruct_series_reference_streaming(
    series_dir_or_files: str | Path | list[Path],
    z_slices_mm: np.ndarray | list[float],
    *,
    geometry: HelicalGeometry | None = None,
    angular_mode: str = "short_scan",
    active_row_range: tuple[float, float] | None = None,
    image_size: int = 512,
    fov_mm: float | None = None,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    inpaint_mode: str = "conjugate_feather",
    lateral_pad_cols: int = 96,
    backend: str = "numpy",
    workers: int = 4,
    show_progress: bool = False,
) -> tuple[SSRBResult, FBPReconstructionResult]:
    """End-to-end bounded-RAM pipeline: DICOM-CT-PD -> Curved-to-Flat + SSRB -> Parker + FBP -> `x_reference`."""
    if isinstance(series_dir_or_files, list):
        files = [Path(p) for p in series_dir_or_files]
        if geometry is None:
            raise ValueError("geometry must be provided when passing an explicit file list")
    else:
        geometry, files = scan_projection_series(
            series_dir_or_files, workers=max(workers, 4), show_progress=show_progress
        )

    ssrb = rebin_series_ssrb_streaming(
        files,
        z_slices_mm,
        geometry=geometry,
        angular_mode=angular_mode,
        active_row_range=active_row_range,
        workers=workers,
        show_progress=show_progress,
    )
    recon = reconstruct_fanbeam_fbp(
        ssrb,
        image_size=image_size,
        fov_mm=fov_mm,
        center_xy_mm=center_xy_mm,
        water_mu_mm_inv=geometry.water_mu_mm_inv,
        filter_name=filter_name,
        cutoff_ratio=cutoff_ratio,
        inpaint_mode=inpaint_mode,
        lateral_pad_cols=lateral_pad_cols,
        backend=backend,
        show_progress=show_progress,
    )
    return ssrb, recon
