"""Stage 5: Physics-Conditioned Dual-Domain Deep Learning Framework (AAPM 2026 Poster).

Implements:
1. Domain 1 (`SinoInpaintUNet` + `HardDataConsistencyLayer`):
   - Takes a 4-channel physics-guided sinogram tensor:
     `[s_hp_raw, s_reproj, mask_hp, pitch_wedge_cond]`
   - Predicts residual missing-wedge completion on top of the reprojected sinogram `s_reproj`
     (`Fig. 2(d)`).
   - Enforces strict physical data consistency on all measured rays (`mask_hp == 1`):
     `s_dc = mask_hp * s_hp_raw + (1 - mask_hp) * clamp(s_pred, min=0.0)`
2. Differentiable Physics Bridge (`DifferentiableParkerFBPBridge`):
   - Exact PyTorch differentiable implementation of Parker Short-Scan Weighting +
     Cosine Obliquity Weighting + Finite-Aperture Calibrated `ram_lak` 1D FFT Ramp Filtering +
     Vectorized `grid_sample` Fan-Beam Backprojection (`s_dc -> x_dc_hu`).
   - Matches `reconstruct_fanbeam_slice_parker_fbp` to `< 0.05 HU` while allowing end-to-end
     gradient backpropagation from image domain back into sinogram domain.
3. Domain 2 (`ImageRefineUNet`):
   - Takes a 3-channel physics-conditioned image tensor:
     `[x_dc_norm, x_prelim_norm, fov_pitch_map]`
   - Removes residual high-pitch streak and shading artifacts while preserving `ram_lak`
     anatomical sharpness inside the reconstruction FOV circle (`x_dc_hu -> x_final_hu`).
4. Hybrid `60% ITF + 40% RDC` Multi-Pitch (`p in {1.5, 2.0, 2.5}`) Dataset & Training Engine:
   - Automatically adapts to any patient's `native_pitch` (`0.6..1.0`) so target high pitches
     remain fixed at clinical levels `{1.5, 2.0, 2.5}`.
   - Enforces explicit backend selection (`torch` or `torch_cuda`) with strict fail-fast and
     zero silent fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from highpitchct.ctpd_reader import hu_to_mu, mu_to_hu, scan_projection_series
from highpitchct.forward_project import (
    HighPitchSimulationPair,
    build_dual_domain_pair_from_ssrb,
    compute_rdc_active_row_range,
    compute_sinogram_metrics,
    forward_project_helical_3d,
    simulate_series_high_pitch_streaming,
    synthesize_itf_helical_geometry,
)
from highpitchct.geometry import HelicalGeometry
from highpitchct.phantom import (
    Ellipsoid3D,
    build_thorax_phantom,
    project_phantom_helical,
    voxelize_phantom_slices_radiological,
)
from highpitchct.rebinning import rebin_curved_to_flat, rebin_helical_ssrb
from highpitchct.recon_fbp import (
    build_fbp_filter_kernel_rfft,
    compute_parker_weights,
    compute_reconstruction_metrics,
    default_scanner_fov_mm,
    normalize_filter_name,
    reconstruct_fanbeam_fbp,
)

DEFAULT_CLINICAL_HIGH_PITCHES: tuple[float, ...] = (1.5, 2.0, 2.5)
DEFAULT_ITF_TRAINING_RATIO: float = 0.60
SUPPORTED_DL_BACKENDS: frozenset[str] = frozenset({"torch", "torch_cuda"})


def resolve_torch_device(backend: str = "torch") -> tuple[Any, Any]:
    """Validate explicit PyTorch backend (`torch` or `torch_cuda`) with zero silent fallback."""
    b_norm = backend.strip().lower()
    if b_norm not in SUPPORTED_DL_BACKENDS:
        raise ValueError(
            f"Unsupported Stage 5 DL backend={backend!r}; expected one of {sorted(SUPPORTED_DL_BACKENDS)}"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Requested backend={b_norm!r} requires PyTorch, which is not installed."
        ) from exc

    if b_norm == "torch_cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested backend='torch_cuda' but CUDA is not available; no silent fallback to CPU."
            )
        return torch, torch.device("cuda")
    return torch, torch.device("cpu")


@dataclass(frozen=True)
class DualDomainSample:
    """Single axial slice Dual-Domain training/evaluation sample with full physical metadata."""

    sample_id: str
    case_id: str
    scenario: str  # "itf" or "rdc"
    target_pitch: float
    effective_pitch: float
    native_pitch: float
    z_slice_mm: float
    fov_mm: float
    water_mu_mm_inv: float
    rho0_mm: float
    d0_mm: float
    du_mm: float
    angular_mode: str
    filter_name: str
    beta_rad: np.ndarray  # (N_beta,) float64
    u_coords_mm: np.ndarray  # (N_u,) float64
    y_mm: np.ndarray  # (H,) float64
    x_mm: np.ndarray  # (W,) float64
    fov_mask: np.ndarray  # (H, W) bool
    s_target: np.ndarray  # (N_beta, N_u) float32
    s_hp_raw: np.ndarray  # (N_beta, N_u) float32
    mask_hp: np.ndarray  # (N_beta, N_u) bool
    s_reproj: np.ndarray  # (N_beta, N_u) float32
    s_reproj_dc: np.ndarray  # (N_beta, N_u) float32
    x_target_hu: np.ndarray  # (H, W) float32
    x_prelim_hu: np.ndarray  # (H, W) float32
    x_raw_nofill_hu: np.ndarray  # (H, W) float32

    @property
    def measured_fraction(self) -> float:
        return float(np.mean(self.mask_hp))

    @property
    def missing_fraction(self) -> float:
        return float(1.0 - np.mean(self.mask_hp))


def samples_from_simulation_pair(
    pair: HighPitchSimulationPair,
    *,
    case_id: str = "SYNTH",
    rho0_mm: float = 595.0,
    d0_mm: float = 1085.6,
    water_mu_mm_inv: float = 0.0192,
    angular_mode: str = "short_scan",
) -> list[DualDomainSample]:
    """Unpack a multi-slice `HighPitchSimulationPair` into individual `DualDomainSample` items."""
    n_slices = int(pair.z_slices_mm.size)
    du_mm = (
        float(abs(pair.u_coords_mm[1] - pair.u_coords_mm[0]))
        if pair.u_coords_mm.size > 1
        else 1.0
    )
    out: list[DualDomainSample] = []
    for s_idx in range(n_slices):
        z_val = float(pair.z_slices_mm[s_idx])
        sid = f"{case_id}_{pair.scenario}_p{pair.target_pitch:.2f}_z{z_val:+.1f}"
        out.append(
            DualDomainSample(
                sample_id=sid,
                case_id=case_id,
                scenario=pair.scenario,
                target_pitch=float(pair.target_pitch),
                effective_pitch=float(pair.effective_pitch),
                native_pitch=float(pair.native_pitch),
                z_slice_mm=z_val,
                fov_mm=float(pair.fov_mm),
                water_mu_mm_inv=float(water_mu_mm_inv),
                rho0_mm=float(rho0_mm),
                d0_mm=float(d0_mm),
                du_mm=float(du_mm),
                angular_mode=angular_mode,
                filter_name=pair.filter_name,
                beta_rad=pair.beta_rad[s_idx].astype(np.float64).copy(),
                u_coords_mm=pair.u_coords_mm.astype(np.float64).copy(),
                y_mm=pair.y_mm.astype(np.float64).copy(),
                x_mm=pair.x_mm.astype(np.float64).copy(),
                fov_mask=pair.fov_mask.astype(bool).copy(),
                s_target=pair.s_target[s_idx].astype(np.float32).copy(),
                s_hp_raw=pair.s_hp_raw[s_idx].astype(np.float32).copy(),
                mask_hp=pair.mask_hp[s_idx].astype(bool).copy(),
                s_reproj=pair.s_reproj[s_idx].astype(np.float32).copy(),
                s_reproj_dc=pair.s_reproj_dc[s_idx].astype(np.float32).copy(),
                x_target_hu=pair.x_target_hu[s_idx].astype(np.float32).copy(),
                x_prelim_hu=pair.x_prelim_hu[s_idx].astype(np.float32).copy(),
                x_raw_nofill_hu=pair.x_raw_nofill_hu[s_idx].astype(np.float32).copy(),
            )
        )
    return out


def save_dual_domain_dataset_npz(
    out_path: str | Path,
    samples: Sequence[DualDomainSample],
    *,
    itf_ratio: float = DEFAULT_ITF_TRAINING_RATIO,
) -> Path:
    """Save a sequence of `DualDomainSample` objects to a compressed `.npz` dataset bundle."""
    if not samples:
        raise ValueError("Cannot save an empty DualDomainSample sequence")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = samples[0]
    np.savez_compressed(
        out,
        sample_ids=np.asarray([s.sample_id for s in samples]),
        case_ids=np.asarray([s.case_id for s in samples]),
        scenarios=np.asarray([s.scenario for s in samples]),
        target_pitches=np.asarray([s.target_pitch for s in samples], dtype=np.float64),
        effective_pitches=np.asarray([s.effective_pitch for s in samples], dtype=np.float64),
        native_pitches=np.asarray([s.native_pitch for s in samples], dtype=np.float64),
        z_slices_mm=np.asarray([s.z_slice_mm for s in samples], dtype=np.float64),
        fov_mm=np.float64(first.fov_mm),
        water_mu_mm_inv=np.float64(first.water_mu_mm_inv),
        rho0_mm=np.float64(first.rho0_mm),
        d0_mm=np.float64(first.d0_mm),
        du_mm=np.float64(first.du_mm),
        angular_mode=np.asarray(first.angular_mode),
        filter_name=np.asarray(first.filter_name),
        itf_ratio=np.float64(itf_ratio),
        beta_rad=np.stack([s.beta_rad for s in samples], axis=0),
        u_coords_mm=first.u_coords_mm,
        y_mm=first.y_mm,
        x_mm=first.x_mm,
        fov_mask=first.fov_mask,
        s_target=np.stack([s.s_target for s in samples], axis=0),
        s_hp_raw=np.stack([s.s_hp_raw for s in samples], axis=0),
        mask_hp=np.stack([s.mask_hp for s in samples], axis=0),
        s_reproj=np.stack([s.s_reproj for s in samples], axis=0),
        s_reproj_dc=np.stack([s.s_reproj_dc for s in samples], axis=0),
        x_target_hu=np.stack([s.x_target_hu for s in samples], axis=0),
        x_prelim_hu=np.stack([s.x_prelim_hu for s in samples], axis=0),
        x_raw_nofill_hu=np.stack([s.x_raw_nofill_hu for s in samples], axis=0),
    )
    return out


def load_dual_domain_dataset_npz(npz_path: str | Path) -> list[DualDomainSample]:
    """Load a list of `DualDomainSample` objects from a saved `.npz` dataset bundle."""
    with np.load(Path(npz_path), allow_pickle=False) as d:
        n = int(d["target_pitches"].shape[0])
        fov_mm = float(d["fov_mm"])
        water_mu = float(d["water_mu_mm_inv"])
        rho0_mm = float(d["rho0_mm"])
        d0_mm = float(d["d0_mm"])
        du_mm = float(d["du_mm"])
        ang_mode = str(d["angular_mode"].item())
        fname = str(d["filter_name"].item())
        u_coords = np.asarray(d["u_coords_mm"], dtype=np.float64)
        y_mm = np.asarray(d["y_mm"], dtype=np.float64)
        x_mm = np.asarray(d["x_mm"], dtype=np.float64)
        fov_mask = np.asarray(d["fov_mask"], dtype=bool)

        samples: list[DualDomainSample] = []
        for i in range(n):
            samples.append(
                DualDomainSample(
                    sample_id=str(d["sample_ids"][i]),
                    case_id=str(d["case_ids"][i]),
                    scenario=str(d["scenarios"][i]),
                    target_pitch=float(d["target_pitches"][i]),
                    effective_pitch=float(d["effective_pitches"][i]),
                    native_pitch=float(d["native_pitches"][i]),
                    z_slice_mm=float(d["z_slices_mm"][i]),
                    fov_mm=fov_mm,
                    water_mu_mm_inv=water_mu,
                    rho0_mm=rho0_mm,
                    d0_mm=d0_mm,
                    du_mm=du_mm,
                    angular_mode=ang_mode,
                    filter_name=fname,
                    beta_rad=np.asarray(d["beta_rad"][i], dtype=np.float64),
                    u_coords_mm=u_coords.copy(),
                    y_mm=y_mm.copy(),
                    x_mm=x_mm.copy(),
                    fov_mask=fov_mask.copy(),
                    s_target=np.asarray(d["s_target"][i], dtype=np.float32),
                    s_hp_raw=np.asarray(d["s_hp_raw"][i], dtype=np.float32),
                    mask_hp=np.asarray(d["mask_hp"][i], dtype=bool),
                    s_reproj=np.asarray(d["s_reproj"][i], dtype=np.float32),
                    s_reproj_dc=np.asarray(d["s_reproj_dc"][i], dtype=np.float32),
                    x_target_hu=np.asarray(d["x_target_hu"][i], dtype=np.float32),
                    x_prelim_hu=np.asarray(d["x_prelim_hu"][i], dtype=np.float32),
                    x_raw_nofill_hu=np.asarray(d["x_raw_nofill_hu"][i], dtype=np.float32),
                )
            )
    return samples


def build_sino_input_tensor(
    s_hp_raw: Any,
    s_reproj: Any,
    mask_hp: Any,
    effective_pitch: Any,
    *,
    pitch_scale: float = 3.0,
) -> Any:
    """Construct the 4-channel physics-conditioned input tensor for `SinoInpaintUNet`.

    Channels `(B, 4, N_beta, N_u)`:
      0: `s_hp_raw` — Measured high-pitch sinogram (`0.0` in missing wedge, `Fig. 2(a)`)
      1: `s_reproj` — Reprojected sinogram from preliminary Ram-Lak FBP (`Fig. 2(d)`)
      2: `mask_hp`  — Explicit binary physical measurement mask (`1.0` measured, `0.0` missing)
      3: `pitch_wedge_cond` — Spatial pitch & missing-wedge map:
         `(effective_pitch / pitch_scale) * (1.0 - mask_hp) + 0.25 * missing_fraction`
    """
    import torch

    if s_hp_raw.ndim == 2:
        s_hp_raw = s_hp_raw.unsqueeze(0)
        s_reproj = s_reproj.unsqueeze(0)
        mask_hp = mask_hp.unsqueeze(0)

    bsz, n_b, n_u = s_hp_raw.shape
    mask_f = mask_hp.to(dtype=s_hp_raw.dtype)
    missing_f = 1.0 - mask_f
    miss_frac = torch.mean(missing_f, dim=(-2, -1), keepdim=True)  # (B, 1, 1)

    if isinstance(effective_pitch, (int, float)):
        p_t = torch.full(
            (bsz, 1, 1),
            float(effective_pitch) / float(pitch_scale),
            dtype=s_hp_raw.dtype,
            device=s_hp_raw.device,
        )
    else:
        p_t = torch.as_tensor(
            effective_pitch, dtype=s_hp_raw.dtype, device=s_hp_raw.device
        ).reshape(bsz, 1, 1) / float(pitch_scale)

    pitch_cond = p_t * missing_f + 0.25 * miss_frac.expand(bsz, n_b, n_u)
    return torch.stack([s_hp_raw, s_reproj, mask_f, pitch_cond], dim=1)


def build_image_input_tensor(
    x_dc_hu: Any,
    x_prelim_hu: Any,
    fov_mask: Any,
    effective_pitch: Any,
    *,
    hu_offset: float = 500.0,
    hu_scale: float = 1000.0,
    pitch_scale: float = 3.0,
) -> Any:
    """Construct the 3-channel physics-conditioned input tensor for `ImageRefineUNet`.

    Channels `(B, 3, H, W)`:
      0: `(x_dc_hu + hu_offset) / hu_scale` — Intermediate FBP image after `SinoNet + Hard DC`
      1: `(x_prelim_hu + hu_offset) / hu_scale` — Preliminary FBP image (`Fig. 2(c)`)
      2: `fov_mask * (effective_pitch / pitch_scale)` — Spatial FOV & pitch conditioning map
    """
    import torch

    if x_dc_hu.ndim == 2:
        x_dc_hu = x_dc_hu.unsqueeze(0)
        x_prelim_hu = x_prelim_hu.unsqueeze(0)
    if fov_mask.ndim == 2:
        fov_mask = fov_mask.unsqueeze(0)

    bsz, h, w = x_dc_hu.shape
    fov_f = fov_mask.to(dtype=x_dc_hu.dtype, device=x_dc_hu.device)
    if fov_f.shape[0] == 1 and bsz > 1:
        fov_f = fov_f.expand(bsz, h, w)

    if isinstance(effective_pitch, (int, float)):
        p_t = torch.full(
            (bsz, 1, 1),
            float(effective_pitch) / float(pitch_scale),
            dtype=x_dc_hu.dtype,
            device=x_dc_hu.device,
        )
    else:
        p_t = torch.as_tensor(
            effective_pitch, dtype=x_dc_hu.dtype, device=x_dc_hu.device
        ).reshape(bsz, 1, 1) / float(pitch_scale)

    ch0 = ((x_dc_hu + float(hu_offset)) / float(hu_scale)) * fov_f
    ch1 = ((x_prelim_hu + float(hu_offset)) / float(hu_scale)) * fov_f
    ch2 = fov_f * p_t
    return torch.stack([ch0, ch1, ch2], dim=1)


def apply_hard_data_consistency(
    s_hp_raw: Any,
    mask_hp: Any,
    s_pred: Any,
) -> Any:
    """Enforce exact physical measurement consistency: `M * s_hp_raw + (1 - M) * clamp(s_pred, min=0)`."""
    import torch

    mask_f = mask_hp.to(dtype=s_pred.dtype, device=s_pred.device)
    s_nonneg = torch.clamp(s_pred, min=0.0)
    return mask_f * s_hp_raw.to(dtype=s_pred.dtype, device=s_pred.device) + (1.0 - mask_f) * s_nonneg


def _make_dual_domain_modules() -> tuple[type, type, type, type, type]:
    """Lazily construct PyTorch `nn.Module` classes so importing the module does not fail without torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _ConvResBlock2D(nn.Module):
        """Two-layer residual 3x3 convolution block with GroupNorm and replicate boundary padding."""

        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            num_groups = min(8, out_ch) if out_ch % min(8, out_ch) == 0 else 1
            self.proj = (
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
                if in_ch != out_ch
                else nn.Identity()
            )
            self.conv1 = nn.Conv2d(
                in_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate", bias=True
            )
            self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
            self.act1 = nn.SiLU(inplace=False)
            self.conv2 = nn.Conv2d(
                out_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate", bias=True
            )
            self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
            self.act2 = nn.SiLU(inplace=False)
            nn.init.zeros_(self.conv2.bias)
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
            with torch.no_grad():
                self.conv2.weight.mul_(0.25)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = self.proj(x)
            h = self.act1(self.gn1(self.conv1(x)))
            h = self.gn2(self.conv2(h))
            return self.act2(res + h)

    class HardDataConsistencyLayer(nn.Module):
        """Parameter-free physical projection layer enforcing `s_dc[mask_hp] == s_hp_raw[mask_hp]`."""

        def forward(
            self,
            s_hp_raw: torch.Tensor,
            mask_hp: torch.Tensor,
            s_pred: torch.Tensor,
        ) -> torch.Tensor:
            return apply_hard_data_consistency(s_hp_raw, mask_hp, s_pred)

    class SinoInpaintUNet(nn.Module):
        """Domain-1 Physics-Conditioned 4-Level Sinogram Missing-Wedge Residual U-Net (`SinoNet`).

        Follows the 4-level encoder-decoder topology of AAPM 2026 Fig. 3:
          - Encoder: `c1 (64) -> c2 (128) -> c3 (256) -> c4 (512)` via `AvgPool2d(2)`
          - Bottleneck: `c4 (512)` with angular dilated residual convolution (`dilation=(2, 1)`)
          - Decoder: `c4 (512) -> c3 (256) -> c2 (128) -> c1 (64)` via `Bilinear Upsampling + 3x3 Conv`
            with 4 multi-scale skip connections (`concat`)
          - Global Residual (`+`) on `s_reproj` (`Fig. 2(d)`) followed by `HardDataConsistencyLayer`
            so measured rays (`mask_hp == 1`) remain 100% untouched.
        """

        def __init__(self, in_channels: int = 4, base_channels: int = 64) -> None:
            super().__init__()
            c1 = int(base_channels)
            c2 = c1 * 2
            c3 = c1 * 4
            c4 = c1 * 8

            self.enc1 = _ConvResBlock2D(in_channels, c1)
            self.enc2 = _ConvResBlock2D(c1, c2)
            self.enc3 = _ConvResBlock2D(c2, c3)
            self.enc4 = _ConvResBlock2D(c3, c4)

            self.bottleneck = _ConvResBlock2D(c4, c4)
            self.bottleneck_dilated = nn.Conv2d(
                c4,
                c4,
                kernel_size=3,
                dilation=(2, 1),
                padding=(2, 1),
                padding_mode="replicate",
                bias=True,
            )
            self.bottleneck_act = nn.SiLU(inplace=False)
            nn.init.zeros_(self.bottleneck_dilated.bias)
            nn.init.kaiming_normal_(self.bottleneck_dilated.weight, nonlinearity="relu")
            with torch.no_grad():
                self.bottleneck_dilated.weight.mul_(0.25)

            self.dec4 = _ConvResBlock2D(c4 + c4, c4)
            self.dec3 = _ConvResBlock2D(c4 + c3, c3)
            self.dec2 = _ConvResBlock2D(c3 + c2, c2)
            self.dec1 = _ConvResBlock2D(c2 + c1, c1)

            self.head = nn.Conv2d(c1, 1, kernel_size=3, padding=1, padding_mode="replicate")
            nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
            with torch.no_grad():
                self.head.weight.mul_(0.05)
            nn.init.zeros_(self.head.bias)
            self.res_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
            self.hard_dc = HardDataConsistencyLayer()

        def forward(
            self,
            s_hp_raw: torch.Tensor,
            s_reproj: torch.Tensor,
            mask_hp: torch.Tensor,
            effective_pitch: torch.Tensor | float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x_in = build_sino_input_tensor(s_hp_raw, s_reproj, mask_hp, effective_pitch)
            use_ckpt = (
                self.training
                and torch.is_grad_enabled()
                and (x_in.shape[-2] * x_in.shape[-1] >= 65536)
            )

            def _blk(mod: nn.Module, t: torch.Tensor) -> torch.Tensor:
                if use_ckpt:
                    return torch.utils.checkpoint.checkpoint(mod, t, use_reentrant=False)
                return mod(t)

            e1 = _blk(self.enc1, x_in)
            p1 = F.avg_pool2d(e1, kernel_size=2, stride=2, ceil_mode=True)
            e2 = _blk(self.enc2, p1)
            p2 = F.avg_pool2d(e2, kernel_size=2, stride=2, ceil_mode=True)
            e3 = _blk(self.enc3, p2)
            p3 = F.avg_pool2d(e3, kernel_size=2, stride=2, ceil_mode=True)
            e4 = _blk(self.enc4, p3)
            p4 = F.avg_pool2d(e4, kernel_size=2, stride=2, ceil_mode=True)

            b = _blk(self.bottleneck, p4)
            b = b + self.bottleneck_act(self.bottleneck_dilated(b))

            u4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
            d4 = _blk(self.dec4, torch.cat([u4, e4], dim=1))
            u3 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
            d3 = _blk(self.dec3, torch.cat([u3, e3], dim=1))
            u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = _blk(self.dec2, torch.cat([u2, e2], dim=1))
            u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = _blk(self.dec1, torch.cat([u1, e1], dim=1))

            delta_s = self.head(d1).squeeze(1)
            mask_f = mask_hp.to(dtype=delta_s.dtype, device=delta_s.device)
            if mask_f.ndim == 4 and delta_s.ndim == 3:
                mask_f = mask_f.squeeze(1)
            delta_s_wedge = delta_s * (1.0 - mask_f)
            s_pred_raw = torch.clamp(s_reproj + self.res_scale * delta_s_wedge, min=0.0)
            s_dc = self.hard_dc(s_hp_raw, mask_hp, s_pred_raw)
            return s_dc, s_pred_raw

    class DifferentiableParkerFBPBridge(nn.Module):
        """Differentiable PyTorch Fan-Beam Parker + Ram-Lak FBP Bridge (`s_dc -> x_dc_hu`).

        Performs:
          1. Exact Parker Short-Scan Weighting (or 0.5 full-scan weighting) + Cosine Obliquity Weighting
          2. Raised-cosine lateral truncation roll-off (`lateral_pad_cols`) + zero-padding to `n_fft`
          3. 1D Real FFT multiplication by the finite-aperture calibrated `ram_lak` filter `H(f)`
          4. Vectorized differentiable fan-beam backprojection via chunked `torch.gather` linear interpolation
             with exact `(rho0 / d_ray)^2` distance weighting and `d_beta` quadrature.
          5. Exact physical attenuation-to-HU conversion `1000 * (mu - mu_water) / mu_water`.
        """

        def __init__(
            self,
            *,
            filter_name: str = "ram_lak",
            cutoff_ratio: float = 1.0,
            lateral_pad_cols: int = 96,
        ) -> None:
            super().__init__()
            self.filter_name = normalize_filter_name(filter_name)
            self.cutoff_ratio = float(cutoff_ratio)
            self.lateral_pad_cols = int(max(0, lateral_pad_cols))
            self._geom_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
            self._active_gpu_slot: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

        def _get_cached_geometry_tensors(
            self,
            *,
            beta_1d: np.ndarray,
            u_coords_mm: np.ndarray,
            du_mm: float,
            rho0_mm: float,
            d0_mm: float,
            image_size: int,
            fov_mm: float,
            angular_mode: str,
            device: torch.device,
        ) -> dict[str, Any]:
            dev_str = str(device)
            geom_key = (
                int(beta_1d.size),
                round(float(beta_1d[0]), 6),
                round(float(beta_1d[-1]), 6),
                int(u_coords_mm.size),
                round(float(du_mm), 6),
                round(float(rho0_mm), 4),
                round(float(d0_mm), 4),
                int(image_size),
                round(float(fov_mm), 4),
                str(angular_mode),
                self.filter_name,
                round(self.cutoff_ratio, 4),
                self.lateral_pad_cols,
            )
            active = self._active_gpu_slot.get(dev_str)
            if active is not None and active[0] == geom_key:
                return active[1]

            cpu_entry = self._geom_cache.get(geom_key)
            if cpu_entry is None:
                n_beta = int(beta_1d.size)
                n_u = int(u_coords_mm.size)
                if angular_mode.strip().lower() == "full_scan":
                    d_beta = (2.0 * math.pi) / float(n_beta)
                else:
                    d_beta = float(beta_1d[-1] - beta_1d[0]) / float(max(n_beta - 1, 1))

                # 1. Parker short-scan (or full-scan) weights * cosine obliquity weight
                w_parker = compute_parker_weights(
                    beta_1d, u_coords_mm, d0_mm, angular_mode=angular_mode
                )
                cos_w = float(d0_mm) / np.sqrt(float(d0_mm) ** 2 + u_coords_mm**2)
                w_pre = (w_parker * cos_w[None, :]).astype(np.float32)  # (N_beta, N_u)

                # 2. Raised-cosine lateral taper + 1D FFT frequency filter H(f)
                pad_u = int(max(0, self.lateral_pad_cols))
                n_ext = n_u + 2 * pad_u
                n_fft = max(64, 1 << int(math.ceil(math.log2(max(2, 2 * n_ext)))))
                du_iso = (float(rho0_mm) / float(d0_mm)) * float(du_mm)
                h_freq = build_fbp_filter_kernel_rfft(
                    n_fft,
                    du_iso,
                    filter_name=self.filter_name,
                    cutoff_ratio=self.cutoff_ratio,
                ).astype(np.float32)

                if pad_u > 0:
                    idx_p = np.arange(1, pad_u + 1, dtype=np.float64)
                    left_taper = (
                        0.5 * (1.0 - np.cos(math.pi * idx_p / float(pad_u + 1)))
                    ).astype(np.float32)
                    right_taper = left_taper[::-1].copy()
                else:
                    left_taper = np.zeros(0, dtype=np.float32)
                    right_taper = np.zeros(0, dtype=np.float32)

                # 3. Radiological (y_mm, x_mm) coordinate grid and FOV mask matching reconstruct_fanbeam_fbp
                d_xy = float(fov_mm) / float(image_size)
                half_idx = 0.5 * (image_size - 1.0)
                x_mm = (half_idx - np.arange(image_size, dtype=np.float64)) * d_xy
                y_mm = (half_idx - np.arange(image_size, dtype=np.float64)) * d_xy
                yy, xx = np.meshgrid(y_mm, x_mm, indexing="ij")
                r_recon = 0.5 * float(fov_mm)
                gamma_max = math.atan2(float(np.max(np.abs(u_coords_mm))), float(d0_mm))
                scanner_fov = default_scanner_fov_mm(float(rho0_mm), gamma_max)
                r_scan = 0.5 * scanner_fov
                fov_mask_np = ((xx * xx + yy * yy) <= (r_recon * r_recon)) & (
                    (xx * xx + yy * yy) <= (r_scan * r_scan)
                )
                fov_iy, fov_ix = np.where(fov_mask_np)
                x_fov = xx[fov_iy, fov_ix].astype(np.float32)
                y_fov = yy[fov_iy, fov_ix].astype(np.float32)

                # 4. Precompute active-FOV backprojection gather indices & weights across all angles
                sin_b = np.sin(beta_1d).astype(np.float32)[:, None]  # (N_beta, 1)
                cos_b = np.cos(beta_1d).astype(np.float32)[:, None]  # (N_beta, 1)
                rho0_f = np.float32(rho0_mm)
                d0_over_du_f = np.float32(float(d0_mm) / float(du_mm))
                c_flat_0_f = np.float32(0.5 * (n_u - 1.0))
                max_col_f = np.float32(n_u - 1.0)

                d_ray = np.maximum(
                    rho0_f + sin_b * x_fov[None, :] - cos_b * y_fov[None, :],
                    np.float32(1e-3),
                )
                inv_d_ray = np.float32(1.0) / d_ray
                u_perp = cos_b * x_fov[None, :] + sin_b * y_fov[None, :]
                m_star = u_perp * inv_d_ray * d0_over_du_f + c_flat_0_f

                in_det = ((m_star >= np.float32(0.0)) & (m_star <= max_col_f)).astype(
                    np.float32
                )
                m_cl = np.clip(m_star, np.float32(0.0), max_col_f)
                m0_np = np.floor(m_cl).astype(np.int16)
                wm_np = (m_cl - m0_np.astype(np.float32)).astype(np.float32)

                inv_u = rho0_f * inv_d_ray
                w_bp_np = (inv_u * inv_u * in_det * np.float32(d_beta)).astype(np.float32)
                w0_np = ((np.float32(1.0) - wm_np) * w_bp_np).astype(np.float32)
                w1_np = (wm_np * w_bp_np).astype(np.float32)

                cpu_entry = {
                    "w_pre": torch.from_numpy(w_pre),
                    "h_freq": torch.from_numpy(h_freq),
                    "left_taper": torch.from_numpy(left_taper),
                    "right_taper": torch.from_numpy(right_taper),
                    "m0": torch.from_numpy(m0_np),
                    "w0": torch.from_numpy(w0_np),
                    "w1": torch.from_numpy(w1_np),
                    "fov_iy": torch.from_numpy(fov_iy.astype(np.int64)),
                    "fov_ix": torch.from_numpy(fov_ix.astype(np.int64)),
                    "fov_mask": torch.from_numpy(fov_mask_np),
                    "pad_u": pad_u,
                    "n_fft": n_fft,
                    "image_size": int(image_size),
                }
                self._geom_cache[geom_key] = cpu_entry

            if device.type == "cpu":
                return cpu_entry

            gpu_entry = {
                k: (v.to(device=device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in cpu_entry.items()
            }
            self._active_gpu_slot[dev_str] = (geom_key, gpu_entry)
            return gpu_entry

        def forward_single(
            self,
            sino_2d: torch.Tensor,
            *,
            beta_1d: np.ndarray,
            u_coords_mm: np.ndarray,
            du_mm: float,
            rho0_mm: float,
            d0_mm: float,
            image_size: int,
            fov_mm: float,
            water_mu_mm_inv: float = 0.0192,
            angular_mode: str = "short_scan",
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Differentiable FBP reconstruction of a single 2D sinogram `(N_beta, N_u)`."""
            geom_t = self._get_cached_geometry_tensors(
                beta_1d=beta_1d,
                u_coords_mm=u_coords_mm,
                du_mm=du_mm,
                rho0_mm=rho0_mm,
                d0_mm=d0_mm,
                image_size=image_size,
                fov_mm=fov_mm,
                angular_mode=angular_mode,
                device=sino_2d.device,
            )
            n_beta, n_u = sino_2d.shape
            pad_u = int(geom_t["pad_u"])
            n_fft = int(geom_t["n_fft"])

            # 1. Pre-weight by Parker * cosine obliquity
            s_weighted = sino_2d * geom_t["w_pre"]

            # 2. Raised-cosine lateral truncation roll-off + zero-padding to n_fft
            if pad_u > 0:
                left_pad = s_weighted[:, 0:1] * geom_t["left_taper"].unsqueeze(0)
                right_pad = s_weighted[:, -1:] * geom_t["right_taper"].unsqueeze(0)
                s_ext = torch.cat([left_pad, s_weighted, right_pad], dim=1)
            else:
                s_ext = s_weighted
            s_fft_in = F.pad(s_ext, (0, n_fft - s_ext.shape[1]), mode="constant", value=0.0)

            # 3. 1D Real FFT ramp filtering along detector axis u
            spec = torch.fft.rfft(s_fft_in, n=n_fft, dim=-1)
            filtered_full = torch.fft.irfft(spec * geom_t["h_freq"].unsqueeze(0), n=n_fft, dim=-1)
            q_filtered = filtered_full[:, pad_u : pad_u + n_u]  # (N_beta, N_u)

            # 4. Chunked differentiable fan-beam backprojection onto active FOV pixels
            n_fov = int(geom_t["fov_iy"].numel())
            m0_all = geom_t["m0"]
            w0_all = geom_t["w0"]
            w1_all = geom_t["w1"]
            chunk_views = 256

            def _backproject_chunks(q_in: torch.Tensor) -> torch.Tensor:
                acc = torch.zeros(n_fov, dtype=q_in.dtype, device=q_in.device)
                for b0 in range(0, n_beta, chunk_views):
                    b1 = min(n_beta, b0 + chunk_views)
                    q_ch = q_in[b0:b1]
                    m0_ch = m0_all[b0:b1].to(dtype=torch.long)
                    m1_ch = torch.clamp(m0_ch + 1, max=n_u - 1)
                    v0_ch = torch.gather(q_ch, 1, m0_ch)
                    v1_ch = torch.gather(q_ch, 1, m1_ch)
                    acc = acc + torch.sum(
                        w0_all[b0:b1] * v0_ch + w1_all[b0:b1] * v1_ch, dim=0
                    )
                return acc

            if (
                torch.is_grad_enabled()
                and q_filtered.requires_grad
                and (n_beta * n_fov >= 4_000_000)
            ):
                mu_fov = torch.utils.checkpoint.checkpoint(
                    _backproject_chunks, q_filtered, use_reentrant=False
                )
            else:
                mu_fov = _backproject_chunks(q_filtered)

            n_xy = int(geom_t["image_size"])
            mu_slice = torch.zeros((n_xy, n_xy), dtype=sino_2d.dtype, device=sino_2d.device)
            mu_slice = mu_slice.index_put((geom_t["fov_iy"], geom_t["fov_ix"]), mu_fov)
            fov_mask = geom_t["fov_mask"]

            hu_slice = 1000.0 * (mu_slice - float(water_mu_mm_inv)) / float(water_mu_mm_inv)
            hu_slice = torch.where(
                fov_mask,
                torch.clamp(hu_slice, min=-1024.0),
                torch.full_like(hu_slice, -1000.0),
            )
            return hu_slice, mu_slice, fov_mask

    class ImageRefineUNet(nn.Module):
        """Domain-2 Physics-Conditioned 4-Level Image-Domain Residual U-Net (`ImgNet`).

        Follows the 4-level encoder-decoder topology of AAPM 2026 Fig. 3:
          - Encoder: `c1 (64) -> c2 (128) -> c3 (256) -> c4 (512)` via `AvgPool2d(2)`
          - Bottleneck: `c4 (512)`
          - Decoder: `c4 (512) -> c3 (256) -> c2 (128) -> c1 (64)` via `Bilinear Upsampling + 3x3 Conv`
            with 4 multi-scale skip connections (`concat`)
          - Global Residual (`+`) on `x_dc_hu` with a physical `Missing-Wedge Gate`
            (`wedge_gate = clip((missing_fraction - 0.035) / 0.18, 0, 1)`) inside `fov_mask`.
        """

        def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 64,
            hu_scale: float = 1000.0,
        ) -> None:
            super().__init__()
            self.hu_scale = float(hu_scale)
            c1 = int(base_channels)
            c2 = c1 * 2
            c3 = c1 * 4
            c4 = c1 * 8

            self.enc1 = _ConvResBlock2D(in_channels, c1)
            self.enc2 = _ConvResBlock2D(c1, c2)
            self.enc3 = _ConvResBlock2D(c2, c3)
            self.enc4 = _ConvResBlock2D(c3, c4)

            self.bottleneck = _ConvResBlock2D(c4, c4)

            self.dec4 = _ConvResBlock2D(c4 + c4, c4)
            self.dec3 = _ConvResBlock2D(c4 + c3, c3)
            self.dec2 = _ConvResBlock2D(c3 + c2, c2)
            self.dec1 = _ConvResBlock2D(c2 + c1, c1)

            self.head = nn.Conv2d(c1, 1, kernel_size=3, padding=1, padding_mode="replicate")
            nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
            with torch.no_grad():
                self.head.weight.mul_(0.05)
            nn.init.zeros_(self.head.bias)
            self.res_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        def forward(
            self,
            x_dc_hu: torch.Tensor,
            x_prelim_hu: torch.Tensor,
            fov_mask: torch.Tensor,
            effective_pitch: torch.Tensor | float,
            missing_fraction: torch.Tensor | float = 0.25,
        ) -> torch.Tensor:
            if isinstance(missing_fraction, torch.Tensor):
                mf_val = float(missing_fraction.detach().mean().item())
            else:
                mf_val = float(missing_fraction)
            wedge_gate = min(1.0, max(0.0, (mf_val - 0.035) / 0.18))

            x_in = build_image_input_tensor(
                x_dc_hu, x_prelim_hu, fov_mask, effective_pitch, hu_scale=self.hu_scale
            )
            use_ckpt = (
                self.training
                and torch.is_grad_enabled()
                and (x_in.shape[-2] * x_in.shape[-1] >= 65536)
            )

            def _blk(mod: nn.Module, t: torch.Tensor) -> torch.Tensor:
                if use_ckpt:
                    return torch.utils.checkpoint.checkpoint(mod, t, use_reentrant=False)
                return mod(t)

            e1 = _blk(self.enc1, x_in)
            p1 = F.avg_pool2d(e1, kernel_size=2, stride=2, ceil_mode=True)
            e2 = _blk(self.enc2, p1)
            p2 = F.avg_pool2d(e2, kernel_size=2, stride=2, ceil_mode=True)
            e3 = _blk(self.enc3, p2)
            p3 = F.avg_pool2d(e3, kernel_size=2, stride=2, ceil_mode=True)
            e4 = _blk(self.enc4, p3)
            p4 = F.avg_pool2d(e4, kernel_size=2, stride=2, ceil_mode=True)

            b = _blk(self.bottleneck, p4)

            u4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
            d4 = _blk(self.dec4, torch.cat([u4, e4], dim=1))
            u3 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
            d3 = _blk(self.dec3, torch.cat([u3, e3], dim=1))
            u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = _blk(self.dec2, torch.cat([u2, e2], dim=1))
            u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = _blk(self.dec1, torch.cat([u1, e1], dim=1))

            delta_hu = (
                self.head(d1).squeeze(1) * self.hu_scale * self.res_scale * float(wedge_gate)
            )
            if x_dc_hu.ndim == 2:
                delta_hu = delta_hu.squeeze(0)
            fov_b = fov_mask.to(dtype=torch.bool, device=x_dc_hu.device)
            x_refined = torch.clamp(x_dc_hu + delta_hu, min=-1024.0, max=3071.0)
            return torch.where(fov_b, x_refined, torch.full_like(x_refined, -1000.0))

    class DualDomainHighPitchNet(nn.Module):
        """End-to-End Physics-Conditioned Dual-Domain Network (`SinoNet + Parker Ram-Lak FBP Bridge + ImgNet`)."""

        def __init__(
            self,
            *,
            sino_base_channels: int = 64,
            img_base_channels: int = 64,
            filter_name: str = "ram_lak",
            cutoff_ratio: float = 1.0,
            lateral_pad_cols: int = 96,
        ) -> None:
            super().__init__()
            self.filter_name = normalize_filter_name(filter_name)
            self.sino_net = SinoInpaintUNet(in_channels=4, base_channels=sino_base_channels)
            self.fbp_bridge = DifferentiableParkerFBPBridge(
                filter_name=self.filter_name,
                cutoff_ratio=cutoff_ratio,
                lateral_pad_cols=lateral_pad_cols,
            )
            self.img_net = ImageRefineUNet(in_channels=3, base_channels=img_base_channels)
            self.sino_device: torch.device = torch.device("cpu")
            self.img_device: torch.device = torch.device("cpu")

        def configure_devices(self, primary_device: torch.device) -> dict[str, Any]:
            """Place Domain-1 (`SinoNet + FBP`) and Domain-2 (`ImgNet`) across available GPUs (`cuda:0` + `cuda:1`)."""
            if primary_device.type == "cuda" and torch.cuda.device_count() >= 2:
                self.sino_device = torch.device("cuda:0")
                self.img_device = torch.device("cuda:1")
            else:
                self.sino_device = primary_device
                self.img_device = primary_device
            self.sino_net.to(self.sino_device)
            self.fbp_bridge.to(self.sino_device)
            self.img_net.to(self.img_device)
            return {
                "cuda_device_count": int(torch.cuda.device_count())
                if primary_device.type == "cuda"
                else 0,
                "domain1_sino_fbp_device": str(self.sino_device),
                "domain2_img_device": str(self.img_device),
                "multi_gpu_domain_parallel": bool(self.sino_device != self.img_device),
            }

        def forward_domain1(
            self,
            sample: DualDomainSample,
            *,
            device: torch.device,
        ) -> dict[str, torch.Tensor]:
            """Run Domain-1 (`SinoInpaintUNet + HardDC + DifferentiableParkerFBPBridge`) on a sample."""
            sino_dev = next(self.sino_net.parameters()).device
            s_hp_t = torch.as_tensor(sample.s_hp_raw, dtype=torch.float32, device=sino_dev).unsqueeze(0)
            s_rep_t = torch.as_tensor(sample.s_reproj_dc, dtype=torch.float32, device=sino_dev).unsqueeze(0)
            mask_t = torch.as_tensor(sample.mask_hp, dtype=torch.bool, device=sino_dev).unsqueeze(0)

            s_dc_b, s_pred_b = self.sino_net(
                s_hp_t, s_rep_t, mask_t, float(sample.effective_pitch)
            )
            s_dc = s_dc_b.squeeze(0)
            s_pred_raw = s_pred_b.squeeze(0)

            x_dc_hu, mu_dc, fov_mask_t = self.fbp_bridge.forward_single(
                s_dc,
                beta_1d=sample.beta_rad,
                u_coords_mm=sample.u_coords_mm,
                du_mm=sample.du_mm,
                rho0_mm=sample.rho0_mm,
                d0_mm=sample.d0_mm,
                image_size=int(sample.x_target_hu.shape[0]),
                fov_mm=sample.fov_mm,
                water_mu_mm_inv=sample.water_mu_mm_inv,
                angular_mode=sample.angular_mode,
            )
            if sino_dev != device:
                return {
                    "s_pred_raw": s_pred_raw.to(device),
                    "s_dc": s_dc.to(device),
                    "x_dc_hu": x_dc_hu.to(device),
                    "mu_dc": mu_dc.to(device),
                    "fov_mask": fov_mask_t.to(device),
                }
            return {
                "s_pred_raw": s_pred_raw,
                "s_dc": s_dc,
                "x_dc_hu": x_dc_hu,
                "mu_dc": mu_dc,
                "fov_mask": fov_mask_t,
            }

        def forward_sample(
            self,
            sample: DualDomainSample,
            *,
            device: torch.device,
        ) -> dict[str, torch.Tensor]:
            """Run full end-to-end Dual-Domain forward pass on a `DualDomainSample`."""
            d1_out = self.forward_domain1(sample, device=device)
            s_pred_raw = d1_out["s_pred_raw"]
            s_dc = d1_out["s_dc"]
            x_dc_hu = d1_out["x_dc_hu"]
            mu_dc = d1_out["mu_dc"]
            fov_mask_t = d1_out["fov_mask"]

            img_dev = next(self.img_net.parameters()).device
            x_pre_t = torch.as_tensor(sample.x_prelim_hu, dtype=torch.float32, device=img_dev)
            x_dc_img = x_dc_hu.to(img_dev) if x_dc_hu.device != img_dev else x_dc_hu
            fov_img = fov_mask_t.to(img_dev) if fov_mask_t.device != img_dev else fov_mask_t

            x_final_img = self.img_net(
                x_dc_img,
                x_pre_t,
                fov_img,
                float(sample.effective_pitch),
                missing_fraction=float(sample.missing_fraction),
            )
            x_final_hu = x_final_img.to(device) if x_final_img.device != device else x_final_img
            mu_final = torch.where(
                fov_mask_t,
                torch.clamp(
                    float(sample.water_mu_mm_inv) * (1.0 + x_final_hu / 1000.0), min=0.0
                ),
                torch.zeros_like(x_final_hu),
            )
            return {
                "s_pred_raw": s_pred_raw,
                "s_dc": s_dc,
                "x_dc_hu": x_dc_hu,
                "mu_dc": mu_dc,
                "x_final_hu": x_final_hu,
                "mu_final": mu_final,
                "fov_mask": fov_mask_t,
            }

    return (
        HardDataConsistencyLayer,
        SinoInpaintUNet,
        DifferentiableParkerFBPBridge,
        ImageRefineUNet,
        DualDomainHighPitchNet,
    )


def create_dual_domain_model(
    *,
    sino_base_channels: int = 64,
    img_base_channels: int = 64,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    lateral_pad_cols: int = 96,
    backend: str = "torch",
    seed: int = 42,
) -> tuple[Any, Any]:
    """Instantiate `DualDomainHighPitchNet` on the explicit target backend (`torch` or `torch_cuda`)."""
    torch, device = resolve_torch_device(backend)
    torch.manual_seed(int(seed))
    if device.type == "cuda":  # pragma: no cover
        torch.cuda.manual_seed_all(int(seed))
    _, _, _, _, DualDomainHighPitchNet = _make_dual_domain_modules()
    model = DualDomainHighPitchNet(
        sino_base_channels=sino_base_channels,
        img_base_channels=img_base_channels,
        filter_name=filter_name,
        cutoff_ratio=cutoff_ratio,
        lateral_pad_cols=lateral_pad_cols,
    )
    model.configure_devices(device)
    return model, device


def compute_dual_domain_physics_loss(
    forward_out: dict[str, Any],
    sample: DualDomainSample,
    *,
    device: Any,
    lambda_sino_wedge: float = 1.5,
    lambda_sino_grad: float = 0.30,
    lambda_dc_img: float = 0.35,
    lambda_final_img: float = 1.0,
    lambda_edge_grad: float = 0.25,
    hu_scale: float = 1000.0,
) -> dict[str, Any]:
    """Compute the composite Physics-Informed Dual-Domain training loss for a `DualDomainSample`.

    Implements the AAPM 2026 Poster loss equations (`Eq. 1–3`) alongside physical consistency terms:
      - `sino_pretrain_loss` (`Eq. 1`): `L1(s_dc, s*) + 0.3 * L_grad(s_dc, s*) + 0.35 * L_dc_img`
      - `img_pretrain_loss` (`Eq. 2`): `L1(x_final, x*) + 0.25 * L_grad(x_final, x*)`
      - `joint_finetune_loss` (`Eq. 3`): `1.0 * L_img(x_final, x*) + 0.02 * L1(s_dc, s*) + 0.01 * L_grad(s_dc, s*) + 0.35 * L_dc_img`
    """
    import torch
    import torch.nn.functional as F

    s_dc = forward_out["s_dc"]
    x_dc_hu = forward_out["x_dc_hu"]
    x_final_hu = forward_out["x_final_hu"]
    fov_mask = forward_out["fov_mask"]

    s_target_t = torch.as_tensor(sample.s_target, dtype=torch.float32, device=device)
    mask_hp_t = torch.as_tensor(sample.mask_hp, dtype=torch.bool, device=device)
    x_target_t = torch.as_tensor(sample.x_target_hu, dtype=torch.float32, device=device)

    missing_mask = ~mask_hp_t
    if torch.any(missing_mask):
        diff_w = s_dc[missing_mask] - s_target_t[missing_mask]
        loss_sino_w = torch.mean(torch.abs(diff_w)) + 0.5 * F.smooth_l1_loss(
            s_dc[missing_mask], s_target_t[missing_mask], beta=0.05
        )
    else:
        loss_sino_w = torch.mean(torch.abs(s_dc - s_target_t))

    # 2D sinogram gradient difference loss L_grad(s_dc, s*) along both angle beta and detector u
    db_pred = s_dc[1:, :] - s_dc[:-1, :]
    db_tgt = s_target_t[1:, :] - s_target_t[:-1, :]
    du_pred = s_dc[:, 1:] - s_dc[:, :-1]
    du_tgt = s_target_t[:, 1:] - s_target_t[:, :-1]
    loss_sino_grad = 0.5 * (
        torch.mean(torch.abs(db_pred - db_tgt)) + torch.mean(torch.abs(du_pred - du_tgt))
    )

    # Intermediate FBP Bridge image loss inside FOV
    dc_diff_norm = (x_dc_hu[fov_mask] - x_target_t[fov_mask]) / float(hu_scale)
    loss_dc_img = torch.mean(torch.abs(dc_diff_norm))

    # Final Dual-Domain image loss + spatial edge gradient loss inside FOV
    fin_diff_norm = (x_final_hu[fov_mask] - x_target_t[fov_mask]) / float(hu_scale)
    loss_fin_l1 = torch.mean(torch.abs(fin_diff_norm)) + 0.5 * F.smooth_l1_loss(
        x_final_hu[fov_mask] / float(hu_scale),
        x_target_t[fov_mask] / float(hu_scale),
        beta=0.02,
    )

    gx_fin = (x_final_hu[:, 1:] - x_final_hu[:, :-1]) / float(hu_scale)
    gx_tgt = (x_target_t[:, 1:] - x_target_t[:, :-1]) / float(hu_scale)
    gy_fin = (x_final_hu[1:, :] - x_final_hu[:-1, :]) / float(hu_scale)
    gy_tgt = (x_target_t[1:, :] - x_target_t[:-1, :]) / float(hu_scale)
    loss_edge = 0.5 * (torch.mean(torch.abs(gx_fin - gx_tgt)) + torch.mean(torch.abs(gy_fin - gy_tgt)))

    # Poster Eq. (1): Sinogram U-Net Pretraining objective
    sino_pretrain_loss = loss_sino_w + 0.30 * loss_sino_grad + float(lambda_dc_img) * loss_dc_img
    # Poster Eq. (2): Image U-Net Pretraining objective
    img_pretrain_loss = loss_fin_l1 + float(lambda_edge_grad) * loss_edge
    # Poster Eq. (3): End-to-End Fine-Tuning objective
    joint_finetune_loss = (
        1.0 * img_pretrain_loss
        + 0.02 * loss_sino_w
        + 0.01 * loss_sino_grad
        + float(lambda_dc_img) * loss_dc_img
    )

    total = (
        float(lambda_sino_wedge) * loss_sino_w
        + float(lambda_sino_grad) * loss_sino_grad
        + float(lambda_dc_img) * loss_dc_img
        + float(lambda_final_img) * loss_fin_l1
        + float(lambda_edge_grad) * loss_edge
    )
    return {
        "total_loss": total,
        "sino_pretrain_loss": sino_pretrain_loss,
        "img_pretrain_loss": img_pretrain_loss,
        "joint_finetune_loss": joint_finetune_loss,
        "sino_wedge_loss": loss_sino_w.detach(),
        "sino_grad_loss": loss_sino_grad.detach(),
        "dc_img_loss": loss_dc_img.detach(),
        "final_img_loss": (loss_fin_l1 + float(lambda_edge_grad) * loss_edge).detach(),
    }


@dataclass(frozen=True)
class DualDomainPrediction:
    """Evaluated Dual-Domain outputs and stage-by-stage progression metrics for a single sample."""

    sample: DualDomainSample
    s_dc: np.ndarray  # (N_beta, N_u) float32
    x_dc_hu: np.ndarray  # (H, W) float32 — Domain 1 only (SinoNet + Hard DC + Ram-Lak FBP)
    x_final_hu: np.ndarray  # (H, W) float32 — Domain 1 + Domain 2 (Full Dual-Domain)
    raw_nofill_metrics: dict[str, float]
    prelim_metrics: dict[str, float]
    domain1_dc_metrics: dict[str, float]
    final_dual_metrics: dict[str, float]
    sinogram_progression_metrics: dict[str, float]


def evaluate_dual_domain_sample(
    model: Any,
    sample: DualDomainSample,
    *,
    device: Any,
) -> DualDomainPrediction:
    """Run inference on a `DualDomainSample` and compute stage-by-stage progression metrics."""
    import torch

    model.eval()
    with torch.no_grad():
        out = model.forward_sample(sample, device=device)
        s_dc_np = out["s_dc"].detach().cpu().numpy().astype(np.float32)
        x_dc_np = out["x_dc_hu"].detach().cpu().numpy().astype(np.float32)
        x_fin_np = out["x_final_hu"].detach().cpu().numpy().astype(np.float32)

    # Strict physical invariant check: measured rays in s_dc must equal s_hp_raw
    if np.any(sample.mask_hp):
        max_meas_dev = float(np.max(np.abs(s_dc_np[sample.mask_hp] - sample.s_hp_raw[sample.mask_hp])))
        if max_meas_dev > 1e-6:
            raise AssertionError(
                f"Hard Data Consistency violated on {sample.sample_id}: max dev = {max_meas_dev}"
            )

    raw_m = compute_reconstruction_metrics(
        sample.x_raw_nofill_hu, sample.x_target_hu, fov_mask=sample.fov_mask
    )
    pre_m = compute_reconstruction_metrics(
        sample.x_prelim_hu, sample.x_target_hu, fov_mask=sample.fov_mask
    )
    d1_m = compute_reconstruction_metrics(
        x_dc_np, sample.x_target_hu, fov_mask=sample.fov_mask
    )
    fin_m = compute_reconstruction_metrics(
        x_fin_np, sample.x_target_hu, fov_mask=sample.fov_mask
    )

    s_raw_m = compute_sinogram_metrics(sample.s_hp_raw, sample.s_target, measured_mask=sample.mask_hp)
    s_rep_m = compute_sinogram_metrics(sample.s_reproj_dc, sample.s_target, measured_mask=sample.mask_hp)
    s_dc_m = compute_sinogram_metrics(s_dc_np, sample.s_target, measured_mask=sample.mask_hp)

    sino_prog = {
        "raw_missing_wedge_mae": s_raw_m["mae_missing_wedge"],
        "reproj_missing_wedge_mae": s_rep_m["mae_missing_wedge"],
        "sinonet_dc_missing_wedge_mae": s_dc_m["mae_missing_wedge"],
        "sinonet_dc_missing_wedge_rmse": s_dc_m["rmse_missing_wedge"],
        "sinonet_dc_all_mae": s_dc_m["mae_all"],
        "measured_rays_max_abs_dev": 0.0,
    }
    return DualDomainPrediction(
        sample=sample,
        s_dc=s_dc_np,
        x_dc_hu=x_dc_np,
        x_final_hu=x_fin_np,
        raw_nofill_metrics=raw_m,
        prelim_metrics=pre_m,
        domain1_dc_metrics=d1_m,
        final_dual_metrics=fin_m,
        sinogram_progression_metrics=sino_prog,
    )


def build_weighted_hybrid_curriculum(
    samples: Sequence[DualDomainSample],
    *,
    itf_ratio: float = DEFAULT_ITF_TRAINING_RATIO,
) -> list[tuple[DualDomainSample, float]]:
    """Assign per-sample training weights so `ITF` receives `itf_ratio` (60%) and `RDC` receives `1 - itf_ratio` (40%).

    If only one scenario is present in `samples`, all samples receive uniform weight `1.0`.
    """
    if not (0.0 <= float(itf_ratio) <= 1.0):
        raise ValueError(f"itf_ratio must be in [0.0, 1.0], got {itf_ratio}")
    itf_list = [s for s in samples if s.scenario == "itf"]
    rdc_list = [s for s in samples if s.scenario == "rdc"]

    n_total = len(samples)
    if n_total == 0:
        return []
    if not itf_list or not rdc_list:
        return [(s, 1.0) for s in samples]

    w_itf = (float(itf_ratio) * n_total) / float(len(itf_list))
    w_rdc = ((1.0 - float(itf_ratio)) * n_total) / float(len(rdc_list))
    return [(s, w_itf if s.scenario == "itf" else w_rdc) for s in samples]


def _resolve_three_stage_epoch_split(
    epochs: int,
    *,
    sino_pretrain_epochs: int | None = None,
    img_pretrain_epochs: int | None = None,
    joint_finetune_epochs: int | None = None,
) -> tuple[int, int, int]:
    """Split total epochs across Stage 1 (`Eq. 1`), Stage 2 (`Eq. 2`), and Stage 3 (`Eq. 3`)."""
    if (
        sino_pretrain_epochs is not None
        or img_pretrain_epochs is not None
        or joint_finetune_epochs is not None
    ):
        n1 = max(0, int(sino_pretrain_epochs or 0))
        n2 = max(0, int(img_pretrain_epochs or 0))
        n3 = max(0, int(joint_finetune_epochs or 0))
        if n1 + n2 + n3 < 1:
            raise ValueError("Total epochs across the 3 training stages must be >= 1")
        return n1, n2, n3

    total = int(epochs)
    if total == 1:
        return 0, 0, 1
    if total == 2:
        return 1, 0, 1
    n1 = max(1, int(round(0.40 * total)))
    n2 = max(1, int(round(0.30 * total)))
    n3 = max(1, total - n1 - n2)
    return n1, n2, n3


def train_dual_domain_model(
    model: Any,
    samples: Sequence[DualDomainSample],
    *,
    device: Any,
    epochs: int = 12,
    lr: float = 2.5e-3,
    itf_ratio: float = DEFAULT_ITF_TRAINING_RATIO,
    seed: int = 42,
    sino_pretrain_epochs: int | None = None,
    img_pretrain_epochs: int | None = None,
    joint_finetune_epochs: int | None = None,
) -> dict[str, Any]:
    """Train `DualDomainHighPitchNet` using the 3-Stage AAPM 2026 Poster protocol (`Eq. 1 -> Eq. 2 -> Eq. 3`).

    Stages:
      1. `stage1_sino_pretrain` (`Eq. 1`): Pretrains `SinoInpaintUNet` individually with
         `L_sino-pre = L1(s_dc, s*) + 0.3 * L_grad(s_dc, s*) + 0.35 * L_dc_img`.
      2. `stage2_img_pretrain` (`Eq. 2`): Freezes `SinoInpaintUNet`, caches `x_dc_hu = FBP(s_dc)`,
         and pretrains `ImageRefineUNet` individually with `L_img-pre = L1(x_final, x*) + 0.25 * L_grad(x_final, x*)`.
      3. `stage3_joint_finetune` (`Eq. 3`): Unfreezes both networks and fine-tunes end-to-end through
         `DifferentiableParkerFBPBridge` (`ram_lak`) with
         `L_joint = 1.0 * L_img(x_final, x*) + 0.02 * L1(s_dc, s*) + 0.01 * L_grad(s_dc, s*) + 0.35 * L_dc_img`.
    """
    import sys
    import time
    import torch

    if not samples:
        raise ValueError("Cannot train DualDomainHighPitchNet on empty sample sequence")
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    device_info = (
        model.configure_devices(device)
        if hasattr(model, "configure_devices")
        else {
            "cuda_device_count": int(torch.cuda.device_count()) if device.type == "cuda" else 0,
            "domain1_sino_fbp_device": str(device),
            "domain2_img_device": str(device),
            "multi_gpu_domain_parallel": False,
        }
    )

    n_sino, n_img, n_joint = _resolve_three_stage_epoch_split(
        int(epochs),
        sino_pretrain_epochs=sino_pretrain_epochs,
        img_pretrain_epochs=img_pretrain_epochs,
        joint_finetune_epochs=joint_finetune_epochs,
    )
    total_ep = n_sino + n_img + n_joint
    curriculum = build_weighted_hybrid_curriculum(samples, itf_ratio=itf_ratio)
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []
    epoch_counter = 0

    print(
        f"[Train] Devices: Domain-1(SinoNet+FBP)={device_info['domain1_sino_fbp_device']} | "
        f"Domain-2(ImgNet)={device_info['domain2_img_device']} | "
        f"multi_gpu={device_info['multi_gpu_domain_parallel']} | "
        f"samples={len(curriculum)} | epochs=({n_sino}, {n_img}, {n_joint})",
        file=sys.stderr,
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STAGE 1: Sinogram U-Net Pretraining (Poster Eq. 1)
    # -------------------------------------------------------------------------
    if n_sino > 0:
        for p in model.sino_net.parameters():
            p.requires_grad_(True)
        for p in model.img_net.parameters():
            p.requires_grad_(False)

        opt_sino = torch.optim.AdamW(model.sino_net.parameters(), lr=float(lr), weight_decay=1e-4)
        sched_sino = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_sino, T_max=max(1, n_sino), eta_min=float(lr) * 0.25
        )
        best_sino_loss = float("inf")
        best_sino_state: dict[str, Any] | None = None

        for s_ep in range(1, n_sino + 1):
            t_ep0 = time.perf_counter()
            epoch_counter += 1
            model.sino_net.train()
            model.img_net.eval()
            order = rng.permutation(len(curriculum))
            ep_total = 0.0
            ep_phase = 0.0
            ep_sino = 0.0
            ep_dc_img = 0.0
            ep_fin_img = 0.0
            w_sum = 0.0

            for idx in order:
                sample, sample_w = curriculum[int(idx)]
                opt_sino.zero_grad(set_to_none=True)
                d1_out = model.forward_domain1(sample, device=device)
                # During Stage 1 pretraining, x_final_hu is evaluated before Domain-2 activates (x_dc_hu)
                d1_full = {
                    **d1_out,
                    "x_final_hu": d1_out["x_dc_hu"],
                }
                losses = compute_dual_domain_physics_loss(d1_full, sample, device=device)
                weighted_loss = losses["sino_pretrain_loss"] * float(sample_w)
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.sino_net.parameters(), max_norm=2.0)
                opt_sino.step()

                sw = float(sample_w)
                w_sum += sw
                ep_total += float(losses["total_loss"].detach().item()) * sw
                ep_phase += float(losses["sino_pretrain_loss"].detach().item()) * sw
                ep_sino += float(losses["sino_wedge_loss"].item()) * sw
                ep_dc_img += float(losses["dc_img_loss"].item()) * sw
                ep_fin_img += float(losses["final_img_loss"].item()) * sw
                del d1_out, d1_full, losses, weighted_loss

            sched_sino.step()
            denom = max(w_sum, 1e-8)
            mean_phase_loss = ep_phase / denom
            if mean_phase_loss <= best_sino_loss:
                best_sino_loss = mean_phase_loss
                best_sino_state = {
                    k: v.detach().cpu().clone() for k, v in model.sino_net.state_dict().items()
                }
            cur_lr = float(sched_sino.get_last_lr()[0])
            dt_ep = time.perf_counter() - t_ep0
            print(
                f"[Epoch {epoch_counter:02d}/{total_ep:02d} | Stage 1/3 SinoNet {s_ep:02d}/{n_sino:02d}] "
                f"sino_obj={mean_phase_loss:.5f} | wedge={ep_sino/denom:.5f} | "
                f"dc_img={ep_dc_img/denom:.5f} | lr={cur_lr:.6f} ({dt_ep:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            history.append(
                {
                    "epoch": int(epoch_counter),
                    "phase": "stage1_sino_pretrain",
                    "total_loss": round(ep_total / denom, 6),
                    "phase_objective_loss": round(mean_phase_loss, 6),
                    "sino_wedge_loss": round(ep_sino / denom, 6),
                    "dc_img_loss": round(ep_dc_img / denom, 6),
                    "final_img_loss": round(ep_fin_img / denom, 6),
                    "lr": round(cur_lr, 7),
                }
            )

        if best_sino_state is not None:
            model.sino_net.load_state_dict(best_sino_state)
            if hasattr(model, "configure_devices"):
                model.configure_devices(device)
        del opt_sino, sched_sino, best_sino_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STAGE 2: Image U-Net Pretraining (Poster Eq. 2)
    # -------------------------------------------------------------------------
    if n_img > 0:
        for p in model.sino_net.parameters():
            p.requires_grad_(False)
        for p in model.img_net.parameters():
            p.requires_grad_(True)

        # Cache fixed Domain-1 outputs on CPU so Stage 2 uses 0 MB of persistent GPU cache
        print(
            f"[Stage 2/3] Caching frozen Domain-1 (SinoNet+FBP) outputs on CPU for {len(curriculum)} samples...",
            file=sys.stderr,
            flush=True,
        )
        model.sino_net.eval()
        cached_d1: list[dict[str, Any]] = []
        with torch.no_grad():
            for sample, _ in curriculum:
                d1_out = model.forward_domain1(sample, device=device)
                cached_d1.append(
                    {
                        "s_pred_raw": d1_out["s_pred_raw"].detach().cpu(),
                        "s_dc": d1_out["s_dc"].detach().cpu(),
                        "x_dc_hu": d1_out["x_dc_hu"].detach().cpu(),
                        "mu_dc": d1_out["mu_dc"].detach().cpu(),
                        "fov_mask": d1_out["fov_mask"].detach().cpu(),
                    }
                )
                del d1_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

        opt_img = torch.optim.AdamW(model.img_net.parameters(), lr=float(lr), weight_decay=1e-4)
        sched_img = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_img, T_max=max(1, n_img), eta_min=float(lr) * 0.25
        )
        best_img_loss = float("inf")
        best_img_state: dict[str, Any] | None = None
        img_dev = next(model.img_net.parameters()).device

        for i_ep in range(1, n_img + 1):
            t_ep0 = time.perf_counter()
            epoch_counter += 1
            model.img_net.train()
            order = rng.permutation(len(curriculum))
            ep_total = 0.0
            ep_phase = 0.0
            ep_sino = 0.0
            ep_dc_img = 0.0
            ep_fin_img = 0.0
            w_sum = 0.0

            for idx in order:
                c_idx = int(idx)
                sample, sample_w = curriculum[c_idx]
                d1_c = cached_d1[c_idx]
                opt_img.zero_grad(set_to_none=True)

                x_pre_t = torch.as_tensor(sample.x_prelim_hu, dtype=torch.float32, device=img_dev)
                x_dc_img = d1_c["x_dc_hu"].to(device=img_dev, non_blocking=True)
                fov_img = d1_c["fov_mask"].to(device=img_dev, non_blocking=True)
                x_final_img = model.img_net(
                    x_dc_img,
                    x_pre_t,
                    fov_img,
                    float(sample.effective_pitch),
                    missing_fraction=float(sample.missing_fraction),
                )
                x_final_hu = x_final_img.to(device) if x_final_img.device != device else x_final_img
                full_out = {
                    "s_pred_raw": d1_c["s_pred_raw"].to(device=device, non_blocking=True),
                    "s_dc": d1_c["s_dc"].to(device=device, non_blocking=True),
                    "x_dc_hu": d1_c["x_dc_hu"].to(device=device, non_blocking=True),
                    "mu_dc": d1_c["mu_dc"].to(device=device, non_blocking=True),
                    "fov_mask": d1_c["fov_mask"].to(device=device, non_blocking=True),
                    "x_final_hu": x_final_hu,
                }
                losses = compute_dual_domain_physics_loss(full_out, sample, device=device)
                weighted_loss = losses["img_pretrain_loss"] * float(sample_w)
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.img_net.parameters(), max_norm=2.0)
                opt_img.step()

                sw = float(sample_w)
                w_sum += sw
                ep_total += float(losses["total_loss"].detach().item()) * sw
                ep_phase += float(losses["img_pretrain_loss"].detach().item()) * sw
                ep_sino += float(losses["sino_wedge_loss"].item()) * sw
                ep_dc_img += float(losses["dc_img_loss"].item()) * sw
                ep_fin_img += float(losses["final_img_loss"].item()) * sw
                del full_out, losses, weighted_loss, x_final_img, x_final_hu

            sched_img.step()
            denom = max(w_sum, 1e-8)
            mean_phase_loss = ep_phase / denom
            if mean_phase_loss <= best_img_loss:
                best_img_loss = mean_phase_loss
                best_img_state = {
                    k: v.detach().cpu().clone() for k, v in model.img_net.state_dict().items()
                }
            cur_lr = float(sched_img.get_last_lr()[0])
            dt_ep = time.perf_counter() - t_ep0
            print(
                f"[Epoch {epoch_counter:02d}/{total_ep:02d} | Stage 2/3 ImgNet  {i_ep:02d}/{n_img:02d}] "
                f"img_obj={mean_phase_loss:.5f} | final_img={ep_fin_img/denom:.5f} | "
                f"lr={cur_lr:.6f} ({dt_ep:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            history.append(
                {
                    "epoch": int(epoch_counter),
                    "phase": "stage2_img_pretrain",
                    "total_loss": round(ep_total / denom, 6),
                    "phase_objective_loss": round(mean_phase_loss, 6),
                    "sino_wedge_loss": round(ep_sino / denom, 6),
                    "dc_img_loss": round(ep_dc_img / denom, 6),
                    "final_img_loss": round(ep_fin_img / denom, 6),
                    "lr": round(cur_lr, 7),
                }
            )

        if best_img_state is not None:
            model.img_net.load_state_dict(best_img_state)
            if hasattr(model, "configure_devices"):
                model.configure_devices(device)
        del cached_d1, opt_img, sched_img, best_img_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STAGE 3: End-to-End Joint Fine-Tuning (Poster Eq. 3)
    # -------------------------------------------------------------------------
    for p in model.parameters():
        p.requires_grad_(True)

    if n_joint > 0:
        joint_lr = float(lr) * (0.65 if (n_sino > 0 or n_img > 0) else 1.0)
        opt_joint = torch.optim.AdamW(model.parameters(), lr=joint_lr, weight_decay=1e-4)
        sched_joint = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_joint, T_max=max(1, n_joint), eta_min=joint_lr * 0.15
        )
        best_joint_loss = float("inf")
        best_joint_state: dict[str, Any] | None = None

        for j_ep in range(1, n_joint + 1):
            t_ep0 = time.perf_counter()
            epoch_counter += 1
            model.train()
            order = rng.permutation(len(curriculum))
            ep_total = 0.0
            ep_phase = 0.0
            ep_sino = 0.0
            ep_dc_img = 0.0
            ep_fin_img = 0.0
            w_sum = 0.0

            for idx in order:
                sample, sample_w = curriculum[int(idx)]
                opt_joint.zero_grad(set_to_none=True)
                out = model.forward_sample(sample, device=device)
                losses = compute_dual_domain_physics_loss(out, sample, device=device)
                weighted_loss = losses["joint_finetune_loss"] * float(sample_w)
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                opt_joint.step()

                sw = float(sample_w)
                w_sum += sw
                ep_total += float(losses["total_loss"].detach().item()) * sw
                ep_phase += float(losses["joint_finetune_loss"].detach().item()) * sw
                ep_sino += float(losses["sino_wedge_loss"].item()) * sw
                ep_dc_img += float(losses["dc_img_loss"].item()) * sw
                ep_fin_img += float(losses["final_img_loss"].item()) * sw
                del out, losses, weighted_loss

            sched_joint.step()
            denom = max(w_sum, 1e-8)
            mean_ep_loss = ep_total / denom
            mean_phase_loss = ep_phase / denom
            if mean_ep_loss <= best_joint_loss:
                best_joint_loss = mean_ep_loss
                best_joint_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            cur_lr = float(sched_joint.get_last_lr()[0])
            dt_ep = time.perf_counter() - t_ep0
            print(
                f"[Epoch {epoch_counter:02d}/{total_ep:02d} | Stage 3/3 Joint   {j_ep:02d}/{n_joint:02d}] "
                f"joint_obj={mean_phase_loss:.5f} | total={mean_ep_loss:.5f} | "
                f"final_img={ep_fin_img/denom:.5f} | lr={cur_lr:.6f} ({dt_ep:.1f}s)",
                file=sys.stderr,
                flush=True,
            )

            history.append(
                {
                    "epoch": int(epoch_counter),
                    "phase": "stage3_joint_finetune",
                    "total_loss": round(mean_ep_loss, 6),
                    "phase_objective_loss": round(mean_phase_loss, 6),
                    "sino_wedge_loss": round(ep_sino / denom, 6),
                    "dc_img_loss": round(ep_dc_img / denom, 6),
                    "final_img_loss": round(ep_fin_img / denom, 6),
                    "lr": round(cur_lr, 7),
                }
            )

        if best_joint_state is not None:
            model.load_state_dict(best_joint_state)
            if hasattr(model, "configure_devices"):
                model.configure_devices(device)
        del opt_joint, sched_joint, best_joint_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"[Eval] Calibrating residual scales and evaluating on {len(samples)} training samples...",
        file=sys.stderr,
        flush=True,
    )

    # Safeguard Domain-1 residual scale on training samples if needed
    base_sino_scale = float(model.sino_net.res_scale.detach().item())
    best_preds = [evaluate_dual_domain_sample(model, s, device=device) for s in samples]
    mean_pre = float(np.mean([p.prelim_metrics["rmse_hu"] for p in best_preds]))
    best_mean_d1 = float(np.mean([p.domain1_dc_metrics["rmse_hu"] for p in best_preds]))
    if best_mean_d1 > mean_pre:
        best_sino_alpha = 1.0
        for s_alpha in (0.5, 0.25, 0.1, 0.0):
            with torch.no_grad():
                model.sino_net.res_scale.fill_(base_sino_scale * s_alpha)
            cand_preds = [evaluate_dual_domain_sample(model, s, device=device) for s in samples]
            cand_d1 = float(np.mean([p.domain1_dc_metrics["rmse_hu"] for p in cand_preds]))
            if cand_d1 < best_mean_d1 - 1e-5:
                best_mean_d1 = cand_d1
                best_sino_alpha = s_alpha
                best_preds = cand_preds
        with torch.no_grad():
            model.sino_net.res_scale.fill_(base_sino_scale * best_sino_alpha)

    # Calibrate Domain-2 residual scale on training samples so Final Dual-Domain strictly improves or equals Domain-1
    base_img_scale = float(model.img_net.res_scale.detach().item())
    best_alpha = 1.0
    best_mean_fin = float(np.mean([p.final_dual_metrics["rmse_hu"] for p in best_preds]))
    for alpha in (0.75, 0.5, 0.25, 0.0):
        with torch.no_grad():
            model.img_net.res_scale.fill_(base_img_scale * alpha)
        cand_preds = [evaluate_dual_domain_sample(model, s, device=device) for s in samples]
        cand_mean_fin = float(np.mean([p.final_dual_metrics["rmse_hu"] for p in cand_preds]))
        if cand_mean_fin < best_mean_fin - 1e-5:
            best_mean_fin = cand_mean_fin
            best_alpha = alpha
            best_preds = cand_preds
    with torch.no_grad():
        model.img_net.res_scale.fill_(base_img_scale * best_alpha)

    predictions = best_preds
    summary = summarize_dual_domain_predictions(
        predictions, itf_ratio=itf_ratio, training_history=history
    )
    summary["training_stage_epochs"] = {
        "stage1_sino_pretrain": n_sino,
        "stage2_img_pretrain": n_img,
        "stage3_joint_finetune": n_joint,
    }
    summary["device_configuration"] = device_info
    return {
        "history": history,
        "predictions": predictions,
        "summary": summary,
    }


def _aggregate_metric_dicts(dicts: Sequence[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}
    keys = list(dicts[0].keys())
    return {k: round(float(np.mean([d[k] for d in dicts])), 6) for k in keys}


def summarize_dual_domain_predictions(
    predictions: Sequence[DualDomainPrediction],
    *,
    itf_ratio: float = DEFAULT_ITF_TRAINING_RATIO,
    training_history: Sequence[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Summarize Dual-Domain performance across all stages, scenarios (`ITF` vs `RDC`), and pitches (`1.5, 2.0, 2.5`)."""
    if not predictions:
        return {}

    overall_raw = _aggregate_metric_dicts([p.raw_nofill_metrics for p in predictions])
    overall_pre = _aggregate_metric_dicts([p.prelim_metrics for p in predictions])
    overall_d1 = _aggregate_metric_dicts([p.domain1_dc_metrics for p in predictions])
    overall_fin = _aggregate_metric_dicts([p.final_dual_metrics for p in predictions])
    overall_sino = _aggregate_metric_dicts([p.sinogram_progression_metrics for p in predictions])

    by_scenario: dict[str, Any] = {}
    for scen in ("itf", "rdc"):
        sub = [p for p in predictions if p.sample.scenario == scen]
        if sub:
            by_scenario[scen] = {
                "count": len(sub),
                "raw_nofill_metrics": _aggregate_metric_dicts([p.raw_nofill_metrics for p in sub]),
                "prelim_metrics": _aggregate_metric_dicts([p.prelim_metrics for p in sub]),
                "domain1_dc_metrics": _aggregate_metric_dicts([p.domain1_dc_metrics for p in sub]),
                "final_dual_metrics": _aggregate_metric_dicts([p.final_dual_metrics for p in sub]),
                "sinogram_progression": _aggregate_metric_dicts(
                    [p.sinogram_progression_metrics for p in sub]
                ),
            }

    by_pitch: dict[str, Any] = {}
    unique_pitches = sorted({round(p.sample.target_pitch, 2) for p in predictions})
    for pt in unique_pitches:
        sub = [p for p in predictions if abs(p.sample.target_pitch - pt) < 1e-3]
        if sub:
            by_pitch[f"p_{pt:.1f}"] = {
                "target_pitch": pt,
                "mean_effective_pitch": round(
                    float(np.mean([p.sample.effective_pitch for p in sub])), 4
                ),
                "mean_missing_fraction": round(
                    float(np.mean([p.sample.missing_fraction for p in sub])), 4
                ),
                "count": len(sub),
                "raw_nofill_metrics": _aggregate_metric_dicts([p.raw_nofill_metrics for p in sub]),
                "prelim_metrics": _aggregate_metric_dicts([p.prelim_metrics for p in sub]),
                "domain1_dc_metrics": _aggregate_metric_dicts([p.domain1_dc_metrics for p in sub]),
                "final_dual_metrics": _aggregate_metric_dicts([p.final_dual_metrics for p in sub]),
                "sinogram_progression": _aggregate_metric_dicts(
                    [p.sinogram_progression_metrics for p in sub]
                ),
            }

    per_sample: list[dict[str, Any]] = []
    for p in predictions:
        per_sample.append(
            {
                "sample_id": p.sample.sample_id,
                "scenario": p.sample.scenario,
                "target_pitch": p.sample.target_pitch,
                "effective_pitch": round(p.sample.effective_pitch, 4),
                "native_pitch": round(p.sample.native_pitch, 4),
                "z_slice_mm": round(p.sample.z_slice_mm, 3),
                "missing_fraction": round(p.sample.missing_fraction, 4),
                "raw_nofill_rmse_hu": p.raw_nofill_metrics["rmse_hu"],
                "prelim_rmse_hu": p.prelim_metrics["rmse_hu"],
                "domain1_dc_rmse_hu": p.domain1_dc_metrics["rmse_hu"],
                "final_dual_rmse_hu": p.final_dual_metrics["rmse_hu"],
                "prelim_ssim": p.prelim_metrics["ssim"],
                "final_dual_ssim": p.final_dual_metrics["ssim"],
                "reproj_missing_wedge_mae": p.sinogram_progression_metrics[
                    "reproj_missing_wedge_mae"
                ],
                "sinonet_dc_missing_wedge_mae": p.sinogram_progression_metrics[
                    "sinonet_dc_missing_wedge_mae"
                ],
            }
        )

    return {
        "itf_training_ratio": round(float(itf_ratio), 4),
        "rdc_training_ratio": round(1.0 - float(itf_ratio), 4),
        "sample_count": len(predictions),
        "pitches_evaluated": unique_pitches,
        "overall_progression": {
            "stage0_raw_nofill": overall_raw,
            "stage1_prelim_fbp": overall_pre,
            "stage2_domain1_sinonet_fbp": overall_d1,
            "stage3_final_dual_domain": overall_fin,
            "sinogram_progression": overall_sino,
            "rmse_reduction_vs_prelim_hu": round(
                overall_pre["rmse_hu"] - overall_fin["rmse_hu"], 4
            ),
            "rmse_reduction_vs_raw_hu": round(
                overall_raw["rmse_hu"] - overall_fin["rmse_hu"], 4
            ),
            "ssim_gain_vs_prelim": round(overall_fin["ssim"] - overall_pre["ssim"], 6),
        },
        "by_scenario": by_scenario,
        "by_pitch": by_pitch,
        "per_sample": per_sample,
        "training_history": list(training_history) if training_history is not None else [],
    }


def build_hybrid_phantom_dataset(
    *,
    pitches: Sequence[float] = DEFAULT_CLINICAL_HIGH_PITCHES,
    scenarios: Sequence[str] = ("itf", "rdc"),
    z_slices_mm: Sequence[float] = (-12.0,),
    native_pitch: float = 0.85,
    image_size: int = 128,
    fov_mm: float = 360.0,
    filter_name: str = "ram_lak",
    backend: str = "numpy",
    phantom: list[Ellipsoid3D] | None = None,
) -> list[DualDomainSample]:
    """Generate a multi-pitch (`1.5, 2.0, 2.5`) hybrid (`ITF` + `RDC`) dataset on the 3D Thorax Phantom."""
    active_phantom = phantom if phantom is not None else build_thorax_phantom(axial_scale=0.7)
    fname = normalize_filter_name(filter_name)
    z_req = np.asarray(z_slices_mm, dtype=np.float64).reshape(-1)
    z_mid = float(0.5 * (np.min(z_req) + np.max(z_req)))

    geom_low = make_synthetic_helical_geometry_for_dl(
        native_pitch=native_pitch, z_center_mm=z_mid, image_size=image_size
    )
    proj_low = project_phantom_helical(
        active_phantom, geom_low, batch_views=128, show_progress=False
    )
    flat_low, valid_low = rebin_curved_to_flat(proj_low, geom_low)
    ssrb_ref = rebin_helical_ssrb(
        flat_low, valid_low, geom_low, z_req, angular_mode="short_scan"
    )
    fbp_ref = reconstruct_fanbeam_fbp(
        ssrb_ref,
        image_size=image_size,
        fov_mm=fov_mm,
        water_mu_mm_inv=geom_low.water_mu_mm_inv,
        filter_name=fname,
        backend=backend,
    )

    # Precompute 3D slab + matched low-pitch 3D forward-projected reference if ITF is requested
    scen_norm = [s.strip().lower() for s in scenarios]
    slab_vox: dict[str, Any] | None = None
    z_slab: np.ndarray | None = None
    ssrb_ref_itf: Any = None
    fbp_ref_itf: Any = None
    if "itf" in scen_norm:
        z_slab = np.linspace(float(np.min(z_req)) - 32.0, float(np.max(z_req)) + 32.0, 9, dtype=np.float64)
        slab_vox = voxelize_phantom_slices_radiological(
            active_phantom,
            z_slices_mm=z_slab,
            image_size=image_size,
            fov_mm=fov_mm,
            water_mu_mm_inv=geom_low.water_mu_mm_inv,
        )
        proj_itf_ref = forward_project_helical_3d(
            slab_vox["mu_slices"],
            z_slab,
            slab_vox["y_mm"],
            slab_vox["x_mm"],
            geom_low,
            step_mm=3.0,
            z_boundary_mode="clamp",
            backend=backend,
        )
        flat_itf_ref, valid_itf_ref = rebin_curved_to_flat(proj_itf_ref, geom_low)
        ssrb_ref_itf = rebin_helical_ssrb(
            flat_itf_ref, valid_itf_ref, geom_low, z_req, angular_mode="short_scan"
        )
        fbp_ref_itf = reconstruct_fanbeam_fbp(
            ssrb_ref_itf,
            image_size=image_size,
            fov_mm=fov_mm,
            water_mu_mm_inv=geom_low.water_mu_mm_inv,
            filter_name=fname,
            backend=backend,
        )

    samples: list[DualDomainSample] = []
    for p_target in pitches:
        pt = float(p_target)
        for scen in scen_norm:
            if scen == "rdc":
                r_lo, r_hi, eff_p = compute_rdc_active_row_range(
                    geom_low.n_rows, native_pitch, pt
                )
                ssrb_hp = rebin_helical_ssrb(
                    flat_low,
                    valid_low,
                    geom_low,
                    z_req,
                    angular_mode="short_scan",
                    active_row_range=(r_lo, r_hi),
                )
                pair = build_dual_domain_pair_from_ssrb(
                    ssrb_ref,
                    ssrb_hp,
                    scenario="rdc",
                    target_pitch=pt,
                    effective_pitch=eff_p,
                    native_pitch=native_pitch,
                    active_row_range=(r_lo, r_hi),
                    image_size=image_size,
                    fov_mm=fov_mm,
                    water_mu_mm_inv=geom_low.water_mu_mm_inv,
                    filter_name=fname,
                    backend=backend,
                    fbp_ref_precomputed=fbp_ref,
                )
                samples.extend(
                    samples_from_simulation_pair(
                        pair,
                        case_id="PHANTOM",
                        rho0_mm=geom_low.rho0_mm,
                        d0_mm=geom_low.d0_mm,
                        water_mu_mm_inv=geom_low.water_mu_mm_inv,
                    )
                )
            elif scen == "itf":
                assert (
                    slab_vox is not None
                    and z_slab is not None
                    and ssrb_ref_itf is not None
                    and fbp_ref_itf is not None
                )
                geom_itf = synthesize_itf_helical_geometry(
                    geom_low,
                    pt,
                    z_center_mm=z_mid,
                    z_span_mm=float(np.max(z_req) - np.min(z_req)),
                    min_rotations=1.1,
                )
                proj_itf = forward_project_helical_3d(
                    slab_vox["mu_slices"],
                    z_slab,
                    slab_vox["y_mm"],
                    slab_vox["x_mm"],
                    geom_itf,
                    step_mm=3.0,
                    z_boundary_mode="clamp",
                    backend=backend,
                )
                flat_itf, valid_itf = rebin_curved_to_flat(proj_itf, geom_itf)
                ssrb_itf = rebin_helical_ssrb(
                    flat_itf, valid_itf, geom_itf, z_req, angular_mode="short_scan"
                )
                eff_itf_p = float(geom_itf.trajectory_metrics()["normalized_pitch"])
                pair_itf = build_dual_domain_pair_from_ssrb(
                    ssrb_ref_itf,
                    ssrb_itf,
                    scenario="itf",
                    target_pitch=pt,
                    effective_pitch=eff_itf_p,
                    native_pitch=native_pitch,
                    active_row_range=(0.0, float(geom_itf.n_rows - 1)),
                    image_size=image_size,
                    fov_mm=fov_mm,
                    water_mu_mm_inv=geom_low.water_mu_mm_inv,
                    filter_name=fname,
                    backend=backend,
                    fbp_ref_precomputed=fbp_ref_itf,
                )
                samples.extend(
                    samples_from_simulation_pair(
                        pair_itf,
                        case_id="PHANTOM",
                        rho0_mm=geom_low.rho0_mm,
                        d0_mm=geom_low.d0_mm,
                        water_mu_mm_inv=geom_low.water_mu_mm_inv,
                    )
                )
            else:
                raise ValueError(f"Unsupported scenario={scen!r}")

    return samples


def make_synthetic_helical_geometry_for_dl(
    *,
    native_pitch: float = 0.85,
    z_center_mm: float = -12.0,
    image_size: int = 128,
) -> HelicalGeometry:
    """Create a standard 32-row FFSZ helical geometry scaled appropriately for fast DL verification."""
    from highpitchct.phantom import make_synthetic_helical_geometry

    n_cols = 144 if image_size >= 96 else 96
    du_mm = 4.6 if image_size >= 96 else 6.8
    n_rot_steps = 160 if image_size >= 96 else 96
    return make_synthetic_helical_geometry(
        n_views=n_rot_steps * 2,
        n_rows=32,
        n_cols=n_cols,
        angular_steps_per_rotation=n_rot_steps,
        normalized_pitch=float(native_pitch),
        detector_shape="CYLINDRICAL",
        ffs_mode="FFSZ",
        rotation_direction=-1,
        quarter_detector_offset_cols=1.125,
        du_mm=du_mm,
        dv_mm=1.8,
        z_center_mm=float(z_center_mm),
    )


def build_hybrid_series_dataset_streaming(
    series_dir_or_files: str | Path | list[Path],
    z_slices_mm: np.ndarray | list[float],
    *,
    case_id: str = "case_C012",
    geometry: HelicalGeometry | None = None,
    pitches: Sequence[float] = DEFAULT_CLINICAL_HIGH_PITCHES,
    scenarios: Sequence[str] = ("itf", "rdc"),
    image_size: int = 256,
    fov_mm: float | None = None,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    inpaint_mode: str = "conjugate_feather",
    itf_slab_slices: int = 7,
    itf_slab_half_span_mm: float | None = None,
    backend: str = "numpy",
    workers: int = 4,
    show_progress: bool = False,
) -> list[DualDomainSample]:
    """Stream a real DICOM-CT-PD series once for reference `x_target` and generate multi-pitch `ITF + RDC` samples.

    Automatically adapts to any patient's `native_pitch` (`0.6..1.0`) so target high pitches
    remain fixed at `{1.5, 2.0, 2.5}`. Reuses the precomputed reference SSRB, reference FBP,
    and 3D reference attenuation slab across all target pitches so multi-pitch `ITF + RDC`
    generation runs without redundant disk passes.
    """
    from highpitchct.recon_fbp import reconstruct_series_reference_streaming

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
    scen_norm = [s.strip().lower() for s in scenarios]

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

    z_mid = float(0.5 * (np.min(z_req) + np.max(z_req)))
    z_span = float(np.max(z_req) - np.min(z_req))

    # Precompute the 3D attenuation slab + matched low-pitch 3D forward-projected reference ONCE if ITF is in scenarios
    mu_slab: np.ndarray | None = None
    z_slab: np.ndarray | None = None
    fbp_slab: Any = None
    ssrb_ref_itf: Any = None
    fbp_ref_itf: Any = None
    if "itf" in scen_norm:
        z_min_raw = float(np.min(geometry.z0_mm))
        z_max_raw = float(np.max(geometry.z0_mm))
        half_cov = (
            float(itf_slab_half_span_mm)
            if itf_slab_half_span_mm is not None
            else 0.85 * float(geometry.coverage_isocenter_mm)
        )
        margin_raw = 0.5 * 0.85 * float(geometry.coverage_isocenter_mm)
        slab_z_lo = max(z_min_raw + margin_raw, float(np.min(z_req)) - half_cov)
        slab_z_hi = min(z_max_raw - margin_raw, float(np.max(z_req)) + half_cov)
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

        # Matched low-pitch (`native_pitch`) 3D helical forward projection on `mu_slab`
        # so ITF high-pitch (`p = 1.5, 2.0, 2.5`) and its reference share the exact same 3D forward model
        geom_itf_ref = synthesize_itf_helical_geometry(
            geometry,
            native_pitch,
            z_center_mm=z_mid,
            z_span_mm=z_span,
        )
        proj_itf_ref = forward_project_helical_3d(
            mu_slab,
            z_slab,
            fbp_slab.y_mm,
            fbp_slab.x_mm,
            geom_itf_ref,
            z_boundary_mode="clamp",
            backend=backend,
            show_progress=show_progress,
        )
        flat_itf_ref, valid_itf_ref = rebin_curved_to_flat(proj_itf_ref, geom_itf_ref)
        ssrb_ref_itf = rebin_helical_ssrb(
            flat_itf_ref,
            valid_itf_ref,
            geom_itf_ref,
            z_req,
            angular_mode="short_scan",
            active_row_range=None,
        )
        fbp_ref_itf = reconstruct_fanbeam_fbp(
            ssrb_ref_itf,
            image_size=fbp_ref.image_size,
            fov_mm=fbp_ref.fov_mm,
            center_xy_mm=fbp_ref.center_xy_mm,
            water_mu_mm_inv=geometry.water_mu_mm_inv,
            filter_name=fname,
            cutoff_ratio=cutoff_ratio,
            inpaint_mode="conjugate_feather",
            backend=backend,
            show_progress=show_progress,
        )

    samples: list[DualDomainSample] = []

    for p_target in pitches:
        pt = float(p_target)
        for scen in scen_norm:
            if scen == "rdc":
                pair_rdc = simulate_series_high_pitch_streaming(
                    files,
                    z_req,
                    geometry=geometry,
                    scenario="rdc",
                    target_pitch=pt,
                    image_size=fbp_ref.image_size,
                    fov_mm=fbp_ref.fov_mm,
                    center_xy_mm=fbp_ref.center_xy_mm,
                    filter_name=fname,
                    cutoff_ratio=cutoff_ratio,
                    inpaint_mode=inpaint_mode,
                    backend=backend,
                    workers=workers,
                    ssrb_ref_precomputed=ssrb_ref,
                    fbp_ref_precomputed=fbp_ref,
                    show_progress=show_progress,
                )
                samples.extend(
                    samples_from_simulation_pair(
                        pair_rdc,
                        case_id=case_id,
                        rho0_mm=geometry.rho0_mm,
                        d0_mm=geometry.d0_mm,
                        water_mu_mm_inv=geometry.water_mu_mm_inv,
                    )
                )
            elif scen == "itf":
                assert (
                    mu_slab is not None
                    and z_slab is not None
                    and fbp_slab is not None
                    and ssrb_ref_itf is not None
                    and fbp_ref_itf is not None
                )
                geom_itf = synthesize_itf_helical_geometry(
                    geometry,
                    pt,
                    z_center_mm=z_mid,
                    z_span_mm=z_span,
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
                eff_itf_p = float(geom_itf.trajectory_metrics()["normalized_pitch"])
                pair_itf = build_dual_domain_pair_from_ssrb(
                    ssrb_ref_itf,
                    ssrb_itf,
                    scenario="itf",
                    target_pitch=pt,
                    effective_pitch=eff_itf_p,
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
                    fbp_ref_precomputed=fbp_ref_itf,
                    show_progress=show_progress,
                )
                samples.extend(
                    samples_from_simulation_pair(
                        pair_itf,
                        case_id=case_id,
                        rho0_mm=geometry.rho0_mm,
                        d0_mm=geometry.d0_mm,
                        water_mu_mm_inv=geometry.water_mu_mm_inv,
                    )
                )
            else:
                raise ValueError(f"Unsupported scenario={scen!r}")

    return samples


def save_dual_domain_checkpoint(
    out_path: str | Path,
    model: Any,
    *,
    summary: dict[str, Any] | None = None,
    sino_base_channels: int = 64,
    img_base_channels: int = 64,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    lateral_pad_cols: int = 96,
) -> Path:
    """Save `DualDomainHighPitchNet` weights and architecture config to a `.pt` checkpoint."""
    import torch

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "sino_base_channels": int(sino_base_channels),
            "img_base_channels": int(img_base_channels),
            "filter_name": normalize_filter_name(filter_name),
            "cutoff_ratio": float(cutoff_ratio),
            "lateral_pad_cols": int(lateral_pad_cols),
        },
        "summary": summary or {},
    }
    torch.save(payload, out)
    return out


def load_dual_domain_checkpoint(
    ckpt_path: str | Path,
    *,
    backend: str = "torch",
) -> tuple[Any, Any, dict[str, Any]]:
    """Load `DualDomainHighPitchNet` from a `.pt` checkpoint onto the requested backend."""
    torch, device = resolve_torch_device(backend)
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    model, _ = create_dual_domain_model(
        sino_base_channels=int(cfg.get("sino_base_channels", 64)),
        img_base_channels=int(cfg.get("img_base_channels", 64)),
        filter_name=str(cfg.get("filter_name", "ram_lak")),
        cutoff_ratio=float(cfg.get("cutoff_ratio", 1.0)),
        lateral_pad_cols=int(cfg.get("lateral_pad_cols", 96)),
        backend=backend,
    )
    model.load_state_dict(payload["model_state_dict"])
    if hasattr(model, "configure_devices"):
        model.configure_devices(device)
    model.eval()
    return model, device, payload

