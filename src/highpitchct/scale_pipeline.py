"""Workstation-Scale (v1.0) Multi-Patient Full-Volume Sharding & Accelerated Resumable Training Engine.

Designed for high-VRAM / high-RAM research servers (e.g., NVIDIA RTX PRO 6000 96 GB / A100 80 GB
with 128+ GB RAM) while remaining fully compatible with multi-GPU T4 and CPU test environments.

Key Capabilities:
  1. Strict 3-Way Patient Cohort Splitting (`25 Train / 5 Val / 5 Test` default) with automated
     zero-leakage verification (`Train ∩ Val ∩ Test == ∅`).
  2. Full-Patient Interior Axial Slice Extraction (`~238 slices/patient × 6 scenarios/pitches ×
     35 patients ≈ 50,000 samples`) — never cropped to 2 slices/patient.
  3. Single-Pass Resumable Patient Shard Generation (`<data_root>/shards/<split>/<case_id>/`):
     - Reads raw DICOM projections ONCE per patient via parallel `iter_projection_chunks`
     - GPU Curved-to-Flat detector rebinning + parallel multi-slice SSRB + batched GPU Parker FBP
     - Batched multi-view GPU 3D Helical Cone-Beam Forward Projection (`ITF`) + 2D Reprojection
     - Live `stderr` telemetry reporting patient `%`, substep `%` (`DICOM/C2F`, `SSRB/FBP`,
       `RDC`, `Helical3D`), elapsed time, and `ETA`
     - Atomic `<case_id>.tmp -> <case_id>` commit with `shard_complete.json` for instant resume.
  4. 100-Epoch Accelerated Resumable Dual-Domain Training (`train_sharded_dual_domain`):
     - Geometry-homogeneous `ShardAwareShuffledBatchSampler` + `DataLoader` (`pin_memory`, workers)
     - `TF32` + `AMP` (`bfloat16` on Ampere/Ada/Hopper/Blackwell, `float16 + GradScaler` on Turing,
       while `HardDC`, `DifferentiableParkerFBPBridge`, and HU losses stay in strict `float32`)
     - 4-Channel Activated `ImageRefineUNet` (`[x_dc, x_prelim, fov_pitch, diff_streak]`) + Stage-2
       memory-mapped Domain-1 `x_dc_hu` disk cache + radiological flip augmentation
     - Per-Epoch Validation (`Val`) printing `RMSE [HU]`, `MAE [HU]`, `PSNR [dB]`, `SSIM`, `lr`,
       overall completion `%`, and `ETA` after every epoch (`1..100`)
     - Atomic `latest_training_state.pt` + `best_val_dual_domain.pt` after every epoch for exact
       interruption recovery (`--resume`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

from highpitchct.ctpd_reader import iter_projection_chunks, scan_projection_series
from highpitchct.dual_domain_dl import (
    DEFAULT_CLINICAL_HIGH_PITCHES,
    DEFAULT_ITF_TRAINING_RATIO,
    DualDomainSample,
    build_hybrid_phantom_dataset,
    compute_dual_domain_physics_loss_batch,
    create_dual_domain_model,
    resolve_torch_device,
    save_dual_domain_checkpoint,
)
from highpitchct.forward_project import (
    _blend_data_consistent_reprojected_sinogram,
    compute_rdc_active_row_range,
    forward_project_fanbeam_2d,
    forward_project_helical_3d,
    synthesize_itf_helical_geometry,
)
from highpitchct.geometry import HelicalGeometry
from highpitchct.phantom import build_thorax_phantom
from highpitchct.rebinning import SSRBResult, rebin_curved_to_flat, rebin_helical_ssrb
from highpitchct.recon_fbp import (
    compute_batch_reconstruction_metrics_torch,
    default_scanner_fov_mm,
    extrapolate_sinogram,
    normalize_filter_name,
    reconstruct_fanbeam_fbp,
)

SHARD_SCHEMA_VERSION: str = "1.0.0"
REQUIRED_SHARD_FILES: tuple[str, ...] = (
    "shard_complete.json",
    "u_coords_mm.npy",
    "y_mm.npy",
    "x_mm.npy",
    "fov_mask.npy",
    "z_slices_mm.npy",
    "x_target_hu.npy",
    "s_target.npy",
    "sample_z_idx.npy",
    "sample_scen_idx.npy",
    "target_pitch.npy",
    "effective_pitch.npy",
    "missing_fraction.npy",
    "beta_rad.npy",
    "s_hp_raw.npy",
    "s_reproj_dc.npy",
    "mask_hp.npy",
    "x_prelim_hu.npy",
)


def format_duration_hms(seconds: float) -> str:
    """Format non-negative duration in seconds as `HH:MM:SS` (or `MM:SS` under 1 hour)."""
    if not math.isfinite(seconds) or seconds < 0.0:
        return "--:--"
    sec_i = int(round(seconds))
    h, rem = divmod(sec_i, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _normalize_case_token(cid: str) -> str:
    tok = cid.strip()
    if tok.lower().startswith("case_"):
        return f"case_{tok[5:]}"
    return f"case_{tok}"


def _case_anatomy_bucket(case_id: str) -> str:
    bare = case_id[5:] if case_id.lower().startswith("case_") else case_id
    return bare[:1].upper() if bare else "X"


def discover_local_patient_cases(raw_tcia_root: str | Path) -> list[dict[str, Any]]:
    """Discover downloaded patient cases with `projection_full` under `<data_root>/raw/tcia`."""
    root = Path(raw_tcia_root).expanduser().resolve()
    if not root.exists():
        return []

    records: list[dict[str, Any]] = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        proj_parent = cdir / "projection_full"
        if not proj_parent.is_dir():
            continue
        proj_subdirs = sorted(p for p in proj_parent.iterdir() if p.is_dir())
        if not proj_subdirs:
            continue
        img_parent = cdir / "reconstruction_published"
        img_dir: Path | None = None
        if img_parent.is_dir():
            img_subdirs = sorted(p for p in img_parent.iterdir() if p.is_dir())
            if img_subdirs:
                img_dir = img_subdirs[0]
        records.append(
            {
                "case_id": cdir.name,
                "projection_dir": proj_subdirs[0],
                "image_dir": img_dir,
            }
        )
    return records


def plan_three_way_patient_split(
    available_case_ids: Sequence[str],
    *,
    n_train: int = 25,
    n_val: int = 5,
    n_test: int = 5,
    train_case_ids: Sequence[str] | None = None,
    val_case_ids: Sequence[str] | None = None,
    test_case_ids: Sequence[str] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Partition unique patients into strictly disjoint `train / val / test` splits.

    - Uses stratified anatomy prefix (`C*` chest, `L*` abdomen, `N*` head) round-robin ordering
      seeded deterministically by `seed` when explicit lists are not supplied.
    - Selects `(n_train, n_val, n_test)` (default `25 / 5 / 5 = 35` patients) when enough cases
      exist, or scales proportionally while guaranteeing `>=1` patient in each of `train/val/test`
      when `3 <= len(available_case_ids) < n_train + n_val + n_test`.
    """
    unique_avail = sorted(dict.fromkeys(str(c).strip() for c in available_case_ids if str(c).strip()))
    avail_map = {c.lower(): c for c in unique_avail}
    for c in unique_avail:
        bare = c[5:] if c.lower().startswith("case_") else c
        avail_map.setdefault(bare.lower(), c)
        avail_map.setdefault(f"case_{bare.lower()}", c)

    def _resolve_list(req: Sequence[str] | None, label: str) -> list[str] | None:
        if req is None:
            return None
        resolved: list[str] = []
        for item in req:
            tok = str(item).strip()
            if not tok:
                continue
            matched = avail_map.get(tok.lower()) or avail_map.get(_normalize_case_token(tok).lower())
            if matched is None:
                raise ValueError(
                    f"Requested {label} case {tok!r} not found among available cases {unique_avail}"
                )
            if matched not in resolved:
                resolved.append(matched)
        return resolved

    exp_tr = _resolve_list(train_case_ids, "train")
    exp_va = _resolve_list(val_case_ids, "val")
    exp_te = _resolve_list(test_case_ids, "test")

    if exp_tr is not None or exp_va is not None or exp_te is not None:
        used = set(exp_tr or []) | set(exp_va or []) | set(exp_te or [])
        remaining = [c for c in unique_avail if c not in used]
        if exp_te is None:
            n_te_take = min(max(1, n_test), max(1, len(remaining) // 3))
            exp_te = remaining[-n_te_take:]
            remaining = remaining[:-n_te_take]
        if exp_va is None:
            n_va_take = min(max(1, n_val), max(1, len(remaining) // 2))
            exp_va = remaining[-n_va_take:]
            remaining = remaining[:-n_va_take]
        if exp_tr is None:
            exp_tr = remaining[: max(1, n_train)] if len(remaining) >= n_train else remaining
        audit = verify_three_way_zero_leakage(exp_tr, exp_va, exp_te)
        if not audit["overall_zero_leakage_pass"]:
            raise ValueError(f"Invalid explicit 3-way patient split: {audit['notes']}")
        return {
            "train_case_ids": list(exp_tr),
            "val_case_ids": list(exp_va),
            "test_case_ids": list(exp_te),
        }

    if len(unique_avail) < 3:
        raise ValueError(
            f"Strict 3-way patient split (Train/Val/Test) requires at least 3 distinct patients; "
            f"found {len(unique_avail)}: {unique_avail}"
        )

    rng = np.random.default_rng(int(seed))
    buckets: dict[str, list[str]] = defaultdict(list)
    for cid in unique_avail:
        buckets[_case_anatomy_bucket(cid)].append(cid)
    for b_list in buckets.values():
        perm = rng.permutation(len(b_list))
        shuffled = [b_list[int(i)] for i in perm]
        b_list[:] = shuffled

    # Interleave anatomy buckets so Train, Val, and Test all share balanced anatomical regions
    ordered: list[str] = []
    b_keys = sorted(buckets.keys())
    max_len = max(len(v) for v in buckets.values())
    for idx in range(max_len):
        for bk in b_keys:
            if idx < len(buckets[bk]):
                ordered.append(buckets[bk][idx])

    req_total = int(n_train) + int(n_val) + int(n_test)
    if len(ordered) >= req_total:
        sel = ordered[:req_total]
        tr_out = sorted(sel[: int(n_train)])
        va_out = sorted(sel[int(n_train) : int(n_train) + int(n_val)])
        te_out = sorted(sel[int(n_train) + int(n_val) : req_total])
    else:
        n_tot = len(ordered)
        n_te_eff = max(1, int(round(n_tot * (float(n_test) / float(max(req_total, 1))))))
        n_va_eff = max(1, int(round(n_tot * (float(n_val) / float(max(req_total, 1))))))
        while n_tot - (n_te_eff + n_va_eff) < 1:
            if n_te_eff > 1:
                n_te_eff -= 1
            elif n_va_eff > 1:
                n_va_eff -= 1
            else:
                break
        n_tr_eff = n_tot - n_va_eff - n_te_eff
        tr_out = sorted(ordered[:n_tr_eff])
        va_out = sorted(ordered[n_tr_eff : n_tr_eff + n_va_eff])
        te_out = sorted(ordered[n_tr_eff + n_va_eff :])

    verify_three_way_zero_leakage(tr_out, va_out, te_out)
    return {
        "train_case_ids": tr_out,
        "val_case_ids": va_out,
        "test_case_ids": te_out,
    }


def verify_three_way_zero_leakage(
    train_case_ids: Sequence[str],
    val_case_ids: Sequence[str],
    test_case_ids: Sequence[str],
    *,
    train_sample_count: int = 0,
    val_sample_count: int = 0,
    test_sample_count: int = 0,
) -> dict[str, Any]:
    """Audit strict 3-way patient-level zero data leakage (`Train ∩ Val ∩ Test == ∅`)."""
    s_tr = {str(c).strip() for c in train_case_ids if str(c).strip()}
    s_va = {str(c).strip() for c in val_case_ids if str(c).strip()}
    s_te = {str(c).strip() for c in test_case_ids if str(c).strip()}

    ov_tr_va = sorted(s_tr & s_va)
    ov_tr_te = sorted(s_tr & s_te)
    ov_va_te = sorted(s_va & s_te)
    non_empty = bool(s_tr and s_va and s_te)
    passed = bool(non_empty and not ov_tr_va and not ov_tr_te and not ov_va_te)

    return {
        "split_mode": "three_way_patient_split",
        "train_case_ids": sorted(s_tr),
        "val_case_ids": sorted(s_va),
        "test_case_ids": sorted(s_te),
        "train_patient_count": len(s_tr),
        "val_patient_count": len(s_va),
        "test_patient_count": len(s_te),
        "train_val_overlap": ov_tr_va,
        "train_test_overlap": ov_tr_te,
        "val_test_overlap": ov_va_te,
        "total_patient_overlap_count": len(ov_tr_va) + len(ov_tr_te) + len(ov_va_te),
        "train_sample_count": int(train_sample_count),
        "val_sample_count": int(val_sample_count),
        "test_sample_count": int(test_sample_count),
        "total_sample_count": int(train_sample_count + val_sample_count + test_sample_count),
        "overall_zero_leakage_pass": passed,
        "notes": (
            f"Strict 3-way inter-patient split verified: "
            f"{len(s_tr)} Train / {len(s_va)} Val / {len(s_te)} Test patients with 0 overlap."
            if passed
            else f"Zero-leakage check FAILED (non_empty={non_empty}, "
            f"overlaps={ov_tr_va + ov_tr_te + ov_va_te})."
        ),
    }


def compute_patient_full_slice_grid(
    geometry: HelicalGeometry,
    *,
    slice_spacing_mm: float | None = None,
    target_slices_per_patient: int | None = None,
    max_slices_per_patient: int | None = None,
) -> tuple[np.ndarray, float]:
    """Compute the full-volume interior reconstructible axial slice grid `z_req` `[mm]`.

    Extracts the entire interior helical scan span `[z_lo, z_hi]` where short-scan SSRB has
    complete physical support, rather than cropping 2 isolated slices.
    Returns `(z_req_mm, dz_mm)` with strictly uniform spacing `dz_mm` suitable for 3D Helical
    cone-beam forward projection (`ITF`) and multi-slice SSRB (`RDC`).
    """
    z_min_raw = float(np.min(geometry.z0_mm))
    z_max_raw = float(np.max(geometry.z0_mm))
    raw_span = max(0.0, z_max_raw - z_min_raw)
    margin_mm = min(1.15 * float(geometry.coverage_isocenter_mm), 0.16 * raw_span)
    z_lo = z_min_raw + margin_mm
    z_hi = z_max_raw - margin_mm
    usable_span = max(0.0, z_hi - z_lo)

    if usable_span < 1.0:
        z_mid = 0.5 * (z_min_raw + z_max_raw)
        return np.array([z_mid], dtype=np.float64), 1.25

    if target_slices_per_patient is not None and int(target_slices_per_patient) >= 2:
        n_slices = int(target_slices_per_patient)
    elif slice_spacing_mm is not None and float(slice_spacing_mm) > 0.0:
        n_slices = max(2, int(round(usable_span / float(slice_spacing_mm))) + 1)
    else:
        # Default clinical thin-slice spacing (~1.25 mm -> ~235-250 full-volume slices/patient,
        # yielding ~50,000 samples across 35 patients × 6 high-pitch scenarios!)
        default_dz = max(0.8, min(1.25, float(geometry.dv_iso_mm) * 2.0))
        n_slices = max(2, int(round(usable_span / default_dz)) + 1)

    if max_slices_per_patient is not None and int(max_slices_per_patient) >= 1:
        n_slices = min(n_slices, int(max_slices_per_patient))

    if n_slices == 1:
        return np.array([0.5 * (z_lo + z_hi)], dtype=np.float64), max(usable_span, 1.25)

    z_req = np.linspace(z_lo, z_hi, n_slices, dtype=np.float64)
    dz_mm = float(z_req[1] - z_req[0])
    return z_req, dz_mm


def _infer_clinical_fov_from_image_dir(image_dir: Path | None) -> float | None:
    """Fast single-header DICOM read to match clinical CT FOV (`PixelSpacing * Columns`) in <2 ms."""
    if image_dir is None or not image_dir.exists():
        return None
    import pydicom

    for fpath in sorted(image_dir.iterdir()):
        if not fpath.is_file() or fpath.name.startswith("."):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=False)
            ps = getattr(ds, "PixelSpacing", None)
            cols = getattr(ds, "Columns", None)
            if ps is not None and cols is not None and len(ps) >= 2:
                fov = float(cols) * float(ps[1])
                if fov > 50.0:
                    return fov
        except Exception:
            continue
    return None


def is_shard_complete(shard_dir: str | Path) -> bool:
    """Return True iff `shard_dir` contains a valid `shard_complete.json` and all `.npy` arrays."""
    sdir = Path(shard_dir)
    if not sdir.is_dir():
        return False
    for fname in REQUIRED_SHARD_FILES:
        if not (sdir / fname).is_file():
            return False
    try:
        meta = json.loads((sdir / "shard_complete.json").read_text(encoding="utf-8"))
        return bool(
            isinstance(meta, dict)
            and meta.get("schema_version") == SHARD_SCHEMA_VERSION
            and int(meta.get("num_samples", 0)) > 0
        )
    except Exception:
        return False


def _slice_ssrb_result(ssrb: SSRBResult, start: int, stop: int) -> SSRBResult:
    """Slice an `SSRBResult` along the axial slice axis `[start:stop]` without copying geometry."""
    return SSRBResult(
        sinograms=ssrb.sinograms[start:stop],
        measured_mask=ssrb.measured_mask[start:stop],
        z_slices_mm=ssrb.z_slices_mm[start:stop],
        beta_rad=ssrb.beta_rad[start:stop],
        phi_center_rad=ssrb.phi_center_rad[start:stop],
        u_coords_mm=ssrb.u_coords_mm,
        du_mm=ssrb.du_mm,
        rho0_mm=ssrb.rho0_mm,
        d0_mm=ssrb.d0_mm,
        gamma_max_rad=ssrb.gamma_max_rad,
        angular_span_rad=ssrb.angular_span_rad,
        angular_mode=ssrb.angular_mode,
        active_row_range=ssrb.active_row_range,
    )


def _reconstruct_high_pitch_arrays_chunked(
    ssrb_ref: SSRBResult,
    ssrb_hp: SSRBResult,
    *,
    fbp_ref_mu: np.ndarray,
    fov_mm: float,
    image_size: int,
    water_mu_mm_inv: float,
    filter_name: str,
    cutoff_ratio: float,
    inpaint_mode: str,
    backend: str,
    slice_chunk: int = 64,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Compute `(s_hp_raw, mask_hp, s_reproj_dc, x_prelim_hu)` in fast GPU slice chunks."""
    n_slices = int(ssrb_hp.z_slices_mm.size)
    n_beta = int(ssrb_hp.sinograms.shape[1])
    n_u = int(ssrb_hp.sinograms.shape[2])

    s_hp_raw_all = np.where(ssrb_hp.measured_mask, ssrb_hp.sinograms, np.float32(0.0)).astype(
        np.float32
    )
    mask_hp_all = np.asarray(ssrb_hp.measured_mask, dtype=bool)
    s_reproj_dc_all = np.empty((n_slices, n_beta, n_u), dtype=np.float32)
    x_prelim_all = np.empty((n_slices, image_size, image_size), dtype=np.float32)

    same_grid = (
        ssrb_hp.sinograms.shape == ssrb_ref.sinograms.shape
        and np.allclose(ssrb_hp.beta_rad, ssrb_ref.beta_rad, atol=1e-4)
        and np.allclose(ssrb_hp.u_coords_mm, ssrb_ref.u_coords_mm, atol=1e-4)
    )
    active_beta_all = ssrb_ref.beta_rad if same_grid else ssrb_hp.beta_rad
    active_u_coords = ssrb_ref.u_coords_mm if same_grid else ssrb_hp.u_coords_mm
    s_target_override: np.ndarray | None = (
        None if same_grid else np.empty((n_slices, n_beta, n_u), dtype=np.float32)
    )

    for s0 in range(0, n_slices, slice_chunk):
        s1 = min(n_slices, s0 + slice_chunk)
        sub_hp = _slice_ssrb_result(ssrb_hp, s0, s1)
        fbp_pre = reconstruct_fanbeam_fbp(
            sub_hp,
            image_size=image_size,
            fov_mm=fov_mm,
            water_mu_mm_inv=water_mu_mm_inv,
            filter_name=filter_name,
            cutoff_ratio=cutoff_ratio,
            inpaint_mode=inpaint_mode,
            backend=backend,
            show_progress=False,
        )
        x_prelim_all[s0:s1] = fbp_pre.hu_slices
        mu_pre_pos = np.maximum(fbp_pre.mu_slices, np.float32(0.0))
        s_rep = forward_project_fanbeam_2d(
            mu_pre_pos,
            active_beta_all[s0:s1],
            active_u_coords,
            rho0_mm=ssrb_hp.rho0_mm,
            d0_mm=ssrb_hp.d0_mm,
            fov_mm=fov_mm,
            backend=backend,
            show_progress=False,
        )
        s_reproj_dc_all[s0:s1] = _blend_data_consistent_reprojected_sinogram(
            s_hp_raw_all[s0:s1],
            mask_hp_all[s0:s1],
            s_rep,
            s_hp_extrap=fbp_pre.extrapolated_sinograms,
        )
        if s_target_override is not None:
            mu_tgt_pos = np.maximum(fbp_ref_mu[s0:s1], np.float32(0.0))
            s_target_override[s0:s1] = forward_project_fanbeam_2d(
                mu_tgt_pos,
                active_beta_all[s0:s1],
                active_u_coords,
                rho0_mm=ssrb_hp.rho0_mm,
                d0_mm=ssrb_hp.d0_mm,
                fov_mm=fov_mm,
                backend=backend,
                show_progress=False,
            )
        if progress_cb is not None:
            progress_cb(s1, n_slices)

    out: dict[str, np.ndarray] = {
        "s_hp_raw": s_hp_raw_all,
        "mask_hp": mask_hp_all,
        "s_reproj_dc": s_reproj_dc_all,
        "x_prelim_hu": x_prelim_all,
        "beta_rad": active_beta_all.astype(np.float64),
        "u_coords_mm": active_u_coords.astype(np.float64),
    }
    if s_target_override is not None:
        out["s_target_override"] = s_target_override
    return out


def _write_shard_atomic(
    shard_dir: Path,
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> Path:
    """Write all `.npy` arrays + `shard_complete.json` into a `.tmp` dir and atomically replace."""
    shard_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = shard_dir.parent / f".{shard_dir.name}.tmp_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name, arr in arrays.items():
            np.save(tmp_dir / f"{name}.npy", arr)
        (tmp_dir / "shard_complete.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        os.replace(tmp_dir, shard_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return shard_dir


def build_patient_full_volume_shard(
    case_id: str,
    series_dir: str | Path,
    shard_dir: str | Path,
    *,
    split_name: str,
    image_dir: str | Path | None = None,
    pitches: Sequence[float] = DEFAULT_CLINICAL_HIGH_PITCHES,
    scenarios: Sequence[str] = ("itf", "rdc"),
    slice_spacing_mm: float | None = None,
    target_slices_per_patient: int | None = None,
    max_slices_per_patient: int | None = None,
    image_size: int = 256,
    fov_mm: float | None = None,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    inpaint_mode: str = "conjugate_feather",
    backend: str = "torch_cuda",
    workers: int = 8,
    compress_float16_sinograms: bool = False,
    patient_idx: int = 1,
    total_patients: int = 1,
    cohort_t0: float | None = None,
    force_rebuild: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Reconstruct all interior patient slices & build an atomic on-disk Dual-Domain shard.

    Reports live `stderr` percentage progress (`%`) and `ETA` for every substep:
      - `DICOM+Curved->Flat`
      - `Ref SSRB+FBP`
      - `RDC High-Pitch`
      - `ITF Helical3D`
    """
    sdir = Path(shard_dir).expanduser().resolve()
    if not force_rebuild and is_shard_complete(sdir):
        meta = json.loads((sdir / "shard_complete.json").read_text(encoding="utf-8"))
        if show_progress:
            pct_cohort = 100.0 * float(patient_idx) / float(max(total_patients, 1))
            print(
                f"[Cohort {patient_idx:02d}/{total_patients:02d} ({pct_cohort:5.1f}%)] "
                f"{case_id} ({split_name.upper()}): verified existing shard "
                f"({meta['num_samples']} samples, {meta['num_slices']} slices) — resumed.",
                file=sys.stderr,
                flush=True,
            )
        return {**meta, "shard_dir": str(sdir), "resumed_from_disk": True}

    t_pat0 = time.perf_counter()
    t_cohort_start = t_pat0 if cohort_t0 is None else float(cohort_t0)
    fname = normalize_filter_name(filter_name)
    scen_list = [s.strip().lower() for s in scenarios if s.strip()]
    pitch_list = [float(p) for p in pitches]
    sim_backend = "torch_cuda" if backend == "torch_cuda" else ("torch" if backend == "torch" else "numpy")
    if sim_backend in {"torch", "torch_cuda"}:
        resolve_torch_device(sim_backend)

    def _emit_step(substep_pct: float, stage_msg: str) -> None:
        if not show_progress:
            return
        overall_frac = (float(patient_idx - 1) + max(0.0, min(1.0, substep_pct / 100.0))) / float(
            max(total_patients, 1)
        )
        elapsed_all = max(1e-3, time.perf_counter() - t_cohort_start)
        eta_all = (
            elapsed_all * (1.0 - overall_frac) / max(overall_frac, 1e-4)
            if overall_frac > 0.01
            else float("nan")
        )
        print(
            f"[Patient {patient_idx:02d}/{total_patients:02d} | Cohort {overall_frac*100.0:5.1f}% | "
            f"Patient {substep_pct:5.1f}% | ETA {format_duration_hms(eta_all)}] "
            f"{case_id} ({split_name.upper()}): {stage_msg}",
            file=sys.stderr,
            flush=True,
        )

    _emit_step(2.0, "Scanning DICOM-CT-PD v4 projection headers...")
    geom, files = scan_projection_series(
        Path(series_dir), workers=max(1, workers), show_progress=False
    )
    traj = geom.trajectory_metrics()
    native_pitch = float(traj["normalized_pitch"])
    z_req, dz_mm = compute_patient_full_slice_grid(
        geom,
        slice_spacing_mm=slice_spacing_mm,
        target_slices_per_patient=target_slices_per_patient,
        max_slices_per_patient=max_slices_per_patient,
    )
    n_req = int(z_req.size)

    eff_fov_mm = fov_mm
    if eff_fov_mm is None and image_dir is not None:
        eff_fov_mm = _infer_clinical_fov_from_image_dir(Path(image_dir))
    if eff_fov_mm is None:
        gamma_max_est = math.atan2(0.5 * float(geom.n_cols) * float(geom.du_mm), float(geom.d0_mm))
        eff_fov_mm = default_scanner_fov_mm(float(geom.rho0_mm), gamma_max_est)
    eff_fov_mm = float(eff_fov_mm)

    # Build strictly uniform z_slab = [z_guard_lo, z_req, z_guard_hi] so 1 pass serves BOTH RDC & 3D ITF!
    if "itf" in scen_list and n_req >= 1:
        z_min_raw = float(np.min(geom.z0_mm))
        z_max_raw = float(np.max(geom.z0_mm))
        safe_pad = 0.42 * float(geom.coverage_isocenter_mm)
        n_guard = max(4, int(math.ceil(0.85 * float(geom.coverage_isocenter_mm) / max(dz_mm, 0.5))))
        g_lo = z_req[0] - np.arange(n_guard, 0, -1, dtype=np.float64) * dz_mm
        g_lo = g_lo[g_lo >= (z_min_raw + safe_pad)]
        g_hi = z_req[-1] + np.arange(1, n_guard + 1, dtype=np.float64) * dz_mm
        g_hi = g_hi[g_hi <= (z_max_raw - safe_pad)]
        z_slab = np.concatenate([g_lo, z_req, g_hi])
        req_i0 = int(g_lo.size)
    else:
        z_slab = z_req.copy()
        req_i0 = 0
    req_i1 = req_i0 + n_req

    # 1. Single-pass DICOM read + GPU Curved-to-Flat rebinning (2% -> 22%)
    n_views = int(geom.n_views)
    flat_low = np.empty((n_views, geom.n_rows, geom.n_cols), dtype=np.float32)
    valid_low = np.empty((n_views, geom.n_rows, geom.n_cols), dtype=bool)
    chunk_views = 512
    last_c2f_log = 0.0
    for v0, v1, raw_chunk in iter_projection_chunks(
        files, geom, chunk_views=chunk_views, workers=max(1, workers), show_progress=False
    ):
        f_ch, m_ch = rebin_curved_to_flat(
            raw_chunk, geom.slice_views(v0, v1), backend=sim_backend
        )
        flat_low[v0:v1] = f_ch
        valid_low[v0:v1] = m_ch
        frac_v = float(v1) / float(max(n_views, 1))
        if frac_v - last_c2f_log >= 0.24 or v1 == n_views:
            last_c2f_log = frac_v
            _emit_step(
                2.0 + 20.0 * frac_v,
                f"DICOM + GPU Curved->Flat views {v1}/{n_views} ({frac_v*100.0:.0f}%)",
            )

    # 2. Full-volume Reference SSRB + Batched GPU FBP on z_slab (22% -> 38%)
    _emit_step(24.0, f"Reference SSRB across {z_slab.size} axial slices (usable={n_req}, dz={dz_mm:.2f} mm)...")
    ssrb_slab = rebin_helical_ssrb(
        flat_low,
        valid_low,
        geom,
        z_slab,
        angular_mode="short_scan",
        workers=max(1, workers),
    )

    n_slab = int(z_slab.size)
    mu_slab = np.empty((n_slab, image_size, image_size), dtype=np.float32)
    hu_slab = np.empty((n_slab, image_size, image_size), dtype=np.float32)
    y_mm_ref: np.ndarray | None = None
    x_mm_ref: np.ndarray | None = None
    fov_mask_ref: np.ndarray | None = None
    fbp_chunk = 64
    for s0 in range(0, n_slab, fbp_chunk):
        s1 = min(n_slab, s0 + fbp_chunk)
        fbp_part = reconstruct_fanbeam_fbp(
            _slice_ssrb_result(ssrb_slab, s0, s1),
            image_size=image_size,
            fov_mm=eff_fov_mm,
            water_mu_mm_inv=geom.water_mu_mm_inv,
            filter_name=fname,
            cutoff_ratio=cutoff_ratio,
            inpaint_mode="conjugate_feather",
            backend=sim_backend,
            show_progress=False,
        )
        mu_slab[s0:s1] = fbp_part.mu_slices
        hu_slab[s0:s1] = fbp_part.hu_slices
        if y_mm_ref is None:
            y_mm_ref = fbp_part.y_mm.copy()
            x_mm_ref = fbp_part.x_mm.copy()
            fov_mask_ref = fbp_part.fov_mask.copy()
        _emit_step(
            26.0 + 12.0 * (float(s1) / float(max(n_slab, 1))),
            f"Reference Parker+Ram-Lak FBP {s1}/{n_slab} slices ({100.0*s1/max(n_slab,1):.0f}%)",
        )

    assert y_mm_ref is not None and x_mm_ref is not None and fov_mask_ref is not None
    ssrb_ref_rdc = _slice_ssrb_result(ssrb_slab, req_i0, req_i1)
    x_target_rdc = hu_slab[req_i0:req_i1].copy()
    mu_target_rdc = mu_slab[req_i0:req_i1].copy()
    s_target_rdc = extrapolate_sinogram(
        ssrb_ref_rdc.sinograms,
        ssrb_ref_rdc.measured_mask,
        mode="conjugate_feather",
        beta_rad=ssrb_ref_rdc.beta_rad,
        u_coords_mm=ssrb_ref_rdc.u_coords_mm,
        d0_mm=ssrb_ref_rdc.d0_mm,
        workers=max(1, workers),
    )
    del ssrb_slab

    # Containers to stack all scenario × pitch blocks cleanly
    scen_order: list[str] = []
    x_targets_by_scen: list[np.ndarray] = []
    s_targets_by_scen: list[np.ndarray] = []

    sample_z_idx_parts: list[np.ndarray] = []
    sample_scen_idx_parts: list[np.ndarray] = []
    target_pitch_parts: list[np.ndarray] = []
    effective_pitch_parts: list[np.ndarray] = []
    missing_frac_parts: list[np.ndarray] = []
    beta_parts: list[np.ndarray] = []
    s_hp_raw_parts: list[np.ndarray] = []
    s_reproj_dc_parts: list[np.ndarray] = []
    mask_hp_parts: list[np.ndarray] = []
    x_prelim_parts: list[np.ndarray] = []

    sino_dtype = np.float16 if compress_float16_sinograms else np.float32
    u_coords_saved = ssrb_ref_rdc.u_coords_mm.astype(np.float64).copy()
    du_saved = float(abs(u_coords_saved[1] - u_coords_saved[0])) if u_coords_saved.size > 1 else float(ssrb_ref_rdc.du_mm)

    # 3. Scenario A: RDC High-Pitch Simulation while `flat_low` is in memory (38% -> 58%)
    if "rdc" in scen_list:
        rdc_scen_idx = len(scen_order)
        scen_order.append("rdc")
        x_targets_by_scen.append(x_target_rdc)
        s_targets_by_scen.append(s_target_rdc.astype(sino_dtype))

        for p_idx, p_target in enumerate(pitch_list):
            p_base = 38.0 + 20.0 * (float(p_idx) / float(max(len(pitch_list), 1)))
            p_span = 20.0 / float(max(len(pitch_list), 1))
            r_lo, r_hi, p_eff = compute_rdc_active_row_range(geom.n_rows, native_pitch, p_target)
            _emit_step(p_base + 0.15 * p_span, f"RDC p={p_target:.1f} (eff={p_eff:.2f}): multi-slice SSRB...")
            ssrb_hp_rdc = rebin_helical_ssrb(
                flat_low,
                valid_low,
                geom,
                z_req,
                angular_mode="short_scan",
                active_row_range=(r_lo, r_hi),
                workers=max(1, workers),
            )
            rdc_out = _reconstruct_high_pitch_arrays_chunked(
                ssrb_ref_rdc,
                ssrb_hp_rdc,
                fbp_ref_mu=mu_target_rdc,
                fov_mm=eff_fov_mm,
                image_size=image_size,
                water_mu_mm_inv=geom.water_mu_mm_inv,
                filter_name=fname,
                cutoff_ratio=cutoff_ratio,
                inpaint_mode=inpaint_mode,
                backend=sim_backend,
                slice_chunk=64,
                progress_cb=lambda done, tot, _pb=p_base, _ps=p_span, _pt=p_target: _emit_step(
                    _pb + _ps * (0.20 + 0.80 * float(done) / float(max(tot, 1))),
                    f"RDC p={_pt:.1f}: Prelim FBP + 2D Reproj {done}/{tot} slices ({100.0*done/max(tot,1):.0f}%)",
                ),
            )
            miss_f = (1.0 - np.mean(rdc_out["mask_hp"], axis=(1, 2))).astype(np.float32)
            sample_z_idx_parts.append(np.arange(n_req, dtype=np.int32))
            sample_scen_idx_parts.append(np.full(n_req, rdc_scen_idx, dtype=np.int8))
            target_pitch_parts.append(np.full(n_req, p_target, dtype=np.float32))
            effective_pitch_parts.append(np.full(n_req, p_eff, dtype=np.float32))
            missing_frac_parts.append(miss_f)
            beta_parts.append(rdc_out["beta_rad"])
            s_hp_raw_parts.append(rdc_out["s_hp_raw"].astype(sino_dtype))
            s_reproj_dc_parts.append(rdc_out["s_reproj_dc"].astype(sino_dtype))
            mask_hp_parts.append(rdc_out["mask_hp"].astype(np.uint8))
            x_prelim_parts.append(rdc_out["x_prelim_hu"].astype(np.float32))
            del ssrb_hp_rdc, rdc_out

    # Free raw flat projection volume immediately to reclaim ~1.5-2.5 GB CPU RAM before 3D Helical ITF
    del flat_low, valid_low

    # 4. Scenario B: ITF 3D Helical Forward Projection (`Helical3D`) (58% -> 96%)
    if "itf" in scen_list:
        mu_slab_nonneg = np.maximum(mu_slab, np.float32(0.0))
        z_mid = float(0.5 * (z_req[0] + z_req[-1]))
        z_span = float(max(0.0, z_req[-1] - z_req[0]))

        # 4a. Self-consistent low-pitch (`native_pitch`) 3D Helical reference (`58% -> 68%`)
        geom_itf_ref = synthesize_itf_helical_geometry(
            geom, native_pitch, z_center_mm=z_mid, z_span_mm=z_span
        )
        last_ref_h3d = [-1.0]

        def _ref_h3d_cb(done_v: int, tot_v: int) -> None:
            frac = float(done_v) / float(max(tot_v, 1))
            if frac - last_ref_h3d[0] >= 0.24 or done_v == tot_v:
                last_ref_h3d[0] = frac
                _emit_step(
                    58.0 + 7.0 * frac,
                    f"Helical3D Ref (native p={native_pitch:.2f}): {done_v}/{tot_v} views ({frac*100.0:.0f}%)",
                )

        proj_itf_ref = forward_project_helical_3d(
            mu_slab_nonneg,
            z_slab,
            y_mm_ref,
            x_mm_ref,
            geom_itf_ref,
            z_boundary_mode="clamp",
            backend=sim_backend,
            show_progress=False,
            progress_callback=_ref_h3d_cb,
        )
        flat_itf_ref, valid_itf_ref = rebin_curved_to_flat(
            proj_itf_ref, geom_itf_ref, backend=sim_backend
        )
        del proj_itf_ref
        ssrb_ref_itf = rebin_helical_ssrb(
            flat_itf_ref,
            valid_itf_ref,
            geom_itf_ref,
            z_req,
            angular_mode="short_scan",
            workers=max(1, workers),
        )
        del flat_itf_ref, valid_itf_ref

        x_target_itf = np.empty((n_req, image_size, image_size), dtype=np.float32)
        mu_target_itf = np.empty((n_req, image_size, image_size), dtype=np.float32)
        s_target_itf = np.empty_like(ssrb_ref_itf.sinograms, dtype=np.float32)
        for s0 in range(0, n_req, fbp_chunk):
            s1 = min(n_req, s0 + fbp_chunk)
            fbp_itf_part = reconstruct_fanbeam_fbp(
                _slice_ssrb_result(ssrb_ref_itf, s0, s1),
                image_size=image_size,
                fov_mm=eff_fov_mm,
                water_mu_mm_inv=geom.water_mu_mm_inv,
                filter_name=fname,
                cutoff_ratio=cutoff_ratio,
                inpaint_mode="conjugate_feather",
                backend=sim_backend,
                show_progress=False,
            )
            x_target_itf[s0:s1] = fbp_itf_part.hu_slices
            mu_target_itf[s0:s1] = fbp_itf_part.mu_slices
            s_target_itf[s0:s1] = fbp_itf_part.extrapolated_sinograms

        itf_scen_idx = len(scen_order)
        scen_order.append("itf")
        x_targets_by_scen.append(x_target_itf)
        s_targets_by_scen.append(s_target_itf.astype(sino_dtype))

        # 4b. High-Pitch ITF Helical3D (`p in pitches`, `68% -> 96%`)
        for p_idx, p_target in enumerate(pitch_list):
            p_base = 68.0 + 28.0 * (float(p_idx) / float(max(len(pitch_list), 1)))
            p_span = 28.0 / float(max(len(pitch_list), 1))
            geom_itf = synthesize_itf_helical_geometry(
                geom, p_target, z_center_mm=z_mid, z_span_mm=z_span
            )
            eff_itf_p = float(geom_itf.trajectory_metrics()["normalized_pitch"])
            last_hp_h3d = [-1.0]

            def _hp_h3d_cb(
                done_v: int,
                tot_v: int,
                _pb: float = p_base,
                _ps: float = p_span,
                _pt: float = p_target,
            ) -> None:
                frac = float(done_v) / float(max(tot_v, 1))
                if frac - last_hp_h3d[0] >= 0.24 or done_v == tot_v:
                    last_hp_h3d[0] = frac
                    _emit_step(
                        _pb + 0.50 * _ps * frac,
                        f"Helical3D ITF p={_pt:.1f}: {done_v}/{tot_v} views ({frac*100.0:.0f}%)",
                    )

            proj_itf = forward_project_helical_3d(
                mu_slab_nonneg,
                z_slab,
                y_mm_ref,
                x_mm_ref,
                geom_itf,
                z_boundary_mode="clamp",
                backend=sim_backend,
                show_progress=False,
                progress_callback=_hp_h3d_cb,
            )
            flat_itf, valid_itf = rebin_curved_to_flat(proj_itf, geom_itf, backend=sim_backend)
            del proj_itf
            ssrb_hp_itf = rebin_helical_ssrb(
                flat_itf,
                valid_itf,
                geom_itf,
                z_req,
                angular_mode="short_scan",
                workers=max(1, workers),
            )
            del flat_itf, valid_itf

            itf_out = _reconstruct_high_pitch_arrays_chunked(
                ssrb_ref_itf,
                ssrb_hp_itf,
                fbp_ref_mu=mu_target_itf,
                fov_mm=eff_fov_mm,
                image_size=image_size,
                water_mu_mm_inv=geom.water_mu_mm_inv,
                filter_name=fname,
                cutoff_ratio=cutoff_ratio,
                inpaint_mode=inpaint_mode,
                backend=sim_backend,
                slice_chunk=64,
                progress_cb=lambda done, tot, _pb=p_base, _ps=p_span, _pt=p_target: _emit_step(
                    _pb + _ps * (0.52 + 0.48 * float(done) / float(max(tot, 1))),
                    f"ITF p={_pt:.1f}: Prelim FBP + 2D Reproj {done}/{tot} slices ({100.0*done/max(tot,1):.0f}%)",
                ),
            )
            miss_f = (1.0 - np.mean(itf_out["mask_hp"], axis=(1, 2))).astype(np.float32)
            sample_z_idx_parts.append(np.arange(n_req, dtype=np.int32))
            sample_scen_idx_parts.append(np.full(n_req, itf_scen_idx, dtype=np.int8))
            target_pitch_parts.append(np.full(n_req, p_target, dtype=np.float32))
            effective_pitch_parts.append(np.full(n_req, eff_itf_p, dtype=np.float32))
            missing_frac_parts.append(miss_f)
            beta_parts.append(itf_out["beta_rad"])
            s_hp_raw_parts.append(itf_out["s_hp_raw"].astype(sino_dtype))
            s_reproj_dc_parts.append(itf_out["s_reproj_dc"].astype(sino_dtype))
            mask_hp_parts.append(itf_out["mask_hp"].astype(np.uint8))
            x_prelim_parts.append(itf_out["x_prelim_hu"].astype(np.float32))
            del ssrb_hp_itf, itf_out

        del mu_slab_nonneg, ssrb_ref_itf

    del mu_slab, hu_slab

    arrays_to_save = {
        "u_coords_mm": u_coords_saved,
        "y_mm": y_mm_ref.astype(np.float64),
        "x_mm": x_mm_ref.astype(np.float64),
        "fov_mask": fov_mask_ref.astype(bool),
        "z_slices_mm": z_req.astype(np.float64),
        "x_target_hu": np.stack(x_targets_by_scen, axis=0).astype(np.float32),
        "s_target": np.stack(s_targets_by_scen, axis=0).astype(sino_dtype),
        "sample_z_idx": np.concatenate(sample_z_idx_parts, axis=0),
        "sample_scen_idx": np.concatenate(sample_scen_idx_parts, axis=0),
        "target_pitch": np.concatenate(target_pitch_parts, axis=0),
        "effective_pitch": np.concatenate(effective_pitch_parts, axis=0),
        "missing_fraction": np.concatenate(missing_frac_parts, axis=0),
        "beta_rad": np.concatenate(beta_parts, axis=0),
        "s_hp_raw": np.concatenate(s_hp_raw_parts, axis=0),
        "s_reproj_dc": np.concatenate(s_reproj_dc_parts, axis=0),
        "mask_hp": np.concatenate(mask_hp_parts, axis=0),
        "x_prelim_hu": np.concatenate(x_prelim_parts, axis=0),
    }
    num_samples = int(arrays_to_save["sample_z_idx"].shape[0])
    n_beta_saved = int(arrays_to_save["s_hp_raw"].shape[1])
    n_u_saved = int(arrays_to_save["s_hp_raw"].shape[2])

    meta_out: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "case_id": str(case_id),
        "split": str(split_name),
        "num_slices": n_req,
        "num_samples": num_samples,
        "slice_spacing_mm": round(dz_mm, 5),
        "z_range_mm": [round(float(z_req[0]), 4), round(float(z_req[-1]), 4)],
        "scenarios": scen_order,
        "pitches": pitch_list,
        "native_pitch": round(native_pitch, 5),
        "image_size": int(image_size),
        "n_beta": n_beta_saved,
        "n_u": n_u_saved,
        "du_mm": round(du_saved, 6),
        "rho0_mm": round(float(geom.rho0_mm), 4),
        "d0_mm": round(float(geom.d0_mm), 4),
        "fov_mm": round(eff_fov_mm, 4),
        "water_mu_mm_inv": round(float(geom.water_mu_mm_inv), 6),
        "angular_mode": "short_scan",
        "filter_name": fname,
        "cutoff_ratio": round(float(cutoff_ratio), 4),
        "inpaint_mode": str(inpaint_mode),
        "sinogram_dtype": "float16" if compress_float16_sinograms else "float32",
        "build_seconds": round(time.perf_counter() - t_pat0, 2),
    }

    _emit_step(98.0, f"Committing {num_samples} samples atomically to disk...")
    _write_shard_atomic(sdir, metadata=meta_out, arrays=arrays_to_save)
    shard_bytes = sum(f.stat().st_size for f in sdir.iterdir() if f.is_file())
    meta_out["shard_size_mb"] = round(shard_bytes / (1024.0 * 1024.0), 2)
    _emit_step(
        100.0,
        f"Completed shard ({num_samples} samples, {meta_out['shard_size_mb']:.1f} MB) in {meta_out['build_seconds']:.1f}s",
    )
    return {**meta_out, "shard_dir": str(sdir), "resumed_from_disk": False}


def write_samples_to_shard(
    samples: Sequence[DualDomainSample],
    shard_dir: str | Path,
    *,
    split_name: str,
    compress_float16_sinograms: bool = False,
) -> dict[str, Any]:
    """Convert in-memory `DualDomainSample` items into the exact v1.0 memory-mapped shard layout."""
    if not samples:
        raise ValueError("write_samples_to_shard requires at least 1 sample")
    first = samples[0]
    sdir = Path(shard_dir).expanduser().resolve()
    sino_dtype = np.float16 if compress_float16_sinograms else np.float32

    unique_z = sorted({round(float(s.z_slice_mm), 5) for s in samples})
    z_arr = np.asarray(unique_z, dtype=np.float64)
    z_map = {z_val: i for i, z_val in enumerate(unique_z)}

    unique_scen = list(dict.fromkeys(s.scenario.strip().lower() for s in samples))
    scen_map = {sc: i for i, sc in enumerate(unique_scen)}

    h, w = first.x_target_hu.shape
    n_beta, n_u = first.s_hp_raw.shape
    x_targets = np.zeros((len(unique_scen), len(unique_z), h, w), dtype=np.float32)
    s_targets = np.zeros((len(unique_scen), len(unique_z), n_beta, n_u), dtype=sino_dtype)

    z_idx_list: list[int] = []
    sc_idx_list: list[int] = []
    for s in samples:
        zi = z_map[round(float(s.z_slice_mm), 5)]
        si = scen_map[s.scenario.strip().lower()]
        z_idx_list.append(zi)
        sc_idx_list.append(si)
        x_targets[si, zi] = s.x_target_hu.astype(np.float32)
        s_targets[si, zi] = s.s_target.astype(sino_dtype)

    arrays = {
        "u_coords_mm": first.u_coords_mm.astype(np.float64),
        "y_mm": first.y_mm.astype(np.float64),
        "x_mm": first.x_mm.astype(np.float64),
        "fov_mask": first.fov_mask.astype(bool),
        "z_slices_mm": z_arr,
        "x_target_hu": x_targets,
        "s_target": s_targets,
        "sample_z_idx": np.asarray(z_idx_list, dtype=np.int32),
        "sample_scen_idx": np.asarray(sc_idx_list, dtype=np.int8),
        "target_pitch": np.asarray([s.target_pitch for s in samples], dtype=np.float32),
        "effective_pitch": np.asarray([s.effective_pitch for s in samples], dtype=np.float32),
        "missing_fraction": np.asarray([s.missing_fraction for s in samples], dtype=np.float32),
        "beta_rad": np.stack([s.beta_rad.astype(np.float64) for s in samples], axis=0),
        "s_hp_raw": np.stack([s.s_hp_raw.astype(sino_dtype) for s in samples], axis=0),
        "s_reproj_dc": np.stack([s.s_reproj_dc.astype(sino_dtype) for s in samples], axis=0),
        "mask_hp": np.stack([s.mask_hp.astype(np.uint8) for s in samples], axis=0),
        "x_prelim_hu": np.stack([s.x_prelim_hu.astype(np.float32) for s in samples], axis=0),
    }
    pitches = sorted({round(float(s.target_pitch), 3) for s in samples})
    dz = float(z_arr[1] - z_arr[0]) if z_arr.size > 1 else 1.25
    meta = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "case_id": str(first.case_id),
        "split": str(split_name),
        "num_slices": int(z_arr.size),
        "num_samples": len(samples),
        "slice_spacing_mm": round(dz, 5),
        "z_range_mm": [round(float(z_arr[0]), 4), round(float(z_arr[-1]), 4)],
        "scenarios": unique_scen,
        "pitches": pitches,
        "native_pitch": round(float(first.native_pitch), 5),
        "image_size": int(h),
        "n_beta": int(n_beta),
        "n_u": int(n_u),
        "du_mm": round(float(first.du_mm), 6),
        "rho0_mm": round(float(first.rho0_mm), 4),
        "d0_mm": round(float(first.d0_mm), 4),
        "fov_mm": round(float(first.fov_mm), 4),
        "water_mu_mm_inv": round(float(first.water_mu_mm_inv), 6),
        "angular_mode": str(first.angular_mode),
        "filter_name": str(first.filter_name),
        "cutoff_ratio": 1.0,
        "inpaint_mode": "conjugate_feather",
        "sinogram_dtype": "float16" if compress_float16_sinograms else "float32",
        "build_seconds": 0.0,
    }
    _write_shard_atomic(sdir, metadata=meta, arrays=arrays)
    return {**meta, "shard_dir": str(sdir), "resumed_from_disk": False}


def build_synthetic_cohort_shards(
    shards_root: str | Path,
    *,
    n_train_cases: int = 2,
    n_val_cases: int = 1,
    n_test_cases: int = 1,
    slices_per_case: int = 2,
    image_size: int = 64,
    pitches: Sequence[float] = (1.5, 2.0),
    scenarios: Sequence[str] = ("itf", "rdc"),
    backend: str = "numpy",
) -> dict[str, Any]:
    """Generate distinct synthetic patient shards (`Train / Val / Test`) for fast end-to-end verification."""
    root = Path(shards_root).expanduser().resolve()
    sim_backend = "torch_cuda" if backend == "torch_cuda" else "numpy"

    specs: list[tuple[str, str, float, float]] = []
    for i in range(n_train_cases):
        specs.append(("train", f"synth_train_{i+1:02d}", 0.72 + 0.06 * i, 0.82 + 0.04 * i))
    for i in range(n_val_cases):
        specs.append(("val", f"synth_val_{i+1:02d}", 0.78 + 0.05 * i, 0.86))
    for i in range(n_test_cases):
        specs.append(("test", f"synth_test_{i+1:02d}", 0.84 + 0.05 * i, 0.90))

    z_slices = np.linspace(-14.0, -6.0, max(1, int(slices_per_case)), dtype=np.float64).tolist()
    shard_dirs: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for split_name, cid, ax_scale, nat_p in specs:
        sdir = root / split_name / cid
        if not is_shard_complete(sdir):
            ph = build_thorax_phantom(axial_scale=ax_scale)
            raw_samples = build_hybrid_phantom_dataset(
                phantom=ph,
                z_slices_mm=z_slices,
                native_pitch=nat_p,
                pitches=pitches,
                scenarios=scenarios,
                image_size=image_size,
                fov_mm=340.0 * ax_scale / 0.80,
                backend=sim_backend,
            )
            relabelled = [
                DualDomainSample(
                    **{
                        **s.__dict__,
                        "case_id": cid,
                        "sample_id": f"{cid}_{s.scenario}_p{s.target_pitch:.2f}_z{s.z_slice_mm:+.1f}",
                    }
                )
                for s in raw_samples
            ]
            write_samples_to_shard(
                relabelled,
                sdir,
                split_name=split_name,
                compress_float16_sinograms=(split_name == "train"),
            )
        shard_dirs[split_name].append(sdir)

    tr_c = [p.name for p in shard_dirs["train"]]
    va_c = [p.name for p in shard_dirs["val"]]
    te_c = [p.name for p in shard_dirs["test"]]
    audit = verify_three_way_zero_leakage(tr_c, va_c, te_c)
    return {
        "shards_root": str(root),
        "train_shard_dirs": shard_dirs["train"],
        "val_shard_dirs": shard_dirs["val"],
        "test_shard_dirs": shard_dirs["test"],
        "zero_leakage_audit": audit,
    }


def build_cohort_shards_resumable(
    patient_records: Sequence[dict[str, Any]],
    shards_root: str | Path,
    *,
    n_train: int = 25,
    n_val: int = 5,
    n_test: int = 5,
    train_case_ids: Sequence[str] | None = None,
    val_case_ids: Sequence[str] | None = None,
    test_case_ids: Sequence[str] | None = None,
    target_total_samples: int | None = 50000,
    slice_spacing_mm: float | None = None,
    max_slices_per_patient: int | None = None,
    pitches: Sequence[float] = DEFAULT_CLINICAL_HIGH_PITCHES,
    scenarios: Sequence[str] = ("itf", "rdc"),
    image_size: int = 256,
    fov_mm: float | None = None,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    inpaint_mode: str = "conjugate_feather",
    backend: str = "torch_cuda",
    workers: int = 8,
    seed: int = 42,
    force_rebuild: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Plan 3-way split, build/resume all patient shards on disk, and verify zero data leakage."""
    rec_by_id = {str(r["case_id"]): r for r in patient_records}
    split_plan = plan_three_way_patient_split(
        list(rec_by_id.keys()),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        train_case_ids=train_case_ids,
        val_case_ids=val_case_ids,
        test_case_ids=test_case_ids,
        seed=seed,
    )

    ordered_jobs: list[tuple[str, str]] = []
    for sp_key, c_key in (
        ("train", "train_case_ids"),
        ("val", "val_case_ids"),
        ("test", "test_case_ids"),
    ):
        for cid in split_plan[c_key]:
            ordered_jobs.append((sp_key, cid))

    n_patients_total = len(ordered_jobs)
    n_combos = max(1, len([s for s in scenarios if s.strip()]) * len(list(pitches)))
    target_slices_per_pat: int | None = None
    if (
        slice_spacing_mm is None
        and target_total_samples is not None
        and int(target_total_samples) > 0
    ):
        target_slices_per_pat = max(
            2, int(round(float(target_total_samples) / float(n_patients_total * n_combos)))
        )

    root = Path(shards_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cohort_t0 = time.perf_counter()

    shard_dirs: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    shard_summaries: list[dict[str, Any]] = []
    sample_counts = {"train": 0, "val": 0, "test": 0}

    for idx, (sp_name, cid) in enumerate(ordered_jobs, start=1):
        rec = rec_by_id[cid]
        sdir = root / sp_name / cid
        info = build_patient_full_volume_shard(
            cid,
            rec["projection_dir"],
            sdir,
            split_name=sp_name,
            image_dir=rec.get("image_dir"),
            pitches=pitches,
            scenarios=scenarios,
            slice_spacing_mm=slice_spacing_mm,
            target_slices_per_patient=target_slices_per_pat,
            max_slices_per_patient=max_slices_per_patient,
            image_size=image_size,
            fov_mm=fov_mm,
            filter_name=filter_name,
            cutoff_ratio=cutoff_ratio,
            inpaint_mode=inpaint_mode,
            backend=backend,
            workers=workers,
            compress_float16_sinograms=(sp_name == "train"),
            patient_idx=idx,
            total_patients=n_patients_total,
            cohort_t0=cohort_t0,
            force_rebuild=force_rebuild,
            show_progress=show_progress,
        )
        shard_dirs[sp_name].append(sdir)
        shard_summaries.append(info)
        sample_counts[sp_name] += int(info["num_samples"])

    audit = verify_three_way_zero_leakage(
        split_plan["train_case_ids"],
        split_plan["val_case_ids"],
        split_plan["test_case_ids"],
        train_sample_count=sample_counts["train"],
        val_sample_count=sample_counts["val"],
        test_sample_count=sample_counts["test"],
    )
    manifest_out = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "shards_root": str(root),
        "zero_leakage_audit": audit,
        "shard_build_elapsed_seconds": round(time.perf_counter() - cohort_t0, 2),
        "shard_summaries": shard_summaries,
    }
    (root / "cohort_shards_manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )
    return {
        **manifest_out,
        "train_shard_dirs": shard_dirs["train"],
        "val_shard_dirs": shard_dirs["val"],
        "test_shard_dirs": shard_dirs["test"],
    }


class ShardedDualDomainDataset:
    """Memory-mapped multi-patient Dual-Domain dataset (`< 1.5 GB` active RSS on `50,000+` slices).

    Supports two fast modes:
      - `mode='full'`: yields full Dual-Domain sinogram + image tensors (Stage 1, Stage 3, Val, Test)
      - `mode='stage2_cached'`: yields cached Domain-1 `x_dc_hu` + `x_prelim_hu` + `x_target_hu`
        with optional radiological 2D flip augmentation for ultra-fast Stage 2 `ImgNet` training.
    """

    def __init__(
        self,
        shard_dirs: Sequence[str | Path],
        *,
        itf_ratio: float = DEFAULT_ITF_TRAINING_RATIO,
        augment_flips: bool = False,
        seed: int = 42,
    ) -> None:
        self.shard_dirs = [Path(p).expanduser().resolve() for p in shard_dirs]
        if not self.shard_dirs:
            raise ValueError("ShardedDualDomainDataset requires at least one shard directory")
        self.itf_ratio = float(itf_ratio)
        self.augment_flips = bool(augment_flips)
        self.mode: str = "full"
        self._rng_seed = int(seed)

        self.shards: list[dict[str, Any]] = []
        self._shard_offsets: list[int] = [0]
        self.geometry_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        self._stage2_cache_mmaps: list[np.ndarray | None] = []

        # Compute scenario curriculum weights matching `60% ITF + 40% RDC`
        total_itf = 0
        total_rdc = 0
        raw_metas: list[dict[str, Any]] = []
        for sdir in self.shard_dirs:
            if not is_shard_complete(sdir):
                raise FileNotFoundError(f"Incomplete or missing patient shard directory: {sdir}")
            meta = json.loads((sdir / "shard_complete.json").read_text(encoding="utf-8"))
            raw_metas.append(meta)
            sc_idx = np.load(sdir / "sample_scen_idx.npy", mmap_mode="r")
            for idx_s, sc_name in enumerate(meta["scenarios"]):
                cnt = int(np.sum(sc_idx == idx_s))
                if sc_name == "itf":
                    total_itf += cnt
                elif sc_name == "rdc":
                    total_rdc += cnt

        total_n = max(1, total_itf + total_rdc)
        if total_itf > 0 and total_rdc > 0:
            w_itf = float(self.itf_ratio) * float(total_n) / float(total_itf)
            w_rdc = (1.0 - float(self.itf_ratio)) * float(total_n) / float(total_rdc)
        else:
            w_itf = 1.0
            w_rdc = 1.0

        running = 0
        for s_i, (sdir, meta) in enumerate(zip(self.shard_dirs, raw_metas)):
            u_coords = np.load(sdir / "u_coords_mm.npy")
            y_mm = np.load(sdir / "y_mm.npy")
            x_mm = np.load(sdir / "x_mm.npy")
            fov_mask = np.load(sdir / "fov_mask.npy")
            z_slices = np.load(sdir / "z_slices_mm.npy")
            sample_z_idx = np.load(sdir / "sample_z_idx.npy")
            sample_scen_idx = np.load(sdir / "sample_scen_idx.npy")
            target_pitch = np.load(sdir / "target_pitch.npy")
            effective_pitch = np.load(sdir / "effective_pitch.npy")
            missing_fraction = np.load(sdir / "missing_fraction.npy")

            scen_list = list(meta["scenarios"])
            weights = np.ones(sample_z_idx.shape[0], dtype=np.float32)
            for k_sc, sc_name in enumerate(scen_list):
                weights[sample_scen_idx == k_sc] = np.float32(
                    w_itf if sc_name == "itf" else w_rdc
                )

            g_key = (
                int(meta["n_beta"]),
                int(meta["n_u"]),
                round(float(meta["du_mm"]), 6),
                round(float(meta["rho0_mm"]), 4),
                round(float(meta["d0_mm"]), 4),
                int(meta["image_size"]),
                round(float(meta["fov_mm"]), 4),
                round(float(meta["water_mu_mm_inv"]), 6),
                str(meta["angular_mode"]),
            )
            n_here = int(sample_z_idx.shape[0])
            self.geometry_groups[g_key].extend(range(running, running + n_here))
            running += n_here
            self._shard_offsets.append(running)

            self.shards.append(
                {
                    "shard_idx": s_i,
                    "shard_dir": sdir,
                    "meta": meta,
                    "geom_key": g_key,
                    "u_coords_mm": u_coords,
                    "y_mm": y_mm,
                    "x_mm": x_mm,
                    "fov_mask": fov_mask,
                    "z_slices_mm": z_slices,
                    "sample_z_idx": sample_z_idx,
                    "sample_scen_idx": sample_scen_idx,
                    "target_pitch": target_pitch,
                    "effective_pitch": effective_pitch,
                    "missing_fraction": missing_fraction,
                    "sample_weight": weights,
                    # Lazy per-worker mmap handles (opened on first access inside worker process)
                    "_mmaps": None,
                }
            )
            self._stage2_cache_mmaps.append(None)

        self.total_samples = running

    def __len__(self) -> int:
        return self.total_samples

    @property
    def case_ids(self) -> list[str]:
        return [str(s["meta"]["case_id"]) for s in self.shards]

    def attach_stage2_d1_cache(self, cache_paths: Sequence[Path]) -> None:
        """Attach memory-mapped Stage 2 Domain-1 `x_dc_hu` `.npy` files (1 per shard)."""
        if len(cache_paths) != len(self.shards):
            raise ValueError("cache_paths length must match number of shards")
        self._stage2_cache_paths = [Path(p) for p in cache_paths]
        self._stage2_cache_mmaps = [None] * len(self.shards)
        self.mode = "stage2_cached"

    def set_full_mode(self) -> None:
        self.mode = "full"

    def locate_index(self, global_idx: int) -> tuple[int, int]:
        if global_idx < 0 or global_idx >= self.total_samples:
            raise IndexError(f"Sample index {global_idx} out of range [0, {self.total_samples})")
        # Binary search over `_shard_offsets`
        lo, hi = 0, len(self.shards) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._shard_offsets[mid + 1] <= global_idx:
                lo = mid + 1
            elif self._shard_offsets[mid] > global_idx:
                hi = mid - 1
            else:
                return mid, global_idx - self._shard_offsets[mid]
        raise IndexError(global_idx)

    def _ensure_shard_mmaps(self, shard_idx: int) -> dict[str, np.ndarray]:
        sh = self.shards[shard_idx]
        mm = sh["_mmaps"]
        if mm is None:
            sdir = sh["shard_dir"]
            mm = {
                "x_target_hu": np.load(sdir / "x_target_hu.npy", mmap_mode="r"),
                "s_target": np.load(sdir / "s_target.npy", mmap_mode="r"),
                "beta_rad": np.load(sdir / "beta_rad.npy", mmap_mode="r"),
                "s_hp_raw": np.load(sdir / "s_hp_raw.npy", mmap_mode="r"),
                "s_reproj_dc": np.load(sdir / "s_reproj_dc.npy", mmap_mode="r"),
                "mask_hp": np.load(sdir / "mask_hp.npy", mmap_mode="r"),
                "x_prelim_hu": np.load(sdir / "x_prelim_hu.npy", mmap_mode="r"),
            }
            sh["_mmaps"] = mm
        return mm

    def get_dual_domain_sample(self, global_idx: int) -> DualDomainSample:
        """Reconstruct a full `DualDomainSample` dataclass instance for Visual QA or single-slice inspection."""
        s_i, loc = self.locate_index(global_idx)
        sh = self.shards[s_i]
        mm = self._ensure_shard_mmaps(s_i)
        meta = sh["meta"]
        zi = int(sh["sample_z_idx"][loc])
        si = int(sh["sample_scen_idx"][loc])
        scen = str(meta["scenarios"][si])
        z_val = float(sh["z_slices_mm"][zi])
        pt = float(sh["target_pitch"][loc])
        cid = str(meta["case_id"])
        s_rep_dc = np.asarray(mm["s_reproj_dc"][loc], dtype=np.float32).copy()
        x_pre = np.asarray(mm["x_prelim_hu"][loc], dtype=np.float32).copy()
        return DualDomainSample(
            sample_id=f"{cid}_{scen}_p{pt:.2f}_z{z_val:+.1f}",
            case_id=cid,
            scenario=scen,
            target_pitch=pt,
            effective_pitch=float(sh["effective_pitch"][loc]),
            native_pitch=float(meta["native_pitch"]),
            z_slice_mm=z_val,
            fov_mm=float(meta["fov_mm"]),
            water_mu_mm_inv=float(meta["water_mu_mm_inv"]),
            rho0_mm=float(meta["rho0_mm"]),
            d0_mm=float(meta["d0_mm"]),
            du_mm=float(meta["du_mm"]),
            angular_mode=str(meta["angular_mode"]),
            filter_name=str(meta["filter_name"]),
            beta_rad=np.asarray(mm["beta_rad"][loc], dtype=np.float64).copy(),
            u_coords_mm=sh["u_coords_mm"].copy(),
            y_mm=sh["y_mm"].copy(),
            x_mm=sh["x_mm"].copy(),
            fov_mask=sh["fov_mask"].copy(),
            s_target=np.asarray(mm["s_target"][si, zi], dtype=np.float32).copy(),
            s_hp_raw=np.asarray(mm["s_hp_raw"][loc], dtype=np.float32).copy(),
            mask_hp=np.asarray(mm["mask_hp"][loc], dtype=bool).copy(),
            s_reproj=s_rep_dc,
            s_reproj_dc=s_rep_dc,
            x_target_hu=np.asarray(mm["x_target_hu"][si, zi], dtype=np.float32).copy(),
            x_prelim_hu=x_pre,
            x_raw_nofill_hu=x_pre,
        )

    def __getitem__(self, global_idx: int) -> dict[str, Any]:
        s_i, loc = self.locate_index(global_idx)
        sh = self.shards[s_i]
        mm = self._ensure_shard_mmaps(s_i)
        meta = sh["meta"]
        zi = int(sh["sample_z_idx"][loc])
        si = int(sh["sample_scen_idx"][loc])

        x_tgt = np.asarray(mm["x_target_hu"][si, zi], dtype=np.float32).copy()
        x_pre = np.asarray(mm["x_prelim_hu"][loc], dtype=np.float32).copy()

        if self.mode == "stage2_cached":
            c_mm = self._stage2_cache_mmaps[s_i]
            if c_mm is None:
                c_mm = np.load(self._stage2_cache_paths[s_i], mmap_mode="r")
                self._stage2_cache_mmaps[s_i] = c_mm
            x_dc = np.asarray(c_mm[loc], dtype=np.float32).copy()
            if self.augment_flips:
                # Deterministic-friendly fast worker RNG using OS entropy mixed with index
                flip_code = int.from_bytes(os.urandom(1), "little") & 3
                if flip_code & 1:
                    x_dc = np.ascontiguousarray(x_dc[:, ::-1])
                    x_pre = np.ascontiguousarray(x_pre[:, ::-1])
                    x_tgt = np.ascontiguousarray(x_tgt[:, ::-1])
                if flip_code & 2:
                    x_dc = np.ascontiguousarray(x_dc[::-1, :])
                    x_pre = np.ascontiguousarray(x_pre[::-1, :])
                    x_tgt = np.ascontiguousarray(x_tgt[::-1, :])
            return {
                "mode": "stage2_cached",
                "global_idx": int(global_idx),
                "shard_idx": int(s_i),
                "x_dc_hu": x_dc,
                "x_prelim_hu": x_pre,
                "x_target_hu": x_tgt,
                "effective_pitch": np.float32(sh["effective_pitch"][loc]),
                "target_pitch": np.float32(sh["target_pitch"][loc]),
                "missing_fraction": np.float32(sh["missing_fraction"][loc]),
                "sample_weight": np.float32(sh["sample_weight"][loc]),
                "scenario": str(meta["scenarios"][si]),
                "case_id": str(meta["case_id"]),
            }

        return {
            "mode": "full",
            "global_idx": int(global_idx),
            "shard_idx": int(s_i),
            "s_hp_raw": np.asarray(mm["s_hp_raw"][loc], dtype=np.float32).copy(),
            "s_reproj_dc": np.asarray(mm["s_reproj_dc"][loc], dtype=np.float32).copy(),
            "s_target": np.asarray(mm["s_target"][si, zi], dtype=np.float32).copy(),
            "mask_hp": np.asarray(mm["mask_hp"][loc], dtype=bool).copy(),
            "beta_rad": np.asarray(mm["beta_rad"][loc], dtype=np.float64).copy(),
            "x_prelim_hu": x_pre,
            "x_target_hu": x_tgt,
            "effective_pitch": np.float32(sh["effective_pitch"][loc]),
            "target_pitch": np.float32(sh["target_pitch"][loc]),
            "missing_fraction": np.float32(sh["missing_fraction"][loc]),
            "sample_weight": np.float32(sh["sample_weight"][loc]),
            "scenario": str(meta["scenarios"][si]),
            "case_id": str(meta["case_id"]),
        }


class ShardAwareShuffledBatchSampler:
    """Groups mini-batches within identical scanner geometries and shuffles all batches globally."""

    def __init__(
        self,
        dataset: ShardedDualDomainDataset,
        *,
        batch_size: int = 16,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.dataset.geometry_groups.values()
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        batches: list[list[int]] = []
        for g_key in sorted(self.dataset.geometry_groups.keys()):
            idx_arr = np.asarray(self.dataset.geometry_groups[g_key], dtype=np.int64)
            if self.shuffle and idx_arr.size > 1:
                idx_arr = idx_arr[rng.permutation(idx_arr.size)]
            for b0 in range(0, idx_arr.size, self.batch_size):
                batches.append(idx_arr[b0 : b0 + self.batch_size].tolist())
        if self.shuffle and len(batches) > 1:
            order = rng.permutation(len(batches))
            batches = [batches[int(k)] for k in order]
        yield from batches


def make_sharded_collate_fn(dataset: ShardedDualDomainDataset) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a fast `DataLoader` collate function attaching shared geometry metadata per mini-batch."""

    def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        sh0 = dataset.shards[int(items[0]["shard_idx"])]
        meta = sh0["meta"]
        mode = str(items[0]["mode"])
        common = {
            "mode": mode,
            "global_idx": [int(it["global_idx"]) for it in items],
            "case_ids": [str(it["case_id"]) for it in items],
            "scenarios": [str(it["scenario"]) for it in items],
            "x_prelim_hu": torch.from_numpy(np.stack([it["x_prelim_hu"] for it in items], axis=0)),
            "x_target_hu": torch.from_numpy(np.stack([it["x_target_hu"] for it in items], axis=0)),
            "effective_pitch": torch.tensor(
                [float(it["effective_pitch"]) for it in items], dtype=torch.float32
            ),
            "target_pitch": torch.tensor(
                [float(it["target_pitch"]) for it in items], dtype=torch.float32
            ),
            "missing_fraction": torch.tensor(
                [float(it["missing_fraction"]) for it in items], dtype=torch.float32
            ),
            "sample_weight": torch.tensor(
                [float(it["sample_weight"]) for it in items], dtype=torch.float32
            ),
            "fov_mask": torch.from_numpy(sh0["fov_mask"]),
            "u_coords_mm": sh0["u_coords_mm"],
            "du_mm": float(meta["du_mm"]),
            "rho0_mm": float(meta["rho0_mm"]),
            "d0_mm": float(meta["d0_mm"]),
            "image_size": int(meta["image_size"]),
            "fov_mm": float(meta["fov_mm"]),
            "water_mu_mm_inv": float(meta["water_mu_mm_inv"]),
            "angular_mode": str(meta["angular_mode"]),
        }
        if mode == "stage2_cached":
            common["x_dc_hu"] = torch.from_numpy(np.stack([it["x_dc_hu"] for it in items], axis=0))
            return common

        common["s_hp_raw"] = torch.from_numpy(np.stack([it["s_hp_raw"] for it in items], axis=0))
        common["s_reproj_dc"] = torch.from_numpy(
            np.stack([it["s_reproj_dc"] for it in items], axis=0)
        )
        common["s_target"] = torch.from_numpy(np.stack([it["s_target"] for it in items], axis=0))
        common["mask_hp"] = torch.from_numpy(np.stack([it["mask_hp"] for it in items], axis=0))
        common["beta_rad"] = torch.from_numpy(np.stack([it["beta_rad"] for it in items], axis=0))
        return common

    return _collate


def _select_amp_configuration(device: Any, amp_mode: str = "auto") -> tuple[bool, Any, bool]:
    """Return `(use_amp, amp_dtype, use_grad_scaler)` for the active PyTorch device."""
    import torch

    m = amp_mode.strip().lower()
    if device.type != "cuda" or m in {"off", "none", "fp32", "false"}:
        return False, torch.float32, False
    if m in {"bf16", "bfloat16"}:
        return True, torch.bfloat16, False
    if m in {"fp16", "float16"}:
        return True, torch.float16, True
    # Auto: prefer bfloat16 on Ampere+ (compute capability >= 8.0: A100, RTX PRO 6000, H100, L4),
    # fallback to float16 + GradScaler on Turing (T4 = 7.5).
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def compute_sharded_prelim_metrics(
    dataset: ShardedDualDomainDataset,
    *,
    device: Any,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict[str, float]:
    """Compute exact `Prelim FBP` (`RMSE [HU]`, `MAE [HU]`, `PSNR [dB]`, `SSIM`) once on `dataset`."""
    import torch
    from torch.utils.data import DataLoader

    prev_mode = dataset.mode
    dataset.set_full_mode()
    try:
        sampler = ShardAwareShuffledBatchSampler(
            dataset, batch_size=batch_size, shuffle=False, seed=0
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=max(0, int(num_workers)),
            pin_memory=(device.type == "cuda"),
            collate_fn=make_sharded_collate_fn(dataset),
        )
        sum_rmse = 0.0
        sum_mae = 0.0
        sum_psnr = 0.0
        sum_ssim = 0.0
        n_tot = 0
        with torch.no_grad():
            for batch in loader:
                bsz = len(batch["global_idx"])
                x_pre = batch["x_prelim_hu"].to(device=device, dtype=torch.float32, non_blocking=True)
                x_tgt = batch["x_target_hu"].to(device=device, dtype=torch.float32, non_blocking=True)
                fmask = batch["fov_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
                m = compute_batch_reconstruction_metrics_torch(x_pre, x_tgt, fov_mask=fmask)
                sum_rmse += m["rmse_hu"] * bsz
                sum_mae += m["mae_hu"] * bsz
                sum_psnr += m["psnr_db"] * bsz
                sum_ssim += m["ssim"] * bsz
                n_tot += bsz
        denom = float(max(n_tot, 1))
        return {
            "rmse_hu": round(sum_rmse / denom, 4),
            "mae_hu": round(sum_mae / denom, 4),
            "psnr_db": round(sum_psnr / denom, 4),
            "ssim": round(sum_ssim / denom, 6),
        }
    finally:
        dataset.mode = prev_mode


def evaluate_sharded_dataset(
    model: Any,
    dataset: ShardedDualDomainDataset,
    *,
    device: Any,
    batch_size: int = 16,
    num_workers: int = 2,
    amp_mode: str = "auto",
    cached_prelim_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run batched zero-leakage evaluation on `dataset` (`Prelim`, `Domain-1`, `Final Dual-Domain`)."""
    import torch
    from torch.utils.data import DataLoader

    prev_mode = dataset.mode
    dataset.set_full_mode()
    use_amp, amp_dtype, _ = _select_amp_configuration(device, amp_mode)
    prelim_m = (
        cached_prelim_metrics
        if cached_prelim_metrics is not None
        else compute_sharded_prelim_metrics(
            dataset, device=device, batch_size=max(batch_size, 16), num_workers=num_workers
        )
    )

    try:
        model.eval()
        sampler = ShardAwareShuffledBatchSampler(
            dataset, batch_size=batch_size, shuffle=False, seed=0
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=max(0, int(num_workers)),
            pin_memory=(device.type == "cuda"),
            collate_fn=make_sharded_collate_fn(dataset),
        )
        d1_sums = {"rmse_hu": 0.0, "mae_hu": 0.0, "psnr_db": 0.0, "ssim": 0.0}
        fin_sums = {"rmse_hu": 0.0, "mae_hu": 0.0, "psnr_db": 0.0, "ssim": 0.0}
        max_meas_dev = 0.0
        n_tot = 0

        with torch.no_grad():
            for batch in loader:
                bsz = len(batch["global_idx"])
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    out = model.forward_batch_tensors(batch, device=device)

                x_tgt = batch["x_target_hu"].to(device=device, dtype=torch.float32, non_blocking=True)
                fmask = out["fov_mask"].to(device=device, dtype=torch.bool, non_blocking=True)

                m_d1 = compute_batch_reconstruction_metrics_torch(
                    out["x_dc_hu"], x_tgt, fov_mask=fmask
                )
                m_fin = compute_batch_reconstruction_metrics_torch(
                    out["x_final_hu"], x_tgt, fov_mask=fmask
                )
                for k in d1_sums:
                    d1_sums[k] += m_d1[k] * bsz
                    fin_sums[k] += m_fin[k] * bsz

                s_hp = batch["s_hp_raw"].to(device=device, dtype=torch.float32, non_blocking=True)
                m_hp = batch["mask_hp"].to(device=device, dtype=torch.bool, non_blocking=True)
                if torch.any(m_hp):
                    dev_val = float(torch.max(torch.abs(out["s_dc"][m_hp] - s_hp[m_hp])).item())
                    max_meas_dev = max(max_meas_dev, dev_val)
                n_tot += bsz

        denom = float(max(n_tot, 1))
        d1_m = {
            k: round(v / denom, 6 if k == "ssim" else 4) for k, v in d1_sums.items()
        }
        fin_m = {
            k: round(v / denom, 6 if k == "ssim" else 4) for k, v in fin_sums.items()
        }
        return {
            "sample_count": n_tot,
            "prelim_fbp": prelim_m,
            "domain1_sino_dc": d1_m,
            "final_dual_domain": fin_m,
            "rmse_gain_vs_prelim_hu": round(prelim_m["rmse_hu"] - fin_m["rmse_hu"], 4),
            "domain2_vs_domain1_rmse_gain_hu": round(d1_m["rmse_hu"] - fin_m["rmse_hu"], 4),
            "psnr_gain_vs_prelim_db": round(fin_m["psnr_db"] - prelim_m["psnr_db"], 4),
            "ssim_gain_vs_prelim": round(fin_m["ssim"] - prelim_m["ssim"], 6),
            "measured_rays_max_abs_dev": round(max_meas_dev, 8),
        }
    finally:
        dataset.mode = prev_mode


def _build_stage2_d1_disk_cache(
    model: Any,
    train_dataset: ShardedDualDomainDataset,
    cache_dir: Path,
    *,
    device: Any,
    batch_size: int,
    amp_mode: str,
    signature: str,
    show_progress: bool = True,
) -> list[Path]:
    """Compute and atomically cache frozen Domain-1 `x_dc_hu` (`float32`) per training shard on disk."""
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    sig_file = cache_dir / "cache_signature.json"
    expected_paths = [
        cache_dir / f"shard_{s['meta']['case_id']}_x_dc_hu.npy" for s in train_dataset.shards
    ]
    if sig_file.is_file() and all(p.is_file() for p in expected_paths):
        try:
            saved_sig = json.loads(sig_file.read_text(encoding="utf-8")).get("signature")
            if saved_sig == signature:
                if show_progress:
                    print(
                        f"[Stage 2 Cache] Reusing verified disk-backed Domain-1 cache ({len(expected_paths)} shards).",
                        file=sys.stderr,
                        flush=True,
                    )
                return expected_paths
        except Exception:
            pass

    use_amp, amp_dtype, _ = _select_amp_configuration(device, amp_mode)
    model.eval()
    t0 = time.perf_counter()
    done_samples = 0
    total_samples = len(train_dataset)

    with torch.no_grad():
        for sh_i, sh in enumerate(train_dataset.shards):
            mm = train_dataset._ensure_shard_mmaps(sh_i)
            meta = sh["meta"]
            n_sh = int(sh["sample_z_idx"].shape[0])
            h = int(meta["image_size"])
            x_dc_arr = np.empty((n_sh, h, h), dtype=np.float32)

            for b0 in range(0, n_sh, batch_size):
                b1 = min(n_sh, b0 + batch_size)
                sub_batch = {
                    "s_hp_raw": torch.from_numpy(
                        np.asarray(mm["s_hp_raw"][b0:b1], dtype=np.float32).copy()
                    ),
                    "s_reproj_dc": torch.from_numpy(
                        np.asarray(mm["s_reproj_dc"][b0:b1], dtype=np.float32).copy()
                    ),
                    "mask_hp": torch.from_numpy(
                        np.asarray(mm["mask_hp"][b0:b1], dtype=bool).copy()
                    ),
                    "effective_pitch": torch.from_numpy(
                        np.asarray(sh["effective_pitch"][b0:b1], dtype=np.float32).copy()
                    ),
                    "beta_rad": torch.from_numpy(
                        np.asarray(mm["beta_rad"][b0:b1], dtype=np.float64).copy()
                    ),
                    "u_coords_mm": sh["u_coords_mm"],
                    "du_mm": float(meta["du_mm"]),
                    "rho0_mm": float(meta["rho0_mm"]),
                    "d0_mm": float(meta["d0_mm"]),
                    "image_size": h,
                    "fov_mm": float(meta["fov_mm"]),
                    "water_mu_mm_inv": float(meta["water_mu_mm_inv"]),
                    "angular_mode": str(meta["angular_mode"]),
                }
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    d1_out = model.forward_domain1_batch(sub_batch, device=device)
                x_dc_arr[b0:b1] = d1_out["x_dc_hu"].detach().cpu().numpy()
                done_samples += b1 - b0

            tmp_p = expected_paths[sh_i].with_suffix(".tmp.npy")
            np.save(tmp_p, x_dc_arr)
            os.replace(tmp_p, expected_paths[sh_i])
            if show_progress:
                pct = 100.0 * float(done_samples) / float(max(total_samples, 1))
                el = max(1e-3, time.perf_counter() - t0)
                eta = el * (1.0 - pct / 100.0) / max(pct / 100.0, 1e-4)
                print(
                    f"[Stage 2 Cache {pct:5.1f}% | ETA {format_duration_hms(eta)}] "
                    f"Cached Domain-1 x_dc_hu for {meta['case_id']} ({done_samples}/{total_samples} slices)",
                    file=sys.stderr,
                    flush=True,
                )

    sig_file.write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")
    return expected_paths


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def train_sharded_dual_domain(
    train_dataset: ShardedDualDomainDataset,
    val_dataset: ShardedDualDomainDataset | None,
    *,
    checkpoint_dir: str | Path,
    test_dataset: ShardedDualDomainDataset | None = None,
    epochs: int = 100,
    stage_epochs: tuple[int, int, int] | None = None,
    batch_size: int = 16,
    eval_batch_size: int | None = None,
    lr: float = 1.5e-3,
    sino_base_channels: int = 64,
    img_base_channels: int = 64,
    img_in_channels: int = 4,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    backend: str = "torch_cuda",
    amp_mode: str = "auto",
    num_workers: int = 4,
    resume: bool = True,
    stop_after_epoch: int | None = None,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train 100-epoch Dual-Domain network on sharded volumes with per-epoch `Val` & exact `--resume`."""
    torch, device = resolve_torch_device(backend)
    fname = normalize_filter_name(filter_name)
    ckpt_root = Path(checkpoint_dir).expanduser().resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)
    latest_state_pt = ckpt_root / "latest_training_state.pt"
    best_model_pt = ckpt_root / "best_val_dual_domain.pt"
    stage2_cache_dir = ckpt_root / "_stage2_d1_cache"

    # Enable TensorFloat-32 matmul/conv acceleration on Ampere/Ada/Hopper/Blackwell GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    total_ep = max(1, int(epochs))
    if stage_epochs is not None:
        n_sino, n_img, n_joint = (int(stage_epochs[0]), int(stage_epochs[1]), int(stage_epochs[2]))
        total_ep = max(1, n_sino + n_img + n_joint)
    elif total_ep < 3:
        n_sino, n_img, n_joint = 0, 0, total_ep
    else:
        n_sino = max(1, int(round(0.40 * total_ep)))
        n_img = max(1, int(round(0.30 * total_ep)))
        n_joint = max(1, total_ep - n_sino - n_img)
        if n_sino + n_img + n_joint != total_ep:
            n_sino = total_ep - n_img - n_joint

    model, _ = create_dual_domain_model(
        sino_base_channels=sino_base_channels,
        img_base_channels=img_base_channels,
        img_in_channels=img_in_channels,
        filter_name=fname,
        cutoff_ratio=cutoff_ratio,
        backend=backend,
        seed=seed,
    )
    dev_cfg = model.configure_devices(device)
    use_amp, amp_dtype, use_scaler = _select_amp_configuration(device, amp_mode)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    opt_sino = torch.optim.AdamW(model.sino_net.parameters(), lr=float(lr), weight_decay=1e-4)
    sched_sino = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_sino, T_max=max(1, n_sino), eta_min=float(lr) * 0.25
    )
    opt_img = torch.optim.AdamW(model.img_net.parameters(), lr=float(lr), weight_decay=1e-4)
    sched_img = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_img, T_max=max(1, n_img), eta_min=float(lr) * 0.20
    )
    opt_joint = torch.optim.AdamW(
        [
            {"params": model.sino_net.parameters(), "lr": float(lr) * 0.15},
            {"params": model.img_net.parameters(), "lr": float(lr) * 0.30},
        ],
        weight_decay=1e-4,
    )
    sched_joint = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_joint, T_max=max(1, n_joint), eta_min=float(lr) * 0.03
    )

    completed_epoch = 0
    cumulative_elapsed_sec = 0.0
    history: list[dict[str, Any]] = []
    best_val_d1_rmse = float("inf")
    best_sino_state: dict[str, Any] | None = None
    best_val_stage2_rmse = float("inf")
    best_img_state: dict[str, Any] | None = None
    best_val_dual_rmse = float("inf")
    best_val_epoch = 0
    best_model_state: dict[str, Any] | None = None
    stage2_sig = f"ep{n_sino}_seed{seed}_n{len(train_dataset)}"

    if resume and latest_state_pt.is_file():
        saved = torch.load(latest_state_pt, map_location="cpu", weights_only=False)
        if (
            int(saved.get("total_epochs", -1)) == total_ep
            and tuple(saved.get("stage_epochs", ())) == (n_sino, n_img, n_joint)
        ):
            model.load_state_dict(saved["model_state_dict"])
            model.configure_devices(device)
            opt_sino.load_state_dict(saved["opt_sino_state"])
            sched_sino.load_state_dict(saved["sched_sino_state"])
            opt_img.load_state_dict(saved["opt_img_state"])
            sched_img.load_state_dict(saved["sched_img_state"])
            opt_joint.load_state_dict(saved["opt_joint_state"])
            sched_joint.load_state_dict(saved["sched_joint_state"])
            if use_scaler and saved.get("scaler_state") is not None:
                scaler.load_state_dict(saved["scaler_state"])
            completed_epoch = int(saved["completed_epoch"])
            cumulative_elapsed_sec = float(saved.get("cumulative_elapsed_sec", 0.0))
            history = list(saved.get("history", []))
            best_val_d1_rmse = float(saved.get("best_val_d1_rmse", float("inf")))
            best_sino_state = saved.get("best_sino_state")
            best_val_stage2_rmse = float(saved.get("best_val_stage2_rmse", float("inf")))
            best_img_state = saved.get("best_img_state")
            best_val_dual_rmse = float(saved.get("best_val_dual_rmse", float("inf")))
            best_val_epoch = int(saved.get("best_val_epoch", 0))
            best_model_state = saved.get("best_model_state")
            stage2_sig = str(saved.get("stage2_sig", stage2_sig))
            if show_progress:
                print(
                    f"[Resume] Restored training state at Epoch {completed_epoch}/{total_ep} "
                    f"(best Val RMSE={best_val_dual_rmse:.3f} HU @ ep {best_val_epoch}).",
                    file=sys.stderr,
                    flush=True,
                )

    ev_bs = int(eval_batch_size) if eval_batch_size is not None else max(int(batch_size), 16)
    val_prelim_cached: dict[str, float] | None = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_prelim_cached = compute_sharded_prelim_metrics(
            val_dataset, device=device, batch_size=ev_bs, num_workers=min(num_workers, 4)
        )
        if show_progress:
            print(
                f"[Val Baseline ({len(val_dataset)} slices across {len(val_dataset.shards)} Val patients)] "
                f"Prelim FBP: RMSE={val_prelim_cached['rmse_hu']:.2f} HU | "
                f"MAE={val_prelim_cached['mae_hu']:.2f} HU | "
                f"PSNR={val_prelim_cached['psnr_db']:.2f} dB | "
                f"SSIM={val_prelim_cached['ssim']:.4f}",
                file=sys.stderr,
                flush=True,
            )

    from torch.utils.data import DataLoader

    session_t0 = time.perf_counter()
    session_epochs_done = 0

    for ep in range(completed_epoch + 1, total_ep + 1):
        t_ep0 = time.perf_counter()

        # Determine active curriculum stage (1: SinoNet, 2: ImgNet, 3: Joint End-to-End)
        if ep <= n_sino:
            stage_id = 1
            stage_label = f"Stage 1/3 SinoNet {ep:02d}/{n_sino:02d}"
            train_dataset.set_full_mode()
            train_dataset.augment_flips = False
            for p in model.sino_net.parameters():
                p.requires_grad_(True)
            for p in model.img_net.parameters():
                p.requires_grad_(False)
            model.sino_net.train()
            model.img_net.eval()
            active_opt = opt_sino
            active_sched = sched_sino
        elif ep <= n_sino + n_img:
            stage_id = 2
            s2_ep = ep - n_sino
            stage_label = f"Stage 2/3 ImgNet  {s2_ep:02d}/{n_img:02d}"
            if s2_ep == 1 and best_sino_state is not None:
                model.sino_net.load_state_dict(best_sino_state)
                model.configure_devices(device)
                stage2_sig = f"ep{n_sino}_bestd1_{best_val_d1_rmse:.4f}_n{len(train_dataset)}"
            if train_dataset.mode != "stage2_cached":
                c_paths = _build_stage2_d1_disk_cache(
                    model,
                    train_dataset,
                    stage2_cache_dir,
                    device=device,
                    batch_size=ev_bs,
                    amp_mode=amp_mode,
                    signature=stage2_sig,
                    show_progress=show_progress,
                )
                train_dataset.attach_stage2_d1_cache(c_paths)
            train_dataset.augment_flips = True
            for p in model.sino_net.parameters():
                p.requires_grad_(False)
            for p in model.img_net.parameters():
                p.requires_grad_(True)
            model.sino_net.eval()
            model.img_net.train()
            active_opt = opt_img
            active_sched = sched_img
        else:
            stage_id = 3
            s3_ep = ep - n_sino - n_img
            stage_label = f"Stage 3/3 Joint   {s3_ep:02d}/{n_joint:02d}"
            if s3_ep == 1 and best_img_state is not None:
                model.img_net.load_state_dict(best_img_state)
                model.configure_devices(device)
            train_dataset.set_full_mode()
            train_dataset.augment_flips = False
            for p in model.sino_net.parameters():
                p.requires_grad_(True)
            for p in model.img_net.parameters():
                p.requires_grad_(True)
            model.sino_net.train()
            model.img_net.train()
            active_opt = opt_joint
            active_sched = sched_joint

        sampler = ShardAwareShuffledBatchSampler(
            train_dataset, batch_size=batch_size, shuffle=True, seed=seed
        )
        sampler.set_epoch(ep)
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=max(0, int(num_workers)),
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
            collate_fn=make_sharded_collate_fn(train_dataset),
        )

        ep_obj_sum = 0.0
        ep_w_sum = 0.0
        n_batches = max(1, len(sampler))
        log_step_every = max(1, n_batches // 4)

        for b_idx, batch in enumerate(loader, start=1):
            active_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                if stage_id == 1:
                    d1_out = model.forward_domain1_batch(batch, device=device)
                    fwd_out = {**d1_out, "x_final_hu": d1_out["x_dc_hu"]}
                    losses = compute_dual_domain_physics_loss_batch(fwd_out, batch, device=device)
                    obj_loss = losses["sino_pretrain_loss"]
                elif stage_id == 2:
                    img_dev = next(model.img_net.parameters()).device
                    x_dc_b = batch["x_dc_hu"].to(
                        device=img_dev, dtype=torch.float32, non_blocking=True
                    )
                    x_pre_b = batch["x_prelim_hu"].to(
                        device=img_dev, dtype=torch.float32, non_blocking=True
                    )
                    fov_b = batch["fov_mask"].to(
                        device=img_dev, dtype=torch.bool, non_blocking=True
                    )
                    eff_p_b = batch["effective_pitch"].to(
                        device=img_dev, dtype=torch.float32, non_blocking=True
                    )
                    miss_b = batch["missing_fraction"].to(
                        device=img_dev, dtype=torch.float32, non_blocking=True
                    )
                    x_fin_b = model.img_net(
                        x_dc_b, x_pre_b, fov_b, eff_p_b, missing_fraction=miss_b
                    )
                    fwd_out = {
                        "x_dc_hu": x_dc_b.to(device, non_blocking=True),
                        "x_final_hu": x_fin_b.to(device, non_blocking=True),
                        "fov_mask": fov_b.to(device, non_blocking=True),
                    }
                    losses = compute_dual_domain_physics_loss_batch(fwd_out, batch, device=device)
                    obj_loss = losses["img_pretrain_loss"]
                else:
                    fwd_out = model.forward_batch_tensors(batch, device=device)
                    losses = compute_dual_domain_physics_loss_batch(fwd_out, batch, device=device)
                    obj_loss = losses["joint_finetune_loss"]

            if use_scaler:
                scaler.scale(obj_loss).backward()
                scaler.unscale_(active_opt)
                if stage_id in (1, 3):
                    torch.nn.utils.clip_grad_norm_(model.sino_net.parameters(), max_norm=2.0)
                if stage_id in (2, 3):
                    torch.nn.utils.clip_grad_norm_(model.img_net.parameters(), max_norm=2.0)
                scaler.step(active_opt)
                scaler.update()
            else:
                obj_loss.backward()
                if stage_id in (1, 3):
                    torch.nn.utils.clip_grad_norm_(model.sino_net.parameters(), max_norm=2.0)
                if stage_id in (2, 3):
                    torch.nn.utils.clip_grad_norm_(model.img_net.parameters(), max_norm=2.0)
                active_opt.step()

            bsz_w = float(torch.sum(batch["sample_weight"]).item())
            ep_obj_sum += float(obj_loss.detach().item()) * bsz_w
            ep_w_sum += bsz_w

            if show_progress and n_batches >= 12 and (b_idx % log_step_every == 0) and b_idx < n_batches:
                in_ep_pct = 100.0 * float(b_idx) / float(n_batches)
                print(
                    f"  -> [Epoch {ep:03d}/{total_ep:03d} batch {b_idx}/{n_batches} ({in_ep_pct:4.0f}%)] "
                    f"running_loss={ep_obj_sum/max(ep_w_sum,1e-8):.5f}",
                    file=sys.stderr,
                    flush=True,
                )

        active_sched.step()
        cur_lr = float(active_sched.get_last_lr()[0])
        train_loss_val = ep_obj_sum / max(ep_w_sum, 1e-8)

        # Per-epoch Validation (`Val`) across all 4 metrics: RMSE [HU], MAE [HU], PSNR [dB], SSIM
        val_eval: dict[str, Any] | None = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_eval = evaluate_sharded_dataset(
                model,
                val_dataset,
                device=device,
                batch_size=ev_bs,
                num_workers=min(num_workers, 4),
                amp_mode=amp_mode,
                cached_prelim_metrics=val_prelim_cached,
            )
            val_d1_rmse = float(val_eval["domain1_sino_dc"]["rmse_hu"])
            val_dual_rmse = float(val_eval["final_dual_domain"]["rmse_hu"])
        else:
            val_d1_rmse = train_loss_val
            val_dual_rmse = train_loss_val

        if stage_id == 1 and val_d1_rmse <= best_val_d1_rmse:
            best_val_d1_rmse = val_d1_rmse
            best_sino_state = {
                k: v.detach().cpu().clone() for k, v in model.sino_net.state_dict().items()
            }
        if stage_id == 2 and val_dual_rmse <= best_val_stage2_rmse:
            best_val_stage2_rmse = val_dual_rmse
            best_img_state = {
                k: v.detach().cpu().clone() for k, v in model.img_net.state_dict().items()
            }

        is_new_best = False
        if val_dual_rmse <= best_val_dual_rmse or best_model_state is None:
            best_val_dual_rmse = val_dual_rmse
            best_val_epoch = ep
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            is_new_best = True
            save_dual_domain_checkpoint(
                best_model_pt,
                model,
                summary={
                    "best_epoch": ep,
                    "stage_id": stage_id,
                    "val_metrics": val_eval,
                },
                sino_base_channels=sino_base_channels,
                img_base_channels=img_base_channels,
                filter_name=fname,
                cutoff_ratio=cutoff_ratio,
            )

        dt_ep = time.perf_counter() - t_ep0
        cumulative_elapsed_sec += dt_ep
        session_epochs_done += 1
        completed_epoch = ep

        avg_ep_sec = (time.perf_counter() - session_t0) / float(max(session_epochs_done, 1))
        rem_sec = avg_ep_sec * float(max(0, total_ep - ep))
        pct_total = 100.0 * float(ep) / float(total_ep)

        ep_record: dict[str, Any] = {
            "epoch": ep,
            "stage_id": stage_id,
            "stage_label": stage_label,
            "progress_pct": round(pct_total, 2),
            "train_objective_loss": round(train_loss_val, 6),
            "lr": round(cur_lr, 7),
            "epoch_seconds": round(dt_ep, 2),
            "eta_seconds": round(rem_sec, 1),
            "val": val_eval,
            "is_best_val": is_new_best,
        }
        history.append(ep_record)

        # Save atomic resumable training state after EVERY epoch
        _atomic_torch_save(
            latest_state_pt,
            {
                "completed_epoch": completed_epoch,
                "total_epochs": total_ep,
                "stage_epochs": (n_sino, n_img, n_joint),
                "model_state_dict": model.state_dict(),
                "opt_sino_state": opt_sino.state_dict(),
                "sched_sino_state": sched_sino.state_dict(),
                "opt_img_state": opt_img.state_dict(),
                "sched_img_state": sched_img.state_dict(),
                "opt_joint_state": opt_joint.state_dict(),
                "sched_joint_state": sched_joint.state_dict(),
                "scaler_state": scaler.state_dict() if use_scaler else None,
                "best_val_d1_rmse": best_val_d1_rmse,
                "best_sino_state": best_sino_state,
                "best_val_stage2_rmse": best_val_stage2_rmse,
                "best_img_state": best_img_state,
                "best_val_dual_rmse": best_val_dual_rmse,
                "best_val_epoch": best_val_epoch,
                "best_model_state": best_model_state,
                "stage2_sig": stage2_sig,
                "cumulative_elapsed_sec": cumulative_elapsed_sec,
                "history": history,
            },
        )

        if show_progress:
            if val_eval is not None:
                vp = val_eval["prelim_fbp"]
                vd1 = val_eval["domain1_sino_dc"]
                vf = val_eval["final_dual_domain"]
                star = " ★BEST" if is_new_best else ""
                print(
                    f"[Epoch {ep:03d}/{total_ep:03d} ({pct_total:5.1f}%) | {stage_label} | "
                    f"{dt_ep:.1f}s | ETA {format_duration_hms(rem_sec)}] "
                    f"loss={train_loss_val:.5f} lr={cur_lr:.6f} || "
                    f"VAL RMSE: {vp['rmse_hu']:.2f}->{vd1['rmse_hu']:.2f}->{vf['rmse_hu']:.2f} HU | "
                    f"MAE: {vf['mae_hu']:.2f} HU | PSNR: {vf['psnr_db']:.2f} dB | "
                    f"SSIM: {vp['ssim']:.4f}->{vf['ssim']:.4f}{star}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[Epoch {ep:03d}/{total_ep:03d} ({pct_total:5.1f}%) | {stage_label} | "
                    f"{dt_ep:.1f}s | ETA {format_duration_hms(rem_sec)}] "
                    f"loss={train_loss_val:.5f} lr={cur_lr:.6f}",
                    file=sys.stderr,
                    flush=True,
                )

        if stop_after_epoch is not None and ep >= int(stop_after_epoch):
            break

    # Restore best validation model weights before final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.configure_devices(device)

    final_val_metrics = (
        evaluate_sharded_dataset(
            model,
            val_dataset,
            device=device,
            batch_size=ev_bs,
            num_workers=min(num_workers, 4),
            amp_mode=amp_mode,
            cached_prelim_metrics=val_prelim_cached,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    final_test_metrics = (
        evaluate_sharded_dataset(
            model,
            test_dataset,
            device=device,
            batch_size=ev_bs,
            num_workers=min(num_workers, 4),
            amp_mode=amp_mode,
        )
        if test_dataset is not None and len(test_dataset) > 0
        else None
    )

    save_dual_domain_checkpoint(
        best_model_pt,
        model,
        summary={
            "best_val_epoch": best_val_epoch,
            "val_metrics": final_val_metrics,
            "test_metrics": final_test_metrics,
        },
        sino_base_channels=sino_base_channels,
        img_base_channels=img_base_channels,
        filter_name=fname,
        cutoff_ratio=cutoff_ratio,
    )

    return {
        "model": model,
        "device": device,
        "device_configuration": dev_cfg,
        "amp_enabled": use_amp,
        "amp_dtype": str(amp_dtype),
        "epochs_completed": completed_epoch,
        "stage_epochs": {"sino_pretrain": n_sino, "img_pretrain": n_img, "joint_finetune": n_joint},
        "best_val_epoch": best_val_epoch,
        "cumulative_training_seconds": round(cumulative_elapsed_sec, 2),
        "best_checkpoint_pt": str(best_model_pt),
        "latest_resume_state_pt": str(latest_state_pt),
        "val_metrics": final_val_metrics,
        "unseen_test_metrics": final_test_metrics,
        "history": history,
    }
