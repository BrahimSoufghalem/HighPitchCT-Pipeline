"""Stage 7: Comprehensive 6-Method Thesis Ablation Study & Quantitative Evaluation Matrix.

Evaluates and compares six high-pitch helical CT reconstruction paradigms on strictly held-out
unseen `Test` patients across scenarios (`ITF`, `RDC`) and clinical target pitches (`1.5, 2.0, 2.5`):

  - `M0_raw_nofill_fbp`: High-Pitch SSRB (Zero-Filled Wedge) + Parker + Ram-Lak FBP
  - `M1_angular_feather_fbp`: High-Pitch SSRB + Angular Feather Extrapolation + Parker + Ram-Lak FBP
  - `M2_physics_conjugate_prelim_fbp`: Physics Conjugate-Ray + Feather Preliminary FBP (`x_prelim`)
  - `M3_image_only_dl`: Pure Image-Domain U-Net (`Prelim FBP -> ImageRefineUNet`, no `SinoNet`)
  - `M4_sino_only_dl`: Pure Sinogram-Domain DL (`SinoInpaintUNet + Hard DC + Parker Ram-Lak FBP`)
  - `M5_full_dual_domain_dl`: Proposed Full Physics-Conditioned Dual-Domain Network
    (`SinoNet + Hard DC + Differentiable Parker Ram-Lak FBP + 4-Ch Activated ImgNet`)

Exports publication-ready JSON reports, CSV thesis tables, an 8-panel visual comparison + HU line
profile PNG, and pitch degradation curves (`RMSE [HU]` & `SSIM` vs pitch).
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from highpitchct.dual_domain_dl import (
    _make_dual_domain_modules,
    compute_dual_domain_physics_loss_batch,
    load_dual_domain_checkpoint,
    resolve_torch_device,
)
from highpitchct.rebinning import SSRBResult
from highpitchct.recon_fbp import (
    compute_batch_reconstruction_metrics_torch,
    compute_reconstruction_metrics,
    reconstruct_fanbeam_fbp,
)
from highpitchct.scale_pipeline import (
    ShardAwareShuffledBatchSampler,
    ShardedDualDomainDataset,
    _atomic_torch_save,
    _select_amp_configuration,
    format_duration_hms,
    make_sharded_collate_fn,
)
from highpitchct.visual_qa import (
    save_stage7_ablation_visual_qa,
    save_stage7_pitch_degradation_curves,
)

ABLATION_METHOD_ORDER: tuple[tuple[str, str], ...] = (
    ("M0_raw_nofill_fbp", "M0: Raw No-Fill FBP"),
    ("M1_angular_feather_fbp", "M1: Angular Feather FBP"),
    ("M2_physics_conjugate_prelim_fbp", "M2: Physics Conjugate-Ray Prelim FBP"),
    ("M3_image_only_dl", "M3: Image-Only U-Net (Prelim -> DL)"),
    ("M4_sino_only_dl", "M4: Sino-Only DL (SinoNet + Hard DC + FBP)"),
    ("M5_full_dual_domain_dl", "M5: Full Dual-Domain DL (Proposed)"),
)


def train_or_load_m3_image_only_baseline(
    train_dataset: ShardedDualDomainDataset | None,
    val_dataset: ShardedDualDomainDataset | None,
    *,
    checkpoint_path: str | Path,
    img_base_channels: int = 64,
    m3_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1.5e-3,
    backend: str = "torch_cuda",
    amp_mode: str = "auto",
    num_workers: int = 4,
    resume: bool = True,
    seed: int = 42,
    show_progress: bool = True,
) -> Any | None:
    """Train or load a dedicated standalone `M3: Image-Only U-Net` (`x_prelim -> x_target`)."""
    torch, device = resolve_torch_device(backend)
    ckpt_pt = Path(checkpoint_path).expanduser().resolve()
    _, _, _, ImageRefineUNet, _ = _make_dual_domain_modules()

    if resume and ckpt_pt.is_file():
        saved = torch.load(ckpt_pt, map_location="cpu", weights_only=False)
        if bool(saved.get("completed", False)):
            m3_net = ImageRefineUNet(
                in_channels=3, base_channels=int(saved.get("img_base_channels", img_base_channels))
            ).to(device)
            m3_net.load_state_dict(saved["state_dict"])
            m3_net.eval()
            if show_progress:
                print(
                    f"[Stage 7 M3] Loaded cached standalone Image-Only U-Net -> {ckpt_pt}",
                    file=sys.stderr,
                    flush=True,
                )
            return m3_net

    if train_dataset is None or len(train_dataset) == 0 or int(m3_epochs) <= 0:
        return None

    from torch.utils.data import DataLoader

    torch.manual_seed(int(seed) + 7)
    m3_net = ImageRefineUNet(in_channels=3, base_channels=int(img_base_channels)).to(device)
    use_amp, amp_dtype, use_scaler = _select_amp_configuration(device, amp_mode)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    opt = torch.optim.AdamW(m3_net.parameters(), lr=float(lr), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, int(m3_epochs)), eta_min=float(lr) * 0.15
    )

    prev_mode = train_dataset.mode
    prev_aug = train_dataset.augment_flips
    train_dataset.set_full_mode()
    train_dataset.augment_flips = False

    best_rmse = float("inf")
    best_state: dict[str, Any] | None = None
    t0 = time.perf_counter()

    try:
        for ep in range(1, int(m3_epochs) + 1):
            m3_net.train()
            sampler = ShardAwareShuffledBatchSampler(
                train_dataset, batch_size=batch_size, shuffle=True, seed=seed + 300
            )
            sampler.set_epoch(ep)
            loader = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                num_workers=max(0, int(num_workers)),
                pin_memory=(device.type == "cuda"),
                collate_fn=make_sharded_collate_fn(train_dataset),
            )
            loss_sum = 0.0
            w_sum = 0.0
            for batch in loader:
                opt.zero_grad(set_to_none=True)
                x_pre = batch["x_prelim_hu"].to(device=device, dtype=torch.float32, non_blocking=True)
                fov_b = batch["fov_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
                eff_p = batch["effective_pitch"].to(device=device, dtype=torch.float32, non_blocking=True)
                miss_f = batch["missing_fraction"].to(device=device, dtype=torch.float32, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    x_pred = m3_net(x_pre, x_pre, fov_b, eff_p, missing_fraction=miss_f)
                    losses = compute_dual_domain_physics_loss_batch(
                        {"x_dc_hu": x_pre, "x_final_hu": x_pred, "fov_mask": fov_b},
                        batch,
                        device=device,
                    )
                    obj = losses["img_pretrain_loss"]
                if use_scaler:
                    scaler.scale(obj).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(m3_net.parameters(), max_norm=2.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    obj.backward()
                    torch.nn.utils.clip_grad_norm_(m3_net.parameters(), max_norm=2.0)
                    opt.step()
                bw = float(torch.sum(batch["sample_weight"]).item())
                loss_sum += float(obj.detach().item()) * bw
                w_sum += bw
            sched.step()

            # Validate on Val dataset if available
            eval_metric = loss_sum / max(w_sum, 1e-8)
            if val_dataset is not None and len(val_dataset) > 0:
                m3_net.eval()
                v_sampler = ShardAwareShuffledBatchSampler(
                    val_dataset, batch_size=max(batch_size, 16), shuffle=False, seed=0
                )
                v_loader = DataLoader(
                    val_dataset,
                    batch_sampler=v_sampler,
                    num_workers=max(0, min(int(num_workers), 4)),
                    pin_memory=(device.type == "cuda"),
                    collate_fn=make_sharded_collate_fn(val_dataset),
                )
                r_sum = 0.0
                v_cnt = 0
                with torch.no_grad():
                    for vb in v_loader:
                        bsz = len(vb["global_idx"])
                        xp = vb["x_prelim_hu"].to(device=device, dtype=torch.float32, non_blocking=True)
                        xt = vb["x_target_hu"].to(device=device, dtype=torch.float32, non_blocking=True)
                        fm = vb["fov_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
                        ep_v = vb["effective_pitch"].to(device=device, dtype=torch.float32, non_blocking=True)
                        mf_v = vb["missing_fraction"].to(device=device, dtype=torch.float32, non_blocking=True)
                        with torch.amp.autocast(
                            device_type=device.type, dtype=amp_dtype, enabled=use_amp
                        ):
                            yp = m3_net(xp, xp, fm, ep_v, missing_fraction=mf_v)
                        mm = compute_batch_reconstruction_metrics_torch(yp, xt, fov_mask=fm)
                        r_sum += mm["rmse_hu"] * bsz
                        v_cnt += bsz
                eval_metric = r_sum / float(max(v_cnt, 1))

            if eval_metric <= best_rmse or best_state is None:
                best_rmse = eval_metric
                best_state = {k: v.detach().cpu().clone() for k, v in m3_net.state_dict().items()}

            if show_progress:
                pct = 100.0 * float(ep) / float(m3_epochs)
                el = time.perf_counter() - t0
                eta = el * float(m3_epochs - ep) / float(max(ep, 1))
                print(
                    f"[Stage 7 M3 Image-Only {ep:02d}/{m3_epochs:02d} ({pct:5.1f}%) | "
                    f"ETA {format_duration_hms(eta)}] loss={loss_sum/max(w_sum,1e-8):.5f} "
                    f"val_score={eval_metric:.3f}",
                    file=sys.stderr,
                    flush=True,
                )

        if best_state is not None:
            m3_net.load_state_dict(best_state)
        m3_net.eval()
        _atomic_torch_save(
            ckpt_pt,
            {
                "completed": True,
                "img_base_channels": int(img_base_channels),
                "m3_epochs": int(m3_epochs),
                "best_score": best_rmse,
                "state_dict": m3_net.state_dict(),
            },
        )
        return m3_net
    finally:
        train_dataset.mode = prev_mode
        train_dataset.augment_flips = prev_aug


def _aggregate_metric_rows(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    if not rows:
        return {"count": 0, "rmse_hu": 0.0, "mae_hu": 0.0, "psnr_db": 0.0, "ssim": 0.0}
    n = float(len(rows))
    return {
        "count": len(rows),
        "rmse_hu": round(sum(r["rmse_hu"] for r in rows) / n, 4),
        "mae_hu": round(sum(r["mae_hu"] for r in rows) / n, 4),
        "psnr_db": round(sum(r["psnr_db"] for r in rows) / n, 4),
        "ssim": round(sum(r["ssim"] for r in rows) / n, 6),
    }


def run_stage7_ablation_study(
    test_dataset: ShardedDualDomainDataset,
    *,
    dual_model: Any | None = None,
    dual_checkpoint_path: str | Path | None = None,
    train_dataset: ShardedDualDomainDataset | None = None,
    val_dataset: ShardedDualDomainDataset | None = None,
    output_dir: str | Path,
    figures_dir: str | Path | None = None,
    m3_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1.5e-3,
    backend: str = "torch_cuda",
    amp_mode: str = "auto",
    num_workers: int = 4,
    resume: bool = True,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Execute the full 6-method Stage 7 Ablation Study on held-out unseen `Test` shards."""
    t_start = time.perf_counter()
    torch, device = resolve_torch_device(backend)
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (
        Path(figures_dir).expanduser().resolve()
        if figures_dir is not None
        else (out_dir / "figures")
    )
    fig_dir.mkdir(parents=True, exist_ok=True)

    if dual_model is None:
        if dual_checkpoint_path is None:
            raise ValueError("Either dual_model or dual_checkpoint_path must be supplied to Stage 7")
        dual_model, device, _ = load_dual_domain_checkpoint(
            dual_checkpoint_path, backend=backend
        )
    dual_model.configure_devices(device)
    dual_model.eval()

    img_base_ch = int(dual_model.img_net.enc1.conv1.out_channels)
    m3_net = train_or_load_m3_image_only_baseline(
        train_dataset,
        val_dataset,
        checkpoint_path=out_dir / "stage7_m3_image_only_baseline.pt",
        img_base_channels=img_base_ch,
        m3_epochs=int(m3_epochs),
        batch_size=batch_size,
        lr=lr,
        backend=backend,
        amp_mode=amp_mode,
        num_workers=num_workers,
        resume=resume,
        seed=seed,
        show_progress=show_progress,
    )

    fbp_backend = "torch_cuda" if backend == "torch_cuda" else ("torch" if backend == "torch" else "numpy")
    use_amp, amp_dtype, _ = _select_amp_configuration(device, amp_mode)
    test_dataset.set_full_mode()

    from torch.utils.data import DataLoader

    sampler = ShardAwareShuffledBatchSampler(
        test_dataset, batch_size=batch_size, shuffle=False, seed=0
    )
    loader = DataLoader(
        test_dataset,
        batch_sampler=sampler,
        num_workers=max(0, min(int(num_workers), 4)),
        pin_memory=(device.type == "cuda"),
        collate_fn=make_sharded_collate_fn(test_dataset),
    )

    overall_buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    scen_buckets: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    pitch_buckets: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    sp_buckets: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    patient_buckets: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    # Capture a representative ITF high-pitch (p ~ 2.0 or max available) slice for the 8-panel Visual QA
    rep_bundle: dict[str, Any] | None = None
    max_meas_ray_dev = 0.0
    n_done = 0
    n_total = len(test_dataset)
    t_eval0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            bsz = len(batch["global_idx"])
            s_hp_np = batch["s_hp_raw"].numpy()
            mask_np = batch["mask_hp"].numpy()
            beta_np = batch["beta_rad"].numpy()
            u_coords = batch["u_coords_mm"]
            n_u = int(u_coords.size)

            # Construct temporary SSRBResult batch to reconstruct M0 (none) & M1 (angular_feather) on GPU
            ssrb_batch = SSRBResult(
                sinograms=s_hp_np,
                measured_mask=mask_np,
                z_slices_mm=np.arange(bsz, dtype=np.float64),
                beta_rad=beta_np,
                phi_center_rad=0.5 * (beta_np[:, 0] + beta_np[:, -1]),
                u_coords_mm=u_coords,
                du_mm=float(batch["du_mm"]),
                rho0_mm=float(batch["rho0_mm"]),
                d0_mm=float(batch["d0_mm"]),
                gamma_max_rad=float(np.max(np.abs(np.arctan2(u_coords, float(batch["d0_mm"]))))),
                angular_span_rad=float(beta_np[0, -1] - beta_np[0, 0]),
                angular_mode=str(batch["angular_mode"]),
                active_row_range=(0.0, float(max(1, n_u - 1))),
            )
            fbp_m0 = reconstruct_fanbeam_fbp(
                ssrb_batch,
                image_size=int(batch["image_size"]),
                fov_mm=float(batch["fov_mm"]),
                water_mu_mm_inv=float(batch["water_mu_mm_inv"]),
                filter_name=str(dual_model.filter_name),
                inpaint_mode="none",
                backend=fbp_backend,
                show_progress=False,
            )
            fbp_m1 = reconstruct_fanbeam_fbp(
                ssrb_batch,
                image_size=int(batch["image_size"]),
                fov_mm=float(batch["fov_mm"]),
                water_mu_mm_inv=float(batch["water_mu_mm_inv"]),
                filter_name=str(dual_model.filter_name),
                inpaint_mode="angular_feather",
                backend=fbp_backend,
                show_progress=False,
            )

            x_m0 = torch.from_numpy(fbp_m0.hu_slices).to(device=device, dtype=torch.float32)
            x_m1 = torch.from_numpy(fbp_m1.hu_slices).to(device=device, dtype=torch.float32)
            x_m2 = batch["x_prelim_hu"].to(device=device, dtype=torch.float32, non_blocking=True)
            x_tgt = batch["x_target_hu"].to(device=device, dtype=torch.float32, non_blocking=True)
            fov_b = batch["fov_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
            eff_p = batch["effective_pitch"].to(device=device, dtype=torch.float32, non_blocking=True)
            miss_f = batch["missing_fraction"].to(device=device, dtype=torch.float32, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                dual_out = dual_model.forward_batch_tensors(batch, device=device)
                if m3_net is not None:
                    x_m3 = m3_net(x_m2, x_m2, fov_b, eff_p, missing_fraction=miss_f)
                else:
                    img_dev = next(dual_model.img_net.parameters()).device
                    x_m3 = dual_model.img_net(
                        x_m2.to(img_dev),
                        x_m2.to(img_dev),
                        fov_b.to(img_dev),
                        eff_p.to(img_dev),
                        missing_fraction=miss_f.to(img_dev),
                    ).to(device)

            x_m4 = dual_out["x_dc_hu"]
            x_m5 = dual_out["x_final_hu"]

            s_hp_dev = batch["s_hp_raw"].to(device=device, dtype=torch.float32, non_blocking=True)
            m_hp_dev = batch["mask_hp"].to(device=device, dtype=torch.bool, non_blocking=True)
            if torch.any(m_hp_dev):
                dev_val = float(torch.max(torch.abs(dual_out["s_dc"][m_hp_dev] - s_hp_dev[m_hp_dev])).item())
                max_meas_ray_dev = max(max_meas_ray_dev, dev_val)

            preds_by_method = {
                "M0_raw_nofill_fbp": x_m0,
                "M1_angular_feather_fbp": x_m1,
                "M2_physics_conjugate_prelim_fbp": x_m2,
                "M3_image_only_dl": x_m3,
                "M4_sino_only_dl": x_m4,
                "M5_full_dual_domain_dl": x_m5,
            }
            per_slice_metrics: dict[str, list[dict[str, float]]] = {}
            for m_key, pred_t in preds_by_method.items():
                batch_m = compute_batch_reconstruction_metrics_torch(
                    pred_t, x_tgt, fov_mask=fov_b, return_per_slice=True
                )
                per_slice_metrics[m_key] = [
                    {
                        "rmse_hu": float(batch_m["per_slice_rmse_hu"][bi]),
                        "mae_hu": float(batch_m["per_slice_mae_hu"][bi]),
                        "psnr_db": float(batch_m["per_slice_psnr_db"][bi]),
                        "ssim": float(batch_m["per_slice_ssim"][bi]),
                    }
                    for bi in range(bsz)
                ]

            for i in range(bsz):
                sc_i = str(batch["scenarios"][i]).lower()
                pt_i = float(batch["target_pitch"][i].item())
                cid_i = str(batch["case_ids"][i])
                p_key = f"p{pt_i:.1f}"
                sp_key = f"{sc_i}_p{pt_i:.1f}"

                for m_key, _ in ABLATION_METHOD_ORDER:
                    mi = per_slice_metrics[m_key][i]
                    overall_buckets[m_key].append(mi)
                    scen_buckets[sc_i][m_key].append(mi)
                    pitch_buckets[p_key][m_key].append(mi)
                    sp_buckets[sp_key][m_key].append(mi)
                    patient_buckets[cid_i][m_key].append(mi)

                prefer_this = rep_bundle is None or (
                    sc_i == "itf"
                    and abs(pt_i - 2.0) < abs(float(rep_bundle["target_pitch"]) - 2.0)
                )
                if prefer_this:
                    g_idx = int(batch["global_idx"][i])
                    dd_s = test_dataset.get_dual_domain_sample(g_idx)
                    rep_slices = {
                        mk: pt_t[i].detach().cpu().numpy().astype(np.float32)
                        for mk, pt_t in preds_by_method.items()
                    }
                    rep_m = {
                        mk: compute_reconstruction_metrics(
                            rep_slices[mk], dd_s.x_target_hu, fov_mask=dd_s.fov_mask
                        )
                        for mk in rep_slices
                    }
                    rep_bundle = {
                        "case_id": dd_s.case_id,
                        "scenario": dd_s.scenario,
                        "target_pitch": dd_s.target_pitch,
                        "z_slice_mm": dd_s.z_slice_mm,
                        "fov_mm": dd_s.fov_mm,
                        "fov_mask": dd_s.fov_mask,
                        "x_target_hu": dd_s.x_target_hu,
                        "method_slices_hu": rep_slices,
                        "method_metrics": rep_m,
                    }

            n_done += bsz
            if show_progress:
                pct = 100.0 * float(n_done) / float(max(n_total, 1))
                el = max(1e-3, time.perf_counter() - t_eval0)
                eta = el * (1.0 - pct / 100.0) / max(pct / 100.0, 1e-4)
                print(
                    f"[Stage 7 Ablation {pct:5.1f}% | ETA {format_duration_hms(eta)}] "
                    f"Evaluated M0..M5 on {n_done}/{n_total} unseen Test slices",
                    file=sys.stderr,
                    flush=True,
                )

    def _summarize_group(bucket: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
        m2_agg = _aggregate_metric_rows(bucket.get("M2_physics_conjugate_prelim_fbp", []))
        m2_rmse = float(m2_agg["rmse_hu"])
        out_g: dict[str, dict[str, float]] = {}
        for m_key, m_label in ABLATION_METHOD_ORDER:
            agg = _aggregate_metric_rows(bucket.get(m_key, []))
            out_g[m_key] = {
                "label": m_label,
                **agg,
                "delta_rmse_vs_m2_prelim_hu": round(float(agg["rmse_hu"]) - m2_rmse, 4),
            }
        return out_g

    overall_summary = _summarize_group(overall_buckets)
    by_scenario = {sc: _summarize_group(b) for sc, b in sorted(scen_buckets.items())}
    by_pitch = {pk: _summarize_group(b) for pk, b in sorted(pitch_buckets.items())}
    by_scenario_pitch = {spk: _summarize_group(b) for spk, b in sorted(sp_buckets.items())}
    by_patient = {cid: _summarize_group(b) for cid, b in sorted(patient_buckets.items())}

    # Export clean thesis CSV table (`overall` + `by_scenario` + `by_scenario_pitch`)
    csv_path = out_dir / "stage7_ablation_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "stratum",
                "method_id",
                "method_label",
                "sample_count",
                "rmse_hu",
                "mae_hu",
                "psnr_db",
                "ssim",
                "delta_rmse_vs_m2_prelim_hu",
            ]
        )
        for m_key, _ in ABLATION_METHOD_ORDER:
            row = overall_summary[m_key]
            writer.writerow(
                [
                    "OVERALL",
                    m_key,
                    row["label"],
                    row["count"],
                    row["rmse_hu"],
                    row["mae_hu"],
                    row["psnr_db"],
                    row["ssim"],
                    row["delta_rmse_vs_m2_prelim_hu"],
                ]
            )
        for sp_key, grp in by_scenario_pitch.items():
            for m_key, _ in ABLATION_METHOD_ORDER:
                row = grp[m_key]
                writer.writerow(
                    [
                        sp_key.upper(),
                        m_key,
                        row["label"],
                        row["count"],
                        row["rmse_hu"],
                        row["mae_hu"],
                        row["psnr_db"],
                        row["ssim"],
                        row["delta_rmse_vs_m2_prelim_hu"],
                    ]
                )

    grid_png_str: str | None = None
    curves_png_str: str | None = None
    if rep_bundle is not None:
        grid_png = save_stage7_ablation_visual_qa(
            fig_dir / "stage7_ablation_visual_qa.png",
            **rep_bundle,
        )
        grid_png_str = str(grid_png)
    curves_png = save_stage7_pitch_degradation_curves(
        fig_dir / "stage7_pitch_degradation_curves.png",
        by_scenario_pitch=by_scenario_pitch,
    )
    curves_png_str = str(curves_png)

    m0_r = float(overall_summary["M0_raw_nofill_fbp"]["rmse_hu"])
    m2_r = float(overall_summary["M2_physics_conjugate_prelim_fbp"]["rmse_hu"])
    m5_r = float(overall_summary["M5_full_dual_domain_dl"]["rmse_hu"])
    ablation_pass = bool(
        n_total > 0
        and max_meas_ray_dev == 0.0
        and m2_r <= m0_r
        and m5_r <= m2_r + 1e-4
    )

    report: dict[str, Any] = {
        "stage": "Stage 7 Ablation Study (v1.0.0)",
        "overall_pass": ablation_pass,
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
        "test_patient_ids": test_dataset.case_ids,
        "test_sample_count": n_total,
        "m3_standalone_trained": bool(m3_net is not None),
        "m3_epochs": int(m3_epochs) if m3_net is not None else 0,
        "measured_rays_max_abs_dev": round(max_meas_ray_dev, 8),
        "overall_ablation_matrix": overall_summary,
        "by_scenario": by_scenario,
        "by_pitch": by_pitch,
        "by_scenario_pitch": by_scenario_pitch,
        "by_patient": by_patient,
        "ablation_csv": str(csv_path),
        "visual_qa_grid_png": grid_png_str,
        "pitch_curves_png": curves_png_str,
    }
    report_json = out_dir / "stage7_ablation_report.json"
    report["report_json"] = str(report_json)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
