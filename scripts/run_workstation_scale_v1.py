#!/usr/bin/env python3
"""All-in-one Workstation Launcher (v1.0.0) for RTX PRO 6000 (96 GB) / A100 (80 GB).

Performs end-to-end, fully resumable execution:
  1. Hardware & VRAM / CPU RAM telemetry check (auto-tunes batch size & CPU workers).
  2. Discovers 35 patient cases (CHEST + ABDOMEN) on TCIA `LDCT-and-Projection-data`.
  3. Downloads raw DICOM-CT-PD projection & CT series in `full` workstation profile (resumable).
  4. Reconstructs full-volume patient shards (~50,000 slices across 25 Train / 5 Val / 5 Test)
     with GPU Curved-to-Flat, multi-threaded SSRB, Batched GPU Parker FBP, and GPU Helical3D
     reporting live `%` and `ETA` per patient and per substep.
  5. Trains the 4-channel Activated Dual-Domain Network for 100 epochs with AMP (`bf16`), `TF32`,
     per-epoch `Val` evaluation (`RMSE`, `MAE`, `PSNR`, `SSIM`, `%`, `ETA`), and atomic `--resume`.
  6. Runs Stage 7 6-method Thesis Ablation Study (`M0..M5`) on the 5 unseen Test patients.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _detect_optimal_batch_size(default_batch: int = 24) -> tuple[int, int, str]:
    """Select optimal mini-batch size, eval batch size, and AMP mode based on GPU VRAM."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 4, 8, "off"
        vram_gib = torch.cuda.get_device_properties(0).total_memory / float(1024**3)
        bf16_ok = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        amp_mode = "bf16" if bf16_ok else "fp16"
        if vram_gib >= 70.0:
            return max(default_batch, 24), 32, amp_mode
        if vram_gib >= 38.0:
            return 16, 24, amp_mode
        if vram_gib >= 20.0:
            return 8, 16, amp_mode
        return 4, 8, amp_mode
    except Exception:
        return default_batch, default_batch, "auto"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("HPCT_DATA_ROOT", "/workspace/highpitchct-data"),
        help="Writable workstation directory for TCIA data, shards, checkpoints, and reports",
    )
    parser.add_argument("--n-train", type=int, default=25)
    parser.add_argument("--n-val", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--target-total-samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto-tune for GPU VRAM")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--m3-epochs", type=int, default=15)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root).expanduser().resolve()
    manifest_csv = data_root / "manifests" / "cohort_35_patients.csv"
    data_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_str = str(repo_root / "src")
    env["PYTHONPATH"] = f"{src_str}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_str

    cpu_count = os.cpu_count() or 8
    io_workers = max(4, min(16, cpu_count - 2))
    loader_workers = max(2, min(8, cpu_count // 2))
    auto_bs, eval_bs, amp_mode = _detect_optimal_batch_size()
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else auto_bs
    total_cases = int(args.n_train) + int(args.n_val) + int(args.n_test)

    print(
        f"=== HighPitchCT v1.0.0 Workstation Pipeline ===\n"
        f"  data_root    : {data_root}\n"
        f"  cohort split : {args.n_train} Train / {args.n_val} Val / {args.n_test} Test ({total_cases} patients)\n"
        f"  target slices: ~{args.target_total_samples}\n"
        f"  training     : {args.epochs} epochs | batch_size={batch_size} | eval_batch={eval_bs} | amp={amp_mode}\n"
        f"  workers      : dicom/ssrb={io_workers} | dataloader={loader_workers}",
        flush=True,
    )

    if not args.skip_download:
        if not manifest_csv.is_file():
            print(f"\n=== [1/3] Discovering {total_cases} Patients on TCIA (CHEST,ABDOMEN) ===", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "highpitchct.cli",
                    "discover-tcia",
                    "--body-part",
                    "CHEST,ABDOMEN",
                    "--roles",
                    "projection_full,reconstruction_published",
                    "--max-cases",
                    str(total_cases),
                    "--output",
                    str(manifest_csv),
                ],
                env=env,
                check=True,
            )

        print("\n=== [2/3] Downloading / Verifying Patient Series (Resumable Full Profile) ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "highpitchct.cli",
                "download",
                "--series-file",
                str(manifest_csv),
                "--data-root",
                str(data_root),
                "--profile",
                "full",
                "--cleanup-zip",
                "--workers",
                str(min(8, io_workers)),
            ],
            env=env,
            check=True,
        )

    print(
        "\n=== [3/3] Full-Volume Sharding + 100-Epoch Val Training + Stage 7 Ablation (Resumable) ===",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "highpitchct.cli",
            "stage6-scale",
            "--data-root",
            str(data_root),
            "--n-train",
            str(args.n_train),
            "--n-val",
            str(args.n_val),
            "--n-test",
            str(args.n_test),
            "--target-total-samples",
            str(args.target_total_samples),
            "--pitches",
            "1.5,2.0,2.5",
            "--scenario",
            "both",
            "--itf-ratio",
            "0.60",
            "--image-size",
            str(args.image_size),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(batch_size),
            "--eval-batch-size",
            str(eval_bs),
            "--lr",
            str(args.lr),
            "--sino-channels",
            "64",
            "--img-channels",
            "64",
            "--img-in-channels",
            "4",
            "--backend",
            "torch_cuda",
            "--amp",
            amp_mode,
            "--workers",
            str(io_workers),
            "--num-loader-workers",
            str(loader_workers),
            "--run-stage7-ablation",
            "--m3-epochs",
            str(args.m3_epochs),
            "--resume",
        ],
        env=env,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
