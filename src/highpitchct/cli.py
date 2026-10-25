"""Command-line interface for Stage 0 (smart Kaggle/Full download & DICOM-CT-PD indexing)."""

from __future__ import annotations

import argparse
import json

from .index_dicom import index_dicom_tree
from .manifest import read_series_manifest, write_series_manifest
from .runtime import print_runtime_report, resolve_environment_profile
from .tcia_client import DEFAULT_COLLECTION, discover_tcia_series, download_manifest_with_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpct",
        description="HighPitchCT Stage-0 CLI (smart Kaggle <=18 GiB & Full workstation modes)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser(
        "discover-tcia",
        help="Query NBIA public API and generate a reproducible series CSV manifest",
    )
    discover.add_argument("--collection", default=DEFAULT_COLLECTION)
    discover.add_argument(
        "--body-part",
        help="Optional BodyPartExamined filter, e.g., CHEST or ABDOMEN",
    )
    discover.add_argument(
        "--roles",
        default="projection_full,reconstruction_published",
        help="Comma-separated roles to include (default: projection_full,reconstruction_published)",
    )
    discover.add_argument("--max-cases", type=int, help="Optional maximum number of patient cases")
    discover.add_argument("--output", required=True, help="Output CSV path")
    discover.add_argument("--timeout", type=int, default=120)

    validate = sub.add_parser(
        "validate-manifest", help="Validate a series CSV without downloading"
    )
    validate.add_argument("--series-file", required=True)

    download = sub.add_parser(
        "download",
        help="Download TCIA series with smart Kaggle (<=18 GiB) vs Full disk quota management",
    )
    download.add_argument("--series-file", required=True)
    download.add_argument("--data-root", help="Writable root for raw/ and manifests/")
    download.add_argument(
        "--incoming-root",
        help="Staging directory for .zip/.part files (defaults to /tmp/highpitchct-incoming on Kaggle)",
    )
    download.add_argument(
        "--profile",
        choices=["auto", "kaggle", "full"],
        default=None,
        help="Execution profile: auto (default), kaggle (<=18 GiB cap + /tmp staging), or full (no cap)",
    )
    download.add_argument(
        "--max-disk-gb",
        type=float,
        default=None,
        help="Explicit disk quota cap in GiB for data-root (overrides profile default)",
    )
    zip_group = download.add_mutually_exclusive_group()
    zip_group.add_argument(
        "--cleanup-zip",
        action="store_true",
        help="Delete verified .zip archive after extraction (default on Kaggle)",
    )
    zip_group.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep .zip archive after extraction (default on Full profile)",
    )
    download.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel series download workers across admitted patients (default: 4)",
    )
    download.add_argument("--retries", type=int, default=5)
    download.add_argument("--timeout", type=int, default=120)
    download.add_argument("--no-extract", action="store_true")
    download.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live tqdm progress bars and stderr status logs",
    )

    index = sub.add_parser(
        "index",
        help="Create header-only DICOM & DICOM-CT-PD v4 inventory and series audit",
    )
    index.add_argument("--data-root", help="Writable root where manifests/ will be saved")
    index.add_argument(
        "--scan-dir",
        action="append",
        dest="scan_dirs",
        help="Optional directory to scan (e.g., /kaggle/input/... or external drive); repeatable",
    )
    index.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel threads for header-only DICOM scanning (default: 8)",
    )
    index.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live tqdm progress bar during indexing",
    )

    environment = sub.add_parser(
        "env", help="Report Kaggle/Full profile, disk quota, RAM, disk space, and GPUs"
    )
    environment.add_argument("--data-root")
    environment.add_argument("--profile", choices=["auto", "kaggle", "full"], default=None)
    environment.add_argument("--max-disk-gb", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "discover-tcia":
        roles = {r.strip() for r in args.roles.split(",") if r.strip()}
        specs = discover_tcia_series(
            collection=args.collection,
            body_part=args.body_part,
            roles=roles,
            max_cases=args.max_cases,
            timeout=args.timeout,
        )
        out_path = write_series_manifest(args.output, specs, enabled=True)
        total_expected = sum(s.expected_bytes or 0 for s in specs)
        cases = sorted({s.case_id for s in specs})
        print(
            json.dumps(
                {
                    "output_csv": str(out_path),
                    "cases_discovered": len(cases),
                    "series_discovered": len(specs),
                    "total_expected_gb": round(total_expected / (1024**3), 3),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate-manifest":
        specs = read_series_manifest(args.series_file)
        cases = sorted({s.case_id for s in specs})
        total_expected = sum(s.expected_bytes or 0 for s in specs)
        print(
            json.dumps(
                {
                    "enabled_cases": len(cases),
                    "enabled_series": len(specs),
                    "known_expected_gb": round(total_expected / (1024**3), 3)
                    if total_expected > 0
                    else None,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "env":
        print_runtime_report(
            args.data_root, profile=args.profile, max_disk_gb=args.max_disk_gb
        )
        return 0

    if args.command == "index":
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        env_prof = resolve_environment_profile(data_root=args.data_root)
        summary = index_dicom_tree(
            env_prof.data_root,
            scan_dirs=args.scan_dirs,
            workers=args.workers,
            show_progress=not args.no_progress,
        )
        # Print concise summary without dumping thousands of rows to stdout
        compact = {k: v for k, v in summary.items() if k != "series_summaries"}
        compact["series_validation_statuses"] = {
            s["series_instance_uid"]: {
                "kind": s["dicom_kind"],
                "files": s["file_count"],
                "pitch": s["estimated_normalized_pitch"],
                "status": s["validation_status"],
                "notes": s["validation_notes"],
            }
            for s in summary["series_summaries"]
        }
        print(json.dumps(compact, indent=2))
        return 0

    # download command
    specs = read_series_manifest(args.series_file)
    if not specs:
        raise SystemExit("No enabled series in the manifest")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    cleanup_override: bool | None = None
    if args.cleanup_zip:
        cleanup_override = True
    elif args.keep_zip:
        cleanup_override = False

    env_prof = resolve_environment_profile(
        profile=args.profile,
        data_root=args.data_root,
        incoming_root=args.incoming_root,
        max_disk_gb=args.max_disk_gb,
        cleanup_zip=cleanup_override,
    )
    summary = download_manifest_with_profile(
        specs,
        env_prof,
        workers=args.workers,
        retries=args.retries,
        timeout=args.timeout,
        extract=not args.no_extract,
        show_progress=not args.no_progress,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 1 if summary["failed_series"] > 0 else 0
