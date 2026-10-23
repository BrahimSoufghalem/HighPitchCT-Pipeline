"""Command-line interface for Stage 0."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .index_dicom import index_dicom_tree
from .manifest import read_series_manifest
from .tcia_client import download_series


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hpct", description="HighPitchCT Stage-0 tools")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-manifest", help="Validate a series CSV without downloading")
    validate.add_argument("--series-file", required=True)

    download = sub.add_parser("download", help="Download exact TCIA series from a manifest")
    download.add_argument("--series-file", required=True)
    download.add_argument("--data-root", default="data")
    download.add_argument("--workers", type=int, default=2)
    download.add_argument("--retries", type=int, default=5)
    download.add_argument("--timeout", type=int, default=120)
    download.add_argument("--no-extract", action="store_true")

    index = sub.add_parser("index", help="Create a header-only DICOM inventory")
    index.add_argument("--data-root", default="data")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-manifest":
        specs = read_series_manifest(args.series_file)
        print(json.dumps({"enabled_series": len(specs)}, indent=2))
        return 0

    if args.command == "index":
        print(json.dumps(index_dicom_tree(args.data_root), indent=2))
        return 0

    specs = read_series_manifest(args.series_file)
    if not specs:
        raise SystemExit("No enabled series in the manifest")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(
                download_series,
                spec,
                args.data_root,
                retries=args.retries,
                timeout=args.timeout,
                extract=not args.no_extract,
            ): spec
            for spec in specs
        }
        for future in as_completed(jobs):
            spec = jobs[future]
            try:
                result = future.result()
                print(json.dumps(result, sort_keys=True))
            except Exception as error:
                failed += 1
                print(json.dumps({"series": spec.series_instance_uid, "error": str(error)}))
    return 1 if failed else 0
