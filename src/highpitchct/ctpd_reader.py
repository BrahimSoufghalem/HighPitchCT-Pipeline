"""Memory-safe DICOM-CT-PD v4 projection reader, CT image reader, and orientation verifier."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .ctpd_tags import (
    classify_dicom_sop,
    decode_named_ctpd_tag,
    decode_tag,
    extract_referenced_raw_series_uid,
)
from .geometry import HelicalGeometry, unwrap_angles_rad
from .index_dicom import _collect_candidates
from .progress import ProgressReporter


def hu_to_mu(
    hu: np.ndarray | float,
    water_mu_mm_inv: float = 0.0192,
    *,
    clip_non_negative: bool = False,
) -> np.ndarray | float:
    """Convert Hounsfield Units [HU] to linear attenuation mu [mm^-1]."""
    if water_mu_mm_inv <= 0.0:
        raise ValueError(f"water_mu_mm_inv must be positive, got {water_mu_mm_inv}")
    if isinstance(hu, (int, float, np.floating, np.integer)) and not isinstance(hu, np.ndarray):
        mu_val = float(water_mu_mm_inv) * (1.0 + float(hu) / 1000.0)
        return max(0.0, mu_val) if clip_non_negative else mu_val

    arr = np.asarray(hu)
    out_dtype = np.float64 if arr.dtype == np.float64 else np.float32
    mu = arr.astype(out_dtype, copy=False) * out_dtype(water_mu_mm_inv / 1000.0) + out_dtype(
        water_mu_mm_inv
    )
    if clip_non_negative:
        mu = np.maximum(mu, out_dtype(0.0))
    return mu


def mu_to_hu(
    mu: np.ndarray | float,
    water_mu_mm_inv: float = 0.0192,
) -> np.ndarray | float:
    """Convert linear attenuation mu [mm^-1] to Hounsfield Units [HU]."""
    if water_mu_mm_inv <= 0.0:
        raise ValueError(f"water_mu_mm_inv must be positive, got {water_mu_mm_inv}")
    if isinstance(mu, (int, float, np.floating, np.integer)) and not isinstance(mu, np.ndarray):
        return 1000.0 * (float(mu) - float(water_mu_mm_inv)) / float(water_mu_mm_inv)

    arr = np.asarray(mu)
    out_dtype = np.float64 if arr.dtype == np.float64 else np.float32
    inv_w = out_dtype(1000.0 / water_mu_mm_inv)
    return (arr.astype(out_dtype, copy=False) - out_dtype(water_mu_mm_inv)) * inv_w


def scan_projection_series(
    series_dir: str | Path,
    *,
    series_uid: str | None = None,
    workers: int = 8,
    show_progress: bool = False,
) -> tuple[HelicalGeometry, list[Path]]:
    """Scan DICOM-CT-PD v4 headers in parallel, validate trajectory, and build HelicalGeometry.

    Does NOT load PixelData into memory. Sorts views strictly by InstanceNumber.
    """
    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError("pydicom is required to scan DICOM-CT-PD series") from error

    sdir = Path(series_dir).expanduser().resolve()
    if not sdir.exists():
        raise FileNotFoundError(f"Projection series directory does not exist: {sdir}")

    candidates = _collect_candidates(sdir)
    if not candidates:
        raise ValueError(f"No candidate DICOM files found under {sdir}")

    def _parse_header(item: tuple[Path, Path, int]) -> dict[str, Any] | None:
        _, path, _ = item
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except Exception:
            return None
        if classify_dicom_sop(ds) != "dicom_ct_pd_projection":
            return None
        suid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        if series_uid is not None and suid != series_uid:
            return None

        inst_raw = getattr(ds, "InstanceNumber", None)
        if inst_raw is None:
            raise ValueError(f"Missing InstanceNumber in projection file {path}")
        inst = int(inst_raw)

        phi0 = decode_named_ctpd_tag(ds, "phi0_rad")
        z0 = decode_named_ctpd_tag(ds, "z0_mm")
        if phi0 is None or z0 is None:
            raise ValueError(f"Missing (7031,1001) phi0 or (7031,1002) z0 in {path}")

        dphi = decode_named_ctpd_tag(ds, "dphi_rad")
        dz = decode_named_ctpd_tag(ds, "dz_mm")
        drho = decode_named_ctpd_tag(ds, "drho_mm")

        return {
            "path": path,
            "suid": suid,
            "inst": inst,
            "rows_tag": int(getattr(ds, "Rows", 0)),
            "cols_tag": int(getattr(ds, "Columns", 0)),
            "phi0": float(phi0),
            "z0": float(z0),
            "dphi": None if dphi is None else float(dphi),
            "dz": None if dz is None else float(dz),
            "drho": None if drho is None else float(drho),
        }

    records: list[dict[str, Any]] = []
    effective_workers = max(1, min(int(workers), len(candidates)))
    with ProgressReporter(
        total=len(candidates),
        desc="📐 Scanning CT-PD geometry",
        unit="view",
        unit_scale=False,
        colour="cyan",
        enabled=show_progress,
    ) as bar:
        if effective_workers == 1:
            for item in candidates:
                rec = _parse_header(item)
                bar.update(1)
                if rec is not None:
                    records.append(rec)
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                for rec in pool.map(_parse_header, candidates, chunksize=64):
                    bar.update(1)
                    if rec is not None:
                        records.append(rec)

    if not records:
        raise ValueError(f"No DICOM-CT-PD projection files found under {sdir}")

    found_uids = sorted({r["suid"] for r in records})
    if len(found_uids) > 1:
        raise ValueError(
            f"Multiple projection SeriesInstanceUIDs found under {sdir}: {found_uids}. "
            "Pass an explicit series directory or series_uid (no silent fallback)."
        )
    active_uid = found_uids[0]

    records.sort(key=lambda r: r["inst"])
    inst_arr = np.array([r["inst"] for r in records], dtype=np.int32)
    unique_count = int(np.unique(inst_arr).size)
    if unique_count != inst_arr.size:
        raise ValueError(
            f"Series {active_uid} contains {inst_arr.size - unique_count} duplicate InstanceNumbers"
        )
    expected_span = int(inst_arr[-1] - inst_arr[0] + 1)
    if expected_span != inst_arr.size:
        raise ValueError(
            f"Series {active_uid} is missing {expected_span - inst_arr.size} views "
            f"between InstanceNumber {inst_arr[0]} and {inst_arr[-1]}"
        )

    # Read full static scanner metadata from the first sorted file
    ds0 = pydicom.dcmread(records[0]["path"], stop_before_pixels=True, force=False)
    n_rows = decode_named_ctpd_tag(ds0, "detector_rows")
    n_cols = decode_named_ctpd_tag(ds0, "detector_cols")
    du_mm = decode_named_ctpd_tag(ds0, "du_mm")
    dv_mm = decode_named_ctpd_tag(ds0, "dv_mm")
    rho0_mm = decode_named_ctpd_tag(ds0, "rho0_mm")
    d0_mm = decode_named_ctpd_tag(ds0, "d0_mm")
    center_1based = decode_named_ctpd_tag(ds0, "detector_central_element_1based")
    detector_shape = str(decode_named_ctpd_tag(ds0, "detector_shape") or "").strip().upper()
    ffs_mode = str(decode_named_ctpd_tag(ds0, "ffs_mode") or "FFSNONE").strip().upper()
    n_rot = decode_named_ctpd_tag(ds0, "angular_steps_per_rotation")
    water_mu = decode_named_ctpd_tag(ds0, "water_attenuation_mm_inv")
    log_flag = str(decode_named_ctpd_tag(ds0, "log_flag") or "YES").strip().upper()
    nominal_pitch = decode_tag(ds0, 0x0018, 0x9311, "FD")
    slope0 = float(getattr(ds0, "RescaleSlope", 1.0))
    intercept0 = float(getattr(ds0, "RescaleIntercept", 0.0))

    if None in (n_rows, n_cols, du_mm, dv_mm, rho0_mm, d0_mm, center_1based, n_rot):
        raise ValueError(f"Series {active_uid} is missing required DICOM-CT-PD v4 geometry tags")
    if not isinstance(center_1based, list) or len(center_1based) < 2:
        raise ValueError(
            f"Invalid (7031,1033) DetectorCentralElement={center_1based!r}; expected [Col X, Row Y]"
        )

    n_rows_i = int(n_rows)
    n_cols_i = int(n_cols)
    for rec in records:
        if rec["rows_tag"] != n_cols_i or rec["cols_tag"] != n_rows_i:
            raise ValueError(
                f"Unexpected DICOM (Rows, Columns)=({rec['rows_tag']}, {rec['cols_tag']}) in "
                f"{rec['path']}; DICOM-CT-PD v4 requires (Rows={n_cols_i}, Columns={n_rows_i})"
            )

    if ffs_mode != "FFSNONE":
        for rec in records:
            if rec["dphi"] is None or rec["dz"] is None or rec["drho"] is None:
                raise ValueError(
                    f"Series {active_uid} declares ffs_mode={ffs_mode!r} but view "
                    f"InstanceNumber={rec['inst']} is missing FFS shift tags (7033,100B/C/D)"
                )
        dphi_arr = np.array([r["dphi"] for r in records], dtype=np.float64)
        dz_arr = np.array([r["dz"] for r in records], dtype=np.float64)
        drho_arr = np.array([r["drho"] for r in records], dtype=np.float64)
    else:
        dphi_arr = np.array([0.0 if r["dphi"] is None else r["dphi"] for r in records], dtype=np.float64)
        dz_arr = np.array([0.0 if r["dz"] is None else r["dz"] for r in records], dtype=np.float64)
        drho_arr = np.array([0.0 if r["drho"] is None else r["drho"] for r in records], dtype=np.float64)

    phi0_arr = np.array([r["phi0"] for r in records], dtype=np.float64)
    phi0_unwrap = unwrap_angles_rad(phi0_arr)
    z0_arr = np.array([r["z0"] for r in records], dtype=np.float64)

    # Convert 1-based [Col X, Row Y] -> 0-based (center_col_0based, center_row_0based)
    center_col_0 = float(center_1based[0]) - 1.0
    center_row_0 = float(center_1based[1]) - 1.0

    geometry = HelicalGeometry(
        series_instance_uid=active_uid,
        detector_shape=detector_shape,
        n_rows=n_rows_i,
        n_cols=n_cols_i,
        du_mm=float(du_mm),
        dv_mm=float(dv_mm),
        rho0_mm=float(rho0_mm),
        d0_mm=float(d0_mm),
        center_col_0based=center_col_0,
        center_row_0based=center_row_0,
        ffs_mode=ffs_mode,
        angular_steps_per_rotation=int(n_rot),
        water_mu_mm_inv=float(water_mu) if water_mu is not None else 0.0192,
        nominal_spiral_pitch=float(nominal_pitch) if nominal_pitch is not None else None,
        log_flag=log_flag,
        rescale_slope=slope0,
        rescale_intercept=intercept0,
        instance_numbers=inst_arr,
        phi0_rad=phi0_arr,
        phi0_unwrapped_rad=phi0_unwrap,
        z0_mm=z0_arr,
        dphi_rad=dphi_arr,
        dz_mm=dz_arr,
        drho_mm=drho_arr,
    )
    sorted_paths = [r["path"] for r in records]
    return geometry, sorted_paths


def _extract_raw_matrix_fortran(
    ds: Any,
    *,
    n_rows: int,
    n_cols: int,
    order: str = "F",
) -> np.ndarray:
    """Decode raw 16-bit PixelData into (n_rows, n_cols) using explicit Fortran order."""
    rows_tag = int(getattr(ds, "Rows", 0))
    cols_tag = int(getattr(ds, "Columns", 0))
    if rows_tag != n_cols or cols_tag != n_rows:
        raise ValueError(
            f"Expected DICOM (Rows, Columns)=({n_cols}, {n_rows}), got ({rows_tag}, {cols_tag})"
        )

    bits_alloc = int(getattr(ds, "BitsAllocated", 16))
    pixel_repr = int(getattr(ds, "PixelRepresentation", 0))
    raw_bytes = getattr(ds, "PixelData", None)
    expected_nbytes = n_rows * n_cols * (bits_alloc // 8)

    if isinstance(raw_bytes, (bytes, bytearray)) and len(raw_bytes) == expected_nbytes and bits_alloc == 16:
        dtype = np.dtype("<i2") if pixel_repr == 1 else np.dtype("<u2")
        return np.frombuffer(raw_bytes, dtype=dtype).reshape((n_rows, n_cols), order=order)

    # Fallback for compressed transfer syntaxes via pydicom.pixel_array
    px = ds.pixel_array
    if px.shape != (n_cols, n_rows):
        raise ValueError(f"Unexpected pixel_array shape {px.shape}, expected ({n_cols}, {n_rows})")
    if order == "F":
        return px.T
    return px.reshape((n_rows, n_cols), order="C")


def read_single_projection(
    path: str | Path,
    *,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
    apply_log_if_needed: bool = True,
) -> np.ndarray:
    """Read a single DICOM-CT-PD projection as float32 line integral [row_idx, col_idx].

    Applies g = raw * RescaleSlope + RescaleIntercept and respects (7039,1009) LogFlag
    so already-log-transformed projections (LogFlag == 'YES') are NEVER logged twice.
    """
    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError("pydicom is required to read DICOM-CT-PD projections") from error

    ds = pydicom.dcmread(Path(path), force=False)
    n_rows = expected_rows if expected_rows is not None else int(decode_named_ctpd_tag(ds, "detector_rows"))
    n_cols = expected_cols if expected_cols is not None else int(decode_named_ctpd_tag(ds, "detector_cols"))

    raw_rc = _extract_raw_matrix_fortran(ds, n_rows=n_rows, n_cols=n_cols, order="F")
    slope = np.float32(float(getattr(ds, "RescaleSlope", 1.0)))
    intercept = np.float32(float(getattr(ds, "RescaleIntercept", 0.0)))
    proj = np.ascontiguousarray(raw_rc, dtype=np.float32)
    proj *= slope
    proj += intercept

    log_flag = str(decode_named_ctpd_tag(ds, "log_flag") or "YES").strip().upper()
    if log_flag == "NO" and apply_log_if_needed:
        if float(np.max(proj)) <= 0.0:
            raise ValueError(f"Cannot apply -log to non-positive transmission data in {path}")
        proj = -np.log(np.maximum(proj, np.float32(1e-7)))
    elif log_flag not in {"YES", "NO"}:
        raise ValueError(f"Unsupported (7039,1009) LogFlag={log_flag!r} in {path}")

    return proj


def iter_projection_chunks(
    files: list[Path],
    geometry: HelicalGeometry,
    *,
    chunk_views: int = 256,
    workers: int = 4,
    show_progress: bool = False,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Stream projection views in bounded-RAM float32 chunks of shape (K, n_rows, n_cols)."""
    if len(files) != geometry.n_views:
        raise ValueError(f"File count ({len(files)}) != geometry.n_views ({geometry.n_views})")
    if chunk_views < 1:
        raise ValueError("chunk_views must be >= 1")

    n_total = len(files)
    effective_workers = max(1, min(int(workers), chunk_views))

    def _read_idx(fpath: Path) -> np.ndarray:
        return read_single_projection(
            fpath,
            expected_rows=geometry.n_rows,
            expected_cols=geometry.n_cols,
        )

    with ProgressReporter(
        total=n_total,
        desc="📖 Streaming CT-PD projections",
        unit="view",
        unit_scale=False,
        colour="blue",
        enabled=show_progress,
    ) as bar:
        if effective_workers == 1:
            for start in range(0, n_total, chunk_views):
                stop = min(n_total, start + chunk_views)
                chunk = np.empty((stop - start, geometry.n_rows, geometry.n_cols), dtype=np.float32)
                for k, fpath in enumerate(files[start:stop]):
                    chunk[k] = _read_idx(fpath)
                    bar.update(1)
                yield start, stop, chunk
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                for start in range(0, n_total, chunk_views):
                    stop = min(n_total, start + chunk_views)
                    chunk = np.empty(
                        (stop - start, geometry.n_rows, geometry.n_cols), dtype=np.float32
                    )
                    for k, arr in enumerate(pool.map(_read_idx, files[start:stop])):
                        chunk[k] = arr
                        bar.update(1)
                    yield start, stop, chunk


def load_projections_memmap(
    series_dir: str | Path,
    memmap_path: str | Path,
    *,
    series_uid: str | None = None,
    chunk_views: int = 256,
    workers: int = 4,
    show_progress: bool = False,
) -> tuple[HelicalGeometry, np.memmap]:
    """Stream an entire DICOM-CT-PD series into a disk-backed float32 memmap (<100 MiB RAM)."""
    geometry, files = scan_projection_series(
        series_dir,
        series_uid=series_uid,
        workers=max(workers, 4),
        show_progress=show_progress,
    )
    mm_path = Path(memmap_path).expanduser().resolve()
    mm_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(
        mm_path,
        dtype=np.float32,
        mode="w+",
        shape=(geometry.n_views, geometry.n_rows, geometry.n_cols),
    )
    for start, stop, chunk in iter_projection_chunks(
        files,
        geometry,
        chunk_views=chunk_views,
        workers=workers,
        show_progress=show_progress,
    ):
        mm[start:stop] = chunk
    mm.flush()
    geometry.to_npz(mm_path.with_suffix(mm_path.suffix + ".geometry.npz"))
    return geometry, mm


def _bilinear_sample_2d(
    img: np.ndarray,
    rows_f: np.ndarray,
    cols_f: np.ndarray,
) -> np.ndarray:
    """Vectorized 2D bilinear interpolation on img[row, col] for valid interior coordinates."""
    r0 = np.floor(rows_f).astype(np.int64)
    c0 = np.floor(cols_f).astype(np.int64)
    r1 = np.minimum(r0 + 1, img.shape[0] - 1)
    c1 = np.minimum(c0 + 1, img.shape[1] - 1)
    wr = rows_f - r0
    wc = cols_f - c0
    v00 = img[r0, c0]
    v01 = img[r0, c1]
    v10 = img[r1, c0]
    v11 = img[r1, c1]
    return (
        (1.0 - wr) * (1.0 - wc) * v00
        + (1.0 - wr) * wc * v01
        + wr * (1.0 - wc) * v10
        + wr * wc * v11
    )


def _detect_ffs_period(geometry: HelicalGeometry) -> int:
    """Detect repeating FFS state period (1 for FFSNONE, 2 for FFSZ/FFSXY, 4 for FFSXYZ)."""
    max_p = min(8, geometry.n_views // 2)
    states = np.round(
        np.stack([geometry.drho_mm, geometry.dphi_rad, geometry.dz_mm], axis=-1),
        decimals=4,
    )
    for period in (1, 2, 4, 8):
        if period > max_p:
            break
        n_check = min(geometry.n_views - period, 64)
        if n_check > 0 and np.allclose(states[:n_check], states[period : period + n_check], atol=2e-4):
            return period
    return 1


def _evaluate_conjugate_hypothesis(
    block_a: np.ndarray,
    idx_a: np.ndarray,
    block_b: np.ndarray,
    idx_b: np.ndarray,
    geometry: HelicalGeometry,
    *,
    flip_rows: bool,
    flip_cols: bool,
    center_col_override: float | None = None,
) -> tuple[float, int]:
    """Compute exact 3D helical half-turn conjugate-ray MAE for a geometry hypothesis."""
    geom = (
        geometry.with_modified_centers(center_col_0based=center_col_override)
        if center_col_override is not None
        else geometry
    )

    # All views in idx_a and idx_b share the exact same FFS state, so horizontal ray
    # geometry rotates rigidly around (0,0) by phi0_rad!
    ref_view = int(idx_a[0])
    phi_ref = float(geom.phi0_rad[ref_view])
    cos_r = float(np.cos(-phi_ref))
    sin_r = float(np.sin(-phi_ref))

    src_ref, dirs_ref, _ = geom.ray_origins_and_directions(
        ref_view, flip_rows=flip_rows, flip_cols=flip_cols
    )
    # Use central row to extract horizontal (x, y) geometry per column
    mid_r = geom.n_rows // 2
    det_ref = geom.detector_element_cartesian_mm(
        ref_view, flip_rows=flip_rows, flip_cols=flip_cols
    )[mid_r]  # (C, 3)

    # Rotate reference view back to canonical phi0 == 0
    sx0 = src_ref[0] * cos_r - src_ref[1] * sin_r
    sy0 = src_ref[0] * sin_r + src_ref[1] * cos_r
    dx0 = det_ref[:, 0] * cos_r - det_ref[:, 1] * sin_r
    dy0 = det_ref[:, 0] * sin_r + det_ref[:, 1] * cos_r

    vx0 = dx0 - sx0
    vy0 = dy0 - sy0
    l_xy = np.hypot(vx0, vy0)  # (C,)
    ux0 = vx0 / l_xy
    uy0 = vy0 / l_xy

    # Horizontal ray angle beta0(c) and signed origin-to-line distance xi0(c)
    beta0 = np.unwrap(np.arctan2(ux0, -uy0))  # (C,)
    xi0 = sx0 * uy0 - sy0 * ux0  # (C,)
    s0 = -(sx0 * ux0 + sy0 * uy0)  # horizontal distance from source to chord midpoint (C,)

    # Ensure monotonic interpolation lookup for xi0 -> col index
    cols_grid = np.arange(geom.n_cols, dtype=np.float64)
    if xi0[-1] >= xi0[0]:
        xi_sorted = xi0
        cols_for_xi = cols_grid
    else:
        xi_sorted = xi0[::-1]
        cols_for_xi = cols_grid[::-1]

    # Test interior columns around the central 55% of the detector
    col_margin = max(4, int(0.22 * geom.n_cols))
    test_cols = np.arange(col_margin, geom.n_cols - col_margin, dtype=np.int64)
    xi_target = -xi0[test_cols]
    in_xi = (xi_target >= xi_sorted[0]) & (xi_target <= xi_sorted[-1])
    test_cols = test_cols[in_xi]
    if test_cols.size == 0:
        return float("nan"), 0

    c2_f = np.interp(xi_target[in_xi], xi_sorted, cols_for_xi)
    beta2 = np.interp(c2_f, cols_grid, beta0)
    s0_1 = s0[test_cols]
    s0_2 = np.interp(c2_f, cols_grid, s0)
    l_xy_1 = l_xy[test_cols]
    l_xy_2 = np.interp(c2_f, cols_grid, l_xy)

    phi_a = geom.phi0_unwrapped_rad[idx_a]
    phi_b = geom.phi0_unwrapped_rad[idx_b]
    rot_dir = 1.0 if geom.phi0_unwrapped_rad[-1] >= geom.phi0_unwrapped_rad[0] else -1.0

    # Prepare monotonic lookup on phi_b -> position in [0 .. len(idx_b)-1]
    b_pos_grid = np.arange(idx_b.size, dtype=np.float64)
    if phi_b[-1] >= phi_b[0]:
        phi_b_sorted = phi_b
        b_pos_sorted = b_pos_grid
    else:
        phi_b_sorted = phi_b[::-1]
        b_pos_sorted = b_pos_grid[::-1]

    row_sign = -1.0 if flip_rows else 1.0
    row_center_eff = (geom.n_rows - 1.0 - geom.center_row_0based) if flip_rows else geom.center_row_0based

    abs_diffs: list[np.ndarray] = []
    # Sample views in the interior middle half of block_a so conjugate angles fall inside block_b
    a_start = idx_a.size // 4
    a_stop = max(a_start + 1, (3 * idx_a.size) // 4)

    for ka in range(a_start, a_stop):
        i1 = int(idx_a[ka])
        phi_target = phi_a[ka] + rot_dir * np.pi + (beta0[test_cols] - beta2)
        valid_phi = (phi_target >= phi_b_sorted[0]) & (phi_target <= phi_b_sorted[-1])
        if not np.any(valid_phi):
            continue

        c1_sub = test_cols[valid_phi]
        c2_sub = c2_f[valid_phi]
        b_pos_f = np.interp(phi_target[valid_phi], phi_b_sorted, b_pos_sorted)
        kb0 = np.floor(b_pos_f).astype(np.int64)
        kb1 = np.minimum(kb0 + 1, idx_b.size - 1)
        wb = b_pos_f - kb0

        z_s1 = float(geom.z0_mm[i1] + geom.dz_mm[i1])
        z_01 = float(geom.z0_mm[i1])
        z_s2 = (1.0 - wb) * (geom.z0_mm[idx_b[kb0]] + geom.dz_mm[idx_b[kb0]]) + wb * (
            geom.z0_mm[idx_b[kb1]] + geom.dz_mm[idx_b[kb1]]
        )
        z_02 = (1.0 - wb) * geom.z0_mm[idx_b[kb0]] + wb * geom.z0_mm[idx_b[kb1]]

        # Exact 3D helical conjugate slope m along horizontal ray parameter u:
        m_slope = (z_s2 - z_s1) / (s0_1[valid_phi] + s0_2[valid_phi])
        v1_mm = (z_s1 - z_01) + m_slope * l_xy_1[valid_phi]
        v2_mm = (z_s2 - z_02) - m_slope * l_xy_2[valid_phi]

        # Map v(r) -> continuous 0-based row coordinate r:
        # True: v(r) = (r0 - r) * dv  => r = r0 - v / dv
        # Flipped: v_flip(r) = (r0 - (N-1-r)) * dv => r = (N-1-r0) + v / dv
        r1_f = row_center_eff - row_sign * (v1_mm / geom.dv_mm)
        r2_f = row_center_eff - row_sign * (v2_mm / geom.dv_mm)

        valid_r = (
            (r1_f >= 0.0)
            & (r1_f <= geom.n_rows - 1.0)
            & (r2_f >= 0.0)
            & (r2_f <= geom.n_rows - 1.0)
            & (c2_sub >= 0.0)
            & (c2_sub <= geom.n_cols - 1.0)
        )
        if not np.any(valid_r):
            continue

        c1_ok = c1_sub[valid_r].astype(np.float64)
        c2_ok = c2_sub[valid_r]
        r1_ok = r1_f[valid_r]
        r2_ok = r2_f[valid_r]
        kb0_ok = kb0[valid_r]
        kb1_ok = kb1[valid_r]
        wb_ok = wb[valid_r]

        g1 = _bilinear_sample_2d(block_a[ka], r1_ok, c1_ok)
        # Sample block_b at bracketing views kb0 and kb1
        g2 = np.empty_like(g1)
        for u_kb in np.unique(kb0_ok):
            mask_u = kb0_ok == u_kb
            kb_next = int(kb1_ok[mask_u][0])
            val0 = _bilinear_sample_2d(block_b[int(u_kb)], r2_ok[mask_u], c2_ok[mask_u])
            if kb_next == int(u_kb):
                g2[mask_u] = val0
            else:
                val1 = _bilinear_sample_2d(block_b[kb_next], r2_ok[mask_u], c2_ok[mask_u])
                g2[mask_u] = (1.0 - wb_ok[mask_u]) * val0 + wb_ok[mask_u] * val1

        abs_diffs.append(np.abs(g1 - g2))

    if not abs_diffs:
        return float("nan"), 0
    all_diffs = np.concatenate(abs_diffs)
    return float(np.mean(all_diffs)), int(all_diffs.size)


def verify_projection_orientation(
    series_dir_or_files: str | Path | list[Path],
    geometry: HelicalGeometry | None = None,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Run 3 quantitative physical proofs of PixelData layout, row sign, and column center.

    Proof 1: Fortran order (pixel_array.T) vs C-order axial smoothness ratio.
    Proof 2: Exact 3D Helical Half-Turn Conjugate-Ray consistency comparing:
             - True geometry (row 0 = +z, col -> +e_u, c0 = X_1based - 1.0)
             - Flipped axial rows (row 0 = -z)
             - Flipped transverse columns (col -> -e_u)
             - Unconverted 1-based central column bug (c0 + 1.0)
             - Mirrored central column ((N_cols - 1) - c0, when asymmetric)
    Proof 3: Direct Full-Turn (2*pi) isocenter axial overlap when normalized_pitch < 0.96
             and total_rotations >= 1.15.
    """
    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError("pydicom is required for verify_projection_orientation") from error

    if isinstance(series_dir_or_files, list):
        files = [Path(p) for p in series_dir_or_files]
        if geometry is None:
            raise ValueError("geometry must be provided when passing an explicit file list")
    else:
        geometry, files = scan_projection_series(series_dir_or_files, workers=max(workers, 4))

    # --- Proof 1: Fortran-order vs C-order PixelData layout smoothness ---
    sample_indices = np.linspace(0, geometry.n_views - 1, num=min(16, geometry.n_views), dtype=int)
    grad_row_f_list: list[float] = []
    grad_row_c_list: list[float] = []
    pydicom_t_exact = True
    val_min = float("inf")
    val_max = float("-inf")

    for idx in sample_indices:
        ds = pydicom.dcmread(files[int(idx)], force=False)
        raw_f = _extract_raw_matrix_fortran(
            ds, n_rows=geometry.n_rows, n_cols=geometry.n_cols, order="F"
        )
        raw_c = _extract_raw_matrix_fortran(
            ds, n_rows=geometry.n_rows, n_cols=geometry.n_cols, order="C"
        )
        if not np.array_equal(raw_f, ds.pixel_array.T):
            pydicom_t_exact = False

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        gf = raw_f.astype(np.float32) * slope + intercept
        gc = raw_c.astype(np.float32) * slope + intercept

        val_min = min(val_min, float(np.min(gf)))
        val_max = max(val_max, float(np.max(gf)))
        grad_row_f_list.append(float(np.mean(np.abs(np.diff(gf, axis=0)))))
        grad_row_c_list.append(float(np.mean(np.abs(np.diff(gc, axis=0)))))

    mean_grad_f = float(np.mean(grad_row_f_list))
    mean_grad_c = float(np.mean(grad_row_c_list))
    f_over_c_ratio = mean_grad_f / max(mean_grad_c, 1e-12)
    layout_pass = bool(pydicom_t_exact and f_over_c_ratio < 0.65)

    # --- Proof 2: Exact 3D Helical Half-Turn Conjugate-Ray Verification ---
    ffs_period = _detect_ffs_period(geometry)
    half_turn = geometry.angular_steps_per_rotation // 2
    # Align half_turn to a multiple of ffs_period so both blocks share identical FFS state
    half_turn_aligned = max(ffs_period, (half_turn // ffs_period) * ffs_period)

    conjugate_report: dict[str, Any] = {"available": False}
    axial_row_pass = True
    transverse_col_pass = True

    if geometry.n_views >= half_turn_aligned + 16 * ffs_period:
        # Choose up to 2 anchor windows across the middle of the scan
        block_same_state_len = min(
            64,
            max(16, (geometry.n_views - half_turn_aligned) // (2 * ffs_period)),
        )
        span_raw = block_same_state_len * ffs_period
        max_start = geometry.n_views - half_turn_aligned - span_raw
        anchor_starts = sorted(
            {
                int((max_start * frac) // ffs_period) * ffs_period
                for frac in (0.25, 0.65)
                if max_start >= 0
            }
        )
        if not anchor_starts:
            anchor_starts = [0]

        def _load_subset(indices: np.ndarray) -> np.ndarray:
            out = np.empty((indices.size, geometry.n_rows, geometry.n_cols), dtype=np.float32)
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                for k, arr in enumerate(
                    pool.map(
                        lambda p: read_single_projection(
                            p, expected_rows=geometry.n_rows, expected_cols=geometry.n_cols
                        ),
                        [files[int(i)] for i in indices],
                    )
                ):
                    out[k] = arr
            return out

        acc = {
            "true": [0.0, 0],
            "flip_rows": [0.0, 0],
            "flip_cols": [0.0, 0],
            "bug_1based_c0": [0.0, 0],
            "mirrored_c0": [0.0, 0],
        }
        mirrored_c0 = (geometry.n_cols - 1.0) - geometry.center_col_0based
        c0_is_asymmetric = abs(mirrored_c0 - geometry.center_col_0based) >= 0.25

        for a0 in anchor_starts:
            idx_a = np.arange(a0, a0 + span_raw, ffs_period, dtype=np.int64)
            idx_b = idx_a + half_turn_aligned
            idx_b = idx_b[idx_b < geometry.n_views]
            if idx_a.size < 8 or idx_b.size < 8:
                continue
            block_a = _load_subset(idx_a)
            block_b = _load_subset(idx_b)

            for key, kwargs in [
                ("true", {"flip_rows": False, "flip_cols": False, "center_col_override": None}),
                ("flip_rows", {"flip_rows": True, "flip_cols": False, "center_col_override": None}),
                ("flip_cols", {"flip_rows": False, "flip_cols": True, "center_col_override": None}),
                (
                    "bug_1based_c0",
                    {
                        "flip_rows": False,
                        "flip_cols": False,
                        "center_col_override": geometry.center_col_0based + 1.0,
                    },
                ),
            ]:
                mae_val, cnt = _evaluate_conjugate_hypothesis(
                    block_a, idx_a, block_b, idx_b, geometry, **kwargs
                )
                if cnt > 0 and not np.isnan(mae_val):
                    acc[key][0] += mae_val * cnt
                    acc[key][1] += cnt

            if c0_is_asymmetric:
                mae_m, cnt_m = _evaluate_conjugate_hypothesis(
                    block_a,
                    idx_a,
                    block_b,
                    idx_b,
                    geometry,
                    flip_rows=False,
                    flip_cols=False,
                    center_col_override=mirrored_c0,
                )
                if cnt_m > 0 and not np.isnan(mae_m):
                    acc["mirrored_c0"][0] += mae_m * cnt_m
                    acc["mirrored_c0"][1] += cnt_m

        if acc["true"][1] > 0:
            mae_true = acc["true"][0] / acc["true"][1]
            mae_flip_r = (
                acc["flip_rows"][0] / acc["flip_rows"][1] if acc["flip_rows"][1] > 0 else float("nan")
            )
            mae_flip_c = (
                acc["flip_cols"][0] / acc["flip_cols"][1] if acc["flip_cols"][1] > 0 else float("nan")
            )
            mae_1based = (
                acc["bug_1based_c0"][0] / acc["bug_1based_c0"][1]
                if acc["bug_1based_c0"][1] > 0
                else float("nan")
            )
            mae_mirror = (
                acc["mirrored_c0"][0] / acc["mirrored_c0"][1]
                if acc["mirrored_c0"][1] > 0
                else None
            )

            axial_row_pass = bool(not np.isnan(mae_flip_r) and mae_true < mae_flip_r)
            transverse_col_pass = bool(
                not np.isnan(mae_flip_c)
                and mae_true < mae_flip_c
                and (np.isnan(mae_1based) or mae_true < mae_1based)
                and (mae_mirror is None or mae_true < mae_mirror)
            )
            conjugate_report = {
                "available": True,
                "ffs_period_detected": ffs_period,
                "evaluated_ray_pairs": int(acc["true"][1]),
                "mae_true_geometry": round(mae_true, 6),
                "mae_flipped_axial_rows": round(mae_flip_r, 6),
                "mae_flipped_detector_cols": round(mae_flip_c, 6),
                "mae_unconverted_1based_c0": round(mae_1based, 6),
                "mae_mirrored_c0": round(mae_mirror, 6) if mae_mirror is not None else None,
                "axial_row_sign_pass": axial_row_pass,
                "transverse_col_and_c0_pass": transverse_col_pass,
            }

    # --- Proof 3: Direct Full-Turn (2*pi) Isocenter Axial Overlap (when pitch < 0.96) ---
    traj = geometry.trajectory_metrics()
    full_turn_report: dict[str, Any] = {"available": False}
    n_rot = geometry.angular_steps_per_rotation
    n_rot_aligned = max(ffs_period, (n_rot // ffs_period) * ffs_period)

    if (
        float(traj["normalized_pitch"]) < 0.96
        and geometry.n_views >= n_rot_aligned + 16
        and float(traj["normalized_pitch"]) > 0.05
    ):
        sample_v1 = np.linspace(
            0, geometry.n_views - n_rot_aligned - 1, num=min(12, geometry.n_views - n_rot_aligned), dtype=int
        )
        c_mid = int(round(geometry.center_col_0based))
        col_sl = slice(max(0, c_mid - 8), min(geometry.n_cols, c_mid + 9))
        diffs_true_ft: list[float] = []
        diffs_flip_ft: list[float] = []

        for v1 in sample_v1:
            v2 = int(v1) + n_rot_aligned
            dz_turn = float(geometry.z0_mm[v2] - geometry.z0_mm[v1])
            dr_shift = (geometry.d0_mm / geometry.rho0_mm) * dz_turn / geometry.dv_mm

            p1 = read_single_projection(
                files[int(v1)], expected_rows=geometry.n_rows, expected_cols=geometry.n_cols
            )[:, col_sl]
            p2 = read_single_projection(
                files[v2], expected_rows=geometry.n_rows, expected_cols=geometry.n_cols
            )[:, col_sl]

            # Under TRUE row 0 = +z: r2 = r1 + dr_shift
            r1_grid = np.arange(geometry.n_rows, dtype=np.float64)
            r2_true = r1_grid + dr_shift
            mask_t = (r2_true >= 0.0) & (r2_true <= geometry.n_rows - 1.0)
            if np.any(mask_t):
                r2_idx = r2_true[mask_t]
                r2_0 = np.floor(r2_idx).astype(np.int64)
                r2_1 = np.minimum(r2_0 + 1, geometry.n_rows - 1)
                w = (r2_idx - r2_0)[:, None]
                p2_interp = (1.0 - w) * p2[r2_0] + w * p2[r2_1]
                diffs_true_ft.append(float(np.mean(np.abs(p1[mask_t] - p2_interp))))

            # Under FLIPPED row 0 = -z: r2 = r1 - dr_shift
            r2_flip = r1_grid - dr_shift
            mask_f = (r2_flip >= 0.0) & (r2_flip <= geometry.n_rows - 1.0)
            if np.any(mask_f):
                r2_idx = r2_flip[mask_f]
                r2_0 = np.floor(r2_idx).astype(np.int64)
                r2_1 = np.minimum(r2_0 + 1, geometry.n_rows - 1)
                w = (r2_idx - r2_0)[:, None]
                p2_interp = (1.0 - w) * p2[r2_0] + w * p2[r2_1]
                diffs_flip_ft.append(float(np.mean(np.abs(p1[mask_f] - p2_interp))))

        if diffs_true_ft and diffs_flip_ft:
            ft_true = float(np.mean(diffs_true_ft))
            ft_flip = float(np.mean(diffs_flip_ft))
            ft_pass = bool(ft_true < ft_flip)
            axial_row_pass = bool(axial_row_pass and ft_pass)
            full_turn_report = {
                "available": True,
                "sampled_turn_pairs": int(len(diffs_true_ft)),
                "mae_true_row0_plus_z": round(ft_true, 6),
                "mae_flipped_row0_minus_z": round(ft_flip, 6),
                "pass": ft_pass,
            }

    overall_pass = bool(layout_pass and axial_row_pass and transverse_col_pass)
    return {
        "overall_pass": overall_pass,
        "value_range_line_integral": [round(val_min, 6), round(val_max, 6)],
        "log_flag_respected": geometry.log_flag == "YES",
        "layout_proof": {
            "pydicom_pixel_array_T_matches_fortran": pydicom_t_exact,
            "mean_axial_diff_fortran": round(mean_grad_f, 6),
            "mean_axial_diff_c_order": round(mean_grad_c, 6),
            "fortran_over_c_ratio": round(f_over_c_ratio, 6),
            "pass": layout_pass,
        },
        "helical_conjugate_ray_proof": conjugate_report,
        "full_turn_axial_overlap_proof": full_turn_report,
    }


def read_ct_image_volume(
    series_dir: str | Path,
    *,
    series_uid: str | None = None,
    water_mu_mm_inv: float = 0.0192,
) -> dict[str, Any]:
    """Read a published CT image series into z-sorted HU and mu [mm^-1] 3D volumes.

    NOTE: Published vendor reconstructions in TCIA have a different FrameOfReferenceUID
    from raw DICOM-CT-PD projections and use vendor kernels (e.g., B50f). Use ONLY for
    visual sanity / axial span comparison, NEVER as pixel-wise supervised Ground Truth.
    """
    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError("pydicom is required to read CT image volumes") from error

    sdir = Path(series_dir).expanduser().resolve()
    candidates = _collect_candidates(sdir)
    if not candidates:
        raise ValueError(f"No DICOM files found in {sdir}")

    slices: list[tuple[float, int, Any]] = []
    found_uids: set[str] = set()
    for _, path, _ in candidates:
        try:
            ds = pydicom.dcmread(path, force=False)
        except Exception:
            continue
        if classify_dicom_sop(ds) != "ct_image":
            continue
        suid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        if series_uid is not None and suid != series_uid:
            continue
        found_uids.add(suid)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None or len(ipp) < 3:
            raise ValueError(f"CT slice {path} is missing ImagePositionPatient")
        z_val = float(ipp[2])
        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
        slices.append((z_val, inst, ds))

    if not slices:
        raise ValueError(f"No CT Image Storage slices found in {sdir}")
    if len(found_uids) > 1:
        raise ValueError(
            f"Multiple CT image SeriesInstanceUIDs found under {sdir}: {sorted(found_uids)}"
        )

    # Sort strictly in ascending patient z [mm]
    slices.sort(key=lambda item: (item[0], item[1]))
    ds0 = slices[0][2]
    rows = int(ds0.Rows)
    cols = int(ds0.Columns)

    hu_vol = np.empty((len(slices), rows, cols), dtype=np.float32)
    z_coords = np.empty(len(slices), dtype=np.float64)
    for k, (z_val, _, ds) in enumerate(slices):
        slope = np.float32(float(getattr(ds, "RescaleSlope", 1.0)))
        intercept = np.float32(float(getattr(ds, "RescaleIntercept", -1024.0)))
        hu_vol[k] = ds.pixel_array.astype(np.float32) * slope + intercept
        z_coords[k] = z_val

    mu_vol = hu_to_mu(hu_vol, water_mu_mm_inv=water_mu_mm_inv, clip_non_negative=True)
    ps = [float(x) for x in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
    dz_diffs = np.diff(z_coords) if z_coords.size > 1 else np.array([0.0])

    return {
        "series_instance_uid": next(iter(found_uids)),
        "frame_of_reference_uid": str(getattr(ds0, "FrameOfReferenceUID", "") or ""),
        "referenced_raw_series_uid": extract_referenced_raw_series_uid(ds0),
        "convolution_kernel": str(getattr(ds0, "ConvolutionKernel", "") or ""),
        "shape_zyx": tuple(int(x) for x in hu_vol.shape),
        "pixel_spacing_yx_mm": (ps[0], ps[1]),
        "slice_thickness_mm": float(getattr(ds0, "SliceThickness", 0.0) or 0.0),
        "median_dz_mm": float(np.median(dz_diffs)),
        "image_position_xy_mm": (
            float(ds0.ImagePositionPatient[0]),
            float(ds0.ImagePositionPatient[1]),
        ),
        "z_mm": z_coords,
        "hu_volume": hu_vol,
        "mu_volume": mu_vol,
        "role_warning": "vendor_reconstruction_visual_sanity_only_not_training_gt",
    }
