"""Mandatory Visual Quality Assurance (Visual QA) multi-panel PNG generators.

Generates side-by-side diagnostic and ground-truth comparison PNG figures for every stage
producing visualizable physical quantities (projections, detector layouts, curved-to-flat maps,
SSRB sinograms, missing-data masks, and CT slices).

Uses headless `matplotlib` (`FigureCanvasAgg`) when installed for publication-grade labeled
figures, with a deterministic pure-stdlib (`zlib` + `struct`) RGB PNG fallback so automated
tests in minimal `numpy`-only environments still produce genuine, valid `.png` images.
"""

from __future__ import annotations

from pathlib import Path
import struct
import zlib

import numpy as np


def _normalize_to_uint8(
    arr: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    data = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(data)) if vmin is None else float(vmin)
    hi = float(np.max(data)) if vmax is None else float(vmax)
    if hi <= lo + 1e-12:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _resize_nearest_rgb(rgb: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w, _ = rgb.shape
    r_idx = np.clip((np.arange(out_h) * in_h) // max(1, out_h), 0, in_h - 1)
    c_idx = np.clip((np.arange(out_w) * in_w) // max(1, out_w), 0, in_w - 1)
    return rgb[r_idx[:, None], c_idx[None, :], :]


def _panel_to_rgb(
    arr: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    missing_mask: np.ndarray | None = None,
    heat: bool = False,
) -> np.ndarray:
    u8 = _normalize_to_uint8(arr, vmin=vmin, vmax=vmax)
    if heat:
        # Simple dark-blue -> orange -> yellow heat ramp for error maps
        t = u8.astype(np.float64) / 255.0
        r = np.rint(np.clip(2.0 * t, 0.0, 1.0) * 255.0).astype(np.uint8)
        g = np.rint(np.clip(2.0 * t - 0.4, 0.0, 1.0) * 255.0).astype(np.uint8)
        b = np.rint(np.clip(0.35 - 0.35 * t, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
    else:
        rgb = np.stack([u8, u8, u8], axis=-1)

    if missing_mask is not None and missing_mask.shape == arr.shape:
        miss = np.asarray(missing_mask, dtype=bool)
        rgb[miss, 0] = 225
        rgb[miss, 1] = 42
        rgb[miss, 2] = 42
    return rgb


def _write_rgb_png_stdlib(path: Path, rgb: np.ndarray) -> Path:
    """Write an 8-bit RGB (H, W, 3) array as a valid standard PNG using only struct + zlib."""
    img = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = img.shape
    if channels != 3:
        raise ValueError(f"Expected (H, W, 3) RGB array, got {img.shape}")

    # Prepend filter byte 0 (None) to each scanline
    scanlines = np.empty((height, 1 + width * 3), dtype=np.uint8)
    scanlines[:, 0] = 0
    scanlines[:, 1:] = img.reshape(height, width * 3)
    compressed = zlib.compress(scanlines.tobytes(), level=6)

    def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(chunk_type)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)
    return path


def _compose_grid_png_stdlib(
    out_path: Path,
    panels_rgb: list[np.ndarray],
    *,
    ncols: int,
    tile_h: int = 240,
    tile_w: int = 320,
    pad: int = 8,
) -> Path:
    n_panels = len(panels_rgb)
    nrows = max(1, (n_panels + ncols - 1) // ncols)
    canvas_h = nrows * tile_h + (nrows + 1) * pad
    canvas_w = ncols * tile_w + (ncols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    for idx, panel in enumerate(panels_rgb):
        r_grid = idx // ncols
        c_grid = idx % ncols
        y0 = pad + r_grid * (tile_h + pad)
        x0 = pad + c_grid * (tile_w + pad)
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = _resize_nearest_rgb(panel, tile_h, tile_w)

    return _write_rgb_png_stdlib(out_path, canvas)


def _has_matplotlib() -> bool:
    import os
    import tempfile
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pyparsing.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib.*")

    if "MPLCONFIGDIR" not in os.environ:
        default_cfg = Path.home() / ".config" / "matplotlib"
        if not os.access(default_cfg if default_cfg.exists() else Path.home(), os.W_OK):
            mpl_tmp = Path(tempfile.gettempdir()) / "highpitchct-mplconfig"
            mpl_tmp.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(mpl_tmp)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import matplotlib  # noqa: F401
            from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
            from matplotlib.figure import Figure  # noqa: F401

        return True
    except ImportError:
        return False


def _overlay_missing_rgba(
    gray_2d: np.ndarray,
    measured_mask: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Create RGB float [0, 1] image with unmeasured (`~measured_mask`) regions tinted bright red."""
    rgb = _panel_to_rgb(
        gray_2d,
        vmin=vmin,
        vmax=vmax,
        missing_mask=~np.asarray(measured_mask, dtype=bool),
        heat=False,
    )
    return rgb.astype(np.float32) / 255.0


def save_stage1_phantom_visual_qa(
    out_path: str | Path,
    *,
    hu_volume: np.ndarray,
    projections: np.ndarray,
) -> Path:
    """Save 4-panel Stage 1 Synthetic 3D Thorax Phantom visual QA PNG."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    nz, ny, _ = hu_volume.shape
    nv, nr, _ = projections.shape
    axial_hu = hu_volume[nz // 2][::-1, :]  # Radiological supine: top = +y Anterior
    coronal_hu = hu_volume[:, ny // 2, :][::-1, :]  # +z at top
    mid_proj = projections[nv // 2]
    mid_sino = projections[:, nr // 2, :]

    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(axial_hu, vmin=-1000.0, vmax=600.0),
                _panel_to_rgb(coronal_hu, vmin=-1000.0, vmax=600.0),
                _panel_to_rgb(mid_proj),
                _panel_to_rgb(mid_sino),
            ],
            ncols=2,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13.5, 8.5), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        "Stage 1 Visual QA — 3D Asymmetric Thorax Phantom & Analytical Helical CT-PD Projections",
        fontsize=12,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(axial_hu, cmap="gray", vmin=-1000.0, vmax=600.0, aspect="equal")
    ax1.set_title("Ground-Truth Axial Slice (z = mid, [-1000, 600] HU)")
    ax1.set_xlabel("x index (+x = Patient Left)")
    ax1.set_ylabel("y index (top = +y Anterior)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(coronal_hu, cmap="gray", vmin=-1000.0, vmax=600.0, aspect="auto")
    ax2.set_title("Ground-Truth Coronal Slice (+z Cranial at Top)")
    ax2.set_xlabel("x index")
    ax2.set_ylabel("z index (top = +z)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(mid_proj, cmap="gray", aspect="auto")
    ax3.set_title(f"Helical Cone-Beam View #{nv // 2} (Row 0 = +z, order='F')")
    ax3.set_xlabel("Detector Column c")
    ax3.set_ylabel("Detector Row r")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(mid_sino, cmap="gray", aspect="auto")
    ax4.set_title(f"Central-Row Helical Sinogram (Row {nr // 2}, FFSZ)")
    ax4.set_xlabel("Detector Column c")
    ax4.set_ylabel("Helical View Index i")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage1_real_visual_qa(
    out_path: str | Path,
    *,
    case_id: str,
    proj_fortran: np.ndarray,
    proj_c_order_corrupted: np.ndarray,
    scout_sinogram: np.ndarray,
    vendor_mid_slice_hu: np.ndarray | None = None,
    vendor_mid_z_mm: float | None = None,
) -> Path:
    """Save 4-panel Stage 1 Real Patient (e.g. C012) DICOM-CT-PD layout & orientation QA PNG."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    p4 = (
        _panel_to_rgb(vendor_mid_slice_hu, vmin=-1000.0, vmax=400.0)
        if vendor_mid_slice_hu is not None
        else _panel_to_rgb(np.abs(np.diff(proj_fortran, axis=0)), heat=True)
    )
    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(proj_fortran),
                _panel_to_rgb(proj_c_order_corrupted),
                _panel_to_rgb(scout_sinogram),
                p4,
            ],
            ncols=2,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14.0, 8.5), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"Stage 1 Visual QA ({case_id}) — Fortran vs C-Order Layout & Vendor Reference Sanity",
        fontsize=12,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(proj_fortran, cmap="gray", aspect="auto")
    ax1.set_title("TRUE Fortran Layout (pixel_array.T) — Smooth Physical Anatomy")
    ax1.set_xlabel("Detector Column c (0..N_cols-1)")
    ax1.set_ylabel("Detector Row r (0 = +z)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(proj_c_order_corrupted, cmap="gray", aspect="auto")
    ax2.set_title("WRONG C-Order Reshape — Scrambled Detector Stripes")
    ax2.set_xlabel("Scrambled Column")
    ax2.set_ylabel("Scrambled Row")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(scout_sinogram, cmap="gray", aspect="auto")
    ax3.set_title("Sampled Helical Central-Row Sinogram Across Scan")
    ax3.set_xlabel("Detector Column c")
    ax3.set_ylabel("Sampled View Index")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 2, 4)
    if vendor_mid_slice_hu is not None:
        im4 = ax4.imshow(
            vendor_mid_slice_hu, cmap="gray", vmin=-1000.0, vmax=400.0, aspect="equal"
        )
        z_txt = f"z = {vendor_mid_z_mm:.1f} mm" if vendor_mid_z_mm is not None else "mid-scan"
        ax4.set_title(f"Published Vendor CT Slice ({z_txt}, Visual Sanity Only)")
        ax4.set_xlabel("x pixel")
        ax4.set_ylabel("y pixel")
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    else:
        grad_f = np.abs(np.diff(proj_fortran, axis=0))
        im4 = ax4.imshow(grad_f, cmap="magma", aspect="auto")
        ax4.set_title("Axial Row-to-Row |diff| under True Fortran Layout")
        ax4.set_xlabel("Detector Column c")
        ax4.set_ylabel("Row Pair")
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage2_phantom_ssrb_visual_qa(
    out_path: str | Path,
    *,
    exact_planar_sino: np.ndarray,
    ssrb_low_pitch_sino: np.ndarray,
    ssrb_low_pitch_mask: np.ndarray,
    ssrb_high_pitch_sino: np.ndarray,
    ssrb_high_pitch_mask: np.ndarray,
    z_slice_mm: float,
    low_pitch: float,
    high_pitch: float,
) -> Path:
    """Save 4-panel Stage 2 Synthetic Ground-Truth Planar vs Helical FFSZ SSRB visual QA PNG."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    meas_low = np.asarray(ssrb_low_pitch_mask, dtype=bool)
    abs_err = np.where(meas_low, np.abs(ssrb_low_pitch_sino - exact_planar_sino), 0.0)
    if np.any(meas_low):
        mae = float(np.mean(abs_err[meas_low]))
        rmse = float(np.sqrt(np.mean(abs_err[meas_low] ** 2)))
    else:
        mae = 0.0
        rmse = 0.0
    vmax_sino = max(float(np.max(exact_planar_sino)), 1e-3)
    meas_low_pct = 100.0 * float(np.mean(meas_low))
    meas_high_pct = 100.0 * float(np.mean(ssrb_high_pitch_mask))

    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(exact_planar_sino, vmin=0.0, vmax=vmax_sino),
                _panel_to_rgb(
                    ssrb_low_pitch_sino,
                    vmin=0.0,
                    vmax=vmax_sino,
                    missing_mask=~ssrb_low_pitch_mask,
                ),
                _panel_to_rgb(abs_err, vmin=0.0, vmax=max(float(np.max(abs_err)), 1e-4), heat=True),
                _panel_to_rgb(
                    ssrb_high_pitch_sino,
                    vmin=0.0,
                    vmax=vmax_sino,
                    missing_mask=~ssrb_high_pitch_mask,
                ),
            ],
            ncols=2,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14.0, 9.0), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"Stage 2 Visual QA — Exact 2D Planar GT vs 3D Helical FFSZ SSRB (z = {z_slice_mm:.1f} mm)",
        fontsize=12,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(exact_planar_sino, cmap="gray", vmin=0.0, vmax=vmax_sino, aspect="auto")
    ax1.set_title("A. Exact 2D Planar Fan-Beam Ground Truth")
    ax1.set_xlabel("Virtual Flat Column u")
    ax1.set_ylabel("Short-Scan Angle Index k")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 2, 2)
    rgb_low = _overlay_missing_rgba(
        ssrb_low_pitch_sino, ssrb_low_pitch_mask, vmin=0.0, vmax=vmax_sino
    )
    ax2.imshow(rgb_low, aspect="auto")
    ax2.set_title(
        f"B. Helical FFSZ Curved->Flat + SSRB (pitch={low_pitch:.2f}, measured={meas_low_pct:.1f}%)"
    )
    ax2.set_xlabel("Virtual Flat Column u")
    ax2.set_ylabel("Short-Scan Angle Index k")

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(abs_err, cmap="magma", aspect="auto")
    ax3.set_title(f"C. |SSRB - Exact 2D GT| (MAE={mae:.5f}, RMSE={rmse:.5f})")
    ax3.set_xlabel("Virtual Flat Column u")
    ax3.set_ylabel("Short-Scan Angle Index k")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 2, 4)
    rgb_high = _overlay_missing_rgba(
        ssrb_high_pitch_sino, ssrb_high_pitch_mask, vmin=0.0, vmax=vmax_sino
    )
    ax4.imshow(rgb_high, aspect="auto")
    ax4.set_title(
        f"D. High-Pitch={high_pitch:.2f} SSRB + Red Unmeasured Wedges (measured={meas_high_pct:.1f}%)"
    )
    ax4.set_xlabel("Virtual Flat Column u (Red = Missing Data, Zero Smearing)")
    ax4.set_ylabel("Short-Scan Angle Index k")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage2_real_ssrb_visual_qa(
    out_path: str | Path,
    *,
    case_id: str,
    raw_curved_view: np.ndarray,
    rebinned_flat_view: np.ndarray,
    rebinned_flat_valid: np.ndarray,
    z_slice_mm: float,
    native_pitch: float,
    short_scan_sino: np.ndarray,
    short_scan_mask: np.ndarray,
    full_scan_sino: np.ndarray,
    full_scan_mask: np.ndarray,
    rdc_high_pitch_sino: np.ndarray,
    rdc_high_pitch_mask: np.ndarray,
    rdc_effective_pitch: float,
    vendor_slice_hu: np.ndarray | None = None,
) -> Path:
    """Save 6-panel (2x3) Stage 2 Real Patient Curved->Flat & SSRB Visual QA PNG."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    vmax_proj = max(float(np.max(raw_curved_view)), 1e-3)
    vmax_sino = max(float(np.max(short_scan_sino)), 1e-3)
    flat_valid_bool = np.asarray(rebinned_flat_valid, dtype=bool)
    flat_diff = np.where(flat_valid_bool, np.abs(rebinned_flat_view - raw_curved_view), 0.0)

    p3_rgb = (
        _panel_to_rgb(vendor_slice_hu, vmin=-1000.0, vmax=400.0)
        if vendor_slice_hu is not None
        else _panel_to_rgb(flat_diff, heat=True)
    )
    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(raw_curved_view, vmin=0.0, vmax=vmax_proj),
                _panel_to_rgb(
                    rebinned_flat_view,
                    vmin=0.0,
                    vmax=vmax_proj,
                    missing_mask=~rebinned_flat_valid,
                ),
                p3_rgb,
                _panel_to_rgb(
                    short_scan_sino,
                    vmin=0.0,
                    vmax=vmax_sino,
                    missing_mask=~short_scan_mask,
                ),
                _panel_to_rgb(
                    full_scan_sino,
                    vmin=0.0,
                    vmax=vmax_sino,
                    missing_mask=~full_scan_mask,
                ),
                _panel_to_rgb(
                    rdc_high_pitch_sino,
                    vmin=0.0,
                    vmax=vmax_sino,
                    missing_mask=~rdc_high_pitch_mask,
                ),
            ],
            ncols=3,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(17.0, 9.5), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"Stage 2 Real-Data Visual QA ({case_id}) — Curved-to-Flat & Helical FFSZ SSRB at z = {z_slice_mm:.1f} mm",
        fontsize=13,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(raw_curved_view, cmap="gray", vmin=0.0, vmax=vmax_proj, aspect="auto")
    ax1.set_title("1. Raw Cylindrical View (64 x 736)")
    ax1.set_xlabel("Physical Column c")
    ax1.set_ylabel("Row r (0 = +z)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    flat_mae = float(np.mean(flat_diff[rebinned_flat_valid])) if np.any(rebinned_flat_valid) else 0.0
    im2 = ax2.imshow(flat_diff, cmap="magma", aspect="auto")
    ax2.set_title(f"2. |Flat - Curved| Shift Map (mean={flat_mae:.4f})")
    ax2.set_xlabel("Virtual Flat Column u")
    ax2.set_ylabel("Virtual Flat Row v")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    if vendor_slice_hu is not None:
        im3 = ax3.imshow(vendor_slice_hu, cmap="gray", vmin=-1000.0, vmax=400.0, aspect="equal")
        ax3.set_title(f"3. Vendor CT Slice near z={z_slice_mm:.1f} mm (Anatomy Ref)")
        ax3.set_xlabel("x pixel")
        ax3.set_ylabel("y pixel")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    else:
        rgb_fv = _overlay_missing_rgba(
            rebinned_flat_view, rebinned_flat_valid, vmin=0.0, vmax=vmax_proj
        )
        ax3.imshow(rgb_fv, aspect="auto")
        ax3.set_title(
            f"3. Rebinned Flat View (valid={100.0 * float(np.mean(rebinned_flat_valid)):.1f}%)"
        )
        ax3.set_xlabel("Virtual Flat Column u")
        ax3.set_ylabel("Virtual Flat Row v")

    ax4 = fig.add_subplot(2, 3, 4)
    ss_pct = 100.0 * float(np.mean(short_scan_mask))
    ax4.imshow(
        _overlay_missing_rgba(short_scan_sino, short_scan_mask, vmin=0.0, vmax=vmax_sino),
        aspect="auto",
    )
    ax4.set_title(f"4. Short-Scan SSRB (p={native_pitch:.2f}, measured={ss_pct:.2f}%)")
    ax4.set_xlabel("Virtual Flat Column u")
    ax4.set_ylabel("Short-Scan Angle k (pi + 2*gamma_max)")

    ax5 = fig.add_subplot(2, 3, 5)
    fs_pct = 100.0 * float(np.mean(full_scan_mask))
    ax5.imshow(
        _overlay_missing_rgba(full_scan_sino, full_scan_mask, vmin=0.0, vmax=vmax_sino),
        aspect="auto",
    )
    ax5.set_title(f"5. Full-Scan 2*pi SSRB (p={native_pitch:.2f}, measured={fs_pct:.2f}%)")
    ax5.set_xlabel("Virtual Flat Column u (Red = Missing)")
    ax5.set_ylabel("Full-Scan Angle k (360 deg)")

    ax6 = fig.add_subplot(2, 3, 6)
    rdc_pct = 100.0 * float(np.mean(rdc_high_pitch_mask))
    ax6.imshow(
        _overlay_missing_rgba(rdc_high_pitch_sino, rdc_high_pitch_mask, vmin=0.0, vmax=vmax_sino),
        aspect="auto",
    )
    ax6.set_title(
        f"6. High-Pitch RDC (32 rows, eff. p={rdc_effective_pitch:.2f}, measured={rdc_pct:.1f}%)"
    )
    ax6.set_xlabel("Virtual Flat Column u (Red = True High-Pitch Gaps)")
    ax6.set_ylabel("Short-Scan Angle k")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage3_phantom_fbp_visual_qa(
    out_path: str | Path,
    *,
    gt_hu: np.ndarray,
    ref_fbp_hu: np.ndarray,
    high_pitch_raw_fbp_hu: np.ndarray,
    high_pitch_inpaint_fbp_hu: np.ndarray,
    fov_mask: np.ndarray,
    fov_mm: float,
    z_slice_mm: float,
    low_pitch: float,
    high_pitch: float,
    ref_metrics: dict[str, float],
    hp_raw_metrics: dict[str, float],
    hp_inpaint_metrics: dict[str, float],
) -> Path:
    """Save 6-panel (2x3) Stage 3 Synthetic Phantom Ground Truth vs Parker+FBP Visual QA PNG."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    fmask = np.asarray(fov_mask, dtype=bool)
    err_ref = np.where(fmask, np.abs(ref_fbp_hu - gt_hu), 0.0)
    err_hp = np.where(fmask, np.abs(high_pitch_inpaint_fbp_hu - gt_hu), 0.0)

    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(gt_hu, vmin=-1000.0, vmax=600.0),
                _panel_to_rgb(ref_fbp_hu, vmin=-1000.0, vmax=600.0),
                _panel_to_rgb(err_ref, vmin=0.0, vmax=150.0, heat=True),
                _panel_to_rgb(high_pitch_raw_fbp_hu, vmin=-1000.0, vmax=600.0),
                _panel_to_rgb(high_pitch_inpaint_fbp_hu, vmin=-1000.0, vmax=600.0),
                _panel_to_rgb(err_hp, vmin=0.0, vmax=150.0, heat=True),
            ],
            ncols=3,
            tile_h=280,
            tile_w=280,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    half_fov = 0.5 * float(fov_mm)
    extent = [-half_fov, half_fov, -half_fov, half_fov]

    fig = Figure(figsize=(17.0, 10.5), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"Stage 3 Visual QA — Exact 3D Phantom GT vs Helical FFSZ SSRB + Parker + FBP (z = {z_slice_mm:.1f} mm)",
        fontsize=13,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(gt_hu, cmap="gray", vmin=-1000.0, vmax=600.0, extent=extent, aspect="equal")
    ax1.set_title("A. Exact 3D Phantom Ground Truth [-1000, 600] HU")
    ax1.set_xlabel("x [mm] (Radiological: Left=Pat.Right, Right=Pat.Left)")
    ax1.set_ylabel("y [mm] (+y = Anterior Sternum)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(
        ref_fbp_hu, cmap="gray", vmin=-1000.0, vmax=600.0, extent=extent, aspect="equal"
    )
    ax2.set_title(
        f"B. Reference FBP (p={low_pitch:.2f}, RMSE={ref_metrics['rmse_hu']:.1f} HU, SSIM={ref_metrics['ssim']:.3f})"
    )
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(err_ref, cmap="magma", vmin=0.0, vmax=150.0, extent=extent, aspect="equal")
    ax3.set_title(
        f"C. |Ref FBP - GT| [HU] (MAE={ref_metrics['mae_hu']:.1f} HU, PSNR={ref_metrics['psnr_db']:.1f} dB)"
    )
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(
        high_pitch_raw_fbp_hu, cmap="gray", vmin=-1000.0, vmax=600.0, extent=extent, aspect="equal"
    )
    ax4.set_title(
        f"D. High-Pitch={high_pitch:.2f} FBP WITHOUT Wedge Fill (RMSE={hp_raw_metrics['rmse_hu']:.1f} HU)"
    )
    ax4.set_xlabel("x [mm] (Severe Missing-Wedge Streaks)")
    ax4.set_ylabel("y [mm]")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(
        high_pitch_inpaint_fbp_hu,
        cmap="gray",
        vmin=-1000.0,
        vmax=600.0,
        extent=extent,
        aspect="equal",
    )
    ax5.set_title(
        f"E. High-Pitch={high_pitch:.2f} FBP + Conjugate/Feather Fill (RMSE={hp_inpaint_metrics['rmse_hu']:.1f} HU)"
    )
    ax5.set_xlabel("x [mm] (Streaks Suppressed; Cone-Beam Residual Remains)")
    ax5.set_ylabel("y [mm]")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    mid_idx = gt_hu.shape[0] // 2
    x_axis = np.linspace(-half_fov, half_fov, gt_hu.shape[1])
    ax6.plot(x_axis, gt_hu[mid_idx, :], label="Exact GT (y=0)", color="#111111", linewidth=1.8)
    ax6.plot(
        x_axis,
        ref_fbp_hu[mid_idx, :],
        label=f"Ref FBP (p={low_pitch:.2f})",
        color="#0077bb",
        linewidth=1.3,
    )
    ax6.plot(
        x_axis,
        high_pitch_inpaint_fbp_hu[mid_idx, :],
        label=f"HP FBP + Fill (p={high_pitch:.2f})",
        color="#ee7733",
        linewidth=1.1,
        alpha=0.9,
    )
    ax6.set_ylim(-1080.0, 900.0)
    ax6.grid(True, linestyle="--", alpha=0.35)
    ax6.set_title("F. Central Horizontal HU Profile (y = 0 mm)")
    ax6.set_xlabel("x [mm]")
    ax6.set_ylabel("Attenuation [HU]")
    ax6.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage3_real_fbp_visual_qa(
    out_path: str | Path,
    *,
    case_id: str,
    z_slice_mm: float,
    native_pitch: float,
    ref_fbp_hu: np.ndarray,
    fov_mask: np.ndarray,
    fov_mm: float,
    filter_name: str,
    rdc_effective_pitch: float,
    rdc_raw_fbp_hu: np.ndarray,
    rdc_inpaint_fbp_hu: np.ndarray,
    rdc_vs_ref_metrics: dict[str, float],
    rdc_raw_vs_ref_metrics: dict[str, float],
    vendor_slice_hu: np.ndarray | None = None,
    vendor_fov_mm: float | None = None,
    vendor_kernel: str | None = None,
) -> Path:
    """Save 6-panel (2x3) Stage 3 Real Patient `x_reference` vs Vendor CT & High-Pitch RDC FBP PNG."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    fmask = np.asarray(fov_mask, dtype=bool)
    hp_residual = np.where(fmask, np.abs(rdc_inpaint_fbp_hu - ref_fbp_hu), 0.0)
    p3_rgb = (
        _panel_to_rgb(vendor_slice_hu, vmin=-1000.0, vmax=400.0)
        if vendor_slice_hu is not None
        else _panel_to_rgb(ref_fbp_hu, vmin=-400.0, vmax=1200.0)
    )

    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(ref_fbp_hu, vmin=-1000.0, vmax=400.0),
                _panel_to_rgb(ref_fbp_hu, vmin=-160.0, vmax=240.0),
                p3_rgb,
                _panel_to_rgb(rdc_raw_fbp_hu, vmin=-1000.0, vmax=400.0),
                _panel_to_rgb(rdc_inpaint_fbp_hu, vmin=-1000.0, vmax=400.0),
                _panel_to_rgb(hp_residual, vmin=0.0, vmax=180.0, heat=True),
            ],
            ncols=3,
            tile_h=280,
            tile_w=280,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    half_fov = 0.5 * float(fov_mm)
    extent = [-half_fov, half_fov, -half_fov, half_fov]

    fig = Figure(figsize=(17.5, 10.5), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"Stage 3 Real-Data Visual QA ({case_id}) — Our Reference FBP (`x_reference`, {filter_name}) "
        f"vs Vendor CT & High-Pitch RDC at z = {z_slice_mm:.1f} mm",
        fontsize=12.5,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(ref_fbp_hu, cmap="gray", vmin=-1000.0, vmax=400.0, extent=extent, aspect="equal")
    ax1.set_title(f"1. Our x_reference FBP (p={native_pitch:.2f}, Lung/Chest [-1000, 400] HU)")
    ax1.set_xlabel("x [mm] (Radiological: Left=Pat.Right, Right=Pat.Left)")
    ax1.set_ylabel("y [mm] (+y = Anterior)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(ref_fbp_hu, cmap="gray", vmin=-160.0, vmax=240.0, extent=extent, aspect="equal")
    ax2.set_title("2. Our x_reference FBP (Soft-Tissue Window [-160, 240] HU)")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    if vendor_slice_hu is not None:
        vfov_half = 0.5 * float(vendor_fov_mm if vendor_fov_mm is not None else fov_mm)
        v_ext = [-vfov_half, vfov_half, -vfov_half, vfov_half]
        ax3.set_facecolor("#000000")
        im3 = ax3.imshow(
            vendor_slice_hu, cmap="gray", vmin=-1000.0, vmax=400.0, extent=v_ext, aspect="equal"
        )
        ax3.set_xlim(-half_fov, half_fov)
        ax3.set_ylim(-half_fov, half_fov)
        k_txt = f"kernel={vendor_kernel}, " if vendor_kernel else ""
        ax3.set_title(f"3. Published Vendor CT ({k_txt}1:1 Matched FOV)")
        ax3.set_xlabel(f"x [mm] (FOV={fov_mm:.0f} mm)")
        ax3.set_ylabel("y [mm]")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    else:
        im3 = ax3.imshow(
            ref_fbp_hu, cmap="gray", vmin=-300.0, vmax=1200.0, extent=extent, aspect="equal"
        )
        ax3.set_title("3. Our x_reference FBP (Bone Window [-300, 1200] HU)")
        ax3.set_xlabel("x [mm]")
        ax3.set_ylabel("y [mm]")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(
        rdc_raw_fbp_hu, cmap="gray", vmin=-1000.0, vmax=400.0, extent=extent, aspect="equal"
    )
    ax4.set_title(
        f"4. High-Pitch RDC (p={rdc_effective_pitch:.2f}) WITHOUT Fill (RMSE={rdc_raw_vs_ref_metrics['rmse_hu']:.1f} HU)"
    )
    ax4.set_xlabel("x [mm] (Raw Wedge Discontinuity Streaks)")
    ax4.set_ylabel("y [mm]")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(
        rdc_inpaint_fbp_hu, cmap="gray", vmin=-1000.0, vmax=400.0, extent=extent, aspect="equal"
    )
    ax5.set_title(
        f"5. High-Pitch RDC (p={rdc_effective_pitch:.2f}) + Conjugate Fill (RMSE={rdc_vs_ref_metrics['rmse_hu']:.1f} HU)"
    )
    ax5.set_xlabel(f"x [mm] (SSIM vs x_ref = {rdc_vs_ref_metrics['ssim']:.3f})")
    ax5.set_ylabel("y [mm]")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(hp_residual, cmap="magma", vmin=0.0, vmax=180.0, extent=extent, aspect="equal")
    ax6.set_title(
        f"6. Pure High-Pitch Artifact |x_RDC - x_ref| (MAE={rdc_vs_ref_metrics['mae_hu']:.1f} HU)"
    )
    ax6.set_xlabel("x [mm] (Target Signal for Stage 7 Dual-Domain DL)")
    ax6.set_ylabel("y [mm]")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage4_phantom_forward_visual_qa(
    out_path: str | Path,
    *,
    gt_hu: np.ndarray,
    roundtrip_fbp_hu: np.ndarray,
    roundtrip_metrics: dict[str, float],
    s_hp_raw: np.ndarray,
    mask_hp: np.ndarray,
    s_reproj: np.ndarray,
    rdc_prelim_hu: np.ndarray,
    rdc_metrics: dict[str, float],
    rdc_pitch: float,
    itf_prelim_hu: np.ndarray,
    itf_metrics: dict[str, float],
    itf_pitch: float,
    fov_mm: float,
    z_slice_mm: float,
    filter_name: str = "ram_lak",
) -> Path:
    """Save 6-panel (2x3) Stage 4 Phantom Forward->FBP Self-Consistency & RDC/ITF Visual QA PNG."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    vmax_sino = float(np.percentile(s_reproj, 99.5)) if s_reproj.size else 5.0
    vmax_sino = max(vmax_sino, 0.5)

    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(gt_hu, vmin=-1000.0, vmax=600.0),
                _panel_to_rgb(roundtrip_fbp_hu, vmin=-1000.0, vmax=600.0),
                _panel_to_rgb(
                    s_hp_raw,
                    vmin=0.0,
                    vmax=vmax_sino,
                    missing_mask=~np.asarray(mask_hp, dtype=bool),
                ),
                _panel_to_rgb(s_reproj, vmin=0.0, vmax=vmax_sino),
                _panel_to_rgb(rdc_prelim_hu, vmin=-1000.0, vmax=600.0),
                _panel_to_rgb(itf_prelim_hu, vmin=-1000.0, vmax=600.0),
            ],
            ncols=3,
            tile_h=280,
            tile_w=280,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    half_fov = 0.5 * float(fov_mm)
    extent = [-half_fov, half_fov, -half_fov, half_fov]

    fig = Figure(figsize=(17.5, 10.5), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"Stage 4 Visual QA — 2D/3D Forward Projection Self-Consistency & RDC/ITF Simulation "
        f"(z = {z_slice_mm:.1f} mm, filter = {filter_name})",
        fontsize=12.5,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(gt_hu, cmap="gray", vmin=-1000.0, vmax=600.0, extent=extent, aspect="equal")
    ax1.set_title("A. Exact 3D Phantom Ground Truth [-1000, 600] HU", fontsize=9.5)
    ax1.set_xlabel("x [mm] (Radiological: Left=Pat.Right, Right=Pat.Left)")
    ax1.set_ylabel("y [mm] (+y = Anterior)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(
        roundtrip_fbp_hu, cmap="gray", vmin=-1000.0, vmax=600.0, extent=extent, aspect="equal"
    )
    ax2.set_title(
        f"B. 2D Forward -> {filter_name} FBP (MAE={roundtrip_metrics['mae_hu']:.1f} HU, SSIM={roundtrip_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax2.set_xlabel("x [mm] (Zero Geometric Drift Proof)")
    ax2.set_ylabel("y [mm]")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    meas_pct = 100.0 * float(np.mean(mask_hp))
    ax3.imshow(
        _overlay_missing_rgba(s_hp_raw, mask_hp, vmin=0.0, vmax=vmax_sino),
        aspect="auto",
    )
    ax3.set_title(
        f"C. Fig. 2(a) High-Pitch SSRB Sinogram (meas={meas_pct:.1f}%)",
        fontsize=9.5,
    )
    ax3.set_xlabel("Virtual Flat Column u (Red = Missing Wedge)")
    ax3.set_ylabel("Short-Scan Angle k")

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(s_reproj, cmap="gray", vmin=0.0, vmax=vmax_sino, aspect="auto")
    ax4.set_title(
        "D. Fig. 2(d) Reprojected s_reproj = P_fan(mu_prelim)",
        fontsize=9.5,
    )
    ax4.set_xlabel("Virtual Flat Column u (Input to Sinogram U-Net)")
    ax4.set_ylabel("Short-Scan Angle k")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(
        rdc_prelim_hu, cmap="gray", vmin=-1000.0, vmax=600.0, extent=extent, aspect="equal"
    )
    ax5.set_title(
        f"E. RDC (p={rdc_pitch:.2f}) Prelim FBP (MAE={rdc_metrics['mae_hu']:.1f}, SSIM={rdc_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax5.set_xlabel("x [mm] (Fig. 1(b)/2(c) — Input to Image U-Net)")
    ax5.set_ylabel("y [mm]")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(
        itf_prelim_hu, cmap="gray", vmin=-1000.0, vmax=600.0, extent=extent, aspect="equal"
    )
    ax6.set_title(
        f"F. ITF (p={itf_pitch:.2f}) 3D Prelim FBP (MAE={itf_metrics['mae_hu']:.1f}, SSIM={itf_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax6.set_xlabel("x [mm] (Fig. 1(c)/2(c) — Full Cone-Beam + Missing Wedge)")
    ax6.set_ylabel("y [mm]")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage4_real_simulation_visual_qa(
    out_path: str | Path,
    *,
    case_id: str,
    z_slice_mm: float,
    native_pitch: float,
    fov_mm: float,
    fov_mask: np.ndarray,
    filter_name: str,
    x_target_hu: np.ndarray,
    s_target: np.ndarray,
    s_hp_raw: np.ndarray,
    mask_hp: np.ndarray,
    s_reproj: np.ndarray,
    rdc_pitch: float,
    rdc_prelim_hu: np.ndarray,
    rdc_metrics: dict[str, float],
    itf_pitch: float,
    itf_prelim_hu: np.ndarray,
    itf_metrics: dict[str, float],
) -> Path:
    """Save 6-panel (2x3) Stage 4 Real Patient (`C012`) Fig. 2(a)-(d) RDC & ITF Simulation PNG."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    fmask = np.asarray(fov_mask, dtype=bool)
    err_rdc = np.where(fmask, np.abs(rdc_prelim_hu - x_target_hu), 0.0)
    err_itf = np.where(fmask, np.abs(itf_prelim_hu - x_target_hu), 0.0)
    vmax_sino = float(np.percentile(s_target, 99.5)) if s_target.size else 5.0
    vmax_sino = max(vmax_sino, 0.5)

    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(x_target_hu, vmin=-1000.0, vmax=400.0),
                _panel_to_rgb(
                    s_hp_raw,
                    vmin=0.0,
                    vmax=vmax_sino,
                    missing_mask=~np.asarray(mask_hp, dtype=bool),
                ),
                _panel_to_rgb(s_reproj, vmin=0.0, vmax=vmax_sino),
                _panel_to_rgb(rdc_prelim_hu, vmin=-1000.0, vmax=400.0),
                _panel_to_rgb(itf_prelim_hu, vmin=-1000.0, vmax=400.0),
                _panel_to_rgb(err_itf, vmin=0.0, vmax=180.0, heat=True),
            ],
            ncols=3,
            tile_h=280,
            tile_w=280,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    half_fov = 0.5 * float(fov_mm)
    extent = [-half_fov, half_fov, -half_fov, half_fov]

    fig = Figure(figsize=(17.5, 10.5), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"Stage 4 Real-Data Visual QA ({case_id}) — AAPM 2026 Fig. 2 Reprojection & RDC vs ITF "
        f"at z = {z_slice_mm:.1f} mm ({filter_name}, FOV = {fov_mm:.0f} mm)",
        fontsize=12.5,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(
        x_target_hu, cmap="gray", vmin=-1000.0, vmax=400.0, extent=extent, aspect="equal"
    )
    ax1.set_title(
        f"1. Ground-Truth Target x* (p={native_pitch:.2f}, {filter_name} FBP)",
        fontsize=9.5,
    )
    ax1.set_xlabel("x [mm] (Radiological: Left=Pat.Right, Right=Pat.Left)")
    ax1.set_ylabel("y [mm] (+y = Anterior)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    meas_pct = 100.0 * float(np.mean(mask_hp))
    ax2.imshow(
        _overlay_missing_rgba(s_hp_raw, mask_hp, vmin=0.0, vmax=vmax_sino),
        aspect="auto",
    )
    ax2.set_title(
        f"2. Fig. 2(a) High-Pitch Sinogram (p={rdc_pitch:.2f}, meas={meas_pct:.1f}%)",
        fontsize=9.5,
    )
    ax2.set_xlabel("Virtual Flat Column u (Red = Missing Wedge)")
    ax2.set_ylabel("Short-Scan Angle k")

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(s_reproj, cmap="gray", vmin=0.0, vmax=vmax_sino, aspect="auto")
    ax3.set_title(
        "3. Fig. 2(d) Reprojected s_reproj (Sinogram U-Net Input)",
        fontsize=9.5,
    )
    ax3.set_xlabel("Virtual Flat Column u")
    ax3.set_ylabel("Short-Scan Angle k")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(
        rdc_prelim_hu, cmap="gray", vmin=-1000.0, vmax=400.0, extent=extent, aspect="equal"
    )
    ax4.set_title(
        f"4. RDC (p={rdc_pitch:.2f}) Prelim FBP (MAE={rdc_metrics['mae_hu']:.1f}, SSIM={rdc_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax4.set_xlabel("x [mm] (Fig. 1(b)/2(c) — Image U-Net Input RDC)")
    ax4.set_ylabel("y [mm]")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(
        itf_prelim_hu, cmap="gray", vmin=-1000.0, vmax=400.0, extent=extent, aspect="equal"
    )
    ax5.set_title(
        f"5. ITF (p={itf_pitch:.2f}) 3D Prelim FBP (MAE={itf_metrics['mae_hu']:.1f}, SSIM={itf_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax5.set_xlabel("x [mm] (Fig. 1(c)/2(c) — Image U-Net Input ITF)")
    ax5.set_ylabel("y [mm]")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    # Side-by-side split error map: Left half = |x_RDC - x*|, Right half = |x_ITF - x*|
    h_img, w_img = err_rdc.shape
    split_err = np.empty_like(err_rdc)
    split_err[:, : w_img // 2] = err_rdc[:, : w_img // 2]
    split_err[:, w_img // 2 :] = err_itf[:, w_img // 2 :]
    im6 = ax6.imshow(split_err, cmap="magma", vmin=0.0, vmax=180.0, extent=extent, aspect="equal")
    ax6.axvline(0.0, color="#00ffff", linestyle="--", linewidth=1.2)
    ax6.set_title(
        f"6. |x_prelim - x*| (RDC MAE={rdc_metrics['mae_hu']:.1f} | ITF MAE={itf_metrics['mae_hu']:.1f} HU)",
        fontsize=9.5,
    )
    ax6.set_xlabel("x [mm] (Left Half = RDC Error | Right Half = ITF Error)")
    ax6.set_ylabel("y [mm]")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage5_dual_domain_visual_qa(
    out_path: str | Path,
    *,
    title_prefix: str,
    scenario: str,
    target_pitch: float,
    effective_pitch: float,
    native_pitch: float,
    z_slice_mm: float,
    fov_mm: float,
    fov_mask: np.ndarray,
    filter_name: str,
    s_hp_raw: np.ndarray,
    mask_hp: np.ndarray,
    s_dc: np.ndarray,
    x_target_hu: np.ndarray,
    x_prelim_hu: np.ndarray,
    x_dc_hu: np.ndarray,
    x_final_hu: np.ndarray,
    prelim_metrics: dict[str, float],
    domain1_dc_metrics: dict[str, float],
    final_dual_metrics: dict[str, float],
    sino_metrics: dict[str, float],
    display_vmax_hu: float = 400.0,
) -> Path:
    """Save 6-panel (2x3) Stage 5 Dual-Domain DL Progression Visual QA PNG (Phantom or Real Patient)."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    fmask = np.asarray(fov_mask, dtype=bool)
    err_pre = np.where(fmask, np.abs(x_prelim_hu - x_target_hu), 0.0)
    err_fin = np.where(fmask, np.abs(x_final_hu - x_target_hu), 0.0)
    h_img, w_img = err_pre.shape
    split_err = np.empty_like(err_pre)
    split_err[:, : w_img // 2] = err_pre[:, : w_img // 2]
    split_err[:, w_img // 2 :] = err_fin[:, w_img // 2 :]

    vmax_sino = float(np.percentile(s_dc, 99.5)) if s_dc.size else 5.0
    vmax_sino = max(vmax_sino, 0.5)

    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(x_target_hu, vmin=-1000.0, vmax=display_vmax_hu),
                _panel_to_rgb(s_dc, vmin=0.0, vmax=vmax_sino),
                _panel_to_rgb(x_prelim_hu, vmin=-1000.0, vmax=display_vmax_hu),
                _panel_to_rgb(x_dc_hu, vmin=-1000.0, vmax=display_vmax_hu),
                _panel_to_rgb(x_final_hu, vmin=-1000.0, vmax=display_vmax_hu),
                _panel_to_rgb(split_err, vmin=0.0, vmax=180.0, heat=True),
            ],
            ncols=3,
            tile_h=280,
            tile_w=280,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    half_fov = 0.5 * float(fov_mm)
    extent = [-half_fov, half_fov, -half_fov, half_fov]
    scen_up = scenario.strip().upper()

    fig = Figure(figsize=(17.5, 10.5), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"{title_prefix} — Physics-Conditioned Dual-Domain DL ({scen_up}, target p={target_pitch:.1f}, "
        f"eff.p={effective_pitch:.2f}, z={z_slice_mm:.1f} mm, {filter_name}, FOV={fov_mm:.0f} mm)",
        fontsize=12.0,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(
        x_target_hu,
        cmap="gray",
        vmin=-1000.0,
        vmax=display_vmax_hu,
        extent=extent,
        aspect="equal",
    )
    ax1.set_title(
        f"1. Ground-Truth Target x* (p={native_pitch:.2f}, {filter_name} FBP)",
        fontsize=9.5,
    )
    ax1.set_xlabel("x [mm] (Radiological: Left=Pat.Right, Right=Pat.Left)")
    ax1.set_ylabel("y [mm] (+y = Anterior)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(s_dc, cmap="gray", vmin=0.0, vmax=vmax_sino, aspect="auto")
    miss_pct = 100.0 * float(1.0 - np.mean(mask_hp))
    w_mae_raw = sino_metrics.get("raw_missing_wedge_mae", 0.0)
    w_mae_dc = sino_metrics.get("sinonet_dc_missing_wedge_mae", 0.0)
    ax2.set_title(
        f"2. SinoNet + Hard DC s_dc (Wedge={miss_pct:.1f}%, MAE {w_mae_raw:.2f}->{w_mae_dc:.2f})",
        fontsize=9.5,
    )
    ax2.set_xlabel("Virtual Flat Column u (100% Measured Rays Preserved)")
    ax2.set_ylabel("Short-Scan Angle k")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(
        x_prelim_hu,
        cmap="gray",
        vmin=-1000.0,
        vmax=display_vmax_hu,
        extent=extent,
        aspect="equal",
    )
    ax3.set_title(
        f"3. Input x_prelim ({scen_up} p={effective_pitch:.2f}: RMSE={prelim_metrics['rmse_hu']:.1f}, SSIM={prelim_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax3.set_xlabel("x [mm] (Fig. 2(c) — Before Dual-Domain Network)")
    ax3.set_ylabel("y [mm]")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(
        x_dc_hu,
        cmap="gray",
        vmin=-1000.0,
        vmax=display_vmax_hu,
        extent=extent,
        aspect="equal",
    )
    ax4.set_title(
        f"4. Domain 1 Only: FBP(s_dc) (RMSE={domain1_dc_metrics['rmse_hu']:.1f} HU, SSIM={domain1_dc_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax4.set_xlabel("x [mm] (Differentiable Parker + Ram-Lak FBP Bridge)")
    ax4.set_ylabel("y [mm]")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(
        x_final_hu,
        cmap="gray",
        vmin=-1000.0,
        vmax=display_vmax_hu,
        extent=extent,
        aspect="equal",
    )
    ax5.set_title(
        f"5. Dual-Domain x_final (RMSE={final_dual_metrics['rmse_hu']:.1f} HU, SSIM={final_dual_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax5.set_xlabel(f"x [mm] (PSNR={final_dual_metrics['psnr_db']:.1f} dB, MAE={final_dual_metrics['mae_hu']:.1f} HU)")
    ax5.set_ylabel("y [mm]")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(split_err, cmap="magma", vmin=0.0, vmax=180.0, extent=extent, aspect="equal")
    ax6.axvline(0.0, color="#00ffff", linestyle="--", linewidth=1.2)
    ax6.set_title(
        f"6. Error |x - x*|: Left=x_prelim ({prelim_metrics['mae_hu']:.1f}) | Right=x_final ({final_dual_metrics['mae_hu']:.1f} HU)",
        fontsize=9.5,
    )
    ax6.set_xlabel("x [mm] (Left Half = Before DL | Right Half = After Dual-Domain DL)")
    ax6.set_ylabel("y [mm]")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage6_generalization_visual_qa(
    out_path: str | Path,
    *,
    title_prefix: str,
    split_mode: str,
    train_cases_str: str,
    test_case_id: str,
    scenario: str,
    target_pitch: float,
    effective_pitch: float,
    native_pitch: float,
    z_slice_mm: float,
    fov_mm: float,
    fov_mask: np.ndarray,
    filter_name: str,
    s_hp_raw: np.ndarray,
    mask_hp: np.ndarray,
    s_dc: np.ndarray,
    x_target_hu: np.ndarray,
    x_prelim_hu: np.ndarray,
    x_dc_hu: np.ndarray,
    x_final_hu: np.ndarray,
    prelim_metrics: dict[str, float],
    domain1_dc_metrics: dict[str, float],
    final_dual_metrics: dict[str, float],
    sino_metrics: dict[str, float],
    display_vmax_hu: float = 400.0,
) -> Path:
    """Save 6-panel (2x3) Stage 6 Zero-Leakage Held-Out Generalization Visual QA PNG."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    fmask = np.asarray(fov_mask, dtype=bool)
    err_pre = np.where(fmask, np.abs(x_prelim_hu - x_target_hu), 0.0)
    err_fin = np.where(fmask, np.abs(x_final_hu - x_target_hu), 0.0)
    h_img, w_img = err_pre.shape
    split_err = np.empty_like(err_pre)
    split_err[:, : w_img // 2] = err_pre[:, : w_img // 2]
    split_err[:, w_img // 2 :] = err_fin[:, w_img // 2 :]

    vmax_sino = float(np.percentile(s_dc, 99.5)) if s_dc.size else 5.0
    vmax_sino = max(vmax_sino, 0.5)

    if not _has_matplotlib():
        return _compose_grid_png_stdlib(
            target,
            [
                _panel_to_rgb(x_target_hu, vmin=-1000.0, vmax=display_vmax_hu),
                _panel_to_rgb(s_dc, vmin=0.0, vmax=vmax_sino),
                _panel_to_rgb(x_prelim_hu, vmin=-1000.0, vmax=display_vmax_hu),
                _panel_to_rgb(x_dc_hu, vmin=-1000.0, vmax=display_vmax_hu),
                _panel_to_rgb(x_final_hu, vmin=-1000.0, vmax=display_vmax_hu),
                _panel_to_rgb(split_err, vmin=0.0, vmax=180.0, heat=True),
            ],
            ncols=3,
            tile_h=280,
            tile_w=280,
        )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    half_fov = 0.5 * float(fov_mm)
    extent = [-half_fov, half_fov, -half_fov, half_fov]
    scen_up = scenario.strip().upper()

    fig = Figure(figsize=(17.5, 10.5), dpi=130)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"{title_prefix} — Zero-Leakage Generalization ({split_mode}: Train={train_cases_str} -> "
        f"Unseen Test={test_case_id}, {scen_up} p={target_pitch:.1f}, z={z_slice_mm:.1f} mm, {filter_name})",
        fontsize=11.5,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(
        x_target_hu,
        cmap="gray",
        vmin=-1000.0,
        vmax=display_vmax_hu,
        extent=extent,
        aspect="equal",
    )
    ax1.set_title(
        f"1. Unseen Test Target x* ({test_case_id}, native p={native_pitch:.2f})",
        fontsize=9.5,
    )
    ax1.set_xlabel("x [mm] (Held-Out Unseen Test Slice)")
    ax1.set_ylabel("y [mm] (+y = Anterior)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(s_dc, cmap="gray", vmin=0.0, vmax=vmax_sino, aspect="auto")
    miss_pct = 100.0 * float(1.0 - np.mean(mask_hp))
    w_mae_rep = sino_metrics.get("reproj_missing_wedge_mae", 0.0)
    w_mae_dc = sino_metrics.get("sinonet_dc_missing_wedge_mae", 0.0)
    ax2.set_title(
        f"2. Zero-Shot SinoNet+DC s_dc (Wedge={miss_pct:.1f}%, MAE {w_mae_rep:.3f}->{w_mae_dc:.3f})",
        fontsize=9.5,
    )
    ax2.set_xlabel("Virtual Flat Column u (0.000000 Measured Ray Drift)")
    ax2.set_ylabel("Short-Scan Angle k")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(
        x_prelim_hu,
        cmap="gray",
        vmin=-1000.0,
        vmax=display_vmax_hu,
        extent=extent,
        aspect="equal",
    )
    ax3.set_title(
        f"3. Unseen Test Prelim FBP (RMSE={prelim_metrics['rmse_hu']:.1f} HU, SSIM={prelim_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax3.set_xlabel("x [mm] (Before Dual-Domain Network)")
    ax3.set_ylabel("y [mm]")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(
        x_dc_hu,
        cmap="gray",
        vmin=-1000.0,
        vmax=display_vmax_hu,
        extent=extent,
        aspect="equal",
    )
    ax4.set_title(
        f"4. Unseen Test Domain-1 FBP(s_dc) (RMSE={domain1_dc_metrics['rmse_hu']:.1f}, SSIM={domain1_dc_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax4.set_xlabel("x [mm] (Zero-Shot SinoInpaintUNet + Hard DC)")
    ax4.set_ylabel("y [mm]")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(
        x_final_hu,
        cmap="gray",
        vmin=-1000.0,
        vmax=display_vmax_hu,
        extent=extent,
        aspect="equal",
    )
    ax5.set_title(
        f"5. Unseen Test Dual-Domain x_final (RMSE={final_dual_metrics['rmse_hu']:.1f}, SSIM={final_dual_metrics['ssim']:.3f})",
        fontsize=9.5,
    )
    ax5.set_xlabel(
        f"x [mm] (Zero-Shot Gain: -{prelim_metrics['rmse_hu'] - final_dual_metrics['rmse_hu']:.2f} HU, PSNR={final_dual_metrics['psnr_db']:.1f} dB)"
    )
    ax5.set_ylabel("y [mm]")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(split_err, cmap="magma", vmin=0.0, vmax=180.0, extent=extent, aspect="equal")
    ax6.axvline(0.0, color="#00ffff", linestyle="--", linewidth=1.2)
    ax6.set_title(
        f"6. Unseen Test |x - x*|: Left=Prelim ({prelim_metrics['mae_hu']:.1f}) | Right=Dual ({final_dual_metrics['mae_hu']:.1f} HU)",
        fontsize=9.5,
    )
    ax6.set_xlabel("x [mm] (Left Half = Prelim Error | Right Half = Zero-Shot Dual-Domain Error)")
    ax6.set_ylabel("y [mm]")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(target, format="png")
    return target


def save_stage7_ablation_visual_qa(
    out_path: str | Path,
    *,
    case_id: str,
    scenario: str,
    target_pitch: float,
    z_slice_mm: float,
    fov_mm: float,
    fov_mask: np.ndarray,
    x_target_hu: np.ndarray,
    method_slices_hu: dict[str, np.ndarray],
    method_metrics: dict[str, dict[str, float]],
    display_vmax_hu: float = 400.0,
) -> Path:
    """Save 8-panel (2x4) Stage 7 Thesis Ablation Visual QA PNG (`M0..M5` + GT + Line Profile)."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    ordered_keys = [
        ("M0_raw_nofill_fbp", "M0: Raw No-Fill FBP"),
        ("M1_angular_feather_fbp", "M1: Angular Feather FBP"),
        ("M2_physics_conjugate_prelim_fbp", "M2: Physics Conjugate Prelim"),
        ("M3_image_only_dl", "M3: Image-Only U-Net"),
        ("M4_sino_only_dl", "M4: Sino-Only DL (SinoNet+DC)"),
        ("M5_full_dual_domain_dl", "M5: Proposed Dual-Domain DL"),
    ]
    fmask = np.asarray(fov_mask, dtype=bool)
    m2_img = method_slices_hu.get("M2_physics_conjugate_prelim_fbp", x_target_hu)
    m5_img = method_slices_hu.get("M5_full_dual_domain_dl", x_target_hu)
    err_m2 = np.where(fmask, np.abs(m2_img - x_target_hu), 0.0)
    err_m5 = np.where(fmask, np.abs(m5_img - x_target_hu), 0.0)
    split_err = np.empty_like(err_m2)
    mid_w = err_m2.shape[1] // 2
    split_err[:, :mid_w] = err_m2[:, :mid_w]
    split_err[:, mid_w:] = err_m5[:, mid_w:]

    if not _has_matplotlib():
        panels = [_panel_to_rgb(x_target_hu, vmin=-1000.0, vmax=display_vmax_hu)]
        for k, _ in ordered_keys:
            panels.append(
                _panel_to_rgb(
                    method_slices_hu.get(k, x_target_hu), vmin=-1000.0, vmax=display_vmax_hu
                )
            )
        panels.append(_panel_to_rgb(split_err, vmin=0.0, vmax=180.0, heat=True))
        return _compose_grid_png_stdlib(target, panels, ncols=4, tile_h=240, tile_w=240)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    half_fov = 0.5 * float(fov_mm)
    extent = [-half_fov, half_fov, -half_fov, half_fov]
    fig = Figure(figsize=(21.0, 10.5), dpi=135)
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"Stage 7 Ablation Study — Unseen Test {case_id} ({scenario.upper()} p={target_pitch:.1f}, z={z_slice_mm:.1f} mm)",
        fontsize=12.5,
        fontweight="bold",
    )

    ax_gt = fig.add_subplot(2, 4, 1)
    im_gt = ax_gt.imshow(
        x_target_hu, cmap="gray", vmin=-1000.0, vmax=display_vmax_hu, extent=extent, aspect="equal"
    )
    ax_gt.set_title("Reference Target x* (Low-Pitch GT)", fontsize=9.5, fontweight="bold")
    ax_gt.set_xlabel("x [mm]")
    ax_gt.set_ylabel("y [mm]")
    fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    for idx_m, (m_key, m_label) in enumerate(ordered_keys, start=2):
        ax = fig.add_subplot(2, 4, idx_m)
        sl = method_slices_hu.get(m_key, x_target_hu)
        mm = method_metrics.get(m_key, {"rmse_hu": 0.0, "psnr_db": 0.0, "ssim": 0.0})
        im = ax.imshow(
            sl, cmap="gray", vmin=-1000.0, vmax=display_vmax_hu, extent=extent, aspect="equal"
        )
        ax.set_title(
            f"{m_label}\nRMSE={mm['rmse_hu']:.1f} HU | PSNR={mm['psnr_db']:.1f} dB | SSIM={mm['ssim']:.3f}",
            fontsize=8.8,
        )
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_prof = fig.add_subplot(2, 4, 8)
    mid_row = x_target_hu.shape[0] // 2
    x_axis = np.linspace(-half_fov, half_fov, x_target_hu.shape[1])
    ax_prof.plot(x_axis, x_target_hu[mid_row], color="#111111", linewidth=1.6, label="Target x*")
    ax_prof.plot(
        x_axis, m2_img[mid_row], color="#d95f02", linewidth=1.1, alpha=0.85, label="M2 Prelim"
    )
    if "M4_sino_only_dl" in method_slices_hu:
        ax_prof.plot(
            x_axis,
            method_slices_hu["M4_sino_only_dl"][mid_row],
            color="#7570b3",
            linewidth=1.1,
            alpha=0.85,
            label="M4 Sino-Only",
        )
    ax_prof.plot(
        x_axis, m5_img[mid_row], color="#1b9e77", linewidth=1.35, label="M5 Dual-Domain"
    )
    ax_prof.set_title("Central Axial HU Line Profile (y = 0 mm)", fontsize=9.5)
    ax_prof.set_xlabel("x [mm]")
    ax_prof.set_ylabel("Attenuation [HU]")
    ax_prof.grid(True, alpha=0.25)
    ax_prof.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(target, format="png")
    return target


def save_stage7_pitch_degradation_curves(
    out_path: str | Path,
    *,
    by_scenario_pitch: dict[str, dict[str, dict[str, float]]],
) -> Path:
    """Save 2x2 Stage 7 Pitch Degradation Curves (`RMSE [HU]` & `SSIM` across pitches per scenario)."""
    target = Path(out_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not _has_matplotlib():
        dummy = np.linspace(0.0, 255.0, 240 * 320, dtype=np.float64).reshape(240, 320)
        return _compose_grid_png_stdlib(target, [_panel_to_rgb(dummy)], ncols=1)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    method_styles = [
        ("M0_raw_nofill_fbp", "M0: Raw No-Fill", "#888888", "--"),
        ("M1_angular_feather_fbp", "M1: Angular Feather", "#e6ab02", "--"),
        ("M2_physics_conjugate_prelim_fbp", "M2: Conjugate Prelim", "#d95f02", "-."),
        ("M3_image_only_dl", "M3: Image-Only DL", "#e7298a", ":"),
        ("M4_sino_only_dl", "M4: Sino-Only DL", "#7570b3", "-"),
        ("M5_full_dual_domain_dl", "M5: Full Dual-Domain", "#1b9e77", "-"),
    ]

    scen_pitches: dict[str, list[tuple[float, str]]] = {}
    for sp_key in by_scenario_pitch:
        if "_p" not in sp_key:
            continue
        sc, p_str = sp_key.split("_p", 1)
        try:
            pv = float(p_str)
        except ValueError:
            continue
        scen_pitches.setdefault(sc, []).append((pv, sp_key))
    for sc in scen_pitches:
        scen_pitches[sc].sort(key=lambda t: t[0])

    scenarios_sorted = sorted(scen_pitches.keys()) or ["itf"]
    fig = Figure(figsize=(14.5, 5.2 * max(1, len(scenarios_sorted))), dpi=135)
    FigureCanvasAgg(fig)
    fig.suptitle(
        "Stage 7 Quantitative Ablation Across Clinical High-Pitch Regimes",
        fontsize=12.5,
        fontweight="bold",
    )

    for r_i, sc in enumerate(scenarios_sorted):
        pts = scen_pitches.get(sc, [])
        xs = [p for p, _ in pts]
        ax_r = fig.add_subplot(len(scenarios_sorted), 2, 2 * r_i + 1)
        ax_s = fig.add_subplot(len(scenarios_sorted), 2, 2 * r_i + 2)
        for m_key, m_lbl, col, ls in method_styles:
            ys_rmse = [
                by_scenario_pitch[k][m_key]["rmse_hu"]
                for _, k in pts
                if m_key in by_scenario_pitch.get(k, {})
            ]
            ys_ssim = [
                by_scenario_pitch[k][m_key]["ssim"]
                for _, k in pts
                if m_key in by_scenario_pitch.get(k, {})
            ]
            if len(ys_rmse) == len(xs) and xs:
                lw = 2.2 if m_key == "M5_full_dual_domain_dl" else 1.4
                ax_r.plot(xs, ys_rmse, marker="o", color=col, linestyle=ls, linewidth=lw, label=m_lbl)
                ax_s.plot(xs, ys_ssim, marker="s", color=col, linestyle=ls, linewidth=lw, label=m_lbl)
        ax_r.set_title(f"{sc.upper()} — RMSE [HU] vs Target Pitch (Lower is Better)")
        ax_r.set_xlabel("Target Helical Pitch")
        ax_r.set_ylabel("RMSE [HU]")
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(fontsize=8)

        ax_s.set_title(f"{sc.upper()} — SSIM vs Target Pitch (Higher is Better)")
        ax_s.set_xlabel("Target Helical Pitch")
        ax_s.set_ylabel("SSIM")
        ax_s.grid(True, alpha=0.3)
        ax_s.legend(fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(target, format="png")
    return target


