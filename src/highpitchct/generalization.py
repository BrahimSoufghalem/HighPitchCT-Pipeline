"""Stage 6 — Multi-Patient / Multi-Zone Generalization & Zero-Leakage Out-of-Sample Evaluation.

This module implements audit-grade out-of-sample generalization protocols for the
Physics-Conditioned Dual-Domain High-Pitch Helical CT Network (`DualDomainHighPitchNet`):

1. **Strict Zero-Leakage Split Verification (`verify_zero_leakage_split`):**
   - **`patient_split` mode (Inter-Patient Generalization):** Enforces strict patient-level
     disjointness (`train_case_ids ∩ test_case_ids == ∅`) and sample ID disjointness.
   - **`axial_guardband` mode (Intra-Patient Multi-Zone Axial Generalization):** Partitions
     a patient's helical scan into an upper/mid-thorax training zone and a lower/basal unseen
     test zone separated by a physical **Helical Cone-Beam Dead-Zone Guard Band**
     (`Delta_z_guard >= min_guardband_mm`, default `45.0 mm`), ensuring that neither target
     slices nor 3D ITF cone-beam slabs overlap between train and test sets (`slab_gap_mm > 0`).
2. **Synthetic Multi-Phantom Generalization (`build_multi_phantom_generalization_split`):**
   - Trains on morphologically distinct 3D thorax phantoms (`TRAIN_PHANTOM_A` at `native_pitch=0.85`
     and `TRAIN_PHANTOM_B` at `native_pitch=0.65`) and evaluates zero-shot on an unseen third
     phantom (`TEST_PHANTOM_UNSEEN` at `native_pitch=0.90` with unseen pulmonary nodules and
     shifted mediastinal/vertebral geometry).
3. **Out-of-Sample Frozen Evaluation (`run_generalization_experiment`):**
   - Trains `DualDomainHighPitchNet` (4-level `64->128->256->512` Residual U-Nets, 3-stage
     `Eq. 1 -> Eq. 2 -> Eq. 3` curriculum, `60% ITF + 40% RDC` across `p in {1.5, 2.0, 2.5}`)
     exclusively on `train_samples`, freezes all parameters, and evaluates zero-shot on held-out
     `test_samples` without any test-set calibration or weight adaptation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from highpitchct.ctpd_reader import read_ct_image_volume, scan_projection_series
from highpitchct.dual_domain_dl import (
    DEFAULT_CLINICAL_HIGH_PITCHES,
    DEFAULT_ITF_TRAINING_RATIO,
    DualDomainPrediction,
    DualDomainSample,
    build_hybrid_phantom_dataset,
    build_hybrid_series_dataset_streaming,
    create_dual_domain_model,
    evaluate_dual_domain_sample,
    summarize_dual_domain_predictions,
    train_dual_domain_model,
)
from highpitchct.phantom import Ellipsoid3D, build_thorax_phantom
from highpitchct.recon_fbp import normalize_filter_name


@dataclass(frozen=True)
class ZeroLeakageAuditReport:
    """Formal mathematical audit proving zero data leakage between training and held-out test sets."""

    split_mode: str  # "patient_split" or "axial_guardband"
    train_case_ids: list[str]
    test_case_ids: list[str]
    patient_id_overlap_count: int
    sample_id_overlap_count: int
    train_sample_count: int
    test_sample_count: int
    train_z_slices_mm: list[float]
    test_z_slices_mm: list[float]
    min_train_test_z_separation_mm: float | None
    required_guardband_mm: float
    itf_slab_half_span_mm: float
    min_itf_slab_gap_mm: float | None
    overall_zero_leakage_pass: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_zero_leakage_split(
    train_samples: Sequence[DualDomainSample],
    test_samples: Sequence[DualDomainSample],
    *,
    split_mode: str,
    min_guardband_mm: float = 40.0,
    itf_slab_half_span_mm: float = 18.0,
) -> ZeroLeakageAuditReport:
    """Verify strict mathematical and physical separation between `train_samples` and `test_samples`.

    Raises `ValueError` immediately if any leakage invariant is violated (No Silent Fallback).
    """
    if not train_samples:
        raise ValueError("train_samples must not be empty for Stage 6 generalization")
    if not test_samples:
        raise ValueError("test_samples must not be empty for Stage 6 generalization")

    mode = split_mode.strip().lower()
    if mode not in ("patient_split", "axial_guardband"):
        raise ValueError(
            f"Unsupported split_mode={split_mode!r}; expected 'patient_split' or 'axial_guardband'"
        )

    train_cases = sorted({str(s.case_id) for s in train_samples})
    test_cases = sorted({str(s.case_id) for s in test_samples})
    case_overlap = set(train_cases).intersection(set(test_cases))

    train_ids = {str(s.sample_id) for s in train_samples}
    test_ids = {str(s.sample_id) for s in test_samples}
    sample_overlap = train_ids.intersection(test_ids)
    if sample_overlap:
        raise ValueError(
            f"Zero-Leakage violation: {len(sample_overlap)} sample_ids appear in both train and test: "
            f"{sorted(sample_overlap)[:5]}"
        )

    train_z = sorted({round(float(s.z_slice_mm), 3) for s in train_samples})
    test_z = sorted({round(float(s.z_slice_mm), 3) for s in test_samples})

    if mode == "patient_split":
        if case_overlap:
            raise ValueError(
                f"Zero-Leakage patient_split violation: patient case_ids {sorted(case_overlap)} "
                f"appear in both train_cases={train_cases} and test_cases={test_cases}"
            )
        return ZeroLeakageAuditReport(
            split_mode="patient_split",
            train_case_ids=train_cases,
            test_case_ids=test_cases,
            patient_id_overlap_count=0,
            sample_id_overlap_count=0,
            train_sample_count=len(train_samples),
            test_sample_count=len(test_samples),
            train_z_slices_mm=train_z,
            test_z_slices_mm=test_z,
            min_train_test_z_separation_mm=None,
            required_guardband_mm=0.0,
            itf_slab_half_span_mm=round(float(itf_slab_half_span_mm), 3),
            min_itf_slab_gap_mm=None,
            overall_zero_leakage_pass=True,
            notes=(
                f"Strict inter-patient split verified: train_cases={train_cases} and "
                f"test_cases={test_cases} share 0 patient IDs and 0 sample IDs."
            ),
        )

    # mode == "axial_guardband": check per-case axial distance between train and test slices
    min_sep = float("inf")
    for s_tr in train_samples:
        for s_te in test_samples:
            if s_tr.case_id == s_te.case_id:
                dist = abs(float(s_tr.z_slice_mm) - float(s_te.z_slice_mm))
                if dist < min_sep:
                    min_sep = dist

    if min_sep == float("inf"):
        min_sep_val: float | None = None
        min_slab_gap: float | None = None
    else:
        min_sep_val = round(float(min_sep), 4)
        min_slab_gap = round(float(min_sep - 2.0 * float(itf_slab_half_span_mm)), 4)
        if min_sep < float(min_guardband_mm) - 1e-4:
            raise ValueError(
                f"Zero-Leakage axial_guardband violation: minimum axial separation between train "
                f"({train_z}) and test ({test_z}) is {min_sep:.2f} mm < required guardband "
                f"{min_guardband_mm:.2f} mm."
            )
        if min_slab_gap < -1e-4:
            raise ValueError(
                f"Zero-Leakage 3D ITF slab overlap violation: min_sep={min_sep:.2f} mm with "
                f"itf_slab_half_span_mm={itf_slab_half_span_mm:.2f} mm leaves negative slab gap "
                f"{min_slab_gap:.2f} mm."
            )

    return ZeroLeakageAuditReport(
        split_mode="axial_guardband",
        train_case_ids=train_cases,
        test_case_ids=test_cases,
        patient_id_overlap_count=len(case_overlap),
        sample_id_overlap_count=0,
        train_sample_count=len(train_samples),
        test_sample_count=len(test_samples),
        train_z_slices_mm=train_z,
        test_z_slices_mm=test_z,
        min_train_test_z_separation_mm=min_sep_val,
        required_guardband_mm=round(float(min_guardband_mm), 3),
        itf_slab_half_span_mm=round(float(itf_slab_half_span_mm), 3),
        min_itf_slab_gap_mm=min_slab_gap,
        overall_zero_leakage_pass=True,
        notes=(
            f"Strict axial guard-band split verified: train_z={train_z} mm and test_z={test_z} mm "
            f"are separated by >= {min_sep_val} mm (3D ITF slab gap = {min_slab_gap} mm >= 0)."
        ),
    )


def plan_axial_guardband_slices(
    z_min_raw_mm: float,
    z_max_raw_mm: float,
    coverage_isocenter_mm: float,
    *,
    n_train_slices: int = 2,
    n_test_slices: int = 1,
    min_guardband_mm: float = 45.0,
) -> dict[str, Any]:
    """Plan spatially disjoint Cranial/Mid-Thorax training slices and Caudal/Basal test slices.

    Guarantees that:
      1. All slices lie safely inside the full-coverage interior helical scan interval.
      2. `min(test_z) - max(train_z) >= min_guardband_mm`.
      3. `itf_slab_half_span_mm <= 0.45 * (min(test_z) - max(train_z))` so the 3D ITF cone-beam
         slabs for the training zone and test zone have strictly positive separation (`gap > 0`).
    """
    if int(n_train_slices) < 1 or int(n_test_slices) < 1:
        raise ValueError("n_train_slices and n_test_slices must both be >= 1")

    margin = max(24.0, 0.90 * float(coverage_isocenter_mm))
    z_lo = float(z_min_raw_mm) + margin
    z_hi = float(z_max_raw_mm) - margin
    usable_span = z_hi - z_lo
    if usable_span < float(min_guardband_mm) + 12.0:
        raise ValueError(
            f"Usable interior axial span ({usable_span:.1f} mm) is too short for "
            f"min_guardband_mm={min_guardband_mm:.1f} mm."
        )

    # Allocate 45% of remaining span beyond guardband to training zone, 25% to test zone
    rem_span = usable_span - float(min_guardband_mm)
    train_span = max(0.0, min(36.0, 0.60 * rem_span)) if int(n_train_slices) > 1 else 0.0
    test_span = max(0.0, min(24.0, 0.35 * rem_span)) if int(n_test_slices) > 1 else 0.0

    train_z_start = z_lo + 0.08 * rem_span
    train_z_end = train_z_start + train_span
    test_z_end = z_hi - 0.08 * rem_span
    test_z_start = test_z_end - test_span

    actual_guard = test_z_start - train_z_end
    if actual_guard < float(min_guardband_mm):
        # Push zones toward z_lo and z_hi to honor min_guardband_mm exactly
        train_z_start = z_lo
        train_z_end = z_lo + train_span
        test_z_end = z_hi
        test_z_start = z_hi - test_span
        actual_guard = test_z_start - train_z_end

    if actual_guard < float(min_guardband_mm) - 1e-4:
        raise ValueError(
            f"Could not satisfy min_guardband_mm={min_guardband_mm:.1f} mm within [{z_lo:.1f}, {z_hi:.1f}] mm"
        )

    train_z = (
        np.array([0.5 * (train_z_start + train_z_end)], dtype=np.float64)
        if int(n_train_slices) == 1
        else np.linspace(train_z_start, train_z_end, int(n_train_slices), dtype=np.float64)
    )
    test_z = (
        np.array([0.5 * (test_z_start + test_z_end)], dtype=np.float64)
        if int(n_test_slices) == 1
        else np.linspace(test_z_start, test_z_end, int(n_test_slices), dtype=np.float64)
    )

    slab_half_span = min(0.45 * float(actual_guard), 0.65 * float(coverage_isocenter_mm))
    slab_half_span = max(12.0, float(slab_half_span))

    return {
        "train_z_slices_mm": [round(float(z), 3) for z in train_z],
        "test_z_slices_mm": [round(float(z), 3) for z in test_z],
        "actual_guardband_mm": round(float(actual_guard), 3),
        "itf_slab_half_span_mm": round(float(slab_half_span), 3),
        "itf_slab_gap_mm": round(float(actual_guard - 2.0 * slab_half_span), 3),
    }


def build_variant_thorax_phantom(variant: str = "train_a") -> list[Ellipsoid3D]:
    """Build morphologically distinct 3D Thorax Phantoms representing different patient anatomies.

    Variants:
      - `'train_a'`: Standard thorax habitus (`axial_scale=0.68`).
      - `'train_b'`: Broader patient habitus (`axial_scale=0.76`) with shifted cardiac mass &
        hypertrophic spine.
      - `'test_unseen'`: Held-out unseen patient anatomy (`axial_scale=0.72`) featuring two
        high-contrast pulmonary nodules and a calcified aortic plaque not present in training phantoms.
    """
    v = variant.strip().lower()
    if v == "train_a":
        return build_thorax_phantom(axial_scale=0.68)
    if v == "train_b":
        base = build_thorax_phantom(axial_scale=0.76)
        extra = [
            # Enlarged left ventricular apex & lateral pleural thickening
            Ellipsoid3D(
                name="train_b_cardiac_shift",
                center_xyz_mm=(-28.0, 26.0, 0.0),
                semi_axes_xyz_mm=(26.0, 22.0, 28.0),
                delta_hu=55.0,
                yaw_rad=0.21,
            ),
            Ellipsoid3D(
                name="train_b_dorsal_muscle",
                center_xyz_mm=(0.0, -92.0, 4.0),
                semi_axes_xyz_mm=(65.0, 18.0, 32.0),
                delta_hu=65.0,
            ),
        ]
        return base + extra
    if v == "test_unseen":
        base = build_thorax_phantom(axial_scale=0.72)
        unseen_structures = [
            # Right-lung solid pulmonary nodule (unseen in train_a / train_b)
            Ellipsoid3D(
                name="unseen_right_lung_nodule",
                center_xyz_mm=(58.0, 18.0, 2.0),
                semi_axes_xyz_mm=(14.0, 12.0, 16.0),
                delta_hu=620.0,
            ),
            # Left-lung posterior ground-glass nodule
            Ellipsoid3D(
                name="unseen_left_lung_nodule",
                center_xyz_mm=(-54.0, -24.0, 2.0),
                semi_axes_xyz_mm=(16.0, 14.0, 18.0),
                delta_hu=480.0,
            ),
            # Calcified descending aortic plaque
            Ellipsoid3D(
                name="unseen_aortic_calcification",
                center_xyz_mm=(-18.0, -34.0, 2.0),
                semi_axes_xyz_mm=(9.0, 9.0, 20.0),
                delta_hu=380.0,
            ),
        ]
        return base + unseen_structures
    raise ValueError(f"Unknown phantom variant={variant!r}; expected 'train_a', 'train_b', or 'test_unseen'")


def _relabel_samples_case_id(
    samples: Sequence[DualDomainSample],
    new_case_id: str,
) -> list[DualDomainSample]:
    """Return copies of `DualDomainSample` entries with updated `case_id` and `sample_id` prefix."""
    from dataclasses import replace

    out: list[DualDomainSample] = []
    for s in samples:
        sid = f"{new_case_id}_{s.scenario}_p{s.target_pitch:.2f}_z{s.z_slice_mm:+.1f}"
        out.append(replace(s, sample_id=sid, case_id=new_case_id))
    return out


def build_multi_phantom_generalization_split(
    *,
    pitches: Sequence[float] = DEFAULT_CLINICAL_HIGH_PITCHES,
    scenarios: Sequence[str] = ("itf", "rdc"),
    image_size: int = 96,
    fov_mm: float = 360.0,
    filter_name: str = "ram_lak",
    backend: str = "torch",
) -> dict[str, Any]:
    """Build a multi-phantom zero-leakage generalization dataset across distinct anatomies and native pitches.

    - Training patients:
        1. `PHANTOM_TRAIN_A`: standard thorax (`native_pitch = 0.85`, `z = -12.0 mm`)
        2. `PHANTOM_TRAIN_B`: broad habitus (`native_pitch = 0.65`, `z = +8.0 mm`)
    - Unseen Test patient:
        3. `PHANTOM_TEST_UNSEEN`: thorax with unseen pulmonary nodules & aortic calcification
           (`native_pitch = 0.90`, `z = +2.0 mm`)
    """
    fname = normalize_filter_name(filter_name)
    samples_a = _relabel_samples_case_id(
        build_hybrid_phantom_dataset(
            pitches=pitches,
            scenarios=scenarios,
            z_slices_mm=(-12.0,),
            native_pitch=0.85,
            image_size=image_size,
            fov_mm=fov_mm,
            filter_name=fname,
            backend=backend,
            phantom=build_variant_thorax_phantom("train_a"),
        ),
        "PHANTOM_TRAIN_A",
    )
    samples_b = _relabel_samples_case_id(
        build_hybrid_phantom_dataset(
            pitches=pitches,
            scenarios=scenarios,
            z_slices_mm=(8.0,),
            native_pitch=0.65,
            image_size=image_size,
            fov_mm=fov_mm,
            filter_name=fname,
            backend=backend,
            phantom=build_variant_thorax_phantom("train_b"),
        ),
        "PHANTOM_TRAIN_B",
    )
    train_samples = samples_a + samples_b

    test_samples = _relabel_samples_case_id(
        build_hybrid_phantom_dataset(
            pitches=pitches,
            scenarios=scenarios,
            z_slices_mm=(2.0,),
            native_pitch=0.90,
            image_size=image_size,
            fov_mm=fov_mm,
            filter_name=fname,
            backend=backend,
            phantom=build_variant_thorax_phantom("test_unseen"),
        ),
        "PHANTOM_TEST_UNSEEN",
    )

    audit = verify_zero_leakage_split(
        train_samples,
        test_samples,
        split_mode="patient_split",
    )
    return {
        "split_mode": "patient_split",
        "train_samples": train_samples,
        "test_samples": test_samples,
        "zero_leakage_audit": audit,
    }


def build_real_series_generalization_split(
    targets: Sequence[tuple[str, Path, Path | None]],
    *,
    split_mode: str = "auto",
    train_case_ids: Sequence[str] | None = None,
    test_case_ids: Sequence[str] | None = None,
    n_train_slices_per_case: int = 2,
    n_test_slices_per_case: int = 1,
    min_guardband_mm: float = 45.0,
    pitches: Sequence[float] = DEFAULT_CLINICAL_HIGH_PITCHES,
    scenarios: Sequence[str] = ("itf", "rdc"),
    image_size: int = 256,
    fov_mm: float | None = None,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    inpaint_mode: str = "conjugate_feather",
    itf_slab_slices: int = 7,
    backend: str = "torch",
    workers: int = 4,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Build a real-patient zero-leakage generalization split (`patient_split` or `axial_guardband`).

    Rules:
      - If `split_mode == 'patient_split'` (or `'auto'` with >= 2 distinct patients):
        assigns distinct patients to `train_samples` and `test_samples` with zero patient overlap.
      - If `split_mode == 'axial_guardband'` (or `'auto'` with 1 patient such as `C012`):
        partitions each patient's scan into a Cranial/Mid-Thorax training zone and a Caudal/Basal
        unseen test zone separated by `>= min_guardband_mm` (default `45.0 mm`) with non-overlapping
        3D ITF slabs.
      - Never performs silent fallbacks: if `patient_split` is explicitly requested with < 2 patients,
        raises `ValueError`.
    """
    if not targets:
        raise ValueError("No projection series targets provided to build_real_series_generalization_split")

    fname = normalize_filter_name(filter_name)
    req_mode = split_mode.strip().lower()
    if req_mode not in ("auto", "patient_split", "axial_guardband"):
        raise ValueError(
            f"Unsupported split_mode={split_mode!r}; expected 'auto', 'patient_split', or 'axial_guardband'"
        )

    if req_mode == "auto":
        active_mode = "patient_split" if len(targets) >= 2 else "axial_guardband"
    else:
        active_mode = req_mode

    if active_mode == "patient_split" and len(targets) < 2:
        raise ValueError(
            f"Explicit split_mode='patient_split' requires at least 2 local patient series, "
            f"but only {len(targets)} found ({[t[0] for t in targets]}). "
            "Download a second patient or use --split-mode axial_guardband."
        )

    def _resolve_case_fov_and_center(
        geom: Any, idir: Path | None
    ) -> tuple[float | None, tuple[float, float]]:
        vendor_fov_mm: float | None = None
        if idir is not None and idir.exists():
            vol_meta = read_ct_image_volume(
                idir, water_mu_mm_inv=geom.water_mu_mm_inv
            )
            ps_yx = vol_meta["pixel_spacing_yx_mm"]
            vendor_fov_mm = float(vol_meta["shape_zyx"][2]) * float(ps_yx[1])
        case_fov = (
            fov_mm
            if fov_mm is not None
            else (
                vendor_fov_mm
                if vendor_fov_mm is not None and vendor_fov_mm > 0.0
                else None
            )
        )
        return case_fov, (0.0, 0.0)

    train_samples: list[DualDomainSample] = []
    test_samples: list[DualDomainSample] = []
    case_metadata: list[dict[str, Any]] = []

    if active_mode == "patient_split":
        by_cid = {cid: (cid, pdir, idir) for cid, pdir, idir in targets}
        all_cids = [cid for cid, _, _ in targets]

        if train_case_ids or test_case_ids:
            tr_cids = [
                c if c in by_cid else f"case_{c}"
                for c in (train_case_ids or [])
            ]
            te_cids = [
                c if c in by_cid else f"case_{c}"
                for c in (test_case_ids or [])
            ]
            if not tr_cids and te_cids:
                tr_cids = [c for c in all_cids if c not in set(te_cids)]
            if not te_cids and tr_cids:
                te_cids = [c for c in all_cids if c not in set(tr_cids)]
            missing_tr = [c for c in tr_cids if c not in by_cid]
            missing_te = [c for c in te_cids if c not in by_cid]
            if missing_tr or missing_te:
                raise ValueError(
                    f"Explicit train/test case IDs not found locally (missing train={missing_tr}, "
                    f"missing test={missing_te}; available={all_cids}). Refusing silent fallback."
                )
        else:
            tr_cids = all_cids[:-1]
            te_cids = [all_cids[-1]]

        if set(tr_cids).intersection(set(te_cids)):
            raise ValueError(
                f"train_case_ids={tr_cids} and test_case_ids={te_cids} must be strictly disjoint"
            )

        for role_name, cid_list, n_slices in (
            ("train", tr_cids, int(n_train_slices_per_case)),
            ("test", te_cids, int(n_test_slices_per_case)),
        ):
            for cid in cid_list:
                _, pdir, idir = by_cid[cid]
                geom, files = scan_projection_series(
                    pdir, workers=max(workers, 4), show_progress=show_progress
                )
                case_fov, c_xy = _resolve_case_fov_and_center(geom, idir)
                z_min_raw = float(np.min(geom.z0_mm))
                z_max_raw = float(np.max(geom.z0_mm))
                margin = max(25.0, 0.9 * float(geom.coverage_isocenter_mm))
                z_lo = z_min_raw + margin
                z_hi = z_max_raw - margin
                z_req = (
                    np.array([0.5 * (z_lo + z_hi)], dtype=np.float64)
                    if n_slices <= 1
                    else np.linspace(z_lo, z_hi, n_slices, dtype=np.float64)
                )
                s_list = build_hybrid_series_dataset_streaming(
                    files,
                    z_req,
                    case_id=cid,
                    geometry=geom,
                    pitches=pitches,
                    scenarios=scenarios,
                    image_size=image_size,
                    fov_mm=case_fov,
                    center_xy_mm=c_xy,
                    filter_name=fname,
                    cutoff_ratio=cutoff_ratio,
                    inpaint_mode=inpaint_mode,
                    itf_slab_slices=itf_slab_slices,
                    backend=backend,
                    workers=workers,
                    show_progress=show_progress,
                )
                if role_name == "train":
                    train_samples.extend(s_list)
                else:
                    test_samples.extend(s_list)
                case_metadata.append(
                    {
                        "case_id": cid,
                        "split_role": role_name,
                        "series_dir": str(pdir),
                        "native_normalized_pitch": round(
                            float(geom.trajectory_metrics()["normalized_pitch"]), 5
                        ),
                        "z_slices_mm": [round(float(z), 3) for z in z_req],
                        "sample_count": len(s_list),
                    }
                )

        audit = verify_zero_leakage_split(
            train_samples,
            test_samples,
            split_mode="patient_split",
        )
        return {
            "split_mode": "patient_split",
            "train_samples": train_samples,
            "test_samples": test_samples,
            "zero_leakage_audit": audit,
            "case_metadata": case_metadata,
        }

    # active_mode == "axial_guardband"
    slab_half_used = 18.0
    for cid, pdir, idir in targets:
        geom, files = scan_projection_series(
            pdir, workers=max(workers, 4), show_progress=show_progress
        )
        case_fov, c_xy = _resolve_case_fov_and_center(geom, idir)
        z_min_raw = float(np.min(geom.z0_mm))
        z_max_raw = float(np.max(geom.z0_mm))
        plan = plan_axial_guardband_slices(
            z_min_raw,
            z_max_raw,
            float(geom.coverage_isocenter_mm),
            n_train_slices=int(n_train_slices_per_case),
            n_test_slices=int(n_test_slices_per_case),
            min_guardband_mm=float(min_guardband_mm),
        )
        slab_half_used = float(plan["itf_slab_half_span_mm"])
        z_train_arr = np.asarray(plan["train_z_slices_mm"], dtype=np.float64)
        z_test_arr = np.asarray(plan["test_z_slices_mm"], dtype=np.float64)

        tr_s = build_hybrid_series_dataset_streaming(
            files,
            z_train_arr,
            case_id=cid,
            geometry=geom,
            pitches=pitches,
            scenarios=scenarios,
            image_size=image_size,
            fov_mm=case_fov,
            center_xy_mm=c_xy,
            filter_name=fname,
            cutoff_ratio=cutoff_ratio,
            inpaint_mode=inpaint_mode,
            itf_slab_slices=itf_slab_slices,
            itf_slab_half_span_mm=slab_half_used,
            backend=backend,
            workers=workers,
            show_progress=show_progress,
        )
        te_s = build_hybrid_series_dataset_streaming(
            files,
            z_test_arr,
            case_id=cid,
            geometry=geom,
            pitches=pitches,
            scenarios=scenarios,
            image_size=image_size,
            fov_mm=case_fov,
            center_xy_mm=c_xy,
            filter_name=fname,
            cutoff_ratio=cutoff_ratio,
            inpaint_mode=inpaint_mode,
            itf_slab_slices=itf_slab_slices,
            itf_slab_half_span_mm=slab_half_used,
            backend=backend,
            workers=workers,
            show_progress=show_progress,
        )
        train_samples.extend(tr_s)
        test_samples.extend(te_s)
        case_metadata.append(
            {
                "case_id": cid,
                "split_role": "axial_guardband_train_and_test",
                "series_dir": str(pdir),
                "native_normalized_pitch": round(
                    float(geom.trajectory_metrics()["normalized_pitch"]), 5
                ),
                "train_z_slices_mm": plan["train_z_slices_mm"],
                "test_z_slices_mm": plan["test_z_slices_mm"],
                "actual_guardband_mm": plan["actual_guardband_mm"],
                "itf_slab_half_span_mm": plan["itf_slab_half_span_mm"],
                "itf_slab_gap_mm": plan["itf_slab_gap_mm"],
                "train_sample_count": len(tr_s),
                "test_sample_count": len(te_s),
            }
        )

    audit = verify_zero_leakage_split(
        train_samples,
        test_samples,
        split_mode="axial_guardband",
        min_guardband_mm=float(min_guardband_mm),
        itf_slab_half_span_mm=slab_half_used,
    )
    return {
        "split_mode": "axial_guardband",
        "train_samples": train_samples,
        "test_samples": test_samples,
        "zero_leakage_audit": audit,
        "case_metadata": case_metadata,
    }


def run_generalization_experiment(
    train_samples: Sequence[DualDomainSample],
    test_samples: Sequence[DualDomainSample],
    *,
    split_mode: str = "axial_guardband",
    min_guardband_mm: float = 40.0,
    itf_slab_half_span_mm: float = 18.0,
    sino_base_channels: int = 64,
    img_base_channels: int = 64,
    filter_name: str = "ram_lak",
    cutoff_ratio: float = 1.0,
    epochs: int = 20,
    lr: float = 2.0e-3,
    itf_ratio: float = DEFAULT_ITF_TRAINING_RATIO,
    backend: str = "torch",
    seed: int = 42,
    sino_pretrain_epochs: int | None = None,
    img_pretrain_epochs: int | None = None,
    joint_finetune_epochs: int | None = None,
) -> dict[str, Any]:
    """Execute a complete Stage 6 Zero-Leakage Training + Held-Out Generalization Evaluation.

    1. Verifies zero data leakage between `train_samples` and `test_samples`.
    2. Instantiates and trains `DualDomainHighPitchNet` exclusively on `train_samples` using the
       3-stage sequential curriculum (`Eq. 1 -> Eq. 2 -> Eq. 3`).
    3. Freezes the trained model and evaluates zero-shot on `test_samples` without any test-set
       adaptation.
    4. Returns both training and held-out test summaries, predictions, and generalization gap metrics.
    """
    audit = verify_zero_leakage_split(
        train_samples,
        test_samples,
        split_mode=split_mode,
        min_guardband_mm=min_guardband_mm,
        itf_slab_half_span_mm=itf_slab_half_span_mm,
    )

    fname = normalize_filter_name(filter_name)
    model, device = create_dual_domain_model(
        sino_base_channels=sino_base_channels,
        img_base_channels=img_base_channels,
        filter_name=fname,
        cutoff_ratio=cutoff_ratio,
        backend=backend,
        seed=seed,
    )

    train_res = train_dual_domain_model(
        model,
        train_samples,
        device=device,
        epochs=epochs,
        lr=lr,
        itf_ratio=itf_ratio,
        seed=seed,
        sino_pretrain_epochs=sino_pretrain_epochs,
        img_pretrain_epochs=img_pretrain_epochs,
        joint_finetune_epochs=joint_finetune_epochs,
    )
    train_preds: list[DualDomainPrediction] = train_res["predictions"]
    train_summary = train_res["summary"]

    # Frozen zero-shot out-of-sample evaluation on held-out test_samples
    test_preds: list[DualDomainPrediction] = [
        evaluate_dual_domain_sample(model, s, device=device) for s in test_samples
    ]
    test_summary = summarize_dual_domain_predictions(
        test_preds,
        itf_ratio=itf_ratio,
        training_history=None,
    )

    tr_prog = train_summary["overall_progression"]
    te_prog = test_summary["overall_progression"]

    te_raw_rmse = float(te_prog["stage0_raw_nofill"]["rmse_hu"])
    te_pre_rmse = float(te_prog["stage1_prelim_fbp"]["rmse_hu"])
    te_d1_rmse = float(te_prog["stage2_domain1_sinonet_fbp"]["rmse_hu"])
    te_fin_rmse = float(te_prog["stage3_final_dual_domain"]["rmse_hu"])
    te_pre_ssim = float(te_prog["stage1_prelim_fbp"]["ssim"])
    te_fin_ssim = float(te_prog["stage3_final_dual_domain"]["ssim"])
    te_meas_dev = float(te_prog["sinogram_progression"]["measured_rays_max_abs_dev"])

    tr_fin_rmse = float(tr_prog["stage3_final_dual_domain"]["rmse_hu"])

    unseen_monotonic_pass = bool(
        te_fin_rmse <= te_d1_rmse + 0.15
        and te_fin_rmse < te_pre_rmse
        and te_pre_rmse < te_raw_rmse
        and te_fin_ssim >= te_pre_ssim - 1e-4
        and te_meas_dev == 0.0
    )

    gen_metrics = {
        "split_mode": audit.split_mode,
        "zero_leakage_verified": bool(audit.overall_zero_leakage_pass),
        "train_sample_count": len(train_samples),
        "unseen_test_sample_count": len(test_samples),
        "train_final_rmse_hu": round(tr_fin_rmse, 4),
        "unseen_test_raw_nofill_rmse_hu": round(te_raw_rmse, 4),
        "unseen_test_prelim_rmse_hu": round(te_pre_rmse, 4),
        "unseen_test_domain1_dc_rmse_hu": round(te_d1_rmse, 4),
        "unseen_test_final_dual_rmse_hu": round(te_fin_rmse, 4),
        "unseen_test_rmse_reduction_vs_prelim_hu": round(te_pre_rmse - te_fin_rmse, 4),
        "unseen_test_rmse_reduction_vs_raw_hu": round(te_raw_rmse - te_fin_rmse, 4),
        "unseen_test_prelim_ssim": round(te_pre_ssim, 6),
        "unseen_test_final_dual_ssim": round(te_fin_ssim, 6),
        "unseen_test_ssim_gain_vs_prelim": round(te_fin_ssim - te_pre_ssim, 6),
        "unseen_test_measured_rays_max_abs_dev": te_meas_dev,
        "unseen_test_monotonic_pass": unseen_monotonic_pass,
    }

    return {
        "model": model,
        "device": device,
        "zero_leakage_audit": audit.to_dict(),
        "generalization_metrics": gen_metrics,
        "train_summary": train_summary,
        "unseen_test_summary": test_summary,
        "train_predictions": train_preds,
        "unseen_test_predictions": test_preds,
    }
