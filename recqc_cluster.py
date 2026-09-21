"""ReCQC 3.0 clustering helpers.

This module contains the non-GUI backend used by the new ReCQC clustering module.
It intentionally does not import or reuse the standalone legacy GUI.  It only reuses the
SPTC/PMTC algorithmic backend for 13C clustering and provides an HSQC extension that keeps
the same logic: cross-sample candidate tracks -> score/set-pack tracks -> PMTC-style
presence-mask/trend partitioning, with QG-PMTC guarded refinement when selected.

Supported input formats
-----------------------
13C clustering files:
    CPPM,intensity[,area]
    ppm,intensity[,area]
    or no-header numeric text: Cppm intensity [area]

HSQC clustering files:
    CPPM,HPPM,intensity[,area]
    c_ppm,h_ppm,intensity[,area]
    or no-header numeric text: Cppm Hppm intensity [area]

The HSQC input is therefore exactly the 13C clustering peak-list format with one extra
H-shift column immediately after CPPM, as requested for ReCQC 3.0.
"""
from __future__ import annotations

import csv
import io
import math
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from services.continuous_sample_correction import maybe_prepare_corrected_samples
from single_spectrum.pipeline import (
    SingleSpectrumCalibrationConfig,
    SingleSpectrumGMMConfig,
    SingleSpectrumPipelineConfig,
    run_single_spectrum_gmm,
    run_single_spectrum_pipeline,
)


DEFAULT_REGION_WINDOWS: List[Tuple[float, float, float]] = [
    (0.0, 220.0, 0.80),
]


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "original_t": "TTC",
    "v4_baseline": "SPTC",
    "v5_pmtc": "PMTC",
    "v5_pmtc_r85": "PMTC",
    "v5_enum": "QG-PMTC",
    "v5_enum_mask": "DMP",
    "v5_frontend_mask": "SP-Mask",
    "v5_pmtc_mask": "SP-Mask",
    "v5_mask": "SP-Mask",
}

MODEL_FULL_NAMES: Dict[str, str] = {
    "original_t": "tolerance-based trajectory clustering",
    "v4_baseline": "set-packing trajectory clustering",
    "v5_pmtc": "presence-mask and trend clustering",
    "v5_pmtc_r85": "presence-mask and trend clustering",
    "v5_enum": "quality-guarded PMTC",
    "v5_enum_mask": "direct mask-pooling ablation",
    "v5_frontend_mask": "set-packing mask-only ablation",
    "v5_pmtc_mask": "set-packing mask-only ablation",
    "v5_mask": "set-packing mask-only ablation",
}

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "original_t": "ppm tolerance-driven early trajectory connection",
    "v4_baseline": "seed-based candidate trajectory generation and conflict-constrained set-packing",
    "v5_pmtc": "presence-mask binning and intensity-profile-based bucket splitting",
    "v5_pmtc_r85": "presence-mask binning and intensity-profile-based bucket splitting",
    "v5_enum": "SPTC front end, PMTC initial labels and quality-guarded merge/split refinement",
    "v5_enum_mask": "early direct mask-pooling output without PMTC/guarded refinement",
    "v5_frontend_mask": "set-packing front end followed only by presence-mask grouping",
    "v5_pmtc_mask": "set-packing front end followed only by presence-mask grouping",
    "v5_mask": "set-packing front end followed only by presence-mask grouping",
}

MODEL_DISPLAY_ALIASES: Dict[str, str] = {
    "ttc": "original_t",
    "tolerance_based_trajectory_clustering": "original_t",
    "sptc": "v4_baseline",
    "set_packing_trajectory_clustering": "v4_baseline",
    "pmtc": "v5_pmtc",
    "presence_mask_and_trend_clustering": "v5_pmtc",
    "qg_pmtc": "v5_enum",
    "quality_guarded_pmtc": "v5_enum",
    "sp_mask": "v5_frontend_mask",
    "set_packing_mask_only_ablation": "v5_frontend_mask",
    "dmp": "v5_enum_mask",
    "direct_mask_pooling_ablation": "v5_enum_mask",
}


def normalize_model_key(model: str | None) -> str:
    key = str(model or "v5_pmtc").strip().lower().replace("-", "_").replace(" ", "_")
    return MODEL_DISPLAY_ALIASES.get(key, key)


def model_display_label(model: str | None, *, include_full_name: bool = True) -> str:
    key = normalize_model_key(model)
    short = MODEL_DISPLAY_NAMES.get(key, str(model or key))
    full = MODEL_FULL_NAMES.get(key, "")
    if include_full_name and full:
        return f"{short} ({full})"
    return short


def model_key_from_display(value: str | None) -> str:
    if value is None:
        return "v5_enum"
    raw = str(value).strip()
    key = normalize_model_key(raw.split("(", 1)[0].strip())
    if key != raw.lower().replace("-", "_").replace(" ", "_"):
        return key
    for model_key, label in MODEL_DISPLAY_NAMES.items():
        if raw == label or raw.startswith(f"{label} "):
            return model_key
    return normalize_model_key(raw)


@dataclass
class ClusterBlock:
    cluster_id: str
    kind: str  # "C" or "HSQC"
    values: List[float] | List[Tuple[float, float]]
    n_tracks: int
    presence_mask: str = ""
    details: str = ""


@dataclass
class HSQCPeak:
    peak_id: str
    sample_id: str
    c_ppm: float
    h_ppm: float
    intensity: float = 1.0
    area: Optional[float] = None


@dataclass
class HSQCTrack:
    track_id: str
    members: Dict[str, HSQCPeak]
    center_c: float
    center_h: float
    c_span: float
    h_span: float
    score: float
    reciprocal_fraction: float = 0.0


# ----------------------------- parsing / formatting -----------------------------

def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def format_c_values(values: Iterable[float]) -> str:
    vals = sorted(float(v) for v in values)
    return ", ".join(f"{v:.3f}" for v in vals)


def format_hsqc_points(points: Iterable[Tuple[float, float]]) -> str:
    vals = sorted((float(c), float(h)) for c, h in points)
    return ", ".join(f"({c:.3f}, {h:.3f})" for c, h in vals)


def parse_c_values(text: str) -> List[float]:
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace("，", ","))
    return [float(x) for x in nums]


def parse_hsqc_points(text: str) -> List[List[float]]:
    text = text.replace("（", "(").replace("）", ")").replace("，", ",")
    pairs = re.findall(
        r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)",
        text,
    )
    if pairs:
        return [[float(c), float(h)] for c, h in pairs]
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]


def parse_region_windows(text: str) -> List[Tuple[float, float, float]]:
    """Parse GUI region tolerance text.

    Accepted lines include:
        0-50:0.20
        0,50,0.20
        0 50 0.20
    """
    regions: List[Tuple[float, float, float]] = []
    for raw in text.replace("，", ",").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";", "//")):
            continue
        range_match = re.match(
            r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*-\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*[:=,;\s]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$",
            line,
        )
        if range_match:
            nums = [float(x) for x in range_match.groups()]
        else:
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
        if len(nums) >= 3:
            lo, hi, tol = nums[0], nums[1], nums[2]
            if hi <= lo or tol <= 0:
                raise ValueError(f"Invalid ppm region tolerance line: {raw}")
            regions.append((lo, hi, tol))
    if not regions:
        raise ValueError("No valid ppm region tolerances were found.")
    regions.sort(key=lambda x: x[0])
    return regions


def _normalize_col(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace(" ", "_").replace("-", "_").replace("/", "_")
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    s = s.replace("δ", "delta")
    return s


def _find_key(keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm = {_normalize_col(k): k for k in keys}
    for cand in candidates:
        cand_n = _normalize_col(cand)
        if cand_n in norm:
            return norm[cand_n]
    # also allow loose contains matching for exported tables such as "C ppm".
    for cand in candidates:
        cand_n = _normalize_col(cand)
        for nk, original in norm.items():
            if cand_n and (cand_n in nk or nk in cand_n):
                return original
    return None


def _tokenize_line(line: str) -> List[str]:
    return [t for t in re.split(r"[,;\t\s]+", line.strip()) if t != ""]


def _read_delimited_rows(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith(("#", ";", "//"))]
    if not lines:
        return []

    first_tokens = _tokenize_line(lines[0])
    first_has_header = any(re.search(r"[A-Za-zΑ-ωµμδ]|ppm|int|height|area|cppm|hppm", tok, re.I) for tok in first_tokens)

    if first_has_header:
        header = first_tokens
        rows: List[dict] = []
        for ln in lines[1:]:
            toks = _tokenize_line(ln)
            if not toks:
                continue
            rows.append({header[i]: toks[i] for i in range(min(len(header), len(toks)))})
        return rows

    # no header: expose positional columns col1, col2, col3 ...
    rows = []
    for ln in lines:
        nums = []
        for t in _tokenize_line(ln):
            val = _safe_float(t)
            if val is not None:
                nums.append(val)
        if nums:
            rows.append({f"col{i + 1}": nums[i] for i in range(len(nums))})
    return rows


def _normalized_c_peaklist(path: str, out_path: Path) -> None:
    rows = _read_delimited_rows(Path(path))
    if not rows:
        raise ValueError(f"No peaks were read from {Path(path).name}")
    keys = list(rows[0].keys())
    c_key = _find_key(keys, [
        "CPPM", "C_PPM", "C ppm", "c_ppm", "ppm_c", "carbon_ppm", "carbon", "13c", "13c_ppm",
        "ppm", "shift", "chemical_shift", "position", "delta", "col1",
    ])
    i_key = _find_key(keys, ["intensity", "height", "amp", "amplitude", "peak_height", "area", "integral", "volume", "col2"])
    a_key = _find_key(keys, ["area", "integral", "volume", "col3"])
    if c_key is None:
        raise ValueError(f"Cannot find C/CPPM column in {Path(path).name}.")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ppm", "intensity", "area"])
        w.writeheader()
        for row in rows:
            c = _safe_float(row.get(c_key))
            if c is None:
                continue
            inten = _safe_float(row.get(i_key)) if i_key else None
            area = _safe_float(row.get(a_key)) if a_key else None
            if inten is None and area is not None:
                inten = area
            if area is None and inten is not None:
                area = inten
            w.writerow({"ppm": c, "intensity": inten if inten is not None else 1.0, "area": area if area is not None else (inten if inten is not None else 1.0)})


# ----------------------------- 13C / bundled V5 backend -----------------------------

def _build_v5_config(
    *,
    model: str = "v5_pmtc",
    region_windows: Optional[List[Tuple[float, float, float]]] = None,
    span_tol: float = 1.00,
    residual_gate: float = 0.15,
    top_k_per_seed: int = 5,
    max_per_sample: int = 3,
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None,
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None,
    pmtc_min_cluster_size: int = 3,
    setpacking_model_cost: float = 1.10,
    setpacking_high_mask_bonus: float = 0.0,
    guarded_quality_max_cluster_rise: Optional[int] = None,
):
    from nmr_trendtrack.config import AppConfig

    config = AppConfig()
    config.model.name = normalize_model_key(model or "v5_pmtc")
    config.align.ppm_window_by_region = list(region_windows or DEFAULT_REGION_WINDOWS)
    config.align.max_track_span_ppm = float(span_tol)
    config.model.residual_gate = float(residual_gate)
    config.model.top_k_per_seed = int(top_k_per_seed)
    config.model.max_per_sample = int(max_per_sample)
    config.model.exact_limit = int(exact_limit)
    config.model.beam_width = int(beam_width)
    config.model.node_limit = int(node_limit)
    config.model.pmtc_min_cluster_size = int(pmtc_min_cluster_size)
    config.model.setpacking_model_cost = float(setpacking_model_cost)
    config.model.setpacking_high_mask_bonus = float(setpacking_high_mask_bonus)
    if guarded_quality_max_cluster_rise is not None:
        config.model.guarded_quality_max_cluster_rise = int(guarded_quality_max_cluster_rise)
    if pmtc_max_tracks_by_n_samples:
        config.model.pmtc_max_tracks_by_n_samples = {int(k): int(v) for k, v in pmtc_max_tracks_by_n_samples.items()}
    if pmtc_frac_limit_by_n_samples:
        config.model.pmtc_frac_limit_by_n_samples = {int(k): float(v) for k, v in pmtc_frac_limit_by_n_samples.items()}
    return config


def _run_switchable_model_global_strict(samples, config):
    """Run the switchable 13C pipeline with the strict global set-packing front end."""
    from nmr_trendtrack.models import run_switchable_model
    from nmr_trendtrack.models.three_model_pipeline import _v4_modules

    _base, gsp = _v4_modules()
    missing = object()
    original_pack = getattr(gsp, "mask_stratified_global_set_packing", missing)
    try:
        gsp.mask_stratified_global_set_packing = gsp.componentwise_global_set_packing
        return run_switchable_model(samples, config)
    finally:
        if original_pack is missing:
            try:
                delattr(gsp, "mask_stratified_global_set_packing")
            except AttributeError:
                pass
        else:
            gsp.mask_stratified_global_set_packing = original_pack


def run_c_v5_clustering(
    sample_files: Sequence[str],
    model: str = "v5_pmtc",
    *,
    region_windows: Optional[List[Tuple[float, float, float]]] = None,
    span_tol: float = 1.20,
    residual_gate: float = 1.00,
    top_k_per_seed: int = 6,
    max_per_sample: int = 3,
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None,
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None,
    pmtc_min_cluster_size: int = 3,
    setpacking_model_cost: float = 1.10,
    setpacking_high_mask_bonus: float = 0.10,
    guarded_quality_max_cluster_rise: int = 1,
    correction_options: Optional[dict] = None,
    output_dir: Optional[str | Path] = None,
) -> List[ClusterBlock]:
    """Run the selected 13C cross-spectrum backend and convert its state to editable C blocks."""
    if len(sample_files) < 2:
        raise ValueError("Cross-spectrum clustering needs at least two sample peak-list files.")
    if span_tol <= 0 or residual_gate <= 0:
        raise ValueError("C span tolerance and residual gate must be positive.")

    from nmr_trendtrack.contracts import Sample

    with tempfile.TemporaryDirectory(prefix="recqc30_ccluster_") as tmp:
        tmpdir = Path(tmp)
        norm_files: List[Path] = []
        for i, path in enumerate(sample_files, 1):
            out = tmpdir / f"S{i}.csv"
            _normalized_c_peaklist(path, out)
            norm_files.append(out)

        samples = [
            Sample(sample_id=f"S{i + 1}", order_index=i, source_type="peaklist", peaklist_path=str(path))
            for i, path in enumerate(norm_files)
        ]
        correction_note = "correction=off"
        if correction_options and correction_options.get("enabled"):
            runtime_dir = Path(output_dir) / "cross_sample_correction" if output_dir else tmpdir / "cross_sample_correction"
            correction_result = maybe_prepare_corrected_samples(
                samples=[
                    {
                        "sample_id": s.sample_id,
                        "order_index": s.order_index,
                        "source_type": s.source_type,
                        "peaklist_path": s.peaklist_path,
                    }
                    for s in samples
                ],
                runtime_dir=runtime_dir,
                options=correction_options,
            )
            samples = [
                Sample(
                    sample_id=str(row["sample_id"]),
                    order_index=int(row.get("order_index", i)),
                    source_type=str(row.get("source_type", "peaklist")),
                    peaklist_path=str(row["peaklist_path"]),
                )
                for i, row in enumerate(correction_result.samples)
            ]
            meta = correction_result.metadata or {}
            correction_note = (
                f"correction=on; mode={meta.get('calibration_mode', '')}; "
                f"model={correction_result.model_path}"
            )
        config = _build_v5_config(
            model=model,
            region_windows=region_windows,
            span_tol=span_tol,
            residual_gate=residual_gate,
            top_k_per_seed=top_k_per_seed,
            max_per_sample=max_per_sample,
            exact_limit=exact_limit,
            beam_width=beam_width,
            node_limit=node_limit,
            pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples,
            pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples,
            pmtc_min_cluster_size=pmtc_min_cluster_size,
            setpacking_model_cost=setpacking_model_cost,
            setpacking_high_mask_bonus=setpacking_high_mask_bonus,
            guarded_quality_max_cluster_rise=guarded_quality_max_cluster_rise,
        )
        state = _run_switchable_model_global_strict(samples, config)

    membership = {m.track_id: (m.final_cluster_id or m.component_cluster_id or m.best_cluster_id or "C_unassigned") for m in state.memberships}
    groups: Dict[str, List] = defaultdict(list)
    for track in state.tracks:
        groups[membership.get(track.track_id, "C_unassigned")].append(track)

    blocks: List[ClusterBlock] = []
    ordered_ids = list(state.ordered_sample_ids)
    sample_order_txt = ",".join(ordered_ids)
    region_txt = "; ".join(f"{lo:g}-{hi:g}:{win:g}" for lo, hi, win in (region_windows or DEFAULT_REGION_WINDOWS))
    for cid in sorted(groups, key=lambda x: (str(x).lower().endswith("unassigned"), str(x))):
        tracks = groups[cid]
        vals = [float(t.center_ppm) for t in tracks]
        mask_counts: Dict[str, int] = defaultdict(int)
        for t in tracks:
            mask = "".join("1" if sid in t.members else "0" for sid in ordered_ids)
            mask_counts[mask] += 1
        mask = max(mask_counts.items(), key=lambda kv: kv[1])[0] if mask_counts else ""
        normalized_model = normalize_model_key(model)
        if normalized_model in {"v5_enum_mask", "v5_enumerated_mask", "enumerated_v5_mask"}:
            backend_note = model_display_label("v5_enum_mask")
        elif normalized_model in {"v5_enum", "v5_enumerated", "enumerated_v5"}:
            backend_note = model_display_label("v5_enum")
        elif normalized_model in {"v4", "v4_baseline", "baseline"}:
            backend_note = model_display_label("v4_baseline")
        elif normalized_model in {"original", "original_t", "t", "t_mixture"}:
            backend_note = model_display_label("original_t")
        else:
            backend_note = model_display_label("v5_pmtc")
        mask_counts_txt = ",".join(f"{m}:{n}" for m, n in sorted(mask_counts.items(), reverse=True))
        details = (
            f"{backend_note} frontend=global_strict; tracks={len(tracks)}; dominant_mask={mask}; "
            f"sample_order={sample_order_txt}; mask_counts={mask_counts_txt}; "
            f"region_tol=[{region_txt}]; span<={span_tol:g}; residual_gate={residual_gate:g}"
        )
        details = f"{details}; {correction_note}"
        blocks.append(ClusterBlock(cluster_id=str(cid), kind="C", values=vals, n_tracks=len(tracks), presence_mask=mask, details=details))
    return blocks


# ----------------------------- HSQC / PMTC-style backend -----------------------------

def load_hsqc_peaklist(path: str, sample_id: str) -> List[HSQCPeak]:
    p = Path(path)
    rows = _read_delimited_rows(p)
    if not rows:
        return []
    keys = list(rows[0].keys())
    c_key = _find_key(keys, [
        "CPPM", "C_PPM", "C ppm", "c_ppm", "ppm_c", "cshift", "c_shift", "carbon", "carbon_ppm",
        "13c", "13c_ppm", "f1ppm", "f1", "x", "c", "col1",
    ])
    h_key = _find_key(keys, [
        "HPPM", "H_PPM", "H ppm", "h_ppm", "ppm_h", "hshift", "h_shift", "proton", "proton_ppm",
        "1h", "1h_ppm", "f2ppm", "f2", "y", "h", "col2",
    ])
    i_key = _find_key(keys, ["intensity", "height", "amp", "amplitude", "peak_height", "area", "integral", "volume", "col3"])
    a_key = _find_key(keys, ["area", "integral", "volume", "col4"])
    if c_key is None or h_key is None:
        raise ValueError(
            f"Cannot find HSQC C/H columns in {p.name}. Use CPPM,HPPM,intensity or c_ppm,h_ppm,intensity."
        )

    peaks: List[HSQCPeak] = []
    for idx, row in enumerate(rows, 1):
        c = _safe_float(row.get(c_key))
        h = _safe_float(row.get(h_key))
        if c is None or h is None:
            continue
        inten = _safe_float(row.get(i_key)) if i_key else None
        area = _safe_float(row.get(a_key)) if a_key else None
        if inten is None and area is not None:
            inten = area
        peaks.append(
            HSQCPeak(
                peak_id=f"{sample_id}_P{idx:04d}",
                sample_id=sample_id,
                c_ppm=float(c),
                h_ppm=float(h),
                intensity=float(inten if inten is not None else 1.0),
                area=float(area) if area is not None else None,
            )
        )
    peaks.sort(key=lambda x: (x.c_ppm, x.h_ppm))
    return peaks


def _hsqc_signal_value(peak: HSQCPeak) -> float:
    value = _safe_float(getattr(peak, "intensity", None))
    if value is not None:
        return float(value)
    area = _safe_float(getattr(peak, "area", None))
    return float(area) if area is not None else 1.0


def _apply_hsqc_intensity_correction(
    peaks_by_sample: Dict[str, List[HSQCPeak]],
    sample_ids: Sequence[str],
    correction_options: Optional[dict],
    output_dir: Optional[str | Path],
) -> str:
    if not correction_options or not correction_options.get("enabled"):
        return "correction=off"

    runtime_dir = Path(output_dir) / "hsqc_intensity_correction" if output_dir else Path(tempfile.mkdtemp(prefix="recqc30_hsqc_corr_"))
    input_root = runtime_dir / "hsqc_correction_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, sid in enumerate(sample_ids):
        path = input_root / f"{sid}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ppm", "intensity", "area", "width_hz"])
            w.writeheader()
            for peak in peaks_by_sample.get(sid, []):
                w.writerow({
                    "ppm": peak.c_ppm,
                    "intensity": _hsqc_signal_value(peak),
                    "area": "" if peak.area is None else peak.area,
                    "width_hz": 1.0,
                })
        samples.append({"sample_id": sid, "order_index": i, "source_type": "peaklist", "peaklist_path": str(path)})

    result = maybe_prepare_corrected_samples(samples=samples, runtime_dir=runtime_dir, options=correction_options)
    for sample in result.samples:
        sid = str(sample["sample_id"])
        corrected_rows = _read_delimited_rows(Path(str(sample["peaklist_path"])))
        for peak, row in zip(peaks_by_sample.get(sid, []), corrected_rows):
            corrected = _safe_float(row.get("intensity"))
            if corrected is not None:
                setattr(peak, "intensity_raw", peak.intensity)
                setattr(peak, "correction_delta", _safe_float(row.get("delta")))
                peak.intensity = float(corrected)

    meta = result.metadata or {}
    return (
        f"correction=on; mode={meta.get('calibration_mode', '')}; "
        f"used_area={meta.get('used_area', '')}; model={result.model_path}"
    )


def _hsqc_dist(a: HSQCPeak, b: HSQCPeak, c_tol: float, h_tol: float) -> float:
    return math.sqrt(((a.c_ppm - b.c_ppm) / max(c_tol, 1e-12)) ** 2 + ((a.h_ppm - b.h_ppm) / max(h_tol, 1e-12)) ** 2)


def _best_match(source: HSQCPeak, target_peaks: Sequence[HSQCPeak], c_tol: float, h_tol: float) -> Optional[HSQCPeak]:
    opts = []
    for p in target_peaks:
        if abs(p.c_ppm - source.c_ppm) <= c_tol and abs(p.h_ppm - source.h_ppm) <= h_tol:
            opts.append((_hsqc_dist(source, p, c_tol, h_tol), p))
    return min(opts, key=lambda x: x[0])[1] if opts else None


def _candidate_track_from_seed(
    seed: HSQCPeak,
    peaks_by_sample: Dict[str, List[HSQCPeak]],
    sample_ids: Sequence[str],
    *,
    c_tol: float,
    h_tol: float,
    c_span_tol: float,
    h_span_tol: float,
    min_track_size: int,
    candidate_min_score: float,
    pair_score_weight: float,
    reciprocal_best_bonus: float,
) -> Optional[HSQCTrack]:
    members: Dict[str, HSQCPeak] = {}
    dists = []
    reciprocal_hits = 0
    pair_tests = 0

    for sid in sample_ids:
        best = _best_match(seed, peaks_by_sample.get(sid, []), c_tol, h_tol)
        if best is None:
            continue
        members[sid] = best
        dists.append(_hsqc_dist(seed, best, c_tol, h_tol))
        if sid != seed.sample_id:
            pair_tests += 1
            back = _best_match(best, peaks_by_sample.get(seed.sample_id, []), c_tol, h_tol)
            if back is not None and back.peak_id == seed.peak_id:
                reciprocal_hits += 1

    if len(members) < max(2, int(min_track_size)):
        return None
    c_vals = [p.c_ppm for p in members.values()]
    h_vals = [p.h_ppm for p in members.values()]
    c_span = max(c_vals) - min(c_vals) if len(c_vals) > 1 else 0.0
    h_span = max(h_vals) - min(h_vals) if len(h_vals) > 1 else 0.0
    if c_span > c_span_tol or h_span > h_span_tol:
        return None

    center_c = sum(c_vals) / len(c_vals)
    center_h = sum(h_vals) / len(h_vals)
    coverage = len(members) / max(1, len(sample_ids))
    closeness = 1.0 / (1.0 + (sum(dists) / max(1, len(dists))))
    reciprocal_fraction = reciprocal_hits / max(1, pair_tests)
    span_penalty = 0.18 * (c_span / max(c_span_tol, 1e-12) + h_span / max(h_span_tol, 1e-12))
    score = coverage + pair_score_weight * 0.25 * closeness + reciprocal_best_bonus * reciprocal_fraction - span_penalty
    if score < candidate_min_score:
        return None
    return HSQCTrack(
        track_id="",
        members=members,
        center_c=center_c,
        center_h=center_h,
        c_span=c_span,
        h_span=h_span,
        score=score,
        reciprocal_fraction=reciprocal_fraction,
    )


def build_hsqc_tracks(
    peaks_by_sample: Dict[str, List[HSQCPeak]],
    sample_ids: Sequence[str],
    *,
    c_tol: float = 1.0,
    h_tol: float = 0.1,
    c_span_tol: float = 1.0,
    h_span_tol: float = 0.1,
    min_track_size: int = 2,
    candidate_min_score: float = 0.40,
    pair_score_weight: float = 1.20,
    reciprocal_best_bonus: float = 0.18,
    keep_singletons: bool = True,
) -> List[HSQCTrack]:
    candidates: Dict[Tuple[str, ...], HSQCTrack] = {}
    for sid in sample_ids:
        for seed in peaks_by_sample.get(sid, []):
            cand = _candidate_track_from_seed(
                seed,
                peaks_by_sample,
                sample_ids,
                c_tol=c_tol,
                h_tol=h_tol,
                c_span_tol=c_span_tol,
                h_span_tol=h_span_tol,
                min_track_size=min_track_size,
                candidate_min_score=candidate_min_score,
                pair_score_weight=pair_score_weight,
                reciprocal_best_bonus=reciprocal_best_bonus,
            )
            if cand is None:
                continue
            key = tuple(sorted(p.peak_id for p in cand.members.values()))
            old = candidates.get(key)
            if old is None or cand.score > old.score:
                candidates[key] = cand

    used: set[str] = set()
    selected: List[HSQCTrack] = []
    ordered_candidates = sorted(
        candidates.values(),
        key=lambda t: (len(t.members), t.score, t.reciprocal_fraction, -t.c_span, -t.h_span),
        reverse=True,
    )
    for cand in ordered_candidates:
        ids = {p.peak_id for p in cand.members.values()}
        if used & ids:
            continue
        selected.append(cand)
        used.update(ids)

    if keep_singletons:
        # Keep unmatched peaks visible so the user can still inspect/edit them, but they go to unassigned.
        for sid in sample_ids:
            for p in peaks_by_sample.get(sid, []):
                if p.peak_id not in used:
                    selected.append(
                        HSQCTrack(
                            track_id="",
                            members={sid: p},
                            center_c=p.c_ppm,
                            center_h=p.h_ppm,
                            c_span=0.0,
                            h_span=0.0,
                            score=0.0,
                            reciprocal_fraction=0.0,
                        )
                    )
                    used.add(p.peak_id)

    selected.sort(key=lambda t: (t.center_c, t.center_h))
    for i, t in enumerate(selected, 1):
        t.track_id = f"T{i:04d}"
    return selected


def _presence_mask_track(track: HSQCTrack, sample_ids: Sequence[str]) -> str:
    return "".join("1" if sid in track.members else "0" for sid in sample_ids)


def _track_log_values(track: HSQCTrack, sample_ids: Sequence[str]) -> List[float]:
    vals: List[Optional[float]] = []
    for sid in sample_ids:
        p = track.members.get(sid)
        intensity = None if p is None else _hsqc_signal_value(p)
        vals.append(None if intensity is None else math.log(max(abs(float(intensity)), 1e-12)))
    obs = [v for v in vals if v is not None]
    med = median(obs) if obs else 0.0
    return [0.0 if v is None else float(v - med) for v in vals]


def _track_feature_hsqc(track: HSQCTrack, sample_ids: Sequence[str]) -> List[float]:
    v = _track_log_values(track, sample_ids)
    steps = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    return v + steps + [0.20 * track.center_c / 220.0, 0.20 * track.center_h / 12.0, 0.15 * min(1.5, track.score)]


def _euclid(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a)))


def _normalize_features(X: List[List[float]]) -> List[List[float]]:
    if not X:
        return X
    d = len(X[0])
    means = [sum(x[j] for x in X) / len(X) for j in range(d)]
    sds = []
    for j, m in enumerate(means):
        var = sum((x[j] - m) ** 2 for x in X) / max(1, len(X) - 1)
        sds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return [[(x[j] - means[j]) / sds[j] for j in range(d)] for x in X]


def _kmeans_labels(X: List[List[float]], k: int, max_iter: int = 40) -> List[int]:
    n = len(X)
    if k <= 1 or n <= 1:
        return [0] * n
    k = min(k, n)
    d = len(X[0])
    meanv = [sum(x[j] for x in X) / n for j in range(d)]
    first = min(range(n), key=lambda i: _euclid(X[i], meanv))
    centroids = [list(X[first])]
    while len(centroids) < k:
        far = max(range(n), key=lambda i: min(_euclid(X[i], c) for c in centroids))
        centroids.append(list(X[far]))
    labels = [-1] * n
    for _ in range(max_iter):
        new = [min(range(k), key=lambda c: _euclid(X[i], centroids[c])) for i in range(n)]
        if new == labels:
            break
        labels = new
        for c in range(k):
            ids = [i for i, lab in enumerate(labels) if lab == c]
            if ids:
                centroids[c] = [sum(X[i][j] for i in ids) / len(ids) for j in range(d)]
    remap = {old: i for i, old in enumerate(sorted(set(labels)))}
    return [remap[l] for l in labels]


def _split_hsqc_indices(indices: List[int], tracks: List[HSQCTrack], sample_ids: Sequence[str], k: int) -> List[List[int]]:
    if len(indices) <= 1 or k <= 1:
        return [indices]
    X = _normalize_features([_track_feature_hsqc(tracks[i], sample_ids) for i in indices])
    labels = _kmeans_labels(X, k)
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in zip(indices, labels):
        groups[lab].append(idx)
    out = [g for _, g in sorted(groups.items()) if g]
    if len(out) == 1 and k > 1 and len(indices) > 1:
        ordered = sorted(indices, key=lambda i: (tracks[i].center_c, tracks[i].center_h))
        chunk = max(1, math.ceil(len(ordered) / k))
        out = [ordered[i : i + chunk] for i in range(0, len(ordered), chunk)]
    return out


def _cluster_label_prefix(labels: Sequence[str]) -> str:
    for preferred in ("H", "C", "R", "M"):
        if any(str(label).startswith(preferred) for label in labels):
            return preferred
    return "H"


def _matched_low_indices_by_hsqc_greedy(
    low_indices: Sequence[int],
    high_indices: Sequence[int],
    tracks: Sequence[HSQCTrack],
    c_tol: float,
    h_tol: float,
) -> set[int]:
    pairs: List[Tuple[float, int, int]] = []
    c_scale = max(float(c_tol), 1e-12)
    h_scale = max(float(h_tol), 1e-12)
    for li in low_indices:
        low = tracks[li]
        for hi in high_indices:
            high = tracks[hi]
            dc = abs(float(low.center_c) - float(high.center_c))
            dh = abs(float(low.center_h) - float(high.center_h))
            if dc <= c_tol and dh <= h_tol:
                pairs.append((dc / c_scale + dh / h_scale, li, hi))
    pairs.sort()
    used_low: set[int] = set()
    used_high: set[int] = set()
    for _d, li, hi in pairs:
        if li in used_low or hi in used_high:
            continue
        used_low.add(li)
        used_high.add(hi)
    return used_low


def _apply_hsqc_mask_residual_filter(
    tracks: List[HSQCTrack],
    labels: Sequence[str],
    sample_ids: Sequence[str],
    *,
    c_tol: float,
    h_tol: float,
    min_remaining: int = 5,
    max_cover_ratio: float = 1.50,
    pool_single_sample_masks: bool = True,
) -> Tuple[List[HSQCTrack], List[str], Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, (track, label) in enumerate(zip(tracks, labels)):
        mask = _presence_mask_track(track, sample_ids)
        if pool_single_sample_masks and _mask_popcount(mask) == 1:
            key = (f"single_{mask}", mask)
        else:
            key = (str(label), mask)
        grouped[key].append(i)

    pieces = [
        {"source_label": key[0], "mask": key[1], "idxs": idxs}
        for key, idxs in grouped.items()
        if idxs
    ]
    pieces.sort(key=lambda p: (-_mask_popcount(str(p["mask"])), min(p["idxs"])))

    accepted: List[dict] = []
    dropped = 0
    matched_tracks = 0
    prefix = _cluster_label_prefix(labels)
    for piece in pieces:
        mask = str(piece["mask"])
        idxs = list(piece["idxs"])
        high_indices = [
            idx
            for acc in accepted
            if _mask_contains(str(acc["mask"]), mask)
            for idx in acc["idxs"]
        ]
        matched = _matched_low_indices_by_hsqc_greedy(idxs, high_indices, tracks, c_tol, h_tol) if high_indices else set()
        matched_tracks += len(matched)
        cover_size_ok = bool(high_indices) and len(high_indices) <= max_cover_ratio * max(1, len(idxs))
        if cover_size_ok and len(matched) == len(idxs):
            dropped += 1
            continue
        accepted.append({"mask": mask, "idxs": idxs, "source_label": piece["source_label"]})

    accepted.sort(key=lambda p: (-_mask_popcount(str(p["mask"])), min(p["idxs"])))
    output_label_by_index: Dict[int, str] = {}
    for ci, piece in enumerate(accepted, 1):
        label = f"{prefix}{ci:02d}"
        for idx in piece["idxs"]:
            output_label_by_index[idx] = label

    kept_indices = sorted(output_label_by_index)
    filtered_tracks = [tracks[i] for i in kept_indices]
    filtered_labels = [output_label_by_index[i] for i in kept_indices]
    return filtered_tracks, filtered_labels, {
        "hsqc_mask_residual_enabled": True,
        "hsqc_mask_residual_mode": "full_coverage_one_to_one",
        "hsqc_mask_residual_c_tol": float(c_tol),
        "hsqc_mask_residual_h_tol": float(h_tol),
        "hsqc_mask_residual_min_remaining": int(min_remaining),
        "hsqc_mask_residual_max_cover_ratio": float(max_cover_ratio),
        "hsqc_mask_residual_pool_single": bool(pool_single_sample_masks),
        "hsqc_mask_residual_input_clusters": len(pieces),
        "hsqc_mask_residual_output_clusters": len(accepted),
        "hsqc_mask_residual_dropped_clusters": dropped,
        "hsqc_mask_residual_matched_tracks": matched_tracks,
        "hsqc_mask_residual_input_tracks": len(tracks),
        "hsqc_mask_residual_output_tracks": len(filtered_tracks),
    }


def _apply_hsqc_sample_specific_residual_peak_filter(
    tracks: List[HSQCTrack],
    labels: Sequence[str],
    sample_ids: Sequence[str],
) -> Tuple[List[HSQCTrack], List[str], Dict[str, object]]:
    used_by_multi: Dict[str, set[str]] = defaultdict(set)
    for track in tracks:
        if _mask_popcount(_presence_mask_track(track, sample_ids)) <= 1:
            continue
        for sid, peak in track.members.items():
            if peak.peak_id:
                used_by_multi[str(sid)].add(str(peak.peak_id))

    filtered_tracks: List[HSQCTrack] = []
    filtered_labels: List[str] = []
    removed_single = 0
    kept_single = 0
    kept_by_sample: Dict[str, int] = defaultdict(int)
    removed_by_sample: Dict[str, int] = defaultdict(int)
    for track, label in zip(tracks, labels):
        mask = _presence_mask_track(track, sample_ids)
        if _mask_popcount(mask) == 1:
            sid = next((sample for sample in sample_ids if sample in track.members), None)
            if sid is not None:
                peak_id = str(track.members[sid].peak_id)
                if peak_id and peak_id in used_by_multi.get(str(sid), set()):
                    removed_single += 1
                    removed_by_sample[str(sid)] += 1
                    continue
                kept_single += 1
                kept_by_sample[str(sid)] += 1
        filtered_tracks.append(track)
        filtered_labels.append(str(label))

    return filtered_tracks, filtered_labels, {
        "hsqc_sample_specific_residual_enabled": True,
        "hsqc_sample_specific_residual_removed_single_tracks": removed_single,
        "hsqc_sample_specific_residual_kept_single_tracks": kept_single,
        "hsqc_sample_specific_residual_kept_by_sample": ",".join(f"{sid}:{kept_by_sample.get(sid, 0)}" for sid in sample_ids),
        "hsqc_sample_specific_residual_removed_by_sample": ",".join(f"{sid}:{removed_by_sample.get(sid, 0)}" for sid in sample_ids),
        "hsqc_sample_specific_residual_input_tracks": len(tracks),
        "hsqc_sample_specific_residual_output_tracks": len(filtered_tracks),
    }


def _pmtc_hsqc_labels(
    tracks: List[HSQCTrack],
    sample_ids: Sequence[str],
    *,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None,
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None,
    pmtc_min_cluster_size: int = 3,
) -> List[str]:
    n = len(tracks)
    ns = len(sample_ids)
    max_tracks_map = pmtc_max_tracks_by_n_samples or {3: 32, 4: 28, 5: 26}
    frac_map = pmtc_frac_limit_by_n_samples or {3: 0.55, 4: 0.47, 5: 0.41}
    max_tracks = int(max_tracks_map.get(ns, max_tracks_map.get(str(ns), 26)))
    frac_limit = float(frac_map.get(ns, frac_map.get(str(ns), 0.41)))
    min_cluster_size = int(pmtc_min_cluster_size)

    buckets: Dict[str, List[int]] = defaultdict(list)
    unassigned: List[int] = []
    for i, t in enumerate(tracks):
        if len(t.members) >= 2:
            buckets[_presence_mask_track(t, sample_ids)].append(i)
        else:
            unassigned.append(i)

    clusters: List[List[int]] = []
    for _, idxs in sorted(buckets.items()):
        if len(idxs) <= max(min_cluster_size, max_tracks):
            clusters.append(list(idxs))
        else:
            k = max(1, math.ceil(len(idxs) / max_tracks))
            clusters.extend(_split_hsqc_indices(list(idxs), tracks, sample_ids, k))

    max_size = max(max_tracks, int(math.ceil(frac_limit * max(n, 1))))
    for _ in range(20):
        changed = False
        new_clusters: List[List[int]] = []
        for c in clusters:
            if len(c) > max_size and len(c) > 2 * min_cluster_size:
                k = math.ceil(len(c) / max_size)
                parts = _split_hsqc_indices(c, tracks, sample_ids, k)
                if len(parts) == 1 and parts[0] == c:
                    new_clusters.append(c)
                else:
                    new_clusters.extend(parts)
                    changed = True
            else:
                new_clusters.append(c)
        clusters = new_clusters
        if not changed:
            break

    labels = ["HSQC_unassigned"] * n
    clusters = sorted([c for c in clusters if c], key=lambda c: min(c))
    for ci, c in enumerate(clusters, 1):
        for i in c:
            labels[i] = f"H{ci:02d}"
    for i in unassigned:
        labels[i] = "HSQC_unassigned"
    return labels


def run_hsqc_v5_style_clustering(
    sample_files: Sequence[str],
    *,
    c_tol: float = 1.0,
    h_tol: float = 0.1,
    c_span_tol: float = 1.0,
    h_span_tol: float = 0.1,
    min_track_size: int = 2,
    candidate_min_score: float = 0.40,
    pair_score_weight: float = 1.20,
    reciprocal_best_bonus: float = 0.18,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None,
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None,
    pmtc_min_cluster_size: int = 3,
) -> List[ClusterBlock]:
    if len(sample_files) < 2:
        raise ValueError("HSQC PMTC-style clustering needs at least two sample peak-list files.")
    if min(c_tol, h_tol, c_span_tol, h_span_tol) <= 0:
        raise ValueError("HSQC C/H tolerance and C/H span tolerances must be positive.")

    sample_ids = [f"S{i + 1}" for i in range(len(sample_files))]
    peaks_by_sample = {sid: load_hsqc_peaklist(path, sid) for sid, path in zip(sample_ids, sample_files)}
    if not any(peaks_by_sample.values()):
        raise ValueError("No HSQC peaks were read. Use CPPM,HPPM,intensity or c_ppm,h_ppm,intensity.")

    tracks = build_hsqc_tracks(
        peaks_by_sample,
        sample_ids,
        c_tol=c_tol,
        h_tol=h_tol,
        c_span_tol=c_span_tol,
        h_span_tol=h_span_tol,
        min_track_size=min_track_size,
        candidate_min_score=candidate_min_score,
        pair_score_weight=pair_score_weight,
        reciprocal_best_bonus=reciprocal_best_bonus,
        keep_singletons=True,
    )
    labels = _pmtc_hsqc_labels(
        tracks,
        sample_ids,
        pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples,
        pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples,
        pmtc_min_cluster_size=pmtc_min_cluster_size,
    )
    groups: Dict[str, List[HSQCTrack]] = defaultdict(list)
    for lab, track in zip(labels, tracks):
        groups[lab].append(track)

    blocks: List[ClusterBlock] = []
    for cid in sorted(groups, key=lambda x: (x.endswith("unassigned"), x)):
        ts = groups[cid]
        vals = [(float(t.center_c), float(t.center_h)) for t in ts]
        mask_counts: Dict[str, int] = defaultdict(int)
        for t in ts:
            mask_counts[_presence_mask_track(t, sample_ids)] += 1
        mask = max(mask_counts.items(), key=lambda kv: kv[1])[0] if mask_counts else ""
        avg_score = sum(t.score for t in ts) / len(ts) if ts else 0.0
        details = (
            f"HSQC PMTC-style tracks={len(ts)}; C_tol={c_tol:g}; H_tol={h_tol:g}; "
            f"C_span≤{c_span_tol:g}; H_span≤{h_span_tol:g}; min_track_size={min_track_size}; "
            f"candidate_min_score={candidate_min_score:g}; avg_score={avg_score:.3f}; dominant_mask={mask}"
        )
        details = (
            f"HSQC PMTC-style tracks={len(ts)}; C_tol={c_tol:g}; H_tol={h_tol:g}; "
            f"C_span<={c_span_tol:g}; H_span<={h_span_tol:g}; min_track_size={min_track_size}; "
            f"candidate_min_score={candidate_min_score:g}; avg_score={avg_score:.3f}; dominant_mask={mask}"
        )
        blocks.append(ClusterBlock(cluster_id=str(cid), kind="HSQC", values=vals, n_tracks=len(ts), presence_mask=mask, details=details))
    return blocks

# =============================================================================
# ReCQC 3.0 additions: single-spectrum 13C clustering and full 2D HSQC PMTC/QG-PMTC flow
# =============================================================================

@dataclass
class CPeak1D:
    peak_id: str
    sample_id: str
    ppm: float
    intensity: float = 1.0
    area: Optional[float] = None


def load_c_peaklist(path: str, sample_id: str = "S1") -> List[CPeak1D]:
    rows = _read_delimited_rows(Path(path))
    if not rows:
        return []
    keys = list(rows[0].keys())
    c_key = _find_key(keys, [
        "CPPM", "C_PPM", "C ppm", "c_ppm", "ppm_c", "carbon_ppm", "13c_ppm",
        "ppm", "shift", "chemical_shift", "position", "delta", "col1",
    ])
    i_key = _find_key(keys, ["intensity", "height", "amp", "amplitude", "peak_height", "area", "integral", "volume", "col2"])
    a_key = _find_key(keys, ["area", "integral", "volume", "col3"])
    if c_key is None:
        raise ValueError(f"Cannot find C/CPPM column in {Path(path).name}.")
    peaks: List[CPeak1D] = []
    for idx, row in enumerate(rows, 1):
        c = _safe_float(row.get(c_key))
        if c is None:
            continue
        inten = _safe_float(row.get(i_key)) if i_key else None
        area = _safe_float(row.get(a_key)) if a_key else None
        if inten is None and area is not None:
            inten = area
        peaks.append(CPeak1D(f"{sample_id}_P{idx:04d}", sample_id, float(c), float(inten if inten is not None else 1.0), area))
    peaks.sort(key=lambda p: p.ppm)
    return peaks


def _ppm_window_at(ppm: float, region_windows: Sequence[Tuple[float, float, float]]) -> float:
    for lo, hi, win in region_windows:
        if lo <= ppm < hi:
            return float(win)
    return float(region_windows[-1][2] if region_windows else 0.30)


def _dict_track_mask(track: dict, ordered_ids: Sequence[str]) -> str:
    members = track.get("members", {}) or {}
    return "".join("1" if members.get(sid) else "0" for sid in ordered_ids)


def _mask_popcount(mask: str) -> int:
    return str(mask).count("1")


def _mask_contains(mask: str, anchor: str) -> bool:
    return all(a == "0" or m == "1" for m, a in zip(str(mask), str(anchor)))


def _all_nonempty_masks(n_samples: int) -> List[str]:
    masks = []
    for value in range(1, 2 ** int(n_samples)):
        masks.append("".join("1" if value & (1 << (n_samples - 1 - i)) else "0" for i in range(n_samples)))
    return sorted(masks, key=lambda m: (_mask_popcount(m), m), reverse=True)


def _project_track_to_common_mask(track: dict, common_mask: str, ordered_ids: Sequence[str]) -> Optional[dict]:
    members = track.get("members", {}) or {}
    projected_members = {
        sid: dict(members[sid])
        for sid, bit in zip(ordered_ids, str(common_mask))
        if bit == "1" and sid in members
    }
    if not projected_members:
        return None
    projected = dict(track)
    projected["members"] = projected_members
    projected["member_ids"] = tuple(sorted(str(p.get("peak_id", "")) for p in projected_members.values()))
    projected["source_strict_mask"] = _dict_track_mask(track, ordered_ids)
    projected["common_mask"] = str(common_mask)
    projected["source_track_id"] = str(track.get("track_id", ""))
    return projected


def _common_mask_track_center(track: dict, common_mask: str, ordered_ids: Sequence[str]) -> float:
    vals = []
    members = track.get("members", {}) or {}
    for sid, bit in zip(ordered_ids, str(common_mask)):
        if bit != "1":
            continue
        peak = members.get(sid)
        if not isinstance(peak, dict):
            continue
        ppm = _safe_float(peak.get("ppm_corr"))
        if ppm is None:
            ppm = _safe_float(peak.get("ppm"))
        if ppm is not None:
            vals.append(ppm)
    if not vals:
        for peak in members.values():
            if not isinstance(peak, dict):
                continue
            ppm = _safe_float(peak.get("ppm_corr"))
            if ppm is None:
                ppm = _safe_float(peak.get("ppm"))
            if ppm is not None:
                vals.append(ppm)
    return float(median(vals)) if vals else 0.0


def run_c_presence_mask_clustering(
    sample_files: Sequence[str],
    *,
    model: str = "v5_pmtc",
    region_windows: Optional[List[Tuple[float, float, float]]] = None,
    span_tol: float = 1.00,
    residual_gate: float = 1.00,
    top_k_per_seed: int = 6,
    max_per_sample: int = 3,
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
    setpacking_model_cost: float = 1.10,
    setpacking_high_mask_bonus: float = 0.0,
    correction_options: Optional[dict] = None,
    output_dir: Optional[str | Path] = None,
) -> List[ClusterBlock]:
    """SP-Mask ablation: group generated cross-sample 13C tracks by common mask.

    Track generation is intentionally shared with the SPTC/PMTC front end.  This mode
    changes only the final partitioning step: two-or-more-sample masks are interpreted
    inclusively.  For example, mask 00011 means "shared by S4/S5", so tracks from
    00011, 10011, ..., 11111 can contribute after a second set-packing pass inside
    that common-mask pool.  Singleton masks keep strict "sample-specific residual"
    semantics to avoid over-broad single-sample groups.
    """
    if len(sample_files) < 2:
        raise ValueError("Presence-mask C clustering needs at least two sample peak-list files.")
    if span_tol <= 0:
        raise ValueError("C span tolerance must be positive.")

    from nmr_trendtrack.contracts import Sample
    from nmr_trendtrack.models.three_model_pipeline import (
        _apply_mask_residual_filter,
        _apply_sample_specific_residual_peak_filter,
        _run_v4_frontend,
        _v4_modules,
    )

    with tempfile.TemporaryDirectory(prefix="recqc30_presence_c_") as tmp:
        tmpdir = Path(tmp)
        sample_ids = [f"S{i + 1}" for i in range(len(sample_files))]
        norm_files: List[Path] = []
        for i, path in enumerate(sample_files, 1):
            out = tmpdir / f"S{i}.csv"
            _normalized_c_peaklist(path, out)
            norm_files.append(out)

        samples = [
            Sample(sample_id=f"S{i + 1}", order_index=i, source_type="peaklist", peaklist_path=str(path))
            for i, path in enumerate(norm_files)
        ]
        correction_note = "correction=off"
        if correction_options and correction_options.get("enabled"):
            runtime_dir = Path(output_dir) / "presence_mask_correction" if output_dir else tmpdir / "presence_mask_correction"
            correction_result = maybe_prepare_corrected_samples(
                samples=[
                    {
                        "sample_id": s.sample_id,
                        "order_index": s.order_index,
                        "source_type": s.source_type,
                        "peaklist_path": s.peaklist_path,
                    }
                    for s in samples
                ],
                runtime_dir=runtime_dir,
                options=correction_options,
            )
            samples = [
                Sample(
                    sample_id=str(row["sample_id"]),
                    order_index=int(row.get("order_index", i)),
                    source_type=str(row.get("source_type", "peaklist")),
                    peaklist_path=str(row["peaklist_path"]),
                )
                for i, row in enumerate(correction_result.samples)
            ]
            meta = correction_result.metadata or {}
            correction_note = (
                f"correction=on; mode={meta.get('calibration_mode', '')}; "
                f"model={correction_result.model_path}"
            )

        config = _build_v5_config(
            model=model,
            region_windows=region_windows,
            span_tol=span_tol,
            residual_gate=residual_gate,
            top_k_per_seed=top_k_per_seed,
            max_per_sample=max_per_sample,
            exact_limit=exact_limit,
            beam_width=beam_width,
            node_limit=node_limit,
            setpacking_model_cost=setpacking_model_cost,
            setpacking_high_mask_bonus=setpacking_high_mask_bonus,
        )
        normalized_model = normalize_model_key(model or "v5_pmtc")
        if normalized_model in {
            "v5",
            "v5a",
            "v5_pmtc",
            "v5_pmtc_r85",
            "v5_enum",
            "v5_enumerated",
            "enumerated_v5",
            "v4",
            "v4_baseline",
            "baseline",
            "default",
        }:
            ordered_ids, _peaks_original, _shifts, tracks, diag = _run_v4_frontend(samples, config)
            backend_note = model_display_label("v4_baseline")
        else:
            from nmr_trendtrack.models import run_switchable_model

            state = run_switchable_model(samples, config)
            ordered_ids = list(state.ordered_sample_ids)
            tracks = []
            for tr in state.tracks:
                members = {}
                for sid, peak in tr.members.items():
                    members[sid] = {
                        "sample": sid,
                        "peak_id": peak.peak_id,
                        "ppm": float(peak.ppm_raw),
                        "ppm_corr": float(peak.corrected_ppm()),
                        "intensity": float(peak.intensity if peak.intensity is not None else (peak.area or 1.0)),
                        "area": float(peak.area if peak.area is not None else peak.intensity),
                    }
                tracks.append({
                    "track_id": tr.track_id,
                    "member_ids": tuple(sorted(p["peak_id"] for p in members.values())),
                    "members": members,
                    "score": float(tr.quality_score),
                })
            diag = {"n_candidate_tracks": "", "n_selected_tracks": len(tracks), "source_model": normalized_model}
            backend_note = model_display_label(normalized_model)

        def _track_center(track: dict) -> float:
            vals = []
            for peak in track.get("members", {}).values():
                ppm = _safe_float(peak.get("ppm_corr")) if isinstance(peak, dict) else None
                if ppm is None and isinstance(peak, dict):
                    ppm = _safe_float(peak.get("ppm"))
                if ppm is not None:
                    vals.append(ppm)
            return float(median(vals)) if vals else 0.0

        _base_mod, setpacking_mod = _v4_modules()
        groups: Dict[str, List[dict]] = {}
        pool_sizes: Dict[str, int] = {}
        source_mask_counts: Dict[str, Dict[str, int]] = {}

        for common_mask in _all_nonempty_masks(len(ordered_ids)):
            pool: List[dict] = []
            for track in tracks:
                strict_mask = _dict_track_mask(track, ordered_ids)
                if _mask_popcount(common_mask) >= 2:
                    if not _mask_contains(strict_mask, common_mask):
                        continue
                elif strict_mask != common_mask:
                    continue
                projected = _project_track_to_common_mask(track, common_mask, ordered_ids)
                if projected is not None:
                    pool.append(projected)
            if not pool:
                continue
            selected, _pack_info = setpacking_mod.componentwise_global_set_packing(
                pool,
                exact_limit=exact_limit,
                beam_width=beam_width,
                node_limit=node_limit,
            )
            if not selected:
                continue
            groups[common_mask] = selected
            pool_sizes[common_mask] = len(pool)
            counts: Dict[str, int] = defaultdict(int)
            for track in selected:
                counts[str(track.get("source_strict_mask", _dict_track_mask(track, ordered_ids)))] += 1
            source_mask_counts[common_mask] = dict(counts)

        mask_only_tracks: List[dict] = []
        mask_only_labels: List[str] = []
        for mask in sorted(groups, key=lambda m: (_mask_popcount(m), m), reverse=True):
            for track in groups[mask]:
                mask_only_tracks.append(track)
                mask_only_labels.append(f"M{mask}")

        mask_only_tracks, mask_only_labels, residual_diag = _apply_mask_residual_filter(
            mask_only_tracks,
            mask_only_labels,
            ordered_ids,
            config,
        )
        mask_only_tracks, mask_only_labels, single_diag = _apply_sample_specific_residual_peak_filter(
            mask_only_tracks,
            mask_only_labels,
            ordered_ids,
            config,
        )

        cleaned_groups: Dict[str, List[dict]] = defaultdict(list)
        for label, track in zip(mask_only_labels, mask_only_tracks):
            cleaned_groups[str(label)].append(track)

        blocks: List[ClusterBlock] = []
        rows: List[dict] = []
        sample_order_txt = ",".join(ordered_ids)
        cleanup_txt = (
            f"mask_cleanup={residual_diag.get('mask_residual_output_tracks', '')}; "
            f"sample_specific_removed={single_diag.get('sample_specific_residual_removed_single_tracks', '')}"
        )
        for cid in sorted(
            cleaned_groups,
            key=lambda label: (
                -_mask_popcount(_dict_track_mask(cleaned_groups[label][0], ordered_ids)) if cleaned_groups[label] else 0,
                str(label),
            ),
        ):
            group = cleaned_groups[cid]
            mask_counts: Dict[str, int] = defaultdict(int)
            source_counts: Dict[str, int] = defaultdict(int)
            vals = []
            for track in group:
                mask = _dict_track_mask(track, ordered_ids)
                mask_counts[mask] += 1
                source_counts[str(track.get("source_strict_mask", mask))] += 1
                vals.append(_common_mask_track_center(track, mask, ordered_ids))
            dominant_mask = max(mask_counts.items(), key=lambda kv: kv[1])[0] if mask_counts else ""
            source_counts_txt = ",".join(f"{m}:{n}" for m, n in sorted(source_counts.items(), reverse=True) if m)
            mask_counts_txt = ",".join(f"{m}:{n}" for m, n in sorted(mask_counts.items(), reverse=True))
            details = (
                f"{model_display_label('v5_frontend_mask')} common-mask via {backend_note} tracks={len(group)}; "
                f"dominant_mask={dominant_mask}; sample_order={sample_order_txt}; "
                f"mask_counts={mask_counts_txt}; source_strict_masks={source_counts_txt}; "
                f"span<={span_tol:g}; no PMTC/QG refinement; {cleanup_txt}; "
                f"candidate_tracks={diag.get('n_candidate_tracks', diag.get('candidate_tracks', ''))}; {correction_note}"
            )
            blocks.append(ClusterBlock(str(cid), "C", vals, len(vals), dominant_mask, details))
            rows.append({
                "cluster_id": str(cid),
                "common_mask": dominant_mask,
                "n_tracks": len(vals),
                "ppm_values": ";".join(f"{v:.4f}" for v in vals),
                "mask_counts": mask_counts_txt,
                "source_strict_masks": source_counts_txt,
                "details": details,
            })

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            if rows:
                with (out / "presence_mask_only_clusters.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
        return blocks


def run_c_common_mask_backend_clustering(
    sample_files: Sequence[str],
    *,
    model: str = "v5_enum",
    region_windows: Optional[List[Tuple[float, float, float]]] = None,
    span_tol: float = 1.00,
    residual_gate: float = 1.00,
    top_k_per_seed: int = 6,
    max_per_sample: int = 3,
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None,
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None,
    pmtc_min_cluster_size: int = 3,
    setpacking_model_cost: float = 1.10,
    setpacking_high_mask_bonus: float = 0.0,
    guarded_quality_max_cluster_rise: int = 4,
    correction_options: Optional[dict] = None,
    output_dir: Optional[str | Path] = None,
) -> List[ClusterBlock]:
    """Run common-mask projected tracks through PMTC or QG-PMTC.

    The front end first builds inclusive common-mask pools. For mask 00011, tracks
    from 00011, 10011, ..., 11111 are projected onto S4/S5 and repacked inside
    that common-mask pool. The resulting tracks are then clustered by PMTC or
    QG-PMTC and passed through the same residual-mask/sample-specific cleanup as
    the strict PMTC backends.
    """
    if len(sample_files) < 2:
        raise ValueError("Common-mask backend clustering needs at least two sample peak-list files.")
    if span_tol <= 0 or residual_gate <= 0:
        raise ValueError("C span tolerance and residual gate must be positive.")
    if pmtc_max_tracks_by_n_samples is None:
        pmtc_max_tracks_by_n_samples = {3: 28, 4: 24, 5: 22}
    if pmtc_frac_limit_by_n_samples is None:
        pmtc_frac_limit_by_n_samples = {3: 0.50, 4: 0.43, 5: 0.36}

    from nmr_trendtrack.contracts import Sample
    from nmr_trendtrack.models.three_model_pipeline import (
        _apply_mask_residual_filter,
        _apply_sample_specific_residual_peak_filter,
        _pmtc_labels,
        _run_v4_frontend,
        _v4_modules,
        guarded_recall_quality_labels,
    )

    with tempfile.TemporaryDirectory(prefix="recqc30_common_backend_c_") as tmp:
        tmpdir = Path(tmp)
        norm_files: List[Path] = []
        for i, path in enumerate(sample_files, 1):
            out = tmpdir / f"S{i}.csv"
            _normalized_c_peaklist(path, out)
            norm_files.append(out)

        samples = [
            Sample(sample_id=f"S{i + 1}", order_index=i, source_type="peaklist", peaklist_path=str(path))
            for i, path in enumerate(norm_files)
        ]
        correction_note = "correction=off"
        if correction_options and correction_options.get("enabled"):
            runtime_dir = Path(output_dir) / "common_mask_correction" if output_dir else tmpdir / "common_mask_correction"
            correction_result = maybe_prepare_corrected_samples(
                samples=[
                    {
                        "sample_id": s.sample_id,
                        "order_index": s.order_index,
                        "source_type": s.source_type,
                        "peaklist_path": s.peaklist_path,
                    }
                    for s in samples
                ],
                runtime_dir=runtime_dir,
                options=correction_options,
            )
            samples = [
                Sample(
                    sample_id=str(row["sample_id"]),
                    order_index=int(row.get("order_index", i)),
                    source_type=str(row.get("source_type", "peaklist")),
                    peaklist_path=str(row["peaklist_path"]),
                )
                for i, row in enumerate(correction_result.samples)
            ]
            meta = correction_result.metadata or {}
            correction_note = (
                f"correction=on; mode={meta.get('calibration_mode', '')}; "
                f"model={correction_result.model_path}"
            )

        config = _build_v5_config(
            model=model,
            region_windows=region_windows,
            span_tol=span_tol,
            residual_gate=residual_gate,
            top_k_per_seed=top_k_per_seed,
            max_per_sample=max_per_sample,
            exact_limit=exact_limit,
            beam_width=beam_width,
            node_limit=node_limit,
            pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples,
            pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples,
            pmtc_min_cluster_size=pmtc_min_cluster_size,
            setpacking_model_cost=setpacking_model_cost,
            setpacking_high_mask_bonus=setpacking_high_mask_bonus,
            guarded_quality_max_cluster_rise=guarded_quality_max_cluster_rise,
        )
        ordered_ids, _peaks_original, _shifts, strict_tracks, diag = _run_v4_frontend(samples, config)
        _base_mod, setpacking_mod = _v4_modules()

        common_tracks: List[dict] = []
        pool_sizes: Dict[str, int] = {}
        selected_sizes: Dict[str, int] = {}
        source_mask_counts: Dict[str, Dict[str, int]] = {}
        for common_mask in _all_nonempty_masks(len(ordered_ids)):
            pool: List[dict] = []
            for track in strict_tracks:
                strict_mask = _dict_track_mask(track, ordered_ids)
                if _mask_popcount(common_mask) >= 2:
                    if not _mask_contains(strict_mask, common_mask):
                        continue
                elif strict_mask != common_mask:
                    continue
                projected = _project_track_to_common_mask(track, common_mask, ordered_ids)
                if projected is not None:
                    pool.append(projected)
            if not pool:
                continue
            selected, _pack_info = setpacking_mod.componentwise_global_set_packing(
                pool,
                exact_limit=exact_limit,
                beam_width=beam_width,
                node_limit=node_limit,
            )
            if not selected:
                continue
            for track in selected:
                track["common_mask"] = common_mask
            common_tracks.extend(selected)
            pool_sizes[common_mask] = len(pool)
            selected_sizes[common_mask] = len(selected)
            counts: Dict[str, int] = defaultdict(int)
            for track in selected:
                counts[str(track.get("source_strict_mask", _dict_track_mask(track, ordered_ids)))] += 1
            source_mask_counts[common_mask] = dict(counts)

        normalized_model = normalize_model_key(model or "v5_enum")
        pmtc_labels, pmtc_diag = _pmtc_labels(common_tracks, ordered_ids, config)
        if normalized_model in {"v5", "v5a", "v5_pmtc", "v5_pmtc_r85", "default"}:
            labels = pmtc_labels
            backend_note = model_display_label("v5_pmtc")
            backend_diag = dict(pmtc_diag)
        elif normalized_model in {"v5_enum", "v5_enumerated", "enumerated_v5"}:
            labels, backend_diag = guarded_recall_quality_labels(
                common_tracks,
                ordered_ids,
                config,
                current_labels=pmtc_labels,
                merge_threshold=float(getattr(config.model, "guarded_quality_merge_threshold", 0.80)),
                hac_threshold=float(getattr(config.model, "guarded_quality_hac_threshold", 1.00)),
            )
            backend_note = model_display_label("v5_enum")
        else:
            raise ValueError("Common-mask backend supports PMTC and QG-PMTC only.")

        common_tracks, labels, residual_diag = _apply_mask_residual_filter(common_tracks, labels, ordered_ids, config)
        common_tracks, labels, single_diag = _apply_sample_specific_residual_peak_filter(
            common_tracks,
            labels,
            ordered_ids,
            config,
        )

    groups: Dict[str, List[dict]] = defaultdict(list)
    for track, label in zip(common_tracks, labels):
        groups[str(label)].append(track)

    blocks: List[ClusterBlock] = []
    rows: List[dict] = []
    sample_order_txt = ",".join(ordered_ids)
    region_txt = "; ".join(f"{lo:g}-{hi:g}:{win:g}" for lo, hi, win in (region_windows or DEFAULT_REGION_WINDOWS))
    diag_txt = (
        f"common_frontend_tracks={sum(selected_sizes.values())}; "
        f"candidate_tracks={diag.get('n_candidate_tracks', '')}; "
        f"strict_selected={diag.get('n_selected_tracks', '')}; "
        f"mask_cleanup={residual_diag.get('mask_residual_output_tracks', '')}; "
        f"sample_specific_removed={single_diag.get('sample_specific_residual_removed_single_tracks', '')}"
    )
    for cid in sorted(groups, key=lambda x: (str(x).lower().endswith("unassigned"), str(x))):
        tracks = groups[cid]
        vals = []
        mask_counts: Dict[str, int] = defaultdict(int)
        source_counts: Dict[str, int] = defaultdict(int)
        for track in tracks:
            mask = _dict_track_mask(track, ordered_ids)
            mask_counts[mask] += 1
            source_counts[str(track.get("source_strict_mask", ""))] += 1
            vals.append(_common_mask_track_center(track, mask, ordered_ids))
        dominant_mask = max(mask_counts.items(), key=lambda kv: kv[1])[0] if mask_counts else ""
        mask_counts_txt = ",".join(f"{m}:{n}" for m, n in sorted(mask_counts.items(), reverse=True))
        source_counts_txt = ",".join(f"{m}:{n}" for m, n in sorted(source_counts.items(), reverse=True) if m)
        details = (
            f"Common-mask front end + {backend_note} tracks={len(tracks)}; dominant_mask={dominant_mask}; "
            f"sample_order={sample_order_txt}; mask_counts={mask_counts_txt}; source_strict_masks={source_counts_txt}; "
            f"region_tol=[{region_txt}]; span<={span_tol:g}; residual_gate={residual_gate:g}; "
            f"{diag_txt}; {correction_note}"
        )
        blocks.append(ClusterBlock(str(cid), "C", vals, len(vals), dominant_mask, details))
        rows.append({
            "cluster_id": str(cid),
            "dominant_mask": dominant_mask,
            "n_tracks": len(vals),
            "ppm_values": ";".join(f"{v:.4f}" for v in vals),
            "mask_counts": mask_counts_txt,
            "source_strict_masks": source_counts_txt,
            "details": details,
        })

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if rows:
            with (out / "common_mask_backend_clusters.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return blocks


def _split_c_single_segment(segment: List[CPeak1D], max_peaks_per_cluster: int, intensity_split_weight: float) -> List[List[CPeak1D]]:
    if len(segment) <= max_peaks_per_cluster:
        return [segment]
    ordered = sorted(segment, key=lambda p: p.ppm)
    n_parts = int(math.ceil(len(ordered) / max(1, max_peaks_per_cluster)))
    logs = [math.log(max(abs(float(p.area if p.area is not None else p.intensity)), 1e-12)) for p in ordered]
    gaps = []
    for i in range(len(ordered) - 1):
        gaps.append(((ordered[i + 1].ppm - ordered[i].ppm) + float(intensity_split_weight) * abs(logs[i + 1] - logs[i]), i))
    cuts = sorted(i + 1 for _, i in sorted(gaps, reverse=True)[: max(0, n_parts - 1)])
    out: List[List[CPeak1D]] = []
    start = 0
    for cut in cuts + [len(ordered)]:
        part = ordered[start:cut]
        start = cut
        for j in range(0, len(part), max_peaks_per_cluster):
            chunk = part[j:j + max_peaks_per_cluster]
            if chunk:
                out.append(chunk)
    return out


def run_c_single_spectrum_gap_clustering(
    sample_files: Sequence[str],
    *,
    region_windows: Optional[List[Tuple[float, float, float]]] = None,
    single_gap_factor: float = 1.8,
    max_peaks_per_cluster: int = 30,
    min_cluster_size: int = 1,
    intensity_split_weight: float = 0.10,
) -> List[ClusterBlock]:
    """Exploratory 13C single-spectrum grouping integrated into the fourth module.

    This mode is deliberately separate from V5 cross-sample clustering.  With one spectrum
    there is no co-variation evidence, so the result is an editable C_untyped candidate
    grouping based on region-aware ppm gaps, cluster-size control, and optional intensity-gap
    splitting.  Each block can still be edited and sent to ReCQC C_untyped database analysis.
    """
    if not sample_files:
        raise ValueError("Please select at least one 13C peak-list file.")
    if single_gap_factor <= 0 or max_peaks_per_cluster <= 0 or min_cluster_size <= 0:
        raise ValueError("Single-spectrum gap factor, max peaks/cluster and min cluster size must be positive.")
    regions = region_windows or DEFAULT_REGION_WINDOWS
    blocks: List[ClusterBlock] = []
    for si, path in enumerate(sample_files, 1):
        sid = f"S{si}"
        peaks = load_c_peaklist(path, sid)
        if not peaks:
            continue
        segments: List[List[CPeak1D]] = []
        cur = [peaks[0]]
        for a, b in zip(peaks, peaks[1:]):
            center = 0.5 * (a.ppm + b.ppm)
            gate = _ppm_window_at(center, regions) * float(single_gap_factor)
            if b.ppm - a.ppm > gate:
                segments.append(cur)
                cur = [b]
            else:
                cur.append(b)
        segments.append(cur)
        local_idx = 0
        for seg in segments:
            for part in _split_c_single_segment(seg, int(max_peaks_per_cluster), float(intensity_split_weight)):
                if len(part) < int(min_cluster_size):
                    continue
                local_idx += 1
                vals = [p.ppm for p in part]
                span = max(vals) - min(vals) if len(vals) > 1 else 0.0
                details = (
                    f"Single-spectrum exploratory C cluster; sample={Path(path).name}; peaks={len(part)}; "
                    f"ppm_span={span:.3f}; gap_factor={single_gap_factor:g}; max_peaks/cluster={max_peaks_per_cluster}"
                )
                blocks.append(ClusterBlock(f"{sid}_C{local_idx:02d}", "C", vals, len(part), "single", details))
    if not blocks:
        raise ValueError("No single-spectrum C clusters were produced.")
    return blocks


def _single_spectrum_pipeline_config(
    *,
    use_type: bool = False,
    use_area: bool = False,
    calibration_mode: str = "height_stabilize",
    shrink_strength: float = 1.235,
    delta_clip: Optional[float] = 0.63,
    ppm_bin_width: float = 9.75,
    width_bin_width: float = 0.35,
    gmm_min_components: int = 1,
    gmm_max_components: int = 4,
    gmm_use_log: bool = True,
) -> SingleSpectrumPipelineConfig:
    return SingleSpectrumPipelineConfig(
        calibration=SingleSpectrumCalibrationConfig(
            use_type=bool(use_type),
            use_area=bool(use_area),
            calibration_mode=str(calibration_mode or "height_stabilize"),
            shrink_strength=float(shrink_strength),
            delta_clip=None if delta_clip is None else float(delta_clip),
            ppm_bin_width=float(ppm_bin_width),
            width_bin_width=float(width_bin_width),
        ),
        gmm=SingleSpectrumGMMConfig(
            min_components=int(gmm_min_components),
            max_components=int(gmm_max_components),
            use_log=bool(gmm_use_log),
        ),
    )


def _cluster_shift_values(value) -> List[float]:
    if value is None:
        return []
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))]


def run_c_single_spectrum_clustering(
    sample_files: Sequence[str],
    *,
    region_windows: Optional[List[Tuple[float, float, float]]] = None,
    single_gap_factor: float = 1.8,
    max_peaks_per_cluster: int = 30,
    min_cluster_size: int = 1,
    intensity_split_weight: float = 0.10,
    correction_enabled: bool = False,
    use_type: bool = False,
    use_area: bool = False,
    calibration_mode: str = "height_stabilize",
    shrink_strength: float = 1.235,
    delta_clip: Optional[float] = 0.63,
    ppm_bin_width: float = 9.75,
    width_bin_width: float = 0.35,
    gmm_min_components: int = 1,
    gmm_max_components: int = 4,
    gmm_use_log: bool = True,
    output_dir: Optional[str | Path] = None,
) -> List[ClusterBlock]:
    """Run the restored no_brancha single-spectrum correction + PeakGMM pipeline.

    The old ppm-gap exploratory splitter is intentionally kept above for reference, but
    this definition is the active GUI/backend entry point.  It preserves the original
    outputs: calibration model/result files, corrected tables, GMM detailed rows,
    GMM clusters, and summary JSON.
    """
    if not sample_files:
        raise ValueError("Please select at least one 13C peak-list file.")
    if gmm_min_components <= 0 or gmm_max_components < gmm_min_components:
        raise ValueError("GMM component limits must satisfy 1 <= min <= max.")

    root = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="recqc30_single_spectrum_"))
    root.mkdir(parents=True, exist_ok=True)
    cfg = _single_spectrum_pipeline_config(
        use_type=use_type,
        use_area=use_area,
        calibration_mode=calibration_mode,
        shrink_strength=shrink_strength,
        delta_clip=delta_clip,
        ppm_bin_width=ppm_bin_width,
        width_bin_width=width_bin_width,
        gmm_min_components=gmm_min_components,
        gmm_max_components=gmm_max_components,
        gmm_use_log=gmm_use_log,
    )

    blocks: List[ClusterBlock] = []
    combined_rows: List[dict] = []
    for file_index, path in enumerate(sample_files, 1):
        in_path = Path(path)
        out = root / f"{file_index:02d}_{in_path.stem}"
        if correction_enabled:
            result = run_single_spectrum_pipeline(in_path, out_dir=out, config=cfg)
            clusters_df = result["clusters"]
            summary = result["summary"]
            correction_text = f"correction=on; model={summary.get('calibration_model_path')}"
        else:
            result = run_single_spectrum_gmm(
                in_path,
                out_dir=out,
                min_components=gmm_min_components,
                max_components=gmm_max_components,
                use_log=gmm_use_log,
            )
            clusters_df = result["clusters"]
            correction_text = "correction=off"

        for row_index, row in enumerate(clusters_df.to_dict("records"), 1):
            vals = _cluster_shift_values(row.get("shift_values", ""))
            if len(vals) < int(min_cluster_size):
                continue
            sample_label = str(row.get("sample", f"S{file_index}"))
            cluster_label = str(row.get("cluster_id", row_index))
            cid = f"{in_path.stem}_{sample_label}_GMM{cluster_label}"
            details = (
                f"Single-spectrum original pipeline; source={in_path.name}; sample={sample_label}; "
                f"num_peaks={int(row.get('num_peaks', len(vals)))}; mean_intensity={float(row.get('mean_intensity', 0.0)):.4g}; "
                f"GMM k={gmm_min_components}-{gmm_max_components}; use_log={bool(gmm_use_log)}; "
                f"{correction_text}; output={out}"
            )
            blocks.append(ClusterBlock(cid, "C", vals, len(vals), "single", details))
            combined_rows.append({
                "source_file": str(in_path),
                "cluster_id": cid,
                "sample": sample_label,
                "num_peaks": len(vals),
                "shift_values": ",".join(f"{v:.4f}" for v in vals),
                "output_dir": str(out),
                "correction_enabled": bool(correction_enabled),
            })

    if combined_rows:
        with (root / "single_spectrum_combined_clusters.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
            w.writeheader()
            w.writerows(combined_rows)
    if not blocks:
        raise ValueError("No single-spectrum C clusters were produced.")
    return blocks


def _hc(p: HSQCPeak) -> float:
    return float(getattr(p, "h_corr", p.h_ppm))


def _cc(p: HSQCPeak) -> float:
    return float(getattr(p, "c_corr", p.c_ppm))


def _set_corr(p: HSQCPeak, c: float, h: float) -> None:
    setattr(p, "c_corr", c)
    setattr(p, "h_corr", h)


def _estimate_hsqc_2d_coarse_shift(peaks_by_sample: Dict[str, List[HSQCPeak]], sample_ids: Sequence[str], c_win: float, h_win: float) -> Dict[str, Tuple[float, float]]:
    shifts: Dict[str, Tuple[float, float]] = {}
    if not sample_ids:
        return shifts
    ref_id = sample_ids[0]
    ref = peaks_by_sample.get(ref_id, [])
    shifts[ref_id] = (0.0, 0.0)
    for p in ref:
        _set_corr(p, p.c_ppm, p.h_ppm)
    for sid in sample_ids[1:]:
        cd: List[float] = []
        hd: List[float] = []
        for p in peaks_by_sample.get(sid, []):
            best = None
            best_d = float("inf")
            for q in ref:
                dc = q.c_ppm - p.c_ppm
                dh = q.h_ppm - p.h_ppm
                if abs(dc) <= c_win and abs(dh) <= h_win:
                    d = math.sqrt((dc / max(c_win, 1e-12)) ** 2 + (dh / max(h_win, 1e-12)) ** 2)
                    if d < best_d:
                        best_d, best = d, (dc, dh)
            if best is not None:
                cd.append(best[0]); hd.append(best[1])
        cs = max(-c_win, min(c_win, float(median(cd)))) if cd else 0.0
        hs = max(-h_win, min(h_win, float(median(hd)))) if hd else 0.0
        shifts[sid] = (cs, hs)
        for p in peaks_by_sample.get(sid, []):
            _set_corr(p, p.c_ppm + cs, p.h_ppm + hs)
    return shifts


def _build_hsqc_2d_candidates(peaks_by_sample: Dict[str, List[HSQCPeak]], sample_ids: Sequence[str], c_tol: float, h_tol: float, c_span_tol: float, h_span_tol: float, reciprocal_best_bonus: float):
    rows = []
    for i, a_sid in enumerate(sample_ids):
        for b_sid in sample_ids[i + 1:]:
            pair = []
            for a in peaks_by_sample.get(a_sid, []):
                for b in peaks_by_sample.get(b_sid, []):
                    dc = abs(_cc(a) - _cc(b)); dh = abs(_hc(a) - _hc(b))
                    if dc <= c_tol and dh <= h_tol and dc <= c_span_tol and dh <= h_span_tol:
                        d = math.sqrt((dc / max(c_tol, 1e-12)) ** 2 + (dh / max(h_tol, 1e-12)) ** 2)
                        pair.append((d, a, b))
            best_a: Dict[str, str] = {}
            best_b: Dict[str, str] = {}
            for d, a, b in sorted(pair, key=lambda x: x[0]):
                best_a.setdefault(a.peak_id, b.peak_id)
                best_b.setdefault(b.peak_id, a.peak_id)
            for d, a, b in pair:
                rec = best_a.get(a.peak_id) == b.peak_id and best_b.get(b.peak_id) == a.peak_id
                score = max(0.0, 1.0 - 0.5 * d) + (reciprocal_best_bonus if rec else 0.0)
                rows.append({"a": a.peak_id, "b": b.peak_id, "score": score, "reciprocal": rec})
    return rows


def _hsqc_maps(cands):
    compat: Dict[str, set] = defaultdict(set)
    pair_score: Dict[Tuple[str, str], float] = {}
    rec: Dict[Tuple[str, str], bool] = {}
    for c in cands:
        a, b = c["a"], c["b"]
        compat[a].add(b); compat[b].add(a)
        key = tuple(sorted((a, b)))
        pair_score[key] = max(pair_score.get(key, 0.0), float(c["score"]))
        rec[key] = rec.get(key, False) or bool(c["reciprocal"])
    return compat, pair_score, rec


def _hsqc_components_from_candidates(cands):
    graph: Dict[str, set] = defaultdict(set)
    for c in cands:
        graph[c["a"]].add(c["b"]); graph[c["b"]].add(c["a"])
    comps = []
    seen = set()
    for node in list(graph):
        if node in seen:
            continue
        stack = [node]; seen.add(node); comp = set()
        while stack:
            cur = stack.pop(); comp.add(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt); stack.append(nxt)
        comps.append(comp)
    return comps


def _score_hsqc_members(members: Dict[str, HSQCPeak], sample_ids: Sequence[str], c_tol: float, h_tol: float, c_span_tol: float, h_span_tol: float, pair_score, rec_pair, pair_score_weight: float):
    cvals = [_cc(p) for p in members.values()]
    hvals = [_hc(p) for p in members.values()]
    if not cvals:
        return float("-inf"), 0, 0, 0, 0, 0
    center_c = float(median(cvals)); center_h = float(median(hvals))
    cspan = max(cvals) - min(cvals) if len(cvals) > 1 else 0.0
    hspan = max(hvals) - min(hvals) if len(hvals) > 1 else 0.0
    if cspan > c_span_tol or hspan > h_span_tol:
        return float("-inf"), center_c, center_h, cspan, hspan, 0
    for p in members.values():
        if abs(_cc(p) - center_c) > 1.35 * c_tol or abs(_hc(p) - center_h) > 1.35 * h_tol:
            return float("-inf"), center_c, center_h, cspan, hspan, 0
    coverage = len(members); missing = len(sample_ids) - coverage
    score = 2.15 * coverage + 0.75 * max(0, coverage - 2) - 0.50 * missing
    score -= 1.80 * cspan / max(c_span_tol, 1e-12)
    score -= 1.35 * hspan / max(h_span_tol, 1e-12)
    ids = [p.peak_id for p in members.values()]
    ps = []; recs = 0; nt = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            key = tuple(sorted((ids[i], ids[j])))
            if key in pair_score:
                ps.append(pair_score[key]); nt += 1; recs += 1 if rec_pair.get(key) else 0
    if ps:
        score += pair_score_weight * (sum(ps) / len(ps)) * max(1, len(ps))
    vals = []
    for sid in sample_ids:
        p = members.get(sid)
        vals.append(None if p is None else math.log(max(abs(_hsqc_signal_value(p)), 1e-12)))
    steps = [b - a for a, b in zip(vals, vals[1:]) if a is not None and b is not None]
    if len(steps) >= 2:
        med = float(median(steps))
        score -= 0.30 * max(abs(s - med) for s in steps)
        score -= 0.65 * sum(1 for i in range(len(steps)) for j in range(i + 1, len(steps)) if abs(steps[i]) > 0.35 and abs(steps[j]) > 0.35 and steps[i] * steps[j] < 0)
    return float(score), center_c, center_h, cspan, hspan, recs / max(1, nt)


def _enumerate_hsqc_2d_tracks_for_component(comp, lookup, compat, pair_score, rec_pair, sample_ids, c_tol, h_tol, c_span_tol, h_span_tol, min_track_size, candidate_min_score, pair_score_weight, top_k_per_seed, max_per_sample, node_limit):
    by_sample: Dict[str, List[HSQCPeak]] = defaultdict(list)
    for pid in comp:
        by_sample[lookup[pid].sample_id].append(lookup[pid])
    if len(by_sample) < max(2, min_track_size):
        return []
    mc = float(median([_cc(p) for arr in by_sample.values() for p in arr])); mh = float(median([_hc(p) for arr in by_sample.values() for p in arr]))
    for sid in list(by_sample):
        by_sample[sid].sort(key=lambda p: (math.sqrt(((_cc(p) - mc) / max(c_tol, 1e-12)) ** 2 + ((_hc(p) - mh) / max(h_tol, 1e-12)) ** 2), -abs(_hsqc_signal_value(p))))
        by_sample[sid] = by_sample[sid][:max(1, int(max_per_sample))]
    order = sorted([s for s in sample_ids if s in by_sample], key=lambda s: len(by_sample[s]))
    best: Dict[Tuple[str, ...], HSQCTrack] = {}
    nodes = 0
    def add(members):
        if len(members) < max(2, min_track_size):
            return
        score, cc, hh, cs, hs, rf = _score_hsqc_members(members, sample_ids, c_tol, h_tol, c_span_tol, h_span_tol, pair_score, rec_pair, pair_score_weight)
        if not math.isfinite(score) or score < candidate_min_score:
            return
        key = tuple(sorted(p.peak_id for p in members.values()))
        tr = HSQCTrack("", dict(members), cc, hh, cs, hs, score, rf)
        if key not in best or tr.score > best[key].score:
            best[key] = tr
    def can_add(members, p):
        if p.sample_id in members:
            return False
        if members and not any(q.peak_id in compat.get(p.peak_id, set()) for q in members.values()):
            return False
        test = dict(members); test[p.sample_id] = p
        return math.isfinite(_score_hsqc_members(test, sample_ids, c_tol, h_tol, c_span_tol, h_span_tol, pair_score, rec_pair, pair_score_weight)[0])
    def dfs(pos, members):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            return
        if len(members) + (len(order) - pos) < max(2, min_track_size):
            return
        if pos >= len(order):
            add(members); return
        sid = order[pos]
        dfs(pos + 1, members)
        for p in by_sample[sid]:
            if can_add(members, p):
                members[sid] = p; dfs(pos + 1, members); members.pop(sid, None)
    dfs(0, {})
    keep = max(1, int(top_k_per_seed)) * max(1, sum(len(v) for v in by_sample.values()))
    return sorted(best.values(), key=lambda t: (t.score, len(t.members), t.reciprocal_fraction), reverse=True)[:keep]


def _pack_hsqc_2d_tracks(cands: List[HSQCTrack], beam_width: int, exact_limit: int, node_limit: int) -> List[HSQCTrack]:
    ordered = sorted(cands, key=lambda t: (t.score, len(t.members), t.reciprocal_fraction, -t.c_span, -t.h_span), reverse=True)
    if not ordered:
        return []
    beams: List[Tuple[float, set, List[HSQCTrack]]] = [(0.0, set(), [])]
    nodes = 0
    # exact branch-and-bound is used for small components/candidate lists.
    if len(ordered) <= exact_limit:
        best_score = float("-inf"); best_sel: List[HSQCTrack] = []
        def rec(i, used, sel, score):
            nonlocal nodes, best_score, best_sel
            nodes += 1
            if nodes > node_limit:
                return
            if i >= len(ordered):
                if score > best_score:
                    best_score = score; best_sel = list(sel)
                return
            rec(i + 1, used, sel, score)
            ids = {p.peak_id for p in ordered[i].members.values()}
            if not used & ids:
                sel.append(ordered[i]); rec(i + 1, used | ids, sel, score + ordered[i].score); sel.pop()
        rec(0, set(), [], 0.0)
        return sorted(best_sel, key=lambda t: (t.center_c, t.center_h))
    for tr in ordered:
        new = list(beams)
        ids = {p.peak_id for p in tr.members.values()}
        for score, used, sel in beams:
            nodes += 1
            if nodes > node_limit:
                break
            if not (used & ids):
                new.append((score + tr.score, used | ids, sel + [tr]))
        new.sort(key=lambda x: (x[0], len(x[2])), reverse=True)
        beams = new[:max(1, beam_width)]
        if nodes > node_limit:
            break
    selected = list(beams[0][2]) if beams else []
    for tr in ordered:
        ids = {p.peak_id for p in tr.members.values()}
        conflicts = [s for s in selected if ids & {p.peak_id for p in s.members.values()}]
        if not conflicts:
            if tr not in selected:
                selected.append(tr)
        elif tr not in selected and tr.score > sum(c.score for c in conflicts) + 0.05:
            selected = [s for s in selected if s not in conflicts] + [tr]
    return sorted(selected, key=lambda t: (t.center_c, t.center_h))


def build_hsqc_v5_full_tracks(peaks_by_sample, sample_ids, *, c_tol=1.0, h_tol=0.1, c_span_tol=1.0, h_span_tol=0.1, min_track_size=2, candidate_min_score=0.40, pair_score_weight=1.20, reciprocal_best_bonus=0.18, top_k_per_seed=6, max_per_sample=3, exact_limit=12, beam_width=120, node_limit=100000, keep_singletons=True):
    shifts = _estimate_hsqc_2d_coarse_shift(peaks_by_sample, sample_ids, max(2.0, 2 * c_tol), max(0.25, 2 * h_tol))
    cands = _build_hsqc_2d_candidates(peaks_by_sample, sample_ids, c_tol, h_tol, c_span_tol, h_span_tol, reciprocal_best_bonus)
    compat, pair_score, rec_pair = _hsqc_maps(cands)
    lookup = _hsqc_peak_lookup(peaks_by_sample)
    comps = _hsqc_components_from_candidates(cands)
    all_cands: List[HSQCTrack] = []
    per_comp_limit = max(1000, int(node_limit) // max(1, len(comps)))
    for comp in comps:
        all_cands.extend(_enumerate_hsqc_2d_tracks_for_component(comp, lookup, compat, pair_score, rec_pair, sample_ids, c_tol, h_tol, c_span_tol, h_span_tol, int(min_track_size), float(candidate_min_score), float(pair_score_weight), int(top_k_per_seed), int(max_per_sample), per_comp_limit))
    selected = _pack_hsqc_2d_tracks(all_cands, int(beam_width), int(exact_limit), int(node_limit))
    used = {p.peak_id for t in selected for p in t.members.values()}
    if keep_singletons:
        for sid in sample_ids:
            for p in peaks_by_sample.get(sid, []):
                if p.peak_id not in used:
                    selected.append(HSQCTrack("", {sid: p}, _cc(p), _hc(p), 0.0, 0.0, 0.0, 0.0))
                    used.add(p.peak_id)
    selected.sort(key=lambda t: (t.center_c, t.center_h))
    for i, t in enumerate(selected, 1):
        t.track_id = f"T{i:04d}"
    return selected, shifts, len(cands)


def _legacy_hsqc_v5_full_clustering_pmtc_only(sample_files: Sequence[str], *, c_tol: float = 1.0, h_tol: float = 0.1, c_span_tol: float = 1.0, h_span_tol: float = 0.1, min_track_size: int = 2, candidate_min_score: float = 0.40, pair_score_weight: float = 1.20, reciprocal_best_bonus: float = 0.18, top_k_per_seed: int = 6, max_per_sample: int = 3, exact_limit: int = 12, beam_width: int = 120, node_limit: int = 100000, pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None, pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None, pmtc_min_cluster_size: int = 3, correction_options: Optional[dict] = None, output_dir: Optional[str | Path] = None) -> List[ClusterBlock]:
    if len(sample_files) < 2:
        raise ValueError("HSQC full 2D clustering needs at least two sample peak-list files.")
    if min(c_tol, h_tol, c_span_tol, h_span_tol) <= 0:
        raise ValueError("HSQC C/H tolerance and span tolerances must be positive.")
    sample_ids = [f"S{i + 1}" for i in range(len(sample_files))]
    peaks_by_sample = {sid: load_hsqc_peaklist(path, sid) for sid, path in zip(sample_ids, sample_files)}
    if not any(peaks_by_sample.values()):
        raise ValueError("No HSQC peaks were read. Use CPPM,HPPM,intensity or c_ppm,h_ppm,intensity.")
    correction_note = _apply_hsqc_intensity_correction(peaks_by_sample, sample_ids, correction_options, output_dir)
    tracks, shifts, n_pair = build_hsqc_v5_full_tracks(peaks_by_sample, sample_ids, c_tol=c_tol, h_tol=h_tol, c_span_tol=c_span_tol, h_span_tol=h_span_tol, min_track_size=min_track_size, candidate_min_score=candidate_min_score, pair_score_weight=pair_score_weight, reciprocal_best_bonus=reciprocal_best_bonus, top_k_per_seed=top_k_per_seed, max_per_sample=max_per_sample, exact_limit=exact_limit, beam_width=beam_width, node_limit=node_limit, keep_singletons=True)
    labels = _pmtc_hsqc_labels(tracks, sample_ids, pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples, pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples, pmtc_min_cluster_size=pmtc_min_cluster_size)
    groups: Dict[str, List[HSQCTrack]] = defaultdict(list)
    for lab, tr in zip(labels, tracks):
        groups[lab].append(tr)
    blocks: List[ClusterBlock] = []
    shift_txt = "; ".join(f"{sid}:ΔC={shifts.get(sid, (0,0))[0]:.3f},ΔH={shifts.get(sid, (0,0))[1]:.3f}" for sid in sample_ids)
    for cid in sorted(groups, key=lambda x: (x.endswith("unassigned"), x)):
        ts = groups[cid]
        vals = [(float(t.center_c), float(t.center_h)) for t in ts]
        mask_counts: Dict[str, int] = defaultdict(int)
        for t in ts:
            mask_counts[_presence_mask_track(t, sample_ids)] += 1
        mask = max(mask_counts.items(), key=lambda kv: kv[1])[0] if mask_counts else ""
        avg = sum(t.score for t in ts) / len(ts) if ts else 0.0
        details = f"HSQC full 2D-V5 tracks={len(ts)}; pair_candidates={n_pair}; C_tol={c_tol:g}; H_tol={h_tol:g}; C_span≤{c_span_tol:g}; H_span≤{h_span_tol:g}; top_k={top_k_per_seed}; max/sample={max_per_sample}; beam={beam_width}; avg_score={avg:.3f}; dominant_mask={mask}; coarse_shift=[{shift_txt}]"
        details = f"{details}; {correction_note}"
        blocks.append(ClusterBlock(str(cid), "HSQC", vals, len(ts), mask, details))
    return blocks


def _hsqc_track_to_backend_dict(track: HSQCTrack, sample_ids: Sequence[str]) -> dict:
    members = {}
    member_ids = []
    for sid in sample_ids:
        peak = track.members.get(sid)
        if peak is None:
            continue
        c_val = float(_cc(peak))
        h_val = float(_hc(peak))
        signal = float(_hsqc_signal_value(peak))
        area = float(peak.area) if peak.area is not None else signal
        members[sid] = {
            "sample": sid,
            "peak_id": str(peak.peak_id),
            "ppm": c_val,
            "ppm_corr": c_val,
            "c_ppm": c_val,
            "c_ppm_corr": c_val,
            "h_ppm": h_val,
            "h_ppm_corr": h_val,
            "intensity": signal,
            "area": area,
        }
        member_ids.append(str(peak.peak_id))
    return {
        "track_id": track.track_id,
        "member_ids": tuple(sorted(member_ids)),
        "members": members,
        "score": float(track.score),
        "center_ppm": float(track.center_c),
        "center_c": float(track.center_c),
        "center_h": float(track.center_h),
        "c_span": float(track.c_span),
        "h_span": float(track.h_span),
        "reciprocal_fraction": float(track.reciprocal_fraction),
    }


def _hsqc_labels_for_model(
    tracks: List[HSQCTrack],
    sample_ids: Sequence[str],
    *,
    model: str,
    presence_only: bool,
    top_k_per_seed: int,
    max_per_sample: int,
    exact_limit: int,
    beam_width: int,
    node_limit: int,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]],
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]],
    pmtc_min_cluster_size: int,
    guarded_quality_max_cluster_rise: Optional[int] = None,
) -> Tuple[List[str], Dict[str, object]]:
    normalized_model = normalize_model_key(model or "v5_enum")
    mask_models = {
        "v5_frontend_mask",
        "v5_pmtc_mask",
        "v5_mask",
        "v5_enum_mask",
        "v5_enumerated_mask",
        "enumerated_v5_mask",
    }
    if presence_only or normalized_model in mask_models:
        labels = [f"M{_presence_mask_track(track, sample_ids)}" for track in tracks]
        return labels, {
            "backend": model_display_label("v5_frontend_mask", include_full_name=False),
            "model": normalized_model,
            "n_clusters": len(set(labels)),
            "mask_only": True,
        }

    cfg = _build_v5_config(
        model=normalized_model,
        top_k_per_seed=top_k_per_seed,
        max_per_sample=max_per_sample,
        exact_limit=exact_limit,
        beam_width=beam_width,
        node_limit=node_limit,
        pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples,
        pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples,
        pmtc_min_cluster_size=pmtc_min_cluster_size,
        guarded_quality_max_cluster_rise=guarded_quality_max_cluster_rise,
    )

    if normalized_model in {"v5", "v5a", "v5_pmtc", "v5_pmtc_r85", "default"}:
        labels = _pmtc_hsqc_labels(
            tracks,
            sample_ids,
            pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples,
            pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples,
            pmtc_min_cluster_size=pmtc_min_cluster_size,
        )
        return labels, {
            "backend": model_display_label("v5_pmtc", include_full_name=False),
            "model": normalized_model,
            "n_clusters": len(set(labels)),
        }

    backend_tracks = [_hsqc_track_to_backend_dict(track, sample_ids) for track in tracks]
    if normalized_model in {"v5_enum", "v5_enumerated", "enumerated_v5"}:
        from nmr_trendtrack.models.three_model_pipeline import guarded_recall_quality_labels

        current_labels = _pmtc_hsqc_labels(
            tracks,
            sample_ids,
            pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples,
            pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples,
            pmtc_min_cluster_size=pmtc_min_cluster_size,
        )
        labels, guarded_diag = guarded_recall_quality_labels(
            backend_tracks,
            list(sample_ids),
            cfg,
            current_labels=current_labels,
            merge_threshold=float(getattr(cfg.model, "guarded_quality_merge_threshold", 0.80)),
            hac_threshold=float(getattr(cfg.model, "guarded_quality_hac_threshold", 1.00)),
        )
        diag: Dict[str, object] = {
            "backend": model_display_label("v5_enum", include_full_name=False),
            "model": normalized_model,
            "n_clusters": len(set(labels)),
        }
        diag.update({f"guarded_{k}": v for k, v in guarded_diag.items()})
        return labels, diag

    if normalized_model in {"v4", "v4_baseline", "baseline"}:
        from nmr_trendtrack.models.three_model_pipeline import _v4_modules

        _base, gsp = _v4_modules()
        labels, v4_diag = gsp.light_joint_cluster(backend_tracks, list(sample_ids), {"mask_weight": 0.38})
        diag = {
            "backend": model_display_label("v4_baseline", include_full_name=False),
            "model": normalized_model,
            "n_clusters": len(set(labels)),
        }
        diag.update({f"v4_{k}": v for k, v in v4_diag.items()})
        return labels, diag

    raise ValueError(
        f"Unsupported HSQC cross-spectrum model: {normalized_model}. "
        "Use QG-PMTC (v5_enum), PMTC (v5_pmtc), SPTC (v4_baseline), or SP-Mask."
    )


def run_hsqc_cross_clustering(
    sample_files: Sequence[str],
    *,
    model: str = "v5_enum",
    presence_only: bool = False,
    c_tol: float = 1.0,
    h_tol: float = 0.1,
    c_span_tol: float = 1.0,
    h_span_tol: float = 0.1,
    min_track_size: int = 2,
    candidate_min_score: float = 0.40,
    pair_score_weight: float = 1.20,
    reciprocal_best_bonus: float = 0.18,
    top_k_per_seed: int = 6,
    max_per_sample: int = 3,
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None,
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None,
    pmtc_min_cluster_size: int = 3,
    guarded_quality_max_cluster_rise: int = 1,
    correction_options: Optional[dict] = None,
    output_dir: Optional[str | Path] = None,
) -> List[ClusterBlock]:
    if len(sample_files) < 2:
        raise ValueError("HSQC cross-spectrum clustering needs at least two sample peak-list files.")
    if min(c_tol, h_tol, c_span_tol, h_span_tol) <= 0:
        raise ValueError("HSQC C/H tolerance and span tolerances must be positive.")

    sample_ids = [f"S{i + 1}" for i in range(len(sample_files))]
    peaks_by_sample = {sid: load_hsqc_peaklist(path, sid) for sid, path in zip(sample_ids, sample_files)}
    if not any(peaks_by_sample.values()):
        raise ValueError("No HSQC peaks were read. Use CPPM,HPPM,intensity or c_ppm,h_ppm,intensity.")

    correction_note = _apply_hsqc_intensity_correction(peaks_by_sample, sample_ids, correction_options, output_dir)
    tracks, shifts, n_pair = build_hsqc_v5_full_tracks(
        peaks_by_sample,
        sample_ids,
        c_tol=c_tol,
        h_tol=h_tol,
        c_span_tol=c_span_tol,
        h_span_tol=h_span_tol,
        min_track_size=min_track_size,
        candidate_min_score=candidate_min_score,
        pair_score_weight=pair_score_weight,
        reciprocal_best_bonus=reciprocal_best_bonus,
        top_k_per_seed=top_k_per_seed,
        max_per_sample=max_per_sample,
        exact_limit=exact_limit,
        beam_width=beam_width,
        node_limit=node_limit,
        keep_singletons=True,
    )
    labels, model_diag = _hsqc_labels_for_model(
        tracks,
        sample_ids,
        model=model,
        presence_only=presence_only,
        top_k_per_seed=top_k_per_seed,
        max_per_sample=max_per_sample,
        exact_limit=exact_limit,
        beam_width=beam_width,
        node_limit=node_limit,
        pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples,
        pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples,
        pmtc_min_cluster_size=pmtc_min_cluster_size,
        guarded_quality_max_cluster_rise=guarded_quality_max_cluster_rise,
    )
    cleanup_c_tol = max(float(c_tol) * 0.50, 1e-12)
    cleanup_h_tol = max(float(h_tol) * 0.50, 1e-12)
    tracks, labels, residual_diag = _apply_hsqc_mask_residual_filter(
        tracks,
        labels,
        sample_ids,
        c_tol=cleanup_c_tol,
        h_tol=cleanup_h_tol,
    )
    tracks, labels, single_diag = _apply_hsqc_sample_specific_residual_peak_filter(
        tracks,
        labels,
        sample_ids,
    )
    model_diag = dict(model_diag)
    model_diag.update(
        {
            "n_clusters": len(set(labels)),
            "hsqc_mask_cleanup": residual_diag.get("hsqc_mask_residual_output_tracks", ""),
            "hsqc_mask_cleanup_clusters": residual_diag.get("hsqc_mask_residual_output_clusters", ""),
            "hsqc_mask_cleanup_dropped": residual_diag.get("hsqc_mask_residual_dropped_clusters", ""),
            "hsqc_sample_specific_removed": single_diag.get("hsqc_sample_specific_residual_removed_single_tracks", ""),
        }
    )

    groups: Dict[str, List[HSQCTrack]] = defaultdict(list)
    for lab, tr in zip(labels, tracks):
        groups[lab].append(tr)

    blocks: List[ClusterBlock] = []
    rows: List[dict] = []
    shift_txt = "; ".join(
        f"{sid}:dC={shifts.get(sid, (0, 0))[0]:.3f},dH={shifts.get(sid, (0, 0))[1]:.3f}"
        for sid in sample_ids
    )
    sample_order_txt = ",".join(sample_ids)
    diag_txt = "; ".join(f"{k}={v}" for k, v in sorted(model_diag.items()) if k != "model")
    for cid in sorted(groups, key=lambda x: (x.endswith("unassigned"), x)):
        ts = groups[cid]
        vals = [(float(t.center_c), float(t.center_h)) for t in ts]
        mask_counts: Dict[str, int] = defaultdict(int)
        for t in ts:
            mask_counts[_presence_mask_track(t, sample_ids)] += 1
        mask = max(mask_counts.items(), key=lambda kv: kv[1])[0] if mask_counts else ""
        avg = sum(t.score for t in ts) / len(ts) if ts else 0.0
        mask_counts_txt = ",".join(f"{m}:{n}" for m, n in sorted(mask_counts.items(), reverse=True))
        no_pmtc_note = "; presence-mask grouping only; no PMTC/QG refinement" if presence_only or str(cid).startswith("M") else ""
        details = (
            f"HSQC full 2D cross-spectrum model={model_display_label(model)}; tracks={len(ts)}; "
            f"pair_candidates={n_pair}; sample_order={sample_order_txt}; "
            f"C_tol={c_tol:g}; H_tol={h_tol:g}; C_span<={c_span_tol:g}; H_span<={h_span_tol:g}; "
            f"min_track_size={min_track_size}; candidate_min_score={candidate_min_score:g}; "
            f"pair_score_weight={pair_score_weight:g}; reciprocal_bonus={reciprocal_best_bonus:g}; "
            f"top_k={top_k_per_seed}; max/sample={max_per_sample}; exact_limit={exact_limit}; "
            f"beam={beam_width}; node_limit={node_limit}; avg_score={avg:.3f}; "
            f"dominant_mask={mask}; mask_counts={mask_counts_txt}; coarse_shift=[{shift_txt}]; "
            f"{diag_txt}{no_pmtc_note}; {correction_note}"
        )
        blocks.append(ClusterBlock(str(cid), "HSQC", vals, len(ts), mask, details))
        rows.append({
            "cluster_id": str(cid),
            "presence_mask": mask,
            "n_tracks": len(ts),
            "points": ";".join(f"({c:.4f},{h:.4f})" for c, h in vals),
            "details": details,
        })

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if rows:
            out_name = "hsqc_presence_mask_only_clusters.csv" if presence_only else "hsqc_cross_clusters.csv"
            with (out / out_name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return blocks


def run_hsqc_presence_mask_clustering(sample_files: Sequence[str], **kwargs) -> List[ClusterBlock]:
    kwargs["presence_only"] = True
    return run_hsqc_cross_clustering(sample_files, **kwargs)


def _project_hsqc_track_to_common_mask(
    track: HSQCTrack,
    common_mask: str,
    sample_ids: Sequence[str],
) -> Optional[HSQCTrack]:
    projected_members = {
        sid: track.members[sid]
        for sid, bit in zip(sample_ids, str(common_mask))
        if bit == "1" and sid in track.members
    }
    if not projected_members:
        return None
    cvals = [float(_cc(p)) for p in projected_members.values()]
    hvals = [float(_hc(p)) for p in projected_members.values()]
    center_c = float(median(cvals))
    center_h = float(median(hvals))
    c_span = max(cvals) - min(cvals) if len(cvals) > 1 else 0.0
    h_span = max(hvals) - min(hvals) if len(hvals) > 1 else 0.0
    projected = HSQCTrack(
        "",
        projected_members,
        center_c,
        center_h,
        c_span,
        h_span,
        float(track.score),
        float(track.reciprocal_fraction),
    )
    projected.source_strict_mask = _presence_mask_track(track, sample_ids)
    projected.common_mask = str(common_mask)
    projected.source_track_id = str(track.track_id)
    return projected


def run_hsqc_common_mask_backend_clustering(
    sample_files: Sequence[str],
    *,
    model: str = "v5_enum",
    presence_only: bool = False,
    c_tol: float = 1.0,
    h_tol: float = 0.1,
    c_span_tol: float = 1.0,
    h_span_tol: float = 0.1,
    min_track_size: int = 2,
    candidate_min_score: float = 0.40,
    pair_score_weight: float = 1.20,
    reciprocal_best_bonus: float = 0.18,
    top_k_per_seed: int = 6,
    max_per_sample: int = 3,
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None,
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None,
    pmtc_min_cluster_size: int = 3,
    guarded_quality_max_cluster_rise: int = 4,
    correction_options: Optional[dict] = None,
    output_dir: Optional[str | Path] = None,
) -> List[ClusterBlock]:
    if len(sample_files) < 2:
        raise ValueError("HSQC common-mask clustering needs at least two sample peak-list files.")
    if min(c_tol, h_tol, c_span_tol, h_span_tol) <= 0:
        raise ValueError("HSQC C/H tolerance and span tolerances must be positive.")

    sample_ids = [f"S{i + 1}" for i in range(len(sample_files))]
    peaks_by_sample = {sid: load_hsqc_peaklist(path, sid) for sid, path in zip(sample_ids, sample_files)}
    if not any(peaks_by_sample.values()):
        raise ValueError("No HSQC peaks were read. Use CPPM,HPPM,intensity or c_ppm,h_ppm,intensity.")

    correction_note = _apply_hsqc_intensity_correction(peaks_by_sample, sample_ids, correction_options, output_dir)
    strict_tracks, shifts, n_pair = build_hsqc_v5_full_tracks(
        peaks_by_sample,
        sample_ids,
        c_tol=c_tol,
        h_tol=h_tol,
        c_span_tol=c_span_tol,
        h_span_tol=h_span_tol,
        min_track_size=min_track_size,
        candidate_min_score=candidate_min_score,
        pair_score_weight=pair_score_weight,
        reciprocal_best_bonus=reciprocal_best_bonus,
        top_k_per_seed=top_k_per_seed,
        max_per_sample=max_per_sample,
        exact_limit=exact_limit,
        beam_width=beam_width,
        node_limit=node_limit,
        keep_singletons=True,
    )

    common_tracks: List[HSQCTrack] = []
    pool_sizes: Dict[str, int] = {}
    selected_sizes: Dict[str, int] = {}
    source_mask_counts: Dict[str, Dict[str, int]] = {}
    for common_mask in _all_nonempty_masks(len(sample_ids)):
        pool: List[HSQCTrack] = []
        for track in strict_tracks:
            strict_mask = _presence_mask_track(track, sample_ids)
            if _mask_popcount(common_mask) >= 2:
                if not _mask_contains(strict_mask, common_mask):
                    continue
            elif strict_mask != common_mask:
                continue
            projected = _project_hsqc_track_to_common_mask(track, common_mask, sample_ids)
            if projected is not None:
                pool.append(projected)
        if not pool:
            continue
        selected = _pack_hsqc_2d_tracks(pool, int(beam_width), int(exact_limit), int(node_limit))
        if not selected:
            continue
        for track in selected:
            track.common_mask = common_mask
        common_tracks.extend(selected)
        pool_sizes[common_mask] = len(pool)
        selected_sizes[common_mask] = len(selected)
        counts: Dict[str, int] = defaultdict(int)
        for track in selected:
            counts[str(getattr(track, "source_strict_mask", _presence_mask_track(track, sample_ids)))] += 1
        source_mask_counts[common_mask] = dict(counts)

    labels, model_diag = _hsqc_labels_for_model(
        common_tracks,
        sample_ids,
        model=model,
        presence_only=presence_only,
        top_k_per_seed=top_k_per_seed,
        max_per_sample=max_per_sample,
        exact_limit=exact_limit,
        beam_width=beam_width,
        node_limit=node_limit,
        pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples,
        pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples,
        pmtc_min_cluster_size=pmtc_min_cluster_size,
        guarded_quality_max_cluster_rise=guarded_quality_max_cluster_rise,
    )
    cleanup_c_tol = max(float(c_tol) * 0.50, 1e-12)
    cleanup_h_tol = max(float(h_tol) * 0.50, 1e-12)
    common_tracks, labels, residual_diag = _apply_hsqc_mask_residual_filter(
        common_tracks,
        labels,
        sample_ids,
        c_tol=cleanup_c_tol,
        h_tol=cleanup_h_tol,
    )
    common_tracks, labels, single_diag = _apply_hsqc_sample_specific_residual_peak_filter(
        common_tracks,
        labels,
        sample_ids,
    )
    model_diag = dict(model_diag)
    model_diag.update(
        {
            "n_clusters": len(set(labels)),
            "hsqc_mask_cleanup": residual_diag.get("hsqc_mask_residual_output_tracks", ""),
            "hsqc_mask_cleanup_clusters": residual_diag.get("hsqc_mask_residual_output_clusters", ""),
            "hsqc_mask_cleanup_dropped": residual_diag.get("hsqc_mask_residual_dropped_clusters", ""),
            "hsqc_sample_specific_removed": single_diag.get("hsqc_sample_specific_residual_removed_single_tracks", ""),
        }
    )

    groups: Dict[str, List[HSQCTrack]] = defaultdict(list)
    for lab, tr in zip(labels, common_tracks):
        groups[lab].append(tr)

    blocks: List[ClusterBlock] = []
    rows: List[dict] = []
    shift_txt = "; ".join(
        f"{sid}:dC={shifts.get(sid, (0, 0))[0]:.3f},dH={shifts.get(sid, (0, 0))[1]:.3f}"
        for sid in sample_ids
    )
    sample_order_txt = ",".join(sample_ids)
    diag_txt = "; ".join(f"{k}={v}" for k, v in sorted(model_diag.items()) if k != "model")
    for cid in sorted(groups, key=lambda x: (x.endswith("unassigned"), x)):
        ts = groups[cid]
        vals = [(float(t.center_c), float(t.center_h)) for t in ts]
        mask_counts: Dict[str, int] = defaultdict(int)
        source_counts: Dict[str, int] = defaultdict(int)
        pool_total = 0
        selected_total = 0
        for t in ts:
            mask = _presence_mask_track(t, sample_ids)
            mask_counts[mask] += 1
            source_counts[str(getattr(t, "source_strict_mask", ""))] += 1
            pool_total += int(pool_sizes.get(str(getattr(t, "common_mask", mask)), 0))
            selected_total += int(selected_sizes.get(str(getattr(t, "common_mask", mask)), 0))
        dominant_mask = max(mask_counts.items(), key=lambda kv: kv[1])[0] if mask_counts else ""
        mask_counts_txt = ",".join(f"{m}:{n}" for m, n in sorted(mask_counts.items(), reverse=True))
        source_counts_txt = ",".join(f"{m}:{n}" for m, n in sorted(source_counts.items(), reverse=True) if m)
        no_pmtc_note = "; common-mask output only; no PMTC/QG refinement" if presence_only or str(cid).startswith("M") else ""
        details = (
            f"HSQC common-mask front end + {model_display_label(model)} tracks={len(ts)}; "
            f"pair_candidates={n_pair}; sample_order={sample_order_txt}; "
            f"C_tol={c_tol:g}; H_tol={h_tol:g}; C_span<={c_span_tol:g}; H_span<={h_span_tol:g}; "
            f"min_track_size={min_track_size}; candidate_min_score={candidate_min_score:g}; "
            f"pair_score_weight={pair_score_weight:g}; reciprocal_bonus={reciprocal_best_bonus:g}; "
            f"top_k={top_k_per_seed}; max/sample={max_per_sample}; exact_limit={exact_limit}; "
            f"beam={beam_width}; node_limit={node_limit}; dominant_mask={dominant_mask}; "
            f"mask_counts={mask_counts_txt}; source_strict_masks={source_counts_txt}; "
            f"common_pool_sum={pool_total}; common_selected_sum={selected_total}; "
            f"coarse_shift=[{shift_txt}]; {diag_txt}{no_pmtc_note}; {correction_note}"
        )
        blocks.append(ClusterBlock(str(cid), "HSQC", vals, len(ts), dominant_mask, details))
        rows.append({
            "cluster_id": str(cid),
            "presence_mask": dominant_mask,
            "n_tracks": len(ts),
            "points": ";".join(f"({c:.4f},{h:.4f})" for c, h in vals),
            "mask_counts": mask_counts_txt,
            "source_strict_masks": source_counts_txt,
            "details": details,
        })

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if rows:
            out_name = "hsqc_common_mask_only_clusters.csv" if presence_only else "hsqc_common_mask_backend_clusters.csv"
            with (out / out_name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return blocks


def run_hsqc_v5_full_clustering(
    sample_files: Sequence[str],
    *,
    model: str = "v5_pmtc",
    presence_only: bool = False,
    c_tol: float = 1.0,
    h_tol: float = 0.1,
    c_span_tol: float = 1.0,
    h_span_tol: float = 0.1,
    min_track_size: int = 2,
    candidate_min_score: float = 0.40,
    pair_score_weight: float = 1.20,
    reciprocal_best_bonus: float = 0.18,
    top_k_per_seed: int = 6,
    max_per_sample: int = 3,
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None,
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None,
    pmtc_min_cluster_size: int = 3,
    guarded_quality_max_cluster_rise: int = 1,
    correction_options: Optional[dict] = None,
    output_dir: Optional[str | Path] = None,
) -> List[ClusterBlock]:
    """Compatibility wrapper for the old HSQC V5 entry point.

    The implementation now shares the same model switch as 13C cross-spectrum
    clustering. The default remains PMTC (internal key v5_pmtc) to preserve existing callers.
    """
    return run_hsqc_cross_clustering(
        sample_files,
        model=model,
        presence_only=presence_only,
        c_tol=c_tol,
        h_tol=h_tol,
        c_span_tol=c_span_tol,
        h_span_tol=h_span_tol,
        min_track_size=min_track_size,
        candidate_min_score=candidate_min_score,
        pair_score_weight=pair_score_weight,
        reciprocal_best_bonus=reciprocal_best_bonus,
        top_k_per_seed=top_k_per_seed,
        max_per_sample=max_per_sample,
        exact_limit=exact_limit,
        beam_width=beam_width,
        node_limit=node_limit,
        pmtc_max_tracks_by_n_samples=pmtc_max_tracks_by_n_samples,
        pmtc_frac_limit_by_n_samples=pmtc_frac_limit_by_n_samples,
        pmtc_min_cluster_size=pmtc_min_cluster_size,
        guarded_quality_max_cluster_rise=guarded_quality_max_cluster_rise,
        correction_options=correction_options,
        output_dir=output_dir,
    )


# Override the earlier compatibility function: GUI code that still calls the old name now gets the full 2D V5 backend.
def run_hsqc_v5_style_clustering(*args, **kwargs) -> List[ClusterBlock]:
    return run_hsqc_v5_full_clustering(*args, **kwargs)

# Late helper needed by the appended full 2D HSQC backend.
def _hsqc_peak_lookup(peaks_by_sample: Dict[str, List[HSQCPeak]]) -> Dict[str, HSQCPeak]:
    return {p.peak_id: p for arr in peaks_by_sample.values() for p in arr}

# Override HSQC PMTC labels with trend-aware small-bucket splitting.  The older helper only
# split oversized presence-mask buckets; this keeps the V5 spirit by using intensity-trend
# features whenever a bucket has enough tracks to form at least two interpretable clusters.
def _pmtc_hsqc_labels(
    tracks: List[HSQCTrack],
    sample_ids: Sequence[str],
    *,
    pmtc_max_tracks_by_n_samples: Optional[Dict[int, int]] = None,
    pmtc_frac_limit_by_n_samples: Optional[Dict[int, float]] = None,
    pmtc_min_cluster_size: int = 3,
) -> List[str]:
    n = len(tracks)
    ns = len(sample_ids)
    max_tracks_map = pmtc_max_tracks_by_n_samples or {3: 32, 4: 28, 5: 26}
    frac_map = pmtc_frac_limit_by_n_samples or {3: 0.55, 4: 0.47, 5: 0.41}
    max_tracks = int(max_tracks_map.get(ns, max_tracks_map.get(str(ns), 26)))
    frac_limit = float(frac_map.get(ns, frac_map.get(str(ns), 0.41)))
    min_size = max(1, int(pmtc_min_cluster_size))

    buckets: Dict[str, List[int]] = defaultdict(list)
    unassigned: List[int] = []
    for i, t in enumerate(tracks):
        if len(t.members) >= 2:
            buckets[_presence_mask_track(t, sample_ids)].append(i)
        else:
            unassigned.append(i)

    def feature_sd(idxs: List[int]) -> float:
        if len(idxs) <= 1:
            return 0.0
        X = [_track_feature_hsqc(tracks[i], sample_ids) for i in idxs]
        if not X:
            return 0.0
        d = len(X[0])
        vals = []
        for j in range(d):
            m = sum(x[j] for x in X) / len(X)
            vals.append(math.sqrt(sum((x[j] - m) ** 2 for x in X) / max(1, len(X) - 1)))
        return max(vals) if vals else 0.0

    clusters: List[List[int]] = []
    for _, idxs in sorted(buckets.items()):
        idxs = list(idxs)
        size_k = max(1, math.ceil(len(idxs) / max(1, max_tracks)))
        frac_k = max(1, math.ceil(len(idxs) / max(1, int(math.ceil(frac_limit * max(n, 1))))))
        trend_k = 1
        if len(idxs) >= 2 * min_size and feature_sd(idxs) > 0.45:
            trend_k = min(4, max(2, len(idxs) // min_size))
        k = max(size_k, frac_k, trend_k)
        parts = _split_hsqc_indices(idxs, tracks, sample_ids, k) if k > 1 else [idxs]
        # Keep undersized fragments attached to nearest larger fragment by order to avoid unusable one-peak clusters.
        good = [p for p in parts if len(p) >= min_size]
        small = [p for p in parts if len(p) < min_size]
        if good and small:
            for sm in small:
                nearest = min(good, key=lambda g: abs(sum(tracks[i].center_c for i in g) / len(g) - sum(tracks[i].center_c for i in sm) / len(sm)))
                nearest.extend(sm)
            parts = good
        clusters.extend([p for p in parts if p])

    labels = ["HSQC_unassigned"] * n
    clusters = sorted(clusters, key=lambda c: min(c) if c else 10**9)
    for ci, c in enumerate(clusters, 1):
        for i in c:
            labels[i] = f"H{ci:02d}"
    for i in unassigned:
        labels[i] = "HSQC_unassigned"
    return labels
