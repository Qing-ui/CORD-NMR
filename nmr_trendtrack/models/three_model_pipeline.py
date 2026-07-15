from __future__ import annotations

import csv
import importlib.util
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

from nmr_trendtrack.config import AppConfig
from nmr_trendtrack.contracts import (
    ClusterPrototype,
    ComponentClusterPrototype,
    FinalClusterPrototype,
    JointState,
    Membership,
    Peak,
    Sample,
    Track,
    TrendVector,
)


_THIS_DIR = Path(__file__).resolve().parent
_V4_DIR = _THIS_DIR / "v4_reference"


def _load_peaklist_for_sample_light(sample: Sample) -> List[Peak]:
    """Lightweight CSV/TSV peak loader used by SPTC/PMTC to avoid pandas dependency.

    Expected columns: ppm plus intensity or area. If area is absent, intensity is reused.
    """
    if not sample.peaklist_path:
        raise ValueError(f"Sample {sample.sample_id} missing peaklist_path")
    path = Path(sample.peaklist_path)
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    # CSV first; if it fails to expose ppm, fallback to whitespace parsing.
    rows = []
    try:
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames and any(str(c).strip().lower() == "ppm" for c in reader.fieldnames):
            for r in reader:
                norm = {str(k).strip().lower(): v for k, v in r.items()}
                ppm = float(norm.get("ppm") or norm.get("shift") or norm.get("chemical_shift"))
                intensity = float(norm.get("intensity") or norm.get("height") or norm.get("area") or 1.0)
                area_val = norm.get("area")
                area = float(area_val) if area_val not in (None, "") else intensity
                rows.append((ppm, intensity, area))
    except Exception:
        rows = []
    if not rows:
        import re
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith(("#", ";", "//")):
                continue
            toks = re.split(r"[,;\t\s]+", line.strip())
            nums = []
            for t in toks:
                try:
                    nums.append(float(t))
                except Exception:
                    pass
            if len(nums) >= 2:
                rows.append((nums[0], nums[1], nums[2] if len(nums) > 2 else nums[1]))
    peaks = [
        Peak(
            peak_id=f"{sample.sample_id}_P{i:03d}",
            sample_id=sample.sample_id,
            ppm_raw=float(ppm),
            ppm_corr=float(ppm),
            intensity=float(intensity),
            area=float(area),
        )
        for i, (ppm, intensity, area) in enumerate(sorted(rows, key=lambda x: x[0]), 1)
    ]
    return peaks


def _load_v4_module(filename: str, module_name: str):
    path = _V4_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy-loaded so TTC does not pay the import cost.
_V4_BASE = None
_V4_GSP = None


def _v4_modules():
    global _V4_BASE, _V4_GSP
    if _V4_BASE is None or _V4_GSP is None:
        _V4_BASE = _load_v4_module("13c_joint_align_cluster.py", "nmr_trendtrack_v4_base")
        _V4_GSP = _load_v4_module("13c_global_setpacking_best3.py", "nmr_trendtrack_v4_gsp")
    return _V4_BASE, _V4_GSP


def _ordered_ids(samples: List[Sample]) -> List[str]:
    return [s.sample_id for s in sorted(samples, key=lambda x: x.order_index)]


def _rows_from_peaklists(samples: List[Sample]) -> Tuple[List[str], List[dict], Dict[str, List[Peak]]]:
    ordered = sorted(samples, key=lambda x: x.order_index)
    rows: List[dict] = []
    peak_objs: Dict[str, List[Peak]] = {}
    for s in ordered:
        peaks = _load_peaklist_for_sample_light(s)
        peaks = sorted(peaks, key=lambda p: p.ppm_raw)
        peak_objs[s.sample_id] = peaks
        for i, p in enumerate(peaks, 1):
            pid = p.peak_id or f"{s.sample_id}_P{i:03d}"
            rows.append(
                {
                    "sample": s.sample_id,
                    "peak_id": pid,
                    "ppm": float(p.ppm_raw),
                    "ppm_corr": float(p.corrected_ppm()),
                    "area": float(p.area if p.area is not None else p.intensity),
                    "intensity": float(p.intensity if p.intensity is not None else (p.area or 1.0)),
                    # Empty truth placeholders keep the legacy V4 diagnostic helpers safe.
                    "component": "",
                    "source_id": pid,
                    "atom_id": "",
                    "template_ppm": "",
                    "concentration": "",
                }
            )
    return [s.sample_id for s in ordered], rows, peak_objs


def _peak_from_v4_member(p: dict) -> Peak:
    return Peak(
        peak_id=str(p.get("peak_id") or ""),
        sample_id=str(p.get("sample") or ""),
        ppm_raw=float(p.get("ppm", p.get("ppm_corr", 0.0))),
        ppm_corr=float(p.get("ppm_corr", p.get("ppm", 0.0))),
        intensity=float(p.get("intensity", p.get("area", 1.0)) or 1.0),
        area=float(p.get("area", p.get("intensity", 1.0)) or 1.0),
    )


def _to_contract_tracks(v4_tracks: List[dict], ordered_sample_ids: List[str]) -> List[Track]:
    out: List[Track] = []
    for i, t in enumerate(v4_tracks, 1):
        members = {sid: _peak_from_v4_member(p) for sid, p in t.get("members", {}).items()}
        vals = [p.corrected_ppm() for p in members.values()]
        center = float(median(vals)) if vals else 0.0
        span = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
        mask = tuple(1 if sid in members else 0 for sid in ordered_sample_ids)
        out.append(
            Track(
                track_id=str(t.get("track_id") or f"T{i:04d}"),
                members=members,
                center_ppm=center,
                ppm_span=span,
                presence_mask=mask,
                quality_score=float(t.get("score", 0.0) or 0.0),
            )
        )
    return out


def _trend_vec_for_track(track: Track, ordered_sample_ids: List[str]) -> TrendVector:
    vals = []
    for sid in ordered_sample_ids:
        p = track.members.get(sid)
        vals.append(None if p is None else math.log(max(float(p.intensity if p.intensity is not None else (p.area or 1.0)), 1e-12)))
    steps: List[float | None] = []
    valid: List[bool] = []
    for a, b in zip(vals, vals[1:]):
        if a is None or b is None:
            steps.append(None)
            valid.append(False)
        else:
            steps.append(float(b - a))
            valid.append(True)
    return TrendVector(track_id=track.track_id, presence_mask=track.presence_mask, step_log_fc=steps, valid_steps=valid)


def _make_state(
    samples: List[Sample],
    ordered_sample_ids: List[str],
    peaks_original: Dict[str, List[Peak]],
    shifts: Dict[str, float],
    v4_tracks: List[dict],
    labels: List[str],
    *,
    objective_value: float = 0.0,
    iterations: int = 1,
) -> JointState:
    tracks = _to_contract_tracks(v4_tracks, ordered_sample_ids)
    trend_vectors = [_trend_vec_for_track(t, ordered_sample_ids) for t in tracks]

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[str(lab)].append(i)

    prototypes: List[ClusterPrototype] = []
    component_protos: List[ComponentClusterPrototype] = []
    final_protos: List[FinalClusterPrototype] = []
    memberships: List[Membership] = []

    for lab, idxs in sorted(groups.items()):
        if not idxs:
            continue
        mask_counts = Counter(tracks[i].presence_mask for i in idxs)
        mask = mask_counts.most_common(1)[0][0]
        n_steps = max(0, len(ordered_sample_ids) - 1)
        mean_steps = []
        scale_steps = []
        for j in range(n_steps):
            vals = [trend_vectors[i].step_log_fc[j] for i in idxs if trend_vectors[i].step_log_fc[j] is not None]
            if vals:
                m = sum(float(v) for v in vals) / len(vals)
                var = sum((float(v) - m) ** 2 for v in vals) / max(1, len(vals) - 1)
                mean_steps.append(float(m))
                scale_steps.append(max(math.sqrt(var), 1e-6))
            else:
                mean_steps.append(0.0)
                scale_steps.append(1.0)
        prototypes.append(ClusterPrototype(cluster_id=lab, presence_mask=mask, mean_step_log_fc=mean_steps, step_scale=scale_steps, n_tracks=len(idxs), weight=float(len(idxs))))
        component_protos.append(ComponentClusterPrototype(component_cluster_id=lab, presence_mask=mask, n_tracks=len(idxs), source_cluster_ids=[lab]))
        final_protos.append(FinalClusterPrototype(cluster_id=lab, presence_mask=mask, n_tracks=len(idxs), source_cluster_ids=[lab], merge_mode="three_model_switch"))

    for i, t in enumerate(tracks):
        lab = str(labels[i]) if i < len(labels) else "C_unassigned"
        memberships.append(
            Membership(
                track_id=t.track_id,
                cluster_probs={lab: 1.0},
                best_cluster_id=lab,
                second_cluster_id=None,
                assigned_label="pure",
                component_cluster_id=lab,
                final_cluster_id=lab,
            )
        )

    return JointState(
        samples=samples,
        ordered_sample_ids=ordered_sample_ids,
        peaks_original=peaks_original,
        peaks_corrected=peaks_original,
        shifts=shifts,
        tracks=tracks,
        trend_vectors=trend_vectors,
        cluster_prototypes=prototypes,
        component_cluster_prototypes=component_protos,
        final_cluster_prototypes=final_protos,
        memberships=memberships,
        objective_value=objective_value,
        outer_iterations_completed=iterations,
        best_iteration=iterations,
        converged=True,
    )


def _setpacking_track_weight(model_cost: float, high_mask_bonus: float):
    def weight(track: dict) -> float:
        n = len(track["member_ids"])
        score = max(0.0, float(track.get("score", 0.0)))
        if n <= 1:
            return 0.02
        return score * n - float(model_cost) + float(high_mask_bonus) * max(0, n - 2) ** 2

    return weight


def _run_v4_frontend(samples: List[Sample], config: AppConfig) -> Tuple[List[str], Dict[str, List[Peak]], Dict[str, float], List[dict], dict]:
    base, gsp = _v4_modules()
    ordered_ids, rows, peaks_original = _rows_from_peaklists(samples)

    # CORD-NMR exposes the V5/V4-front-end tolerances in the GUI.
    # The original reference file stores them as module globals, so patch them
    # before running alignment/track enumeration.
    region_windows = list(getattr(config.align, "ppm_window_by_region", []) or [])
    if region_windows:
        base.REGIONS = [(float(lo), float(hi), float(win)) for lo, hi, win in region_windows]
        base.REGION_NAMES = [f"{int(lo)}-{int(hi)}" for lo, hi, _ in base.REGIONS]
    base.RAW_SPAN_LIMIT = float(getattr(config.align, "max_track_span_ppm", 0.50) or 0.50)

    residual_gate = float(getattr(config.model, "residual_gate", 0.15))
    top_k = int(getattr(config.model, "top_k_per_seed", 6))
    max_per_sample = int(getattr(config.model, "max_per_sample", 3))
    exact_limit = int(getattr(config.model, "exact_limit", 12))
    beam_width = int(getattr(config.model, "beam_width", 120))
    node_limit = int(getattr(config.model, "node_limit", 100000))
    model_cost = float(getattr(config.model, "setpacking_model_cost", 1.10) or 1.10)
    high_mask_bonus = float(getattr(config.model, "setpacking_high_mask_bonus", 0.0) or 0.0)
    gsp.track_weight = _setpacking_track_weight(model_cost, high_mask_bonus)

    corrected, shift_model, shift_diag = base.apply_shift(rows, ordered_ids)
    cand = base.enumerate_candidate_tracks(
        corrected,
        ordered_ids,
        residual_gate=residual_gate,
        top_k_per_seed=top_k,
        max_per_sample=max_per_sample,
        min_samples=2,
    )
    pack_fn = getattr(gsp, "mask_stratified_global_set_packing", gsp.componentwise_global_set_packing)
    selected, pack_info = pack_fn(cand, exact_limit=exact_limit, beam_width=beam_width, node_limit=node_limit)
    shifts = {sid: 0.0 for sid in ordered_ids}
    for sid in ordered_ids:
        vals = []
        for reg in getattr(base, "REGION_NAMES", []):
            try:
                vals.append(float(shift_model.get(sid, {}).get(reg, 0.0)))
            except Exception:
                pass
        shifts[sid] = float(sum(vals) / len(vals)) if vals else 0.0
    diag = {
        "n_candidate_tracks": len(cand),
        "n_selected_tracks": len(selected),
        "setpacking_model_cost": model_cost,
        "setpacking_high_mask_bonus": high_mask_bonus,
        **pack_info,
    }
    return ordered_ids, peaks_original, shifts, selected, diag


# ----- PMTC backend -----

def _track_mask(track: dict, samples: List[str]) -> str:
    return "".join("1" if track.get("members", {}).get(s) else "0" for s in samples)


def _mask_popcount(mask: str) -> int:
    return str(mask or "").count("1")


def _mask_superset(high: str, low: str) -> bool:
    high = str(high or "")
    low = str(low or "")
    if len(high) != len(low) or high == low:
        return False
    return all(h == "1" or l == "0" for h, l in zip(high, low))


def _track_center_ppm(track: dict) -> float:
    vals = [
        float(p.get("ppm_corr", p.get("ppm", 0.0)))
        for p in track.get("members", {}).values()
    ]
    return float(median(vals)) if vals else 0.0


def _track_log_vec(track: dict, samples: List[str]) -> List[float]:
    vals = []
    for s in samples:
        p = track.get("members", {}).get(s)
        vals.append(None if not p else math.log(max(float(p.get("intensity", p.get("area", 1.0)) or 1.0), 1e-12)))
    obs = [v for v in vals if v is not None]
    med = median(obs) if obs else 0.0
    return [0.0 if v is None else float(v - med) for v in vals]


def _track_feature(track: dict, samples: List[str]) -> List[float]:
    v = _track_log_vec(track, samples)
    steps = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    return v + steps + [0.15 * min(1.5, float(track.get("score", 0) or 0))]


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


def _euclid(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a)))


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


def _split_indices_by_feature(indices: List[int], tracks: List[dict], samples: List[str], k: int) -> List[List[int]]:
    if len(indices) <= 1 or k <= 1:
        return [indices]
    X = _normalize_features([_track_feature(tracks[i], samples) for i in indices])
    labels = _kmeans_labels(X, k)
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in zip(indices, labels):
        groups[lab].append(idx)
    out = [g for _, g in sorted(groups.items()) if g]
    # Safety fallback: if k-means degenerates to one group, deterministic ppm slicing prevents infinite size-control loops.
    if len(out) == 1 and k > 1 and len(indices) > 1:
        ordered = sorted(indices, key=lambda i: median([p.get("ppm_corr", p.get("ppm", 0.0)) for p in tracks[i].get("members", {}).values()]))
        chunk = max(1, math.ceil(len(ordered) / k))
        out = [ordered[i : i + chunk] for i in range(0, len(ordered), chunk)]
    return out


def _matched_low_indices_by_ppm(
    low_indices: Sequence[int],
    high_indices: Sequence[int],
    tracks: Sequence[dict],
    tol: float,
) -> set[int]:
    high_centers = [_track_center_ppm(tracks[hi]) for hi in high_indices]
    matched: set[int] = set()
    for li in low_indices:
        lc = _track_center_ppm(tracks[li])
        if any(abs(lc - hc) <= tol for hc in high_centers):
            matched.add(li)
    return matched


def _matched_low_indices_by_ppm_greedy(
    low_indices: Sequence[int],
    high_indices: Sequence[int],
    tracks: Sequence[dict],
    tol: float,
) -> set[int]:
    """One-to-one ppm matching: a high-mask track can explain at most one low-mask track."""
    pairs = []
    for li in low_indices:
        lc = _track_center_ppm(tracks[li])
        for hi in high_indices:
            hc = _track_center_ppm(tracks[hi])
            d = abs(lc - hc)
            if d <= tol:
                pairs.append((d, li, hi))
    pairs.sort()
    used_low: set[int] = set()
    used_high: set[int] = set()
    for _d, li, hi in pairs:
        if li in used_low or hi in used_high:
            continue
        used_low.add(li)
        used_high.add(hi)
    return used_low


def _label_prefix(labels: Sequence[str]) -> str:
    for preferred in ("H", "C", "R", "M"):
        if any(str(label).startswith(preferred) for label in labels):
            return preferred
    return "C"


def _apply_mask_residual_filter(
    tracks: List[dict],
    labels: Sequence[str],
    samples: List[str],
    config: AppConfig,
) -> Tuple[List[dict], List[str], dict]:
    mode = str(getattr(config.model, "mask_residual_mode", "full_coverage_one_to_one") or "full_coverage_one_to_one").lower()
    if (not bool(getattr(config.model, "enable_mask_residual_filter", True))) or mode in {"off", "none", "disabled"}:
        return tracks, list(labels), {"mask_residual_enabled": False, "mask_residual_mode": mode}

    tol = float(getattr(config.model, "mask_residual_match_tol", 0.50) or 0.50)
    min_remaining = int(getattr(config.model, "mask_residual_min_remaining", 5) or 5)
    max_cover_ratio = float(getattr(config.model, "mask_residual_max_cover_ratio", 1.50) or 1.50)
    pool_single = bool(getattr(config.model, "pool_single_sample_masks", True))

    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, (track, label) in enumerate(zip(tracks, labels)):
        mask = _track_mask(track, samples)
        if pool_single and _mask_popcount(mask) == 1:
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
    residualized = 0
    matched_tracks = 0
    prefix = _label_prefix(labels)
    for piece in pieces:
        mask = str(piece["mask"])
        idxs = list(piece["idxs"])
        high_indices = [
            idx
            for acc in accepted
            if _mask_superset(str(acc["mask"]), mask)
            for idx in acc["idxs"]
        ]
        cover_size_ok = bool(high_indices) and len(high_indices) <= max_cover_ratio * max(1, len(idxs))
        if mode in {"full_coverage_delete", "full", "delete_full"}:
            matched = _matched_low_indices_by_ppm(idxs, high_indices, tracks, tol) if high_indices else set()
            matched_tracks += len(matched)
            if cover_size_ok and len(matched) == len(idxs):
                dropped += 1
                continue
            accepted.append({"mask": mask, "idxs": idxs, "source_label": piece["source_label"]})
            continue

        if mode in {"full_coverage_one_to_one", "full_one_to_one", "delete_full_one_to_one"}:
            matched = _matched_low_indices_by_ppm_greedy(idxs, high_indices, tracks, tol) if high_indices else set()
            matched_tracks += len(matched)
            if cover_size_ok and len(matched) == len(idxs):
                dropped += 1
                continue
            accepted.append({"mask": mask, "idxs": idxs, "source_label": piece["source_label"]})
            continue

        matched = _matched_low_indices_by_ppm_greedy(idxs, high_indices, tracks, tol) if high_indices else set()
        residual = [idx for idx in idxs if idx not in matched]
        matched_tracks += len(matched)
        if matched:
            residualized += 1
        if matched and len(residual) < min_remaining:
            dropped += 1
            continue
        if not residual:
            dropped += 1
            continue
        accepted.append({"mask": mask, "idxs": residual, "source_label": piece["source_label"]})

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
        "mask_residual_enabled": True,
        "mask_residual_mode": mode,
        "mask_residual_match_tol": tol,
        "mask_residual_min_remaining": min_remaining,
        "mask_residual_max_cover_ratio": max_cover_ratio,
        "mask_residual_pool_single": pool_single,
        "mask_residual_input_clusters": len(pieces),
        "mask_residual_output_clusters": len(accepted),
        "mask_residual_dropped_clusters": dropped,
        "mask_residual_residualized_clusters": residualized,
        "mask_residual_matched_tracks": matched_tracks,
        "mask_residual_input_tracks": len(tracks),
        "mask_residual_output_tracks": len(filtered_tracks),
    }


def _apply_sample_specific_residual_peak_filter(
    tracks: List[dict],
    labels: Sequence[str],
    samples: List[str],
    config: AppConfig,
) -> Tuple[List[dict], List[str], dict]:
    """Keep single-sample tracks only when their raw peak is not used by a multi-sample track."""
    if not bool(getattr(config.model, "enable_sample_specific_residual_peaks", True)):
        return tracks, list(labels), {"sample_specific_residual_enabled": False}

    used_by_multi: Dict[str, set[str]] = defaultdict(set)
    for track in tracks:
        if _mask_popcount(_track_mask(track, samples)) <= 1:
            continue
        for sid, peak in track.get("members", {}).items():
            peak_id = str(peak.get("peak_id", ""))
            if peak_id:
                used_by_multi[str(sid)].add(peak_id)

    filtered_tracks: List[dict] = []
    filtered_labels: List[str] = []
    removed_single = 0
    kept_single = 0
    kept_by_sample: Dict[str, int] = defaultdict(int)
    removed_by_sample: Dict[str, int] = defaultdict(int)
    for track, label in zip(tracks, labels):
        mask = _track_mask(track, samples)
        if _mask_popcount(mask) == 1:
            members = track.get("members", {}) or {}
            sid = next((sample for sample in samples if members.get(sample)), None)
            if sid is not None:
                peak = members.get(sid) or {}
                peak_id = str(peak.get("peak_id", ""))
                if peak_id and peak_id in used_by_multi.get(str(sid), set()):
                    removed_single += 1
                    removed_by_sample[str(sid)] += 1
                    continue
                kept_single += 1
                kept_by_sample[str(sid)] += 1
        filtered_tracks.append(track)
        filtered_labels.append(str(label))

    return filtered_tracks, filtered_labels, {
        "sample_specific_residual_enabled": True,
        "sample_specific_residual_removed_single_tracks": removed_single,
        "sample_specific_residual_kept_single_tracks": kept_single,
        "sample_specific_residual_kept_by_sample": ",".join(f"{sid}:{kept_by_sample.get(sid, 0)}" for sid in samples),
        "sample_specific_residual_removed_by_sample": ",".join(f"{sid}:{removed_by_sample.get(sid, 0)}" for sid in samples),
        "sample_specific_residual_input_tracks": len(tracks),
        "sample_specific_residual_output_tracks": len(filtered_tracks),
    }


def _pmtc_labels(tracks: List[dict], samples: List[str], config: AppConfig) -> Tuple[List[str], dict]:
    n = len(tracks)
    ns = len(samples)
    max_tracks_map = getattr(config.model, "pmtc_max_tracks_by_n_samples", {}) or {3: 32, 4: 28, 5: 26}
    frac_map = getattr(config.model, "pmtc_frac_limit_by_n_samples", {}) or {3: 0.55, 4: 0.47, 5: 0.41}
    min_cluster_size = int(getattr(config.model, "pmtc_min_cluster_size", 3))
    max_tracks = int(max_tracks_map.get(ns, max_tracks_map.get(str(ns), 26)))
    frac_limit = float(frac_map.get(ns, frac_map.get(str(ns), 0.41)))

    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(tracks):
        buckets[_track_mask(t, samples)].append(i)

    clusters: List[List[int]] = []
    for mask, idxs in sorted(buckets.items()):
        if _mask_popcount(mask) <= 1:
            clusters.append(list(idxs))
        elif len(idxs) <= max(min_cluster_size, max_tracks):
            clusters.append(list(idxs))
        else:
            k = max(1, math.ceil(len(idxs) / max_tracks))
            clusters.extend(_split_indices_by_feature(list(idxs), tracks, samples, k))

    max_size = max(max_tracks, int(math.ceil(frac_limit * max(n, 1))))
    guard = 0
    while guard < 20:
        guard += 1
        changed = False
        new_clusters: List[List[int]] = []
        for c in clusters:
            mask = _track_mask(tracks[c[0]], samples) if c else ""
            if _mask_popcount(mask) <= 1:
                new_clusters.append(c)
            elif len(c) > max_size and len(c) > 2 * min_cluster_size:
                k = math.ceil(len(c) / max_size)
                parts = _split_indices_by_feature(c, tracks, samples, k)
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

    labels = [""] * n
    clusters = sorted([c for c in clusters if c], key=lambda c: min(c))
    for ci, c in enumerate(clusters, 1):
        for i in c:
            labels[i] = f"C{ci:02d}"
    return labels, {"backend": "PMTC", "n_clusters": len(clusters), "bucket_count": len(buckets), "max_tracks": max_tracks, "frac_limit": frac_limit}


# ----- Guarded recall-quality backend -----

def _feature_sse(X: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not X:
        return 0.0
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[int(label)].append(i)
    total = 0.0
    d = len(X[0])
    for idxs in groups.values():
        if not idxs:
            continue
        centroid = [sum(float(X[i][j]) for i in idxs) / len(idxs) for j in range(d)]
        for i in idxs:
            dist = _euclid(list(X[i]), centroid)
            total += dist * dist
    return float(total)


def _silhouette_score(X: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[int(label)].append(i)
    if len(groups) <= 1 or len(X) <= 2:
        return 0.0
    scores: List[float] = []
    for i, label in enumerate(labels):
        same = [j for j in groups[int(label)] if j != i]
        a = sum(_euclid(list(X[i]), list(X[j])) for j in same) / len(same) if same else 0.0
        other = []
        for other_label, idxs in groups.items():
            if other_label == int(label) or not idxs:
                continue
            other.append(sum(_euclid(list(X[i]), list(X[j])) for j in idxs) / len(idxs))
        b = min(other) if other else 0.0
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 1e-12 else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _cluster_quality(tracks: Sequence[dict], labels: Sequence[str], samples: Sequence[str]) -> Tuple[float, float, float]:
    if not tracks:
        return 0.0, 0.0, 0.0
    X = _normalize_features([_track_feature(track, list(samples)) for track in tracks])
    label_ids = {label: i for i, label in enumerate(sorted(set(labels)))}
    ints = [label_ids[label] for label in labels]
    sse = _feature_sse(X, ints)
    silhouette = _silhouette_score(X, ints)

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[str(label)].append(i)

    overmerge = 0.0
    for idxs in groups.values():
        if len(idxs) <= 2:
            continue
        local_labels = [0] * len(idxs)
        local_sse = _feature_sse([X[i] for i in idxs], local_labels) / max(1, len(idxs))
        overmerge += max(0.0, local_sse - 0.42) ** 2 * len(idxs)
    return float(sse / max(1, len(tracks))), float(overmerge), float(silhouette)


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 1.0
    return float(1.0 - dot / (na * nb))


def _complete_link_distance(cluster_a: Sequence[int], cluster_b: Sequence[int], vectors: Sequence[Sequence[float]]) -> float:
    return max(_cosine_distance(vectors[i], vectors[j]) for i in cluster_a for j in cluster_b)


def _directional_hac_split_indices(
    indices: List[int],
    tracks: List[dict],
    samples: List[str],
    *,
    max_tracks: int,
    threshold: float,
) -> List[List[int]]:
    if len(indices) <= 1:
        return [indices]
    vectors = [_track_log_vec(tracks[i], samples) for i in indices]
    clusters: List[List[int]] = [[i] for i in range(len(indices))]
    while len(clusters) > 1:
        best_pair = None
        best_dist = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = _complete_link_distance(clusters[i], clusters[j], vectors)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        if best_pair is None or best_dist > threshold:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    parts = [[indices[i] for i in cluster] for cluster in clusters]
    final: List[List[int]] = []
    for part in parts:
        if len(part) > max_tracks:
            final.extend(_split_indices_by_feature(part, tracks, samples, math.ceil(len(part) / max_tracks)))
        else:
            final.append(part)
    return [part for part in final if part]


def directional_hac_labels(
    tracks: List[dict],
    samples: List[str],
    config: AppConfig,
    *,
    threshold: float | None = None,
) -> Tuple[List[str], dict]:
    ns = len(samples)
    max_tracks_map = getattr(config.model, "pmtc_max_tracks_by_n_samples", {}) or {3: 32, 4: 28, 5: 26}
    frac_map = getattr(config.model, "pmtc_frac_limit_by_n_samples", {}) or {3: 0.55, 4: 0.47, 5: 0.41}
    min_cluster_size = int(getattr(config.model, "pmtc_min_cluster_size", 3))
    max_tracks = int(max_tracks_map.get(ns, max_tracks_map.get(str(ns), 26)))
    frac_limit = float(frac_map.get(ns, frac_map.get(str(ns), 0.41)))
    if threshold is None:
        threshold = float(getattr(config.model, "guarded_quality_hac_threshold", 1.0))

    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, track in enumerate(tracks):
        buckets[_track_mask(track, samples)].append(i)

    clusters: List[List[int]] = []
    hac_splits = 0
    for mask, idxs in sorted(buckets.items()):
        if _mask_popcount(mask) <= 1:
            parts = [list(idxs)]
        else:
            parts = _directional_hac_split_indices(list(idxs), tracks, samples, max_tracks=max_tracks, threshold=float(threshold))
        tiny = [part for part in parts if len(part) < min_cluster_size]
        regular = [part for part in parts if len(part) >= min_cluster_size]
        if tiny and regular:
            vectors = [_track_log_vec(tracks[i], samples) for i in idxs]
            for part in tiny:
                target = min(
                    range(len(regular)),
                    key=lambda j: _complete_link_distance(
                        [idxs.index(i) for i in part if i in idxs],
                        [idxs.index(i) for i in regular[j] if i in idxs],
                        vectors,
                    ),
                )
                regular[target].extend(part)
            parts = regular
        hac_splits += max(0, len(parts) - 1)
        clusters.extend(parts)

    max_size = max(max_tracks, int(math.ceil(frac_limit * max(len(tracks), 1))))
    final_clusters: List[List[int]] = []
    for cluster in clusters:
        mask = _track_mask(tracks[cluster[0]], samples) if cluster else ""
        if _mask_popcount(mask) <= 1:
            final_clusters.append(cluster)
        elif len(cluster) > max_size:
            final_clusters.extend(_split_indices_by_feature(cluster, tracks, samples, math.ceil(len(cluster) / max_size)))
        else:
            final_clusters.append(cluster)
    clusters = final_clusters

    labels = [""] * len(tracks)
    clusters = sorted([cluster for cluster in clusters if cluster], key=lambda c: min(c))
    for ci, cluster in enumerate(clusters, 1):
        for i in cluster:
            labels[i] = f"H{ci:02d}"
    return labels, {
        "backend": "directional_hac",
        "n_clusters": len(clusters),
        "bucket_count": len(buckets),
        "hac_splits": hac_splits,
        "threshold": float(threshold),
    }


def merge_current_by_profile_labels(
    tracks: List[dict],
    sample_ids: List[str],
    cfg: AppConfig,
    *,
    threshold: float,
    current_labels: Sequence[str] | None = None,
) -> Tuple[List[str], dict]:
    if current_labels is None:
        current_labels, _diag = _pmtc_labels(tracks, sample_ids, cfg)
    groups: List[List[int]] = []
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(current_labels):
        by_label[str(label)].append(i)
    for _, idxs in sorted(by_label.items(), key=lambda kv: min(kv[1])):
        groups.append(list(idxs))

    vectors = [_track_log_vec(track, sample_ids) for track in tracks]

    def centroid(idxs: Sequence[int]) -> List[float]:
        d = len(vectors[0]) if vectors else 0
        return [sum(float(vectors[i][j]) for i in idxs) / len(idxs) for j in range(d)]

    merges = 0
    while len(groups) > 1:
        cents = [centroid(group) for group in groups]
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                dist = _cosine_distance(cents[i], cents[j])
                if best is None or dist < best[0]:
                    best = (dist, i, j)
        if best is None or best[0] > threshold:
            break
        _dist, i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]
        merges += 1

    out = [""] * len(tracks)
    for ci, group in enumerate(sorted(groups, key=lambda g: min(g)), 1):
        for i in group:
            out[i] = f"R{ci:02d}"
    return out, {
        "backend": "merge_current_by_profile",
        "n_clusters": len(set(out)),
        "merge_threshold": threshold,
        "merges": merges,
    }


def _cluster_stats(tracks: Sequence[dict], labels: Sequence[str], sample_ids: Sequence[str]) -> dict:
    cohesion, overmerge, silhouette = _cluster_quality(tracks, labels, sample_ids)
    return {
        "cohesion": float(cohesion),
        "overmerge": float(overmerge),
        "silhouette": float(silhouette),
        "n_clusters": len(set(labels)),
    }


def _merge_guard_reason(current_stats: dict, merge_stats: dict) -> str:
    cluster_drop = int(current_stats["n_clusters"]) - int(merge_stats["n_clusters"])
    cohesion_delta = merge_stats["cohesion"] - current_stats["cohesion"]
    overmerge_delta = merge_stats["overmerge"] - current_stats["overmerge"]
    silhouette_delta = merge_stats["silhouette"] - current_stats["silhouette"]
    if (
        0 < cluster_drop <= 1
        and overmerge_delta <= -0.25
        and silhouette_delta >= 0.02
        and cohesion_delta <= 0.15
    ):
        return "small_safe_merge"
    if (
        2 <= cluster_drop <= 8
        and overmerge_delta <= -90.0
        and silhouette_delta >= 0.05
        and cohesion_delta <= 0.15
    ):
        return "large_overmerge_relief"
    return ""


def guarded_recall_quality_labels(
    tracks: List[dict],
    sample_ids: List[str],
    cfg: AppConfig,
    current_labels: Sequence[str] | None = None,
    *,
    merge_threshold: float = 0.80,
    hac_threshold: float = 1.0,
) -> Tuple[List[str], dict]:
    if current_labels is None:
        current_labels, current_diag = _pmtc_labels(tracks, sample_ids, cfg)
    else:
        current_labels = list(current_labels)
        current_diag = {"backend": "state_labels"}

    current_stats = _cluster_stats(tracks, current_labels, sample_ids)

    merged_labels, merge_diag = merge_current_by_profile_labels(
        tracks,
        sample_ids,
        cfg,
        threshold=merge_threshold,
        current_labels=current_labels,
    )
    merge_stats = _cluster_stats(tracks, merged_labels, sample_ids)
    cluster_drop = current_stats["n_clusters"] - merge_stats["n_clusters"]
    merge_reason = _merge_guard_reason(current_stats, merge_stats)
    if merge_reason:
        diag = dict(merge_diag)
        diag.update(
            {
                "backend": "QG-PMTC",
                "selected_strategy": f"merge_current_t{str(merge_threshold).replace('.', 'p')}",
                "selection_reason": merge_reason,
                "current_backend": current_diag.get("backend", ""),
                "cluster_drop": cluster_drop,
                "current_overmerge": round(current_stats["overmerge"], 6),
                "selected_overmerge": round(merge_stats["overmerge"], 6),
                "current_silhouette": round(current_stats["silhouette"], 6),
                "selected_silhouette": round(merge_stats["silhouette"], 6),
            }
        )
        return merged_labels, diag

    hac_labels, hac_diag = directional_hac_labels(tracks, sample_ids, cfg, threshold=hac_threshold)
    hac_stats = _cluster_stats(tracks, hac_labels, sample_ids)
    cluster_rise = hac_stats["n_clusters"] - current_stats["n_clusters"]
    sil_gain = hac_stats["silhouette"] - current_stats["silhouette"]
    cohesion_drop = current_stats["cohesion"] - hac_stats["cohesion"]
    hac_overmerge_slack = max(0.20, 0.10 * max(1.0, current_stats["overmerge"]))
    max_cluster_rise = int(getattr(cfg.model, "guarded_quality_max_cluster_rise", 4))
    hac_pass = (
        cluster_rise <= max_cluster_rise
        and hac_stats["overmerge"] <= current_stats["overmerge"] + hac_overmerge_slack
        and (
            current_stats["overmerge"] - hac_stats["overmerge"] >= max(1.0, 0.50 * current_stats["overmerge"])
            or (sil_gain >= 0.10 and cohesion_drop >= -0.05)
        )
    )
    if hac_pass:
        diag = dict(hac_diag)
        diag.update(
            {
                "backend": "QG-PMTC",
                "selected_strategy": f"hac_t{str(hac_threshold).replace('.', 'p')}",
                "selection_reason": "quality_split",
                "current_backend": current_diag.get("backend", ""),
                "cluster_rise": cluster_rise,
                "max_cluster_rise": max_cluster_rise,
                "current_overmerge": round(current_stats["overmerge"], 6),
                "selected_overmerge": round(hac_stats["overmerge"], 6),
                "current_silhouette": round(current_stats["silhouette"], 6),
                "selected_silhouette": round(hac_stats["silhouette"], 6),
            }
        )
        return hac_labels, diag

    return list(current_labels), {
        "backend": "QG-PMTC",
        "selected_strategy": "current_state",
        "selection_reason": "guard_rejected",
        "n_clusters": current_stats["n_clusters"],
        "current_backend": current_diag.get("backend", ""),
        "current_overmerge": round(current_stats["overmerge"], 6),
        "current_silhouette": round(current_stats["silhouette"], 6),
        "merge_clusters": merge_stats["n_clusters"],
        "hac_clusters": hac_stats["n_clusters"],
        "max_cluster_rise": max_cluster_rise,
    }


def run_v5_pmtc(samples: List[Sample], config: AppConfig) -> JointState:
    ordered_ids, peaks_original, shifts, selected, diag = _run_v4_frontend(samples, config)
    labels, cdiag = _pmtc_labels(selected, ordered_ids, config)
    selected, labels, rdiag = _apply_mask_residual_filter(selected, labels, ordered_ids, config)
    selected, labels, sdiag = _apply_sample_specific_residual_peak_filter(selected, labels, ordered_ids, config)
    cdiag = {**cdiag, **rdiag, **sdiag}
    state = _make_state(samples, ordered_ids, peaks_original, shifts, selected, labels, objective_value=float(cdiag.get("n_clusters", 0)), iterations=1)
    state.meta = {
        **(diag if isinstance(diag, dict) else {"frontend_diag": diag}),
        **{f"pmtc_{k}": v for k, v in cdiag.items()},
        "backend": "PMTC",
    }  # type: ignore[attr-defined]
    return state


def run_v5_frontend_mask_only(samples: List[Sample], config: AppConfig) -> JointState:
    ordered_ids, peaks_original, shifts, selected, diag = _run_v4_frontend(samples, config)
    labels = [f"M{_track_mask(track, ordered_ids)}" for track in selected]
    selected, labels, rdiag = _apply_mask_residual_filter(selected, labels, ordered_ids, config)
    selected, labels, sdiag = _apply_sample_specific_residual_peak_filter(selected, labels, ordered_ids, config)
    state = _make_state(samples, ordered_ids, peaks_original, shifts, selected, labels, objective_value=float(len(set(labels))), iterations=1)
    state.meta = {
        **(diag if isinstance(diag, dict) else {"frontend_diag": diag}),
        **{f"mask_only_{k}": v for k, v in {**rdiag, **sdiag}.items()},
        "backend": "SP-Mask",
    }  # type: ignore[attr-defined]
    return state


def _covered_peak_keys(tracks: Sequence[dict]) -> set[Tuple[str, str]]:
    covered: set[Tuple[str, str]] = set()
    for track in tracks:
        for sid, peak in track.get("members", {}).items():
            peak_id = str(peak.get("peak_id", ""))
            if peak_id:
                covered.add((str(sid), peak_id))
    return covered


def _append_uncovered_singletons(
    tracks: List[dict],
    ordered_ids: List[str],
    peaks_original: Dict[str, List[Peak]],
) -> Tuple[List[dict], List[int]]:
    out = list(tracks)
    uncovered_indices: List[int] = []
    covered = _covered_peak_keys(out)
    fill_idx = 0
    for sid in ordered_ids:
        for peak in peaks_original.get(sid, []):
            key = (sid, str(peak.peak_id))
            if key in covered:
                continue
            fill_idx += 1
            member = {
                "sample": sid,
                "peak_id": str(peak.peak_id),
                "ppm": float(peak.ppm_raw),
                "ppm_corr": float(peak.corrected_ppm()),
                "area": float(peak.area if peak.area is not None else peak.intensity),
                "intensity": float(peak.intensity if peak.intensity is not None else (peak.area or 1.0)),
            }
            uncovered_indices.append(len(out))
            out.append(
                {
                    "track_id": f"FILL{fill_idx:04d}",
                    "member_ids": (str(peak.peak_id),),
                    "members": {sid: member},
                    "score": 0.0,
                    "kind": "uncovered_singleton",
                    "alignment_error": 0.0,
                }
            )
            covered.add(key)
    return out, uncovered_indices


def run_guarded_recall_quality(samples: List[Sample], config: AppConfig) -> JointState:
    ordered_ids, peaks_original, shifts, selected, diag = _run_v4_frontend(samples, config)
    current_labels, _current_diag = _pmtc_labels(selected, ordered_ids, config)
    labels, cdiag = guarded_recall_quality_labels(
        selected,
        ordered_ids,
        config,
        current_labels=current_labels,
        merge_threshold=float(getattr(config.model, "guarded_quality_merge_threshold", 0.80)),
        hac_threshold=float(getattr(config.model, "guarded_quality_hac_threshold", 1.00)),
    )
    selected, labels, rdiag = _apply_mask_residual_filter(selected, labels, ordered_ids, config)
    selected, labels, sdiag = _apply_sample_specific_residual_peak_filter(selected, labels, ordered_ids, config)
    cdiag = {**cdiag, **rdiag, **sdiag}
    state = _make_state(samples, ordered_ids, peaks_original, shifts, selected, labels, objective_value=float(cdiag.get("n_clusters", 0)), iterations=1)
    state.meta = {
        **(diag if isinstance(diag, dict) else {"frontend_diag": diag}),
        **{f"guarded_{k}": v for k, v in cdiag.items()},
        "backend": "QG-PMTC",
    }  # type: ignore[attr-defined]
    return state


def run_switchable_model(samples: List[Sample], config: AppConfig) -> JointState:
    """Run the selected cross-spectrum clustering backend.

    Supported names:
      - v5_pmtc: PMTC, presence-mask binning and intensity-profile bucket splitting
      - v5_enum: QG-PMTC, SPTC front end plus PMTC labels and guarded merge/split refinement
      - v5_frontend_mask: SP-Mask, SPTC front end grouped only by presence mask
    """
    name = str(getattr(config.model, "name", "v5_pmtc") or "v5_pmtc").lower().replace("-", "_")
    if name == "v5_pmtc":
        return run_v5_pmtc(samples, config)
    if name == "v5_frontend_mask":
        return run_v5_frontend_mask_only(samples, config)
    if name == "v5_enum":
        return run_guarded_recall_quality(samples, config)
    raise ValueError(f"Unsupported clustering model: {name}. Use v5_enum, v5_pmtc, or v5_frontend_mask.")
