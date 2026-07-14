from __future__ import annotations

import math
from collections import defaultdict, deque
from itertools import product
from statistics import median
from typing import Dict, List, Sequence, Tuple

from nmr_trendtrack.config import AppConfig
from nmr_trendtrack.contracts import JointState, Peak, Sample
from nmr_trendtrack.models.three_model_pipeline import (
    _cluster_quality,
    _make_state,
    _pmtc_labels,
    _run_v4_frontend,
    _rows_from_peaklists,
    _v4_modules,
    guarded_recall_quality_labels,
)


def _track_key(track: dict) -> Tuple[str, ...]:
    return tuple(sorted(str(x) for x in track.get("member_ids", [])))


def _track_mask(track: dict, ordered_sample_ids: Sequence[str]) -> str:
    members = track.get("members", {})
    return "".join("1" if members.get(sid) else "0" for sid in ordered_sample_ids)


def _track_center(track: dict) -> float:
    vals = []
    for peak in track.get("members", {}).values():
        vals.append(float(peak.get("ppm_corr", peak.get("ppm", 0.0))))
    return float(median(vals)) if vals else 0.0


def _alignment_error(track: dict) -> float:
    vals = [float(p.get("ppm_corr", p.get("ppm", 0.0))) for p in track.get("members", {}).values()]
    raw_vals = [float(p.get("ppm", p.get("ppm_corr", 0.0))) for p in track.get("members", {}).values()]
    corr_span = max(vals) - min(vals) if len(vals) > 1 else 0.0
    raw_span = max(raw_vals) - min(raw_vals) if len(raw_vals) > 1 else 0.0
    return float(corr_span + 0.25 * raw_span)


def _score_members(base, members: Dict[str, dict], ordered_sample_ids: Sequence[str], residual_gate: float) -> float:
    try:
        return float(base.score_track(members, list(ordered_sample_ids), residual_gate=residual_gate))
    except Exception:
        return -_alignment_error({"members": members})


def _patch_v4_globals(base, config: AppConfig) -> None:
    region_windows = list(getattr(config.align, "ppm_window_by_region", []) or [])
    if region_windows:
        base.REGIONS = [(float(lo), float(hi), float(win)) for lo, hi, win in region_windows]
        base.REGION_NAMES = [f"{int(lo)}-{int(hi)}" for lo, hi, _ in base.REGIONS]
    base.RAW_SPAN_LIMIT = float(getattr(config.align, "max_track_span_ppm", 0.50) or 0.50)


def _ppm_window_at(ppm: float, config: AppConfig) -> float:
    for lo, hi, win in list(getattr(config.align, "ppm_window_by_region", []) or []):
        if float(lo) <= ppm < float(hi):
            return float(win)
    default = getattr(config.align, "ppm_window_default", None)
    if default is not None:
        return float(default)
    return float(getattr(config.align, "max_track_span_ppm", 0.50) or 0.50)


def _make_track(member_peaks: Sequence[dict], score: float, kind: str) -> dict:
    members = {str(p["sample"]): p for p in member_peaks}
    key = tuple(sorted(str(p["peak_id"]) for p in member_peaks))
    return {
        "member_ids": key,
        "members": members,
        "score": float(score),
        "kind": kind,
        "alignment_error": _alignment_error({"members": members}),
    }


def _enumerate_all_mask_candidates(
    corrected_rows: Sequence[dict],
    ordered_sample_ids: Sequence[str],
    base,
    config: AppConfig,
    *,
    include_drop1: bool = False,
) -> Tuple[List[dict], dict]:
    by_sample: Dict[str, List[dict]] = defaultdict(list)
    for row in corrected_rows:
        by_sample[str(row["sample"])].append(row)

    corr_window = float(getattr(config.align, "max_track_span_ppm", 0.50) or 0.50)
    raw_window = float(getattr(config.align, "max_track_span_ppm", 0.50) or 0.50)
    residual_gate = float(getattr(config.model, "residual_gate", 0.15) or 0.15)
    candidate_by_key: Dict[Tuple[str, ...], dict] = {}
    seed_product_total = 0
    seed_product_max = 0

    for seed in corrected_rows:
        choices: List[List[dict | None]] = []
        for sid in ordered_sample_ids:
            if sid == seed["sample"]:
                choices.append([seed])
                continue
            opts: List[dict | None] = [None]
            seed_corr = float(seed.get("ppm_corr", seed.get("ppm", 0.0)))
            seed_raw = float(seed.get("ppm", seed_corr))
            for peak in by_sample[sid]:
                peak_corr = float(peak.get("ppm_corr", peak.get("ppm", 0.0)))
                peak_raw = float(peak.get("ppm", peak_corr))
                if abs(peak_corr - seed_corr) <= corr_window and abs(peak_raw - seed_raw) <= raw_window:
                    opts.append(peak)
            choices.append(opts)

        seed_product = 1
        for opts in choices:
            seed_product *= len(opts)
        seed_product_total += seed_product
        seed_product_max = max(seed_product_max, seed_product)

        for combo in product(*choices):
            present = [p for p in combo if p is not None]
            if not present or seed not in present:
                continue
            if len({p["sample"] for p in present}) != len(present):
                continue
            corr_vals = [float(p.get("ppm_corr", p.get("ppm", 0.0))) for p in present]
            raw_vals = [float(p.get("ppm", p.get("ppm_corr", 0.0))) for p in present]
            if len(present) > 1 and (max(corr_vals) - min(corr_vals) > corr_window or max(raw_vals) - min(raw_vals) > raw_window):
                continue

            members = {str(p["sample"]): p for p in present}
            score = _score_members(base, members, ordered_sample_ids, residual_gate)
            track = _make_track(present, score, "enumerated_all_masks")
            key = _track_key(track)
            prev = candidate_by_key.get(key)
            if prev is None or (float(track["score"]), -float(track["alignment_error"])) > (
                float(prev.get("score", 0.0)),
                -float(prev.get("alignment_error", 0.0)),
            ):
                candidate_by_key[key] = track

    drop1_added = 0
    drop1_replaced = 0
    drop1_seen = 0
    if include_drop1:
        for parent in list(candidate_by_key.values()):
            members = list(parent.get("members", {}).values())
            if len(members) <= 1:
                continue
            for drop_idx in range(len(members)):
                child_peaks = [p for i, p in enumerate(members) if i != drop_idx]
                if not child_peaks:
                    continue
                child_members = {str(p["sample"]): p for p in child_peaks}
                score = _score_members(base, child_members, ordered_sample_ids, residual_gate)
                child = _make_track(child_peaks, score, "enumerated_drop1")
                key = _track_key(child)
                prev = candidate_by_key.get(key)
                drop1_seen += 1
                if prev is None:
                    candidate_by_key[key] = child
                    drop1_added += 1
                elif (float(child["score"]), -float(child["alignment_error"])) > (
                    float(prev.get("score", 0.0)),
                    -float(prev.get("alignment_error", 0.0)),
                ):
                    candidate_by_key[key] = child
                    drop1_replaced += 1

    return list(candidate_by_key.values()), {
        "seed_product_total": seed_product_total,
        "seed_product_max": seed_product_max,
        "candidate_tracks": len(candidate_by_key),
        "drop1_candidates_seen": drop1_seen,
        "drop1_candidates_added": drop1_added,
        "drop1_candidates_replaced": drop1_replaced,
    }


def _components_from_candidates(all_peak_ids: Sequence[str], candidates: Sequence[dict]) -> List[set[str]]:
    graph: Dict[str, set[str]] = {str(pid): set() for pid in all_peak_ids}
    for track in candidates:
        ids = list(_track_key(track))
        if len(ids) < 2:
            continue
        first = ids[0]
        for pid in ids[1:]:
            graph.setdefault(first, set()).add(pid)
            graph.setdefault(pid, set()).add(first)

    components: List[set[str]] = []
    seen: set[str] = set()
    for peak_id in all_peak_ids:
        pid = str(peak_id)
        if pid in seen:
            continue
        queue = deque([pid])
        seen.add(pid)
        comp: set[str] = set()
        while queue:
            cur = queue.popleft()
            comp.add(cur)
            for nxt in graph.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    components.sort(key=lambda c: (min(c), len(c)))
    return components


def _singleton_tracks(rows: Sequence[dict], base, ordered_sample_ids: Sequence[str], residual_gate: float) -> Dict[str, dict]:
    out = {}
    for row in rows:
        members = {str(row["sample"]): row}
        score = _score_members(base, members, ordered_sample_ids, residual_gate)
        out[str(row["peak_id"])] = _make_track([row], score, "singleton_fallback")
    return out


def _exact_cover_component(
    component_peak_ids: set[str],
    component_candidates: Sequence[dict],
    singletons: Dict[str, dict],
    *,
    max_options: int,
    node_limit: int,
) -> Tuple[List[dict], dict]:
    candidates_by_peak: Dict[str, List[dict]] = defaultdict(list)
    usable: Dict[Tuple[str, ...], dict] = {}
    for cand in component_candidates:
        key = _track_key(cand)
        if not key or not set(key) <= component_peak_ids:
            continue
        usable[key] = cand
    for pid in component_peak_ids:
        usable.setdefault((pid,), singletons[pid])

    ordered_candidates = sorted(
        usable.values(),
        key=lambda t: (-len(_track_key(t)), float(t.get("alignment_error", 0.0)), -float(t.get("score", 0.0))),
    )
    for cand in ordered_candidates:
        for pid in _track_key(cand):
            candidates_by_peak[pid].append(cand)

    best_tracks: List[dict] | None = None
    best_obj: Tuple[int, float, float] | None = None
    nodes = 0
    timed_out = False

    def objective(chosen: Sequence[dict]) -> Tuple[int, float, float]:
        return (
            len(chosen),
            sum(float(t.get("alignment_error", 0.0)) for t in chosen),
            -sum(float(t.get("score", 0.0)) for t in chosen),
        )

    def search(uncovered: set[str], chosen: List[dict], used: set[str]) -> None:
        nonlocal best_tracks, best_obj, nodes, timed_out
        if nodes >= node_limit:
            timed_out = True
            return
        nodes += 1
        if best_obj is not None and len(chosen) > best_obj[0]:
            return
        if not uncovered:
            obj = objective(chosen)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_tracks = list(chosen)
            return

        pivot = min(uncovered, key=lambda pid: len([c for c in candidates_by_peak[pid] if not (set(_track_key(c)) & used)]))
        options = [c for c in candidates_by_peak[pivot] if not (set(_track_key(c)) & used)]
        options = options[:max_options]
        for cand in options:
            ids = set(_track_key(cand))
            if ids & used:
                continue
            search(uncovered - ids, chosen + [cand], used | ids)

    search(set(component_peak_ids), [], set())
    if best_tracks is None:
        best_tracks = [singletons[pid] for pid in sorted(component_peak_ids)]

    return best_tracks, {
        "n_peaks": len(component_peak_ids),
        "candidate_tracks": len(usable),
        "nodes": nodes,
        "timed_out": timed_out,
        "selected_tracks": len(best_tracks),
        "alignment_error": sum(float(t.get("alignment_error", 0.0)) for t in best_tracks),
    }


def _cover_objective(chosen: Sequence[dict]) -> Tuple[int, float, float]:
    return (
        len(chosen),
        sum(float(t.get("alignment_error", _alignment_error(t))) for t in chosen),
        -sum(float(t.get("score", 0.0) or 0.0) for t in chosen),
    )


def _cover_signature(chosen: Sequence[dict]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(sorted(_track_key(track) for track in chosen))


def _copy_track(track: dict, kind: str) -> dict:
    out = dict(track)
    out["members"] = dict(track.get("members", {}))
    out["member_ids"] = _track_key(out)
    out.setdefault("kind", kind)
    out.setdefault("alignment_error", _alignment_error(out))
    return out


def _add_cover_option(
    options: List[List[dict]],
    seen: set[Tuple[Tuple[str, ...], ...]],
    forced_signatures: set[Tuple[Tuple[str, ...], ...]],
    component_peak_ids: set[str],
    chosen: Sequence[dict],
    *,
    forced: bool = False,
) -> bool:
    used: set[str] = set()
    normalized: List[dict] = []
    for track in chosen:
        key = set(_track_key(track))
        if not key or not key <= component_peak_ids or key & used:
            return False
        used |= key
        normalized.append(track)
    if used != component_peak_ids:
        return False
    normalized = sorted(normalized, key=lambda t: (_track_key(t), _track_center(t)))
    sig = _cover_signature(normalized)
    if forced:
        forced_signatures.add(sig)
    if sig in seen:
        return False
    seen.add(sig)
    options.append(normalized)
    return True


def _enumerate_cover_options(
    component_peak_ids: set[str],
    component_candidates: Sequence[dict],
    singletons: Dict[str, dict],
    *,
    max_options: int,
    node_limit: int,
    forced_options: Sequence[Sequence[dict]] | None = None,
) -> Tuple[List[List[dict]], dict]:
    component_peak_ids = {str(pid) for pid in component_peak_ids}
    candidates_by_peak: Dict[str, List[dict]] = defaultdict(list)
    usable: Dict[Tuple[str, ...], dict] = {}
    for cand in component_candidates:
        key = _track_key(cand)
        if not key or not set(key) <= component_peak_ids:
            continue
        usable[key] = cand
    for pid in component_peak_ids:
        usable.setdefault((pid,), singletons[pid])

    ordered_candidates = sorted(
        usable.values(),
        key=lambda t: (-len(_track_key(t)), float(t.get("alignment_error", 0.0)), -float(t.get("score", 0.0)), _track_key(t)),
    )
    key_sets: Dict[Tuple[str, ...], set[str]] = {}
    for cand in ordered_candidates:
        key = _track_key(cand)
        key_sets[key] = set(key)
        for pid in key:
            candidates_by_peak[pid].append(cand)

    max_options = max(1, int(max_options))
    node_limit = max(1, int(node_limit))
    options: List[List[dict]] = []
    seen: set[Tuple[Tuple[str, ...], ...]] = set()
    forced_signatures: set[Tuple[Tuple[str, ...], ...]] = set()
    forced_count = 0
    for forced in forced_options or []:
        if _add_cover_option(options, seen, forced_signatures, component_peak_ids, forced, forced=True):
            forced_count += 1

    nodes = 0
    timed_out = False

    def search(uncovered: set[str], chosen: List[dict], used: set[str]) -> None:
        nonlocal nodes, timed_out
        if len(options) >= max_options + forced_count:
            return
        if nodes >= node_limit:
            timed_out = True
            return
        nodes += 1
        if not uncovered:
            _add_cover_option(options, seen, forced_signatures, component_peak_ids, chosen)
            return

        def available_count(pid: str) -> int:
            return sum(1 for cand in candidates_by_peak[pid] if not (key_sets[_track_key(cand)] & used))

        pivot = min(uncovered, key=lambda pid: (available_count(pid), pid))
        candidates = [cand for cand in candidates_by_peak[pivot] if not (key_sets[_track_key(cand)] & used)]
        for cand in candidates:
            ids = key_sets[_track_key(cand)]
            if ids & used:
                continue
            search(uncovered - ids, chosen + [cand], used | ids)
            if timed_out or len(options) >= max_options + forced_count:
                break

    search(set(component_peak_ids), [], set())
    if not options:
        fallback = [singletons[pid] for pid in sorted(component_peak_ids)]
        _add_cover_option(options, seen, forced_signatures, component_peak_ids, fallback)

    forced_kept = [opt for opt in options if _cover_signature(opt) in forced_signatures]
    normal = [opt for opt in options if _cover_signature(opt) not in forced_signatures]
    normal.sort(key=_cover_objective)
    keep_count = max(0, max_options - len(forced_kept))
    kept = forced_kept + normal[:keep_count]
    kept.sort(key=_cover_objective)

    return kept, {
        "n_peaks": len(component_peak_ids),
        "candidate_tracks": len(usable),
        "nodes": nodes,
        "timed_out": timed_out,
        "cover_options": len(kept),
        "forced_v5_options": len(forced_kept),
        "best_cover_tracks": len(kept[0]) if kept else 0,
        "best_alignment_error": sum(float(t.get("alignment_error", 0.0)) for t in kept[0]) if kept else 0.0,
    }


def _forced_v5_cover_option(component_peak_ids: set[str], v5_tracks: Sequence[dict], singletons: Dict[str, dict]) -> List[dict] | None:
    component_peak_ids = {str(pid) for pid in component_peak_ids}
    chosen: List[dict] = []
    used: set[str] = set()
    for track in v5_tracks:
        key = set(_track_key(track))
        if not key or not key <= component_peak_ids:
            continue
        if key & used:
            return None
        chosen.append(track)
        used |= key
    if not chosen:
        return None
    for pid in sorted(component_peak_ids - used):
        chosen.append(singletons[pid])
    return chosen


def _mask_cluster_score(tracks: Sequence[dict], ordered_sample_ids: Sequence[str], config: AppConfig) -> Tuple[int, int, int, int, float, float]:
    if not tracks:
        return (0, 0, 0, 0, 0.0, 0.0)
    labels, _diag = _pmtc_labels(list(tracks), list(ordered_sample_ids), config)
    per_mask: Dict[str, set[str]] = defaultdict(set)
    for track, label in zip(tracks, labels):
        per_mask[_track_mask(track, ordered_sample_ids)].add(str(label))
    split_penalty = sum(max(0, len(mask_labels) - 1) for mask_labels in per_mask.values())
    total_clusters = len(set(labels))
    alignment = sum(float(t.get("alignment_error", _alignment_error(t))) for t in tracks)
    score_sum = sum(float(t.get("score", 0.0) or 0.0) for t in tracks)
    return (total_clusters, split_penalty, -len(per_mask), len(tracks), alignment, -score_sum)


def _mask_labels(tracks: Sequence[dict], ordered_sample_ids: Sequence[str]) -> List[str]:
    return [f"M{_track_mask(track, ordered_sample_ids)}" for track in tracks]


def _mask_only_guarded_score(
    tracks: Sequence[dict],
    ordered_sample_ids: Sequence[str],
    *,
    baseline_n_tracks: int | None,
) -> Tuple[int, float, int, float, float]:
    labels = _mask_labels(tracks, ordered_sample_ids)
    if tracks:
        cohesion, overmerge, silhouette = _cluster_quality(tracks, labels, ordered_sample_ids)
    else:
        cohesion, overmerge, silhouette = 0.0, 0.0, 0.0
    n_masks = len(set(labels))
    quality = -silhouette + 0.60 * cohesion + 0.20 * overmerge + 0.01 * n_masks
    if baseline_n_tracks is None:
        track_deficit = 0
        track_delta = 0
    else:
        track_deficit = max(0, int(baseline_n_tracks) - len(tracks))
        track_delta = abs(len(tracks) - int(baseline_n_tracks))
    alignment = sum(float(track.get("alignment_error", _alignment_error(track))) for track in tracks)
    score_sum = sum(float(track.get("score", 0.0) or 0.0) for track in tracks)
    return (track_deficit, float(quality), track_delta, float(alignment), float(-score_sum))


def _select_mask_guarded_option(
    options: Sequence[Sequence[dict]],
    ordered_sample_ids: Sequence[str],
    *,
    forced_option: Sequence[dict] | None = None,
) -> Tuple[List[dict], dict]:
    baseline_n = len(forced_option) if forced_option else None
    scored = [
        (_mask_only_guarded_score(option, ordered_sample_ids, baseline_n_tracks=baseline_n), list(option))
        for option in options
        if option
    ]
    if forced_option:
        forced = list(forced_option)
        scored.append((_mask_only_guarded_score(forced, ordered_sample_ids, baseline_n_tracks=baseline_n), forced))
    if not scored:
        return [], {"mask_guarded_selected": "empty"}
    score, selected = min(scored, key=lambda item: item[0])
    forced_sig = _cover_signature(forced_option or [])
    selected_from_forced = bool(forced_option) and _cover_signature(selected) == forced_sig
    return selected, {
        "mask_guarded_option_count": len(scored),
        "mask_guarded_score": score,
        "mask_guarded_baseline_tracks": baseline_n if baseline_n is not None else "",
        "mask_guarded_selected_tracks": len(selected),
        "mask_guarded_selected_from_v5": selected_from_forced,
    }


def _select_options_by_mask_min_clusters(
    component_options: Sequence[Sequence[Sequence[dict]]],
    ordered_sample_ids: Sequence[str],
    config: AppConfig,
    protected_solutions: Sequence[Sequence[dict]] | None = None,
) -> Tuple[List[dict], dict]:
    beam_width = max(1, int(getattr(config.model, "enum_cluster_beam_width", 400) or 400))
    beam: List[Tuple[List[dict], Tuple[int, int, int, int, float, float]]] = [([], _mask_cluster_score([], ordered_sample_ids, config))]
    max_beam_seen = 1
    for options in component_options:
        if not options:
            continue
        expanded: List[Tuple[List[dict], Tuple[int, int, int, int, float, float]]] = []
        for prefix, _prefix_score in beam:
            for option in options:
                combined = prefix + list(option)
                expanded.append((combined, _mask_cluster_score(combined, ordered_sample_ids, config)))
        expanded.sort(key=lambda item: item[1])
        deduped: List[Tuple[List[dict], Tuple[int, int, int, int, float, float]]] = []
        seen: set[Tuple[Tuple[str, ...], ...]] = set()
        for tracks, score in expanded:
            sig = _cover_signature(tracks)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append((tracks, score))
            if len(deduped) >= beam_width:
                break
        beam = deduped or beam
        max_beam_seen = max(max_beam_seen, len(beam))

    protected_scored: List[Tuple[List[dict], Tuple[int, int, int, int, float, float]]] = []
    for solution in protected_solutions or []:
        protected_scored.append((list(solution), _mask_cluster_score(solution, ordered_sample_ids, config)))

    selected, score = min(beam + protected_scored, key=lambda item: item[1])
    protected_sigs = {_cover_signature(tracks) for tracks, _score in protected_scored}
    protected_scores = [score for _tracks, score in protected_scored]
    return selected, {
        "enum_backend": "mask_min_cluster_beam",
        "enum_cluster_beam_width": beam_width,
        "enum_cluster_beam_max_seen": max_beam_seen,
        "enum_cluster_score": score,
        "protected_solution_count": len(protected_scored),
        "protected_solution_best_score": min(protected_scores) if protected_scores else None,
        "selected_from_protected_solution": _cover_signature(selected) in protected_sigs,
    }


def _guarded_cover_score(
    tracks: Sequence[dict],
    ordered_sample_ids: Sequence[str],
    config: AppConfig,
) -> Tuple[Tuple[float, float, float, float, int, int, float, float], dict]:
    if not tracks:
        return ((0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0), {"n_clusters": 0})
    current_labels, current_diag = _pmtc_labels(list(tracks), list(ordered_sample_ids), config)
    labels, guarded_diag = guarded_recall_quality_labels(
        list(tracks),
        list(ordered_sample_ids),
        config,
        current_labels=current_labels,
        merge_threshold=float(getattr(config.model, "guarded_quality_merge_threshold", 0.80)),
        hac_threshold=float(getattr(config.model, "guarded_quality_hac_threshold", 1.00)),
    )
    cohesion, overmerge, silhouette = _cluster_quality(tracks, labels, ordered_sample_ids)
    n_clusters = len(set(labels))
    quality = -float(silhouette) + 0.60 * float(cohesion) + 0.20 * float(overmerge) + 0.01 * float(n_clusters)
    alignment = sum(float(track.get("alignment_error", _alignment_error(track))) for track in tracks)
    score_sum = sum(float(track.get("score", 0.0) or 0.0) for track in tracks)
    diag = {
        "quality": float(quality),
        "cohesion": float(cohesion),
        "overmerge": float(overmerge),
        "silhouette": float(silhouette),
        "n_clusters": int(n_clusters),
        "current_backend": current_diag.get("backend", ""),
        "guarded_strategy": guarded_diag.get("selected_strategy", ""),
        "guarded_reason": guarded_diag.get("selection_reason", ""),
    }
    return (
        (
            float(quality),
            float(overmerge),
            float(-silhouette),
            float(cohesion),
            int(n_clusters),
            len(tracks),
            float(alignment),
            float(-score_sum),
        ),
        diag,
    )


def _select_options_by_guarded_quality(
    component_options: Sequence[Sequence[Sequence[dict]]],
    ordered_sample_ids: Sequence[str],
    config: AppConfig,
    protected_solutions: Sequence[Sequence[dict]] | None = None,
) -> Tuple[List[dict], dict]:
    beam_width = max(1, int(getattr(config.model, "enum_cluster_beam_width", 400) or 400))
    empty_score, _empty_diag = _guarded_cover_score([], ordered_sample_ids, config)
    beam: List[Tuple[List[dict], Tuple[float, float, float, float, int, int, float, float], dict]] = [([], empty_score, {})]
    max_beam_seen = 1
    scored_inputs = 0
    for options in component_options:
        if not options:
            continue
        expanded: List[Tuple[List[dict], Tuple[float, float, float, float, int, int, float, float], dict]] = []
        for prefix, _prefix_score, _prefix_diag in beam:
            for option in options:
                combined = prefix + list(option)
                score, score_diag = _guarded_cover_score(combined, ordered_sample_ids, config)
                scored_inputs += 1
                expanded.append((combined, score, score_diag))
        expanded.sort(key=lambda item: item[1])
        deduped: List[Tuple[List[dict], Tuple[float, float, float, float, int, int, float, float], dict]] = []
        seen: set[Tuple[Tuple[str, ...], ...]] = set()
        for tracks, score, score_diag in expanded:
            sig = _cover_signature(tracks)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append((tracks, score, score_diag))
            if len(deduped) >= beam_width:
                break
        beam = deduped or beam
        max_beam_seen = max(max_beam_seen, len(beam))

    protected_scored: List[Tuple[List[dict], Tuple[float, float, float, float, int, int, float, float], dict]] = []
    for solution in protected_solutions or []:
        score, score_diag = _guarded_cover_score(solution, ordered_sample_ids, config)
        protected_scored.append((list(solution), score, score_diag))

    selected, score, score_diag = min(beam + protected_scored, key=lambda item: item[1])
    protected_sigs = {_cover_signature(tracks) for tracks, _score, _diag in protected_scored}
    protected_scores = [score for _tracks, score, _diag in protected_scored]
    return selected, {
        "enum_backend": "guarded_quality_cover_beam",
        "enum_cluster_beam_width": beam_width,
        "enum_cluster_beam_max_seen": max_beam_seen,
        "enum_scored_cover_inputs": scored_inputs + len(protected_scored),
        "enum_cluster_score": score,
        "enum_cluster_quality": score_diag.get("quality"),
        "enum_cluster_cohesion": score_diag.get("cohesion"),
        "enum_cluster_overmerge": score_diag.get("overmerge"),
        "enum_cluster_silhouette": score_diag.get("silhouette"),
        "enum_guarded_strategy": score_diag.get("guarded_strategy", ""),
        "enum_guarded_reason": score_diag.get("guarded_reason", ""),
        "protected_solution_count": len(protected_scored),
        "protected_solution_best_score": min(protected_scores) if protected_scores else None,
        "selected_from_protected_solution": _cover_signature(selected) in protected_sigs,
    }


def run_enumerated_frontend(samples: List[Sample], config: AppConfig) -> Tuple[List[str], Dict[str, List[Peak]], Dict[str, float], List[dict], dict]:
    base, _gsp = _v4_modules()
    _patch_v4_globals(base, config)
    ordered_ids, rows, peaks_original = _rows_from_peaklists(samples)
    corrected, shift_model, shift_diag = base.apply_shift(rows, ordered_ids)

    include_drop1 = bool(getattr(config.model, "enum_include_drop1", True))
    candidates, enum_diag = _enumerate_all_mask_candidates(corrected, ordered_ids, base, config, include_drop1=include_drop1)
    singletons = _singleton_tracks(corrected, base, ordered_ids, float(getattr(config.model, "residual_gate", 0.15) or 0.15))
    all_peak_ids = [str(row["peak_id"]) for row in corrected]
    components = _components_from_candidates(all_peak_ids, candidates)

    max_options = int(getattr(config.model, "enum_max_options_per_component", 160) or 160)
    node_limit = int(getattr(config.model, "enum_node_limit", 350000) or 350000)
    selected: List[dict] = []
    component_diags: List[dict] = []
    for ci, component in enumerate(components, 1):
        comp_candidates = [c for c in candidates if set(_track_key(c)) <= component]
        chosen, cdiag = _exact_cover_component(
            component,
            comp_candidates,
            singletons,
            max_options=max_options,
            node_limit=node_limit,
        )
        cdiag["component_id"] = ci
        component_diags.append(cdiag)
        selected.extend(chosen)

    selected = sorted(selected, key=lambda t: (_track_mask(t, ordered_ids), _track_center(t), _track_key(t)))
    for i, track in enumerate(selected, 1):
        track["track_id"] = f"EV5{i:04d}"

    shifts = {sid: 0.0 for sid in ordered_ids}
    for sid in ordered_ids:
        vals = []
        for reg in getattr(base, "REGION_NAMES", []):
            try:
                vals.append(float(shift_model.get(sid, {}).get(reg, 0.0)))
            except Exception:
                pass
        shifts[sid] = float(sum(vals) / len(vals)) if vals else 0.0

    selected_ids = [pid for track in selected for pid in _track_key(track)]
    dup_count = len(selected_ids) - len(set(selected_ids))
    missing_count = len(set(all_peak_ids) - set(selected_ids))
    shift_diag_map = shift_diag if isinstance(shift_diag, dict) else {"shift_diag": shift_diag}
    diag = {
        **shift_diag_map,
        **enum_diag,
        "backend": "enumerated_v5_exact_cover",
        "include_drop1": include_drop1,
        "n_components": len(components),
        "n_selected_tracks": len(selected),
        "missing_peak_refs": missing_count,
        "duplicate_peak_refs": dup_count,
        "component_timeouts": sum(1 for d in component_diags if d.get("timed_out")),
        "component_option_cap": max_options,
    }
    return ordered_ids, peaks_original, shifts, selected, diag


def run_enumerated_cluster_frontend(samples: List[Sample], config: AppConfig) -> Tuple[List[str], Dict[str, List[Peak]], Dict[str, float], List[dict], dict]:
    base, _gsp = _v4_modules()
    _patch_v4_globals(base, config)
    ordered_ids, rows, peaks_original = _rows_from_peaklists(samples)
    corrected, shift_model, shift_diag = base.apply_shift(rows, ordered_ids)

    include_drop1 = bool(getattr(config.model, "enum_include_drop1", True))
    candidates, enum_diag = _enumerate_all_mask_candidates(corrected, ordered_ids, base, config, include_drop1=include_drop1)
    singletons = _singleton_tracks(corrected, base, ordered_ids, float(getattr(config.model, "residual_gate", 0.15) or 0.15))
    all_peak_ids = [str(row["peak_id"]) for row in corrected]
    components = _components_from_candidates(all_peak_ids, candidates)
    _v5_ordered_ids, _v5_peaks_original, _v5_shifts, v5_tracks, _v5_diag = _run_v4_frontend(samples, config)

    max_options = int(getattr(config.model, "enum_max_options_per_component", 160) or 160)
    node_limit = int(getattr(config.model, "enum_node_limit", 350000) or 350000)
    component_options: List[List[List[dict]]] = []
    component_diags: List[dict] = []
    v5_full_cover_protected: List[dict] = []
    for ci, component in enumerate(components, 1):
        comp_candidates = [c for c in candidates if set(_track_key(c)) <= component]
        forced = _forced_v5_cover_option(component, v5_tracks, singletons)
        forced_options = [forced] if forced is not None else []
        if forced is not None:
            v5_full_cover_protected.extend(_copy_track(track, "v5_full_cover_protected") for track in forced)
        else:
            v5_full_cover_protected.extend(_copy_track(singletons[pid], "v5_full_cover_singleton") for pid in sorted(component))
        options, cdiag = _enumerate_cover_options(
            component,
            comp_candidates,
            singletons,
            max_options=max_options,
            node_limit=node_limit,
            forced_options=forced_options,
        )
        cdiag["component_id"] = ci
        component_diags.append(cdiag)
        component_options.append(options)

    selected, selector_diag = _select_options_by_guarded_quality(
        component_options,
        ordered_ids,
        config,
        protected_solutions=[v5_full_cover_protected],
    )

    selected = sorted(selected, key=lambda t: (_track_mask(t, ordered_ids), _track_center(t), _track_key(t)))
    for i, track in enumerate(selected, 1):
        track["track_id"] = f"EV5C{i:04d}"

    shifts = {sid: 0.0 for sid in ordered_ids}
    for sid in ordered_ids:
        vals = []
        for reg in getattr(base, "REGION_NAMES", []):
            try:
                vals.append(float(shift_model.get(sid, {}).get(reg, 0.0)))
            except Exception:
                pass
        shifts[sid] = float(sum(vals) / len(vals)) if vals else 0.0

    selected_ids = [pid for track in selected for pid in _track_key(track)]
    dup_count = len(selected_ids) - len(set(selected_ids))
    missing_count = len(set(all_peak_ids) - set(selected_ids))
    diag = {
        **(shift_diag if isinstance(shift_diag, dict) else {"shift_diag": shift_diag}),
        **enum_diag,
        **selector_diag,
        "backend": "enumerated_v5_guarded_cover_search",
        "include_drop1": include_drop1,
        "n_components": len(components),
        "n_selected_tracks": len(selected),
        "missing_peak_refs": missing_count,
        "duplicate_peak_refs": dup_count,
        "component_timeouts": sum(1 for d in component_diags if d.get("timed_out")),
        "component_option_cap": max_options,
        "forced_v5_components": sum(1 for d in component_diags if d.get("forced_v5_options")),
        "v5_protected_tracks": len(v5_full_cover_protected),
        "selected_v5_track_overlap": sum(1 for t in selected if _track_key(t) in {_track_key(v) for v in v5_full_cover_protected}),
    }
    return ordered_ids, peaks_original, shifts, selected, diag


def run_enumerated_v5_pmtc(samples: List[Sample], config: AppConfig) -> JointState:
    ordered_ids, peaks_original, shifts, selected, diag = run_enumerated_cluster_frontend(samples, config)
    labels, cdiag = _pmtc_labels(selected, ordered_ids, config)
    state = _make_state(
        samples,
        ordered_ids,
        peaks_original,
        shifts,
        selected,
        labels,
        objective_value=float(cdiag.get("n_clusters", 0)),
        iterations=1,
    )
    state.objective_value = float(cdiag.get("n_clusters", 0))
    state.meta = {**diag, **{f"pmtc_{k}": v for k, v in cdiag.items()}}  # type: ignore[attr-defined]
    return state


def run_enumerated_v5_mask_only(samples: List[Sample], config: AppConfig) -> JointState:
    ordered_ids, peaks_original, shifts, selected, diag = run_enumerated_frontend(samples, config)
    labels = _mask_labels(selected, ordered_ids)
    return _make_state(samples, ordered_ids, peaks_original, shifts, selected, labels, objective_value=float(len(set(labels))), iterations=1)


def _select_one_mask_pool_cover(
    mask: str,
    mask_candidates: Sequence[dict],
    singletons: Dict[str, dict],
    *,
    target_peak_ids: set[str] | None = None,
    max_options: int,
    node_limit: int,
) -> Tuple[List[dict], dict]:
    if target_peak_ids is None:
        target_peak_ids = set()
        for cand in mask_candidates:
            target_peak_ids.update(_track_key(cand))
    else:
        target_peak_ids = set(target_peak_ids)
    if not target_peak_ids:
        return [], {
            "mask": mask,
            "target_peak_refs": 0,
            "selected_tracks": 0,
            "missing_peak_refs": 0,
            "duplicate_peak_refs": 0,
            "component_count": 0,
        }

    components = _components_from_candidates(sorted(target_peak_ids), mask_candidates)
    selected: List[dict] = []
    component_diags: List[dict] = []
    for ci, component in enumerate(components, 1):
        comp_candidates = [cand for cand in mask_candidates if set(_track_key(cand)) <= component]
        chosen, cdiag = _exact_cover_component(
            component,
            comp_candidates,
            singletons,
            max_options=max_options,
            node_limit=node_limit,
        )
        cdiag["component_id"] = ci
        component_diags.append(cdiag)
        selected.extend(chosen)

    selected_ids = [pid for track in selected for pid in _track_key(track)]
    return selected, {
        "mask": mask,
        "target_peak_refs": len(target_peak_ids),
        "selected_peak_refs": len(selected_ids),
        "selected_tracks": len(selected),
        "missing_peak_refs": len(target_peak_ids - set(selected_ids)),
        "duplicate_peak_refs": len(selected_ids) - len(set(selected_ids)),
        "component_count": len(components),
        "component_timeouts": sum(1 for row in component_diags if row.get("timed_out")),
    }


def run_enumerated_mask_pool_frontend(
    samples: List[Sample],
    config: AppConfig,
) -> Tuple[List[str], Dict[str, List[Peak]], Dict[str, float], List[dict], List[str], dict]:
    base, _gsp = _v4_modules()
    _patch_v4_globals(base, config)
    ordered_ids, rows, peaks_original = _rows_from_peaklists(samples)
    corrected, shift_model, shift_diag = base.apply_shift(rows, ordered_ids)

    candidates, enum_diag = _enumerate_all_mask_candidates(corrected, ordered_ids, base, config, include_drop1=True)
    singletons = _singleton_tracks(corrected, base, ordered_ids, float(getattr(config.model, "residual_gate", 0.15) or 0.15))
    max_options = int(getattr(config.model, "enum_max_options_per_component", 160) or 160)
    node_limit = int(getattr(config.model, "enum_node_limit", 350000) or 350000)
    all_peak_ids = [str(row["peak_id"]) for row in corrected]
    seed_components = _components_from_candidates(all_peak_ids, candidates)
    seed_selected: List[dict] = []
    seed_diags: List[dict] = []
    for ci, component in enumerate(seed_components, 1):
        comp_candidates = [cand for cand in candidates if set(_track_key(cand)) <= component]
        chosen, cdiag = _exact_cover_component(
            component,
            comp_candidates,
            singletons,
            max_options=max_options,
            node_limit=node_limit,
        )
        cdiag["component_id"] = ci
        seed_diags.append(cdiag)
        seed_selected.extend(chosen)
    active_masks = {_track_mask(track, ordered_ids) for track in seed_selected}
    seed_target_by_mask: Dict[str, set[str]] = defaultdict(set)
    for track in seed_selected:
        seed_target_by_mask[_track_mask(track, ordered_ids)].update(_track_key(track))

    candidates_by_mask: Dict[str, List[dict]] = defaultdict(list)
    for cand in candidates:
        mask = _track_mask(cand, ordered_ids)
        if mask in active_masks:
            candidates_by_mask[mask].append(cand)

    selected: List[dict] = []
    labels: List[str] = []
    mask_diags: List[dict] = []
    for mask in sorted(candidates_by_mask, reverse=True):
        mask_tracks, mask_diag = _select_one_mask_pool_cover(
            mask,
            candidates_by_mask[mask],
            singletons,
            target_peak_ids=seed_target_by_mask[mask],
            max_options=max_options,
            node_limit=node_limit,
        )
        mask_diags.append(mask_diag)
        mask_tracks = sorted(mask_tracks, key=lambda track: (_track_center(track), _track_key(track)))
        for track in mask_tracks:
            selected.append(_copy_track(track, f"mask_pool_{mask}"))
            labels.append(f"M{mask}")

    for i, track in enumerate(selected, 1):
        track["track_id"] = f"EV5MP{i:04d}"

    shifts = {sid: 0.0 for sid in ordered_ids}
    for sid in ordered_ids:
        vals = []
        for reg in getattr(base, "REGION_NAMES", []):
            try:
                vals.append(float(shift_model.get(sid, {}).get(reg, 0.0)))
            except Exception:
                pass
        shifts[sid] = float(sum(vals) / len(vals)) if vals else 0.0

    shift_diag_map = shift_diag if isinstance(shift_diag, dict) else {"shift_diag": shift_diag}
    diag = {
        **shift_diag_map,
        **enum_diag,
        "backend": "enumerated_v5_mask_pool_direct",
        "n_active_masks": len(active_masks),
        "active_masks": ",".join(sorted(active_masks, reverse=True)),
        "n_selected_tracks": len(selected),
        "seed_selected_tracks": len(seed_selected),
        "seed_component_timeouts": sum(1 for row in seed_diags if row.get("timed_out")),
        "mask_pool_target_peak_refs": sum(int(row.get("target_peak_refs", 0) or 0) for row in mask_diags),
        "mask_pool_missing_peak_refs": sum(int(row.get("missing_peak_refs", 0) or 0) for row in mask_diags),
        "mask_pool_duplicate_peak_refs": sum(int(row.get("duplicate_peak_refs", 0) or 0) for row in mask_diags),
        "mask_pool_component_timeouts": sum(int(row.get("component_timeouts", 0) or 0) for row in mask_diags),
        "mask_pool_details": mask_diags,
        "component_option_cap": max_options,
    }
    return ordered_ids, peaks_original, shifts, selected, labels, diag


def run_enumerated_v5_mask_pool_direct(samples: List[Sample], config: AppConfig) -> JointState:
    ordered_ids, peaks_original, shifts, selected, labels, diag = run_enumerated_mask_pool_frontend(samples, config)
    state = _make_state(samples, ordered_ids, peaks_original, shifts, selected, labels, objective_value=float(len(set(labels))), iterations=1)
    state.meta = diag  # type: ignore[attr-defined]
    return state


def run_enumerated_v5_mask_guarded(samples: List[Sample], config: AppConfig) -> JointState:
    base, _gsp = _v4_modules()
    _patch_v4_globals(base, config)
    ordered_ids, rows, peaks_original = _rows_from_peaklists(samples)
    corrected, shift_model, shift_diag = base.apply_shift(rows, ordered_ids)

    candidates, enum_diag = _enumerate_all_mask_candidates(corrected, ordered_ids, base, config, include_drop1=True)
    singletons = _singleton_tracks(corrected, base, ordered_ids, float(getattr(config.model, "residual_gate", 0.15) or 0.15))
    all_peak_ids = [str(row["peak_id"]) for row in corrected]
    components = _components_from_candidates(all_peak_ids, candidates)
    _v5_ordered_ids, _v5_peaks_original, _v5_shifts, v5_tracks, _v5_diag = _run_v4_frontend(samples, config)

    max_options = int(getattr(config.model, "enum_max_options_per_component", 160) or 160)
    node_limit = int(getattr(config.model, "enum_node_limit", 350000) or 350000)
    selected: List[dict] = []
    component_diags: List[dict] = []
    for ci, component in enumerate(components, 1):
        comp_candidates = [candidate for candidate in candidates if set(_track_key(candidate)) <= component]
        forced = _forced_v5_cover_option(component, v5_tracks, singletons)
        forced_options = [forced] if forced is not None else []
        options, cdiag = _enumerate_cover_options(
            component,
            comp_candidates,
            singletons,
            max_options=max_options,
            node_limit=node_limit,
            forced_options=forced_options,
        )
        chosen, select_diag = _select_mask_guarded_option(options, ordered_ids, forced_option=forced)
        cdiag.update(select_diag)
        cdiag["component_id"] = ci
        component_diags.append(cdiag)
        selected.extend(chosen)

    selected = sorted(selected, key=lambda track: (_track_mask(track, ordered_ids), _track_center(track), _track_key(track)))
    for i, track in enumerate(selected, 1):
        track["track_id"] = f"EV5M{i:04d}"

    shifts = {sid: 0.0 for sid in ordered_ids}
    for sid in ordered_ids:
        vals = []
        for reg in getattr(base, "REGION_NAMES", []):
            try:
                vals.append(float(shift_model.get(sid, {}).get(reg, 0.0)))
            except Exception:
                pass
        shifts[sid] = float(sum(vals) / len(vals)) if vals else 0.0

    labels = _mask_labels(selected, ordered_ids)
    selected_ids = [pid for track in selected for pid in _track_key(track)]
    diag = {
        **(shift_diag if isinstance(shift_diag, dict) else {"shift_diag": shift_diag}),
        **enum_diag,
        "backend": "enumerated_v5_mask_guarded",
        "n_components": len(components),
        "n_selected_tracks": len(selected),
        "missing_peak_refs": len(set(all_peak_ids) - set(selected_ids)),
        "duplicate_peak_refs": len(selected_ids) - len(set(selected_ids)),
        "component_timeouts": sum(1 for diag_row in component_diags if diag_row.get("timed_out")),
        "component_option_cap": max_options,
        "selected_v5_components": sum(1 for diag_row in component_diags if diag_row.get("mask_guarded_selected_from_v5")),
    }
    state = _make_state(samples, ordered_ids, peaks_original, shifts, selected, labels, objective_value=float(len(set(labels))), iterations=1)
    state.meta = diag  # type: ignore[attr-defined]
    return state


def run_enumerated_v5_selected_mask_only(samples: List[Sample], config: AppConfig) -> JointState:
    state = run_enumerated_v5_pmtc(samples, config)
    selected: List[dict] = []
    for track in state.tracks:
        members = {}
        for sid, peak in track.members.items():
            members[str(sid)] = {
                "sample": str(sid),
                "peak_id": str(peak.peak_id),
                "ppm": float(peak.ppm_raw),
                "ppm_corr": float(peak.corrected_ppm()),
                "intensity": float(peak.intensity if peak.intensity is not None else (peak.area or 1.0)),
                "area": float(peak.area if peak.area is not None else peak.intensity),
            }
        selected.append(
            {
                "track_id": str(track.track_id),
                "member_ids": tuple(sorted(p["peak_id"] for p in members.values())),
                "members": members,
                "score": float(track.quality_score),
            }
        )
    labels = [f"M{_track_mask(track, state.ordered_sample_ids)}" for track in selected]
    out = _make_state(
        samples,
        list(state.ordered_sample_ids),
        state.peaks_original,
        state.shifts,
        selected,
        labels,
        objective_value=float(len(set(labels))),
        iterations=1,
    )
    out.meta = getattr(state, "meta", {})  # type: ignore[attr-defined]
    return out
