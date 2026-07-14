from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import math
import numpy as np

from nmr_trendtrack.config import AlignConfig
from nmr_trendtrack.contracts import AlignmentCandidate, Peak, Sample, Track
from nmr_trendtrack.align.ppm_window import get_ppm_window
from nmr_trendtrack.preprocess.intensity_clean import choose_signal_value
from nmr_trendtrack.component.local_square_nmf import fit_local_square_component_model, score_track_against_components


def _build_peak_lookup(peaks_by_sample: Dict[str, List[Peak]]) -> Dict[str, Peak]:
    return {p.peak_id: p for peaks in peaks_by_sample.values() for p in peaks}


def _build_compatibility(candidates: List[AlignmentCandidate]) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], float]]:
    compat: Dict[str, Set[str]] = defaultdict(set)
    pair_score: Dict[Tuple[str, str], float] = {}
    for c in candidates:
        compat[c.peak_a_id].add(c.peak_b_id)
        compat[c.peak_b_id].add(c.peak_a_id)
        key = tuple(sorted((c.peak_a_id, c.peak_b_id)))
        pair_score[key] = max(pair_score.get(key, 0.0), c.score_ppm)
    return compat, pair_score


def _build_peak_graph(candidates: List[AlignmentCandidate]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    for c in candidates:
        graph[c.peak_a_id].add(c.peak_b_id)
        graph[c.peak_b_id].add(c.peak_a_id)
    return graph


def _connected_components(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    seen: Set[str] = set()
    comps: List[Set[str]] = []
    for node in list(graph):
        if node in seen:
            continue
        stack = [node]
        comp: Set[str] = set()
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _track_center(members: Dict[str, Peak]) -> float:
    vals = [p.corrected_ppm() for p in members.values()]
    return float(np.median(vals)) if vals else 0.0


def _track_soft_ppm_penalty(members: Dict[str, Peak], cfg: AlignConfig) -> Tuple[float, float, float]:
    center = _track_center(members)
    if not members:
        return 0.0, 0.0, center
    corr_ppms = [p.corrected_ppm() for p in members.values()]
    span = max(corr_ppms) - min(corr_ppms)
    hard_win = get_ppm_window(center, cfg)
    soft_win = hard_win * cfg.center_soft_factor
    penalty = 0.0
    for p in members.values():
        d = abs(p.corrected_ppm() - center)
        if d > soft_win:
            return float("inf"), span, center
        if d > hard_win:
            penalty += cfg.center_soft_penalty * ((d - hard_win) / max(soft_win - hard_win, 1e-8))
    return penalty, span, center


def _track_trend_penalty(members: Dict[str, Peak], ordered_sample_ids: List[str], sample_scales: Optional[Dict[str, float]], cfg: AlignConfig) -> float:
    if len(members) < 3:
        return 0.0
    scales = sample_scales or {}
    vals = []
    for sid in ordered_sample_ids:
        peak = members.get(sid)
        if peak is None:
            vals.append(None)
            continue
        vals.append(choose_signal_value(peak, use_area=False) / max(scales.get(sid, 1.0), 1e-8))
    steps: List[float] = []
    for i in range(len(vals) - 1):
        a, b = vals[i], vals[i + 1]
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        steps.append(float(math.log((b + 1e-8) / (a + 1e-8))))
    if len(steps) < 2:
        return 0.0
    med = float(np.median(steps))
    step_range = max(steps) - min(steps)
    step_dev = max(abs(s - med) for s in steps)
    sign_conflicts = 0
    sig = 0.35
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            if abs(steps[i]) > sig and abs(steps[j]) > sig and steps[i] * steps[j] < 0:
                sign_conflicts += 1
    penalty = cfg.trend_sign_conflict_penalty * sign_conflicts
    if step_range > cfg.trend_max_step_gap:
        penalty += cfg.trend_step_gap_penalty * (step_range - cfg.trend_max_step_gap)
    if step_dev > cfg.trend_max_step_gap:
        penalty += 0.5 * cfg.trend_step_gap_penalty * (step_dev - cfg.trend_max_step_gap)
    return penalty


def _pair_quality_bonus(members: Dict[str, Peak], pair_scores: Dict[Tuple[str, str], float], cfg: AlignConfig) -> float:
    peak_ids = [p.peak_id for p in members.values()]
    vals: List[float] = []
    for i in range(len(peak_ids)):
        for j in range(i + 1, len(peak_ids)):
            vals.append(pair_scores.get(tuple(sorted((peak_ids[i], peak_ids[j]))), 0.0))
    if not vals:
        return 0.0
    return cfg.pair_score_weight * float(np.mean(vals)) * max(1, len(vals))


def _shape_penalty(members: Dict[str, Peak], cfg: AlignConfig) -> float:
    width_logs: List[float] = []
    shape_logs: List[float] = []
    for peak in members.values():
        if peak.width_hz is not None and peak.width_hz > 0:
            width_logs.append(float(math.log(peak.width_hz)))
        if peak.area is not None and peak.intensity > 0 and peak.area > 0:
            shape_logs.append(float(math.log(peak.area / peak.intensity)))
    penalty = 0.0
    if len(width_logs) >= 2:
        penalty += cfg.track_width_penalty_weight * float(np.std(width_logs, ddof=0))
    if len(shape_logs) >= 2:
        penalty += cfg.track_shape_penalty_weight * float(np.std(shape_logs, ddof=0))
    return penalty


def _build_local_support_scores(peaks_by_sample: Dict[str, List[Peak]], cfg: AlignConfig) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for peaks in peaks_by_sample.values():
        peaks_sorted = sorted(peaks, key=lambda p: p.corrected_ppm())
        for peak in peaks_sorted:
            neighbors = [q for q in peaks_sorted if abs(q.corrected_ppm() - peak.corrected_ppm()) <= cfg.local_support_window_ppm]
            neighbors = sorted(neighbors, key=lambda q: q.intensity, reverse=True)
            if not neighbors:
                scores[peak.peak_id] = 0.0
                continue
            max_int = max(float(q.intensity) for q in neighbors)
            prominence = float(peak.intensity) / max(max_int, 1e-8)
            rank = 1 + next((i for i, q in enumerate(neighbors) if q.peak_id == peak.peak_id), 0)
            rank_term = 1.0 / rank
            scores[peak.peak_id] = 0.65 * prominence + 0.35 * rank_term
    return scores


def _local_support_bonus(members: Dict[str, Peak], support_scores: Dict[str, float], cfg: AlignConfig) -> float:
    if not members:
        return 0.0
    vals = [support_scores.get(p.peak_id, 0.0) for p in members.values()]
    return cfg.local_support_weight * float(np.mean(vals))


def _score_track_geometry(
    members: Dict[str, Peak],
    ordered_sample_ids: List[str],
    cfg: AlignConfig,
    sample_scales: Optional[Dict[str, float]] = None,
    pair_scores: Optional[Dict[Tuple[str, str], float]] = None,
    support_scores: Optional[Dict[str, float]] = None,
    component_prior_bonus: float = 0.0,
) -> Tuple[float, float, float]:
    coverage = len(members)
    soft_ppm_penalty, span, center = _track_soft_ppm_penalty(members, cfg)
    if math.isinf(soft_ppm_penalty):
        return float("-inf"), span, center
    missing = len(ordered_sample_ids) - coverage
    coverage_bonus = 2.0 * coverage + 0.9 * max(0, coverage - 2)
    score = coverage_bonus - 2.0 * span - soft_ppm_penalty - 0.55 * missing
    score -= _track_trend_penalty(members, ordered_sample_ids, sample_scales, cfg)
    score -= _shape_penalty(members, cfg)
    if pair_scores is not None:
        score += _pair_quality_bonus(members, pair_scores, cfg)
    if support_scores is not None:
        score += _local_support_bonus(members, support_scores, cfg)
    score += component_prior_bonus
    return score, span, center


def enumerate_candidate_tracks(samples: List[Sample], peaks_by_sample: Dict[str, List[Peak]], candidates: List[AlignmentCandidate], cfg: AlignConfig, sample_scales: Optional[Dict[str, float]] = None) -> List[Track]:
    peak_lookup = _build_peak_lookup(peaks_by_sample)
    compat, pair_scores = _build_compatibility(candidates)
    graph = _build_peak_graph(candidates)
    support_scores = _build_local_support_scores(peaks_by_sample, cfg)
    ordered_sample_ids = [s.sample_id for s in sorted(samples, key=lambda s: s.order_index)]
    components = _connected_components(graph)
    all_tracks: List[Track] = []
    track_counter = 0
    for comp in components:
        comp_by_sample: Dict[str, List[Peak]] = defaultdict(list)
        for pid in comp:
            peak = peak_lookup[pid]
            comp_by_sample[peak.sample_id].append(peak)
        if sum(len(v) for v in comp_by_sample.values()) > cfg.max_component_size:
            peaks_sorted = sorted([peak_lookup[pid] for pid in comp], key=lambda p: p.corrected_ppm())
            chunks: List[List[Peak]] = [[]]
            for p in peaks_sorted:
                if chunks[-1] and abs(p.corrected_ppm() - chunks[-1][-1].corrected_ppm()) > cfg.max_track_span_ppm * cfg.center_soft_factor:
                    chunks.append([])
                chunks[-1].append(p)
            comp_subsets = [set(p.peak_id for p in chunk) for chunk in chunks if chunk]
        else:
            comp_subsets = [comp]
        for subset in comp_subsets:
            subset_by_sample: Dict[str, List[Peak]] = defaultdict(list)
            for pid in subset:
                subset_by_sample[peak_lookup[pid].sample_id].append(peak_lookup[pid])
            sample_order = [sid for sid in ordered_sample_ids if sid in subset_by_sample]
            if len(sample_order) < cfg.min_track_size:
                continue
            sample_order = sorted(sample_order, key=lambda sid: len(subset_by_sample[sid]))
            component_model = None
            if cfg.enable_local_square_component_prior:
                component_model = fit_local_square_component_model(subset_by_sample, ordered_sample_ids, sample_scales, max_components=cfg.local_square_component_max_components)
            best_tracks: Dict[Tuple[str, ...], Track] = {}

            def can_add(current: Dict[str, Peak], cand: Peak) -> bool:
                if cand.sample_id in current:
                    return False
                if current and cfg.require_one_compat_edge and not any(other.peak_id in compat.get(cand.peak_id, set()) for other in current.values()):
                    return False
                proposed = dict(current)
                proposed[cand.sample_id] = cand
                pen, span, _ = _track_soft_ppm_penalty(proposed, cfg)
                if math.isinf(pen):
                    return False
                if span > cfg.max_track_span_ppm * cfg.center_soft_factor:
                    return False
                return True

            def optimistic_remaining(idx: int, current_size: int) -> int:
                return current_size + max(0, len(sample_order) - idx)

            def backtrack(idx: int, current: Dict[str, Peak]):
                nonlocal track_counter
                if optimistic_remaining(idx, len(current)) < cfg.min_track_size:
                    return
                if idx == len(sample_order):
                    if len(current) >= cfg.min_track_size:
                        key = tuple(sorted(p.peak_id for p in current.values()))
                        component_bonus = 0.0
                        if component_model is not None:
                            component_bonus = cfg.local_square_component_weight * score_track_against_components(current, component_model, sample_scales)
                        score, span, center = _score_track_geometry(current, ordered_sample_ids, cfg, sample_scales, pair_scores, support_scores, component_bonus)
                        if score == float("-inf"):
                            return
                        if key not in best_tracks or score > best_tracks[key].quality_score:
                            track_counter += 1
                            best_tracks[key] = Track(track_id=f"tr_{track_counter:06d}", members=dict(current), center_ppm=center, ppm_span=span, quality_score=score)
                    return
                sid = sample_order[idx]
                options = sorted(subset_by_sample[sid], key=lambda p: p.corrected_ppm())
                for peak in options:
                    if can_add(current, peak):
                        current[peak.sample_id] = peak
                        backtrack(idx + 1, current)
                        current.pop(peak.sample_id, None)
                backtrack(idx + 1, current)

            backtrack(0, {})
            ranked = sorted(best_tracks.values(), key=lambda t: (-t.quality_score, t.ppm_span, -len(t.members)))
            all_tracks.extend(ranked[: cfg.max_tracks_per_component])
    return all_tracks
