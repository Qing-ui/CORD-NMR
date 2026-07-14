from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

import math

from nmr_trendtrack.config import AppConfig
from nmr_trendtrack.contracts import ClusterPrototype, JointState, Membership, Peak, Sample, Track
from nmr_trendtrack.io import load_peaklist_for_sample
from nmr_trendtrack.preprocess.sample_order import ordered_sample_ids as build_ordered_sample_ids
from nmr_trendtrack.align import estimate_warp_maps, apply_warp_maps, update_warp_maps_from_tracks, build_alignment_candidates, enumerate_candidate_tracks, select_tracks_via_set_packing
from nmr_trendtrack.trend import attach_presence_masks, estimate_sample_scales, normalize_track_intensities, compute_trend_vector
from nmr_trendtrack.cluster import bucket_by_presence_mask, fit_all_buckets, refine_natural_clusters, build_component_clusters


def initialize_state(samples: List[Sample], config: AppConfig) -> JointState:
    peaks_original: Dict[str, List[Peak]] = {s.sample_id: load_peaklist_for_sample(s) for s in samples}
    if config.align.allow_global_shift:
        shifts, warp_maps = estimate_warp_maps(samples, peaks_original, config.align)
    else:
        shifts = {s.sample_id: 0.0 for s in samples}
        warp_maps = {s.sample_id: [] for s in samples}
    peaks_corrected = apply_warp_maps(peaks_original, warp_maps, shifts)
    return JointState(samples=samples, ordered_sample_ids=build_ordered_sample_ids(samples), peaks_original=peaks_original, peaks_corrected=peaks_corrected, shifts=shifts, warp_maps=warp_maps)


def _membership_label_map(memberships: List[Membership]) -> Dict[str, str]:
    return {m.track_id: m.assigned_label for m in memberships}


def _prototype_map(prototypes: List[ClusterPrototype]) -> Dict[Tuple[int, ...], List[ClusterPrototype]]:
    out: Dict[Tuple[int, ...], List[ClusterPrototype]] = {}
    for p in prototypes:
        out.setdefault(p.presence_mask, []).append(p)
    return out


def _trend_bonus_for_track(track: Track, ordered_sample_ids: List[str], sample_scales: Dict[str, float], prototypes_by_mask: Dict[Tuple[int, ...], List[ClusterPrototype]], use_area: bool, eps: float, weight: float) -> float:
    if not prototypes_by_mask:
        return 0.0
    mask = tuple(1 if sid in track.members else 0 for sid in ordered_sample_ids)
    protos = prototypes_by_mask.get(mask)
    if not protos:
        return 0.0
    norm = {}
    for sid, peak in track.members.items():
        value = peak.area if use_area and peak.area is not None else peak.intensity
        norm[sid] = value / max(sample_scales.get(sid, 1.0), 1e-8)
    vals = [norm.get(sid) for sid in ordered_sample_ids]
    steps = []
    for i in range(len(vals) - 1):
        a, b = vals[i], vals[i + 1]
        if a is None or b is None:
            steps.append(None)
        else:
            steps.append(math.log((b + eps) / (a + eps)))
    best = None
    for proto in protos:
        total = 0.0
        count = 0
        for i, x in enumerate(steps):
            if x is None:
                continue
            scale = max(proto.step_scale[i], 1e-6)
            total += math.log1p(((x - proto.mean_step_log_fc[i]) / scale) ** 2)
            count += 1
        if count == 0:
            continue
        score = -total / count
        if best is None or score > best:
            best = score
    return 0.0 if best is None else weight * float(best)


def _compute_objective(state: JointState) -> float:
    align_term = sum((tr.quality_score + tr.trend_bonus) for tr in state.tracks)
    purity_bonus = 0.0
    shared_penalty = 0.0
    for m in state.memberships:
        if m.assigned_label == 'pure':
            purity_bonus += 0.25
        elif m.assigned_label == 'shared':
            shared_penalty += 0.15
    cluster_penalty = 0.02 * len(state.cluster_prototypes)
    return align_term + purity_bonus - shared_penalty - cluster_penalty


def _tracks_signature(tracks: List[Track]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(sorted(tuple(sorted(t.member_peak_ids())) for t in tracks))


def _single_run(samples: List[Sample], config: AppConfig) -> JointState:
    state = initialize_state(samples, config)
    best_state = deepcopy(state)
    best_state.objective_value = float('-inf')
    prev_obj = None
    prev_sig = None
    for it in range(config.optimize.n_outer_iters):
        prototypes_by_mask = _prototype_map(state.cluster_prototypes)
        state.peaks_corrected = apply_warp_maps(state.peaks_original, state.warp_maps, state.shifts)
        candidates = build_alignment_candidates(state.samples, state.peaks_corrected, config.align)
        cand_tracks = enumerate_candidate_tracks(state.samples, state.peaks_corrected, candidates, config.align, sample_scales=(state.sample_scales or {sid: 1.0 for sid in state.ordered_sample_ids}))
        for tr in cand_tracks:
            tr.trend_bonus = _trend_bonus_for_track(tr, state.ordered_sample_ids, state.sample_scales or {sid: 1.0 for sid in state.ordered_sample_ids}, prototypes_by_mask, config.trend.use_area_instead_of_height, config.trend.epsilon, config.optimize.trend_bonus_weight)
        state.tracks = select_tracks_via_set_packing(cand_tracks)
        attach_presence_masks(state.tracks, state.ordered_sample_ids)
        memberships_by_track = _membership_label_map(state.memberships)
        if state.tracks:
            state.shifts, state.warp_maps = update_warp_maps_from_tracks(state.samples, state.peaks_original, state.tracks, memberships_by_track, state.warp_maps, state.shifts, config.align)
        state.sample_scales = estimate_sample_scales(state.tracks, state.ordered_sample_ids, config.trend.normalization_method, config.trend.use_area_instead_of_height)
        normalized = normalize_track_intensities(state.tracks, state.sample_scales, config.trend.use_area_instead_of_height)
        state.trend_vectors = [compute_trend_vector(tr, state.ordered_sample_ids, normalized, config.trend.epsilon) for tr in state.tracks]
        buckets = bucket_by_presence_mask(state.trend_vectors)
        state.cluster_prototypes, state.memberships = fit_all_buckets(buckets, config.cluster)
        state.cluster_prototypes, state.memberships = refine_natural_clusters(state.trend_vectors, state.tracks, state.cluster_prototypes, state.memberships)
        state.component_cluster_prototypes, state.final_cluster_prototypes, state.memberships = build_component_clusters(state.trend_vectors, state.memberships)
        state.objective_value = _compute_objective(state)
        state.outer_iterations_completed = it + 1
        cur_sig = _tracks_signature(state.tracks)
        if state.objective_value > best_state.objective_value:
            best_state = deepcopy(state)
            best_state.best_iteration = it + 1
        if prev_obj is not None and abs(state.objective_value - prev_obj) < config.optimize.convergence_tol and cur_sig == prev_sig:
            state.converged = True
            best_state.converged = True
            break
        prev_obj = state.objective_value
        prev_sig = cur_sig
    best_state.outer_iterations_completed = state.outer_iterations_completed
    best_state.converged = state.converged
    return best_state


def run_joint_optimization(samples: List[Sample], config: AppConfig) -> JointState:
    best = None
    for _ in range(config.optimize.n_starts):
        state = _single_run(samples, config)
        if best is None or state.objective_value > best.objective_value:
            best = state
    assert best is not None
    return best
