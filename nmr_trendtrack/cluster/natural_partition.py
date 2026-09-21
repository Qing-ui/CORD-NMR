from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
import math
import re

from nmr_trendtrack.contracts import ClusterPrototype, Membership, Track, TrendVector

FULL_MASK_MERGE_MAX_TRACKS = 24
FULL_MASK_CROWD_RADIUS_PPM = 0.6
FULL_MASK_CROWD_MIN_FRACTION = 0.5
LARGE_BUCKET_MIN_TRACKS = 30
SEMANTIC_NORM_MAX = 0.80
LOCAL_ATTACH_RADIUS_PPM = 4.0
LOCAL_ATTACH_NORM_MAX = 0.30
ATTENUATED_FAMILY_ATTACH_NORM_MAX = 0.25
ATTENUATED_FAMILY_ATTACH_MIN_FRACTION = 0.5
FAMILY_SIGN_ZERO_BAND = 0.12

_CLUSTER_ID_RE = re.compile(r"^(?P<prefix>b\d+_f\d+)_a\d+$")


def _mask_to_str(mask: Tuple[int, ...]) -> str:
    return ''.join(str(int(v)) for v in mask)


def _is_full_mask(mask: Tuple[int, ...]) -> bool:
    return bool(mask) and all(int(v) == 1 for v in mask)


def _nearest_distance(center_ppm: float, others: List[float]) -> float | None:
    if not others:
        return None
    return min(abs(center_ppm - other) for other in others)


def _sign_signature(values: List[float], zero_band: float = FAMILY_SIGN_ZERO_BAND) -> Tuple[int, ...]:
    out = []
    for x in values:
        if x > zero_band:
            out.append(1)
        elif x < -zero_band:
            out.append(-1)
        else:
            out.append(0)
    return tuple(out)


def _prototype_norm(proto: ClusterPrototype) -> float:
    return float(math.sqrt(sum(float(v) * float(v) for v in proto.mean_step_log_fc)))


def _build_refine_plan(
    tracks: List[Track],
    memberships: List[Membership],
    mask_by_track: Dict[str, Tuple[int, ...]],
    prototype_by_cluster: Dict[str, ClusterPrototype],
) -> Dict[str, str]:
    """Return a refined track->cluster mapping before final output.

    This keeps the old semantic consolidation heuristics, but applies them inside
    the natural cluster partition itself instead of emitting a second synthetic
    final layer like merge_111_*. The returned ids are canonical existing cluster
    ids, so downstream output can use best_cluster_id directly.
    """
    track_by_id = {track.track_id: track for track in tracks}
    membership_by_track = {m.track_id: m for m in memberships}

    tracks_by_mask: Dict[Tuple[int, ...], List[str]] = defaultdict(list)
    for track_id, mask in mask_by_track.items():
        tracks_by_mask[mask].append(track_id)

    refined_cluster_for_track: Dict[str, str] = {}

    def choose_canonical(source_cluster_ids: List[str]) -> str:
        def key(cid: str) -> Tuple[int, float, str]:
            proto = prototype_by_cluster.get(cid)
            n_tracks = proto.n_tracks if proto is not None else 0
            norm = _prototype_norm(proto) if proto is not None else float('inf')
            return (n_tracks, -norm, cid)
        return max(source_cluster_ids, key=key)

    for mask, track_ids in tracks_by_mask.items():
        source_clusters: Dict[str, List[str]] = defaultdict(list)
        for track_id in track_ids:
            m = membership_by_track.get(track_id)
            if m is None or not m.best_cluster_id:
                continue
            source_clusters[m.best_cluster_id].append(track_id)
        if len(source_clusters) <= 1:
            continue

        other_centers = [
            track_by_id[other_id].center_ppm
            for other_mask, other_ids in tracks_by_mask.items()
            if other_mask != mask
            for other_id in other_ids
            if other_id in track_by_id
        ]

        # 1) small crowded full-mask buckets: keep same natural cluster id, but consolidate
        # subclusters to an existing canonical id instead of inventing merge_111_*.
        if len(track_ids) <= FULL_MASK_MERGE_MAX_TRACKS and other_centers:
            affected_clusters: List[str] = []
            for source_cluster_id, source_track_ids in source_clusters.items():
                crowded = 0
                total = 0
                for track_id in source_track_ids:
                    track = track_by_id.get(track_id)
                    if track is None:
                        continue
                    total += 1
                    nearest = _nearest_distance(track.center_ppm, other_centers)
                    if nearest is not None and nearest <= FULL_MASK_CROWD_RADIUS_PPM:
                        crowded += 1
                if total > 0 and (crowded / total) >= FULL_MASK_CROWD_MIN_FRACTION:
                    affected_clusters.append(source_cluster_id)
            if len(affected_clusters) > 1:
                canonical = choose_canonical(affected_clusters)
                for source_cluster_id in affected_clusters:
                    for track_id in source_clusters[source_cluster_id]:
                        refined_cluster_for_track[track_id] = canonical
                continue

        cluster_stats = []
        for source_cluster_id, source_track_ids in source_clusters.items():
            proto = prototype_by_cluster.get(source_cluster_id)
            if proto is None:
                continue
            norm = _prototype_norm(proto)
            sign = _sign_signature(proto.mean_step_log_fc)
            cluster_stats.append((source_cluster_id, source_track_ids, proto, norm, sign))
        if not cluster_stats:
            continue

        # 2) one-step semantic consolidation inside the same sign group.
        if len(cluster_stats[0][2].mean_step_log_fc) == 1:
            sign_groups: Dict[Tuple[int, ...], List[Tuple[str, List[str], ClusterPrototype, float, Tuple[int, ...]]]] = defaultdict(list)
            for stat in cluster_stats:
                sign_groups[stat[4]].append(stat)
            merged_any = False
            for _, items in sign_groups.items():
                if len(items) <= 1:
                    continue
                total_tracks = sum(len(ids) for _, ids, _, _, _ in items)
                strongest = max(norm for _, _, _, norm, _ in items)
                largest = max(len(ids) for _, ids, _, _, _ in items)
                second = sorted((len(ids) for _, ids, _, _, _ in items), reverse=True)[1]
                if total_tracks < 12 or total_tracks > 35 or strongest < 1.2 or largest > 30 or second < 5:
                    continue
                merged_any = True
                canonical = choose_canonical([sid for sid, _, _, _, _ in items])
                for source_cluster_id, source_track_ids, _, _, _ in items:
                    for track_id in source_track_ids:
                        refined_cluster_for_track[track_id] = canonical
            if merged_any:
                continue
            continue

        # 3) large same-sign semantic consolidation with strong-family protection.
        if len(track_ids) < LARGE_BUCKET_MIN_TRACKS:
            continue

        sign_groups: Dict[Tuple[int, ...], List[Tuple[str, List[str], ClusterPrototype, float, Tuple[int, ...]]]] = defaultdict(list)
        for stat in cluster_stats:
            sign_groups[stat[4]].append(stat)
        dominant_items = max(sign_groups.values(), key=lambda items: sum(len(ids) for _, ids, _, _, _ in items))
        dominant_cluster_ids = [sid for sid, _, _, _, _ in dominant_items]
        canonical = choose_canonical(dominant_cluster_ids)
        merge_seed_clusters = [sid for sid, _, _, norm, _ in dominant_items if norm <= SEMANTIC_NORM_MAX]
        if canonical not in merge_seed_clusters:
            merge_seed_clusters.append(canonical)

        merged_track_centers: List[float] = []
        for source_cluster_id in merge_seed_clusters:
            for track_id in source_clusters[source_cluster_id]:
                refined_cluster_for_track[track_id] = canonical
                track = track_by_id.get(track_id)
                if track is not None:
                    merged_track_centers.append(track.center_ppm)

        for source_cluster_id, source_track_ids, _, norm, sign in cluster_stats:
            if source_cluster_id in merge_seed_clusters:
                continue
            local_hits = 0
            total = 0
            for track_id in source_track_ids:
                track = track_by_id.get(track_id)
                if track is None:
                    continue
                total += 1
                nearest = _nearest_distance(track.center_ppm, merged_track_centers)
                if nearest is not None and nearest <= LOCAL_ATTACH_RADIUS_PPM:
                    local_hits += 1
            frac = (local_hits / total) if total else 0.0
            is_flattened = all(v == 0 for v in sign)
            if len(source_track_ids) <= 2 and norm <= LOCAL_ATTACH_NORM_MAX and frac > 0.0:
                for track_id in source_track_ids:
                    refined_cluster_for_track[track_id] = canonical
                continue
            if (norm <= ATTENUATED_FAMILY_ATTACH_NORM_MAX or is_flattened) and frac >= ATTENUATED_FAMILY_ATTACH_MIN_FRACTION:
                for track_id in source_track_ids:
                    refined_cluster_for_track[track_id] = canonical

    return refined_cluster_for_track


def _family_id_from_cluster_id(cluster_id: str) -> str | None:
    m = _CLUSTER_ID_RE.match(cluster_id)
    if not m:
        return None
    return m.group('prefix')


def _aggregate_prototypes(
    trend_vectors: List[TrendVector],
    memberships: List[Membership],
    base_prototypes: Dict[str, ClusterPrototype],
) -> List[ClusterPrototype]:
    tv_by_id = {tv.track_id: tv for tv in trend_vectors}
    track_ids_by_cluster: Dict[str, List[str]] = defaultdict(list)
    source_ids_by_cluster: Dict[str, List[str]] = defaultdict(list)
    for m in memberships:
        if not m.best_cluster_id:
            continue
        track_ids_by_cluster[m.best_cluster_id].append(m.track_id)
        if m.second_cluster_id:
            source_ids_by_cluster[m.best_cluster_id].append(m.second_cluster_id)
    out: List[ClusterPrototype] = []
    for cluster_id in sorted(track_ids_by_cluster):
        ids = track_ids_by_cluster[cluster_id]
        rows = [tv_by_id[tid] for tid in ids if tid in tv_by_id]
        if not rows:
            continue
        d = max(len(tv.step_log_fc) for tv in rows)
        means: List[float] = []
        scales: List[float] = []
        for j in range(d):
            vals = [float(tv.step_log_fc[j]) for tv in rows if j < len(tv.step_log_fc) and tv.step_log_fc[j] is not None]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
                scale = math.sqrt(max(var, 1e-4))
            else:
                mean = 0.0
                scale = 1.0
            means.append(float(mean))
            scales.append(float(scale))
        proto0 = base_prototypes.get(cluster_id)
        amp_center = float(math.sqrt(sum(m * m for m in means))) if means else 0.0
        out.append(
            ClusterPrototype(
                cluster_id=cluster_id,
                presence_mask=rows[0].presence_mask,
                mean_step_log_fc=means,
                step_scale=scales,
                n_tracks=len(ids),
                weight=1.0,
                family_id=(proto0.family_id if proto0 is not None else _family_id_from_cluster_id(cluster_id)),
                family_direction=(proto0.family_direction if proto0 is not None else None),
                amplitude_center=(proto0.amplitude_center if proto0 is not None else amp_center),
                amplitude_scale=(proto0.amplitude_scale if proto0 is not None else 1.0),
                amplitude_rank=(proto0.amplitude_rank if proto0 is not None else 0),
            )
        )
    return out


def refine_natural_clusters(
    trend_vectors: List[TrendVector],
    tracks: List[Track],
    prototypes: List[ClusterPrototype],
    memberships: List[Membership],
) -> Tuple[List[ClusterPrototype], List[Membership]]:
    """Refine best_cluster_id before final output.

    The public pipeline should export natural clusters directly, without a second
    synthetic merge layer. We therefore apply the semantic consolidation plan here
    and rewrite best_cluster_id / cluster_probs to the canonical natural cluster id.
    """
    if not memberships:
        return prototypes, memberships
    mask_by_track: Dict[str, Tuple[int, ...]] = {tv.track_id: tv.presence_mask for tv in trend_vectors}
    prototype_by_cluster: Dict[str, ClusterPrototype] = {p.cluster_id: p for p in prototypes}
    refined_track_cluster = _build_refine_plan(tracks, memberships, mask_by_track, prototype_by_cluster)
    if not refined_track_cluster:
        return prototypes, memberships

    # Build canonical mapping at the source-cluster level when possible.
    source_to_canonical: Dict[str, str] = {}
    for m in memberships:
        if not m.best_cluster_id:
            continue
        canonical = refined_track_cluster.get(m.track_id, m.best_cluster_id)
        source_to_canonical.setdefault(m.best_cluster_id, canonical)

    for m in memberships:
        if not m.best_cluster_id:
            continue
        canonical = refined_track_cluster.get(m.track_id, source_to_canonical.get(m.best_cluster_id, m.best_cluster_id))
        if canonical == m.best_cluster_id:
            continue
        new_probs: Dict[str, float] = defaultdict(float)
        for cid, prob in m.cluster_probs.items():
            new_probs[source_to_canonical.get(cid, cid)] += float(prob)
        m.cluster_probs = dict(sorted(new_probs.items()))
        ranked = sorted(m.cluster_probs.items(), key=lambda kv: kv[1], reverse=True)
        m.best_cluster_id = ranked[0][0] if ranked else canonical
        m.second_cluster_id = ranked[1][0] if len(ranked) > 1 else None
        m.family_id = _family_id_from_cluster_id(m.best_cluster_id) or m.family_id

    refined_prototypes = _aggregate_prototypes(trend_vectors, memberships, prototype_by_cluster)
    return refined_prototypes, memberships
