"""Weighted set-packing by connected peak-conflict components."""


from __future__ import annotations


import importlib.util
from collections import Counter, defaultdict
from pathlib import Path
from typing import List


_base_path = Path(__file__).with_name("13c_joint_align_cluster.py")
_spec = importlib.util.spec_from_file_location("base13c", _base_path)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def relabel(tracks: List[dict]) -> List[dict]:
    out = []
    for i, t in enumerate(tracks, 1):
        tt = dict(t)
        tt["track_id"] = f"T{i:04d}"
        out.append(tt)
    return out


def track_weight(track: dict) -> float:
    """Objective used by set-packing.

    It intentionally does not use component truth.  The candidate score is an
    average fit/coverage quality, so multiplying by the number of assigned
    peaks converts it into total explained evidence.  A fixed model-cost then
    discourages explaining one coherent signal as several fragmented tracks.
    """
    n = len(track["member_ids"])
    score = max(0.0, float(track.get("score", 0.0)))
    if n <= 1:
        # Keep singleton tracks available as a low-value coverage fallback, but
        # do not let them make fragmented alternatives look artificially good.
        return 0.02
    return score * n - 1.10


def conflict_components(candidate_tracks: List[dict]) -> List[List[int]]:
    """Split candidate-track conflict graph into connected components.

    Nodes are candidate tracks. Two nodes conflict if they share any input peak.
    """
    dsu = DSU(len(candidate_tracks))
    by_peak = defaultdict(list)
    for i, t in enumerate(candidate_tracks):
        for pid in t["member_ids"]:
            by_peak[pid].append(i)
    for ids in by_peak.values():
        for j in ids[1:]:
            dsu.union(ids[0], j)

    comps = defaultdict(list)
    for i in range(len(candidate_tracks)):
        comps[dsu.find(i)].append(i)
    return list(comps.values())


def exact_component(candidate_tracks: List[dict], idxs: List[int], node_limit: int = 100000, *, weight_fn=None):
    """Branch-and-bound exact weighted set-packing for one small conflict component."""
    weight = weight_fn or track_weight
    idxs = sorted(idxs, key=lambda i: weight(candidate_tracks[i]), reverse=True)
    weights = [weight(candidate_tracks[i]) for i in idxs]

    suffix = [0.0] * (len(idxs) + 1)
    for k in range(len(idxs) - 1, -1, -1):
        suffix[k] = suffix[k + 1] + max(0.0, weights[k])

    best_score = -1e18
    best_sel: List[int] = []
    nodes = 0

    def rec(pos: int, used: set, score: float, sel: List[int]):
        nonlocal best_score, best_sel, nodes
        nodes += 1
        if nodes > node_limit:
            raise TimeoutError("branch-and-bound node limit exceeded")

        if score + suffix[pos] <= best_score + 1e-12:
            return

        if pos == len(idxs):
            if score > best_score:
                best_score = score
                best_sel = list(sel)
            return

        i = idxs[pos]
        ids = set(candidate_tracks[i]["member_ids"])

        # include
        if not (ids & used):
            sel.append(i)
            rec(pos + 1, used | ids, score + weights[pos], sel)
            sel.pop()

        # exclude
        rec(pos + 1, used, score, sel)

    try:
        rec(0, set(), 0.0, [])
        return best_sel, "exact"
    except TimeoutError:
        return None, "timeout"


def beam_component(candidate_tracks: List[dict], idxs: List[int], beam_width: int = 120, *, weight_fn=None) -> List[int]:
    """Beam-search approximate weighted set-packing for one large conflict component."""
    weight = weight_fn or track_weight
    idxs = sorted(idxs, key=lambda i: weight(candidate_tracks[i]), reverse=True)
    states = [(0.0, tuple(), frozenset())]  # score, selected indices, used peak ids

    for i in idxs:
        ids = frozenset(candidate_tracks[i]["member_ids"])
        w = weight(candidate_tracks[i])
        new_states = list(states)
        for score, sel, used in states:
            if not (ids & used):
                new_states.append((score + w, sel + (i,), used | ids))

        # Deduplicate by used-peak set and keep best beam.
        best = {}
        for st in new_states:
            if st[2] not in best or st[0] > best[st[2]][0]:
                best[st[2]] = st
        states = sorted(best.values(), key=lambda x: x[0], reverse=True)[:beam_width]

    return list(states[0][1]) if states else []


def componentwise_global_set_packing(
    candidate_tracks: List[dict],
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
    *,
    weight_fn=None,
):
    """Global-ish set-packing by conflict-component decomposition.

    Small conflict components are solved exactly by branch-and-bound. Large
    components use beam search. This is substantially less greedy than the old
    "sort candidates and pick non-conflicting" strategy while remaining fast.
    """
    comps = conflict_components(candidate_tracks)
    selected = []
    mode_counts = Counter()
    sizes = []

    for comp in comps:
        sizes.append(len(comp))
        if len(comp) <= exact_limit:
            sel, mode = exact_component(candidate_tracks, comp, node_limit=node_limit, weight_fn=weight_fn)
            if sel is None:
                sel = beam_component(candidate_tracks, comp, beam_width=beam_width, weight_fn=weight_fn)
                mode = "beam_timeout"
        else:
            sel = beam_component(candidate_tracks, comp, beam_width=beam_width, weight_fn=weight_fn)
            mode = "beam_large"

        selected.extend(dict(candidate_tracks[i]) for i in sel)
        mode_counts[mode] += 1

    info = {
        "n_conflict_components": len(comps),
        "max_component_candidates": max(sizes) if sizes else 0,
        "mean_component_candidates": round(sum(sizes) / len(sizes), 3) if sizes else 0.0,
    }
    info.update({f"solve_{k}": v for k, v in mode_counts.items()})
    return relabel(selected), info


def connected_components(distance_matrix: List[List[float]], threshold: float):
    n = len(distance_matrix)
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i][j] <= threshold:
                dsu.union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    return list(groups.values())


def light_joint_cluster(tracks: List[dict], samples: List[str], cfg: dict):
    """Lightweight joint trend/profile clustering.

    It combines:
    - row-centered log-intensity distance
    - presence-mask compatibility
    - alignment confidence
    It auto-selects a cut threshold by an internal objective.
    """
    n = len(tracks)
    if n == 0:
        return [], {}

    mask_weight = cfg.get("mask_weight", 0.38)
    D = [[0.0] * n for _ in range(n)]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            d = base.track_distance(tracks[i], tracks[j], samples, mask_weight=mask_weight)
            D[i][j] = D[j][i] = d
            vals.append(d)

    vals = sorted(vals)
    quantiles = [0.02, 0.05, 0.09, 0.14, 0.22, 0.32]
    thresholds = sorted(
        set(
            [vals[min(len(vals) - 1, int(q * (len(vals) - 1)))] for q in quantiles if vals]
            + [0.24, 0.36, 0.52, 0.80, 1.0]
        )
    )

    best = None
    for thr in thresholds:
        clusters = connected_components(D, thr)
        if len(clusters) == n and n > 12:
            continue
        labels = [None] * n
        for ci, c in enumerate(clusters):
            for i in c:
                labels[i] = ci

        sil = base.silhouette(D, labels)
        tiny = sum(len(c) for c in clusters if len(c) < 3) / max(n, 1)
        maxfrac = max(len(c) for c in clusters) / max(n, 1)
        score = sil - 0.045 * len(clusters) - 0.45 * tiny - 1.20 * max(0.0, maxfrac - 0.55)

        if best is None or score > best[0]:
            best = (score, clusters, thr, sil, tiny, maxfrac)

    if best is None:
        best = (0, [[i] for i in range(n)], None, 0, 1, 1)

    labels = [""] * n
    for ci, c in enumerate(best[1], 1):
        for i in c:
            labels[i] = f"C{ci:02d}"

    diag = {
        "k": len(best[1]),
        "threshold": best[2],
        "internal_score": round(best[0], 4),
        "silhouette": round(best[3], 4),
        "tiny_frac": round(best[4], 4),
        "max_cluster_frac": round(best[5], 4),
    }
    return labels, diag
