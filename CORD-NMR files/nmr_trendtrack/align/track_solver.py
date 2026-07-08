from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from nmr_trendtrack.contracts import Track


def _conflict_components(tracks: List[Track]) -> List[List[int]]:
    peak_to_tracks: Dict[str, List[int]] = defaultdict(list)
    for i, tr in enumerate(tracks):
        for pid in tr.member_peak_ids():
            peak_to_tracks[pid].append(i)

    graph: Dict[int, Set[int]] = defaultdict(set)
    for idxs in peak_to_tracks.values():
        for i in idxs:
            graph.setdefault(i, set())
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                graph[a].add(b)
                graph[b].add(a)
    seen: Set[int] = set()
    comps: List[List[int]] = []
    for node in range(len(tracks)):
        if node in seen:
            continue
        stack = [node]
        comp = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in graph.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _solve_component(indices: List[int], tracks: List[Track]) -> List[int]:
    comp_tracks = [tracks[i] for i in indices]
    n = len(comp_tracks)
    member_sets = [set(t.member_peak_ids()) for t in comp_tracks]
    scores = [t.quality_score + t.trend_bonus for t in comp_tracks]
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    ordered_scores = [scores[i] for i in order]
    suffix_pos = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_pos[i] = suffix_pos[i + 1] + max(0.0, ordered_scores[i])

    best_score = float("-inf")
    best_choice: List[int] = []

    def dfs(pos: int, used_peaks: Set[str], cur_score: float, chosen: List[int]) -> None:
        nonlocal best_score, best_choice
        if cur_score + suffix_pos[pos] <= best_score + 1e-12:
            return
        if pos == n:
            if cur_score > best_score:
                best_score = cur_score
                best_choice = chosen.copy()
            return
        idx = order[pos]
        track_set = member_sets[idx]
        # branch include
        if used_peaks.isdisjoint(track_set):
            used_peaks.update(track_set)
            chosen.append(indices[idx])
            dfs(pos + 1, used_peaks, cur_score + scores[idx], chosen)
            chosen.pop()
            for pid in track_set:
                used_peaks.remove(pid)
        # branch skip
        dfs(pos + 1, used_peaks, cur_score, chosen)

    dfs(0, set(), 0.0, [])
    return best_choice


def select_tracks_via_set_packing(candidate_tracks: List[Track]) -> List[Track]:
    if not candidate_tracks:
        return []
    comps = _conflict_components(candidate_tracks)
    selected_indices: List[int] = []
    for comp in comps:
        selected_indices.extend(_solve_component(comp, candidate_tracks))
    selected = [candidate_tracks[i] for i in sorted(selected_indices)]
    return sorted(selected, key=lambda t: (t.center_ppm, -len(t.members)))
