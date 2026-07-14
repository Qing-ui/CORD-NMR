#!/usr/bin/env python3
"""
Best-3 13C cross-spectrum clustering implementation.

This module implements the third / best tested track-selection strategy:
    conflict-component global set-packing

Pipeline:
    templates + scenario definitions
    -> simulated multi-sample peak lists
    -> regional shift correction
    -> top-K candidate track generation
    -> conflict graph decomposition
    -> exact set-packing for small conflict components
       beam-search set-packing for large conflict components
    -> lightweight joint trend/profile clustering
    -> 1 ppm recovery evaluation

No truth labels are used for candidate selection or clustering. Truth labels are
only used in simulated tests for evaluation output.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import argparse
from collections import defaultdict, Counter
from statistics import median
from typing import Dict, List, Tuple, Iterable, Any

# Import local utility/base implementation. It includes simulation, shift
# correction, candidate generation, track distances, diagnostics, etc.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import importlib.util
_base_path = os.path.join(THIS_DIR, "13c_joint_align_cluster.py")
_spec = importlib.util.spec_from_file_location("base13c", _base_path)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

COMPS = base.COMPS


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


def parse_scenario_definitions(path: str) -> List[Tuple[str, List[Tuple[str, List[str]]]]]:
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            combos = []
            for part in r["combo_text"].split(";"):
                sample, comps = part.strip().split(":", 1)
                combos.append((sample.strip(), [x.strip() for x in comps.split("+") if x.strip()]))
            out.append((r["scenario"], combos))
    return out


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


def exact_component(candidate_tracks: List[dict], idxs: List[int], node_limit: int = 100000):
    """Branch-and-bound exact weighted set-packing for one small conflict component."""
    idxs = sorted(idxs, key=lambda i: track_weight(candidate_tracks[i]), reverse=True)
    weights = [track_weight(candidate_tracks[i]) for i in idxs]

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


def beam_component(candidate_tracks: List[dict], idxs: List[int], beam_width: int = 120) -> List[int]:
    """Beam-search approximate weighted set-packing for one large conflict component."""
    idxs = sorted(idxs, key=lambda i: track_weight(candidate_tracks[i]), reverse=True)
    states = [(0.0, tuple(), frozenset())]  # score, selected indices, used peak ids

    for i in idxs:
        ids = frozenset(candidate_tracks[i]["member_ids"])
        w = track_weight(candidate_tracks[i])
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


def _sample_sort_key(sample_id: str):
    text = str(sample_id)
    i = len(text)
    while i > 0 and text[i - 1].isdigit():
        i -= 1
    suffix = text[i:]
    return (text[:i], int(suffix) if suffix else -1, text)


def _track_presence_mask(track: dict, samples: List[str]) -> str:
    members = track.get("members", {}) or {}
    return "".join("1" if members.get(sid) else "0" for sid in samples)


def mask_stratified_global_set_packing(
    candidate_tracks: List[dict],
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
):
    """Set-packing independently inside each presence mask.

    Tracks with the same mask still compete for shared raw peaks, so each mask
    gets a non-conflicting best explanation.  Different masks are allowed to
    reuse the same raw peak; this preserves alternative explanations such as a
    complete 111 trajectory and its lower-mask sub-trajectories for downstream
    mask display or quality-guarded selection.
    """
    if not candidate_tracks:
        return [], {
            "mask_stratified_pack": True,
            "n_mask_buckets": 0,
            "n_selected_tracks_all_masks": 0,
        }

    samples = sorted(
        {
            sid
            for track in candidate_tracks
            for sid in (track.get("members", {}) or {}).keys()
        },
        key=_sample_sort_key,
    )
    by_mask = defaultdict(list)
    for track in candidate_tracks:
        by_mask[_track_presence_mask(track, samples)].append(track)

    selected: List[dict] = []
    mode_counts = Counter()
    info = {
        "mask_stratified_pack": True,
        "n_mask_buckets": len(by_mask),
        "mask_candidate_counts": ",".join(
            f"{mask}:{len(group)}" for mask, group in sorted(by_mask.items(), reverse=True)
        ),
    }

    for mask, group in sorted(by_mask.items(), reverse=True):
        sel, subinfo = componentwise_global_set_packing(
            group,
            exact_limit=exact_limit,
            beam_width=beam_width,
            node_limit=node_limit,
        )
        selected.extend(sel)
        info[f"mask_{mask}_candidates"] = len(group)
        info[f"mask_{mask}_selected"] = len(sel)
        for key, value in subinfo.items():
            if key.startswith("solve_"):
                mode_counts[f"{mask}_{key[6:]}"] += int(value)

    info["n_selected_tracks_all_masks"] = len(selected)
    info.update({f"mask_solve_{key}": value for key, value in mode_counts.items()})
    return relabel(selected), info


def componentwise_global_set_packing(
    candidate_tracks: List[dict],
    exact_limit: int = 12,
    beam_width: int = 120,
    node_limit: int = 100000,
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
            sel, mode = exact_component(candidate_tracks, comp, node_limit=node_limit)
            if sel is None:
                sel = beam_component(candidate_tracks, comp, beam_width=beam_width)
                mode = "beam_timeout"
        else:
            sel = beam_component(candidate_tracks, comp, beam_width=beam_width)
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

    This is the same clustering backend used in the last comparison for fair
    evaluation. It combines:
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


def track_center_corr(track: dict):
    vals = [p.get("ppm_corr", p["ppm"]) for p in track["members"].values()]
    return median(vals) if vals else None


def component_recovery_1ppm(tracks: List[dict], labels: List[str], templates: dict, combos):
    """Corrected 1 ppm recovery evaluation.

    1 ppm is used only for target atom recovery in the best cluster. It is not
    used for clustering and does not add non-target cross-hits into the target
    ratio denominator.
    """
    sample_comps = {s: set(c) for s, c in combos}
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)

    recalls = []
    ratios = []
    compressions = []
    good = 0
    details = []

    for comp in COMPS:
        atoms = templates[comp]
        best = None

        for lab, idxs in groups.items():
            centers = [track_center_corr(tracks[i]) for i in idxs]
            recovered = sum(
                1
                for atom in atoms
                if any(c is not None and abs(c - atom["ppm"]) <= 1.0 for c in centers)
            )
            target_ratio = recovered / max(len(idxs), 1)
            if best is None or (recovered, target_ratio) > (best[0], best[1]):
                best = (recovered, target_ratio, idxs, lab)

        recovered, target_ratio, idxs, lab = best if best else (0, 0.0, [], "")
        recall = recovered / max(len(atoms), 1)

        true_comps = set()
        for i in idxs:
            true_comps.update(p["component"] for p in tracks[i]["members"].values())

        avg_mix = sum(len(sample_comps[s]) for s in sample_comps if comp in sample_comps[s]) / max(
            1, sum(1 for s in sample_comps if comp in sample_comps[s])
        )
        compression = avg_mix / max(1, len(true_comps))

        status = (
            "good"
            if recall >= 0.8 and target_ratio >= 0.7
            else ("mixed" if recall >= 0.8 else ("fragmented" if target_ratio >= 0.7 else "poor"))
        )
        if status == "good":
            good += 1

        recalls.append(recall)
        ratios.append(target_ratio)
        compressions.append(compression)
        details.append(
            {
                "component": comp,
                "best_cluster": lab,
                "recovered_atoms_1ppm": recovered,
                "total_atoms": len(atoms),
                "recall_1ppm": round(recall, 4),
                "target_ratio": round(target_ratio, 4),
                "cluster_n_tracks": len(idxs),
                "cluster_true_components": "+".join(sorted(true_comps)),
                "compression": round(compression, 4),
                "status": status,
            }
        )

    return (
        {
            "mean_recall_1ppm": round(sum(recalls) / len(recalls), 4),
            "mean_target_ratio": round(sum(ratios) / len(ratios), 4),
            "mean_compression": round(sum(compressions) / len(compressions), 4),
            "n_good_components": good,
        },
        details,
    )


def run_scenario(templates: dict, name: str, combos, outdir: str, seed: int, cfg: dict):
    samples = [s for s, _ in combos]

    weights = base.concentration_curves(combos, seed=seed, near=("near" in name))
    shifts = base.make_regional_shifts(samples, seed=seed + 11, stress=True)
    rows = base.simulate_peaklists(templates, weights, shifts, seed=seed + 101)
    corrected, shift_model, shift_diag = base.apply_shift(rows, samples)

    candidate_tracks = base.enumerate_candidate_tracks(
        corrected,
        samples,
        residual_gate=cfg.get("residual_gate", 0.18),
        top_k_per_seed=cfg.get("top_k_per_seed", 8),
        max_per_sample=cfg.get("max_per_sample_candidates", 2),
        min_samples=2,
    )

    selected_tracks, pack_info = componentwise_global_set_packing(
        candidate_tracks,
        exact_limit=cfg.get("exact_limit", 12),
        beam_width=cfg.get("beam_width", 120),
        node_limit=cfg.get("node_limit", 100000),
    )

    labels, cluster_diag = light_joint_cluster(selected_tracks, samples, cfg)
    eval_metrics = base.evaluate_tracks_and_clusters(selected_tracks, labels)
    recovery, recovery_details = component_recovery_1ppm(selected_tracks, labels, templates, combos)

    summary = {
        "scenario": name,
        "n_samples": len(samples),
        "method": "componentwise_global_set_packing_best3",
        "n_input_peaks": len(rows),
        "n_candidate_tracks": len(candidate_tracks),
    }
    summary.update(eval_metrics)
    summary.update(recovery)
    summary.update(cluster_diag)
    summary.update(pack_info)

    scen_dir = os.path.join(outdir, name)
    os.makedirs(scen_dir, exist_ok=True)
    base.write_csv(
        os.path.join(scen_dir, "mixing_matrix.csv"),
        [{"sample": s, **{c: weights[s].get(c, 0) for c in COMPS}} for s in samples],
    )
    base.write_csv(os.path.join(scen_dir, "peaklists_corrected.csv"), corrected)
    base.write_csv(os.path.join(scen_dir, "shift_diagnostics.csv"), shift_diag)
    base.write_csv(os.path.join(scen_dir, "tracks.csv"), base.selected_to_rows(selected_tracks, samples, labels))
    base.write_csv(os.path.join(scen_dir, "cluster_diagnostics.csv"), base.cluster_diagnostics(selected_tracks, labels, samples))
    base.write_csv(os.path.join(scen_dir, "component_recovery_1ppm.csv"), recovery_details)
    base.write_csv(os.path.join(scen_dir, "scenario_summary.csv"), [summary])

    return summary


def aggregate(rows: List[dict]) -> dict:
    if not rows:
        return {}
    def avg(k):
        return round(sum(float(r.get(k, 0) or 0) for r in rows) / len(rows), 4)

    return {
        "method": "componentwise_global_set_packing_best3",
        "n_scenarios": len(rows),
        "avg_n_clusters": avg("n_clusters"),
        "avg_mixed_tracks": avg("mixed_tracks"),
        "avg_cluster_ari": avg("cluster_ari"),
        "avg_pair_f1": avg("pair_f1"),
        "avg_recall_1ppm": avg("mean_recall_1ppm"),
        "avg_target_ratio": avg("mean_target_ratio"),
        "avg_compression": avg("mean_compression"),
        "total_good_components": sum(int(r.get("n_good_components", 0)) for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-csv", default=os.path.join(os.path.dirname(THIS_DIR), "data", "standard_templates.csv"))
    ap.add_argument("--scenario-csv", default=os.path.join(os.path.dirname(THIS_DIR), "data", "scenario_definitions.csv"))
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(THIS_DIR), "results", "best3_final_tests"))
    ap.add_argument("--scenarios", default="01_N3_all3,05_N3_bridge,08_N4_all4",
                    help="Comma-separated scenario names, or 'all'.")
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--residual-gate", type=float, default=0.18)
    ap.add_argument("--top-k-per-seed", type=int, default=8)
    ap.add_argument("--max-per-sample-candidates", type=int, default=2)
    ap.add_argument("--exact-limit", type=int, default=12)
    ap.add_argument("--beam-width", type=int, default=120)
    ap.add_argument("--mask-weight", type=float, default=0.38)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    templates = base.load_templates(args.template_csv)
    scenarios = parse_scenario_definitions(args.scenario_csv)
    wanted = None if args.scenarios.strip().lower() == "all" else set(x.strip() for x in args.scenarios.split(",") if x.strip())

    cfg = {
        "residual_gate": args.residual_gate,
        "top_k_per_seed": args.top_k_per_seed,
        "max_per_sample_candidates": args.max_per_sample_candidates,
        "exact_limit": args.exact_limit,
        "beam_width": args.beam_width,
        "mask_weight": args.mask_weight,
    }

    rows = []
    for idx, (name, combos) in enumerate(scenarios):
        if wanted is not None and name not in wanted:
            continue
        t0 = time.time()
        row = run_scenario(templates, name, combos, args.outdir, seed=args.seed + idx * 137, cfg=cfg)
        row["runtime_s"] = round(time.time() - t0, 3)
        rows.append(row)
        print(f"[done] {name}: k={row.get('n_clusters')} mixed={row.get('mixed_tracks')} "
              f"recall={row.get('mean_recall_1ppm')} target_ratio={row.get('mean_target_ratio')} "
              f"runtime={row['runtime_s']}s", flush=True)

    base.write_csv(os.path.join(args.outdir, "best3_final_summary.csv"), rows)
    base.write_csv(os.path.join(args.outdir, "best3_aggregate_summary.csv"), [aggregate(rows)])
    print(json.dumps({"outdir": args.outdir, "aggregate": aggregate(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
