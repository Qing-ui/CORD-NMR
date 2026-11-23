"""Enumerate strictly contiguous carbon peak trajectories."""

from __future__ import annotations

import heapq
import math
from statistics import median

import numpy as np


def enumerate_strict_trajectories(
    alignment, rows, samples, *, residual_gate, top_k_per_seed, max_per_sample, min_samples=2
):
    """Keep the best candidates per starting peak, plus singleton fallbacks."""
    if top_k_per_seed < 1 or max_per_sample < 1 or residual_gate <= 0:
        raise ValueError("Candidate limits and corrected shift gate must be positive.")

    peaks_by_sample = alignment.group_by_sample(rows)
    sample_index = {sample: i for i, sample in enumerate(samples)}
    candidates = {}
    n_samples = len(samples)
    sigma = max(0.025, residual_gate / 2.5)
    raw_limit = float(alignment.RAW_SPAN_LIMIT)
    corrected_limit = residual_gate * 1.30
    grid_step = 0.01
    # The Gaussian's maximum derivative bounds the error between grid points.
    grid_margin = math.exp(-0.5) / sigma * grid_step / 2.0

    for seed in rows:
        seed_index = sample_index[seed["sample"]]
        choices = alignment.build_candidates_for_peak(
            seed, peaks_by_sample, samples,
            residual_gate=residual_gate, max_per_sample=max_per_sample,
        )
        peak_options = [[p for p in options if p is not None] for options in choices]
        leftmost = rightmost = seed_index
        while leftmost > 0 and peak_options[leftmost - 1]:
            leftmost -= 1
        while rightmost + 1 < n_samples and peak_options[rightmost + 1]:
            rightmost += 1

        intervals = [
            (left, right)
            for left in range(leftmost, seed_index + 1)
            for right in range(seed_index, rightmost + 1)
            if right - left + 1 >= min_samples
        ]
        intervals.sort(
            key=lambda interval: (
                interval[1] - interval[0] + 1,
                -abs((interval[0] + interval[1]) / 2 - seed_index),
            ),
            reverse=True,
        )
        if not intervals:
            continue

        seed_corr = float(seed["ppm_corr"])
        seed_raw = float(seed["ppm"])
        grid = np.arange(
            seed_corr - residual_gate, seed_corr + residual_gate + grid_step * 0.5,
            grid_step, dtype=float,
        )
        gaussian = {
            str(p["peak_id"]): np.exp(-0.5 * ((grid - float(p["ppm_corr"])) / sigma) ** 2)
            for options in peak_options[leftmost : rightmost + 1]
            for p in options
        }
        best_by_sample = {
            i: np.maximum.reduce([gaussian[str(p["peak_id"])] for p in peak_options[i]])
            for i in range(leftmost, rightmost + 1) if i != seed_index
        }
        best = []
        seen = set()

        for left, right in intervals:
            indices = [i for i in range(left, right + 1) if i != seed_index]
            indices.sort(key=lambda i: (len(peak_options[i]), i))
            size = right - left + 1
            coverage = 0.25 * size / n_samples
            suffix = [np.zeros(len(grid), dtype=float) for _ in range(len(indices) + 1)]
            for pos in range(len(indices) - 1, -1, -1):
                suffix[pos] = suffix[pos + 1] + best_by_sample[indices[pos]]

            def search(pos, selected, fixed, raw_min, raw_max, corr_min, corr_max):
                if len(best) == top_k_per_seed:
                    penalty = (
                        max(0.0, (raw_max - raw_min - raw_limit) * 4.0)
                        + max(0.0, (corr_max - corr_min - residual_gate) * 5.0)
                    )
                    bound = float(np.max((fixed + suffix[pos]) / size)) + grid_margin + coverage - penalty
                    if bound < best[0][0] - 1e-13:
                        return
                if pos == len(indices):
                    ids = tuple(sorted(str(p["peak_id"]) for p in selected))
                    if ids in seen:
                        return
                    seen.add(ids)
                    members = {p["sample"]: p for p in selected}
                    score = float(alignment.score_track(members, samples, residual_gate=residual_gate))
                    item = (score, len(ids), ids, members)
                    if len(best) < top_k_per_seed:
                        heapq.heappush(best, item)
                    elif item[:3] > best[0][:3]:
                        heapq.heapreplace(best, item)
                    return

                center = median(float(p["ppm_corr"]) for p in selected)
                options = sorted(
                    peak_options[indices[pos]],
                    key=lambda p: (abs(float(p["ppm_corr"]) - center), str(p["peak_id"])),
                )
                for peak in options:
                    raw = float(peak["ppm"])
                    corr = float(peak["ppm_corr"])
                    lo_raw, hi_raw = min(raw_min, raw), max(raw_max, raw)
                    lo_corr, hi_corr = min(corr_min, corr), max(corr_max, corr)
                    if hi_raw - lo_raw > raw_limit or hi_corr - lo_corr > corrected_limit:
                        continue
                    search(
                        pos + 1, selected + [peak], fixed + gaussian[str(peak["peak_id"])],
                        lo_raw, hi_raw, lo_corr, hi_corr,
                    )

            search(0, [seed], gaussian[str(seed["peak_id"])], seed_raw, seed_raw, seed_corr, seed_corr)

        for score, _, ids, members in sorted(best, key=lambda item: item[:3], reverse=True):
            if ids not in candidates or score > candidates[ids]["score"]:
                candidates[ids] = {"member_ids": ids, "members": members, "score": score, "kind": "multi"}

    for peak in rows:
        ids = (peak["peak_id"],)
        candidates.setdefault(
            ids,
            {"member_ids": ids, "members": {peak["sample"]: peak}, "score": 0.15, "kind": "singleton"},
        )
    return [dict(track, track_id=f"CT{i:05d}") for i, track in enumerate(candidates.values(), 1)]
