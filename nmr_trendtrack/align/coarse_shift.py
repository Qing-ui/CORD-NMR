from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Tuple

import numpy as np

from nmr_trendtrack.config import AlignConfig
from nmr_trendtrack.contracts import Peak, Sample, Track


def _weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    wts = np.asarray(list(weights), dtype=float)
    if vals.size == 0:
        return 0.0
    if np.all(wts <= 0):
        return float(np.median(vals))
    order = np.argsort(vals)
    vals = vals[order]
    wts = np.maximum(wts[order], 0.0)
    cum = np.cumsum(wts)
    cutoff = cum[-1] / 2.0
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    return float(vals[min(idx, len(vals) - 1)])


def _sorted_peaks(peaks: List[Peak]) -> List[Peak]:
    return sorted(peaks, key=lambda p: p.ppm_raw)


def _match_peaks_monotone(ref_peaks: List[Peak], qry_peaks: List[Peak], max_delta: float) -> List[Tuple[Peak, Peak]]:
    ref = _sorted_peaks(ref_peaks)
    qry = _sorted_peaks(qry_peaks)
    n, m = len(qry), len(ref)
    gap = -0.35
    dp = np.full((n + 1, m + 1), -1e18, dtype=float)
    back = np.full((n + 1, m + 1), 0, dtype=int)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap
        back[i, 0] = 2
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap
        back[0, j] = 3
    for i in range(1, n + 1):
        qi = qry[i - 1]
        for j in range(1, m + 1):
            rj = ref[j - 1]
            best = dp[i - 1, j] + gap
            code = 2
            left = dp[i, j - 1] + gap
            if left > best:
                best = left
                code = 3
            d = abs(qi.ppm_raw - rj.ppm_raw)
            if d <= max_delta:
                prox = 2.0 - (d / max_delta)
                match = dp[i - 1, j - 1] + prox + 0.05
                if match > best:
                    best = match
                    code = 1
            dp[i, j] = best
            back[i, j] = code
    i, j = n, m
    pairs: List[Tuple[Peak, Peak]] = []
    while i > 0 or j > 0:
        code = int(back[i, j])
        if code == 1:
            qi = qry[i - 1]
            rj = ref[j - 1]
            if abs(qi.ppm_raw - rj.ppm_raw) <= max_delta:
                pairs.append((qi, rj))
            i -= 1
            j -= 1
        elif code == 2:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _build_anchor_map(matches: List[Tuple[Peak, Peak]]) -> Tuple[List[Tuple[float, float]], float]:
    if not matches:
        return [], 0.0
    anchors = [(q.ppm_raw, r.ppm_raw - q.ppm_raw) for q, r in matches]
    anchors.sort(key=lambda x: x[0])
    vals = [d for _, d in anchors]
    summary = _weighted_median(vals, [1.0] * len(vals))
    collapsed: List[Tuple[float, float]] = []
    cur_x, cur_vals = anchors[0][0], [anchors[0][1]]
    for x, d in anchors[1:]:
        if abs(x - cur_x) <= 0.05:
            cur_vals.append(d)
        else:
            collapsed.append((cur_x, float(np.median(cur_vals))))
            cur_x, cur_vals = x, [d]
    collapsed.append((cur_x, float(np.median(cur_vals))))
    return collapsed, float(summary)


def _interp_shift(ppm: float, anchors: List[Tuple[float, float]], default_shift: float) -> float:
    if not anchors:
        return default_shift
    if len(anchors) == 1:
        return anchors[0][1]
    xs = [x for x, _ in anchors]
    ys = [y for _, y in anchors]
    if ppm <= xs[0]:
        return ys[0]
    if ppm >= xs[-1]:
        return ys[-1]
    idx = int(np.searchsorted(xs, ppm, side="right"))
    x0, y0 = xs[idx - 1], ys[idx - 1]
    x1, y1 = xs[idx], ys[idx]
    if abs(x1 - x0) < 1e-8:
        return y0
    t = (ppm - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


def estimate_warp_maps(samples: List[Sample], peaks_by_sample: Dict[str, List[Peak]], cfg: AlignConfig) -> Tuple[Dict[str, float], Dict[str, List[Tuple[float, float]]]]:
    ref_sample_id = sorted(samples, key=lambda s: s.order_index)[0].sample_id
    ref_peaks = peaks_by_sample[ref_sample_id]
    shifts: Dict[str, float] = {ref_sample_id: 0.0}
    warp_maps: Dict[str, List[Tuple[float, float]]] = {ref_sample_id: []}
    for sample in samples:
        sid = sample.sample_id
        if sid == ref_sample_id:
            continue
        matches = _match_peaks_monotone(ref_peaks, peaks_by_sample[sid], cfg.coarse_match_window)
        anchors, summary = _build_anchor_map(matches)
        shifts[sid] = summary
        warp_maps[sid] = anchors
    return shifts, warp_maps


def estimate_global_shifts(samples: List[Sample], peaks_by_sample: Dict[str, List[Peak]], cfg: AlignConfig) -> Dict[str, float]:
    shifts, _ = estimate_warp_maps(samples, peaks_by_sample, cfg)
    return shifts


def apply_warp_maps(peaks_by_sample: Dict[str, List[Peak]], warp_maps: Dict[str, List[Tuple[float, float]]], shifts: Dict[str, float]) -> Dict[str, List[Peak]]:
    out: Dict[str, List[Peak]] = {}
    for sample_id, peaks in peaks_by_sample.items():
        default_shift = float(shifts.get(sample_id, 0.0))
        anchors = warp_maps.get(sample_id, [])
        corrected = []
        for p in peaks:
            new_p = deepcopy(p)
            new_p.ppm_corr = p.ppm_raw + _interp_shift(p.ppm_raw, anchors, default_shift)
            corrected.append(new_p)
        corrected.sort(key=lambda x: x.corrected_ppm())
        out[sample_id] = corrected
    return out


def apply_global_shifts(peaks_by_sample: Dict[str, List[Peak]], shifts: Dict[str, float]) -> Dict[str, List[Peak]]:
    return apply_warp_maps(peaks_by_sample, {}, shifts)


def update_warp_maps_from_tracks(samples: List[Sample], peaks_original: Dict[str, List[Peak]], tracks: List[Track], memberships_by_track: Dict[str, str], old_warp_maps: Dict[str, List[Tuple[float, float]]], old_shifts: Dict[str, float], cfg: AlignConfig) -> Tuple[Dict[str, float], Dict[str, List[Tuple[float, float]]]]:
    ordered_samples = sorted(samples, key=lambda s: s.order_index)
    ref_sample_id = ordered_samples[0].sample_id
    peak_lookup = {p.peak_id: p for peaks in peaks_original.values() for p in peaks}
    new_shifts = dict(old_shifts)
    new_maps = {sid: list(anchors) for sid, anchors in old_warp_maps.items()}
    for sample in ordered_samples:
        sid = sample.sample_id
        if sid == ref_sample_id:
            new_shifts[sid] = 0.0
            new_maps[sid] = []
            continue
        anchors = []
        for tr in tracks:
            if sid not in tr.members or ref_sample_id not in tr.members:
                continue
            if memberships_by_track.get(tr.track_id) not in {"pure", "shared"}:
                continue
            raw_q = peak_lookup[tr.members[sid].peak_id].ppm_raw
            raw_r = peak_lookup[tr.members[ref_sample_id].peak_id].ppm_raw
            anchors.append((raw_q, raw_r - raw_q))
        if not anchors:
            continue
        anchors.sort(key=lambda x: x[0])
        xs = sorted({x for x, _ in anchors} | {x for x, _ in old_warp_maps.get(sid, [])})
        blended = []
        for x in xs:
            target_vals = [d for xp, d in anchors if abs(xp - x) <= 0.05]
            target = float(np.median(target_vals)) if target_vals else _interp_shift(x, anchors, old_shifts.get(sid, 0.0))
            current = _interp_shift(x, old_warp_maps.get(sid, []), old_shifts.get(sid, 0.0))
            delta = float(np.clip(target - current, -cfg.shift_step_limit, cfg.shift_step_limit))
            blended.append((x, current + delta))
        new_maps[sid] = blended
        new_shifts[sid] = float(np.median([d for _, d in blended])) if blended else old_shifts.get(sid, 0.0)
    return new_shifts, new_maps


def update_global_shifts_from_tracks(samples: List[Sample], peaks_original: Dict[str, List[Peak]], tracks: List[Track], memberships_by_track: Dict[str, str], old_shifts: Dict[str, float], cfg: AlignConfig) -> Dict[str, float]:
    new_shifts, _ = update_warp_maps_from_tracks(samples, peaks_original, tracks, memberships_by_track, {}, old_shifts, cfg)
    return new_shifts
