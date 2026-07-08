from __future__ import annotations

import math
from typing import Dict, List, Tuple

from nmr_trendtrack.config import AlignConfig
from nmr_trendtrack.contracts import AlignmentCandidate, Peak, Sample
from nmr_trendtrack.align.ppm_window import get_ppm_window


def _shape_value(peak: Peak) -> float | None:
    if peak.area is None or peak.intensity is None:
        return None
    if peak.area <= 0 or peak.intensity <= 0:
        return None
    return float(peak.area / peak.intensity)


def _similarity_terms(pa: Peak, pb: Peak, win: float, cfg: AlignConfig) -> Tuple[float, float, float]:
    dppm = abs(pb.corrected_ppm() - pa.corrected_ppm())
    ppm_score = max(0.0, 1.0 - dppm / max(win, 1e-8))

    width_score = 0.5
    if pa.width_hz and pb.width_hz and pa.width_hz > 0 and pb.width_hz > 0:
        width_gap = abs(math.log(pb.width_hz / pa.width_hz))
        width_score = float(math.exp(-width_gap / 0.45))

    shape_score = 0.5
    sa = _shape_value(pa)
    sb = _shape_value(pb)
    if sa is not None and sb is not None and sa > 0 and sb > 0:
        shape_gap = abs(math.log(sb / sa))
        shape_score = float(math.exp(-shape_gap / 0.60))

    combo = ppm_score + cfg.width_similarity_weight * width_score + cfg.shape_similarity_weight * shape_score
    return combo, width_score, shape_score


def _pairwise_candidates(peaks_a_sorted: List[Peak], peaks_b_sorted: List[Peak], sa: Sample, sb: Sample, cfg: AlignConfig) -> List[AlignmentCandidate]:
    raw_rows: List[Tuple[Peak, Peak, float, float, float, float]] = []
    ia = 0
    ib = 0
    while ia < len(peaks_a_sorted) and ib < len(peaks_b_sorted):
        pa = peaks_a_sorted[ia]
        pb = peaks_b_sorted[ib]
        win = max(get_ppm_window(pa.corrected_ppm(), cfg), get_ppm_window(pb.corrected_ppm(), cfg))
        delta = pb.corrected_ppm() - pa.corrected_ppm()
        if delta < -win:
            ib += 1
            continue
        if delta > win:
            ia += 1
            continue
        jb = ib
        while jb < len(peaks_b_sorted):
            pb2 = peaks_b_sorted[jb]
            win2 = max(get_ppm_window(pa.corrected_ppm(), cfg), get_ppm_window(pb2.corrected_ppm(), cfg))
            d2 = abs(pb2.corrected_ppm() - pa.corrected_ppm())
            if d2 <= win2:
                combo, width_score, shape_score = _similarity_terms(pa, pb2, win2, cfg)
                if combo >= cfg.candidate_min_score:
                    raw_rows.append((pa, pb2, win2, d2, width_score, shape_score))
                jb += 1
            else:
                if pb2.corrected_ppm() > pa.corrected_ppm():
                    break
                jb += 1
        ia += 1

    if not raw_rows:
        return []

    best_for_a: Dict[str, float] = {}
    best_for_b: Dict[str, float] = {}
    for pa, pb, win, d2, width_score, shape_score in raw_rows:
        ppm_score = max(0.0, 1.0 - d2 / max(win, 1e-8))
        combo = ppm_score + cfg.width_similarity_weight * width_score + cfg.shape_similarity_weight * shape_score
        best_for_a[pa.peak_id] = max(best_for_a.get(pa.peak_id, float("-inf")), combo)
        best_for_b[pb.peak_id] = max(best_for_b.get(pb.peak_id, float("-inf")), combo)

    candidates: List[AlignmentCandidate] = []
    for pa, pb, win, d2, width_score, shape_score in raw_rows:
        ppm_score = max(0.0, 1.0 - d2 / max(win, 1e-8))
        combo = ppm_score + cfg.width_similarity_weight * width_score + cfg.shape_similarity_weight * shape_score
        recip = (abs(combo - best_for_a.get(pa.peak_id, combo)) < 1e-12) and (abs(combo - best_for_b.get(pb.peak_id, combo)) < 1e-12)
        final_score = combo + (cfg.reciprocal_best_bonus if recip else 0.0)
        candidates.append(
            AlignmentCandidate(
                peak_a_id=pa.peak_id,
                peak_b_id=pb.peak_id,
                sample_a=sa.sample_id,
                sample_b=sb.sample_id,
                ppm_delta_raw=abs(pb.ppm_raw - pa.ppm_raw),
                ppm_delta_corrected=d2,
                allowed_window=win,
                score_ppm=final_score,
            )
        )
    return candidates


def build_alignment_candidates(
    samples: List[Sample],
    peaks_by_sample: Dict[str, List[Peak]],
    cfg: AlignConfig,
) -> List[AlignmentCandidate]:
    ordered = sorted(samples, key=lambda s: s.order_index)
    candidates: List[AlignmentCandidate] = []
    for i, sa in enumerate(ordered):
        peaks_a_sorted = sorted(peaks_by_sample[sa.sample_id], key=lambda p: p.corrected_ppm())
        for sb in ordered[i + 1 :]:
            peaks_b_sorted = sorted(peaks_by_sample[sb.sample_id], key=lambda p: p.corrected_ppm())
            candidates.extend(_pairwise_candidates(peaks_a_sorted, peaks_b_sorted, sa, sb, cfg))
    return candidates
