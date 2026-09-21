from __future__ import annotations

from typing import Dict, List

import numpy as np

from nmr_trendtrack.contracts import Track
from nmr_trendtrack.preprocess.intensity_clean import choose_signal_value


_IDENTITY_METHODS = {"none", "raw", "identity", "off"}
_ROBUST_METHODS = {"median_ratio", "trimmed_median_ratio", "winsorized_median_ratio"}


def _identity_scales(ordered_sample_ids: List[str]) -> Dict[str, float]:
    return {sid: 1.0 for sid in ordered_sample_ids}


def estimate_sample_scales(
    tracks: List[Track],
    ordered_sample_ids: List[str],
    method: str = "none",
    use_area: bool = False,
) -> Dict[str, float]:
    if not ordered_sample_ids:
        return {}
    method_key = str(method or "none").strip().lower()
    if method_key in _IDENTITY_METHODS:
        return _identity_scales(ordered_sample_ids)

    ref = ordered_sample_ids[0]
    scales = {ref: 1.0}
    ref_tracks = [tr for tr in tracks if ref in tr.members]
    for sid in ordered_sample_ids[1:]:
        ratios = []
        for tr in ref_tracks:
            if sid not in tr.members:
                continue
            a = choose_signal_value(tr.members[ref], use_area)
            b = choose_signal_value(tr.members[sid], use_area)
            if a > 0 and b > 0:
                ratios.append(b / a)
        if ratios:
            vals = np.asarray(ratios, dtype=float)
            if method_key == "trimmed_median_ratio":
                lo = np.quantile(vals, 0.10)
                hi = np.quantile(vals, 0.90)
                vals = vals[(vals >= lo) & (vals <= hi)]
            elif method_key == "winsorized_median_ratio":
                lo = np.quantile(vals, 0.10)
                hi = np.quantile(vals, 0.90)
                vals = np.clip(vals, lo, hi)
            scale = float(np.median(vals)) if len(vals) else 1.0
            scales[sid] = scale if scale > 0 else 1.0
        else:
            scales[sid] = 1.0
    return scales


def normalize_track_intensities(
    tracks: List[Track],
    sample_scales: Dict[str, float],
    use_area: bool = False,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tr in tracks:
        out[tr.track_id] = {}
        for sid, peak in tr.members.items():
            scale = sample_scales.get(sid, 1.0)
            value = choose_signal_value(peak, use_area)
            out[tr.track_id][sid] = value / max(scale, 1e-8)
    return out
