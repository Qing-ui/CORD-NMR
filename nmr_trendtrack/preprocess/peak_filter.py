from __future__ import annotations

from typing import Dict, List

from nmr_trendtrack.contracts import Peak


def filter_peaks(
    peaks_by_sample: Dict[str, List[Peak]],
    min_intensity: float | None = None,
    ppm_min: float | None = None,
    ppm_max: float | None = None,
) -> Dict[str, List[Peak]]:
    out: Dict[str, List[Peak]] = {}
    for sample_id, peaks in peaks_by_sample.items():
        kept = []
        for p in peaks:
            if min_intensity is not None and p.intensity < min_intensity:
                continue
            if ppm_min is not None and p.ppm_raw < ppm_min:
                continue
            if ppm_max is not None and p.ppm_raw > ppm_max:
                continue
            kept.append(p)
        out[sample_id] = kept
    return out
