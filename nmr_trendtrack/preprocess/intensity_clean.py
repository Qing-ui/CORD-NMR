from __future__ import annotations

from typing import Dict, List

from nmr_trendtrack.contracts import Peak


def choose_signal_value(peak: Peak, use_area: bool = False) -> float:
    if use_area and peak.area is not None:
        return float(peak.area)
    return float(peak.intensity)


def sanitize_nonpositive(values: Dict[str, Dict[str, float]], floor: float = 1e-8) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, sub in values.items():
        out[key] = {k: (v if v > floor else floor) for k, v in sub.items()}
    return out
