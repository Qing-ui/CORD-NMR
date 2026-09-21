from __future__ import annotations

import math
from typing import Dict, List

from nmr_trendtrack.contracts import Track, TrendVector


def compute_trend_vector(
    track: Track,
    ordered_sample_ids: List[str],
    normalized_intensity: Dict[str, Dict[str, float]],
    eps: float,
) -> TrendVector:
    vals = [normalized_intensity.get(track.track_id, {}).get(sid) for sid in ordered_sample_ids]
    step_log_fc = []
    valid_steps = []
    for i in range(len(vals) - 1):
        a, b = vals[i], vals[i + 1]
        if a is None or b is None:
            step_log_fc.append(None)
            valid_steps.append(False)
        else:
            step_log_fc.append(float(math.log((b + eps) / (a + eps))))
            valid_steps.append(True)
    return TrendVector(
        track_id=track.track_id,
        presence_mask=track.presence_mask,
        step_log_fc=step_log_fc,
        valid_steps=valid_steps,
    )
