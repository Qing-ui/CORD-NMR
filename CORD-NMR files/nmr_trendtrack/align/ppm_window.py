from __future__ import annotations

from nmr_trendtrack.config import AlignConfig


def get_ppm_window(ppm: float, cfg: AlignConfig) -> float:
    for lo, hi, win in cfg.ppm_window_by_region:
        if lo <= ppm < hi:
            return float(win)
    return float(cfg.ppm_window_default)
