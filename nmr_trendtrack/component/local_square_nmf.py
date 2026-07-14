from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import numpy as np
from sklearn.decomposition import NMF
from nmr_trendtrack.contracts import Peak
from nmr_trendtrack.preprocess.intensity_clean import choose_signal_value

@dataclass
class LocalSquareComponentModel:
    ordered_sample_ids: List[str]
    grid_ppm: np.ndarray
    loadings: np.ndarray
    profiles: np.ndarray
    rel_error: float


def _peak_sigma_ppm(peak: Peak) -> float:
    if peak.width_hz is None or peak.width_hz <= 0:
        return 0.05
    return float(min(0.12, max(0.03, 0.028 * float(peak.width_hz))))


def _build_local_spectrum_matrix(peaks_by_sample: Dict[str, List[Peak]], ordered_sample_ids: List[str], sample_scales: Optional[Dict[str, float]], grid_step_ppm: float = 0.03, margin_ppm: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    all_peaks = [p for sid in ordered_sample_ids for p in peaks_by_sample.get(sid, [])]
    if not all_peaks:
        return np.zeros((len(ordered_sample_ids), 0), dtype=float), np.zeros((0,), dtype=float)
    lo = min(p.corrected_ppm() for p in all_peaks) - margin_ppm
    hi = max(p.corrected_ppm() for p in all_peaks) + margin_ppm
    n_grid = max(24, int(math.ceil((hi - lo) / grid_step_ppm)) + 1)
    grid = np.linspace(lo, hi, n_grid)
    X = np.zeros((len(ordered_sample_ids), n_grid), dtype=float)
    scales = sample_scales or {}
    for i, sid in enumerate(ordered_sample_ids):
        row = np.zeros(n_grid, dtype=float)
        for peak in peaks_by_sample.get(sid, []):
            amp = choose_signal_value(peak, use_area=False) / max(scales.get(sid, 1.0), 1e-8)
            if amp <= 0:
                continue
            sigma = _peak_sigma_ppm(peak)
            d = (grid - peak.corrected_ppm()) / sigma
            row += float(amp) * np.exp(-0.5 * d * d)
        mx = float(np.max(row)) if row.size else 0.0
        if mx > 0:
            row /= mx
        X[i] = row
    return X, grid


def fit_local_square_component_model(peaks_by_sample: Dict[str, List[Peak]], ordered_sample_ids: List[str], sample_scales: Optional[Dict[str, float]], max_components: int = 3) -> Optional[LocalSquareComponentModel]:
    nonempty = [sid for sid in ordered_sample_ids if peaks_by_sample.get(sid)]
    if len(nonempty) < 2:
        return None
    X, grid = _build_local_spectrum_matrix(peaks_by_sample, ordered_sample_ids, sample_scales)
    if X.shape[1] < 8 or float(np.max(X)) <= 0:
        return None
    X_sq = np.square(np.clip(X, 0.0, None))
    denom = float(np.linalg.norm(X_sq))
    if denom <= 1e-10:
        return None
    n_samples = X_sq.shape[0]
    max_k = max(1, min(max_components, n_samples, max(1, len(nonempty))))
    best = None
    best_score = float('inf')
    for k in range(1, max_k + 1):
        try:
            nmf = NMF(n_components=k, init='nndsvda', solver='cd', beta_loss='frobenius', max_iter=600, tol=1e-4, random_state=0)
            W = nmf.fit_transform(X_sq)
            H = nmf.components_
        except Exception:
            continue
        rel_err = float(np.linalg.norm(X_sq - W @ H) / denom)
        score = rel_err + 0.035 * k
        if score < best_score - 1e-8:
            best_score = score
            best = LocalSquareComponentModel(list(ordered_sample_ids), grid, W.copy(), H.copy(), rel_err)
    return best


def _interp_profile(profile: np.ndarray, grid_ppm: np.ndarray, ppm: float) -> float:
    if profile.size == 0:
        return 0.0
    mx = float(np.max(profile))
    if mx <= 1e-10:
        return 0.0
    return max(0.0, float(np.interp(ppm, grid_ppm, profile, left=0.0, right=0.0)) / mx)


def score_track_against_components(members: Dict[str, Peak], model: Optional[LocalSquareComponentModel], sample_scales: Optional[Dict[str, float]]) -> float:
    if model is None or not members:
        return 0.0
    scales = sample_scales or {}
    y = []
    mask = []
    for sid in model.ordered_sample_ids:
        peak = members.get(sid)
        if peak is None:
            y.append(0.0); mask.append(0.0)
        else:
            y.append(float(max(choose_signal_value(peak, use_area=False) / max(scales.get(sid, 1.0), 1e-8), 0.0))); mask.append(1.0)
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=float)
    if float(np.max(y)) <= 0:
        return 0.0
    y /= max(float(np.linalg.norm(y)), 1e-8)
    best = 0.0
    for k in range(model.loadings.shape[1]):
        w = np.asarray(model.loadings[:, k], dtype=float)
        if float(np.max(w)) <= 0:
            continue
        w = w * np.where(mask > 0, 1.0, 0.65)
        w /= max(float(np.linalg.norm(w)), 1e-8)
        cosine = float(np.dot(y, w))
        profile = np.asarray(model.profiles[k], dtype=float)
        vals = [_interp_profile(profile, model.grid_ppm, p.corrected_ppm()) for p in members.values()]
        if not vals:
            continue
        affinity = 0.7 * cosine + 0.2 * float(np.mean(vals)) + 0.1 * float(np.min(vals))
        if affinity > best:
            best = affinity
    return max(0.0, best)
