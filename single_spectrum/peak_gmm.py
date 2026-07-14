from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


@dataclass
class PeakGMMIntensityAutoClusterer:
    min_components: int = 1
    max_components: int = 4
    use_log: bool = True
    random_state: int = 42

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        x = np.clip(x, 1e-8, None)
        if self.use_log:
            x = np.log(x)
        return x

    def fit_predict(self, intensities: np.ndarray) -> tuple[np.ndarray, GaussianMixture]:
        X = self._prepare(intensities)
        best_model = None
        best_bic = np.inf
        max_k = min(self.max_components, len(X))
        for k in range(max(1, self.min_components), max_k + 1):
            model = GaussianMixture(n_components=k, random_state=self.random_state)
            model.fit(X)
            bic = model.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_model = model
        if best_model is None:
            raise RuntimeError('Failed to fit GMM model.')
        labels = best_model.predict(X)
        means = best_model.means_.ravel()
        order = np.argsort(means)
        remap = {old: new for new, old in enumerate(order)}
        labels = np.array([remap[int(v)] for v in labels], dtype=int)
        return labels, best_model


def run_peakgmm(
    df: pd.DataFrame,
    min_components: int = 1,
    max_components: int = 4,
    use_log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if 'height_corr' in work.columns:
        work['intensity'] = work['height_corr']
    elif 'height' in work.columns and 'intensity' not in work.columns:
        work['intensity'] = work['height']
    if 'shift' not in work.columns and 'ppm' in work.columns:
        work['shift'] = work['ppm']

    rows = []
    cluster_rows = []
    clusterer = PeakGMMIntensityAutoClusterer(
        min_components=min_components,
        max_components=max_components,
        use_log=use_log,
    )
    for sample, part in work.groupby('sample'):
        labels, model = clusterer.fit_predict(part['intensity'].to_numpy())
        part = part.copy()
        part['cluster_id'] = labels
        rows.append(part)
        for cid, grp in part.groupby('cluster_id'):
            cluster_rows.append({
                'sample': sample,
                'cluster_id': int(cid),
                'num_peaks': int(len(grp)),
                'shift_values': ','.join(f'{x:.4f}' for x in grp['shift'].tolist()),
                'mean_intensity': float(grp['intensity'].mean()),
            })
    detailed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    clusters = pd.DataFrame(cluster_rows)
    return detailed, clusters
