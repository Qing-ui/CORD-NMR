from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from nmr_trendtrack.config import ClusterConfig
from nmr_trendtrack.contracts import ClusterPrototype, Membership, TrendVector
from nmr_trendtrack.cluster.shared_logic import assign_label_from_probs


def _matrix_from_bucket(bucket: List[TrendVector]) -> np.ndarray:
    rows = []
    for tv in bucket:
        rows.append([np.nan if v is None else float(v) for v in tv.step_log_fc])
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.asarray(rows, dtype=float)


def _impute_nan(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X.copy()
    Y = X.copy()
    col_means = []
    for j in range(Y.shape[1]):
        col = Y[:, j]
        obs = ~np.isnan(col)
        col_means.append(float(np.mean(col[obs])) if np.any(obs) else 0.0)
    col_means = np.asarray(col_means, dtype=float)
    inds = np.where(np.isnan(Y))
    Y[inds] = np.take(col_means, inds[1])
    return Y


def _prepare_matrix(X: np.ndarray, zscore: bool) -> np.ndarray:
    Y = _impute_nan(X)
    if Y.size == 0 or not zscore:
        return Y
    mu = np.mean(Y, axis=0, keepdims=True)
    sd = np.std(Y, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Y - mu) / sd


def _corr_distance_matrix(Y: np.ndarray) -> np.ndarray:
    n = Y.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    if Y.shape[1] == 0:
        return np.zeros((n, n), dtype=float)
    R = np.corrcoef(Y)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(R, 1.0)
    D = (1.0 - R) / 2.0
    D = np.clip(D, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D


def _euclidean_distance_matrix(Y: np.ndarray) -> np.ndarray:
    n = Y.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    D = squareform(pdist(Y, metric="euclidean"))
    maxv = float(np.max(D)) if D.size else 0.0
    if maxv > 0:
        D = D / maxv
    return D


def _distance_matrix(Y: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    metric = getattr(cfg, "hac_metric", "euclidean")
    if metric == "signed_corr":
        return _corr_distance_matrix(Y)
    if metric == "hybrid":
        Dc = _corr_distance_matrix(Y)
        De = _euclidean_distance_matrix(Y)
        w = float(getattr(cfg, "hac_corr_weight", 0.20))
        return w * Dc + (1.0 - w) * De
    return _euclidean_distance_matrix(Y)


def _hac_labels(Y: np.ndarray, k: int, cfg: ClusterConfig) -> np.ndarray:
    n = Y.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    if k <= 1:
        return np.zeros(n, dtype=int)
    k = min(k, n)
    linkage_method = getattr(cfg, "hac_linkage", "ward")
    metric = getattr(cfg, "hac_metric", "euclidean")
    try:
        if linkage_method == "ward" and metric == "euclidean":
            Z = linkage(Y, method="ward", metric="euclidean")
        else:
            D = _distance_matrix(Y, cfg)
            condensed = squareform(np.clip(D, 0.0, None), checks=False)
            Z = linkage(condensed, method=linkage_method)
        labels_1based = fcluster(Z, t=k, criterion="maxclust")
        return labels_1based - 1
    except Exception:
        return np.arange(n) % k


def _cluster_sse(Y: np.ndarray, labels: np.ndarray) -> float:
    if Y.size == 0:
        return 0.0
    sse = 0.0
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        C = Y[idx]
        center = np.mean(C, axis=0, keepdims=True)
        sse += float(np.sum((C - center) ** 2))
    return sse


def _select_k_hac_bic(Y: np.ndarray, max_k: int, min_cluster_size: int, cfg: ClusterConfig) -> Tuple[int, np.ndarray]:
    n = Y.shape[0]
    p = Y.shape[1] if Y.ndim == 2 else 1
    if n <= 1:
        return 1, np.zeros(n, dtype=int)
    upper = min(max_k, n)
    best_k = 1
    best_labels = np.zeros(n, dtype=int)
    best_bic = float("inf")
    for k in range(1, upper + 1):
        labels = _hac_labels(Y, k, cfg)
        counts = np.bincount(labels) if len(labels) else np.array([0])
        if k > 1 and np.any(counts < max(1, min_cluster_size)):
            continue
        sse = max(_cluster_sse(Y, labels), 1e-8)
        bic = n * math.log(sse / max(n, 1)) + (k * (p + 1)) * math.log(max(n, 2))
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = labels
    return best_k, best_labels


def _cluster_meta_from_prototypes(prototypes: List[ClusterPrototype]) -> Dict[str, Dict[str, object]]:
    return {
        p.cluster_id: {
            "family_id": p.family_id,
            "amplitude_rank": p.amplitude_rank,
            "amplitude_center": p.amplitude_center,
        }
        for p in prototypes
    }


def _soft_probs_from_labels(Y: np.ndarray, labels: np.ndarray, temperature: float) -> Tuple[List[Dict[int, float]], Dict[int, np.ndarray]]:
    cluster_ids = sorted(int(x) for x in np.unique(labels))
    centers = {cid: np.mean(Y[labels == cid], axis=0) for cid in cluster_ids}
    out: List[Dict[int, float]] = []
    temp = max(float(temperature), 1e-6)
    for i in range(len(Y)):
        entries = []
        for cid in cluster_ids:
            dist = float(np.linalg.norm(Y[i] - centers[cid]))
            entries.append((cid, dist))
        logits = np.asarray([-dist / temp for _, dist in entries], dtype=float)
        logits -= np.max(logits)
        ex = np.exp(logits)
        pr = ex / np.sum(ex)
        out.append({cid: float(p) for (cid, _), p in zip(entries, pr)})
    return out, centers


def fit_hac_for_bucket(bucket: List[TrendVector], cfg: ClusterConfig) -> Tuple[List[ClusterPrototype], List[Membership]]:
    if not bucket:
        return [], []
    X = _matrix_from_bucket(bucket)
    n, d = X.shape
    if n == 0:
        return [], []
    if n == 1:
        cid = "f0_a0"
        proto = ClusterPrototype(
            cluster_id=cid,
            presence_mask=bucket[0].presence_mask,
            mean_step_log_fc=[float(v) for v in _impute_nan(X)[0].tolist()] if d else [],
            step_scale=[1.0 for _ in range(d)],
            n_tracks=1,
            weight=1.0,
            family_id="f0",
            family_direction=[0.0 for _ in range(d)],
            amplitude_center=0.0,
            amplitude_scale=1.0,
            amplitude_rank=0,
        )
        mem = Membership(track_id=bucket[0].track_id, cluster_probs={cid: 1.0}, best_cluster_id=cid, second_cluster_id=None, assigned_label="pure", family_id="f0")
        return [proto], [mem]

    Y = _prepare_matrix(X, getattr(cfg, "hac_zscore", True))
    k, labels = _select_k_hac_bic(Y, max(getattr(cfg, "max_clusters_per_mask", 4), 1), max(getattr(cfg, "min_cluster_size", 3), 1), cfg)
    probs_raw, centers = _soft_probs_from_labels(Y, labels, getattr(cfg, "hac_temperature", 0.35))

    prototypes: List[ClusterPrototype] = []
    cid_map: Dict[int, str] = {}
    for j, cid in enumerate(sorted(int(x) for x in np.unique(labels))):
        idx = np.where(labels == cid)[0]
        cid_map[cid] = f"f{j}_a0"
        block = _impute_nan(X[idx])
        mean_vec = np.mean(block, axis=0) if len(block) else np.zeros((d,), dtype=float)
        scale_vec = np.std(block, axis=0) if len(block) else np.ones((d,), dtype=float)
        scale_vec = np.maximum(scale_vec, math.sqrt(cfg.min_variance))
        norm = float(np.linalg.norm(mean_vec))
        direction = (mean_vec / norm).tolist() if norm > 1e-8 else [0.0 for _ in range(d)]
        prototypes.append(
            ClusterPrototype(
                cluster_id=cid_map[cid],
                presence_mask=bucket[0].presence_mask,
                mean_step_log_fc=[float(v) for v in mean_vec.tolist()],
                step_scale=[float(v) for v in scale_vec.tolist()],
                n_tracks=int(len(idx)),
                weight=float(len(idx) / max(n, 1)),
                family_id=f"f{j}",
                family_direction=[float(v) for v in direction],
                amplitude_center=float(np.linalg.norm(centers[cid])) if len(centers[cid]) else 0.0,
                amplitude_scale=1.0,
                amplitude_rank=0,
            )
        )

    cluster_meta = _cluster_meta_from_prototypes(prototypes)
    memberships: List[Membership] = []
    for tv, raw_probs in zip(bucket, probs_raw):
        probs = {cid_map[c]: p for c, p in raw_probs.items()}
        label, best, second, family_id = assign_label_from_probs(
            probs,
            cluster_meta,
            cfg.shared_prob_min,
            cfg.shared_gap_max,
            cfg.uncertain_prob_max,
            cfg.allow_shared_reuse,
        )
        memberships.append(Membership(track_id=tv.track_id, cluster_probs=probs, best_cluster_id=best, second_cluster_id=second, assigned_label=label, family_id=family_id))
    return prototypes, memberships


def fit_all_buckets(
    buckets: Dict[Tuple[int, ...], List[TrendVector]],
    cfg: ClusterConfig,
) -> Tuple[List[ClusterPrototype], List[Membership]]:
    all_prototypes: List[ClusterPrototype] = []
    all_memberships: List[Membership] = []
    bucket_index = 0
    for _, bucket in sorted(buckets.items(), key=lambda kv: (sum(kv[0]), kv[0])):
        protos, members = fit_hac_for_bucket(bucket, cfg)
        mapping: Dict[str, str] = {}
        fam_mapping: Dict[str, str] = {}
        for p in protos:
            old_family = p.family_id or "f0"
            if old_family not in fam_mapping:
                fam_mapping[old_family] = f"b{bucket_index}_{old_family}"
            old_cid = p.cluster_id
            new_cid = f"b{bucket_index}_{old_cid}"
            mapping[old_cid] = new_cid
            p.cluster_id = new_cid
            p.family_id = fam_mapping[old_family]
        for m in members:
            m.cluster_probs = {mapping.get(k, k): v for k, v in m.cluster_probs.items()}
            m.best_cluster_id = mapping.get(m.best_cluster_id, m.best_cluster_id)
            m.second_cluster_id = mapping.get(m.second_cluster_id, m.second_cluster_id)
            if m.family_id is not None:
                m.family_id = fam_mapping.get(m.family_id, m.family_id)
        all_prototypes.extend(protos)
        all_memberships.extend(members)
        bucket_index += 1
    return all_prototypes, all_memberships
