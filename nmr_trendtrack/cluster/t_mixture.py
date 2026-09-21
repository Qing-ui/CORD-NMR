from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

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


def _row_normalize(Y: np.ndarray) -> np.ndarray:
    if Y.size == 0:
        return Y.copy()
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return Y / norms


def _init_farthest(X: np.ndarray, k: int, directional: bool) -> np.ndarray:
    n = X.shape[0]
    if k >= n:
        centers = X[:k].copy()
        return _row_normalize(centers) if directional else centers
    centers = [X[0]]
    while len(centers) < k:
        C = np.asarray(centers, dtype=float)
        if directional:
            Cn = _row_normalize(C)
            sims = np.max(np.stack([_row_normalize(X) @ c for c in Cn], axis=1), axis=1)
            idx = int(np.argmin(sims))
        else:
            d2 = np.min(np.stack([np.sum((X - c) ** 2, axis=1) for c in C], axis=1), axis=1)
            idx = int(np.argmax(d2))
        centers.append(X[idx])
    C = np.asarray(centers, dtype=float)
    return _row_normalize(C) if directional else C


def _spherical_kmeans(V: np.ndarray, k: int, max_iters: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
    n, d = V.shape
    if n == 0:
        return np.zeros((0, d)), np.zeros((0,), dtype=int), 0.0
    centers = _init_farthest(V, k, directional=True)
    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iters):
        sims = V @ centers.T
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            pts = V[labels == j]
            if len(pts) == 0:
                centers[j] = V[int(np.random.randint(0, n))]
            else:
                centers[j] = pts.mean(axis=0)
        centers = _row_normalize(centers)
    score = float(np.sum(np.max(V @ centers.T, axis=1)))
    return centers, labels, score


def _euclidean_kmeans(X: np.ndarray, k: int, max_iters: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
    n, d = X.shape
    if n == 0:
        return np.zeros((0, d)), np.zeros((0,), dtype=int), 0.0
    centers = _init_farthest(X, k, directional=False)
    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iters):
        d2 = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], axis=1)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            pts = X[labels == j]
            if len(pts) == 0:
                centers[j] = X[int(np.random.randint(0, n))]
            else:
                centers[j] = pts.mean(axis=0)
    final_d2 = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], axis=1)
    score = -float(np.sum(np.min(final_d2, axis=1)))
    return centers, labels, score


def _sign_signature(vec: np.ndarray, zero_band: float) -> Tuple[int, ...]:
    sig = []
    for x in vec:
        if x > zero_band:
            sig.append(1)
        elif x < -zero_band:
            sig.append(-1)
        else:
            sig.append(0)
    return tuple(sig)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def _merge_similar_family_labels(centers: np.ndarray, labels: np.ndarray, cfg: ClusterConfig) -> Tuple[np.ndarray, np.ndarray]:
    if centers.shape[0] <= 1:
        return centers, labels
    parent = list(range(centers.shape[0]))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(centers.shape[0]):
        for j in range(i + 1, centers.shape[0]):
            same_sign = _sign_signature(centers[i], cfg.family_sign_zero_band) == _sign_signature(centers[j], cfg.family_sign_zero_band)
            cos_ok = _cosine_similarity(centers[i], centers[j]) >= cfg.family_cosine_merge_min
            dist_ok = float(np.linalg.norm(centers[i] - centers[j])) <= cfg.family_distance_merge_max
            if same_sign and (cos_ok or dist_ok):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(centers.shape[0]):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == centers.shape[0]:
        return centers, labels

    new_centers = []
    remap: Dict[int, int] = {}
    for new_idx, old_group in enumerate(groups.values()):
        block = centers[old_group]
        center = block.mean(axis=0)
        if cfg.use_directional_family_clustering:
            center = _row_normalize(center.reshape(1, -1))[0]
        new_centers.append(center)
        for old_idx in old_group:
            remap[old_idx] = new_idx
    new_labels = np.asarray([remap[int(x)] for x in labels], dtype=int)
    return np.asarray(new_centers, dtype=float), new_labels


def _choose_family_partition(F: np.ndarray, cfg: ClusterConfig) -> Tuple[np.ndarray, np.ndarray]:
    n, d = F.shape
    if n == 0:
        return np.zeros((0, d)), np.zeros((0,), dtype=int)
    max_families = cfg.max_families_per_mask
    if n >= 30:
        max_families = max(max_families, min(7, max(4, n // 5)))
    elif n >= 20:
        max_families = max(max_families, min(5, max(3, n // 6)))
    max_k = min(max_families, cfg.max_clusters_per_mask if cfg.max_clusters_per_mask > 0 else max_families, max(1, n // max(1, cfg.min_cluster_size)))
    max_k = max(1, max_k)
    best_score = float("-inf")
    best = (F[:1].copy(), np.zeros(n, dtype=int))
    for k in range(1, max_k + 1):
        if cfg.use_directional_family_clustering:
            centers, labels, score = _spherical_kmeans(_row_normalize(F), k)
        else:
            centers, labels, score = _euclidean_kmeans(F, k)
        penalty = 0.5 * (k - 1) * max(d, 1) * math.log(max(n, 2)) / max(n, 1)
        adj = score - penalty
        if adj > best_score:
            best_score = adj
            best = (centers, labels)
    centers, labels = best
    if cfg.merge_similar_families and n < 24:
        centers, labels = _merge_similar_family_labels(centers, labels, cfg)
    return centers, labels


def _student_t_logpdf_1d(x: float, mu: float, var: float, nu: float) -> float:
    var = max(var, 1e-8)
    delta = ((x - mu) ** 2) / var
    return float(
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(var)
        - 0.5 * math.log(nu * math.pi)
        - ((nu + 1.0) / 2.0) * math.log1p(delta / nu)
    )


def _fit_1d_t_mixture(values: np.ndarray, k: int, cfg: ClusterConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(values)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0)), float("-inf")
    vals = np.asarray(values, dtype=float)
    if k == 1:
        mu = np.array([float(np.median(vals))])
        var = np.array([max(float(np.var(vals)), cfg.min_variance)])
        pi = np.array([1.0])
        resp = np.ones((n, 1), dtype=float)
        ll = sum(_student_t_logpdf_1d(float(x), mu[0], var[0], cfg.student_t_nu) for x in vals)
        bic = -2.0 * ll + 2 * math.log(max(n, 2))
        return mu, var, pi, resp, -bic

    qs = np.linspace(0.1, 0.9, k)
    mu = np.quantile(vals, qs)
    global_var = max(float(np.var(vals)), cfg.min_variance)
    var = np.full(k, global_var, dtype=float)
    pi = np.full(k, 1.0 / k, dtype=float)
    nu = cfg.student_t_nu
    prev_ll = -np.inf
    resp = np.full((n, k), 1.0 / k, dtype=float)
    for _ in range(cfg.max_em_iters):
        log_resp = np.zeros((n, k), dtype=float)
        delta = np.zeros((n, k), dtype=float)
        for i, x in enumerate(vals):
            for c in range(k):
                log_resp[i, c] = math.log(max(pi[c], 1e-12)) + _student_t_logpdf_1d(float(x), float(mu[c]), float(var[c]), nu)
                delta[i, c] = ((x - mu[c]) ** 2) / max(var[c], 1e-8)
        max_log = np.max(log_resp, axis=1, keepdims=True)
        ex = np.exp(log_resp - max_log)
        resp = ex / np.sum(ex, axis=1, keepdims=True)
        u = (nu + 1.0) / np.maximum(nu + delta, 1e-8)
        for c in range(k):
            rc = resp[:, c]
            pi[c] = max(float(np.mean(rc)), 1e-12)
            w = rc * u[:, c]
            denom = np.sum(w)
            if denom <= 1e-12:
                continue
            mu[c] = np.sum(w * vals) / denom
            var_num = np.sum(w * (vals - mu[c]) ** 2)
            var[c] = max(var_num / max(np.sum(rc), 1e-12), cfg.min_variance)
        pi /= np.sum(pi)
        ll = 0.0
        for x in vals:
            parts = [math.log(max(pi[c], 1e-12)) + _student_t_logpdf_1d(float(x), float(mu[c]), float(var[c]), nu) for c in range(k)]
            m = max(parts)
            ll += m + math.log(sum(math.exp(v - m) for v in parts))
        if abs(ll - prev_ll) < cfg.em_tol:
            prev_ll = ll
            break
        prev_ll = ll
    order = np.argsort(mu)
    mu = mu[order]
    var = var[order]
    pi = pi[order]
    resp = resp[:, order]
    params = 3 * k - 1
    bic = -2.0 * prev_ll + params * math.log(max(n, 2))
    return mu, var, pi, resp, -bic


def _student_t_pdf_1d(x: np.ndarray, mu: float, var: float, nu: float) -> np.ndarray:
    var = max(var, 1e-8)
    delta = ((x - mu) ** 2) / var
    coef = math.exp(math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0))
    coef /= math.sqrt(var * nu * math.pi)
    return coef * np.power(1.0 + delta / nu, -((nu + 1.0) / 2.0))


def _amplitude_separation(mu: np.ndarray, var: np.ndarray) -> float:
    if len(mu) <= 1:
        return float('inf')
    sep = []
    for i in range(len(mu) - 1):
        pooled = math.sqrt(max(var[i] + var[i + 1], 1e-8))
        sep.append(abs(float(mu[i + 1] - mu[i])) / pooled)
    return float(min(sep)) if sep else float('inf')


def _amplitude_overlap(mu: np.ndarray, var: np.ndarray, pi: np.ndarray, cfg: ClusterConfig) -> float:
    if len(mu) <= 1:
        return 0.0
    lows = [float(m - 6.0 * math.sqrt(max(v, cfg.min_variance))) for m, v in zip(mu, var)]
    highs = [float(m + 6.0 * math.sqrt(max(v, cfg.min_variance))) for m, v in zip(mu, var)]
    grid = np.linspace(min(lows), max(highs), 801)
    comps = []
    for m, v, w in zip(mu, var, pi):
        comps.append(float(w) * _student_t_pdf_1d(grid, float(m), float(v), cfg.student_t_nu))
    stack = np.stack(comps, axis=1)
    total = np.sum(stack, axis=1)
    total = np.maximum(total, 1e-12)
    overlap_vals = []
    for i in range(len(mu) - 1):
        a = stack[:, i]
        b = stack[:, i + 1]
        overlap_vals.append(float(np.trapz(np.minimum(a, b), grid) / np.trapz(total, grid)))
    return float(max(overlap_vals)) if overlap_vals else 0.0


def _choose_amplitude_model(values: np.ndarray, cfg: ClusterConfig, hard_limit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(values)
    if not cfg.enable_amplitude_binning:
        return _fit_1d_t_mixture(values, 1, cfg)[:4]
    max_k = min(hard_limit, cfg.max_amplitude_bins_per_family, max(1, n // max(1, cfg.min_cluster_size)))
    max_k = max(1, max_k)
    candidates = []
    for k in range(1, max_k + 1):
        mu, var, pi, resp, score = _fit_1d_t_mixture(values, k, cfg)
        sep = _amplitude_separation(mu, var)
        ov = _amplitude_overlap(mu, var, pi, cfg)
        candidates.append((k, score, sep, ov, mu, var, pi, resp))

    one = next(c for c in candidates if c[0] == 1)
    chosen = one
    one_score = float(one[1])
    best_aug_score = one_score
    for cand in candidates:
        k, score, sep, ov, mu, var, pi, resp = cand
        if k == 1:
            continue
        gain = float(score - one_score)
        sep_ok = sep >= cfg.amplitude_split_min_separation
        ov_ok = ov <= cfg.amplitude_split_max_overlap
        aug_score = float(score - max(0.0, ov - cfg.amplitude_split_max_overlap) * 10.0)
        if gain >= cfg.amplitude_split_min_gain and sep_ok and ov_ok and aug_score > best_aug_score:
            chosen = cand
            best_aug_score = aug_score

    _, _, _, _, mu, var, pi, resp = chosen
    return mu, var, pi, resp


def _cluster_meta_from_prototypes(prototypes: Sequence[ClusterPrototype]) -> Dict[str, Dict[str, object]]:
    return {
        p.cluster_id: {
            "family_id": p.family_id,
            "amplitude_rank": p.amplitude_rank,
            "amplitude_center": p.amplitude_center,
        }
        for p in prototypes
    }


def _collapse_to_single_cluster_if_consistent(bucket: List[TrendVector], X: np.ndarray, cfg: ClusterConfig) -> Tuple[List[ClusterPrototype], List[Membership]] | None:
    if not bucket:
        return None
    # Only collapse small, genuinely uniform buckets. Large same-mask buckets may contain
    # multiple semantic components that share sign but differ in trend magnitude.
    if len(bucket) > 20:
        return None
    totals = []
    for tv in bucket:
        vals = [float(v) for v in tv.step_log_fc if v is not None]
        if not vals:
            continue
        totals.append(float(sum(vals)))
    if not totals:
        return None
    zero_band = max(0.25, float(cfg.family_sign_zero_band))
    pos = sum(v > zero_band for v in totals)
    neg = sum(v < -zero_band for v in totals)
    same_sign = (pos == 0) or (neg == 0)
    if not same_sign:
        return None
    if pos == 0 and neg == 0:
        return None
    Y = _impute_nan(X)
    mean_vec = np.mean(Y, axis=0) if len(Y) else np.zeros((X.shape[1],), dtype=float)
    scale_vec = np.std(Y, axis=0) if len(Y) else np.ones((X.shape[1],), dtype=float)
    scale_vec = np.maximum(scale_vec, math.sqrt(cfg.min_variance))
    cid = 'f0_a0'
    proto = ClusterPrototype(
        cluster_id=cid,
        presence_mask=bucket[0].presence_mask,
        mean_step_log_fc=[float(v) for v in mean_vec.tolist()],
        step_scale=[float(v) for v in scale_vec.tolist()],
        n_tracks=len(bucket),
        weight=1.0,
        family_id='f0',
        family_direction=[float(v) for v in _row_normalize(mean_vec.reshape(1, -1))[0].tolist()] if len(mean_vec) else [],
        amplitude_center=float(np.linalg.norm(mean_vec)) if len(mean_vec) else 0.0,
        amplitude_scale=1.0,
        amplitude_rank=0,
    )
    members = [Membership(track_id=tv.track_id, cluster_probs={cid: 1.0}, best_cluster_id=cid, second_cluster_id=None, assigned_label='pure', family_id='f0') for tv in bucket]
    return [proto], members


def fit_t_mixture_for_bucket(bucket: List[TrendVector], cfg: ClusterConfig) -> Tuple[List[ClusterPrototype], List[Membership]]:
    if not bucket:
        return [], []
    X = _matrix_from_bucket(bucket)
    n, d = X.shape
    if n == 0:
        return [], []
    collapsed = _collapse_to_single_cluster_if_consistent(bucket, X, cfg)
    if collapsed is not None:
        return collapsed
    Y = _impute_nan(X)
    if d == 0:
        cid = "f0_a0"
        proto = ClusterPrototype(cluster_id=cid, presence_mask=bucket[0].presence_mask, mean_step_log_fc=[], step_scale=[], n_tracks=n, weight=1.0, family_id="f0", family_direction=[], amplitude_center=0.0, amplitude_scale=1.0, amplitude_rank=0)
        members = [Membership(track_id=tv.track_id, cluster_probs={cid: 1.0}, best_cluster_id=cid, second_cluster_id=None, assigned_label="pure", family_id="f0") for tv in bucket]
        return [proto], members

    F = _row_normalize(Y) if cfg.use_directional_family_clustering else Y
    fam_centers, fam_labels = _choose_family_partition(F, cfg)
    n_fam = max(1, fam_centers.shape[0])
    prototypes: List[ClusterPrototype] = []
    family_amp_params: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    family_member_idx: Dict[str, np.ndarray] = {}
    family_dirs: Dict[str, np.ndarray] = {}
    family_feature_centers: Dict[str, np.ndarray] = {}
    total_cluster_budget = max(cfg.max_clusters_per_mask, n_fam)
    remaining_budget = total_cluster_budget
    for f in range(n_fam):
        fam_id = f"f{f}"
        idx = np.where(fam_labels == f)[0]
        if len(idx) == 0:
            continue
        family_member_idx[fam_id] = idx
        feature_center = fam_centers[f]
        family_feature_centers[fam_id] = feature_center
        raw_mean = np.mean(Y[idx], axis=0)
        raw_dir = _row_normalize(raw_mean.reshape(1, -1))[0]
        family_dirs[fam_id] = raw_dir
        amps = Y[idx] @ raw_dir
        fam_budget = max(1, min(cfg.max_amplitude_bins_per_family, remaining_budget - (n_fam - f - 1)))
        mu, var, pi, resp = _choose_amplitude_model(amps, cfg, fam_budget)
        family_amp_params[fam_id] = (mu, var, pi)
        remaining_budget -= len(mu)
        for j in range(len(mu)):
            w = resp[:, j]
            if np.sum(w) <= 1e-12:
                mean_vec = np.nanmean(X[idx], axis=0)
                scale_vec = np.nanstd(X[idx], axis=0)
                n_tracks = 0
                weight = 0.0
            else:
                mean_vec = []
                scale_vec = []
                for col in range(d):
                    col_vals = X[idx, col]
                    obs = ~np.isnan(col_vals)
                    if not np.any(obs):
                        mean_vec.append(0.0)
                        scale_vec.append(1.0)
                    else:
                        ww = w[obs]
                        vv = col_vals[obs]
                        denom = np.sum(ww)
                        if denom <= 1e-12:
                            mean_vec.append(float(np.mean(vv)))
                            scale_vec.append(max(float(np.std(vv)), math.sqrt(cfg.min_variance)))
                        else:
                            m = float(np.sum(ww * vv) / denom)
                            s = math.sqrt(max(float(np.sum(ww * (vv - m) ** 2) / denom), cfg.min_variance))
                            mean_vec.append(m)
                            scale_vec.append(s)
                n_tracks = int(np.sum(np.argmax(resp, axis=1) == j))
                weight = float(np.mean(resp[:, j]))
            if not isinstance(mean_vec, list):
                mean_vec = [float(v) for v in mean_vec.tolist()]
                scale_vec = [float(max(v, math.sqrt(cfg.min_variance))) for v in scale_vec.tolist()]
            cid = f"{fam_id}_a{j}"
            prototypes.append(
                ClusterPrototype(
                    cluster_id=cid,
                    presence_mask=bucket[0].presence_mask,
                    mean_step_log_fc=[float(v) for v in mean_vec],
                    step_scale=[float(v) for v in scale_vec],
                    n_tracks=n_tracks,
                    weight=weight,
                    family_id=fam_id,
                    family_direction=[float(v) for v in raw_dir.tolist()],
                    amplitude_center=float(mu[j]),
                    amplitude_scale=float(math.sqrt(max(var[j], cfg.min_variance))),
                    amplitude_rank=j,
                )
            )

    cluster_meta = _cluster_meta_from_prototypes(prototypes)
    memberships: List[Membership] = []
    family_ids = [f"f{i}" for i in range(n_fam) if f"f{i}" in family_amp_params]
    for i, tv in enumerate(bucket):
        row = Y[i]
        cluster_logits: Dict[str, float] = {}
        for fam_id in family_ids:
            if cfg.use_directional_family_clustering:
                v = _row_normalize(row.reshape(1, -1))[0]
                fam_logit = cfg.family_kappa * float(v @ family_feature_centers[fam_id])
            else:
                center = family_feature_centers[fam_id]
                fam_logit = -0.5 * float(np.sum((row - center) ** 2))
            mu, var, pi = family_amp_params[fam_id]
            raw_dir = family_dirs[fam_id]
            amp = float(row @ raw_dir)
            for j in range(len(mu)):
                cid = f"{fam_id}_a{j}"
                cluster_logits[cid] = fam_logit + math.log(max(float(pi[j]), 1e-12)) + _student_t_logpdf_1d(amp, float(mu[j]), float(var[j]), cfg.student_t_nu)
        if not cluster_logits:
            memberships.append(Membership(track_id=tv.track_id, cluster_probs={}, best_cluster_id=None, second_cluster_id=None, assigned_label="uncertain", family_id=None))
            continue
        m = max(cluster_logits.values())
        exps = {cid: math.exp(val - m) for cid, val in cluster_logits.items()}
        z = sum(exps.values())
        probs = {cid: val / z for cid, val in exps.items()}
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
        protos, members = fit_t_mixture_for_bucket(bucket, cfg)
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
