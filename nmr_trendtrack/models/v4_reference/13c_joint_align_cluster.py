#!/usr/bin/env python3
import csv, math, random, os, zipfile, json
from collections import defaultdict, Counter
from statistics import median, mean
from itertools import product

COMPS = [f"D1-{i}" for i in range(1, 7)]
REGIONS = [(0.0, 50.0, 0.20), (50.0, 110.0, 0.30), (110.0, 165.0, 0.20), (165.0, 220.0, 0.50)]
RAW_SPAN_LIMIT = 0.50
REGION_NAMES = [f"{int(a)}-{int(b)}" for a,b,_ in REGIONS]
REGION_SHIFT_AMPS = {"0-50": 0.14, "50-110": 0.22, "110-165": 0.16, "165-220": 0.38}


def write_csv(path, rows, fields=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def load_templates(path):
    d = defaultdict(list)
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            comp = r.get('component') or r.get('compound')
            if not comp:
                continue
            ppm = float(r['ppm'])
            rel = float(r.get('rel_response') or r.get('rel_response_max') or r.get('area_raw') or 1.0)
            src = r.get('source_id') or r.get('atom_id') or f"{comp}:A{len(d[comp])+1:02d}"
            atom = r.get('atom_id') or src
            d[comp].append({'component': comp, 'source_id': src, 'atom_id': atom, 'ppm': ppm, 'rel_response': rel})
    return dict(d)


def region_name(ppm):
    for lo, hi, _ in REGIONS:
        if lo <= ppm < hi:
            return f"{int(lo)}-{int(hi)}"
    return REGION_NAMES[-1]


def tol_for_ppm(ppm):
    for lo, hi, tol in REGIONS:
        if lo <= ppm < hi:
            return tol
    return REGIONS[-1][2]


def scenario_library():
    return {
        'S3_bridge_135_123_234': [
            ('M1_135', ['D1-1', 'D1-3', 'D1-5']),
            ('M2_123', ['D1-1', 'D1-2', 'D1-3']),
            ('M3_234', ['D1-2', 'D1-3', 'D1-4']),
        ],
        'S4_balanced_3to5': [
            ('M1_123', ['D1-1', 'D1-2', 'D1-3']),
            ('M2_2345', ['D1-2', 'D1-3', 'D1-4', 'D1-5']),
            ('M3_1356', ['D1-1', 'D1-3', 'D1-5', 'D1-6']),
            ('M4_12456', ['D1-1', 'D1-2', 'D1-4', 'D1-5', 'D1-6']),
        ],
        'S5_mixed_3to5': [
            ('M1_1234', ['D1-1', 'D1-2', 'D1-3', 'D1-4']),
            ('M2_2345', ['D1-2', 'D1-3', 'D1-4', 'D1-5']),
            ('M3_1356', ['D1-1', 'D1-3', 'D1-5', 'D1-6']),
            ('M4_12456', ['D1-1', 'D1-2', 'D1-4', 'D1-5', 'D1-6']),
            ('M5_246', ['D1-2', 'D1-4', 'D1-6']),
        ],
        'S5_near_coelution_3to5': [
            ('M1_135', ['D1-1', 'D1-3', 'D1-5']),
            ('M2_1245', ['D1-1', 'D1-2', 'D1-4', 'D1-5']),
            ('M3_2346', ['D1-2', 'D1-3', 'D1-4', 'D1-6']),
            ('M4_12356', ['D1-1', 'D1-2', 'D1-3', 'D1-5', 'D1-6']),
            ('M5_2456', ['D1-2', 'D1-4', 'D1-5', 'D1-6']),
        ],
    }


def concentration_curves(combos, seed=0, near=False):
    rng = random.Random(seed)
    n = len(combos)
    xs = [i / max(n-1, 1) for i in range(n)]
    curves = {}
    if near:
        base_a = [0.85 + 0.32*math.sin(2*math.pi*(x+0.1)) + 0.10*x for x in xs]
        base_b = [0.78 + 0.35*math.cos(2*math.pi*(0.85*x-0.05)) - 0.08*x for x in xs]
        curves['D1-1'] = base_a
        curves['D1-5'] = [0.96*v + 0.03*math.sin(4*math.pi*x) for v,x in zip(base_a,xs)]
        curves['D1-2'] = base_b
        curves['D1-4'] = [0.93*v + 0.04*math.cos(3*math.pi*x+0.2) for v,x in zip(base_b,xs)]
        curves['D1-3'] = [0.75 + 0.37*math.sin(2*math.pi*(1.35*x+0.22)) for x in xs]
        curves['D1-6'] = [0.90 + 0.25*math.cos(2*math.pi*(1.20*x+0.31)) for x in xs]
    else:
        phase = {'D1-1': 0.07, 'D1-2': 0.19, 'D1-3': 0.34, 'D1-4': 0.54, 'D1-5': 0.72, 'D1-6': 0.91}
        freq = {'D1-1': 1.0, 'D1-2': 1.25, 'D1-3': 0.75, 'D1-4': 1.55, 'D1-5': 1.1, 'D1-6': 1.35}
        slope = {'D1-1': 0.18, 'D1-2': -0.10, 'D1-3': 0.05, 'D1-4': -0.16, 'D1-5': 0.12, 'D1-6': -0.20}
        for comp in COMPS:
            curves[comp] = [0.80 + 0.34*math.sin(2*math.pi*(freq[comp]*x+phase[comp])) + 0.12*math.cos(2*math.pi*(0.6*x+phase[comp])) + slope[comp]*(x-0.5) for x in xs]
    out = {}
    used = set()
    for i, (sample, comps) in enumerate(combos):
        out[sample] = {}
        for comp in comps:
            raw = curves[comp][i]
            vals = curves[comp]
            mn, mx = min(vals), max(vals)
            val = 0.35 + (raw - mn) / (mx - mn + 1e-12) * 1.05 + rng.uniform(-0.012, 0.012)
            val = round(max(0.25, min(1.45, val)), 3)
            while val in used:
                val = round(val + 0.001, 3)
            used.add(val)
            out[sample][comp] = val
    return out


def make_regional_shifts(samples, seed=0, stress=True):
    rng = random.Random(seed)
    shifts = {}
    for i, s in enumerate(samples):
        shifts[s] = {}
        x = (i - (len(samples)-1)/2.0) / max(len(samples)-1, 1)
        for reg, amp in REGION_SHIFT_AMPS.items():
            base = rng.uniform(-amp*0.22, amp*0.22)
            smooth = x * amp * (0.75 if stress else 0.35)
            shifts[s][reg] = max(-amp, min(amp, base + smooth + rng.uniform(-amp*0.08, amp*0.08)))
    return shifts


def simulate_peaklists(templates, weights, shifts, seed=0):
    rng = random.Random(seed)
    rows = []
    for sample, comp_weights in weights.items():
        sample_rows = []
        for comp, conc in comp_weights.items():
            for p in templates[comp]:
                reg = region_name(p['ppm'])
                ppm = p['ppm'] + shifts[sample][reg] + rng.gauss(0, 0.0035)
                area = p['rel_response'] * conc * math.exp(rng.gauss(0, 0.025)) * 1_000_000
                sample_rows.append({
                    'sample': sample, 'ppm': round(ppm, 5), 'area': round(area, 3), 'intensity': round(area/8.0, 3),
                    'component': comp, 'source_id': p['source_id'], 'atom_id': p['atom_id'],
                    'template_ppm': round(p['ppm'], 5), 'concentration': conc, 'region_template': region_name(p['ppm'])
                })
        sample_rows.sort(key=lambda r: r['ppm'])
        for i, r in enumerate(sample_rows, 1):
            r['peak_id'] = f"{sample}_P{i:03d}"
            rows.append(r)
    return rows


def group_by_sample(rows):
    d = defaultdict(list)
    for r in rows:
        d[r['sample']].append(r)
    return d


def dense_shift_for_pair(ref_peaks, mov_peaks, reg):
    ref = [p for p in ref_peaks if region_name(p['ppm']) == reg]
    mov = [p for p in mov_peaks if region_name(p['ppm']) == reg]
    if not ref or not mov:
        return 0.0, 0, 999.0
    mid = sum(float(x) for x in reg.split('-')) / 2
    tol = tol_for_ppm(mid)
    deltas = []
    for a in ref:
        for b in mov:
            d = b['ppm'] - a['ppm']
            if abs(d) <= tol:
                deltas.append(d)
    if not deltas:
        return 0.0, 0, 999.0
    deltas.sort()
    win = max(0.035, min(0.11, tol*0.32))
    best = []
    j = 0
    for i, x in enumerate(deltas):
        while j < len(deltas) and deltas[j] - x <= win:
            j += 1
        cand = deltas[i:j]
        if len(cand) > len(best):
            best = cand
    shift = median(best) if best else median(deltas)
    res = [abs(d-shift) for d in best]
    return shift, len(best), median(res) if res else 999.0


def estimate_regional_shift(peaks_by_sample, samples):
    ref = samples[0]
    cumulative = {ref: {reg: 0.0 for reg in REGION_NAMES}}
    diagnostics = []
    # all pair estimates
    pair_est = {}
    for a in samples:
        for b in samples:
            if a == b:
                continue
            pair_est[(a,b)] = {}
            for reg in REGION_NAMES:
                pair_est[(a,b)][reg] = dense_shift_for_pair(peaks_by_sample[a], peaks_by_sample[b], reg)
    for reg in REGION_NAMES:
        assigned = {ref}
        while len(assigned) < len(samples):
            best = None
            for a in list(assigned):
                for b in samples:
                    if b in assigned:
                        continue
                    shift, n, res = pair_est[(a,b)][reg]
                    score = n / (res + 0.02)
                    if best is None or score > best[0]:
                        best = (score, a, b, shift, n, res)
            if best is None or best[4] < 2:
                for b in samples:
                    if b not in assigned:
                        cumulative.setdefault(b, {})[reg] = 0.0
                        assigned.add(b)
                break
            _, a, b, shift, n, res = best
            cumulative.setdefault(b, {})[reg] = cumulative[a][reg] + shift
            assigned.add(b)
            diagnostics.append({'region': reg, 'anchor_from': a, 'sample': b, 'pair_shift': round(shift, 5), 'cumulative_shift': round(cumulative[b][reg], 5), 'n_anchors': n, 'anchor_residual_median': round(res, 5)})
    return cumulative, diagnostics


def apply_shift(rows, samples):
    data = group_by_sample(rows)
    shift_model, diag = estimate_regional_shift(data, samples)
    out = []
    for r in rows:
        rr = dict(r)
        reg = region_name(rr['ppm'])
        rr['region'] = reg
        rr['shift_estimate'] = round(shift_model.get(rr['sample'], {}).get(reg, 0.0), 5)
        rr['ppm_corr'] = round(rr['ppm'] - rr['shift_estimate'], 5)
        out.append(rr)
    return out, shift_model, diag


def build_candidates_for_peak(seed, peaks_by_sample, samples, residual_gate=0.15, max_per_sample=3):
    choices = []
    seed_ppm = seed['ppm_corr']
    seed_reg = region_name(seed['ppm'])
    for s in samples:
        if s == seed['sample']:
            choices.append([seed])
            continue
        cand = []
        for p in peaks_by_sample[s]:
            if abs(p['ppm_corr'] - seed_ppm) <= residual_gate and abs(p['ppm'] - seed['ppm']) <= RAW_SPAN_LIMIT:
                cand.append((abs(p['ppm_corr'] - seed_ppm), p))
        cand.sort(key=lambda x: x[0])
        options = [p for _, p in cand[:max_per_sample]]
        choices.append([None] + options)
    return choices


def score_track(members, samples, residual_gate=0.15):
    present = [p for p in members.values() if p]
    if not present:
        return -999.0
    corr_vals = [p['ppm_corr'] for p in present]
    raw_vals = [p['ppm'] for p in present]
    center = median(corr_vals)
    res = [abs(x-center) for x in corr_vals]
    sigma = max(0.025, residual_gate / 2.5)
    ppm_score = sum(math.exp(-0.5 * (r/sigma)**2) for r in res) / len(res)
    cov = len(present) / len(samples)
    raw_span = max(raw_vals) - min(raw_vals) if len(raw_vals) > 1 else 0.0
    corr_span = max(corr_vals) - min(corr_vals) if len(corr_vals) > 1 else 0.0
    span_pen = max(0.0, (raw_span - RAW_SPAN_LIMIT) * 4.0) + max(0.0, (corr_span - residual_gate) * 5.0)
    return ppm_score + 0.25 * cov - span_pen


def enumerate_candidate_tracks(rows, samples, residual_gate=0.15, top_k_per_seed=6, max_per_sample=3, min_samples=2):
    peaks_by_sample = group_by_sample(rows)
    candidates = {}
    for seed in rows:
        choices = build_candidates_for_peak(seed, peaks_by_sample, samples, residual_gate=residual_gate, max_per_sample=max_per_sample)
        seed_idx = samples.index(seed['sample'])
        seed_hyps = []
        for combo in product(*choices):
            if combo[seed_idx] is None:
                continue
            present = [p for p in combo if p is not None]
            if len(present) < min_samples:
                continue
            if len(set(p['sample'] for p in present)) != len(present):
                continue
            raw_vals = [p['ppm'] for p in present]
            corr_vals = [p['ppm_corr'] for p in present]
            if len(present) > 1 and (max(raw_vals)-min(raw_vals) > RAW_SPAN_LIMIT or max(corr_vals)-min(corr_vals) > residual_gate*1.30):
                continue
            members = {p['sample']: p for p in present}
            ids = tuple(sorted(p['peak_id'] for p in present))
            score = score_track(members, samples, residual_gate=residual_gate)
            seed_hyps.append((score, ids, members))
        seed_hyps.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        for score, ids, members in seed_hyps[:top_k_per_seed]:
            if ids not in candidates or score > candidates[ids]['score']:
                candidates[ids] = {'member_ids': ids, 'members': members, 'score': score, 'kind': 'multi'}
    # singleton fallback for complete coverage; not preferred but available for uncertain/single sample groups
    for p in rows:
        ids = (p['peak_id'],)
        if ids not in candidates:
            candidates[ids] = {'member_ids': ids, 'members': {p['sample']: p}, 'score': 0.15, 'kind': 'singleton'}
    out = []
    for i, t in enumerate(candidates.values(), 1):
        t = dict(t)
        t['track_id'] = f"CT{i:05d}"
        out.append(t)
    return out


def peak_ids_of_track(t):
    return set(t['member_ids'])


def initial_set_packing(candidate_tracks, samples):
    selected = []
    used = set()
    ordered = sorted(candidate_tracks, key=lambda t: (t['score'] + 0.10*len(t['member_ids']), len(t['member_ids'])), reverse=True)
    for t in ordered:
        ids = peak_ids_of_track(t)
        if ids & used:
            continue
        selected.append(dict(t))
        used |= ids
    # make sure no peak lost: selected already includes singleton candidates; if greedy selected multi, singletons conflict and skip
    for i, t in enumerate(selected, 1):
        t['track_id'] = f"T{i:04d}"
    return selected


def track_vector(track, samples):
    vals = []
    for s in samples:
        p = track['members'].get(s)
        vals.append(None if p is None else math.log(max(p['area'], 1e-12)))
    return vals


def mask_tuple(track, samples):
    return tuple(1 if track['members'].get(s) else 0 for s in samples)


def row_centered(vals):
    obs = [v for v in vals if v is not None]
    if not obs:
        return vals
    med = median(obs)
    return [None if v is None else v-med for v in vals]


def mask_jaccard(m1, m2):
    a = {i for i,v in enumerate(m1) if v}
    b = {i for i,v in enumerate(m2) if v}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def track_distance(t1, t2, samples, mask_weight=0.35):
    v1 = row_centered(track_vector(t1, samples)); v2 = row_centered(track_vector(t2, samples))
    diffs = []
    for a,b in zip(v1, v2):
        if a is not None and b is not None:
            diffs.append(a-b)
    if len(diffs) >= 2:
        rms = math.sqrt(sum(d*d for d in diffs)/len(diffs))
        dtrend = min(2.5, rms) / 2.5
    elif len(diffs) == 1:
        dtrend = 0.65
    else:
        dtrend = 1.0
    mj = mask_jaccard(mask_tuple(t1,samples), mask_tuple(t2,samples))
    dalign = 0.05 * ((1.0 - min(1.0, t1.get('score', 0))) + (1.0 - min(1.0, t2.get('score', 0))))
    return dtrend + mask_weight * (1.0 - mj) + dalign


def fit_rank1_residual(track_indices, tracks, samples):
    if len(track_indices) <= 1:
        return 0.0, {}
    rows = {i: track_vector(tracks[i], samples) for i in track_indices}
    alpha = {i: 0.0 for i in track_indices}
    beta = {s: 0.0 for s in range(len(samples))}
    for _ in range(30):
        for i in track_indices:
            vals = [v - beta[j] for j,v in enumerate(rows[i]) if v is not None]
            if vals:
                alpha[i] = median(vals)
        for j in range(len(samples)):
            vals = [rows[i][j] - alpha[i] for i in track_indices if rows[i][j] is not None]
            if vals:
                beta[j] = median(vals)
        b0 = median(beta.values()) if beta else 0.0
        for j in beta:
            beta[j] -= b0
        for i in alpha:
            alpha[i] += b0
    per_track = {}
    allres = []
    for i in track_indices:
        res = []
        for j,v in enumerate(rows[i]):
            if v is not None:
                res.append(v - alpha[i] - beta[j])
        medabs = median([abs(x) for x in res]) if res else 0.0
        per_track[i] = medabs
        allres.extend(res)
    return (median([abs(x) for x in allres]) if allres else 0.0), per_track


def hierarchical_partitions(tracks, samples, mask_weight=0.35, max_k=14):
    n = len(tracks)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            D[i][j] = D[j][i] = track_distance(tracks[i], tracks[j], samples, mask_weight=mask_weight)
    clusters = [[i] for i in range(n)]
    parts = {n: [list(c) for c in clusters]}
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            ca = clusters[a]
            for b in range(a+1, len(clusters)):
                cb = clusters[b]
                dist = sum(D[i][j] for i in ca for j in cb) / (len(ca)*len(cb))
                if best is None or dist < best[0]:
                    best = (dist, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
        if len(clusters) <= max_k:
            parts[len(clusters)] = [list(c) for c in clusters]
    return parts, D


def silhouette(D, labels):
    n = len(labels)
    labs = sorted(set(labels))
    if n < 3 or len(labs) < 2 or len(labs) >= n:
        return -1.0
    groups = defaultdict(list)
    for i,l in enumerate(labels):
        groups[l].append(i)
    vals = []
    for i,l in enumerate(labels):
        own = groups[l]
        a = sum(D[i][j] for j in own if j != i) / (len(own)-1) if len(own) > 1 else 0.0
        b = None
        for l2, ids in groups.items():
            if l2 == l:
                continue
            dd = sum(D[i][j] for j in ids) / len(ids)
            if b is None or dd < b:
                b = dd
        if b is not None:
            den = max(a,b)
            vals.append((b-a)/den if den > 1e-12 else 0.0)
    return sum(vals)/len(vals) if vals else -1.0


def partition_cost(clusters, tracks, samples, D, lambda_k=0.08, lambda_tiny=0.08, min_main_size=3):
    n = len(tracks)
    labels = [None]*n
    for k,c in enumerate(clusters):
        for i in c:
            labels[i] = k
    sil = silhouette(D, labels)
    rank_loss = 0.0
    obs_clusters = 0
    tiny_tracks = 0
    max_frac = 0.0
    for c in clusters:
        max_frac = max(max_frac, len(c)/max(n,1))
        r, _ = fit_rank1_residual(c, tracks, samples)
        rank_loss += min(1.5, r) * len(c) / max(n,1)
        obs_clusters += 1
        if len(c) < min_main_size:
            tiny_tracks += len(c)
    tiny_frac = tiny_tracks / max(n,1)
    # Internal model-selection score (higher is better).
    # It balances fit quality, separation, and model complexity without using truth labels.
    score = (
        sil
        - 0.20 * rank_loss
        - 1.70 * max(0.0, max_frac - 0.45)
        - 0.45 * tiny_frac
        - 0.045 * obs_clusters
    )
    # Return negative score as cost so the existing minimizer selects the best partition.
    cost = -score
    return cost, {'silhouette': sil, 'rank_loss': rank_loss, 'tiny_frac': tiny_frac, 'max_cluster_frac': max_frac, 'internal_score': score}


def joint_cluster(tracks, samples, cfg):
    if not tracks:
        return [], {}
    best = None
    # Parameter grid is part of algorithm: it searches internal objective, not true labels.
    for mask_w in cfg.get('mask_weight_grid', [0.25, 0.35, 0.50]):
        parts, D = hierarchical_partitions(tracks, samples, mask_weight=mask_w, max_k=min(cfg.get('max_k',14), max(2,len(tracks))))
        for lam_k in cfg.get('lambda_k_grid', [0.035, 0.055, 0.08, 0.12]):
            for lam_tiny in cfg.get('lambda_tiny_grid', [0.04, 0.08, 0.14]):
                for k, clusters in parts.items():
                    if k < 2 and len(tracks) > 6:
                        continue
                    cost, diag = partition_cost(clusters, tracks, samples, D, lambda_k=lam_k, lambda_tiny=lam_tiny, min_main_size=cfg.get('min_main_size',3))
                    # discourage implausibly many clusters but not as expected component count
                    cost += 0.01 * max(0, k - 10)
                    if best is None or cost < best[0]:
                        best = (cost, clusters, {'mask_weight': mask_w, 'lambda_k': lam_k, 'lambda_tiny': lam_tiny, 'k': k, **diag}, D)
    clusters = best[1]
    labels = [None]*len(tracks)
    for ci,c in enumerate(clusters,1):
        for i in c:
            labels[i] = f'C{ci:02d}'
    return labels, best[2]


def selected_to_rows(tracks, samples, labels=None, status=None):
    rows = []
    for i,t in enumerate(tracks):
        comps = []; srcs = []; raw=[]; corr=[]
        r = {'track_id': t['track_id'], 'candidate_id': t.get('candidate_id', t.get('track_id')), 'kind': t.get('kind',''), 'n_peaks': len(t['member_ids']), 'alignment_score': round(t.get('score',0),4), 'cluster_id': labels[i] if labels else ''}
        for s in samples:
            p = t['members'].get(s)
            r[f'peak_{s}'] = p['peak_id'] if p else ''
            r[f'ppm_{s}'] = p['ppm'] if p else ''
            r[f'ppm_corr_{s}'] = p.get('ppm_corr','') if p else ''
            r[f'area_{s}'] = p['area'] if p else ''
            r[f'component_{s}'] = p['component'] if p else ''
            r[f'source_{s}'] = p['source_id'] if p else ''
            if p:
                comps.append(p['component']); srcs.append(p['source_id']); raw.append(p['ppm']); corr.append(p.get('ppm_corr',p['ppm']))
        cc = Counter(comps); sc = Counter(srcs)
        r['presence_mask'] = ''.join('1' if t['members'].get(s) else '0' for s in samples)
        r['major_component'] = cc.most_common(1)[0][0] if cc else ''
        r['major_source'] = sc.most_common(1)[0][0] if sc else ''
        r['components_seen'] = '+'.join(sorted(cc))
        r['sources_seen'] = '+'.join(sorted(sc))
        r['component_pure'] = len(cc)==1
        r['source_pure'] = len(sc)==1
        r['ppm_span_raw'] = round(max(raw)-min(raw),5) if len(raw)>1 else 0.0
        r['ppm_span_corr'] = round(max(corr)-min(corr),5) if len(corr)>1 else 0.0
        if status:
            r['membership_status'] = status.get(t['track_id'], 'core')
        rows.append(r)
    return rows


def cluster_diagnostics(tracks, labels, samples):
    d = defaultdict(list)
    for i,l in enumerate(labels): d[l].append(i)
    out = []
    for l, idx in sorted(d.items()):
        comps = Counter(tracks[i]['members'][next(iter(tracks[i]['members']))]['component'] if False else selected_major_comp(tracks[i]) for i in idx)
        r, per = fit_rank1_residual(idx, tracks, samples)
        masks = Counter(''.join(str(x) for x in mask_tuple(tracks[i],samples)) for i in idx)
        n_pure = sum(1 for i in idx if is_component_pure(tracks[i]))
        out.append({'cluster_id': l, 'n_tracks': len(idx), 'major_component': comps.most_common(1)[0][0] if comps else '', 'component_counts': ';'.join(f'{k}:{v}' for k,v in comps.most_common()), 'mask_counts': ';'.join(f'{k}:{v}' for k,v in masks.most_common()), 'rank1_median_abs_residual': round(r,4), 'pure_track_fraction': round(n_pure/len(idx),4) if idx else 0})
    return out


def selected_major_comp(t):
    comps=[p['component'] for p in t['members'].values()]
    return Counter(comps).most_common(1)[0][0] if comps else 'NA'


def is_component_pure(t):
    return len({p['component'] for p in t['members'].values()}) <= 1


def adjusted_rand(labels_true, labels_pred):
    n = len(labels_true)
    if n < 2:
        return 0.0
    def comb2(x): return x*(x-1)//2
    ct = Counter(labels_true); cp = Counter(labels_pred)
    cont = Counter(zip(labels_true, labels_pred))
    sum_cont = sum(comb2(v) for v in cont.values())
    sum_true = sum(comb2(v) for v in ct.values())
    sum_pred = sum(comb2(v) for v in cp.values())
    total = comb2(n)
    if total == 0:
        return 0.0
    expected = sum_true * sum_pred / total
    denom = 0.5*(sum_true+sum_pred) - expected
    return (sum_cont - expected) / denom if abs(denom) > 1e-12 else 0.0


def pairwise_pr(labels_true, labels_pred):
    tp = fp = fn = 0
    n = len(labels_true)
    for i in range(n):
        for j in range(i+1,n):
            same_t = labels_true[i] == labels_true[j]
            same_p = labels_pred[i] == labels_pred[j]
            if same_p and same_t: tp += 1
            elif same_p and not same_t: fp += 1
            elif same_t and not same_p: fn += 1
    prec = tp/(tp+fp) if tp+fp else 1.0
    rec = tp/(tp+fn) if tp+fn else 1.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return prec, rec, f1


def evaluate_tracks_and_clusters(tracks, labels):
    true_comp = [selected_major_comp(t) for t in tracks]
    pred = list(labels)
    mixed = sum(1 for t in tracks if not is_component_pure(t))
    pure_frac = 1 - mixed/max(len(tracks),1)
    ari = adjusted_rand(true_comp, pred)
    prec, rec, f1 = pairwise_pr(true_comp, pred)
    return {'n_tracks': len(tracks), 'n_clusters': len(set(pred)), 'mixed_tracks': mixed, 'pure_track_fraction': round(pure_frac,4), 'cluster_ari': round(ari,4), 'pair_precision': round(prec,4), 'pair_recall': round(rec,4), 'pair_f1': round(f1,4)}


def total_objective(tracks, labels, samples):
    groups = defaultdict(list)
    for i,l in enumerate(labels): groups[l].append(i)
    cost = 0.0
    for l, idx in groups.items():
        r,_ = fit_rank1_residual(idx, tracks, samples)
        cost += min(1.5, r) * len(idx)/max(len(tracks),1) + 0.06
        if len(idx) < 3: cost += 0.05*(3-len(idx))
    cost -= 0.12 * sum(min(1.0,t.get('score',0)) for t in tracks)/max(len(tracks),1)
    return cost


def relabel_tracks(tracks):
    for i,t in enumerate(tracks,1):
        t['track_id'] = f'T{i:04d}'
    return tracks


def fill_uncovered_with_singletons(selected, candidate_tracks, all_peak_ids):
    used = set()
    for t in selected: used |= peak_ids_of_track(t)
    singleton_by_peak = {}
    for t in candidate_tracks:
        if len(t['member_ids']) == 1:
            singleton_by_peak[t['member_ids'][0]] = t
    for pid in all_peak_ids:
        if pid not in used and pid in singleton_by_peak:
            selected.append(dict(singleton_by_peak[pid]))
            used.add(pid)
    return relabel_tracks(selected)


def find_suspicious(tracks, labels, samples, max_candidates=25):
    groups = defaultdict(list)
    for i,l in enumerate(labels): groups[l].append(i)
    suspicious = []
    for l, idx in groups.items():
        r, per = fit_rank1_residual(idx, tracks, samples)
        vals = list(per.values())
        med = median(vals) if vals else 0.0
        mad = median([abs(v-med) for v in vals]) if vals else 0.0
        gate = med + max(0.12, 2.5*mad)
        for i in idx:
            score = 0.0
            if per.get(i,0) > gate: score += per.get(i,0) - gate
            if tracks[i].get('score',0) < 0.65: score += 0.2
            if not is_component_pure(tracks[i]): score += 0.5
            if score > 0:
                suspicious.append((score, i))
    suspicious.sort(reverse=True)
    return [i for _,i in suspicious[:max_candidates]]


def iterative_joint_refinement(candidate_tracks, samples, cfg):
    all_peak_ids = set()
    for t in candidate_tracks:
        all_peak_ids |= peak_ids_of_track(t)
    selected = initial_set_packing(candidate_tracks, samples)
    selected = fill_uncovered_with_singletons(selected, candidate_tracks, all_peak_ids)
    labels, diag = joint_cluster(selected, samples, cfg)
    audit = []
    for rnd in range(1, cfg.get('max_rounds', 2)+1):
        changed = False
        suspicious = find_suspicious(selected, labels, samples, max_candidates=cfg.get('max_suspicious_per_round', 12))
        selected_peak_sets = [peak_ids_of_track(t) for t in selected]
        label_groups = defaultdict(list)
        for i,l in enumerate(labels):
            label_groups[l].append(i)
        for idx in suspicious:
            old = selected[idx]
            old_ids = peak_ids_of_track(old)
            cluster_ids = [i for i in label_groups.get(labels[idx], []) if i != idx]
            if not cluster_ids:
                continue
            old_cluster = cluster_ids + [idx]
            old_r, _ = fit_rank1_residual(old_cluster, selected, samples)
            old_local_cost = old_r - 0.03 * min(1.0, old.get('score', 0.0))
            alts = [t for t in candidate_tracks if peak_ids_of_track(t) & old_ids and t['member_ids'] != old['member_ids']]
            alts.sort(key=lambda t: (len(t['member_ids']), t.get('score',0)), reverse=True)
            best = None
            checked = 0
            for alt in alts[:cfg.get('max_alternatives_per_track', 20)]:
                alt_ids = peak_ids_of_track(alt)
                conflict = False
                for j, ids in enumerate(selected_peak_sets):
                    if j == idx:
                        continue
                    if alt_ids & ids:
                        conflict = True; break
                if conflict:
                    continue
                trial_tracks = list(selected)
                trial_tracks[idx] = dict(alt)
                new_r, _ = fit_rank1_residual(cluster_ids + [idx], trial_tracks, samples)
                new_local_cost = new_r - 0.03 * min(1.0, alt.get('score', 0.0))
                gain = old_local_cost - new_local_cost
                checked += 1
                if best is None or gain > best[0]:
                    best = (gain, alt)
            if best is not None and best[0] > cfg.get('min_realign_gain', 0.01):
                gain, alt = best
                audit.append({'round': rnd, 'old_track': old.get('track_id'), 'old_members': '+'.join(old['member_ids']), 'new_members': '+'.join(alt['member_ids']), 'gain': round(gain,5), 'accepted': True, 'checked_alternatives': checked, 'mode': 'local_rank1_replacement'})
                selected[idx] = dict(alt)
                selected_peak_sets[idx] = peak_ids_of_track(alt)
                changed = True
            else:
                audit.append({'round': rnd, 'old_track': old.get('track_id'), 'old_members': '+'.join(old['member_ids']), 'new_members': '', 'gain': round(best[0],5) if best else '', 'accepted': False, 'checked_alternatives': checked, 'mode': 'local_rank1_replacement'})
        if not changed:
            break
        selected = fill_uncovered_with_singletons(selected, candidate_tracks, all_peak_ids)
        selected = relabel_tracks(selected)
        labels, diag = joint_cluster(selected, samples, cfg)
    labels, diag = joint_cluster(selected, samples, cfg)
    return selected, labels, diag, audit


def run_scenario(templates, name, combos, outdir, seed=0, cfg=None):
    cfg = cfg or {}
    samples = [s for s,_ in combos]
    weights = concentration_curves(combos, seed=seed, near=('near' in name))
    shifts = make_regional_shifts(samples, seed=seed+11, stress=True)
    peak_rows = simulate_peaklists(templates, weights, shifts, seed=seed+101)
    corrected, shift_model, shift_diag = apply_shift(peak_rows, samples)
    cand = enumerate_candidate_tracks(corrected, samples, residual_gate=cfg.get('residual_gate',0.15), top_k_per_seed=cfg.get('top_k_per_seed',6), max_per_sample=cfg.get('max_per_sample_candidates',3), min_samples=2)
    selected, labels, cluster_diag, audit = iterative_joint_refinement(cand, samples, cfg)
    track_rows = selected_to_rows(selected, samples, labels)
    clus_rows = cluster_diagnostics(selected, labels, samples)
    eval_metrics = evaluate_tracks_and_clusters(selected, labels)
    eval_metrics.update({'scenario': name, 'n_samples': len(samples), 'n_input_peaks': len(peak_rows), 'n_candidate_tracks': len(cand), 'selected_internal_k': cluster_diag.get('k'), 'mask_weight': cluster_diag.get('mask_weight'), 'lambda_k': cluster_diag.get('lambda_k'), 'lambda_tiny': cluster_diag.get('lambda_tiny'), 'accepted_realignments': sum(1 for a in audit if a['accepted'])})
    scen_dir = os.path.join(outdir, name)
    os.makedirs(scen_dir, exist_ok=True)
    write_csv(os.path.join(scen_dir, 'mixing_matrix.csv'), [{'sample':s, **{c: weights[s].get(c,0) for c in COMPS}} for s in samples])
    write_csv(os.path.join(scen_dir, 'true_regional_shifts.csv'), [{'sample':s, **{reg: round(shifts[s][reg],5) for reg in REGION_NAMES}} for s in samples])
    write_csv(os.path.join(scen_dir, 'shift_diagnostics.csv'), shift_diag)
    write_csv(os.path.join(scen_dir, 'peaklists_corrected.csv'), corrected)
    write_csv(os.path.join(scen_dir, 'candidate_tracks_sample.csv'), selected_to_rows(cand[:200], samples))
    write_csv(os.path.join(scen_dir, 'final_tracks.csv'), track_rows)
    write_csv(os.path.join(scen_dir, 'final_cluster_diagnostics.csv'), clus_rows)
    write_csv(os.path.join(scen_dir, 'realignment_audit.csv'), audit)
    write_csv(os.path.join(scen_dir, 'scenario_summary.csv'), [eval_metrics])
    return eval_metrics


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--template-csv', required=True)
    ap.add_argument('--outdir', default='cord_nmr_output')
    ap.add_argument('--seed', type=int, default=20260426)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    templates = load_templates(args.template_csv)
    cfg = {
        'residual_gate': 0.15,
        'top_k_per_seed': 3,
        'max_per_sample_candidates': 2,
        'max_rounds': 1,
        'min_realign_gain': 0.012,
        'min_total_gain': 0.003,
        'max_suspicious_per_round': 4,
        'max_alternatives_per_track': 5,
        'mask_weight_grid': [0.22, 0.35, 0.50],
        'lambda_k_grid': [0.055],
        'lambda_tiny_grid': [0.08],
        'min_main_size': 3,
        'max_k': 14,
    }
    summaries = []
    for i, (name, combos) in enumerate(scenario_library().items()):
        summaries.append(run_scenario(templates, name, combos, args.outdir, seed=args.seed+i*100, cfg=cfg))
    write_csv(os.path.join(args.outdir, 'final_algorithm_summary.csv'), summaries)
    # package
    zip_path = args.outdir.rstrip('/').rstrip('\\') + '.zip'
    if os.path.exists(zip_path): os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(__file__, arcname='final_joint_align_cluster.py')
        for root, dirs, files in os.walk(args.outdir):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, arcname=os.path.relpath(p, args.outdir))
    print(json.dumps({'outdir': args.outdir, 'zip': zip_path, 'summary': summaries}, indent=2))

if __name__ == '__main__':
    main()
