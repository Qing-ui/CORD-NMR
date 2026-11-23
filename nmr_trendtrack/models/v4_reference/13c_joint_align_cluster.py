"""Regional alignment and peak-track scoring for continuous carbon spectra."""


import math
from collections import defaultdict
from statistics import median, mean


REGIONS = [(0.0, 50.0, 0.20), (50.0, 110.0, 0.30), (110.0, 165.0, 0.20), (165.0, 220.0, 0.50)]
RAW_SPAN_LIMIT = 0.50
REGION_NAMES = [f"{int(a)}-{int(b)}" for a,b,_ in REGIONS]


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
