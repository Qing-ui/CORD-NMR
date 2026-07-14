from __future__ import annotations

import pandas as pd


def _greedy_unique_matches(target: pd.DataFrame, ref: pd.DataFrame, shift_col: str = "ppm", tol_ppm: float = 0.8) -> dict[int, int]:
    tgt = target.reset_index(drop=True)
    rrf = ref.reset_index(drop=True)
    if tgt.empty or rrf.empty:
        return {}
    pairs: list[tuple[float, int, int]] = []
    for tidx, trow in tgt.iterrows():
        diffs = (rrf[shift_col] - float(trow[shift_col])).abs()
        close = diffs[diffs <= tol_ppm]
        for ridx, dist in close.items():
            pairs.append((float(dist), int(tidx), int(ridx)))
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))
    used_t: set[int] = set()
    used_r: set[int] = set()
    matches: dict[int, int] = {}
    for _, tidx, ridx in pairs:
        if tidx in used_t or ridx in used_r:
            continue
        used_t.add(tidx)
        used_r.add(ridx)
        matches[tidx] = ridx
    return matches


def infer_dept_types(allc: pd.DataFrame, d135: pd.DataFrame, d90: pd.DataFrame, tol_ppm: float = 0.8) -> pd.DataFrame:
    work = allc.copy().reset_index(drop=True)
    d135 = d135.copy().reset_index(drop=True)
    d90 = d90.copy().reset_index(drop=True)
    out_types = [None] * len(work)
    sample_iter = work.groupby("sample", sort=False) if "sample" in work.columns else [(None, work)]
    for sample, group in sample_iter:
        group_local = group.reset_index()
        if sample is None:
            d135_s = d135.reset_index(drop=True)
            d90_s = d90.reset_index(drop=True)
        else:
            d135_s = (d135[d135["sample"] == sample] if "sample" in d135.columns else d135).reset_index(drop=True)
            d90_s = (d90[d90["sample"] == sample] if "sample" in d90.columns else d90).reset_index(drop=True)
        m135 = _greedy_unique_matches(group_local[["ppm"]], d135_s[[c for c in d135_s.columns if c in {"ppm", "height"}]], shift_col="ppm", tol_ppm=tol_ppm)
        m90 = _greedy_unique_matches(group_local[["ppm"]], d90_s[[c for c in d90_s.columns if c in {"ppm", "height"}]], shift_col="ppm", tol_ppm=tol_ppm)
        for local_tidx, row in group_local.iterrows():
            global_pos = int(row["index"])
            ridx135 = m135.get(int(local_tidx))
            ridx90 = m90.get(int(local_tidx))
            if ridx135 is None:
                out_types[global_pos] = "Cq"
                continue
            if ridx90 is not None:
                out_types[global_pos] = "CH"
                continue
            height135 = float(d135_s.loc[ridx135, "height"]) if "height" in d135_s.columns else 0.0
            out_types[global_pos] = "CH3" if height135 >= 0 else "CH2"
    work["type"] = out_types
    return work
