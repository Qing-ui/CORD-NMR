from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from nmr_trendtrack.contracts import Peak, Sample


_PEAK_ID_COUNTER = 0


def _next_peak_id(sample_id: str) -> str:
    global _PEAK_ID_COUNTER
    _PEAK_ID_COUNTER += 1
    return f"{sample_id}_pk{_PEAK_ID_COUNTER:06d}"


def _normalize_columns(cols: Iterable[str]) -> list[str]:
    out = []
    for c in cols:
        s = str(c).strip().lower().replace(" ", "_")
        s = s.replace("(", "").replace(")", "")
        out.append(s)
    return out


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = _normalize_columns(df.columns)
    for cand in candidates:
        cand_n = cand.strip().lower().replace(" ", "_")
        if cand_n in cols:
            return df.columns[cols.index(cand_n)]
    return None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt", ".list", ".peak", ".peaks"}:
        text = path.read_text(errors="ignore")
        # Try CSV sniffer first.
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
            return pd.read_csv(io.StringIO(text), sep=dialect.delimiter)
        except Exception:
            pass
        # Fallback: whitespace separated, skip comment lines.
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(('#', ';', '//'))]
        if not lines:
            return pd.DataFrame(columns=["ppm", "intensity"])
        # Try header parse.
        first = re.split(r"\s+", lines[0].strip())
        has_header = any(re.search(r"ppm|int|height|amp|area", tok.lower()) for tok in first)
        if has_header:
            return pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+", engine="python")
        # No header: assume first two numeric columns = ppm, intensity.
        rows = []
        for ln in lines:
            toks = re.split(r"\s+", ln.strip())
            nums = []
            for t in toks:
                try:
                    nums.append(float(t))
                except ValueError:
                    continue
            if len(nums) >= 2:
                rows.append({"ppm": nums[0], "intensity": nums[1], "area": nums[2] if len(nums) > 2 else None})
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported peak list format: {path}")


def _to_peaks(df: pd.DataFrame, sample: Sample) -> List[Peak]:
    ppm_col = _find_column(df, ["ppm", "shift", "chemical_shift", "position", "f2ppm", "delta"])
    intensity_col = _find_column(df, ["intensity", "height", "amp", "amplitude", "peak_height"])
    area_col = _find_column(df, ["area", "integral"])
    width_col = _find_column(df, ["width_hz", "width", "linewidth"])
    snr_col = _find_column(df, ["snr", "signal_to_noise"])

    if ppm_col is None or intensity_col is None:
        raise ValueError(
            f"Could not find required columns in peak list for sample {sample.sample_id}. "
            f"Need ppm and intensity/height columns. Found columns: {list(df.columns)}"
        )

    peaks: List[Peak] = []
    for _, row in df.iterrows():
        try:
            ppm = float(row[ppm_col])
            intensity = float(row[intensity_col])
        except Exception:
            continue
        peaks.append(
            Peak(
                peak_id=_next_peak_id(sample.sample_id),
                sample_id=sample.sample_id,
                ppm_raw=ppm,
                intensity=float(intensity),
                area=float(row[area_col]) if area_col and pd.notna(row[area_col]) else None,
                width_hz=float(row[width_col]) if width_col and pd.notna(row[width_col]) else None,
                snr=float(row[snr_col]) if snr_col and pd.notna(row[snr_col]) else None,
                ppm_corr=ppm,
            )
        )
    peaks.sort(key=lambda p: p.ppm_raw)
    return peaks


def load_peaklist_for_sample(sample: Sample) -> List[Peak]:
    if not sample.peaklist_path:
        raise ValueError(f"Sample {sample.sample_id} missing peaklist_path")
    path = Path(sample.peaklist_path)
    df = _read_table(path)
    return _to_peaks(df, sample)
