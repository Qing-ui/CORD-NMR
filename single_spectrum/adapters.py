from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd

COLUMN_ALIASES = {
    "sample": ("sample", "sample_id", "spec", "spectrum"),
    "ppm": ("ppm", "shift", "delta", "chemical_shift", "peak_ppm", "x", "ppm1", "cppm", "c_ppm", "c ppm", "carbon_ppm", "13c_ppm"),
    "height": ("height", "intensity", "amp", "amplitude", "peak_height", "y"),
    "width": ("width", "width_ppm", "fwhm", "peak_width"),
    "area": ("area", "integral"),
    "type": ("type", "peak_type", "dept_type"),
}


def _find_column(df: pd.DataFrame, logical_name: str) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES[logical_name]:
        if alias.lower() in lowered:
            return str(lowered[alias.lower()])
    return None


def normalize_peak_table(df: pd.DataFrame, default_sample: str = "sample_0") -> pd.DataFrame:
    work = df.copy()
    rename_map: dict[str, str] = {}
    for logical in ("sample", "ppm", "height", "width", "area", "type"):
        src = _find_column(work, logical)
        if src and src != logical:
            rename_map[src] = logical
    if rename_map:
        work = work.rename(columns=rename_map)
    if "sample" not in work.columns:
        work["sample"] = default_sample
    if "ppm" not in work.columns:
        raise ValueError("Peak table requires a ppm/shift column.")
    if "height" not in work.columns:
        raise ValueError("Peak table requires a height/intensity column.")
    if "width" not in work.columns:
        work["width"] = 1.0
    work["sample"] = work["sample"].astype(str)
    work["ppm"] = pd.to_numeric(work["ppm"], errors="coerce")
    work["height"] = pd.to_numeric(work["height"], errors="coerce")
    work["width"] = pd.to_numeric(work["width"], errors="coerce").fillna(1.0)
    if "area" in work.columns:
        work["area"] = pd.to_numeric(work["area"], errors="coerce")
    if "type" in work.columns:
        work["type"] = work["type"].astype(str)
    return work.dropna(subset=["ppm", "height"]).reset_index(drop=True)


def _looks_like_data_header(df: pd.DataFrame) -> bool:
    for col in df.columns:
        try:
            float(str(col).strip())
        except Exception:
            return False
    return True


def _read_numeric_text_table(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith(("#", ";", "//")):
            continue
        vals = []
        for tok in re.split(r"[,;\t\s]+", line.strip()):
            try:
                vals.append(float(tok))
            except Exception:
                pass
        if vals:
            rows.append(vals)
    if not rows:
        raise ValueError(f"Unsupported or empty peak table: {path}")
    width = max(len(r) for r in rows)
    names = ["ppm", "height", "area", "width"][:width]
    names.extend(f"col{i + 1}" for i in range(len(names), width))
    return pd.DataFrame([r + [None] * (width - len(r)) for r in rows], columns=names)


def read_generic_peak_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if _looks_like_data_header(df):
            df = _read_numeric_text_table(path)
    elif suffix in {".tsv", ".txt"}:
        try:
            df = pd.read_csv(path, sep="\t")
            if _looks_like_data_header(df) or not any(_find_column(df, name) for name in ("ppm", "height")):
                df = _read_numeric_text_table(path)
        except Exception:
            df = _read_numeric_text_table(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".xml":
        root = ET.parse(path).getroot()
        rows = []
        for node in root.findall('.//Peak1D'):
            rows.append({
                "sample": path.stem,
                "ppm": float(node.attrib.get("F1", "nan")),
                "height": float(node.attrib.get("intensity", "0")),
                "type": node.attrib.get("type", ""),
                "width": float(node.attrib.get("width", "1.0") or "1.0"),
            })
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported peak table format: {path}")
    return normalize_peak_table(df, default_sample=path.stem)
