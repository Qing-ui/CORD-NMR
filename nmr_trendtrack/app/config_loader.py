from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from nmr_trendtrack.config import AppConfig, AlignConfig, TrendConfig, ClusterConfig, OptimizeConfig, ModelConfig
from nmr_trendtrack.contracts import Sample


def _as_tuple_list(items: List[List[float]]) -> List[tuple[float, float, float]]:
    return [tuple(map(float, row)) for row in items]


def _int_key_dict(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except Exception:
            out[k] = v
    return out


def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    align_data = data.get("align", {})
    if "ppm_window_by_region" in align_data:
        align_data = dict(align_data)
        align_data["ppm_window_by_region"] = _as_tuple_list(align_data["ppm_window_by_region"])
    model_data = dict(data.get("model", {}))
    if "pmtc_max_tracks_by_n_samples" in model_data:
        model_data["pmtc_max_tracks_by_n_samples"] = _int_key_dict(model_data["pmtc_max_tracks_by_n_samples"])
    if "pmtc_frac_limit_by_n_samples" in model_data:
        model_data["pmtc_frac_limit_by_n_samples"] = _int_key_dict(model_data["pmtc_frac_limit_by_n_samples"])
    return AppConfig(
        align=AlignConfig(**align_data),
        trend=TrendConfig(**data.get("trend", {})),
        cluster=ClusterConfig(**data.get("cluster", {})),
        optimize=OptimizeConfig(**data.get("optimize", {})),
        model=ModelConfig(**model_data),
    )


def load_manifest(path: str | Path) -> List[Sample]:
    manifest_path = Path(path)
    manifest: Dict[str, Any] = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    samples = []
    for item in manifest.get("samples", []):
        peaklist_path = item.get("peaklist_path")
        spectrum_path = item.get("spectrum_path")
        if peaklist_path is not None:
            peak_path = Path(peaklist_path)
            if peak_path.is_absolute():
                peaklist_path = str(peak_path)
            else:
                joined = (manifest_path.parent / peak_path).resolve()
                alt_joined = (manifest_path.parent.parent / peak_path).resolve()
                if joined.exists():
                    peaklist_path = str(joined)
                elif alt_joined.exists():
                    peaklist_path = str(alt_joined)
                else:
                    peaklist_path = str(peak_path)
        if spectrum_path is not None:
            spec_path = Path(spectrum_path)
            if spec_path.is_absolute():
                spectrum_path = str(spec_path)
            else:
                joined = (manifest_path.parent / spec_path).resolve()
                alt_joined = (manifest_path.parent.parent / spec_path).resolve()
                if joined.exists():
                    spectrum_path = str(joined)
                elif alt_joined.exists():
                    spectrum_path = str(alt_joined)
                else:
                    spectrum_path = str(spec_path)
        samples.append(
            Sample(
                sample_id=str(item["sample_id"]),
                order_index=int(item["order_index"]),
                source_type=str(item.get("source_type", "csv")),
                peaklist_path=peaklist_path,
                spectrum_path=spectrum_path,
                meta=dict(item.get("meta", {})),
            )
        )
    return sorted(samples, key=lambda s: s.order_index)
