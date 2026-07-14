from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .adapters import normalize_peak_table, read_generic_peak_table
from .corrector import PeakCorrector
from .dept import infer_dept_types
from .peak_gmm import run_peakgmm


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_model(model: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(model, indent=2), encoding="utf-8")


def load_model(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class SingleSpectrumCalibrationConfig:
    use_type: bool = False
    use_area: bool = False
    calibration_mode: str = "height_stabilize"
    shrink_strength: float = 1.235
    delta_clip: float | None = 0.63
    dept_tol: float = 0.8
    ppm_bin_width: float = 9.75
    width_bin_width: float = 0.35


@dataclass
class SingleSpectrumGMMConfig:
    min_components: int = 1
    max_components: int = 4
    use_log: bool = True


@dataclass
class SingleSpectrumPipelineConfig:
    calibration: SingleSpectrumCalibrationConfig = field(default_factory=SingleSpectrumCalibrationConfig)
    gmm: SingleSpectrumGMMConfig = field(default_factory=SingleSpectrumGMMConfig)


def _normalize_sample_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if "sample" not in work.columns:
        work["sample"] = "sample_0"
    raw = work["sample"]
    if raw.isna().any():
        raise ValueError("'sample' column contains missing values.")
    labels = raw.astype(str).str.strip()
    if labels.eq("").any() or labels.str.lower().isin({"nan", "none"}).any():
        raise ValueError("'sample' column contains missing or blank values.")
    unique_labels = list(pd.unique(labels))
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    work["original_sample"] = labels
    work["sample"] = labels.map(mapping).astype(int)
    mapping_df = pd.DataFrame({
        "sample": list(mapping.values()),
        "original_sample": list(mapping.keys()),
    })
    return work, mapping_df


def _prepare_table(path: str | Path) -> pd.DataFrame:
    return normalize_peak_table(read_generic_peak_table(path), default_sample=Path(path).stem)


def _resolve_input_mode(*, input_path: str | Path | None, allc_path: str | Path | None, d135_path: str | Path | None, d90_path: str | Path | None) -> str:
    if input_path and any([allc_path, d135_path, d90_path]):
        raise ValueError("Use either single-table mode or DEPT three-spectra mode, not both.")
    if input_path:
        return "single_table"
    if all([allc_path, d135_path, d90_path]):
        return "three_spectra"
    raise ValueError("Provide either one input table or all three DEPT files.")


def _load_training_dataframe(
    config: SingleSpectrumCalibrationConfig,
    *,
    input_path: str | Path | None = None,
    allc_path: str | Path | None = None,
    d135_path: str | Path | None = None,
    d90_path: str | Path | None = None,
    use_type_override: bool | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    mode = _resolve_input_mode(input_path=input_path, allc_path=allc_path, d135_path=d135_path, d90_path=d90_path)
    effective_use_type = config.use_type if use_type_override is None else bool(use_type_override)
    if mode == "single_table":
        df = _prepare_table(input_path)
        df, mapping = _normalize_sample_labels(df)
        if effective_use_type and "type" not in df.columns:
            raise ValueError("Model requires type-aware correction, but input table has no usable 'type' column.")
        source_stem = Path(input_path).with_suffix("").as_posix()
        return mode, df.copy(), mapping, source_stem

    allc = _prepare_table(allc_path)
    d135 = _prepare_table(d135_path)
    d90 = _prepare_table(d90_path)
    allc, mapping = _normalize_sample_labels(allc)
    # Keep shared numeric sample ids across DEPT tables.
    id_map = {row.original_sample: int(row.sample) for row in mapping.itertuples(index=False)}
    d135["sample"] = d135["sample"].astype(str).map(id_map)
    d90["sample"] = d90["sample"].astype(str).map(id_map)
    d135 = d135.dropna(subset=["sample"]).copy(); d135["sample"] = d135["sample"].astype(int)
    d90 = d90.dropna(subset=["sample"]).copy(); d90["sample"] = d90["sample"].astype(int)
    training_df = infer_dept_types(allc, d135, d90, tol_ppm=config.dept_tol) if effective_use_type else allc.copy()
    source_stem = Path(allc_path).with_suffix("").as_posix()
    return mode, training_df, mapping, source_stem


def train_intensity_correction_model(
    input_path=None,
    *,
    allc_path=None,
    d135_path=None,
    d90_path=None,
    config: SingleSpectrumCalibrationConfig | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    cfg = config or SingleSpectrumCalibrationConfig()
    mode, training_df, mapping, source_stem = _load_training_dataframe(
        cfg, input_path=input_path, allc_path=allc_path, d135_path=d135_path, d90_path=d90_path
    )
    corr = PeakCorrector(ppm_bin_width=cfg.ppm_bin_width, width_bin_width=cfg.width_bin_width)
    model = corr.train(
        training_df,
        use_type=cfg.use_type,
        use_area=cfg.use_area,
        calibration_mode=cfg.calibration_mode,
        shrink_strength=cfg.shrink_strength,
        delta_clip=cfg.delta_clip,
    )
    result = {
        "mode": mode,
        "n_rows": int(len(training_df)),
        "n_samples": int(training_df["sample"].nunique()) if "sample" in training_df.columns else 1,
        "used_type": bool(model.get("used_type", False)),
        "used_area": bool(model.get("used_area", False)),
        "config": asdict(cfg),
        "model": model,
        "sample_mapping": mapping,
        "used_type_column": bool("type" in training_df.columns and training_df["type"].notna().any()),
    }
    if out_dir is not None:
        out = _ensure_dir(out_dir)
        save_model(model, out / "model.json")
        mapping.to_csv(out / "sample_id_mapping.csv", index=False)
        hidden = training_df.copy()
        if "original_sample" in hidden.columns:
            hidden = hidden.drop(columns=["original_sample"])
        hidden.to_csv(out / "_training_input_prepared.csv", index=False)
        _write_json(out / "training_summary.json", {k: v for k, v in result.items() if k not in {"model", "sample_mapping"}})
    return result


def apply_intensity_correction_model(
    input_path=None,
    *,
    model_path,
    out_dir=None,
    allc_path=None,
    d135_path=None,
    d90_path=None,
    config: SingleSpectrumCalibrationConfig | None = None,
) -> dict:
    cfg = config or SingleSpectrumCalibrationConfig()
    model = load_model(model_path)
    required_type = bool(model.get("used_type", False))
    mode, apply_df, mapping, source_stem = _load_training_dataframe(
        cfg,
        input_path=input_path,
        allc_path=allc_path,
        d135_path=d135_path,
        d90_path=d90_path,
        use_type_override=required_type,
    )
    corr = PeakCorrector(
        ppm_bin_width=float(model.get("ppm_bin_width", cfg.ppm_bin_width)),
        width_bin_width=float(model.get("width_bin_width", cfg.width_bin_width)),
    )
    corrected = corr.correct_table(apply_df, model)
    final_out = corrected[["sample", "ppm", "height_corr"]].rename(columns={"ppm": "shift", "height_corr": "intensity"})
    summary = {
        "mode": mode,
        "n_rows": int(len(corrected)),
        "n_samples": int(corrected["sample"].nunique()),
        "model_path": str(model_path),
        "used_type_column": bool("type" in apply_df.columns and apply_df["type"].notna().any()),
    }
    if out_dir is not None:
        out = _ensure_dir(out_dir)
        final_out.to_csv(out / "corrected_1d_for_clustering.csv", index=False)
        corrected.to_csv(out / "corrected_peaks.csv", index=False)
        mapping.to_csv(out / "sample_id_mapping.csv", index=False)
        hidden = corrected.copy()
        if "original_sample" in hidden.columns:
            hidden = hidden.drop(columns=["original_sample"])
        hidden.to_csv(out / "_calibration_corrected_detailed.csv", index=False)
        _write_json(out / "correction_summary.json", summary)
    return {"summary": summary, "corrected": corrected, "model": model, "final_output": final_out, "sample_mapping": mapping}


def run_calibration_pipeline(
    config: SingleSpectrumCalibrationConfig,
    *,
    input_path: str | Path | None = None,
    allc_path: str | Path | None = None,
    d135_path: str | Path | None = None,
    d90_path: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    out = _ensure_dir(out_dir) if out_dir is not None else None
    train_result = train_intensity_correction_model(
        input_path=input_path,
        allc_path=allc_path,
        d135_path=d135_path,
        d90_path=d90_path,
        config=config,
        out_dir=out,
    )
    model_path = (out / "model.json") if out is not None else Path("model.json")
    apply_result = apply_intensity_correction_model(
        input_path=input_path,
        allc_path=allc_path,
        d135_path=d135_path,
        d90_path=d90_path,
        model_path=model_path,
        out_dir=out,
        config=config,
    )
    return {
        **apply_result,
        "model": train_result["model"],
        "config": asdict(config),
        "used_type_column": apply_result["summary"]["used_type_column"],
    }


def run_single_spectrum_gmm(input_path, *, out_dir=None, min_components=1, max_components=4, use_log=True) -> dict:
    df = normalize_peak_table(read_generic_peak_table(input_path))
    detailed, clusters = run_peakgmm(df, min_components=min_components, max_components=max_components, use_log=use_log)
    summary = {
        "n_rows": int(len(detailed)),
        "n_samples": int(detailed["sample"].nunique()) if not detailed.empty else 0,
        "n_clusters": int(len(clusters)),
        "config": {"min_components": min_components, "max_components": max_components, "use_log": use_log},
    }
    if out_dir is not None:
        out = _ensure_dir(out_dir)
        detailed.to_csv(out / "single_spectrum_gmm_detailed.csv", index=False)
        clusters.to_csv(out / "single_spectrum_gmm_clusters.csv", index=False)
        _write_json(out / "single_spectrum_gmm_summary.json", summary)
    return {"summary": summary, "detailed": detailed, "clusters": clusters}


def run_single_spectrum_pipeline(input_path, *, out_dir, config: SingleSpectrumPipelineConfig | None = None, model_path=None) -> dict:
    cfg = config or SingleSpectrumPipelineConfig()
    out = _ensure_dir(out_dir)
    df = normalize_peak_table(read_generic_peak_table(input_path))
    if model_path is None:
        train_result = train_intensity_correction_model(input_path=input_path, config=cfg.calibration, out_dir=out / "calibration_train")
        model_path = out / "calibration_train" / "model.json"
    else:
        train_result = None
    apply_result = apply_intensity_correction_model(input_path=input_path, model_path=model_path, out_dir=out / "calibration_apply", config=cfg.calibration)
    corrected = apply_result["corrected"]
    detailed, clusters = run_peakgmm(corrected, min_components=cfg.gmm.min_components, max_components=cfg.gmm.max_components, use_log=cfg.gmm.use_log)
    detailed.to_csv(out / "single_spectrum_pipeline_detailed.csv", index=False)
    clusters.to_csv(out / "single_spectrum_pipeline_clusters.csv", index=False)
    summary = {
        "input_rows": int(len(df)),
        "corrected_rows": int(len(corrected)),
        "n_samples": int(corrected["sample"].nunique()),
        "n_clusters": int(len(clusters)),
        "calibration_model_path": str(model_path),
        "train_mode": None if train_result is None else train_result["mode"],
        "gmm": asdict(cfg.gmm),
        "calibration": asdict(cfg.calibration),
    }
    _write_json(out / "single_spectrum_pipeline_summary.json", summary)
    return {"summary": summary, "corrected": corrected, "detailed": detailed, "clusters": clusters, "train_result": train_result}
