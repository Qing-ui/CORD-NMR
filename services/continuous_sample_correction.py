from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - used only when PyYAML is unavailable.
    yaml = None

from single_spectrum.adapters import normalize_peak_table, read_generic_peak_table
from single_spectrum.corrector import PeakCorrector


@dataclass
class ContinuousCorrectionOptions:
    enabled: bool = False
    use_type: bool = False
    use_area: bool = True
    calibration_mode: str = "auto"
    shrink_strength: float = 1.235
    delta_clip: float | None = 0.63
    ppm_bin_width: float = 9.75
    width_bin_width: float = 0.35


@dataclass
class ContinuousCorrectionResult:
    samples: list[dict]
    model_path: Path | None = None
    summary_path: Path | None = None
    corrected_manifest_path: Path | None = None
    corrected_root: Path | None = None
    metadata: dict | None = None


def _sample_peak_path(sample: dict) -> Path:
    value = sample.get("peaklist_path") or sample.get("path")
    if not value:
        raise ValueError(f"Sample entry missing peaklist_path/path: {sample}")
    return Path(str(value))


def _normalize_sample_records(samples: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in samples:
        row = dict(item)
        if "sample_id" not in row:
            raise ValueError(f"Sample entry missing sample_id: {item}")
        row.setdefault("order_index", len(rows))
        row.setdefault("source_type", "csv")
        row["peaklist_path"] = str(_sample_peak_path(row))
        rows.append(row)
    return rows


def _load_training_frame(samples: list[dict]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    per_sample: dict[str, pd.DataFrame] = {}
    frames: list[pd.DataFrame] = []
    for sample in samples:
        sid = str(sample["sample_id"])
        df = normalize_peak_table(read_generic_peak_table(sample["peaklist_path"]), default_sample=sid)
        df["sample"] = sid
        per_sample[sid] = df
        frames.append(df)
    if not frames:
        raise ValueError("At least one sample is required for correction.")
    return pd.concat(frames, ignore_index=True), per_sample


def _resolved_calibration_mode(options: ContinuousCorrectionOptions, train_df: pd.DataFrame) -> str:
    if options.calibration_mode != "auto":
        return options.calibration_mode
    has_area = "area" in train_df.columns and train_df["area"].notna().sum() >= max(3, int(len(train_df) * 0.3))
    return "area_match" if options.use_area and has_area else "height_stabilize"


def _write_corrected_samples(
    *,
    samples: list[dict],
    per_sample: dict[str, pd.DataFrame],
    corrector: PeakCorrector,
    model: dict,
    corrected_root: Path,
) -> list[dict]:
    corrected_root.mkdir(parents=True, exist_ok=True)
    out_samples: list[dict] = []
    for sample in samples:
        sid = str(sample["sample_id"])
        corrected = corrector.correct_table(per_sample[sid], model)
        export = pd.DataFrame({
            "ppm": corrected["ppm"],
            "intensity": corrected["height_corr"],
            "area": corrected["area"] if "area" in corrected.columns else pd.NA,
            "width_hz": corrected["width"] if "width" in corrected.columns else pd.NA,
            "type": corrected["type"] if "type" in corrected.columns else pd.NA,
            "intensity_raw": corrected["height"],
            "delta": corrected["delta"],
        })
        if export["area"].isna().all():
            export = export.drop(columns=["area"])
        if "area" in corrected.columns and corrected["area"].notna().any():
            export["area_raw"] = corrected["area"]
        if export["width_hz"].isna().all():
            export = export.drop(columns=["width_hz"])
        if export["type"].isna().all():
            export = export.drop(columns=["type"])
        out_path = corrected_root / f"{sid}.csv"
        export.to_csv(out_path, index=False)
        new_sample = dict(sample)
        new_sample["peaklist_path"] = str(out_path)
        out_samples.append(new_sample)
    return out_samples


def maybe_prepare_corrected_samples(
    *,
    samples: Iterable[dict],
    runtime_dir: str | Path,
    options: ContinuousCorrectionOptions | dict | None,
) -> ContinuousCorrectionResult:
    sample_rows = _normalize_sample_records(samples)
    if options is None:
        options = ContinuousCorrectionOptions(enabled=False)
    elif isinstance(options, dict):
        options = ContinuousCorrectionOptions(**options)
    if not options.enabled:
        return ContinuousCorrectionResult(samples=sample_rows, metadata={"enabled": False})

    runtime_dir = Path(runtime_dir)
    corrected_root = runtime_dir / "corrected_inputs"
    train_df, per_sample = _load_training_frame(sample_rows)
    mode = _resolved_calibration_mode(options, train_df)
    corrector = PeakCorrector(ppm_bin_width=options.ppm_bin_width, width_bin_width=options.width_bin_width)
    model = corrector.train(
        train_df,
        use_type=options.use_type,
        use_area=options.use_area,
        calibration_mode=mode,
        shrink_strength=options.shrink_strength,
        delta_clip=options.delta_clip,
    )
    out_samples = _write_corrected_samples(
        samples=sample_rows,
        per_sample=per_sample,
        corrector=corrector,
        model=model,
        corrected_root=corrected_root,
    )
    model_path = runtime_dir / "correction_model.json"
    model_path.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path = runtime_dir / "corrected_manifest.yaml"
    manifest_payload = {"samples": out_samples}
    if yaml is not None:
        manifest_text = yaml.safe_dump(manifest_payload, sort_keys=False, allow_unicode=True)
    else:
        manifest_text = json.dumps(manifest_payload, indent=2, ensure_ascii=False)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    summary = {
        "enabled": True,
        "n_samples": len(out_samples),
        "n_rows": int(len(train_df)),
        "calibration_mode": mode,
        "used_area": bool(model.get("used_area", False)),
        "used_type": bool(model.get("used_type", False)),
        "target_source": str(model.get("target_source", "")),
        "options": asdict(options),
        "model_path": str(model_path),
        "corrected_manifest_path": str(manifest_path),
        "corrected_root": str(corrected_root),
    }
    summary_path = runtime_dir / "correction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return ContinuousCorrectionResult(
        samples=out_samples,
        model_path=model_path,
        summary_path=summary_path,
        corrected_manifest_path=manifest_path,
        corrected_root=corrected_root,
        metadata=summary,
    )
