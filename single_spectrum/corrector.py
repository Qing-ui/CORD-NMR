from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PeakCorrector:
    ppm_bin_width: float = 9.75
    width_bin_width: float = 0.35

    @staticmethod
    def _prepare_work(df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if "shift" in work.columns and "ppm" not in work.columns:
            work["ppm"] = work["shift"]
        if "intensity" in work.columns and "height" not in work.columns:
            work["height"] = work["intensity"]
        if "sample" not in work.columns:
            work["sample"] = "sample_0"
        if "width" not in work.columns:
            work["width"] = 1.0
        work["width"] = pd.to_numeric(work["width"], errors="coerce").fillna(1.0).clip(lower=1e-6)
        work["ppm"] = pd.to_numeric(work["ppm"], errors="coerce")
        work["height"] = pd.to_numeric(work["height"], errors="coerce")
        work = work.dropna(subset=["ppm", "height"]).reset_index(drop=True)
        return work

    @staticmethod
    def _infer_target(
        work: pd.DataFrame,
        *,
        use_type: bool,
        use_area: bool,
        calibration_mode: str,
        shrink_strength: float,
    ) -> tuple[pd.Series, str, bool]:
        min_area_count = max(3, int(np.ceil(len(work) * 0.3)))
        has_area = "area" in work.columns and work["area"].notna().sum() >= min_area_count
        type_available = bool(use_type and "type" in work.columns and work["type"].notna().all())

        mode = (calibration_mode or "area_match").strip().lower()
        if mode not in {"area_match", "height_stabilize"}:
            raise ValueError("calibration_mode must be either 'area_match' or 'height_stabilize'.")

        if mode == "area_match":
            if use_area and has_area:
                y = np.log(work["area"].clip(lower=1e-8)) - np.log(work["height"].clip(lower=1e-8))
                return y, "area", True
            mode = "height_stabilize"

        if mode == "height_stabilize":
            shrink = float(shrink_strength)
            if not np.isfinite(shrink) or shrink < 0:
                raise ValueError("shrink_strength must be a finite non-negative number.")
            if type_available:
                ref = work.groupby(["sample", "type"])["height"].transform("median")
                fallback = work.groupby("sample")["height"].transform("median")
                ref = ref.fillna(fallback).clip(lower=1e-8)
                source = "height_sample_type_median"
            else:
                ref = work.groupby("sample")["height"].transform("median").clip(lower=1e-8)
                source = "height_sample_median"
            y = shrink * (np.log(ref) - np.log(work["height"].clip(lower=1e-8)))
            return y, source, False

        raise AssertionError("unreachable")

    def train(
        self,
        df: pd.DataFrame,
        use_type: bool = False,
        use_area: bool = True,
        calibration_mode: str = "height_stabilize",
        shrink_strength: float = 1.235,
        delta_clip: float | None = 0.63,
    ) -> dict[str, Any]:
        work = self._prepare_work(df)
        y, target_source, used_area = self._infer_target(
            work,
            use_type=use_type,
            use_area=use_area,
            calibration_mode=calibration_mode,
            shrink_strength=shrink_strength,
        )
        work["_delta_target"] = y

        ppm_min, ppm_max = float(work["ppm"].min()), float(work["ppm"].max())
        width_min, width_max = float(work["width"].min()), float(work["width"].max())
        ppm_edges = np.arange(ppm_min, ppm_max + self.ppm_bin_width, self.ppm_bin_width)
        width_edges = np.arange(width_min, width_max + self.width_bin_width, self.width_bin_width)
        if len(ppm_edges) < 2:
            ppm_edges = np.array([ppm_min, ppm_min + self.ppm_bin_width], dtype=float)
        if len(width_edges) < 2:
            width_edges = np.array([width_min, width_min + self.width_bin_width], dtype=float)

        work["_ppm_bin"] = pd.cut(work["ppm"], bins=ppm_edges, include_lowest=True, labels=False)
        work["_width_bin"] = pd.cut(work["width"], bins=width_edges, include_lowest=True, labels=False)

        global_bias = float(work["_delta_target"].median())
        work["_residual"] = work["_delta_target"] - global_bias

        ppm_bias = work.groupby("_ppm_bin")["_residual"].median().dropna().to_dict()
        work["_residual"] = work["_residual"] - work["_ppm_bin"].map(ppm_bias).fillna(0.0)

        width_bias = work.groupby("_width_bin")["_residual"].median().dropna().to_dict()
        work["_residual"] = work["_residual"] - work["_width_bin"].map(width_bias).fillna(0.0)

        type_bias: dict[str, float] = {}
        type_used = bool(use_type and "type" in work.columns and work["type"].notna().any())
        if type_used:
            by_type = work.groupby("type")["_residual"].median().dropna().to_dict()
            type_bias = {str(k): float(v) for k, v in by_type.items()}

        clip_value = None if delta_clip is None else float(delta_clip)
        if clip_value is not None and (not np.isfinite(clip_value) or clip_value <= 0):
            raise ValueError("delta_clip must be a positive finite number or None.")

        return {
            "version": 3,
            "mode": "with_type" if type_used else "basic",
            "used_area": bool(used_area),
            "used_type": type_used,
            "calibration_mode": calibration_mode,
            "shrink_strength": float(shrink_strength),
            "delta_clip": clip_value,
            "target_source": target_source,
            "global_bias": global_bias,
            "ppm_bin_width": self.ppm_bin_width,
            "width_bin_width": self.width_bin_width,
            "ppm_min": ppm_min,
            "width_min": width_min,
            "ppm_edges": [float(x) for x in ppm_edges],
            "width_edges": [float(x) for x in width_edges],
            "ppm_bias": {str(int(k)): float(v) for k, v in ppm_bias.items()},
            "width_bias": {str(int(k)): float(v) for k, v in width_bias.items()},
            "type_bias": type_bias,
        }

    @staticmethod
    def _predict_delta(work: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
        ppm_bias = model.get("ppm_bias", {})
        width_bias = model.get("width_bias", {})
        type_bias = model.get("type_bias", {})

        if "ppm_edges" in model:
            ppm_edges = np.asarray(model["ppm_edges"], dtype=float)
            ppm_bins = pd.cut(work["ppm"], bins=ppm_edges, include_lowest=True, labels=False)
        else:
            ppm_bw = float(model["ppm_bin_width"])
            ppm_min = float(model["ppm_min"])
            ppm_bins = np.floor((work["ppm"] - ppm_min) / ppm_bw).astype(int)
            if ppm_bias:
                valid_ppm = np.array(sorted(int(k) for k in ppm_bias.keys()), dtype=int)
                ppm_bins = np.clip(ppm_bins, valid_ppm.min(), valid_ppm.max())

        if "width_edges" in model:
            width_edges = np.asarray(model["width_edges"], dtype=float)
            width_bins = pd.cut(work["width"], bins=width_edges, include_lowest=True, labels=False)
        else:
            width_bw = float(model["width_bin_width"])
            width_min = float(model["width_min"])
            width_bins = np.floor((work["width"] - width_min) / width_bw).astype(int)
            if width_bias:
                valid_width = np.array(sorted(int(k) for k in width_bias.keys()), dtype=int)
                width_bins = np.clip(width_bins, valid_width.min(), valid_width.max())

        delta = np.full(len(work), float(model["global_bias"]), dtype=float)
        delta += np.array([float(ppm_bias.get(str(int(x)), 0.0)) if pd.notna(x) else 0.0 for x in ppm_bins])
        delta += np.array([float(width_bias.get(str(int(x)), 0.0)) if pd.notna(x) else 0.0 for x in width_bins])
        if "type" in work.columns and type_bias:
            delta += np.array([float(type_bias.get(str(x), 0.0)) for x in work["type"].astype(str)])
        clip_value = model.get("delta_clip")
        if clip_value is not None and str(model.get("calibration_mode", "area_match")).lower() == "height_stabilize":
            delta = np.clip(delta, -float(clip_value), float(clip_value))
        return delta

    def correct_table(self, df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
        work = self._prepare_work(df)
        delta = self._predict_delta(work, model)
        work["delta"] = delta
        work["height_corr"] = work["height"].clip(lower=1e-8) * np.exp(delta)
        return work
