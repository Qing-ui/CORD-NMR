from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
DEFAULT_NMR_PREDICTOR_ROOT = PROJECT_ROOT / "external" / "NMR-Predictor-Portable"


@dataclass(frozen=True)
class NMRPredictorStatus:
    root: Path
    script: Path
    nmrnet_python: Path
    cascade2_python: Path
    missing_required: List[str]
    missing_optional: List[str]

    @property
    def ready_for_nmrnet(self) -> bool:
        return not self.missing_required

    @property
    def ready_for_cascade2(self) -> bool:
        return not self.missing_required and not self.missing_optional


@dataclass(frozen=True)
class NMRPredictionLaunch:
    command: List[str]
    cwd: Path
    env: Dict[str, str]
    output_dir: Path
    expected_final_sdf: Path


@dataclass(frozen=True)
class SDFAnnotationResult:
    path: Path
    molecule_count: int


def default_output_dir(input_path: str | Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(input_path).stem or "nmr_prediction"
    return PROJECT_ROOT / "results" / "nmr_prediction_runs" / f"{stamp}_{stem}"


def detect_input_type(input_path: str | Path) -> str:
    suffix = Path(input_path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".sdf", ".sd"}:
        return "sdf"
    raise ValueError("Input file must be SDF or CSV.")


def predictor_paths(root: str | Path | None = None) -> tuple[Path, Path, Path, Path]:
    root_path = Path(root) if root else DEFAULT_NMR_PREDICTOR_ROOT
    root_path = root_path.expanduser()
    script = root_path / "app" / "script" / "predict_nmr_unified.py"
    nmrnet_python = root_path / "envs" / "nmrnet" / "python.exe"
    cascade2_python = root_path / "envs" / "cascade2" / "python.exe"
    return root_path, script, nmrnet_python, cascade2_python


def describe_nmr_predictor_root(root: str | Path | None = None) -> NMRPredictorStatus:
    root_path, script, nmrnet_python, cascade2_python = predictor_paths(root)
    missing_required: List[str] = []
    missing_optional: List[str] = []
    if not script.exists():
        missing_required.append(str(script))
    if not nmrnet_python.exists():
        missing_required.append(str(nmrnet_python))
    if not cascade2_python.exists():
        missing_optional.append(str(cascade2_python))
    return NMRPredictorStatus(
        root=root_path,
        script=script,
        nmrnet_python=nmrnet_python,
        cascade2_python=cascade2_python,
        missing_required=missing_required,
        missing_optional=missing_optional,
    )


def annotate_prediction_sdf(path: str | Path) -> SDFAnnotationResult:
    """Add sequential ID and formula-weight fields to a predicted SDF file."""
    sdf_path = Path(path).expanduser()
    if not sdf_path.exists():
        raise FileNotFoundError(f"Predicted SDF does not exist: {sdf_path}")

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    molecules = [Chem.Mol(mol) for mol in supplier if mol is not None]
    del supplier
    if not molecules:
        raise ValueError(f"No valid molecules found in predicted SDF: {sdf_path}")

    suffix = "".join(sdf_path.suffixes) or ".sdf"
    tmp_path = sdf_path.with_name(f"{sdf_path.stem}.annotating{suffix}")
    writer = Chem.SDWriter(str(tmp_path))
    try:
        for index, mol in enumerate(molecules, start=1):
            mol.SetProp("ID", str(index))
            mol.SetProp("FW", f"{Descriptors.MolWt(mol):.4f}")
            writer.write(mol)
    finally:
        writer.close()
    tmp_path.replace(sdf_path)
    return SDFAnnotationResult(path=sdf_path, molecule_count=len(molecules))


def build_nmr_prediction_launch(
    *,
    root: str | Path | None,
    input_path: str | Path,
    output_dir: str | Path,
    mode: str = "CH",
    input_type: Optional[str] = None,
    c_engine: str = "nmrnet",
    max_conformers: int = 9,
    max_iters: int = 300,
    forcefield: str = "auto",
    time_limit_seconds: float = 20.0,
    coord_route: str = "standard",
    route_initial_confs: int = 27,
    route_prune_rms_thresh: float = 0.5,
    route_coarse_steps: int = 10,
    route_keep_top_k: int = 9,
    route_fine_steps: int = 300,
    optimize_existing: bool = True,
    allow_2d_if_h_nonzero: bool = True,
    smiles_column: str = "smiles",
    id_column: str = "id",
    flush_every: int = 10,
    cascade_batch_size: int = 32,
) -> NMRPredictionLaunch:
    mode = str(mode).upper()
    if mode not in {"C", "H", "CH"}:
        raise ValueError("Prediction mode must be C, H, or CH.")
    c_engine = str(c_engine).lower()
    if c_engine not in {"nmrnet", "cascade2"}:
        raise ValueError("C engine must be nmrnet or cascade2.")
    forcefield = str(forcefield).lower()
    if forcefield not in {"auto", "mmff", "uff"}:
        raise ValueError("Forcefield must be auto, mmff, or uff.")
    coord_route = str(coord_route).lower()
    if coord_route not in {"standard", "staged27"}:
        raise ValueError("3D route must be standard or staged27.")

    input_file = Path(input_path).expanduser()
    if not input_file.exists():
        raise ValueError(f"Input file does not exist: {input_file}")
    resolved_input_type = (input_type or detect_input_type(input_file)).lower()
    if resolved_input_type not in {"sdf", "csv"}:
        raise ValueError("Input type must be sdf or csv.")

    status = describe_nmr_predictor_root(root)
    missing = list(status.missing_required)
    if mode in {"C", "CH"} and c_engine == "cascade2" and status.missing_optional:
        missing.extend(status.missing_optional)
    if missing:
        raise FileNotFoundError(
            "NMR predictor portable root is incomplete. Missing:\n" + "\n".join(missing)
        )

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(status.nmrnet_python),
        str(status.script),
        "--mode",
        mode,
        "--input-type",
        resolved_input_type,
        "--input",
        str(input_file),
        "--output-dir",
        str(out_dir),
        "--c-engine",
        c_engine,
        "--h-engine",
        "nmrnet",
        "--smiles-column",
        str(smiles_column or "smiles"),
        "--id-column",
        str(id_column or "id"),
        "--max-conformers",
        str(int(max_conformers)),
        "--max-iters",
        str(int(max_iters)),
        "--forcefield",
        forcefield,
        "--flush-every",
        str(int(flush_every)),
        "--time-limit-seconds",
        str(float(time_limit_seconds)),
        "--coord-route",
        coord_route,
        "--route-initial-confs",
        str(int(route_initial_confs)),
        "--route-prune-rms-thresh",
        str(float(route_prune_rms_thresh)),
        "--route-coarse-steps",
        str(int(route_coarse_steps)),
        "--route-keep-top-k",
        str(int(route_keep_top_k)),
        "--route-fine-steps",
        str(int(route_fine_steps)),
        "--cascade-batch-size",
        str(int(cascade_batch_size)),
    ]
    command.append("--optimize-existing" if optimize_existing else "--no-optimize-existing")
    command.append("--allow-2d-if-h-nonzero" if allow_2d_if_h_nonzero else "--no-allow-2d-if-h-nonzero")

    env = dict(os.environ)
    env["NMR_PREDICTOR_HOME"] = str(status.root)
    env["NMRNET_PYTHON"] = str(status.nmrnet_python)
    env["CASCADE2_PYTHON"] = str(status.cascade2_python)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_MAX_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    final_name = {"C": "predicted_C.sdf", "H": "predicted_H.sdf", "CH": "predicted_CH.sdf"}[mode]
    return NMRPredictionLaunch(
        command=command,
        cwd=status.root / "app",
        env=env,
        output_dir=out_dir,
        expected_final_sdf=out_dir / final_name,
    )
