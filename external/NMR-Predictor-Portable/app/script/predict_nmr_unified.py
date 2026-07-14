from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

import shared_rdkit_prep as prep


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(os.environ["NMR_PREDICTOR_HOME"]) if "NMR_PREDICTOR_HOME" in os.environ else REPO_ROOT.parent
NMRNET_PYTHON = Path(os.environ["NMRNET_PYTHON"]) if "NMRNET_PYTHON" in os.environ else (BUNDLE_ROOT / "envs" / "nmrnet" / "python.exe")
CASCADE2_PYTHON = Path(os.environ["CASCADE2_PYTHON"]) if "CASCADE2_PYTHON" in os.environ else (BUNDLE_ROOT / "envs" / "cascade2" / "python.exe")


@dataclass
class MergeSample:
    molecule_id: str
    mol: Chem.Mol
    original_props: dict[str, str]


def run_command(command: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def safe_stem(name: str) -> str:
    raw = str(name).strip()
    if raw.lower().endswith(".sdf"):
        raw = Path(raw).stem
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", raw).strip()
    return cleaned or "molecule"


def run_nmrnet(args, output_dir: Path, nucleus: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(NMRNET_PYTHON),
        str(REPO_ROOT / "script" / "predict_liquid_batch.py"),
        "--input-type",
        args.input_type,
        "--input",
        str(args.input),
        "--output-dir",
        str(output_dir),
        "--nucleus",
        nucleus,
        "--max-conformers",
        str(args.max_conformers),
        "--max-iters",
        str(args.max_iters),
        "--forcefield",
        args.forcefield,
        "--flush-every",
        str(args.flush_every),
        "--time-limit-seconds",
        str(args.time_limit_seconds),
        "--coord-route",
        args.coord_route,
        "--route-initial-confs",
        str(args.route_initial_confs),
        "--route-prune-rms-thresh",
        str(args.route_prune_rms_thresh),
        "--route-coarse-steps",
        str(args.route_coarse_steps),
        "--route-keep-top-k",
        str(args.route_keep_top_k),
        "--route-fine-steps",
        str(args.route_fine_steps),
        "--write-combined",
    ]
    if args.prepared_cache_sdf is not None:
        command += ["--prepared-cache-sdf", str(args.prepared_cache_sdf)]
    if args.input_type == "csv":
        command += ["--id-column", args.id_column, "--smiles-column", args.smiles_column]
    command.append("--optimize-existing-coordinates" if args.optimize_existing else "--no-optimize-existing-coordinates")
    command.append("--allow-2d-if-h-nonzero" if args.allow_2d_if_h_nonzero else "--no-allow-2d-if-h-nonzero")
    run_command(command, REPO_ROOT)
    return output_dir / "annotated_predictions.sdf"


def run_cascade2(args, output_dir: Path) -> Path:
    if args.input_type != "sdf" and args.prepared_cache_sdf is None:
        raise ValueError("CASCADE-2.0 路线处理 CSV 时需要先生成共享 3D cache SDF。")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(CASCADE2_PYTHON),
        str(REPO_ROOT / "script" / "cascade2_predict_sdf.py"),
        "--input",
        str(args.prepared_cache_sdf if args.prepared_cache_sdf is not None else args.input),
        "--output-dir",
        str(output_dir),
        "--max-conformers",
        str(args.max_conformers),
        "--max-iters",
        str(args.max_iters),
        "--forcefield",
        args.forcefield,
        "--batch-size",
        str(args.cascade_batch_size),
        "--time-limit-seconds",
        str(args.time_limit_seconds),
        "--coord-route",
        args.coord_route,
        "--route-initial-confs",
        str(args.route_initial_confs),
        "--route-prune-rms-thresh",
        str(args.route_prune_rms_thresh),
        "--route-coarse-steps",
        str(args.route_coarse_steps),
        "--route-keep-top-k",
        str(args.route_keep_top_k),
        "--route-fine-steps",
        str(args.route_fine_steps),
    ]
    if args.prepared_cache_sdf is not None:
        command += ["--prepared-cache-sdf", str(args.prepared_cache_sdf)]
    if args.optimize_existing:
        command.append("--optimize-existing-coordinates")
    run_command(command, REPO_ROOT)
    return output_dir / "cascade2_annotated_predictions.sdf"


def count_valid_sdf_records(sdf_path: Path) -> int:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    return sum(1 for mol in supplier if mol is not None)


def iter_sdf_records(sdf_path: Path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    base_name = sdf_path.stem
    valid_index = 0
    for index, mol in enumerate(supplier):
        if mol is None:
            continue
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            raw_name = mol.GetProp("_Name").strip()
            molecule_id = f"{base_name}_{index + 1:04d}_{raw_name}"
        else:
            molecule_id = f"{base_name}_{index + 1:04d}"
        smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
        yield valid_index, molecule_id, mol, smiles
        valid_index += 1


def build_prepared_cache(args, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache_path.unlink()
    failed_rows: list[str] = []
    if args.input_type == "csv":
        raw = prep.load_csv_molecules(args.input, args.smiles_column, args.id_column)
        source = "csv_smiles"
        total_count = len(raw)
        raw_iter = ((index, molecule_id, mol, smiles) for index, (molecule_id, mol, smiles) in enumerate(raw))
    else:
        source = "sdf"
        total_count = count_valid_sdf_records(args.input)
        raw_iter = iter_sdf_records(args.input)
    print(f"building shared 3D cache: {cache_path}", flush=True)
    print(f"loaded records for cache: {total_count}", flush=True)
    writer = Chem.SDWriter(str(cache_path))
    written = 0
    chunk_written = 0
    try:
        for index, molecule_id, mol, smiles in raw_iter:
            current_index = index + 1
            if chunk_written == 0:
                chunk_end = min(chunk_written + args.flush_every + written, total_count)
                print(f"cache chunk {written + 1}-{chunk_end}/{total_count}", flush=True)
            if index == 0 or current_index % 25 == 0:
                print(f"cache preparing {current_index}/{total_count}: {molecule_id}", flush=True)
            try:
                prepared = prep.prepare_single_molecule(
                    molecule_id=molecule_id,
                    mol=mol,
                    source=source,
                    original_smiles=smiles,
                    random_seed=42 + index,
                    max_iters=args.max_iters,
                    forcefield=args.forcefield,
                    max_conformers=args.max_conformers,
                    time_limit_seconds=args.time_limit_seconds,
                    prefer_existing_coordinates=(args.input_type == "sdf"),
                    allow_2d_if_h_nonzero=args.allow_2d_if_h_nonzero,
                    optimize_existing_coordinates=args.optimize_existing,
                    coord_route=args.coord_route,
                    route_initial_confs=args.route_initial_confs,
                    route_prune_rms_thresh=args.route_prune_rms_thresh,
                    route_coarse_steps=args.route_coarse_steps,
                    route_keep_top_k=args.route_keep_top_k,
                    route_fine_steps=args.route_fine_steps,
                )
                writer.write(prep.prepared_molecule_to_cache_mol(prepared))
                written += 1
            except Exception as exc:
                failed_rows.append(f"{molecule_id}\t{type(exc).__name__}\t{exc}")
            chunk_written += 1
            if chunk_written >= args.flush_every:
                chunk_written = 0
                gc.collect()
        gc.collect()
    finally:
        writer.close()
        if args.input_type == "csv":
            del raw
        gc.collect()
    if failed_rows:
        failed_path = args.output_dir / "prepared_cache_failed.tsv"
        with failed_path.open("w", encoding="utf-8") as handle:
            handle.write("molecule_id\terror_type\terror_message\n")
            handle.write("\n".join(failed_rows))
            handle.write("\n")
    print(f"shared 3D cache saved: {cache_path} ({written} molecules)", flush=True)
    return cache_path


def load_sdf_map(path: Path) -> dict[str, MergeSample]:
    samples: dict[str, MergeSample] = {}
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if not mol_id:
            continue
        props = {name: mol.GetProp(name) for name in mol.GetPropNames()}
        samples[mol_id] = MergeSample(molecule_id=mol_id, mol=Chem.Mol(mol), original_props=props)
    return samples


def load_nmrnet_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"atom_symbol": "atom_type", "predicted_shift_ppm": "shift_ppm"})
    if "atom_index_explicit_h_0based" in df.columns:
        df["atom_index"] = df["atom_index_explicit_h_0based"]
    return df


def load_cascade_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.rename(columns={"cascade2_13c_shift_ppm": "shift_ppm"})


def build_properties(sample: MergeSample, group: pd.DataFrame, mode: str) -> dict[str, str]:
    props: dict[str, str] = {}
    if mode in {"C", "CH"}:
        carbon_rows = group[group["atom_type"] == "C"].copy().sort_values("atom_index").reset_index(drop=True)
        predicted_c13_lines = []
        quaternaries: list[str] = []
        tertiaries: list[str] = []
        secondaries: list[str] = []
        primaries: list[str] = []
        for serial, row in carbon_rows.iterrows():
            atom_index = int(row["atom_index"])
            atom_index_1based = atom_index + 1
            shift = float(row["shift_ppm"])
            predicted_c13_lines.append(f"{serial}[{atom_index_1based}]\t{shift:.2f}")
            atom = sample.mol.GetAtomWithIdx(atom_index)
            hydrogen_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
            line = f"{atom_index_1based}\t{shift:.2f}"
            if hydrogen_count == 0:
                quaternaries.append(line)
            elif hydrogen_count == 1:
                tertiaries.append(line)
            elif hydrogen_count == 2:
                secondaries.append(line)
            elif hydrogen_count == 3:
                primaries.append(line)
        props["Predicted 13C shifts"] = "\n".join(predicted_c13_lines)
        props["Quaternaries"] = "\n".join(quaternaries)
        props["Tertiaries"] = "\n".join(tertiaries)
        props["Secondaries"] = "\n".join(secondaries)
        props["Primaries"] = "\n".join(primaries)

    if mode in {"H", "CH"}:
        hydrogen_rows = group[group["atom_type"] == "H"].copy().sort_values("atom_index").reset_index(drop=True)
        hydrogen_lines: list[str] = []
        attached_counter: dict[int, int] = {}
        for _, row in hydrogen_rows.iterrows():
            h_atom_index = int(row["atom_index"])
            h_atom_index_1based = h_atom_index + 1
            shift = float(row["shift_ppm"])
            h_atom = sample.mol.GetAtomWithIdx(h_atom_index)
            neighbors = [n.GetIdx() for n in h_atom.GetNeighbors()]
            attached_index = neighbors[0] if neighbors else h_atom_index
            suffix_idx = attached_counter.get(attached_index, 0)
            suffix = chr(ord("a") + suffix_idx)
            attached_counter[attached_index] = suffix_idx + 1
            hydrogen_lines.append(f"{attached_index + 1}{suffix}[{h_atom_index_1based}]\t{shift:.2f}")
        props["HydrogenShifts"] = "\n".join(hydrogen_lines)
    return props


def write_outputs(base_samples: dict[str, MergeSample], merged_df: pd.DataFrame, output_dir: Path, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_writer = Chem.SDWriter(str(output_dir / "annotated_predictions.sdf"))
    for output_index, (molecule_id, sample) in enumerate(base_samples.items(), start=1):
        group = merged_df[merged_df["molecule_id"] == molecule_id].copy()
        props = build_properties(sample, group, mode)
        out_mol = Chem.Mol(sample.mol)
        prep.strip_cache_properties(out_mol)
        prep.strip_embedding_properties(out_mol)
        for key, value in sample.original_props.items():
            out_mol.SetProp(key, str(value))
        out_mol.SetProp("_Name", molecule_id)
        out_mol.SetProp("ID", str(output_index))
        out_mol.SetProp("FW", f"{Descriptors.MolWt(out_mol):.4f}")
        for key, value in props.items():
            out_mol.SetProp(key, value)
        combined_writer.write(out_mol)

        simple = group[["atom_index", "atom_type", "shift_ppm"]].copy().sort_values(["atom_index", "atom_type"])
        simple.to_csv(output_dir / f"{safe_stem(molecule_id)}.csv", index=False, encoding="utf-8-sig")
        writer = Chem.SDWriter(str(output_dir / f"{safe_stem(molecule_id)}.sdf"))
        writer.write(out_mol)
        writer.close()
    combined_writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified NMR entry for NMRNet and CASCADE-2.0.")
    parser.add_argument("--mode", choices=["C", "H", "CH"], required=True)
    parser.add_argument("--input-type", choices=["sdf", "csv"], default="sdf")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c-engine", choices=["nmrnet", "cascade2"], default="nmrnet")
    parser.add_argument("--h-engine", choices=["nmrnet"], default="nmrnet")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--max-conformers", type=int, default=9)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--forcefield", choices=["auto", "mmff", "uff"], default="auto")
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--time-limit-seconds", type=float, default=20.0)
    parser.add_argument("--coord-route", choices=["standard", "staged27"], default="standard")
    parser.add_argument("--route-initial-confs", type=int, default=27)
    parser.add_argument("--route-prune-rms-thresh", type=float, default=0.5)
    parser.add_argument("--route-coarse-steps", type=int, default=10)
    parser.add_argument("--route-keep-top-k", type=int, default=9)
    parser.add_argument("--route-fine-steps", type=int, default=300)
    parser.set_defaults(optimize_existing=True)
    parser.add_argument("--optimize-existing", dest="optimize_existing", action="store_true")
    parser.add_argument("--no-optimize-existing", dest="optimize_existing", action="store_false")
    parser.set_defaults(allow_2d_if_h_nonzero=True)
    parser.add_argument("--allow-2d-if-h-nonzero", dest="allow_2d_if_h_nonzero", action="store_true")
    parser.add_argument("--no-allow-2d-if-h-nonzero", dest="allow_2d_if_h_nonzero", action="store_false")
    parser.add_argument("--cascade-batch-size", type=int, default=32)
    parser.add_argument("--prepared-cache-sdf", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    needs_shared_cache = (
        args.mode == "CH"
        or (args.mode == "C" and args.c_engine == "cascade2" and args.input_type == "csv")
    )
    if needs_shared_cache and args.prepared_cache_sdf is None:
        args.prepared_cache_sdf = args.output_dir / "_prepared_3d_cache.sdf"
        build_prepared_cache(args, args.prepared_cache_sdf)
    merged_frames: list[pd.DataFrame] = []
    base_samples: dict[str, MergeSample] | None = None

    if args.mode in {"H", "CH"}:
        nmrnet_h_dir = args.output_dir / "nmrnet_H"
        nmrnet_h_sdf = run_nmrnet(args, nmrnet_h_dir, nucleus="H")
        base_samples = load_sdf_map(nmrnet_h_sdf)
        h_df = load_nmrnet_rows(nmrnet_h_dir / "all_predictions.csv")
        merged_frames.append(h_df[h_df["atom_type"] == "H"].copy())

    if args.mode in {"C", "CH"}:
        if args.c_engine == "nmrnet":
            nmrnet_c_dir = args.output_dir / "nmrnet_C"
            nmrnet_c_sdf = run_nmrnet(args, nmrnet_c_dir, nucleus="C")
            if base_samples is None:
                base_samples = load_sdf_map(nmrnet_c_sdf)
            c_df = load_nmrnet_rows(nmrnet_c_dir / "all_predictions.csv")
            merged_frames.append(c_df[c_df["atom_type"] == "C"].copy())
        else:
            cascade_dir = args.output_dir / "cascade2_C"
            cascade_sdf = run_cascade2(args, cascade_dir)
            if base_samples is None:
                base_samples = load_sdf_map(cascade_sdf)
            c_df = load_cascade_rows(cascade_dir / "cascade2_13c_predictions.csv")
            merged_frames.append(c_df.copy())

    if base_samples is None:
        raise RuntimeError("No predictions were produced.")

    merged = pd.concat(merged_frames, ignore_index=True) if merged_frames else pd.DataFrame()
    write_outputs(base_samples, merged, args.output_dir, args.mode)

    final_name = {
        "C": "predicted_C.sdf",
        "H": "predicted_H.sdf",
        "CH": "predicted_CH.sdf",
    }[args.mode]
    shutil.copy2(args.output_dir / "annotated_predictions.sdf", args.output_dir / final_name)
    print(f"Done. Output dir: {args.output_dir}", flush=True)
    print(f"Final merged SDF: {args.output_dir / final_name}", flush=True)


if __name__ == "__main__":
    main()
