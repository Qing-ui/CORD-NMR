from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import time
import gc

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unicore.data import (  # noqa: E402
    AppendTokenDataset,
    Dictionary,
    NestedDictionaryDataset,
    PrependTokenDataset,
    RightPadDataset,
    RightPadDataset2D,
    TokenizeDataset,
)
from uninmr.data import (  # noqa: E402
    CroppingDataset,
    DistanceDataset,
    EdgeTypeDataset,
    FilterDataset,
    KeyDataset,
    NormalizeDataset,
    PrependAndAppend2DDataset,
    RightPadDataset2D0,
    SelectTokenDataset,
    ToTorchDataset,
)
from uninmr.models import UniMatModel  # noqa: E402
from uninmr.utils import TargetScaler, parse_select_atom  # noqa: E402
import shared_rdkit_prep as prep


@dataclass
class PreparedMolecule:
    molecule_id: str
    source: str
    original_smiles: str | None
    canonical_smiles: str
    original_props: dict[str, str]
    mol: Chem.Mol
    atoms: list[str]
    coordinates: np.ndarray
    atoms_target: np.ndarray
    original_atom_index: list[int | None]
    attached_to_original_atom_index: list[int | None]
    is_original_atom: list[bool]


class ListDataset(Dataset):
    def __init__(self, data_list: list[dict]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        return self.data_list[idx]


def prepend_and_append(dataset, pre_token, app_token):
    dataset = PrependTokenDataset(dataset, pre_token)
    return AppendTokenDataset(dataset, app_token)


def load_csv_molecules(csv_path: Path, smiles_column: str, id_column: str | None) -> list[tuple[str, Chem.Mol, str]]:
    df = pd.read_csv(csv_path)
    if smiles_column not in df.columns:
        raise ValueError(f"CSV 缺少列: {smiles_column}")

    molecules: list[tuple[str, Chem.Mol, str]] = []
    for row_index, row in df.iterrows():
        smiles = str(row[smiles_column]).strip()
        if not smiles or smiles.lower() == "nan":
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"第 {row_index + 1} 行 SMILES 无法解析: {smiles}")
        molecule_id = str(row[id_column]) if id_column and id_column in df.columns else f"row_{row_index}"
        molecules.append((molecule_id, mol, smiles))
    return molecules


def load_sdf_molecules(sdf_path: Path) -> list[tuple[str, Chem.Mol, str | None]]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    molecules: list[tuple[str, Chem.Mol, str | None]] = []
    for index, mol in enumerate(supplier):
        if mol is None:
            continue
        base_name = sdf_path.stem
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            raw_name = mol.GetProp("_Name").strip()
            molecule_id = f"{base_name}_{index + 1:04d}_{raw_name}"
        else:
            molecule_id = f"{base_name}_{index + 1:04d}"
        smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
        molecules.append((molecule_id, mol, smiles))
    return molecules


def get_default_max_iters(opt_level: str) -> int:
    return {
        "none": 0,
        "quick": 100,
        "balanced": 300,
        "thorough": 1000,
    }[opt_level]


def minimize_with_deadline(ff, max_iters: int, deadline: float) -> None:
    remaining = max(0, int(max_iters))
    chunk = 10
    while remaining > 0:
        if time.monotonic() >= deadline:
            break
        this_round = min(chunk, remaining)
        ff.Minimize(maxIts=this_round)
        remaining -= this_round


def optimize_conformer_and_get_energy(
    mol: Chem.Mol,
    conf_id: int,
    max_iters: int,
    forcefield: str,
    deadline: float,
) -> float:
    if forcefield == "mmff":
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise RuntimeError("MMFF 参数不完整，无法使用 MMFF")
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        if ff is None:
            raise RuntimeError("MMFF 力场创建失败")
        if max_iters > 0:
            minimize_with_deadline(ff, max_iters, deadline)
        return float(ff.CalcEnergy())
    if forcefield == "uff":
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        if ff is None:
            raise RuntimeError("UFF 力场创建失败")
        if max_iters > 0:
            minimize_with_deadline(ff, max_iters, deadline)
        return float(ff.CalcEnergy())
    if AllChem.MMFFHasAllMoleculeParams(mol):
        return optimize_conformer_and_get_energy(mol, conf_id, max_iters, "mmff", deadline)
    return optimize_conformer_and_get_energy(mol, conf_id, max_iters, "uff", deadline)


def get_original_3d_mol_if_present(mol: Chem.Mol) -> Chem.Mol | None:
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer(0)
    if not conf.Is3D():
        return None
    fallback = Chem.Mol(mol)
    fallback.RemoveAllConformers()
    fallback.AddConformer(conf, assignId=True)
    fallback.SetProp("embedding_fallback", "original_sdf_coordinates")
    fallback.SetIntProp("generated_conformer_count", 0)
    fallback.SetIntProp("selected_conformer_id", 0)
    fallback.SetDoubleProp("selected_conformer_energy", 0.0)
    return fallback


def has_nonzero_hydrogen_coordinates(mol: Chem.Mol) -> bool:
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer(0)
    hydrogen_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            hydrogen_count += 1
            pos = conf.GetAtomPosition(atom.GetIdx())
            if abs(pos.x) > 1e-8 or abs(pos.y) > 1e-8 or abs(pos.z) > 1e-8:
                return True
    return hydrogen_count == 0


def optimize_existing_coordinates_if_possible(
    mol: Chem.Mol,
    max_iters: int,
    forcefield: str,
    deadline: float,
) -> Chem.Mol:
    work = Chem.Mol(mol)
    try:
        energy = optimize_conformer_and_get_energy(work, 0, max_iters, forcefield, deadline)
        work.SetDoubleProp("selected_conformer_energy", float(energy))
        work.SetIntProp("selected_conformer_id", 0)
        work.SetIntProp("generated_conformer_count", 1)
        work.SetProp("embedding_fallback", "optimized_existing_coordinates")
        return work
    except Exception:
        fallback = Chem.Mol(mol)
        fallback.SetProp("embedding_fallback", "original_coordinates_optimization_failed")
        fallback.SetIntProp("generated_conformer_count", 0)
        fallback.SetIntProp("selected_conformer_id", 0)
        fallback.SetDoubleProp("selected_conformer_energy", 0.0)
        return fallback


def get_original_coordinates_if_allowed(
    mol: Chem.Mol,
    allow_2d_if_h_nonzero: bool,
    optimize_existing: bool,
    max_iters: int,
    forcefield: str,
    deadline: float,
) -> Chem.Mol | None:
    if mol.GetNumConformers() == 0:
        if not allow_2d_if_h_nonzero:
            return None
        fallback = Chem.Mol(mol)
        AllChem.Compute2DCoords(fallback, clearConfs=True)
        if not has_nonzero_hydrogen_coordinates(fallback):
            return None
        if optimize_existing:
            fallback = optimize_existing_coordinates_if_possible(fallback, max_iters, forcefield, deadline)
            if not has_nonzero_hydrogen_coordinates(fallback):
                return None
            return fallback
        fallback.SetProp("embedding_fallback", "computed_2d_coordinates_with_nonzero_h")
        fallback.SetIntProp("generated_conformer_count", 0)
        fallback.SetIntProp("selected_conformer_id", 0)
        fallback.SetDoubleProp("selected_conformer_energy", 0.0)
        return fallback
    conf = mol.GetConformer(0)
    if not conf.Is3D() and allow_2d_if_h_nonzero and not has_nonzero_hydrogen_coordinates(mol):
        fallback = Chem.Mol(mol)
        AllChem.Compute2DCoords(fallback, clearConfs=True)
        if not has_nonzero_hydrogen_coordinates(fallback):
            return None
        if optimize_existing:
            fallback = optimize_existing_coordinates_if_possible(fallback, max_iters, forcefield, deadline)
            if not has_nonzero_hydrogen_coordinates(fallback):
                return None
            return fallback
        fallback.SetProp("embedding_fallback", "computed_2d_coordinates_with_nonzero_h")
        fallback.SetIntProp("generated_conformer_count", 0)
        fallback.SetIntProp("selected_conformer_id", 0)
        fallback.SetDoubleProp("selected_conformer_energy", 0.0)
        return fallback
    if not conf.Is3D() and not allow_2d_if_h_nonzero:
        return None
    fallback = Chem.Mol(mol)
    fallback.RemoveAllConformers()
    fallback.AddConformer(conf, assignId=True)
    if optimize_existing:
        return optimize_existing_coordinates_if_possible(fallback, max_iters, forcefield, deadline)
    fallback.SetProp(
        "embedding_fallback",
        "original_sdf_coordinates" if conf.Is3D() else "original_2d_sdf_coordinates_with_nonzero_h",
    )
    fallback.SetIntProp("generated_conformer_count", 0)
    fallback.SetIntProp("selected_conformer_id", 0)
    fallback.SetDoubleProp("selected_conformer_energy", 0.0)
    return fallback


def validate_usable_coordinates(mol: Chem.Mol, allow_2d_if_h_nonzero: bool) -> None:
    if mol.GetNumConformers() == 0:
        raise RuntimeError("没有可用构象坐标")
    conf = mol.GetConformer(0)
    if not conf.Is3D() and not allow_2d_if_h_nonzero:
        raise RuntimeError("最终构象不是 3D 坐标")
    h_positions = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            pos = conf.GetAtomPosition(atom.GetIdx())
            h_positions.append((pos.x, pos.y, pos.z))
    if h_positions and all(abs(x) < 1e-8 and abs(y) < 1e-8 and abs(z) < 1e-8 for x, y, z in h_positions):
        raise RuntimeError("所有氢坐标均为 0，拒绝输出无效预测")


def embed_and_optimize_3d(
    mol: Chem.Mol,
    random_seed: int,
    max_iters: int,
    forcefield: str,
    max_conformers: int,
    time_limit_seconds: float,
    prefer_existing_coordinates: bool,
    allow_2d_if_h_nonzero: bool,
    optimize_existing_coordinates: bool,
) -> Chem.Mol:
    start_time = time.monotonic()
    deadline = start_time + time_limit_seconds
    best_mol = None
    best_conf_id = None
    best_energy = None
    success_count = 0

    if prefer_existing_coordinates and mol.GetNumConformers() > 0 and mol.GetConformer(0).Is3D():
        fallback = get_original_coordinates_if_allowed(
            mol,
            allow_2d_if_h_nonzero,
            optimize_existing_coordinates,
            max_iters,
            forcefield,
            deadline,
        )
        if fallback is not None:
            return fallback

    for conf_try in range(max_conformers):
        if time.monotonic() >= deadline:
            break

        work = Chem.Mol(mol)
        work.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed + conf_try
        params.useRandomCoords = conf_try > 0
        params.pruneRmsThresh = 0.2
        result = AllChem.EmbedMolecule(work, params)
        if result != 0:
            continue
        try:
            energy = optimize_conformer_and_get_energy(work, 0, max_iters, forcefield, deadline)
        except Exception:
            continue
        success_count += 1
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_conf_id = conf_try
            best_mol = Chem.Mol(work)

    if best_mol is not None:
        best_mol.SetDoubleProp("selected_conformer_energy", float(best_energy))
        best_mol.SetIntProp("selected_conformer_id", int(best_conf_id if best_conf_id is not None else 0))
        best_mol.SetIntProp("generated_conformer_count", int(success_count))
        return best_mol

    fallback = get_original_coordinates_if_allowed(
        mol,
        allow_2d_if_h_nonzero,
        optimize_existing_coordinates,
        max_iters,
        forcefield,
        deadline,
    )
    if fallback is not None:
        return fallback

    raise RuntimeError("RDKit 3D 嵌入失败")


def prepare_single_molecule(
    molecule_id: str,
    mol: Chem.Mol,
    source: str,
    original_smiles: str | None,
    random_seed: int,
    max_iters: int,
    forcefield: str,
    max_conformers: int,
    time_limit_seconds: float,
    prefer_existing_coordinates: bool,
    allow_2d_if_h_nonzero: bool,
    optimize_existing_coordinates: bool,
) -> PreparedMolecule:
    work = Chem.Mol(mol)
    Chem.SanitizeMol(work)
    original_props = {prop_name: mol.GetProp(prop_name) for prop_name in mol.GetPropNames()}

    for atom in work.GetAtoms():
        atom.SetIntProp("orig_idx", atom.GetIdx())

    work = Chem.AddHs(work, addCoords=work.GetNumConformers() > 0)
    work = embed_and_optimize_3d(
        work,
        random_seed=random_seed,
        max_iters=max_iters,
        forcefield=forcefield,
        max_conformers=max_conformers,
        time_limit_seconds=time_limit_seconds,
        prefer_existing_coordinates=prefer_existing_coordinates,
        allow_2d_if_h_nonzero=allow_2d_if_h_nonzero,
        optimize_existing_coordinates=optimize_existing_coordinates,
    )
    validate_usable_coordinates(work, allow_2d_if_h_nonzero)
    conformer = work.GetConformer()

    atoms: list[str] = []
    original_atom_index: list[int | None] = []
    attached_to_original_atom_index: list[int | None] = []
    is_original_atom: list[bool] = []

    for atom in work.GetAtoms():
        atoms.append(atom.GetSymbol())
        if atom.HasProp("orig_idx"):
            original_atom_index.append(atom.GetIntProp("orig_idx"))
            attached_to_original_atom_index.append(atom.GetIntProp("orig_idx"))
            is_original_atom.append(True)
        else:
            original_atom_index.append(None)
            parent_orig = None
            for neighbor in atom.GetNeighbors():
                if neighbor.HasProp("orig_idx"):
                    parent_orig = neighbor.GetIntProp("orig_idx")
                    break
            attached_to_original_atom_index.append(parent_orig)
            is_original_atom.append(False)

    coordinates = np.array(conformer.GetPositions(), dtype=np.float32)
    atoms_target = np.zeros(len(atoms), dtype=np.float32)

    return PreparedMolecule(
        molecule_id=molecule_id,
        source=source,
        original_smiles=original_smiles,
        canonical_smiles=Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(work))),
        original_props=original_props,
        mol=work,
        atoms=atoms,
        coordinates=coordinates,
        atoms_target=atoms_target,
        original_atom_index=original_atom_index,
        attached_to_original_atom_index=attached_to_original_atom_index,
        is_original_atom=is_original_atom,
    )


def build_nested_dataset(samples: list[PreparedMolecule], dictionary: Dictionary, args, selected_token, nucleus: str):
    data_list = [
        {
            "atoms": sample.atoms,
            "coordinates": sample.coordinates,
            "atoms_target": sample.atoms_target,
            "atoms_target_mask": np.array([1 if atom == nucleus else 0 for atom in sample.atoms], dtype=np.int64),
        }
        for sample in samples
    ]
    dataset = ListDataset(data_list)

    dataset = CroppingDataset(dataset, args.seed, "atoms", "coordinates", args.max_atoms)
    dataset = NormalizeDataset(dataset, "coordinates")

    token_dataset = KeyDataset(dataset, "atoms")
    token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=args.max_seq_len)
    atoms_target_mask_dataset = KeyDataset(dataset, "atoms_target_mask")
    select_atom_dataset = SelectTokenDataset(
        token_dataset=token_dataset,
        token_mask_dataset=atoms_target_mask_dataset,
        selected_token=selected_token,
    )
    filter_list = [0 if torch.all(select_atom_dataset[i] == 0) else 1 for i in range(len(select_atom_dataset))]

    dataset = FilterDataset(dataset, filter_list)
    token_dataset = FilterDataset(token_dataset, filter_list)
    select_atom_dataset = FilterDataset(select_atom_dataset, filter_list)

    filtered_samples = [sample for sample, keep in zip(samples, filter_list) if keep == 1]

    coord_dataset = KeyDataset(dataset, "coordinates")
    token_dataset = prepend_and_append(token_dataset, dictionary.bos(), dictionary.eos())
    select_atom_dataset = prepend_and_append(select_atom_dataset, dictionary.pad(), dictionary.pad())
    coord_dataset = ToTorchDataset(coord_dataset, "float32")

    distance_dataset = DistanceDataset(coord_dataset)
    distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
    distance_dataset = RightPadDataset2D(distance_dataset, pad_idx=0)
    coord_dataset = prepend_and_append(coord_dataset, 0.0, 0.0)
    edge_type = EdgeTypeDataset(token_dataset, len(dictionary))

    tgt_dataset = KeyDataset(dataset, "atoms_target")
    tgt_dataset = ToTorchDataset(tgt_dataset, dtype="float32")
    tgt_dataset = prepend_and_append(tgt_dataset, dictionary.pad(), dictionary.pad())

    nested = NestedDictionaryDataset(
        {
            "net_input": {
                "select_atom": RightPadDataset(select_atom_dataset, pad_idx=dictionary.pad()),
                "src_tokens": RightPadDataset(token_dataset, pad_idx=dictionary.pad()),
                "src_coord": RightPadDataset2D0(coord_dataset, pad_idx=0),
                "src_distance": distance_dataset,
                "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0),
            },
            "target": {
                "finetune_target": RightPadDataset(tgt_dataset, pad_idx=0),
            },
        }
    )
    return nested, filtered_samples


def load_model(model_path: Path, dict_path: Path, scaler_dir: Path, selected_atom: str):
    if not model_path.exists():
        raise FileNotFoundError(f"缺少模型文件: {model_path}")
    if not dict_path.exists():
        raise FileNotFoundError(f"缺少字典文件: {dict_path}")
    scaler_path = scaler_dir / "target_scaler.ss"
    if not scaler_path.exists():
        raise FileNotFoundError(f"缺少 scaler 文件: {scaler_path}")

    state = torch.load(model_path, map_location="cpu")
    args = state["args"]
    args.cpu = True
    args.fp16 = False
    args.batch_size = 1
    args.num_workers = 0
    args.required_batch_size_multiple = 1
    args.data_buffer_size = 1
    args.saved_dir = str(scaler_dir)
    args.selected_atom = selected_atom
    args.global_distance = False
    args.atom_descriptor = 0

    dictionary = Dictionary.load(str(dict_path))
    dictionary.add_symbol("[MASK]", is_special=True)
    selected_token = parse_select_atom(dictionary, selected_atom)
    target_scaler = TargetScaler(str(scaler_dir))

    state["model"] = {
        (
            key.replace("classification_heads", "node_classification_heads")
            if key.startswith("classification_heads")
            else key
        ): value
        for key, value in state["model"].items()
    }

    model = UniMatModel(args, dictionary)
    model.register_node_classification_head(
        args.classification_head_name,
        num_classes=args.num_classes,
        extra_dim=args.atom_descriptor,
    )
    model.load_state_dict(state["model"], strict=False)
    model.float()
    model.eval()
    return model, dictionary, selected_token, target_scaler, args


def get_default_assets() -> dict[str, Path]:
    bundle_root = Path(os.environ["NMR_PREDICTOR_HOME"]) if "NMR_PREDICTOR_HOME" in os.environ else REPO_ROOT.parent
    liquid_root = bundle_root / "models" / "nmrnet" / "liquid"
    if not liquid_root.exists():
        liquid_root = REPO_ROOT.parent / "model_assets" / "weights" / "finetune" / "liquid"
    return {
        "liquid_root": liquid_root,
        "dict_path": liquid_root / "mol_dict.txt",
        "h_dir": liquid_root / "H_mol_pre_all_h_220816_global_0_kener_gauss_atomdes_0_unimol_large_atom_regloss_mae_lr_5e-3_bs_16_0.06_400",
        "c_dir": liquid_root / "C_mol_pre_all_h_220816_global_0_kener_gauss_atomdes_0_unimol_large_atom_regloss_mae_lr_1e-3_bs_16_0.06_200",
    }


def run_single_nucleus_prediction(
    prepared_samples: list[PreparedMolecule],
    nucleus: str,
    model_path: Path,
    dict_path: Path,
    scaler_dir: Path,
) -> pd.DataFrame:
    model, dictionary, selected_token, target_scaler, model_args = load_model(
        model_path=model_path,
        dict_path=dict_path,
        scaler_dir=scaler_dir,
        selected_atom=nucleus,
    )
    filtered_input = [sample for sample in prepared_samples if any(atom == nucleus for atom in sample.atoms)]
    nested, filtered_samples = build_nested_dataset(filtered_input, dictionary, model_args, selected_token, nucleus)
    dataloader = DataLoader(nested, batch_size=1, shuffle=False)

    rows: list[dict] = []
    with torch.no_grad():
        for sample_info, batch in zip(filtered_samples, dataloader):
            model_inputs = {
                key.replace("net_input.", ""): value
                for key, value in batch.items()
                if key.startswith("net_input.")
            }
            net_output = model(
                **model_inputs,
                features_only=True,
                classification_head_name=model_args.classification_head_name,
            )
            pred = target_scaler.inverse_transform(
                net_output[0].view(-1, model_args.num_classes).cpu()
            ).astype("float32").reshape(-1)

            selected_atom_indices = [
                atom_index
                for atom_index, atom_symbol in enumerate(sample_info.atoms)
                if atom_symbol == nucleus
            ]
            if len(pred) == len(sample_info.atoms) + 2:
                selected_predictions = [
                    (atom_index, float(pred[1 + atom_index]))
                    for atom_index in selected_atom_indices
                ]
            elif len(pred) == len(sample_info.atoms):
                selected_predictions = [
                    (atom_index, float(pred[atom_index]))
                    for atom_index in selected_atom_indices
                ]
            elif len(pred) == len(selected_atom_indices):
                selected_predictions = [
                    (atom_index, float(pred_value))
                    for atom_index, pred_value in zip(selected_atom_indices, pred)
                ]
            else:
                raise RuntimeError(
                    f"Unexpected prediction length for {sample_info.molecule_id} "
                    f"{nucleus}: got {len(pred)}, atoms={len(sample_info.atoms)}, "
                    f"selected={len(selected_atom_indices)}"
                )

            for atom_index, predicted_shift in selected_predictions:
                atom_symbol = sample_info.atoms[atom_index]
                rows.append(
                    {
                        "molecule_id": sample_info.molecule_id,
                        "source": sample_info.source,
                        "nucleus": nucleus,
                        "atom_index_explicit_h_0based": atom_index,
                        "atom_index_explicit_h_1based": atom_index + 1,
                        "original_atom_index_0based": sample_info.original_atom_index[atom_index],
                        "original_atom_index_1based": (
                            None
                            if sample_info.original_atom_index[atom_index] is None
                            else sample_info.original_atom_index[atom_index] + 1
                        ),
                        "attached_to_original_atom_index_0based": sample_info.attached_to_original_atom_index[atom_index],
                        "attached_to_original_atom_index_1based": (
                            None
                            if sample_info.attached_to_original_atom_index[atom_index] is None
                            else sample_info.attached_to_original_atom_index[atom_index] + 1
                        ),
                        "is_original_atom": sample_info.is_original_atom[atom_index],
                        "atom_symbol": atom_symbol,
                        "predicted_shift_ppm": predicted_shift,
                        "original_smiles": sample_info.original_smiles,
                        "canonical_smiles": sample_info.canonical_smiles,
                    }
                )
    return pd.DataFrame(rows)


def load_input_records(args) -> tuple[list[tuple[str, Chem.Mol, str | None]], str]:
    if args.input_type == "csv":
        raw = load_csv_molecules(args.input, args.smiles_column, args.id_column)
        source = "csv_smiles"
    else:
        raw = load_sdf_molecules(args.input)
        source = "sdf"
    return raw, source


def count_valid_sdf_records(sdf_path: Path) -> int:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    return sum(1 for mol in supplier if mol is not None)


def iter_prepared_cache_chunks(sdf_path: Path, chunk_size: int):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    chunk: list[PreparedMolecule] = []
    for mol in supplier:
        if mol is None:
            continue
        chunk.append(prep.prepared_molecule_from_cache_mol(mol))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def prepare_input_molecules(
    raw: list[tuple[str, Chem.Mol, str | None]],
    source: str,
    args,
    start_index: int = 0,
    total_count: int | None = None,
) -> tuple[list[PreparedMolecule], list[str]]:
    prepared: list[PreparedMolecule] = []
    failed_rows: list[str] = []
    for index, (molecule_id, mol, smiles) in enumerate(raw):
        current_index = start_index + index + 1
        total_display = total_count if total_count is not None else len(raw)
        if index == 0 or current_index % 25 == 0:
            print(f"preparing {current_index}/{total_display}: {molecule_id}", flush=True)
        try:
            prepared.append(
                prep.prepare_single_molecule(
                    molecule_id=molecule_id,
                    mol=mol,
                    source=source,
                    original_smiles=smiles,
                    random_seed=args.random_seed + index,
                    max_iters=args.max_iters,
                    forcefield=args.forcefield,
                    max_conformers=args.max_conformers,
                    time_limit_seconds=args.time_limit_seconds,
                    prefer_existing_coordinates=(args.input_type == "sdf" and args.prefer_sdf_coordinates),
                    allow_2d_if_h_nonzero=args.allow_2d_if_h_nonzero,
                    optimize_existing_coordinates=args.optimize_existing_coordinates,
                    coord_route=args.coord_route,
                    route_initial_confs=args.route_initial_confs,
                    route_prune_rms_thresh=args.route_prune_rms_thresh,
                    route_coarse_steps=args.route_coarse_steps,
                    route_keep_top_k=args.route_keep_top_k,
                    route_fine_steps=args.route_fine_steps,
                )
            )
        except Exception as exc:
            failed_rows.append(f"{molecule_id}\t{type(exc).__name__}\t{exc}")
    return prepared, failed_rows


def safe_stem(name: str) -> str:
    raw = str(name).strip()
    if raw.lower().endswith(".sdf"):
        raw = Path(raw).stem
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", raw).strip()
    return cleaned or "molecule"


def build_sdf_prediction_properties(sample: PreparedMolecule, group: pd.DataFrame) -> dict[str, str]:
    group = group.sort_values("atom_index_explicit_h_0based").reset_index(drop=True)
    carbon_rows = group[group["atom_symbol"] == "C"].copy().reset_index(drop=True)
    hydrogen_rows = group[group["atom_symbol"] == "H"].copy().reset_index(drop=True)

    predicted_c13_lines: list[str] = []
    quaternaries: list[str] = []
    tertiaries: list[str] = []
    secondaries: list[str] = []
    primaries: list[str] = []

    for serial, row in carbon_rows.iterrows():
        atom_index = int(row["atom_index_explicit_h_0based"])
        atom_index_1based = atom_index + 1
        shift = float(row["predicted_shift_ppm"])
        predicted_c13_lines.append(f"{serial}[{atom_index_1based}]\t{shift:.2f}")

        atom = sample.mol.GetAtomWithIdx(atom_index)
        hydrogen_count = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)
        classified_line = f"{atom_index_1based}\t{shift:.2f}"
        if hydrogen_count == 0:
            quaternaries.append(classified_line)
        elif hydrogen_count == 1:
            tertiaries.append(classified_line)
        elif hydrogen_count == 2:
            secondaries.append(classified_line)
        elif hydrogen_count == 3:
            primaries.append(classified_line)

    hydrogen_lines: list[str] = []
    attached_counter: dict[int, int] = {}
    for _, row in hydrogen_rows.iterrows():
        h_atom_index = int(row["atom_index_explicit_h_0based"])
        h_atom_index_1based = h_atom_index + 1
        shift = float(row["predicted_shift_ppm"])
        h_atom = sample.mol.GetAtomWithIdx(h_atom_index)
        neighbor_indices = [neighbor.GetIdx() for neighbor in h_atom.GetNeighbors()]
        attached_index = neighbor_indices[0] if neighbor_indices else h_atom_index
        suffix_index = attached_counter.get(attached_index, 0)
        suffix = chr(ord("a") + suffix_index)
        attached_counter[attached_index] = suffix_index + 1
        hydrogen_lines.append(f"{attached_index + 1}{suffix}[{h_atom_index_1based}]\t{shift:.2f}")

    return {
        "Predicted 13C shifts": "\n".join(predicted_c13_lines),
        "Quaternaries": "\n".join(quaternaries),
        "Tertiaries": "\n".join(tertiaries),
        "Secondaries": "\n".join(secondaries),
        "Primaries": "\n".join(primaries),
        "HydrogenShifts": "\n".join(hydrogen_lines),
    }


def apply_properties_to_mol(mol: Chem.Mol, sample: PreparedMolecule, sdf_props: dict[str, str]) -> None:
    prep.strip_cache_properties(mol)
    prep.strip_embedding_properties(mol)
    for key, value in sample.original_props.items():
        mol.SetProp(key, str(value))
    mol.SetProp("_Name", sample.molecule_id)
    mol.SetProp("canonical_smiles", sample.canonical_smiles)
    if sample.original_smiles:
        mol.SetProp("original_smiles", sample.original_smiles)
    for key, value in sdf_props.items():
        mol.SetProp(key, value)
    if mol.HasProp("selected_conformer_energy"):
        mol.SetProp("selected_conformer_energy", str(mol.GetDoubleProp("selected_conformer_energy")))
    if mol.HasProp("selected_conformer_id"):
        mol.SetProp("selected_conformer_id", str(mol.GetIntProp("selected_conformer_id")))
    if mol.HasProp("generated_conformer_count"):
        mol.SetProp("generated_conformer_count", str(mol.GetIntProp("generated_conformer_count")))


def write_final_sdfs(prepared: list[PreparedMolecule], result: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in prepared:
        out_path = output_dir / f"{safe_stem(sample.molecule_id)}.sdf"
        mol = Chem.Mol(sample.mol)
        group = result[result["molecule_id"] == sample.molecule_id].copy()
        sdf_props = build_sdf_prediction_properties(sample, group)
        apply_properties_to_mol(mol, sample, sdf_props)
        writer = Chem.SDWriter(str(out_path))
        writer.write(mol)
        writer.close()


def append_annotated_combined_sdf(prepared: list[PreparedMolecule], result: pd.DataFrame, writer) -> None:
    for sample in prepared:
        mol = Chem.Mol(sample.mol)
        group = result[result["molecule_id"] == sample.molecule_id].copy()
        sdf_props = build_sdf_prediction_properties(sample, group)
        apply_properties_to_mol(mol, sample, sdf_props)
        writer.write(mol)


def write_annotated_combined_sdf(prepared: list[PreparedMolecule], result: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "annotated_predictions.sdf"
    writer = Chem.SDWriter(str(out_path))
    append_annotated_combined_sdf(prepared, result, writer)
    writer.close()


def write_per_structure_csvs(result: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for molecule_id, group in result.groupby("molecule_id", sort=False):
        simple = group.loc[:, [
            "atom_index_explicit_h_0based",
            "atom_symbol",
            "predicted_shift_ppm",
        ]].copy()
        simple.columns = [
            "atom_index",
            "atom_type",
            "shift_ppm",
        ]
        simple = simple.sort_values(["atom_index", "atom_type"]).reset_index(drop=True)
        out_path = output_dir / f"{safe_stem(molecule_id)}.csv"
        simple.to_csv(out_path, index=False, encoding="utf-8-sig")


def append_failed_rows(failed_rows: list[str], output_dir: Path) -> None:
    if not failed_rows:
        return
    failed_path = output_dir / "failed_molecules.tsv"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not failed_path.exists()
    with failed_path.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write("molecule_id\terror_type\terror_message\n")
        handle.write("\n".join(failed_rows))
        handle.write("\n")


def parse_args():
    defaults = get_default_assets()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-type", choices=["csv", "sdf"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nucleus", choices=["H", "C", "both"], default="both")
    parser.add_argument("--smiles-column", type=str, default="smiles")
    parser.add_argument("--id-column", type=str, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--opt-level", choices=["none", "quick", "balanced", "thorough"], default="balanced")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--forcefield", choices=["auto", "mmff", "uff"], default="auto")
    parser.add_argument("--max-conformers", type=int, default=9)
    parser.add_argument("--time-limit-seconds", type=float, default=20.0)
    parser.add_argument("--coord-route", choices=["standard", "staged27"], default="standard")
    parser.add_argument("--route-initial-confs", type=int, default=27)
    parser.add_argument("--route-prune-rms-thresh", type=float, default=0.5)
    parser.add_argument("--route-coarse-steps", type=int, default=10)
    parser.add_argument("--route-keep-top-k", type=int, default=9)
    parser.add_argument("--route-fine-steps", type=int, default=300)
    parser.set_defaults(prefer_sdf_coordinates=True)
    parser.add_argument("--prefer-sdf-coordinates", dest="prefer_sdf_coordinates", action="store_true")
    parser.add_argument("--no-prefer-sdf-coordinates", dest="prefer_sdf_coordinates", action="store_false")
    parser.set_defaults(optimize_existing_coordinates=True)
    parser.add_argument("--optimize-existing-coordinates", dest="optimize_existing_coordinates", action="store_true")
    parser.add_argument("--no-optimize-existing-coordinates", dest="optimize_existing_coordinates", action="store_false")
    parser.set_defaults(allow_2d_if_h_nonzero=True)
    parser.add_argument("--allow-2d-if-h-nonzero", dest="allow_2d_if_h_nonzero", action="store_true")
    parser.add_argument("--no-allow-2d-if-h-nonzero", dest="allow_2d_if_h_nonzero", action="store_false")
    parser.add_argument("--write-combined", action="store_true")
    parser.add_argument("--prepared-cache-sdf", type=Path, default=None)
    parser.add_argument("--dict-path", type=Path, default=defaults["dict_path"])
    parser.add_argument("--h-model-path", type=Path, default=defaults["h_dir"] / "cv_seed_42_fold_0" / "checkpoint_best.pt")
    parser.add_argument("--h-scaler-dir", type=Path, default=defaults["h_dir"])
    parser.add_argument("--c-model-path", type=Path, default=defaults["c_dir"] / "cv_seed_42_fold_0" / "checkpoint_best.pt")
    parser.add_argument("--c-scaler-dir", type=Path, default=defaults["c_dir"])
    parser.add_argument("--flush-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    args.max_conformers = max(1, min(9, int(args.max_conformers)))
    args.flush_every = max(1, int(args.flush_every))
    args.route_initial_confs = max(1, int(args.route_initial_confs))
    args.route_keep_top_k = max(1, min(int(args.route_keep_top_k), args.route_initial_confs))
    if args.max_iters is None:
        args.max_iters = get_default_max_iters(args.opt_level)
    nuclei = ["H", "C"] if args.nucleus == "both" else [args.nucleus]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_writer = Chem.SDWriter(str(args.output_dir / "annotated_predictions.sdf"))
    combined_csv_path = args.output_dir / "all_predictions.csv"
    if args.write_combined and combined_csv_path.exists():
        combined_csv_path.unlink()

    if args.prepared_cache_sdf is not None:
        total_count = count_valid_sdf_records(args.prepared_cache_sdf)
        prepared_chunks = iter_prepared_cache_chunks(args.prepared_cache_sdf, args.flush_every)
        source = "prepared_cache"
    else:
        raw, source = load_input_records(args)
        total_count = len(raw)
        prepared_chunks = None
    print(f"loaded records: {total_count}", flush=True)

    if prepared_chunks is not None:
        chunk_start = 0
        for prepared in prepared_chunks:
            chunk_end = min(chunk_start + len(prepared), total_count)
            print(f"processing chunk {chunk_start + 1}-{chunk_end}/{total_count}", flush=True)
            failed_rows: list[str] = []
            outputs: list[pd.DataFrame] = []
            for nucleus in nuclei:
                if nucleus == "H":
                    outputs.append(
                        run_single_nucleus_prediction(
                            prepared_samples=prepared,
                            nucleus="H",
                            model_path=args.h_model_path,
                            dict_path=args.dict_path,
                            scaler_dir=args.h_scaler_dir,
                        )
                    )
                else:
                    outputs.append(
                        run_single_nucleus_prediction(
                            prepared_samples=prepared,
                            nucleus="C",
                            model_path=args.c_model_path,
                            dict_path=args.dict_path,
                            scaler_dir=args.c_scaler_dir,
                        )
                    )
            result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
            write_final_sdfs(prepared, result, args.output_dir)
            append_annotated_combined_sdf(prepared, result, combined_writer)
            write_per_structure_csvs(result, args.output_dir)
            if args.write_combined:
                result.to_csv(
                    combined_csv_path,
                    mode="a",
                    header=not combined_csv_path.exists(),
                    index=False,
                    encoding="utf-8-sig",
                )
            append_failed_rows(failed_rows, args.output_dir)
            del outputs
            del result
            del prepared
            gc.collect()
            chunk_start = chunk_end
    else:
        for chunk_start in range(0, total_count, args.flush_every):
            chunk_end = min(chunk_start + args.flush_every, total_count)
            print(f"processing chunk {chunk_start + 1}-{chunk_end}/{total_count}", flush=True)
            prepared, failed_rows = prepare_input_molecules(
                raw[chunk_start:chunk_end],
                source=source,
                args=args,
                start_index=chunk_start,
                total_count=total_count,
            )
            outputs: list[pd.DataFrame] = []
            for nucleus in nuclei:
                if nucleus == "H":
                    outputs.append(
                        run_single_nucleus_prediction(
                            prepared_samples=prepared,
                            nucleus="H",
                            model_path=args.h_model_path,
                            dict_path=args.dict_path,
                            scaler_dir=args.h_scaler_dir,
                        )
                    )
                else:
                    outputs.append(
                        run_single_nucleus_prediction(
                            prepared_samples=prepared,
                            nucleus="C",
                            model_path=args.c_model_path,
                            dict_path=args.dict_path,
                            scaler_dir=args.c_scaler_dir,
                        )
                    )

            result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
            write_final_sdfs(prepared, result, args.output_dir)
            append_annotated_combined_sdf(prepared, result, combined_writer)
            write_per_structure_csvs(result, args.output_dir)
            if args.write_combined:
                result.to_csv(
                    combined_csv_path,
                    mode="a",
                    header=not combined_csv_path.exists(),
                    index=False,
                    encoding="utf-8-sig",
                )
            append_failed_rows(failed_rows, args.output_dir)
            del outputs
            del result
            del prepared
            gc.collect()
        del raw
        gc.collect()

    combined_writer.close()
    print(f"saved dir: {args.output_dir}")
    print(f"saved combined sdf: {args.output_dir / 'annotated_predictions.sdf'}")


if __name__ == "__main__":
    main()
