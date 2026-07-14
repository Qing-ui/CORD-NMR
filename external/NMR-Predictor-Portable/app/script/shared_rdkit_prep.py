from __future__ import annotations

import json
import re
import time
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolDescriptors


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


MACROCYCLE_THRESHOLD = 9
SMALL_RING_MAX = 7
FLEXIBLE_ROTATABLE_BOND_CUTOFF = 8
VERY_FLEXIBLE_ROTATABLE_BOND_CUTOFF = 14
VERY_LARGE_HEAVY_ATOM_CUTOFF = 70
EMBEDDING_METADATA_KEYS = [
    "selected_conformer_energy",
    "selected_conformer_id",
    "generated_conformer_count",
    "embedding_route",
    "embedding_profile",
    "embedding_max_ring_size",
    "embedding_rotatable_bonds",
    "embedding_heavy_atoms",
    "embedding_fragment_count",
    "embedding_flexible",
    "embedding_very_flexible",
]
CACHE_PROP_PREFIX = "__prepared_cache__"


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


def pick_best_conf_as_single_conformer(mol: Chem.Mol, conf_id: int, energy: float, generated_count: int, tag: str) -> Chem.Mol:
    selected = Chem.Mol(mol)
    best_conf = mol.GetConformer(conf_id)
    selected.RemoveAllConformers()
    selected.AddConformer(best_conf, assignId=True)
    selected.SetDoubleProp("selected_conformer_energy", float(energy))
    selected.SetIntProp("selected_conformer_id", int(conf_id))
    selected.SetIntProp("generated_conformer_count", int(generated_count))
    selected.SetProp("embedding_route", tag)
    return selected


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


def build_2d_fallback_with_nonzero_h(
    mol: Chem.Mol,
    optimize_existing: bool,
    max_iters: int,
    forcefield: str,
    deadline: float,
    reason: str,
) -> Chem.Mol | None:
    fallback = Chem.Mol(mol)
    AllChem.Compute2DCoords(fallback, clearConfs=True)
    if not has_nonzero_hydrogen_coordinates(fallback):
        return None
    if optimize_existing:
        fallback = optimize_existing_coordinates_if_possible(fallback, max_iters, forcefield, deadline)
        if not has_nonzero_hydrogen_coordinates(fallback):
            return None
    fallback.SetProp("embedding_fallback", reason)
    if not fallback.HasProp("selected_conformer_id"):
        fallback.SetIntProp("selected_conformer_id", 0)
    if not fallback.HasProp("generated_conformer_count"):
        fallback.SetIntProp("generated_conformer_count", 0)
    if not fallback.HasProp("selected_conformer_energy"):
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
        return build_2d_fallback_with_nonzero_h(
            mol,
            optimize_existing,
            max_iters,
            forcefield,
            deadline,
            "computed_2d_coordinates_with_nonzero_h",
        )

    conf = mol.GetConformer(0)
    if not conf.Is3D() and allow_2d_if_h_nonzero and not has_nonzero_hydrogen_coordinates(mol):
        return build_2d_fallback_with_nonzero_h(
            mol,
            optimize_existing,
            max_iters,
            forcefield,
            deadline,
            "computed_2d_coordinates_with_nonzero_h",
        )

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


def classify_embedding_profile(mol: Chem.Mol) -> dict[str, object]:
    base = Chem.RemoveHs(Chem.Mol(mol))
    ring_sizes = [len(ring) for ring in base.GetRingInfo().AtomRings()]
    max_ring_size = max(ring_sizes, default=0)
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(base)
    heavy_atoms = base.GetNumHeavyAtoms()
    fragment_count = len(Chem.GetMolFrags(base))
    has_macrocycle = max_ring_size >= MACROCYCLE_THRESHOLD
    has_small_ring = any(3 <= size <= SMALL_RING_MAX for size in ring_sizes)
    flexible = rotatable_bonds >= FLEXIBLE_ROTATABLE_BOND_CUTOFF
    very_flexible = (
        rotatable_bonds >= VERY_FLEXIBLE_ROTATABLE_BOND_CUTOFF
        or heavy_atoms >= VERY_LARGE_HEAVY_ATOM_CUTOFF
    )
    if has_macrocycle:
        preset = "macrocycle"
    elif has_small_ring:
        preset = "small_ring"
    else:
        preset = "general"
    return {
        "preset": preset,
        "max_ring_size": max_ring_size,
        "rotatable_bonds": rotatable_bonds,
        "heavy_atoms": heavy_atoms,
        "fragment_count": fragment_count,
        "flexible": flexible,
        "very_flexible": very_flexible,
    }


def build_embed_params(
    mol: Chem.Mol,
    profile: dict[str, object],
    random_seed: int,
    use_random_coords: bool,
    prune_rms_thresh: float,
    timeout_seconds: float,
):
    preset = str(profile["preset"])
    if preset == "macrocycle":
        params = rdDistGeom.ETKDGv3()
    elif preset == "small_ring":
        params = rdDistGeom.srETKDGv3()
    else:
        params = rdDistGeom.ETKDGv2()
    params.randomSeed = random_seed
    params.useRandomCoords = bool(use_random_coords or profile["very_flexible"])
    params.pruneRmsThresh = prune_rms_thresh
    params.trackFailures = True
    params.embedFragmentsSeparately = bool(profile["fragment_count"] > 1)
    params.enforceChirality = True
    if hasattr(params, "timeout"):
        params.timeout = max(1, int(timeout_seconds))
    return params


def annotate_embedding_metadata(mol: Chem.Mol, profile: dict[str, object]) -> None:
    mol.SetProp("embedding_profile", str(profile["preset"]))
    mol.SetIntProp("embedding_max_ring_size", int(profile["max_ring_size"]))
    mol.SetIntProp("embedding_rotatable_bonds", int(profile["rotatable_bonds"]))
    mol.SetIntProp("embedding_heavy_atoms", int(profile["heavy_atoms"]))
    mol.SetIntProp("embedding_fragment_count", int(profile["fragment_count"]))
    mol.SetProp("embedding_flexible", "yes" if bool(profile["flexible"]) else "no")
    mol.SetProp("embedding_very_flexible", "yes" if bool(profile["very_flexible"]) else "no")


def export_embedding_metadata(mol: Chem.Mol) -> dict[str, tuple[str, object]]:
    metadata: dict[str, tuple[str, object]] = {}
    for key in EMBEDDING_METADATA_KEYS:
        if not mol.HasProp(key):
            continue
        if key in {"selected_conformer_id", "generated_conformer_count", "embedding_max_ring_size", "embedding_rotatable_bonds", "embedding_heavy_atoms", "embedding_fragment_count"}:
            metadata[key] = ("int", mol.GetIntProp(key))
        elif key in {"selected_conformer_energy"}:
            metadata[key] = ("double", mol.GetDoubleProp(key))
        else:
            metadata[key] = ("str", mol.GetProp(key))
    return metadata


def import_embedding_metadata(mol: Chem.Mol, metadata: dict[str, tuple[str, object]]) -> Chem.Mol:
    for key, (kind, value) in metadata.items():
        if kind == "int":
            mol.SetIntProp(key, int(value))
        elif kind == "double":
            mol.SetDoubleProp(key, float(value))
        else:
            mol.SetProp(key, str(value))
    return mol


def _encode_optional_int_list(values: list[int | None]) -> str:
    return ",".join("" if value is None else str(int(value)) for value in values)


def _decode_optional_int_list(text: str) -> list[int | None]:
    if not text:
        return []
    values: list[int | None] = []
    for item in text.split(","):
        values.append(None if item == "" else int(item))
    return values


def _encode_bool_list(values: list[bool]) -> str:
    return "".join("1" if value else "0" for value in values)


def _decode_bool_list(text: str) -> list[bool]:
    return [char == "1" for char in text]


def prepared_molecule_to_cache_mol(sample: PreparedMolecule) -> Chem.Mol:
    mol = Chem.Mol(sample.mol)
    mol.SetProp("_Name", sample.molecule_id)
    mol.SetProp(f"{CACHE_PROP_PREFIX}source", sample.source)
    mol.SetProp(f"{CACHE_PROP_PREFIX}original_smiles", sample.original_smiles or "")
    mol.SetProp(f"{CACHE_PROP_PREFIX}canonical_smiles", sample.canonical_smiles)
    mol.SetProp(
        f"{CACHE_PROP_PREFIX}original_props_json",
        json.dumps(sample.original_props, ensure_ascii=False),
    )
    mol.SetProp(
        f"{CACHE_PROP_PREFIX}original_atom_index",
        _encode_optional_int_list(sample.original_atom_index),
    )
    mol.SetProp(
        f"{CACHE_PROP_PREFIX}attached_to_original_atom_index",
        _encode_optional_int_list(sample.attached_to_original_atom_index),
    )
    mol.SetProp(
        f"{CACHE_PROP_PREFIX}is_original_atom",
        _encode_bool_list(sample.is_original_atom),
    )
    mol.SetProp(
        f"{CACHE_PROP_PREFIX}embedding_metadata_json",
        json.dumps(export_embedding_metadata(sample.mol), ensure_ascii=False),
    )
    return mol


def prepared_molecule_from_cache_mol(mol: Chem.Mol) -> PreparedMolecule:
    work = Chem.Mol(mol)
    molecule_id = work.GetProp("_Name") if work.HasProp("_Name") else "molecule"
    source = work.GetProp(f"{CACHE_PROP_PREFIX}source") if work.HasProp(f"{CACHE_PROP_PREFIX}source") else "sdf"
    original_smiles = work.GetProp(f"{CACHE_PROP_PREFIX}original_smiles") if work.HasProp(f"{CACHE_PROP_PREFIX}original_smiles") else None
    if original_smiles == "":
        original_smiles = None
    canonical_smiles = work.GetProp(f"{CACHE_PROP_PREFIX}canonical_smiles") if work.HasProp(f"{CACHE_PROP_PREFIX}canonical_smiles") else Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(work)))
    original_props = {}
    if work.HasProp(f"{CACHE_PROP_PREFIX}original_props_json"):
        original_props = json.loads(work.GetProp(f"{CACHE_PROP_PREFIX}original_props_json"))
    original_atom_index = _decode_optional_int_list(work.GetProp(f"{CACHE_PROP_PREFIX}original_atom_index")) if work.HasProp(f"{CACHE_PROP_PREFIX}original_atom_index") else []
    attached_to_original_atom_index = _decode_optional_int_list(work.GetProp(f"{CACHE_PROP_PREFIX}attached_to_original_atom_index")) if work.HasProp(f"{CACHE_PROP_PREFIX}attached_to_original_atom_index") else []
    is_original_atom = _decode_bool_list(work.GetProp(f"{CACHE_PROP_PREFIX}is_original_atom")) if work.HasProp(f"{CACHE_PROP_PREFIX}is_original_atom") else []
    atoms = [atom.GetSymbol() for atom in work.GetAtoms()]
    if not original_atom_index:
        original_atom_index = list(range(len(atoms)))
    if not attached_to_original_atom_index:
        attached_to_original_atom_index = list(original_atom_index)
    if not is_original_atom:
        is_original_atom = [atom.GetAtomicNum() != 1 for atom in work.GetAtoms()]
    if work.HasProp(f"{CACHE_PROP_PREFIX}embedding_metadata_json"):
        raw_metadata = json.loads(work.GetProp(f"{CACHE_PROP_PREFIX}embedding_metadata_json"))
        import_embedding_metadata(work, {key: (value[0], value[1]) for key, value in raw_metadata.items()})
    coordinates = np.array(work.GetConformer().GetPositions(), dtype=np.float32)
    atoms_target = np.zeros(len(atoms), dtype=np.float32)
    return PreparedMolecule(
        molecule_id=molecule_id,
        source=source,
        original_smiles=original_smiles,
        canonical_smiles=canonical_smiles,
        original_props=original_props,
        mol=work,
        atoms=atoms,
        coordinates=coordinates,
        atoms_target=atoms_target,
        original_atom_index=original_atom_index,
        attached_to_original_atom_index=attached_to_original_atom_index,
        is_original_atom=is_original_atom,
    )


def strip_cache_properties(mol: Chem.Mol) -> Chem.Mol:
    for prop_name in list(mol.GetPropNames()):
        if prop_name.startswith(CACHE_PROP_PREFIX):
            mol.ClearProp(prop_name)
    return mol


def strip_embedding_properties(mol: Chem.Mol) -> Chem.Mol:
    for prop_name in list(mol.GetPropNames()):
        if prop_name.startswith("embedding_"):
            mol.ClearProp(prop_name)
    return mol


def try_embed_multiple_confs(
    mol: Chem.Mol,
    num_confs: int,
    params,
) -> list[int]:
    try:
        return list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, int(num_confs)), params=params))
    except Exception:
        return []


def try_embed_single_conf_with_retries(
    mol: Chem.Mol,
    profile: dict[str, object],
    random_seed: int,
    max_attempts: int,
    prune_rms_thresh: float,
    timeout_seconds: float,
) -> tuple[Chem.Mol, list[int]]:
    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    added_conf_ids: list[int] = []
    for conf_try in range(max(1, int(max_attempts))):
        temp = Chem.Mol(mol)
        temp.RemoveAllConformers()
        params = build_embed_params(
            mol=temp,
            profile=profile,
            random_seed=random_seed + conf_try,
            use_random_coords=bool(conf_try > 0 or profile["flexible"]),
            prune_rms_thresh=prune_rms_thresh,
            timeout_seconds=timeout_seconds,
        )
        try:
            result = AllChem.EmbedMolecule(temp, params)
        except Exception:
            continue
        if result != 0 or temp.GetNumConformers() == 0:
            continue
        work.AddConformer(temp.GetConformer(0), assignId=True)
        added_conf_ids.append(work.GetNumConformers() - 1)
    return work, added_conf_ids


def compute_embedding_result(
    mol: Chem.Mol,
    random_seed: int,
    max_iters: int,
    forcefield: str,
    max_conformers: int,
    time_limit_seconds: float,
    coord_route: str,
    route_initial_confs: int,
    route_prune_rms_thresh: float,
    route_coarse_steps: int,
    route_keep_top_k: int,
    route_fine_steps: int,
) -> Chem.Mol | None:
    deadline = time.monotonic() + max(1.0, float(time_limit_seconds))
    profile = classify_embedding_profile(mol)
    if coord_route == "staged27":
        work = Chem.Mol(mol)
        work.RemoveAllConformers()
        params = build_embed_params(
            mol=work,
            profile=profile,
            random_seed=random_seed,
            use_random_coords=bool(profile["flexible"]),
            prune_rms_thresh=route_prune_rms_thresh,
            timeout_seconds=time_limit_seconds,
        )
        conf_ids = try_embed_multiple_confs(work, route_initial_confs, params)
        if not conf_ids:
            work, conf_ids = try_embed_single_conf_with_retries(
                mol=mol,
                profile=profile,
                random_seed=random_seed,
                max_attempts=route_initial_confs,
                prune_rms_thresh=route_prune_rms_thresh,
                timeout_seconds=time_limit_seconds,
            )
        coarse_rank = []
        for conf_id in conf_ids:
            if time.monotonic() >= deadline:
                break
            try:
                coarse_energy = optimize_conformer_and_get_energy(work, conf_id, route_coarse_steps, forcefield, deadline)
                coarse_rank.append((coarse_energy, conf_id))
            except Exception:
                continue
        coarse_rank.sort(key=lambda x: x[0])
        selected_ids = [conf_id for _, conf_id in coarse_rank[: max(1, route_keep_top_k)]]
        fine_rank = []
        for conf_id in selected_ids:
            if time.monotonic() >= deadline:
                break
            try:
                fine_energy = optimize_conformer_and_get_energy(work, conf_id, route_fine_steps, forcefield, deadline)
                fine_rank.append((fine_energy, conf_id))
            except Exception:
                continue
        if fine_rank:
            fine_rank.sort(key=lambda x: x[0])
            best_energy, best_conf_id = fine_rank[0]
            result = pick_best_conf_as_single_conformer(work, best_conf_id, best_energy, len(conf_ids), "staged27")
            annotate_embedding_metadata(result, profile)
            return result
        return None

    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    params = build_embed_params(
        mol=work,
        profile=profile,
        random_seed=random_seed,
        use_random_coords=bool(profile["flexible"]),
        prune_rms_thresh=0.2,
        timeout_seconds=time_limit_seconds,
    )
    conf_ids = try_embed_multiple_confs(work, max_conformers, params)
    if not conf_ids:
        work, conf_ids = try_embed_single_conf_with_retries(
            mol=mol,
            profile=profile,
            random_seed=random_seed,
            max_attempts=max_conformers,
            prune_rms_thresh=0.2,
            timeout_seconds=time_limit_seconds,
        )
    best_conf_id = None
    best_energy = None
    success_count = 0
    for conf_id in conf_ids:
        if time.monotonic() >= deadline:
            break
        try:
            energy = optimize_conformer_and_get_energy(work, conf_id, max_iters, forcefield, deadline)
        except Exception:
            continue
        success_count += 1
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_conf_id = conf_id

    if best_conf_id is not None and best_energy is not None:
        result = pick_best_conf_as_single_conformer(work, best_conf_id, best_energy, success_count, "standard")
        annotate_embedding_metadata(result, profile)
        return result
    return None


def run_embedding_with_hard_timeout(
    mol: Chem.Mol,
    random_seed: int,
    max_iters: int,
    forcefield: str,
    max_conformers: int,
    hard_timeout_seconds: float,
    coord_route: str,
    route_initial_confs: int,
    route_prune_rms_thresh: float,
    route_coarse_steps: int,
    route_keep_top_k: int,
    route_fine_steps: int,
) -> tuple[Chem.Mol | None, str]:
    helper_path = Path(__file__).with_name("rdkit_embed_worker.py")
    payload = {
        "mol": mol,
        "random_seed": random_seed,
        "max_iters": max_iters,
        "forcefield": forcefield,
        "max_conformers": max_conformers,
        "time_limit_seconds": hard_timeout_seconds,
        "coord_route": coord_route,
        "route_initial_confs": route_initial_confs,
        "route_prune_rms_thresh": route_prune_rms_thresh,
        "route_coarse_steps": route_coarse_steps,
        "route_keep_top_k": route_keep_top_k,
        "route_fine_steps": route_fine_steps,
    }
    input_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    output_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    input_handle.close()
    output_handle.close()
    input_path = Path(input_handle.name)
    output_path = Path(output_handle.name)
    try:
        input_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(
            [sys.executable, str(helper_path), "--input-pkl", str(input_path), "--output-pkl", str(output_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(1.0, float(hard_timeout_seconds)),
            creationflags=creationflags,
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            return None, f"exit_{completed.returncode}"
        status, result_payload = pickle.loads(output_path.read_bytes())
        if status == "ok":
            payload = pickle.loads(result_payload)
            result = payload["mol"]
            import_embedding_metadata(result, payload.get("metadata", {}))
            return result, "ok"
        return None, status
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


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
    coord_route: str = "standard",
    route_initial_confs: int = 27,
    route_prune_rms_thresh: float = 0.5,
    route_coarse_steps: int = 10,
    route_keep_top_k: int = 9,
    route_fine_steps: int = 300,
) -> Chem.Mol:
    start_time = time.monotonic()
    deadline = start_time + time_limit_seconds

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

    if coord_route == "staged27":
        result, status = run_embedding_with_hard_timeout(
            mol,
            random_seed=random_seed,
            max_iters=max_iters,
            forcefield=forcefield,
            max_conformers=max_conformers,
            hard_timeout_seconds=max(1.0, deadline - time.monotonic()),
            coord_route="staged27",
            route_initial_confs=route_initial_confs,
            route_prune_rms_thresh=route_prune_rms_thresh,
            route_coarse_steps=route_coarse_steps,
            route_keep_top_k=route_keep_top_k,
            route_fine_steps=route_fine_steps,
        )
        if result is not None:
            return result
        if status == "timeout":
            fallback = build_2d_fallback_with_nonzero_h(
                mol,
                optimize_existing_coordinates,
                max_iters,
                forcefield,
                deadline,
                "hard_timeout_to_2d_with_nonzero_h",
            )
            if fallback is not None:
                return fallback

    result, status = run_embedding_with_hard_timeout(
        mol,
        random_seed=random_seed,
        max_iters=max_iters,
        forcefield=forcefield,
        max_conformers=max_conformers,
        hard_timeout_seconds=max(1.0, deadline - time.monotonic()),
        coord_route="standard",
        route_initial_confs=route_initial_confs,
        route_prune_rms_thresh=route_prune_rms_thresh,
        route_coarse_steps=route_coarse_steps,
        route_keep_top_k=route_keep_top_k,
        route_fine_steps=route_fine_steps,
    )
    if result is not None:
        return result
    if status == "timeout":
        fallback = build_2d_fallback_with_nonzero_h(
            mol,
            optimize_existing_coordinates,
            max_iters,
            forcefield,
            deadline,
            "hard_timeout_to_2d_with_nonzero_h",
        )
        if fallback is not None:
            return fallback

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

    raise RuntimeError("RDKit 3D embedding failed")


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
    coord_route: str = "standard",
    route_initial_confs: int = 27,
    route_prune_rms_thresh: float = 0.5,
    route_coarse_steps: int = 10,
    route_keep_top_k: int = 9,
    route_fine_steps: int = 300,
) -> PreparedMolecule:
    work = Chem.Mol(mol)
    Chem.SanitizeMol(work)
    original_props = {prop_name: mol.GetProp(prop_name) for prop_name in mol.GetPropNames()}
    original_atom_count = work.GetNumAtoms()
    work = Chem.AddHs(work, addCoords=work.GetNumConformers() > 0)
    for atom_idx in range(original_atom_count):
        work.GetAtomWithIdx(atom_idx).SetIntProp("orig_idx", atom_idx)
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
        coord_route=coord_route,
        route_initial_confs=route_initial_confs,
        route_prune_rms_thresh=route_prune_rms_thresh,
        route_coarse_steps=route_coarse_steps,
        route_keep_top_k=route_keep_top_k,
        route_fine_steps=route_fine_steps,
    )
    for atom_idx in range(min(original_atom_count, work.GetNumAtoms())):
        work.GetAtomWithIdx(atom_idx).SetIntProp("orig_idx", atom_idx)
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
