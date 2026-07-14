from __future__ import annotations

import argparse
import gc
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from shared_rdkit_prep import (
    PreparedMolecule,
    get_default_max_iters,
    load_sdf_molecules,
    prepared_molecule_from_cache_mol,
    prepare_single_molecule,
)


BUNDLE_ROOT = Path(os.environ["NMR_PREDICTOR_HOME"]) if "NMR_PREDICTOR_HOME" in os.environ else Path(__file__).resolve().parents[2]
CASCADE_ROOT = Path(os.environ["CASCADE2_HOME"]) if "CASCADE2_HOME" in os.environ else BUNDLE_ROOT / "models" / "cascade2"
DEFAULT_MODEL_DIR = BUNDLE_ROOT / "models" / "cascade2" / "Predict_SMILES_FF_GPR"
if not DEFAULT_MODEL_DIR.exists():
    DEFAULT_MODEL_DIR = CASCADE_ROOT / "models" / "Predict_SMILES_FF_GPR"

def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()


def compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


def ragged_const(inp_arr):
    import tensorflow as tf
    return tf.ragged.constant(np.expand_dims(inp_arr, axis=0), ragged_rank=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-conformers", type=int, default=9)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--opt-level", choices=["none", "quick", "balanced", "thorough"], default="balanced")
    parser.add_argument("--forcefield", choices=["auto", "mmff", "uff"], default="auto")
    parser.add_argument("--optimize-existing-coordinates", action="store_true", default=True)
    parser.add_argument("--allow-2d-if-h-nonzero", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--time-limit-seconds", type=float, default=20.0)
    parser.add_argument("--coord-route", choices=["standard", "staged27"], default="standard")
    parser.add_argument("--route-initial-confs", type=int, default=27)
    parser.add_argument("--route-prune-rms-thresh", type=float, default=0.5)
    parser.add_argument("--route-coarse-steps", type=int, default=10)
    parser.add_argument("--route-keep-top-k", type=int, default=9)
    parser.add_argument("--route-fine-steps", type=int, default=300)
    parser.add_argument("--prepared-cache-sdf", type=Path, default=None)
    args = parser.parse_args()
    if args.max_iters is None:
        args.max_iters = get_default_max_iters(args.opt_level)
    args.max_conformers = max(1, min(9, int(args.max_conformers)))
    args.route_initial_confs = max(1, int(args.route_initial_confs))
    args.route_keep_top_k = max(1, min(int(args.route_keep_top_k), args.route_initial_confs))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(args.model_dir))
    sys.path.insert(0, str(args.model_dir / "modules"))
    os.chdir(args.model_dir)

    import tensorflow as tf
    from nfp.preprocessing import GraphSequence

    from kgcnn.layers.casting import ChangeTensorType
    from kgcnn.layers.conv.painn_conv import PAiNNUpdate, EquivariantInitialize, PAiNNconv
    from kgcnn.layers.geom import NodeDistanceEuclidean, EdgeDirectionNormalized, CosCutOffEnvelope, NodePosition, ShiftPeriodicLattice
    from kgcnn.layers.modules import LazyAdd, OptionalInputEmbedding
    from kgcnn.layers.mlp import GraphMLP, MLP
    from modules.pooling import PoolingNodes
    from modules.bessel_basis import BesselBasisLayer
    from model import make_model

    tf.get_logger().setLevel("ERROR")
    tf.keras.backend.set_floatx("float64")

    with open(args.model_dir / "preprocessor_orig.p", "rb") as handle:
        input_data = pickle.load(handle)
    preprocessor = input_data["preprocessor"]
    model = make_model()
    model.load_weights(args.model_dir / "best_model_val_mae.h5")

    class RBFSequence(GraphSequence):
        def process_data(self, batch_data):
            offset = compute_stacked_offsets(batch_data["n_pro"], batch_data["n_atom"])
            offset = np.where(batch_data["atom_index"] >= 0, offset, 0)
            batch_data["atom_index"] += offset
            for feature in ["node_attributes", "node_coordinates", "edge_indices", "atom_index", "n_pro"]:
                batch_data[feature] = ragged_const(batch_data[feature])
            for feature in ["n_atom", "n_bond", "distance", "bond", "node_graph_indices"]:
                if feature in batch_data:
                    del batch_data[feature]
            return batch_data

    prepared = []
    failures = []
    if args.prepared_cache_sdf is not None:
        supplier = Chem.SDMolSupplier(str(args.prepared_cache_sdf), removeHs=False)
        cache_count = 0
        for index, mol in enumerate(supplier):
            if mol is None:
                continue
            cache_count += 1
            prepared_mol = prepared_molecule_from_cache_mol(mol)
            if index == 0 or cache_count % 25 == 0:
                print(f"preparing {cache_count}: {prepared_mol.molecule_id}", flush=True)
            prepared.append((prepared_mol.molecule_id, prepared_mol))
    else:
        records = load_sdf_molecules(args.input)
        for index, (molecule_id, mol, smiles) in enumerate(records):
            if index == 0 or (index + 1) % 25 == 0:
                print(f"preparing {index + 1}/{len(records)}: {molecule_id}", flush=True)
            try:
                prepared.append((
                    molecule_id,
                    prepare_single_molecule(
                        molecule_id=molecule_id,
                        mol=mol,
                        source="sdf",
                        original_smiles=smiles,
                        random_seed=42 + index,
                        max_conformers=args.max_conformers,
                        max_iters=args.max_iters,
                        forcefield=args.forcefield,
                        time_limit_seconds=args.time_limit_seconds,
                        prefer_existing_coordinates=True,
                        allow_2d_if_h_nonzero=True,
                        optimize_existing_coordinates=args.optimize_existing_coordinates,
                        coord_route=args.coord_route,
                        route_initial_confs=args.route_initial_confs,
                        route_prune_rms_thresh=args.route_prune_rms_thresh,
                        route_coarse_steps=args.route_coarse_steps,
                        route_keep_top_k=args.route_keep_top_k,
                        route_fine_steps=args.route_fine_steps,
                    ),
                ))
            except Exception as exc:
                failures.append((molecule_id, type(exc).__name__, str(exc)))

    rows = []
    predict_items = []
    for molecule_id, prepared_mol in prepared:
        mol = prepared_mol.mol
        carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        if carbon_indices:
            predict_items.append({"molecule_id": molecule_id, "Mol": mol, "atom_index": np.array(carbon_indices, dtype=int)})

    if predict_items:
        inp_df = pd.DataFrame(predict_items)
        def mol_iter(df):
            for _, row in df.iterrows():
                yield row["Mol"], row["atom_index"]
        inputs_test = preprocessor.predict(mol_iter(inp_df))
        test_sequence = RBFSequence(inputs_test, batch_size=args.batch_size)
        predictions = []
        uncertainty = []
        for x in test_sequence:
            dist = model(x)
            predictions.extend(dist.mean().numpy().flatten())
            uncertainty.extend(dist.stddev().numpy().flatten())
        cursor = 0
        for item in predict_items:
            for atom_index in item["atom_index"]:
                pred = predictions[cursor]
                std = uncertainty[cursor]
                cursor += 1
                shift = round(float(pred) * 50.484337 + 99.798111, 2)
                conf_width = float(std) * 1.96 * 50.484337
                rows.append({
                    "molecule_id": item["molecule_id"],
                    "atom_index": int(atom_index),
                    "atom_type": "C",
                    "cascade2_13c_shift_ppm": shift,
                    "cascade2_13c_confidence_ppm": round(conf_width, 2),
                })

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "cascade2_13c_predictions.csv", index=False, encoding="utf-8-sig")
    writer = Chem.SDWriter(str(args.output_dir / "cascade2_annotated_predictions.sdf"))
    for molecule_id, prepared_mol in prepared:
        group = result[result["molecule_id"] == molecule_id].sort_values("atom_index")
        lines = []
        for serial, row in enumerate(group.itertuples(index=False)):
            lines.append(f"{serial}[{int(row.atom_index) + 1}]\t{float(row.cascade2_13c_shift_ppm):.2f}")
        out_mol = Chem.Mol(prepared_mol.mol)
        out_mol.SetProp("_Name", molecule_id)
        out_mol.SetProp("CASCADE2 Predicted 13C shifts", "\n".join(lines))
        writer.write(out_mol)
    writer.close()
    del prepared
    gc.collect()
    if failures:
        pd.DataFrame(failures, columns=["molecule_id", "error_type", "error_message"]).to_csv(
            args.output_dir / "cascade2_failed_molecules.tsv", sep="\t", index=False, encoding="utf-8"
        )
    print(f"saved: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
