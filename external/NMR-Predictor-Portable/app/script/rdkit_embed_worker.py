from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from shared_rdkit_prep import compute_embedding_result, export_embedding_metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    args = parser.parse_args()

    payload = pickle.loads(Path(args.input_pkl).read_bytes())
    try:
        result = compute_embedding_result(
            mol=payload["mol"],
            random_seed=payload["random_seed"],
            max_iters=payload["max_iters"],
            forcefield=payload["forcefield"],
            max_conformers=payload["max_conformers"],
            time_limit_seconds=payload["time_limit_seconds"],
            coord_route=payload["coord_route"],
            route_initial_confs=payload["route_initial_confs"],
            route_prune_rms_thresh=payload["route_prune_rms_thresh"],
            route_coarse_steps=payload["route_coarse_steps"],
            route_keep_top_k=payload["route_keep_top_k"],
            route_fine_steps=payload["route_fine_steps"],
        )
        if result is None:
            out = ("none", b"")
        else:
            out = (
                "ok",
                pickle.dumps(
                    {
                        "mol": result,
                        "metadata": export_embedding_metadata(result),
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
            )
    except Exception as exc:
        out = ("error", repr(exc).encode("utf-8", errors="replace"))

    Path(args.output_pkl).write_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
