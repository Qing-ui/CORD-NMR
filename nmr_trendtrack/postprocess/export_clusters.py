from __future__ import annotations

from pathlib import Path

import pandas as pd

from nmr_trendtrack.contracts import JointState


def write_cluster_prototypes(state: JointState, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for proto in state.cluster_prototypes:
        row = {
            "cluster_id": proto.cluster_id,
            "family_id": proto.family_id,
            "presence_mask": "".join(map(str, proto.presence_mask)),
            "n_tracks": proto.n_tracks,
            "weight": proto.weight,
            "amplitude_center": proto.amplitude_center,
            "amplitude_scale": proto.amplitude_scale,
            "amplitude_rank": proto.amplitude_rank,
        }
        if proto.family_direction is not None:
            for i, v in enumerate(proto.family_direction):
                row[f"family_dir_{i+1}"] = v
        for i, v in enumerate(proto.mean_step_log_fc):
            row[f"step_{i+1}_mean"] = v
        for i, v in enumerate(proto.step_scale):
            row[f"step_{i+1}_scale"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "cluster_prototypes.csv", index=False)


def write_component_cluster_prototypes(state: JointState, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for proto in state.component_cluster_prototypes:
        rows.append({
            "component_cluster_id": proto.component_cluster_id,
            "presence_mask": "".join(map(str, proto.presence_mask)),
            "n_tracks": proto.n_tracks,
            "source_cluster_ids": "|".join(proto.source_cluster_ids),
        })
    pd.DataFrame(rows).to_csv(out / "component_cluster_prototypes.csv", index=False)


def write_final_cluster_prototypes(state: JointState, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for proto in state.final_cluster_prototypes:
        rows.append({
            "cluster_id": proto.cluster_id,
            "presence_mask": "".join(map(str, proto.presence_mask)),
            "n_tracks": proto.n_tracks,
            "merge_mode": proto.merge_mode,
            "source_cluster_ids": "|".join(proto.source_cluster_ids),
        })
    pd.DataFrame(rows).to_csv(out / "final_cluster_prototypes.csv", index=False)
