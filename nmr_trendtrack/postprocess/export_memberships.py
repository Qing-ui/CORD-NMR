from __future__ import annotations

from pathlib import Path

import pandas as pd

from nmr_trendtrack.contracts import JointState


def write_memberships(state: JointState, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tv in state.trend_vectors:
        row = {
            "track_id": tv.track_id,
            "presence_mask": "".join(map(str, tv.presence_mask)),
        }
        for i, v in enumerate(tv.step_log_fc):
            row[f"step_{i+1}_log_fc"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "trend_vectors.csv", index=False)

    rows = []
    for m in state.memberships:
        row = {
            "track_id": m.track_id,
            "cluster_id": m.final_cluster_id,
            "family_id": m.family_id,
            "best_cluster_id": m.best_cluster_id,
            "second_cluster_id": m.second_cluster_id,
            "component_cluster_id": m.component_cluster_id,
            "assigned_label": m.assigned_label,
        }
        for cid, prob in sorted(m.cluster_probs.items()):
            row[f"prob_{cid}"] = prob
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "memberships.csv", index=False)
