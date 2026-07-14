from __future__ import annotations

import json
from pathlib import Path

from nmr_trendtrack.contracts import JointState


def write_summary(state: JointState, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_counts = {}
    for m in state.memberships:
        label_counts[m.assigned_label] = label_counts.get(m.assigned_label, 0) + 1
    component_counts = {}
    final_counts = {}
    for m in state.memberships:
        if m.component_cluster_id is not None:
            component_counts[m.component_cluster_id] = component_counts.get(m.component_cluster_id, 0) + 1
        if m.final_cluster_id is not None:
            final_counts[m.final_cluster_id] = final_counts.get(m.final_cluster_id, 0) + 1
    summary = {
        "n_samples": len(state.samples),
        "n_tracks": len(state.tracks),
        "n_cluster_prototypes": len(state.cluster_prototypes),
        "n_component_cluster_prototypes": len(state.component_cluster_prototypes),
        "n_final_cluster_prototypes": len(state.final_cluster_prototypes),
        "component_cluster_counts": component_counts,
        "final_cluster_counts": final_counts,
        "label_counts": label_counts,
        "sample_scales": state.sample_scales,
        "shifts": state.shifts,
        "objective_value": state.objective_value,
        "outer_iterations_completed": state.outer_iterations_completed,
        "best_iteration": state.best_iteration,
        "converged": state.converged,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
