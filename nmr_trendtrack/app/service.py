from __future__ import annotations

from pathlib import Path

from nmr_trendtrack.app.config_loader import load_config, load_manifest
from nmr_trendtrack.models import run_switchable_model
from nmr_trendtrack.postprocess import (
    write_summary,
    write_tracks_tables,
    write_cluster_prototypes,
    write_component_cluster_prototypes,
    write_final_cluster_prototypes,
    write_memberships,
)


def run_joint_from_files(config_path: str, manifest_path: str, output_dir: str, model: str | None = None) -> None:
    config = load_config(config_path)
    if model:
        config.model.name = model
    samples = load_manifest(manifest_path)
    state = run_switchable_model(samples, config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_tracks_tables(state, output_dir)
    write_cluster_prototypes(state, output_dir)
    write_component_cluster_prototypes(state, output_dir)
    write_final_cluster_prototypes(state, output_dir)
    write_memberships(state, output_dir)
    write_summary(state, output_dir)
