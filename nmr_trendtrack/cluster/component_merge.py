from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from nmr_trendtrack.contracts import (
    ComponentClusterPrototype,
    FinalClusterPrototype,
    Membership,
    TrendVector,
)


def _mask_to_str(mask: Tuple[int, ...]) -> str:
    return ''.join(str(int(v)) for v in mask)


def build_component_clusters(
    trend_vectors: List[TrendVector],
    memberships: List[Membership],
) -> Tuple[List[ComponentClusterPrototype], List[FinalClusterPrototype], List[Membership]]:
    """Build mask-level summaries and final natural-cluster summaries.

    Final cluster ids are now the refined best_cluster_id directly. This keeps the
    output semantics simple: the exported cluster_id is the natural partition, not
    a second synthetic merge layer.
    """
    mask_by_track: Dict[str, Tuple[int, ...]] = {tv.track_id: tv.presence_mask for tv in trend_vectors}
    source_clusters_by_component: Dict[str, set[str]] = defaultdict(set)
    tracks_by_component: Dict[str, List[str]] = defaultdict(list)
    tracks_by_final: Dict[str, List[str]] = defaultdict(list)

    for m in memberships:
        mask = mask_by_track.get(m.track_id, ())
        mask_str = _mask_to_str(mask)
        component_id = f'cmask_{mask_str}'
        m.component_cluster_id = component_id
        tracks_by_component[component_id].append(m.track_id)
        if m.best_cluster_id:
            source_clusters_by_component[component_id].add(m.best_cluster_id)

        final_id = m.best_cluster_id or component_id
        m.final_cluster_id = final_id
        tracks_by_final[final_id].append(m.track_id)

    component_out: List[ComponentClusterPrototype] = []
    for cid in sorted(tracks_by_component):
        mask_str = cid.replace('cmask_', '')
        mask = tuple(int(ch) for ch in mask_str) if mask_str else ()
        component_out.append(
            ComponentClusterPrototype(
                component_cluster_id=cid,
                presence_mask=mask,
                n_tracks=len(tracks_by_component[cid]),
                source_cluster_ids=sorted(source_clusters_by_component.get(cid, set())),
            )
        )

    final_out: List[FinalClusterPrototype] = []
    for cid in sorted(tracks_by_final):
        mask = mask_by_track.get(tracks_by_final[cid][0], ()) if tracks_by_final[cid] else ()
        final_out.append(
            FinalClusterPrototype(
                cluster_id=cid,
                presence_mask=mask,
                n_tracks=len(tracks_by_final[cid]),
                source_cluster_ids=[cid],
                merge_mode='natural_cluster',
            )
        )

    return component_out, final_out, memberships
