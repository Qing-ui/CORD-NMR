from __future__ import annotations

from typing import Dict, Optional, Tuple


def assign_label_from_probs(
    probs: Dict[str, float],
    cluster_meta: Dict[str, Dict[str, object]],
    shared_prob_min: float,
    shared_gap_max: float,
    uncertain_prob_max: float,
    allow_shared_reuse: bool = True,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    if not probs:
        return "uncertain", None, None, None
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    best_id, p1 = ranked[0]
    second_id, p2 = ranked[1] if len(ranked) > 1 else (None, 0.0)
    best_family = None if best_id is None else cluster_meta.get(best_id, {}).get("family_id")
    if p1 < uncertain_prob_max:
        return "uncertain", best_id, second_id, best_family  # type: ignore[arg-type]
    if allow_shared_reuse and second_id is not None and p2 >= shared_prob_min and (p1 - p2) <= shared_gap_max:
        meta1 = cluster_meta.get(best_id, {})
        meta2 = cluster_meta.get(second_id, {})
        fam1 = meta1.get("family_id")
        fam2 = meta2.get("family_id")
        rank1 = meta1.get("amplitude_rank")
        rank2 = meta2.get("amplitude_rank")
        if fam1 is not None and fam1 == fam2 and rank1 is not None and rank2 is not None and abs(int(rank1) - int(rank2)) == 1:
            return "shared", best_id, second_id, fam1  # type: ignore[arg-type]
    return "pure", best_id, second_id, best_family  # type: ignore[arg-type]
