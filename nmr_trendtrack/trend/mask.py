from __future__ import annotations

from typing import List, Tuple

from nmr_trendtrack.contracts import Track


def build_presence_mask(track: Track, ordered_sample_ids: List[str]) -> Tuple[int, ...]:
    return tuple(1 if sid in track.members else 0 for sid in ordered_sample_ids)


def attach_presence_masks(tracks: list[Track], ordered_sample_ids: list[str]) -> None:
    for tr in tracks:
        tr.presence_mask = build_presence_mask(tr, ordered_sample_ids)
