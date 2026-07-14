from __future__ import annotations

from typing import List

from nmr_trendtrack.contracts import Sample


def ordered_sample_ids(samples: List[Sample]) -> List[str]:
    return [s.sample_id for s in sorted(samples, key=lambda s: s.order_index)]
