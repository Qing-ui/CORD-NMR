from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from nmr_trendtrack.contracts import TrendVector


def bucket_by_presence_mask(trend_vectors: List[TrendVector]) -> Dict[Tuple[int, ...], List[TrendVector]]:
    buckets = defaultdict(list)
    for tv in trend_vectors:
        buckets[tv.presence_mask].append(tv)
    return dict(buckets)
