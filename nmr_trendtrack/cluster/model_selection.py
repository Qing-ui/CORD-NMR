from __future__ import annotations

from typing import Iterable


def choose_k_from_bucket_size(bucket_size: int, max_k: int, min_cluster_size: int) -> range:
    if bucket_size <= 1:
        return range(1, 2)
    upper = min(max_k, max(1, bucket_size // max(1, min_cluster_size)))
    upper = max(1, upper)
    return range(1, upper + 1)
