from .bucket import bucket_by_presence_mask
from .t_mixture import fit_all_buckets as fit_all_buckets_t, fit_t_mixture_for_bucket
from .hac_cluster import fit_all_buckets as fit_all_buckets_hac, fit_hac_for_bucket
from .shared_logic import assign_label_from_probs
from .natural_partition import refine_natural_clusters
from .component_merge import build_component_clusters


def fit_all_buckets(buckets, cfg):
    backend = getattr(cfg, 'cluster_backend', 't_mixture')
    if backend == 'hac':
        return fit_all_buckets_hac(buckets, cfg)
    return fit_all_buckets_t(buckets, cfg)


__all__ = [
    'bucket_by_presence_mask',
    'fit_all_buckets',
    'fit_all_buckets_t',
    'fit_all_buckets_hac',
    'fit_t_mixture_for_bucket',
    'fit_hac_for_bucket',
    'assign_label_from_probs',
    'refine_natural_clusters',
    'build_component_clusters',
]
