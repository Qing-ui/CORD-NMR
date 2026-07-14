from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class AlignConfig:
    ppm_window_default: float = 0.18
    ppm_window_by_region: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 60.0, 0.20), (60.0, 120.0, 0.18), (120.0, 220.0, 0.15)]
    )
    coarse_match_window: float = 2.0
    max_track_span_ppm: float = 0.24
    min_track_size: int = 2
    max_missing_samples: int = 2
    allow_global_shift: bool = True
    shift_step_limit: float = 0.01
    max_component_size: int = 40
    max_tracks_per_component: int = 1500
    max_candidates_per_peak_pair: int = 50000
    center_soft_factor: float = 1.35
    center_soft_penalty: float = 1.2
    require_one_compat_edge: bool = True
    trend_sign_conflict_penalty: float = 1.0
    trend_step_gap_penalty: float = 0.6
    trend_max_step_gap: float = 1.2
    candidate_min_score: float = 0.42
    width_similarity_weight: float = 0.30
    shape_similarity_weight: float = 0.22
    reciprocal_best_bonus: float = 0.18
    pair_score_weight: float = 1.20
    track_width_penalty_weight: float = 0.28
    track_shape_penalty_weight: float = 0.18
    local_support_window_ppm: float = 1.0
    local_support_weight: float = 1.35
    enable_local_square_component_prior: bool = True
    local_square_component_max_components: int = 3
    local_square_component_weight: float = 1.10


@dataclass
class TrendConfig:
    epsilon: float = 1e-8
    use_area_instead_of_height: bool = False
    normalization_method: str = "none"
    min_valid_steps: int = 1


@dataclass
class ClusterConfig:
    max_clusters_per_mask: int = 4
    min_cluster_size: int = 3
    shared_prob_min: float = 0.30
    shared_gap_max: float = 0.08
    uncertain_prob_max: float = 0.45
    student_t_nu: float = 4.0
    max_em_iters: int = 80
    em_tol: float = 1e-4
    min_variance: float = 1e-4
    max_families_per_mask: int = 3
    max_amplitude_bins_per_family: int = 1
    family_kappa: float = 8.0
    family_cosine_merge_min: float = 0.92
    family_distance_merge_max: float = 0.55
    family_sign_zero_band: float = 0.12
    use_directional_family_clustering: bool = False
    merge_similar_families: bool = True
    prefer_raw_trend_space: bool = True
    enable_amplitude_binning: bool = False
    amplitude_boundary_tau: float = 0.35
    amplitude_split_min_gain: float = 2.0
    amplitude_split_min_separation: float = 1.1
    amplitude_split_max_overlap: float = 0.42
    allow_shared_reuse: bool = True
    cluster_backend: str = "t_mixture"
    hac_metric: str = "euclidean"
    hac_linkage: str = "ward"
    hac_corr_weight: float = 0.20
    hac_temperature: float = 0.35
    hac_zscore: bool = True


@dataclass
class OptimizeConfig:
    n_outer_iters: int = 100
    n_starts: int = 1
    convergence_tol: float = 1e-4
    trend_bonus_weight: float = 0.75


@dataclass
class ModelConfig:
    # Default GUI/CLI model: V5-PMTC-R85, from raw blind peaklists.
    name: str = "v5_pmtc"

    # V4 front-end parameters. Raw-span is fixed at <=0.50 ppm in the V4 reference implementation.
    residual_gate: float = 0.15
    top_k_per_seed: int = 5
    max_per_sample: int = 3
    exact_limit: int = 12
    beam_width: int = 120
    node_limit: int = 100000
    setpacking_model_cost: float = 1.10
    setpacking_high_mask_bonus: float = 0.0

    # V5-PMTC-R85 thresholds used for the new45/span0.50 paper comparison.
    pmtc_max_tracks_by_n_samples: Dict[int, int] = field(default_factory=lambda: {3: 32, 4: 28, 5: 26})
    pmtc_frac_limit_by_n_samples: Dict[int, float] = field(default_factory=lambda: {3: 0.55, 4: 0.47, 5: 0.41})
    pmtc_min_cluster_size: int = 3

    # Optional enumerated V5 front-end. It generates all presence masks inside
    # the configured ppm/span window, then exact-covers each local window.
    enum_max_options_per_component: int = 160
    enum_node_limit: int = 350000
    enum_cluster_beam_width: int = 400

    # Guarded recall-quality backend. It starts from V5-PMTC labels and only
    # accepts profile merges or HAC splits when internal quality guards pass.
    guarded_quality_merge_threshold: float = 0.80
    guarded_quality_hac_threshold: float = 1.00
    guarded_quality_max_cluster_rise: int = 4

    # Final mask cleanup. By default, a lower-mask cluster is removed only when
    # it is fully explained by higher-mask clusters under one-to-one ppm matching.
    enable_mask_residual_filter: bool = True
    mask_residual_mode: str = "full_coverage_one_to_one"
    mask_residual_match_tol: float = 0.50
    mask_residual_min_remaining: int = 5
    mask_residual_max_cover_ratio: float = 1.50
    pool_single_sample_masks: bool = True
    enable_sample_specific_residual_peaks: bool = True


@dataclass
class AppConfig:
    align: AlignConfig = field(default_factory=AlignConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
