from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Sample:
    sample_id: str
    order_index: int
    source_type: str
    peaklist_path: Optional[str] = None
    spectrum_path: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class Peak:
    peak_id: str
    sample_id: str
    ppm_raw: float
    intensity: float
    area: Optional[float] = None
    width_hz: Optional[float] = None
    snr: Optional[float] = None
    quality_flag: Optional[str] = None
    ppm_corr: Optional[float] = None

    def corrected_ppm(self) -> float:
        return self.ppm_raw if self.ppm_corr is None else self.ppm_corr


@dataclass
class AlignmentCandidate:
    peak_a_id: str
    peak_b_id: str
    sample_a: str
    sample_b: str
    ppm_delta_raw: float
    ppm_delta_corrected: float
    allowed_window: float
    score_ppm: float


@dataclass
class Track:
    track_id: str
    members: Dict[str, Peak]
    center_ppm: float
    ppm_span: float
    presence_mask: Tuple[int, ...] = ()
    quality_score: float = 0.0
    is_outlier: bool = False
    trend_bonus: float = 0.0

    def member_peak_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(p.peak_id for p in self.members.values()))


@dataclass
class TrendVector:
    track_id: str
    presence_mask: Tuple[int, ...]
    step_log_fc: List[Optional[float]]
    valid_steps: List[bool]


@dataclass
class ClusterPrototype:
    cluster_id: str
    presence_mask: Tuple[int, ...]
    mean_step_log_fc: List[float]
    step_scale: List[float]
    n_tracks: int = 0
    weight: float = 0.0
    family_id: Optional[str] = None
    family_direction: Optional[List[float]] = None
    amplitude_center: Optional[float] = None
    amplitude_scale: Optional[float] = None
    amplitude_rank: Optional[int] = None


@dataclass
class Membership:
    track_id: str
    cluster_probs: Dict[str, float]
    best_cluster_id: Optional[str]
    second_cluster_id: Optional[str]
    assigned_label: str
    family_id: Optional[str] = None
    component_cluster_id: Optional[str] = None
    final_cluster_id: Optional[str] = None


@dataclass
class ComponentClusterPrototype:
    component_cluster_id: str
    presence_mask: Tuple[int, ...]
    n_tracks: int = 0
    source_cluster_ids: List[str] = field(default_factory=list)


@dataclass
class FinalClusterPrototype:
    cluster_id: str
    presence_mask: Tuple[int, ...]
    n_tracks: int = 0
    source_cluster_ids: List[str] = field(default_factory=list)
    merge_mode: str = 'keep_best_cluster'


@dataclass
class JointState:
    samples: List[Sample]
    ordered_sample_ids: List[str]
    peaks_original: Dict[str, List[Peak]]
    peaks_corrected: Dict[str, List[Peak]]
    shifts: Dict[str, float]
    warp_maps: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    tracks: List[Track] = field(default_factory=list)
    trend_vectors: List[TrendVector] = field(default_factory=list)
    cluster_prototypes: List[ClusterPrototype] = field(default_factory=list)
    component_cluster_prototypes: List[ComponentClusterPrototype] = field(default_factory=list)
    final_cluster_prototypes: List[FinalClusterPrototype] = field(default_factory=list)
    memberships: List[Membership] = field(default_factory=list)
    sample_scales: Dict[str, float] = field(default_factory=dict)
    objective_value: float = float("-inf")
    outer_iterations_completed: int = 0
    best_iteration: int = 0
    converged: bool = False
