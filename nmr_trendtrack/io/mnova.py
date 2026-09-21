from __future__ import annotations

from nmr_trendtrack.contracts import Sample
from nmr_trendtrack.io.peaklist import load_peaklist_for_sample


def load_mnova_peaklist(sample: Sample):
    """Load an Mnova-exported peak list via the generic table parser."""
    return load_peaklist_for_sample(sample)
