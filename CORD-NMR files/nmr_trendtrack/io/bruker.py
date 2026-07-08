from __future__ import annotations

from nmr_trendtrack.contracts import Sample
from nmr_trendtrack.io.peaklist import load_peaklist_for_sample


def load_bruker_peaklist(sample: Sample):
    """Load a Bruker peak list export via the generic table parser.

    This project targets peak-list driven alignment and clustering, not raw FID processing.
    Provide a Bruker-exported peak list in text/csv/tsv form.
    """
    return load_peaklist_for_sample(sample)
