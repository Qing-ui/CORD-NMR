from __future__ import annotations

from pathlib import Path

import pandas as pd

from nmr_trendtrack.contracts import JointState


def write_tracks_tables(state: JointState, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tracks_rows = []
    member_rows = []
    for tr in state.tracks:
        tracks_rows.append(
            {
                "track_id": tr.track_id,
                "center_ppm": tr.center_ppm,
                "ppm_span": tr.ppm_span,
                "presence_mask": "".join(map(str, tr.presence_mask)),
                "quality_score": tr.quality_score,
                "trend_bonus": tr.trend_bonus,
                "n_members": len(tr.members),
            }
        )
        for sid, peak in sorted(tr.members.items()):
            member_rows.append(
                {
                    "track_id": tr.track_id,
                    "sample_id": sid,
                    "peak_id": peak.peak_id,
                    "ppm_raw": peak.ppm_raw,
                    "ppm_corr": peak.corrected_ppm(),
                    "intensity": peak.intensity,
                    "area": peak.area,
                }
            )
    pd.DataFrame(tracks_rows).to_csv(out / "tracks.csv", index=False)
    pd.DataFrame(member_rows).to_csv(out / "track_members.csv", index=False)
