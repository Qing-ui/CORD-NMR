# CORD-NMR

CORD-NMR is a Windows desktop application for natural-product NMR workflows.
The GUI combines carbon and HSQC dereplication, continuous-series and
single-spectrum clustering, NMR prediction, and atom-level assignment.

Current public release: **CORD-NMR 1.0.2**.

The continuous-series implementation in this repository is synchronized with
the final CORD-NMR method. The continuous ¹³C workflow uses the
strict-contiguous SPTC front end (`gap = 0`) with up to 20 candidates per
starting peak. Both global strict and common-mask support mask-only, PMTC,
and QG-PMTC. The samples must be supplied in fraction order.

This repository contains the software needed to inspect and reproduce the GUI
logic. It deliberately does not contain the manuscript, supporting information,
research datasets, exploratory work, generated results, or user databases.

## Included workflows

- Carbon-only, HSQC-only, and combined library scoring with ReCQC 2.0 Speed.
- Continuous-series and single-spectrum clustering.
- NMRNet and CASCADE-2.0 prediction bridges.
- Predicted-to-experimental NMR assignment workspace.

Cross-spectrum intensity correction is off by default. When enabled, clustering
uses the corrected intensities. To use peak types in the correction, include a
`type`, `peak_type`, or `dept_type` column with assignments such as `CH` and
`CH2` in the input peak lists.

## Install from source

Python 3.10 is the supported baseline. On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python gui.py
```

For a source checkout, start the application with `python gui.py` after
activating the environment above. The Release launcher `run_gui.bat` uses the
isolated interpreter written by `Install-CORD-NMR.bat`.

## Prediction runtime

The core GUI and scoring/clustering workflows install from this repository.
NMRNet and CASCADE-2.0 additionally require their model weights and specialized
Python environments. These large runtime components are excluded from Git.
After extracting the Windows Release, double-click `Install-CORD-NMR.bat` once.
The installer creates an isolated GUI environment plus separate NMRNet and
CASCADE-2.0 environments, downloads the verified inference-only model assets,
and connects all three environments to `run_gui.bat`. It does not download
training data or unrelated models, and it does not require a preinstalled
Python distribution.

The expected runtime layout is documented in
[`external/NMR-Predictor-Portable/README.md`](external/NMR-Predictor-Portable/README.md).
The GUI can also use a compatible predictor runtime selected from its interface.
Third-party copyrights, licenses, sources, and citations are documented in
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md).

## Runtime data

SDF, CSV, and Excel inputs remain on the user's computer. CORD-NMR creates
`chem_data.db`, result directories, plots, and prediction outputs locally while
the application runs. These files are ignored by Git and are not part of the
public software distribution.

## Repository layout

```text
gui.py                         desktop application entry point
Carbon* / HSQC* / Combine*     dereplication scoring and result views
recqc_cluster.py               GUI-facing clustering orchestration
nmr_trendtrack/                continuous-series clustering implementation
single_spectrum/               single-spectrum correction and clustering
services/                      correction, prediction, and assignment services
external/.../app/              predictor bridge source (no weights or envs)
scripts/check_publication.py   repository publication guard
```

See [`docs/PUBLICATION_SCOPE.md`](docs/PUBLICATION_SCOPE.md) for the explicit
inclusion and exclusion policy.

The final implementation contract is summarized in
[`docs/FINAL_METHOD_CONTRACT.md`](docs/FINAL_METHOD_CONTRACT.md).

## Recommended reading

Deng C, Dong T, Zhang M, et al. Computer-Assisted Analytical Workflows for
Natural Product Dereplication, Structure Elucidation, and Bioactivity-Oriented
Prioritization. *Analytical Chemistry*. https://doi.org/10.1021/acs.analchem.6c02967

## License

The CORD-NMR application is distributed under [`LICENSE`](LICENSE).
The integrated ReCQC 2.0 Speed scoring modules retain their
[research and education license](docs/LICENSE-ReCQC-2.0-Speed.txt).
CORD-NMR adapts those modules to its scoring tabs and the direct search
actions in clustering results. The SDF reader preserves explicit atom
indices used by the prediction tags. The upstream source is
[ReCQC commit `80d873b`](https://github.com/Qing-ui/ReCQC/commit/80d873bdf7944f576c76f0f1b031f82bc0d09558).
