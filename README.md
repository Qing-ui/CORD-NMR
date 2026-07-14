# CORD-NMR

CORD-NMR is a Windows desktop application for natural-product NMR workflows.
The GUI combines carbon and HSQC dereplication, continuous-series and
single-spectrum clustering, NMR prediction, and atom-level assignment.

Current public release: **CORD-NMR 1.0.0**.

This repository contains the software needed to inspect and reproduce the GUI
logic. It deliberately does not contain the manuscript, supporting information,
research datasets, exploratory work, generated results, or user databases.

## Included workflows

- Carbon-only, HSQC-only, and combined scoring.
- Continuous-series and single-spectrum clustering.
- NMRNet and CASCADE-2.0 prediction bridges.
- Predicted-to-experimental NMR assignment workspace.

## Install from source

Python 3.10 is the supported baseline. On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python gui.py
```

`run_gui.bat` starts the same GUI using the active `python` command.

## Prediction runtime

The core GUI and scoring/clustering workflows install from this repository.
NMRNet and CASCADE-2.0 additionally require their model weights and specialized
Python environments. These large runtime components are excluded from Git and
are supplied only in a curated portable Release package.

The expected runtime layout is documented in
[`external/NMR-Predictor-Portable/README.md`](external/NMR-Predictor-Portable/README.md).
The GUI can also use a compatible predictor runtime selected from its interface.

## Runtime data

SDF, CSV, and Excel inputs remain on the user's computer. CORD-NMR creates
`chem_data.db`, result directories, plots, and prediction outputs locally while
the application runs. These files are ignored by Git and are not part of the
public software distribution.

## Repository layout

```text
gui.py                         desktop application entry point
Carbon* / HSQC* / Combine*     dereplication scoring and result views
cord_nmr_cluster.py            GUI-facing clustering orchestration
nmr_trendtrack/                continuous-series clustering implementation
single_spectrum/               single-spectrum correction and clustering
services/                      correction, prediction, and assignment services
external/.../app/              predictor bridge source (no weights or envs)
scripts/check_publication.py   repository publication guard
```

See [`docs/PUBLICATION_SCOPE.md`](docs/PUBLICATION_SCOPE.md) for the explicit
inclusion and exclusion policy.

## License

Copyright is reserved. Public visibility does not grant permission to copy,
modify, redistribute, sublicense, or use the software commercially. See
[`LICENSE`](LICENSE).
