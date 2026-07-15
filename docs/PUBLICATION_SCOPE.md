# Public repository scope

## Included

- GUI source and launch script.
- Carbon, HSQC, and combined scoring modules.
- GUI-reachable continuous-series and single-spectrum clustering code.
- Assignment and prediction bridge source required by GUI workflows.
- Dependency and runtime-layout documentation.

## Excluded

- Manuscript, supporting information, drafts, slides, and submission files.
- Experimental, benchmark, training, and unpublished research datasets.
- Exploratory notebooks, temporary scripts, intermediate results, and plots.
- Legacy clustering backends, benchmark-only ablations, compatibility wrappers,
  and export helpers that are not reachable from the current GUI.
- User-provided SDF, CSV, Excel, and generated SQLite databases.
- Conda environments, caches, compiled bytecode, and model weights from Git.
- Portable ZIP archives from Git history; curated runtime archives belong only
  in GitHub Releases.

The publication check in `scripts/check_publication.py` enforces these rules
before repository updates and in continuous integration.
