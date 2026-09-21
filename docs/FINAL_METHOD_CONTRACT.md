# Final CORD-NMR method contract

This repository contains the implementation corresponding to the final
resubmission manuscript. It is the adopted method, not a collection of the
parameter or clustering experiments considered during revision.

The shared continuous-spectrum front end is set-packing trajectory clustering
(SPTC). Candidate trajectories satisfy the corrected-shift and raw-span
constraints, contain at most one peak per sample, and span at least two
samples. Candidates that share an input peak are mutually conflicting. The
production benchmark uses the strict-contiguous rule (`gap = 0`) and retains
20 candidates per seed in the benchmark evaluator. Connected conflict
components are solved with the bounded exact/beam-search implementation.

Two route organisations are retained: `global strict` preserves the original
occurrence masks, while `common-mask` projects high-mask trajectories into
shared sub-mask views. Each route supports `mask-only`, `PMTC`, and
`QG-PMTC`. PMTC bins trajectories by occurrence mask and organises each bucket
using centred log-intensity profiles and trend features. The recorded
sample-number-specific PMTC caps and fraction limits remain part of the final
benchmark contract. QG-PMTC starts from PMTC labels and accepts profile-based
merges or directional-HAC splits only when the unsupervised quality guards
support them; otherwise it returns the PMTC state. Coverage and
sample-specific residual cleanup, followed by the final minimum-cluster
filter, is applied consistently to all backend outputs.

The repository does not include the manuscript, supporting information,
benchmark truth labels, or exploratory revision outputs. The final benchmark
data and result tables remain in the controlled analysis package used to
prepare the manuscript.

## Deliberately excluded exploratory variants

The following are not adopted and must not be used to reproduce the reported
manuscript results: deleting PMTC size controls, forcing one set of PMTC caps
for every sample number, replacing PMTC splitting with BIC or permutation
tests, and any other pilot clustering variant. They remain outside this
repository's production method.
