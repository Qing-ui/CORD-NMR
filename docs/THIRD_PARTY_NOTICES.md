# Third-party prediction components

CORD-NMR can install and invoke the third-party prediction components listed
below. Their copyrights remain with their respective authors. CORD-NMR does
not change their licenses or provide any additional warranty.

## NMRNet

- Project: NMRNet
- Authors: Fanjie Xu et al.
- Source: https://github.com/Colin-Jay/NMRNet
- Model record: https://zenodo.org/records/19142375
- Code license: MIT License, copyright AI4EC
- Model record license: Creative Commons Attribution 4.0 International
- Citation: https://doi.org/10.1038/s43588-025-00783-z

The CORD-NMR runtime uses only the released liquid-state carbon and proton
fine-tuned weights, their target scalers, and the atom dictionary required for
inference.

## CASCADE-2.0

- Project: CASCADE-2.0
- Authors: Abhijeet Bhadauria, Zhitao Feng, Mihai Popescu, and Robert Paton
- Source project: https://github.com/patonlab/CASCADE
- Preprint: https://doi.org/10.26434/chemrxiv-2025-r8m9m
- License: MIT License, copyright Abhijeet Bhadauria and Robert Paton / Paton Lab

The CORD-NMR runtime uses only the `Predict_SMILES_FF_GPR` inference model and
the Python modules required to load it.

## Uni-Core

- Project: Uni-Core
- Source: https://github.com/dptech-corp/Uni-Core
- License: MIT License, copyright DP Technology

CORD-NMR downloads a pinned Uni-Core source revision during NMRNet runtime
installation. Optional fused CUDA extensions are not required for the default
Windows CPU runtime.

## Disclaimer

Third-party components are provided under their original licenses and without
warranty. Users are responsible for complying with the applicable license
terms, attribution requirements, export controls, institutional policies, and
any restrictions that apply to their data or intended use.
