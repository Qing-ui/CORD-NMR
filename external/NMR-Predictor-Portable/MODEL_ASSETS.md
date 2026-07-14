# Prediction model assets

Model weights are distributed as GitHub Release assets rather than committed
to Git. Run `Install-CORD-NMR.bat` from an extracted CORD-NMR release to create
the required environments and download the verified model files.

Installed layout:

```text
external/NMR-Predictor-Portable/
  app/
  models/
    nmrnet/liquid/
    cascade2/Predict_SMILES_FF_GPR/
  licenses/
  runtime-paths.json
```

The isolated GUI and prediction environments are installed under
`%LOCALAPPDATA%\CORD-NMR\envs` to keep TensorFlow paths below the Windows path
length limit. `runtime-paths.json` records their absolute interpreter paths for
the GUI launcher and prediction bridge.

See `docs/THIRD_PARTY_NOTICES.md` for sources, licenses, and citations.
