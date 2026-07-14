# NMR predictor runtime

The tracked `app/` directory contains the prediction bridge source used by the
CORD-NMR GUI. Large Python environments and model weights are intentionally not
stored in Git.

After extracting a CORD-NMR release on 64-bit Windows, double-click
`Install-CORD-NMR.bat` in the application root. The installer creates a private
GUI environment and separate NMRNet and CASCADE-2.0 environments, downloads
only the required model files, verifies their SHA-256 hashes, and places them
in the layout below.

The portable release uses this layout:

```text
external/NMR-Predictor-Portable/
  app/
  models/
    nmrnet/
    cascade2/
  runtime-paths.json
```

To avoid Windows path-length failures in TensorFlow, the Python environments
are installed under `%LOCALAPPDATA%\CORD-NMR\envs`. `runtime-paths.json` records
the GUI, NMRNet, and CASCADE-2.0 interpreter paths and links them to the
extracted CORD-NMR release. The model files remain inside the release directory.

The GUI also accepts another runtime location through its predictor-directory
selector. Model weights and third-party runtime components remain subject to
their respective upstream terms.

See [`MODEL_ASSETS.md`](MODEL_ASSETS.md) and
[`../../docs/THIRD_PARTY_NOTICES.md`](../../docs/THIRD_PARTY_NOTICES.md) for the
installed assets, sources, licenses, and citations.
