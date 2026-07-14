# NMR predictor runtime

The tracked `app/` directory contains the prediction bridge source used by the
CORD-NMR GUI. Large Python environments and model weights are intentionally not
stored in Git.

The portable release uses this layout:

```text
external/NMR-Predictor-Portable/
  app/
  envs/
    nmrnet/
    cascade2/
  models/
    nmrnet/
    cascade2/
```

The GUI also accepts another runtime location through its predictor-directory
selector. Model weights and third-party runtime components remain subject to
their respective upstream terms.
