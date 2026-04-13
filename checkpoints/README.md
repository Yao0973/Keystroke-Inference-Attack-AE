# Checkpoints Directory

Model artifacts are now grouped under `checkpoints/models/`:

- `keystroke_morphology_mlp.pth`
- `norm_params.pth`
- `kinematic_params.pth`

The artifact runner directly uses `keystroke_morphology_mlp.pth` and recomputes the auxiliary normalization and kinematic statistics from bundled training CSVs for portability.
