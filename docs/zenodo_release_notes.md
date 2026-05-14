# Zenodo Release Notes

## Artifact Title

`Exploiting Body-Coupled Leakage for Keystroke Inference on Smartphones - Artifact`

## Authors

`Tao Gu, Yao Wang`

## Version

`v1.0.0`

## Short Description

This archival release contains the ACM CCS-style artifact-evaluation package for a paper on keystroke/PIN inference from side-channel signals. The release includes released checkpoints, evaluation datasets already bundled with the repository, reviewer-friendly execution scripts, Docker metadata, and documentation for quick testing and main-result reproduction.

## Keywords

- security
- side-channel analysis
- keystroke inference
- PIN inference
- artifact evaluation
- machine learning
- signal processing

## License

MIT

## Relationship to the Paper

This artifact is the evaluation package associated with the accepted conference paper. It provides the released-checkpoint reproduction path used for artifact evaluation and documents the assumptions made where the original repository did not encode final paper table/figure numbering.

## Recommended Files to Include in the Archival Release

- `README.md`
- `LICENSE`
- `CITATION.cff`
- `requirements.txt`
- `environment.yml`
- `Dockerfile`
- `sitecustomize.py`
- `scripts/`
- `src/`
- `docs/`
- `configs/`
- `data/`
- `checkpoints/`
- `outputs/README.md` and `outputs/precomputed/` files needed for provenance

## Notes Before Upload

- Confirm that all release-specific fields are final.
- Confirm that the final archival snapshot matches the camera-ready paper numbering.
- Verify that no private data or accidental local-only files are present.
- Do not include local regression logs or transient cache directories such as `outputs/regression/`, `.artifact_cache/`, or `__pycache__/` in the archival release.
