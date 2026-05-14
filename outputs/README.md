# Outputs Directory

This directory is the canonical location for generated artifact outputs.

Expected subdirectories:

- `outputs/quick_test/`
- `outputs/table1/`
- `outputs/figure9/`
- `outputs/figure10/`
- `outputs/table7_ablation/`
- `outputs/robustness/`
- `outputs/main_results/`
- `outputs/precomputed/`

`outputs/precomputed/` stores historical result files that were already present before the artifact cleanup. Local regression logs and transient rerun outputs are useful during development but are not required for archival artifact evaluation.

Each target writes machine-readable CSV/JSON outputs and, when relevant, PNG/PDF figures.
