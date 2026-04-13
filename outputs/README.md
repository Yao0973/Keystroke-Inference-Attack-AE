# Outputs Directory

This directory is the canonical location for generated artifact outputs.

Expected subdirectories:

- `outputs/quick_test/`
- `outputs/table1/`
- `outputs/table2/`
- `outputs/figure1/`
- `outputs/ablation/`
- `outputs/robustness/`
- `outputs/main_results/`
- `outputs/precomputed/`

`outputs/precomputed/` stores historical result files that were already present before the artifact cleanup.

Each target writes machine-readable CSV/JSON outputs and, when relevant, PNG/PDF figures.
