# Exploiting Body-Coupled Leakage for Keystroke Inference on Smartphones

This repository has been repackaged as an ACM CCS-style artifact-evaluation bundle for a security/AI-agent paper on keystroke/PIN inference. The artifact keeps the original research scripts intact for provenance, but adds a reviewer-friendly execution layer under `scripts/`, `src/`, `docs/`, and `configs/` so an external evaluator can run a quick sanity check and reproduce the main released-checkpoint results with minimal setup.

The artifact is intentionally conservative. It does not invent missing datasets, results, or figure numbering. Where the original flat repository did not encode the final camera-ready table/figure mapping, this artifact uses explicit placeholders and marks those assumptions in [configs/reproduction_targets.json](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/configs/reproduction_targets.json) and [docs/repository_inventory.md](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/docs/repository_inventory.md).

## Project Overview

The repository studies side-channel keystroke inference for PIN entry using signal-processing, keystroke morphology classification, and temporal joint inference.

Core included components:

- Single-keystroke classifier evaluation using shipped checkpoints and `data/classification/test_data.csv`
- Multi-digit PIN reconstruction for 4/6/8/11/16-digit datasets using shipped checkpoints
- Robustness evaluation over hand-size, battery-level, and background-load datasets
- Quick feature-extraction sanity check on `data/raw_signals/PIN_163589.csv`

Core artifact entry points:

- `scripts/quick_test.sh`
- `scripts/reproduce_main_results.sh`
- `scripts/reproduce_table1.sh`
- `scripts/reproduce_table2.sh`
- `scripts/reproduce_figure1.sh`
- `scripts/reproduce_ablation.sh`
- `scripts/reproduce_robustness.sh`

The new Python orchestration code lives under `src/keystroke_artifact/`. Legacy research scripts are now grouped under `legacy/`, datasets under `data/`, and checkpoints under `checkpoints/models/`.

## Cleaned Repository Layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── Dockerfile
├── scripts/
├── src/keystroke_artifact/
├── configs/
├── docs/
├── data/
│   ├── classification/
│   ├── pin_reconstruction/
│   ├── robustness/
│   ├── raw_signals/
│   └── derived/
├── checkpoints/models/
├── outputs/
└── legacy/
    ├── processing/
    ├── training/
    ├── inference/
    ├── robustness/
    ├── visualization/
    └── orchestration/
```

## Environment Requirements

Recommended software environment:

- OS: Linux or macOS
- Python: 3.10 recommended
- Package manager: `pip` or `conda`
- Container option: Docker with a recent Linux host

Exact pinned Python package versions for the artifact path are in [requirements.txt](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/requirements.txt) and [environment.yml](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/environment.yml).

## Hardware Requirements

Recommended reviewer hardware:

- Quick test: 2 CPU cores, 4 GB RAM, no GPU required
- Full released-checkpoint reproduction: 4 CPU cores, 8 GB RAM, no GPU required
- Free disk space: at least 3 GB for the repository, environment, and generated outputs

GPU acceleration is optional and not required for the packaged artifact workflow.

## Software Requirements

The artifact path depends on:

- NumPy
- pandas
- SciPy
- matplotlib
- seaborn
- scikit-learn
- PyTorch
- torchvision
- tqdm
- psutil

The original repository also contains training and plotting scripts with additional presentation-oriented behavior, but they are not required for the core AEC workflow.

## Installation

### Option A: Native `pip` environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Option B: Conda environment

```bash
conda env create -f environment.yml
conda activate keystroke-artifact
pip install -e .
```

### Option C: Docker

```bash
docker build -t keystroke-artifact-aec .
docker run --rm -it keystroke-artifact-aec bash scripts/quick_test.sh
```

`docker-compose.yml` is intentionally omitted because the artifact is single-container and does not require multi-service orchestration.

## Quick Start

Run the self-contained sanity check:

```bash
bash scripts/quick_test.sh
```

Expected quick-test behavior:

- runs legacy feature extraction on `data/raw_signals/PIN_163589.csv`
- evaluates the shipped single-keystroke classifier on `data/classification/test_data.csv`
- evaluates 6-digit PIN reconstruction on `data/pin_reconstruction/full_test_data_6digit.csv`
- generates a confusion-matrix figure
- writes machine-readable outputs under `outputs/quick_test/`

## Full Reproduction

One-command full reproduction:

```bash
bash scripts/reproduce_main_results.sh
```

Individual targets:

```bash
bash scripts/reproduce_table1.sh
bash scripts/reproduce_table2.sh
bash scripts/reproduce_figure1.sh
bash scripts/reproduce_ablation.sh
bash scripts/reproduce_robustness.sh
```

## Expected Runtime

Approximate runtimes on a 4-core CPU-only machine:

- `scripts/quick_test.sh`: 5 minutes or less in typical conditions
- `scripts/reproduce_table1.sh`: under 2 minutes
- `scripts/reproduce_table2.sh`: under 10 minutes
- `scripts/reproduce_figure1.sh`: under 2 minutes
- `scripts/reproduce_ablation.sh`: under 10 minutes
- `scripts/reproduce_robustness.sh`: 10 to 30 minutes
- `scripts/reproduce_main_results.sh`: 20 to 45 minutes

These are practical estimates, not hard guarantees.

## Script-to-Paper Mapping

Because the repository does not encode the final camera-ready numbering, the following mappings are placeholders and should be updated before archival submission if the paper uses different identifiers:

| Script | Placeholder Paper Object | Output Directory | Notes |
| --- | --- | --- | --- |
| `scripts/reproduce_table1.sh` | Table 1 | `outputs/table1/` | Single-keystroke classification metrics |
| `scripts/reproduce_table2.sh` | Table 2 | `outputs/table2/` | PIN reconstruction across 4/6/8/11/16-digit datasets |
| `scripts/reproduce_figure1.sh` | Figure 1 | `outputs/figure1/` | Confusion matrix generated from `data/classification/test_data.csv` |
| `scripts/reproduce_ablation.sh` | Ablation | `outputs/ablation/` | 6-digit MLP-only vs joint-inference comparison |
| `scripts/reproduce_robustness.sh` | Robustness appendix / extra results | `outputs/robustness/` | Hand-size and battery/background-load evaluations |
| `scripts/reproduce_main_results.sh` | Main released-checkpoint artifact package | `outputs/main_results/` | Runs all packaged targets above |

See [configs/reproduction_targets.json](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/configs/reproduction_targets.json) for the same mapping in machine-readable form.

## Expected Outputs

Representative outputs include:

- CSV summaries under `outputs/*/tables/`
- JSON manifests under `outputs/*/`
- PNG/PDF figures under `outputs/*/figures/`
- `outputs/quick_test/SUCCESS.txt` for a simple pass/fail marker

The artifact runner writes deterministic outputs when using the shipped classifier checkpoint and the included datasets. For portability, the new runner recomputes normalization and kinematic statistics from the bundled training CSVs instead of depending on serialized auxiliary `.pth` files.

## Repository Audit Summary

Key findings from the repository audit:

- Main legacy entry points: `legacy/inference/*.py`, `legacy/visualization/Test_MLP_*.py`, `legacy/robustness/Impact_*.py`, `legacy/processing/keystroke_segmentation+feature_extraction.py`
- Experiment/training scripts: `MLP_training.py`, `fit_kinematic_model.py`, `mlp_training_sample_impact.py`, `sensitivity_analysis_variance.py`, `overhead_calculation.py`
- Config state: the original repository had almost no external configuration files; most settings were hard-coded inside scripts
- Included datasets: training/test CSVs, multi-digit reconstruction CSVs, hand-size robustness CSVs, and 120 battery/background-load CSVs
- Included checkpoints: `checkpoints/models/keystroke_morphology_mlp.pth`, `checkpoints/models/norm_params.pth`, `checkpoints/models/kinematic_params.pth`
- Plot/table generation status: several legacy plot scripts were interactive or partially hard-coded, so the artifact package now generates table/figure outputs through the new runner instead

Detailed inventory and identified blockers are in [docs/repository_inventory.md](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/docs/repository_inventory.md).

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy._core'`

Some bundled legacy `.pth` files were serialized in an environment that references `numpy._core`. This artifact includes [sitecustomize.py](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/sitecustomize.py) to help legacy scripts, and the new artifact runner avoids depending on those serialized auxiliary files by recomputing the needed statistics from the bundled training CSVs.

### Matplotlib or font-cache permission warnings

The artifact uses a repository-local cache directory via `sitecustomize.py`, so the wrapper scripts should avoid home-directory cache issues. If you need the older research code, prefer the wrapper scripts in `scripts/`; the moved `legacy/` tree is kept mainly for provenance.

### Reviewer launched scripts from another directory

The shell wrappers `cd` into the repository root before execution. If running Python commands manually, do the same.

### Full retraining does not match exactly

The core AEC path is checkpoint-based. Training scripts are preserved, but exact retraining parity is not guaranteed because the original repository does not ship a complete, externally configurable training pipeline.

## Limitations and Nondeterminism

- The packaged artifact reproduces the released-checkpoint evaluation path, not a full from-scratch retraining campaign.
- The original repository does not encode the final paper table/figure numbering. Placeholder mappings are documented and should be updated before final submission.
- Several legacy plot scripts remain presentation-oriented and are not used as the primary AEC entry points.
- Training-related experiments may show small variation if run manually outside the packaged artifact path.

## Contact

- Corresponding author: Tao Gu
- Email: tao.gu@mq.edu.au
- Artifact maintainer: Yao Wang (wangyao@xidian.edu.cn)

## Additional Documentation

- [docs/repository_inventory.md](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/docs/repository_inventory.md)
- [docs/artifact_checklist.md](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/docs/artifact_checklist.md)
- [docs/zenodo_release_notes.md](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/docs/zenodo_release_notes.md)
- [docs/aec_response_template.md](/Users/an/Downloads/Keystroke-Inference-Attack-Code-master/docs/aec_response_template.md)
