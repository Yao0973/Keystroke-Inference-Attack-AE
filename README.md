# Exploiting Body-Coupled Leakage for Keystroke Inference on Smartphones

This repository is an ACM CCS-style artifact-evaluation bundle for the paper *Exploiting Body-Coupled Leakage for Keystroke Inference on Smartphones*. The artifact keeps the original research scripts intact for provenance, but adds a reviewer-friendly execution layer under `scripts/`, `src/`, `docs/`, and `configs/` so an external evaluator can run a quick sanity check and reproduce the main released-checkpoint results with minimal setup.

The artifact is intentionally conservative. It does not invent missing datasets, results, or figure numbering. Where the original flat repository did not encode the final camera-ready table/figure mapping, this artifact uses explicit documented assumptions in [configs/reproduction_targets.json](configs/reproduction_targets.json) and [docs/repository_inventory.md](docs/repository_inventory.md).

## Artifact Availability

Public artifact repository:

`https://github.com/Yao0973/Keystroke-Inference-Attack-AE`

Archival Zenodo DOI:

`https://doi.org/10.5281/zenodo.21237223`

Reviewed CCS 2026-A AE snapshot:

`https://github.com/Yao0973/Keystroke-Inference-Attack-AE/releases/tag/ccs2026-ae-v1.3`

For the paper artifact link, cite the archival Zenodo DOI above. The GitHub release is the reviewed source snapshot corresponding to the artifact-evaluation workflow; it is public and includes the bundled datasets, released checkpoints, reproduction scripts, plotting scripts, and documentation.

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
- `scripts/reproduce_figure9.sh`
- `scripts/reproduce_figure10.sh`
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

Exact pinned Python package versions for the artifact path are in [requirements.txt](requirements.txt) and [environment.yml](environment.yml).

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
bash scripts/reproduce_figure9.sh
bash scripts/reproduce_figure10.sh
bash scripts/reproduce_ablation.sh
bash scripts/reproduce_robustness.sh
```

## Model and Training Configuration

The paper uses a lightweight MLP to map a five-dimensional keystroke feature vector to probabilities over the ten numeric-key classes. The released artifact contains the implementation and the configuration used for the reported experiments.

| Layer | Input dimension | Output dimension | Activation / normalization | Dropout |
| --- | ---: | ---: | --- | ---: |
| FC block 1 | 5 | 64 | ReLU + batch normalization | 0.3 |
| FC block 2 | 64 | 128 | ReLU + batch normalization | 0.3 |
| FC block 3 | 128 | 64 | ReLU + batch normalization | 0.2 |
| Output layer | 64 | 10 | Softmax | — |

| Training parameter | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 5e-3 |
| Learning-rate schedule | CosineAnnealingLR (T_max=35) |
| Batch size | 128 |
| Training epochs | 35 |
| Loss | Weighted cross-entropy |
| Random seed | 2025 |
| Reference environment | PyTorch 2.9.1; Intel Core i7-10700 @ 2.90 GHz; 64 GB DDR4 RAM |

### User-independent evaluation protocol

The participant split is performed before model training, rather than at the sample level. Data from 16 participants (8,000 keystroke samples) form the training set; data from six disjoint participants (3,000 keystroke samples) form the held-out test set. This protocol evaluates generalization to unseen participants rather than per-user calibration.

The primary artifact workflow evaluates the released checkpoint on the bundled held-out data. The repository preserves training code and the configuration above, but the artifact does not claim bit-for-bit parity for full retraining across all environments; see [Limitations and Nondeterminism](#limitations-and-nondeterminism).

## Expected Runtime

Approximate runtimes on a 4-core CPU-only machine:

- `scripts/quick_test.sh`: 5 minutes or less in typical conditions
- `scripts/reproduce_table1.sh`: under 2 minutes
- `scripts/reproduce_figure9.sh`: under 2 minutes
- `scripts/reproduce_figure10.sh`: under 10 minutes
- `scripts/reproduce_ablation.sh`: under 10 minutes
- `scripts/reproduce_robustness.sh`: 10 to 30 minutes
- `scripts/reproduce_main_results.sh`: 20 to 45 minutes

These are practical estimates, not hard guarantees.

## Script-to-Paper Mapping

The following mappings were checked against `ccs2026a-paper1084.pdf`:

| Script | Reproduced Object | Output Directory | Notes |
| --- | --- | --- | --- |
| `scripts/reproduce_table1.sh` | Table 1 | `outputs/table1/` | Attack success rate within 1-5 attempts for 4/6/8-digit PINs |
| `scripts/reproduce_figure9.sh` | Figure 9 | `outputs/figure9/` | Confusion matrix generated from `data/classification/test_data.csv` |
| `scripts/reproduce_figure10.sh` | Figure 10 | `outputs/figure10/` | Sequence-length recovery and Top-k sensitivity plots |
| `scripts/reproduce_ablation.sh` | Appendix Table 7 | `outputs/table7_ablation/` | Morphology/spatial/temporal ablation on 6-digit PIN recovery |
| `scripts/reproduce_robustness.sh` | Figure 11(a-b) subset | `outputs/robustness/` | Hand-size/posture and battery/background-load robustness bundled with this artifact |
| `scripts/reproduce_main_results.sh` | Main released-checkpoint artifact package | `outputs/main_results/` | Runs all packaged targets above |

Paper Table 2 is a related-work comparison table rather than a computational result. Figures 11(c-d), 12, 13, 15, 16, and 17 depend on additional device/charger/physical-variation or training-sweep artifacts that are discussed in the paper but are not fully bundled as primary AEC reproduction targets.

`scripts/reproduce_figure10.sh` first regenerates the Figure 10 CSV/JSON data and then calls the plotting script:

```bash
python3 scripts/plot_figure10.py \
  --length-csv outputs/figure10/tables/figure10_sequence_length_recovery.csv \
  --topk-csv outputs/figure10/tables/figure10_topk_sensitivity.csv \
  --output-dir outputs/figure10
```

The resulting figure files are `outputs/figure10/figures/figure10_performance.png` and `outputs/figure10/figures/figure10_performance.pdf`.

See [configs/reproduction_targets.json](configs/reproduction_targets.json) for the same mapping in machine-readable form.

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

Detailed inventory and identified blockers are in [docs/repository_inventory.md](docs/repository_inventory.md).

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy._core'`

Some bundled legacy `.pth` files were serialized in an environment that references `numpy._core`. This artifact includes [sitecustomize.py](sitecustomize.py) to help legacy scripts, and the new artifact runner avoids depending on those serialized auxiliary files by recomputing the needed statistics from the bundled training CSVs.

### Matplotlib or font-cache permission warnings

The artifact uses a repository-local cache directory via `sitecustomize.py`, so the wrapper scripts should avoid home-directory cache issues. If you need the older research code, prefer the wrapper scripts in `scripts/`; the moved `legacy/` tree is kept mainly for provenance.

### Reviewer launched scripts from another directory

The shell wrappers `cd` into the repository root before execution. If running Python commands manually, do the same.

### Full retraining does not match exactly

The core AEC path is checkpoint-based. Training scripts are preserved, but exact retraining parity is not guaranteed because the original repository does not ship a complete, externally configurable training pipeline.

## Limitations and Nondeterminism

- The packaged artifact reproduces the released-checkpoint evaluation path, not a full from-scratch retraining campaign.
- The computational reproduction targets are mapped to the submitted paper numbering above; some qualitative/setup and related-work figures are documented but not regenerated by the core AEC workflow.
- Several legacy plot scripts remain presentation-oriented and are not used as the primary AEC entry points.
- Training-related experiments may show small variation if run manually outside the packaged artifact path.

## Contact

- Corresponding author: Tao Gu
- Email: tao.gu@mq.edu.au
- Artifact maintainer: Yao Wang (wangyao@xidian.edu.cn)

## Additional Documentation

- [docs/repository_inventory.md](docs/repository_inventory.md)
- [docs/artifact_checklist.md](docs/artifact_checklist.md)
- [docs/zenodo_release_notes.md](docs/zenodo_release_notes.md)
- [docs/aec_response_template.md](docs/aec_response_template.md)
