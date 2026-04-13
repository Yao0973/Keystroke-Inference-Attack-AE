# Artifact Checklist

## Included in the Artifact

- Original research scripts preserved under `legacy/`
- Released checkpoints under `checkpoints/models/`
- Included evaluation datasets and robustness datasets already present in the repository
- Artifact wrapper scripts under `scripts/`
- Artifact runner package under `src/keystroke_artifact/`
- Reviewer documentation under `docs/`
- Environment files: `requirements.txt`, `environment.yml`, `Dockerfile`

## What a Reviewer Can Reproduce

- A quick end-to-end sanity check via `scripts/quick_test.sh`
- Single-keystroke classification metrics and per-digit recall
- PIN reconstruction accuracy across the included 4/6/8/11/16-digit datasets
- Confusion-matrix figure generated from the shipped classifier and `data/classification/test_data.csv`
- 6-digit ablation comparing MLP-only and joint-inference settings
- Hand-size and battery/background-load robustness summaries

## What Requires Larger Compute or More Time

- Full robustness reproduction is slower than the quick test because it evaluates 120 battery/background-load files plus 6 hand-size files.
- Optional legacy training scripts may take substantially longer and are not the default AEC workflow.

## What Requires Network Access

- Native installation with `pip install -r requirements.txt`
- Conda environment creation from `environment.yml`
- Docker image build if dependencies are not already cached locally

The packaged released-checkpoint evaluation itself does not require network access once the environment exists.

## What Is Optional

- `scripts/reproduce_robustness.sh`
- Legacy training scripts such as `MLP_training.py` and `mlp_training_sample_impact.py`
- Legacy presentation-oriented perturbation plotting scripts
- Docker workflow if the reviewer prefers native execution

## Outputs That Should Match the Paper

The artifact currently assumes the following placeholder mapping:

- Table 1: single-keystroke classification metrics
- Table 2: PIN reconstruction metrics across included PIN lengths
- Figure 1: single-keystroke confusion matrix
- Ablation: 6-digit MLP-only vs joint-inference comparison

Replace these placeholders with final camera-ready numbering before submission if needed.

## Expected Numeric Tolerance

- Released-checkpoint quick test and evaluation scripts should be deterministic in normal CPU-only execution.
- Exact matches should usually be achievable for checkpoint-based metrics.
- A tolerance of `+- 0.1` percentage points is reasonable for checkpoint-based summaries if minor library-version differences appear.
- If the optional legacy training scripts are run manually, small run-to-run variation is expected and a wider tolerance such as `+- 1.0` percentage point is more realistic.
