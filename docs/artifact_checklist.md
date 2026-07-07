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
- Paper Figure 9: single-keystroke confusion matrix, Top-1, Top-3, and per-digit recall
- Paper Figure 10: sequence-length recovery and Top-k sensitivity plots
- Paper Table 1: 4/6/8-digit attack success rates within 1-5 attempts
- Appendix Table 7: 6-digit component ablation values
- Paper Figure 11(a-b) subset: hand-size/posture and battery/background-load robustness summaries

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

The artifact currently documents the following reproduction-target mapping checked against `ccs2026a-paper1084.pdf`:

- Table 1: attack success rate within N attempts for 4/6/8-digit PINs
- Figure 9: single-keystroke confusion matrix
- Figure 10: MLP-only and physics-guided recovery across sequence lengths, plus Top-k sensitivity
- Figure 11(a-b): bundled robustness subset
- Appendix Table 7: morphology/spatial/temporal ablation
- Table 2: related-work comparison, not a computational reproduction target

Figures 11(c-d), 12, 13, 15, 16, and 17 require additional device/charger/physical-variation or training-sweep artifacts and are not part of the core released-checkpoint AEC workflow.

## Expected Numeric Tolerance

- Released-checkpoint quick test and evaluation scripts should be deterministic in normal CPU-only execution.
- Exact matches should usually be achievable for checkpoint-based metrics.
- A tolerance of `+- 0.1` percentage points is reasonable for checkpoint-based summaries if minor library-version differences appear.
- If the optional legacy training scripts are run manually, small run-to-run variation is expected and a wider tolerance such as `+- 1.0` percentage point is more realistic.
