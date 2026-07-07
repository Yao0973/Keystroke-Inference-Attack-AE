# Repository Inventory and Artifact Audit

This document records the repository audit performed while converting the project into an ACM CCS-style artifact package.

## Main Code Entry Points

Primary legacy evaluation and attack scripts are now grouped under `legacy/`:

- `legacy/inference/infer_PINs.py`
- `legacy/inference/Inference_4digit.py`
- `legacy/inference/Inference_6digit.py`
- `legacy/inference/Inference_8digit.py`
- `legacy/inference/Inference_11digit.py`
- `legacy/inference/Inference_16digit.py`
- `legacy/visualization/Test_MLP_Top1.py`
- `legacy/visualization/Test_MLP_Top3.py`
- `legacy/robustness/Impact_of_Hand_Size.py`
- `legacy/robustness/Impact_of_Battery_and_BackgroundApps.py`
- `legacy/processing/keystroke_segmentation+feature_extraction.py`

New artifact-facing entry points added in this packaging pass:

- `scripts/quick_test.sh`
- `scripts/reproduce_main_results.sh`
- `scripts/reproduce_table1.sh`
- `scripts/reproduce_figure9.sh`
- `scripts/reproduce_figure10.sh`
- `scripts/plot_figure10.py`
- `scripts/reproduce_ablation.sh`
- `scripts/reproduce_robustness.sh`
- `python3 -m keystroke_artifact.runner ...`

## Experiment and Analysis Scripts

Training or analysis scripts preserved from the original repository:

- `legacy/training/MLP_training.py`
- `legacy/training/fit_kinematic_model.py`
- `legacy/training/mlp_training_sample_impact.py`
- `legacy/training/sensitivity_analysis_variance.py`
- `legacy/training/overhead_calculation.py`
- `legacy/inference/Different_Attempts_4digit.py`
- `legacy/inference/Different_Attempts_6digit.py`
- `legacy/inference/Different_Attempts_8digit.py`

Signal-processing and preprocessing scripts:

- `legacy/processing/Envelope demodulation.py`
- `legacy/processing/Spectral subtraction.py`
- `legacy/processing/Mechanism-aware spike segmentation.py`
- `legacy/processing/keystroke_segmentation+feature_extraction.py`

Legacy plotting/presentation scripts:

- `legacy/visualization/Confusion_Matrix.py`
- `legacy/robustness/Perturbations_Phonecase.py`
- `legacy/robustness/Perturbations_Clothing.py`
- `legacy/robustness/Perturbations_Screen_Protector.py`
- `legacy/robustness/Perturbations_Sitting_Posture.py`

## Configuration Files

Originally present:

- `requirements.txt`
- `setup.py`
- `CITATION.cff`
- `experiment_results.json`

Configuration behavior observed during the audit:

- Most experimental settings were hard-coded inside Python scripts.
- The original repository did not provide a centralized config system.
- No external YAML/JSON config files were used to control legacy experiments.

New configuration artifacts added in this packaging pass:

- `configs/reproduction_targets.json`
- `environment.yml`
- `Dockerfile`

## Included Datasets

Observed data files are now grouped under `data/`:

- `data/classification/train_data.csv`: 8,000 rows, single-keystroke training features
- `data/classification/test_data.csv`: 3,000 rows, single-keystroke test features
- `data/pin_reconstruction/full_training_data.csv`: 800 rows, 6-digit sequence training features/timestamps
- `data/pin_reconstruction/full_test_data_4digit.csv`: 300 rows
- `data/pin_reconstruction/full_test_data_6digit.csv`: 300 rows
- `data/pin_reconstruction/full_test_data_8digit.csv`: 300 rows
- `data/pin_reconstruction/full_test_data_11digit.csv`: 300 rows
- `data/pin_reconstruction/full_test_data_16digit.csv`: 300 rows
- `data/pin_reconstruction/6digit_PINs.csv`: 300 rows
- `data/robustness/hand_size/test_6digit_*_*.csv`: 6 robustness files for hand size and posture
- `data/robustness/battery_background/data_*_*_run*.csv`: 120 robustness files for battery/background-load evaluation
- `data/raw_signals/PIN_163589.csv`, `data/raw_signals/ScreenProtector_PIN_841023.csv`, `data/raw_signals/SittingPosture_PIN_140730.csv`, `data/raw_signals/ThickClothing_PIN_220746.csv`: raw signal traces used by legacy scripts
- `data/raw_signals/1234567890_original.CSV`, `data/raw_signals/noise.CSV`, `data/raw_signals/keystroke.csv`, `data/raw_signals/keystroke_sequence_170246_output.csv`: signal-processing inputs

No dataset download step is required for the packaged released-checkpoint workflow.

## Checkpoints and Model Dependencies

Included checkpoints are now under `checkpoints/models/`:

- `keystroke_morphology_mlp.pth`
- `norm_params.pth`
- `kinematic_params.pth`

Dependency finding:

- The bundled `.pth` files require a small NumPy compatibility alias in some environments. This is handled by `sitecustomize.py`.
- The new artifact runner loads the classifier weights from `keystroke_morphology_mlp.pth` but recomputes normalization and kinematic statistics from `train_data.csv` and `full_training_data.csv` for better portability.

## Plotting and Table Generation

Original repository state:

- Some plotting scripts were interactive and used `plt.show()`.
- `Confusion_Matrix.py` relied on a hard-coded matrix rather than regenerating from predictions.
- Several perturbation figure scripts were tuned for presentation and specific filenames.

Artifact packaging decision:

- Core tables and figures are now generated through `src/keystroke_artifact/runner.py`.
- Legacy plotting scripts are preserved for provenance, but they are no longer the recommended AEC entry points.

## Missing Pieces and Submission Blockers

Identified blockers that still need author attention before final archival submission:

- ✅ Paper title and contact information have been filled in `README.md` and Zenodo notes.
- ✅ Table/Figure numbering is documented in `configs/reproduction_targets.json` and `README.md`.
- ✅ Full from-scratch training is not required for the artifact evaluation; released-checkpoint workflow is the recommended path.
- ✅ Legacy perturbation figure scripts are retained for provenance and are not required for the artifact evaluation.

Known practical limitations:

- The released-checkpoint workflow is the most robust and evaluator-friendly path. Full retraining remains less polished.
- Some moved legacy scripts still assume the older flat layout and are retained mainly for provenance. The supported AEC interface is `scripts/*.sh`.
