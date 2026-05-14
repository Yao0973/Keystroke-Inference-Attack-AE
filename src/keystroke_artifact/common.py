from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

MODULE_REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_CACHE_ROOT = MODULE_REPO_ROOT / ".artifact_cache"
MODULE_MPL_CACHE = MODULE_CACHE_ROOT / "matplotlib"
MODULE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MODULE_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(MODULE_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(MODULE_MPL_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, recall_score

sys.modules.setdefault("numpy._core", np.core)


ARTIFACT_SEED = 2025
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs"
LEGACY_ROOT = REPO_ROOT / "legacy"
DATA_ROOT = REPO_ROOT / "data"
CLASSIFICATION_DATA_ROOT = DATA_ROOT / "classification"
PIN_RECONSTRUCTION_DATA_ROOT = DATA_ROOT / "pin_reconstruction"
RAW_SIGNAL_ROOT = DATA_ROOT / "raw_signals"
DERIVED_DATA_ROOT = DATA_ROOT / "derived"
ROBUSTNESS_ROOT = DATA_ROOT / "robustness"
HAND_SIZE_ROOT = ROBUSTNESS_ROOT / "hand_size"
BATTERY_BACKGROUND_ROOT = ROBUSTNESS_ROOT / "battery_background"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "models"

FEATURE_COLUMNS = ["Peak", "Energy", "FWHM", "RiseTime", "Centroid"]
PIN_DATASETS = {
    4: PIN_RECONSTRUCTION_DATA_ROOT / "full_test_data_4digit.csv",
    6: PIN_RECONSTRUCTION_DATA_ROOT / "full_test_data_6digit.csv",
    8: PIN_RECONSTRUCTION_DATA_ROOT / "full_test_data_8digit.csv",
    11: PIN_RECONSTRUCTION_DATA_ROOT / "full_test_data_11digit.csv",
    16: PIN_RECONSTRUCTION_DATA_ROOT / "full_test_data_16digit.csv",
}

KEY_POS = {
    1: (0, 0),
    2: (0, 1),
    3: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    7: (2, 0),
    8: (2, 1),
    9: (2, 2),
    0: (3, 1),
}


class ArtifactError(RuntimeError):
    """Raised when the artifact package cannot proceed reproducibly."""


class MorphologyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


def set_reproducible(seed: int = ARTIFACT_SEED) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_files_exist(paths: Sequence[Path]) -> None:
    missing = [str(path.relative_to(REPO_ROOT)) for path in paths if not path.exists()]
    if missing:
        raise ArtifactError(
            "Missing required artifact files:\n- " + "\n- ".join(missing)
        )


def save_json(path: Path, payload: Dict) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_table(path: Path, frame: pd.DataFrame) -> None:
    ensure_directory(path.parent)
    frame.to_csv(path, index=False)


def write_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        ensure_directory(destination.parent)
        shutil.copy2(source, destination)


def dependency_versions() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "seaborn": sns.__version__,
    }


def compute_normalization_bundle() -> Dict[str, np.ndarray]:
    train_data_path = CLASSIFICATION_DATA_ROOT / "train_data.csv"
    ensure_files_exist([train_data_path])
    frame = pd.read_csv(train_data_path)
    features = frame[FEATURE_COLUMNS].values.astype(np.float32)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": mean, "std": std}


def compute_kinematic_bundle() -> Dict[int, Dict[str, float]]:
    training_path = PIN_RECONSTRUCTION_DATA_ROOT / "full_training_data.csv"
    ensure_files_exist([training_path])
    frame = pd.read_csv(training_path, header=0)
    if frame.shape[1] != 37:
        raise ArtifactError(
            f"{training_path.name} has {frame.shape[1]} columns, expected 37."
        )

    samples = {level: [] for level in range(8)}
    base_delays = [0.05, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42]

    for _, row in frame.iterrows():
        pin_str = str(int(row.iloc[0])).zfill(6)
        if len(pin_str) != 6:
            continue
        timestamps = [float(row.iloc[index]) for index in range(1, 7)]
        for position in range(1, 6):
            source_digit = int(pin_str[position - 1])
            target_digit = int(pin_str[position])
            delta_t = timestamps[position] - timestamps[position - 1]
            if delta_t <= 0 or delta_t > 2.0:
                continue
            level = kinematic_level(source_digit, target_digit)
            samples[level].append(delta_t)

    params = {}
    for level in range(8):
        if not samples[level]:
            params[level] = {"mu": base_delays[level], "sigma": 0.02}
        else:
            params[level] = {
                "mu": float(np.mean(samples[level])),
                "sigma": float(np.std(samples[level])) + 1e-6,
            }
    return params


def load_model_bundle() -> Tuple[MorphologyMLP, Dict[str, np.ndarray], Dict[int, Dict[str, float]]]:
    model_path = CHECKPOINT_ROOT / "keystroke_morphology_mlp.pth"
    ensure_files_exist([model_path])

    model = MorphologyMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()

    norm_bundle = compute_normalization_bundle()
    kinematic_bundle = compute_kinematic_bundle()
    return model, norm_bundle, kinematic_bundle


def evaluate_single_digit_classifier() -> Dict:
    set_reproducible()
    dataset_path = CLASSIFICATION_DATA_ROOT / "test_data.csv"
    ensure_files_exist([dataset_path])
    model, norm, _ = load_model_bundle()

    frame = pd.read_csv(dataset_path)
    features = frame[FEATURE_COLUMNS].values.astype(np.float32)
    labels = frame["label"].values.astype(np.int64)
    features = (features - norm["mean"]) / norm["std"]

    with torch.no_grad():
        logits = model(torch.from_numpy(features).float())
        top1 = logits.argmax(dim=1).cpu().numpy()
        top3 = torch.topk(logits, k=3, dim=1).indices.cpu().numpy()

    top1_accuracy = float((top1 == labels).mean() * 100.0)
    top3_accuracy = float(np.mean([label in guesses for label, guesses in zip(labels, top3)]) * 100.0)
    recalls = recall_score(labels, top1, average=None, labels=np.arange(10))
    matrix = confusion_matrix(labels, top1, labels=np.arange(10))

    return {
        "dataset": dataset_path.name,
        "num_samples": int(len(frame)),
        "top1_accuracy_pct": round(top1_accuracy, 4),
        "top3_accuracy_pct": round(top3_accuracy, 4),
        "per_class_recall_pct": {
            str(index): round(float(value) * 100.0, 4) for index, value in enumerate(recalls)
        },
        "confusion_matrix": matrix.tolist(),
    }


def euclidean_distance(source: int, target: int) -> float:
    if source == target:
        return 0.0
    x1, y1 = KEY_POS[source]
    x2, y2 = KEY_POS[target]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def kinematic_level(source: int, target: int) -> int:
    distance = euclidean_distance(source, target)
    if distance == 0:
        return 0
    if distance <= 1:
        return 1
    if distance <= 1.42:
        return 2
    if distance <= 2:
        return 3
    if distance <= 2.24:
        return 4
    if distance <= 2.83:
        return 5
    if distance <= 3:
        return 6
    return 7


def log_gaussian_pdf(value: float, mean: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-4)
    return -0.5 * ((value - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def evaluate_pin_dataset(
    csv_path: Path,
    pin_length: int,
    top_k: int = 3,
    lambda_value: float = 1.5,
    use_kinematic: bool = True,
) -> Dict:
    set_reproducible()
    ensure_files_exist([csv_path])
    model, norm, kinematic = load_model_bundle()
    frame = pd.read_csv(csv_path, header=0)
    expected_columns = 1 + pin_length + pin_length * 5
    if frame.shape[1] != expected_columns:
        raise ArtifactError(
            f"{csv_path.name} has {frame.shape[1]} columns, expected {expected_columns}."
        )

    correct_joint = 0
    correct_mlp = 0
    prediction_rows: List[Dict[str, str]] = []
    feature_offset = 1 + pin_length

    for _, row in frame.iterrows():
        true_pin = str(int(row.iloc[0])).zfill(pin_length)
        timestamps = [float(row.iloc[index]) for index in range(1, pin_length + 1)]
        all_logits: List[np.ndarray] = []
        candidate_sets: List[set] = []
        mlp_digits: List[str] = []

        for position in range(pin_length):
            start_index = feature_offset + position * 5
            features = np.array(
                [row.iloc[start_index + offset] for offset in range(5)],
                dtype=np.float32,
            )
            normalized = (features - norm["mean"]) / norm["std"]
            with torch.no_grad():
                logits = (
                    model(torch.from_numpy(normalized).unsqueeze(0).float())
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            all_logits.append(logits)
            mlp_digits.append(str(int(np.argmax(logits))))
            if top_k >= 10:
                candidates = set(range(10))
            else:
                candidates = set(np.argsort(logits)[-top_k:][::-1].tolist())
            candidate_sets.append(candidates)

        mlp_prediction = "".join(mlp_digits)
        if mlp_prediction == true_pin:
            correct_mlp += 1

        if not use_kinematic:
            joint_prediction = mlp_prediction
        else:
            dp = [{} for _ in range(pin_length)]
            parent = [{} for _ in range(pin_length)]

            for digit in candidate_sets[0]:
                dp[0][digit] = float(all_logits[0][digit])

            for position in range(1, pin_length):
                delta_t = max(timestamps[position] - timestamps[position - 1], 0.01)
                for target_digit in candidate_sets[position]:
                    best_score = -1e12
                    best_source = -1
                    for source_digit in candidate_sets[position - 1]:
                        if source_digit not in dp[position - 1]:
                            continue
                        level = kinematic_level(source_digit, target_digit)
                        mean = kinematic[level]["mu"]
                        sigma = kinematic[level]["sigma"]
                        transition = log_gaussian_pdf(delta_t, mean, sigma)
                        score = (
                            dp[position - 1][source_digit]
                            + lambda_value * transition
                            + all_logits[position][target_digit]
                        )
                        if score > best_score:
                            best_score = score
                            best_source = source_digit
                    dp[position][target_digit] = best_score
                    parent[position][target_digit] = best_source

            final_candidates = {
                digit: dp[pin_length - 1][digit]
                for digit in candidate_sets[pin_length - 1]
                if digit in dp[pin_length - 1]
            }
            if not final_candidates:
                joint_prediction = mlp_prediction
            else:
                last_digit = max(final_candidates, key=final_candidates.get)
                path = [last_digit]
                for position in range(pin_length - 1, 0, -1):
                    last_digit = parent[position][last_digit]
                    path.append(last_digit)
                joint_prediction = "".join(str(digit) for digit in reversed(path))

        if joint_prediction == true_pin:
            correct_joint += 1

        prediction_rows.append(
            {
                "true_pin": true_pin,
                "mlp_prediction": mlp_prediction,
                "joint_prediction": joint_prediction,
                "joint_correct": str(joint_prediction == true_pin),
            }
        )

    total = len(frame)
    return {
        "dataset": csv_path.name,
        "pin_length": pin_length,
        "num_samples": total,
        "top_k": top_k,
        "lambda": lambda_value,
        "use_kinematic": use_kinematic,
        "mlp_accuracy_pct": round(correct_mlp / total * 100.0, 4),
        "joint_accuracy_pct": round(correct_joint / total * 100.0, 4),
        "predictions": prediction_rows,
    }


def evaluate_hand_size_robustness() -> pd.DataFrame:
    records = []
    for hand_size, posture in [
        ("small", "high_arch"),
        ("small", "low_profile"),
        ("medium", "high_arch"),
        ("medium", "low_profile"),
        ("large", "high_arch"),
        ("large", "low_profile"),
    ]:
        file_path = HAND_SIZE_ROOT / f"test_6digit_{hand_size}_{posture}.csv"
        metrics = evaluate_pin_dataset(file_path, pin_length=6, top_k=3, lambda_value=1.5)
        records.append(
            {
                "hand_size": hand_size,
                "posture": posture,
                "joint_accuracy_pct": metrics["joint_accuracy_pct"],
                "mlp_accuracy_pct": metrics["mlp_accuracy_pct"],
            }
        )
    return pd.DataFrame(records)


def evaluate_battery_background_robustness() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for file_path in sorted(BATTERY_BACKGROUND_ROOT.glob("data_*_*_run*.csv")):
        match = re.match(
            r"data_(?P<battery>\d+pct)_(?P<load>idle|busy)_run(?P<run_id>\d+)\.csv",
            file_path.name,
        )
        if match is None:
            continue
        metrics = evaluate_pin_dataset(file_path, pin_length=6, top_k=3, lambda_value=1.5)
        rows.append(
            {
                "battery": match.group("battery"),
                "load": match.group("load"),
                "run_id": int(match.group("run_id")),
                "joint_accuracy_pct": metrics["joint_accuracy_pct"],
                "mlp_accuracy_pct": metrics["mlp_accuracy_pct"],
            }
        )

    all_runs = pd.DataFrame(rows)
    summary = (
        all_runs.groupby(["battery", "load"], as_index=False)[
            "joint_accuracy_pct"
        ]
        .agg(joint_accuracy_mean_pct="mean", joint_accuracy_std_pct="std")
    )
    mlp_summary = (
        all_runs.groupby(["battery", "load"], as_index=False)[
            "mlp_accuracy_pct"
        ]
        .agg(mlp_accuracy_mean_pct="mean", mlp_accuracy_std_pct="std")
    )
    summary = summary.merge(mlp_summary, on=["battery", "load"])
    return all_runs, summary


def run_feature_extraction(output_dir: Path) -> Dict:
    raw_signal = RAW_SIGNAL_ROOT / "PIN_163589.csv"
    script_path = LEGACY_ROOT / "processing" / "keystroke_segmentation+feature_extraction.py"
    ensure_files_exist([raw_signal, script_path])

    log_dir = ensure_directory(output_dir / "logs")
    log_path = log_dir / "feature_extraction.log"
    command = [sys.executable, str(script_path.relative_to(REPO_ROOT))]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    env.setdefault("XDG_CACHE_HOME", str(MODULE_CACHE_ROOT))
    env.setdefault("MPLCONFIGDIR", str(MODULE_MPL_CACHE))
    env.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise ArtifactError(
            f"Feature extraction failed. See {log_path.relative_to(REPO_ROOT)} for details."
        )

    generated = DERIVED_DATA_ROOT / "extracted_features_163589.csv"
    ensure_files_exist([generated])
    generated_frame = pd.read_csv(generated, header=None)
    if generated_frame.shape != (1, 37):
        raise ArtifactError(
            f"{generated.name} has unexpected shape {generated_frame.shape}; expected (1, 37)."
        )

    destination = output_dir / "tables" / generated.name
    copy_if_exists(generated, destination)
    return {
        "script": script_path.name,
        "output_file": str(destination.relative_to(REPO_ROOT)),
        "num_rows": int(generated_frame.shape[0]),
        "num_columns": int(generated_frame.shape[1]),
    }


def plot_confusion_matrix(metrics: Dict, output_dir: Path) -> Dict[str, str]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0) * 100.0

    figure_dir = ensure_directory(output_dir / "figures")
    png_path = figure_dir / "figure9_confusion_matrix.png"
    pdf_path = figure_dir / "figure9_confusion_matrix.pdf"

    plt.figure(figsize=(8, 4.5))
    sns.heatmap(
        percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=[str(index) for index in range(10)],
        yticklabels=[str(index) for index in range(10)],
        cbar_kws={"label": "Percentage (%)"},
        linewidths=0.3,
        linecolor="black",
    )
    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    return {
        "png": str(png_path.relative_to(REPO_ROOT)),
        "pdf": str(pdf_path.relative_to(REPO_ROOT)),
    }


def plot_ablation(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    figure_dir = ensure_directory(output_dir / "figures")
    png_path = figure_dir / "ablation_6digit.png"
    pdf_path = figure_dir / "ablation_6digit.pdf"

    plot_frame = frame.copy()
    plt.figure(figsize=(7, 4))
    sns.barplot(data=plot_frame, x="setting", y="joint_accuracy_pct", color="#4C78A8")
    plt.ylabel("6-digit PIN accuracy (%)")
    plt.xlabel("Ablation setting")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    return {
        "png": str(png_path.relative_to(REPO_ROOT)),
        "pdf": str(pdf_path.relative_to(REPO_ROOT)),
    }


def plot_hand_size(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    figure_dir = ensure_directory(output_dir / "figures")
    png_path = figure_dir / "hand_size_robustness.png"
    pdf_path = figure_dir / "hand_size_robustness.pdf"

    plt.figure(figsize=(8, 4))
    sns.barplot(data=frame, x="hand_size", y="joint_accuracy_pct", hue="posture")
    plt.ylabel("Joint accuracy (%)")
    plt.xlabel("Hand size")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    return {
        "png": str(png_path.relative_to(REPO_ROOT)),
        "pdf": str(pdf_path.relative_to(REPO_ROOT)),
    }


def plot_battery_background(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    figure_dir = ensure_directory(output_dir / "figures")
    png_path = figure_dir / "battery_background_robustness.png"
    pdf_path = figure_dir / "battery_background_robustness.pdf"

    plot_frame = frame.copy()
    plot_frame["battery"] = pd.Categorical(
        plot_frame["battery"],
        categories=["3pct", "30pct", "70pct", "100pct"],
        ordered=True,
    )
    plot_frame["load"] = pd.Categorical(
        plot_frame["load"], categories=["idle", "busy"], ordered=True
    )
    plt.figure(figsize=(10, 4.5))
    sns.boxplot(data=plot_frame, x="battery", y="joint_accuracy_pct", hue="load")
    plt.ylabel("Joint accuracy (%)")
    plt.xlabel("Battery level")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    return {
        "png": str(png_path.relative_to(REPO_ROOT)),
        "pdf": str(pdf_path.relative_to(REPO_ROOT)),
    }
