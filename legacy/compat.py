from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keystroke_artifact.common import load_model_bundle  # noqa: E402

DATA_ROOT = REPO_ROOT / "data"
CLASSIFICATION_ROOT = DATA_ROOT / "classification"
PIN_ROOT = DATA_ROOT / "pin_reconstruction"
RAW_SIGNAL_ROOT = DATA_ROOT / "raw_signals"
DERIVED_ROOT = DATA_ROOT / "derived"
HAND_SIZE_ROOT = DATA_ROOT / "robustness" / "hand_size"
BATTERY_BACKGROUND_ROOT = DATA_ROOT / "robustness" / "battery_background"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "models"
LEGACY_OUTPUT_ROOT = REPO_ROOT / "outputs" / "legacy"


def classification_path(name: str) -> Path:
    return CLASSIFICATION_ROOT / name


def pin_reconstruction_path(name: str) -> Path:
    return PIN_ROOT / name


def raw_signal_path(name: str) -> Path:
    return RAW_SIGNAL_ROOT / name


def derived_path(name: str) -> Path:
    return DERIVED_ROOT / name


def hand_size_path(name: str) -> Path:
    return HAND_SIZE_ROOT / name


def checkpoint_path(name: str) -> Path:
    return CHECKPOINT_ROOT / name


def battery_background_files(pattern: str = "data_*_*_run*.csv") -> list[Path]:
    return sorted(BATTERY_BACKGROUND_ROOT.glob(pattern))


def ensure_output_dir(*parts: str) -> Path:
    path = LEGACY_OUTPUT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def legacy_output_path(*parts: str) -> Path:
    if len(parts) == 1:
        ensure_output_dir()
    else:
        ensure_output_dir(*parts[:-1])
    return LEGACY_OUTPUT_ROOT.joinpath(*parts)
