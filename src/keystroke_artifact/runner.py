from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .common import (
    OUTPUT_ROOT,
    PIN_DATASETS,
    ArtifactError,
    CHECKPOINT_ROOT,
    CLASSIFICATION_DATA_ROOT,
    dependency_versions,
    ensure_directory,
    ensure_files_exist,
    evaluate_battery_background_robustness,
    evaluate_hand_size_robustness,
    evaluate_pin_dataset,
    evaluate_single_digit_classifier,
    plot_ablation,
    plot_battery_background,
    plot_confusion_matrix,
    plot_hand_size,
    run_feature_extraction,
    save_json,
    save_table,
    write_text,
)


def default_output_dir(name: str) -> Path:
    return OUTPUT_ROOT / name


def run_env_check(output_dir: Path) -> dict:
    ensure_directory(output_dir)
    ensure_files_exist(
        [
            CHECKPOINT_ROOT / "keystroke_morphology_mlp.pth",
            CHECKPOINT_ROOT / "norm_params.pth",
            CHECKPOINT_ROOT / "kinematic_params.pth",
            CLASSIFICATION_DATA_ROOT / "test_data.csv",
            PIN_DATASETS[6],
        ]
    )
    payload = {"dependency_versions": dependency_versions(), "status": "ok"}
    save_json(output_dir / "env_check.json", payload)
    return payload


def command_quick_test(output_dir: Path) -> dict:
    ensure_directory(output_dir)
    feature_metrics = run_feature_extraction(output_dir)
    single_digit_metrics = evaluate_single_digit_classifier()
    figure_paths = plot_confusion_matrix(single_digit_metrics, output_dir)
    six_digit_metrics = evaluate_pin_dataset(PIN_DATASETS[6], pin_length=6, top_k=3, lambda_value=1.5)

    quick_summary = {
        "feature_extraction": feature_metrics,
        "single_digit_classifier": {
            "top1_accuracy_pct": single_digit_metrics["top1_accuracy_pct"],
            "top3_accuracy_pct": single_digit_metrics["top3_accuracy_pct"],
        },
        "six_digit_joint_inference": {
            "mlp_accuracy_pct": six_digit_metrics["mlp_accuracy_pct"],
            "joint_accuracy_pct": six_digit_metrics["joint_accuracy_pct"],
            "top_k": six_digit_metrics["top_k"],
            "lambda": six_digit_metrics["lambda"],
        },
        "generated_figures": figure_paths,
    }

    save_json(output_dir / "quick_test_summary.json", quick_summary)
    save_table(
        output_dir / "tables" / "quick_test_6digit_predictions.csv",
        pd.DataFrame(six_digit_metrics["predictions"]),
    )
    write_text(
        output_dir / "SUCCESS.txt",
        (
            "Quick test completed successfully.\n"
            f"- Single-digit Top-1 accuracy: {single_digit_metrics['top1_accuracy_pct']:.2f}%\n"
            f"- Single-digit Top-3 accuracy: {single_digit_metrics['top3_accuracy_pct']:.2f}%\n"
            f"- 6-digit joint accuracy: {six_digit_metrics['joint_accuracy_pct']:.2f}%\n"
        ),
    )
    return quick_summary


def command_table1(output_dir: Path) -> dict:
    ensure_directory(output_dir)
    metrics = evaluate_single_digit_classifier()
    recall_frame = pd.DataFrame(
        [
            {"digit": int(digit), "recall_pct": value}
            for digit, value in metrics["per_class_recall_pct"].items()
        ]
    )
    summary_frame = pd.DataFrame(
        [
            {
                "dataset": metrics["dataset"],
                "num_samples": metrics["num_samples"],
                "top1_accuracy_pct": metrics["top1_accuracy_pct"],
                "top3_accuracy_pct": metrics["top3_accuracy_pct"],
            }
        ]
    )
    save_table(output_dir / "tables" / "table1_single_digit_metrics.csv", summary_frame)
    save_table(output_dir / "tables" / "table1_per_digit_recall.csv", recall_frame)
    save_json(output_dir / "table1_single_digit_metrics.json", metrics)
    return metrics


def command_table2(output_dir: Path) -> dict:
    ensure_directory(output_dir)
    rows = []
    for pin_length, dataset_path in PIN_DATASETS.items():
        metrics = evaluate_pin_dataset(dataset_path, pin_length=pin_length, top_k=3, lambda_value=1.5)
        rows.append(
            {
                "pin_length": pin_length,
                "dataset": dataset_path.name,
                "num_samples": metrics["num_samples"],
                "mlp_accuracy_pct": metrics["mlp_accuracy_pct"],
                "joint_accuracy_pct": metrics["joint_accuracy_pct"],
                "top_k": metrics["top_k"],
                "lambda": metrics["lambda"],
            }
        )
        save_table(
            output_dir / "tables" / f"table2_predictions_{pin_length}digit.csv",
            pd.DataFrame(metrics["predictions"]),
        )
    frame = pd.DataFrame(rows).sort_values("pin_length").reset_index(drop=True)
    save_table(output_dir / "tables" / "table2_pin_reconstruction.csv", frame)
    payload = {"rows": frame.to_dict(orient="records")}
    save_json(output_dir / "table2_pin_reconstruction.json", payload)
    return payload


def command_figure1(output_dir: Path) -> dict:
    ensure_directory(output_dir)
    metrics = evaluate_single_digit_classifier()
    figure_paths = plot_confusion_matrix(metrics, output_dir)
    payload = {
        "source_dataset": metrics["dataset"],
        "top1_accuracy_pct": metrics["top1_accuracy_pct"],
        "figure_paths": figure_paths,
    }
    save_json(output_dir / "figure1_confusion_matrix.json", payload)
    return payload


def command_ablation(output_dir: Path) -> dict:
    ensure_directory(output_dir)
    settings = [
        ("mlp_only", False, 3),
        ("joint_top3", True, 3),
        ("joint_top10", True, 10),
    ]
    rows = []
    for setting_name, use_kinematic, top_k in settings:
        metrics = evaluate_pin_dataset(
            PIN_DATASETS[6],
            pin_length=6,
            top_k=top_k,
            lambda_value=1.5,
            use_kinematic=use_kinematic,
        )
        rows.append(
            {
                "setting": setting_name,
                "use_kinematic": use_kinematic,
                "top_k": top_k,
                "mlp_accuracy_pct": metrics["mlp_accuracy_pct"],
                "joint_accuracy_pct": metrics["joint_accuracy_pct"],
            }
        )
    frame = pd.DataFrame(rows)
    figure_paths = plot_ablation(frame, output_dir)
    save_table(output_dir / "tables" / "ablation_6digit.csv", frame)
    payload = {"rows": frame.to_dict(orient="records"), "figure_paths": figure_paths}
    save_json(output_dir / "ablation_6digit.json", payload)
    return payload


def command_robustness(output_dir: Path) -> dict:
    ensure_directory(output_dir)
    hand_size = evaluate_hand_size_robustness()
    all_runs, summary = evaluate_battery_background_robustness()
    hand_size_figures = plot_hand_size(hand_size, output_dir)
    battery_figures = plot_battery_background(all_runs, output_dir)
    save_table(output_dir / "tables" / "robustness_hand_size.csv", hand_size)
    save_table(output_dir / "tables" / "robustness_battery_background_all_runs.csv", all_runs)
    save_table(output_dir / "tables" / "robustness_battery_background_summary.csv", summary)
    payload = {
        "hand_size_robustness_rows": hand_size.to_dict(orient="records"),
        "battery_background_summary_rows": summary.to_dict(orient="records"),
        "figure_paths": {
            "hand_size": hand_size_figures,
            "battery_background": battery_figures,
        },
    }
    save_json(output_dir / "robustness_summary.json", payload)
    return payload


def command_main_results(output_dir: Path) -> dict:
    ensure_directory(output_dir)
    payload = {
        "table1": command_table1(output_dir / "table1"),
        "table2": command_table2(output_dir / "table2"),
        "figure1": command_figure1(output_dir / "figure1"),
        "ablation": command_ablation(output_dir / "ablation"),
        "robustness": command_robustness(output_dir / "robustness"),
    }
    save_json(output_dir / "main_results_manifest.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Artifact-evaluation runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in [
        "env-check",
        "quick-test",
        "table1",
        "table2",
        "figure1",
        "ablation",
        "robustness",
        "main-results",
    ]:
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--output-dir", type=Path, default=default_output_dir(name.replace("-", "_")))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "env-check":
            payload = run_env_check(args.output_dir)
        elif args.command == "quick-test":
            payload = command_quick_test(args.output_dir)
        elif args.command == "table1":
            payload = command_table1(args.output_dir)
        elif args.command == "table2":
            payload = command_table2(args.output_dir)
        elif args.command == "figure1":
            payload = command_figure1(args.output_dir)
        elif args.command == "ablation":
            payload = command_ablation(args.output_dir)
        elif args.command == "robustness":
            payload = command_robustness(args.output_dir)
        elif args.command == "main-results":
            payload = command_main_results(args.output_dir)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except ArtifactError as exc:
        print(f"[artifact-error] {exc}", file=sys.stderr)
        return 1

    print(f"[artifact-ok] Completed '{args.command}'. Outputs saved under {args.output_dir}")
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                continue
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
