#!/usr/bin/env python3
"""
Automated experiment runner for keystroke inference research.
This script runs the complete experimental pipeline as described in the paper.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json
import re
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import checkpoint_path, legacy_output_path, pin_reconstruction_path, raw_signal_path

def extract_results_from_output(stdout, stderr, results_dict):
    """Extract numerical results from command output."""
    combined_output = stdout + stderr

    # Extract accuracy percentages
    accuracy_patterns = [
        r'accuracy[:\s]+([\d.]+)%',
        r'准确率[:\s]+([\d.]+)%',
        r'Top-1.*?:\s*([\d.]+)%',
        r'Top-3.*?:\s*([\d.]+)%',
        r'([\d.]+)%'
    ]

    for pattern in accuracy_patterns:
        matches = re.findall(pattern, combined_output, re.IGNORECASE)
        if matches:
            results_dict['accuracy'] = [float(m) for m in matches]
            break

    # Extract success rates
    success_patterns = [
        r'success rate[:\s]+([\d.]+)%',
        r'成功率[:\s]+([\d.]+)%'
    ]

    for pattern in success_patterns:
        matches = re.findall(pattern, combined_output, re.IGNORECASE)
        if matches:
            results_dict['success_rate'] = [float(m) for m in matches]
            break

def run_command(cmd, description, results_dict=None, cwd=None):
    """Run a command and print status."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd or REPO_ROOT),
        )
        print(f"✅ {description} completed successfully")

        # Extract results if available
        if results_dict is not None:
            extract_results_from_output(result.stdout, result.stderr, results_dict)

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")

        if results_dict is not None:
            results_dict['status'] = 'failed'
            results_dict['error'] = str(e)

        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_modules = [
        'numpy', 'pandas', 'torch', 'scipy', 'matplotlib', 'sklearn', 'tqdm'
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print(f"❌ Missing required dependencies: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False

    print("✅ All dependencies are available")
    return True

def check_data_files():
    """Check if required data and model files exist."""
    required_files = [
        checkpoint_path('keystroke_morphology_mlp.pth'),
        checkpoint_path('norm_params.pth'),
        checkpoint_path('kinematic_params.pth'),
        pin_reconstruction_path('6digit_PINs.csv'),
    ]

    missing = []
    for file in required_files:
        if not file.exists():
            missing.append(str(file.relative_to(REPO_ROOT)))

    if missing:
        print(f"⚠️  Missing data/model files: {', '.join(missing)}")
        print("Some experiments may not run. Please ensure all files are present.")
        return False

    print("✅ All required data and model files are present")
    return True

def run_feature_extraction():
    """Run feature extraction from raw signals."""
    raw_signal = raw_signal_path('PIN_163589.csv')
    script_path = REPO_ROOT / 'legacy' / 'processing' / 'keystroke_segmentation+feature_extraction.py'
    if not raw_signal.exists():
        print(f"⚠️  {raw_signal} not found, skipping feature extraction")
        return True

    return run_command(
        [sys.executable, str(script_path)],
        "Extracting features from raw signals",
        cwd=REPO_ROOT,
    )

def run_inference():
    """Run PIN inference experiments."""
    results = {}
    script_path = REPO_ROOT / 'legacy' / 'inference' / 'infer_PINs.py'
    success = run_command(
        [sys.executable, str(script_path)],
        "Running PIN inference experiments",
        results,
        cwd=REPO_ROOT,
    )

    if success and ('accuracy' in results or 'success_rate' in results):
        print(f"📊 PIN Inference Results:")
        if 'accuracy' in results:
            print(f"   - Overall Accuracy: {results['accuracy'][-1]:.2f}%")
        if 'success_rate' in results:
            print(f"   - Success Rate: {results['success_rate'][-1]:.2f}%")

    return success, results

def run_robustness_tests():
    """Run robustness evaluation tests."""
    test_files = [
        REPO_ROOT / 'legacy' / 'robustness' / 'Impact_of_Hand_Size.py',
        REPO_ROOT / 'legacy' / 'robustness' / 'Impact_of_Battery_and_BackgroundApps.py',
    ]

    success = True
    results = {}
    for test_file in test_files:
        if test_file.exists():
            test_results = {}
            test_success = run_command(
                [sys.executable, str(test_file)],
                f"Running {test_file.name}",
                test_results,
                cwd=REPO_ROOT,
            )
            success &= test_success
            results[test_file.name] = test_results
        else:
            print(f"⚠️  {test_file} not found, skipping")

    return success, results

def run_different_attempts():
    """Run different attack attempt analyses."""
    test_files = [
        REPO_ROOT / 'legacy' / 'inference' / 'Different_Attempts_6digit.py',
        REPO_ROOT / 'legacy' / 'inference' / 'Different_Attempts_4digit.py',
        REPO_ROOT / 'legacy' / 'inference' / 'Different_Attempts_8digit.py',
    ]

    success = True
    results = {}
    for test_file in test_files:
        if test_file.exists():
            test_results = {}
            test_success = run_command(
                [sys.executable, str(test_file)],
                f"Running {test_file.name}",
                test_results,
                cwd=REPO_ROOT,
            )
            success &= test_success
            results[test_file.name] = test_results
        else:
            print(f"⚠️  {test_file} not found, skipping")

    return success, results

def generate_plots():
    """Generate visualization plots."""
    plot_files = [
        REPO_ROOT / 'legacy' / 'visualization' / 'Confusion_Matrix.py'
    ]

    success = True
    for plot_file in plot_files:
        if plot_file.exists():
            success &= run_command(
                [sys.executable, str(plot_file)],
                f"Generating plots with {plot_file.name}",
                cwd=REPO_ROOT,
            )
        else:
            print(f"⚠️  {plot_file} not found, skipping")

    return success

def main():
    parser = argparse.ArgumentParser(description='Run keystroke inference experiments')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency and file checks')
    parser.add_argument('--quick', action='store_true',
                       help='Run only core experiments (skip robustness tests)')
    parser.add_argument('--plots-only', action='store_true',
                       help='Generate plots only')

    args = parser.parse_args()

    # Initialize results tracking
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': {},
        'summary': {}
    }

    print("🚀 Starting keystroke inference experiments")
    print("=" * 50)

    # Check environment
    if not args.skip_checks:
        if not check_dependencies():
            return 1
        if not check_data_files():
            print("⚠️  Continuing with available files...")

    # Run experiments based on arguments
    if args.plots_only:
        results['experiments']['plots'] = {'status': 'running'}
        success = generate_plots()
        results['experiments']['plots']['status'] = 'completed' if success else 'failed'
    else:
        success = True

        # Core pipeline
        results['experiments']['feature_extraction'] = {'status': 'running'}
        if run_feature_extraction():
            results['experiments']['feature_extraction']['status'] = 'completed'
        else:
            results['experiments']['feature_extraction']['status'] = 'failed'
            success = False

        results['experiments']['inference'] = {'status': 'running'}
        inference_success, inference_results = run_inference()
        if inference_success:
            results['experiments']['inference']['status'] = 'completed'
            # Merge inference results
            results['experiments']['inference'].update(inference_results)
        else:
            results['experiments']['inference']['status'] = 'failed'
            success = False

        if not args.quick:
            # Extended tests
            results['experiments']['robustness'] = {'status': 'running'}
            robustness_success, robustness_results = run_robustness_tests()
            if robustness_success:
                results['experiments']['robustness']['status'] = 'completed'
                results['experiments']['robustness'].update(robustness_results)
            else:
                results['experiments']['robustness']['status'] = 'failed'
                success = False

            results['experiments']['attempts'] = {'status': 'running'}
            attempts_success, attempts_results = run_different_attempts()
            if attempts_success:
                results['experiments']['attempts']['status'] = 'completed'
                results['experiments']['attempts'].update(attempts_results)
            else:
                results['experiments']['attempts']['status'] = 'failed'
                success = False

        # Generate visualizations
        results['experiments']['plots'] = {'status': 'running'}
        if generate_plots():
            results['experiments']['plots']['status'] = 'completed'
        else:
            results['experiments']['plots']['status'] = 'failed'
            success = False

    # Save results to file
    results['summary'] = {
        'overall_success': success,
        'completed_experiments': sum(1 for exp in results['experiments'].values()
                                   if exp.get('status') == 'completed'),
        'total_experiments': len(results['experiments']),
        'has_accuracy_data': any('accuracy' in exp for exp in results['experiments'].values()
                                if isinstance(exp, dict))
    }

    # Save detailed results
    output_path = legacy_output_path('orchestration', 'experiment_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 50)
    if success:
        print("🎉 All experiments completed successfully!")
        print(f"\n📊 Results saved to: {output_path}")
        print("\n📈 Performance Summary:")

        # Display key metrics
        for exp_name, exp_data in results['experiments'].items():
            if isinstance(exp_data, dict) and 'accuracy' in exp_data:
                acc = exp_data['accuracy']
                print(f"- {exp_name.title()}: {acc}% accuracy")
            elif isinstance(exp_data, dict) and 'success_rate' in exp_data:
                sr = exp_data['success_rate']
                print(f"- {exp_name.title()}: {sr}% success rate")

        print("\n📁 Check output files and plots in the current directory.")
    else:
        print("❌ Some experiments failed. Check the output above for details.")
        print(f"📊 Partial results saved to: {output_path}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
