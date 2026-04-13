# evaluate_robustness_correct_se.py
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from math import log, pi
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import hand_size_path, legacy_output_path, load_model_bundle

# === Keyboard Layout & Kinematics ===
KEY_POS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 9: (2, 2),
    0: (3, 1)
}

def euclidean_dist(u, v):
    if u == v:
        return 0.0
    x1, y1 = KEY_POS[u]
    x2, y2 = KEY_POS[v]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def kinematic_level(u, v):
    d = euclidean_dist(u, v)
    if d == 0:
        return 0
    elif d <= 1:
        return 1
    elif d <= 1.42:
        return 2
    elif d <= 2:
        return 3
    elif d <= 2.24:
        return 4
    elif d <= 2.83:
        return 5
    elif d <= 3:
        return 6
    else:
        return 7

class MorphologyMLP(nn.Module):
    def __init__(self):
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
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

def log_gaussian_pdf(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - log(sigma) - 0.5 * log(2 * pi)

def evaluate_single_file(csv_path, model, norm_params, kinematic_params,
                         LAMBDA=1.5, TOP_K=3):
    df = pd.read_csv(csv_path, header=0)
    expected_cols = 1 + 6 + 30  # pin + t0~t5 + f0~f5_x5
    assert df.shape[1] == expected_cols, f"Column count error: {df.shape[1]} vs {expected_cols}"

    results = []

    for idx, row in df.iterrows():
        pin_str = str(int(row.iloc[0])).zfill(6)
        timestamps = [float(row.iloc[i]) for i in range(1, 7)]

        all_logits = []
        candidate_sets = []

        for i in range(6):
            start_idx = 7 + i * 5
            feats = np.array([
                row.iloc[start_idx],
                row.iloc[start_idx + 1],
                row.iloc[start_idx + 2],
                row.iloc[start_idx + 3],
                row.iloc[start_idx + 4]
            ], dtype=np.float32)

            feats_norm = (feats - norm_params['mean']) / norm_params['std']
            with torch.no_grad():
                logits = model(torch.from_numpy(feats_norm).unsqueeze(0)).squeeze(0).cpu().numpy()
            all_logits.append(logits)

            if TOP_K == 10:
                candidates = set(range(10))
            else:
                topk_idx = np.argsort(logits)[-TOP_K:][::-1].tolist()
                candidates = set(topk_idx)
            candidate_sets.append(candidates)

        # ========== Joint Inference (Force Enable) ==========
        dp = [{} for _ in range(6)]
        parent = [{} for _ in range(6)]

        for d in candidate_sets[0]:
            dp[0][d] = float(all_logits[0][d])

        for pos in range(1, 6):
            dt = max(timestamps[pos] - timestamps[pos - 1], 0.01)
            for v in candidate_sets[pos]:
                best_score = -1e12
                best_u = -1
                for u in candidate_sets[pos - 1]:
                    if u not in dp[pos - 1]:
                        continue
                    k_level = kinematic_level(u, v)
                    mu = kinematic_params[k_level]['mu']
                    sigma = max(kinematic_params[k_level]['sigma'], 1e-4)
                    log_trans = log_gaussian_pdf(dt, mu, sigma)
                    score = dp[pos - 1][u] + LAMBDA * log_trans + all_logits[pos][v]
                    if score > best_score:
                        best_score = score
                        best_u = u
                dp[pos][v] = best_score
                parent[pos][v] = best_u

        final_candidates = {d: dp[5][d] for d in candidate_sets[5] if d in dp[5]}
        if final_candidates:
            last = max(final_candidates, key=final_candidates.get)
            path = [last]
            for pos in range(5, 0, -1):
                last = parent[pos][last]
                path.append(last)
            pred_pin = ''.join(str(d) for d in reversed(path))
        else:
            pred_digits = [str(int(np.argmax(logits))) for logits in all_logits]
            pred_pin = ''.join(pred_digits)

        is_correct = (pred_pin == pin_str)
        results.append(is_correct)

    return results  # Return a column of True/False


def main():
    # ====== Configuration Section ======
    LAMBDA = 1.5
    TOP_K = 3
    # ===================

    model, norm_params, kinematic_params = load_model_bundle()

    configs = [
        ('small', 'high_arch'),
        ('small', 'low_profile'),
        ('medium', 'high_arch'),
        ('medium', 'low_profile'),
        ('large', 'high_arch'),
        ('large', 'low_profile')
    ]

    all_records = []

    print("🚀 Starting batch evaluation of 6 robustness test files...\n")

    for hand, posture in configs:
        filename = hand_size_path(f"test_6digit_{hand}_{posture}.csv")
        if not filename.exists():
            print(f"⚠️ Skip {filename} (file does not exist)")
            continue

        try:
            is_correct_list = evaluate_single_file(
                filename, model, norm_params, kinematic_params,
                LAMBDA=LAMBDA, TOP_K=TOP_K
            )
            n = len(is_correct_list)
            acc_mean = np.mean(is_correct_list)
            # Theoretical standard error SE = sqrt(p(1-p)/n)
            se = np.sqrt(acc_mean * (1 - acc_mean) / n) if n > 0 else 0.0

            print(f"✅ {hand:>6} + {posture:<12} → Acc: {acc_mean:.2%}, SE: {se:.2%}")

            all_records.append({
                'Hand Size': hand,
                'Posture': posture,
                'JointAccMean': acc_mean * 100,   # Convert to percentage
                'JointAccSE': se * 100            # Convert to percentage
            })

        except Exception as e:
            print(f"❌ Evaluation failed {filename}: {e}")
            continue

    # Convert to DataFrame and save
    df_summary = pd.DataFrame(all_records)
    df_summary = df_summary.sort_values(['Hand Size', 'Posture']).reset_index(drop=True)

    print("\n" + "="*60)
    print("📊 Joint model evaluation results (correct SE calculation)")
    print(df_summary.to_string(index=False, float_format="%.2f"))
    print("="*60)

    output_path = legacy_output_path("robustness", "robustness_joint_summary_correct_se.csv")
    df_summary.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to '{output_path}'")


if __name__ == '__main__':
    main()
