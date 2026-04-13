# evaluate_8digit.py
# script for evaluating.*inference
# generated.*compatible）

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

from legacy.compat import load_model_bundle, pin_reconstruction_path

# === Keyboard Layout（consistent with training）===
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

def main():
    # ====== Configuration Section ======
    USE_KINEMATIC = True
    LAMBDA = 1.5
    TOP_K = 3
    DATA_FILE = 'full_test_data_8digit.csv'
    PIN_LENGTH = 8
    # ===================

    data_path = pin_reconstruction_path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(f"❌ missing file: {data_path}")

    model, norm, kinematic = load_model_bundle()

    # load data
    df = pd.read_csv(data_path, header=0)
    print(f"📊 Successfully loaded {len(df)} test samples")

    # verify column count: 1(pin) + 8(ts) + 8*5(features) = 49
    expected_cols = 1 + PIN_LENGTH + PIN_LENGTH * 5
    if df.shape[1] != expected_cols:
        raise ValueError(f"❌ CSV column count error：expected.*actual {df.shape[1]}")

    correct_joint = 0
    correct_mlp = 0
    total = len(df)

    for idx, row in df.iterrows():
        pin_str = str(int(row.iloc[0])).zfill(PIN_LENGTH)
        timestamps = [float(row.iloc[i]) for i in range(1, 1 + PIN_LENGTH)]

        all_logits = []
        mlp_pred_digits = []
        candidate_sets = []

        for i in range(PIN_LENGTH):
            start_idx = 1 + PIN_LENGTH + i * 5
            feats = np.array([
                row.iloc[start_idx],
                row.iloc[start_idx + 1],
                row.iloc[start_idx + 2],
                row.iloc[start_idx + 3],
                row.iloc[start_idx + 4]
            ], dtype=np.float32)

            # ✅ normalize original physical features
            feats_norm = (feats - norm['mean']) / norm['std']

            with torch.no_grad():
                logits = model(torch.from_numpy(feats_norm).unsqueeze(0)).squeeze(0).cpu().numpy()
            all_logits.append(logits)

            digit = int(np.argmax(logits))
            mlp_pred_digits.append(str(digit))

            if TOP_K == 10:
                candidates = set(range(10))
            else:
                topk_idx = np.argsort(logits)[-TOP_K:][::-1].tolist()
                candidates = set(topk_idx)
            candidate_sets.append(candidates)

        mlp_pred_pin = ''.join(mlp_pred_digits)
        if mlp_pred_pin == pin_str:
            correct_mlp += 1

        # ========== joint inference.*dynamic programming）==========
        if not USE_KINEMATIC:
            pred_pin = mlp_pred_pin
        else:
            dp = [{} for _ in range(PIN_LENGTH)]
            parent = [{} for _ in range(PIN_LENGTH)]

            # initialize position
            for d in candidate_sets[0]:
                dp[0][d] = float(all_logits[0][d])

            # dynamic programming recursion
            for pos in range(1, PIN_LENGTH):
                dt = max(timestamps[pos] - timestamps[pos - 1], 0.01)  # prevent.*too small
                for v in candidate_sets[pos]:
                    best_score = -1e12
                    best_u = -1
                    for u in candidate_sets[pos - 1]:
                        if u not in dp[pos - 1]:
                            continue
                        k_level = kinematic_level(u, v)
                        mu = kinematic[k_level]['mu']
                        sigma = max(kinematic[k_level]['sigma'], 1e-4)
                        log_trans = log_gaussian_pdf(dt, mu, sigma)
                        score = dp[pos - 1][u] + LAMBDA * log_trans + all_logits[pos][v]
                        if score > best_score:
                            best_score = score
                            best_u = u
                    dp[pos][v] = best_score
                    parent[pos][v] = best_u

            # backtrack optimal path
            final_candidates = {d: dp[PIN_LENGTH-1][d] for d in candidate_sets[PIN_LENGTH-1] if d in dp[PIN_LENGTH-1]}
            if not final_candidates:
                pred_pin = mlp_pred_pin  # fallback
            else:
                last = max(final_candidates, key=final_candidates.get)
                path = [last]
                for pos in range(PIN_LENGTH-1, 0, -1):
                    last = parent[pos][last]
                    path.append(last)
                pred_pin = ''.join(str(d) for d in reversed(path))

        if pred_pin == pin_str:
            correct_joint += 1

    # Output Results
    acc_joint = correct_joint / total
    acc_mlp = correct_mlp / total

    print("\n" + "=" * 60)
    print("🔐 8-digit PIN inference results")
    print("🔍 MLP-only Accuracy (Top-1):       {:.2%} ({}/{})".format(acc_mlp, correct_mlp, total))
    if USE_KINEMATIC:
        mode_desc = "Full Viterbi" if TOP_K == 10 else f"Top-{TOP_K}"
        print("🎯 Joint Model Accuracy:          {:.2%} ({}/{})".format(acc_joint, correct_joint, total))
        print(f"   (λ={LAMBDA}, Mode={mode_desc})")
    else:
        print("⚠️  Joint Model = MLP-only")
    print("=" * 60)

if __name__ == '__main__':
    main()
