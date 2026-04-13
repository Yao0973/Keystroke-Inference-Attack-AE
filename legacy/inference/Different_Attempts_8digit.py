# evaluate_8digit_topn.py
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from math import log, pi
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import legacy_output_path, load_model_bundle, pin_reconstruction_path

# === Keyboard Layout ===
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
    TOP_K_PER_STEP = 5      # MLP candidates per step（for pruning）
    BEAM_WIDTH = 10         # Beam Search width（must ≥ MAX_N）
    MAX_N = 5               # Calculate Top-1 to Top-5 success rate
    DATA_FILE = 'full_test_data_8digit.csv'
    PIN_LENGTH = 8
    # ===================

    assert BEAM_WIDTH >= MAX_N, f"❌ BEAM_WIDTH ({BEAM_WIDTH}) must ≥ MAX_N ({MAX_N})"

    data_path = pin_reconstruction_path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Missing file: {data_path}")

    model, norm, kinematic = load_model_bundle()
    df = pd.read_csv(data_path, header=0)
    print(f"📊 Successfully loaded {len(df)} test samples")

    expected_cols = 1 + PIN_LENGTH + PIN_LENGTH * 5
    if df.shape[1] != expected_cols:
        raise ValueError(f"❌ CSV Column count error：expected.*actual {df.shape[1]}")

    total = len(df)
    correct_mlp = 0
    hits_at_n = [0] * MAX_N

    for idx, row in df.iterrows():
        pin_str = str(int(row.iloc[0])).zfill(PIN_LENGTH)
        ground_truth = tuple(int(d) for d in pin_str)
        timestamps = [float(row.iloc[i]) for i in range(1, 1 + PIN_LENGTH)]

        all_logits = []
        mlp_pred_digits = []
        candidate_sets = []

        # Step 1: Get logits and Top-K candidates per step
        for i in range(PIN_LENGTH):
            start_idx = 1 + PIN_LENGTH + i * 5
            feats = np.array([
                row.iloc[start_idx],
                row.iloc[start_idx + 1],
                row.iloc[start_idx + 2],
                row.iloc[start_idx + 3],
                row.iloc[start_idx + 4]
            ], dtype=np.float32)
            feats_norm = (feats - norm['mean']) / norm['std']
            with torch.no_grad():
                logits = model(torch.from_numpy(feats_norm).unsqueeze(0)).squeeze(0).cpu().numpy()
            all_logits.append(logits)

            digit = int(np.argmax(logits))
            mlp_pred_digits.append(str(digit))

            topk_idx = np.argsort(logits)[-TOP_K_PER_STEP:][::-1].tolist()
            candidate_sets.append(topk_idx)

        # MLP-only accuracy
        mlp_pred_pin = ''.join(mlp_pred_digits)
        if mlp_pred_pin == pin_str:
            correct_mlp += 1

        # Step 2: Beam Search generates Top-BEAM_WIDTH sequences
        if not USE_KINEMATIC:
            # no kinematic：use Cartesian product directly（but.*too large.*fallback to MLP top-k product）
            from itertools import product
            candidate_sequences = list(product(*candidate_sets))
            scored_seqs = [(sum(all_logits[i][d] for i, d in enumerate(seq)), seq) for seq in candidate_sequences]
            scored_seqs.sort(key=lambda x: x[0], reverse=True)
            top_beam = [seq for _, seq in scored_seqs[:BEAM_WIDTH]]
        else:
            # initialize beam: list of (score, path_as_tuple)
            beam = [(float(all_logits[0][d]), (d,)) for d in candidate_sets[0]]
            beam.sort(key=lambda x: x[0], reverse=True)
            beam = beam[:BEAM_WIDTH]

            # extend bit by bit
            for pos in range(1, PIN_LENGTH):
                dt = max(timestamps[pos] - timestamps[pos - 1], 0.01)
                new_candidates = []
                for score, path in beam:
                    last_digit = path[-1]
                    for next_d in candidate_sets[pos]:
                        k_level = kinematic_level(last_digit, next_d)
                        mu = kinematic[k_level]['mu']
                        sigma = max(kinematic[k_level]['sigma'], 1e-4)
                        log_trans = log_gaussian_pdf(dt, mu, sigma)
                        new_score = score + LAMBDA * log_trans + all_logits[pos][next_d]
                        new_path = path + (next_d,)
                        new_candidates.append((new_score, new_path))
                # keep Top-BEAM_WIDTH
                new_candidates.sort(key=lambda x: x[0], reverse=True)
                beam = new_candidates[:BEAM_WIDTH]

            top_beam = [path for _, path in beam]

        # Step 3: Take Top-N and check hits
        top_n_seqs = top_beam[:MAX_N]
        for n in range(MAX_N):
            if ground_truth in top_n_seqs[:n + 1]:
                hits_at_n[n] += 1

    # Output results
    acc_mlp = correct_mlp / total
    success_rates = [hits / total * 100 for hits in hits_at_n]

    print("\n" + "=" * 60)
    print("🔐 8-digit PIN Top-N attack success rate")
    print("🔍 MLP-only Accuracy (Top-1): {:.2%} ({}/{})".format(acc_mlp, correct_mlp, total))
    for n in range(MAX_N):
        print("🎯 Top-{} Success Rate: {:.2f}%".format(n + 1, success_rates[n]))
    print("=" * 60)

    # Save results for plotting
    output_path = legacy_output_path('inference', 'topn_results_8digit.npz')
    np.savez(output_path,
             attempts=np.arange(1, MAX_N + 1),
             success_rates=success_rates)
    print(f"💾 Saved Top-N results to {output_path}")

if __name__ == '__main__':
    main()
