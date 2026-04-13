# sensitivity_analysis.py
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import load_model_bundle, pin_reconstruction_path

# === Core modules copied from evaluate_6digit.py ===

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
    from math import log, pi
    return -0.5 * ((x - mu) / sigma) ** 2 - log(sigma) - 0.5 * log(2 * pi)

# === Constants needed for spatial prior (inferred from your previous code) ===
RANK_ORDER = [1, 4, 7, 2, 5, 8, 3, 6, 9, 0]
RANK_MAP = {k: i for i, k in enumerate(RANK_ORDER)}

# === Sensitivity Analysis Main Functions ===

def load_model_and_data():
    data_path = pin_reconstruction_path('full_test_data_6digit.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Required file missing: {data_path}")

    model, norm_params, kinematic_params_orig = load_model_bundle()
    df = pd.read_csv(data_path, header=0)
    print(f"📊 Successfully loaded {len(df)} test samples")
    return model, norm_params, kinematic_params_orig, df

def evaluate_with_sigma_scale(df, model, norm_params, kinematic_params_orig, sigma_scale,
                              use_spatial=True, lambda_val=1.5, top_k=3):
    # Build scaled kinematic_params
    kinematic_params_scaled = {}
    for level in range(8):
        mu = kinematic_params_orig[level]['mu']
        sigma = kinematic_params_orig[level]['sigma'] * sigma_scale
        kinematic_params_scaled[level] = {'mu': mu, 'sigma': sigma}

    total = len(df)
    correct = 0
    SPATIAL_PENALTY = -0.1
    NOISE_TOLERANCE = 0.3
    
    for idx, row in df.iterrows():
        pin_str = str(int(row.iloc[0])).zfill(6)
        timestamps = [float(row.iloc[i]) for i in range(1, 7)]
        
        all_logits = []
        obs_amplitudes = []
        candidate_sets = []
        
        for i in range(6):
            start_idx = 7 + i * 5
            raw_feats = np.array([row.iloc[start_idx + j] for j in range(5)], dtype=np.float32)
            obs_amplitudes.append(raw_feats[0])
            
            feats_norm = (raw_feats - norm_params['mean']) / norm_params['std']
            with torch.no_grad():
                logits = model(torch.from_numpy(feats_norm).unsqueeze(0)).squeeze(0).cpu().numpy()
            all_logits.append(logits)
            
            # Top-K candidate set
            if top_k == 10:
                candidates = set(range(10))
            else:
                topk_idx = np.argsort(logits)[-top_k:][::-1].tolist()
                candidates = set(topk_idx)
            candidate_sets.append(candidates)

        # Viterbi decoding
        dp = [{} for _ in range(6)]
        parent = [{} for _ in range(6)]
        
        for d in candidate_sets[0]:
            dp[0][d] = float(all_logits[0][d])
        
        for pos in range(1, 6):
            dt = max(timestamps[pos] - timestamps[pos-1], 0.01)
            for v in candidate_sets[pos]:
                best_score = -1e12
                best_u = -1
                for u in candidate_sets[pos-1]:
                    if u not in dp[pos-1]: 
                        continue
                    score = dp[pos-1][u] + all_logits[pos][v]
                    
                    # Spatial constraint
                    if use_spatial:
                        rank_u = RANK_MAP.get(u, 9)
                        rank_v = RANK_MAP.get(v, 9)
                        delta_I = obs_amplitudes[pos-1] - obs_amplitudes[pos]
                        expected_sign = np.sign(rank_v - rank_u)
                        actual_sign = np.sign(delta_I)
                        if (actual_sign != expected_sign) and (abs(delta_I) > NOISE_TOLERANCE):
                            score += SPATIAL_PENALTY

                    # Temporal constraint (using scaled sigma)
                    k_level = kinematic_level(u, v)
                    mu = kinematic_params_scaled[k_level]['mu']
                    sigma = max(kinematic_params_scaled[k_level]['sigma'], 1e-4)
                    log_trans = log_gaussian_pdf(dt, mu, sigma)
                    score += lambda_val * log_trans
                    
                    if score > best_score:
                        best_score = score
                        best_u = u
                
                if best_u != -1:
                    dp[pos][v] = best_score
                    parent[pos][v] = best_u
        
        # Backtrack optimal path
        final_candidates = {d: dp[5][d] for d in candidate_sets[5] if d in dp[5]}
        if not final_candidates:
            pred_digits = [str(np.argmax(l)) for l in all_logits]
            pred_pin = "".join(pred_digits)
        else:
            last = max(final_candidates, key=final_candidates.get)
            path = [last]
            for pos in range(5, 0, -1):
                last = parent[pos][last]
                path.append(last)
            pred_pin = ''.join(str(d) for d in reversed(path))
        
        if pred_pin == pin_str:
            correct += 1
    
    return correct / total

def main():
    model, norm_params, kinematic_params_orig, df = load_model_and_data()
    
    # 测试不同的 sigma 缩放因子
    scales = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    accuracies = []
    
    print("\n🔄 Starting Fitts' Law parameter sensitivity analysis...")
    for s in scales:
        print(f"\n   Testing σ scaling factor: {s}x")
        acc = evaluate_with_sigma_scale(
            df, model, norm_params, kinematic_params_orig,
            sigma_scale=s,
            use_spatial=True,      # Use complete system
            lambda_val=1.5,        # Consistent with main experiment
            top_k=3                # Consistent with main experiment
        )
        accuracies.append(acc)
        print(f"   → Accuracy: {acc:.2%}")
    
    # 绘图
    plt.figure(figsize=(7, 4.5))
    plt.plot(scales, accuracies, marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
    plt.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.2, label='Original σ (Main Experiment)')
    plt.xlabel('σ Scaling Factor (Time Variance)', fontsize=12)
    plt.ylabel('PIN Inference Accuracy (Top-1)', fontsize=12)
    plt.title('Robustness Analysis for Typing Speed Variations', fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()
    plt.tight_layout()
    # plt.savefig('sigma_sensitivity.pdf', bbox_inches='tight', dpi=300)
    # print("\n✅ Sensitivity curve saved as: sigma_sensitivity.pdf")

    # Print table
    print("\n" + "="*45)
    print("σ Scaling Factor | Accuracy")
    print("-"*45)
    for s, acc in zip(scales, accuracies):
        mark = " ← Main experiment config" if s == 1.0 else ""
        print(f"{s:9.1f}x | {acc:8.2%}{mark}")
    print("="*45)

if __name__ == '__main__':
    main()
