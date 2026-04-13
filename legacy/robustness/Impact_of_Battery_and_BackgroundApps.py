# evaluate_for_boxplot.py
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

from legacy.compat import battery_background_files, legacy_output_path, load_model_bundle

# --- Reuse your model and inference logic ---
KEY_POS = {1: (0,0), 2: (0,1), 3: (0,2), 4: (1,0), 5: (1,1), 6: (1,2), 7: (2,0), 8: (2,1), 9: (2,2), 0: (3,1)}

def euclidean_dist(u, v):
    if u == v: return 0.0
    x1, y1 = KEY_POS[u]; x2, y2 = KEY_POS[v]
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def kinematic_level(u, v):
    d = euclidean_dist(u, v)
    if d == 0: return 0
    elif d <= 1: return 1
    elif d <= 1.42: return 2
    elif d <= 2: return 3
    elif d <= 2.24: return 4
    elif d <= 2.83: return 5
    elif d <= 3: return 6
    else: return 7

class MorphologyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,10)
        )
    def forward(self, x): return self.net(x)

def log_gaussian_pdf(x, mu, sigma):
    return -0.5*((x-mu)/sigma)**2 - np.log(sigma) - 0.5*np.log(2*np.pi)

def evaluate_single_file(csv_path, model, norm_params, kinematic_params, LAMBDA=1.5, TOP_K=3):
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        pin_str = str(int(row.iloc[0])).zfill(6)
        timestamps = [float(row.iloc[i]) for i in range(1,7)]
        all_logits = []
        candidate_sets = []
        for i in range(6):
            feats = np.array([row.iloc[7+i*5+j] for j in range(5)], dtype=np.float32)
            feats_norm = (feats - norm_params['mean']) / norm_params['std']
            with torch.no_grad():
                logits = model(torch.from_numpy(feats_norm).unsqueeze(0)).squeeze().cpu().numpy()
            all_logits.append(logits)
            topk_idx = np.argsort(logits)[-TOP_K:][::-1].tolist()
            candidate_sets.append(set(topk_idx))
        
        dp = [{} for _ in range(6)]
        parent = [{} for _ in range(6)]
        for d in candidate_sets[0]:
            dp[0][d] = float(all_logits[0][d])
        for pos in range(1,6):
            dt = max(timestamps[pos]-timestamps[pos-1], 0.01)
            for v in candidate_sets[pos]:
                best_score, best_u = -1e12, -1
                for u in candidate_sets[pos-1]:
                    if u not in dp[pos-1]: continue
                    k = kinematic_level(u,v)
                    mu, sigma = kinematic_params[k]['mu'], max(kinematic_params[k]['sigma'],1e-4)
                    log_trans = log_gaussian_pdf(dt, mu, sigma)
                    score = dp[pos-1][u] + LAMBDA*log_trans + all_logits[pos][v]
                    if score > best_score:
                        best_score, best_u = score, u
                dp[pos][v] = best_score
                parent[pos][v] = best_u
        final = {d:dp[5][d] for d in candidate_sets[5] if d in dp[5]}
        if final:
            last = max(final, key=final.get)
            path = [last]
            for p in range(5,0,-1): last = parent[p][last]; path.append(last)
            pred = ''.join(str(d) for d in reversed(path))
        else:
            pred = ''.join(str(int(np.argmax(l))) for l in all_logits)
        results.append(pred == pin_str)
    return np.mean(results) * 100  # Return percentage

def main():
    # Load model
    model, norm_params, kinematic_params = load_model_bundle()
    
    # Collect all files
    files = battery_background_files()
    results = []
    
    for f in sorted(files):
        parts = os.path.basename(f).replace('.csv','').split('_')
        battery = parts[1]
        load = parts[2]
        run_id = int(parts[3][3:])
        
        acc = evaluate_single_file(f, model, norm_params, kinematic_params)
        battery_label = {'3pct':'3%', '30pct':'30%', '70pct':'70%', '100pct':'100%'}[battery]
        results.append({
            'Battery': battery_label,
            'Load': 'Idle' if load=='idle' else 'Busy',
            'Accuracy': acc
        })
        print(f"Evaluated: {f} → {acc:.2f}%")
    
    df = pd.DataFrame(results)
    # Set classification order to ensure correct sorting
    df['Battery'] = pd.Categorical(df['Battery'], categories=['3%', '30%', '70%', '100%'], ordered=True)
    df['Load'] = pd.Categorical(df['Load'], categories=['Idle', 'Busy'], ordered=True)
    boxplot_path = legacy_output_path("robustness", "boxplot_data.csv")
    df.to_csv(boxplot_path, index=False)
    print(f"\n✅ All evaluations completed, results saved to {boxplot_path}")
    
    # Calculate average accuracy and standard deviation (error)
    stats_df = df.groupby(['Battery', 'Load'])['Accuracy'].agg(['mean', 'std']).unstack()
    # Adjust column name format
    stats_df.columns = [f'{load}_{stat}' for stat, load in stats_df.columns]
    # Save statistical results
    stats_path = legacy_output_path("robustness", "accuracy_stats.csv")
    stats_df.to_csv(stats_path)
    print(f"📊 Accuracy statistical results saved to {stats_path}")

if __name__ == '__main__':
    main()
