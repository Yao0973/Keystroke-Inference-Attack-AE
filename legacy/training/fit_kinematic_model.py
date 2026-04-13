# fit_kinematic_model.py (revised version)
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PIN_DATA_ROOT = REPO_ROOT / "data" / "pin_reconstruction"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "models"

KEY_POS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 9: (2, 2),
    0: (3, 1)
}

def euclidean_dist(u, v):
    if u == v: return 0.0
    x1, y1 = KEY_POS[u]
    x2, y2 = KEY_POS[v]
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

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

def main():
    input_file = PIN_DATA_ROOT / 'full_training_data.csv'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ {input_file} does not exist")

    # Critical correction: read CSV with header (first row is header)
    df = pd.read_csv(input_file, header=0)
    print(f"📂 Loaded {len(df)} training samples, each with {df.shape[1]} columns")

    if df.shape[1] != 37:
        raise ValueError(f"Column count error! Expected 37 columns, actual {df.shape[1]} columns")

    samples = {k: [] for k in range(8)}

    for idx, row in df.iterrows():
        pin_str = str(int(row[0])).zfill(6)
        if len(pin_str) != 6:
            continue
        try:
            pin = [int(c) for c in pin_str]
        except:
            continue

        # Timestamps in columns 1-6 (t0~t5)
        try:
            ts = [float(row[i]) for i in range(1, 7)]
        except:
            continue

        # Collect adjacent pairs
        for i in range(1, 6):
            u = pin[i-1]
            v = pin[i]
            dt = ts[i] - ts[i-1]
            if dt <= 0 or dt > 2.0:
                continue
            k = kinematic_level(u, v)
            samples[k].append(dt)

    total_pairs = sum(len(v) for v in samples.values())
    print(f"✅ Collected {total_pairs} valid (u→v, Δt) samples in total")

    # Fit parameters
    kinematic_params = {}
    print("\n📊 Fitting results:")
    print("k |   μ (mean)   |  σ (std)   | sample count")
    print("--|--------------|------------|--------")
    
    BASE_DELAYS = [0.05, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42]
    for k in range(8):
        if len(samples[k]) == 0:
            mu = BASE_DELAYS[k]
            sigma = 0.02
            count = 0
        else:
            mu = float(np.mean(samples[k]))
            sigma = float(np.std(samples[k])) + 1e-6
            count = len(samples[k])
        kinematic_params[k] = {'mu': mu, 'sigma': sigma}
        print(f"{k} |   {mu:.6f}   |  {sigma:.6f}  | {count}")

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    output_file = CHECKPOINT_ROOT / 'kinematic_params.pth'
    torch.save(kinematic_params, output_file)
    print(f"\n💾 Saved temporal model parameters to: {output_file}")

if __name__ == '__main__':
    main()
