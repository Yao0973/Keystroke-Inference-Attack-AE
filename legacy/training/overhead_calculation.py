import time
import torch
import torch.nn as nn
import numpy as np
import tracemalloc
import os
from math import log, pi

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency for nicer RSS reporting
    psutil = None

# ==========================================
# 1. Core Classes and Functions Reuse 
# ==========================================

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
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

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
            nn.Linear(5, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

def log_gaussian_pdf(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - log(sigma) - 0.5 * log(2 * pi)

# ==========================================
# 2. Simulate Single Inference Logic (Core Algorithm Encapsulation)
# ==========================================

def inference_one_sequence(model, features_seq, timestamps, norm, kinematic_params, TOP_K=3, LAMBDA=1.5):
    """
    Simulate processing a complete 6-digit PIN sequence
    features_seq: shape (6, 5)
    timestamps: list of 6 floats
    """
    all_logits = []
    candidate_sets = []

    # --- Phase 1: Feature Extraction and MLP Inference ---
    # Simulate keystrokes arriving one by one
    for i in range(6):
        feats = features_seq[i]

        # Normalization (simulated computation)
        feats_norm = (feats - norm['mean']) / norm['std']

        # MLP Inference
        with torch.no_grad():
            # unsqueeze(0) simulates batch_size=1
            input_tensor = torch.from_numpy(feats_norm).unsqueeze(0)
            logits = model(input_tensor).squeeze(0).cpu().numpy()

        all_logits.append(logits)

        # Top-K Candidate Generation
        if TOP_K >= 10:
            candidates = set(range(10))
        else:
            topk_idx = np.argsort(logits)[-TOP_K:][::-1].tolist()
            candidates = set(topk_idx)
        candidate_sets.append(candidates)

    # --- Phase 2: Viterbi Joint Inference ---
    dp = [{} for _ in range(6)]
    parent = [{} for _ in range(6)]

    # Initialization
    for d in candidate_sets[0]:
        dp[0][d] = float(all_logits[0][d])

    # Recursion
    for pos in range(1, 6):
        dt = max(timestamps[pos] - timestamps[pos - 1], 0.01)
        for v in candidate_sets[pos]:
            best_score = -1e12
            best_u = -1
            for u in candidate_sets[pos - 1]:
                if u not in dp[pos - 1]:
                    continue

                # Physical constraint calculation
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

    # Backtracking (simple simulation, negligible overhead)
    final_candidates = {d: dp[5][d] for d in candidate_sets[5] if d in dp[5]}
    # (Omit specific path construction as it has negligible overhead)
    return final_candidates

# ==========================================
# 3. Benchmark Testing Main Program
# ==========================================

def run_benchmark():
    # --- A. Environment Setup ---
    print("🚀 Initializing benchmark testing...")
    device = torch.device('cpu') # Force CPU, simulating constrained device
    model = MorphologyMLP().to(device)
    model.eval()

    # Mock Parameters
    norm_mock = {'mean': np.zeros(5, dtype=np.float32), 'std': np.ones(5, dtype=np.float32)}
    # Mock kinematic parameters (8 levels)
    kinematic_mock = {i: {'mu': 0.2, 'sigma': 0.05} for i in range(8)}

    # Mock input data (one 6-digit PIN)
    # 6 keystrokes, each with 5-dimensional features
    dummy_features = np.random.randn(6, 5).astype(np.float32)
    # 6 timestamps, 0.2s intervals
    dummy_timestamps = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

    # --- B. Latency Testing ---
    print("\n⏱️  Measuring latency...")

    # Warm-up
    for _ in range(50):
        inference_one_sequence(model, dummy_features, dummy_timestamps, norm_mock, kinematic_mock)

    # Formal testing
    num_loops = 1000
    t_start = time.perf_counter()
    for _ in range(num_loops):
        inference_one_sequence(model, dummy_features, dummy_timestamps, norm_mock, kinematic_mock)
    t_end = time.perf_counter()

    avg_latency_ms = (t_end - t_start) * 1000 / num_loops
    print(f"✅ Average 6-digit PIN inference latency: {avg_latency_ms:.4f} ms")

    # --- C. Memory Testing ---
    print("\n💾 Measuring memory usage...")
    tracemalloc.start()

    # Run one inference
    inference_one_sequence(model, dummy_features, dummy_timestamps, norm_mock, kinematic_mock)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    print(f"✅ Peak Memory Allocation: {peak_mb:.4f} MB")

    # Process-level memory
    if psutil is not None:
        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024 * 1024)
        print(f"✅ Total Process Memory (RSS): {rss_mb:.2f} MB")
    else:
        rss_mb = peak_mb
        print("ℹ️ psutil is not installed; using tracemalloc peak as the memory proxy.")

    print("\n" + "="*40)
    print(f"Latency (End-to-End): < {avg_latency_ms + 1:.0f} ms") # Slight rounding
    print(f"Memory Overhead:      ~{rss_mb:.0f} MB")
    print("="*40)

if __name__ == '__main__':
    run_benchmark()
