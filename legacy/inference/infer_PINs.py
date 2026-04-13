import torch
import torch.nn as nn
import numpy as np
import csv
import os
import sys
from math import log, pi
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import load_model_bundle, pin_reconstruction_path

# ----------------------------
# 模型定义（与训练时一致）
# ----------------------------
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

# ----------------------------
# 键盘布局 & 动力学特征
# ----------------------------
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
    if d == 0: return 0
    elif d <= 1: return 1
    elif d <= 1.42: return 2
    elif d <= 2: return 3
    elif d <= 2.24: return 4
    elif d <= 2.83: return 5
    elif d <= 3: return 6
    else: return 7

def log_gaussian_pdf(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - log(sigma) - 0.5 * log(2 * pi)

# ----------------------------
# 主函数
# ----------------------------
def main():
    dataset_path = pin_reconstruction_path('6digit_PINs.csv')
    if not dataset_path.exists():
        raise FileNotFoundError(f"缺少数据文件: {dataset_path}")

    print("🧠 Loading model and normalization parameters...")
    model, norm, kinematic = load_model_bundle()

    # 配置
    USE_KINEMATIC = True
    LAMBDA = 1.5
    TOP_K = 10  # 使用全部10个数字作为候选

    # 读取 CSV 文件
    print(f"📂 Reading {dataset_path} ...")
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    print(f"✅ Loaded {len(rows)} samples in total\n")
    print(f"{'Index':<4} | {'True PIN':<8} | {'Predicted PIN':<8} | Result")
    print("-" * 40)

    correct = 0
    for idx, row in enumerate(rows):
        if len(row) != 37:
            print(f"⚠️ Row {idx} has incorrect format, skipping.")
            continue
        
        # 解析数据
        true_pin = str(int(float(row[0]))).zfill(6)
        data = np.array([float(x) for x in row[1:]], dtype=np.float32)
        timestamps = data[:6]           # 6 个时间戳
        features = data[6:].reshape(6, 5)  # 6 个按键 × 5 维特征

        # 获取每个位置的 logits 和候选集
        all_logits = []
        candidate_sets = []
        for i in range(6):
            feats = features[i]
            # ⚠️ 修复点：norm['mean'] 已是 numpy 数组，无需 .numpy()
            feats_norm = (feats - norm['mean']) / norm['std']
            with torch.no_grad():
                logits = model(torch.from_numpy(feats_norm).unsqueeze(0).float()).squeeze(0).cpu().numpy()
            all_logits.append(logits)
            candidates = set(range(10)) if TOP_K == 10 else set(np.argsort(logits)[-TOP_K:].tolist())
            candidate_sets.append(candidates)

        # ====== 联合推理（带动力学）======
        if not USE_KINEMATIC:
            pred_pin = ''.join(str(int(np.argmax(logits))) for logits in all_logits)
        else:
            # DP 初始化
            dp = [{} for _ in range(6)]
            parent = [{} for _ in range(6)]
            for d in candidate_sets[0]:
                dp[0][d] = float(all_logits[0][d])
            
            # DP 递推
            for pos in range(1, 6):
                dt = max(timestamps[pos] - timestamps[pos - 1], 0.01)
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
            
            # 回溯路径
            final_candidates = {d: dp[5][d] for d in candidate_sets[5] if d in dp[5]}
            if not final_candidates:
                pred_pin = ''.join(str(int(np.argmax(logits))) for logits in all_logits)
            else:
                last_digit = max(final_candidates, key=final_candidates.get)
                path = [last_digit]
                for p in range(5, 0, -1):
                    last_digit = parent[p][last_digit]
                    path.append(last_digit)
                pred_pin = ''.join(str(d) for d in reversed(path))

        # 判断是否正确
        match = "✅" if pred_pin == true_pin else "❌"
        if pred_pin == true_pin:
            correct += 1

        print(f"{idx:<4} | {true_pin}   | {pred_pin}   | {match}")

    # 输出最终准确率
    accuracy = correct / len(rows) if rows else 0
    print("-" * 40)
    print(f"🎯 Overall Accuracy: {accuracy:.2%} ({correct}/{len(rows)})")

if __name__ == "__main__":
    main()
