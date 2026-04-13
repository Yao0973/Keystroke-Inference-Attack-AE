import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import classification_path

# ==========================================
# 1. Define MLP model 
# ==========================================
class MorphologyMLP(nn.Module):
    def __init__(self, input_dim=5, num_classes=10):
        super().__init__()
        # Extracted network structure from original file
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
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

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Training and evaluation functions
# ==========================================
def train_and_evaluate(train_loader, X_test_tensor, y_test_tensor, device):
    model = MorphologyMLP().to(device)
    
    # Use class weights from original file (Class Weights)
    class_weights = torch.tensor([
        140.0, 15.0, 50.0, 10.0, 35.0, 30.0, 
        30.0, 50.0, 150.0, 100.0
    ], device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    # Use Cosine Annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    model.train()
    # Train for 35 epochs (refer to original file)
    for epoch in range(35): 
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluation (Top-3 Accuracy)
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
        _, top3_pred = logits.topk(3, dim=1)
        y_true_reshaped = y_test_tensor.to(device).view(-1, 1)
        correct_top3 = (top3_pred == y_true_reshaped).sum().item()
        acc_top3 = correct_top3 / y_test_tensor.size(0) * 100.0
        
    return acc_top3

# ==========================================
# 3. Main experiment logic
# ==========================================
def run_experiment():
    # 1. Load data
    print("Loading data...")
    train_df = pd.read_csv(classification_path('train_data.csv'))
    test_df = pd.read_csv(classification_path('test_data.csv'))

    features = ['Peak', 'Energy', 'FWHM', 'RiseTime', 'Centroid']
    target = 'label'

    # 2. Data normalization (MinMaxScaling) - important for MLP
    scaler = MinMaxScaler()
    X_train_all = scaler.fit_transform(train_df[features].values)
    y_train_all = train_df[target].values
    
    X_test = scaler.transform(test_df[features].values)
    y_test = test_df[target].values

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device use: {device}")

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # 3. Experiment parameters
    # Number of training samples per key for the model
    sample_sizes = [5, 10, 20, 30, 40, 50, 60, 70, 100, 200, 400, 800]  

    n_runs = 5 # Repeat 5 times for each sample size to calculate standard deviation
    
    results_mean = []
    results_std = []

    print("Starting training efficiency experiment...")
    for n in sample_sizes:
        accuracies = []
        print(f"Testing training data sample size N={n}...")
        for run in range(n_runs):
            # Build few-shot training set: randomly sample N from each class
            indices = []
            for cls in range(10):
                cls_indices = np.where(y_train_all == cls)[0]
                if len(cls_indices) < n:
                     indices.extend(cls_indices) # Take all if insufficient samples
                else:
                    selected = np.random.choice(cls_indices, n, replace=False)
                    indices.extend(selected)
            
            X_subset = X_train_all[indices]
            y_subset = y_train_all[indices]

            # Create DataLoader
            dataset = TensorDataset(torch.FloatTensor(X_subset), torch.LongTensor(y_subset))
            train_loader = DataLoader(dataset, batch_size=64, shuffle=True) # Reduce batch size for small samples

            # Train and evaluate
            acc = train_and_evaluate(train_loader, X_test_tensor, y_test_tensor, device)
            accuracies.append(acc)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results_mean.append(mean_acc)
        results_std.append(std_acc)
        print(f"N={n}: Mean Top-3 Acc = {mean_acc:.2f}%, Std = {std_acc:.2f}")
        # Save normalization parameters during training phase (speculative code)
        # torch.save(scaler, 'norm_params.pth')  # Save MinMaxScaler parameters
   


if __name__ == "__main__":
    run_experiment()
