import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import recall_score, confusion_matrix
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compat import classification_path, load_model_bundle

# ----------------------------
# 1. Fixed Random Seed (for reproducibility in metrics)
# ----------------------------
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# 2. MLP Model Definition (must match training)
# ----------------------------
class MorphologyMLP(nn.Module):
    def __init__(self, input_dim=5, num_classes=10):
        super().__init__()
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

# ----------------------------
# 3. Load Test Data from CSV
# ----------------------------
def load_test_data(csv_path=None):
    csv_path = classification_path('test_data.csv') if csv_path is None else csv_path
    df = pd.read_csv(csv_path)
    feature_cols = ['Peak', 'Energy', 'FWHM', 'RiseTime', 'Centroid']
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    return X, y

# ----------------------------
# 4. Main Evaluation
# ----------------------------
if __name__ == "__main__":
    test_data_path = classification_path('test_data.csv')
    print(f"📂 Loading test data from '{test_data_path}'...")
    X_test, y_test = load_test_data()

    model, norm_params, _ = load_model_bundle()
    mean = norm_params['mean']
    std = norm_params['std']
    X_test = (X_test - mean) / std

    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_te = torch.from_numpy(X_test).to(device)
    y_te = torch.from_numpy(y_test).to(device)

    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=len(X_te))

    model = model.to(device)
    model.eval()

    print("🧪 Running inference on test set...")
    with torch.no_grad():
        x, y = next(iter(test_loader))
        logits = model(x)
        pred = logits.argmax(dim=1)
        overall_acc = (pred == y).float().mean().item()
        recalls = recall_score(y.cpu().numpy(), pred.cpu().numpy(), average=None)
        cm = confusion_matrix(y.cpu().numpy(), pred.cpu().numpy())

    # Print results
    print("\n" + "="*60)
    print(f"✅ Final Test Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print("\nDigit | Actual Recall")
    print("-" * 25)
    for i in range(10):
        actual = recalls[i] * 100
        print(f"  {i}   |    {actual:>6.2f}%")
    print("="*60)

    print("\n📌 Confusion Matrix (rows: true, cols: predicted):")
    print(cm)

    # Optional: Save confusion matrix as CSV
    # pd.DataFrame(cm).to_csv('confusion_matrix.csv', index=False)
    # print("\n💾 Confusion matrix saved to 'confusion_matrix.csv'")
