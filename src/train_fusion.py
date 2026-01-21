import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from dotenv import load_dotenv
import numpy as np
import wandb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# FIXED: Added missing imports here
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.tracking.wandb_tracker import WandbTracker
from src.tracking import set_tracker, init, log, finish

# --- CONFIG ---
BASE_DIR = "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection"
FEATURES_PATH = os.path.join(BASE_DIR, "outputs/extracted_features.json")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/fusion_head.pth")

class NeuralFusionHead(nn.Module):
    def __init__(self):
        super(NeuralFusionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train():
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    
    tracker = WandbTracker(
        project="fake-news-detection", 
        run_name="neural_fusion_v2_metrics",
        config={
            "architecture": "MLP-3-16-8-1",
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32
        }
    )
    set_tracker(tracker)
    init()

    with open(FEATURES_PATH, "r") as f:
        data = json.load(f)

    X = torch.tensor([item["features"] for item in data], dtype=torch.float32)
    y = torch.tensor([item["label"] for item in data], dtype=torch.float32).view(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    model = NeuralFusionHead()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    wandb.watch(model, log="all", log_freq=10)

    print("--- Starting Fusion Training with Full Metrics ---")
    for epoch in range(100):
        model.train()
        train_preds, train_true = [], []
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_preds.extend((outputs > 0.5).float().cpu().numpy())
            train_true.extend(batch_y.cpu().numpy())

        # Validation per epoch
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_preds_np = (test_outputs > 0.5).float().cpu().numpy()
            y_test_np = y_test.cpu().numpy()
            
            # Metrics calculation
            t_acc = accuracy_score(train_true, train_preds)
            v_acc = accuracy_score(y_test_np, test_preds_np)
            v_f1 = f1_score(y_test_np, test_preds_np)

        log({
            "train/loss": epoch_loss / len(train_loader),
            "train/accuracy": t_acc,
            "test/accuracy": v_acc,
            "test/f1_score": v_f1
        }, step=epoch)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Acc {t_acc:.4f} | Test Acc {v_acc:.4f}")

    # Final visual reporting
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test).cpu().numpy().flatten()
        final_preds = (final_outputs > 0.5).astype(int)
        y_true = y_test.cpu().numpy().flatten().astype(int)
        
        # Binary probability formatting for WandB
        probas_2d = np.vstack([1 - final_outputs, final_outputs]).T

        log({
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=final_preds,
                class_names=["Real", "Fake"]
            ),
            "roc": wandb.plot.roc_curve(y_true, probas_2d),
            "pr": wandb.plot.pr_curve(y_true, probas_2d)
        })

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Training Complete. Model saved to {MODEL_SAVE_PATH}")
    finish()

if __name__ == "__main__":
    train()