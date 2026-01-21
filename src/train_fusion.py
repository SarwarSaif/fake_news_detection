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
from sklearn.metrics import classification_report

from src.tracking.wandb_tracker import WandbTracker
from src.tracking import set_tracker, init, log, finish

# --- CONFIG ---
BASE_DIR = "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection"
FEATURES_PATH = os.path.join(BASE_DIR, "outputs/extracted_features.json")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/fusion_head.pth")

# 1. Define the Fusion Model
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
    # 2. Initialize Tracking
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    
    tracker = WandbTracker(
        project="fake-news-detection", 
        run_name="neural_fusion_training",
        config={
            "architecture": "MLP-3-16-8-1",
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32
        }
    )
    set_tracker(tracker)
    init()

    # 3. Load the extracted features
    with open(FEATURES_PATH, "r") as f:
        data = json.load(f)

    X = torch.tensor([item["features"] for item in data], dtype=torch.float32)
    y = torch.tensor([item["label"] for item in data], dtype=torch.float32).view(-1, 1)

    # 4. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    # 5. Initialize Model
    model = NeuralFusionHead()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Watch gradients
    wandb.watch(model, log="all", log_freq=10)

    print("--- Training Fusion Head ---")
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Log training loss to W&B
        avg_loss = epoch_loss / len(train_loader)
        log({"train_loss": avg_loss}, step=epoch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    # 6. Final Evaluation & Advanced Tracking
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_preds = (test_outputs > 0.5).float()
        
        # Calculate Accuracy
        correct = (test_preds == y_test).float().sum()
        accuracy = correct / y_test.shape[0]
        
        # Log Summary Metric
        wandb.run.summary["final_test_accuracy"] = accuracy.item()
        print(f"\n✅ Final Test Accuracy: {accuracy.item()*100:.2f}%")

        # Log Confusion Matrix and PR Curves
        y_true_np = y_test.cpu().numpy().flatten()
        y_pred_np = test_preds.cpu().numpy().flatten()
        y_probas_np = test_outputs.cpu().numpy().flatten()

        # Format probabilities for wandb (requires [N, 2] for binary)
        probas_2d = np.vstack([1 - y_probas_np, y_probas_np]).T

        log({
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_np, 
                preds=y_pred_np,
                class_names=["Real", "Fake"]
            ),
            "pr_curve": wandb.plot.pr_curve(
                y_true_np, 
                probas_2d, 
                labels=["Real", "Fake"]
            ),
            "roc_curve": wandb.plot.roc_curve(
                y_true_np, 
                probas_2d, 
                labels=["Real", "Fake"]
            )
        })

    # 7. Save Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"🚀 Fusion Head saved to {MODEL_SAVE_PATH}")
    finish()

if __name__ == "__main__":
    train()