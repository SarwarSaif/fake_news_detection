import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, TensorDataset

class NeuralFusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train():
    # 1. Load Pre-extracted Features
    with open("features.json", "r") as f:
        data = json.load(f)
    
    X = torch.tensor([d["features"] for d in data], dtype=torch.float32)
    y = torch.tensor([d["label"] for d in data], dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Training Setup
    model = NeuralFusionHead().to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 3. Training Loop
    print("Training Neural Fusion Head...")
    for epoch in range(50):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to("cuda"), by.to("cuda")
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "fusion_head.pth")
    print("Training complete. Weights saved as fusion_head.pth")

if __name__ == "__main__":
    train()