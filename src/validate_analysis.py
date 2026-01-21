import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# Import your model class from your training script
from train_fusion import NeuralFusionHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_NAMES = ["Stage 1: Subjectivity", "Stage 2: Reasoning", "Stage 3: CLIP Sim"]

def validate_and_analyze(features_path="test_features.json", weights_path="fusion_head.pth"):
    # 1. Load and Prepare Data
    with open(features_path, "r") as f:
        data = json.load(f)
    
    X = np.array([d["features"] for d in data])
    y = np.array([d["label"] for d in data])

    # 2. Load Model
    model = NeuralFusionHead().to(DEVICE)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Wrapper for Scikit-Learn compatibility
    def model_predict(X_np):
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            outputs = model(X_tensor)
            return (outputs.cpu().numpy() > 0.5).astype(int)

    # 3. Basic Metrics
    y_pred = model_predict(X)
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y, y_pred):.4f}")
    print("\nFull Report:")
    print(classification_report(y, y_pred, target_names=["Real", "Fake"]))

    # 4. Feature Importance (Permutation Method)
    # We shuffle one feature at a time and see how much Accuracy drops
    print("--- Computing Feature Importance ---")
    r = permutation_importance(model_predict, X, y, n_repeats=10, random_state=42)
    
    # 5. Visualizing Importance
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Importance': r.importances_mean,
        'Std': r.importances_std
    }).sort_values(by='Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.errorbar(x=importance_df['Importance'], y=importance_df['Feature'], 
                 xerr=importance_df['Std'], fmt='none', c='black', capsize=5)
    plt.title('Stage Contribution to Final Decision (MMFakeBench)')
    plt.xlabel('Decrease in Accuracy when Shuffled')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    
    # 6. Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    
    print("\nAnalysis complete! Visuals saved to 'feature_importance.png' and 'confusion_matrix.png'.")

if __name__ == "__main__":
    validate_and_analyze()