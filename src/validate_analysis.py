import torch
import json
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# --- CONFIG ---
BASE_DIR = "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection"
INPUT_FILE = os.path.join(BASE_DIR, "outputs/extracted_features.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs/test_features.json")
WEIGHT_FILE = os.path.join(BASE_DIR, "models/fusion_head.pth")

def create_test_file():
    print(f"Loading features from {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    
    # Split the data exactly as done in train_fusion.py
    # This ensures the test set is consistent
    _, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"Saving {len(test_data)} test samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(test_data, f, indent=4)
    
    print("✅ Done! You can now run your validation script.")


# Import your model class from your training script
from src.train_fusion import NeuralFusionHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_NAMES = ["Stage 1: Subjectivity", "Stage 2: Reasoning", "Stage 3: CLIP Sim"]

# 1. Create a Wrapper Class
class SklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = [0, 1]

    def fit(self, X, y):
        return self # Not needed for analysis, but required for Sklearn

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            return (outputs.cpu().numpy() > 0.5).astype(int).flatten()

def validate_and_analyze(features_path=OUTPUT_FILE, weights_path=WEIGHT_FILE):
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
    wrapped_model = SklearnWrapper(model)

    # def model_predict(X_np):
    #     X_tensor = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    #     with torch.no_grad():
    #         outputs = model(X_tensor)
    #         return (outputs.cpu().numpy() > 0.5).astype(int)

    # 3. Basic Metrics
    y_pred = wrapped_model.predict(X)
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y, y_pred):.4f}")
    print("\nFull Report:")
    print(classification_report(y, y_pred, target_names=["Real", "Fake"]))

    # 4. Feature Importance (Permutation Method)
    # We shuffle one feature at a time and see how much Accuracy drops
    print("--- Computing Feature Importance ---")
    r = permutation_importance(wrapped_model, X, y, n_repeats=10, random_state=42)
    
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
    plt.savefig("outputs/test/feature_importance.png")
    
    # 6. Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title('Confusion Matrix')
    plt.savefig("outputs/test/confusion_matrix.png")
    
    print("\nAnalysis complete! Visuals saved to 'feature_importance.png' and 'confusion_matrix.png'.")

if __name__ == "__main__":
    # create_test_file() # Run for the first time if test file is not created 
    validate_and_analyze()