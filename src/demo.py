import torch
import torch.nn as nn
import requests
from PIL import Image
from io import BytesIO
import os

# Import your stage classes and the Fusion Head from previous scripts
from main import SubjectivityAnalyzer, SemanticReasoner, CLIPLoraEncoder
from train_fusion import NeuralFusionHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FakeNewsInference:
    def __init__(self, fusion_weights="fusion_head.pth"):
        print("--- Initializing Stage 1: Subjectivity (DeBERTa) ---")
        self.stage1 = SubjectivityAnalyzer()
        print("--- Initializing Stage 2: Reasoning (Gemma) ---")
        self.stage2 = SemanticReasoner()
        print("--- Initializing Stage 3: Consistency (CLIP-LoRA) ---")
        self.stage3 = CLIPLoraEncoder()
        
        print("--- Loading Fusion Head Weights ---")
        self.fusion_head = NeuralFusionHead().to(DEVICE)
        self.fusion_head.load_state_dict(torch.load(fusion_weights))
        self.fusion_head.eval()

    def load_image(self, path_or_url):
        if path_or_url.startswith(('http://', 'https://')):
            response = requests.get(path_or_url)
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(path_or_url).convert("RGB")

    def predict(self, text, image_source):
        # 1. Prepare Data
        # Stage 3 expects a path, so we temporarily save if it's a URL
        temp_img_path = "temp_inference.jpg"
        img = self.load_image(image_source)
        img.save(temp_img_path)

        # 2. Extract Stage Features
        print("\nProcessing functional stages...")
        f1 = self.stage1.get_score([text])
        f2 = self.stage2.get_stance([text])
        f3 = self.stage3.get_similarity([temp_img_path], [text])

        # 3. Final Fusion Verdict
        with torch.no_grad():
            prob = self.fusion_head(f1, f2, f3).item()
        
        # 4. Clean up
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        return {
            "verdict": "FAKE" if prob > 0.5 else "REAL",
            "confidence": prob if prob > 0.5 else 1 - prob,
            "stage_scores": {
                "subjectivity": f1.item(),
                "reasoning_logit": f2.item(),
                "visual_consistency": f3.item()
            }
        }

if __name__ == "__main__":
    # Example usage
    detector = FakeNewsInference()
    
    headline = "Scientists discover that chocolate cures all diseases overnight!"
    img_url = "https://images.unsplash.com/photo-1548900912-381002e07f4e" # Generic chocolate image

    result = detector.predict(headline, img_url)
    
    print("\n" + "="*30)
    print(f"VERDICT: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("-" * 30)
    print(f"Text Subjectivity: {result['stage_scores']['subjectivity']:.4f}")
    print(f"Logical Consistency Logit: {result['stage_scores']['reasoning_logit']:.4f}")
    print(f"Visual-Textual Alignment: {result['stage_scores']['visual_consistency']:.4f}")
    print("="*30)