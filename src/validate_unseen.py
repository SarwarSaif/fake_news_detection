# 1. UN_SLOTH MUST BE FIRST
import unsloth 
from unsloth import FastLanguageModel

import os
import json
import torch
import gc
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

# Transformers & PEFT Imports
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model

# Local Project Imports
from src.train_fusion import NeuralFusionHead
from src.data.mmfakebench import MMFakeBenchDataset

# --- CONFIG ---
BASE_DIR = "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection"
VAL_JSON = os.path.join(BASE_DIR, "data/MMFakeBench2/MMFakeBench_val/source/MMFakeBench_val.json")
VAL_IMAGE_ROOT = os.path.join(BASE_DIR, "data/MMFakeBench2/MMFakeBench_val")

MODEL_WEIGHTS = os.path.join(BASE_DIR, "models/fusion_head.pth")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models") # <--- ADDED THIS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --------------------------
# Stage 1: Subjectivity (DeBERTa)
# --------------------------
class SubjectivityAnalyzer:
    def __init__(self):
        model_weights = "MatteoFasulo/mdeberta-v3-base-subjectivity-english"
        tokenizer_base = "microsoft/mdeberta-v3-base"
        print(f"--- Stage 1: Loading {model_weights} ---")
        vocab_path = hf_hub_download(repo_id=tokenizer_base, filename="spm.model", cache_dir=MODEL_SAVE_DIR)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_base, vocab_file=vocab_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_weights, cache_dir=MODEL_SAVE_DIR).to(DEVICE).eval()
        
    def get_score(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=1)[:, 1].view(-1, 1)

# --------------------------
# Stage 2: Reasoning (Gemma-2)
# --------------------------
class SemanticReasoner:
    def __init__(self):
        print("--- Stage 2: Loading Gemma-2-9B (4-bit) ---")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-2-9b-it-bnb-4bit",
            max_seq_length=2048,
            load_in_4bit=True,
            cache_dir=os.path.join(BASE_DIR, "unsloth_compiled_cache")
        )
        FastLanguageModel.for_inference(self.model)

    def get_stance(self, texts):
        prompts = [f"Claim: {t}\nDoes this contain factual contradictions? Answer Yes or No:" for t in texts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
            return outputs.scores[0].max(dim=1)[0].view(-1, 1)

# --------------------------
# Stage 3: Consistency (CLIP-LoRA)
# --------------------------
class CLIPLoraEncoder:
    def __init__(self):
        print("--- Stage 3: Loading CLIP-LoRA ---")
        base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODEL_SAVE_DIR)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        config = LoraConfig(r=16, target_modules=["visual_projection", "text_projection"])
        self.model = get_peft_model(base_model, config).to(DEVICE).eval()

    def get_similarity(self, image_paths, texts):
        processed_images = []
        for p in image_paths:
            try:
                processed_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                processed_images.append(Image.new('RGB', (224, 224), color='gray'))

        inputs = self.processor(text=texts, images=processed_images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            out = self.model(**inputs)
            sim = torch.sum(out.image_embeds * out.text_embeds, dim=1)
            return sim.view(-1, 1)

# --------------------------
# Main Logic
# --------------------------
def validate_unseen():
    # 1. Initialize Feature Extractors
    s1 = SubjectivityAnalyzer()
    gc.collect(); torch.cuda.empty_cache()
    
    s2 = SemanticReasoner()
    gc.collect(); torch.cuda.empty_cache()
    
    s3 = CLIPLoraEncoder()
    gc.collect(); torch.cuda.empty_cache()

    # 2. Initialize Fusion Head
    fusion_model = NeuralFusionHead().to(DEVICE)
    fusion_model.load_state_dict(torch.load(MODEL_WEIGHTS))
    fusion_model.eval()

    # 3. Load New Data
    dataset = MMFakeBenchDataset(json_path=VAL_JSON, root_dir=VAL_IMAGE_ROOT)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    correct = 0

    print(f"--- Processing {len(dataset)} Unseen Samples ---")
    for step, batch in enumerate(tqdm(loader)):
        texts = batch["text"]
        img_paths = batch["image_path"]
        label_raw = batch["label"][0]
        # Standardize labels to 0/1
        ground_truth = 1 if label_raw in ["Fake", "fake", "1"] else 0

        try:
            f1 = s1.get_score(texts)
            f2 = s2.get_stance(texts)
            f3 = s3.get_similarity(img_paths, texts)
            
            features = torch.cat([f1, f2, f3], dim=1).to(DEVICE)

            with torch.no_grad():
                prob = fusion_model(features)
                prediction = 1 if prob.item() > 0.5 else 0

            if prediction == ground_truth:
                correct += 1

            results.append({
                "id": step,
                "probability": prob.item(),
                "prediction": "Fake" if prediction == 1 else "Real",
                "actual": "Fake" if ground_truth == 1 else "Real"
            })

        except Exception as e:
            print(f"Error at step {step}: {e}")
            continue

    final_acc = (correct / len(results)) * 100 if len(results) > 0 else 0
    print(f"\n✅ Final Accuracy on Unseen Data: {final_acc:.2f}%")

    with open(os.path.join(BASE_DIR, "outputs/unseen_validation_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    validate_unseen()