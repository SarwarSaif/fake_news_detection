# 1. UN_SLOTH MUST BE FIRST
import unsloth 
from unsloth import FastLanguageModel

# 2. Standard Imports
import os
import json
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# 3. Transformers & Tracking
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
from src.data.mmfakebench import MMFakeBenchDataset
from src.tracking import set_tracker, init, log, finish
from src.tracking.wandb_tracker import WandbTracker

# --- DIRECTORY CONFIG ---
# Corrected based on your specific Google Drive structure
BASE_DIR = "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection"
DATA_ROOT = os.path.join(BASE_DIR, "data/MMFakeBench2")
ANN_PATH = os.path.join(DATA_ROOT, "MMFakeBench_test.json") 
IMAGE_ROOT = os.path.join(DATA_ROOT, "MMFakeBench_test") 

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 

# --------------------------
# Path Validation
# --------------------------
def validate_paths():
    print("--- Validating Paths ---")
    critical_paths = [ANN_PATH, IMAGE_ROOT]
    for p in critical_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"CRITICAL PATH MISSING: {p}")
    print("✅ All paths verified.")

# --------------------------
# Stage 1: Subjectivity (DeBERTa)
# --------------------------
class SubjectivityAnalyzer:
    def __init__(self):
        model_weights = "MatteoFasulo/mdeberta-v3-base-subjectivity-english"
        tokenizer_base = "microsoft/mdeberta-v3-base"
        
        print(f"--- Stage 1: Loading {model_weights} ---")
        # Fix for EntryNotFoundError: Download from official MS repo
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
                print(f"⚠️ Error loading image {p}: {e}")
                processed_images.append(Image.new('RGB', (224, 224), color='gray'))

        inputs = self.processor(text=texts, images=processed_images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            out = self.model(**inputs)
            sim = torch.sum(out.image_embeds * out.text_embeds, dim=1)
            return sim.view(-1, 1)

# --------------------------
# Execution Logic
# --------------------------
def main():
    validate_paths()
    
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    
    tracker = WandbTracker(project="fake-news-detection", run_name="e2e_mmfakebench_run")
    set_tracker(tracker)
    init()

    # Init Models with memory clearing between loads
    s1 = SubjectivityAnalyzer()
    gc.collect(); torch.cuda.empty_cache()
    
    s2 = SemanticReasoner()
    gc.collect(); torch.cuda.empty_cache()
    
    s3 = CLIPLoraEncoder()
    gc.collect(); torch.cuda.empty_cache()

    # Load Data
    dataset = MMFakeBenchDataset(json_path=ANN_PATH, root_dir=IMAGE_ROOT)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    output_file = os.path.join(OUTPUT_DIR, "extracted_features.json")

    # Check if we already have progress
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            extracted_data = json.load(f)
        start_step = len(extracted_data)
        print(f"⏩ Resuming from step {start_step}...")
    else:
        extracted_data = []
        start_step = 0

    print(f"--- Starting Extraction Loop on {DEVICE} ---")
    for step, batch in enumerate(tqdm(loader)):

        # Skip steps we already processed
        if step < start_step:
            continue
        texts = batch["text"]
        
        # Since MMFakeBenchDataset already joined the root_dir, 
        # the batch['image_path'] is already absolute.
        img_paths = batch["image_path"] 
        
        try:
            f1 = s1.get_score(texts)
            f2 = s2.get_stance(texts)
            f3 = s3.get_similarity(img_paths, texts)

            for i in range(len(texts)):
                extracted_data.append({
                    "id": batch.get("id", [f"sample_{step}_{i}"])[i],
                    "features": [f1[i].item(), f2[i].item(), f3[i].item()],
                    "label": 1 if batch["label"][i] == "Fake" else 0
                })
            # Save every 50 steps so you never lose more than a few minutes of work
            if step % 50 == 0:
                with open(output_file, "w") as f:
                    json.dump(extracted_data, f)

        except Exception as e:
            print(f"⚠️ Skipping step {step} due to error: {e}")
            continue

        # Log periodically to WandB
        if step % 10 == 0:
            log({"processed_steps": step})
    
    print(f"✅ Success! Features saved to {output_file}")
    finish()

if __name__ == "__main__":
    main()
