# 1. UN_SLOTH MUST BE FIRST
import unsloth 
from unsloth import FastLanguageModel

# 2. Standard Imports
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

# --- OPTIMIZED CONFIG ---
BATCH_SIZE = 8  # Increased from 1
NUM_WORKERS = 4 # Parallel CPU loading
USE_BF16 = torch.cuda.is_bf16_supported() # L4 supports Bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- DIRECTORY CONFIG ---
# Corrected based on your specific Google Drive structure
BASE_DIR = "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection"
DATA_ROOT = os.path.join(BASE_DIR, "data/MMFakeBench2")
ANN_PATH = os.path.join(DATA_ROOT, "MMFakeBench_test.json") 
IMAGE_ROOT = os.path.join(DATA_ROOT, "MMFakeBench_test") 

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")

# main.py
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "extracted_features_v2.json")
WANDB_RUN_META = os.path.join(OUTPUT_DIR, "wandb_run.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)



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

FACTCHECK_PROMPT_TEMPLATE = """
You are an assistant that helps a forensic analyst verify claims.
You are provided a textual message.

RULES:
- Use reliable, well-established world knowledge only.
- If the claim cannot be verified, return "insufficient_evidence".
- Do not hallucinate.
- Be concise and factual.

Task: Produce a valid JSON object with:

1) extracted_claims: list of short factual claims.
2) claim_evaluations: list of objects:
   - claim
   - evaluation: one of ["supports", "contradicts", "insufficient_evidence"]
   - confidence: float 0.0–1.0
3) final_verdict: one of ["likely_real", "likely_fake", "uncertain"]

Output JSON ONLY.
Text:
\"\"\"{text}\"\"\"
"""
class SemanticReasoner:

    def __init__(self):
        print("--- Stage 2: Loading Gemma-2-9B (Optimized) ---")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-2-9b-it-bnb-4bit",
            max_seq_length=2048,
            load_in_4bit=True,
            # L4 specific optimization
            attn_implementation="flash_attention_2" if USE_BF16 else "sdpa"
        )
        FastLanguageModel.for_inference(self.model)
        self.tokenizer.padding_side = "left" # Required for batch generation

    # def __init__(self):
    #     print("--- Stage 2: Loading Gemma-2-9B (4-bit) ---")
    #     self.model, self.tokenizer = FastLanguageModel.from_pretrained(
    #         model_name="unsloth/gemma-2-9b-it-bnb-4bit",
    #         max_seq_length=2048,
    #         load_in_4bit=True,
    #         cache_dir=os.path.join(BASE_DIR, "unsloth_compiled_cache")
    #     )
    #     FastLanguageModel.for_inference(self.model)

    def _extract_reasoning_score(self, json_obj):
        """
        Convert structured reasoning → scalar score
        Higher = more likely Fake
        """
        evals = json_obj.get("claim_evaluations", [])
        if len(evals) == 0:
            return 0.5  # neutral uncertainty

        contradiction = sum(e["evaluation"] == "contradicts" for e in evals)
        support = sum(e["evaluation"] == "supports" for e in evals)
        insufficient = sum(e["evaluation"] == "insufficient_evidence" for e in evals)

        avg_conf = sum(e["confidence"] for e in evals) / len(evals)

        # Reasoning score (bounded, stable)
        score = (
            0.6 * (contradiction / len(evals)) +
            0.2 * (insufficient / len(evals)) -
            0.4 * (support / len(evals))
        )

        return float(torch.clamp(torch.tensor(score * avg_conf + 0.5), 0, 1))
    
    def get_stance(self, texts):
        prompts = [FACTCHECK_PROMPT_TEMPLATE.format(text=t) for t in texts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            # Batch generate is MUCH faster than one-by-one
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # def get_stance(self, texts):
    #     prompts = [
    #         FACTCHECK_PROMPT_TEMPLATE.format(text=t)
    #         for t in texts
    #     ]

    #     inputs = self.tokenizer(
    #         prompts,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True
    #     ).to(DEVICE)

    #     with torch.no_grad():
    #         outputs = self.model.generate(
    #             **inputs,
    #             max_new_tokens=512,
    #             do_sample=False,
    #             temperature=0.2
    #         )

    #     decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        scores = []
        for out in decoded:
            try:
                parsed = json.loads(out)
                scores.append(self._extract_reasoning_score(parsed))
            except Exception:
                # JSON failure → uncertainty
                scores.append(0.5)

        return torch.tensor(scores, device=DEVICE).view(-1, 1)


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

def get_wandb_resume_state():
    """
    Determines whether to resume a W&B run based on output existence.
    """
    if os.path.exists(OUTPUT_FILE) and os.path.exists(WANDB_RUN_META):
        with open(WANDB_RUN_META, "r") as f:
            meta = json.load(f)
        return meta["run_id"], True
    return None, False


# --------------------------
# Execution Logic
# --------------------------
def main():
    validate_paths()
    
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    
    # tracker = WandbTracker(project="fake-news-detection", run_name="e2e_mmfakebench_run")
    # set_tracker(tracker)
    # init()

    run_id, is_resume = get_wandb_resume_state()

    tracker = WandbTracker(
        project="fake-news-detection",
        run_name="e2e_mmfakebench_run",
        run_id=run_id,
        resume=is_resume,
    )

    set_tracker(tracker)
    init()

    if not is_resume:
        with open(WANDB_RUN_META, "w") as f:
            json.dump({"run_id": tracker.run.id}, f)


    # Init Models with memory clearing between loads
    s1 = SubjectivityAnalyzer()
    gc.collect(); torch.cuda.empty_cache()
    
    s2 = SemanticReasoner()
    gc.collect(); torch.cuda.empty_cache()
    
    s3 = CLIPLoraEncoder()
    gc.collect(); torch.cuda.empty_cache()

    # Load Data
    dataset = MMFakeBenchDataset(json_path=ANN_PATH, root_dir=IMAGE_ROOT)
    # Optimized DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, # Faster CPU -> GPU transfer
        prefetch_factor=2
    )
    # loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    output_file = os.path.join(OUTPUT_DIR, "extracted_features_v2.json")

    # Check if we already have progress
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            extracted_data = json.load(f)
        start_step = len(extracted_data)
        print(f"⏩ Resuming from step {start_step}...")
    else:
        extracted_data = []
        start_step = 0

    # print(f"--- Starting Extraction Loop on {DEVICE} ---")
    print(f"--- Starting Parallel Extraction Loop on {DEVICE} ---")

    for step, batch in enumerate(tqdm(loader)):
        if step * BATCH_SIZE < start_step: continue
        

    # for step, batch in enumerate(tqdm(loader)):

    #     # Skip steps we already processed
    #     if step < start_step:
    #         continue
    #     texts = batch["text"]
        
        # Since MMFakeBenchDataset already joined the root_dir, 
        # the batch['image_path'] is already absolute.
        img_paths = batch["image_path"] 
        
        try:
            # Batch inference across all stages
            with torch.cuda.amp.autocast(enabled=True): # Use Mixed Precision
                f1 = s1.get_score(batch["text"])
                f2 = s2.get_stance(batch["text"])
                f3 = s3.get_similarity(batch["image_path"], batch["text"])
            # f1 = s1.get_score(texts)
            # f2 = s2.get_stance(texts)
            # f3 = s3.get_similarity(img_paths, texts)

            # Collect results for the entire batch
            for i in range(len(batch["text"])):
                extracted_data.append({
                    "id": batch.get("id", [f"s_{step}_{i}"])[i],
                    "features": [f1[i].item(), f2[i].item(), f3[i].item()],
                    "label": 1 if batch["label"][i] == "Fake" else 0
                })
            

            # for i in range(len(texts)):
            #     extracted_data.append({
            #         "id": batch.get("id", [f"sample_{step}_{i}"])[i],
            #         "features": [f1[i].item(), f2[i].item(), f3[i].item()],
            #         "label": 1 if batch["label"][i] == "Fake" else 0
            #     })
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
