# src/models/fact_checker.py
import os
import json
import argparse
import time
from typing import Tuple, Dict, Any, List
from dotenv import load_dotenv

import pandas as pd
import torch

from unsloth import FastModel

# Dataset imports
from src.data.mmfakebench import MMFakeBenchDataset
from src.data.gossipcop import GossipCopDataset

from src.tracking import set_tracker, init, finish
from src.tracking.wandb_tracker import WandbTracker

# -----------------------
# Config / Defaults
# -----------------------
DEFAULT_MODEL_NAME = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Model loader / generator
# -----------------------
def load_gemma_model(model_name: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE):
    """
    Load the Gemma-3N model (FastModel wrapper) and tokenizer.
    Returns (model, tokenizer, device).
    """
    print(f"[fact_checker] Loading model {model_name} on {device} ...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,  # None means fp16 typically, depending on unsloth
        max_seq_length=2048,
        load_in_4bit=True,
        full_finetuning=False,
    )
    return model, tokenizer, device

def gemma_generate(model, tokenizer, messages: List[dict], device: str = DEFAULT_DEVICE, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    """
    Generate textual output using the Gemma model with the chat-image template.
    `messages` should follow the same structure you used earlier, e.g.:
      [{"role":"user","content":[{"type":"image","image":path},{"type":"text","text":"Prompt..."}]}]
    Returns the generated text (string). Tries to decode returned sequences; falls back to streaming approach if necessary.
    """
    # Prepare inputs
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    # Generation call - attempt to decode returned sequences
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=64,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )
            # Many generate implementations return a dict-like with "sequences"
            seqs = None
            if isinstance(outputs, dict) and "sequences" in outputs:
                seqs = outputs["sequences"]
            elif hasattr(outputs, "sequences"):
                seqs = outputs.sequences
            if seqs is not None:
                # decode
                decoded = tokenizer.batch_decode(seqs, skip_special_tokens=True)
                # If multiple sequences, join the last one
                return decoded[-1] if isinstance(decoded, list) else decoded
        except Exception as e:
            # Fall back to a streaming approach or simpler generate call
            print(f"[fact_checker] Warning: primary generate call failed: {e}. Trying fallback generate.")
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=64,
                )
                # Attempt to decode whatever returned
                if isinstance(outputs, (list, tuple)):
                    # if outputs is a list of token ids
                    return tokenizer.decode(outputs[0], skip_special_tokens=True)
                elif hasattr(outputs, "sequences"):
                    return tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[-1]
            except Exception as e2:
                raise RuntimeError(f"Gemma generate fallback also failed: {e2}")

    raise RuntimeError("Could not decode generation output from model.generate()")
# -----------------------
# Image preprocessing utility
# -----------------------   

from PIL import Image

def preprocess_image(image_path: str, target_size=(256, 256)):
    """
    Open image, resize it, return a PIL.Image object (not BytesIO).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    return img



# -----------------------
# JSON parsing utility
# -----------------------   
import json
import re

def safe_parse_json(raw_text: str):
    """
    Try to extract the first valid JSON object from model output.
    Handles trailing commas, markdown, and extra quotes.
    """
    # Remove markdown blocks and leading 'model' words
    raw_clean = re.sub(r"```json|```|^model\s*", "", raw_text, flags=re.MULTILINE).strip()
    
    # Find first { ... } block
    brace_matches = list(re.finditer(r"\{.*\}", raw_clean, re.DOTALL))
    if not brace_matches:
        return {"error": "no_json_found", "raw": raw_text}

    json_text = brace_matches[0].group(0)

    # Remove trailing commas before closing braces/brackets
    json_text = re.sub(r",(\s*[\]}])", r"\1", json_text)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        return {"error": "json_parse_error", "exception": str(e), "raw": raw_text}

# -----------------------
# Fact-checking prompt
# -----------------------
FACTCHECK_PROMPT_TEMPLATE = """
You are an assistant that helps a forensic analyst verify claims. 
You are provided: (1) an image (accessible by file path), and (2) an associated textual message.

RULES:
- Use BOTH the image+text evidence AND reliable, well-established world knowledge.
- Clearly separate evidence sources:
  * If support/contradiction comes from the image or text, reference those explicitly.
  * If support/contradiction comes from general world knowledge, state that explicitly.
- If neither the image, text, nor reliable world knowledge can settle the claim, return evaluation = "insufficient_evidence".
- Do not hallucinate uncertain or speculative knowledge. Only use high-confidence, factual knowledge (e.g., geography, widely known institutions, major events).
- Evidence must be short, explicit, and traceable to either "image", "text", or "world_knowledge".

Task: Produce a structured JSON object (parsable by machines) with these fields:

1) image_description: concise summary of the visible content (objects, people, setting, any readable text). Max 200 words.
2) extracted_claims: list of short factual claims explicitly asserted in the message text.
3) claim_evaluations: list of objects, each with:
   - claim: the extracted claim string
   - evaluation: one of ["supports", "contradicts", "insufficient_evidence"]
   - evidence: explanation with source labels ("image", "text", "world_knowledge")
   - confidence: float 0.0–1.0
4) forensic_checklist: list of short, actionable items for human verification 
   (examples: "check EXIF/metadata", "reverse image search", "verify institutional location from trusted sources").
5) final_verdict: one of ["likely_real", "likely_fake", "uncertain"].
6) verdict_rationale: short rationale (max 200 words), explicitly distinguishing between image/text evidence and world knowledge.
7) analysis_steps: numbered list of reasoning steps followed.

CONSTRAINTS:
- Output must be valid JSON ONLY, no extra commentary.
- Be concise but precise.
- Always mark which parts of the evidence come from: image, text, or world_knowledge.

Now analyze the image and text provided and return the JSON.
"""


# -----------------------
# Fact-checking function
# -----------------------
def fact_check_pair(model, tokenizer, image_path: str, text: str, device: str = DEFAULT_DEVICE, max_new_tokens: int = 512) -> Tuple[Dict[str, Any], str]:
    """
    Fact-check a single (image_path, text) pair.
    Returns (parsed_json_result, raw_model_text).
    The parsed_json_result may be partial if the model returned non-JSON content; we try to extract the first JSON blob.
    """
    

    # Preprocess
    image_pil = preprocess_image(image_path)

    # Compose messages: send image + the prompt + appended text
    system_prompt = FACTCHECK_PROMPT_TEMPLATE
    user_content = [
        {"type": "image", "image": image_pil},
        {"type": "text", "text": f"Associated message text:\n{text}"}
    ]
    messages = [{"role": "user", "content": user_content + [{"type":"text", "text": system_prompt}]}]

    raw_out = gemma_generate(model, tokenizer, messages, device=device, max_new_tokens=max_new_tokens, temperature=0.8)

    # attempt to parse JSON from model output (find first {...})
    parsed = None
    raw = raw_out.strip()
    try:
        # Find the first JSON object in raw text
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_text = raw[first_brace:last_brace+1]
            # parsed = json.loads(json_text)
            parsed = safe_parse_json(raw)

        else:
            parsed = {"error": "no_json_found", "raw": raw}
    except Exception as e:
        parsed = {"error": "json_parse_error", "exception": str(e), "raw": raw}

    return parsed, raw

# -----------------------
# Utilities: dataset loader, runner
# -----------------------
DATASETS = [
    {
        "name": "MMFakeBench_val",
        "cls": MMFakeBenchDataset,
        "base_dir": "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection/data/MMFakeBench/MMFakeBench_val",
        "annotation_file": "source/MMFakeBench_val.json"
    },
    {
        "name": "GossipCop_val",
        "cls": GossipCopDataset,
        "base_dir": "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection/data/gossipcop",
        "annotation_file": None
    }
]

def load_dataset_by_name(name: str):
    ds = next((d for d in DATASETS if d["name"] == name), None)
    if ds is None:
        raise ValueError(f"Dataset {name} not found in config")
    if ds["annotation_file"]:
        return ds["cls"](json_path=os.path.join(ds["base_dir"], ds["annotation_file"]), root_dir=ds["base_dir"]), ds["base_dir"]
    else:
        return ds["cls"](root_dir=ds["base_dir"]), ds["base_dir"]

# -----------------------
# Runner
# -----------------------
def run_fact_checks_on_dataset(dataset_name: str, label: str = None, max_samples: int = None, model_name: str = DEFAULT_MODEL_NAME, out_root: str = "outputs/fact_checks", device: str = DEFAULT_DEVICE):
    model, tokenizer, device = load_gemma_model(model_name=model_name, device=device)
    dataset, base_dir = load_dataset_by_name(dataset_name)

    os.makedirs(out_root, exist_ok=True)
    run_dir = os.path.join(base_dir, out_root, f"{dataset_name}_{label or 'all'}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    load_dotenv()
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    tracker = WandbTracker(
        project="fake-news-detection",
        run_name=f"gemma_fact_check_{dataset_name}",
        config={"dataset": dataset_name, "label": label, "model": model_name},
        group="fact-check"
    )
    set_tracker(tracker)
    init()

    summary_rows, count, skipped = [], 0, 0
    label_norm = label.lower() if label else None

    for idx, sample in enumerate(dataset):
        sample_label = sample.get("label", "").lower()
        if label_norm and sample_label != label_norm:
            continue

        image_path = sample.get("image_path") or sample.get("image")
        if image_path and not os.path.isabs(image_path):
            image_path = os.path.join(base_dir, image_path.lstrip("/"))
        text = sample.get("text", "")

        if not text.strip():
            print(f"[fact_checker] ⚠️ Skipping idx={idx}, empty text.")
            skipped += 1
            continue
        if not image_path or not os.path.exists(image_path):
            print(f"[fact_checker] ⚠️ Skipping idx={idx}, missing image {image_path}.")
            skipped += 1
            continue

        print(f"[fact_checker] Processing idx={idx}, label={sample_label}")
        parsed, raw = fact_check_pair(model, tokenizer, image_path, text, device=device, max_new_tokens=2048)

        sample_id = str(sample.get("id", f"{sample_label}_{idx}"))
        with open(os.path.join(run_dir, f"{sample_id}.json"), "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        with open(os.path.join(run_dir, f"{sample_id}_raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)

        verdict = parsed.get("final_verdict") if isinstance(parsed, dict) else None
        confidence = None
        if isinstance(parsed, dict) and parsed.get("claim_evaluations"):
            try:
                vals = [float(c.get("confidence", 0.0)) for c in parsed["claim_evaluations"]]
                confidence = sum(vals)/len(vals) if vals else None
            except Exception:
                pass

        summary_rows.append({
            "id": sample_id,
            "image_path": image_path,
            "label": sample_label,
            "verdict": verdict,
            "confidence": confidence,
        })

        tracker.log_llm_interaction(
            sample_id=sample_id,
            dataset=dataset_name,
            image_path=image_path,
            text=text,
            raw_output=raw,
            parsed=parsed,
            verdict=verdict,
            confidence=confidence,
            step=idx,
        )

        count += 1
        if max_samples and count >= max_samples:
            break

    tracker.finish()
    pd.DataFrame(summary_rows).to_csv(os.path.join(run_dir, "summary.csv"), index=False, encoding="utf-8")
    print(f"[fact_checker] ✅ Done. Processed={count}, Skipped={skipped}. Results in {run_dir}")
    return run_dir

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemma-3N fact-checks on dataset images + text.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., MMFakeBench_val)")
    parser.add_argument("--label", type=str, default=None, help="Optional label filter: fake or real")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--out_root", type=str, default="outputs/fact_checks")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    run_fact_checks_on_dataset(args.dataset, label=args.label, max_samples=args.max_samples, model_name=args.model_name, out_root=args.out_root, device=args.device)

# python3 -m src.models.fact_checker --dataset MMFakeBench_val --label fake --max_samples 50
