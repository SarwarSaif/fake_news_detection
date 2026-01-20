# src/evaluation/batch_processor_cache.py
import os
import csv
import threading
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
from src.tracking import log
from src.evaluation.similarity import SIMILARITY_METRICS
from src.utils.embedding_cache import EmbeddingCache

def _get_sample_id(sample: dict) -> str:
    """Determine unique id for a sample."""
    if "id" in sample and sample["id"] is not None:
        return str(sample["id"])
    img = sample.get("image_path") or sample.get("image")
    if isinstance(img, str):
        return os.path.splitext(os.path.basename(img))[0]
    return str(abs(hash(str(sample.get("text", ""))))[:12])

def process_dataset_in_batches_with_cache(
    dataset,
    batch_size,
    encoder,
    similarity="cosine",
    label=None,
    cache_dir="cache",
    results_csv=None,
    clear_cache=False,
    save_cache_every_n_batches: int = 1
):
    """Batch processing with FP16, vectorized similarity, async cache saving."""
    os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
    if results_csv and not os.path.exists(results_csv):
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "image_path", "label", "similarity", "score"])
            writer.writeheader()

    cache = EmbeddingCache(cache_dir)
    if clear_cache:
        cache.clear()
        print("[process] Cleared cache")

    sim_fn = similarity if callable(similarity) else SIMILARITY_METRICS.get(similarity, SIMILARITY_METRICS["cosine"])
    indices = [i for i in range(len(dataset)) if dataset[i]["label"] == label] if label else list(range(len(dataset)))

    all_scores = []
    batch_count = 0

    for start in tqdm(range(0, len(indices), batch_size), desc=f"Processing Batches ({label or 'all'})"):
        batch_indices = indices[start:start + batch_size]
        samples = [dataset[i] for i in batch_indices]
        ids = [_get_sample_id(s) for s in samples]
        missing_ids = cache.get_missing_ids(ids)

        # --- Prepare images & texts for missing entries ---
        imgs_to_encode, txts_to_encode, metas_to_add, ids_valid = [], [], [], []
        for s, _id in zip(samples, ids):
            if _id not in missing_ids:
                continue
            img_obj = None
            if "image" in s and not isinstance(s.get("image"), str):
                img_obj = s.get("image")
            else:
                try:
                    img_obj = Image.open(s["image_path"]).convert("RGB")
                except:
                    img_obj = None
            imgs_to_encode.append(img_obj)
            txts_to_encode.append(s.get("text", ""))
            metas_to_add.append({"id": _id, "image_path": s.get("image_path"), "text": s.get("text"), "label": s.get("label")})
            ids_valid.append(_id)

        # --- Filter valid entries ---
        valid_idx = [i for i, im in enumerate(imgs_to_encode) if im is not None and txts_to_encode[i] is not None]
        if valid_idx:
            imgs_valid = [imgs_to_encode[i] for i in valid_idx]
            txts_valid = [txts_to_encode[i] for i in valid_idx]
            metas_valid = [metas_to_add[i] for i in valid_idx]
            ids_valid = [ids_valid[i] for i in valid_idx]

            # --- Encode batch with FP16 ---
            img_embeds, txt_embeds = encoder.encode(imgs_valid, txts_valid)
            img_embeds = img_embeds.detach().cpu() if img_embeds is not None else None
            txt_embeds = txt_embeds.detach().cpu() if txt_embeds is not None else None

            cache.add_batch(ids_valid, [t for t in img_embeds] if img_embeds is not None else [None]*len(ids_valid),
                            [t for t in txt_embeds] if txt_embeds is not None else [None]*len(ids_valid),
                            metas_valid, save=False)

        # --- Compute vectorized similarity ---
        img_list, txt_list = cache.get_embeddings(ids)
        img_tensor = torch.stack([e for e in img_list if e is not None])
        txt_tensor = torch.stack([e for e in txt_list if e is not None])
        batch_scores = F.cosine_similarity(img_tensor, txt_tensor)

        # Assign scores back to all_scores
        all_scores.extend(batch_scores.tolist())

        # --- Save results to CSV ---
        if results_csv:
            rows = []
            for s, _id, sc in zip(samples, ids, batch_scores.tolist()):
                rows.append({
                    "id": _id,
                    "text": s.get("text"),
                    "image_path": s.get("image_path"),
                    "label": s.get("label"),
                    "similarity": similarity,
                    "score": sc
                })
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(results_csv, mode="a", header=False, index=False, encoding="utf-8")

        # --- Async cache save ---
        batch_count += 1
        if batch_count % save_cache_every_n_batches == 0:
            threading.Thread(target=cache.save).start()

    # Final cache save
    cache.save()
    return all_scores
