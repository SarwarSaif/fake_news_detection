from tqdm import tqdm
import pandas as pd
from PIL import Image
from src.tracking import log
from src.evaluation.similarity import SIMILARITY_METRICS

def process_dataset_in_batches(dataset, batch_size, encoder, similarity="cosine", label=None, save_path=None):
    """
    Process a dataset in batches using the given encoder and similarity metric.
    Images are lazy-loaded to avoid exhausting system memory.
    Logs batch- and dataset-level metrics via the tracker.

    Args:
        dataset: Dataset supporting __getitem__ that returns a dict with keys "text", "image_path", "label".
        batch_size (int): Number of samples per batch.
        encoder: Encoder with an encode(images, texts) method.
        similarity (str|callable): Either a callable similarity function or a string key in SIMILARITY_METRICS.
        label (str, optional): If provided, only process samples with this label.
        save_path (str, optional): If provided, saves detailed results as CSV.

    Returns:
        tuple: (scores, records)
            scores (list[float]): Similarity scores for the processed subset.
            records (list[dict]): Detailed results per sample.
    """
    scores = []
    records = []

    # Pick similarity function
    sim_fn = similarity if callable(similarity) else SIMILARITY_METRICS.get(
        similarity, SIMILARITY_METRICS["cosine"]
    )

    # Filter dataset indices by label
    if label is not None:
        indices = [i for i in range(len(dataset)) if dataset[i]["label"] == label]
    else:
        indices = list(range(len(dataset)))

    # Batch processing
    for i in tqdm(range(0, len(indices), batch_size), desc=f"Processing Batches ({label or 'all'})"):
        batch_indices = indices[i:i + batch_size]
        batch = [dataset[j] for j in batch_indices]

        images, texts, metas = [], [], []
        for s in batch:
            try:
                img = Image.open(s["image_path"]).convert("RGB")
                if hasattr(encoder, "transform_image") and encoder.transform_image:
                    img = encoder.transform_image(img)
                images.append(img)
                texts.append(s["text"])
                metas.append(s)
            except Exception as e:
                print(f"Warning: Could not load image {s['image_path']} -> {e}")

        if not images:  # if no valid images
            scores.extend([0.0] * len(batch))
            continue

        # Encode and compute similarity
        img_embeds, txt_embeds = encoder.encode(images, texts)
        batch_scores = sim_fn(img_embeds, txt_embeds)
        batch_scores_list = batch_scores.cpu().tolist()

        # Log batch-level stats
        log({
            f"{similarity}/{label or 'all'}/batch_mean": sum(batch_scores_list) / len(batch_scores_list)
        })

        scores.extend(batch_scores_list)

        # Save detailed per-sample records
        for m, sc in zip(metas, batch_scores_list):
            records.append({
                "text": m["text"],
                "image_path": m["image_path"],
                "label": m["label"],
                "score": sc,
                "similarity": similarity,
            })

    # Log dataset-level stats
    if scores:
        log({
            f"{similarity}/{label or 'all'}/mean": sum(scores) / len(scores),
            f"{similarity}/{label or 'all'}/max": max(scores),
            f"{similarity}/{label or 'all'}/min": min(scores),
        })

    # Save CSV if requested
    if save_path and records:
        df = pd.DataFrame(records)
        df.to_csv(save_path, index=False)
        print(f"[INFO] Saved results to {save_path}")

    return scores #, records
