# =========================
# Main
# =========================
import os
import argparse
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import wandb
from collections import defaultdict
from torchvision import transforms

from src.tracking import set_tracker, init, log, finish
from src.tracking.wandb_tracker import WandbTracker
from src.models.clip_encoder import CLIPEncoder
# from src.utils.image_loader import load_image 

# Dataset imports
from src.data.mmfakebench import MMFakeBenchDataset
from src.data.gossipcop import GossipCopDataset  

# --------------------------
# Config
# --------------------------
BATCH_SIZE = 128  # Tuned for GPU
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

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models/clip-vit-base-patch32")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# --------------------------
# Cache manager
# --------------------------
def get_cache_path(base_dir, sim_type, label):
    cache_dir = os.path.join(base_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{sim_type}_{label}.json")

def load_cache(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_cache(path, cache):
    with open(path, "w") as f:
        json.dump(cache, f)

# --------------------------
# Dataset stats
# --------------------------
def compute_dataset_stats(dataset):
    total = len(dataset)
    class_counts = defaultdict(int)
    text_lengths = defaultdict(list)

    for sample in dataset:
        label = sample["label"]
        class_counts[label] += 1
        text_lengths[label].append(len(sample["text"].split()))

    stats = {
        "dataset/total_samples": total,
        "dataset/class_distribution": dict(class_counts)
    }

    for label, lengths in text_lengths.items():
        stats.update({
            f"dataset/{label}/avg_text_length": sum(lengths)/len(lengths),
            f"dataset/{label}/max_text_length": max(lengths),
            f"dataset/{label}/min_text_length": min(lengths),
        })

    return stats

# --------------------------
# Log similarity distributions
# --------------------------
def log_similarity_distribution(sim_type, fake_scores, real_scores, dataset_name):
    # Flatten in case any batch lists are inside
    fake_scores = [s for score in fake_scores for s in (score if isinstance(score, (list, tuple)) else [score])]
    real_scores = [s for score in real_scores for s in (score if isinstance(score, (list, tuple)) else [score])]

    fig, ax = plt.subplots(figsize=(6, 5))
    data = [fake_scores, real_scores]
    labels = [f"{sim_type.capitalize()} - Fake", f"{sim_type.capitalize()} - Real"]
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_title(f"{dataset_name}: {sim_type.capitalize()} Similarity")
    ax.set_ylabel("Similarity Score")
    wandb.log({f"{dataset_name}/{sim_type}_distribution": wandb.Image(fig)})
    plt.close(fig)


# --------------------------
# Run experiment (optimized)
# --------------------------
def run_experiment(dataset_info, clear_cache=False):
    name = dataset_info["name"]
    base_dir = dataset_info["base_dir"]
    ann_file = dataset_info["annotation_file"]
    dataset_cls = dataset_info["cls"]

    # --- Load dataset ---
    if ann_file:
        ann_path = os.path.join(base_dir, ann_file)
        dataset = dataset_cls(json_path=ann_path, root_dir=base_dir)
    else:
        dataset = dataset_cls(root_dir=base_dir)

    # --- Log dataset stats ---
    stats = compute_dataset_stats(dataset)
    log(stats)

    # --- Initialize encoder ---
    encoder = CLIPEncoder(
        transform_image=transforms.Compose([transforms.Resize((224, 224))]),
        local_dir=LOCAL_MODEL_DIR
    )

    # --- Precompute all text embeddings ---
    all_texts = [s["text"] for s in dataset]
    _, text_embeds = encoder.encode(None, all_texts)  # FP16 handled inside
    text_embeds = text_embeds.detach().cpu()  # offload to CPU

    # --- Similarity evaluation ---
    similarity_types = ["cosine", "euclidean"]
    for sim in similarity_types:
        # caches for both labels
        all_scores = {"Fake": {}, "Real": {}}
        for label in ["Fake", "Real"]:
            cache_path = get_cache_path(base_dir, sim, label)
            cache = {} if clear_cache else load_cache(cache_path)
            scores = []

            # --- Filter samples by label ---
            label_samples = [s for s in dataset if s["label"] == label]
            num_samples = len(label_samples)

            for start in range(0, num_samples, BATCH_SIZE):
                batch_samples = label_samples[start:start+BATCH_SIZE]

                # Load images
                batch_images = []
                valid_indices = []
                for i, s in enumerate(batch_samples):
                    img_path = s.get("image_path")
                    if img_path:
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(base_dir, img_path.lstrip("/"))
                        if os.path.exists(img_path):
                            batch_images.append(img_path)
                            valid_indices.append(i)
                        else:
                            batch_images.append(None)
                    else:
                        batch_images.append(None)

                # Prepare texts
                batch_texts = [s["text"] for s in batch_samples]

                # Compute batch similarity
                sim_scores = encoder.compute_similarity(batch_images, batch_texts, similarity=sim)  # vectorized
                # sim_scores = sim_scores.detach().cpu().tolist()
                

                # Assign scores to cache & local list
                for idx, s_idx in enumerate(valid_indices):
                    uid = str(batch_samples[s_idx].get("id", f"{label}_{start+s_idx}"))
                    score = sim_scores[idx]
                    cache[uid] = {"similarity_score": score}
                    scores.append(score)

                # Async cache save every 50 samples
                import threading
                if len(scores) % 50 == 0:
                    threading.Thread(target=save_cache, args=(cache_path, cache.copy())).start()

            # Final save
            save_cache(cache_path, cache)
            all_scores[label] = scores

            # # Collect fake/real scores for visualization
            # fake_scores = [v["similarity_score"] for k, v in cache.items() if "Fake" in k]
            # real_scores = [v["similarity_score"] for k, v in cache.items() if "Real" in k]
            # log_similarity_distribution(sim, fake_scores, real_scores, name)
            # print(f"{name} - {sim.capitalize()} ({label}): {scores[:5]}")
        # log **once per similarity type**, combining Fake + Real
        log_similarity_distribution(sim, all_scores["Fake"], all_scores["Real"], name)

# --------------------------
# Main execution
# --------------------------
if __name__ == "__main__":
    load_dotenv()
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_cache", action="store_true", help="Clear cache and recompute all embeddings")
    args = parser.parse_args()

    for ds_info in DATASETS:
        dataset_name = ds_info["name"]

        tracker = WandbTracker(
            project="fake-news-detection",
            run_name=f"clip-eval_{dataset_name}",
            config={
                "encoder": "openai/clip-vit-base-patch32",
                "batch_size": BATCH_SIZE,
                "clear_cache": args.clear_cache,
                "dataset": dataset_name,
            },
            group="clip-eval"
        )
        set_tracker(tracker)
        init()

        print(f"\n=== Running experiment for {dataset_name} ===")
        run_experiment(ds_info, clear_cache=args.clear_cache)

        finish()
