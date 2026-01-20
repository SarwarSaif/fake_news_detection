# src/utils/visualize_outliers.py
import os
import shutil
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import textwrap

# -----------------------------
# Dataset config
# -----------------------------
from src.data.mmfakebench import MMFakeBenchDataset
from src.data.gossipcop import GossipCopDataset

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

# -----------------------------
# Per-sample visualization
# -----------------------------
def visualize_outlier_sample(img_path, text, score, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # --- Image ---
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Could not load {img_path}: {e}")
            img = Image.new("RGB", (256, 256), color=(200, 200, 200))
            text = f"[MISSING IMAGE]\n{text}"
    else:
        print(f"⚠️ Missing image: {img_path}")
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        text = f"[MISSING IMAGE]\n{text}"

    ax[0].imshow(img)
    ax[0].axis("off")

    # --- Text + metadata ---
    wrapped_text = "\n".join(textwrap.wrap(str(text), width=80))
    info = f"Path: {img_path}\nScore: {score:.4f}\n\n{wrapped_text}"

    ax[1].text(0, 1, info, fontsize=10, va="top", ha="left", wrap=True)
    ax[1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main function
# -----------------------------
def main(dataset_name, label, similarity, score_min=None, score_max=None, clean=False):
    # Resolve dataset base_dir
    dataset_info = next((d for d in DATASETS if d["name"] == dataset_name), None)
    if dataset_info is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS config.")
    
    base_dir = dataset_info["base_dir"]
    csv_filename = f"{similarity}_{label}_results.csv"
    csv_path = os.path.join(base_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Filter by score
    if score_min is not None:
        df = df[df['score'] >= score_min]
    if score_max is not None:
        df = df[df['score'] <= score_max]
    
    out_dir = os.path.join(base_dir, "outputs", similarity, "standard", label)
    # out_dir = os.path.join(base_dir, "outputs", similarity, "outliers", label)
    if clean and os.path.exists(out_dir):
        shutil.rmtree(out_dir)  # remove old outputs
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in df.iterrows():
        img_path = row['image_path']
        text = row['text']
        score = row['score']

        save_path = os.path.join(out_dir, f"outlier_{idx}.png")
        visualize_outlier_sample(img_path, text, score, save_path)

    print(f"✅ Saved {len(df)} outliers to {out_dir}")


# -----------------------------
# Command-line interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save outlier images+texts from CSV results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., MMFakeBench_val)")
    parser.add_argument("--label", type=str, required=True, choices=["fake", "real"], help="Label to filter")
    parser.add_argument("--similarity", type=str, required=True, choices=["cosine", "euclidean"], help="Similarity type")
    parser.add_argument("--score_min", type=float, default=None, help="Minimum similarity score filter")
    parser.add_argument("--score_max", type=float, default=None, help="Maximum similarity score filter")
    parser.add_argument("--clean", action="store_true", help="Remove old outliers before saving new ones")
    args = parser.parse_args()

    main(
        args.dataset,
        args.label,
        args.similarity,
        score_min=args.score_min,
        score_max=args.score_max,
        clean=args.clean
    )
