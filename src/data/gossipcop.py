# src/data/gossipcop.py
import os
import json
from torch.utils.data import Dataset

class GossipCopDataset(Dataset):
    """
    GossipCop dataset loader.
    Expects directory structure:

    gossipcop/
      fake/
        img/*.jpg
        text/*.json
      real/
        img/*.jpg
        text/*.json
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        for label in ["fake", "real"]:
            img_dir = os.path.join(root_dir, label, "img")
            text_dir = os.path.join(root_dir, label, "text")

            img_files = {os.path.splitext(f)[0]: os.path.join(img_dir, f)
                         for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))}
            text_files = {os.path.splitext(f)[0]: os.path.join(text_dir, f)
                          for f in os.listdir(text_dir) if f.endswith(".json")}

            # Match images with text files by ID
            common_ids = set(img_files.keys()) & set(text_files.keys())
            for sample_id in common_ids:
                with open(text_files[sample_id], "r", encoding="utf-8") as f:
                    text_data = json.load(f)

                self.samples.append({
                    "image_path": img_files[sample_id],
                    "text": text_data.get("text", ""),
                    "label": label.capitalize(),  # "Fake" or "Real"
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
