from torch.utils.data import Dataset
from PIL import Image
import os

class MMFakeBenchDataset(Dataset):
    def __init__(self, json_path, root_dir, transform_text=None, transform_image=None):
        import json
        with open(json_path, "r") as f:
            self.samples = json.load(f)
        self.root_dir = root_dir
        self.transform_text = transform_text
        self.transform_image = transform_image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        if self.transform_text:
            text = self.transform_text(text)

        label = sample.get("gt_answers")
        if label == "True":
            label = "Real"

        # Only return path, load later in batch
        return {
            "text": text,
            "image_path": os.path.join(self.root_dir, sample["image_path"].lstrip("/")),
            "label": label,
            "meta": sample
        }
    
    

