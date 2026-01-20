# =========================
# Base dataset class
# =========================
import os
import torch
from torch.utils.data import Dataset
import json

class BaseMultimodalDataset(Dataset):
    def __init__(self, transform_text=None, transform_image=None):
        self.samples = []
        self.transform_text = transform_text
        self.transform_image = transform_image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")