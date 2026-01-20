# src/utils/embedding_cache.py
import os
import torch
import pandas as pd
from typing import Dict, List, Optional

class EmbeddingCache:
    """
    Per-sample embedding cache.

    Layout:
      <cache_dir>/image_embeds.pt   -> dict {id: tensor(cpu)}
      <cache_dir>/text_embeds.pt    -> dict {id: tensor(cpu)}
      <cache_dir>/meta.csv         -> id, image_path, text, label, any other meta

    Notes:
      - Embeddings saved as CPU tensors (torch.save). Loading entire dict into memory;
        acceptable for medium-size datasets. For very large datasets, consider LMDB/HDF5.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.image_path = os.path.join(self.cache_dir, "image_embeds.pt")
        self.text_path = os.path.join(self.cache_dir, "text_embeds.pt")
        self.meta_path = os.path.join(self.cache_dir, "meta.csv")

        # in-memory dicts
        self.image_embeds: Dict[str, torch.Tensor] = {}
        self.text_embeds: Dict[str, torch.Tensor] = {}
        self.meta: Dict[str, dict] = {}

        self._loaded = False
        self.load()

    def load(self):
        # load saved embeddings/meta if present
        if os.path.exists(self.image_path):
            try:
                d = torch.load(self.image_path, map_location="cpu")
                if isinstance(d, dict):
                    self.image_embeds = d
            except Exception as e:
                print(f"[EmbeddingCache] Failed to load image embeds: {e}")

        if os.path.exists(self.text_path):
            try:
                d = torch.load(self.text_path, map_location="cpu")
                if isinstance(d, dict):
                    self.text_embeds = d
            except Exception as e:
                print(f"[EmbeddingCache] Failed to load text embeds: {e}")

        if os.path.exists(self.meta_path):
            try:
                df = pd.read_csv(self.meta_path)
                for _, row in df.iterrows():
                    _id = str(row["id"])
                    self.meta[_id] = row.to_dict()
            except Exception as e:
                print(f"[EmbeddingCache] Failed to load meta: {e}")

        self._loaded = True

    def has_both(self, _id: str) -> bool:
        return (_id in self.image_embeds) and (_id in self.text_embeds)

    def get_missing_ids(self, ids: List[str]) -> List[str]:
        return [i for i in ids if not self.has_both(i)]

    def add_batch(self, ids: List[str], image_embeds: List[torch.Tensor],
                  text_embeds: List[torch.Tensor], metas: Optional[List[dict]] = None,
                  save: bool = True):
        """
        Add embeddings for a batch of ids. Embeddings moved to CPU.
        If metas provided, merge into meta store (metas must align with ids).
        If save=True, persist to disk after update.
        """
        for idx, _id in enumerate(ids):
            ie = image_embeds[idx] if image_embeds is not None else None
            te = text_embeds[idx] if text_embeds is not None else None

            if ie is not None:
                # convert to CPU to save memory on GPU
                self.image_embeds[_id] = ie.detach().cpu().clone()
            if te is not None:
                self.text_embeds[_id] = te.detach().cpu().clone()

            if metas is not None:
                self.meta[_id] = metas[idx].copy()

        if save:
            self.save()

    def get_embeddings(self, ids: List[str]):
        """
        Returns (image_tensors_list, text_tensors_list), both lists aligned with input ids.
        If an id missing an embedding, returns None at that position.
        """
        images = [self.image_embeds.get(i, None) for i in ids]
        texts = [self.text_embeds.get(i, None) for i in ids]
        return images, texts

    def save(self):
        # Save full dicts. Overwrites atomically by writing to tmp then moving.
        tmp_img = self.image_path + ".tmp"
        tmp_txt = self.text_path + ".tmp"
        tmp_meta = self.meta_path + ".tmp"

        try:
            torch.save(self.image_embeds, tmp_img)
            os.replace(tmp_img, self.image_path)
        except Exception as e:
            print(f"[EmbeddingCache] Error saving image_embeds: {e}")

        try:
            torch.save(self.text_embeds, tmp_txt)
            os.replace(tmp_txt, self.text_path)
        except Exception as e:
            print(f"[EmbeddingCache] Error saving text_embeds: {e}")

        try:
            # meta -> DataFrame
            if self.meta:
                df = pd.DataFrame.from_dict(self.meta, orient="index")
                df = df.reset_index().rename(columns={"index": "id"})
                df.to_csv(tmp_meta, index=False)
                os.replace(tmp_meta, self.meta_path)
            else:
                # if empty, remove file
                if os.path.exists(self.meta_path):
                    os.remove(self.meta_path)
        except Exception as e:
            print(f"[EmbeddingCache] Error saving meta: {e}")

    def clear(self):
        # Remove files and clear in-memory stores
        for p in (self.image_path, self.text_path, self.meta_path):
            if os.path.exists(p):
                os.remove(p)
        self.image_embeds.clear()
        self.text_embeds.clear()
        self.meta.clear()
        self._loaded = True

    def get_cached_ids(self) -> List[str]:
        # return ids that have both embeddings
        return [i for i in self.image_embeds.keys() if i in self.text_embeds]

    def __len__(self):
        return len(self.get_cached_ids())
