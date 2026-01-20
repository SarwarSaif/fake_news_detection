import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.cuda.amp import autocast

class CLIPEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", local_dir=None, transform_image=None):
        """Wrapper around CLIP for encoding text and images with batching and FP16 support."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.local_dir = local_dir

        # Load model + processor
        if local_dir and os.path.exists(local_dir) and os.listdir(local_dir):
            self.model = CLIPModel.from_pretrained(local_dir).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(local_dir)
        else:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
                self.model.save_pretrained(local_dir)
                self.processor.save_pretrained(local_dir)

        self.transform_image = transform_image

    @staticmethod
    def chunk_text(text, tokenizer, max_length=77):
        """Split long text into <=77 token chunks for CLIP text encoder."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = [tokenizer.decode(tokens[i:i+max_length]) for i in range(0, len(tokens), max_length)]
        return chunks if chunks else [""]

    def encode(self, images, texts):
        """
        Encode images and texts into normalized embeddings.
        Supports batch processing and FP16.
        """
        img_embeds = None
        txt_embeds = None

        # --- Process images ---
        if images:
            processed_imgs = []
            for img in images:
                if img is None:
                    continue
                if isinstance(img, str):
                    if not os.path.exists(img):
                        print(f"⚠️ Image path does not exist: {img}")
                        continue
                    try:
                        img = Image.open(img).convert("RGB")
                    except:
                        continue
                if self.transform_image:
                    img = self.transform_image(img)
                processed_imgs.append(img)

            if processed_imgs:
                with torch.no_grad(), autocast():
                    inputs = self.processor(images=processed_imgs, return_tensors="pt").to(self.device)
                    outputs = self.model.get_image_features(**inputs)
                    img_embeds = F.normalize(outputs, p=2, dim=1)

        # --- Process texts ---
        if texts:
            txt_embeds_list = []
            for text in texts:
                chunks = self.chunk_text(text, self.processor.tokenizer)
                chunk_inputs = self.processor(text=chunks, return_tensors="pt", padding=True, truncation=True)
                chunk_inputs = {k: v.to(self.device) for k, v in chunk_inputs.items()}
                with torch.no_grad(), autocast():
                    chunk_outputs = self.model.get_text_features(**chunk_inputs)
                    text_embed = F.normalize(chunk_outputs.mean(dim=0, keepdim=True), p=2, dim=1)
                    txt_embeds_list.append(text_embed)
            txt_embeds = torch.cat(txt_embeds_list, dim=0)

        return img_embeds, txt_embeds

    def compute_similarity(self, images, texts, similarity="cosine"):
        """
        Compute similarity between image and text embeddings.
        Returns a list of scores.
        """
        img_embeds, txt_embeds = self.encode(images, texts)
        if img_embeds is None:
            return [None] * txt_embeds.size(0)

        if similarity == "cosine":
            # Vectorized cosine similarity
            sim = F.cosine_similarity(img_embeds, txt_embeds)
        elif similarity == "euclidean":
            sim = -torch.norm(img_embeds - txt_embeds, dim=1)
        else:
            raise ValueError(f"Unsupported similarity type: {similarity}")

        return sim.cpu().numpy().tolist()  # always a list

