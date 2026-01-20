from PIL import Image
from torchvision import transforms

def load_image(path, transform=None):
    """
    Safely load an image from the given path and apply optional transform.
    Returns None if the image cannot be loaded.
    """
    try:
        img = Image.open(path).convert("RGB")
        if transform:
            img = transform(img)
        return img
    except Exception as e:
        print(f"⚠️ Could not load image {path}: {e}")
        return None
