import torch
import torch.nn.functional as F

def cosine_similarity(img_embeds, txt_embeds):
    return torch.diag(torch.matmul(img_embeds, txt_embeds.T))

def euclidean_distance(img_embeds, txt_embeds):
    return -torch.norm(img_embeds - txt_embeds, p=2, dim=1)  # negative for "similarity"

def manhattan_distance(img_embeds, txt_embeds):
    return -torch.norm(img_embeds - txt_embeds, p=1, dim=1)

SIMILARITY_METRICS = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
}
